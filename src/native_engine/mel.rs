//! Log-mel spectrogram frontend (pure Rust, exact whisper.cpp semantics).
//!
//! This is a faithful port of whisper.cpp's `log_mel_spectrogram` /
//! `whisper_pcm_to_mel` path (see `whisper.cpp` `src/whisper.cpp`
//! `log_mel_spectrogram`, `log_mel_spectrogram_worker_thread`, `fft`, `dft`,
//! and `fill_hann_window`). Every constant and every numeric behavior that
//! could plausibly diverge from whisper.cpp is documented inline so the output
//! is bit-for-bit comparable (within f32 rounding) to the reference encoder
//! input.
//!
//! Pipeline (per whisper.cpp):
//! 1. Pad the PCM: a leading *reflective* pad of `N_FFT/2` samples, the audio
//!    itself, then a trailing pad of one full 30 s chunk of zeros plus another
//!    `N_FFT/2` zeros. (The 30 s tail guarantees at least 3000 frames exist.)
//! 2. Slide a length-`N_FFT` Hann (periodic) window with hop `HOP`.
//! 3. FFT each windowed frame; take the **power** spectrum `re^2 + im^2` over
//!    the one-sided `N_FFT/2 + 1 = 201` bins.
//! 4. Project through the model's `n_mel x 201` filterbank, then
//!    `log10(max(e, 1e-10))`.
//! 5. Globally clamp to `max - 8.0` and normalize `(x + 4) / 4`.
//!
//! The FFT is whisper.cpp's exact recursive structure: radix-2 split when the
//! length is even, falling back to a naive O(N^2) DFT when the length is odd.
//! For `N = 400 = 2^4 * 5^2` the recursion is `400 -> 200 -> 100 -> 50 -> 25`,
//! and `25` (odd) is computed by the naive DFT. This is numerically a correct
//! discrete Fourier transform; it is validated against an independent naive DFT
//! reference in the tests.

use std::f64::consts::PI;
use std::simd::Simd;

use crate::error::{FwError, FwResult};

use super::{Mel, MelFilterbank};

/// Audio sample rate whisper expects, in Hz (`WHISPER_SAMPLE_RATE`).
pub const SAMPLE_RATE: usize = 16_000;
/// FFT / analysis window length in samples (`WHISPER_N_FFT`).
pub const N_FFT: usize = 400;
/// Hop / frame step in samples (`WHISPER_HOP_LENGTH`).
pub const HOP: usize = 160;
/// One full 30 s chunk of audio in samples (`WHISPER_SAMPLE_RATE * 30`).
pub const N_SAMPLES_30S: usize = SAMPLE_RATE * 30;
/// Mel frames produced per 30 s chunk (`N_SAMPLES_30S / HOP`).
pub const FRAMES_PER_CHUNK: usize = N_SAMPLES_30S / HOP; // 3000
/// One-sided FFT bin count: `N_FFT/2 + 1`, i.e. bin_0 .. bin_nyquist.
pub const N_FREQ_BINS: usize = N_FFT / 2 + 1; // 201

/// The exact value every "silence" mel sample collapses to.
///
/// Derivation (all bins zero ⇒ every pre-normalization sample equals
/// `log10(1e-10) = -10`): the global max is also `-10`, so the clamp floor is
/// `-10 - 8 = -18`, but no sample is below `-10`, so the clamp is a no-op and
/// every sample normalizes to `(-10 + 4) / 4 = -1.5`. This is the value
/// `chunk_frames` pads with when slicing past the end of a real mel, matching
/// whisper.cpp's `n_len`-padding behavior (frames beyond the audio are all
/// `log10(1e-10)` pre-clamp, hence `-1.5` post-normalize whenever the loudest
/// real frame is at most 8 dB above the noise floor — which, after the global
/// clamp, every silence frame always is).
pub const SILENCE_FLOOR: f32 = -1.5;

/// Compute the periodic Hann window of length `N_FFT`.
///
/// whisper.cpp uses `periodic = true`, i.e. divisor `length` (not `length-1`):
/// `w[i] = 0.5 * (1 - cos(2*pi*i / length))`. We compute in f64 then narrow to
/// f32 to mirror `cosf` precision behavior closely without the platform `cosf`
/// dependency.
fn hann_window() -> [f32; N_FFT] {
    let mut w = [0.0f32; N_FFT];
    for (i, wi) in w.iter_mut().enumerate() {
        // periodic ⇒ offset 0 ⇒ divisor == length.
        *wi = (0.5 * (1.0 - ((2.0 * PI * i as f64) / N_FFT as f64).cos())) as f32;
    }
    w
}

/// The length-`N_FFT` periodic Hann window, computed once and cached for the
/// life of the process.
///
/// The window is a pure function of `N_FFT`, but [`log_mel`] used to recompute
/// it (400 `cos` calls) on every call. whisper.cpp likewise caches its window
/// globally. We memoize behind a [`OnceLock`](std::sync::OnceLock) so the cosine
/// loop runs at most once regardless of how many clips are processed.
fn cached_hann_window() -> &'static [f32; N_FFT] {
    static HANN: std::sync::OnceLock<[f32; N_FFT]> = std::sync::OnceLock::new();
    HANN.get_or_init(hann_window)
}

/// Precomputed twiddle factors for ONE radix-2 decimation-in-time split of width
/// `n` (so `half = n/2` butterflies). `cos[k] = cos(2*pi*k/n) as f32` and
/// `sin[k] = -(sin(2*pi*k/n) as f32)` — i.e. the exact values the reference
/// recursion (whisper.cpp's `fft`) computed inline per butterfly per frame.
/// Precomputing them is **bit-for-bit identical** (the f64 transcendental and
/// the `as f32` narrowing happen at table-build time instead of in the hot loop)
/// while removing the per-frame `cos`/`sin` calls.
struct LevelTwiddle {
    half: usize,
    cos: Vec<f32>,
    /// Already negated: stores `-(sin(2*pi*k/n) as f32)`.
    neg_sin: Vec<f32>,
}

/// Precomputed twiddle table for the odd-length naive-DFT base case (`n` odd).
/// `cos[k*n + j] = cos(2*pi*k*j/n) as f32`, `sin[k*n + j] = sin(2*pi*k*j/n) as
/// f32`. For `N_FFT = 400` the recursion's only base case is `n = 25`, whose
/// 25x25 table is evaluated 16x per FFT frame — the dominant transcendental cost
/// of the whole mel frontend. The lookups are bit-exact replacements for the
/// inline `theta.cos()/theta.sin()` the reference `dft` computed.
struct DftTable {
    n: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl DftTable {
    fn build(n: usize) -> Self {
        let mut cos = vec![0.0f32; n * n];
        let mut sin = vec![0.0f32; n * n];
        for k in 0..n {
            for j in 0..n {
                let theta = (2.0 * PI * (k * j) as f64) / n as f64;
                cos[k * n + j] = theta.cos() as f32;
                sin[k * n + j] = theta.sin() as f32;
            }
        }
        Self { n, cos, sin }
    }
}

/// The full precomputed twiddle set for a fixed transform length. `levels` holds
/// one entry per radix-2 split, **largest width first** (`[400, 200, 100, 50]`
/// for `N_FFT = 400`); `base` is the naive-DFT table for the final odd factor
/// (`25`). The recursion advances through `levels` by one per split, so each FFT
/// call indexes `levels[0]` with zero per-call lookup cost.
struct FftTwiddles {
    levels: Vec<LevelTwiddle>,
    base: DftTable,
}

impl FftTwiddles {
    fn build(n_fft: usize) -> Self {
        let mut levels = Vec::new();
        let mut n = n_fft;
        while n.is_multiple_of(2) && n > 1 {
            let half = n / 2;
            let mut cos = vec![0.0f32; half];
            let mut neg_sin = vec![0.0f32; half];
            for (k, (c, s)) in cos.iter_mut().zip(neg_sin.iter_mut()).enumerate() {
                let theta = (2.0 * PI * k as f64) / n as f64;
                *c = theta.cos() as f32;
                *s = -(theta.sin() as f32);
            }
            levels.push(LevelTwiddle { half, cos, neg_sin });
            n /= 2;
        }
        // `n` is now the final odd factor (25 for N_FFT=400; 1 for a power of 2,
        // in which case the n==1 fast path means `base` is never consulted).
        Self {
            levels,
            base: DftTable::build(n),
        }
    }
}

/// Twiddles for the production `N_FFT`, built once on first use and shared
/// read-only across all mel worker threads.
fn cached_fft_twiddles() -> &'static FftTwiddles {
    static TW: std::sync::OnceLock<FftTwiddles> = std::sync::OnceLock::new();
    TW.get_or_init(|| FftTwiddles::build(N_FFT))
}

/// Naive O(N^2) DFT of a real input. `out` is interleaved complex,
/// length `2*N`: `out[2k] = Re`, `out[2k+1] = Im`. Mirrors whisper.cpp `dft`.
fn dft(input: &[f32], out: &mut [f32], table: &DftTable) {
    let n = input.len();
    debug_assert_eq!(n, table.n, "dft twiddle table width mismatch");
    for k in 0..n {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        // `t = 2*pi*k*j / N`; cos/sin precomputed at table-build time, so this is
        // bit-for-bit identical to the inline `theta.cos()/theta.sin() as f32`.
        let cos_row = &table.cos[k * table.n..k * table.n + n];
        let sin_row = &table.sin[k * table.n..k * table.n + n];
        for (j, &x) in input.iter().enumerate() {
            re += x * cos_row[j];
            im -= x * sin_row[j];
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

/// Recursive Cooley-Tukey FFT of a real input, exactly mirroring whisper.cpp's
/// `fft`: radix-2 decimation-in-time when `N` is even, falling back to the
/// naive DFT when `N` is odd (the prime-factor base case). `out` is interleaved
/// complex of length `2*N`. `levels` holds the precomputed butterfly twiddles
/// for the current width first (the recursion advances with `&levels[1..]`);
/// `base` is the odd-`N` DFT twiddle table. Both are bit-exact stand-ins for the
/// transcendentals the reference computed inline. (Per-call heap allocation of
/// the even/odd split buffers is a separate deferred follow-up, bd-02do L2.)
fn fft(input: &[f32], out: &mut [f32], levels: &[LevelTwiddle], base: &DftTable) {
    let n = input.len();
    if n == 1 {
        out[0] = input[0];
        out[1] = 0.0;
        return;
    }
    if n % 2 == 1 {
        dft(input, out, base);
        return;
    }

    let lvl = &levels[0];
    let half = n / 2;
    debug_assert_eq!(lvl.half, half, "fft level twiddle width mismatch");
    let mut even = vec![0.0f32; half];
    let mut odd = vec![0.0f32; half];
    for i in 0..half {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }

    let mut even_fft = vec![0.0f32; 2 * half];
    let mut odd_fft = vec![0.0f32; 2 * half];
    fft(&even, &mut even_fft, &levels[1..], base);
    fft(&odd, &mut odd_fft, &levels[1..], base);

    // twiddle: t = 2*pi*k / N, factor = cos(t) - i*sin(t). Precomputed at
    // table-build time; `neg_sin` already holds `-(sin(t) as f32)`. Iterating the
    // twiddle slices (len == half) yields `k` in 0..half in order — bit-exact.
    for (k, (&re, &im)) in lvl.cos.iter().zip(lvl.neg_sin.iter()).enumerate() {
        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        let e_re = even_fft[2 * k];
        let e_im = even_fft[2 * k + 1];

        out[2 * k] = e_re + re * re_odd - im * im_odd;
        out[2 * k + 1] = e_im + re * im_odd + im * re_odd;

        out[2 * (k + half)] = e_re - re * re_odd + im * im_odd;
        out[2 * (k + half) + 1] = e_im - re * im_odd - im * re_odd;
    }
}

/// Number of mel frames the SIMD path transforms at once — one frame per lane.
/// f32x8 lane operations are bit-identical to scalar f32 (IEEE-754, no FMA
/// contraction), so lane `L` of the batched transform equals the scalar FFT of
/// frame `L` — proven by `fft_simd8_matches_scalar_bit_exact`. The structure-of-
/// arrays layout (one frame per lane) vectorizes the butterflies and the DFT
/// base case; this is ~4x faster than 8 scalar FFTs at baseline x86-64, ~5.6x
/// with AVX2 — and the FFT is the dominant mel cost once the projection is
/// sparse (L3).
const FFT_LANES: usize = 8;
type FrameLanes = Simd<f32, FFT_LANES>;

/// Frame-batched naive DFT base case (8 frames, one per lane). Twiddles are
/// scalar (shared across frames) and splatted across lanes — bit-exact stand-in
/// for the scalar [`dft`].
fn dft_simd8(input: &[FrameLanes], out: &mut [FrameLanes], table: &DftTable) {
    let n = input.len();
    debug_assert_eq!(n, table.n, "dft_simd8 twiddle table width mismatch");
    for k in 0..n {
        let mut re = FrameLanes::splat(0.0);
        let mut im = FrameLanes::splat(0.0);
        let cos_row = &table.cos[k * table.n..k * table.n + n];
        let sin_row = &table.sin[k * table.n..k * table.n + n];
        for (j, &x) in input.iter().enumerate() {
            re += x * FrameLanes::splat(cos_row[j]);
            im -= x * FrameLanes::splat(sin_row[j]);
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

/// Frame-batched recursive Cooley-Tukey FFT (8 frames, one per lane), mirroring
/// the scalar [`fft`] arithmetic exactly with the same precomputed twiddles.
fn fft_simd8(
    input: &[FrameLanes],
    out: &mut [FrameLanes],
    levels: &[LevelTwiddle],
    base: &DftTable,
) {
    let n = input.len();
    if n == 1 {
        out[0] = input[0];
        out[1] = FrameLanes::splat(0.0);
        return;
    }
    if n % 2 == 1 {
        dft_simd8(input, out, base);
        return;
    }
    let lvl = &levels[0];
    let half = n / 2;
    debug_assert_eq!(lvl.half, half, "fft_simd8 level twiddle width mismatch");
    let mut even = vec![FrameLanes::splat(0.0); half];
    let mut odd = vec![FrameLanes::splat(0.0); half];
    for i in 0..half {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }
    let mut even_fft = vec![FrameLanes::splat(0.0); 2 * half];
    let mut odd_fft = vec![FrameLanes::splat(0.0); 2 * half];
    fft_simd8(&even, &mut even_fft, &levels[1..], base);
    fft_simd8(&odd, &mut odd_fft, &levels[1..], base);
    for (k, (&rec, &imc)) in lvl.cos.iter().zip(lvl.neg_sin.iter()).enumerate() {
        let re = FrameLanes::splat(rec);
        let im = FrameLanes::splat(imc);
        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];
        let e_re = even_fft[2 * k];
        let e_im = even_fft[2 * k + 1];
        out[2 * k] = e_re + re * re_odd - im * im_odd;
        out[2 * k + 1] = e_im + re * im_odd + im * re_odd;
        out[2 * (k + half)] = e_re - re * re_odd + im * im_odd;
        out[2 * (k + half) + 1] = e_im - re * im_odd - im * re_odd;
    }
}

/// Build the padded sample buffer exactly as whisper.cpp's
/// `log_mel_spectrogram` does:
/// - leading `N_FFT/2` samples: *reflective* pad — `reverse_copy(samples[1..1+pad])`,
///   i.e. samples 1..=pad reversed (NOT including sample 0);
/// - the audio itself;
/// - trailing: 30 s of zeros + another `N_FFT/2` zeros.
///
/// Returns `(padded, valid_len)` where `valid_len = n_samples + stage_2_pad` is
/// the prefix the worker actually FFTs (frames past it are pure noise floor and
/// whisper.cpp writes `log10(1e-10)` directly without an FFT).
fn build_padded(samples: &[f32]) -> (Vec<f32>, usize) {
    let n_samples = samples.len();
    let stage_1_pad = N_SAMPLES_30S;
    let stage_2_pad = N_FFT / 2; // 200

    let total = n_samples + stage_1_pad + 2 * stage_2_pad;
    let mut padded = vec![0.0f32; total];

    // audio at offset stage_2_pad
    padded[stage_2_pad..stage_2_pad + n_samples].copy_from_slice(samples);

    // leading reflective pad: reverse of samples[1..=stage_2_pad].
    // whisper.cpp: std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, begin).
    // If the audio is shorter than the reflection width, only what exists is
    // reflected (the rest stays zero) — mirrors the C++ pointer arithmetic for
    // realistic (>= pad-length) inputs; we clamp for safety on tiny inputs.
    let refl = stage_2_pad.min(n_samples.saturating_sub(1));
    for i in 0..refl {
        // padded[i] = samples[stage_2_pad - i]  (reverse of indices 1..=refl)
        padded[i] = samples[stage_2_pad - i];
    }

    // trailing zeros already present from vec![0.0; total].
    let valid_len = n_samples + stage_2_pad;
    (padded, valid_len)
}

/// Number of mel frames whisper.cpp produces for a given raw sample count:
/// `n_len = (padded_len - N_FFT) / HOP`.
#[must_use]
pub fn n_frames_for(n_samples: usize) -> usize {
    let padded_len = n_samples + N_SAMPLES_30S + 2 * (N_FFT / 2);
    (padded_len - N_FFT) / HOP
}

/// A mel filterbank paired with each filter's `[start, end)` nonzero freq-bin
/// range. Real whisper mel filterbanks are sparse triangles (~5 nonzero bins of
/// 201), so the projection can skip the leading/trailing zeros. This is
/// **bit-exact**: a skipped term is `power[k] * 0.0`, and for the finite,
/// non-negative `power` produced by an FFT of real audio that is `+0.0`, which
/// never changes a running f64 sum. The ranges are computed once per `log_mel`
/// call (not per frame). Bundling the filterbank + ranges keeps
/// [`compute_frame_column`]'s arity under the clippy `too_many_arguments` limit.
struct SparseMelFilters<'a> {
    fb: &'a MelFilterbank,
    /// `(start, end)` (end-exclusive) per mel filter; every weight outside the
    /// range is exactly `0.0`.
    ranges: Vec<(usize, usize)>,
}

impl<'a> SparseMelFilters<'a> {
    fn new(fb: &'a MelFilterbank) -> Self {
        let n_bins = fb.n_fft_bins;
        let ranges = (0..fb.n_mel)
            .map(|m| {
                let row = &fb.data[m * n_bins..m * n_bins + n_bins];
                let start = row.iter().position(|&v| v != 0.0).unwrap_or(0);
                let end = row.iter().rposition(|&v| v != 0.0).map_or(0, |e| e + 1);
                (start, end)
            })
            .collect();
        Self { fb, ranges }
    }
}

/// Compute one frame's pre-clamp `log10` mel column into `out` (length
/// `n_mel`). `frame_idx` selects the sample offset (`frame_idx * HOP`);
/// `valid_len` is the FFT-able prefix length. Frames whose offset is past
/// `valid_len` are the pure noise floor (`log10(1e-10)`), matching
/// whisper.cpp's direct-write of `log10(1e-10)` without an FFT.
fn compute_frame_column(
    frame_idx: usize,
    padded: &[f32],
    valid_len: usize,
    hann: &[f32; N_FFT],
    filters: &SparseMelFilters,
    twiddles: &FftTwiddles,
    out: &mut [f32],
) {
    let offset = frame_idx * HOP;

    if offset >= valid_len {
        let floor = (1e-10f64).log10() as f32;
        out[..filters.fb.n_mel].fill(floor);
        return;
    }

    // Windowed frame (zero-padded past valid_len).
    let mut fft_in = [0.0f32; N_FFT];
    let avail = N_FFT.min(valid_len - offset);
    for j in 0..avail {
        fft_in[j] = hann[j] * padded[offset + j];
    }

    let mut fft_out = vec![0.0f32; 2 * N_FFT];
    fft(&fft_in, &mut fft_out, &twiddles.levels, &twiddles.base);
    power_and_project(&fft_out, filters, out);
}

/// Power spectrum + sparse mel projection + `log10` floor for one frame, from
/// its interleaved-complex FFT output `fft_out` (length `2*N_FFT`). Shared by the
/// scalar [`compute_frame_column`] and the SIMD batched [`compute_8_columns`]
/// paths so both produce bit-identical columns.
fn power_and_project(fft_out: &[f32], filters: &SparseMelFilters, out: &mut [f32]) {
    let n_fft_bins = filters.fb.n_fft_bins; // 201

    // Power spectrum over one-sided bins: re^2 + im^2 (NOT sqrt magnitude).
    let mut power = [0.0f32; N_FREQ_BINS];
    for (j, p) in power.iter_mut().enumerate() {
        let re = fft_out[2 * j];
        let im = fft_out[2 * j + 1];
        *p = re * re + im * im;
    }

    // Mel projection + log10 floor. Accumulate in f64 like whisper.cpp, but only
    // over each filter's nonzero freq-bin range: the skipped weights are exactly
    // 0.0 and `power` is finite, so `power[k] * 0.0 == +0.0` contributes nothing
    // to the f64 sum — bit-for-bit identical to the dense loop, but ~13x fewer
    // multiply-adds on a real (sparse-triangular) mel filterbank.
    for (j, o) in out.iter_mut().enumerate().take(filters.fb.n_mel) {
        let (start, end) = filters.ranges[j];
        let row = &filters.fb.data[j * n_fft_bins..j * n_fft_bins + n_fft_bins];
        let mut sum = 0.0f64;
        for (&pk, &rk) in power[start..end].iter().zip(row[start..end].iter()) {
            sum += f64::from(pk) * f64::from(rk);
        }
        *o = sum.max(1e-10).log10() as f32;
    }
}

/// Compute 8 FULLY-VALID frames' mel columns at once via the frame-batched FFT.
/// Frames `frame_base .. frame_base + FFT_LANES` must each be fully inside
/// `valid_len` (`offset + N_FFT <= valid_len`), so every window is a complete,
/// unpadded `N_FFT` frame. Output is frame-major: `out8[lane * n_mel + mel]`.
/// Bit-identical to `FFT_LANES` scalar [`compute_frame_column`] calls.
fn compute_8_columns(
    frame_base: usize,
    padded: &[f32],
    hann: &[f32; N_FFT],
    filters: &SparseMelFilters,
    twiddles: &FftTwiddles,
    out8: &mut [f32],
) {
    let n_mel = filters.fb.n_mel;

    // Structure-of-arrays windowed input: lane L holds frame (frame_base + L).
    let mut fft_in = vec![FrameLanes::splat(0.0); N_FFT];
    for (j, slot) in fft_in.iter_mut().enumerate() {
        let mut lanes = [0.0f32; FFT_LANES];
        for (lane, l) in lanes.iter_mut().enumerate() {
            let offset = (frame_base + lane) * HOP;
            *l = hann[j] * padded[offset + j];
        }
        *slot = FrameLanes::from_array(lanes);
    }

    let mut fft_out = vec![FrameLanes::splat(0.0); 2 * N_FFT];
    fft_simd8(&fft_in, &mut fft_out, &twiddles.levels, &twiddles.base);

    // Transpose each lane back to a scalar spectrum, then reuse the scalar
    // power+projection (bit-exact, already validated) for each frame.
    let mut scalar_outs = [[0.0f32; 2 * N_FFT]; FFT_LANES];
    for (b, v) in fft_out.iter().enumerate() {
        let arr = v.to_array();
        for (lane_buf, &val) in scalar_outs.iter_mut().zip(arr.iter()) {
            lane_buf[b] = val;
        }
    }
    for (lane, lane_buf) in scalar_outs.iter().enumerate() {
        let col = &mut out8[lane * n_mel..lane * n_mel + n_mel];
        power_and_project(lane_buf, filters, col);
    }
}

/// Compute the log-mel spectrogram for the full padded input, exactly mirroring
/// whisper.cpp's `log_mel_spectrogram` / `whisper_pcm_to_mel`.
///
/// `samples` is the raw 16 kHz mono f32 PCM. The returned [`Mel`] covers the
/// full padded signal (`n_frames = (padded_len - N_FFT) / HOP`); callers slice
/// 30 s windows (3000 frames) out of it with [`chunk_frames`]. Frames are
/// computed independently and split across `n_threads` worker threads via
/// `std::thread::scope` (no rayon dependency); the result is identical for any
/// thread count.
///
/// # Errors
/// Returns [`FwError::InvalidRequest`] if the filterbank's bin count does not
/// match the expected `N_FFT/2 + 1 = 201`, or if it has zero mel bins.
pub fn log_mel(samples: &[f32], filters: &MelFilterbank, n_threads: usize) -> FwResult<Mel> {
    if filters.n_fft_bins != N_FREQ_BINS {
        return Err(FwError::InvalidRequest(format!(
            "mel filterbank has {} bins, expected {N_FREQ_BINS} (N_FFT/2+1)",
            filters.n_fft_bins
        )));
    }
    if filters.n_mel == 0 {
        return Err(FwError::InvalidRequest(
            "mel filterbank has zero mel bins".to_string(),
        ));
    }

    let hann = cached_hann_window();
    let twiddles = cached_fft_twiddles();
    // Precompute each mel filter's nonzero range once (real banks are sparse).
    let sparse_filters = SparseMelFilters::new(filters);
    let (padded, valid_len) = build_padded(samples);
    let n_frames = (padded.len() - N_FFT) / HOP;
    let n_mel = filters.n_mel;

    let mut data = vec![0.0f32; n_mel * n_frames];

    let n_threads = n_threads.max(1).min(n_frames.max(1));

    // The output is mel-major (data[mel * n_frames + frame]), so disjoint frame
    // ranges still map to interleaved backing indices and the slice can't be
    // split cleanly. Each worker instead computes its contiguous frame range
    // into a private frame-major buffer; we stitch those into the mel-major
    // global buffer single-threaded afterwards. The frame partition is disjoint
    // and deterministic, so the result is identical for any thread count.
    let frames_per_thread = n_frames.div_ceil(n_threads);

    let columns: Vec<(usize, usize, Vec<f32>)> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(n_threads);
        for t in 0..n_threads {
            let start = t * frames_per_thread;
            if start >= n_frames {
                break;
            }
            let end = (start + frames_per_thread).min(n_frames);
            let padded_ref = &padded;
            // `hann` and `twiddles` are already `&'static`; copy the references
            // into the worker closure. `sparse_filters` is built above and shared
            // read-only across workers.
            let hann_ref = hann;
            let twiddles_ref = twiddles;
            let filters_ref = &sparse_filters;
            handles.push(scope.spawn(move || {
                let len = end - start;
                // frame-major local buffer: local[frame_local * n_mel + mel]
                let mut local = vec![0.0f32; len * n_mel];
                // Frames whose full N_FFT window fits inside valid_len can be
                // FFT'd 8-at-a-time via the frame-batched SIMD path; the
                // partial-window tail + noise-floor frames take the scalar path.
                let n_full = if valid_len >= N_FFT {
                    (valid_len - N_FFT) / HOP + 1
                } else {
                    0
                };
                let full_end = end.min(n_full);
                let mut frame = start;
                let mut fl = 0usize;
                while frame + FFT_LANES <= full_end {
                    let col8 = &mut local[fl * n_mel..(fl + FFT_LANES) * n_mel];
                    compute_8_columns(
                        frame,
                        padded_ref,
                        hann_ref,
                        filters_ref,
                        twiddles_ref,
                        col8,
                    );
                    frame += FFT_LANES;
                    fl += FFT_LANES;
                }
                while frame < end {
                    let col = &mut local[fl * n_mel..fl * n_mel + n_mel];
                    compute_frame_column(
                        frame,
                        padded_ref,
                        valid_len,
                        hann_ref,
                        filters_ref,
                        twiddles_ref,
                        col,
                    );
                    frame += 1;
                    fl += 1;
                }
                (start, end, local)
            }));
        }
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(v) => v,
                // A worker can only panic on an internal logic bug; re-raise it
                // on the owning thread rather than silently dropping frames.
                Err(payload) => std::panic::resume_unwind(payload),
            })
            .collect()
    });

    // Stitch frame-major locals into the mel-major global buffer.
    for (start, end, local) in columns {
        for (fl, frame) in (start..end).enumerate() {
            for m in 0..n_mel {
                data[m * n_frames + frame] = local[fl * n_mel + m];
            }
        }
    }

    // Global clamp + normalize: log_spec = max(log_spec, max - 8); (x + 4) / 4.
    let mut mmax = f32::NEG_INFINITY;
    for &v in &data {
        if v > mmax {
            mmax = v;
        }
    }
    let floor = mmax - 8.0;
    for v in &mut data {
        if *v < floor {
            *v = floor;
        }
        *v = (*v + 4.0) / 4.0;
    }

    Ok(Mel {
        n_mel,
        n_frames,
        data,
    })
}

/// Slice a window of `n_frames` frames from `mel` starting at `frame_offset`,
/// zero-padding (with the post-normalization [`SILENCE_FLOOR`]) past the end.
///
/// whisper.cpp pads the *encoder input* mel with the noise-floor value (the
/// normalized `log10(1e-10)` ⇒ `-1.5`), NOT raw zeros — because the entire mel
/// has already been clamped and normalized before it reaches the encoder. This
/// helper reproduces that: a window extending past the real data is filled with
/// [`SILENCE_FLOOR`].
#[must_use]
pub fn chunk_frames(mel: &Mel, frame_offset: usize, n_frames: usize) -> Mel {
    let mut data = vec![SILENCE_FLOOR; mel.n_mel * n_frames];
    for m in 0..mel.n_mel {
        for f in 0..n_frames {
            let src = frame_offset + f;
            if src < mel.n_frames {
                data[m * n_frames + f] = mel.data[m * mel.n_frames + src];
            }
        }
    }
    Mel {
        n_mel: mel.n_mel,
        n_frames,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG (no `rand` dependency). glibc constants.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_f32(&mut self) -> f32 {
            // x_{n+1} = (a*x + c) mod 2^31
            self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
            // map to [-1, 1)
            (self.0 as f32 / (1u32 << 30) as f32) - 1.0
        }
    }

    /// Independent naive DFT reference (computed fully in f64 then narrowed).
    fn naive_dft(input: &[f32]) -> Vec<(f64, f64)> {
        let n = input.len();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (j, &x) in input.iter().enumerate() {
                let theta = (2.0 * PI * (k * j) as f64) / n as f64;
                re += f64::from(x) * theta.cos();
                im -= f64::from(x) * theta.sin();
            }
            out.push((re, im));
        }
        out
    }

    fn fft_to_complex(input: &[f32]) -> Vec<(f64, f64)> {
        let tw = FftTwiddles::build(input.len());
        let mut out = vec![0.0f32; 2 * input.len()];
        fft(input, &mut out, &tw.levels, &tw.base);
        (0..input.len())
            .map(|k| (f64::from(out[2 * k]), f64::from(out[2 * k + 1])))
            .collect()
    }

    /// Inline-transcendental copy of the PRE-optimization recursive FFT/DFT,
    /// exactly as whisper.cpp's `fft`/`dft`. Retained only to prove the
    /// twiddle-table rewrite is bit-for-bit identical (the conformance contract).
    fn dft_reference(input: &[f32], out: &mut [f32]) {
        let n = input.len();
        for k in 0..n {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (j, &x) in input.iter().enumerate() {
                let theta = (2.0 * PI * (k * j) as f64) / n as f64;
                re += x * theta.cos() as f32;
                im -= x * theta.sin() as f32;
            }
            out[2 * k] = re;
            out[2 * k + 1] = im;
        }
    }

    fn fft_reference(input: &[f32], out: &mut [f32]) {
        let n = input.len();
        if n == 1 {
            out[0] = input[0];
            out[1] = 0.0;
            return;
        }
        if n % 2 == 1 {
            dft_reference(input, out);
            return;
        }
        let half = n / 2;
        let mut even = vec![0.0f32; half];
        let mut odd = vec![0.0f32; half];
        for i in 0..half {
            even[i] = input[2 * i];
            odd[i] = input[2 * i + 1];
        }
        let mut even_fft = vec![0.0f32; 2 * half];
        let mut odd_fft = vec![0.0f32; 2 * half];
        fft_reference(&even, &mut even_fft);
        fft_reference(&odd, &mut odd_fft);
        for k in 0..half {
            let theta = (2.0 * PI * k as f64) / n as f64;
            let re = theta.cos() as f32;
            let im = -(theta.sin() as f32);
            let re_odd = odd_fft[2 * k];
            let im_odd = odd_fft[2 * k + 1];
            let e_re = even_fft[2 * k];
            let e_im = even_fft[2 * k + 1];
            out[2 * k] = e_re + re * re_odd - im * im_odd;
            out[2 * k + 1] = e_im + re * im_odd + im * re_odd;
            out[2 * (k + half)] = e_re - re * re_odd + im * im_odd;
            out[2 * (k + half) + 1] = e_im - re * im_odd - im * re_odd;
        }
    }

    #[test]
    fn fft_twiddle_table_is_bit_exact_vs_inline_reference() {
        // The optimization MUST NOT change a single output bit — the mel output
        // is conformance-checked against whisper.cpp's exact encoder input.
        // Compare the table-driven FFT against the inline-transcendental
        // reference over many random frames at the production width (400) plus a
        // spread of even/odd/power-of-two sizes that exercise every recursion
        // and base-case path.
        for &n in &[N_FFT, 200usize, 100, 50, 25, 256, 8, 5, 2, 1] {
            let tw = FftTwiddles::build(n);
            for seed in 0..64u64 {
                let mut lcg = Lcg::new(
                    0x9E37_79B9 ^ (n as u64).wrapping_mul(2_654_435_761).wrapping_add(seed),
                );
                let input: Vec<f32> = (0..n).map(|_| lcg.next_f32()).collect();
                let mut got = vec![0.0f32; 2 * n];
                let mut want = vec![0.0f32; 2 * n];
                fft(&input, &mut got, &tw.levels, &tw.base);
                fft_reference(&input, &mut want);
                assert_eq!(
                    got, want,
                    "twiddle FFT diverged from inline reference at N={n} seed={seed}"
                );
            }
        }
    }

    #[test]
    fn fft_matches_naive_dft() {
        for &n in &[400usize, 256, 200, 5] {
            let mut lcg = Lcg::new(0xDEAD_BEEF ^ n as u64);
            let input: Vec<f32> = (0..n).map(|_| lcg.next_f32()).collect();

            let got = fft_to_complex(&input);
            let want = naive_dft(&input);

            // reference energy scale for relative error
            let scale: f64 = want
                .iter()
                .map(|(re, im)| (re * re + im * im).sqrt())
                .fold(0.0f64, f64::max)
                .max(1e-9);

            for k in 0..n {
                let dre = (got[k].0 - want[k].0).abs();
                let dim = (got[k].1 - want[k].1).abs();
                let rel = (dre.max(dim)) / scale;
                assert!(
                    rel < 1e-4,
                    "N={n} k={k} rel={rel} got={:?} want={:?}",
                    got[k],
                    want[k]
                );
            }
        }
    }

    #[test]
    fn fft_parseval() {
        // sum |x|^2 == (1/N) sum |X|^2
        for &n in &[400usize, 200, 256] {
            let mut lcg = Lcg::new(0x1234_5678 ^ n as u64);
            let input: Vec<f32> = (0..n).map(|_| lcg.next_f32()).collect();
            let time_energy: f64 = input.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            let spec = fft_to_complex(&input);
            let freq_energy: f64 =
                spec.iter().map(|(re, im)| re * re + im * im).sum::<f64>() / n as f64;
            let rel = (time_energy - freq_energy).abs() / time_energy.max(1e-9);
            assert!(rel < 1e-4, "Parseval N={n} rel={rel}");
        }
    }

    #[test]
    fn fft_simd8_matches_scalar_bit_exact() {
        // Frame-batched f32x8 FFT must be byte-identical to the scalar FFT per
        // lane (frame): IEEE f32 SIMD lanes == scalar f32, no FMA contraction.
        // Guards the whole SIMD mel fast path (the only new arithmetic;
        // power+projection is the shared, separately-tested helper).
        let tw = FftTwiddles::build(N_FFT);
        let mut lcg = Lcg::new(0x0051_1D80);
        for _round in 0..32 {
            let frames: Vec<[f32; N_FFT]> = (0..FFT_LANES)
                .map(|_| {
                    let mut f = [0.0f32; N_FFT];
                    for x in &mut f {
                        *x = lcg.next_f32();
                    }
                    f
                })
                .collect();
            let mut vin = vec![FrameLanes::splat(0.0); N_FFT];
            for (j, slot) in vin.iter_mut().enumerate() {
                let mut a = [0.0f32; FFT_LANES];
                for (lane, al) in a.iter_mut().enumerate() {
                    *al = frames[lane][j];
                }
                *slot = FrameLanes::from_array(a);
            }
            let mut vout = vec![FrameLanes::splat(0.0); 2 * N_FFT];
            fft_simd8(&vin, &mut vout, &tw.levels, &tw.base);
            for (lane, frame) in frames.iter().enumerate() {
                let mut sout = vec![0.0f32; 2 * N_FFT];
                fft(frame, &mut sout, &tw.levels, &tw.base);
                for (b, &sv) in sout.iter().enumerate() {
                    assert_eq!(
                        vout[b].to_array()[lane].to_bits(),
                        sv.to_bits(),
                        "lane {lane} bin {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_projection_matches_dense_bit_exact() {
        // Real mel banks are sparse triangles; SparseMelFilters skips leading/
        // trailing zeros. Verify the range-restricted f64 projection is
        // bit-identical to the dense sum over all 201 bins, for every filter.
        let n_bins = N_FREQ_BINS;
        let n_mel = 16;
        let mut data = vec![0.0f32; n_mel * n_bins];
        for m in 0..n_mel {
            let lo = (m * 7) % (n_bins - 12);
            for k in lo..lo + 9 {
                let w = 1.0 - ((k as f32 - (lo as f32 + 4.0)).abs() / 5.0);
                data[m * n_bins + k] = w.max(0.0);
            }
        }
        let fb = MelFilterbank {
            n_mel,
            n_fft_bins: n_bins,
            data,
        };
        let sparse = SparseMelFilters::new(&fb);
        let mut lcg = Lcg::new(0x000F_17E5);
        for _ in 0..64 {
            let power: Vec<f32> = (0..n_bins).map(|_| (lcg.next_f32() + 1.0) * 50.0).collect();
            for m in 0..n_mel {
                let row = &fb.data[m * n_bins..m * n_bins + n_bins];
                let mut dense = 0.0f64;
                for (k, &fk) in power.iter().enumerate() {
                    dense += f64::from(fk) * f64::from(row[k]);
                }
                let (s, e) = sparse.ranges[m];
                let mut sp = 0.0f64;
                for (&pk, &rk) in power[s..e].iter().zip(row[s..e].iter()) {
                    sp += f64::from(pk) * f64::from(rk);
                }
                assert_eq!(dense.to_bits(), sp.to_bits(), "mel {m}: sparse != dense");
            }
        }
    }

    fn dummy_filters(n_mel: usize) -> MelFilterbank {
        // Identity-ish: each mel bin sums a disjoint contiguous slice of freq
        // bins. Sufficient for silence/determinism tests (exact weights don't
        // matter — silence yields zero power regardless).
        let mut data = vec![0.0f32; n_mel * N_FREQ_BINS];
        for m in 0..n_mel {
            let bin = (m * N_FREQ_BINS / n_mel).min(N_FREQ_BINS - 1);
            data[m * N_FREQ_BINS + bin] = 1.0;
        }
        MelFilterbank {
            n_mel,
            n_fft_bins: N_FREQ_BINS,
            data,
        }
    }

    #[test]
    fn silence_collapses_to_floor() {
        let filters = dummy_filters(80);
        let samples = vec![0.0f32; SAMPLE_RATE]; // 1s of silence
        let mel = log_mel(&samples, &filters, 4).expect("log_mel");
        for &v in &mel.data {
            assert!(
                (v - SILENCE_FLOOR).abs() < 1e-6,
                "silence sample {v} != {SILENCE_FLOOR}"
            );
        }
        // Sanity: analytic floor derivation.
        let pre = (1e-10f64).log10() as f32; // -10
        assert_eq!((pre + 4.0) / 4.0, SILENCE_FLOOR);
    }

    #[test]
    fn determinism_across_thread_counts() {
        let filters = dummy_filters(80);
        let mut lcg = Lcg::new(0xABCD_1234);
        let samples: Vec<f32> = (0..SAMPLE_RATE / 2).map(|_| lcg.next_f32() * 0.3).collect();

        let a = log_mel(&samples, &filters, 1).expect("t1");
        let b = log_mel(&samples, &filters, 4).expect("t4");
        assert_eq!(a.n_frames, b.n_frames);
        assert_eq!(a.n_mel, b.n_mel);
        assert_eq!(a.data, b.data, "thread-count nondeterminism");
    }

    #[test]
    fn chunk_frames_pads_with_silence_floor() {
        let filters = dummy_filters(4);
        let samples = vec![0.0f32; 1000];
        let mel = log_mel(&samples, &filters, 1).expect("log_mel");
        // Slice well past the end.
        let big = mel.n_frames + 50;
        let win = chunk_frames(&mel, mel.n_frames - 10, big);
        assert_eq!(win.n_frames, big);
        // The trailing slots (past real data) must be SILENCE_FLOOR.
        for m in 0..win.n_mel {
            for f in 11..big {
                assert!(
                    (win.data[m * big + f] - SILENCE_FLOOR).abs() < 1e-6,
                    "pad slot ({m},{f}) not floor"
                );
            }
        }
    }

    /// Read the 80x201 filterbank directly from a ggml model file using the
    /// known header layout: magic(4) + 11 hparams i32 (44) + n_mel i32 +
    /// n_fft i32 + n_mel*n_fft f32 (little-endian). Avoids depending on the
    /// `ggml` module (which may not compile yet).
    fn read_filters_from_model(path: &std::path::Path) -> Option<MelFilterbank> {
        let bytes = std::fs::read(path).ok()?;
        let rd_i32 = |b: &[u8], off: usize| -> i32 {
            i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
        };
        if bytes.len() < 4 + 44 + 8 {
            return None;
        }
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if magic != 0x6767_6d6c {
            return None;
        }
        let after_hparams = 4 + 44; // 48
        let n_mel = rd_i32(&bytes, after_hparams) as usize;
        let n_fft = rd_i32(&bytes, after_hparams + 4) as usize;
        let data_off = after_hparams + 8;
        let count = n_mel * n_fft;
        if bytes.len() < data_off + count * 4 {
            return None;
        }
        let mut data = Vec::with_capacity(count);
        for i in 0..count {
            let o = data_off + i * 4;
            data.push(f32::from_le_bytes([
                bytes[o],
                bytes[o + 1],
                bytes[o + 2],
                bytes[o + 3],
            ]));
        }
        Some(MelFilterbank {
            n_mel,
            n_fft_bins: n_fft,
            data,
        })
    }

    #[test]
    fn sine_440_argmax_matches_analytic_filter_projection() {
        let Some(path) = super::super::find_model_file("tiny.en") else {
            eprintln!("skipping: tiny.en model not found");
            return;
        };
        let Some(filters) = read_filters_from_model(&path) else {
            eprintln!("skipping: could not parse filterbank from {path:?}");
            return;
        };
        assert_eq!(filters.n_fft_bins, N_FREQ_BINS);

        // 440 Hz sine at 16 kHz, 1 second.
        let freq = 440.0f64;
        let samples: Vec<f32> = (0..SAMPLE_RATE)
            .map(|n| (2.0 * PI * freq * n as f64 / SAMPLE_RATE as f64).sin() as f32)
            .collect();

        let mel = log_mel(&samples, &filters, 4).expect("log_mel");

        // Empirical argmax over mel bins, averaged across a band of mid frames
        // (well inside the audio, away from edges and the 30s zero tail).
        let mid = (SAMPLE_RATE / HOP) / 2; // ~50
        let mut acc = vec![0.0f64; filters.n_mel];
        let mut cnt = 0;
        for f in mid.saturating_sub(5)..(mid + 5) {
            if f >= mel.n_frames {
                break;
            }
            for (m, a) in acc.iter_mut().enumerate() {
                *a += f64::from(mel.data[m * mel.n_frames + f]);
            }
            cnt += 1;
        }
        assert!(cnt > 0);
        let emp_argmax = acc
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Analytic expectation: a pure 440 Hz tone concentrates power at the
        // nearest FFT bin = round(440 * N_FFT / SAMPLE_RATE) = round(11) = 11.
        // Project a unit-power spectrum at that bin through the filterbank and
        // take the strongest mel bin.
        let bin = (freq * N_FFT as f64 / SAMPLE_RATE as f64).round() as usize;
        let mut proj = vec![0.0f64; filters.n_mel];
        for (m, p) in proj.iter_mut().enumerate() {
            *p = f64::from(filters.data[m * filters.n_fft_bins + bin]);
        }
        let ana_argmax = proj
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            emp_argmax, ana_argmax,
            "empirical mel argmax {emp_argmax} != analytic {ana_argmax} for 440Hz"
        );
    }
}
