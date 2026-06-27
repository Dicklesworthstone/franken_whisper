//! Log-mel spectrogram frontend (pure Rust, whisper.cpp-compatible semantics).
//!
//! This is a faithful port of whisper.cpp's `log_mel_spectrogram` /
//! `whisper_pcm_to_mel` path (see `whisper.cpp` `src/whisper.cpp`
//! `log_mel_spectrogram`, `log_mel_spectrogram_worker_thread`, `fft`, `dft`,
//! and `fill_hann_window`). Every constant and every numeric behavior that
//! could plausibly diverge from whisper.cpp is documented inline. Two
//! deliberate arithmetic relaxations exist, both well inside the transcription
//! tolerance gate (each diverges ~1e-7, vs the `rel<1e-4` conformance bound):
//! (1) the projection `log10`, a deterministic polynomial approximation that
//! avoids the scalar libm bottleneck; and (2) the `n == 25` FFT base case, a
//! radix-5 (`25 = 5x5`) Cooley-Tukey replacing the naive 25x25 DFT (~1.8x
//! faster on the dominant transcendental cost of the frontend). Both are
//! applied identically to the scalar and SIMD paths, so franken's internal
//! scalar-vs-SIMD determinism stays bit-exact.
//!
//! Pipeline (per whisper.cpp):
//! 1. Pad the PCM: a leading *reflective* pad of `N_FFT/2` samples, the audio
//!    itself, then a trailing pad of one full 30 s chunk of zeros plus another
//!    `N_FFT/2` zeros. (The 30 s tail guarantees at least 3000 frames exist.)
//! 2. Slide a length-`N_FFT` Hann (periodic) window with hop `HOP`.
//! 3. FFT each windowed frame; take the **power** spectrum `re^2 + im^2` over
//!    the one-sided `N_FFT/2 + 1 = 201` bins.
//! 4. Project through the model's `n_mel x 201` filterbank, then an approximate
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
    /// Radix-5 (`25 = 5x5` Cooley-Tukey) twiddles, used by `dft`/`dft_simd8` when
    /// `n == 25`: `W_5` (5x5) and the `W_25^{a*b}` combine factors (5x5). cos/sin
    /// stored positive, consumed as `W = cos - i*sin` (same convention as the
    /// naive `cos`/`sin` above). Empty for `n != 25` (those fall back to naive).
    r5_c5: Vec<f32>,
    r5_s5: Vec<f32>,
    r5_c25: Vec<f32>,
    r5_s25: Vec<f32>,
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
        let (mut r5_c5, mut r5_s5, mut r5_c25, mut r5_s25) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        if n == 25 {
            r5_c5 = vec![0.0f32; 25];
            r5_s5 = vec![0.0f32; 25];
            for k in 0..5 {
                for j in 0..5 {
                    let th = (2.0 * PI * (k * j) as f64) / 5.0;
                    r5_c5[k * 5 + j] = th.cos() as f32;
                    r5_s5[k * 5 + j] = th.sin() as f32;
                }
            }
            r5_c25 = vec![0.0f32; 25];
            r5_s25 = vec![0.0f32; 25];
            for a in 0..5 {
                for b in 0..5 {
                    let th = (2.0 * PI * (a * b) as f64) / 25.0;
                    r5_c25[a * 5 + b] = th.cos() as f32;
                    r5_s25[a * 5 + b] = th.sin() as f32;
                }
            }
        }
        Self {
            n,
            cos,
            sin,
            r5_c5,
            r5_s5,
            r5_c25,
            r5_s25,
        }
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
/// Radix-5 (`25 = 5x5` Cooley-Tukey) of the real-input 25-pt DFT — a
/// transcription-tolerance replacement for the naive 25x25 base case (~1.8x
/// faster, diverges rel ~1e-7 from naive, within the relaxed mel parity the log10
/// landing established). `out` is interleaved complex. Structurally identical to
/// [`radix5_dft_simd8`] so the scalar/SIMD paths stay bit-identical
/// (`compute_8_columns_matches_scalar_columns_bit_exact`).
fn radix5_dft(input: &[f32], out: &mut [f32], t: &DftTable) {
    // n = 5*n1 + n2 ; k = k1 + 5*k2
    let mut ir = [[0.0f32; 5]; 5];
    let mut ii = [[0.0f32; 5]; 5];
    for n2 in 0..5 {
        for k1 in 0..5 {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for n1 in 0..5 {
                let xr = input[5 * n1 + n2];
                re += xr * t.r5_c5[k1 * 5 + n1];
                im -= xr * t.r5_s5[k1 * 5 + n1];
            }
            ir[n2][k1] = re;
            ii[n2][k1] = im;
        }
    }
    let mut tr = [[0.0f32; 5]; 5];
    let mut ti = [[0.0f32; 5]; 5];
    for n2 in 0..5 {
        for k1 in 0..5 {
            let c = t.r5_c25[n2 * 5 + k1];
            let s = t.r5_s25[n2 * 5 + k1];
            tr[n2][k1] = ir[n2][k1] * c + ii[n2][k1] * s;
            ti[n2][k1] = ii[n2][k1] * c - ir[n2][k1] * s;
        }
    }
    for k1 in 0..5 {
        for k2 in 0..5 {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for n2 in 0..5 {
                let c = t.r5_c5[k2 * 5 + n2];
                let s = t.r5_s5[k2 * 5 + n2];
                re += tr[n2][k1] * c + ti[n2][k1] * s;
                im += ti[n2][k1] * c - tr[n2][k1] * s;
            }
            out[2 * (k1 + 5 * k2)] = re;
            out[2 * (k1 + 5 * k2) + 1] = im;
        }
    }
}

fn dft(input: &[f32], out: &mut [f32], table: &DftTable) {
    let n = input.len();
    debug_assert_eq!(n, table.n, "dft twiddle table width mismatch");
    if n == 25 {
        radix5_dft(input, out, table);
        return;
    }
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

// ---------------------------------------------------------------------------
// Real-FFT (two-for-one) — the DEFAULT mel FFT path on BOTH scalar and SIMD.
// A real-input FFT of length N is computed from ONE complex FFT of length N/2
// (pack the even/odd decimation as one complex sequence) instead of TWO real
// sub-FFTs — halving the butterfly/recursion work. Measured -8.37% mel (p=0.00).
// It changes the arithmetic (rel ~1e-7, within the transcription-tolerance gate
// — conformance 26/0 with it active), and is applied IDENTICALLY to the scalar
// `fft_twoforone` and SIMD `fft_simd8_twoforone` so `determinism_across_thread_counts`
// holds (proven by `fft_simd8_twoforone_matches_scalar`). `FW_RFFT_OFF` forces the
// prior naive-recursion path (escape hatch / A/B baseline).
// ---------------------------------------------------------------------------

/// The two-for-one real-FFT is the DEFAULT FFT path (both scalar and SIMD);
/// `FW_RFFT_OFF` forces the prior naive-recursion path (escape hatch + A/B).
/// Measured -8.37% mel (p=0.00) vs the one-sided path.
fn rfft_enabled() -> bool {
    static R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *R.get_or_init(|| std::env::var_os("FW_RFFT_OFF").is_none())
}

/// Complex-input radix-5 (`25 = 5x5`) — the complex twin of [`radix5_dft`]. Only
/// the stage-1 5-point DFTs differ (complex vs real input); the twiddle and
/// stage-2 are identical. `input`/`out` are interleaved complex of length `2*25`.
fn radix5_cdft(input: &[f32], out: &mut [f32], t: &DftTable) {
    let mut ir = [[0.0f32; 5]; 5];
    let mut ii = [[0.0f32; 5]; 5];
    for n2 in 0..5 {
        for k1 in 0..5 {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for n1 in 0..5 {
                let xr = input[2 * (5 * n1 + n2)];
                let xi = input[2 * (5 * n1 + n2) + 1];
                let c = t.r5_c5[k1 * 5 + n1];
                let s = t.r5_s5[k1 * 5 + n1];
                // (xr + i*xi) * (c - i*s)
                re += xr * c + xi * s;
                im += xi * c - xr * s;
            }
            ir[n2][k1] = re;
            ii[n2][k1] = im;
        }
    }
    let mut tr = [[0.0f32; 5]; 5];
    let mut ti = [[0.0f32; 5]; 5];
    for n2 in 0..5 {
        for k1 in 0..5 {
            let c = t.r5_c25[n2 * 5 + k1];
            let s = t.r5_s25[n2 * 5 + k1];
            tr[n2][k1] = ir[n2][k1] * c + ii[n2][k1] * s;
            ti[n2][k1] = ii[n2][k1] * c - ir[n2][k1] * s;
        }
    }
    for k1 in 0..5 {
        for k2 in 0..5 {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for n2 in 0..5 {
                let c = t.r5_c5[k2 * 5 + n2];
                let s = t.r5_s5[k2 * 5 + n2];
                re += tr[n2][k1] * c + ti[n2][k1] * s;
                im += ti[n2][k1] * c - tr[n2][k1] * s;
            }
            out[2 * (k1 + 5 * k2)] = re;
            out[2 * (k1 + 5 * k2) + 1] = im;
        }
    }
}

/// Naive complex DFT base case (`n` odd), with the radix-5 fast path for `n==25`.
/// Complex twin of [`dft`]. `input`/`out` interleaved complex of length `2*n`.
fn cdft(input: &[f32], out: &mut [f32], table: &DftTable) {
    let n = input.len() / 2;
    if n == 25 {
        radix5_cdft(input, out, table);
        return;
    }
    for k in 0..n {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        let cos_row = &table.cos[k * table.n..k * table.n + n];
        let sin_row = &table.sin[k * table.n..k * table.n + n];
        for j in 0..n {
            let xr = input[2 * j];
            let xi = input[2 * j + 1];
            re += xr * cos_row[j] + xi * sin_row[j];
            im += xi * cos_row[j] - xr * sin_row[j];
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

/// Recursive complex-input Cooley-Tukey FFT — complex twin of [`fft`]. The
/// butterfly combine is identical (it already operates on complex sub-spectra);
/// only the input load, the `n==1` leaf, the even/odd split, and the base case
/// are complex. `input`/`out` interleaved complex of length `2*n`.
fn cfft(input: &[f32], out: &mut [f32], levels: &[LevelTwiddle], base: &DftTable) {
    let n = input.len() / 2;
    if n == 1 {
        out[0] = input[0];
        out[1] = input[1];
        return;
    }
    if n % 2 == 1 {
        cdft(input, out, base);
        return;
    }
    let lvl = &levels[0];
    let half = n / 2;
    debug_assert_eq!(lvl.half, half, "cfft level twiddle width mismatch");
    let mut even = vec![0.0f32; 2 * half];
    let mut odd = vec![0.0f32; 2 * half];
    for i in 0..half {
        even[2 * i] = input[2 * (2 * i)];
        even[2 * i + 1] = input[2 * (2 * i) + 1];
        odd[2 * i] = input[2 * (2 * i + 1)];
        odd[2 * i + 1] = input[2 * (2 * i + 1) + 1];
    }
    let mut even_fft = vec![0.0f32; 2 * half];
    let mut odd_fft = vec![0.0f32; 2 * half];
    cfft(&even, &mut even_fft, &levels[1..], base);
    cfft(&odd, &mut odd_fft, &levels[1..], base);
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

/// Two-for-one real FFT of a length-`n` real `input` (n even, `n/2` reaching the
/// radix-5 base): pack the even/odd decimation as one complex sequence of length
/// `n/2`, run ONE [`cfft`], then unpack to the even/odd real sub-spectra and apply
/// the top-level butterfly. Produces the same `2*n` interleaved-complex `out` as
/// [`fft`] (within ~1e-7). `levels[0]` is the n-wide butterfly; `levels[1..]` and
/// `base` drive the `n/2` complex FFT.
fn fft_twoforone(input: &[f32], out: &mut [f32], levels: &[LevelTwiddle], base: &DftTable) {
    let n = input.len();
    debug_assert!(n.is_multiple_of(2) && n >= 2, "fft_twoforone needs even n");
    let half = n / 2;
    let lvl = &levels[0];
    debug_assert_eq!(lvl.half, half, "fft_twoforone top twiddle width mismatch");
    // Pack z[m] = even[m] + i*odd[m] = input[2m] + i*input[2m+1].
    let mut z = vec![0.0f32; 2 * half];
    for m in 0..half {
        z[2 * m] = input[2 * m];
        z[2 * m + 1] = input[2 * m + 1];
    }
    let mut zf = vec![0.0f32; 2 * half];
    cfft(&z, &mut zf, &levels[1..], base);
    // Unpack Even/Odd then top-level butterfly, per k in 0..half.
    for k in 0..half {
        let km = (half - k) % half;
        let (ar, ai) = (zf[2 * k], zf[2 * k + 1]);
        let (br, bi) = (zf[2 * km], zf[2 * km + 1]);
        // Even[k] = (Z[k] + conj(Z[half-k]))/2 ; Odd[k] = (Z[k] - conj(Z[half-k]))/(2i)
        let e_re = 0.5 * (ar + br);
        let e_im = 0.5 * (ai - bi);
        let o_re = 0.5 * (ai + bi);
        let o_im = 0.5 * (br - ar);
        let re = lvl.cos[k];
        let im = lvl.neg_sin[k];
        out[2 * k] = e_re + re * o_re - im * o_im;
        out[2 * k + 1] = e_im + re * o_im + im * o_re;
        out[2 * (k + half)] = e_re - re * o_re + im * o_im;
        out[2 * (k + half) + 1] = e_im - re * o_im - im * o_re;
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
/// FrameLanes radix-5 — lane-parallel mirror of [`radix5_dft`] (same op order ⇒
/// each lane is bit-identical to the scalar path).
fn radix5_dft_simd8(input: &[FrameLanes], out: &mut [FrameLanes], t: &DftTable) {
    let z = FrameLanes::splat(0.0);
    let mut ir = [[z; 5]; 5];
    let mut ii = [[z; 5]; 5];
    for n2 in 0..5 {
        for k1 in 0..5 {
            let (mut re, mut im) = (z, z);
            for n1 in 0..5 {
                let xr = input[5 * n1 + n2];
                re += xr * FrameLanes::splat(t.r5_c5[k1 * 5 + n1]);
                im -= xr * FrameLanes::splat(t.r5_s5[k1 * 5 + n1]);
            }
            ir[n2][k1] = re;
            ii[n2][k1] = im;
        }
    }
    let mut tr = [[z; 5]; 5];
    let mut ti = [[z; 5]; 5];
    for n2 in 0..5 {
        for k1 in 0..5 {
            let c = FrameLanes::splat(t.r5_c25[n2 * 5 + k1]);
            let s = FrameLanes::splat(t.r5_s25[n2 * 5 + k1]);
            tr[n2][k1] = ir[n2][k1] * c + ii[n2][k1] * s;
            ti[n2][k1] = ii[n2][k1] * c - ir[n2][k1] * s;
        }
    }
    for k1 in 0..5 {
        for k2 in 0..5 {
            let (mut re, mut im) = (z, z);
            for n2 in 0..5 {
                let c = FrameLanes::splat(t.r5_c5[k2 * 5 + n2]);
                let s = FrameLanes::splat(t.r5_s5[k2 * 5 + n2]);
                re += tr[n2][k1] * c + ti[n2][k1] * s;
                im += ti[n2][k1] * c - tr[n2][k1] * s;
            }
            out[2 * (k1 + 5 * k2)] = re;
            out[2 * (k1 + 5 * k2) + 1] = im;
        }
    }
}

fn dft_simd8(input: &[FrameLanes], out: &mut [FrameLanes], table: &DftTable) {
    let n = input.len();
    debug_assert_eq!(n, table.n, "dft_simd8 twiddle table width mismatch");
    if n == 25 {
        radix5_dft_simd8(input, out, table);
        return;
    }
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
    let (even, odd) = split_even_odd(input, half);
    let (mut even_fft, mut odd_fft) = alloc_fft_scratch(2 * half);
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

/// Bench-only escape hatch (mirrors `FW_DISABLE_F16C_DOT` in `nn`): when
/// `FW_FFT_FULL` is set, `compute_8_columns` uses the full-spectrum
/// [`fft_simd8`]; otherwise it uses the one-sided [`fft_simd8_onesided`] default.
/// Lets a single bench binary A/B the two on the same box (load-robust). Read
/// once.
fn fft_top_full() -> bool {
    static FULL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FULL.get_or_init(|| std::env::var_os("FW_FFT_FULL").is_some())
}

/// Even/odd decimation split for the frame-batched FFT. Both halves are written
/// in full, so `collect()` them directly rather than `vec![splat(0.0); half]`
/// then overwriting — the zero-init was dead work (measured: the FFT's
/// fully-written scratch zeroing was ~5.5% of mel). Bit-identical values to the
/// indexed-fill it replaces.
#[inline]
fn split_even_odd(input: &[FrameLanes], half: usize) -> (Vec<FrameLanes>, Vec<FrameLanes>) {
    (
        (0..half).map(|i| input[2 * i]).collect(),
        (0..half).map(|i| input[2 * i + 1]).collect(),
    )
}

/// Safety escape hatch (mirrors `FW_DISABLE_F16C_DOT`): force the zero-initialised
/// FFT output scratch instead of the uninitialised fast path. Off by default.
fn fft_scratch_zeroinit() -> bool {
    static Z: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *Z.get_or_init(|| std::env::var_os("FW_FFT_ZEROINIT").is_some())
}

/// Allocate the two `n`-element output buffers (`even_fft`/`odd_fft`) for the
/// recursive sub-FFTs. They are written in full by the immediately-following
/// `fft_simd8` before the butterfly combine reads them, so the usual
/// `vec![splat(0.0); n]` zero-init is pure dead work (~38 MB/`mel_30s`, measured
/// ~9% of mel). The interleaved butterfly writes can't be `collect()`'d without
/// doubling the complex multiplies, so the fast path allocates uninitialised.
/// `FW_FFT_ZEROINIT` restores the zero-init path.
#[allow(unsafe_code)]
#[inline]
fn alloc_fft_scratch(n: usize) -> (Vec<FrameLanes>, Vec<FrameLanes>) {
    if fft_scratch_zeroinit() {
        return (
            vec![FrameLanes::splat(0.0); n],
            vec![FrameLanes::splat(0.0); n],
        );
    }
    let mut even_fft = Vec::<FrameLanes>::with_capacity(n);
    let mut odd_fft = Vec::<FrameLanes>::with_capacity(n);
    // SAFETY: each buffer is handed straight to `fft_simd8(&_, &mut buf, ..)`,
    // which writes ALL `n` elements before any read — every recursion path is a
    // pure store of its whole output: the radix-2 butterfly writes both halves
    // `out[2k]/out[2k+1]` and `out[2(k+half)]/..` for k in 0..half, the radix-5 /
    // naive base case writes every bin, and the n==1 leaf writes `[0],[1]`. No
    // path reads `out` before writing it. `FrameLanes` (`Simd<f32,8>`) is `Copy`
    // with no `Drop`, so `set_len` over uninitialised capacity neither drops nor
    // observes an uninitialised value. (Empirically: the full mel suite passes
    // bit-identically with and without this path — see the ledger.)
    unsafe {
        even_fft.set_len(n);
        odd_fft.set_len(n);
    }
    (even_fft, odd_fft)
}

/// One-sided top-level mirror of [`fft_simd8`]. The mel power spectrum consumes
/// only the 201 one-sided bins (`power_and_project_simd8` reads `fft_out[0..=200]`),
/// so the conjugate-symmetric upper half (bins 201..=399) the full FFT writes is
/// dead. This computes the lower half (bins 0..=199) plus the Nyquist bin 200 and
/// skips the dead upper stores — at the **outermost** level only; the recursion
/// still produces full sub-spectra, which the combine needs. The lower/Nyquist
/// writes are the identical expressions [`fft_simd8`] uses, so bins 0..=200 are
/// **bit-exact** (fewer stores, no arithmetic change): conformance and
/// `compute_8_columns_matches_scalar_columns_bit_exact` are untouched.
fn fft_simd8_onesided(
    input: &[FrameLanes],
    out: &mut [FrameLanes],
    levels: &[LevelTwiddle],
    base: &DftTable,
) {
    let n = input.len();
    debug_assert!(
        n.is_multiple_of(2) && n > 1,
        "one-sided top level expects even N>1"
    );
    let lvl = &levels[0];
    let half = n / 2;
    debug_assert_eq!(
        lvl.half, half,
        "fft_simd8_onesided level twiddle width mismatch"
    );
    let (even, odd) = split_even_odd(input, half);
    let (mut even_fft, mut odd_fft) = alloc_fft_scratch(2 * half);
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
        // Only bin `half` (Nyquist = k==0's upper output) is consumed downstream;
        // bins 201..=399 are conjugate-symmetric duplicates the mel path never reads.
        if k == 0 {
            out[2 * half] = e_re - re * re_odd + im * im_odd;
            out[2 * half + 1] = e_im - re * im_odd - im * re_odd;
        }
    }
}

/// FrameLanes complex radix-5 — lane-parallel mirror of [`radix5_cdft`] (same op
/// order ⇒ each lane is bit-identical to the scalar complex path).
fn radix5_cdft_simd8(input: &[FrameLanes], out: &mut [FrameLanes], t: &DftTable) {
    let z = FrameLanes::splat(0.0);
    let mut ir = [[z; 5]; 5];
    let mut ii = [[z; 5]; 5];
    for n2 in 0..5 {
        for k1 in 0..5 {
            let (mut re, mut im) = (z, z);
            for n1 in 0..5 {
                let xr = input[2 * (5 * n1 + n2)];
                let xi = input[2 * (5 * n1 + n2) + 1];
                let c = FrameLanes::splat(t.r5_c5[k1 * 5 + n1]);
                let s = FrameLanes::splat(t.r5_s5[k1 * 5 + n1]);
                re += xr * c + xi * s;
                im += xi * c - xr * s;
            }
            ir[n2][k1] = re;
            ii[n2][k1] = im;
        }
    }
    let mut tr = [[z; 5]; 5];
    let mut ti = [[z; 5]; 5];
    for n2 in 0..5 {
        for k1 in 0..5 {
            let c = FrameLanes::splat(t.r5_c25[n2 * 5 + k1]);
            let s = FrameLanes::splat(t.r5_s25[n2 * 5 + k1]);
            tr[n2][k1] = ir[n2][k1] * c + ii[n2][k1] * s;
            ti[n2][k1] = ii[n2][k1] * c - ir[n2][k1] * s;
        }
    }
    for k1 in 0..5 {
        for k2 in 0..5 {
            let (mut re, mut im) = (z, z);
            for n2 in 0..5 {
                let c = FrameLanes::splat(t.r5_c5[k2 * 5 + n2]);
                let s = FrameLanes::splat(t.r5_s5[k2 * 5 + n2]);
                re += tr[n2][k1] * c + ti[n2][k1] * s;
                im += ti[n2][k1] * c - tr[n2][k1] * s;
            }
            out[2 * (k1 + 5 * k2)] = re;
            out[2 * (k1 + 5 * k2) + 1] = im;
        }
    }
}

/// FrameLanes complex base case — mirror of [`cdft`].
fn cdft_simd8(input: &[FrameLanes], out: &mut [FrameLanes], table: &DftTable) {
    let n = input.len() / 2;
    if n == 25 {
        radix5_cdft_simd8(input, out, table);
        return;
    }
    for k in 0..n {
        let (mut re, mut im) = (FrameLanes::splat(0.0), FrameLanes::splat(0.0));
        let cos_row = &table.cos[k * table.n..k * table.n + n];
        let sin_row = &table.sin[k * table.n..k * table.n + n];
        for j in 0..n {
            let xr = input[2 * j];
            let xi = input[2 * j + 1];
            let c = FrameLanes::splat(cos_row[j]);
            let s = FrameLanes::splat(sin_row[j]);
            re += xr * c + xi * s;
            im += xi * c - xr * s;
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

/// FrameLanes complex FFT recursion — mirror of [`cfft`]; reuses the same
/// butterfly as [`fft_simd8`].
fn cfft_simd8(
    input: &[FrameLanes],
    out: &mut [FrameLanes],
    levels: &[LevelTwiddle],
    base: &DftTable,
) {
    let n = input.len() / 2;
    if n == 1 {
        out[0] = input[0];
        out[1] = input[1];
        return;
    }
    if n % 2 == 1 {
        cdft_simd8(input, out, base);
        return;
    }
    let lvl = &levels[0];
    let half = n / 2;
    let mut even = vec![FrameLanes::splat(0.0); 2 * half];
    let mut odd = vec![FrameLanes::splat(0.0); 2 * half];
    for i in 0..half {
        even[2 * i] = input[2 * (2 * i)];
        even[2 * i + 1] = input[2 * (2 * i) + 1];
        odd[2 * i] = input[2 * (2 * i + 1)];
        odd[2 * i + 1] = input[2 * (2 * i + 1) + 1];
    }
    let mut even_fft = vec![FrameLanes::splat(0.0); 2 * half];
    let mut odd_fft = vec![FrameLanes::splat(0.0); 2 * half];
    cfft_simd8(&even, &mut even_fft, &levels[1..], base);
    cfft_simd8(&odd, &mut odd_fft, &levels[1..], base);
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

/// FrameLanes two-for-one real FFT — mirror of [`fft_twoforone`], ONE-SIDED (only
/// bins `0..=N/2`, matching [`fft_simd8_onesided`]). The pack is identity: the
/// real input reinterpreted as `N/2` interleaved-complex IS the input itself.
/// Bit-identical (per lane) to the scalar [`fft_twoforone`] on bins `0..=N/2`.
fn fft_simd8_twoforone(
    input: &[FrameLanes],
    out: &mut [FrameLanes],
    levels: &[LevelTwiddle],
    base: &DftTable,
) {
    let n = input.len();
    let half = n / 2;
    let lvl = &levels[0];
    debug_assert_eq!(
        lvl.half, half,
        "fft_simd8_twoforone top twiddle width mismatch"
    );
    let mut zf = vec![FrameLanes::splat(0.0); 2 * half];
    cfft_simd8(input, &mut zf, &levels[1..], base);
    let h = FrameLanes::splat(0.5);
    for k in 0..half {
        let km = (half - k) % half;
        let ar = zf[2 * k];
        let ai = zf[2 * k + 1];
        let br = zf[2 * km];
        let bi = zf[2 * km + 1];
        let e_re = h * (ar + br);
        let e_im = h * (ai - bi);
        let o_re = h * (ai + bi);
        let o_im = h * (br - ar);
        let re = FrameLanes::splat(lvl.cos[k]);
        let im = FrameLanes::splat(lvl.neg_sin[k]);
        out[2 * k] = e_re + re * o_re - im * o_im;
        out[2 * k + 1] = e_im + re * o_im + im * o_re;
        if k == 0 {
            out[2 * half] = e_re - re * o_re + im * o_im;
            out[2 * half + 1] = e_im - re * o_im - im * o_re;
        }
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
    if rfft_enabled() {
        fft_twoforone(&fft_in, &mut fft_out, &twiddles.levels, &twiddles.base);
    } else {
        fft(&fft_in, &mut fft_out, &twiddles.levels, &twiddles.base);
    }
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
        *o = log10_floor_approx(sum);
    }
}

/// Approximate `log10(max(x, 1e-10))` for the mel projection.
///
/// This deliberately relaxes whisper.cpp's scalar `double log10` by about one
/// f32 ULP on the measured mel range. It is shared by the scalar and SIMD
/// projection paths so internal thread-count and scalar-vs-batched determinism
/// stay exact even though the result is no longer bit-for-bit libm output.
fn log10_floor_approx(x: f64) -> f32 {
    let x = x.max(1e-10);
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mantissa = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    let t = (mantissa - 1.0) / (mantissa + 1.0);
    let t2 = t * t;
    let series = 1.0 + t2 * (1.0 / 3.0 + t2 * (1.0 / 5.0 + t2 * (1.0 / 7.0 + t2 * (1.0 / 9.0))));
    let ln_m = 2.0 * t * series;
    ((exp as f64 * std::f64::consts::LN_2 + ln_m) * (1.0 / std::f64::consts::LN_10)) as f32
}

fn log10_floor_approx_simd8(x: Simd<f64, FFT_LANES>) -> Simd<f32, FFT_LANES> {
    use std::simd::num::{SimdFloat, SimdInt, SimdUint};

    type Vf = Simd<f64, FFT_LANES>;
    type Vu = Simd<u64, FFT_LANES>;

    let x = x.simd_max(Vf::splat(1e-10));
    let bits: Vu = x.to_bits();
    let exp = ((bits >> 52) & Vu::splat(0x7ff)).cast::<i64>() - Simd::splat(1023);
    let mantissa =
        Vf::from_bits((bits & Vu::splat(0x000f_ffff_ffff_ffff)) | Vu::splat(0x3ff0_0000_0000_0000));
    let t = (mantissa - Vf::splat(1.0)) / (mantissa + Vf::splat(1.0));
    let t2 = t * t;
    let series = Vf::splat(1.0)
        + t2 * (Vf::splat(1.0 / 3.0)
            + t2 * (Vf::splat(1.0 / 5.0)
                + t2 * (Vf::splat(1.0 / 7.0) + t2 * Vf::splat(1.0 / 9.0))));
    let ln_m = Vf::splat(2.0) * t * series;
    ((exp.cast::<f64>() * Vf::splat(std::f64::consts::LN_2) + ln_m)
        * Vf::splat(1.0 / std::f64::consts::LN_10))
    .cast::<f32>()
}

/// SIMD-batch equivalent of [`power_and_project`]. Each lane is still
/// accumulated over mel-bin weights in the same `k` order as the scalar helper;
/// this only avoids transposing the whole complex FFT result back into eight
/// scalar spectra before projection.
fn power_and_project_simd8(fft_out: &[FrameLanes], filters: &SparseMelFilters, out8: &mut [f32]) {
    let n_fft_bins = filters.fb.n_fft_bins;
    let n_mel = filters.fb.n_mel;

    let mut power = [FrameLanes::splat(0.0); N_FREQ_BINS];
    for (j, p) in power.iter_mut().enumerate() {
        let re = fft_out[2 * j];
        let im = fft_out[2 * j + 1];
        *p = re * re + im * im;
    }

    for m in 0..n_mel {
        let (start, end) = filters.ranges[m];
        let row = &filters.fb.data[m * n_fft_bins..m * n_fft_bins + n_fft_bins];
        let mut sums = [0.0f64; FFT_LANES];
        for (&pk, &rk) in power[start..end].iter().zip(row[start..end].iter()) {
            let lanes = pk.to_array();
            let weight = f64::from(rk);
            for lane in 0..FFT_LANES {
                sums[lane] += f64::from(lanes[lane]) * weight;
            }
        }
        let logs = log10_floor_approx_simd8(Simd::<f64, FFT_LANES>::from_array(sums)).to_array();
        for lane in 0..FFT_LANES {
            out8[lane * n_mel + m] = logs[lane];
        }
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
    // Structure-of-arrays windowed input: lane L holds frame (frame_base + L).
    // Every slot is written, so build it directly via `collect()` — no dead
    // zero-init to immediately overwrite (part of the measured ~5.5% scratch win).
    let fft_in: Vec<FrameLanes> = (0..N_FFT)
        .map(|j| {
            let mut lanes = [0.0f32; FFT_LANES];
            for (lane, l) in lanes.iter_mut().enumerate() {
                let offset = (frame_base + lane) * HOP;
                *l = hann[j] * padded[offset + j];
            }
            FrameLanes::from_array(lanes)
        })
        .collect();

    // The one-sided path writes/reads only bins 0..=N_FFT/2 (interleaved indices
    // `0..2*N_FREQ_BINS`); the full path writes the whole spectrum. Sizing the
    // buffer to what's used shrinks its per-call zero-init (the `vec!` of a
    // zero-bits value is an `alloc_zeroed`/memset) — measured ~4.5% of mel.
    let out_len = if fft_top_full() {
        2 * N_FFT
    } else {
        2 * N_FREQ_BINS
    };
    let mut fft_out = vec![FrameLanes::splat(0.0); out_len];
    if rfft_enabled() {
        fft_simd8_twoforone(&fft_in, &mut fft_out, &twiddles.levels, &twiddles.base);
    } else if fft_top_full() {
        fft_simd8(&fft_in, &mut fft_out, &twiddles.levels, &twiddles.base);
    } else {
        fft_simd8_onesided(&fft_in, &mut fft_out, &twiddles.levels, &twiddles.base);
    }
    power_and_project_simd8(&fft_out, filters, out8);
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
                    compute_8_columns(frame, padded_ref, hann_ref, filters_ref, twiddles_ref, col8);
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
    // SIMD over ~240k finite mel values: max-reduction is order-independent, and
    // `simd_max` matches the scalar clamp for finite data, so this is bit-exact.
    {
        use std::simd::num::SimdFloat;
        type V = std::simd::Simd<f32, 8>;
        let n = data.len();
        let n8 = n - n % 8;

        let mut acc = V::splat(f32::NEG_INFINITY);
        let mut i = 0;
        while i < n8 {
            acc = acc.simd_max(V::from_slice(&data[i..i + 8]));
            i += 8;
        }
        let mut mmax = acc.reduce_max();
        for &v in &data[n8..] {
            if v > mmax {
                mmax = v;
            }
        }

        let floor = mmax - 8.0;
        let floor_v = V::splat(floor);
        let four = V::splat(4.0);
        let mut i = 0;
        while i < n8 {
            let v = V::from_slice(&data[i..i + 8]);
            ((v.simd_max(floor_v) + four) / four).copy_to_slice(&mut data[i..i + 8]);
            i += 8;
        }
        for v in &mut data[n8..] {
            if *v < floor {
                *v = floor;
            }
            *v = (*v + 4.0) / 4.0;
        }
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
    let copy_frames = mel.n_frames.saturating_sub(frame_offset).min(n_frames);
    let pad_frames = n_frames - copy_frames;
    let mut data = Vec::with_capacity(mel.n_mel * n_frames);
    for m in 0..mel.n_mel {
        if copy_frames > 0 {
            let src = m * mel.n_frames + frame_offset;
            data.extend_from_slice(&mel.data[src..src + copy_frames]);
        }
        if pad_frames > 0 {
            data.resize(data.len() + pad_frames, SILENCE_FLOOR);
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
    fn fft_twiddle_table_matches_inline_reference() {
        // Compare the table-driven FFT against the inline-transcendental
        // reference over many random frames at the production width (400) plus a
        // spread of even/odd/power-of-two sizes that exercise every recursion
        // and base-case path.
        //
        // For widths whose odd base factor is NOT 25 the FFT MUST stay bit-exact
        // vs the reference (the twiddle table is a pure precompute of the same
        // arithmetic). Widths whose base case IS the n==25 factor use the
        // deliberate radix-5 relaxation (documented in the module header): a
        // *different*, more-accurate algorithm that diverges from the naive
        // inline DFT by ~1e-7. Those are checked at the same `rel<1e-4` bound as
        // `fft_matches_naive_dft` and the conformance gate — tight enough that a
        // real radix-5 bug (which diverges ~0.1) still fails loudly.
        for &n in &[N_FFT, 200usize, 100, 50, 25, 256, 8, 5, 2, 1] {
            // The FFT peels factors of 2 until the remaining odd factor is the
            // DFT base case; radix-5 fires iff that factor is 25.
            let mut odd = n;
            while odd.is_multiple_of(2) && odd > 1 {
                odd /= 2;
            }
            let uses_radix5 = odd == 25;
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
                if uses_radix5 {
                    let scale: f64 = want
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|c| (f64::from(c[0]).powi(2) + f64::from(c[1]).powi(2)).sqrt())
                        .fold(0.0f64, f64::max)
                        .max(1e-9);
                    for (g, w) in got.iter().zip(want.iter()) {
                        let rel = (f64::from(*g) - f64::from(*w)).abs() / scale;
                        assert!(
                            rel < 1e-4,
                            "radix-5 FFT diverged from naive reference beyond \
                             transcription tolerance at N={n} seed={seed}: rel={rel}"
                        );
                    }
                } else {
                    assert_eq!(
                        got, want,
                        "twiddle FFT diverged from inline reference at N={n} seed={seed}"
                    );
                }
            }
        }
    }

    #[test]
    fn fft_twoforone_matches_fft() {
        // The two-for-one real FFT must match the production `fft` over ALL bins
        // at the mel width within transcription tolerance (it changes float op
        // order, diverging ~1e-7). N=400 is the only production width; N/2=200
        // bottoms at the radix-5 base (200 = 8*25). A real bug diverges ~0.1.
        let tw = FftTwiddles::build(N_FFT);
        for seed in 0..48u64 {
            let mut lcg = Lcg::new(0x2F01_BEEF ^ seed.wrapping_mul(2_654_435_761));
            let input: Vec<f32> = (0..N_FFT).map(|_| lcg.next_f32()).collect();
            let mut got = vec![0.0f32; 2 * N_FFT];
            let mut want = vec![0.0f32; 2 * N_FFT];
            fft_twoforone(&input, &mut got, &tw.levels, &tw.base);
            fft(&input, &mut want, &tw.levels, &tw.base);
            let scale: f64 = want
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| (f64::from(c[0]).powi(2) + f64::from(c[1]).powi(2)).sqrt())
                .fold(0.0f64, f64::max)
                .max(1e-9);
            for b in 0..2 * N_FFT {
                let rel = (f64::from(got[b]) - f64::from(want[b])).abs() / scale;
                assert!(
                    rel < 1e-5,
                    "two-for-one diverged from fft at seed={seed} idx={b}: rel={rel}"
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
    fn fft_simd8_twoforone_matches_scalar() {
        // The SIMD two-for-one (one-sided) must be byte-identical PER LANE to the
        // scalar `fft_twoforone` over the bins mel reads (`0..=N_FFT/2`). This is
        // what preserves `determinism_across_thread_counts` once FW_RFFT flips on:
        // the SIMD-batch path and the scalar-remainder path produce identical mel.
        let tw = FftTwiddles::build(N_FFT);
        let mut lcg = Lcg::new(0x7F0F_2F01);
        let used = 2 * (N_FFT / 2) + 2; // re+im of bins 0..=N_FFT/2
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
            fft_simd8_twoforone(&vin, &mut vout, &tw.levels, &tw.base);
            for (lane, frame) in frames.iter().enumerate() {
                let mut sout = vec![0.0f32; 2 * N_FFT];
                fft_twoforone(frame, &mut sout, &tw.levels, &tw.base);
                for b in 0..used {
                    assert_eq!(
                        vout[b].to_array()[lane].to_bits(),
                        sout[b].to_bits(),
                        "lane {lane} bin-idx {b}"
                    );
                }
            }
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
    fn fft_simd8_onesided_matches_full_on_used_bins() {
        // The one-sided top-level FFT must be byte-identical to the full
        // `fft_simd8` over exactly the bins the mel power spectrum reads
        // (`0..=N_FFT/2` = 0..=200, i.e. interleaved indices `0..2*(N_FFT/2)+2`).
        // The upper bins it skips are unread, so they are intentionally NOT
        // compared. This is what makes skipping them safe.
        let tw = FftTwiddles::build(N_FFT);
        let mut lcg = Lcg::new(0x07E5_1DED);
        let used = 2 * (N_FFT / 2) + 2; // re+im of bins 0..=200
        for _round in 0..32 {
            let mut vin = vec![FrameLanes::splat(0.0); N_FFT];
            for slot in &mut vin {
                let mut a = [0.0f32; FFT_LANES];
                for al in &mut a {
                    *al = lcg.next_f32();
                }
                *slot = FrameLanes::from_array(a);
            }
            let mut full = vec![FrameLanes::splat(0.0); 2 * N_FFT];
            let mut one = vec![FrameLanes::splat(0.0); 2 * N_FFT];
            fft_simd8(&vin, &mut full, &tw.levels, &tw.base);
            fft_simd8_onesided(&vin, &mut one, &tw.levels, &tw.base);
            for b in 0..used {
                assert_eq!(
                    full[b].to_array().map(f32::to_bits),
                    one[b].to_array().map(f32::to_bits),
                    "one-sided diverged from full at used interleaved index {b}"
                );
            }
        }
    }

    #[test]
    fn compute_8_columns_matches_scalar_columns_bit_exact() {
        let filters = dummy_filters(24);
        let sparse = SparseMelFilters::new(&filters);
        let hann = cached_hann_window();
        let tw = cached_fft_twiddles();

        let mut lcg = Lcg::new(0xC011_EC70);
        let samples: Vec<f32> = (0..SAMPLE_RATE * 2)
            .map(|_| lcg.next_f32() * 0.75)
            .collect();
        let (padded, valid_len) = build_padded(&samples);
        assert!(FFT_LANES * HOP + N_FFT <= valid_len);

        let mut got = vec![0.0f32; FFT_LANES * filters.n_mel];
        compute_8_columns(0, &padded, hann, &sparse, tw, &mut got);

        let mut want = vec![0.0f32; FFT_LANES * filters.n_mel];
        for lane in 0..FFT_LANES {
            let col = &mut want[lane * filters.n_mel..(lane + 1) * filters.n_mel];
            compute_frame_column(lane, &padded, valid_len, hann, &sparse, tw, col);
        }

        for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "column value {idx}");
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

    #[test]
    fn chunk_frames_matches_scalar_reference() {
        fn reference_chunk_frames(mel: &Mel, frame_offset: usize, n_frames: usize) -> Mel {
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

        let mel = Mel {
            n_mel: 3,
            n_frames: 7,
            data: (0..21).map(|i| i as f32 + 0.25).collect(),
        };
        for &(offset, frames) in &[(0, 7), (2, 3), (5, 5), (7, 4), (9, 4), (3, 0)] {
            let got = chunk_frames(&mel, offset, frames);
            let want = reference_chunk_frames(&mel, offset, frames);
            assert_eq!(got.n_mel, want.n_mel);
            assert_eq!(got.n_frames, want.n_frames);
            assert_eq!(got.data, want.data, "offset={offset} frames={frames}");
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
