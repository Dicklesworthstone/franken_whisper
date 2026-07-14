//! Criterion benchmarks for the in-process native whisper engine
//! (`src/native_engine`). This is the measurement substrate owed on bead
//! bd-2th6 and the A/B instrument for the round-2 f16-compute optimization
//! passes.
//!
//! # What is covered
//! - `mel_30s`            — log-mel frontend over 30 s of synthetic audio (hermetic).
//! - `encoder_window_*`   — one full 3000-frame encoder window (tiny + large).
//! - `decoder_token_step_*` — ONE decoder `forward_step` at a fixed cache depth.
//! - `logits_gemv_large`  — the `[n_vocab, n_state]` tied output projection,
//!   the direct instrument for the f16-GEMV lever.
//! - `sanitize_downmix`    — built-in decoder ingestion: sanitize + channel
//!   average over decoded interleaved PCM.
//! - `e2e_tiny_jfk`       — full `transcribe_samples` over the jfk fixture.
//!
//! # Model gating
//! Every model-dependent bench is gated on model presence via
//! `native_engine::find_model_file`. When the model (or the jfk fixture) is
//! absent the group prints a visible `SKIP` to stderr and registers no
//! measurements, so CI without models stays green. Only `mel_30s` is fully
//! hermetic (synthetic audio + synthetic filterbank).
//! Set `FRANKEN_WHISPER_JFK_WAV=/path/to/jfk.wav` to point clean worktrees at an
//! external checked fixture without copying binary audio into the worktree.
//!
//! # How to A/B (round-2 f16 passes)
//! Save a pre-change baseline once:
//!
//! ```text
//! cargo bench --bench native_engine_bench -- --save-baseline round2-pre
//! ```
//!
//! Then, after a candidate lever (e.g. f16-resident decoder weights), compare
//! against it:
//!
//! ```text
//! cargo bench --bench native_engine_bench -- --baseline round2-pre
//! ```
//!
//! Criterion prints the per-bench delta and a significance verdict. Target the
//! `logits_gemv_large` and `encoder_window_large` / `decoder_token_step_large`
//! numbers for the f16 levers specifically.

use std::collections::HashSet;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use criterion::{Criterion, criterion_group, criterion_main};
use serde_json::{Value, json};

use franken_whisper::native_engine::decode::{DecodeParams, LoadedModel};
use franken_whisper::native_engine::decoder::{self, DecoderState};
use franken_whisper::native_engine::encoder;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::mel::{self, FRAMES_PER_CHUNK, N_SAMPLES_30S, SAMPLE_RATE};
use franken_whisper::native_engine::tokenizer::Tokenizer;
use franken_whisper::native_engine::{
    Mat, Mel, MelFilterbank, NativeWhisperModel, WhisperHParams, find_model_file,
};

// ---------------------------------------------------------------------------
// Deterministic synthetic-audio generator (no `rand` dependency)
// ---------------------------------------------------------------------------

/// Deterministic LCG mirroring the in-tree test generators (glibc constants).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
        (self.0 as f32 / (1u32 << 30) as f32) - 1.0
    }
}

/// `n` samples of a deterministic sine mix dithered by the LCG. Structured
/// enough to exercise the FFT/mel path realistically, fully reproducible.
fn synthetic_audio(n: usize, seed: u64) -> Vec<f32> {
    let mut lcg = Lcg::new(seed);
    let mut out = Vec::with_capacity(n);
    let sr = SAMPLE_RATE as f32;
    for i in 0..n {
        let t = i as f32 / sr;
        let tone = 0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
            + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            + 0.2 * (2.0 * std::f32::consts::PI * 660.0 * t).sin();
        // Small dither keeps every frame non-degenerate without dominating.
        out.push(0.9 * tone + 0.05 * lcg.next_f32());
    }
    out
}

/// A synthetic `[n_mel, N_FREQ_BINS]` filterbank. The mel-frontend cost is
/// dominated by the FFT, which is independent of the filter coefficients, so a
/// deterministic dense bank is representative for timing while keeping the
/// `mel_30s` bench fully hermetic (no model file required).
fn synthetic_filterbank(n_mel: usize) -> MelFilterbank {
    let n_fft_bins = mel::N_FREQ_BINS;
    let mut lcg = Lcg::new(0x5eed_f117);
    let mut data = vec![0.0f32; n_mel * n_fft_bins];
    for v in &mut data {
        // Non-negative weights, like a real mel filterbank.
        *v = (lcg.next_f32() + 1.0) * 0.5;
    }
    MelFilterbank {
        n_mel,
        n_fft_bins,
        data,
    }
}

/// A REALISTIC sparse triangular mel filterbank (standard Slaney-style mel
/// triangles), matching what a real ggml model carries: each of the `n_mel`
/// filters is nonzero over only ~5 contiguous freq bins of `N_FREQ_BINS=201`,
/// zero elsewhere. Unlike [`synthetic_filterbank`] (dense, every weight nonzero)
/// this exercises the sparse-projection fast path the production engine actually
/// hits — the dense bank hides ~13x of wasted multiply-by-zero work.
fn realistic_mel_filterbank(n_mel: usize) -> MelFilterbank {
    let n_fft_bins = mel::N_FREQ_BINS;
    let sr = SAMPLE_RATE as f64;
    let hz_to_mel = |h: f64| 2595.0 * (1.0 + h / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let m_min = hz_to_mel(0.0);
    let m_max = hz_to_mel(sr / 2.0);
    // n_mel+2 band edges in mel space, mapped back to Hz.
    let edges: Vec<f64> = (0..n_mel + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * (i as f64) / ((n_mel + 1) as f64)))
        .collect();
    let bin_hz = |k: usize| (k as f64) * (sr / 2.0) / ((n_fft_bins - 1) as f64);
    let mut data = vec![0.0f32; n_mel * n_fft_bins];
    for m in 0..n_mel {
        let (lo, ce, hi) = (edges[m], edges[m + 1], edges[m + 2]);
        for k in 0..n_fft_bins {
            let f = bin_hz(k);
            let w = if f >= lo && f <= ce && ce > lo {
                (f - lo) / (ce - lo)
            } else if f > ce && f <= hi && hi > ce {
                (hi - f) / (hi - ce)
            } else {
                0.0
            };
            data[m * n_fft_bins + k] = w as f32;
        }
    }
    MelFilterbank {
        n_mel,
        n_fft_bins,
        data,
    }
}

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// The two model short-names this bench exercises. `tiny.en` is the fast,
/// CI-friendly model; `large-v3-turbo` is the memory/bandwidth-bound target the
/// f16 levers care about.
const MODEL_TINY: &str = "tiny.en";
const MODEL_LARGE: &str = "large-v3-turbo";

/// Load a model by short name, or `None` (and a visible SKIP) when absent.
fn load_model(short_name: &str) -> Option<LoadedModel> {
    let path = find_model_file(short_name)?;
    match GgmlModel::load(&path).and_then(LoadedModel::from_ggml) {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("SKIP: failed to load model {short_name}: {e}");
            None
        }
    }
}

fn jfk_wav_path() -> PathBuf {
    std::env::var_os("FRANKEN_WHISPER_JFK_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/native/jfk.wav"
            ))
        })
}

/// Measure the safe resident-model lever for in-process API use.
///
/// This targets the loaded-model OpenAI API gap: direct parse+weight conversion
/// approximates a short-lived caller that drops the model after each request,
/// while `load_resident` keeps one bounded strong slot and returns an `Arc`
/// clone on repeat calls. `load_resident_canonical` isolates the remaining
/// hot-path overhead once an API server has resolved/canonicalized the model
/// path once during startup.
fn bench_model_residency_tiny(c: &mut Criterion) {
    let Some(path) = find_model_file(MODEL_TINY) else {
        eprintln!("SKIP model_residency_tiny: model {MODEL_TINY} missing");
        return;
    };
    let canonical = path.canonicalize().expect("canonical tiny.en model path");

    let mut group = c.benchmark_group("native_engine/model_residency");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("tiny_parse_weights_nonresident", |b| {
        b.iter(|| {
            let ggml = GgmlModel::load(black_box(path.as_path())).expect("load ggml tiny.en");
            let loaded = LoadedModel::from_ggml(ggml).expect("load weights tiny.en");
            black_box(loaded.hparams.n_vocab)
        });
    });

    let resident = NativeWhisperModel::load_resident(&path).expect("resident tiny.en warmup");
    black_box(resident.loaded().hparams.n_vocab);
    group.bench_function("tiny_resident_cache_lookup", |b| {
        b.iter(|| {
            let model = NativeWhisperModel::load_resident(black_box(path.as_path()))
                .expect("resident load");
            black_box(model.loaded().hparams.n_vocab)
        });
    });

    group.bench_function("tiny_resident_canonical_lookup", |b| {
        b.iter(|| {
            let model = NativeWhisperModel::load_resident_canonical(black_box(canonical.as_path()))
                .expect("resident canonical load");
            black_box(model.loaded().hparams.n_vocab)
        });
    });

    group.finish();
}

/// Read the jfk fixture as mono 16 kHz f32, or `None` (with a SKIP) when absent.
///
/// Uses `hound` (a first-class dependency) rather than the crate-private wav
/// reader so the bench needs no extra visibility widening. The fixture is a
/// 16 kHz mono i16 WAV (verified); we assert that and normalize to `[-1, 1]`.
fn load_jfk_samples() -> Option<Vec<f32>> {
    let path = jfk_wav_path();
    let mut reader = match hound::WavReader::open(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("SKIP: jfk.wav not readable at {} ({e})", path.display());
            return None;
        }
    };
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != SAMPLE_RATE as u32 {
        eprintln!(
            "SKIP: jfk.wav is {}ch @ {}Hz, expected mono 16kHz",
            spec.channels, spec.sample_rate
        );
        return None;
    }
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|v| f32::from(v) / 32768.0))
        .collect::<Result<_, _>>()
        .ok()?;
    Some(samples)
}

/// Compute the model's full first 3000-frame mel window from the jfk audio,
/// padding (with `mel::SILENCE_FLOOR`) to exactly `FRAMES_PER_CHUNK` frames the
/// way the decode loop does. Returns a model-shaped `[n_mels, 3000]` window the
/// encoder accepts.
fn jfk_full_window(model: &LoadedModel, samples: &[f32]) -> Mel {
    let full = mel::log_mel(samples, &model.filters, 8).expect("log_mel");
    // jfk is only 11 s, so the real mel is < 3000 frames; pad to a full window.
    mel::chunk_frames(&full, 0, FRAMES_PER_CHUNK)
}

/// Run the encoder over the jfk full window to obtain a real `[n_ctx, n_state]`
/// acoustic embedding for the decoder/GEMV fixtures.
fn jfk_encoder_out(model: &LoadedModel, samples: &[f32]) -> Mat {
    let window = jfk_full_window(model, samples);
    let noop = || Ok(());
    encoder::forward(&model.encoder, &window, 8, &noop).expect("encoder forward")
}

// ---------------------------------------------------------------------------
// 1. mel_30s — hermetic log-mel frontend over 30 s of synthetic audio
// ---------------------------------------------------------------------------

fn bench_mel_30s(c: &mut Criterion) {
    let mut group = c.benchmark_group("native_engine/mel");
    // 30 s of audio @ 16 kHz. log_mel is O(100 ms); the default sample budget
    // is fine but we cap measurement time so the whole suite stays bounded.
    group.measurement_time(Duration::from_secs(10));

    let audio = synthetic_audio(N_SAMPLES_30S, 0xa11ce);
    // tiny.en/large both use 80-bin mel; large-v3* uses 128. Bench the 80-bin
    // frontend (the common case); the FFT cost dominates and is mel-count
    // independent, so this is representative for both.
    let filters = synthetic_filterbank(80);

    group.bench_function("mel_30s", |b| {
        b.iter(|| {
            let m = mel::log_mel(black_box(&audio), black_box(&filters), 8).expect("log_mel");
            black_box(m.n_frames)
        });
    });

    group.finish();
}

/// `mel_30s` over a REALISTIC sparse triangular filterbank — the production case
/// (real ggml models carry sparse banks). This is the bench where the
/// sparse-projection lever shows up: with a dense bank (`mel_30s`) the projection
/// touches all 201 bins per filter; with a real sparse bank only ~5 are nonzero,
/// so skipping the zeros (bit-exact) removes ~13x of the projection's
/// multiply-adds. Compare `mel_30s` (dense) vs `mel_30s_realistic` to see the
/// projection's share of the frontend.
fn bench_mel_30s_realistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("native_engine/mel");
    group.measurement_time(Duration::from_secs(10));

    let audio = synthetic_audio(N_SAMPLES_30S, 0xa11ce);
    let filters = realistic_mel_filterbank(80);

    group.bench_function("mel_30s_realistic", |b| {
        b.iter(|| {
            let m = mel::log_mel(black_box(&audio), black_box(&filters), 8).expect("log_mel");
            black_box(m.n_frames)
        });
    });

    group.finish();
}

/// Copy one Whisper encoder window out of a row-major full-mel buffer.
///
/// This isolates the decode loop's `mel[:, seek:seek+N_FRAMES]` equivalent:
/// OpenAI Whisper can hand PyTorch a strided view, while the Rust encoder owns a
/// compact `[n_mels, n_frames]` buffer and therefore copies/pads the window.
fn bench_chunk_frames(c: &mut Criterion) {
    let mut group = c.benchmark_group("native_engine/mel");
    let n_mel = 80;
    let n_frames = FRAMES_PER_CHUNK + 1024;
    let mut lcg = Lcg::new(0xc0ffee);
    let data: Vec<f32> = (0..n_mel * n_frames).map(|_| lcg.next_f32()).collect();
    let mel = Mel {
        n_mel,
        n_frames,
        data,
    };

    group.bench_function("chunk_frames_80x3000_mid", |b| {
        b.iter(|| {
            let chunk =
                mel::chunk_frames(black_box(&mel), black_box(512), black_box(FRAMES_PER_CHUNK));
            black_box(chunk.data.len())
        });
    });

    group.finish();
}

/// Prepare one encoder window for conv1: old compact-window path vs fused
/// slice+transpose. This targets the OpenAI view/copy gap recorded for
/// `chunk_frames`: the encoder ultimately needs time-major data, so avoiding
/// the intermediate compact mel-major window is the smallest in-crate
/// zero-copy-style lever.
fn bench_window_to_time_major(c: &mut Criterion) {
    let mut group = c.benchmark_group("native_engine/mel");
    let n_mel = 80;
    let n_frames = FRAMES_PER_CHUNK + 1024;
    let mut lcg = Lcg::new(0xc0ffee);
    let data: Vec<f32> = (0..n_mel * n_frames).map(|_| lcg.next_f32()).collect();
    let mel = Mel {
        n_mel,
        n_frames,
        data,
    };
    let offset = 512;

    group.bench_function("window_to_time_major_old_chunk_then_transpose", |b| {
        b.iter(|| {
            let chunk = mel::chunk_frames(
                black_box(&mel),
                black_box(offset),
                black_box(FRAMES_PER_CHUNK),
            );
            let time_major = encoder::time_major_mel_window(black_box(&chunk));
            black_box(time_major.data.len())
        });
    });

    group.bench_function("window_to_time_major_fused", |b| {
        b.iter(|| {
            let time_major = encoder::time_major_mel_window_from_full_mel(
                black_box(&mel),
                black_box(offset),
                black_box(FRAMES_PER_CHUNK),
            );
            black_box(time_major.data.len())
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. encoder_window_{tiny,large} — one full 3000-frame encoder window
// ---------------------------------------------------------------------------

fn bench_encoder_window(c: &mut Criterion, short_name: &str, label: &str) {
    let (Some(model), Some(samples)) = (load_model(short_name), load_jfk_samples()) else {
        eprintln!("SKIP encoder_window_{label}: model {short_name} or jfk.wav missing");
        return;
    };
    let window = jfk_full_window(&model, &samples);
    let noop = || Ok(());

    let mut group = c.benchmark_group("native_engine/encoder");
    // A full large window is ~4.4 s; keep the sample count small and grant
    // enough measurement time for a handful of iterations.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    group.bench_function(format!("encoder_window_{label}"), |b| {
        b.iter(|| {
            let out = encoder::forward(&model.encoder, black_box(&window), 8, &noop)
                .expect("encoder forward");
            black_box(out.rows)
        });
    });

    group.finish();
}

fn bench_encoder_window_tiny(c: &mut Criterion) {
    bench_encoder_window(c, MODEL_TINY, "tiny");
}

fn bench_encoder_window_large(c: &mut Criterion) {
    bench_encoder_window(c, MODEL_LARGE, "large");
}

// ---------------------------------------------------------------------------
// 3. decoder_token_step_{tiny,large} — ONE forward_step at a fixed cache depth
// ---------------------------------------------------------------------------

/// Tokens advanced per timed iteration (see the DESIGN note). Whisper greedily
/// decodes ~a few dozen tokens per window; an 8-step sequence from a freshly
/// reset cache spans a representative early-to-mid-window self-attention depth.
const TOKEN_STEP_SEQ: usize = 8;

fn bench_decoder_token_step(c: &mut Criterion, short_name: &str, label: &str) {
    let (Some(model), Some(samples)) = (load_model(short_name), load_jfk_samples()) else {
        eprintln!("SKIP decoder_token_step_{label}: model {short_name} or jfk.wav missing");
        return;
    };
    let enc_out = jfk_encoder_out(&model, &samples);
    let w = &model.decoder;
    let sot = model.tokenizer.sot;
    let noop = || Ok(());

    // DESIGN — isolating honest per-token cost.
    //
    // `forward_step` MUTATES the KV cache, growing it by one token each call.
    // To bench per-token cost we cannot use `iter_batched`'s separate
    // setup/routine closures: both would need `&mut st`, and the encoder-backed
    // `DecoderState` is neither cheap to rebuild per iteration (cross-K/V
    // projection over ~1500 enc frames) nor `Clone`. So we use the design the
    // bd-2th6 spec sanctions: each timed iteration is `reset()` + a FIXED
    // sequence of `TOKEN_STEP_SEQ` `forward_step`s, and `Throughput::Elements`
    // makes criterion report the **per-step** figure directly.
    //
    // `DecoderState::reset()` restores the initial (cross-K/V-retained) state —
    // proven by the in-tree `reset_clears_cache` test — so every iteration runs
    // the identical, deterministic token sequence over a cache that grows 0→8.
    // The cross-attention K/V (the window-constant, encoder-derived part) is
    // built once outside the timed region; only the self-attention step cost is
    // measured. The reported per-element time is the per-token marginal cost
    // averaged over depths 0..8 — the quantity the f16-GEMV lever moves.
    let mut st = DecoderState::new(w, &enc_out).expect("decoder state");
    // Fixed, valid token sequence: sot then deterministic small ids.
    let seq: Vec<i32> = (0..TOKEN_STEP_SEQ)
        .map(|i| if i == 0 { sot } else { 1 + (i as i32) })
        .collect();

    let mut group = c.benchmark_group("native_engine/decoder");
    group.throughput(criterion::Throughput::Elements(TOKEN_STEP_SEQ as u64));
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(15));

    group.bench_function(format!("decoder_token_step_{label}"), |b| {
        b.iter(|| {
            st.reset();
            let mut last = 0usize;
            for &tok in &seq {
                let logits =
                    decoder::forward_step(w, &mut st, black_box(&[tok]), &noop).expect("step");
                last = logits.len();
            }
            black_box(last)
        });
    });

    group.finish();
}

fn bench_decoder_token_step_tiny(c: &mut Criterion) {
    bench_decoder_token_step(c, MODEL_TINY, "tiny");
}

fn bench_decoder_token_step_large(c: &mut Criterion) {
    bench_decoder_token_step(c, MODEL_LARGE, "large");
}

// ---------------------------------------------------------------------------
// 3b. self-attention packed K — same-binary, interleaved BASE/candidate A/B
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct PairedRatioStats {
    median: f64,
    p10: f64,
    p90: f64,
    min: f64,
    max: f64,
    cv_pct: f64,
    wins: usize,
}

fn paired_ratio_stats(ratios: &[f64]) -> PairedRatioStats {
    assert!(!ratios.is_empty());
    let mut sorted = ratios.to_vec();
    sorted.sort_by(f64::total_cmp);
    let nearest_rank = |percent: usize| {
        let index = (sorted.len() * percent).div_ceil(100).saturating_sub(1);
        sorted[index.min(sorted.len() - 1)]
    };
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let variance = if ratios.len() > 1 {
        ratios
            .iter()
            .map(|ratio| (ratio - mean).powi(2))
            .sum::<f64>()
            / (ratios.len() - 1) as f64
    } else {
        0.0
    };
    PairedRatioStats {
        median: sorted[sorted.len() / 2],
        p10: nearest_rank(10),
        p90: nearest_rank(90),
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        cv_pct: variance.sqrt() / mean * 100.0,
        wins: ratios.iter().filter(|&&ratio| ratio > 1.0).count(),
    }
}

fn format_ratios(ratios: &[f64]) -> String {
    ratios
        .iter()
        .map(|ratio| format!("{ratio:.6}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn measure_self_attn_arm(
    cache: &mut franken_whisper::native_engine::nn::KvCache,
    q: &Mat,
    k_new: &Mat,
    v_new: &Mat,
    n_head: usize,
    prefill: usize,
    inner_steps: usize,
) -> Duration {
    use franken_whisper::native_engine::nn::attention_with_cache;

    let started = Instant::now();
    for _ in 0..inner_steps {
        let output = attention_with_cache(
            black_box(q),
            black_box(k_new),
            black_box(v_new),
            black_box(n_head),
            black_box(&mut *cache),
        )
        .expect("self-attention step");
        black_box(output.data.as_slice());
        cache.truncate_for_bench(prefill);
    }
    started.elapsed()
}

#[allow(clippy::too_many_arguments)]
fn paired_self_attn_ratios(
    first: &mut franken_whisper::native_engine::nn::KvCache,
    second: &mut franken_whisper::native_engine::nn::KvCache,
    q: &Mat,
    k_new: &Mat,
    v_new: &Mat,
    n_head: usize,
    prefill: usize,
    inner_steps: usize,
    repetitions: usize,
) -> Vec<f64> {
    let measure = |cache: &mut franken_whisper::native_engine::nn::KvCache| {
        measure_self_attn_arm(cache, q, k_new, v_new, n_head, prefill, inner_steps)
    };
    let mut ratios = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let (first_elapsed, second_elapsed) = if repetition % 2 == 0 {
            let first_before = measure(first);
            let second_before = measure(second);
            let second_after = measure(second);
            let first_after = measure(first);
            (first_before + first_after, second_before + second_after)
        } else {
            let second_before = measure(second);
            let first_before = measure(first);
            let first_after = measure(first);
            let second_after = measure(second);
            (first_before + first_after, second_before + second_after)
        };
        ratios.push(first_elapsed.as_secs_f64() / second_elapsed.as_secs_f64());
    }
    ratios
}

fn bench_self_attn_column_keys(c: &mut Criterion) {
    use franken_whisper::native_engine::nn::{KvCache, attention_with_cache};

    // large-v3-turbo decoder geometry: d_model=1280, 20 heads, 448-token text
    // context. The last-token case maximizes the live score loop without using
    // a private replica of the production function.
    const N_STATE: usize = 1280;
    const N_HEAD: usize = 20;
    const CAPACITY: usize = 448;
    const PREFILL: usize = CAPACITY - 1;
    const INNER_STEPS: usize = 16;
    const WARMUP_REPS: usize = 3;
    const PAIRED_REPS: usize = 31;
    const NULL_MEDIAN_MIN: f64 = 0.98;
    const NULL_MEDIAN_MAX: f64 = 1.02;

    let mut lcg = Lcg::new(0xc011_0a7e);
    let prefill_k = Mat::from_vec(
        PREFILL,
        N_STATE,
        (0..PREFILL * N_STATE)
            .map(|_| 0.1 * lcg.next_f32())
            .collect(),
    );
    let prefill_v = Mat::from_vec(
        PREFILL,
        N_STATE,
        (0..PREFILL * N_STATE)
            .map(|_| 0.1 * lcg.next_f32())
            .collect(),
    );
    let q = Mat::from_vec(
        1,
        N_STATE,
        (0..N_STATE).map(|_| 0.1 * lcg.next_f32()).collect(),
    );
    let k_new = Mat::from_vec(
        1,
        N_STATE,
        (0..N_STATE).map(|_| 0.1 * lcg.next_f32()).collect(),
    );
    let v_new = Mat::from_vec(
        1,
        N_STATE,
        (0..N_STATE).map(|_| 0.1 * lcg.next_f32()).collect(),
    );

    let prefill = |cache: &mut KvCache| {
        cache
            .append(black_box(&prefill_k), black_box(&prefill_v))
            .expect("prefill self-attention cache");
    };
    let mut null_first = KvCache::new_row_major_keys_for_bench(CAPACITY, N_STATE);
    let mut null_second = KvCache::new_row_major_keys_for_bench(CAPACITY, N_STATE);
    let mut original = KvCache::new_row_major_keys_for_bench(CAPACITY, N_STATE);
    let mut candidate = KvCache::new_column_major_keys_for_bench(CAPACITY, N_STATE);
    prefill(&mut null_first);
    prefill(&mut null_second);
    prefill(&mut original);
    prefill(&mut candidate);

    // Establish the per-function timing floor first. BASE/BASE uses distinct,
    // identically populated caches and the same ABBA routine as BASE/candidate.
    black_box(paired_self_attn_ratios(
        &mut null_first,
        &mut null_second,
        &q,
        &k_new,
        &v_new,
        N_HEAD,
        PREFILL,
        INNER_STEPS,
        WARMUP_REPS,
    ));
    let null_ratios = paired_self_attn_ratios(
        &mut null_first,
        &mut null_second,
        &q,
        &k_new,
        &v_new,
        N_HEAD,
        PREFILL,
        INNER_STEPS,
        PAIRED_REPS,
    );
    let null_stats = paired_ratio_stats(&null_ratios);
    eprintln!(
        "SELF_K_NULL ratios=[{}] median={:.6} p10={:.6} p90={:.6} min={:.6} \
         max={:.6} cv_pct={:.3} wins={}/{} acceptance=[{NULL_MEDIAN_MIN:.2},{NULL_MEDIAN_MAX:.2}]",
        format_ratios(&null_ratios),
        null_stats.median,
        null_stats.p10,
        null_stats.p90,
        null_stats.min,
        null_stats.max,
        null_stats.cv_pct,
        null_stats.wins,
        PAIRED_REPS,
    );
    let null_valid = (NULL_MEDIAN_MIN..=NULL_MEDIAN_MAX).contains(&null_stats.median);

    // Correctness is checked through the real public path before candidate
    // timing. Both complete 1280-float outputs must match bit-for-bit.
    let original_out = attention_with_cache(
        black_box(&q),
        black_box(&k_new),
        black_box(&v_new),
        black_box(N_HEAD),
        black_box(&mut original),
    )
    .expect("original self-attention step");
    original.truncate_for_bench(PREFILL);
    let candidate_out = attention_with_cache(
        black_box(&q),
        black_box(&k_new),
        black_box(&v_new),
        black_box(N_HEAD),
        black_box(&mut candidate),
    )
    .expect("candidate self-attention step");
    candidate.truncate_for_bench(PREFILL);
    assert_eq!(
        original_out
            .data
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        candidate_out
            .data
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "packed K layout changed turbo-shaped attention output"
    );
    black_box(original_out.data.as_slice());
    black_box(candidate_out.data.as_slice());

    if null_valid {
        black_box(paired_self_attn_ratios(
            &mut original,
            &mut candidate,
            &q,
            &k_new,
            &v_new,
            N_HEAD,
            PREFILL,
            INNER_STEPS,
            WARMUP_REPS,
        ));
        let candidate_ratios = paired_self_attn_ratios(
            &mut original,
            &mut candidate,
            &q,
            &k_new,
            &v_new,
            N_HEAD,
            PREFILL,
            INNER_STEPS,
            PAIRED_REPS,
        );
        let candidate_stats = paired_ratio_stats(&candidate_ratios);
        let verdict = if candidate_stats.median > null_stats.p90 {
            "WIN_ABOVE_NULL_P90"
        } else if candidate_stats.median < null_stats.p10 {
            "REJECT_BELOW_NULL_P10"
        } else {
            "SURFACE_INSIDE_NULL"
        };
        eprintln!(
            "SELF_K_CANDIDATE reach=attention_with_cache parity=bit_exact ratios=[{}] \
             median={:.6} p10={:.6} p90={:.6} min={:.6} max={:.6} cv_pct={:.3} \
             wins={}/{} null_median={:.6} null_floor=[{:.6},{:.6}] verdict={verdict}",
            format_ratios(&candidate_ratios),
            candidate_stats.median,
            candidate_stats.p10,
            candidate_stats.p90,
            candidate_stats.min,
            candidate_stats.max,
            candidate_stats.cv_pct,
            candidate_stats.wins,
            PAIRED_REPS,
            null_stats.median,
            null_stats.p10,
            null_stats.p90,
        );
    } else {
        eprintln!(
            "SELF_K_CANDIDATE verdict=BLOCKED_NULL_MEDIAN null_median={:.6} \
             acceptance=[{NULL_MEDIAN_MIN:.2},{NULL_MEDIAN_MAX:.2}] candidate_not_timed=true",
            null_stats.median,
        );
    }

    // These are profiler-reachability loops only, not A/B evidence. The valid
    // comparison above is interleaved in one routine; these two labels merely
    // give perf enough samples to verify both production symbols are live.
    let mut group = c.benchmark_group("native_engine/self_attn_k_layout");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("profile_only_orig_row_major", |b| {
        b.iter(|| {
            let elapsed = measure_self_attn_arm(
                &mut original,
                &q,
                &k_new,
                &v_new,
                N_HEAD,
                PREFILL,
                1,
            );
            black_box(elapsed);
        });
    });
    group.bench_function("profile_only_candidate_column_major", |b| {
        b.iter(|| {
            let elapsed = measure_self_attn_arm(
                &mut candidate,
                &q,
                &k_new,
                &v_new,
                N_HEAD,
                PREFILL,
                1,
            );
            black_box(elapsed);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// 4. logits_gemv_large — the [n_vocab, n_state] tied output projection
// ---------------------------------------------------------------------------

fn bench_logits_gemv(c: &mut Criterion, short_name: &str, label: &str) {
    let Some(model) = load_model(short_name) else {
        eprintln!("SKIP logits_gemv_{label}: model {short_name} missing");
        return;
    };
    let w = &model.decoder;
    // A fixed, deterministic hidden vector `[1, n_state]` — the decoder's last
    // hidden row feeding the tied output projection. We bench the projection in
    // isolation so the f16-GEMV lever has a direct, allocation-free instrument.
    let mut lcg = Lcg::new(0x109175);
    let n_state = w.n_state();
    let hidden: Vec<f32> = (0..n_state).map(|_| 0.1 * lcg.next_f32()).collect();
    let x_last = Mat::from_vec(1, n_state, hidden);

    let mut group = c.benchmark_group("native_engine/logits");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(12));

    group.bench_function(format!("logits_gemv_{label}"), |b| {
        b.iter(|| {
            let logits = decoder::logits_last(w, black_box(&x_last)).expect("logits_last");
            black_box(logits.len())
        });
    });

    group.finish();
}

fn bench_logits_gemv_large(c: &mut Criterion) {
    bench_logits_gemv(c, MODEL_LARGE, "large");
}

// ---------------------------------------------------------------------------
// 5. e2e_tiny_jfk — full transcribe_samples over the jfk fixture
// ---------------------------------------------------------------------------

fn bench_e2e_tiny_jfk(c: &mut Criterion) {
    let Some(path) = find_model_file(MODEL_TINY) else {
        eprintln!("SKIP e2e_tiny_jfk: model {MODEL_TINY} missing");
        return;
    };
    let Some(samples) = load_jfk_samples() else {
        eprintln!("SKIP e2e_tiny_jfk: jfk.wav missing");
        return;
    };
    let model = match NativeWhisperModel::load(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP e2e_tiny_jfk: model load failed: {e}");
            return;
        }
    };
    let params = DecodeParams {
        language: None,
        translate: false,
        timestamps: true,
        n_threads: 8,
        ..DecodeParams::default()
    };
    let noop = || Ok(());

    let mut group = c.benchmark_group("native_engine/e2e");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("e2e_tiny_jfk", |b| {
        b.iter(|| {
            let out = model
                .transcribe(black_box(&samples), &params, &noop)
                .expect("transcribe");
            black_box(out.segments.len())
        });
    });

    group.finish();
}

// 5a. e2e_tiny_jfk_no_timestamps — same loaded-model tiny JFK path with
// timestamp-token segmentation disabled. This mirrors the large head-to-head
// bench mode and isolates the cost of the timestamped decode policy.
fn bench_e2e_tiny_jfk_no_timestamps(c: &mut Criterion) {
    let Some(path) = find_model_file(MODEL_TINY) else {
        eprintln!("SKIP e2e_tiny_jfk_no_timestamps: model {MODEL_TINY} missing");
        return;
    };
    let Some(samples) = load_jfk_samples() else {
        eprintln!("SKIP e2e_tiny_jfk_no_timestamps: jfk.wav missing");
        return;
    };
    let model = match NativeWhisperModel::load(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP e2e_tiny_jfk_no_timestamps: model load failed: {e}");
            return;
        }
    };
    let params = DecodeParams {
        language: None,
        translate: false,
        timestamps: false,
        n_threads: 8,
        ..DecodeParams::default()
    };
    let noop = || Ok(());

    let mut group = c.benchmark_group("native_engine/e2e");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("e2e_tiny_jfk_no_timestamps", |b| {
        b.iter(|| {
            let out = model
                .transcribe(black_box(&samples), &params, &noop)
                .expect("transcribe");
            black_box(out.segments.len())
        });
    });

    group.finish();
}

// 5b. e2e_large_jfk — full transcribe over jfk with large-v3-turbo. No word
// timestamps (matches whisper.cpp `dtw=0`) for an apples-to-apples head-to-head.
fn bench_e2e_large_jfk(c: &mut Criterion) {
    let Some(path) = find_model_file(MODEL_LARGE) else {
        eprintln!("SKIP e2e_large_jfk: model {MODEL_LARGE} missing");
        return;
    };
    let Some(samples) = load_jfk_samples() else {
        eprintln!("SKIP e2e_large_jfk: jfk.wav missing");
        return;
    };
    let model = match NativeWhisperModel::load(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("SKIP e2e_large_jfk: model load failed: {e}");
            return;
        }
    };
    let params = DecodeParams {
        language: None,
        translate: false,
        timestamps: false,
        n_threads: 8,
        ..DecodeParams::default()
    };
    let noop = || Ok(());

    let mut group = c.benchmark_group("native_engine/e2e");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(90));

    group.bench_function("e2e_large_jfk", |b| {
        b.iter(|| {
            let out = model
                .transcribe(black_box(&samples), &params, &noop)
                .expect("transcribe");
            black_box(out.segments.len())
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. f16_gemv_dequant — isolated f16-resident GEMV (dequant + dot) throughput.
//     Direct instrument for the pass-3 vectorizable-dequant lever: a single
//     `[1280, 1280]` f16 weight (1280 output rows, the large decoder Linear
//     width) dotted against one activation. The kernel internally bulk-SIMD
//     dequantizes each row then runs the vectorized dot8 — this bench is where
//     the 3.0 -> 13.5 GFLOP/s win shows up, free of model-load / encoder noise.
// ---------------------------------------------------------------------------

fn f16_normal_weight(out: usize, inp: usize) -> Vec<ft_core::Float16> {
    // Deterministic normal-range half weight (whisper weights are normal range).
    (0..out * inp)
        .map(|i| {
            let e = 1 + (i % 30) as u16;
            let m = (i * 37 % 1024) as u16;
            let s = ((i % 2) as u16) << 15;
            ft_core::Float16::from_bits(s | (e << 10) | m)
        })
        .collect()
}

fn bench_f16_gemv_dequant(c: &mut Criterion) {
    use franken_whisper::native_engine::nn;

    let mut group = c.benchmark_group("native_engine/f16_gemv");

    // The large decoder Linear width: the bandwidth-class shape where the
    // band-parallel scope path wins big (the pass-3 3.0 → ~16 GFLOP/s case).
    {
        let (out, inp) = (1280usize, 1280usize);
        let w = f16_normal_weight(out, inp);
        let x: Vec<f32> = (0..inp).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut out_buf = vec![0.0f32; out];
        group.throughput(criterion::Throughput::Elements((out * inp) as u64));
        group.bench_function("f16_gemv_dequant_1280x1280", |b| {
            b.iter(|| {
                nn::gemv_f16(black_box(&w), out, inp, black_box(&x), None, &mut out_buf);
                black_box(out_buf[0])
            });
        });
    }

    // The tiny.en per-token self-attention / cross-attention Linear shape
    // ([1,384] x [384,384]). At ~147 k MACs this is ~9 µs of compute, far below
    // the 8-way `thread::scope` spawn/join cost — the round-3 pass-2 lever moves
    // its `PAR_THRESHOLD` above this so it runs serial. This bench is the direct
    // instrument for that spawn-elimination (it must DROP after the lever).
    {
        let (out, inp) = (384usize, 384usize);
        let w = f16_normal_weight(out, inp);
        let x: Vec<f32> = (0..inp).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut out_buf = vec![0.0f32; out];
        group.throughput(criterion::Throughput::Elements((out * inp) as u64));
        group.bench_function("f16_gemv_dequant_384x384", |b| {
            b.iter(|| {
                nn::gemv_f16(black_box(&w), out, inp, black_box(&x), None, &mut out_buf);
                black_box(out_buf[0])
            });
        });
    }

    // BATCHED f16 gemv (tq>1, the compute-bound path used by cross-KV precompute
    // and prompt processing). cross-KV shape for large: tq=1500 enc frames,
    // out=inp=1280. ~2.4 GFLOP — the `FW_BATCH_GEMV_CAP` worker-count lever shows
    // up here (m=1 dispatch-bound cap8 vs compute-bound scaling).
    {
        let (tq, out, inp) = (1500usize, 1280usize, 1280usize);
        let w = f16_normal_weight(out, inp);
        let x: Vec<f32> = (0..tq * inp).map(|i| (i as f32 * 0.0001).sin()).collect();
        let mut out_buf = vec![0.0f32; tq * out];
        group.throughput(criterion::Throughput::Elements((tq * out * inp) as u64));
        group.bench_function("f16_gemv_batch_1500x1280x1280", |b| {
            b.iter(|| {
                nn::gemv_f16_batch(
                    black_box(&w),
                    out,
                    inp,
                    black_box(&x),
                    tq,
                    None,
                    &mut out_buf,
                );
                black_box(out_buf[0])
            });
        });
    }

    group.finish();
}

fn synthetic_mat(rows: usize, cols: usize, seed: u64) -> Mat {
    let mut lcg = Lcg::new(seed);
    Mat::from_vec(
        rows,
        cols,
        (0..rows * cols).map(|_| lcg.next_f32()).collect(),
    )
}

fn synthetic_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut lcg = Lcg::new(seed);
    (0..n).map(|_| lcg.next_f32()).collect()
}

fn bench_i7_qkv_activation_reuse(c: &mut Criterion) {
    use franken_whisper::native_engine::nn;

    let h = synthetic_mat(1500, 1280, 0x51);
    let wq = nn::quantize_mat_to_i7(&synthetic_mat(1280, 1280, 0x52));
    let wk = nn::quantize_mat_to_i7(&synthetic_mat(1280, 1280, 0x53));
    let wv = nn::quantize_mat_to_i7(&synthetic_mat(1280, 1280, 0x54));
    let bq = synthetic_vec(1280, 0x55);
    let bv = synthetic_vec(1280, 0x56);

    let mut group = c.benchmark_group("native_engine/i7_qkv");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_secs(2));

    group.bench_function("three_inline_quantize_1500x1280", |b| {
        b.iter(|| {
            let q =
                nn::matmul_bias_i7(black_box(&h), black_box(&wq), Some(black_box(&bq))).expect("q");
            let k = nn::matmul_bias_i7(black_box(&h), black_box(&wk), None).expect("k");
            let v =
                nn::matmul_bias_i7(black_box(&h), black_box(&wv), Some(black_box(&bv))).expect("v");
            black_box(q.data[0] + k.data[1] + v.data[2])
        });
    });

    group.bench_function("shared_activation_1500x1280", |b| {
        b.iter(|| {
            let hq = nn::quantize_act_i7(black_box(&h));
            let q =
                nn::matmul_bias_i7_quantized(&hq, black_box(&wq), Some(black_box(&bq))).expect("q");
            let k = nn::matmul_bias_i7_quantized(&hq, black_box(&wk), None).expect("k");
            let v =
                nn::matmul_bias_i7_quantized(&hq, black_box(&wv), Some(black_box(&bv))).expect("v");
            black_box(q.data[0] + k.data[1] + v.data[2])
        });
    });

    let hq = nn::quantize_act_i7(&h);
    group.bench_function("headmajor_attention_1500x1280", |b| {
        b.iter(|| {
            let attn = nn::attention_from_i7_qkv(
                black_box(&hq),
                black_box(&wq),
                Some(black_box(&bq)),
                black_box(&wk),
                None,
                black_box(&wv),
                Some(black_box(&bv)),
                black_box(20),
            )
            .expect("headmajor qkv attention");
            black_box(attn.data[0])
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 7. layer_norm — vertical-SIMD per-row normalization (hermetic).
//     Direct instrument for the L5 lever: one encoder-window-shaped
//     `[1500, 384]` layer-norm. No model needed.
// ---------------------------------------------------------------------------

fn bench_layer_norm(c: &mut Criterion) {
    use franken_whisper::native_engine::nn;

    let (rows, cols) = (1500usize, 384usize); // a full tiny.en encoder window
    let mut lcg = Lcg::new(0x1a4e_7c0d);
    let w: Vec<f32> = (0..cols).map(|_| lcg.next_f32() * 0.5 + 1.0).collect();
    let b: Vec<f32> = (0..cols).map(|_| lcg.next_f32() * 0.1).collect();
    let base: Vec<f32> = (0..rows * cols).map(|_| lcg.next_f32()).collect();

    let mut group = c.benchmark_group("native_engine/layer_norm");
    group.throughput(criterion::Throughput::Elements((rows * cols) as u64));
    group.bench_function("layer_norm_1500x384", |bch| {
        bch.iter_batched_ref(
            || Mat::from_vec(rows, cols, base.clone()),
            |m| {
                nn::layer_norm(m, &w, &b, 1e-5);
                black_box(m.data[0])
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// GELU over a tiny.en-large encoder MLP hidden activation ([1500, 1536]).
/// No model needed — runs everywhere.
fn bench_gelu(c: &mut Criterion) {
    use franken_whisper::native_engine::nn;

    let (rows, cols) = (1500usize, 1536usize);
    let mut lcg = Lcg::new(0x9e1c_0e1f);
    let base: Vec<f32> = (0..rows * cols)
        .map(|_| lcg.next_f32() * 6.0 - 3.0)
        .collect();

    let mut group = c.benchmark_group("native_engine/gelu");
    group.throughput(criterion::Throughput::Elements((rows * cols) as u64));
    group.bench_function("gelu_1500x1536", |bch| {
        bch.iter_batched_ref(
            || Mat::from_vec(rows, cols, base.clone()),
            |m| {
                nn::gelu(m);
                black_box(m.data[0])
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Linear resampler: 30 s of 44.1 kHz mono → 16 kHz (the most common decode-path
/// resample; 16 kHz inputs skip it entirely). No model needed — runs everywhere.
fn bench_resample(c: &mut Criterion) {
    use franken_whisper::audio::resample_mono_linear;

    let src_rate = 44_100u32;
    let dst_rate = 16_000u32;
    let input = synthetic_audio((src_rate as usize) * 30, 0x5e5a_3b1c);

    let mut group = c.benchmark_group("native_engine/resample");
    group.throughput(criterion::Throughput::Elements(input.len() as u64));
    group.bench_function("resample_44k_to_16k_30s", |bch| {
        bch.iter(|| {
            let out =
                resample_mono_linear(black_box(&input), black_box(src_rate), black_box(dst_rate));
            black_box(out.len())
        });
    });

    group.finish();
}

/// Stereo → mono downmix of 30 s of interleaved 44.1 kHz audio (the decode-path
/// channel average). No model needed — runs everywhere.
fn bench_downmix(c: &mut Criterion) {
    use franken_whisper::audio::downmix_to_mono;

    let frames = 44_100usize * 30;
    let interleaved = synthetic_audio(frames * 2, 0x0d03_3a17);

    let mut group = c.benchmark_group("native_engine/downmix");
    group.throughput(criterion::Throughput::Elements(interleaved.len() as u64));
    group.bench_function("downmix_stereo_30s", |bch| {
        bch.iter(|| {
            let out = downmix_to_mono(black_box(&interleaved), black_box(2));
            black_box(out.len())
        });
    });

    group.finish();
}

fn legacy_clean_then_downmix_append(
    destination: &mut Vec<f32>,
    interleaved: &[f32],
    channels: usize,
) {
    use franken_whisper::audio::downmix_to_mono;

    let clean: Vec<f32> = interleaved
        .iter()
        .map(|&sample| if sample.is_finite() { sample } else { 0.0 })
        .collect();
    if channels <= 1 {
        destination.extend_from_slice(&clean);
    } else {
        destination.extend_from_slice(&downmix_to_mono(&clean, channels));
    }
}

/// Decoded-packet ingestion path: sanitize IEEE-float PCM and downmix 30 s of
/// stereo 44.1 kHz audio before resampling. No model needed — runs everywhere.
fn bench_sanitize_downmix(c: &mut Criterion) {
    use franken_whisper::audio::append_sanitized_downmix_to_mono;

    let frames = 44_100usize * 30;
    let interleaved = synthetic_audio(frames * 2, 0x35aa_51d7);

    let mut group = c.benchmark_group("native_engine/sanitize_downmix");
    group.throughput(criterion::Throughput::Elements(interleaved.len() as u64));
    group.bench_function("legacy_clean_then_downmix_stereo_30s", |bch| {
        bch.iter_batched_ref(
            || Vec::with_capacity(frames),
            |out| {
                legacy_clean_then_downmix_append(out, black_box(&interleaved), black_box(2));
                black_box(out.len())
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("fused_append_stereo_30s", |bch| {
        bch.iter_batched_ref(
            || Vec::with_capacity(frames),
            |out| {
                append_sanitized_downmix_to_mono(out, black_box(&interleaved), black_box(2));
                black_box(out.len())
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Tokenizer construction (hermetic, real Whisper vocabulary cardinality)
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const TOKENIZER_SUPPRESS_SYMBOLS: &[&str] = &[
    "\"", "#", "(", ")", "*", "+", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^",
    "_", "`", "{", "|", "}", "~", "「", "」", "『", "』", "<<", ">>", "<<<", ">>>", "--",
    "---", "-(", "-[", "('", "(\"", "((", "))", "(((", ")))", "[[", "]]", "{{", "}}", "♪♪",
    "♪♪♪", "♩", "♪", "♫", "♬", "♭", "♮", "♯",
];

fn tokenizer_suppress_patterns() -> Vec<Vec<u8>> {
    let mut patterns = Vec::with_capacity(TOKENIZER_SUPPRESS_SYMBOLS.len() * 2 + 2);
    for symbol in TOKENIZER_SUPPRESS_SYMBOLS {
        patterns.push(symbol.as_bytes().to_vec());
        let mut prefixed = Vec::with_capacity(symbol.len() + 1);
        prefixed.push(b' ');
        prefixed.extend_from_slice(symbol.as_bytes());
        patterns.push(prefixed);
    }
    patterns.push(b" -".to_vec());
    patterns.push(b" '".to_vec());
    patterns
}

fn tokenizer_bench_vocab() -> Vec<Vec<u8>> {
    // ggml tiny.en / large-v3-turbo files store 50,257 base tokens; the
    // remaining declared ids are synthesized control/timestamp tokens.
    const N_FILE_TOKENS: usize = 50_257;
    let mut vocab: Vec<Vec<u8>> = (0..N_FILE_TOKENS)
        .map(|id| format!("token-{id:05}").into_bytes())
        .collect();
    let patterns = tokenizer_suppress_patterns();
    let stride = N_FILE_TOKENS / (patterns.len() + 1);
    for (index, pattern) in patterns.into_iter().enumerate() {
        vocab[(index + 1) * stride] = pattern;
    }
    vocab
}

fn tokenizer_prefilter_bench_vocab() -> Vec<Vec<u8>> {
    let mut vocab = tokenizer_bench_vocab();
    let patterns: HashSet<Vec<u8>> = tokenizer_suppress_patterns().into_iter().collect();
    // Profiled from the on-box ggml-tiny.en vocabulary: 881 / 50,257
    // entries (1.753%) have a first non-prefix-space byte that could match a
    // suppress pattern. Preserve all exact patterns and add non-matching quote
    // prefixes until this hermetic vocabulary has the same candidate count.
    let mut candidates = patterns.len();
    for (id, token) in vocab.iter_mut().enumerate() {
        if candidates >= 881 {
            break;
        }
        if patterns.contains(token.as_slice()) {
            continue;
        }
        *token = format!("\"candidate-{id:05}").into_bytes();
        candidates += 1;
    }
    assert_eq!(candidates, 881);
    vocab
}

fn tokenizer_bench_hparams() -> WhisperHParams {
    WhisperHParams {
        n_vocab: 51_866,
        n_audio_ctx: 1500,
        n_audio_state: 1280,
        n_audio_head: 20,
        n_audio_layer: 32,
        n_text_ctx: 448,
        n_text_state: 1280,
        n_text_head: 20,
        n_text_layer: 4,
        n_mels: 128,
        ftype: 1,
    }
}

fn legacy_build_non_speech(tokens: Vec<Vec<u8>>, patterns: &[Vec<u8>]) -> (Vec<Vec<u8>>, Vec<i32>) {
    let mut ids = Vec::new();
    for pattern in patterns {
        if let Some(id) = tokens.iter().position(|token| token == pattern)
            && let Ok(id) = i32::try_from(id)
            && !ids.contains(&id)
        {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    (tokens, ids)
}

fn single_scan_without_discriminant_prefilter(tokens: Vec<Vec<u8>>) -> (Vec<Vec<u8>>, Vec<i32>) {
    let patterns = tokenizer_suppress_patterns();
    let mut remaining = HashSet::with_capacity(patterns.len());
    remaining.extend(patterns);

    let mut ids = Vec::with_capacity(remaining.len());
    for (id, token) in tokens.iter().enumerate() {
        if remaining.remove(token.as_slice())
            && let Ok(id) = i32::try_from(id)
        {
            ids.push(id);
        }
        if remaining.is_empty() {
            break;
        }
    }
    (tokens, ids)
}

fn bench_tokenizer_from_vocab(c: &mut Criterion) {
    let hparams = tokenizer_bench_hparams();
    let vocab = tokenizer_bench_vocab();
    let patterns = tokenizer_suppress_patterns();
    let expected = legacy_build_non_speech(vocab.clone(), &patterns).1;
    let candidate = Tokenizer::from_vocab(&hparams, vocab.clone());
    assert_eq!(candidate.non_speech_tokens(), expected);

    let mut group = c.benchmark_group("native_engine/tokenizer_from_vocab");
    group.throughput(criterion::Throughput::Elements(vocab.len() as u64));
    group.bench_function("legacy_pattern_major", |bch| {
        bch.iter_batched(
            || vocab.clone(),
            |tokens| black_box(legacy_build_non_speech(tokens, black_box(&patterns))),
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("single_vocab_scan", |bch| {
        bch.iter_batched(
            || vocab.clone(),
            |tokens| black_box(Tokenizer::from_vocab(black_box(&hparams), tokens)),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_tokenizer_suppress_prefilter(c: &mut Criterion) {
    let hparams = tokenizer_bench_hparams();
    let vocab = tokenizer_prefilter_bench_vocab();
    let expected = single_scan_without_discriminant_prefilter(vocab.clone()).1;
    let candidate = Tokenizer::from_vocab(&hparams, vocab.clone());
    assert_eq!(candidate.non_speech_tokens(), expected);

    let mut group = c.benchmark_group("native_engine/tokenizer_suppress_prefilter");
    group.throughput(criterion::Throughput::Elements(vocab.len() as u64));
    group.bench_function("hash_every_token", |bch| {
        bch.iter_batched(
            || vocab.clone(),
            |tokens| black_box(single_scan_without_discriminant_prefilter(tokens)),
            criterion::BatchSize::SmallInput,
        );
    });
    group.bench_function("discriminant_prefilter", |bch| {
        bch.iter_batched(
            || vocab.clone(),
            |tokens| black_box(Tokenizer::from_vocab(black_box(&hparams), tokens)),
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn tokenizer_decode_with_owned_copy(tokenizer: &Tokenizer, ids: &[i32]) -> String {
    let mut bytes = Vec::new();
    for &id in ids {
        if tokenizer.is_special(id) {
            continue;
        }
        if let Some(token) = tokenizer.token_bytes(id) {
            bytes.extend_from_slice(token);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn bench_tokenizer_decode_utf8(c: &mut Criterion) {
    let hparams = tokenizer_bench_hparams();
    let tokenizer = Tokenizer::from_vocab(&hparams, tokenizer_bench_vocab());
    let ids: Vec<i32> = (0..256).collect();
    let expected = tokenizer_decode_with_owned_copy(&tokenizer, &ids);
    assert_eq!(tokenizer.decode(&ids), expected);

    let mut group = c.benchmark_group("native_engine/tokenizer_decode_utf8");
    group.throughput(criterion::Throughput::Bytes(expected.len() as u64));
    group.bench_function("lossy_owned_copy", |bch| {
        bch.iter(|| {
            black_box(tokenizer_decode_with_owned_copy(
                black_box(&tokenizer),
                black_box(&ids),
            ))
        });
    });
    group.bench_function("owned_vec_handoff", |bch| {
        bch.iter(|| black_box(black_box(&tokenizer).decode(black_box(&ids))));
    });
    group.finish();
}

fn timestamp_nodes(n: usize) -> Vec<Value> {
    (0..n)
        .map(|i| {
            let start = i as f64 * 0.25;
            let end = start + 0.2;
            match i % 7 {
                0 => json!({"start": start, "end": end}),
                1 => json!({"start_sec": start, "end_sec": end}),
                2 => json!({"timestamp": [start, end]}),
                3 => json!({"timestamp": {"0": start, "1": end}}),
                4 => json!({"timestamp": {"start": start, "end": end}}),
                5 => json!({"offsets": {"from": start * 1000.0, "to": end * 1000.0}}),
                _ => json!({
                    "offsets": {"from": "invalid", "to": "invalid"},
                    "start": start,
                    "end": end,
                }),
            }
        })
        .collect()
}

fn timestamp_number(value: &Value) -> Option<f64> {
    value.as_f64().or_else(|| value.as_i64().map(|v| v as f64))
}

fn pointer_timestamp_pair(node: &Value) -> (Option<f64>, Option<f64>) {
    let start = if let Some(value) = node.pointer("/offsets/from") {
        timestamp_number(value).map(|value| value / 1000.0)
    } else {
        node.get("start")
            .or_else(|| node.get("start_sec"))
            .or_else(|| node.pointer("/timestamp/0"))
            .or_else(|| node.pointer("/timestamp/start"))
            .and_then(timestamp_number)
    };
    let end = if let Some(value) = node.pointer("/offsets/to") {
        timestamp_number(value).map(|value| value / 1000.0)
    } else {
        node.get("end")
            .or_else(|| node.get("end_sec"))
            .or_else(|| node.pointer("/timestamp/1"))
            .or_else(|| node.pointer("/timestamp/end"))
            .and_then(timestamp_number)
    };
    (start, end)
}

fn direct_timestamp_value<'a>(node: &'a Value, index: usize, key: &str) -> Option<&'a Value> {
    let timestamp = node.get("timestamp")?;
    timestamp.get(index).or_else(|| timestamp.get(key))
}

fn direct_timestamp_pair(node: &Value) -> (Option<f64>, Option<f64>) {
    let start = if let Some(value) = node.get("offsets").and_then(|offsets| offsets.get("from")) {
        timestamp_number(value).map(|value| value / 1000.0)
    } else {
        node.get("start")
            .or_else(|| node.get("start_sec"))
            .or_else(|| direct_timestamp_value(node, 0, "0"))
            .or_else(|| {
                node.get("timestamp")
                    .and_then(|timestamp| timestamp.get("start"))
            })
            .and_then(timestamp_number)
    };
    let end = if let Some(value) = node.get("offsets").and_then(|offsets| offsets.get("to")) {
        timestamp_number(value).map(|value| value / 1000.0)
    } else {
        node.get("end")
            .or_else(|| node.get("end_sec"))
            .or_else(|| direct_timestamp_value(node, 1, "1"))
            .or_else(|| {
                node.get("timestamp")
                    .and_then(|timestamp| timestamp.get("end"))
            })
            .and_then(timestamp_number)
    };
    (start, end)
}

fn bench_timestamp_lookup_ab(c: &mut Criterion) {
    let nodes = timestamp_nodes(500);
    let reference: Vec<_> = nodes.iter().map(pointer_timestamp_pair).collect();
    let direct: Vec<_> = nodes.iter().map(direct_timestamp_pair).collect();
    assert_eq!(
        direct, reference,
        "direct lookups must preserve pointer semantics"
    );

    let mut group = c.benchmark_group("native_engine/timestamp_lookup_ab");
    group.bench_function("pointer_reference", |b| {
        b.iter(|| {
            let mut checksum = 0.0;
            for node in &nodes {
                let (start, end) = pointer_timestamp_pair(black_box(node));
                checksum += start.unwrap_or_default() + end.unwrap_or_default();
            }
            black_box(checksum)
        });
    });
    group.bench_function("direct_fields", |b| {
        b.iter(|| {
            let mut checksum = 0.0;
            for node in &nodes {
                let (start, end) = direct_timestamp_pair(black_box(node));
                checksum += start.unwrap_or_default() + end.unwrap_or_default();
            }
            black_box(checksum)
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_mel_30s,
    bench_mel_30s_realistic,
    bench_model_residency_tiny,
    bench_chunk_frames,
    bench_window_to_time_major,
    bench_encoder_window_tiny,
    bench_encoder_window_large,
    bench_decoder_token_step_tiny,
    bench_decoder_token_step_large,
    bench_self_attn_column_keys,
    bench_logits_gemv_large,
    bench_f16_gemv_dequant,
    bench_i7_qkv_activation_reuse,
    bench_layer_norm,
    bench_gelu,
    bench_resample,
    bench_downmix,
    bench_sanitize_downmix,
    bench_tokenizer_from_vocab,
    bench_tokenizer_suppress_prefilter,
    bench_tokenizer_decode_utf8,
    bench_timestamp_lookup_ab,
    bench_e2e_tiny_jfk,
    bench_e2e_tiny_jfk_no_timestamps,
    bench_e2e_large_jfk,
);
criterion_main!(benches);
