//! Turbo ENCODER thread-scaling probe (BlackThrush, 2026-07-02).
//!
//! The global rayon pool is capped at `host_parallelism().min(16)` (mod.rs
//! `default_threads`), so on a 64-core box the encoder sgemm — ~82% of a turbo
//! transcribe — runs on only 16 cores. matmul thread-count does NOT change the
//! per-element k-accumulation order, so raising it is BYTE-EXACT. This probe
//! loads large-v3-turbo, builds one full 30 s window, and times
//! `encoder::forward` (min-of-N, contention-robust). Control the pool EXTERNALLY
//! so the "16" run mirrors the real transcribe default:
//!   for t in 16 32 48 64; do RAYON_NUM_THREADS=$t encoder_scale_probe 12; done
//!
//! Needs `FRANKEN_WHISPER_MODEL_DIR` (dir holding ggml-large-v3-turbo.bin).
//! Usage: `encoder_scale_probe [iters]`  (default 12).
use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::encoder;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::mel::{self, FRAMES_PER_CHUNK, N_SAMPLES_30S, SAMPLE_RATE};
use std::time::Instant;

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let path = find_model_file("large-v3-turbo")
        .expect("set FRANKEN_WHISPER_MODEL_DIR to the dir holding ggml-large-v3-turbo.bin");
    let model = GgmlModel::load(&path)
        .and_then(LoadedModel::from_ggml)
        .expect("load large-v3-turbo");

    let sr = SAMPLE_RATE as f32;
    let audio: Vec<f32> = (0..N_SAMPLES_30S)
        .map(|i| {
            let t = i as f32 / sr;
            0.9 * (0.5 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin())
        })
        .collect();
    let full = mel::log_mel(&audio, &model.filters, 8).expect("log_mel");
    let window = mel::chunk_frames(&full, 0, FRAMES_PER_CHUNK);
    let noop = || Ok(());

    // Warm the pool + caches.
    for _ in 0..2 {
        let enc = encoder::forward(&model.encoder, &window, 8, &noop).expect("encoder");
        std::hint::black_box(enc.rows);
    }
    let threads = rayon::current_num_threads();
    let mut best = f64::INFINITY;
    let mut acc = 0usize;
    for _ in 0..iters {
        let t = Instant::now();
        let enc = encoder::forward(&model.encoder, &window, 8, &noop).expect("encoder");
        best = best.min(t.elapsed().as_secs_f64());
        acc = acc.wrapping_add(enc.rows + enc.cols);
    }
    std::hint::black_box(acc);
    println!(
        "rayon_threads={threads:<3} turbo encoder::forward best {:.1} ms/window  (iters={iters})",
        best * 1e3
    );
}
