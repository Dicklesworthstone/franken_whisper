//! Contention-immune ENCODER perf probe.
//!
//! Loads `tiny.en` once, builds one mel window once, then loops a FIXED number of
//! `encoder::forward` calls so `perf stat -e instructions:u` / `perf record`
//! isolates the encoder's per-window cost. The encoder is the largest e2e slice
//! and is GEMM-bound (external `ft_kernel_cpu`); this probe surfaces how much is
//! the external sgemm vs franken-side work (layer_norm / softmax / gelu / residual
//! / allocation churn — the place a mel-style reuse lever could hide). Mirrors
//! `mel_perf_probe` / `decoder_perf_probe`. Needs `FRANKEN_WHISPER_MODEL_DIR`.
//!
//! Usage: `encoder_perf_probe [iters]`  (default 40).
use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::encoder;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::mel::{self, FRAMES_PER_CHUNK, N_SAMPLES_30S, SAMPLE_RATE};

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let path =
        find_model_file("tiny.en").expect("set FRANKEN_WHISPER_MODEL_DIR to the ggml models dir");
    let model = GgmlModel::load(&path)
        .and_then(LoadedModel::from_ggml)
        .expect("load tiny.en");

    // Synthetic 30 s audio → one full mel window (the encoder input). Encoder cost
    // is shape-determined, so synthetic content is representative.
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

    let mut acc = 0usize;
    for _ in 0..iters {
        let enc = encoder::forward(&model.encoder, &window, 8, &noop).expect("encoder");
        acc = acc.wrapping_add(enc.rows + enc.cols);
    }
    std::hint::black_box(acc);
    eprintln!("encoder_perf_probe: iters={iters} acc={acc}");
}
