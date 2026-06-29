//! Isolated encoder weight-load probe (cc, 2026-06-29).
//!
//! Times `EncoderWeights::from_ggml` over a ggml model that is parsed exactly
//! ONCE up front, so the measurement isolates the dequant + transpose
//! (`model_weights`) — the biggest cold-start gap vs whisper.cpp — from file
//! I/O (`model_parse`). Reports the BEST of N runs (load is memory-bound; min
//! is the cleanest estimator under a contended box).
//!
//! Usage: `encoder_load_probe <model-short-name> [runs]`
//! Needs `FRANKEN_WHISPER_MODEL_DIR` pointing at the ggml model dir.

use std::time::Instant;

use franken_whisper::native_engine::encoder::EncoderWeights;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "large-v3-turbo".to_string());
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(7);

    let path = find_model_file(&model)
        .ok_or_else(|| format!("model {model} not found (set FRANKEN_WHISPER_MODEL_DIR)"))?;
    let ggml = GgmlModel::load(&path)?; // parse the 1.5 GB blob ONCE

    let mut best = f64::INFINITY;
    for r in 0..runs {
        let t = Instant::now();
        let enc = EncoderWeights::from_ggml(&ggml)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&enc);
        best = best.min(ms);
        eprintln!("encoder_from_ggml run {r}: {ms:.1} ms");
    }
    eprintln!("BEST encoder_from_ggml ({model}): {best:.1} ms");
    Ok(())
}
