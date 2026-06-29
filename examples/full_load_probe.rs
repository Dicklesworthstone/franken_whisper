//! Decoder-load A/B probe (cc, 2026-06-29).
//!
//! Times `DecoderWeights::from_ggml` over a pre-parsed ggml (file I/O excluded),
//! best of N. Toggle the cross-layer-parallel load with `FW_DEC_PAR_LAYERS=0|1`
//! (read once per process) and run both for an interleaved A/B. Run with
//! `FRANKEN_WHISPER_MODEL_DIR` set.
//!
//! Usage: `full_load_probe <model-short-name> [runs]`

use std::time::Instant;

use franken_whisper::native_engine::decoder::DecoderWeights;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "large-v3-turbo".to_string());
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(7);

    let path = find_model_file(&model)
        .ok_or_else(|| format!("model {model} not found (set FRANKEN_WHISPER_MODEL_DIR)"))?;
    let ggml = GgmlModel::load(&path)?; // parse the blob ONCE

    let mut best = f64::INFINITY;
    for _ in 0..runs {
        let t = Instant::now();
        let dec = DecoderWeights::from_ggml(&ggml)?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&dec);
    }
    eprintln!("BEST decoder_from_ggml ({model}): {best:.1} ms");
    Ok(())
}
