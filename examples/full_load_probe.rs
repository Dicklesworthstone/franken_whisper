//! LoadedModel::from_ggml A/B probe (cc, 2026-06-29).
//!
//! Times `LoadedModel::from_ggml` (tokenizer + encoder + decoder weight build)
//! over a pre-parsed, cloned ggml (file I/O excluded), best of N. Toggle the
//! concurrent encoder||decoder load with `FW_CONCURRENT_LOAD=0|1` (read once per
//! process) for an interleaved A/B. Run with `FRANKEN_WHISPER_MODEL_DIR` set.
//!
//! Usage: `full_load_probe <model-short-name> [runs]`

use std::time::Instant;

use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "large-v3-turbo".to_string());
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(7);

    let path = find_model_file(&model)
        .ok_or_else(|| format!("model {model} not found (set FRANKEN_WHISPER_MODEL_DIR)"))?;

    let mut best = f64::INFINITY;
    for _ in 0..runs {
        // Re-parse each iter (UNTIMED): from_ggml consumes the GgmlModel, which is
        // not Clone. The file is page-cached so this is a fast warm read+parse.
        let g = GgmlModel::load(&path)?;
        let t = Instant::now();
        let loaded = LoadedModel::from_ggml(g)?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&loaded);
    }
    eprintln!("BEST LoadedModel::from_ggml ({model}): {best:.1} ms");
    Ok(())
}
