//! Full-load phase breakdown (cc, 2026-06-29).
//!
//! After the 3 landed load wins (parallel read + encoder fused-transpose +
//! decoder one-pass f16), this isolates any REMAINING serial cost:
//!   parse  = `GgmlModel::load` (= parallel read ~120 ms + the vocab/filter/tensor
//!            parse loops)
//!   full   = `NativeWhisperModel::load` (= parse + encoder + decoder + tokenizer)
//! ⇒ weights(enc+dec+tok) = full − parse; parse_loops = parse − read(120);
//!   tokenizer = weights − enc(342) − dec(231).
//! Cache is evicted each iter by dropping the Arc (load() keeps only a Weak), so
//! every measurement is a cold re-parse. Best of N. Run with MODEL_DIR set.

use std::time::Instant;

use franken_whisper::native_engine::decode::LoadedModel;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "large-v3-turbo".to_string());
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);

    let path = find_model_file(&model)
        .ok_or_else(|| format!("model {model} not found (set FRANKEN_WHISPER_MODEL_DIR)"))?;

    // `LoadedModel::from_ggml` (no cache, unlike NativeWhisperModel::load) is the
    // weights phase: encoder + decoder + tokenizer. enc(~342) + dec(~231) are
    // known, so tokenizer = weights − 573.
    let (mut parse_best, mut weights_best) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..runs {
        let t = Instant::now();
        let g = GgmlModel::load(&path)?;
        parse_best = parse_best.min(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        let loaded = LoadedModel::from_ggml(g)?;
        weights_best = weights_best.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&loaded);
        drop(loaded);
    }

    eprintln!(
        "BEST ({model}): parse {parse_best:.1} ms | weights(enc+dec+tok) {weights_best:.1} ms | implied tokenizer = weights − enc(342) − dec(231) ≈ {:.1} ms",
        weights_best - 342.0 - 231.0
    );
    Ok(())
}
