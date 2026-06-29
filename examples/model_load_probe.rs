//! Full model-load breakdown + parallel-read A/B probe (cc, 2026-06-29).
//!
//! `parse` (the 1.5 GB `std::fs::read`) is the biggest cold-start load phase.
//! This A/Bs the serial `std::fs::read` against the new parallel
//! `ggml::read_blob_parallel` (same bytes, asserted), best of N under identical
//! contention, and also reports the post-fix `from_ggml` phases for context.
//! Run with `FRANKEN_WHISPER_MODEL_DIR` set.
//!
//! Usage: `model_load_probe <model-short-name> [runs]`

use std::time::Instant;

use franken_whisper::native_engine::decoder::DecoderWeights;
use franken_whisper::native_engine::encoder::EncoderWeights;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::{self, GgmlModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "large-v3-turbo".to_string());
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(7);

    let path = find_model_file(&model)
        .ok_or_else(|| format!("model {model} not found (set FRANKEN_WHISPER_MODEL_DIR)"))?;

    // Byte-identity check (once) + interleaved A/B best-of-N.
    let serial0 = std::fs::read(&path)?;
    let parallel0 = ggml::read_blob_parallel(&path)?;
    assert_eq!(
        serial0, parallel0,
        "parallel read produced different bytes than std::fs::read"
    );
    eprintln!("byte-identity: OK ({} bytes)", serial0.len());
    drop((serial0, parallel0));

    let (mut ser_best, mut par_best) = (f64::INFINITY, f64::INFINITY);
    for r in 0..runs {
        let t = Instant::now();
        let s = std::fs::read(&path)?;
        let ser_ms = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(s.len());
        drop(s);

        let t = Instant::now();
        let p = ggml::read_blob_parallel(&path)?;
        let par_ms = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(p.len());
        drop(p);

        ser_best = ser_best.min(ser_ms);
        par_best = par_best.min(par_ms);
        eprintln!("run {r}: serial fs::read {ser_ms:.1} ms | parallel read_at {par_ms:.1} ms");
    }

    // Post-fix from_ggml phases (context).
    let ggml_model = GgmlModel::load(&path)?;
    let t = Instant::now();
    let enc = EncoderWeights::from_ggml(&ggml_model)?;
    let enc_ms = t.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(&enc);
    let t = Instant::now();
    let dec = DecoderWeights::from_ggml(&ggml_model)?;
    let dec_ms = t.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(&dec);

    eprintln!(
        "BEST ({model}): serial_read {ser_best:.1} ms | parallel_read {par_best:.1} ms | ratio {:.2}x | (encoder_from_ggml {enc_ms:.1} | decoder_from_ggml {dec_ms:.1})",
        ser_best / par_best
    );
    Ok(())
}
