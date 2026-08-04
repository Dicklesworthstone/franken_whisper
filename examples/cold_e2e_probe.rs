//! Integrated COLD-CLI e2e probe (cc, 2026-06-29).
//!
//! Times the realistic one-shot path a CLI user pays: a cold
//! `NativeWhisperModel::load` (cache empty) immediately followed by the FIRST
//! `transcribe` (cold allocator + first-touch page faults on the f32 weights) —
//! the true cold UX, which the warmed-up `e2e_*_jfk` criterion min does NOT
//! capture. One load+transcribe per process; run the binary N times for N cold
//! samples. Run with `FRANKEN_WHISPER_MODEL_DIR` set.
//!
//! Usage: `cold_e2e_probe <model-short-name>`

use std::time::Instant;

use franken_whisper::native_engine::decode::DecodeParams;
use franken_whisper::native_engine::{NativeWhisperModel, find_model_file};

fn read_jfk() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 {
        return Err("jfk.wav must be mono 16 kHz".into());
    }
    Ok(reader
        .samples::<i16>()
        .map(|s| s.map(|v| f32::from(v) / 32768.0))
        .collect::<Result<_, _>>()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "large-v3-turbo".to_string());
    let path = find_model_file(&model)
        .ok_or_else(|| format!("model {model} not found (set FRANKEN_WHISPER_MODEL_DIR)"))?;
    let samples = read_jfk()?;
    let params = DecodeParams {
        language: None,
        translate: false,
        timestamps: false,
        n_threads: 8,
        ..DecodeParams::default()
    };
    let noop = || Ok(());

    let t = Instant::now();
    let m = NativeWhisperModel::load(&path)?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let out = m.transcribe(&samples, &params, &noop)?;
    let trans_ms = t.elapsed().as_secs_f64() * 1e3;

    eprintln!(
        "COLD ({model}): load {load_ms:.1} ms | transcribe {trans_ms:.1} ms | TOTAL {:.1} ms | segs {}",
        load_ms + trans_ms,
        out.segments.len()
    );
    Ok(())
}
