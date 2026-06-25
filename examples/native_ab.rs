//! End-to-end A/B + conformance harness for the native engine.
//!
//! Loads a whisper ggml model by short name, transcribes the jfk fixture
//! `runs` times, and prints per-run wall times plus a golden-comparable JSON
//! (`{transcript, segments:[[start,end,text]]}`). Used by the round-2 f16
//! optimization pass for:
//!   * interleaved wall A/B vs a pre-change REF binary (min/p25 of N runs), and
//!   * the exact-transcript conformance gate against
//!     `/tmp/fw_golden/{tiny.en,large-v3-turbo}.json`.
//!
//! It exercises the same `transcribe_samples` path production uses, so the
//! `FRANKEN_WHISPER_NATIVE_F16_COMPUTE` runtime switch is honored exactly as in
//! the real engine. Usage:
//!   native_ab <tiny.en|large-v3-turbo> [runs] [threads]
//! Audio is the in-repo `tests/fixtures/native/jfk.wav` (mono 16 kHz).

use std::error::Error;
use std::fmt::Write as _;
use std::io::{Error as IoError, ErrorKind};
use std::time::Instant;

use franken_whisper::native_engine::decode::{DecodeParams, LoadedModel, transcribe_samples};
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;

fn load_jfk_samples() -> Result<Vec<f32>, Box<dyn Error>> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/native/jfk.wav");
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(IoError::new(ErrorKind::InvalidData, "jfk.wav must be mono").into());
    }
    if spec.sample_rate != 16_000 {
        return Err(IoError::new(ErrorKind::InvalidData, "jfk.wav must be 16 kHz").into());
    }
    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.map(|s| f32::from(s) / 32768.0))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(samples)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let model_name = args.next().unwrap_or_else(|| "tiny.en".to_string());
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let n_threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);

    let path = find_model_file(&model_name).ok_or_else(|| {
        IoError::new(
            ErrorKind::NotFound,
            format!("model {model_name} not found in search dirs"),
        )
    })?;
    let model = LoadedModel::from_ggml(GgmlModel::load(&path)?)?;
    let samples = load_jfk_samples()?;

    let params = DecodeParams {
        language: None,
        translate: false,
        timestamps: true,
        n_threads,
        max_text_ctx: None,
        ..DecodeParams::default()
    };
    let noop = || Ok(());

    let mut last = None;
    for r in 0..runs {
        let t = Instant::now();
        let out = transcribe_samples(&model, &samples, &params, &noop)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("RUN {r} threads={n_threads} wall_ms={ms:.2}");
        last = Some(out);
    }

    let out = last.ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            "runs must be at least one to emit a transcript",
        )
    })?;
    let transcript: String = out
        .segments
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    // Emit a golden-comparable JSON document on stdout (segments rounded to the
    // same 2-decimal precision the golden files use).
    let mut json = String::new();
    json.push_str("{\n \"transcript\": ");
    push_json_str(&mut json, &transcript);
    json.push_str(",\n \"segments\": [\n");
    for (i, s) in out.segments.iter().enumerate() {
        let start = fmt_secs(round2(s.start_sec.unwrap_or(0.0)));
        let end = fmt_secs(round2(s.end_sec.unwrap_or(0.0)));
        json.push_str("  [\n");
        write!(&mut json, "   {start},\n   {end},\n   ")?;
        push_json_str(&mut json, s.text.trim());
        json.push_str("\n  ]");
        if i + 1 < out.segments.len() {
            json.push(',');
        }
        json.push('\n');
    }
    json.push_str(" ]\n}");
    println!("{json}");
    Ok(())
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Render a seconds value with at least one fractional digit so whole numbers
/// serialize as `0.0` (matching the golden files) rather than `0`. Rust's
/// default `f64` Display drops the fraction for integral values; the golden
/// gate is byte-exact, so we restore the trailing `.0` here.
fn fmt_secs(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{v:.1}")
    } else {
        // Trim to the shortest representation that round-trips at 2-decimal
        // precision, matching `round2` (e.g. 10.4, 8.99, 10.99).
        let s = format!("{v}");
        if s.contains('.') {
            s
        } else {
            format!("{v:.1}")
        }
    }
}

fn push_json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out.push('"');
}
