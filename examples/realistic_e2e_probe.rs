//! Realistic multi-window e2e probe (cc, 2026-06-29).
//!
//! All prior head-to-heads used single-window jfk (11 s). This measures the real
//! workload: a multi-minute MP3 → native symphonia decode → cold model load →
//! multi-window `transcribe`. Compares franken's full realistic path
//! (decode + load + transcribe) against whisper.cpp on the same file. Exercises
//! the unmeasured axes: per-window scaling and the native-decode-vs-ffmpeg edge.
//!
//! Usage: `realistic_e2e_probe <audio-file> [model-short-name]`

use std::path::Path;
use std::time::Instant;

use franken_whisper::audio::normalize_to_wav;
use franken_whisper::native_engine::decode::DecodeParams;
use franken_whisper::native_engine::{NativeWhisperModel, find_model_file};

fn read_wav(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 16_000 {
        return Err(format!(
            "expected mono 16 kHz, got {}ch @ {}Hz",
            spec.channels, spec.sample_rate
        )
        .into());
    }
    Ok(reader
        .samples::<i16>()
        .map(|s| s.map(|v| f32::from(v) / 32768.0))
        .collect::<Result<_, _>>()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let audio = args
        .next()
        .ok_or("usage: realistic_e2e_probe <audio-file> [model]")?;
    let model = args.next().unwrap_or_else(|| "large-v3-turbo".to_string());

    let work_dir = std::env::var("FW_PROBE_WORKDIR")
        .unwrap_or_else(|_| "/data/tmp/claude-1000/realistic_probe_work".to_string());
    std::fs::create_dir_all(&work_dir)?;

    // 1. Native decode + resample to 16 kHz mono (symphonia primary path).
    let t = Instant::now();
    let wav = normalize_to_wav(Path::new(&audio), Path::new(&work_dir))?;
    let decode_ms = t.elapsed().as_secs_f64() * 1e3;
    let samples = read_wav(&wav)?;
    let secs = samples.len() as f64 / 16_000.0;

    // 2. Cold model load.
    let path = find_model_file(&model).ok_or("model not found (set FRANKEN_WHISPER_MODEL_DIR)")?;
    let t = Instant::now();
    let m = NativeWhisperModel::load(&path)?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;

    // 3. Multi-window transcribe.
    let params = DecodeParams {
        language: None,
        translate: false,
        timestamps: false,
        n_threads: 8,
        ..DecodeParams::default()
    };
    let noop = || Ok(());
    let t = Instant::now();
    let out = m.transcribe(&samples, &params, &noop)?;
    let trans_ms = t.elapsed().as_secs_f64() * 1e3;

    eprintln!(
        "REALISTIC ({model}, {secs:.1}s audio): decode {decode_ms:.0} + load {load_ms:.0} + transcribe {trans_ms:.0} = TOTAL {:.0} ms | segs {}",
        decode_ms + load_ms + trans_ms,
        out.segments.len()
    );
    Ok(())
}
