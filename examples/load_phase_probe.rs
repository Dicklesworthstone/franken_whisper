//! Model-load phase breakdown under a `--threads N` compute pool (bd-ct0y).
//!
//! Runs the load the way a backend run does — inside
//! `native_engine::with_compute_threads(N)` — and reports, per phase, wall
//! time, minor page faults, and user / system CPU time read from
//! `/proc/self/stat` (Linux; 10 ms CPU resolution). System time is the kernel
//! side of the load (page faults on fresh buffers, page-cache copies); user
//! time is conversion and repacking.
//!
//! Modes:
//! - `resident`: whole-file `ggml::read_blob_parallel` + `GgmlModel::from_bytes`
//!   (the `FW_STREAM_LOAD=0` path), then the encoder ∥ decoder weight build.
//! - `stream`: `GgmlModel::load` (on unix the streaming loader unless
//!   `FW_STREAM_LOAD=0`), then the encoder ∥ decoder weight build.
//! - `split`: resident read, then the encoder and the decoder builds one
//!   after the other, timed separately.
//!
//! Usage: `load_phase_probe <ggml-model-path> <threads> [runs] [resident|stream|split]`

use std::time::Instant;

use franken_whisper::native_engine::decoder::DecoderWeights;
use franken_whisper::native_engine::encoder::EncoderWeights;
use franken_whisper::native_engine::ggml::{self, GgmlModel};
use franken_whisper::native_engine::with_compute_threads;

#[derive(Clone, Copy)]
struct Snap {
    at: Instant,
    minflt: u64,
    utime: u64,
    stime: u64,
}

/// Process-wide counters (all threads): minor faults and user/system CPU in
/// clock ticks (fields 10, 14 and 15 of `/proc/self/stat`).
fn snap() -> Snap {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let fields: Vec<u64> = stat
        .rsplit_once(')')
        .map(|(_, rest)| rest)
        .unwrap_or_default()
        .split_whitespace()
        .map(|f| f.parse().unwrap_or(0))
        .collect();
    let field = |n: usize| fields.get(n - 3).copied().unwrap_or(0);
    Snap {
        at: Instant::now(),
        minflt: field(10),
        utime: field(14),
        stime: field(15),
    }
}

fn report(label: &str, a: Snap, b: Snap) {
    eprintln!(
        "{label:<8} wall {:8.1} ms  minflt {:8}  user {:6} ms  sys {:6} ms",
        b.at.duration_since(a.at).as_secs_f64() * 1e3,
        b.minflt - a.minflt,
        (b.utime - a.utime) * 10,
        (b.stime - a.stime) * 10
    );
}

fn load(path: &std::path::Path, mode: &str) -> Result<(), String> {
    let start = snap();
    let model = if mode == "stream" {
        GgmlModel::load(path).map_err(|e| e.to_string())?
    } else {
        let blob = ggml::read_blob_parallel(path).map_err(|e| e.to_string())?;
        report("read", start, snap());
        GgmlModel::from_bytes(blob).map_err(|e| e.to_string())?
    };
    let parsed = snap();
    report("parse", start, parsed);
    if mode == "split" {
        let encoder = EncoderWeights::from_ggml(&model).map_err(|e| e.to_string())?;
        let built = snap();
        report("encoder", parsed, built);
        let decoder = DecoderWeights::from_ggml(&model).map_err(|e| e.to_string())?;
        report("decoder", built, snap());
        std::hint::black_box((&encoder, &decoder));
    } else {
        let (encoder, decoder) = rayon::join(
            || EncoderWeights::from_ggml(&model),
            || DecoderWeights::from_ggml(&model),
        );
        report("weights", parsed, snap());
        let built = (
            encoder.map_err(|e| e.to_string())?,
            decoder.map_err(|e| e.to_string())?,
        );
        std::hint::black_box(&built);
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = std::path::PathBuf::from(args.next().ok_or("model path required")?);
    let threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let runs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let mode = args.next().unwrap_or_else(|| "resident".to_owned());
    if !matches!(mode.as_str(), "resident" | "stream" | "split") {
        return Err(format!("unknown mode {mode:?} (resident|stream|split)").into());
    }
    for run in 0..runs {
        eprintln!("-- run {run} threads {threads} mode {mode}");
        with_compute_threads(threads, || load(&path, &mode))??;
    }
    Ok(())
}
