//! Scale A/B harness: sequential (whisper-cpp-native) vs insanely-fast-native
//! over a long WAV, for round-3 pass-3 (contiguous-range + thread-budget work).
//!
//! Runs the REAL public `Engine::run` for both engines over a given 16 kHz mono
//! WAV and reports wall time, word count, and the IF-vs-seq word-diff rate
//! (the same coarse multiset metric the scale baseline uses). Honest A/B: it
//! prints both walls so the caller can compute the ratio; the model is loaded
//! cold per engine the first time and warm thereafter (the global model cache is
//! process-wide), so run seq first then IF, or pass `--warm` to do a throwaway
//! load first.
//!
//! Usage:
//!   native_if_scale <model> <file.wav> [--workers N]
//! `--workers N` forces the IF batch_size (and thus the worker ceiling) to N for
//! the IF run; omit to use the engine's measured default policy.

use std::path::Path;
use std::time::{Duration, Instant};

use franken_whisper::backend::{Engine, InsanelyFastNativeEngine, WhisperCppNativeEngine};
use franken_whisper::model::{
    BackendKind, BackendParams, InputSource, TranscribeRequest, TranscriptionResult,
};

fn transcript_words(r: &TranscriptionResult) -> Vec<String> {
    r.transcript
        .split_whitespace()
        .map(str::to_lowercase)
        .collect()
}

fn word_diff_rate(a: &[String], b: &[String]) -> f64 {
    use std::collections::HashMap;
    fn count(ws: &[String]) -> HashMap<&String, i64> {
        let mut m: HashMap<&String, i64> = HashMap::new();
        for w in ws {
            *m.entry(w).or_default() += 1;
        }
        m
    }
    let ma = count(a);
    let mb = count(b);
    let mut keys: std::collections::HashSet<&String> = ma.keys().copied().collect();
    keys.extend(mb.keys().copied());
    let diff: i64 = keys
        .iter()
        .map(|k| (ma.get(*k).copied().unwrap_or(0) - mb.get(*k).copied().unwrap_or(0)).abs())
        .sum();
    let total = a.len().max(b.len());
    if total == 0 {
        0.0
    } else {
        diff as f64 / total as f64
    }
}

/// Sequence-aware word-diff: `1 - LCS(a,b) / max(|a|,|b|)` — the fraction of
/// words NOT in the longest common subsequence (order-sensitive, the honest
/// transcript-divergence measure).
fn lcs_diff_rate(a: &[String], b: &[String]) -> f64 {
    let (n, m) = (a.len(), b.len());
    if n == 0 && m == 0 {
        return 0.0;
    }
    let mut prev = vec![0usize; m + 1];
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        for j in 1..=m {
            cur[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1] + 1
            } else {
                prev[j].max(cur[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let lcs = prev[m];
    1.0 - lcs as f64 / n.max(m) as f64
}

fn req(model: &str, wav: &Path, batch_size: Option<u32>) -> TranscribeRequest {
    TranscribeRequest {
        input: InputSource::File {
            path: wav.to_path_buf(),
        },
        backend: BackendKind::InsanelyFast,
        model: Some(model.to_owned()),
        language: None,
        translate: false,
        diarize: false,
        persist: false,
        db_path: std::path::PathBuf::from("state.sqlite3"),
        timeout_ms: None,
        backend_params: BackendParams {
            batch_size,
            ..BackendParams::default()
        },
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: native_if_scale <model> <file.wav> [--workers N]");
    let wav_s = args
        .next()
        .expect("usage: native_if_scale <model> <file.wav> [--workers N]");
    let wav = Path::new(&wav_s);
    let mut workers: Option<u32> = None;
    while let Some(a) = args.next() {
        if a == "--workers" {
            workers = args.next().and_then(|s| s.parse().ok());
        }
    }

    let work = std::env::temp_dir();
    let timeout = Duration::from_secs(36_000);

    // Sequential reference.
    let seq_engine = WhisperCppNativeEngine;
    let r = req(&model, wav, None);
    let t = Instant::now();
    let seq = Engine::run(&seq_engine, &r, wav, &work, timeout, None).expect("seq run");
    let seq_ms = t.elapsed().as_secs_f64() * 1e3;
    let seq_words = transcript_words(&seq);

    // Insanely-fast (contiguous-range) — model already warm in the cache.
    let if_engine = InsanelyFastNativeEngine;
    let r = req(&model, wav, workers);
    let t = Instant::now();
    let iff = Engine::run(&if_engine, &r, wav, &work, timeout, None).expect("if run");
    let if_ms = t.elapsed().as_secs_f64() * 1e3;
    let if_words = transcript_words(&iff);

    let rate = word_diff_rate(&if_words, &seq_words);
    let lcs_rate = lcs_diff_rate(&if_words, &seq_words);
    let ratio = if_ms / seq_ms;
    let n_workers = iff.raw_output["workers"].as_u64().unwrap_or(0);
    let tpw = iff.raw_output["threads_per_worker"].as_u64().unwrap_or(0);
    let seams = iff.raw_output["seams"].as_u64().unwrap_or(0);
    let n_ranges = iff.raw_output["ranges"]
        .as_array()
        .map_or(0, std::vec::Vec::len);

    if let Ok(dir) = std::env::var("DUMP_DIR") {
        let _ = std::fs::write(format!("{dir}/seq_{model}.txt"), &seq.transcript);
        let _ = std::fs::write(format!("{dir}/if_{model}.txt"), &iff.transcript);
    }

    println!(
        "{{\"model\":\"{model}\",\"seq_ms\":{seq_ms:.1},\"if_ms\":{if_ms:.1},\
         \"wall_ratio_if_over_seq\":{ratio:.4},\"seq_words\":{},\"if_words\":{},\
         \"word_diff_rate\":{rate:.4},\"lcs_diff_rate\":{lcs_rate:.4},\"if_workers\":{n_workers},\
         \"if_threads_per_worker\":{tpw},\"if_ranges\":{n_ranges},\"if_seams\":{seams}}}",
        seq_words.len(),
        if_words.len(),
    );
}
