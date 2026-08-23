//! Listen latency/quality A/B harness (bd-rt-latency-harness-3dkh).
//!
//! Drives the REAL `franken_whisper robot listen` binary as a subprocess —
//! the product, not a library shortcut — feeding paced s16le PCM over
//! `--source stdin-pcm`. The harness owns the pacing clock and logs every
//! write's wall time, so audio-time -> wall-time mapping is exact to the
//! 20 ms frame grid; NDJSON arrival times are stamped per line by a reader
//! thread. Latency numbers are therefore end-to-end through the shipped
//! pipe, including serialization and flush.
//!
//! ## Metrics (definitions locked by the bead)
//! - TTFT: wall(first transcript.delta arrival) - wall(injection of the
//!   audio sample at that utterance's speech_started time).
//! - Commit lag (per delta): arrival - wall(injection of audio t1_sec).
//! - Endpoint latency (per utterance): utterance_end arrival - wall(t1_sec).
//! - WER: joined utterance_end texts vs the fixture reference
//!   (`conformance::word_error_rate`, punctuation/case-normalized).
//! - Cost: session_stats mean/p95 step latency, delta/utterance counts.
//!
//! ## Method (AGENTS.md measurement contract)
//! - All arms + A/A nulls interleave in ONE invocation with order
//!   alternation per round. Null gate: per-arm A/A median commit-lag ratio
//!   must land in [0.98, 1.02] for a comparative verdict; otherwise rows
//!   are published as observations, not comparisons.
//! - Pace self-check is a runtime gate, not a mocked test: every feed
//!   asserts actual/nominal wall duration within +-1% or the session row
//!   is marked pace_violation and excluded from verdicts.
//! - No external incumbent arm here (whisper.cpp stream is OPTIONAL per
//!   the bead); all results are SELF-RELATIVE and labeled as such.
//!
//! ## Fixtures
//! Derived deterministically in-memory from the pinned public-domain
//! `tests/fixtures/native/jfk.wav` (16 kHz mono, 11 s) — no new binary
//! fixtures to license or commit:
//! - `short`:    first 6.9 s (single utterance).
//! - `pauses`:   the full clip (3 utterances, natural pauses).
//! - `long`:     the clip tiled 2x (22 s monologue shape).
//! - `noisy`:    full clip + seeded LCG noise at ~15 dB SNR.
//! - `negative`: 8 s of 440 Hz tone (music-only; expect zero output).
//!
//! The 2-minute timeout-path fixture is deferred to a campaign extension
//! (wall-cost; the timeout arm is covered by unit tests).
//!
//! Usage:
//!   listen_latency_ab --fw <path-to-binary> [--rounds 2] [--out report.json]
//!     [--fixtures short,pauses,long,noisy,negative] [--arms alignatt,endpoint-commit]

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use franken_whisper::conformance::word_error_rate;

const SAMPLE_RATE: usize = 16_000;
const FRAME: usize = 320; // 20 ms @ 16 kHz
const JFK: &str = "tests/fixtures/native/jfk.wav";
const JFK_REF: &str = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country";
const JFK_SHORT_REF: &str = "And so my fellow Americans ask not what your country can do";

struct Fixture {
    name: &'static str,
    samples: Vec<f32>,
    reference: String,
}

fn load_jfk() -> Vec<f32> {
    let mut reader = hound::WavReader::open(JFK).expect("open jfk.wav");
    let spec = reader.spec();
    assert_eq!(
        spec.sample_rate as usize, SAMPLE_RATE,
        "jfk.wav must be 16 kHz"
    );
    assert_eq!(spec.channels, 1, "jfk.wav must be mono");
    reader
        .samples::<i16>()
        .map(|s| f32::from(s.expect("pcm read")) / 32768.0)
        .collect()
}

fn build_fixtures(names: &[String]) -> Vec<Fixture> {
    let jfk = load_jfk();
    let mut out = Vec::new();
    for name in names {
        let fixture = match name.as_str() {
            "short" => Fixture {
                name: "short",
                samples: jfk[..(SAMPLE_RATE as f64 * 6.9) as usize].to_vec(),
                reference: JFK_SHORT_REF.to_owned(),
            },
            "pauses" => Fixture {
                name: "pauses",
                samples: jfk.clone(),
                reference: JFK_REF.to_owned(),
            },
            "long" => Fixture {
                name: "long",
                samples: [jfk.clone(), jfk.clone()].concat(),
                reference: format!("{JFK_REF} {JFK_REF}"),
            },
            "noisy" => {
                // Seeded LCG noise at ~15 dB SNR: deterministic across runs.
                let mut state: u64 = 0x5eed_cafe_f00d_1234;
                let speech_rms = (jfk.iter().map(|s| s * s).sum::<f32>() / jfk.len() as f32).sqrt();
                let noise_rms = speech_rms / 10f32.powf(15.0 / 20.0);
                let samples = jfk
                    .iter()
                    .map(|s| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let unit = ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
                        (s + unit * noise_rms * 1.732).clamp(-1.0, 1.0)
                    })
                    .collect();
                Fixture {
                    name: "noisy",
                    samples,
                    reference: JFK_REF.to_owned(),
                }
            }
            "negative" => Fixture {
                name: "negative",
                samples: (0..SAMPLE_RATE * 8)
                    .map(|i| {
                        0.2 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32)
                            .sin()
                    })
                    .collect(),
                reference: String::new(),
            },
            other => panic!("unknown fixture {other}"),
        };
        out.push(fixture);
    }
    out
}

/// One paced session against the shipped binary. Returns raw material for
/// metric extraction: the injection log and arrival-stamped NDJSON lines.
struct SessionRaw {
    /// (audio_sec_completed, wall_sec since session epoch) per 20 ms write.
    injection: Vec<(f64, f64)>,
    /// (arrival wall_sec since session epoch, parsed event) per line.
    lines: Vec<(f64, serde_json::Value)>,
    feed_wall_sec: f64,
    audio_sec: f64,
    exit_ok: bool,
}

fn run_session(fw: &str, policy: &str, samples: &[f32]) -> SessionRaw {
    let mut child: Child = Command::new(fw)
        .args([
            "robot",
            "listen",
            "--source",
            "stdin-pcm",
            "--stdin-rate",
            "16000",
            "--stdin-channels",
            "1",
            "--stdin-format",
            "s16le",
            "--language",
            "en",
            "--policy",
            policy,
            "--stats-interval-sec",
            "5",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn fw robot listen");
    let mut stdin = child.stdin.take().expect("child stdin");
    let stdout = child.stdout.take().expect("child stdout");

    let epoch = Instant::now();
    let reader = std::thread::spawn(move || {
        let mut lines = Vec::new();
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let arrival = epoch.elapsed().as_secs_f64();
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                lines.push((arrival, value));
            }
        }
        lines
    });

    // Paced feed on an absolute schedule (drift-free): frame i targets
    // epoch + i*20ms. Each write logs the audio position it completed.
    let mut injection = Vec::with_capacity(samples.len() / FRAME + 1);
    let mut buf = vec![0u8; FRAME * 2];
    let feed_start = Instant::now();
    for (i, frame) in samples.chunks(FRAME).enumerate() {
        let target = Duration::from_millis(i as u64 * 20);
        if let Some(wait) = target.checked_sub(feed_start.elapsed()) {
            std::thread::sleep(wait);
        }
        for (j, s) in frame.iter().enumerate() {
            let v = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            buf[j * 2..j * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
        if stdin.write_all(&buf[..frame.len() * 2]).is_err() {
            break; // child died; exit status reported below
        }
        injection.push((
            (i * FRAME + frame.len()) as f64 / SAMPLE_RATE as f64,
            epoch.elapsed().as_secs_f64(),
        ));
    }
    let feed_wall_sec = feed_start.elapsed().as_secs_f64();
    drop(stdin); // EOF -> session-end flush
    let status = child.wait().expect("child wait");
    let lines = reader.join().expect("reader join");
    SessionRaw {
        injection,
        lines,
        feed_wall_sec,
        audio_sec: samples.len() as f64 / SAMPLE_RATE as f64,
        exit_ok: status.success(),
    }
}

/// Wall time (session epoch seconds) at which the audio at `audio_sec` had
/// been written to the child's stdin.
fn wall_at_audio(injection: &[(f64, f64)], audio_sec: f64) -> Option<f64> {
    injection
        .iter()
        .find(|(a, _)| *a >= audio_sec)
        .map(|(_, w)| *w)
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    Some(sorted[idx])
}

#[derive(Debug)]
struct SessionMetrics {
    ttft_ms: Option<f64>,
    commit_lag_ms: Vec<f64>,
    endpoint_lag_ms: Vec<f64>,
    deltas: u64,
    utterances: u64,
    wer: Option<f64>,
    hyp_words: usize,
    mean_step_ms: Option<f64>,
    p95_step_ms: Option<f64>,
    pace_error_pct: f64,
    pace_violation: bool,
    exit_ok: bool,
    joined_text: String,
}

fn extract(raw: &SessionRaw, reference: &str) -> SessionMetrics {
    let mut speech_started: BTreeMap<u64, f64> = BTreeMap::new(); // utt -> t_session_sec
    let mut first_delta_arrival: BTreeMap<u64, f64> = BTreeMap::new();
    let mut commit_lag_ms = Vec::new();
    let mut endpoint_lag_ms = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let mut deltas = 0u64;
    let mut utterances = 0u64;
    let mut mean_step_ms = None;
    let mut p95_step_ms = None;
    for (arrival, ev) in &raw.lines {
        match ev["event"].as_str().unwrap_or("") {
            "speech_started" => {
                if let (Some(u), Some(t)) =
                    (ev["utterance_id"].as_u64(), ev["t_session_sec"].as_f64())
                {
                    speech_started.entry(u).or_insert(t);
                }
            }
            "transcript.delta" => {
                deltas += 1;
                if let Some(u) = ev["utterance_id"].as_u64() {
                    first_delta_arrival.entry(u).or_insert(*arrival);
                }
                if let Some(t1) = ev["t1_sec"].as_f64()
                    && let Some(w) = wall_at_audio(&raw.injection, t1)
                {
                    commit_lag_ms.push((arrival - w) * 1000.0);
                }
            }
            "utterance_end" => {
                utterances += 1;
                if let Some(t1) = ev["t1_sec"].as_f64()
                    && let Some(w) = wall_at_audio(&raw.injection, t1)
                {
                    endpoint_lag_ms.push((arrival - w) * 1000.0);
                }
                if let Some(text) = ev["text"].as_str()
                    && !text.is_empty()
                {
                    texts.push(text.to_owned());
                }
            }
            "listen.session_stats" => {
                mean_step_ms = ev["mean_step_latency_ms"].as_f64();
                p95_step_ms = ev["p95_step_latency_ms"].as_f64();
            }
            _ => {}
        }
    }
    // TTFT per utterance = first delta arrival - wall(speech onset audio).
    let mut ttfts: Vec<f64> = Vec::new();
    for (u, arrival) in &first_delta_arrival {
        if let Some(t_onset) = speech_started.get(u)
            && let Some(w) = wall_at_audio(&raw.injection, *t_onset)
        {
            ttfts.push((arrival - w) * 1000.0);
        }
    }
    ttfts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let joined_text = texts.join(" ");
    let wer = if reference.is_empty() {
        None // negative fixture: report hypothesis words instead
    } else {
        Some(word_error_rate(reference, &joined_text).wer)
    };
    let pace_error_pct = (raw.feed_wall_sec / raw.audio_sec - 1.0) * 100.0;
    SessionMetrics {
        ttft_ms: percentile(&ttfts, 0.5),
        commit_lag_ms,
        endpoint_lag_ms,
        deltas,
        utterances,
        wer,
        hyp_words: joined_text.split_whitespace().count(),
        mean_step_ms,
        p95_step_ms,
        pace_error_pct,
        pace_violation: pace_error_pct.abs() > 1.0,
        exit_ok: raw.exit_ok,
        joined_text,
    }
}

fn median(values: &mut [f64]) -> Option<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile(values, 0.5)
}

fn session_row(m: &SessionMetrics) -> serde_json::Value {
    let mut commit = m.commit_lag_ms.clone();
    commit.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut endpoint = m.endpoint_lag_ms.clone();
    endpoint.sort_by(|a, b| a.partial_cmp(b).unwrap());
    serde_json::json!({
        "ttft_ms": m.ttft_ms,
        "commit_lag_ms_p50": percentile(&commit, 0.5),
        "commit_lag_ms_p95": percentile(&commit, 0.95),
        "endpoint_lag_ms_p50": percentile(&endpoint, 0.5),
        "deltas": m.deltas,
        "utterances": m.utterances,
        "wer": m.wer,
        "hyp_words": m.hyp_words,
        "mean_step_ms": m.mean_step_ms,
        "p95_step_ms": m.p95_step_ms,
        "pace_error_pct": m.pace_error_pct,
        "pace_violation": m.pace_violation,
        "exit_ok": m.exit_ok,
        "text": m.joined_text,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| default.to_owned())
    };
    let fw = get("--fw", "target/release/franken_whisper");
    let rounds: usize = get("--rounds", "2").parse().expect("--rounds");
    let out_path = get("--out", "");
    let arm_names: Vec<String> = get("--arms", "alignatt,endpoint-commit")
        .split(',')
        .map(str::to_owned)
        .collect();
    let fixture_names: Vec<String> = get("--fixtures", "short,pauses,long,noisy,negative")
        .split(',')
        .map(str::to_owned)
        .collect();
    assert!(
        std::path::Path::new(&fw).exists(),
        "fw binary not found at {fw} (pass --fw)"
    );

    let fixtures = build_fixtures(&fixture_names);
    let mut report_fixtures = serde_json::Map::new();
    let mut null_ratios = serde_json::Map::new();

    for fixture in &fixtures {
        eprintln!(
            "== fixture {} ({:.1}s) ==",
            fixture.name,
            fixture.samples.len() as f64 / 16000.0
        );
        let mut per_arm: BTreeMap<String, Vec<SessionMetrics>> = BTreeMap::new();
        for round in 0..rounds {
            // Order alternation: monotonic drift lands on both arms equally.
            let mut order: Vec<&String> = arm_names.iter().collect();
            if round % 2 == 1 {
                order.reverse();
            }
            for arm in order {
                eprintln!("   round {round} arm {arm}");
                let raw = run_session(&fw, arm, &fixture.samples);
                let m = extract(&raw, &fixture.reference);
                eprintln!(
                    "     ttft={:?}ms deltas={} utts={} wer={:?} pace_err={:.2}% exit_ok={}",
                    m.ttft_ms, m.deltas, m.utterances, m.wer, m.pace_error_pct, m.exit_ok
                );
                per_arm.entry(arm.clone()).or_default().push(m);
            }
        }
        // A/A nulls on the `long` fixture (the most deltas per session, so
        // the pooled medians rest on a workable sample): two session PAIRS
        // per arm, commit lags pooled across each side before the median —
        // a median-of-3 ratio (first campaign) is load noise, not a null.
        if fixture.name == "long" {
            for arm in &arm_names {
                eprintln!("   A/A null: {arm} (2 pairs, pooled)");
                let mut side_a: Vec<f64> = Vec::new();
                let mut side_b: Vec<f64> = Vec::new();
                for _ in 0..2 {
                    side_a.extend(
                        extract(&run_session(&fw, arm, &fixture.samples), &fixture.reference)
                            .commit_lag_ms,
                    );
                    side_b.extend(
                        extract(&run_session(&fw, arm, &fixture.samples), &fixture.reference)
                            .commit_lag_ms,
                    );
                }
                let ratio = match (median(&mut side_a), median(&mut side_b)) {
                    (Some(x), Some(y)) if y > 0.0 => Some(x / y),
                    _ => None,
                };
                eprintln!(
                    "     null ratio {ratio:?} (n={}/{})",
                    side_a.len(),
                    side_b.len()
                );
                null_ratios.insert(arm.clone(), serde_json::json!(ratio));
            }
        }
        let mut arm_rows = serde_json::Map::new();
        for (arm, sessions) in &per_arm {
            let mut ttfts: Vec<f64> = sessions.iter().filter_map(|s| s.ttft_ms).collect();
            let ttft_median = median(&mut ttfts);
            let mut lags: Vec<f64> = sessions
                .iter()
                .flat_map(|s| s.commit_lag_ms.clone())
                .collect();
            let lag_median = median(&mut lags);
            arm_rows.insert(
                arm.clone(),
                serde_json::json!({
                    "sessions": sessions.iter().map(session_row).collect::<Vec<_>>(),
                    "ttft_ms_median": ttft_median,
                    "commit_lag_ms_median": lag_median,
                }),
            );
        }
        report_fixtures.insert(fixture.name.to_owned(), serde_json::Value::Object(arm_rows));
    }

    let nulls_ok = null_ratios
        .values()
        .all(|v| v.as_f64().is_some_and(|r| (0.98..=1.02).contains(&r)));
    let report = serde_json::json!({
        "harness": "listen_latency_ab",
        "bead": "bd-rt-latency-harness-3dkh",
        "scope": "self_relative", // no external incumbent arm in this run
        "binary": fw,
        "rounds": rounds,
        "arms": arm_names,
        "aa_null_commit_lag_ratio": null_ratios,
        "aa_nulls_within_band": nulls_ok,
        "comparative_verdicts_allowed": nulls_ok,
        "fixtures": report_fixtures,
    });
    // Ledger-row summary fields must exist (bead acceptance).
    for key in ["harness", "scope", "aa_nulls_within_band", "fixtures"] {
        assert!(report.get(key).is_some(), "missing summary field {key}");
    }
    let rendered = serde_json::to_string_pretty(&report).expect("render report");
    if out_path.is_empty() {
        println!("{rendered}");
    } else {
        std::fs::write(&out_path, &rendered).expect("write report");
        eprintln!("report -> {out_path}");
    }
}
