//! Model-gated end-to-end tests for the live confirm lane
//! (bd-rt-confirm-lane-3okr).
//!
//! Runs the REAL `fw` binary over an in-repo fixture via unpaced
//! file-replay (no wall-clock in the loop — CI-safe) and asserts the
//! confirm-lane contract on the captured NDJSON stream. Skips gracefully
//! (the standard model-gated pattern: report missing prerequisites instead
//! of fabricating a pass) when the required model packages are not cached.
//!
//! NOTE (bd-rt-e2e-0zo5): the full listen e2e suite (golden streams, error
//! paths, mutation teeth) extends this file once Waves 3 (persist) and 4
//! (local-agreement) land; this file exists now because the confirm-lane
//! acceptance requires its own model-gated contract test.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use franken_whisper::robot::{NdjsonStreamValidator, StreamOutcome};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn jfk_wav() -> PathBuf {
    manifest_dir()
        .join("tests")
        .join("fixtures")
        .join("native")
        .join("jfk.wav")
}

fn fast_model_ready() -> bool {
    static READY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *READY.get_or_init(|| {
        franken_whisper::model_distribution::resolve_cached_fast_lane_with_cancel(
            franken_whisper::model_distribution::FastLaneModel::TinyEn,
            || false,
        )
        .is_ok()
    })
}

fn quality_model_ready() -> bool {
    static READY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *READY.get_or_init(|| {
        franken_whisper::model_distribution::resolve_cached_whisper_with_cancel(|| false).is_ok()
    })
}

/// Model gate: both lanes authenticate through the same compiled package
/// trust roots used by the listen binary. Returns `false` after printing a
/// SKIP line when either package is missing or corrupt.
fn require_both_models() -> bool {
    let fast_ok = fast_model_ready();
    let quality_ok = quality_model_ready();
    if !fast_ok || !quality_ok {
        eprintln!(
            "SKIP confirm_lane_e2e: missing model package(s) \
             (tiny.en={fast_ok}, large-v3-turbo={quality_ok}); \
             install with `fw pull tiny-en` / `fw pull whisper`"
        );
        return false;
    }
    true
}

fn require_fast_model() -> bool {
    if !fast_model_ready() {
        eprintln!("SKIP confirm_lane_e2e: tiny.en package missing (`fw pull tiny-en`)");
        return false;
    }
    true
}

/// Spawn `fw robot listen` over the fixture (unpaced replay), capture all
/// stdout NDJSON lines plus stderr (for failure artifacts), and return
/// (parsed events, exit code, stderr tail).
fn run_listen(extra_args: &[&str]) -> (Vec<serde_json::Value>, i32, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_fw"));
    cmd.args(["robot", "listen"])
        .arg("--source")
        .arg("file-replay")
        .arg("--input")
        .arg(jfk_wav())
        .args(extra_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn fw robot listen");
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    // Drain stderr on a helper thread so the child never blocks on a full
    // pipe; keep only the tail for failure diagnostics.
    let stderr_tail: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::default();
    let stderr_handle = std::thread::spawn({
        let stderr_tail = std::sync::Arc::clone(&stderr_tail);
        move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                let mut tail = stderr_tail.lock().expect("stderr tail lock");
                tail.push(line);
                if tail.len() > 40 {
                    let excess = tail.len() - 40;
                    tail.drain(..excess);
                }
            }
        }
    });
    let deadline = Instant::now() + Duration::from_secs(600);
    let mut events = Vec::new();
    for line in BufReader::new(stdout).lines() {
        let Ok(line) = line else { break };
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
            events.push(value);
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!("fw robot listen exceeded 600 s (unpaced replay must not hang)");
        }
    }
    let status = child.wait().expect("wait for fw");
    stderr_handle.join().expect("stderr drain thread");
    let stderr_text = stderr_tail.lock().expect("stderr tail lock").join("\n");
    (events, status.code().unwrap_or(-1), stderr_text)
}

fn verdict_event_ids(events: &[serde_json::Value]) -> Vec<u32> {
    events
        .iter()
        .filter(|e| {
            matches!(
                e.get("event").and_then(|v| v.as_str()),
                Some("transcript.confirm") | Some("transcript.correct")
            )
        })
        .filter_map(|e| e.get("utterance_id").and_then(|v| v.as_u64()))
        .map(|id| id as u32)
        .collect()
}

/// The dual-model contract: tiny.en fast lane + large-v3-turbo confirm lane
/// over the JFK fixture. The stream must satisfy the full 1.1.0 listen
/// contract, carry at least one verdict, key every verdict to an
/// already-closed utterance (validator-enforced ordering), and reconcile
/// verdict counts with the final session_stats.
#[test]
fn confirm_lane_e2e_turbo_verifies_fast_lane_utterances() {
    if !require_both_models() {
        return;
    }
    let (events, code, stderr) = run_listen(&[
        "--fast-model",
        "tiny.en",
        "--language",
        "en",
        "--quality-model",
        "large-v3-turbo",
        "--policy",
        "alignatt",
        // No --confirm-drain-sec: unpaced replay drains the confirm lane to
        // idle (bd-fdk6), so every verdict lands however slow the host is.
    ]);
    assert_eq!(code, 0, "listen session must exit 0; stderr:\n{stderr}");
    NdjsonStreamValidator::new(StreamOutcome::Success)
        .validate(&events)
        .expect("confirm-lane stream must satisfy the listen contract");

    let verdicts = verdict_event_ids(&events);
    assert!(
        !verdicts.is_empty(),
        "expected at least one transcript.confirm/correct from the turbo lane"
    );

    // Every verdict carries the full payload contract.
    for event in events.iter().filter(|e| {
        matches!(
            e.get("event").and_then(|v| v.as_str()),
            Some("transcript.confirm") | Some("transcript.correct")
        )
    }) {
        assert!(
            event
                .get("quality_model_id")
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.is_empty()),
            "verdict must name its quality model: {event}"
        );
        assert!(
            event
                .get("drift")
                .and_then(|d| d.get("wer_approx"))
                .and_then(|v| v.as_f64())
                .is_some(),
            "verdict must carry drift.wer_approx: {event}"
        );
        assert!(
            event.get("latency_ms").and_then(|v| v.as_u64()).is_some(),
            "verdict must carry latency_ms: {event}"
        );
        if event.get("event").and_then(|v| v.as_str()) == Some("transcript.correct") {
            assert!(
                event
                    .get("segments")
                    .and_then(|v| v.as_array())
                    .is_some_and(|a| !a.is_empty()),
                "correct verdict must carry the quality segments: {event}"
            );
            assert!(
                event.get("correction_id").is_some(),
                "correct verdict must carry correction_id: {event}"
            );
        }
    }

    // session_stats reconciliation: emitted verdict counts must equal the
    // number of verdict events observed on the stream.
    let final_stats = events
        .iter()
        .rev()
        .find(|e| e.get("event").and_then(|v| v.as_str()) == Some("listen.session_stats"))
        .expect("success stream ends with session_stats");
    let confirmations = final_stats
        .get("confirmations_emitted")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let corrections = final_stats
        .get("corrections_emitted")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert_eq!(
        usize::try_from(confirmations + corrections).unwrap_or(0),
        verdicts.len(),
        "stats verdict counts must reconcile with emitted events"
    );
}

/// `--quality-model none` opts out: zero verdict events, contract intact.
#[test]
fn confirm_lane_e2e_quality_model_none_emits_zero_verdicts() {
    if !require_fast_model() {
        return;
    }
    let (events, code, stderr) = run_listen(&[
        "--fast-model",
        "tiny.en",
        "--language",
        "en",
        "--quality-model",
        "none",
        "--policy",
        "alignatt",
    ]);
    assert_eq!(code, 0, "listen session must exit 0; stderr:\n{stderr}");
    NdjsonStreamValidator::new(StreamOutcome::Success)
        .validate(&events)
        .expect("fast-only stream must satisfy the listen contract");
    let verdicts = verdict_event_ids(&events);
    assert!(
        verdicts.is_empty(),
        "quality-model none must emit zero confirm/correct events, got {verdicts:?}"
    );
    let final_stats = events
        .iter()
        .rev()
        .find(|e| e.get("event").and_then(|v| v.as_str()) == Some("listen.session_stats"))
        .expect("success stream ends with session_stats");
    assert_eq!(
        final_stats
            .get("confirmations_emitted")
            .and_then(|v| v.as_u64()),
        Some(0)
    );
    assert_eq!(
        final_stats
            .get("corrections_emitted")
            .and_then(|v| v.as_u64()),
        Some(0)
    );
}

/// LocalAgreement-2 fallback policy (bd-rt-local-agreement-l5x8): the
/// stream must be contract-valid and append-only, and session_start must
/// announce the policy so agents can tell lanes apart. Event SHAPES are
/// identical to AlignAtt — that is the contract.
#[test]
fn local_agreement_e2e_stream_is_contract_valid_and_labeled() {
    if !require_fast_model() {
        return;
    }
    let (events, code, stderr) = run_listen(&[
        "--fast-model",
        "tiny.en",
        "--language",
        "en",
        "--quality-model",
        "none",
        "--policy",
        "local-agreement",
    ]);
    assert_eq!(code, 0, "listen session must exit 0; stderr:\n{stderr}");
    NdjsonStreamValidator::new(StreamOutcome::Success)
        .validate(&events)
        .expect("local-agreement stream must satisfy the listen contract");
    let start = events
        .iter()
        .find(|e| e.get("event").and_then(|v| v.as_str()) == Some("listen.session_start"))
        .expect("session_start present");
    assert_eq!(
        start.get("policy").and_then(|v| v.as_str()),
        Some("local-agreement")
    );
}

/// Word error rate of `hypothesis` against `reference` after lowercasing and
/// stripping punctuation (word-level Levenshtein / reference length).
fn normalized_wer(reference: &str, hypothesis: &str) -> f64 {
    let words = |text: &str| -> Vec<String> {
        text.split_whitespace()
            .map(|word| {
                word.chars()
                    .filter(|ch| ch.is_alphanumeric() || *ch == '\'')
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            .collect()
    };
    let reference = words(reference);
    let hypothesis = words(hypothesis);
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    for (i, ref_word) in reference.iter().enumerate() {
        let mut current = vec![i + 1; hypothesis.len() + 1];
        for (j, hyp_word) in hypothesis.iter().enumerate() {
            let substitution = previous[j] + usize::from(ref_word != hyp_word);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        previous = current;
    }
    previous[hypothesis.len()] as f64 / reference.len().max(1) as f64
}

const JFK_REFERENCE: &str = "And so my fellow Americans, ask not what your country can do \
                             for you, ask what you can do for your country.";

/// Declared live-transcript quality bound for the tiny.en fast lane on JFK:
/// every emission policy must keep the whole speech. Dropping one phrase
/// (e.g. "what your country can do", 5 of 22 words, the onset regression the
/// voiced-density VAD fixed) costs far more than this bound allows.
const JFK_LIVE_WER_BOUND: f64 = 0.15;

#[test]
fn live_transcript_keeps_all_speech_for_every_policy() {
    if !require_fast_model() {
        return;
    }
    for policy in ["alignatt", "local-agreement", "endpoint-commit"] {
        let (events, code, stderr) = run_listen(&[
            "--fast-model",
            "tiny.en",
            "--language",
            "en",
            "--quality-model",
            "none",
            "--no-persist",
            "--policy",
            policy,
        ]);
        assert_eq!(code, 0, "{policy}: listen must exit 0; stderr:\n{stderr}");
        NdjsonStreamValidator::new(StreamOutcome::Success)
            .validate(&events)
            .unwrap_or_else(|error| panic!("{policy}: contract violation: {error:?}"));
        let committed = events
            .iter()
            .filter(|e| e.get("event").and_then(|v| v.as_str()) == Some("utterance_end"))
            .filter_map(|e| e.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join(" ");
        let wer = normalized_wer(JFK_REFERENCE, &committed);
        assert!(
            wer <= JFK_LIVE_WER_BOUND,
            "{policy}: live WER {wer:.3} exceeds {JFK_LIVE_WER_BOUND}: {committed:?}"
        );
    }
}

/// Spawn `fw robot listen --source stdin-pcm` (tiny.en, no confirm lane, no
/// persistence) and hand stdin to `feed`, which owns writing and closing it.
/// Returns (events, exit code, wall seconds).
fn run_stdin_listen(
    extra_args: &[&str],
    feed: impl FnOnce(std::process::ChildStdin) + Send + 'static,
) -> (Vec<serde_json::Value>, i32, f64) {
    let started = Instant::now();
    let mut child = Command::new(env!("CARGO_BIN_EXE_fw"))
        .args(["robot", "listen", "--source", "stdin-pcm"])
        .args(["--fast-model", "tiny.en", "--language", "en"])
        .args(["--quality-model", "none", "--no-persist"])
        .args(extra_args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn fw robot listen");
    let stdin = child.stdin.take().expect("piped stdin");
    let feeder = std::thread::spawn(move || feed(stdin));
    let stdout = child.stdout.take().expect("piped stdout");
    let events: Vec<serde_json::Value> = BufReader::new(stdout)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| serde_json::from_str(&line).ok())
        .collect();
    let status = child.wait().expect("wait for fw");
    feeder.join().expect("stdin feeder");
    (
        events,
        status.code().unwrap_or(-1),
        started.elapsed().as_secs_f64(),
    )
}

/// bd-rt-e2e-0zo5 error path: a stdin stream that ends inside a PCM frame is
/// malformed input, not a clean end of session.
#[test]
fn stdin_pcm_ending_inside_a_sample_fails_the_session() {
    if !require_fast_model() {
        return;
    }
    let (events, code, _) = run_stdin_listen(&[], |mut stdin| {
        use std::io::Write as _;
        // 0.5 s of s16le silence plus one stray byte (half a sample).
        let mut bytes = vec![0_u8; 16_000];
        bytes.push(0x7f);
        let _ = stdin.write_all(&bytes);
        // Dropping stdin closes the pipe: EOF lands mid-sample.
    });
    assert_ne!(code, 0, "a torn PCM stream must not exit successfully");
    let error = events
        .iter()
        .find(|event| event["event"] == "run_error")
        .unwrap_or_else(|| panic!("expected run_error, got {events:?}"));
    let message = error["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("incomplete PCM frame") || message.contains("trailing byte"),
        "{message}"
    );
    assert!(
        !events
            .iter()
            .any(|event| event["event"] == "listen.session_stats" && event["final"] == true),
        "a failed session must not also report a successful final stats event"
    );
}

/// bd-rt-e2e-0zo5: `--max-seconds` bounds an endless live stream, measured
/// from capture start (model load excluded), and ends the session cleanly.
#[test]
fn max_seconds_bounds_an_endless_stdin_stream() {
    if !require_fast_model() {
        return;
    }
    let (events, code, wall_sec) = run_stdin_listen(&["--max-seconds", "2"], |mut stdin| {
        use std::io::Write as _;
        // 100 ms chunks of silence until the session closes the pipe (or a
        // generous safety cap, so a regression cannot hang the suite).
        let chunk = vec![0_u8; 3_200];
        for _ in 0..3_000 {
            if stdin.write_all(&chunk).is_err() {
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    });
    assert_eq!(
        code, 0,
        "a time-bounded session ends successfully: {events:?}"
    );
    let final_stats = events
        .iter()
        .find(|event| event["event"] == "listen.session_stats" && event["final"] == true)
        .unwrap_or_else(|| panic!("expected final stats, got {events:?}"));
    let audio_sec = final_stats["audio_sec"].as_f64().expect("audio_sec");
    assert!(
        audio_sec >= 1.5,
        "the budget must cover listening time, not model load: audio_sec={audio_sec}"
    );
    assert!(
        wall_sec < 240.0,
        "session must end near its budget: {wall_sec:.1}s"
    );
}

#[test]
fn normalized_wer_counts_dropped_phrases() {
    assert_eq!(normalized_wer(JFK_REFERENCE, JFK_REFERENCE), 0.0);
    let dropped =
        "And so my fellow Americans asked not for you ask what you can do for your country";
    let wer = normalized_wer(JFK_REFERENCE, dropped);
    assert!((wer - 6.0 / 22.0).abs() < 1e-9, "{wer}");
    assert!(wer > JFK_LIVE_WER_BOUND);
}
