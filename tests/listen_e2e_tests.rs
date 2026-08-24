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

/// Model gate: both lanes must resolve exactly the way the binary resolves
/// them (same `resolve_model` code path). Returns `None` after printing a
/// SKIP line when either package is missing.
fn require_both_models() -> bool {
    let fast_ok = franken_whisper::native_engine::resolve_model("tiny.en").is_ok();
    let quality_ok = franken_whisper::native_engine::resolve_model("large-v3-turbo").is_ok();
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
    if franken_whisper::native_engine::resolve_model("tiny.en").is_err() {
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
