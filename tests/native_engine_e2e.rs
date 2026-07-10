//! bd-4slu: end-to-end proof that the rollout machinery drives the *real*
//! native whisper engine through the full library dispatch.
//!
//! These are the rollout-machinery tests. Each scenario spawns the actual
//! `franken_whisper` CLI binary as a subprocess (exactly like
//! `cli_integration.rs`'s `run_transcribe_json_with_stub_env`) and drives the
//! whole pipeline — ingest → normalize → backend dispatch — with the native
//! rollout env vars set. We deliberately spawn a subprocess rather than mutate
//! `std::env` in-process, because env mutation is `unsafe` and crate-forbidden
//! under edition 2024; `.env()` on a child process is the safe equivalent the
//! sibling integration tests already rely on.
//!
//! The crucial trick used to *prove* the native engine ran (and that no bridge
//! adapter could have): we point `FRANKEN_WHISPER_WHISPER_CPP_BIN` (and the
//! insanely-fast / diarization bridge binaries) at `/nonexistent`. In a `sole`
//! or `primary` rollout stage with `FRANKEN_WHISPER_NATIVE_EXECUTION=1`, a
//! transcript can therefore only come from the in-process native engine.
//!
//! Every scenario is **gated**: when the real `tiny.en` ggml model is not
//! resolvable (`find_model_file("tiny.en") == None`), it prints a `SKIP` line
//! and returns success, so CI without the model still passes. Provision the
//! model with `scripts/fetch_test_models.sh`.

use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};

use serde_json::Value;

/// The reference transcript whisper-cli produced for `jfk.wav` with `tiny.en`,
/// read at runtime from `tests/fixtures/native/jfk_tiny_reference.json` (the
/// `-oj` output committed alongside the audio fixture). We do not hard-code it
/// so the fixture stays the single source of truth.
fn reference_transcript() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/native/jfk_tiny_reference.json"
    );
    let bytes = std::fs::read(path).expect("read jfk_tiny_reference.json fixture");
    let json: Value = serde_json::from_slice(&bytes).expect("parse reference json");
    let segments = json["transcription"]
        .as_array()
        .expect("reference `transcription` array");
    let joined = segments
        .iter()
        .map(|seg| seg["text"].as_str().unwrap_or_default().trim())
        .collect::<Vec<_>>()
        .join(" ");
    normalize_ws(&joined)
}

/// Collapse internal whitespace runs to single spaces and trim, so transcript
/// comparison is robust to leading-space / spacing quirks across the reference
/// JSON and the engine's joined-segment output.
fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn transcript_words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split_whitespace()
        .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_owned())
        .filter(|word| !word.is_empty())
        .collect()
}

/// Word-level Levenshtein WER, normalized by reference word count.
fn word_error_rate(reference: &str, candidate: &str) -> f64 {
    let reference = transcript_words(reference);
    let candidate = transcript_words(candidate);
    if reference.is_empty() {
        return if candidate.is_empty() { 0.0 } else { 1.0 };
    }

    let mut prev: Vec<usize> = (0..=candidate.len()).collect();
    let mut curr = vec![0usize; candidate.len() + 1];
    for (i, reference_word) in reference.iter().enumerate() {
        curr[0] = i + 1;
        for (j, candidate_word) in candidate.iter().enumerate() {
            let substitute = prev[j] + usize::from(reference_word != candidate_word);
            let delete = prev[j + 1] + 1;
            let insert = curr[j] + 1;
            curr[j + 1] = substitute.min(delete).min(insert);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[candidate.len()] as f64 / reference.len() as f64
}

/// Absolute path to the in-repo audio fixture.
fn jfk_wav() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/native/jfk.wav"
    ))
}

/// Gate: is the real `tiny.en` model resolvable on this machine? Mirrors the
/// library's own resolver so the gate can never drift from production lookup.
fn tiny_en_available() -> bool {
    franken_whisper::native_engine::find_model_file("tiny.en").is_some()
}

/// Outcome of a CLI transcribe subprocess: the parsed JSON report plus the raw
/// streams and exit status (so error-path tests can inspect all three).
struct CliRun {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

impl CliRun {
    /// Parse the JSON report from stdout. Panics with the full streams on
    /// failure — only call on the success path.
    fn report(&self) -> Value {
        let start = self.stdout.find('{').unwrap_or_else(|| {
            panic!(
                "no JSON object in stdout\nstdout:\n{}\nstderr:\n{}",
                self.stdout, self.stderr
            )
        });
        serde_json::from_str(&self.stdout[start..]).unwrap_or_else(|e| {
            panic!(
                "json parse failed: {e}\nstdout:\n{}\nstderr:\n{}",
                self.stdout, self.stderr
            )
        })
    }
}

/// Spawn `franken_whisper transcribe <args>` with the given extra env vars.
/// Bridge binaries are forced to `/nonexistent` by every caller that wants to
/// prove native execution.
fn run_transcribe(args: &[&str], extra_env: &[(&str, &str)], state_root: &Path) -> CliRun {
    let mut cmd = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"));
    cmd.arg("transcribe");
    cmd.args(args);
    cmd.env("FRANKEN_WHISPER_STATE_DIR", state_root);
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let output = cmd.output().expect("spawn franken_whisper transcribe");
    CliRun {
        status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Locate the `backend.ok` event's payload in a report, or panic with the
/// event list when it is absent.
fn backend_ok_payload(report: &Value) -> &Value {
    let events = report["events"].as_array().expect("report events array");
    for event in events {
        if event["code"].as_str() == Some("backend.ok") {
            return &event["payload"];
        }
    }
    let codes: Vec<&str> = events.iter().filter_map(|e| e["code"].as_str()).collect();
    panic!("no backend.ok event in report; codes seen: {codes:?}");
}

/// Force every bridge backend binary to a path that cannot exist, so any
/// produced transcript provably came from the in-process native engine.
fn bridge_bins_missing() -> [(&'static str, &'static str); 3] {
    [
        ("FRANKEN_WHISPER_WHISPER_CPP_BIN", "/nonexistent"),
        ("FRANKEN_WHISPER_INSANELY_FAST_BIN", "/nonexistent"),
        ("FRANKEN_WHISPER_PYTHON_BIN", "/nonexistent"),
    ]
}

/// Assert the report's transcript matches the whisper-cli reference fixture.
fn assert_transcript_matches_reference(report: &Value) {
    let produced = normalize_ws(report["result"]["transcript"].as_str().unwrap_or_default());
    let reference = reference_transcript();
    assert_eq!(
        produced, reference,
        "native transcript must match whisper-cli reference fixture exactly"
    );
}

fn assert_reference_wer_at_or_below(report: &Value, max_wer: f64, scenario: &str) {
    let produced = normalize_ws(report["result"]["transcript"].as_str().unwrap_or_default());
    let reference = reference_transcript();
    let wer = word_error_rate(&reference, &produced);
    assert!(
        wer <= max_wer,
        "{scenario} WER {wer:.4} exceeds gate {max_wer:.4}\nREFERENCE: {reference}\nPRODUCED:  {produced}"
    );
}

// ===========================================================================
// (a) sole-stage native: native is the ONLY thing that can have run.
// ===========================================================================

#[test]
fn gated_sole_stage_native_is_only_path() {
    if !tiny_en_available() {
        eprintln!("SKIP gated_sole_stage_native_is_only_path: tiny.en model missing");
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();

    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole"),
    ];
    env.extend(bridge_bins_missing());

    let run = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--model",
            "tiny.en",
            "--no-persist",
            "--json",
        ],
        &env,
        state.path(),
    );

    assert!(
        run.status.success(),
        "sole-stage native run failed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    let report = run.report();

    assert_transcript_matches_reference(&report);
    assert_eq!(report["result"]["backend"], "whisper_cpp");

    let payload = backend_ok_payload(&report);
    assert_eq!(
        payload["implementation"], "native",
        "sole stage must run the native implementation, not the bridge"
    );
    assert_eq!(
        payload["execution_mode"], "native_only",
        "sole stage maps to native_only execution mode"
    );
    assert_eq!(payload["native_rollout_stage"], "sole");

    // The native raw_output schema proves real in-process inference ran.
    assert_eq!(
        report["result"]["raw_output"]["engine"],
        "whisper.cpp-native"
    );
    assert_eq!(
        report["result"]["raw_output"]["implementation"],
        "real-inference"
    );
}

// ===========================================================================
// (a2) quality-safe full encoder int8: default-on candidate must preserve JFK.
// ===========================================================================

#[test]
fn gated_quality_safe_encoder_int8_jfk_reference_wer_gate() {
    if !tiny_en_available() {
        eprintln!(
            "SKIP gated_quality_safe_encoder_int8_jfk_reference_wer_gate: tiny.en model missing"
        );
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();

    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole"),
        // Point at the quality-safe full encoder-int8 policy, not the older
        // all-i7 full gate that is still owner-gated for proper-noun drift.
        ("FRANKEN_WHISPER_ENC_INT8", "0"),
        ("FW_ENC_ATTN_OUT_I8I32", "1"),
        // Make the intended default-on int8 subpolicy explicit in the evidence
        // gate; both are currently default-on only inside the int8 encoder path.
        ("FW_ENC_QKV_FUSED", "1"),
        ("FW_ENC_EF_QUANT", "1"),
    ];
    env.extend(bridge_bins_missing());

    let run = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--model",
            "tiny.en",
            "--no-persist",
            "--json",
        ],
        &env,
        state.path(),
    );

    assert!(
        run.status.success(),
        "quality-safe encoder-int8 native run failed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    let report = run.report();

    assert_reference_wer_at_or_below(&report, 0.0, "quality-safe encoder-int8 JFK");
    let produced = normalize_ws(report["result"]["transcript"].as_str().unwrap_or_default());
    assert!(
        !produced.to_lowercase().contains("frank at"),
        "known all-i7 encoder adversarial phrase must not appear in quality-safe int8 output: {produced}"
    );

    let payload = backend_ok_payload(&report);
    assert_eq!(
        payload["implementation"], "native",
        "quality-safe encoder-int8 gate must run the native implementation"
    );
    assert_eq!(payload["execution_mode"], "native_only");
}

// ===========================================================================
// (b) primary-stage preference: native preferred, bridge missing -> native.
// ===========================================================================

#[test]
fn gated_primary_stage_prefers_native() {
    if !tiny_en_available() {
        eprintln!("SKIP gated_primary_stage_prefers_native: tiny.en model missing");
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();

    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "primary"),
    ];
    env.extend(bridge_bins_missing());

    let run = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--model",
            "tiny.en",
            "--no-persist",
            "--json",
        ],
        &env,
        state.path(),
    );

    assert!(
        run.status.success(),
        "primary-stage native run failed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    let report = run.report();

    assert_transcript_matches_reference(&report);
    let payload = backend_ok_payload(&report);
    assert_eq!(
        payload["implementation"], "native",
        "primary stage with bridge missing must resolve to native"
    );
    assert_eq!(payload["execution_mode"], "native_preferred");
    assert_eq!(payload["native_rollout_stage"], "primary");
}

// ===========================================================================
// (c) bridge-only honest unavailability: no native, bridge missing -> error.
// ===========================================================================

#[test]
fn bridge_only_missing_bridge_errors_honestly() {
    // This scenario needs NO model: it asserts the honest failure when the
    // native path is disabled and the bridge binary is absent. It must NOT
    // silently succeed via some hidden path.
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();

    let env = [
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "0"),
        ("FRANKEN_WHISPER_WHISPER_CPP_BIN", "/nonexistent"),
        // Disable bridge->native recovery so this is a clean bridge-only test
        // even on a machine that happens to have the model present.
        ("FRANKEN_WHISPER_BRIDGE_NATIVE_RECOVERY", "0"),
    ];

    let run = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        &env,
        state.path(),
    );

    assert!(
        !run.status.success(),
        "bridge-only with a missing bridge binary must fail, not succeed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    // `transcribe` emits a structured `error: ...` line on stderr (see
    // src/main.rs run() error path) and exits non-zero.
    let combined = format!("{}{}", run.stdout, run.stderr);
    assert!(
        combined.to_lowercase().contains("error"),
        "expected a structured error on stdout/stderr\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
}

// ===========================================================================
// (d) insanely-fast native through the dispatch.
// ===========================================================================

#[test]
fn gated_insanely_fast_native_through_dispatch() {
    if !tiny_en_available() {
        eprintln!("SKIP gated_insanely_fast_native_through_dispatch: tiny.en model missing");
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();

    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole"),
    ];
    env.extend(bridge_bins_missing());

    let run = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "insanely-fast",
            "--model",
            "tiny.en",
            "--no-persist",
            "--json",
        ],
        &env,
        state.path(),
    );

    assert!(
        run.status.success(),
        "insanely-fast native run failed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    let report = run.report();

    assert_transcript_matches_reference(&report);
    assert_eq!(report["result"]["backend"], "insanely_fast");
    let payload = backend_ok_payload(&report);
    assert_eq!(payload["implementation"], "native");
    assert_eq!(payload["execution_mode"], "native_only");
}

// ===========================================================================
// (e) diarization native through the dispatch: transcript + SPEAKER_ labels +
//     honest text-temporal-heuristic diarizer tagging.
// ===========================================================================

#[test]
fn gated_diarization_native_through_dispatch() {
    if !tiny_en_available() {
        eprintln!("SKIP gated_diarization_native_through_dispatch: tiny.en model missing");
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();

    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole"),
    ];
    env.extend(bridge_bins_missing());

    let run = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-diarization",
            "--model",
            "tiny.en",
            "--no-persist",
            "--json",
        ],
        &env,
        state.path(),
    );

    assert!(
        run.status.success(),
        "diarization native run failed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    let report = run.report();

    assert_transcript_matches_reference(&report);
    assert_eq!(report["result"]["backend"], "whisper_diarization");

    let payload = backend_ok_payload(&report);
    assert_eq!(payload["implementation"], "native");

    // Every segment must carry a SPEAKER_ label from the heuristic diarizer.
    let segments = report["result"]["segments"]
        .as_array()
        .expect("segments array");
    assert!(!segments.is_empty(), "diarization produced no segments");
    for seg in segments {
        let speaker = seg["speaker"].as_str().unwrap_or_default();
        assert!(
            speaker.starts_with("SPEAKER_"),
            "segment speaker `{speaker}` must be a SPEAKER_NN label"
        );
    }

    // Honest diarizer provenance: the native raw_output must declare the
    // text-temporal heuristic (NOT a neural diarizer).
    assert_eq!(
        report["result"]["raw_output"]["diarizer"], "text-temporal-heuristic",
        "diarizer must be honestly tagged as the text/temporal heuristic"
    );
}

// ===========================================================================
// (f) double-diarization regression: --backend whisper-diarization --diarize
//     must NOT diarize twice. The backend owns diarization, so the pipeline
//     Diarize stage must emit a `diarize.skip` event with the structured
//     `backend_owns_diarization` reason, while segments still carry the
//     backend's SPEAKER_ labels.
// ===========================================================================

#[test]
fn gated_diarize_flag_with_diarization_backend_skips_pipeline_diarize() {
    if !tiny_en_available() {
        eprintln!(
            "SKIP gated_diarize_flag_with_diarization_backend_skips_pipeline_diarize: tiny.en model missing"
        );
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();

    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole"),
    ];
    env.extend(bridge_bins_missing());

    let run = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-diarization",
            "--diarize",
            "--model",
            "tiny.en",
            "--no-persist",
            "--json",
        ],
        &env,
        state.path(),
    );

    // (a) success
    assert!(
        run.status.success(),
        "diarize-flag + diarization-backend run failed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    let report = run.report();

    // (b) the events array contains a diarize skip with the backend-owns reason.
    let events = report["events"].as_array().expect("report events array");
    let diarize_skip = events
        .iter()
        .find(|e| e["code"].as_str() == Some("diarize.skip"))
        .unwrap_or_else(|| {
            let codes: Vec<&str> = events.iter().filter_map(|e| e["code"].as_str()).collect();
            panic!("no diarize.skip event; codes seen: {codes:?}");
        });
    assert_eq!(
        diarize_skip["payload"]["reason"], "backend_owns_diarization",
        "pipeline diarize stage must be skipped because the backend owns diarization"
    );
    assert_eq!(
        diarize_skip["payload"]["details"]["backend"],
        "whisper_diarization"
    );

    // Defensively prove there was no SECOND diarize pass: exactly one
    // diarize-stage event total, and it is the skip.
    let diarize_events: Vec<&str> = events
        .iter()
        .filter(|e| e["stage"].as_str() == Some("diarize"))
        .filter_map(|e| e["code"].as_str())
        .collect();
    assert_eq!(
        diarize_events,
        vec!["diarize.skip"],
        "the pipeline diarize stage must run exactly once as a skip, never re-diarizing"
    );

    // (c) segments still carry SPEAKER_ labels (from the backend's diarizer).
    let segments = report["result"]["segments"]
        .as_array()
        .expect("segments array");
    assert!(!segments.is_empty(), "diarization produced no segments");
    for seg in segments {
        let speaker = seg["speaker"].as_str().unwrap_or_default();
        assert!(
            speaker.starts_with("SPEAKER_"),
            "segment speaker `{speaker}` must be a SPEAKER_NN label from the backend"
        );
    }
}
