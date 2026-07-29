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

use franken_whisper::conformance::compare_replay_envelopes;
use franken_whisper::storage::RunStore;
use franken_whisper::sync::{self, ConflictPolicy};

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

fn large_v3_turbo_available() -> bool {
    franken_whisper::native_engine::find_model_file("large-v3-turbo").is_some()
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
    for key in [
        "FRANKEN_WHISPER_ENC_INT8",
        "FW_ENC_ATTN_OUT_I8I32",
        "FW_ENC_INT8_ATTN_IN",
        "FW_ENC_INT8_FC1",
        "FW_ENC_WEIGHT_ROUNDTRIP",
    ] {
        cmd.env_remove(key);
    }
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

/// Spawn `franken_whisper robot run <args>` so native failures can be
/// characterized through the line-oriented machine contract.
fn run_robot(args: &[&str], extra_env: &[(&str, &str)], state_root: &Path) -> CliRun {
    let mut cmd = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"));
    cmd.args(["robot", "run"]);
    cmd.args(args);
    cmd.env("FRANKEN_WHISPER_STATE_DIR", state_root);
    for key in [
        "FRANKEN_WHISPER_ENC_INT8",
        "FW_ENC_ATTN_OUT_I8I32",
        "FW_ENC_INT8_ATTN_IN",
        "FW_ENC_INT8_FC1",
        "FW_ENC_WEIGHT_ROUNDTRIP",
    ] {
        cmd.env_remove(key);
    }
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let output = cmd.output().expect("spawn franken_whisper robot run");
    CliRun {
        status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

fn strict_ndjson_lines(run: &CliRun) -> Vec<Value> {
    run.stdout
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("each robot line must be JSON"))
        .collect()
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
    eprintln!("{scenario} wer={wer:.4} gate={max_wer:.4}");
    assert!(
        wer <= max_wer,
        "{scenario} WER {wer:.4} exceeds gate {max_wer:.4}\nREFERENCE: {reference}\nPRODUCED:  {produced}"
    );
}

fn fixture_json_text(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let json: Value =
        serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    if let Some(text) = json["text"].as_str() {
        return normalize_ws(text);
    }
    let segments = json["segments"]
        .as_array()
        .unwrap_or_else(|| panic!("{} missing text and segments", path.display()));
    normalize_ws(
        &segments
            .iter()
            .map(|seg| seg["text"].as_str().unwrap_or_default().trim())
            .collect::<Vec<_>>()
            .join(" "),
    )
}

#[test]
fn whisper_cpp_full_paired_fixture_corpus_wer_delta_budget() {
    let golden = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/golden"
    ));
    let mut pairs = Vec::new();
    for entry in std::fs::read_dir(&golden).expect("read golden fixture dir") {
        let path = entry.expect("dir entry").path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("whisper_cpp_")
            || !name.ends_with("_output.json")
            || name.ends_with("_native_output.json")
        {
            continue;
        }
        let native_name = name.replace("_output.json", "_native_output.json");
        let native = golden.join(native_name);
        if native.is_file() {
            pairs.push((path, native));
        }
    }
    pairs.sort();
    assert!(
        pairs.len() >= 8,
        "expected the full paired whisper.cpp fixture corpus, got {} pairs",
        pairs.len()
    );

    for (reference_path, native_path) in pairs {
        let reference = fixture_json_text(&reference_path);
        let native = fixture_json_text(&native_path);
        let wer = word_error_rate(&reference, &native);
        eprintln!(
            "fixture_corpus pair={} native={} wer_delta={wer:.4}",
            reference_path.file_name().unwrap().to_string_lossy(),
            native_path.file_name().unwrap().to_string_lossy()
        );
        assert!(
            wer <= 0.0,
            "fixture corpus WER drift {wer:.4}: {} vs {}",
            reference_path.display(),
            native_path.display()
        );
    }
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

#[test]
fn gated_robot_acoustic_diarization_accepts_canonical_dtw_projection() {
    if !tiny_en_available() {
        eprintln!(
            "SKIP gated_robot_acoustic_diarization_accepts_canonical_dtw_projection: tiny.en model missing"
        );
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();
    let source_db = state.path().join("source.sqlite3");
    let target_db = state.path().join("recovered.sqlite3");
    let snapshot = state.path().join("snapshot");
    let sync_state = state.path().join("sync-state");

    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole"),
    ];
    env.extend(bridge_bins_missing());

    let run = run_robot(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--model",
            "tiny.en",
            "--diarize",
            "--diarization-engine",
            "acoustic",
            "--db",
            source_db.to_str().expect("utf8 source db"),
        ],
        &env,
        state.path(),
    );

    assert!(
        run.status.success(),
        "canonical native DTW projection must complete: stdout={} stderr={}",
        run.stdout,
        run.stderr,
    );
    let lines = strict_ndjson_lines(&run);
    assert!(
        lines
            .iter()
            .any(|line| line["event"] == "stage" && line["code"] == "backend.ok"),
        "the real native backend must complete before acoustic projection"
    );
    assert!(
        lines
            .iter()
            .any(|line| line["event"] == "stage" && line["code"] == "diarize.ok"),
        "the acoustic diarization stage must accept the canonical timeline"
    );
    assert!(
        !lines.iter().any(|line| line["event"] == "run_error"),
        "successful robot output must not contain run_error"
    );
    let complete = lines
        .iter()
        .find(|line| line["event"] == "run_complete")
        .expect("robot mode must terminate with run_complete");
    assert!(
        complete["diarization"].is_object(),
        "run_complete must expose the typed diarization report"
    );
    let segments = complete["segments"]
        .as_array()
        .expect("run_complete segments array");
    assert!(
        !segments.is_empty(),
        "JFK inference must emit transcript units"
    );
    assert!(
        segments.iter().all(|segment| {
            segment["start_sec"]
                .as_f64()
                .zip(segment["end_sec"].as_f64())
                .is_some_and(|(start, end)| end > start)
        }),
        "every projected transcript unit must retain positive duration"
    );

    let run_id = complete["run_id"].as_str().expect("run_complete run_id");
    let source_store = RunStore::open(&source_db).expect("source store");
    let stored = source_store
        .load_run_details(run_id)
        .expect("stored run query")
        .expect("stored run");
    assert_eq!(
        stored
            .projection_timeline
            .as_ref()
            .expect("stored projection timeline")["schema_version"],
        franken_whisper::conformance::DTW_PROJECTION_SCHEMA_VERSION
    );
    assert_eq!(
        stored
            .projection_timeline
            .as_ref()
            .expect("stored projection timeline")["word_aligned_safe"],
        true
    );
    assert_eq!(
        stored
            .projection_timeline
            .as_ref()
            .expect("stored projection timeline")["fallback_reasons"],
        serde_json::json!([])
    );
    assert_eq!(
        serde_json::to_value(&stored.segments).expect("serialize stored segments"),
        complete["segments"],
        "robot output and SQLite-authoritative segments must agree"
    );
    assert_eq!(
        serde_json::to_value(&stored.diarization).expect("serialize stored diarization"),
        complete["diarization"],
        "robot output and SQLite-authoritative diarization must agree"
    );

    let manifest = sync::export(&source_db, &snapshot, &sync_state).expect("JSONL export");
    assert_eq!(manifest.schema_version, "1.1");
    assert_eq!(manifest.export_format_version, "1.0");
    sync::import(&target_db, &snapshot, &sync_state, ConflictPolicy::Reject)
        .expect("JSONL recovery into a fresh database");

    let recovered = RunStore::open(&target_db)
        .expect("recovered store")
        .load_run_details(run_id)
        .expect("recovered run query")
        .expect("recovered run");
    assert_eq!(
        serde_json::to_value(&recovered.segments).expect("serialize recovered segments"),
        serde_json::to_value(&stored.segments).expect("serialize stored segments")
    );
    assert_eq!(recovered.diarization, stored.diarization);
    assert_eq!(recovered.projection_timeline, stored.projection_timeline);
    assert!(
        compare_replay_envelopes(&stored.replay, &recovered.replay).within_tolerance(),
        "replay envelope must survive SQLite -> JSONL -> fresh SQLite"
    );

    let repeat = run_robot(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--model",
            "tiny.en",
            "--diarize",
            "--diarization-engine",
            "acoustic",
            "--no-persist",
        ],
        &env,
        state.path(),
    );
    assert!(
        repeat.status.success(),
        "deterministic repeat failed: stdout={} stderr={}",
        repeat.stdout,
        repeat.stderr
    );
    let repeat_lines = strict_ndjson_lines(&repeat);
    let repeat_complete = repeat_lines
        .iter()
        .find(|line| line["event"] == "run_complete")
        .expect("repeat run_complete");
    assert_eq!(repeat_complete["segments"], complete["segments"]);
    assert_eq!(repeat_complete["diarization"], complete["diarization"]);

    let human = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--model",
            "tiny.en",
            "--diarize",
            "--diarization-engine",
            "acoustic",
            "--no-persist",
            "--json",
        ],
        &env,
        state.path(),
    );
    assert!(
        human.status.success(),
        "human JSON rendering failed: stdout={} stderr={}",
        human.stdout,
        human.stderr
    );
    assert!(
        human.stdout.lines().count() > 1,
        "human JSON output must remain pretty-printed rather than NDJSON"
    );
    let human_report = human.report();
    assert_eq!(
        human_report["result"]["raw_output"]["projection_timeline"]["schema_version"],
        franken_whisper::conformance::DTW_PROJECTION_SCHEMA_VERSION
    );
    assert!(human_report["result"]["diarization"].is_object());
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

#[test]
fn gated_default_encoder_int8_policy_jfk_reference_wer_gate() {
    if !tiny_en_available() {
        eprintln!(
            "SKIP gated_default_encoder_int8_policy_jfk_reference_wer_gate: tiny.en model missing"
        );
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();

    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole"),
        // Keep the rejected all-i7 owner gate off; the quality-safe arm is now
        // selected by the default policy, not by FW_ENC_ATTN_OUT_I8I32.
        ("FRANKEN_WHISPER_ENC_INT8", "0"),
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
        "default encoder-int8 native run failed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    let report = run.report();

    assert_reference_wer_at_or_below(&report, 0.0, "default encoder-int8 JFK");
    let produced = normalize_ws(report["result"]["transcript"].as_str().unwrap_or_default());
    assert!(
        !produced.to_lowercase().contains("frank at"),
        "default quality-safe int8 must not emit the known all-i7 adversarial phrase: {produced}"
    );
    assert_eq!(
        report["result"]["raw_output"]["encoder_int8_policy"]["action"],
        "quality_safe_int8"
    );
    assert_eq!(
        report["result"]["raw_output"]["encoder_int8_policy"]["reason"],
        "calibrated_model_budget_pass"
    );

    let payload = backend_ok_payload(&report);
    assert_eq!(
        payload["implementation"], "native",
        "default encoder-int8 policy gate must run the native implementation"
    );
    assert_eq!(payload["execution_mode"], "native_only");
}

#[test]
fn gated_default_encoder_int8_large_v3_turbo_jfk_adversarial_probe() {
    if !large_v3_turbo_available() {
        eprintln!(
            "SKIP gated_default_encoder_int8_large_v3_turbo_jfk_adversarial_probe: large-v3-turbo model missing"
        );
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();

    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole"),
        ("FRANKEN_WHISPER_ENC_INT8", "0"),
    ];
    env.extend(bridge_bins_missing());

    let run = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--model",
            "large-v3-turbo",
            "--no-persist",
            "--json",
        ],
        &env,
        state.path(),
    );

    assert!(
        run.status.success(),
        "large-v3-turbo default encoder-int8 native run failed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    let report = run.report();
    assert_reference_wer_at_or_below(&report, 0.05, "large-v3-turbo default encoder-int8 JFK");
    let produced = normalize_ws(report["result"]["transcript"].as_str().unwrap_or_default());
    for sentinel in ["fellow americans", "ask not", "country"] {
        assert!(
            produced.to_lowercase().contains(sentinel),
            "large-v3-turbo adversarial sentinel `{sentinel}` missing from: {produced}"
        );
    }
    assert!(
        !produced.to_lowercase().contains("frank at"),
        "large-v3-turbo default quality-safe int8 must not emit known all-i7 phrase: {produced}"
    );
    assert_eq!(
        report["result"]["raw_output"]["encoder_int8_policy"]["action"],
        "quality_safe_int8"
    );
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
