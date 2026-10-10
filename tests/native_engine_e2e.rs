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
//! A separate unset-environment case proves the shipped defaults select native
//! ASR and native Sortformer diarization without either rollout override.
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

/// Absolute path to the audio fixture. It is gitignored (no media in the
/// repo) and provisioned with the model by `scripts/fetch_test_models.sh`, so
/// a model without it fails here by name instead of inside each scenario.
fn jfk_wav() -> PathBuf {
    let path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/native/jfk.wav"
    ));
    assert!(
        path.is_file(),
        "{} is missing: run scripts/fetch_test_models.sh (it fetches the pinned whisper.cpp sample)",
        path.display()
    );
    path
}

/// Gate: is the real `tiny.en` model resolvable on this machine? Mirrors the
/// library's own resolver so the gate can never drift from production lookup.
fn tiny_en_available() -> bool {
    franken_whisper::native_engine::find_model_file("tiny.en").is_some()
}

fn large_v3_turbo_available() -> bool {
    franken_whisper::native_engine::find_model_file("large-v3-turbo").is_some()
}

/// The default encoder is f32 on every target (bd-int8-encoder-mishears-m1q9):
/// on x86_64 AVX2 builds, where the quality-safe int8 kernels are compiled,
/// both calibrated models measured a corpus WER delta over the 0.0 budget;
/// other targets do not compile the int8 kernels at all.
fn expected_default_encoder_int8_policy() -> (&'static str, &'static str) {
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        ("f32", "calibration_wer_budget_exceeded")
    } else {
        ("f32", "cpu_feature_fallback")
    }
}

fn assert_default_encoder_int8_policy(report: &Value, context: &str) {
    let (expected_action, expected_reason) = expected_default_encoder_int8_policy();
    assert_eq!(
        report["result"]["raw_output"]["encoder_int8_policy"]["action"], expected_action,
        "{context} must report the action supported by this compiled target"
    );
    assert_eq!(
        report["result"]["raw_output"]["encoder_int8_policy"]["reason"], expected_reason,
        "{context} must report why the compiled target selected that action"
    );
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
        "FRANKEN_WHISPER_NATIVE_EXECUTION",
        "FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE",
        "FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL",
        "FRANKEN_WHISPER_BRIDGE_NATIVE_RECOVERY",
        "FRANKEN_WHISPER_STAGE_BUDGET_DIARIZE_MS",
        "FRANKEN_WHISPER_ACOUSTIC_DIARIZATION_ROLLOUT",
        "FW_ACOUSTIC_DIARIZATION_ROLLOUT",
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
        "FRANKEN_WHISPER_NATIVE_EXECUTION",
        "FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE",
        "FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL",
        "FRANKEN_WHISPER_BRIDGE_NATIVE_RECOVERY",
        "FRANKEN_WHISPER_STAGE_BUDGET_DIARIZE_MS",
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
            "--no-diarize",
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
fn gated_unset_environment_defaults_to_native_asr_and_sortformer() {
    if !franken_whisper::model_distribution::cached_whisper_is_ready()
        || !franken_whisper::model_distribution::cached_sortformer_is_ready()
    {
        eprintln!(
            "SKIP gated_unset_environment_defaults_to_native_asr_and_sortformer: native model package missing"
        );
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();
    let env = bridge_bins_missing();

    let run = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--no-persist",
            "--json",
        ],
        &env,
        state.path(),
    );

    assert!(
        run.status.success(),
        "unset-env default-native run failed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    let report = run.report();
    assert_reference_wer_at_or_below(&report, 0.05, "unset-env packaged Whisper default");
    let payload = backend_ok_payload(&report);
    assert_eq!(payload["implementation"], "native");
    assert_eq!(payload["execution_mode"], "native_only");
    assert_eq!(payload["native_rollout_stage"], "sole");
    assert_eq!(
        report["result"]["diarization"]["implementation"],
        "native-sortformer-v1"
    );
    assert_eq!(
        report["result"]["diarization"]["speaker_evidence_mode"],
        "sortformer_activity"
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
// (a2) encoder precision on JFK: the opt-in quality-safe int8 encoder
//      (FW_ENC_ATTN_OUT_I8I32=1) and the default (f32, DISC-010).
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
            "--no-diarize",
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
    // The opt-in still selects the int8 arm now that the default is f32.
    let policy = &report["result"]["raw_output"]["encoder_int8_policy"];
    assert_eq!(policy["action"], "quality_safe_int8");
    assert_eq!(policy["reason"], "operator_forced_quality_safe_int8");

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
        // Keep the rejected all-i7 owner gate off; with FW_ENC_ATTN_OUT_I8I32
        // unset the calibrated default policy picks the encoder (f32 since
        // calibration encoder-int8-calibration-2026-10-08, DISC-010).
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
            "--no-diarize",
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
        "the default encoder must not emit the known all-i7 adversarial phrase: {produced}"
    );
    assert_default_encoder_int8_policy(&report, "tiny.en default encoder-int8 policy");

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
            "--no-diarize",
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
        "large-v3-turbo's default encoder must not emit the known all-i7 phrase: {produced}"
    );
    assert_default_encoder_int8_policy(&report, "large-v3-turbo default encoder-int8 policy");
}

/// bd-zrgn: the macOS Metal encoder must give bit-identical output for the
/// same window, including when Metal reports a command buffer that did not
/// complete. A GPU restart (another process hanging the GPU) aborts every
/// command buffer in flight; the encoder used to read the unwritten outputs
/// anyway, which turned jfk's first turbo window into "We're Americans."
/// (avg_logprob -2.71) in a run that overlapped a restart storm.
///
/// Encodes jfk window 0 (3000 mel frames, the full first window) with
/// large-v3-turbo on the GPU stem route three times clean, then with 1..=3
/// injected failed command buffers (each leaves its outputs unwritten, as an
/// aborted buffer does), and requires every output to match the first bit for
/// bit. Gated on macOS, the turbo model, and a usable Metal GPU.
#[cfg(target_os = "macos")]
#[test]
fn gated_metal_turbo_encoder_output_is_bit_identical_across_runs_and_failed_command_buffers() {
    use franken_whisper::native_engine::{decode::LoadedModel, encoder, ggml::GgmlModel, mel};

    const NAME: &str =
        "gated_metal_turbo_encoder_output_is_bit_identical_across_runs_and_failed_command_buffers";
    let Some(path) = franken_whisper::native_engine::find_model_file("large-v3-turbo") else {
        eprintln!("SKIP {NAME}: large-v3-turbo model missing");
        return;
    };
    if !encoder::gpu_encoder_available() {
        eprintln!("SKIP {NAME}: no usable Metal GPU");
        return;
    }
    let model = GgmlModel::load(&path)
        .and_then(LoadedModel::from_ggml)
        .expect("load large-v3-turbo");
    let mut reader = hound::WavReader::open(jfk_wav()).expect("open jfk.wav");
    let spec = reader.spec();
    assert_eq!(
        (spec.channels, spec.sample_rate, spec.bits_per_sample),
        (1, 16_000, 16),
        "jfk.wav must be 16 kHz mono 16-bit"
    );
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| f32::from(s.expect("jfk.wav sample")) / 32_768.0)
        .collect();
    let full = mel::log_mel(&samples, &model.filters, 4).expect("log-mel");
    assert!(
        full.n_frames >= mel::FRAMES_PER_CHUNK,
        "log_mel pads a full first window"
    );
    let encode = || {
        let enc = encoder::forward_from_full_mel_window(
            &model.encoder,
            &full,
            0,
            mel::FRAMES_PER_CHUNK,
            4,
            &|| Ok(()),
        )
        .expect("encode window 0");
        assert_eq!(
            encoder::last_encoder_route(),
            "gpu_fused_stem",
            "the window must run on the Metal stem route"
        );
        enc.data
    };
    let differing = |got: &[f32], want: &[f32]| {
        assert_eq!(got.len(), want.len(), "encoder output length");
        got.iter()
            .zip(want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count()
    };

    let reference = encode();
    for run in 1..3 {
        assert_eq!(
            differing(&encode(), &reference),
            0,
            "clean run {run} differs from run 0"
        );
    }
    for faults in 1..=3 {
        let before = ft_kernel_metal::command_buffer_failures();
        ft_kernel_metal::inject_command_buffer_faults(faults);
        let got = encode();
        ft_kernel_metal::inject_command_buffer_faults(0);
        assert_eq!(
            differing(&got, &reference),
            0,
            "{faults} failed command buffer(s) changed the encoder output"
        );
        assert!(
            ft_kernel_metal::command_buffer_failures() >= before + u64::from(faults),
            "{faults} injected command-buffer failure(s) were not observed"
        );
    }
}

/// Runs `jfk.wav` through the native engine only, with `extra_env` on top of
/// the rollout settings, and returns the JSON report.
fn native_jfk_report(model: &str, extra_env: &[(&str, &str)]) -> Value {
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();
    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole"),
        ("FRANKEN_WHISPER_ENC_INT8", "0"),
    ];
    env.extend(bridge_bins_missing());
    env.extend_from_slice(extra_env);
    let run = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--model",
            model,
            "--no-diarize",
            "--no-persist",
            "--json",
        ],
        &env,
        state.path(),
    );
    assert!(
        run.status.success(),
        "{model} native run with {extra_env:?} failed\nstdout:\n{}\nstderr:\n{}",
        run.stdout,
        run.stderr
    );
    run.report()
}

/// The per-window decoder statistics of a report: they move with any change to
/// the encoder output, so equal values mean the same encoder arithmetic ran.
fn window_statistics(report: &Value) -> Vec<(Value, Value, Value)> {
    report["result"]["raw_output"]["windows"]
        .as_array()
        .expect("raw_output.windows array")
        .iter()
        .map(|w| {
            (
                w["tokens"].clone(),
                w["avg_logprob"].clone(),
                w["no_speech_prob"].clone(),
            )
        })
        .collect()
}

/// bd-int8-encoder-mishears-m1q9: the default run must BE the f32 encoder, not
/// merely transcribe JFK as well as it. With the old default (int8 on x86 AVX2)
/// the default's window statistics equal the int8 arm's and differ from f32's.
fn assert_default_encoder_is_the_f32_path(model: &str) {
    let default = native_jfk_report(model, &[]);
    let f32 = native_jfk_report(model, &[("FW_ENC_ATTN_OUT_I8I32", "0")]);
    let int8 = native_jfk_report(model, &[("FW_ENC_ATTN_OUT_I8I32", "1")]);

    assert_default_encoder_int8_policy(&default, &format!("{model} default"));
    let measured =
        default["result"]["raw_output"]["encoder_int8_policy"]["measured_corpus_wer_delta"]
            .as_f64()
            .expect("calibrated model reports its measured corpus WER delta");
    assert!(
        measured > 0.0,
        "{model} calibration row must be over the 0.0 budget, got {measured}"
    );
    let f32_policy = &f32["result"]["raw_output"]["encoder_int8_policy"];
    assert_eq!(f32_policy["action"], "f32");
    assert_eq!(f32_policy["reason"], "operator_f32_kill_switch");
    let int8_policy = &int8["result"]["raw_output"]["encoder_int8_policy"];
    assert_eq!(int8_policy["action"], "quality_safe_int8");
    assert_eq!(int8_policy["reason"], "operator_forced_quality_safe_int8");

    assert_eq!(
        default["result"]["transcript"], f32["result"]["transcript"],
        "{model}: the default transcript must be the f32 encoder's"
    );
    assert_eq!(
        window_statistics(&default),
        window_statistics(&f32),
        "{model}: the default must run the f32 encoder (window statistics differ)"
    );
    if cfg!(all(target_arch = "x86_64", target_feature = "avx2")) {
        // Sensitivity: the comparison above can tell the two arms apart.
        assert_ne!(
            window_statistics(&int8),
            window_statistics(&f32),
            "{model}: the int8 arm left every window statistic unchanged"
        );
    }
}

#[test]
fn gated_default_encoder_is_the_f32_path_tiny_en() {
    if !tiny_en_available() {
        eprintln!("SKIP gated_default_encoder_is_the_f32_path_tiny_en: tiny.en model missing");
        return;
    }
    assert_default_encoder_is_the_f32_path("tiny.en");
}

#[test]
fn gated_default_encoder_is_the_f32_path_large_v3_turbo() {
    if !large_v3_turbo_available() {
        eprintln!(
            "SKIP gated_default_encoder_is_the_f32_path_large_v3_turbo: large-v3-turbo model missing"
        );
        return;
    }
    assert_default_encoder_is_the_f32_path("large-v3-turbo");
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
            "--no-diarize",
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
    // silently succeed via some hidden path. Its input is a synthetic WAV, not
    // the provisioned fixture: with a missing input file it would fail (and
    // pass) for the wrong reason.
    let state = tempfile::tempdir().expect("tempdir");
    let wav = state.path().join("silence.wav");
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav, spec).expect("create silence wav");
        for _ in 0..16_000 {
            writer.write_sample(0_i16).expect("write sample");
        }
        writer.finalize().expect("finalize silence wav");
    }

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
            "--no-diarize",
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
            "--no-diarize",
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
            "--no-diarize",
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
// (f) diarization provenance regression: --backend whisper-diarization
//     currently labels native ASR segments with a legacy text/temporal
//     heuristic. Those labels are explicitly not acoustic or external speaker
//     evidence, so --diarize must run the native acoustic stage instead of
//     allowing the heuristic to short-circuit it.
// ===========================================================================

#[test]
fn gated_diarize_flag_with_legacy_backend_runs_acoustic_diarization() {
    if !tiny_en_available() {
        eprintln!(
            "SKIP gated_diarize_flag_with_legacy_backend_runs_acoustic_diarization: tiny.en model missing"
        );
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let wav = jfk_wav();

    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole"),
        ("FRANKEN_WHISPER_ACOUSTIC_DIARIZATION_ROLLOUT", "sole"),
    ];
    env.extend(bridge_bins_missing());

    let run = run_transcribe(
        &[
            "--input",
            wav.to_str().expect("utf8"),
            "--backend",
            "whisper-diarization",
            "--diarize",
            "--diarization-engine",
            "acoustic",
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

    // (b) The legacy heuristic must not masquerade as external speaker
    // evidence or suppress the requested waveform-derived acoustic pass.
    let events = report["events"].as_array().expect("report events array");
    let rollout = events
        .iter()
        .find(|event| event["code"].as_str() == Some("diarize.rollout"))
        .unwrap_or_else(|| {
            let codes: Vec<&str> = events.iter().filter_map(|e| e["code"].as_str()).collect();
            panic!("no diarize.rollout event; codes seen: {codes:?}");
        });
    assert_eq!(
        rollout["payload"]["external_speaker_evidence"], false,
        "legacy text/temporal labels must not satisfy the external-evidence provenance gate"
    );
    assert_eq!(
        rollout["payload"]["resolved_engine"], "acoustic",
        "the explicit test rollout must select native acoustic diarization after rejecting legacy heuristic evidence"
    );

    // The complete stage trace is authoritative: rollout resolution, one
    // start, the explicit no-hint evidence record, one assembled report, the
    // acoustic change summary, and one successful completion. A skip or a
    // second start would fail this exact sequence.
    let diarize_events: Vec<&str> = events
        .iter()
        .filter(|e| e["stage"].as_str() == Some("diarize"))
        .filter_map(|e| e["code"].as_str())
        .collect();
    assert_eq!(
        diarize_events,
        vec![
            "diarize.rollout",
            "diarize.start",
            "diarize.tiny_diarize_hint_evidence",
            "diarize.progress",
            "diarize.change",
            "diarize.ok"
        ],
        "the pipeline must run exactly one authoritative diarization pass before the native acoustic fallback is attached"
    );

    // (c) The attached durable report, not the backend's provisional labels,
    // records which evidence actually produced the final attribution.
    assert_eq!(
        report["result"]["diarization"]["speaker_evidence_mode"],
        "acoustic_v2"
    );
    assert_eq!(
        report["result"]["raw_output"]["diarizer"], "text-temporal-heuristic",
        "backend provenance must continue to disclose its provisional heuristic"
    );

    // This short single-speaker fixture does not contain enough independent
    // recurrence to support an identity profile. The acoustic engine must
    // therefore replace the provisional heuristic labels with explicit
    // unknown attribution instead of preserving counterfeit confidence.
    assert_eq!(
        report["result"]["diarization"]["fallback_status"],
        "speaker_count_unresolved"
    );
    assert_eq!(
        report["result"]["diarization"]["speaker_count"]["unknown_voiced_share"],
        1.0
    );
    let segments = report["result"]["segments"]
        .as_array()
        .expect("segments array");
    assert!(!segments.is_empty(), "diarization produced no segments");
    for seg in segments {
        assert!(
            seg["speaker"].is_null(),
            "unsupported legacy labels must be removed rather than promoted to acoustic evidence"
        );
    }
}

// ===========================================================================
// (g) batch mode (bd-batch-transcribe-rraf): one process, one model load,
//     every input byte-identical to its own single-input run.
// ===========================================================================

/// Copy the first `seconds` of `source` into `dest` (same WAV spec).
fn write_wav_prefix(source: &Path, dest: &Path, seconds: f64) {
    let mut reader = hound::WavReader::open(source).expect("open source wav");
    let spec = reader.spec();
    let keep = (f64::from(spec.sample_rate) * seconds) as usize * usize::from(spec.channels);
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .take(keep)
        .map(|sample| sample.expect("source sample"))
        .collect();
    let mut writer = hound::WavWriter::create(dest, spec).expect("create prefix wav");
    for sample in samples {
        writer.write_sample(sample).expect("write sample");
    }
    writer.finalize().expect("finalize prefix wav");
}

/// The per-input result fields a batch must reproduce exactly: transcript,
/// language, segments (DTW word timestamps and confidences) and the
/// acceleration report the confidences came from.
fn batch_comparable(result: &Value) -> Value {
    serde_json::json!({
        "backend": result["backend"],
        "language": result["language"],
        "transcript": result["transcript"],
        "segments": result["segments"],
        "acceleration": result["acceleration"],
        "word_timestamps": result["raw_output"]["word_timestamps"],
    })
}

#[test]
fn gated_batch_matches_single_input_runs() {
    // tiny.en when provisioned; otherwise the default release package, which
    // also exercises the once-per-batch package authentication.
    let model_args: &[&str] = if tiny_en_available() {
        &["--model", "tiny.en"]
    } else if large_v3_turbo_available() {
        &[]
    } else {
        eprintln!("SKIP gated_batch_matches_single_input_runs: no native whisper model");
        return;
    };
    let state = tempfile::tempdir().expect("tempdir");
    let jfk = jfk_wav();
    let head = state.path().join("jfk_head.wav");
    write_wav_prefix(&jfk, &head, 4.5);
    let missing = state.path().join("missing.wav");
    let env = bridge_bins_missing();
    let mut flags = model_args.to_vec();
    flags.extend([
        "--language",
        "en",
        "--no-diarize",
        "--no-persist",
        "--max-segment-length",
        "1",
        "--split-on-word",
        "--json",
    ]);

    let single = |clip: &Path| -> Value {
        let mut args = vec!["--input", clip.to_str().expect("utf8")];
        args.extend(&flags);
        let run = run_transcribe(&args, &env, state.path());
        assert!(
            run.status.success(),
            "single run failed\nstdout:\n{}\nstderr:\n{}",
            run.stdout,
            run.stderr
        );
        batch_comparable(&run.report()["result"])
    };
    let single_jfk = single(&jfk);
    let single_head = single(&head);
    assert_ne!(single_jfk["transcript"], single_head["transcript"]);
    assert!(
        single_jfk["segments"]
            .as_array()
            .is_some_and(|segments| segments.len() > 5),
        "word mode yields per-word segments"
    );

    // The repeated head clip also exercises the in-process transcript cache.
    let order = [&head, &missing, &jfk, &head];
    let list = state.path().join("inputs.txt");
    std::fs::write(
        &list,
        order
            .iter()
            .map(|path| format!("{}\n", path.display()))
            .collect::<String>(),
    )
    .expect("write input list");
    let mut args = vec!["--inputs-from", list.to_str().expect("utf8")];
    args.extend(&flags);
    let run = run_transcribe(&args, &env, state.path());
    assert_eq!(
        run.status.code(),
        Some(1),
        "the missing input makes the batch exit 1\nstderr:\n{}",
        run.stderr
    );
    let records = strict_ndjson_lines(&run);
    assert_eq!(records.len(), 4, "one record per input:\n{}", run.stdout);
    let expected = [
        Some(&single_head),
        None,
        Some(&single_jfk),
        Some(&single_head),
    ];
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record["index"], index);
        assert_eq!(record["input"], order[index].to_str().expect("utf8"));
        match expected[index] {
            Some(single) => {
                assert_eq!(record["status"], "ok", "{record:#?}");
                assert_eq!(
                    &batch_comparable(&record["report"]["result"]),
                    single,
                    "batch input {index} must be byte-identical to its single-input run"
                );
            }
            None => assert_eq!(record["status"], "error", "{record:#?}"),
        }
    }
}

// ===========================================================================
// bd-threads-flag-unbounded-f4pq: `--threads N` bounds every thread pool
// ===========================================================================

/// Threads a `transcribe` process may hold on top of its `--threads N`
/// compute pool, 12 in total. Each is a fixed, non-compute thread: main (1),
/// the Ctrl-C handler (1), the orchestrator runtime (2 workers and up to 4
/// blocking: 6), the running stage thread (1), a batch or robot run's
/// per-input worker (1), the model-hash warm thread of an unauthenticated
/// model (1), and window pipelining's encoder thread (1; no-timestamps runs
/// only, and it only waits on the pool). Before the fix a `--threads 1` run
/// peaked at 86 threads on a 128-thread host (rayon's host-sized global pool
/// plus per-kernel scoped threads), and well above 1 + 12 on any host with 4
/// or more cores.
#[cfg(target_os = "linux")]
const NON_COMPUTE_THREAD_ALLOWANCE: usize = 12;

/// A CLI run observed from outside through `/proc` while it executed.
#[cfg(target_os = "linux")]
struct ThreadObservedRun {
    run: CliRun,
    /// Highest `Threads:` count sampled from `/proc/<pid>/status`.
    peak_threads: usize,
    /// Every thread id whose name marks it as a compute-pool worker
    /// (`fw-compute-*`) seen at any sample.
    compute_tids: std::collections::BTreeSet<String>,
    samples: usize,
}

/// Spawn `franken_whisper <subcommand> <args>` and poll `/proc/<pid>` every
/// 250 µs until the child exits. Sampling can miss a sub-millisecond spike,
/// so `peak_threads` is a lower bound on the true peak; the pool workers live
/// for the whole run and are always seen.
#[cfg(target_os = "linux")]
fn run_observing_threads(
    subcommand: &[&str],
    args: &[&str],
    extra_env: &[(&str, &str)],
    state_root: &Path,
) -> ThreadObservedRun {
    let stdout_path = state_root.join("observed.stdout");
    let stderr_path = state_root.join("observed.stderr");
    let mut cmd = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"));
    cmd.args(subcommand);
    cmd.args(args);
    cmd.env("FRANKEN_WHISPER_STATE_DIR", state_root);
    // Only `--threads` may size the run: no environment pool overrides, and
    // no opt-in extra load pool.
    for key in [
        "RAYON_NUM_THREADS",
        "FW_LOAD_WORKERS",
        "FRANKEN_WHISPER_NATIVE_EXECUTION",
        "FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE",
        "FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL",
        "FRANKEN_WHISPER_BRIDGE_NATIVE_RECOVERY",
        "FRANKEN_WHISPER_ACOUSTIC_DIARIZATION_ROLLOUT",
        "FW_ACOUSTIC_DIARIZATION_ROLLOUT",
    ] {
        cmd.env_remove(key);
    }
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    cmd.stdout(std::fs::File::create(&stdout_path).expect("stdout file"));
    cmd.stderr(std::fs::File::create(&stderr_path).expect("stderr file"));
    let mut child = cmd.spawn().expect("spawn franken_whisper transcribe");
    let status_path = format!("/proc/{}/status", child.id());
    let task_dir = format!("/proc/{}/task", child.id());
    let mut peak_threads = 0usize;
    let mut compute_tids = std::collections::BTreeSet::new();
    let mut samples = 0usize;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll child") {
            break status;
        }
        if let Some(threads) = std::fs::read_to_string(&status_path).ok().and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("Threads:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
        }) {
            peak_threads = peak_threads.max(threads);
        }
        if samples.is_multiple_of(8)
            && let Ok(entries) = std::fs::read_dir(&task_dir)
        {
            for entry in entries.flatten() {
                let comm = std::fs::read_to_string(entry.path().join("comm")).unwrap_or_default();
                if comm.starts_with("fw-compute") {
                    compute_tids.insert(entry.file_name().to_string_lossy().into_owned());
                }
            }
        }
        samples += 1;
        std::thread::sleep(std::time::Duration::from_micros(250));
    };
    ThreadObservedRun {
        run: CliRun {
            status,
            stdout: std::fs::read_to_string(&stdout_path).expect("read stdout"),
            stderr: std::fs::read_to_string(&stderr_path).expect("read stderr"),
        },
        peak_threads,
        compute_tids,
        samples,
    }
}

/// Assert the observed run stayed inside `threads` compute workers plus the
/// fixed allowance, and that its compute pool had exactly `threads` workers.
#[cfg(target_os = "linux")]
fn assert_threads_bounded(observed: &ThreadObservedRun, threads: usize, scenario: &str) {
    assert!(
        observed.run.status.success(),
        "{scenario} failed\nstdout:\n{}\nstderr:\n{}",
        observed.run.stdout,
        observed.run.stderr
    );
    eprintln!(
        "{scenario}: --threads {threads}: peak {} threads ({} samples), {} compute-pool workers",
        observed.peak_threads,
        observed.samples,
        observed.compute_tids.len()
    );
    assert!(
        observed.samples > 0,
        "{scenario}: the run was never sampled"
    );
    assert!(
        observed.peak_threads <= threads + NON_COMPUTE_THREAD_ALLOWANCE,
        "{scenario}: --threads {threads} peaked at {} threads (bound {threads} + {NON_COMPUTE_THREAD_ALLOWANCE})",
        observed.peak_threads
    );
    assert_eq!(
        observed.compute_tids.len(),
        threads,
        "{scenario}: the run must compute on exactly one {threads}-worker pool, saw compute workers {:?}",
        observed.compute_tids
    );
}

/// `--threads 1` and `--threads 4` hold the whole process to the pool plus
/// the fixed allowance, on the narration consumer's flags (DTW word
/// timestamps), and the transcript is byte-identical at both widths.
#[cfg(target_os = "linux")]
#[test]
fn gated_threads_flag_bounds_peak_threads_and_keeps_the_transcript() {
    if !tiny_en_available() {
        eprintln!(
            "SKIP gated_threads_flag_bounds_peak_threads_and_keeps_the_transcript: tiny.en model missing"
        );
        return;
    }
    let wav = jfk_wav();
    let mut comparable = Vec::new();
    for threads in [1usize, 4] {
        let state = tempfile::tempdir().expect("tempdir");
        let threads_arg = threads.to_string();
        let observed = run_observing_threads(
            &["transcribe"],
            &[
                "--input",
                wav.to_str().expect("utf8"),
                "--model",
                "tiny.en",
                "--language",
                "en",
                "--no-diarize",
                "--no-persist",
                "--max-segment-length",
                "1",
                "--split-on-word",
                "--json",
                "--threads",
                &threads_arg,
            ],
            &bridge_bins_missing(),
            state.path(),
        );
        assert_threads_bounded(&observed, threads, "word-timestamp transcribe");
        let report = observed.run.report();
        assert_transcript_matches_reference(&report);
        comparable.push(batch_comparable(&report["result"]));
    }
    assert_eq!(
        comparable[0], comparable[1],
        "--threads 1 and --threads 4 must produce byte-identical segments"
    );
}

/// A batch reuses one pool for every input (no per-input pool); the native
/// acoustic diarization stage computes on the same bounded pool; and the
/// robot surface's `--threads` bounds a run the same way.
#[cfg(target_os = "linux")]
#[test]
fn gated_threads_flag_bounds_batch_diarization_and_robot_runs() {
    if !tiny_en_available() {
        eprintln!(
            "SKIP gated_threads_flag_bounds_batch_diarization_and_robot_runs: tiny.en model missing"
        );
        return;
    }
    let state = tempfile::tempdir().expect("tempdir");
    let jfk = jfk_wav();
    let head = state.path().join("jfk_head.wav");
    write_wav_prefix(&jfk, &head, 4.5);
    let batch = run_observing_threads(
        &["transcribe"],
        &[
            "--input",
            jfk.to_str().expect("utf8"),
            "--input",
            head.to_str().expect("utf8"),
            "--input",
            jfk.to_str().expect("utf8"),
            "--model",
            "tiny.en",
            "--language",
            "en",
            "--no-diarize",
            "--no-persist",
            "--json",
            "--threads",
            "2",
        ],
        &bridge_bins_missing(),
        state.path(),
    );
    assert_threads_bounded(&batch, 2, "3-input batch");
    assert_eq!(strict_ndjson_lines(&batch.run).len(), 3);

    let state = tempfile::tempdir().expect("tempdir");
    let mut env = vec![
        ("FRANKEN_WHISPER_NATIVE_EXECUTION", "1"),
        ("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole"),
        ("FRANKEN_WHISPER_ACOUSTIC_DIARIZATION_ROLLOUT", "sole"),
    ];
    env.extend(bridge_bins_missing());
    let diarized = run_observing_threads(
        &["transcribe"],
        &[
            "--input",
            jfk.to_str().expect("utf8"),
            "--backend",
            "whisper-diarization",
            "--diarize",
            "--diarization-engine",
            "acoustic",
            "--model",
            "tiny.en",
            "--no-persist",
            "--json",
            "--threads",
            "2",
        ],
        &env,
        state.path(),
    );
    assert_threads_bounded(&diarized, 2, "acoustic diarization run");

    let state = tempfile::tempdir().expect("tempdir");
    let robot = run_observing_threads(
        &["robot", "run"],
        &[
            "--input",
            jfk.to_str().expect("utf8"),
            "--model",
            "tiny.en",
            "--language",
            "en",
            "--no-diarize",
            "--no-persist",
            "--threads",
            "3",
        ],
        &bridge_bins_missing(),
        state.path(),
    );
    assert_threads_bounded(&robot, 3, "robot run");
    let lines = strict_ndjson_lines(&robot.run);
    assert!(
        lines
            .iter()
            .any(|line| line["event"].as_str() == Some("run_complete")),
        "robot run must complete:\n{}",
        robot.run.stdout
    );
}
