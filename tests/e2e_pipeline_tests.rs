//! End-to-end pipeline tests for FrankenWhisperEngine.
//!
//! These tests exercise the full pipeline from engine creation through to
//! RunReport output.  Real backends (whisper.cpp, insanely-fast-whisper) are
//! replaced by mock shell scripts (see tests/mocks/) via environment variables.
//!
//! Because Rust 2024 edition marks `std::env::set_var` as `unsafe` and this
//! crate forbids unsafe code, each test spawns *itself* as a subprocess with
//! the required env vars set via `Command::env()`.  The outer test manages
//! temp dirs and env; the inner test (detected via a sentinel env var) runs
//! the actual engine logic.

mod helpers;

use std::collections::HashMap;
use std::path::PathBuf;

use franken_whisper::FrankenWhisperEngine;
use franken_whisper::model::*;
use franken_whisper::storage::RunStore;

// ---------------------------------------------------------------------------
// Subprocess helpers
// ---------------------------------------------------------------------------

/// Returns true when running inside a re-invoked subprocess.
fn is_subprocess() -> bool {
    std::env::var("__FRANKEN_E2E_SUBPROCESS").is_ok()
}

/// Spawn the current test binary as a subprocess, running only the named test
/// function, with extra env vars set on the child process (no `unsafe`).
fn run_in_subprocess(test_name: &str, env: &HashMap<String, String>) {
    let exe = std::env::current_exe().expect("determine test binary path");
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg(test_name)
        .arg("--exact")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env("__FRANKEN_E2E_SUBPROCESS", "1");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let output = cmd.output().expect("spawn subprocess");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "subprocess test '{test_name}' failed (exit={}):\n\
             --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            output.status
        );
    }
}

/// Build the full env map for an e2e test: mock backend paths + state dir.
fn build_test_env(state_dir: &std::path::Path) -> HashMap<String, String> {
    let mut env = helpers::mock_backend_env();
    env.insert(
        "FRANKEN_WHISPER_STATE_DIR".to_owned(),
        state_dir.display().to_string(),
    );
    env
}

/// Return true when mock scripts are present and executable.
fn mocks_available() -> bool {
    let mocks = helpers::mocks_dir();
    let scripts = [
        mocks.join("mock_whisper_cpp.sh"),
        mocks.join("mock_insanely_fast.sh"),
    ];
    for script in &scripts {
        if !script.exists() {
            return false;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(meta) = std::fs::metadata(script) {
                if meta.permissions().mode() & 0o111 == 0 {
                    return false;
                }
            } else {
                return false;
            }
        }
    }
    true
}

/// Return true when ffmpeg is available on PATH for normalization.
fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Return true when all e2e pipeline external prerequisites are available.
fn pipeline_prereqs_available() -> bool {
    mocks_available() && ffmpeg_available()
}

/// Build a `TranscribeRequest` for end-to-end use.
fn build_e2e_request(
    wav_path: &std::path::Path,
    backend: BackendKind,
    db_path: &std::path::Path,
    persist: bool,
    timeout_ms: Option<u64>,
) -> TranscribeRequest {
    TranscribeRequest {
        input: InputSource::File {
            path: wav_path.to_path_buf(),
        },
        backend,
        model: None,
        language: Some("en".to_owned()),
        translate: false,
        diarize: false,
        persist,
        db_path: db_path.to_path_buf(),
        timeout_ms,
        backend_params: BackendParams::default(),
    }
}

/// Read FRANKEN_WHISPER_STATE_DIR from the subprocess environment.
fn state_dir_from_env() -> PathBuf {
    PathBuf::from(
        std::env::var("FRANKEN_WHISPER_STATE_DIR").expect("STATE_DIR must be set in subprocess"),
    )
}

// ---------------------------------------------------------------------------
// Test 1: Full pipeline with mock whisper.cpp
// ---------------------------------------------------------------------------

#[test]
fn e2e_pipeline_with_mock_whisper_cpp() {
    if !is_subprocess() {
        if !pipeline_prereqs_available() {
            eprintln!("Skipping: mock backends or ffmpeg not available");
            return;
        }
        let state_tmp = tempfile::tempdir().expect("create state temp dir");
        run_in_subprocess(
            "e2e_pipeline_with_mock_whisper_cpp",
            &build_test_env(state_tmp.path()),
        );
        return;
    }

    let state_dir = state_dir_from_env();
    let wav_tmp = tempfile::tempdir().expect("create wav temp dir");
    let wav_path = helpers::generate_test_wav(wav_tmp.path(), "input.wav", 1.0, 440.0);
    let db_path = state_dir.join("e2e_wcpp.sqlite3");

    let engine = FrankenWhisperEngine::new().expect("engine creation should succeed");
    let request = build_e2e_request(&wav_path, BackendKind::WhisperCpp, &db_path, false, None);
    let report = engine
        .transcribe(request)
        .expect("transcribe should succeed");

    assert!(!report.run_id.is_empty(), "run_id must be non-empty");
    assert!(!report.trace_id.is_empty(), "trace_id must be non-empty");
    assert!(
        !report.started_at_rfc3339.is_empty(),
        "started_at must be populated"
    );
    assert!(
        !report.finished_at_rfc3339.is_empty(),
        "finished_at must be populated"
    );
    assert!(
        !report.result.transcript.is_empty(),
        "transcript should contain text from mock backend"
    );
    assert!(
        !report.events.is_empty(),
        "events list should not be empty after a full pipeline run"
    );
    assert_eq!(
        report.result.backend,
        BackendKind::WhisperCpp,
        "resolved backend should be whisper_cpp"
    );
    let align_event = report
        .events
        .iter()
        .find(|event| event.code == "align.ok" || event.code == "align.fallback")
        .expect("full pipeline should report the alignment outcome");
    assert_eq!(
        align_event.payload["method"].as_str(),
        Some("char_density_heuristic"),
        "bridge output without typed DTW proof must retain its honest fallback label"
    );
    assert_eq!(
        align_event.payload["energy_evidence_available"].as_bool(),
        Some(true),
        "the normalized public synthetic WAV should provide transient RMS evidence"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Full pipeline with mock insanely-fast-whisper
// ---------------------------------------------------------------------------

#[test]
fn e2e_pipeline_with_mock_insanely_fast() {
    if !is_subprocess() {
        if !pipeline_prereqs_available() {
            eprintln!("Skipping: mock backends or ffmpeg not available");
            return;
        }
        let state_tmp = tempfile::tempdir().expect("create state temp dir");
        run_in_subprocess(
            "e2e_pipeline_with_mock_insanely_fast",
            &build_test_env(state_tmp.path()),
        );
        return;
    }

    let state_dir = state_dir_from_env();
    let wav_tmp = tempfile::tempdir().expect("create wav temp dir");
    let wav_path = helpers::generate_test_wav(wav_tmp.path(), "input.wav", 1.0, 440.0);
    let db_path = state_dir.join("e2e_insanely.sqlite3");

    let engine = FrankenWhisperEngine::new().expect("engine creation should succeed");
    let request = build_e2e_request(&wav_path, BackendKind::InsanelyFast, &db_path, false, None);
    let report = engine
        .transcribe(request)
        .expect("transcribe should succeed");

    assert!(!report.run_id.is_empty(), "run_id must be non-empty");
    assert!(!report.trace_id.is_empty(), "trace_id must be non-empty");
    assert!(
        !report.result.transcript.is_empty(),
        "transcript should contain text from mock backend"
    );
    assert!(!report.events.is_empty(), "events list should not be empty");
    assert_eq!(
        report.result.backend,
        BackendKind::InsanelyFast,
        "resolved backend should be insanely_fast"
    );
}

// ---------------------------------------------------------------------------
// Test 3: run_id is a valid UUID
// ---------------------------------------------------------------------------

#[test]
fn e2e_pipeline_generates_run_id() {
    if !is_subprocess() {
        if !pipeline_prereqs_available() {
            eprintln!("Skipping: mock backends or ffmpeg not available");
            return;
        }
        let state_tmp = tempfile::tempdir().expect("create state temp dir");
        run_in_subprocess(
            "e2e_pipeline_generates_run_id",
            &build_test_env(state_tmp.path()),
        );
        return;
    }

    let state_dir = state_dir_from_env();
    let wav_tmp = tempfile::tempdir().expect("create wav temp dir");
    let wav_path = helpers::generate_test_wav(wav_tmp.path(), "input.wav", 0.5, 440.0);
    let db_path = state_dir.join("e2e_runid.sqlite3");

    let engine = FrankenWhisperEngine::new().expect("engine creation should succeed");
    let request = build_e2e_request(&wav_path, BackendKind::WhisperCpp, &db_path, false, None);
    let report = engine
        .transcribe(request)
        .expect("transcribe should succeed");

    let parts: Vec<&str> = report.run_id.split('-').collect();
    assert_eq!(
        parts.len(),
        5,
        "run_id should have 5 dash-separated groups, got: {}",
        report.run_id
    );
    assert_eq!(parts[0].len(), 8, "group 1 should be 8 hex chars");
    assert_eq!(parts[1].len(), 4, "group 2 should be 4 hex chars");
    assert_eq!(parts[2].len(), 4, "group 3 should be 4 hex chars");
    assert_eq!(parts[3].len(), 4, "group 4 should be 4 hex chars");
    assert_eq!(parts[4].len(), 12, "group 5 should be 12 hex chars");
    assert!(
        report
            .run_id
            .chars()
            .all(|c| c.is_ascii_hexdigit() || c == '-'),
        "run_id should contain only hex digits and dashes, got: {}",
        report.run_id
    );
    assert!(
        parts[2].starts_with('4'),
        "run_id UUID version nibble should be 4, got group 3: {}",
        parts[2]
    );
}

// ---------------------------------------------------------------------------
// Test 4: trace_id is a hex string
// ---------------------------------------------------------------------------

#[test]
fn e2e_pipeline_populates_trace_id() {
    if !is_subprocess() {
        if !pipeline_prereqs_available() {
            eprintln!("Skipping: mock backends or ffmpeg not available");
            return;
        }
        let state_tmp = tempfile::tempdir().expect("create state temp dir");
        run_in_subprocess(
            "e2e_pipeline_populates_trace_id",
            &build_test_env(state_tmp.path()),
        );
        return;
    }

    let state_dir = state_dir_from_env();
    let wav_tmp = tempfile::tempdir().expect("create wav temp dir");
    let wav_path = helpers::generate_test_wav(wav_tmp.path(), "input.wav", 0.5, 440.0);
    let db_path = state_dir.join("e2e_traceid.sqlite3");

    let engine = FrankenWhisperEngine::new().expect("engine creation should succeed");
    let request = build_e2e_request(&wav_path, BackendKind::WhisperCpp, &db_path, false, None);
    let report = engine
        .transcribe(request)
        .expect("transcribe should succeed");

    assert!(!report.trace_id.is_empty(), "trace_id must be non-empty");
    assert!(
        report.trace_id.chars().all(|c| c.is_ascii_hexdigit()),
        "trace_id should contain only hex digits, got: {}",
        report.trace_id
    );
    assert_eq!(
        report.trace_id,
        report.trace_id.to_ascii_lowercase(),
        "trace_id should be lowercase hex"
    );
}

// ---------------------------------------------------------------------------
// Test 5: Event seq numbers are monotonically increasing
// ---------------------------------------------------------------------------

#[test]
fn e2e_pipeline_events_are_monotonic() {
    if !is_subprocess() {
        if !pipeline_prereqs_available() {
            eprintln!("Skipping: mock backends or ffmpeg not available");
            return;
        }
        let state_tmp = tempfile::tempdir().expect("create state temp dir");
        run_in_subprocess(
            "e2e_pipeline_events_are_monotonic",
            &build_test_env(state_tmp.path()),
        );
        return;
    }

    let state_dir = state_dir_from_env();
    let wav_tmp = tempfile::tempdir().expect("create wav temp dir");
    let wav_path = helpers::generate_test_wav(wav_tmp.path(), "input.wav", 0.5, 440.0);
    let db_path = state_dir.join("e2e_monotonic.sqlite3");

    let engine = FrankenWhisperEngine::new().expect("engine creation should succeed");
    let request = build_e2e_request(&wav_path, BackendKind::WhisperCpp, &db_path, false, None);
    let report = engine
        .transcribe(request)
        .expect("transcribe should succeed");

    assert!(
        report.events.len() >= 2,
        "pipeline should emit at least 2 events, got {}",
        report.events.len()
    );

    helpers::assert_events_ordered(&report.events);

    for (i, event) in report.events.iter().enumerate() {
        assert!(event.seq > 0, "event {i}: seq must be positive");
        assert!(
            !event.ts_rfc3339.is_empty(),
            "event {i}: ts_rfc3339 must be non-empty"
        );
        assert!(
            !event.stage.is_empty(),
            "event {i}: stage must be non-empty"
        );
        assert!(!event.code.is_empty(), "event {i}: code must be non-empty");
        assert!(
            !event.message.is_empty(),
            "event {i}: message must be non-empty"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 6: Pipeline persists to SQLite when requested
// ---------------------------------------------------------------------------

#[test]
fn e2e_pipeline_with_persist() {
    if !is_subprocess() {
        if !pipeline_prereqs_available() {
            eprintln!("Skipping: mock backends or ffmpeg not available");
            return;
        }
        let state_tmp = tempfile::tempdir().expect("create state temp dir");
        run_in_subprocess(
            "e2e_pipeline_with_persist",
            &build_test_env(state_tmp.path()),
        );
        return;
    }

    let state_dir = state_dir_from_env();
    let wav_tmp = tempfile::tempdir().expect("create wav temp dir");
    let wav_path = helpers::generate_test_wav(wav_tmp.path(), "input.wav", 0.5, 440.0);
    let db_path = state_dir.join("e2e_persist.sqlite3");

    let engine = FrankenWhisperEngine::new().expect("engine creation should succeed");
    let request = build_e2e_request(&wav_path, BackendKind::WhisperCpp, &db_path, true, None);
    let report = engine
        .transcribe(request)
        .expect("transcribe should succeed");

    assert!(
        db_path.exists(),
        "SQLite database should exist at {}",
        db_path.display()
    );

    let store = RunStore::open(&db_path).expect("open persisted database");
    let runs = store.list_recent_runs(10).expect("list runs from database");
    assert!(
        !runs.is_empty(),
        "database should contain at least one run after persist"
    );

    assert_eq!(
        runs[0].run_id, report.run_id,
        "persisted run_id should match report run_id"
    );

    let has_persist_event = report
        .events
        .iter()
        .any(|e| e.stage == "persist" && e.code.contains("persist"));
    assert!(
        has_persist_event,
        "events should include a persist-stage event when persist=true"
    );
}

// ---------------------------------------------------------------------------
// Test 7: Pipeline respects timeout_ms
// ---------------------------------------------------------------------------

#[test]
fn e2e_pipeline_with_timeout() {
    if !is_subprocess() {
        if !pipeline_prereqs_available() {
            eprintln!("Skipping: mock backends or ffmpeg not available");
            return;
        }
        let state_tmp = tempfile::tempdir().expect("create state temp dir");
        run_in_subprocess(
            "e2e_pipeline_with_timeout",
            &build_test_env(state_tmp.path()),
        );
        return;
    }

    let state_dir = state_dir_from_env();
    let wav_tmp = tempfile::tempdir().expect("create wav temp dir");
    let wav_path = helpers::generate_test_wav(wav_tmp.path(), "input.wav", 0.5, 440.0);
    let db_path = state_dir.join("e2e_timeout.sqlite3");

    let engine = FrankenWhisperEngine::new().expect("engine creation should succeed");

    let generous_timeout_ms = 120_000;
    let request = build_e2e_request(
        &wav_path,
        BackendKind::WhisperCpp,
        &db_path,
        false,
        Some(generous_timeout_ms),
    );
    let report = engine
        .transcribe(request)
        .expect("transcribe with generous timeout should succeed");

    assert!(
        !report.run_id.is_empty(),
        "run_id must be non-empty when pipeline completes within timeout"
    );
    assert!(
        !report.result.transcript.is_empty(),
        "transcript should be populated when pipeline finishes within timeout"
    );

    let budget_event = report
        .events
        .iter()
        .find(|e| e.code == "orchestration.budgets");
    assert!(
        budget_event.is_some(),
        "should emit an orchestration.budgets event"
    );
    if let Some(evt) = budget_event {
        let payload_timeout = evt.payload.get("request_timeout_ms");
        assert!(
            payload_timeout.is_some(),
            "orchestration.budgets payload should include request_timeout_ms"
        );
        assert_eq!(
            payload_timeout.unwrap().as_u64(),
            Some(generous_timeout_ms),
            "request_timeout_ms in payload should match the requested value"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 8: Non-existent input file produces a clean error
// ---------------------------------------------------------------------------

#[test]
fn e2e_pipeline_nonexistent_input_fails_cleanly() {
    if !is_subprocess() {
        let state_tmp = tempfile::tempdir().expect("create state temp dir");
        run_in_subprocess(
            "e2e_pipeline_nonexistent_input_fails_cleanly",
            &build_test_env(state_tmp.path()),
        );
        return;
    }

    let state_dir = state_dir_from_env();
    let bogus_path = PathBuf::from("/tmp/franken_whisper_does_not_exist_e2e_test.wav");
    assert!(
        !bogus_path.exists(),
        "test precondition: bogus path should not exist"
    );

    let db_path = state_dir.join("e2e_nofile.sqlite3");
    let engine = FrankenWhisperEngine::new().expect("engine creation should succeed");
    let request = build_e2e_request(&bogus_path, BackendKind::WhisperCpp, &db_path, false, None);

    let result = engine.transcribe(request);
    assert!(
        result.is_err(),
        "transcribe should return an error for a non-existent input file"
    );

    let error = result.unwrap_err();
    let error_text = error.to_string();
    assert!(
        error_text.contains("not_exist")
            || error_text.contains("No such file")
            || error_text.contains("i/o failure")
            || error_text.contains("not found")
            || error_text.contains("materialize"),
        "error message should indicate a file-not-found condition, got: {error_text}"
    );
}

// ---------------------------------------------------------------------------
// Test 9: Consecutive runs produce unique run_ids and trace_ids
// ---------------------------------------------------------------------------

#[test]
fn e2e_pipeline_consecutive_runs_have_unique_ids() {
    if !is_subprocess() {
        if !pipeline_prereqs_available() {
            eprintln!("Skipping: mock backends or ffmpeg not available");
            return;
        }
        let state_tmp = tempfile::tempdir().expect("create state temp dir");
        run_in_subprocess(
            "e2e_pipeline_consecutive_runs_have_unique_ids",
            &build_test_env(state_tmp.path()),
        );
        return;
    }

    let state_dir = state_dir_from_env();
    let wav_tmp = tempfile::tempdir().expect("create wav temp dir");
    let wav_path = helpers::generate_test_wav(wav_tmp.path(), "input.wav", 0.5, 440.0);

    let engine = FrankenWhisperEngine::new().expect("engine creation should succeed");

    let db_path_a = state_dir.join("e2e_unique_a.sqlite3");
    let request_a = build_e2e_request(&wav_path, BackendKind::WhisperCpp, &db_path_a, false, None);
    let report_a = engine
        .transcribe(request_a)
        .expect("first transcribe should succeed");

    let db_path_b = state_dir.join("e2e_unique_b.sqlite3");
    let request_b = build_e2e_request(&wav_path, BackendKind::WhisperCpp, &db_path_b, false, None);
    let report_b = engine
        .transcribe(request_b)
        .expect("second transcribe should succeed");

    assert_ne!(
        report_a.run_id, report_b.run_id,
        "consecutive runs must produce different run_ids"
    );
    assert_ne!(
        report_a.trace_id, report_b.trace_id,
        "consecutive runs must produce different trace_ids"
    );
}

// ---------------------------------------------------------------------------
// Test 10: Silent WAV input produces valid output
// ---------------------------------------------------------------------------

#[test]
fn e2e_pipeline_with_silence_wav() {
    if !is_subprocess() {
        if !pipeline_prereqs_available() {
            eprintln!("Skipping: mock backends or ffmpeg not available");
            return;
        }
        let state_tmp = tempfile::tempdir().expect("create state temp dir");
        run_in_subprocess(
            "e2e_pipeline_with_silence_wav",
            &build_test_env(state_tmp.path()),
        );
        return;
    }

    let state_dir = state_dir_from_env();
    let wav_tmp = tempfile::tempdir().expect("create wav temp dir");
    // Generate a silent WAV (0 Hz frequency produces all-zero samples)
    let wav_path = helpers::generate_silence_wav(wav_tmp.path(), "silence.wav", 1.0);
    let db_path = state_dir.join("e2e_silence.sqlite3");

    let engine = FrankenWhisperEngine::new().expect("engine creation should succeed");
    let request = build_e2e_request(&wav_path, BackendKind::WhisperCpp, &db_path, false, None);
    let report = engine
        .transcribe(request)
        .expect("transcribe of silence should succeed");

    assert!(!report.run_id.is_empty(), "run_id must be non-empty");
    assert!(!report.trace_id.is_empty(), "trace_id must be non-empty");
    assert!(
        !report.events.is_empty(),
        "events list should not be empty even for silent input"
    );
    // Silence-only input may short-circuit at VAD, producing an empty
    // transcript without invoking the backend stage.
    let vad_silence = report
        .events
        .iter()
        .any(|event| event.code == "vad.silence");
    if vad_silence {
        assert!(
            report.result.transcript.is_empty(),
            "vad.silence short-circuit should produce empty transcript"
        );
    } else {
        assert!(
            !report.result.transcript.is_empty(),
            "non-silence path should produce transcript output"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 11: Pipeline without persist does not create the database file
// ---------------------------------------------------------------------------

#[test]
fn e2e_pipeline_no_persist_skips_db_creation() {
    if !is_subprocess() {
        if !pipeline_prereqs_available() {
            eprintln!("Skipping: mock backends or ffmpeg not available");
            return;
        }
        let state_tmp = tempfile::tempdir().expect("create state temp dir");
        run_in_subprocess(
            "e2e_pipeline_no_persist_skips_db_creation",
            &build_test_env(state_tmp.path()),
        );
        return;
    }

    let state_dir = state_dir_from_env();
    let wav_tmp = tempfile::tempdir().expect("create wav temp dir");
    let wav_path = helpers::generate_test_wav(wav_tmp.path(), "input.wav", 0.5, 440.0);
    let db_path = state_dir.join("e2e_nopersist.sqlite3");

    assert!(
        !db_path.exists(),
        "test precondition: db file should not exist before pipeline run"
    );

    let engine = FrankenWhisperEngine::new().expect("engine creation should succeed");
    let request = build_e2e_request(&wav_path, BackendKind::WhisperCpp, &db_path, false, None);
    let report = engine
        .transcribe(request)
        .expect("transcribe without persist should succeed");

    assert!(
        !report.run_id.is_empty(),
        "run_id must be non-empty even without persist"
    );
    assert!(
        !db_path.exists(),
        "SQLite database should NOT be created when persist=false, but found: {}",
        db_path.display()
    );

    // Verify no persist-stage event was emitted
    let has_persist_event = report
        .events
        .iter()
        .any(|e| e.stage == "persist" && e.code.contains("persist"));
    assert!(
        !has_persist_event,
        "events should NOT include a persist-stage event when persist=false"
    );
}
