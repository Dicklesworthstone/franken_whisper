use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Stdio};

use serde_json::json;
use tempfile::tempdir;

use franken_whisper::backend;
use franken_whisper::cli::{
    Cli, Command, RunsOutputFormat, TtyAudioCommand, TtyAudioControlCommand, TtyAudioRecoveryPolicy,
};
use franken_whisper::confidential_evaluation::{
    CONFIDENTIAL_EVALUATION_AGGREGATE_SCHEMA_VERSION,
    CONFIDENTIAL_EVALUATION_MANIFEST_SCHEMA_VERSION,
};
use franken_whisper::diarization::{
    DIARIZATION_HYPOTHESIS_SCHEMA_VERSION, DIARIZATION_REFERENCE_SCHEMA_VERSION,
    DiarizationHypothesisDocument, DiarizationReferenceDocument, DiarizationScorerConfig,
    EvaluationTurn,
};
use franken_whisper::model::{
    BackendKind, BackendParams, DecodingParams, DiarizationConfig, DiarizationEngine,
    DiarizationFallbackPolicy, DiarizationRequest, InputSource, OutputFormat, RunEvent, RunReport,
    SpeakerCountRequest, TimestampLevel, TranscribeRequest, TranscriptionResult,
    TranscriptionSegment, VadParams,
};
use franken_whisper::public_corpus::{
    PUBLIC_CORPUS_BUNDLE_SCHEMA_VERSION, PUBLIC_CORPUS_INPUT_SCHEMA_VERSION,
    PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION, PUBLIC_CORPUS_SIDECAR_STUDY_SCHEMA_VERSION,
    PublicCorpusEvaluationStage, PublicCorpusSidecarDisposition, PublicCorpusSidecarLane,
    parse_public_corpus_sidecar_study_evidence,
};
use franken_whisper::storage::RunStore;
use franken_whisper::sync::{self, ConflictPolicy};

fn private_tempdir_in(root: &std::path::Path, prefix: &str) -> tempfile::TempDir {
    let directory = tempfile::Builder::new()
        .prefix(prefix)
        .tempdir_in(root)
        .unwrap_or_else(|error| panic!("private temporary directory {prefix}: {error}"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700))
            .unwrap_or_else(|error| panic!("private temporary directory {prefix}: {error}"));
    }
    directory
}

fn private_external_temp_root(prefix: &str) -> tempfile::TempDir {
    let project = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .canonicalize()
        .expect("canonical project root");
    let platform_temp = std::env::temp_dir();
    if let Ok(platform_temp) = platform_temp.canonicalize()
        && !platform_temp.starts_with(&project)
        && !project.starts_with(&platform_temp)
        && let Ok(directory) = tempfile::Builder::new()
            .prefix(prefix)
            .tempdir_in(&platform_temp)
    {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;

            std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap_or_else(|error| panic!("private external root {prefix}: {error}"));
        }
        return directory;
    }
    let project_parent = project.parent().expect("project parent");
    private_tempdir_in(project_parent, prefix)
}

fn run_agent_command(binary: &str, args: &[&str]) -> std::process::Output {
    ProcessCommand::new(binary)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("agent command {args:?} failed to execute: {error}"))
}

#[cfg(unix)]
fn wait_for_pid_witness(
    path: &std::path::Path,
    deadline: std::time::Instant,
    parent: &mut std::process::Child,
) -> rustix::process::Pid {
    loop {
        if let Ok(text) = std::fs::read_to_string(path)
            && let Ok(raw) = text.parse::<i32>()
            && let Some(pid) = rustix::process::Pid::from_raw(raw)
        {
            return pid;
        }
        match parent.try_wait() {
            Ok(Some(status)) => panic!(
                "PID-witness fixture exited before publishing {}: {status}",
                path.display()
            ),
            Ok(None) => {}
            Err(error) => {
                let _ = parent.kill();
                let _ = parent.wait();
                panic!(
                    "could not inspect PID-witness fixture while waiting for {}: {error}",
                    path.display()
                );
            }
        }
        if std::time::Instant::now() >= deadline {
            let _ = parent.kill();
            let _ = parent.wait();
            panic!(
                "PID witness was not published before the deadline: {}",
                path.display()
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[cfg(unix)]
#[test]
fn comparison_cancel_probe_rejects_an_unleased_process_group_root() {
    use std::os::unix::process::CommandExt as _;

    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .arg("__comparison-cancel-probe")
        .process_group(0)
        .env_remove("FRANKEN_WHISPER_PARENT_LIVENESS_FD")
        .env_remove("FRANKEN_WHISPER_PARENT_LIVENESS_PID")
        .output()
        .expect("run unleased cancellation-probe root");
    assert!(
        !output.status.success(),
        "a fresh process-group root must not authenticate without the inherited liveness lease"
    );
}

#[cfg(unix)]
#[test]
fn abrupt_comparison_parent_death_terminates_the_complete_worker_group() {
    let directory = private_external_temp_root("fw-abrupt-parent-");
    let root_pid_path = directory.path().join("root.pid");
    let descendant_pid_path = directory.path().join("descendant.pid");
    let mut parent = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args([
            "__comparison-cancel-probe",
            "--lease-parent",
            "--root-pid-file",
            root_pid_path.to_str().expect("UTF-8 root PID path"),
            "--descendant-pid-file",
            descendant_pid_path
                .to_str()
                .expect("UTF-8 descendant PID path"),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn abrupt-parent fixture");

    // Debug builds on macOS may spend several seconds initializing each of the
    // three nested CLI processes. The complete integration suite also launches
    // many agent-discovery subprocesses concurrently, so setup gets a generous
    // scheduling window; the post-kill cleanup contract remains two seconds.
    let root_deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let root_pid = wait_for_pid_witness(&root_pid_path, root_deadline, &mut parent);
    let descendant_deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let descendant_pid =
        wait_for_pid_witness(&descendant_pid_path, descendant_deadline, &mut parent);
    assert_eq!(
        rustix::process::getpgid(Some(root_pid)).expect("worker process group"),
        root_pid
    );
    assert_eq!(
        rustix::process::getpgid(Some(descendant_pid)).expect("descendant process group"),
        root_pid
    );
    assert!(
        parent.try_wait().expect("inspect fixture parent").is_none(),
        "fixture parent exited before abrupt termination"
    );

    parent.kill().expect("abruptly terminate fixture parent");
    parent.wait().expect("reap fixture parent");

    let cleanup_deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        match rustix::process::test_kill_process_group(root_pid) {
            Err(error) if error == rustix::io::Errno::SRCH => break,
            _ if std::time::Instant::now() < cleanup_deadline => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            result => {
                let _ =
                    rustix::process::kill_process_group(root_pid, rustix::process::Signal::KILL);
                panic!("worker group survived abrupt parent death: {result:?}");
            }
        }
    }
}

#[test]
fn packaged_binary_names_share_version_help_and_schema_contracts() {
    let long = env!("CARGO_BIN_EXE_franken_whisper");
    let short = env!("CARGO_BIN_EXE_fw");

    for args in [
        &["--version"][..],
        &["--help"][..],
        &["robot", "schema"][..],
    ] {
        let long_output = run_agent_command(long, args);
        let short_output = run_agent_command(short, args);
        assert!(
            long_output.status.success(),
            "long binary failed for {args:?}"
        );
        assert!(
            short_output.status.success(),
            "short binary failed for {args:?}"
        );
        if args == ["--help"] {
            let long_help = std::str::from_utf8(&long_output.stdout)
                .expect("long-form help must be valid UTF-8");
            let short_help = std::str::from_utf8(&short_output.stdout)
                .expect("short-form help must be valid UTF-8");
            assert!(long_help.contains("Usage: franken_whisper <COMMAND>"));
            assert!(short_help.contains("Usage: fw <COMMAND>"));
            assert_eq!(
                long_help.replace(
                    "Usage: franken_whisper <COMMAND>",
                    "Usage: <BINARY> <COMMAND>"
                ),
                short_help.replace("Usage: fw <COMMAND>", "Usage: <BINARY> <COMMAND>"),
                "help contract drift beyond the intentionally invoked binary name"
            );
        } else {
            assert_eq!(
                long_output.stdout, short_output.stdout,
                "stdout drift for {args:?}"
            );
        }
        assert_eq!(
            long_output.stderr, short_output.stderr,
            "stderr drift for {args:?}"
        );
        if args == ["--version"] {
            let version =
                String::from_utf8(long_output.stdout).expect("version report must be valid UTF-8");
            assert_eq!(
                version.lines().next(),
                Some(concat!("franken_whisper ", env!("CARGO_PKG_VERSION"))),
                "Clap must add the command name exactly once"
            );
        }
    }
}

#[test]
fn robot_help_remains_human_help_for_both_binary_names() {
    for binary in [
        env!("CARGO_BIN_EXE_franken_whisper"),
        env!("CARGO_BIN_EXE_fw"),
    ] {
        let output = run_agent_command(binary, &["robot", "--help"]);
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        let stdout = String::from_utf8(output.stdout).expect("help must be UTF-8");
        assert!(stdout.contains("Usage:"));
        assert!(stdout.contains("triage"));
        assert!(!stdout.contains("\"event\":\"run_error\""));
    }
}

#[test]
fn agent_discovery_json_commands_are_single_line_and_path_safe() {
    let binary = env!("CARGO_BIN_EXE_fw");
    for args in [
        &["capabilities", "--json"][..],
        &["models", "--json"][..],
        &["doctor", "--json"][..],
        &["robot", "triage"][..],
        &["robot", "schema"][..],
    ] {
        let output = run_agent_command(binary, args);
        assert!(
            output.status.success(),
            "agent discovery command {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty(), "stderr pollution for {args:?}");
        let stdout = String::from_utf8(output.stdout).expect("UTF-8 agent JSON");
        assert_eq!(
            stdout.lines().count(),
            1,
            "expected one JSON line for {args:?}"
        );
        let value: serde_json::Value =
            serde_json::from_str(stdout.trim()).expect("valid agent discovery JSON");
        assert!(value.is_object(), "object response for {args:?}");
        if args == ["models", "--json"] {
            assert_eq!(value["local_paths_emitted"], false);
            let encoded = serde_json::to_string(&value).expect("re-encode model registry");
            assert!(!encoded.contains("/Users/"));
            assert!(!encoded.contains("/home/"));
            assert!(encoded.contains("github_release_with_license_and_notice"));
        }
    }
}

#[test]
fn release_model_sentinels_ignore_same_named_cwd_files() {
    let runtime = tempdir().expect("isolated runtime directory");
    let cache = tempdir().expect("isolated model cache");
    let home = tempdir().expect("isolated home");

    let mut valid_header = Vec::new();
    valid_header.extend_from_slice(&0x6767_6d6cu32.to_le_bytes());
    for value in [51865i32, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 1] {
        valid_header.extend_from_slice(&value.to_le_bytes());
    }
    for name in ["default", "DEFAULT", "large-v3-turbo"] {
        std::fs::write(runtime.path().join(name), &valid_header).expect("write CWD collision");
    }

    let report_for = |spec: &str| -> serde_json::Value {
        let output = ProcessCommand::new(env!("CARGO_BIN_EXE_fw"))
            .args(["models", "--json"])
            .current_dir(runtime.path())
            .env("HOME", home.path())
            .env("FRANKEN_WHISPER_MODEL_DIR", cache.path())
            .env("FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL", spec)
            .env_remove("FRANKEN_WHISPER_TEST_MODEL_DIR")
            .output()
            .expect("run model report");
        assert!(
            output.status.success(),
            "model report failed for {spec:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).expect("model report JSON")
    };

    for sentinel in ["default", "DEFAULT", "large-v3-turbo"] {
        let report = report_for(sentinel);
        let whisper = report["models"]
            .as_array()
            .expect("models array")
            .iter()
            .find(|entry| entry["id"] == "native-whisper-ggml")
            .expect("Whisper model entry");
        assert_eq!(
            whisper["installed"], false,
            "bare sentinel {sentinel:?} must not resolve through CWD"
        );
        assert_eq!(whisper["default_release_package_installed"], false);
    }

    let missing_binary = runtime.path().join("missing-backend-binary");
    for sentinel in ["default", "large-v3-turbo"] {
        let output = ProcessCommand::new(env!("CARGO_BIN_EXE_fw"))
            .args(["robot", "backends"])
            .current_dir(runtime.path())
            .env("HOME", home.path())
            .env("FRANKEN_WHISPER_MODEL_DIR", cache.path())
            .env("FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL", sentinel)
            .env("FRANKEN_WHISPER_WHISPER_CPP_BIN", &missing_binary)
            .env("FRANKEN_WHISPER_INSANELY_FAST_BIN", &missing_binary)
            .env("FRANKEN_WHISPER_PYTHON_BIN", &missing_binary)
            .env_remove("FRANKEN_WHISPER_TEST_MODEL_DIR")
            .output()
            .expect("run backend discovery");
        assert!(output.status.success(), "backend discovery must succeed");
        let report: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("backend discovery JSON");
        let backends = report["backends"].as_array().expect("backends array");
        for native_name in [
            "whisper.cpp-native",
            "insanely-fast-native",
            "whisper-diarization-native",
        ] {
            let backend = backends
                .iter()
                .find(|entry| entry["name"] == native_name)
                .expect("native backend entry");
            assert_eq!(
                backend["available"], false,
                "CWD collision must not make {native_name} available for {sentinel}"
            );
        }
    }

    let voiced_wav = runtime.path().join("voiced.wav");
    let wav_spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&voiced_wav, wav_spec).expect("create voiced WAV");
    for index in 0..1_600 {
        writer
            .write_sample(if index % 2 == 0 { 8_000_i16 } else { -8_000_i16 })
            .expect("write voiced sample");
    }
    writer.finalize().expect("finalize voiced WAV");

    for sentinel in ["default", "large-v3-turbo"] {
        let output = ProcessCommand::new(env!("CARGO_BIN_EXE_fw"))
            .args([
                "robot",
                "run",
                "--input",
                voiced_wav.to_str().expect("UTF-8 WAV path"),
                "--backend",
                "whisper-cpp",
                "--model",
                sentinel,
                "--no-persist",
                "--no-diarize",
            ])
            .current_dir(runtime.path())
            .env("HOME", home.path())
            .env("FRANKEN_WHISPER_MODEL_DIR", cache.path())
            .env("FRANKEN_WHISPER_NATIVE_EXECUTION", "1")
            .env("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole")
            .env("FRANKEN_WHISPER_WAV_PASSTHROUGH", "1")
            .env_remove("FRANKEN_WHISPER_TEST_MODEL_DIR")
            .output()
            .expect("run native request");
        assert!(!output.status.success(), "missing release package must fail");
        let run_error = String::from_utf8(output.stdout)
            .expect("UTF-8 robot output")
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .find(|event| event["event"] == "run_error")
            .expect("terminal run_error");
        assert_eq!(run_error["code"], "FW-BACKEND-UNAVAILABLE");
        let message = run_error["message"].as_str().expect("error message");
        assert!(
            message.contains("missing expected artifact"),
            "execution resolver must report authenticated package absence: {message}"
        );
        assert!(
            !message.contains(runtime.path().to_string_lossy().as_ref()),
            "execution must not report the same-named CWD file: {message}"
        );
    }

    let explicit = runtime.path().join("default");
    let report = report_for(explicit.to_str().expect("UTF-8 explicit path"));
    let whisper = report["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|entry| entry["id"] == "native-whisper-ggml")
        .expect("Whisper model entry");
    assert_eq!(
        whisper["installed"], true,
        "an explicit path with a reserved basename remains a custom model"
    );
    assert_eq!(whisper["default_release_package_installed"], false);
}

#[test]
fn listen_pinned_fast_lane_names_reject_unauthenticated_cache_files() {
    let runtime = tempdir().expect("isolated runtime directory");
    let cache = tempdir().expect("isolated model cache");
    let home = tempdir().expect("isolated home");
    let cache_root = cache.path().join("whisper");

    for (artifact_version, filename) in [
        (
            franken_whisper::model_distribution::TINY_EN_ARTIFACT_VERSION,
            franken_whisper::model_distribution::TINY_EN_WEIGHTS_FILENAME,
        ),
        (
            franken_whisper::model_distribution::TINY_ARTIFACT_VERSION,
            franken_whisper::model_distribution::TINY_WEIGHTS_FILENAME,
        ),
    ] {
        let directory = cache_root.join(artifact_version);
        std::fs::create_dir_all(&directory).expect("create fast-lane package directory");
        let mut valid_header = [0_u8; 48];
        valid_header[0..4].copy_from_slice(&0x6767_6d6c_u32.to_le_bytes());
        valid_header[44..48].copy_from_slice(&1_i32.to_le_bytes());
        std::fs::write(directory.join(filename), valid_header)
            .expect("write header-valid but unauthenticated package");
    }

    let assert_rejected = |case: &str, extra_args: &[&str], expected_filename: &str| {
        let mut args = vec![
            "robot",
            "listen",
            "--source",
            "stdin-pcm",
            "--quality-model",
            "none",
            "--no-persist",
        ];
        args.extend_from_slice(extra_args);
        let output = ProcessCommand::new(env!("CARGO_BIN_EXE_fw"))
            .args(&args)
            .current_dir(runtime.path())
            .env("HOME", home.path())
            .env("FRANKEN_WHISPER_MODEL_DIR", cache.path())
            .env_remove("FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL")
            .env_remove("FRANKEN_WHISPER_TEST_MODEL_DIR")
            .output()
            .unwrap_or_else(|error| panic!("run listen case {case}: {error}"));

        assert!(!output.status.success(), "{case} must fail closed");
        let events: Vec<serde_json::Value> = String::from_utf8(output.stdout)
            .expect("UTF-8 robot output")
            .lines()
            .map(|line| serde_json::from_str(line).expect("NDJSON event"))
            .collect();
        assert!(
            events.iter().all(|event| event["event"] != "session_start"),
            "{case} emitted readiness before authenticating the fast lane"
        );
        let error = events
            .iter()
            .find(|event| event["event"] == "run_error")
            .unwrap_or_else(|| panic!("{case} missing terminal run_error: {events:?}"));
        assert_eq!(error["code"], "FW-MISSING-ARTIFACT", "{case}");
        assert!(
            error["message"]
                .as_str()
                .is_some_and(|message| message.contains(expected_filename)),
            "{case} did not identify the unauthenticated package: {error}"
        );
    };

    assert_rejected(
        "default multilingual tiny",
        &[],
        franken_whisper::model_distribution::TINY_WEIGHTS_FILENAME,
    );
    assert_rejected(
        "language-selected tiny.en",
        &["--language", "en"],
        franken_whisper::model_distribution::TINY_EN_WEIGHTS_FILENAME,
    );
    assert_rejected(
        "explicit bare tiny",
        &["--fast-model", "tiny"],
        franken_whisper::model_distribution::TINY_WEIGHTS_FILENAME,
    );
    assert_rejected(
        "explicit bare tiny.en",
        &["--fast-model", "tiny.en"],
        franken_whisper::model_distribution::TINY_EN_WEIGHTS_FILENAME,
    );
}

#[test]
fn robot_syntax_errors_are_one_json_line_with_usage_exit_code() {
    let output = run_agent_command(
        env!("CARGO_BIN_EXE_fw"),
        &["robot", "run", "--inpt", "private-call.m4a"],
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 robot syntax error");
    assert_eq!(stdout.lines().count(), 1);
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("JSON syntax error");
    assert_eq!(value["event"], "run_error");
    assert_eq!(value["code"], "FW-INVALID-REQUEST");
    assert!(!stdout.contains("private-call.m4a"));
}

#[test]
fn sortformer_runtime_errors_are_one_path_safe_json_line() {
    let cache = tempfile::tempdir().expect("isolated model cache");
    let private_input = "/private/confidential/customer-call.m4a";
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fw"))
        .args(["sortformer-diarize", "--input", private_input])
        .env("FRANKEN_WHISPER_MODEL_DIR", cache.path())
        .output()
        .expect("run Sortformer command");
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 Sortformer error");
    assert_eq!(stdout.lines().count(), 1);
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("JSON runtime error");
    assert_eq!(value["schema_version"], "sortformer-diarization-v1");
    assert_eq!(
        value["certification"],
        franken_whisper::sortformer_conformance::SORTFORMER_CERTIFICATION_STATUS
    );
    assert_eq!(value["status"], "error");
    assert_eq!(value["local_paths_emitted"], false);
    assert!(!stdout.contains(private_input));
    assert!(!stdout.contains(cache.path().to_string_lossy().as_ref()));
}

#[test]
fn sortformer_syntax_errors_are_one_path_safe_json_line() {
    let private_input = "/private/confidential/customer-call.m4a";
    let output = run_agent_command(
        env!("CARGO_BIN_EXE_fw"),
        &["sortformer-diarize", "--inpt", private_input],
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 Sortformer syntax error");
    assert_eq!(stdout.lines().count(), 1);
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("JSON syntax error");
    assert_eq!(value["schema_version"], "sortformer-diarization-v1");
    assert_eq!(
        value["certification"],
        franken_whisper::sortformer_conformance::SORTFORMER_CERTIFICATION_STATUS
    );
    assert_eq!(value["code"], "FW-INVALID-REQUEST");
    assert_eq!(value["local_paths_emitted"], false);
    assert!(!stdout.contains(private_input));
}

#[test]
fn json_model_pull_syntax_errors_are_one_path_safe_json_line() {
    let output = run_agent_command(
        env!("CARGO_BIN_EXE_fw"),
        &["pull", "sortformer", "--json", "--unknown-private-flag"],
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 pull syntax error");
    assert_eq!(stdout.lines().count(), 1);
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("JSON syntax error");
    assert_eq!(value["schema_version"], "franken-whisper-model-pull-v2");
    assert!(value["model"].is_null());
    assert_eq!(value["code"], "FW-INVALID-REQUEST");
    assert_eq!(value["local_paths_emitted"], false);
    assert!(!stdout.contains("unknown-private-flag"));
}

#[test]
fn json_model_pull_runtime_errors_are_one_path_safe_json_line() {
    let private_relative_cache = "private/customer/cache";
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fw"))
        .args(["pull", "sortformer", "--json"])
        .env("FRANKEN_WHISPER_MODEL_DIR", private_relative_cache)
        .output()
        .expect("run model-pull command");
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 pull runtime error");
    assert_eq!(stdout.lines().count(), 1);
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).expect("JSON runtime error");
    assert_eq!(value["schema_version"], "franken-whisper-model-pull-v2");
    assert_eq!(value["status"], "error");
    assert_eq!(value["local_paths_emitted"], false);
    assert!(!stdout.contains(private_relative_cache));
}

// ---------------------------------------------------------------------------
// Storage: runs get --id
// ---------------------------------------------------------------------------

fn fixture_report(id: &str, db_path: &std::path::Path) -> RunReport {
    RunReport {
        run_id: id.to_owned(),
        started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
        finished_at_rfc3339: "2026-02-22T00:00:05Z".to_owned(),
        input_path: "test.wav".to_owned(),
        normalized_wav_path: "normalized.wav".to_owned(),
        request: TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("test.wav"),
            },
            backend: BackendKind::Auto,
            model: None,
            language: Some("en".to_owned()),
            translate: false,
            diarize: false,
            persist: true,
            db_path: db_path.to_path_buf(),
            timeout_ms: None,
            backend_params: franken_whisper::model::BackendParams::default(),
        },
        result: TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: "integration test transcript".to_owned(),
            language: Some("en".to_owned()),
            segments: vec![
                TranscriptionSegment {
                    start_sec: Some(0.0),
                    end_sec: Some(2.0),
                    text: "integration test".to_owned(),
                    speaker: Some("SPEAKER_00".to_owned()),
                    confidence: Some(0.92),
                },
                TranscriptionSegment {
                    start_sec: Some(2.0),
                    end_sec: Some(4.0),
                    text: "transcript".to_owned(),
                    speaker: None,
                    confidence: None,
                },
            ],
            acceleration: None,
            diarization: None,
            raw_output: json!({"test": true}),
            artifact_paths: vec![],
        },
        events: vec![
            RunEvent {
                seq: 1,
                ts_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
                stage: "ingest".to_owned(),
                code: "ingest.ok".to_owned(),
                message: "materialized".to_owned(),
                payload: json!({}),
            },
            RunEvent {
                seq: 2,
                ts_rfc3339: "2026-02-22T00:00:03Z".to_owned(),
                stage: "backend".to_owned(),
                code: "backend.ok".to_owned(),
                message: "done".to_owned(),
                payload: json!({"segments": 2}),
            },
        ],
        warnings: vec![],
        evidence: vec![],
        trace_id: "00000000000000000000000000000000".to_owned(),
        replay: franken_whisper::model::ReplayEnvelope::default(),
    }
}

#[cfg(unix)]
fn ffmpeg_available() -> bool {
    ProcessCommand::new("ffmpeg")
        .args(["-hide_banner", "-version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(unix)]
fn write_whisper_cpp_stub_binary(dir: &std::path::Path) -> PathBuf {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let stub_path = dir.join("whisper_cpp_stub.sh");
    let script = r#"#!/bin/bash
set -euo pipefail
out_prefix=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -of)
      out_prefix="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${out_prefix}" ]]; then
  echo "missing -of output prefix" >&2
  exit 2
fi
printf '%s\n' '{"text":"stub transcript","language":"en","segments":[{"start":0.0,"end":0.5,"text":"stub transcript","speaker":"SPEAKER_00","confidence":0.9}]}' > "${out_prefix}.json"
"#;
    fs::write(&stub_path, script).expect("write stub");
    let mut perms = fs::metadata(&stub_path).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&stub_path, perms).expect("chmod");
    stub_path
}

#[cfg(unix)]
fn write_whisper_cpp_zero_duration_word_stub_binary(dir: &std::path::Path) -> PathBuf {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let stub_path = dir.join("whisper_cpp_zero_duration_word_stub.sh");
    let script = r#"#!/bin/bash
set -euo pipefail
out_prefix=""
split_on_word=0
max_segment_length=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -of)
      out_prefix="$2"
      shift 2
      ;;
    -sow)
      split_on_word=1
      shift
      ;;
    -ml)
      max_segment_length="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${out_prefix}" || "${split_on_word}" != "1" || "${max_segment_length}" != "100" ]]; then
  echo "missing split-on-word zero-duration regression contract" >&2
  exit 42
fi
printf '%s\n' '{"result":{"language":"en"},"transcription":[{"offsets":{"from":0,"to":1000},"text":" first","confidence":0.91},{"offsets":{"from":1000,"to":1000},"text":" zero-width","confidence":0.72},{"offsets":{"from":1500,"to":2000},"text":" third","confidence":0.83}]}' > "${out_prefix}.json"
"#;
    fs::write(&stub_path, script).expect("write zero-duration word stub");
    let mut perms = fs::metadata(&stub_path).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&stub_path, perms).expect("chmod");
    stub_path
}

#[cfg(unix)]
fn write_whisper_cpp_tiny_diarize_stub_binary(dir: &std::path::Path) -> PathBuf {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let stub_path = dir.join("whisper_cpp_tiny_diarize_stub.sh");
    let script = r#"#!/bin/bash
set -euo pipefail
out_prefix=""
tiny_diarize=0
json_output=0
max_context=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -of)
      out_prefix="$2"
      shift 2
      ;;
    -tdrz)
      tiny_diarize=1
      shift
      ;;
    --tdrz)
      echo "unsupported TinyDiarize flag spelling" >&2
      exit 41
      ;;
    -oj)
      json_output=1
      shift
      ;;
    -mc)
      max_context="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${out_prefix}" || "${out_prefix##*/}" != "whispercpp_output" || "${tiny_diarize}" != "1" || "${json_output}" != "1" || "${max_context}" != "0" ]]; then
  echo "missing TinyDiarize JSON artifact contract" >&2
  exit 42
fi
printf '%s\n' '{"result":{"language":"en"},"transcription":[{"timestamps":{"from":"00:00:00,000","to":"00:00:00,500"},"offsets":{"from":0,"to":500},"text":" synthetic speaker one","speaker_turn_next":true},{"timestamps":{"from":"00:00:00,500","to":"00:00:01,000"},"offsets":{"from":500,"to":1000},"text":" synthetic speaker two","speaker_turn_next":false}]}' > "${out_prefix}.json"
"#;
    fs::write(&stub_path, script).expect("write TinyDiarize stub");
    let mut perms = fs::metadata(&stub_path).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&stub_path, perms).expect("chmod");
    stub_path
}

#[cfg(unix)]
fn write_whisper_cpp_stub_binary_without_speaker(dir: &std::path::Path) -> PathBuf {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let stub_path = dir.join("whisper_cpp_stub_no_speaker.sh");
    let script = r#"#!/bin/bash
set -euo pipefail
out_prefix=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -of)
      out_prefix="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
if [[ -z "${out_prefix}" ]]; then
  echo "missing -of output prefix" >&2
  exit 2
fi
printf '%s\n' '{"text":"stub transcript","language":"en","segments":[{"start":0.0,"end":0.5,"text":"stub transcript","confidence":0.9}]}' > "${out_prefix}.json"
"#;
    fs::write(&stub_path, script).expect("write stub");
    let mut perms = fs::metadata(&stub_path).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&stub_path, perms).expect("chmod");
    stub_path
}

#[cfg(unix)]
fn generate_voiced_wav(path: &std::path::Path) {
    let status = ProcessCommand::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=0.3",
            "-ac",
            "1",
            "-ar",
            "16000",
        ])
        .arg(path)
        .status()
        .expect("spawn ffmpeg");
    assert!(status.success(), "ffmpeg should synthesize voiced wav");
}

#[cfg(unix)]
fn generate_voiced_wav_without_ffmpeg(path: &std::path::Path, duration_ms: u16) {
    use std::f32::consts::TAU;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 8_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
    let total_samples = 8_000usize * usize::from(duration_ms) / 1_000;
    for idx in 0..total_samples {
        let phase = TAU * 440.0 * (idx as f32 / 8_000.0);
        let sample = (phase.sin() * 0.25 * f32::from(i16::MAX)).round() as i16;
        writer.write_sample(sample).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
}

#[cfg(unix)]
fn write_provisioned_ffmpeg_stub(bin_dir: &std::path::Path) -> PathBuf {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    fs::create_dir_all(bin_dir).expect("create provisioned ffmpeg bin dir");
    let stub_path = bin_dir.join("ffmpeg");
    let script = r#"#!/bin/bash
set -euo pipefail
input=""
output=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-i" && $# -ge 2 ]]; then
    input="$2"
    shift 2
    continue
  fi
  output="$1"
  shift
done
if [[ -z "$output" ]]; then
  echo "stub ffmpeg missing required output arg" >&2
  exit 2
fi
if [[ -n "${FRANKEN_WHISPER_TEST_FFMPEG_MARKER:-}" ]]; then
  printf '%s\n' "$input -> $output" >> "${FRANKEN_WHISPER_TEST_FFMPEG_MARKER}"
fi
if [[ -n "${FRANKEN_WHISPER_TEST_FFMPEG_DELAY_SECONDS:-}" ]]; then
  /bin/sleep "${FRANKEN_WHISPER_TEST_FFMPEG_DELAY_SECONDS}"
fi
if [[ -n "$input" && -f "$input" ]]; then
  /bin/cp "$input" "$output"
else
  : > "$output"
fi
"#;
    fs::write(&stub_path, script).expect("write ffmpeg stub");
    let mut perms = fs::metadata(&stub_path).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&stub_path, perms).expect("chmod");
    stub_path
}

#[cfg(unix)]
fn write_provisioned_ffprobe_stub(bin_dir: &std::path::Path) -> PathBuf {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    fs::create_dir_all(bin_dir).expect("create provisioned ffprobe bin dir");
    let stub_path = bin_dir.join("ffprobe");
    let script = r#"#!/bin/bash
set -euo pipefail
input=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -*)
      shift
      ;;
    *)
      input="$1"
      shift
      ;;
  esac
done
if [[ -n "${FRANKEN_WHISPER_TEST_FFPROBE_MARKER:-}" ]]; then
  printf '%s\n' "$input" >> "${FRANKEN_WHISPER_TEST_FFPROBE_MARKER}"
fi
printf '0.25\n'
"#;
    fs::write(&stub_path, script).expect("write ffprobe stub");
    let mut perms = fs::metadata(&stub_path).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&stub_path, perms).expect("chmod");
    stub_path
}

#[cfg(unix)]
fn run_transcribe_json_with_stub(
    args: &[&str],
    stdin_payload: Option<&[u8]>,
    stub_bin: &std::path::Path,
    state_root: &std::path::Path,
) -> serde_json::Value {
    run_transcribe_json_with_stub_env(args, stdin_payload, stub_bin, state_root, &[])
}

#[cfg(unix)]
fn run_transcribe_json_with_stub_env(
    args: &[&str],
    stdin_payload: Option<&[u8]>,
    stub_bin: &std::path::Path,
    state_root: &std::path::Path,
    extra_env: &[(&str, &str)],
) -> serde_json::Value {
    let mut cmd = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"));
    cmd.arg("transcribe");
    cmd.args(args);
    // This helper owns the external bridge fixture contract. Keep the
    // independent default Sortformer stage out of these focused ingest,
    // normalization, backend, and persistence tests; the actual all-native
    // default is covered by native_engine_e2e.
    cmd.arg("--no-diarize");
    cmd.env("FRANKEN_WHISPER_WHISPER_CPP_BIN", stub_bin);
    // These tests assert the external stub's exact transcript and bridge
    // metadata. A locally cached native model must not silently bypass the
    // fixture and make the result depend on developer-machine state.
    cmd.env("FRANKEN_WHISPER_NATIVE_EXECUTION", "0");
    cmd.env("FRANKEN_WHISPER_BRIDGE_NATIVE_RECOVERY", "0");
    cmd.env("FRANKEN_WHISPER_STATE_DIR", state_root);
    for (key, value) in extra_env {
        cmd.env(key, value);
    }
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let output = if let Some(payload) = stdin_payload {
        use std::io::Write;
        cmd.stdin(Stdio::piped());
        let mut child = cmd.spawn().expect("spawn transcribe");
        let mut stdin = child.stdin.take().expect("stdin pipe");
        stdin.write_all(payload).expect("write payload");
        drop(stdin);
        child.wait_with_output().expect("wait transcribe")
    } else {
        cmd.output().expect("run transcribe")
    };

    assert!(
        output.status.success(),
        "transcribe failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_text = String::from_utf8_lossy(&output.stdout);
    let json_start = stdout_text.find('{');
    assert!(
        json_start.is_some(),
        "json report parse failed: no JSON object in stdout\nstdout:\n{}\nstderr:\n{}",
        stdout_text,
        String::from_utf8_lossy(&output.stderr)
    );
    let json_start = json_start.unwrap_or(0);
    let json_payload = &stdout_text[json_start..];
    let parsed = serde_json::from_str(json_payload);
    assert!(
        parsed.is_ok(),
        "json report parse failed: {}\nstdout:\n{}\nstderr:\n{}",
        parsed
            .as_ref()
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default(),
        stdout_text,
        String::from_utf8_lossy(&output.stderr)
    );
    parsed.unwrap()
}

#[test]
fn run_get_by_id_returns_full_details() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store should open");

    let report = fixture_report("integ-run-1", &db_path);
    store.persist_report(&report).expect("persist");

    let details = store
        .load_run_details("integ-run-1")
        .expect("query should succeed")
        .expect("run should exist");

    assert_eq!(details.run_id, "integ-run-1");
    assert_eq!(details.transcript, "integration test transcript");
    assert_eq!(details.segments.len(), 2);
    assert_eq!(details.events.len(), 2);
    assert_eq!(details.events[0].code, "ingest.ok");
    assert_eq!(details.events[1].code, "backend.ok");
}

#[test]
fn runs_json_exposes_only_privacy_safe_projection_provenance() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store should open");
    let mut report = fixture_report("integ-projection-1", &db_path);
    report.result.raw_output = json!({
        "model_path": "PRIVATE_MODEL_PATH_SENTINEL",
        "projection_timeline": {
            "schema_version": franken_whisper::conformance::DTW_PROJECTION_SCHEMA_VERSION,
            "unit": "seconds",
            "interval_semantics": "half_open",
            "timestamp_epsilon_sec":
                franken_whisper::conformance::CANONICAL_PROJECTION_EPSILON_SEC,
            "minimum_duration_sec":
                franken_whisper::conformance::CANONICAL_PROJECTION_MIN_DURATION_SEC,
            "input_engine_segments": 2,
            "input_timed_segments": 2,
            "canonical_units": 2,
            "output_segments": 2,
            "decoder_word_units": 2,
            "interpolated_fallback_units": 0,
            "segment_geometry_fallback_units": 0,
            "interpolated_fallback_segments": 0,
            "segment_geometry_fallback_segments": 0,
            "clamped_units": 0,
            "expanded_units": 0,
            "fallback_reasons": [],
            "word_aligned_safe": true,
            "supported_provenance": [
                "decoder_word_timestamp",
                "segment_interpolation_fallback",
                "segment_geometry_fallback"
            ]
        }
    });
    store.persist_report(&report).expect("persist");

    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args([
            "runs",
            "--db",
            db_path.to_str().expect("utf8 db"),
            "--id",
            "integ-projection-1",
            "--format",
            "json",
        ])
        .output()
        .expect("runs command");
    assert!(
        output.status.success(),
        "runs command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let details: serde_json::Value = serde_json::from_str(&stdout).expect("runs JSON");
    assert_eq!(
        details["projection_timeline"]["schema_version"],
        franken_whisper::conformance::DTW_PROJECTION_SCHEMA_VERSION
    );
    assert!(details.get("raw_output").is_none());
    assert!(
        !stdout.contains("PRIVATE_MODEL_PATH_SENTINEL"),
        "run-history output must not expose unrelated backend raw metadata"
    );
}

#[test]
fn confidential_diarization_eval_cli_emits_only_external_aggregates() {
    // RCH may place the process temporary directory beneath the synchronized
    // checkout. Prefer the platform temp root when it is disjoint; otherwise
    // create one private sibling of the checkout without assuming `/tmp`.
    let external_root = private_external_temp_root("fw-confidential-root-");
    let external_temp_root = external_root.path();
    let input = tempfile::Builder::new()
        .prefix("fw-confidential-input-")
        .tempdir_in(external_temp_root)
        .expect("external input");
    let output = tempfile::Builder::new()
        .prefix("fw-confidential-output-")
        .tempdir_in(external_temp_root)
        .expect("external output");
    let runtime_project =
        private_tempdir_in(external_temp_root, "fw-confidential-evaluation-project-");
    std::fs::create_dir(runtime_project.path().join(".git")).expect("project marker");
    std::fs::write(
        runtime_project.path().join("Cargo.toml"),
        "[package]\nname = \"confidential-evaluation-e2e-root\"\nversion = \"0.0.0\"\n",
    )
    .expect("project manifest");
    let audio_path = input.path().join("PRIVATE_AUDIO_CLI_SENTINEL.m4a");
    let reference_path = input.path().join("PRIVATE_REFERENCE_CLI_SENTINEL.json");
    let hypothesis_path = input.path().join("PRIVATE_HYPOTHESIS_CLI_SENTINEL.json");
    let manifest_path = input.path().join("PRIVATE_MANIFEST_CLI_SENTINEL.json");
    let aggregate_path = output.path().join("aggregate.json");
    std::fs::write(&audio_path, b"synthetic CLI audio bytes").expect("audio");
    let reference = DiarizationReferenceDocument {
        schema_version: DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
        recording_id: "private-cli-recording".to_owned(),
        duration_ms: 1_000,
        turns: vec![EvaluationTurn::labeled(0, 1_000, "private-speaker")],
        ignored_regions: vec![],
        speaker_hints: vec![],
        words: vec![],
    };
    let hypothesis = DiarizationHypothesisDocument {
        schema_version: DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
        recording_id: "private-cli-recording".to_owned(),
        duration_ms: 1_000,
        turns: vec![EvaluationTurn {
            speaker_confidence: Some(0.75),
            ..EvaluationTurn::labeled(0, 1_000, "private-cluster")
        }],
        speaker_count_estimate: None,
        performance: None,
    };
    std::fs::write(
        &reference_path,
        serde_json::to_vec(&reference).expect("reference JSON"),
    )
    .expect("reference");
    std::fs::write(
        &hypothesis_path,
        serde_json::to_vec(&hypothesis).expect("hypothesis JSON"),
    )
    .expect("hypothesis");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&json!({
            "schema_version": CONFIDENTIAL_EVALUATION_MANIFEST_SCHEMA_VERSION,
            "dataset_id": "private-cli-dataset",
            "config": DiarizationScorerConfig::default(),
            "recordings": [{
                "recording_id": "private-cli-recording",
                "audio_path": audio_path,
                "reference_path": reference_path,
                "hypothesis_path": hypothesis_path
            }]
        }))
        .expect("manifest JSON"),
    )
    .expect("manifest");

    let result = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .current_dir(runtime_project.path())
        .args([
            "diarization-eval",
            "--input-root",
            input.path().to_str().expect("UTF-8 input root"),
            "--manifest",
            manifest_path.to_str().expect("UTF-8 manifest"),
            "--output",
            aggregate_path.to_str().expect("UTF-8 aggregate"),
        ])
        .output()
        .expect("confidential evaluator");
    assert!(
        result.status.success(),
        "confidential evaluator failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).expect("UTF-8 stdout");
    let retained = std::fs::read_to_string(aggregate_path).expect("aggregate output");
    let aggregate: serde_json::Value = serde_json::from_str(&stdout).expect("aggregate JSON");
    assert_eq!(
        aggregate["schema_version"],
        CONFIDENTIAL_EVALUATION_AGGREGATE_SCHEMA_VERSION
    );
    assert_eq!(aggregate["recording_count"], 1);
    assert_eq!(
        aggregate["speaker_count_posterior"]["observed_recordings"],
        0
    );
    assert_eq!(
        aggregate["speaker_count_posterior"]["unavailable_recordings"],
        1
    );
    assert_eq!(
        aggregate["speaker_count_posterior"]["unresolved_recordings"],
        1
    );
    assert_eq!(aggregate["word_attribution"]["reference_word_count"], 0);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&retained).expect("retained aggregate"),
        aggregate
    );
    for forbidden in [
        "PRIVATE_",
        "private-cli-recording",
        "private-cli-dataset",
        "private-speaker",
        "private-cluster",
        "audio_path",
        "reference_path",
        "hypothesis_path",
        "transcript",
    ] {
        assert!(!stdout.contains(forbidden));
        assert!(!retained.contains(forbidden));
    }
}

#[cfg_attr(
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
    ignore = "public artifact publication is unsupported on this platform"
)]
#[test]
fn public_diarization_corpus_cli_builds_external_path_free_bundle() {
    use sha2::{Digest, Sha256};

    let external_root = private_external_temp_root("fw-public-corpus-root-");
    let external_temp_root = external_root.path();
    let input = private_tempdir_in(external_temp_root, "fw-public-corpus-input-");
    let output = private_tempdir_in(external_temp_root, "fw-public-corpus-output-");
    let runtime_project = private_tempdir_in(external_temp_root, "fw-public-corpus-project-");
    std::fs::create_dir(runtime_project.path().join(".git")).expect("project marker");
    std::fs::write(
        runtime_project.path().join("Cargo.toml"),
        "[package]\nname = \"public-corpus-e2e-root\"\nversion = \"0.0.0\"\n",
    )
    .expect("project manifest");
    let audio_path = input.path().join("fixture.wav");
    let annotation_path = input.path().join("fixture.rttm");
    let descriptor_path = input.path().join("descriptor.json");
    let bundle_path = output.path().join("bundle.json");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 8_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&audio_path, spec).expect("WAV");
    for _ in 0..16_000 {
        writer.write_sample(0_i16).expect("sample");
    }
    writer.finalize().expect("finalize WAV");
    std::fs::write(
        &annotation_path,
        concat!(
            "SPEAKER public-call 1 0.000 0.600 <NA> <NA> local-a <NA> <NA>\n",
            "SPEAKER public-call 1 0.400 0.500 <NA> <NA> local-b <NA> <NA>\n",
        ),
    )
    .expect("RTTM");
    let audio_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&audio_path).expect("WAV bytes"))
    );
    let annotation_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&annotation_path).expect("RTTM bytes"))
    );
    std::fs::write(
        &descriptor_path,
        serde_json::to_vec_pretty(&json!({
            "schema_version": PUBLIC_CORPUS_INPUT_SCHEMA_VERSION,
            "corpus_key": "aishell-4-openslr111-v1",
            "source_version": "cli-fixture-v1",
            "recordings": [{
                "recording_id": "public-cli-recording",
                "split": "development",
                "origin_recording_id": "public-cli-origin",
                "audio_path": "fixture.wav",
                "audio_sha256": audio_sha256,
                "expected_sample_rate_hz": 8_000,
                "expected_channel_count": 2,
                "selected_channel": 1,
                "annotation_path": "fixture.rttm",
                "annotation_sha256": annotation_sha256,
                "annotation_recording_id": "public-call",
                "annotation_channel": "1",
                "speaker_map": {
                    "local-a": "public-cli-speaker-a",
                    "local-b": "public-cli-speaker-b"
                },
                "ignored_regions": [{
                    "start_ms": 900,
                    "end_ms": 950,
                    "reason_code": "annotation_uncertain"
                }]
            }]
        }))
        .expect("descriptor JSON"),
    )
    .expect("descriptor");

    let result = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .current_dir(runtime_project.path())
        .args([
            "diarization-corpus",
            "build",
            "--input-root",
            input.path().to_str().expect("UTF-8 input root"),
            "--descriptor",
            descriptor_path.to_str().expect("UTF-8 descriptor"),
            "--output",
            bundle_path.to_str().expect("UTF-8 output"),
            "--license-ack",
            "accept-aishell-4-cc-by-sa-4.0",
        ])
        .output()
        .expect("public corpus adapter");
    assert!(
        result.status.success(),
        "public corpus adapter failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).expect("UTF-8 stdout");
    let retained = std::fs::read_to_string(&bundle_path).expect("bundle output");
    let bundle: serde_json::Value = serde_json::from_str(&stdout).expect("bundle JSON");
    assert_eq!(
        bundle["schema_version"],
        PUBLIC_CORPUS_BUNDLE_SCHEMA_VERSION
    );
    assert_eq!(
        bundle["references"][0]["turns"].as_array().unwrap().len(),
        2
    );
    assert_eq!(bundle["recordings"][0]["overlap_turn_count"], 2);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&retained).expect("retained bundle"),
        bundle
    );
    for forbidden in [
        input.path().to_str().expect("input path"),
        output.path().to_str().expect("output path"),
        "audio_path",
        "annotation_path",
        "fixture.wav",
        "fixture.rttm",
        "local-a",
        "local-b",
        "transcript",
    ] {
        assert!(!stdout.contains(forbidden));
        assert!(!retained.contains(forbidden));
    }
}

#[cfg_attr(
    not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
    ignore = "public artifact publication is unsupported on this platform"
)]
#[test]
fn public_diarization_corpus_cli_runs_external_path_free_sidecar_study() {
    use sha2::{Digest, Sha256};

    let external_root = private_external_temp_root("fw-sidecar-corpus-root-");
    let external_temp_root = external_root.path();
    let input = private_tempdir_in(external_temp_root, "fw-sidecar-corpus-input-");
    let output = private_tempdir_in(external_temp_root, "fw-sidecar-corpus-output-");
    let runtime_project = private_tempdir_in(external_temp_root, "fw-sidecar-corpus-project-");
    let canonical_input_root = input.path().canonicalize().expect("canonical input root");
    let canonical_output_root = output.path().canonicalize().expect("canonical output root");
    let canonical_runtime_project_root = runtime_project
        .path()
        .canonicalize()
        .expect("canonical runtime project root");
    std::fs::create_dir(runtime_project.path().join(".git")).expect("project marker");
    std::fs::write(
        runtime_project.path().join("Cargo.toml"),
        "[package]\nname = \"sidecar-corpus-e2e-root\"\nversion = \"0.0.0\"\n",
    )
    .expect("project manifest");

    let audio_name = "sidecar-private-audio.wav";
    let annotation_name = "sidecar-private-reference.rttm";
    let descriptor_name = "sidecar-private-descriptor.json";
    let bundle_name = "sidecar-private-bundle.json";
    let evidence_name = "sidecar-private-evidence.json";
    let audio_path = input.path().join(audio_name);
    let annotation_path = input.path().join(annotation_name);
    let descriptor_path = input.path().join(descriptor_name);
    let bundle_path = output.path().join(bundle_name);
    let evidence_path = output.path().join(evidence_name);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&audio_path, spec).expect("WAV");
    for sample_index in 0..20_000 {
        let seconds = sample_index as f32 / 16_000.0;
        let normalized = if sample_index < 10_000 {
            0.24 * (std::f32::consts::TAU * 140.0 * seconds).sin()
                + 0.06 * (std::f32::consts::TAU * 280.0 * seconds).sin()
        } else {
            0.18 * (std::f32::consts::TAU * 260.0 * seconds).sin()
                + 0.04 * (std::f32::consts::TAU * 520.0 * seconds).sin()
        };
        let sample = (normalized * f32::from(i16::MAX)).round() as i16;
        writer.write_sample(sample).expect("sample");
    }
    writer.finalize().expect("finalize WAV");
    std::fs::write(
        &annotation_path,
        concat!(
            "SPEAKER sidecar-source 1 0.000 0.625 <NA> <NA> sidecar-source-near <NA> <NA>\n",
            "SPEAKER sidecar-source 1 0.625 0.625 <NA> <NA> sidecar-source-far <NA> <NA>\n",
        ),
    )
    .expect("RTTM");
    let audio_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&audio_path).expect("WAV bytes"))
    );
    let annotation_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(&annotation_path).expect("RTTM bytes"))
    );
    std::fs::write(
        &descriptor_path,
        serde_json::to_vec_pretty(&json!({
            "schema_version": PUBLIC_CORPUS_INPUT_SCHEMA_VERSION,
            "corpus_key": "aishell-4-openslr111-v1",
            "source_version": "sidecar-cli-fixture-v1",
            "recordings": [{
                "recording_id": "sidecar-cli-development",
                "split": "development",
                "origin_recording_id": "sidecar-cli-origin",
                "audio_path": audio_name,
                "audio_sha256": audio_sha256,
                "expected_sample_rate_hz": 16_000,
                "expected_channel_count": 1,
                "selected_channel": 1,
                "annotation_path": annotation_name,
                "annotation_sha256": annotation_sha256,
                "annotation_recording_id": "sidecar-source",
                "annotation_channel": "1",
                "speaker_map": {
                    "sidecar-source-near": "sidecar-cli-speaker-near",
                    "sidecar-source-far": "sidecar-cli-speaker-far"
                },
                "ignored_regions": []
            }]
        }))
        .expect("descriptor JSON"),
    )
    .expect("descriptor");

    let invoke = || {
        ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
            .current_dir(runtime_project.path())
            .args([
                "diarization-corpus",
                "sidecar-study",
                "--input-root",
                input.path().to_str().expect("UTF-8 input root"),
                "--descriptor",
                descriptor_path.to_str().expect("UTF-8 descriptor"),
                "--bundle-output",
                bundle_path.to_str().expect("UTF-8 bundle output"),
                "--output",
                evidence_path.to_str().expect("UTF-8 evidence output"),
                "--license-ack",
                "accept-aishell-4-cc-by-sa-4.0",
                "--maximum-recording-duration-ms",
                "1250",
                "--stage",
                "development",
            ])
            .output()
            .expect("public sidecar study")
    };

    let result = invoke();
    assert!(
        result.status.success(),
        "public sidecar study failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let retained_evidence = std::fs::read(&evidence_path).expect("sidecar evidence output");
    assert_eq!(
        result.stdout, retained_evidence,
        "stdout and the retained evidence file must be byte-identical"
    );
    let evidence = parse_public_corpus_sidecar_study_evidence(&retained_evidence)
        .expect("verified sidecar evidence");
    assert_eq!(
        evidence.schema_version,
        PUBLIC_CORPUS_SIDECAR_STUDY_SCHEMA_VERSION
    );
    assert_eq!(
        evidence.schema_version,
        "public-diarization-acoustic-sidecar-study-v6"
    );
    assert_eq!(
        evidence.runner_version,
        PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION
    );
    assert_eq!(
        evidence.runner_version,
        "public-diarization-acoustic-sidecar-study-runner-v8"
    );
    assert_eq!(
        evidence.protocol.pair_calibration_fit_id,
        "public-sidecar-pair-calibration-empirical-grid-v1"
    );
    assert_eq!(
        evidence.protocol.fusion_id,
        "acoustic-sidecar-boundary-fusion-v2"
    );
    assert_eq!(
        evidence.protocol.pair_probability_target_id,
        "public-sidecar-different-speaker-given-selected-comparable-frozen-lag-pair-v1"
    );
    assert_eq!(
        evidence.protocol.pair_population_id,
        "public-sidecar-reference-labeled-frozen-lag-pair-population-v1"
    );
    assert_eq!(
        evidence.protocol.pair_selection_key_id,
        "public-sidecar-conditional-pair-bottom-k-normalized-pcm-sha256-v3"
    );
    assert_eq!(
        evidence.protocol.pair_selection_digest_id,
        "public-sidecar-reference-labeled-selected-pair-sequence-sha256-v2"
    );
    assert_eq!(
        evidence.protocol.pair_scorer_id,
        "public-sidecar-conditional-pair-calibrated-v3"
    );
    assert_eq!(
        evidence.protocol.selection_policy_id,
        "public-sidecar-selection-policy-v3"
    );
    assert_eq!(evidence.protocol.pair_score_bins, 100);
    assert!(
        evidence.variants[1].pair_calibration.is_some(),
        "the two-speaker fixture must retain an independently fitted pair calibration"
    );
    let retained_json: serde_json::Value =
        serde_json::from_slice(&retained_evidence).expect("sidecar wire JSON");
    assert!(!retained_json["variants"][1]["pair_calibration"].is_null());
    assert_eq!(
        evidence.evaluation_stage,
        PublicCorpusEvaluationStage::Development
    );
    assert_eq!(evidence.variants.len(), PublicCorpusSidecarLane::ALL.len());
    let retained_lane_order = evidence
        .variants
        .iter()
        .map(|variant| variant.lane)
        .collect::<Vec<_>>();
    assert_eq!(
        retained_lane_order.as_slice(),
        PublicCorpusSidecarLane::ALL.as_slice()
    );
    assert_eq!(
        evidence.variants[0].disposition,
        PublicCorpusSidecarDisposition::Baseline
    );
    assert!(
        evidence.variants[0].splits[0]
            .coverage
            .pair_selection_sha256
            .is_none(),
        "the unfused baseline must not publish a candidate pair-selection identity"
    );
    assert!(evidence.variants[1..].iter().all(|variant| {
        variant.splits.iter().all(|split| {
            split.coverage.fusion_executed_recording_count
                <= split.coverage.consumed_probability_count
        })
    }));
    assert!(
        evidence.variants[1..]
            .iter()
            .all(|variant| variant.disposition == PublicCorpusSidecarDisposition::Rejected)
    );
    assert!(evidence.selected_candidate_lane.is_none());
    assert!(evidence.adopted_candidate_lane.is_none());
    assert_eq!(evidence.deterministic_accuracy_sha256.len(), 64);
    assert_eq!(evidence.result_sha256.len(), 64);
    assert!(bundle_path.is_file());

    let stdout = String::from_utf8(result.stdout).expect("UTF-8 stdout");
    let retained = String::from_utf8(retained_evidence.clone()).expect("UTF-8 evidence");
    let retained_bundle = std::fs::read_to_string(&bundle_path).expect("UTF-8 bundle");
    for forbidden in [
        input.path().to_str().expect("input path"),
        output.path().to_str().expect("output path"),
        runtime_project.path().to_str().expect("project path"),
        canonical_input_root.to_str().expect("canonical input path"),
        canonical_output_root
            .to_str()
            .expect("canonical output path"),
        canonical_runtime_project_root
            .to_str()
            .expect("canonical project path"),
        audio_name,
        annotation_name,
        descriptor_name,
        bundle_name,
        evidence_name,
        "audio_path",
        "annotation_path",
        "descriptor_path",
    ] {
        assert!(
            !retained_bundle.contains(forbidden),
            "bundle leaked {forbidden}"
        );
    }
    for forbidden in [
        input.path().to_str().expect("input path"),
        output.path().to_str().expect("output path"),
        runtime_project.path().to_str().expect("project path"),
        canonical_input_root.to_str().expect("canonical input path"),
        canonical_output_root
            .to_str()
            .expect("canonical output path"),
        canonical_runtime_project_root
            .to_str()
            .expect("canonical project path"),
        audio_name,
        annotation_name,
        descriptor_name,
        bundle_name,
        evidence_name,
        "sidecar-corpus-e2e-root",
        "sidecar-source",
        "sidecar-source-near",
        "sidecar-source-far",
        "sidecar-cli-development",
        "sidecar-cli-origin",
        "sidecar-cli-speaker-near",
        "sidecar-cli-speaker-far",
        "audio_path",
        "annotation_path",
        "descriptor_path",
        "bundle_output_path",
        "evidence_output_path",
        "locked_development_evidence_path",
        "recording_id",
        "speaker_id",
        "speaker_ref",
        "frame_index",
        "start_ms",
        "end_ms",
        "transcript",
    ] {
        assert!(!stdout.contains(forbidden), "stdout leaked {forbidden}");
        assert!(!retained.contains(forbidden), "file leaked {forbidden}");
    }

    let original_bundle = std::fs::read(&bundle_path).expect("original bundle");
    let second = invoke();
    assert!(!second.status.success(), "create-new replay must fail");
    assert!(
        second.stdout.is_empty(),
        "failed replay must not emit evidence"
    );
    assert_eq!(
        std::fs::read(&bundle_path).expect("unchanged bundle"),
        original_bundle
    );
    assert_eq!(
        std::fs::read(&evidence_path).expect("unchanged evidence"),
        retained_evidence
    );
    let stderr = String::from_utf8(second.stderr).expect("UTF-8 stderr");
    for forbidden in [
        input.path().to_str().expect("input path"),
        output.path().to_str().expect("output path"),
        runtime_project.path().to_str().expect("project path"),
        canonical_input_root.to_str().expect("canonical input path"),
        canonical_output_root
            .to_str()
            .expect("canonical output path"),
        canonical_runtime_project_root
            .to_str()
            .expect("canonical project path"),
        audio_name,
        annotation_name,
        descriptor_name,
        bundle_name,
        evidence_name,
        "sidecar-corpus-e2e-root",
        "sidecar-cli-development",
        "sidecar-cli-speaker-near",
        "sidecar-cli-speaker-far",
    ] {
        assert!(!stderr.contains(forbidden), "stderr leaked {forbidden}");
    }

    let certification_bundle_name = "sidecar-private-certification-bundle.json";
    let certification_evidence_name = "sidecar-private-certification-evidence.json";
    let certification_bundle_path = output.path().join(certification_bundle_name);
    let certification_evidence_path = output.path().join(certification_evidence_name);
    let certification = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .current_dir(runtime_project.path())
        .args([
            "diarization-corpus",
            "sidecar-study",
            "--input-root",
            input.path().to_str().expect("UTF-8 input root"),
            "--descriptor",
            descriptor_path.to_str().expect("UTF-8 descriptor"),
            "--bundle-output",
            certification_bundle_path
                .to_str()
                .expect("UTF-8 certification bundle output"),
            "--output",
            certification_evidence_path
                .to_str()
                .expect("UTF-8 certification evidence output"),
            "--license-ack",
            "accept-aishell-4-cc-by-sa-4.0",
            "--maximum-recording-duration-ms",
            "1250",
            "--stage",
            "certification",
            "--locked-development-evidence",
            evidence_path.to_str().expect("UTF-8 development lock"),
        ])
        .output()
        .expect("rejected-lock certification attempt");
    assert!(
        !certification.status.success(),
        "a rejected development artifact must not unlock certification"
    );
    assert!(
        certification.stdout.is_empty(),
        "failed certification must not emit evidence"
    );
    assert!(!certification_bundle_path.exists());
    assert!(!certification_evidence_path.exists());
    let certification_stderr =
        String::from_utf8(certification.stderr).expect("UTF-8 certification stderr");
    assert!(certification_stderr.contains("sidecar_stage_lock"));
    assert!(!certification_stderr.contains("sidecar_split_missing"));
    for forbidden in [
        input.path().to_str().expect("input path"),
        output.path().to_str().expect("output path"),
        runtime_project.path().to_str().expect("project path"),
        canonical_input_root.to_str().expect("canonical input path"),
        canonical_output_root
            .to_str()
            .expect("canonical output path"),
        canonical_runtime_project_root
            .to_str()
            .expect("canonical project path"),
        audio_name,
        annotation_name,
        descriptor_name,
        bundle_name,
        evidence_name,
        certification_bundle_name,
        certification_evidence_name,
        "sidecar-corpus-e2e-root",
        "sidecar-cli-development",
        "sidecar-cli-speaker-near",
        "sidecar-cli-speaker-far",
    ] {
        assert!(
            !certification_stderr.contains(forbidden),
            "certification stderr leaked {forbidden}"
        );
    }
}

#[test]
fn run_get_nonexistent_returns_none() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store should open");

    let details = store
        .load_run_details("nonexistent")
        .expect("query should succeed");

    assert!(details.is_none());
}

#[test]
fn runs_empty_db_returns_empty_list() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store should open");

    let runs = store.list_recent_runs(10).expect("list should succeed");
    assert!(runs.is_empty());
}

// ---------------------------------------------------------------------------
// Robot backends: diagnostics
// ---------------------------------------------------------------------------

#[test]
fn backend_diagnostics_returns_three_entries() {
    let diags = backend::diagnostics();
    assert_eq!(diags.len(), 3);

    let backend_names: Vec<&str> = diags
        .iter()
        .filter_map(|entry| entry.get("backend").and_then(|v| v.as_str()))
        .collect();

    assert!(backend_names.contains(&"whisper_cpp"));
    assert!(backend_names.contains(&"insanely_fast"));
    assert!(backend_names.contains(&"whisper_diarization"));

    // Each entry has required fields
    for entry in &diags {
        assert!(entry.get("available").is_some());
        assert!(entry.get("backend").is_some());
        assert!(entry.get("unsupported_options").is_some());
        assert!(entry.get("unsupported_options").unwrap().is_array());
    }
}

#[test]
fn backend_diagnostics_is_valid_json() {
    let diags = backend::diagnostics();
    let payload = json!({
        "event": "backends",
        "backends": diags,
    });
    let serialized = serde_json::to_string(&payload).expect("should serialize");
    let _parsed: serde_json::Value = serde_json::from_str(&serialized).expect("should parse back");
}

// ---------------------------------------------------------------------------
// Sync: export empty DB, round-trip, checksum validation
// ---------------------------------------------------------------------------

#[test]
fn sync_export_empty_db_valid_manifest() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("empty.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    let _store = RunStore::open(&db_path).expect("store");
    let manifest = sync::export(&db_path, &export_dir, &state_root).expect("export");

    assert_eq!(manifest.schema_version, "1.1");
    assert_eq!(manifest.row_counts.runs, 0);
    assert_eq!(manifest.row_counts.segments, 0);
    assert_eq!(manifest.row_counts.events, 0);

    // Manifest file should exist and be valid JSON
    let manifest_text =
        std::fs::read_to_string(export_dir.join("manifest.json")).expect("manifest file");
    let parsed: sync::SyncManifest =
        serde_json::from_str(&manifest_text).expect("valid manifest json");
    assert_eq!(parsed.schema_version, "1.1");
}

#[test]
fn sync_round_trip_preserves_data() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("target.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    // Persist two runs
    let store = RunStore::open(&source_db).expect("store");
    store
        .persist_report(&fixture_report("rt-1", &source_db))
        .expect("persist 1");
    store
        .persist_report(&fixture_report("rt-2", &source_db))
        .expect("persist 2");

    // Export
    let manifest = sync::export(&source_db, &export_dir, &state_root).expect("export");
    assert_eq!(manifest.row_counts.runs, 2);
    assert_eq!(manifest.row_counts.segments, 4);
    assert_eq!(manifest.row_counts.events, 4);

    // Import to fresh DB
    let result =
        sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
    assert_eq!(result.runs_imported, 2);
    assert_eq!(result.segments_imported, 4);
    assert_eq!(result.events_imported, 4);
    assert!(result.conflicts.is_empty());

    // Verify target DB
    let target_store = RunStore::open(&target_db).expect("target store");
    let runs = target_store.list_recent_runs(10).expect("list");
    assert_eq!(runs.len(), 2);

    // Verify detailed load
    let details = target_store
        .load_run_details("rt-1")
        .expect("load")
        .expect("exists");
    assert_eq!(details.segments.len(), 2);
    assert_eq!(details.events.len(), 2);
}

#[test]
fn sync_import_rejects_tampered_files() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("source.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    let store = RunStore::open(&db_path).expect("store");
    store
        .persist_report(&fixture_report("tamper-1", &db_path))
        .expect("persist");
    sync::export(&db_path, &export_dir, &state_root).expect("export");

    // Tamper with events.jsonl
    std::fs::write(export_dir.join("events.jsonl"), "tampered\n").expect("tamper");

    let target_db = dir.path().join("target.sqlite3");
    let result = sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("checksum"));
}

#[test]
fn sync_import_missing_manifest_fails() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("target.sqlite3");
    let empty_dir = dir.path().join("empty");
    std::fs::create_dir_all(&empty_dir).expect("mkdir");
    let state_root = dir.path().join("state");

    let result = sync::import(&db_path, &empty_dir, &state_root, ConflictPolicy::Reject);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("manifest.json"));
}

// ---------------------------------------------------------------------------
// Robot mode: NDJSON contract validation
// ---------------------------------------------------------------------------

#[test]
fn robot_report_produces_valid_ndjson() {
    let report = fixture_report("ndjson-1", std::path::Path::new("/tmp/test.db"));
    let mut buf = Vec::new();

    // Simulate emit_robot_report by serializing events + final envelope
    for event in &report.events {
        let line = serde_json::to_string(&json!({
            "event": "stage",
            "run_id": report.run_id,
            "seq": event.seq,
            "ts": event.ts_rfc3339,
            "stage": event.stage,
            "code": event.code,
            "message": event.message,
            "payload": event.payload,
        }))
        .expect("serialize");
        buf.push(line);
    }

    let complete = serde_json::to_string(&json!({
        "event": "run_complete",
        "run_id": report.run_id,
        "backend": report.result.backend,
        "transcript": report.result.transcript,
    }))
    .expect("serialize");
    buf.push(complete);

    // Every line must be valid JSON
    for (index, line) in buf.iter().enumerate() {
        let parsed_result: Result<serde_json::Value, _> = serde_json::from_str(line);
        assert!(
            parsed_result.is_ok(),
            "line {index} invalid JSON: {}",
            parsed_result
                .as_ref()
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default()
        );
        let parsed = parsed_result.unwrap();
        assert!(
            parsed.get("event").is_some(),
            "line {index} missing 'event' field"
        );
    }
}

#[test]
fn robot_routing_history_args_parse_correctly() {
    use clap::Parser;
    let cli = Cli::parse_from([
        "franken_whisper",
        "robot",
        "routing-history",
        "--db",
        "/tmp/test.sqlite3",
        "--run-id",
        "run-abc",
        "--limit",
        "5",
    ]);
    let command = cli.command;
    assert!(
        matches!(
            &command,
            Command::Robot {
                command: franken_whisper::cli::RobotCommand::RoutingHistory(_)
            }
        ),
        "expected Robot RoutingHistory"
    );
    let Command::Robot {
        command: franken_whisper::cli::RobotCommand::RoutingHistory(args),
    } = command
    else {
        return;
    };
    assert_eq!(args.db, PathBuf::from("/tmp/test.sqlite3"));
    assert_eq!(args.run_id.as_deref(), Some("run-abc"));
    assert_eq!(args.limit, 5);
}

#[test]
fn robot_routing_history_extracts_decision_events() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("routing.sqlite3");
    let store = RunStore::open(&db_path).expect("store open");

    // Build a report with a routing decision event in its events.
    let mut report = fixture_report("routing-hist-1", &db_path);
    report.events.push(RunEvent {
        seq: 10,
        ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
        stage: "backend_routing".to_owned(),
        code: "backend.routing.decision_contract".to_owned(),
        message: "evaluated formal backend selection contract".to_owned(),
        payload: json!({
            "mode": "active",
            "chosen_action": "try_whisper_cpp",
            "decision_id": "d-123",
            "calibration_score": 0.85,
            "e_process": 1.18,
            "fallback_active": false,
            "recommended_order": ["whisper_cpp", "insanely_fast", "whisper_diarization"],
        }),
    });

    store.persist_report(&report).expect("persist");

    // Load and verify routing events can be extracted.
    let details = store
        .load_run_details("routing-hist-1")
        .expect("load")
        .expect("run exists");

    let routing_events: Vec<_> = details
        .events
        .iter()
        .filter(|e| e.code == "backend.routing.decision_contract")
        .collect();

    assert_eq!(routing_events.len(), 1);
    assert_eq!(
        routing_events[0].payload["chosen_action"],
        "try_whisper_cpp"
    );
    assert_eq!(routing_events[0].payload["mode"], "active");
    assert_eq!(routing_events[0].payload["calibration_score"], 0.85);
}

#[test]
fn robot_run_invalid_request_emits_run_error() {
    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args(["robot", "run"])
        .output()
        .expect("robot command should execute");

    assert!(
        !output.status.success(),
        "invalid request should return non-zero"
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.is_empty() && trimmed.starts_with('{')
        })
        .map(|line| {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
            assert!(
                parsed.is_ok(),
                "line should be valid json: {}; line={line}",
                parsed
                    .as_ref()
                    .err()
                    .map(|error| error.to_string())
                    .unwrap_or_default()
            );
            parsed.unwrap()
        })
        .collect();

    assert!(!lines.is_empty(), "expected NDJSON output");
    assert_eq!(lines[0]["event"], "run_start");

    let run_error = lines
        .iter()
        .find(|line| line["event"] == "run_error")
        .expect("run_error envelope should be emitted");

    assert_eq!(run_error["code"], "FW-INVALID-REQUEST");
    let message = run_error["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("--input") && message.contains("--stdin") && message.contains("--mic"),
        "expected invalid request message, got: {message}"
    );
}

#[test]
fn robot_run_emits_cancelled_stage_before_terminal_run_error() {
    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let db_path = dir.path().join("storage.sqlite3");

    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args([
            "robot",
            "run",
            "--stdin",
            "--timeout",
            "0",
            "--no-persist",
            "--db",
            db_path.to_str().expect("db path should be utf-8"),
        ])
        .env("FRANKEN_WHISPER_STATE_DIR", &state_root)
        .output()
        .expect("robot command should execute");

    assert!(
        !output.status.success(),
        "timeout/cancellation path should return non-zero"
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    let lines: Vec<&str> = stdout
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.is_empty() && trimmed.starts_with('{')
        })
        .collect();
    assert!(
        lines.len() >= 4,
        "expected at least run_start + 2 stage lines + run_error, got {lines:?}"
    );

    let parsed: Vec<serde_json::Value> = lines
        .iter()
        .map(|line| {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
            assert!(
                parsed.is_ok(),
                "line should be valid json: {}; line={line}",
                parsed
                    .as_ref()
                    .err()
                    .map(|error| error.to_string())
                    .unwrap_or_default()
            );
            parsed.unwrap()
        })
        .collect();

    assert_eq!(parsed[0]["event"], "run_start");

    let budgets_idx = parsed
        .iter()
        .position(|line| line["event"] == "stage" && line["code"] == "orchestration.budgets")
        .expect("orchestration budget line should be present");
    let cancelled_idx = parsed
        .iter()
        .position(|line| line["event"] == "stage" && line["code"] == "orchestration.cancelled")
        .expect("cancellation stage line should be present");
    let error_idx = parsed
        .iter()
        .position(|line| line["event"] == "run_error")
        .expect("terminal run_error should be present");

    assert!(budgets_idx < cancelled_idx);
    assert!(cancelled_idx < error_idx);
    assert_eq!(parsed[error_idx]["code"], "FW-CANCELLED");
    assert_eq!(
        parsed[cancelled_idx]["payload"]["cancellation_evidence"]["reason"],
        "checkpoint deadline exceeded"
    );
}

#[cfg(unix)]
#[test]
fn robot_run_normalize_stage_timeout_maps_to_timeout_error_code() {
    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let db_path = dir.path().join("storage.sqlite3");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let ffmpeg_stub = write_provisioned_ffmpeg_stub(dir.path());
    let input_wav = dir.path().join("timeout-input.wav");
    generate_voiced_wav_without_ffmpeg(&input_wav, 300);

    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args([
            "robot",
            "run",
            "--input",
            input_wav.to_str().expect("input path should be utf-8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--db",
            db_path.to_str().expect("db path should be utf-8"),
        ])
        .env("FRANKEN_WHISPER_STATE_DIR", &state_root)
        .env("FRANKEN_WHISPER_WHISPER_CPP_BIN", &stub_bin)
        .env("FRANKEN_WHISPER_FFMPEG_BIN", &ffmpeg_stub)
        .env("FRANKEN_WHISPER_FORCE_FFMPEG_NORMALIZE", "1")
        .env("FRANKEN_WHISPER_TEST_FFMPEG_DELAY_SECONDS", "0.2")
        .env("FRANKEN_WHISPER_STAGE_BUDGET_NORMALIZE_MS", "10")
        .output()
        .expect("robot command should execute");

    assert!(
        !output.status.success(),
        "forced stage timeout should return non-zero"
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    let lines: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.is_empty() && trimmed.starts_with('{')
        })
        .map(|line| {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
            assert!(
                parsed.is_ok(),
                "line should be valid json: {}; line={line}",
                parsed
                    .as_ref()
                    .err()
                    .map(|error| error.to_string())
                    .unwrap_or_default()
            );
            parsed.unwrap()
        })
        .collect();

    let timeout_stage = lines
        .iter()
        .find(|line| line["event"] == "stage" && line["code"] == "normalize.timeout")
        .expect("normalize.timeout stage event should be emitted");
    assert_eq!(timeout_stage["stage"], "normalize");

    let run_error = lines
        .iter()
        .find(|line| line["event"] == "run_error")
        .expect("run_error envelope should be emitted");
    assert_eq!(run_error["code"], "FW-STAGE-TIMEOUT");
}

#[test]
fn robot_budget_stage_reflects_env_overrides_and_invalid_fallbacks() {
    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let db_path = dir.path().join("storage.sqlite3");

    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args([
            "robot",
            "run",
            "--stdin",
            "--timeout",
            "0",
            "--no-persist",
            "--db",
            db_path.to_str().expect("db path should be utf-8"),
        ])
        .env("FRANKEN_WHISPER_STATE_DIR", &state_root)
        .env("FRANKEN_WHISPER_STAGE_BUDGET_INGEST_MS", "321")
        .env("FRANKEN_WHISPER_STAGE_BUDGET_NORMALIZE_MS", "0")
        .env("FRANKEN_WHISPER_STAGE_BUDGET_BACKEND_MS", "2222")
        .env("FRANKEN_WHISPER_STAGE_BUDGET_ACCELERATION_MS", "bad-value")
        .output()
        .expect("robot command should execute");

    assert!(!output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("stdout should be utf-8");
    let budgets_line = stdout
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.is_empty() && trimmed.starts_with('{')
        })
        .map(|line| {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
            assert!(
                parsed.is_ok(),
                "line should parse as json: {}; line={line}",
                parsed
                    .as_ref()
                    .err()
                    .map(|error| error.to_string())
                    .unwrap_or_default()
            );
            parsed.unwrap()
        })
        .find(|line| line["event"] == "stage" && line["code"] == "orchestration.budgets")
        .expect("orchestration budget stage should be present");

    let budgets = &budgets_line["payload"]["stage_budgets"];
    assert_eq!(budgets["ingest_ms"], 321);
    assert_eq!(budgets["normalize_ms"], 180000);
    assert_eq!(budgets["backend_ms"], 2222);
    assert_eq!(budgets["acceleration_ms"], 20000);
    assert_eq!(budgets["persist_ms"], 20000);
}

// ---------------------------------------------------------------------------
// CLI argument validation
// ---------------------------------------------------------------------------

#[test]
fn transcribe_args_rejects_no_input() {
    use franken_whisper::cli::TranscribeArgs;

    let args = TranscribeArgs {
        input: None,
        stdin: false,
        mic: false,
        mic_seconds: 15,
        mic_device: None,
        mic_ffmpeg_format: None,
        mic_ffmpeg_source: None,
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        no_diarize: false,
        diarization_engine: DiarizationEngine::Auto,
        diarization_fallback: DiarizationFallbackPolicy::Unknown,
        speaker_hints: None,
        enrollment_edge_guard_ms: 100,
        diarization_max_prototypes: 512,
        persist_speaker_profiles: false,
        db: PathBuf::from(".franken_whisper/storage.sqlite3"),
        no_persist: false,
        timeout: None,
        json: false,
        output_txt: false,
        output_vtt: false,
        output_srt: false,
        output_csv: false,
        output_json_full: false,
        output_lrc: false,
        no_timestamps: false,
        detect_language_only: false,
        split_on_word: false,
        best_of: None,
        beam_size: None,
        max_context: None,
        max_segment_length: None,
        temperature: None,
        temperature_increment: None,
        entropy_threshold: None,
        logprob_threshold: None,
        no_speech_threshold: None,
        vad: false,
        vad_model: None,
        vad_threshold: None,
        vad_min_speech_ms: None,
        vad_min_silence_ms: None,
        vad_max_speech_s: None,
        vad_speech_pad_ms: None,
        vad_samples_overlap: None,
        batch_size: None,
        timestamp_level: None,
        speaker_count_hard: None,
        speaker_count_range: None,
        speaker_count_prior: None,
        gpu_device: None,
        flash_attention: false,
        hf_token: None,
        transcript_path: None,
        no_stem: false,
        diarization_model: None,
        suppress_numerals: false,
        threads: None,
        processors: None,
        no_gpu: false,
        prompt: None,
        carry_initial_prompt: false,
        no_fallback: false,
        suppress_nst: false,
        tiny_diarize: false,
        normalize_segment_text: false,
        offset_ms: None,
        duration_ms: None,
        audio_ctx: None,
        word_threshold: None,
        suppress_regex: None,
        speculative: false,
        fast_model: None,
        quality_model: None,
        speculative_window_ms: None,
        speculative_overlap_ms: None,
        correction_tolerance_wer: None,
        no_adaptive: false,
        always_correct: false,
    };

    let result = args.to_request();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("specify one of"));
}

#[test]
fn transcribe_args_rejects_multiple_inputs() {
    use franken_whisper::cli::TranscribeArgs;

    let args = TranscribeArgs {
        input: Some(PathBuf::from("file.wav")),
        stdin: true,
        mic: false,
        mic_seconds: 15,
        mic_device: None,
        mic_ffmpeg_format: None,
        mic_ffmpeg_source: None,
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        no_diarize: false,
        diarization_engine: DiarizationEngine::Auto,
        diarization_fallback: DiarizationFallbackPolicy::Unknown,
        speaker_hints: None,
        enrollment_edge_guard_ms: 100,
        diarization_max_prototypes: 512,
        persist_speaker_profiles: false,
        db: PathBuf::from(".franken_whisper/storage.sqlite3"),
        no_persist: false,
        timeout: None,
        json: false,
        output_txt: false,
        output_vtt: false,
        output_srt: false,
        output_csv: false,
        output_json_full: false,
        output_lrc: false,
        no_timestamps: false,
        detect_language_only: false,
        split_on_word: false,
        best_of: None,
        beam_size: None,
        max_context: None,
        max_segment_length: None,
        temperature: None,
        temperature_increment: None,
        entropy_threshold: None,
        logprob_threshold: None,
        no_speech_threshold: None,
        vad: false,
        vad_model: None,
        vad_threshold: None,
        vad_min_speech_ms: None,
        vad_min_silence_ms: None,
        vad_max_speech_s: None,
        vad_speech_pad_ms: None,
        vad_samples_overlap: None,
        batch_size: None,
        timestamp_level: None,
        speaker_count_hard: None,
        speaker_count_range: None,
        speaker_count_prior: None,
        gpu_device: None,
        flash_attention: false,
        hf_token: None,
        transcript_path: None,
        no_stem: false,
        diarization_model: None,
        suppress_numerals: false,
        threads: None,
        processors: None,
        no_gpu: false,
        prompt: None,
        carry_initial_prompt: false,
        no_fallback: false,
        suppress_nst: false,
        tiny_diarize: false,
        normalize_segment_text: false,
        offset_ms: None,
        duration_ms: None,
        audio_ctx: None,
        word_threshold: None,
        suppress_regex: None,
        speculative: false,
        fast_model: None,
        quality_model: None,
        speculative_window_ms: None,
        speculative_overlap_ms: None,
        correction_tolerance_wer: None,
        no_adaptive: false,
        always_correct: false,
    };

    let result = args.to_request();
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("mutually exclusive")
    );
}

#[test]
fn transcribe_args_parses_hf_token_override_for_insanely_fast() {
    use clap::Parser as _;

    let cli = Cli::parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "sample.wav",
        "--backend",
        "insanely-fast",
        "--diarize",
        "--hf-token",
        "hf_cli_token",
    ]);

    let command = cli.command;
    assert!(
        matches!(&command, Command::Transcribe(_)),
        "expected transcribe command"
    );
    let Command::Transcribe(args) = command else {
        return;
    };
    assert_eq!(args.hf_token.as_deref(), Some("hf_cli_token"));
    let request = args.to_request().expect("valid request");
    assert_eq!(
        request.backend_params.insanely_fast_hf_token.as_deref(),
        Some("hf_cli_token")
    );
}

#[test]
fn transcribe_args_parses_insanely_fast_transcript_path_override() {
    use clap::Parser as _;

    let cli = Cli::parse_from([
        "franken_whisper",
        "transcribe",
        "--input",
        "sample.wav",
        "--backend",
        "insanely-fast",
        "--transcript-path",
        "artifacts/ifw-output.json",
    ]);

    let command = cli.command;
    assert!(
        matches!(&command, Command::Transcribe(_)),
        "expected transcribe command"
    );
    let Command::Transcribe(args) = command else {
        return;
    };
    assert_eq!(
        args.transcript_path.as_deref(),
        Some(PathBuf::from("artifacts/ifw-output.json").as_path())
    );
    let request = args.to_request().expect("valid request");
    assert_eq!(
        request
            .backend_params
            .insanely_fast_transcript_path
            .as_deref(),
        Some(PathBuf::from("artifacts/ifw-output.json").as_path())
    );
}

#[test]
fn runs_command_accepts_json_output_format() {
    use clap::Parser as _;

    let cli = Cli::parse_from(["franken_whisper", "runs", "--format", "json"]);
    let command = cli.command;
    assert!(
        matches!(&command, Command::Runs(_)),
        "expected runs command"
    );
    let Command::Runs(args) = command else {
        return;
    };
    assert_eq!(args.format, RunsOutputFormat::Json);
    assert_eq!(args.limit, 20);
}

#[test]
fn runs_command_defaults_to_plain_output_format() {
    use clap::Parser as _;

    let cli = Cli::parse_from(["franken_whisper", "runs"]);
    let command = cli.command;
    assert!(
        matches!(&command, Command::Runs(_)),
        "expected runs command"
    );
    let Command::Runs(args) = command else {
        return;
    };
    assert_eq!(args.format, RunsOutputFormat::Plain);
}

#[test]
fn runs_command_accepts_ndjson_output_format() {
    use clap::Parser as _;

    let cli = Cli::parse_from(["franken_whisper", "runs", "--format", "ndjson"]);
    let command = cli.command;
    assert!(
        matches!(&command, Command::Runs(_)),
        "expected runs command"
    );
    let Command::Runs(args) = command else {
        return;
    };
    assert_eq!(args.format, RunsOutputFormat::Ndjson);
}

#[test]
fn tty_audio_decode_defaults_to_fail_closed_recovery() {
    use clap::Parser as _;

    let cli = Cli::parse_from([
        "franken_whisper",
        "tty-audio",
        "decode",
        "--output",
        "out.wav",
    ]);
    let command = cli.command;
    assert!(
        matches!(
            &command,
            Command::TtyAudio {
                command: TtyAudioCommand::Decode { .. }
            }
        ),
        "expected tty-audio decode command"
    );
    let Command::TtyAudio {
        command: TtyAudioCommand::Decode { recovery, .. },
    } = command
    else {
        return;
    };
    assert_eq!(recovery, TtyAudioRecoveryPolicy::FailClosed);
}

#[test]
fn tty_audio_decode_accepts_skip_missing_recovery() {
    use clap::Parser as _;

    let cli = Cli::parse_from([
        "franken_whisper",
        "tty-audio",
        "decode",
        "--output",
        "out.wav",
        "--recovery",
        "skip_missing",
    ]);
    let command = cli.command;
    assert!(
        matches!(
            &command,
            Command::TtyAudio {
                command: TtyAudioCommand::Decode { .. }
            }
        ),
        "expected tty-audio decode command"
    );
    let Command::TtyAudio {
        command: TtyAudioCommand::Decode { recovery, .. },
    } = command
    else {
        return;
    };
    assert_eq!(recovery, TtyAudioRecoveryPolicy::SkipMissing);
}

#[test]
fn tty_audio_retransmit_plan_defaults_to_skip_missing_recovery() {
    use clap::Parser as _;

    let cli = Cli::parse_from(["franken_whisper", "tty-audio", "retransmit-plan"]);
    let command = cli.command;
    assert!(
        matches!(
            &command,
            Command::TtyAudio {
                command: TtyAudioCommand::RetransmitPlan { .. }
            }
        ),
        "expected tty-audio retransmit-plan command"
    );
    let Command::TtyAudio {
        command: TtyAudioCommand::RetransmitPlan { recovery },
    } = command
    else {
        return;
    };
    assert_eq!(recovery, TtyAudioRecoveryPolicy::SkipMissing);
}

#[test]
fn tty_audio_control_ack_parses_expected_fields() {
    use clap::Parser as _;

    let cli = Cli::parse_from([
        "franken_whisper",
        "tty-audio",
        "control",
        "ack",
        "--up-to-seq",
        "12",
    ]);
    let command = cli.command;
    assert!(
        matches!(
            &command,
            Command::TtyAudio {
                command: TtyAudioCommand::Control {
                    command: TtyAudioControlCommand::Ack { .. }
                }
            }
        ),
        "expected tty-audio control ack command"
    );
    let Command::TtyAudio {
        command:
            TtyAudioCommand::Control {
                command: TtyAudioControlCommand::Ack { up_to_seq },
            },
    } = command
    else {
        return;
    };
    assert_eq!(up_to_seq, 12);
}

#[test]
fn tty_audio_control_retransmit_loop_defaults_are_deterministic() {
    use clap::Parser as _;

    let cli = Cli::parse_from(["franken_whisper", "tty-audio", "control", "retransmit-loop"]);
    let command = cli.command;
    assert!(
        matches!(
            &command,
            Command::TtyAudio {
                command: TtyAudioCommand::Control {
                    command: TtyAudioControlCommand::RetransmitLoop { .. }
                }
            }
        ),
        "expected tty-audio control retransmit-loop command"
    );
    let Command::TtyAudio {
        command:
            TtyAudioCommand::Control {
                command: TtyAudioControlCommand::RetransmitLoop { recovery, rounds },
            },
    } = command
    else {
        return;
    };
    assert_eq!(recovery, TtyAudioRecoveryPolicy::SkipMissing);
    assert_eq!(rounds, 1);
}

#[test]
fn tty_audio_control_ack_command_emits_control_ndjson() {
    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args(["tty-audio", "control", "ack", "--up-to-seq", "42"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("run command");

    assert!(
        output.status.success(),
        "command failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let payload: serde_json::Value = serde_json::from_slice(&output.stdout).expect("json output");
    assert_eq!(payload["frame_type"], "ack");
    assert_eq!(payload["up_to_seq"], 42);
}

#[test]
fn tty_audio_control_retransmit_loop_emits_request_then_response() {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD_NO_PAD;
    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    use sha2::{Digest, Sha256};
    use std::io::Write;

    fn payload_b64(data: &[u8]) -> String {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(data).expect("write");
        let compressed = encoder.finish().expect("finish");
        STANDARD_NO_PAD.encode(compressed)
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(data);
        hasher.finalize()
    }

    fn sha256_hex(data: &[u8]) -> String {
        let digest = Sha256::digest(data);
        format!("{digest:x}")
    }

    let chunk0 = b"chunk-0";
    let chunk3 = b"chunk-3";
    let chunk4 = b"chunk-4";

    let lines = vec![
        serde_json::json!({
            "protocol_version": 1,
            "seq": 0,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk0),
            "crc32": crc32(chunk0),
            "payload_sha256": sha256_hex(chunk0),
        }),
        serde_json::json!({
            "protocol_version": 1,
            "seq": 3,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk3),
            "crc32": crc32(chunk3),
            "payload_sha256": sha256_hex(chunk3),
        }),
        serde_json::json!({
            "protocol_version": 1,
            "seq": 4,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk4),
            "crc32": crc32(chunk4),
            "payload_sha256": "deadbeef",
        }),
    ];

    let ndjson = lines
        .into_iter()
        .map(|line| serde_json::to_string(&line).expect("serialize"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";

    let mut child = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args(["tty-audio", "control", "retransmit-loop", "--rounds", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn retransmit-loop");

    {
        let mut stdin = child.stdin.take().expect("stdin");
        stdin.write_all(ndjson.as_bytes()).expect("write ndjson");
    }

    let output = child.wait_with_output().expect("wait");
    assert!(
        output.status.success(),
        "retransmit-loop failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let lines = String::from_utf8(output.stdout).expect("utf8");
    let frames: Vec<serde_json::Value> = lines
        .lines()
        .map(|line| serde_json::from_str(line).expect("json line"))
        .collect();

    assert_eq!(frames.len(), 3, "2 requests + 1 response expected");
    assert_eq!(frames[0]["frame_type"], "retransmit_request");
    assert_eq!(frames[0]["sequences"], json!([1, 2, 4]));
    assert_eq!(frames[1]["frame_type"], "retransmit_request");
    assert_eq!(frames[1]["sequences"], json!([1, 2, 4]));
    assert_eq!(frames[2]["frame_type"], "retransmit_response");
    assert_eq!(frames[2]["sequences"], json!([1, 2, 4]));
}

#[test]
fn tty_audio_retransmit_plan_outputs_missing_and_corrupt_ranges() {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD_NO_PAD;
    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    use sha2::{Digest, Sha256};
    use std::io::Write;

    fn payload_b64(data: &[u8]) -> String {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(data).expect("write");
        let compressed = encoder.finish().expect("finish");
        STANDARD_NO_PAD.encode(compressed)
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(data);
        hasher.finalize()
    }

    fn sha256_hex(data: &[u8]) -> String {
        let digest = Sha256::digest(data);
        format!("{digest:x}")
    }

    let chunk0 = b"chunk-0";
    let chunk3 = b"chunk-3";
    let chunk4 = b"chunk-4";

    let lines = vec![
        serde_json::json!({
            "protocol_version": 1,
            "seq": 0,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk0),
            "crc32": crc32(chunk0),
            "payload_sha256": sha256_hex(chunk0),
        }),
        serde_json::json!({
            "protocol_version": 1,
            "seq": 3,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk3),
            "crc32": crc32(chunk3),
            "payload_sha256": sha256_hex(chunk3),
        }),
        serde_json::json!({
            "protocol_version": 1,
            "seq": 4,
            "codec": "mulaw+zlib+b64",
            "sample_rate_hz": 8000,
            "channels": 1,
            "payload_b64": payload_b64(chunk4),
            "crc32": crc32(chunk4),
            "payload_sha256": "deadbeef",
        }),
    ];

    let ndjson = lines
        .into_iter()
        .map(|line| serde_json::to_string(&line).expect("serialize"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";

    let mut child = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args(["tty-audio", "retransmit-plan"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn retransmit-plan");

    {
        let mut stdin = child.stdin.take().expect("stdin");
        stdin.write_all(ndjson.as_bytes()).expect("write ndjson");
    }

    let output = child.wait_with_output().expect("wait");
    assert!(
        output.status.success(),
        "retransmit-plan failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let payload: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("json retransmit-plan");
    assert_eq!(payload["protocol_version"], 1);
    assert_eq!(payload["requested_sequences"], json!([1, 2, 4]));
    assert_eq!(
        payload["requested_ranges"],
        json!([
            {"start_seq": 1, "end_seq": 2},
            {"start_seq": 4, "end_seq": 4}
        ])
    );
    assert_eq!(payload["gap_count"], 1);
    assert_eq!(payload["integrity_failure_count"], 1);
}

// ---------------------------------------------------------------------------
// Phase 3: word-level segment extraction
// ---------------------------------------------------------------------------

#[test]
fn extract_word_level_segments_from_chunks() {
    let input = json!({
        "chunks": [
            {
                "timestamp": [0.0, 2.0],
                "text": "hello world",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.8, "confidence": 0.95},
                    {"word": "world", "start": 0.8, "end": 2.0, "confidence": 0.88}
                ]
            },
            {
                "timestamp": [2.0, 3.5],
                "text": "test",
                "words": [
                    {"word": "test", "start": 2.0, "end": 3.5}
                ]
            }
        ]
    });
    let segments = backend::extract_segments_from_json(&input);
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].text, "hello");
    assert_eq!(segments[0].start_sec, Some(0.0));
    assert_eq!(segments[0].end_sec, Some(0.8));
    assert_eq!(segments[0].confidence, Some(0.95));
    assert_eq!(segments[0].speaker.as_deref(), Some("SPEAKER_00"));
    assert_eq!(segments[1].text, "world");
    assert_eq!(segments[1].speaker.as_deref(), Some("SPEAKER_00"));
    assert_eq!(segments[2].text, "test");
    assert!(segments[2].speaker.is_none());
}

#[test]
fn extract_chunk_level_segments_without_words() {
    let input = json!({
        "chunks": [
            {"timestamp": [0.0, 1.5], "text": "hello"},
            {"timestamp": [1.5, 3.0], "text": "world"}
        ]
    });
    let segments = backend::extract_segments_from_json(&input);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].text, "hello");
    assert_eq!(segments[0].start_sec, Some(0.0));
    assert_eq!(segments[1].text, "world");
}

// ---------------------------------------------------------------------------
// Phase 3: backend params serialization round-trip
// ---------------------------------------------------------------------------

#[test]
fn backend_params_default_round_trips_through_json() {
    let params = BackendParams::default();
    let json_str = serde_json::to_string(&params).expect("serialize");
    assert!(
        !json_str.contains("hf_example_token"),
        "hf token should be redacted from serialized params"
    );
    assert!(
        !json_str.contains("insanely_fast_hf_token"),
        "hf token field should be omitted from serialized params"
    );
    let parsed: BackendParams = serde_json::from_str(&json_str).expect("deserialize");
    assert!(parsed.output_formats.is_empty());
    assert!(parsed.timestamp_level.is_none());
    assert!(parsed.decoding.is_none());
    assert!(parsed.vad.is_none());
    assert!(parsed.insanely_fast_hf_token.is_none());
    assert!(parsed.insanely_fast_transcript_path.is_none());
    assert!(!parsed.no_timestamps);
    assert!(!parsed.detect_language_only);
}

#[test]
fn backend_params_populated_round_trips_through_json() {
    let params = BackendParams {
        output_formats: vec![OutputFormat::Srt, OutputFormat::Vtt, OutputFormat::Lrc],
        timestamp_level: Some(TimestampLevel::Word),
        decoding: Some(DecodingParams {
            best_of: Some(5),
            beam_size: Some(3),
            temperature: Some(0.2),
            ..DecodingParams::default()
        }),
        vad: Some(VadParams {
            threshold: Some(0.5),
            min_speech_duration_ms: Some(250),
            ..VadParams::default()
        }),
        acoustic_diarization: Some(DiarizationRequest {
            speaker_count: SpeakerCountRequest::Range {
                minimum: 2,
                maximum: 5,
            },
            ..DiarizationRequest::default()
        }),
        diarization_config: Some(DiarizationConfig {
            no_stem: true,
            suppress_numerals: true,
            ..DiarizationConfig::default()
        }),
        gpu_device: Some("0".to_owned()),
        flash_attention: Some(true),
        insanely_fast_hf_token: Some("hf_example_token".to_owned()),
        insanely_fast_transcript_path: Some(PathBuf::from("artifacts/ifw.json")),
        no_timestamps: false,
        detect_language_only: false,
        batch_size: Some(16),
        split_on_word: true,
        threads: Some(4),
        processors: Some(2),
        no_gpu: false,
        prompt: Some("medical terminology".to_owned()),
        carry_initial_prompt: true,
        no_fallback: false,
        suppress_nst: true,
        tiny_diarize: false,
        offset_ms: Some(1000),
        duration_ms: Some(30000),
        audio_ctx: Some(0),
        word_threshold: Some(0.01),
        suppress_regex: Some(r"\[.*\]".to_owned()),
        ..BackendParams::default()
    };
    let json_str = serde_json::to_string(&params).expect("serialize");
    assert!(
        !json_str.contains("hf_example_token"),
        "hf token should be redacted from serialized params"
    );
    assert!(
        !json_str.contains("insanely_fast_hf_token"),
        "hf token field should be omitted from serialized params"
    );
    let parsed: BackendParams = serde_json::from_str(&json_str).expect("deserialize");
    assert_eq!(parsed.output_formats.len(), 3);
    assert_eq!(parsed.timestamp_level, Some(TimestampLevel::Word));
    assert_eq!(parsed.decoding.as_ref().unwrap().best_of, Some(5));
    assert_eq!(parsed.vad.as_ref().unwrap().threshold, Some(0.5));
    assert_eq!(
        parsed
            .acoustic_diarization
            .as_ref()
            .map(|request| &request.speaker_count),
        Some(&SpeakerCountRequest::Range {
            minimum: 2,
            maximum: 5
        })
    );
    assert!(parsed.diarization_config.as_ref().unwrap().no_stem);
    assert_eq!(parsed.batch_size, Some(16));
    assert!(
        parsed.insanely_fast_hf_token.is_none(),
        "hf token should not round-trip through serialized params"
    );
    assert_eq!(
        parsed.insanely_fast_transcript_path.as_deref(),
        Some(PathBuf::from("artifacts/ifw.json").as_path())
    );
    assert!(parsed.split_on_word);
    assert_eq!(parsed.threads, Some(4));
    assert_eq!(parsed.processors, Some(2));
    assert!(!parsed.no_gpu);
    assert_eq!(parsed.prompt.as_deref(), Some("medical terminology"));
    assert!(parsed.carry_initial_prompt);
    assert!(parsed.suppress_nst);
    assert_eq!(parsed.offset_ms, Some(1000));
    assert_eq!(parsed.duration_ms, Some(30000));
    assert_eq!(parsed.audio_ctx, Some(0));
    assert_eq!(parsed.word_threshold, Some(0.01));
    assert_eq!(parsed.suppress_regex.as_deref(), Some(r"\[.*\]"));
}

#[test]
fn transcribe_request_missing_backend_params_deserializes_with_default() {
    // Simulates loading old data that lacks the backend_params field.
    let old_json = json!({
        "input": {"kind": "file", "path": "test.wav"},
        "backend": "auto",
        "model": null,
        "language": "en",
        "translate": false,
        "diarize": false,
        "persist": true,
        "db_path": "db.sqlite3",
        "timeout_ms": null
    });
    let request: TranscribeRequest =
        serde_json::from_value(old_json).expect("should deserialize with default backend_params");
    assert!(request.backend_params.output_formats.is_empty());
    assert!(request.backend_params.decoding.is_none());
    assert!(!request.backend_params.no_timestamps);
}

// ---------------------------------------------------------------------------
// Phase 3: CLI args → BackendParams mapping
// ---------------------------------------------------------------------------

#[test]
fn transcribe_args_maps_output_formats_to_backend_params() {
    use franken_whisper::cli::TranscribeArgs;

    let args = TranscribeArgs {
        input: Some(PathBuf::from("test.wav")),
        stdin: false,
        mic: false,
        mic_seconds: 15,
        mic_device: None,
        mic_ffmpeg_format: None,
        mic_ffmpeg_source: None,
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: true,
        no_diarize: false,
        diarization_engine: DiarizationEngine::Auto,
        diarization_fallback: DiarizationFallbackPolicy::Unknown,
        speaker_hints: None,
        enrollment_edge_guard_ms: 100,
        diarization_max_prototypes: 512,
        persist_speaker_profiles: false,
        db: PathBuf::from(".franken_whisper/storage.sqlite3"),
        no_persist: false,
        timeout: None,
        json: false,
        output_txt: true,
        output_vtt: false,
        output_srt: true,
        output_csv: false,
        output_json_full: false,
        output_lrc: true,
        no_timestamps: false,
        detect_language_only: false,
        split_on_word: false,
        best_of: Some(5),
        beam_size: Some(3),
        max_context: None,
        max_segment_length: None,
        temperature: Some(0.1),
        temperature_increment: None,
        entropy_threshold: None,
        logprob_threshold: None,
        no_speech_threshold: None,
        vad: true,
        vad_model: None,
        vad_threshold: Some(0.6),
        vad_min_speech_ms: None,
        vad_min_silence_ms: None,
        vad_max_speech_s: None,
        vad_speech_pad_ms: None,
        vad_samples_overlap: None,
        batch_size: Some(24),
        timestamp_level: Some(TimestampLevel::Word),
        speaker_count_hard: None,
        speaker_count_range: Some("2..4".to_owned()),
        speaker_count_prior: None,
        gpu_device: Some("0".to_owned()),
        flash_attention: true,
        hf_token: None,
        transcript_path: None,
        no_stem: false,
        diarization_model: None,
        suppress_numerals: false,
        threads: None,
        processors: None,
        no_gpu: false,
        prompt: None,
        carry_initial_prompt: false,
        no_fallback: false,
        suppress_nst: false,
        tiny_diarize: false,
        normalize_segment_text: false,
        offset_ms: None,
        duration_ms: None,
        audio_ctx: None,
        word_threshold: None,
        suppress_regex: None,
        speculative: false,
        fast_model: None,
        quality_model: None,
        speculative_window_ms: None,
        speculative_overlap_ms: None,
        correction_tolerance_wer: None,
        no_adaptive: false,
        always_correct: false,
    };

    let request = args.to_request().expect("valid request");
    let bp = &request.backend_params;

    // Output formats.
    assert_eq!(bp.output_formats.len(), 3);
    assert!(bp.output_formats.contains(&OutputFormat::Txt));
    assert!(bp.output_formats.contains(&OutputFormat::Srt));
    assert!(bp.output_formats.contains(&OutputFormat::Lrc));

    // Decoding params.
    let dec = bp.decoding.as_ref().expect("decoding should be Some");
    assert_eq!(dec.best_of, Some(5));
    assert_eq!(dec.beam_size, Some(3));
    assert_eq!(dec.temperature, Some(0.1));

    // VAD params.
    let vad = bp.vad.as_ref().expect("vad should be Some");
    assert_eq!(vad.threshold, Some(0.6));

    // Speaker count request.
    let count = &bp
        .acoustic_diarization
        .as_ref()
        .expect("speaker count request")
        .speaker_count;
    assert_eq!(
        count,
        &SpeakerCountRequest::Range {
            minimum: 2,
            maximum: 4
        }
    );

    // Top-level batch/timestamp/GPU.
    assert_eq!(bp.batch_size, Some(24));
    assert_eq!(bp.timestamp_level, Some(TimestampLevel::Word));
    assert_eq!(bp.gpu_device.as_deref(), Some("0"));
    assert_eq!(bp.flash_attention, Some(true));
}

// ---------------------------------------------------------------------------
// Phase 3: run report with populated backend_params persists + restores
// ---------------------------------------------------------------------------

#[test]
fn run_report_with_backend_params_round_trips_through_storage() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store");

    let mut report = fixture_report("bp-rt-1", &db_path);
    report.request.backend_params = BackendParams {
        output_formats: vec![OutputFormat::Srt],
        batch_size: Some(16),
        timestamp_level: Some(TimestampLevel::Word),
        no_timestamps: true,
        ..BackendParams::default()
    };
    store.persist_report(&report).expect("persist");

    // Export and import to a fresh DB to test full serialization round-trip.
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");
    sync::export(&db_path, &export_dir, &state_root).expect("export");

    let target_db = dir.path().join("target.sqlite3");
    let result =
        sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");
    assert_eq!(result.runs_imported, 1);
}

// ---------------------------------------------------------------------------
// Phase 3: cross-input mode error consistency
// ---------------------------------------------------------------------------

#[test]
fn file_stdin_mic_all_produce_input_source_variants() {
    // Verify all three input modes produce the correct InputSource variant.
    let file_request = TranscribeRequest {
        input: InputSource::File {
            path: PathBuf::from("test.wav"),
        },
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        persist: false,
        db_path: PathBuf::from("db.sqlite3"),
        timeout_ms: None,
        backend_params: BackendParams::default(),
    };
    assert!(matches!(file_request.input, InputSource::File { .. }));

    let stdin_request = TranscribeRequest {
        input: InputSource::Stdin {
            hint_extension: None,
        },
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        persist: false,
        db_path: PathBuf::from("db.sqlite3"),
        timeout_ms: None,
        backend_params: BackendParams::default(),
    };
    assert!(matches!(stdin_request.input, InputSource::Stdin { .. }));

    let mic_request = TranscribeRequest {
        input: InputSource::Microphone {
            seconds: 5,
            device: None,
            ffmpeg_format: None,
            ffmpeg_source: None,
        },
        backend: BackendKind::Auto,
        model: None,
        language: None,
        translate: false,
        diarize: false,
        persist: false,
        db_path: PathBuf::from("db.sqlite3"),
        timeout_ms: None,
        backend_params: BackendParams::default(),
    };
    assert!(matches!(mic_request.input, InputSource::Microphone { .. }));

    // All three should serialize cleanly.
    for request in [&file_request, &stdin_request, &mic_request] {
        let json_str = serde_json::to_string(request).expect("serialize");
        let _parsed: TranscribeRequest =
            serde_json::from_str(&json_str).expect("round-trip deserialize");
    }
}

#[cfg(unix)]
#[test]
fn transcribe_file_input_crosses_ingest_normalize_backend_with_stub_whisper_cpp() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("file_input.wav");
    generate_voiced_wav(&input_wav);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    assert_eq!(report["result"]["backend"], "whisper_cpp");
    assert_eq!(report["result"]["transcript"], "stub transcript");
    assert_eq!(report["result"]["segments"][0]["speaker"], "SPEAKER_00");
}

#[cfg(unix)]
#[test]
fn tiny_diarize_emits_native_json_artifact_and_preserves_speaker_turn_signal() {
    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_tiny_diarize_stub_binary(dir.path());
    let input_wav = dir.path().join("tiny_diarize_input.wav");
    generate_voiced_wav_without_ffmpeg(&input_wav, 250);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--tiny-diarize",
            "--max-context",
            "0",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    assert_eq!(report["result"]["backend"], "whisper_cpp");
    assert_eq!(
        report["result"]["transcript"],
        "synthetic speaker one synthetic speaker two"
    );
    assert_eq!(
        report["result"]["segments"][0]["text"],
        "synthetic speaker one"
    );
    assert_eq!(report["result"]["segments"][0]["start_sec"], 0.0);
    assert_eq!(report["result"]["segments"][0]["end_sec"], 0.5);
    assert_eq!(report["result"]["segments"][1]["start_sec"], 0.5);
    assert_eq!(report["result"]["segments"][1]["end_sec"], 1.0);
    assert_eq!(
        report["result"]["raw_output"]["transcription"][0]["speaker_turn_next"],
        true
    );
    assert_eq!(
        report["result"]["raw_output"]["transcription"][1]["speaker_turn_next"],
        false
    );
    assert!(
        report["result"]["artifact_paths"]
            .as_array()
            .expect("artifact paths")
            .iter()
            .filter_map(serde_json::Value::as_str)
            .any(|path| path.ends_with("/whispercpp_output.json")),
        "expected artifact path basename whispercpp_output.json"
    );
}

#[cfg(unix)]
#[test]
fn split_on_word_bridge_preserves_zero_duration_observation_in_final_json() {
    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_zero_duration_word_stub_binary(dir.path());
    let input_wav = dir.path().join("zero_duration_word_input.wav");
    generate_voiced_wav_without_ffmpeg(&input_wav, 250);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--split-on-word",
            "--max-segment-length",
            "100",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    let segments = report["result"]["segments"]
        .as_array()
        .expect("final result segments");
    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0]["text"], "first");
    assert_eq!(segments[0]["start_sec"], 0.0);
    assert_eq!(segments[0]["end_sec"], 1.0);
    assert_eq!(segments[1]["text"], "zero-width");
    assert_eq!(segments[1]["confidence"], 0.72);
    assert!(segments[1]["start_sec"].is_null());
    assert!(segments[1]["end_sec"].is_null());
    assert_eq!(segments[2]["text"], "third");
    assert_eq!(segments[2]["start_sec"], 1.5);
    assert_eq!(segments[2]["end_sec"], 2.0);

    assert_eq!(
        report["result"]["raw_output"]["transcription"][1]["offsets"],
        serde_json::json!({"from": 1000, "to": 1000}),
        "the adapter must preserve the decoder's raw evidence"
    );
}

#[cfg(unix)]
#[test]
fn robot_tiny_diarize_emits_redacted_hint_evidence_and_native_acoustic_report() {
    let dir = tempdir().expect("private test directory");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_tiny_diarize_stub_binary(dir.path());
    let input_wav = dir.path().join("tiny_diarize_robot_input.wav");
    generate_voiced_wav_without_ffmpeg(&input_wav, 1_250);

    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args([
            "robot",
            "run",
            "--input",
            input_wav.to_str().expect("test input must be valid UTF-8"),
            "--backend",
            "whisper-cpp",
            "--tiny-diarize",
            "--max-context",
            "0",
            "--diarize",
            "--diarization-engine",
            "acoustic",
            "--no-persist",
        ])
        .env("FRANKEN_WHISPER_STATE_DIR", &state_root)
        .env("FRANKEN_WHISPER_WHISPER_CPP_BIN", &stub_bin)
        // This fixture proves propagation of whisper.cpp's TinyDiarize turn
        // markers into the acoustic report. A cached native Whisper model must
        // not bypass the purpose-built bridge response on developer machines.
        .env("FRANKEN_WHISPER_NATIVE_EXECUTION", "0")
        .output()
        .expect("robot TinyDiarize acoustic run must execute");

    assert!(
        output.status.success(),
        "robot TinyDiarize acoustic run must succeed; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8(output.stdout)
        .expect("robot TinyDiarize acoustic output must be valid UTF-8");
    let events: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("robot output line must be valid NDJSON"))
        .collect();

    let hint_evidence = events
        .iter()
        .find(|event| {
            event["event"] == "stage" && event["code"] == "diarize.tiny_diarize_hint_evidence"
        })
        .expect("robot run must emit TinyDiarize hint evidence");
    let serialized_hint_evidence = hint_evidence.to_string();
    for (description, forbidden) in [
        ("fixture transcript data", "synthetic speaker one"),
        ("fixture transcript data", "synthetic speaker two"),
        ("turn metadata", "speaker_turn_next"),
        ("backend raw material", "raw_output"),
    ] {
        assert!(
            !serialized_hint_evidence.contains(forbidden),
            "TinyDiarize hint evidence must not expose {description}"
        );
    }
    assert!(
        hint_evidence["stage"] == "diarize",
        "TinyDiarize hint evidence must be a diarize stage event"
    );
    let payload = hint_evidence["payload"]
        .as_object()
        .expect("TinyDiarize hint evidence payload must be an object");
    assert!(
        payload.len() == 6
            && [
                "source",
                "requested",
                "accepted_count",
                "validation",
                "trace_id",
                "elapsed_ms",
            ]
            .into_iter()
            .all(|field| payload.contains_key(field)),
        "TinyDiarize hint evidence must contain only aggregate evidence and generic event metadata"
    );
    assert!(
        payload["source"] == "whisper_cpp.transcription",
        "TinyDiarize hint evidence source must identify whisper.cpp transcription"
    );
    assert!(
        payload["requested"] == true,
        "TinyDiarize hint evidence must record requested=true"
    );
    assert!(
        payload["accepted_count"] == 1,
        "TinyDiarize hint evidence must record one accepted boundary"
    );
    assert!(
        payload["validation"] == "accepted",
        "TinyDiarize hint evidence must record accepted validation"
    );
    assert!(
        payload["trace_id"]
            .as_str()
            .is_some_and(|value| !value.is_empty()),
        "TinyDiarize hint evidence must retain a non-empty event trace identifier"
    );
    assert!(
        payload["elapsed_ms"].as_u64().is_some(),
        "TinyDiarize hint evidence must retain bounded elapsed-time metadata"
    );
    for forbidden_key in ["offsets", "transcription", "text", "raw_output", "path"] {
        assert!(
            !payload.contains_key(forbidden_key),
            "TinyDiarize hint evidence must not expose `{forbidden_key}`"
        );
    }

    let run_complete = events
        .iter()
        .find(|event| event["event"] == "run_complete")
        .expect("robot run must emit run_complete");
    assert!(
        run_complete["backend"] == "whisper_cpp",
        "run_complete must identify the whisper.cpp backend"
    );
    let diarization_report = serde_json::from_value::<franken_whisper::model::DiarizationReport>(
        run_complete["diarization"].clone(),
    )
    .expect("run_complete must contain a typed native acoustic report");
    diarization_report
        .speaker_count
        .estimate
        .as_ref()
        .expect("native acoustic report must retain its speaker-count estimate")
        .validate()
        .expect("native acoustic speaker-count estimate must satisfy its public contract");
    assert!(
        run_complete["diarization"]["implementation"] == "native-acoustic-v2",
        "run_complete diarization must identify the native acoustic implementation"
    );
    assert!(
        run_complete["diarization"]["contract_version"] == "acoustic-diarization-v3",
        "run_complete diarization must use the acoustic contract"
    );
    assert!(
        run_complete["diarization"]["feature_schema"] == "acoustic-feature-v2",
        "run_complete diarization must use the native acoustic feature schema"
    );
    assert!(
        !run_complete
            .as_object()
            .expect("run_complete must be an object")
            .contains_key("raw_output"),
        "run_complete must not expose backend raw_output"
    );
}

#[cfg(unix)]
#[test]
fn transcribe_bridge_fixture_with_no_diarize_emits_skip_events_and_preserves_segments() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary_without_speaker(dir.path());
    let input_wav = dir.path().join("file_input_no_optional.wav");
    generate_voiced_wav(&input_wav);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    assert_eq!(report["result"]["backend"], "whisper_cpp");
    assert_eq!(report["result"]["transcript"], "stub transcript");
    assert_eq!(report["result"]["segments"][0]["text"], "stub transcript");
    assert!(
        report["result"]["segments"][0]["speaker"].is_null(),
        "speaker should remain null when the fixture helper disables pipeline diarization"
    );

    let events = report["events"].as_array().expect("events");
    let codes = events
        .iter()
        .filter_map(|event| event["code"].as_str())
        .collect::<Vec<_>>();
    for code in [
        "vad.skip",
        "separate.skip",
        "align.skip",
        "punctuate.skip",
        "diarize.skip",
        "persist.skip",
    ] {
        assert!(
            codes.contains(&code),
            "expected skip event `{code}` in {codes:?}"
        );
    }
}

#[cfg(unix)]
#[test]
fn transcribe_file_input_uses_builtin_normalizer_when_ffmpeg_missing() {
    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("file_input_builtin.wav");
    generate_voiced_wav_without_ffmpeg(&input_wav, 250);

    let report = run_transcribe_json_with_stub_env(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
        &[("PATH", "")],
    );

    assert_eq!(report["result"]["backend"], "whisper_cpp");
    assert_eq!(report["result"]["transcript"], "stub transcript");
    let events = report["events"].as_array().expect("events");
    assert!(
        events.iter().any(|event| event["code"] == "ingest.ok"),
        "expected ingest.ok event in report events"
    );
    assert!(
        events.iter().any(|event| event["code"] == "normalize.ok"),
        "expected normalize.ok event in report events"
    );
}

#[cfg(unix)]
#[test]
fn transcribe_file_input_falls_back_to_builtin_when_auto_provision_is_disabled() {
    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("file_input_builtin_no_autoprov.wav");
    generate_voiced_wav_without_ffmpeg(&input_wav, 250);

    let report = run_transcribe_json_with_stub_env(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
        &[("PATH", ""), ("FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG", "0")],
    );

    assert_eq!(report["result"]["backend"], "whisper_cpp");
    assert_eq!(report["result"]["transcript"], "stub transcript");
    let events = report["events"].as_array().expect("events");
    assert!(
        events.iter().any(|event| event["code"] == "ingest.ok"),
        "expected ingest.ok event in report events"
    );
    assert!(
        events.iter().any(|event| event["code"] == "normalize.ok"),
        "expected normalize.ok event in report events"
    );
}

#[cfg(unix)]
#[test]
fn transcribe_file_input_uses_state_dir_provisioned_ffmpeg_when_path_is_missing() {
    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("file_input_provisioned_ffmpeg.wav");
    let provisioned_bin_dir = state_root.join("tools/ffmpeg/bin");
    let ffmpeg_marker = dir.path().join("ffmpeg_invocation.txt");
    let ffprobe_marker = dir.path().join("ffprobe_invocation.txt");
    generate_voiced_wav_without_ffmpeg(&input_wav, 250);
    write_provisioned_ffmpeg_stub(&provisioned_bin_dir);
    write_provisioned_ffprobe_stub(&provisioned_bin_dir);
    let ffmpeg_marker_env = ffmpeg_marker
        .to_str()
        .expect("marker path should be valid utf-8");
    let ffprobe_marker_env = ffprobe_marker
        .to_str()
        .expect("marker path should be valid utf-8");

    let report = run_transcribe_json_with_stub_env(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
        &[
            ("PATH", ""),
            ("FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG", "0"),
            // Force the ffmpeg normalizer so the provisioned stub is actually
            // invoked. Without this, the built-in symphonia decoder handles
            // WAV files natively and ffmpeg is never called.
            ("FRANKEN_WHISPER_FORCE_FFMPEG_NORMALIZE", "1"),
            ("FRANKEN_WHISPER_TEST_FFMPEG_MARKER", ffmpeg_marker_env),
            ("FRANKEN_WHISPER_TEST_FFPROBE_MARKER", ffprobe_marker_env),
        ],
    );

    assert_eq!(report["result"]["backend"], "whisper_cpp");
    assert_eq!(report["result"]["transcript"], "stub transcript");
    let events = report["events"].as_array().expect("events");
    assert!(
        events.iter().any(|event| event["code"] == "normalize.ok"),
        "expected normalize.ok event in report events"
    );
    assert!(
        ffmpeg_marker.is_file(),
        "state-dir provisioned ffmpeg should be invoked when PATH ffmpeg is unavailable"
    );
    assert!(
        !ffprobe_marker.exists(),
        "normalized PCM WAV duration should use the exact header probe without spawning ffprobe"
    );
}

#[cfg(unix)]
#[test]
fn transcribe_mic_input_uses_state_dir_provisioned_ffmpeg_when_path_is_missing() {
    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let provisioned_bin_dir = state_root.join("tools/ffmpeg/bin");
    let ffmpeg_marker = dir.path().join("ffmpeg_invocation_mic.txt");
    let ffprobe_marker = dir.path().join("ffprobe_invocation_mic.txt");
    write_provisioned_ffmpeg_stub(&provisioned_bin_dir);
    write_provisioned_ffprobe_stub(&provisioned_bin_dir);
    let ffmpeg_marker_env = ffmpeg_marker
        .to_str()
        .expect("marker path should be valid utf-8");
    let ffprobe_marker_env = ffprobe_marker
        .to_str()
        .expect("marker path should be valid utf-8");

    let report = run_transcribe_json_with_stub_env(
        &[
            "--mic",
            "--mic-seconds",
            "1",
            "--mic-ffmpeg-format",
            "lavfi",
            "--mic-ffmpeg-source",
            "anullsrc=r=16000:cl=mono",
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
        &[
            ("PATH", ""),
            ("FRANKEN_WHISPER_AUTO_PROVISION_FFMPEG", "0"),
            ("FRANKEN_WHISPER_TEST_FFMPEG_MARKER", ffmpeg_marker_env),
            ("FRANKEN_WHISPER_TEST_FFPROBE_MARKER", ffprobe_marker_env),
        ],
    );

    let events = report["events"].as_array().expect("events");
    assert!(
        events.iter().any(|event| event["code"] == "ingest.ok"),
        "expected ingest.ok event in report events"
    );
    assert!(
        events.iter().any(|event| event["code"] == "normalize.ok"),
        "expected normalize.ok event in report events"
    );
    assert!(
        ffmpeg_marker.is_file(),
        "state-dir provisioned ffmpeg should be invoked for mic flow when PATH ffmpeg is unavailable"
    );
    assert!(
        ffprobe_marker.is_file(),
        "state-dir provisioned ffprobe should be invoked for mic flow when PATH ffprobe is unavailable"
    );
}

#[cfg(unix)]
#[test]
fn transcribe_stdin_input_crosses_ingest_normalize_backend_with_stub_whisper_cpp() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("stdin_input.wav");
    generate_voiced_wav(&input_wav);
    let wav_bytes = std::fs::read(&input_wav).expect("wav bytes");

    let report = run_transcribe_json_with_stub(
        &[
            "--stdin",
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        Some(&wav_bytes),
        &stub_bin,
        &state_root,
    );

    assert_eq!(report["result"]["backend"], "whisper_cpp");
    assert_eq!(report["result"]["transcript"], "stub transcript");
}

#[cfg(unix)]
#[test]
fn transcribe_mic_line_in_input_crosses_ingest_normalize_backend_with_stub_whisper_cpp() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());

    let report = run_transcribe_json_with_stub(
        &[
            "--mic",
            "--mic-seconds",
            "1",
            "--mic-ffmpeg-format",
            "lavfi",
            "--mic-ffmpeg-source",
            "sine=frequency=440:sample_rate=16000:duration=0.3",
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    assert_eq!(report["result"]["backend"], "whisper_cpp");
    assert_eq!(report["result"]["transcript"], "stub transcript");
}

#[cfg(unix)]
#[test]
fn transcribe_happy_path_stage_sequence_contract_is_stable() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let db_path = dir.path().join("happy_path.sqlite3");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("happy_path.wav");
    generate_voiced_wav(&input_wav);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--db",
            db_path.to_str().expect("utf8"),
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    let events = report["events"]
        .as_array()
        .expect("events should be an array");
    assert!(
        !events.is_empty(),
        "report should include stage events in json output"
    );

    let code_index = |code: &str| -> usize {
        let position = events.iter().position(|event| event["code"] == code);
        assert!(
            position.is_some(),
            "expected event code `{code}` in {events:?}"
        );
        position.unwrap()
    };
    let ingest_start = code_index("ingest.start");
    let normalize_start = code_index("normalize.start");
    let backend_start = code_index("backend.start");
    let acceleration_start = code_index("acceleration.start");
    let acceleration_terminal = code_index("acceleration.ok");
    let persist_start = code_index("persist.start");
    let persist_ok = code_index("persist.ok");

    assert!(ingest_start < normalize_start);
    assert!(normalize_start < backend_start);
    assert!(backend_start < acceleration_start);
    assert!(acceleration_start < acceleration_terminal);
    assert!(acceleration_terminal < persist_start);
    assert!(persist_start < persist_ok);
}

#[cfg(unix)]
#[test]
fn transcribe_backend_stage_payload_exposes_execution_metadata() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("backend_meta.wav");
    generate_voiced_wav(&input_wav);

    let report = run_transcribe_json_with_stub_env(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
        &[("FRANKEN_WHISPER_NATIVE_EXECUTION", "0")],
    );

    let events = report["events"].as_array().expect("events should be array");
    let backend_ok = events
        .iter()
        .find(|event| event["code"] == "backend.ok")
        .expect("backend.ok stage should be present");
    let backend_payload = &backend_ok["payload"];

    assert_eq!(backend_payload["resolved_backend"], "whisper_cpp");
    assert_eq!(backend_payload["implementation"], "bridge");
    assert_eq!(backend_payload["execution_mode"], "bridge_only");
    assert_eq!(backend_payload["native_rollout_stage"], "sole");
    assert!(backend_payload["engine_identity"].is_string());
    assert!(
        backend_payload["engine_version"].is_string()
            || backend_payload["engine_version"].is_null()
    );
    assert!(backend_payload["native_fallback_error"].is_null());

    let replay_event = events
        .iter()
        .find(|event| event["code"] == "replay.envelope")
        .expect("replay.envelope stage should be present");
    let replay_payload = &replay_event["payload"];
    assert_eq!(
        replay_payload["implementation"],
        backend_payload["implementation"]
    );
    assert_eq!(
        replay_payload["execution_mode"],
        backend_payload["execution_mode"]
    );
    assert_eq!(
        replay_payload["native_rollout_stage"],
        backend_payload["native_rollout_stage"]
    );
    assert_eq!(
        replay_payload["native_fallback_error"],
        backend_payload["native_fallback_error"]
    );
}

#[cfg(unix)]
#[test]
fn transcribe_bridge_only_mode_recovers_with_native_when_bridge_binary_missing() {
    if !ffmpeg_available() {
        return;
    }
    // The native engine is now real inference (bd-jryr): recovery requires a
    // resolvable model file. Skip when the gated test model is absent.
    if franken_whisper::native_engine::find_model_file("tiny.en").is_none() {
        eprintln!("skipping: ggml-tiny.en.bin not found (see scripts/fetch_test_models.sh)");
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let missing_bridge_bin = dir.path().join("missing-whisper-cli");
    let input_wav = dir.path().join("bridge_recovery.wav");
    generate_voiced_wav(&input_wav);

    let report = run_transcribe_json_with_stub_env(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
        &[
            (
                "FRANKEN_WHISPER_WHISPER_CPP_BIN",
                missing_bridge_bin.to_str().expect("utf8"),
            ),
            ("FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL", "tiny.en"),
            ("FRANKEN_WHISPER_BRIDGE_NATIVE_RECOVERY", "1"),
        ],
    );

    let events = report["events"].as_array().expect("events should be array");
    let backend_ok = events
        .iter()
        .find(|event| event["code"] == "backend.ok")
        .expect("backend.ok stage should be present");
    let backend_payload = &backend_ok["payload"];

    assert_eq!(backend_payload["resolved_backend"], "whisper_cpp");
    assert_eq!(backend_payload["implementation"], "native");
    assert_eq!(
        backend_payload["execution_mode"],
        "bridge_only_native_recovery"
    );
    assert_eq!(backend_payload["native_rollout_stage"], "sole");
    assert!(backend_payload["native_fallback_error"].is_string());

    let replay_event = events
        .iter()
        .find(|event| event["code"] == "replay.envelope")
        .expect("replay.envelope stage should be present");
    let replay_payload = &replay_event["payload"];

    assert_eq!(
        replay_payload["implementation"],
        backend_payload["implementation"]
    );
    assert_eq!(
        replay_payload["execution_mode"],
        backend_payload["execution_mode"]
    );
    assert_eq!(
        replay_payload["native_rollout_stage"],
        backend_payload["native_rollout_stage"]
    );
    assert_eq!(
        replay_payload["native_fallback_error"],
        backend_payload["native_fallback_error"]
    );
}

#[cfg(unix)]
#[test]
fn transcribe_acceleration_context_telemetry_round_trips_in_run_artifacts() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let db_path = dir.path().join("telemetry.sqlite3");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("telemetry.wav");
    generate_voiced_wav(&input_wav);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "whisper-cpp",
            "--db",
            db_path.to_str().expect("utf8"),
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    let events = report["events"].as_array().expect("events should be array");
    let accel_event = events
        .iter()
        .find(|event| event["code"] == "acceleration.context")
        .expect("acceleration.context stage event should be present");
    let payload = &accel_event["payload"];
    assert!(payload["logical_stream_owner_id"].is_string());
    assert!(payload["logical_stream_kind"].is_string());
    assert!(payload["acceleration_backend"].is_string());
    assert!(payload["mode"].is_string());
    let fence = &payload["cancellation_fence"];
    assert_eq!(fence["status"], "open");
    assert!(fence["checked_at_rfc3339"].is_string());
    assert!(fence["budget_remaining_ms"].is_number() || fence["budget_remaining_ms"].is_null());
    assert!(fence["error"].is_null());
    assert!(fence["error_code"].is_null());

    let run_level_context = &report["acceleration_context"];
    assert_eq!(
        run_level_context["logical_stream_owner_id"],
        payload["logical_stream_owner_id"]
    );
    assert_eq!(
        run_level_context["logical_stream_kind"],
        payload["logical_stream_kind"]
    );
    assert_eq!(
        run_level_context["cancellation_fence"],
        payload["cancellation_fence"]
    );

    let evidence = report["evidence"]
        .as_array()
        .expect("run artifact should include evidence array");
    let accel_evidence = evidence
        .iter()
        .find(|entry| entry["logical_stream_owner_id"].is_string())
        .expect("acceleration context should be captured in run evidence");
    assert_eq!(
        accel_evidence["logical_stream_owner_id"],
        payload["logical_stream_owner_id"]
    );
    assert_eq!(
        accel_evidence["logical_stream_kind"],
        payload["logical_stream_kind"]
    );
    assert_eq!(
        accel_evidence["cancellation_fence"],
        payload["cancellation_fence"]
    );

    let run_id = report["run_id"].as_str().expect("run_id should be present");
    let store = RunStore::open(&db_path).expect("store should open");
    let details = store
        .load_run_details(run_id)
        .expect("load should succeed")
        .expect("persisted run should exist");
    let persisted_event = details
        .events
        .iter()
        .find(|event| event.code == "acceleration.context")
        .expect("acceleration.context event should be persisted");
    assert_eq!(
        persisted_event.payload["logical_stream_owner_id"],
        payload["logical_stream_owner_id"]
    );
    assert_eq!(
        persisted_event.payload["logical_stream_kind"],
        payload["logical_stream_kind"]
    );
    assert_eq!(
        persisted_event.payload["cancellation_fence"],
        payload["cancellation_fence"]
    );
}

#[cfg(unix)]
#[test]
fn transcribe_backend_routing_stage_event_has_required_decision_contract_fields() {
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    let stub_bin = write_whisper_cpp_stub_binary(dir.path());
    let input_wav = dir.path().join("routing.wav");
    generate_voiced_wav(&input_wav);

    let report = run_transcribe_json_with_stub(
        &[
            "--input",
            input_wav.to_str().expect("utf8"),
            "--backend",
            "auto",
            "--no-persist",
            "--json",
        ],
        None,
        &stub_bin,
        &state_root,
    );

    let events = report["events"].as_array().expect("events should be array");
    let routing_event = events
        .iter()
        .find(|event| event["code"] == "backend.routing.decision_contract")
        .expect("routing decision contract stage should be present");
    let payload = &routing_event["payload"];

    let required_non_null_fields = [
        "version",
        "schema_version",
        "policy_id",
        "mode",
        "contract",
        "state_space",
        "observed_state",
        "action_set",
        "chosen_action",
        "fallback_active",
        "expected_losses",
        "posterior_snapshot",
        "calibration_score",
        "e_process",
        "ci_width",
        "recommended_order",
        "static_order",
        "availability",
        "provenance",
        "duration_seconds",
        "duration_bucket",
        "diarize",
        "decision_id",
        "trace_id",
    ];
    for field in required_non_null_fields {
        assert!(
            !payload[field].is_null(),
            "routing stage payload missing required field `{field}`"
        );
    }
    assert!(
        payload.get("adaptive_router_state").is_some(),
        "routing stage payload should include adaptive_router_state key"
    );
}

#[test]
fn transcribe_args_maps_mic_line_in_envelope_into_request() {
    use franken_whisper::cli::TranscribeArgs;

    let args = TranscribeArgs {
        input: None,
        stdin: false,
        mic: true,
        mic_seconds: 42,
        mic_device: Some("hw:2,0".to_owned()),
        mic_ffmpeg_format: Some("alsa".to_owned()),
        mic_ffmpeg_source: Some("hw:2,0".to_owned()),
        backend: BackendKind::WhisperCpp,
        model: Some("base.en".to_owned()),
        language: Some("en".to_owned()),
        translate: false,
        diarize: false,
        no_diarize: false,
        diarization_engine: DiarizationEngine::Auto,
        diarization_fallback: DiarizationFallbackPolicy::Unknown,
        speaker_hints: None,
        enrollment_edge_guard_ms: 100,
        diarization_max_prototypes: 512,
        persist_speaker_profiles: false,
        db: PathBuf::from(".franken_whisper/storage.sqlite3"),
        no_persist: true,
        timeout: Some(30),
        json: false,
        output_txt: false,
        output_vtt: false,
        output_srt: false,
        output_csv: false,
        output_json_full: false,
        output_lrc: false,
        no_timestamps: false,
        detect_language_only: false,
        split_on_word: false,
        best_of: None,
        beam_size: None,
        max_context: None,
        max_segment_length: None,
        temperature: None,
        temperature_increment: None,
        entropy_threshold: None,
        logprob_threshold: None,
        no_speech_threshold: None,
        vad: false,
        vad_model: None,
        vad_threshold: None,
        vad_min_speech_ms: None,
        vad_min_silence_ms: None,
        vad_max_speech_s: None,
        vad_speech_pad_ms: None,
        vad_samples_overlap: None,
        batch_size: None,
        timestamp_level: None,
        speaker_count_hard: None,
        speaker_count_range: None,
        speaker_count_prior: None,
        gpu_device: None,
        flash_attention: false,
        hf_token: None,
        transcript_path: None,
        no_stem: false,
        diarization_model: None,
        suppress_numerals: false,
        threads: None,
        processors: None,
        no_gpu: false,
        prompt: None,
        carry_initial_prompt: false,
        no_fallback: false,
        suppress_nst: false,
        tiny_diarize: false,
        normalize_segment_text: false,
        offset_ms: None,
        duration_ms: None,
        audio_ctx: None,
        word_threshold: None,
        suppress_regex: None,
        speculative: false,
        fast_model: None,
        quality_model: None,
        speculative_window_ms: None,
        speculative_overlap_ms: None,
        correction_tolerance_wer: None,
        no_adaptive: false,
        always_correct: false,
    };

    let request = args.to_request().expect("valid mic request");
    let input = request.input;
    assert!(
        matches!(
            &input,
            InputSource::Microphone {
                seconds: _,
                device: _,
                ffmpeg_format: _,
                ffmpeg_source: _
            }
        ),
        "expected microphone input source, got {input:?}"
    );
    if let InputSource::Microphone {
        seconds,
        device,
        ffmpeg_format,
        ffmpeg_source,
    } = input
    {
        assert_eq!(seconds, 42);
        assert_eq!(device.as_deref(), Some("hw:2,0"));
        assert_eq!(ffmpeg_format.as_deref(), Some("alsa"));
        assert_eq!(ffmpeg_source.as_deref(), Some("hw:2,0"));
    } else {
        return;
    }
    assert_eq!(request.backend, BackendKind::WhisperCpp);
    assert!(!request.persist);
    assert_eq!(request.timeout_ms, Some(30_000));
}

// ---------------------------------------------------------------------------
// Conformance: segment invariant enforcement
// ---------------------------------------------------------------------------

#[test]
fn conformance_rejects_out_of_bounds_confidence() {
    use franken_whisper::conformance::validate_segment_invariants;

    let segments = vec![TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: "hello".to_owned(),
        speaker: None,
        confidence: Some(1.5),
    }];
    let error = validate_segment_invariants(&segments).expect_err("should reject confidence > 1.0");
    assert!(error.to_string().contains("confidence"));
}

#[test]
fn conformance_rejects_negative_confidence() {
    use franken_whisper::conformance::validate_segment_invariants;

    let segments = vec![TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: "hello".to_owned(),
        speaker: None,
        confidence: Some(-0.01),
    }];
    let error = validate_segment_invariants(&segments).expect_err("should reject confidence < 0.0");
    assert!(error.to_string().contains("confidence"));
}

#[test]
fn conformance_rejects_empty_speaker_label() {
    use franken_whisper::conformance::validate_segment_invariants;

    let segments = vec![TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: "hello".to_owned(),
        speaker: Some("   ".to_owned()),
        confidence: None,
    }];
    let error = validate_segment_invariants(&segments).expect_err("should reject empty speaker");
    assert!(error.to_string().contains("empty speaker"));
}

#[test]
fn conformance_accepts_valid_segments_with_all_fields() {
    use franken_whisper::conformance::validate_segment_invariants;

    let segments = vec![
        TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "hello".to_owned(),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: Some(0.0),
        },
        TranscriptionSegment {
            start_sec: Some(1.0),
            end_sec: Some(2.0),
            text: "world".to_owned(),
            speaker: Some("SPEAKER_01".to_owned()),
            confidence: Some(1.0),
        },
    ];
    assert!(validate_segment_invariants(&segments).is_ok());
}

#[test]
fn conformance_engine_trait_capabilities_are_consistent() {
    use franken_whisper::backend::{
        Engine, InsanelyFastEngine, WhisperCppEngine, WhisperDiarizationEngine,
    };

    let wcpp = WhisperCppEngine;
    assert_eq!(wcpp.kind(), BackendKind::WhisperCpp);
    assert!(!wcpp.capabilities().supports_diarization);
    assert!(wcpp.capabilities().supports_translation);

    let ifw = InsanelyFastEngine;
    assert_eq!(ifw.kind(), BackendKind::InsanelyFast);
    assert!(ifw.capabilities().supports_diarization);
    assert!(ifw.capabilities().supports_gpu);

    let wd = WhisperDiarizationEngine;
    assert_eq!(wd.kind(), BackendKind::WhisperDiarization);
    assert!(wd.capabilities().supports_diarization);
    assert!(!wd.capabilities().supports_translation);
}

#[test]
fn conformance_all_engines_returns_three_engines() {
    use franken_whisper::backend::all_engines;

    let engines = all_engines();
    assert_eq!(engines.len(), 6);

    let names: Vec<&str> = engines.iter().map(|e| e.name()).collect();
    assert!(names.contains(&"whisper.cpp"));
    assert!(names.contains(&"insanely-fast-whisper"));
    assert!(names.contains(&"whisper-diarization"));
}

#[test]
fn conformance_engine_for_returns_correct_engine() {
    use franken_whisper::backend::engine_for;

    let engine = engine_for(BackendKind::WhisperCpp).expect("should return engine");
    assert_eq!(engine.name(), "whisper.cpp");

    let engine = engine_for(BackendKind::InsanelyFast).expect("should return engine");
    assert_eq!(engine.name(), "insanely-fast-whisper");

    assert!(engine_for(BackendKind::Auto).is_none());
}

// ---------------------------------------------------------------------------
// Sync: ConflictPolicy::Overwrite
// ---------------------------------------------------------------------------

#[test]
fn sync_overwrite_policy_replaces_conflicting_run() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("target.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    // Persist run in source and target (same run_id, different transcripts)
    let store = RunStore::open(&source_db).expect("source store");
    let mut source_report = fixture_report("conflict-1", &source_db);
    source_report.result.transcript = "updated transcript".to_owned();
    store
        .persist_report(&source_report)
        .expect("persist source");

    let target_store = RunStore::open(&target_db).expect("target store");
    target_store
        .persist_report(&fixture_report("conflict-1", &target_db))
        .expect("persist target");

    // Export source
    sync::export(&source_db, &export_dir, &state_root).expect("export");

    // Import with Overwrite policy
    let result = sync::import(
        &target_db,
        &export_dir,
        &state_root,
        ConflictPolicy::Overwrite,
    )
    .expect("import with overwrite");

    assert_eq!(result.runs_imported, 1);

    // Verify overwritten data
    let details = target_store
        .load_run_details("conflict-1")
        .expect("load")
        .expect("exists");
    assert_eq!(details.transcript, "updated transcript");
}

#[test]
fn sync_overwrite_strict_policy_replaces_conflicting_segment_row() {
    use franken_whisper::storage::BlockingConnection as Connection;
    use fsqlite_types::SqliteValue;

    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("target.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    let source_store = RunStore::open(&source_db).expect("source store");
    source_store
        .persist_report(&fixture_report("strict-seg-1", &source_db))
        .expect("persist source");

    let target_store = RunStore::open(&target_db).expect("target store");
    target_store
        .persist_report(&fixture_report("strict-seg-1", &target_db))
        .expect("persist target");

    // Introduce a child-row payload conflict without changing the parent run row.
    let conn = Connection::open(target_db.display().to_string()).expect("conn");
    conn.execute_with_params(
        "UPDATE segments SET text = ?1 WHERE run_id = ?2 AND idx = ?3",
        &[
            SqliteValue::Text("mutated segment text".to_owned().into()),
            SqliteValue::Text("strict-seg-1".to_owned().into()),
            SqliteValue::Integer(0),
        ],
    )
    .expect("mutate target segment");

    sync::export(&source_db, &export_dir, &state_root).expect("export");
    sync::import(
        &target_db,
        &export_dir,
        &state_root,
        ConflictPolicy::OverwriteStrict,
    )
    .expect("strict overwrite import should reconcile child conflict");

    let details = target_store
        .load_run_details("strict-seg-1")
        .expect("load")
        .expect("exists");
    assert_eq!(
        details.segments[0].text, "integration test",
        "strict overwrite should restore source segment payload"
    );
}

#[test]
fn sync_reject_policy_fails_on_conflict() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("target.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    // Same run in both databases
    let store = RunStore::open(&source_db).expect("source store");
    store
        .persist_report(&fixture_report("dup-1", &source_db))
        .expect("persist source");

    let target_store = RunStore::open(&target_db).expect("target store");
    let mut alt_report = fixture_report("dup-1", &target_db);
    alt_report.result.transcript = "different".to_owned();
    target_store
        .persist_report(&alt_report)
        .expect("persist target");

    sync::export(&source_db, &export_dir, &state_root).expect("export");

    let result = sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject);
    assert!(result.is_err(), "reject policy should fail on conflict");
}

// ---------------------------------------------------------------------------
// Storage: list_recent_runs with limit, load_latest_run_details
// ---------------------------------------------------------------------------

#[test]
fn list_recent_runs_respects_limit() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store");

    for i in 0..10 {
        store
            .persist_report(&fixture_report(&format!("limit-{i}"), &db_path))
            .expect("persist");
    }

    let all = store.list_recent_runs(100).expect("list all");
    assert_eq!(all.len(), 10);

    let limited = store.list_recent_runs(3).expect("list limited");
    assert_eq!(limited.len(), 3);
}

#[test]
fn load_latest_run_details_returns_most_recent() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store");

    let mut report1 = fixture_report("first", &db_path);
    report1.started_at_rfc3339 = "2026-02-20T00:00:00Z".to_owned();
    store.persist_report(&report1).expect("persist 1");

    let mut report2 = fixture_report("second", &db_path);
    report2.started_at_rfc3339 = "2026-02-22T00:00:00Z".to_owned();
    store.persist_report(&report2).expect("persist 2");

    let latest = store
        .load_latest_run_details()
        .expect("load latest")
        .expect("should exist");
    assert_eq!(latest.run_id, "second");
}

// ---------------------------------------------------------------------------
// Sync: idempotent re-import
// ---------------------------------------------------------------------------

#[test]
fn sync_double_import_is_idempotent() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("target.sqlite3");
    let export_dir = dir.path().join("export");
    let state_root = dir.path().join("state");

    let store = RunStore::open(&source_db).expect("store");
    store
        .persist_report(&fixture_report("idem-1", &source_db))
        .expect("persist");

    sync::export(&source_db, &export_dir, &state_root).expect("export");

    // First import
    let result1 = sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
        .expect("import 1");
    assert_eq!(result1.runs_imported, 1);

    // Second import with overwrite — same data, should succeed
    let result2 = sync::import(
        &target_db,
        &export_dir,
        &state_root,
        ConflictPolicy::Overwrite,
    )
    .expect("import 2");
    assert_eq!(result2.runs_imported, 1);

    // Verify only 1 run in target
    let target_store = RunStore::open(&target_db).expect("target store");
    let runs = target_store.list_recent_runs(100).expect("list");
    assert_eq!(runs.len(), 1);
}

// ---------------------------------------------------------------------------
// Storage: report with evidence and warnings round-trip
// ---------------------------------------------------------------------------

#[test]
fn run_report_with_evidence_and_warnings_round_trips() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store");

    let mut report = fixture_report("evidence-1", &db_path);
    report.evidence = vec![
        json!({"type": "decision", "action": "try_whisper_cpp"}),
        json!({"type": "calibration", "score": 0.85}),
    ];
    report.warnings = vec![
        "fallback: deterministic route used".to_owned(),
        "divergence: replay drift detected".to_owned(),
    ];
    store.persist_report(&report).expect("persist");

    let details = store
        .load_run_details("evidence-1")
        .expect("load")
        .expect("exists");
    assert_eq!(details.run_id, "evidence-1");
    assert_eq!(details.segments.len(), 2);
    assert_eq!(details.events.len(), 2);
}

// ---------------------------------------------------------------------------
// Robot: health subcommand
// ---------------------------------------------------------------------------

#[test]
fn robot_health_args_parse_correctly() {
    use clap::Parser;
    let cli = Cli::parse_from([
        "franken_whisper",
        "robot",
        "health",
        "--db",
        "/tmp/health_test.sqlite3",
    ]);
    let command = cli.command;
    assert!(
        matches!(
            &command,
            Command::Robot {
                command: franken_whisper::cli::RobotCommand::Health(_)
            }
        ),
        "expected Robot Health"
    );
    let Command::Robot {
        command: franken_whisper::cli::RobotCommand::Health(args),
    } = command
    else {
        return;
    };
    assert_eq!(
        args.db,
        std::path::PathBuf::from("/tmp/health_test.sqlite3")
    );
}

#[test]
fn robot_health_default_db_path() {
    use clap::Parser;
    let cli = Cli::parse_from(["franken_whisper", "robot", "health"]);
    let command = cli.command;
    assert!(
        matches!(
            &command,
            Command::Robot {
                command: franken_whisper::cli::RobotCommand::Health(_)
            }
        ),
        "expected Robot Health"
    );
    let Command::Robot {
        command: franken_whisper::cli::RobotCommand::Health(args),
    } = command
    else {
        return;
    };
    assert_eq!(
        args.db,
        std::path::PathBuf::from(".franken_whisper/storage.sqlite3")
    );
}

#[test]
fn robot_health_produces_valid_ndjson_event() {
    use franken_whisper::robot::{
        HEALTH_REPORT_REQUIRED_FIELDS, build_health_report, health_report_value,
    };

    let report = build_health_report(std::path::Path::new("/tmp/health_integration_test.sqlite3"));
    let value = health_report_value(&report);

    // Verify all required fields are present.
    for field in HEALTH_REPORT_REQUIRED_FIELDS {
        assert!(
            value.get(field).is_some(),
            "health report missing required field `{field}`"
        );
    }

    // Verify event type.
    assert_eq!(value["event"], "health.report");

    // Verify overall_status is one of the valid values.
    let status = value["overall_status"].as_str().unwrap();
    assert!(
        ["ok", "degraded", "unavailable"].contains(&status),
        "unexpected overall_status: {status}"
    );

    // Verify backends is an array.
    assert!(value["backends"].is_array(), "backends should be an array");

    // Verify the output serializes to valid NDJSON.
    let serialized = serde_json::to_string(&value).expect("serialize");
    let _roundtrip: serde_json::Value = serde_json::from_str(&serialized).expect("roundtrip parse");
}

// ---------------------------------------------------------------------------
// Robot: backends subcommand (bd-38s)
// ---------------------------------------------------------------------------

#[test]
fn robot_backends_args_parse_correctly() {
    use clap::Parser;
    let cli = Cli::parse_from(["franken_whisper", "robot", "backends"]);
    let command = cli.command;
    assert!(
        matches!(
            &command,
            Command::Robot {
                command: franken_whisper::cli::RobotCommand::Backends
            }
        ),
        "expected Robot Backends"
    );
}

#[test]
fn robot_backends_report_contains_all_backend_kinds() {
    use franken_whisper::robot::build_backends_report;

    let report = build_backends_report();
    let names: Vec<&str> = report.backends.iter().map(|b| b.name.as_str()).collect();

    assert!(
        names.contains(&"whisper.cpp"),
        "missing whisper.cpp in backends report"
    );
    assert!(
        names.contains(&"insanely-fast-whisper"),
        "missing insanely-fast-whisper in backends report"
    );
    assert!(
        names.contains(&"whisper-diarization"),
        "missing whisper-diarization in backends report"
    );
}

#[test]
fn robot_backends_discovery_event_has_required_fields() {
    use franken_whisper::robot::{
        BACKENDS_DISCOVERY_REQUIRED_FIELDS, backends_discovery_value, build_backends_report,
    };

    let report = build_backends_report();
    let value = backends_discovery_value(&report);

    for field in BACKENDS_DISCOVERY_REQUIRED_FIELDS {
        assert!(
            value.get(field).is_some(),
            "backends discovery missing required field `{field}`"
        );
    }

    assert_eq!(value["event"], "backends.discovery");
    assert!(value["backends"].is_array(), "backends should be an array");

    // Each backend entry should have name, kind, available, and capabilities.
    for entry in value["backends"].as_array().unwrap() {
        assert!(entry.get("name").is_some(), "backend entry missing `name`");
        assert!(entry.get("kind").is_some(), "backend entry missing `kind`");
        assert!(
            entry.get("available").is_some(),
            "backend entry missing `available`"
        );
        assert!(
            entry.get("capabilities").is_some(),
            "backend entry missing `capabilities`"
        );
    }
}

#[test]
fn robot_backends_discovery_serializes_to_valid_ndjson() {
    use franken_whisper::robot::{backends_discovery_value, build_backends_report};

    let report = build_backends_report();
    let value = backends_discovery_value(&report);
    let serialized = serde_json::to_string(&value).expect("serialize backends discovery");

    // Must be valid JSON (single line = valid NDJSON entry).
    let roundtrip: serde_json::Value = serde_json::from_str(&serialized).expect("roundtrip parse");
    assert_eq!(roundtrip["event"], "backends.discovery");

    // Must not contain embedded newlines (valid NDJSON = single line).
    assert!(
        !serialized.contains('\n'),
        "NDJSON must be single-line, but got newlines"
    );
}

// ---------------------------------------------------------------------------
// Robot: schema subcommand (bd-38s)
// ---------------------------------------------------------------------------

#[test]
fn robot_schema_args_parse_correctly() {
    use clap::Parser;
    let cli = Cli::parse_from(["franken_whisper", "robot", "schema"]);
    let command = cli.command;
    assert!(
        matches!(
            &command,
            Command::Robot {
                command: franken_whisper::cli::RobotCommand::Schema
            }
        ),
        "expected Robot Schema"
    );
}

#[test]
fn robot_schema_value_contains_version_and_events() {
    use franken_whisper::robot::robot_schema_value;

    let schema = robot_schema_value();

    assert!(schema.get("version").is_some(), "schema missing `version`");
    assert!(
        schema.get("schema_version").is_some(),
        "schema missing `schema_version`"
    );
    assert!(schema.get("events").is_some(), "schema missing `events`");
    assert_eq!(
        schema["line_oriented"], "ndjson",
        "schema should declare ndjson line-oriented format"
    );
}

#[test]
fn robot_schema_value_documents_all_core_event_types() {
    use franken_whisper::robot::robot_schema_value;

    let schema = robot_schema_value();
    let events = schema["events"]
        .as_object()
        .expect("events should be an object");

    let expected_events = [
        "run_start",
        "stage",
        "run_complete",
        "run_error",
        "backends.discovery",
        "health.report",
    ];

    for event_name in expected_events {
        assert!(
            events.contains_key(event_name),
            "schema missing event type `{event_name}`"
        );
    }
}

#[test]
fn robot_schema_event_entries_have_required_and_example() {
    use franken_whisper::robot::robot_schema_value;

    let schema = robot_schema_value();
    let events = schema["events"]
        .as_object()
        .expect("events should be an object");

    for (name, entry) in events {
        assert!(
            entry.get("required").is_some(),
            "event `{name}` missing `required` field"
        );
        assert!(
            entry.get("example").is_some(),
            "event `{name}` missing `example` field"
        );
    }
}

// ---------------------------------------------------------------------------
// Robot: routing-history with empty DB (bd-38s)
// ---------------------------------------------------------------------------

#[test]
fn robot_routing_history_empty_db_emits_terminal_count() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("empty_routing.sqlite3");
    let store = RunStore::open(&db_path).expect("store open");

    // Query with no runs — should return empty.
    let runs = store.list_recent_runs(10).expect("list runs");
    assert!(runs.is_empty(), "fresh DB should have no runs");

    let output = run_agent_command(
        env!("CARGO_BIN_EXE_fw"),
        &[
            "robot",
            "routing-history",
            "--db",
            db_path.to_str().expect("UTF-8 database path"),
        ],
    );
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 routing history");
    assert_eq!(stdout.lines().count(), 1);
    let terminal: serde_json::Value = serde_json::from_str(stdout.trim()).expect("terminal JSON");
    assert_eq!(terminal["event"], "routing_history.complete");
    assert_eq!(terminal["records"], 0);
}

#[test]
fn robot_routing_history_filters_by_run_id() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("filter_routing.sqlite3");
    let store = RunStore::open(&db_path).expect("store open");

    // Insert two reports with different IDs.
    let report_a = fixture_report("routing-filter-a", &db_path);
    let report_b = fixture_report("routing-filter-b", &db_path);
    store.persist_report(&report_a).expect("persist a");
    store.persist_report(&report_b).expect("persist b");

    // Filtering by specific run_id should return only that run.
    let details = store
        .load_run_details("routing-filter-a")
        .expect("load details");
    assert!(details.is_some(), "should find run routing-filter-a");
    assert_eq!(details.unwrap().run_id, "routing-filter-a");

    // Non-existent run_id should return None.
    let missing = store
        .load_run_details("nonexistent-run-xyz")
        .expect("load missing");
    assert!(missing.is_none(), "non-existent run_id should return None");
}

// ---------------------------------------------------------------------------
// Speculative CLI integration: end-to-end exercise of the dispatch path that
// routes `--speculative` requests through `SpeculativeStreamingPipeline` in
// `execute_backend_speculative()`. The bridged whisper.cpp invocation is
// stubbed to return canned segments so the test stays hermetic and fast.
// ---------------------------------------------------------------------------

#[cfg(unix)]
#[test]
fn speculative_cli_dispatch_emits_partial_confirm_and_stats_events() {
    use std::fs;

    // Skip if ffmpeg/ffprobe aren't available — the speculation pipeline calls
    // `probe_duration_seconds_with_timeout` which shells out to ffprobe, and the
    // normalize stage uses ffmpeg as a fallback (even though our input is
    // already 16 kHz mono and should bypass it).
    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    fs::create_dir_all(&state_root).expect("state root");

    // Pre-normalized 16 kHz mono WAV so the Normalize stage detects passthrough.
    let input_wav = dir.path().join("speculative_input.wav");
    generate_voiced_wav(&input_wav);

    // Stub whisper.cpp: writes a canned JSON-encoded transcript for each invocation.
    // The speculative pipeline calls this twice per window (fast lane, then quality
    // lane); both invocations write to the work directory and the bridge reads the
    // most recent one for each lane.
    let bin_dir = dir.path().join("bin");
    std::fs::create_dir_all(&bin_dir).expect("bin dir");
    let stub_bin = write_whisper_cpp_stub_binary(&bin_dir);

    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args([
            "robot",
            "run",
            "--input",
            input_wav.to_str().expect("utf-8 path"),
            "--backend",
            "whisper-cpp",
            "--no-diarize",
            "--no-persist",
            // Single window so the stub only fires twice (fast + quality).
            "--speculative",
            "--fast-model",
            "tiny",
            "--quality-model",
            "large",
            "--speculative-window-ms",
            "10000",
            "--speculative-overlap-ms",
            "0",
        ])
        .env("FRANKEN_WHISPER_STATE_DIR", &state_root)
        .env("FRANKEN_WHISPER_WHISPER_CPP_BIN", &stub_bin)
        .env("FRANKEN_WHISPER_NATIVE_EXECUTION", "0")
        .env("FRANKEN_WHISPER_BRIDGE_NATIVE_RECOVERY", "0")
        .output()
        .expect("robot transcribe should execute");

    let stdout = String::from_utf8(output.stdout).expect("stdout utf-8");
    let stderr = String::from_utf8(output.stderr).expect("stderr utf-8");
    assert!(
        output.status.success(),
        "speculative dispatch should succeed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    let events: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.is_empty() && trimmed.starts_with('{')
        })
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    assert!(
        events.len() >= 5,
        "expected run_start + several stage lines + run_complete, got {} events",
        events.len()
    );

    // run_start MUST be first; run_complete MUST be last.
    assert_eq!(events.first().expect("first")["event"], "run_start");
    let last = events.last().expect("last");
    assert_eq!(
        last["event"], "run_complete",
        "speculative pipeline should land on run_complete (got {last:?})"
    );

    // The speculative backend stage MUST advertise execution_mode=speculative_streaming
    // and report the sticky resolved backend (so callers don't see "auto" leaking
    // through when the actual engine choice has been fixed for the run).
    let backend_ok = events
        .iter()
        .find(|event| event["event"] == "stage" && event["code"] == "backend.ok")
        .expect("backend.ok must be emitted");
    assert_eq!(
        backend_ok["payload"]["execution_mode"], "speculative_streaming",
        "backend.ok must mark execution_mode=speculative_streaming when --speculative was set"
    );
    assert_eq!(
        backend_ok["payload"]["backend"], "whisper_cpp",
        "backend.ok must report the resolved sticky backend (whisper_cpp), not e.g. \"auto\""
    );
    // bd-spec-cli.replay: ReplayEnvelope-feeding fields must be present so the
    // run-wide ReplayEnvelope is fully populated even for speculative runs.
    assert!(
        backend_ok["payload"]["backend_identity"].is_string(),
        "backend.ok must carry a string backend_identity for the ReplayEnvelope"
    );
    let backend_start = events
        .iter()
        .find(|event| event["event"] == "stage" && event["code"] == "backend.start")
        .expect("backend.start must be emitted");
    assert_eq!(
        backend_start["payload"]["resolved_backend"], "whisper_cpp",
        "backend.start must announce the sticky-resolved backend kind"
    );

    // The speculation pipeline MUST emit at least one transcript.partial event
    // (the fast-lane result) and at least one terminal confirmation or correction
    // (transcript.confirm OR transcript.correct, depending on drift). With the
    // whisper.cpp stub returning identical text for both lanes, drift is zero and
    // we expect a confirm.
    let partial_count = events
        .iter()
        .filter(|event| event["code"] == "transcript.partial")
        .count();
    assert!(
        partial_count >= 1,
        "expected at least one transcript.partial, got 0 (events: {})",
        events
            .iter()
            .map(|event| event["code"].as_str().unwrap_or("?"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let resolved_count = events
        .iter()
        .filter(|event| {
            event["code"] == "transcript.confirm" || event["code"] == "transcript.correct"
        })
        .count();
    assert!(
        resolved_count >= 1,
        "expected at least one transcript.confirm or transcript.correct, got 0"
    );

    // The aggregate speculation_stats event MUST be emitted at the end of the
    // speculative backend phase with the contract-required field set.
    let stats = events
        .iter()
        .find(|event| event["code"] == "transcript.speculation_stats")
        .expect("speculation_stats must be emitted at end of speculative backend phase");
    let stats_payload = &stats["payload"];
    for field in [
        "windows_processed",
        "corrections_emitted",
        "confirmations_emitted",
        "correction_rate",
        "current_window_size_ms",
        "mean_drift_wer",
    ] {
        assert!(
            stats_payload.get(field).is_some(),
            "speculation_stats payload missing required field `{field}`"
        );
    }
    assert_eq!(
        stats_payload["windows_processed"], 1,
        "single-window run should report exactly one window processed"
    );

    // The final transcript must reflect the merged segments from the
    // speculation pipeline (the stub emits "stub transcript").
    assert_eq!(
        last["transcript"], "stub transcript",
        "merged transcript should equal the stub's canned text"
    );
}

#[cfg(unix)]
#[test]
fn speculative_cli_without_flag_uses_single_backend_dispatch() {
    // Sanity counterpart: without `--speculative`, the same input should NOT
    // emit any transcript.partial / speculation_stats events. This guards
    // against future regressions in the dispatch branch that would otherwise
    // silently route every transcription through the speculative path.

    if !ffmpeg_available() {
        return;
    }

    let dir = tempdir().expect("tempdir");
    let state_root = dir.path().join("state");
    std::fs::create_dir_all(&state_root).expect("state root");

    let input_wav = dir.path().join("plain_input.wav");
    generate_voiced_wav(&input_wav);

    let bin_dir = dir.path().join("bin");
    std::fs::create_dir_all(&bin_dir).expect("bin dir");
    let stub_bin = write_whisper_cpp_stub_binary(&bin_dir);

    let output = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"))
        .args([
            "robot",
            "run",
            "--input",
            input_wav.to_str().expect("utf-8"),
            "--backend",
            "whisper-cpp",
            "--no-diarize",
            "--no-persist",
        ])
        .env("FRANKEN_WHISPER_STATE_DIR", &state_root)
        .env("FRANKEN_WHISPER_WHISPER_CPP_BIN", &stub_bin)
        .env("FRANKEN_WHISPER_NATIVE_EXECUTION", "0")
        .env("FRANKEN_WHISPER_BRIDGE_NATIVE_RECOVERY", "0")
        .output()
        .expect("robot transcribe should execute");

    assert!(
        output.status.success(),
        "non-speculative transcribe should succeed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8(output.stdout).expect("stdout utf-8");
    let events: Vec<serde_json::Value> = stdout
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.is_empty() && trimmed.starts_with('{')
        })
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    let backend_ok = events
        .iter()
        .find(|event| event["event"] == "stage" && event["code"] == "backend.ok")
        .expect("backend.ok must be emitted");
    assert_ne!(
        backend_ok["payload"]["execution_mode"], "speculative_streaming",
        "non-speculative request must NOT take the speculative dispatch branch"
    );

    let partial_count = events
        .iter()
        .filter(|event| event["code"] == "transcript.partial")
        .count();
    assert_eq!(
        partial_count, 0,
        "non-speculative request should emit zero transcript.partial events"
    );
    let stats_count = events
        .iter()
        .filter(|event| event["code"] == "transcript.speculation_stats")
        .count();
    assert_eq!(
        stats_count, 0,
        "non-speculative request should emit zero speculation_stats events"
    );
}
