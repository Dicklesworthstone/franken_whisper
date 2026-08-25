//! End-to-end integration test for the `youtube` subcommand, driven entirely
//! by a hermetic `yt-dlp` stub (no network).
//!
//! The stub (`tests/fixtures/youtube/ytdlp_stub.sh`) answers `--version`,
//! `--flat-playlist --dump-json`, `-j` metadata, and best-audio downloads
//! (copying the repo `jfk.wav` as the "downloaded" track). Transcription
//! still needs a real model, so these tests are GATED on `tiny.en` being
//! resolvable and force the native engine (`FRANKEN_WHISPER_NATIVE_EXECUTION=1`,
//! rollout `sole`, all bridge bins missing) — exactly like the other gated
//! e2e tests. When the model is absent they print a `SKIP` line and pass unless
//! `FW_REQUIRE_YOUTUBE_E2E=1` selects the strict closure lane, where absence is
//! a hard failure rather than vacuous credit.

use std::path::Path;
use std::process::{Command as ProcessCommand, Stdio};

fn model_present() -> bool {
    franken_whisper::native_engine::find_model_file("tiny.en").is_some()
}

fn model_present_or_skip() -> bool {
    if model_present() {
        return true;
    }
    assert_ne!(
        std::env::var("FW_REQUIRE_YOUTUBE_E2E").as_deref(),
        Ok("1"),
        "FW_REQUIRE_YOUTUBE_E2E=1 but ggml-tiny.en.bin is not resolvable; run scripts/fetch_test_models.sh"
    );
    eprintln!("SKIP: ggml-tiny.en.bin not resolvable (see scripts/fetch_test_models.sh)");
    false
}

fn stub_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/youtube/ytdlp_stub.sh")
}

struct CliRun {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

/// Run `franken_whisper youtube run <args>` with the stub wired in and the native
/// engine forced (no network, no bridge binaries).
fn run_youtube(args: &[&str], state_root: &Path) -> CliRun {
    let mut cmd = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"));
    cmd.arg("youtube");
    cmd.arg("run");
    cmd.args(args);
    cmd.env("FRANKEN_WHISPER_STATE_DIR", state_root);
    cmd.env("FRANKEN_WHISPER_YTDLP_BIN", stub_path());
    cmd.env("FRANKEN_WHISPER_NATIVE_EXECUTION", "1");
    cmd.env("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole");
    cmd.env("FRANKEN_WHISPER_WHISPER_CPP_BIN", "/nonexistent-bridge");
    cmd.env("FRANKEN_WHISPER_INSANELY_FAST_BIN", "/nonexistent-bridge");
    cmd.env("FRANKEN_WHISPER_PYTHON_BIN", "/nonexistent-bridge");
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());
    let output = cmd.output().expect("spawn franken_whisper youtube");
    CliRun {
        status: output.status,
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

#[test]
fn gated_youtube_batch_file_produces_md_and_json_then_resumes() {
    if !model_present_or_skip() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("out");
    let batch = dir.path().join("urls.txt");
    // Two distinct watch URLs -> the stub derives distinct ids -> 2 videos.
    std::fs::write(
        &batch,
        "# my videos\nhttps://www.youtube.com/watch?v=AAAAAAAAAAA\n\nhttps://youtu.be/BBBBBBBBBBB\n",
    )
    .expect("write batch file");

    let run = run_youtube(
        &[
            "--batch-file",
            batch.to_str().unwrap(),
            "--output-dir",
            out.to_str().unwrap(),
            "--model",
            "tiny.en",
            "--json-summary",
        ],
        dir.path(),
    );
    assert!(
        run.status.success(),
        "youtube run failed: stdout={} stderr={}",
        run.stdout,
        run.stderr
    );

    // Two markdown + two json outputs, named with the derived ids.
    let mds: Vec<_> = std::fs::read_dir(&out)
        .expect("read out dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("md"))
        .collect();
    let jsons: Vec<_> = std::fs::read_dir(&out)
        .expect("read out dir")
        .filter_map(Result::ok)
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            // Exclude the .fw_youtube_manifest.json (a dotfile).
            !name.starts_with('.') && name.ends_with(".json")
        })
        .collect();
    assert_eq!(mds.len(), 2, "expected 2 markdown files, got {mds:?}");
    assert_eq!(jsons.len(), 2, "expected 2 json files");

    // The default slug style preserves each case-sensitive id as the final
    // filename component; both transcripts contain real text.
    let names: Vec<String> = mds
        .iter()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        names.iter().any(|n| n.ends_with("_AAAAAAAAAAA.md")),
        "names missing first id: {names:?}"
    );
    assert!(
        names.iter().any(|n| n.ends_with("_BBBBBBBBBBB.md")),
        "names missing second id: {names:?}"
    );
    for md in &mds {
        let body = std::fs::read_to_string(md.path()).expect("read md");
        assert!(body.starts_with("# "), "md missing H1 header");
        assert!(
            body.contains("youtu.be/"),
            "md missing deep link in {:?}",
            md.path()
        );
    }
    // The JSON carries per-utterance records and the exact native-v2 decoder
    // window evidence; JFK speech must not produce a vacuous empty array.
    for j in &jsons {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(j.path()).expect("read json")).expect("parse");
        assert!(v["video"]["id"].is_string());
        assert!(v["utterances"].is_array());
        let windows = v["windows"].as_array().expect("native windows array");
        assert!(
            !windows.is_empty(),
            "speech sidecar must carry native windows"
        );
        for window in windows {
            assert!(window["window_offset_sec"].as_f64().is_some());
            assert!(window["tokens"].as_u64().is_some());
            assert!(window["avg_logprob"].as_f64().is_some());
            let no_speech = window["no_speech_prob"]
                .as_f64()
                .expect("numeric no_speech_prob");
            assert!((0.0..=1.0).contains(&no_speech));
        }
    }

    // The summary reports 2 done, 0 failed.
    let summary: serde_json::Value = serde_json::from_str(&run.stdout).expect("summary json");
    assert_eq!(summary["done"].as_array().map(Vec::len), Some(2));
    assert_eq!(summary["failed"].as_array().map(Vec::len), Some(0));

    // Manifest exists.
    assert!(out.join(".fw_youtube_manifest.json").exists());

    // ── Idempotent re-run: both videos already done -> both skipped. ──
    let rerun = run_youtube(
        &[
            "--batch-file",
            batch.to_str().unwrap(),
            "--output-dir",
            out.to_str().unwrap(),
            "--model",
            "tiny.en",
            "--json-summary",
        ],
        dir.path(),
    );
    assert!(
        rerun.status.success(),
        "resume run failed: {}",
        rerun.stderr
    );
    let s2: serde_json::Value = serde_json::from_str(&rerun.stdout).expect("rerun summary");
    assert_eq!(
        s2["skipped"].as_array().map(Vec::len),
        Some(2),
        "resume should skip both already-done videos"
    );
    assert_eq!(s2["done"].as_array().map(Vec::len), Some(0));
}

#[test]
fn youtube_private_video_fails_gracefully_without_model() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("out");

    let mut cmd = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"));
    cmd.arg("youtube")
        .arg("run")
        .arg("https://www.youtube.com/watch?v=PRIVATE00001&fw_stub_fail=private")
        .arg("--output-dir")
        .arg(out.to_str().unwrap())
        .arg("--model")
        .arg("tiny.en")
        .arg("--robot")
        .env("FRANKEN_WHISPER_STATE_DIR", dir.path())
        .env("FRANKEN_WHISPER_YTDLP_BIN", stub_path())
        .env("FRANKEN_WHISPER_NATIVE_EXECUTION", "1")
        .env("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole")
        .env("FRANKEN_WHISPER_WHISPER_CPP_BIN", "/nonexistent-bridge")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = cmd.output().expect("spawn");
    assert!(
        !output.status.success(),
        "private-video run should exit non-zero"
    );
    let events: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .expect("robot stdout utf8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("robot NDJSON line"))
        .collect();
    let failed: Vec<_> = events
        .iter()
        .filter(|event| event["event"] == "youtube.failed")
        .collect();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0]["id"], "PRIVATE00001");
    assert!(
        failed[0]["error"]
            .as_str()
            .expect("private error")
            .contains("private"),
        "the failure must come from the private-video contract"
    );
    assert_eq!(
        events.last().expect("terminal")["event"],
        "youtube.run_complete"
    );
    assert_eq!(events.last().expect("terminal")["cancelled"], false);

    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(out.join(".fw_youtube_manifest.json")).expect("manifest"),
    )
    .expect("parse manifest");
    assert_eq!(
        manifest["entries"]["PRIVATE00001"]["state"]["state"],
        "failed"
    );
    assert_eq!(manifest["entries"]["PRIVATE00001"]["state"]["attempts"], 1);
}

#[test]
fn youtube_robot_abort_on_transcription_failure_exits_nonzero_and_persists_failure() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("out");
    let missing_model = dir.path().join("definitely-missing-model.bin");
    let run = run_youtube(
        &[
            "https://www.youtube.com/watch?v=MISSING0001",
            "https://www.youtube.com/watch?v=AFTERFAIL01",
            "--output-dir",
            out.to_str().expect("output path"),
            "--model",
            missing_model.to_str().expect("model path"),
            "--no-diarize",
            "--abort-on-error",
            "--batch-size",
            "1",
            "--robot",
        ],
        dir.path(),
    );
    assert!(!run.status.success(), "missing model must fail the run");

    let events: Vec<serde_json::Value> = run
        .stdout
        .lines()
        .map(|line| serde_json::from_str(line).expect("robot NDJSON line"))
        .collect();
    let failed: Vec<_> = events
        .iter()
        .filter(|event| event["event"] == "youtube.failed")
        .collect();
    assert_eq!(failed.len(), 1, "exactly one per-video failure event");
    assert_eq!(failed[0]["id"], "MISSING0001");
    assert_eq!(failed[0]["title"], "Stub Title MISSING0001");
    assert_eq!(failed[0]["attempts"], 1);
    assert!(
        failed[0]["error"]
            .as_str()
            .expect("failure message")
            .contains("model")
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event["event"] == "youtube.done")
            .count(),
        0
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event["id"] == "AFTERFAIL01")
            .filter(|event| event["event"] != "youtube.discovered")
            .count(),
        0,
        "the later wave must remain undispatched after transcription fails"
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event["event"] == "youtube.run_complete")
            .count(),
        1
    );
    assert_eq!(
        events.last().expect("terminal event")["event"],
        "youtube.run_complete"
    );
    assert_eq!(
        events.last().expect("terminal event")["cancelled"],
        false,
        "--abort-on-error must not be misclassified as Ctrl+C cancellation"
    );

    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(out.join(".fw_youtube_manifest.json")).expect("manifest"),
    )
    .expect("parse manifest");
    assert_eq!(
        manifest["entries"]["MISSING0001"]["state"]["state"],
        "failed"
    );
    assert_eq!(manifest["entries"]["MISSING0001"]["state"]["attempts"], 1);
    assert_eq!(
        manifest["entries"]["AFTERFAIL01"]["state"]["state"],
        "pending"
    );
}

#[cfg(unix)]
#[test]
fn youtube_real_sigint_exits_130() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("out");
    let call_log = dir.path().join("ytdlp-calls.log");
    let mut command = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"));
    command
        .arg("youtube")
        .arg("run")
        .arg("https://www.youtube.com/watch?v=SIGINT00001&fw_stub_fail=429")
        .arg("--output-dir")
        .arg(&out)
        .arg("--model")
        .arg(dir.path().join("unused-missing-model.bin"))
        .arg("--robot")
        .env("FRANKEN_WHISPER_STATE_DIR", dir.path())
        .env("FRANKEN_WHISPER_YTDLP_BIN", stub_path())
        .env("FRANKEN_WHISPER_NATIVE_EXECUTION", "1")
        .env("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole")
        .env("FRANKEN_WHISPER_WHISPER_CPP_BIN", "/nonexistent-bridge")
        .env("FRANKEN_WHISPER_INSANELY_FAST_BIN", "/nonexistent-bridge")
        .env("FRANKEN_WHISPER_PYTHON_BIN", "/nonexistent-bridge")
        .env("STUB_CALL_LOG", &call_log)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("spawn franken_whisper youtube");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let worker_is_active = std::fs::read_to_string(&call_log)
            .is_ok_and(|calls| calls.contains("fw_stub_fail=429"));
        if worker_is_active {
            break;
        }
        if let Some(status) = child.try_wait().expect("inspect youtube CLI") {
            let output = child
                .wait_with_output()
                .expect("collect early youtube exit");
            panic!(
                "youtube CLI exited as {status} before the signal probe became active; stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        if std::time::Instant::now() >= deadline {
            child.kill().expect("terminate timed-out youtube CLI");
            let output = child
                .wait_with_output()
                .expect("collect timed-out youtube CLI");
            panic!(
                "yt-dlp worker path did not become active before SIGINT deadline; stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let signal_status = ProcessCommand::new("kill")
        .arg("-INT")
        .arg(child.id().to_string())
        .status()
        .expect("invoke kill -INT");
    assert!(signal_status.success(), "deliver SIGINT to youtube CLI");

    let output = child
        .wait_with_output()
        .expect("reap interrupted youtube CLI");
    assert_eq!(
        output.status.code(),
        Some(130),
        "real SIGINT must retain the stable interrupted exit code; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[test]
fn youtube_real_sigint_exits_130() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("out");
    let call_log = dir.path().join("ytdlp-calls.log");
    let mut command = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"));
    command
        .arg("youtube")
        .arg("run")
        .arg("https://www.youtube.com/watch?v=SIGINT00001&fw_stub_fail=429")
        .arg("--output-dir")
        .arg(&out)
        .arg("--model")
        .arg(dir.path().join("unused-missing-model.bin"))
        .arg("--robot")
        .env("FRANKEN_WHISPER_STATE_DIR", dir.path())
        .env("FRANKEN_WHISPER_YTDLP_BIN", stub_path())
        .env("FRANKEN_WHISPER_NATIVE_EXECUTION", "1")
        .env("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole")
        .env("FRANKEN_WHISPER_WHISPER_CPP_BIN", "/nonexistent-bridge")
        .env("FRANKEN_WHISPER_INSANELY_FAST_BIN", "/nonexistent-bridge")
        .env("FRANKEN_WHISPER_PYTHON_BIN", "/nonexistent-bridge")
        .env("STUB_CALL_LOG", &call_log)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("spawn franken_whisper youtube");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let worker_is_active = std::fs::read_to_string(&call_log)
            .is_ok_and(|calls| calls.contains("fw_stub_fail=429"));
        if worker_is_active {
            break;
        }
        if let Some(status) = child.try_wait().expect("inspect youtube CLI") {
            let output = child.wait_with_output().expect("collect early youtube exit");
            panic!(
                "youtube CLI exited as {status} before the signal probe became active; stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        if std::time::Instant::now() >= deadline {
            child.kill().expect("terminate timed-out youtube CLI");
            let output = child.wait_with_output().expect("collect timed-out youtube CLI");
            panic!(
                "yt-dlp worker path did not become active before SIGINT deadline; stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let signal_status = ProcessCommand::new("kill")
        .arg("-INT")
        .arg(child.id().to_string())
        .status()
        .expect("invoke kill -INT");
    assert!(signal_status.success(), "deliver SIGINT to youtube CLI");

    let output = child.wait_with_output().expect("reap interrupted youtube CLI");
    assert_eq!(
        output.status.code(),
        Some(130),
        "real SIGINT must retain the stable interrupted exit code; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
