//! End-to-end integration test for the `youtube` subcommand, driven entirely
//! by a hermetic `yt-dlp` stub (no network).
//!
//! The stub (`tests/fixtures/youtube/ytdlp_stub.sh`) answers `--version`,
//! `--flat-playlist --dump-json`, `-j` metadata, and best-audio downloads
//! (copying the repo `jfk.wav` as the "downloaded" track). Transcription
//! still needs a real model, so these tests are GATED on `tiny.en` being
//! resolvable and force the native engine (`FRANKEN_WHISPER_NATIVE_EXECUTION=1`,
//! rollout `sole`, all bridge bins missing) — exactly like the other gated
//! e2e tests. When the model is absent they print a `SKIP` line and pass.

use std::path::Path;
use std::process::{Command as ProcessCommand, Stdio};

fn model_present() -> bool {
    franken_whisper::native_engine::find_model_file("tiny.en").is_some()
}

fn stub_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/youtube/ytdlp_stub.sh")
}

struct CliRun {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

/// Run `franken_whisper youtube <args>` with the stub wired in and the native
/// engine forced (no network, no bridge binaries).
fn run_youtube(args: &[&str], state_root: &Path) -> CliRun {
    let mut cmd = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"));
    cmd.arg("youtube");
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
    if !model_present() {
        eprintln!("SKIP: ggml-tiny.en.bin not resolvable (see scripts/fetch_test_models.sh)");
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

    // The id suffix appears in the names; both transcripts contain real text.
    let names: Vec<String> = mds
        .iter()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        names.iter().any(|n| n.contains("[AAAAAAAAAAA]")),
        "names missing first id: {names:?}"
    );
    assert!(
        names.iter().any(|n| n.contains("[BBBBBBBBBBB]")),
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
    // The JSON carries per-utterance records.
    for j in &jsons {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(j.path()).expect("read json")).expect("parse");
        assert!(v["video"]["id"].is_string());
        assert!(v["utterances"].is_array());
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
fn gated_youtube_private_video_fails_gracefully_and_records_manifest() {
    if !model_present() {
        eprintln!("SKIP: ggml-tiny.en.bin not resolvable");
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("out");

    let mut cmd = ProcessCommand::new(env!("CARGO_BIN_EXE_franken_whisper"));
    cmd.arg("youtube")
        .arg("https://www.youtube.com/watch?v=PRIVATE00001")
        .arg("--output-dir")
        .arg(out.to_str().unwrap())
        .arg("--model")
        .arg("tiny.en")
        .env("FRANKEN_WHISPER_STATE_DIR", dir.path())
        .env("FRANKEN_WHISPER_YTDLP_BIN", stub_path())
        .env("STUB_FAIL_MODE", "private")
        .env("FRANKEN_WHISPER_NATIVE_EXECUTION", "1")
        .env("FRANKEN_WHISPER_NATIVE_ROLLOUT_STAGE", "sole")
        .env("FRANKEN_WHISPER_WHISPER_CPP_BIN", "/nonexistent-bridge")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = cmd.output().expect("spawn");
    // A run where every video failed exits non-zero (private => resolve fails
    // at metadata before any work).
    assert!(
        !output.status.success(),
        "private-video run should exit non-zero"
    );
}
