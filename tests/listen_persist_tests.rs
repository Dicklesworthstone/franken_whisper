//! Acceptance tests for utterance-granular live-session persistence
//! (bd-rt-persist-a66y): crash-durable listen runs in SQLite.
//!
//! All model-backed tests follow the standard model-gated pattern: skip with
//! a printed prerequisite line instead of fabricating a pass. Temp DBs are
//! created under the system temp dir and deliberately left in place
//! (repo rule: no deletions without explicit operator approval).

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use franken_whisper::model::BackendKind;
use franken_whisper::storage::RunStore;

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

fn unique_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "fw_persist_test_{label}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn require_tiny_en() -> bool {
    static READY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*READY.get_or_init(|| {
        franken_whisper::model_distribution::resolve_cached_fast_lane_with_cancel(
            franken_whisper::model_distribution::FastLaneModel::TinyEn,
            || false,
        )
        .is_ok()
    }) {
        eprintln!("SKIP persist test: tiny.en package missing (`fw pull tiny-en`)");
        return false;
    }
    true
}

/// Decode jfk.wav exactly like `listen::load_replay_wav` does (f32 samples)
/// so the streamed session-PCM hash can be reproduced independently.
fn decode_jfk_f32() -> Vec<f32> {
    use hound::SampleFormat;
    let mut reader = hound::WavReader::open(jfk_wav()).expect("open jfk.wav");
    let spec = reader.spec();
    match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.expect("float sample"))
            .collect(),
        SampleFormat::Int => {
            let denom = f32::from(i16::MAX) + 1.0;
            reader
                .samples::<i16>()
                .map(|s| f32::from(s.expect("int sample")) / denom)
                .collect()
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

const LISTEN_BASE_ARGS: &[&str] = &[
    "robot",
    "listen",
    "--source",
    "file-replay",
    "--fast-model",
    "tiny.en",
    "--language",
    "en",
    "--quality-model",
    "none",
];

#[test]
fn persist_file_replay_rows_request_json_and_envelope() {
    if !require_tiny_en() {
        return;
    }
    let dir = unique_dir("e2e_rows");
    let db = dir.join("runs.sqlite3");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_fw"));
    cmd.args(LISTEN_BASE_ARGS)
        .arg("--input")
        .arg(jfk_wav())
        .arg("--db")
        .arg(&db)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = cmd.spawn().expect("spawn fw robot listen");
    let stdout = child.stdout.take().expect("piped stdout");
    let mut run_id = String::new();
    let mut delta_count_stdout = 0u64;
    let mut saw_final_stats = false;
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        let Ok(event) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        match event.get("event").and_then(|v| v.as_str()) {
            Some("listen.session_start") => {
                run_id = event
                    .get("run_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned();
            }
            Some("transcript.delta") => delta_count_stdout += 1,
            Some("listen.session_stats")
                if event.get("final") == Some(&serde_json::json!(true)) =>
            {
                saw_final_stats = true;
                break;
            }
            _ => {}
        }
    }
    let code = child.wait().expect("wait").code().unwrap_or(-1);
    assert_eq!(code, 0, "listen session must exit 0");
    assert!(saw_final_stats, "stream must reach its final stats event");
    assert!(!run_id.is_empty(), "session_start must carry the run id");

    // Independent hash of the fixture PCM (16 kHz mono → resampler is a
    // pass-through for this fixture, so raw decoded == session PCM).
    let samples = decode_jfk_f32();
    let mut pcm_bytes = Vec::with_capacity(samples.len() * 4);
    for sample in &samples {
        pcm_bytes.extend_from_slice(&sample.to_le_bytes());
    }
    let expected_hash = sha256_hex(&pcm_bytes);

    // Downstream-consumer view via the public store API.
    let store = RunStore::open(&db).expect("reopen db");
    let summaries = store.list_recent_runs(10).expect("list runs");
    let listen_run = summaries
        .iter()
        .find(|s| s.backend == BackendKind::NativeListen && s.run_id == run_id);
    assert!(listen_run.is_some(), "listen run row must be listed");
    let summary = listen_run.expect("checked above");
    assert_ne!(
        summary.started_at_rfc3339, summary.finished_at_rfc3339,
        "clean close writes the true end time"
    );
    assert!(
        summary.transcript_preview.contains("fellow"),
        "transcript joins utterances: {:?}",
        summary.transcript_preview
    );

    let request_json = store
        .stored_request_json(&run_id)
        .expect("request reader")
        .expect("row exists");
    assert!(
        request_json.contains("\"fast_model\":\"tiny.en\""),
        "config round-trips: {request_json}"
    );
    let replay_raw = store
        .stored_replay_json(&run_id)
        .expect("replay reader")
        .expect("row exists");
    let replay: serde_json::Value = serde_json::from_str(&replay_raw).expect("valid json");
    assert_eq!(replay["kind"], "live-session");
    assert_eq!(
        replay["pcm_sha256"], expected_hash,
        "streamed PCM hash must match independently computed fixture hash"
    );
    assert_eq!(
        replay["live_note"],
        "audio not retained; hash is an integrity fingerprint, not a replayable reference"
    );

    // Event trail + segment granularity via the details loader (the same
    // path `fw runs show` uses). Deltas persisted 1:1; partials NOT.
    let details = store
        .load_run_details(&run_id)
        .expect("details")
        .expect("row");
    // The segments TABLE is the authoritative utterance store for live
    // runs; result_json.segments is an empty summary by design.
    let persisted_segments = store.listen_segment_count(&run_id).expect("count");
    assert_eq!(
        usize::try_from(persisted_segments).unwrap_or(0),
        delta_count_stdout as usize,
        "every committed delta must be a durable segment row"
    );
    let codes: Vec<&str> = details.events.iter().map(|e| e.code.as_str()).collect();
    assert!(codes.contains(&"listen.session_start"));
    assert!(codes.contains(&"listen.session_stats"));
    assert!(
        !codes.contains(&"transcript.partial"),
        "partials are not durable (spec decision)"
    );
}

#[test]
fn kill_nine_mid_session_keeps_flushed_utterances_and_db_intact() {
    if !require_tiny_en() {
        return;
    }
    let dir = unique_dir("kill9");
    let db = dir.join("runs.sqlite3");

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_fw"));
    cmd.args(LISTEN_BASE_ARGS)
        .arg("--input")
        .arg(jfk_wav())
        .arg("--realtime-pace")
        .arg("--db")
        .arg(&db)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = cmd.spawn().expect("spawn");
    let stdout = child.stdout.take().expect("piped stdout");
    let mut utterance_ends_seen = 0u32;
    let mut run_id = String::new();
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if line.contains("\"listen.session_start\"")
            && let Ok(event) = serde_json::from_str::<serde_json::Value>(&line)
        {
            run_id = event
                .get("run_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned();
        }
        if line.contains("\"utterance_end\"") {
            utterance_ends_seen += 1;
            if utterance_ends_seen >= 2 {
                break;
            }
        }
    }
    assert!(
        utterance_ends_seen >= 2,
        "need two closed utterances before the kill"
    );
    // The durability flush commits just AFTER the utterance_end hits
    // stdout; give it a beat so the kill lands past the second flush.
    std::thread::sleep(Duration::from_millis(1000));
    child.kill().expect("SIGKILL mid-session");
    let _ = child.wait();

    // Give the OS a beat; WAL recovery happens on reopen.
    std::thread::sleep(Duration::from_millis(300));

    let store = RunStore::open(&db).expect("reopen after SIGKILL");
    let integrity = store.query_integrity_check();
    assert_eq!(integrity, "ok", "database must survive SIGKILL uncorrupted");
    let persisted_segments = store.listen_segment_count(&run_id).expect("count");
    assert!(
        persisted_segments >= 2,
        "flushed deltas must be durable after SIGKILL (segments={persisted_segments})"
    );
    let details = store
        .load_run_details(&run_id)
        .expect("details")
        .expect("row");
    let ends = details
        .events
        .iter()
        .filter(|e| e.code == "utterance_end")
        .count();
    assert!(ends >= 2, "two utterance_end events durable (ends={ends})");
}

#[test]
fn concurrent_batch_and_live_share_one_database() {
    if !require_tiny_en() {
        return;
    }
    let dir = unique_dir("concurrent");
    let db = dir.join("runs.sqlite3");
    let db_for_live = db.clone();
    let live = std::thread::spawn(move || {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_fw"));
        cmd.args(LISTEN_BASE_ARGS)
            .arg("--input")
            .arg(jfk_wav())
            .arg("--realtime-pace")
            .arg("--db")
            .arg(&db_for_live)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        cmd.status().expect("live status").code().unwrap_or(-1)
    });

    let batch_code = Command::new(env!("CARGO_BIN_EXE_fw"))
        .args([
            "transcribe",
            "--input",
            jfk_wav().to_str().expect("utf8 path"),
            "--model",
            "tiny.en",
            "--no-diarize",
            "--db",
            db.to_str().expect("utf8 db"),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("batch status")
        .code()
        .unwrap_or(-1);
    assert_eq!(
        batch_code, 0,
        "batch run against the shared db must succeed"
    );
    assert_eq!(
        live.join().expect("join live"),
        0,
        "live session must succeed concurrently"
    );

    let store = RunStore::open(&db).expect("open shared db");
    let summaries = store.list_recent_runs(20).expect("list runs");
    let has_listen = summaries
        .iter()
        .any(|s| s.backend == BackendKind::NativeListen);
    let has_batch = summaries
        .iter()
        .any(|s| s.backend != BackendKind::NativeListen);
    assert!(has_listen, "live run persisted alongside batch");
    assert!(has_batch, "batch run persisted alongside live");
}

#[test]
fn no_persist_flag_touches_no_database_file() {
    if !require_tiny_en() {
        return;
    }
    let dir = unique_dir("no_persist");
    let db = dir.join("must_stay_absent.sqlite3");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_fw"));
    cmd.args(LISTEN_BASE_ARGS)
        .arg("--input")
        .arg(jfk_wav())
        .arg("--no-persist")
        .arg("--db")
        .arg(&db)
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = cmd.spawn().expect("spawn");
    // Drain stdout to completion so the child is not blocked on a full pipe.
    if let Some(stdout) = child.stdout.take() {
        for _ in BufReader::new(stdout).lines() {}
    }
    let code = child.wait().expect("wait").code().unwrap_or(-1);
    assert_eq!(code, 0, "session must still succeed without persistence");
    assert!(
        !db.exists(),
        "--no-persist must not create the database file at {}",
        db.display()
    );
}
