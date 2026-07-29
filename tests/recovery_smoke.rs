use std::path::PathBuf;
use std::{fs, io::Write};

use serde_json::json;
use tempfile::tempdir;

use franken_whisper::model::{
    BackendKind, BackendParams, InputSource, RunEvent, RunReport, TranscribeRequest,
    TranscriptionResult, TranscriptionSegment,
};
use franken_whisper::storage::RunStore;
use franken_whisper::sync::{self, ConflictPolicy};

fn fixture_report(id: &str, db_path: &std::path::Path) -> RunReport {
    RunReport {
        run_id: id.to_owned(),
        trace_id: "00000000000000000000000000000000".to_owned(),
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
            backend_params: BackendParams::default(),
        },
        result: TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: "recovery smoke transcript".to_owned(),
            language: Some("en".to_owned()),
            segments: vec![TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.0),
                text: "hello".to_owned(),
                speaker: None,
                confidence: Some(0.91),
            }],
            acceleration: None,
            diarization: None,
            raw_output: json!({"test": true}),
            artifact_paths: vec![],
        },
        events: vec![RunEvent {
            seq: 1,
            ts_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "done".to_owned(),
            payload: json!({"segments": 1}),
        }],
        warnings: vec![],
        evidence: vec![],
        replay: franken_whisper::model::ReplayEnvelope {
            input_content_hash: Some("recovery-input-hash".to_owned()),
            backend_identity: Some("whisper-cli".to_owned()),
            backend_version: Some("whisper 1.2.3".to_owned()),
            output_payload_hash: Some("recovery-output-hash".to_owned()),
        },
    }
}

fn mutate_export_runs_transcript(export_dir: &std::path::Path, transcript: &str) {
    let runs_path = export_dir.join("runs.jsonl");
    let first_line = fs::read_to_string(&runs_path)
        .expect("runs.jsonl should be readable")
        .lines()
        .next()
        .expect("runs snapshot should include at least one row")
        .to_owned();

    let mut mutated: serde_json::Value =
        serde_json::from_str(&first_line).expect("valid jsonl row");
    mutated["transcript"] = json!(transcript);
    if let Some(result_json) = mutated.get("result_json").and_then(|v| v.as_str()) {
        let mut parsed_result: serde_json::Value =
            serde_json::from_str(result_json).expect("result_json should parse");
        parsed_result["transcript"] = json!(transcript);
        mutated["result_json"] =
            json!(serde_json::to_string(&parsed_result).expect("serialize result_json"));
    }
    fs::write(
        &runs_path,
        format!(
            "{}\n",
            serde_json::to_string(&mutated).expect("serialize row")
        ),
    )
    .expect("write runs snapshot");

    let manifest_path = export_dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest"))
            .expect("manifest json");
    let checksum = sha256_file(&runs_path);
    manifest["checksums"]["runs_jsonl_sha256"] = json!(checksum);
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("manifest serialize"),
    )
    .expect("manifest rewrite");
}

fn sha256_file(path: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};
    let bytes = fs::read(path).expect("read file");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[test]
fn recovery_smoke_round_trip_snapshot_restore() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("target.sqlite3");
    let export_dir = dir.path().join("snapshot");
    let state_root = dir.path().join("state");

    let source_store = RunStore::open(&source_db).expect("source store");
    source_store
        .persist_report(&fixture_report("recovery-1", &source_db))
        .expect("persist");

    sync::export(&source_db, &export_dir, &state_root).expect("export");
    sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject).expect("import");

    let target_store = RunStore::open(&target_db).expect("target store");
    let restored = target_store
        .load_run_details("recovery-1")
        .expect("query")
        .expect("restored row");
    assert_eq!(restored.transcript, "recovery smoke transcript");
    assert_eq!(restored.events.len(), 1);
}

#[test]
fn recovery_smoke_failed_import_preserves_existing_db_state() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("db.sqlite3");
    let export_dir = dir.path().join("snapshot");
    let state_root = dir.path().join("state");

    let store = RunStore::open(&db_path).expect("store");
    store
        .persist_report(&fixture_report("recovery-2", &db_path))
        .expect("persist");

    sync::export(&db_path, &export_dir, &state_root).expect("export");

    // Corrupt one channel so import fails deterministically on checksum.
    std::fs::write(export_dir.join("events.jsonl"), "tampered\n").expect("tamper");

    let result = sync::import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject);
    assert!(result.is_err());

    // Existing DB row remains intact.
    let store = RunStore::open(&db_path).expect("store reopen");
    let run = store
        .load_run_details("recovery-2")
        .expect("query")
        .expect("row should remain");
    assert_eq!(run.transcript, "recovery smoke transcript");
}

#[test]
fn recovery_smoke_export_recovers_from_stale_lock_and_temp_artifacts() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("db.sqlite3");
    let export_dir = dir.path().join("snapshot");
    let state_root = dir.path().join("state");
    let locks_dir = state_root.join("locks");

    let store = RunStore::open(&db_path).expect("store");
    store
        .persist_report(&fixture_report("recovery-stale-lock-1", &db_path))
        .expect("persist");

    fs::create_dir_all(&locks_dir).expect("locks dir");
    let stale_lock = locks_dir.join("sync.lock");
    fs::write(
        &stale_lock,
        r#"{"pid":4294967294,"created_at_rfc3339":"2000-01-01T00:00:00Z","operation":"export"}"#,
    )
    .expect("write stale lock");
    fs::create_dir_all(&export_dir).expect("export dir");
    let tmp_file = export_dir.join("runs.jsonl.tmp");
    let mut file = fs::File::create(&tmp_file).expect("tmp");
    writeln!(file, "{{\"partial\":true}}").expect("write tmp");

    let manifest = sync::export(&db_path, &export_dir, &state_root).expect("export recovery");
    assert_eq!(manifest.row_counts.runs, 1);
    assert!(
        !stale_lock.exists(),
        "stale lock should be replaced and released"
    );
    let archived_stale_lock = fs::read_dir(&locks_dir)
        .expect("read locks")
        .filter_map(Result::ok)
        .any(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("sync.lock.stale.")
        });
    assert!(archived_stale_lock, "stale lock should be archived");
}

#[test]
fn recovery_smoke_import_conflict_emits_conflicts_artifact() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("db.sqlite3");
    let export_dir = dir.path().join("snapshot");
    let state_root = dir.path().join("state");

    let store = RunStore::open(&db_path).expect("store");
    store
        .persist_report(&fixture_report("recovery-conflict-1", &db_path))
        .expect("persist");

    sync::export(&db_path, &export_dir, &state_root).expect("export");
    mutate_export_runs_transcript(&export_dir, "mutated conflict transcript");

    let result = sync::import(&db_path, &export_dir, &state_root, ConflictPolicy::Reject);
    assert!(result.is_err());

    let conflicts_path = export_dir.join("sync_conflicts.jsonl");
    assert!(
        conflicts_path.exists(),
        "conflict artifact should be written"
    );
    let conflicts = fs::read_to_string(conflicts_path).expect("read conflicts");
    assert!(conflicts.contains("\"table\":\"runs\""));
}

#[test]
fn recovery_smoke_rebuild_fresh_db_matches_snapshot_expectations() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let rebuilt_db = dir.path().join("rebuilt.sqlite3");
    let export_dir = dir.path().join("snapshot");
    let state_root = dir.path().join("state");

    let source_store = RunStore::open(&source_db).expect("source store");
    source_store
        .persist_report(&fixture_report("recovery-rebuild-1", &source_db))
        .expect("persist 1");
    source_store
        .persist_report(&fixture_report("recovery-rebuild-2", &source_db))
        .expect("persist 2");

    let manifest = sync::export(&source_db, &export_dir, &state_root).expect("export");
    let import_result = sync::import(
        &rebuilt_db,
        &export_dir,
        &state_root,
        ConflictPolicy::Reject,
    )
    .expect("import");

    assert_eq!(import_result.runs_imported, manifest.row_counts.runs);
    assert_eq!(
        import_result.segments_imported,
        manifest.row_counts.segments
    );
    assert_eq!(import_result.events_imported, manifest.row_counts.events);

    let rebuilt_store = RunStore::open(&rebuilt_db).expect("rebuilt store");
    let runs = rebuilt_store
        .list_recent_runs(10)
        .expect("list rebuilt runs");
    assert_eq!(runs.len(), 2);
    let details = rebuilt_store
        .load_run_details("recovery-rebuild-1")
        .expect("details query")
        .expect("details row");
    assert_eq!(
        details.replay.input_content_hash.as_deref(),
        Some("recovery-input-hash")
    );
}
