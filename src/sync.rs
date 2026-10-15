//! JSONL synchronization and atomic publication of fresh restored databases.
//!
//! The table replay engine retains its existing transaction and contract
//! checks. A previously absent destination is built privately, checkpointed,
//! integrity-checked, and only then published without replacing another file.

use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{FwError, FwResult};
use crate::storage::BlockingConnection;

// Keep the existing table replay, export, locking, and regression tests in one
// implementation module. Its source is unchanged by the publication boundary.
mod engine {
    include!("sync_engine.rs");

    // The fresh-target caller holds the public sync lock through publication.
    // Reuse the same importer without trying to acquire that lock twice.
    pub(super) fn import_with_lock_held(
        db_path: &Path,
        input_dir: &Path,
        conflict_policy: ConflictPolicy,
    ) -> FwResult<ImportResult> {
        import_inner(db_path, input_dir, conflict_policy)
    }
}

pub use engine::*;

/// Import a JSONL snapshot, preserving the existing conflict-policy contract.
///
/// Existing databases are updated by the rollback-safe replay engine. When
/// the destination does not yet exist, import instead builds a private sibling
/// database: malformed contracts, failed checksums, failed index reconstruction,
/// and interrupted replay cannot leave a partial database at the requested path.
/// The original request/result JSON and run identities are not rewritten.
///
/// Publication requires same-filesystem hard links and fails closed if another
/// writer creates the destination or a SQLite sidecar before publication. A
/// directory-fsync failure after publication is reported explicitly; in that
/// case the complete database exists but crash durability is uncertain.
pub fn import(
    db_path: &Path,
    input_dir: &Path,
    state_root: &Path,
    conflict_policy: ConflictPolicy,
) -> FwResult<ImportResult> {
    if db_path == Path::new(":memory:") || entry_exists(db_path)? {
        return engine::import(db_path, input_dir, state_root, conflict_policy);
    }
    import_fresh_with_before_publish(db_path, input_dir, state_root, conflict_policy, || {
        Ok(())
    })
}

fn entry_exists(path: &Path) -> FwResult<bool> {
    // exists() hides permission errors and treats dangling symlinks as absent.
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn sidecar_path(db_path: &Path, suffix: &str) -> PathBuf {
    let mut name = db_path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

fn require_unoccupied_destination(db_path: &Path) -> FwResult<()> {
    if entry_exists(db_path)? {
        return Err(FwError::Storage(format!(
            "fresh restore destination appeared during import: {}; refusing to replace it",
            db_path.display()
        )));
    }
    // Never attach a newly restored main file to an unrelated surviving WAL,
    // rollback journal, or FrankenSQLite commit-certificate sidecar.
    for suffix in ["-wal", "-shm", "-journal", "-wal-cert", "-wal-cert-head"] {
        let sidecar = sidecar_path(db_path, suffix);
        if entry_exists(&sidecar)? {
            return Err(FwError::Storage(format!(
                "fresh restore found an existing database sidecar: {}; preserve or recover it before restoring",
                sidecar.display()
            )));
        }
    }
    Ok(())
}

fn import_fresh_with_before_publish(
    db_path: &Path,
    input_dir: &Path,
    state_root: &Path,
    conflict_policy: ConflictPolicy,
    before_publish: impl FnOnce() -> FwResult<()>,
) -> FwResult<ImportResult> {
    let _lock = SyncLock::acquire(state_root, "import")?;
    require_unoccupied_destination(db_path)?;
    let parent = db_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let staging_dir = tempfile::Builder::new()
        .prefix(".fw-restore-")
        .tempdir_in(parent)?;
    let staged_db = staging_dir.path().join("state.sqlite3");

    // This is the production importer, including checksum/count checks, typed
    // request/report validation, and derived-index rebuild before COMMIT.
    // Conflict artifacts still go to the original snapshot directory.
    let imported = engine::import_with_lock_held(&staged_db, input_dir, conflict_policy)?;
    seal_restored_database(&staged_db)?;
    before_publish()?;
    require_unoccupied_destination(db_path)?;

    // rename() would silently overwrite a racing writer on Unix. A hard link
    // is an atomic create-if-absent operation; staging is on the same filesystem.
    fs::hard_link(&staged_db, db_path).map_err(|error| {
        FwError::Storage(format!(
            "cannot publish fresh restored database {} without replacement: {error}",
            db_path.display()
        ))
    })?;
    #[cfg(unix)]
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            FwError::Storage(format!(
                "restored database {} was published, but directory sync failed; crash durability is uncertain: {error}",
                db_path.display()
            ))
        })?;
    Ok(imported)
}

fn seal_restored_database(db_path: &Path) -> FwResult<()> {
    let connection = BlockingConnection::open(db_path.display().to_string())
        .map_err(|error| FwError::Storage(error.to_string()))?;
    let checkpoint = connection
        .query("PRAGMA wal_checkpoint(TRUNCATE);")
        .map_err(|error| FwError::Storage(format!("restore checkpoint failed: {error}")))?;
    if checkpoint.len() != 1
        || checkpoint[0].get(0).and_then(|value| value.as_integer()) != Some(0)
    {
        return Err(FwError::Storage(
            "restore checkpoint did not complete; refusing to publish a WAL-dependent database"
                .to_owned(),
        ));
    }
    let mode = connection
        .query("PRAGMA journal_mode=DELETE;")
        .map_err(|error| FwError::Storage(format!("restore journal finalization failed: {error}")))?;
    if mode.len() != 1
        || !matches!(mode[0].get(0), Some(fsqlite_types::value::SqliteValue::Text(value)) if value.eq_ignore_ascii_case("delete"))
    {
        return Err(FwError::Storage(
            "restore did not enter DELETE journal mode; refusing to publish dependent sidecars"
                .to_owned(),
        ));
    }
    let integrity = connection
        .query("PRAGMA integrity_check;")
        .map_err(|error| FwError::Storage(format!("restore integrity check failed: {error}")))?;
    if integrity.len() != 1
        || !matches!(integrity[0].get(0), Some(fsqlite_types::value::SqliteValue::Text(value)) if value.as_ref() == "ok")
    {
        return Err(FwError::Storage(
            "restored database failed integrity_check; destination was not published".to_owned(),
        ));
    }
    drop(connection);
    fs::OpenOptions::new().write(true).open(db_path)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod fresh_restore_tests {
    use super::*;
    use crate::model::{
        BackendKind, BackendParams, InputSource, ReplayEnvelope, RunEvent, RunReport,
        TranscribeRequest, TranscriptionResult, TranscriptionSegment,
    };
    use crate::storage::RunStore;
    use serde_json::json;
    use sha2::{Digest, Sha256};

    fn snapshot(root: &Path, with_run: bool) -> PathBuf {
        let db = root.join("source.sqlite3");
        let store = RunStore::open(&db).expect("source store");
        if with_run {
            let report = RunReport {
                run_id: "restore-run".to_owned(),
                trace_id: "00000000000000000000000000000000".to_owned(),
                started_at_rfc3339: "2026-10-15T00:00:00Z".to_owned(),
                finished_at_rfc3339: "2026-10-15T00:00:01Z".to_owned(),
                input_path: "fixture.wav".to_owned(),
                normalized_wav_path: "normalized.wav".to_owned(),
                request: TranscribeRequest {
                    input: InputSource::File {
                        path: PathBuf::from("fixture.wav"),
                    },
                    backend: BackendKind::Auto,
                    model: None,
                    language: None,
                    translate: false,
                    diarize: false,
                    persist: true,
                    db_path: db.clone(),
                    timeout_ms: None,
                    backend_params: BackendParams::default(),
                },
                result: TranscriptionResult {
                    backend: BackendKind::WhisperCpp,
                    transcript: "restore fixture".to_owned(),
                    language: None,
                    segments: vec![TranscriptionSegment {
                        start_sec: Some(0.0),
                        end_sec: Some(1.0),
                        text: "restore fixture".to_owned(),
                        speaker: None,
                        confidence: Some(0.9),
                    }],
                    acceleration: None,
                    diarization: None,
                    raw_output: json!({}),
                    artifact_paths: Vec::new(),
                },
                events: vec![RunEvent {
                    seq: 1,
                    ts_rfc3339: "2026-10-15T00:00:01Z".to_owned(),
                    stage: "backend".to_owned(),
                    code: "backend.ok".to_owned(),
                    message: "fixture complete".to_owned(),
                    payload: json!({"fixture": true}),
                }],
                warnings: Vec::new(),
                evidence: Vec::new(),
                replay: ReplayEnvelope::default(),
            };
            store.persist_report(&report).expect("persist source");
        }
        drop(store);
        let archive = root.join("snapshot");
        export(&db, &archive, &root.join("export-state")).expect("export");
        archive
    }

    #[test]
    fn fresh_restore_publishes_complete_database_and_reopens() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = snapshot(dir.path(), true);
        let target = dir.path().join("restored.sqlite3");
        let state = dir.path().join("state");
        let imported = import(&target, &archive, &state, ConflictPolicy::Reject).expect("restore");
        assert_eq!(imported.runs_imported, 1);
        assert_eq!(imported.segments_imported, 1);
        assert_eq!(imported.events_imported, 1);
        let store = RunStore::open(&target).expect("reopen published database");
        let run = store
            .load_run_details("restore-run")
            .expect("load")
            .expect("run");
        assert_eq!(run.transcript, "restore fixture");
        assert_eq!(run.segments.len(), 1);
        assert_eq!(run.events.len(), 1);
        assert_eq!(store.query_integrity_check(), "ok");
        drop(store);
        assert!(validate_sync(&target, &archive).expect("compare").is_valid);
        // The second import exercises the unchanged existing-database path.
        assert!(import(&target, &archive, &state, ConflictPolicy::Reject).is_ok());
    }

    #[test]
    fn fresh_restore_bad_typed_request_leaves_no_destination_and_can_retry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = snapshot(dir.path(), true);
        let runs = archive.join("runs.jsonl");
        let original = fs::read(&runs).expect("runs");
        let manifest_path = archive.join("manifest.json");
        let original_manifest = fs::read(&manifest_path).expect("manifest");
        let mut row: serde_json::Value = serde_json::from_slice(&original).expect("one run");
        let mut request: serde_json::Value =
            serde_json::from_str(row["request_json"].as_str().expect("request"))
                .expect("request JSON");
        request["diarize"] = json!("not-a-boolean");
        row["request_json"] = json!(request.to_string());
        let bytes = format!("{row}\n");
        fs::write(&runs, &bytes).expect("write malformed contract");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&original_manifest).expect("manifest JSON");
        manifest["checksums"]["runs_jsonl_sha256"] =
            json!(format!("{:x}", Sha256::digest(bytes.as_bytes())));
        fs::write(&manifest_path, manifest.to_string()).expect("refresh checksum");
        let target = dir.path().join("restored.sqlite3");
        let state = dir.path().join("state");
        let error = import(&target, &archive, &state, ConflictPolicy::Reject).expect_err("reject");
        assert!(error.to_string().contains("diarize"), "{error}");
        assert!(!entry_exists(&target).expect("target metadata"));
        assert!(!state.join("locks/sync.lock").exists());
        for suffix in ["-wal", "-shm", "-journal"] {
            assert!(!entry_exists(&sidecar_path(&target, suffix)).expect("sidecar metadata"));
        }
        fs::write(&runs, original).expect("repair runs");
        fs::write(&manifest_path, original_manifest).expect("repair manifest");
        import(&target, &archive, &state, ConflictPolicy::Reject).expect("retry repaired archive");
    }

    #[test]
    fn fresh_restore_empty_snapshot_is_a_valid_database() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = snapshot(dir.path(), false);
        let target = dir.path().join("empty.sqlite3");
        let imported = import(
            &target,
            &archive,
            &dir.path().join("state"),
            ConflictPolicy::Reject,
        )
        .expect("empty restore");
        assert_eq!(imported.runs_imported, 0);
        let store = RunStore::open(&target).expect("empty store");
        assert!(store.list_recent_runs(10).expect("list").is_empty());
        assert_eq!(store.query_integrity_check(), "ok");
    }

    #[test]
    fn fresh_restore_preserves_racing_destination() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = snapshot(dir.path(), true);
        let target = dir.path().join("racing.sqlite3");
        let error = import_fresh_with_before_publish(
            &target,
            &archive,
            &dir.path().join("state"),
            ConflictPolicy::OverwriteStrict,
            || {
                fs::write(&target, b"other writer owns this file")?;
                Ok(())
            },
        )
        .expect_err("never replace a racing writer");
        assert!(error.to_string().contains("appeared during import"), "{error}");
        assert_eq!(
            fs::read(&target).expect("racing file"),
            b"other writer owns this file"
        );
    }

    #[test]
    fn fresh_restore_refuses_orphan_sidecars_without_modifying_them() {
        for suffix in ["-wal", "-shm", "-journal", "-wal-cert", "-wal-cert-head"] {
            let dir = tempfile::tempdir().expect("tempdir");
            let target = dir.path().join("orphan.sqlite3");
            let sidecar = sidecar_path(&target, suffix);
            fs::write(&sidecar, b"recover me first").expect("orphan sidecar");
            let error = import(
                &target,
                &dir.path().join("no-archive"),
                &dir.path().join("state"),
                ConflictPolicy::Reject,
            )
            .expect_err("reject before archive access");
            assert!(error.to_string().contains("existing database sidecar"), "{error}");
            assert!(!target.exists());
            assert_eq!(
                fs::read(&sidecar).expect("sidecar survives"),
                b"recover me first"
            );
        }
    }

    #[test]
    fn fresh_restore_rechecks_sidecars_before_publication() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive = snapshot(dir.path(), true);
        let target = dir.path().join("racing-wal.sqlite3");
        let wal = sidecar_path(&target, "-wal");
        let error = import_fresh_with_before_publish(
            &target,
            &archive,
            &dir.path().join("state"),
            ConflictPolicy::Reject,
            || {
                fs::write(&wal, b"other writer WAL")?;
                Ok(())
            },
        )
        .expect_err("never publish next to a racing WAL");
        assert!(error.to_string().contains("existing database sidecar"), "{error}");
        assert!(!target.exists());
        assert_eq!(fs::read(wal).expect("WAL survives"), b"other writer WAL");
    }
}
