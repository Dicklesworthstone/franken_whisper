// Included in sync::engine. Verification deliberately reuses production row
// replay, typed reconstruction, and history decoding rather than maintaining
// another implementation of the diarization contracts or writing to the DB
// whose indexes are being checked.

/// Result of restoring the contents of an archive into a private database.
/// For incremental archives this verifies the supplied aggregates and declared
/// tombstones, not the completeness or ordering of an external backup chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveVerificationReport {
    pub schema_version: String,
    pub row_counts: RowCounts,
    pub declared_deletions: u64,
    pub diarization_contracts: std::collections::BTreeMap<String, u64>,
    pub derived_row_counts: std::collections::BTreeMap<String, u64>,
    pub is_valid: bool,
}

// Drop the live connection before its directory (including on an error).
struct VerificationRestore {
    store: RunStore,
    _directory: tempfile::TempDir,
    pinned: PinnedImport,
}

// Explicit current-schema column lists make every persisted value observable;
// SELECT * would compare by incidental column order after a schema migration.
const VERIFICATION_INDEXES: [(&str, &str, bool); 4] = [
    (
        "diarization_reports",
        "run_id, implementation, contract_version, feature_schema, normalized_input_sha256, hint_document_sha256, detected_speakers, constraints_satisfied, supported_speaker_count, speaker_count_status, speaker_count_json, hint_evidence_json, fallback_status, diagnostics_json, profile_persistence_opt_in",
        false,
    ),
    (
        "diarization_turns",
        "run_id, idx, start_ms, end_ms, speaker_ref, speaker_confidence, change_confidence, overlap_suspected, hard_hint_attributed",
        true,
    ),
    (
        "speaker_hints",
        "run_id, idx, speaker_ref, start_ms, end_ms, confidence, policy, provenance",
        true,
    ),
    (
        "speaker_profile_summaries",
        "run_id, idx, speaker_ref, frame_count, voiced_duration_ms, reliability, channel_profile_count, anchored, soft_hint_contradiction, profile_persistence_opt_in",
        true,
    ),
];

type VerificationIndexRows = std::collections::BTreeMap<(String, i64), Vec<SqliteValue>>;

fn restore_archive_for_verification(input_dir: &Path) -> FwResult<VerificationRestore> {
    let pinned = pin_import(input_dir)?;
    let directory = tempfile::Builder::new().prefix(".fw-sync-verify-").tempdir()?;
    let db_path = directory.path().join("verification.sqlite3");
    let store = RunStore::open(&db_path)?;
    let connection = store.connection();
    connection
        .execute("BEGIN;")
        .map_err(|error| FwError::Storage(error.to_string()))?;
    let result = (|| -> FwResult<()> {
        let imported = import_tables(
            connection,
            &pinned.paths,
            &pinned.manifest,
            &pinned.deleted_runs,
            ConflictPolicy::Reject,
            sync_batch_import_enabled(),
        )?;
        if !imported.conflicts.is_empty() {
            return Err(FwError::Storage(
                "archive conflicts with itself during private reconstruction".to_owned(),
            ));
        }
        store.rebuild_diarization_index()?;
        validate_restored_runs(&store, &pinned.run_ids)?;
        connection
            .execute("COMMIT;")
            .map_err(|error| FwError::Storage(error.to_string()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = connection.execute("ROLLBACK;");
    }
    result?;
    let replay = validate_sync_snapshot(connection, pinned._directory.path(), || Ok(()))?;
    if !replay.is_valid {
        return Err(FwError::Storage(
            "private reconstruction did not preserve the archive's canonical rows".to_owned(),
        ));
    }
    if store.query_integrity_check() != "ok" {
        return Err(FwError::Storage(
            "privately reconstructed archive failed integrity_check".to_owned(),
        ));
    }
    Ok(VerificationRestore {
        store,
        _directory: directory,
        pinned,
    })
}

/// Check snapshot restorability without a source or destination database.
/// Checksums, row counts, archive references, typed contracts, derived-index
/// reconstruction, and production history decoding must all succeed. Neither
/// the archive nor any caller database is written; only private scratch files
/// are created. This does not run ASR or require acoustic/model artifacts.
pub fn verify_archive(input_dir: &Path) -> FwResult<ArchiveVerificationReport> {
    let restored = restore_archive_for_verification(input_dir)?;
    let mut derived_row_counts = std::collections::BTreeMap::new();
    let mut diarization_contracts = std::collections::BTreeMap::new();
    for (table, columns, has_index) in VERIFICATION_INDEXES {
        let rows = verification_index_rows(restored.store.connection(), table, columns, has_index)?;
        derived_row_counts.insert(table.to_owned(), rows.len() as u64);
        if table == "diarization_reports" {
            for row in rows.values() {
                let contract =
                    sqlite_required_text(row.get(2), "diarization_reports.contract_version")?;
                *diarization_contracts.entry(contract).or_insert(0) += 1;
            }
        }
    }
    Ok(ArchiveVerificationReport {
        schema_version: "franken-whisper-archive-verification-v1".to_owned(),
        row_counts: restored.pinned.manifest.row_counts.clone(),
        declared_deletions: restored.pinned.deleted_runs.len() as u64,
        diarization_contracts,
        derived_row_counts,
        is_valid: true,
    })
}

pub(super) fn validate_verified_sync(
    db_path: &Path,
    jsonl_dir: &Path,
) -> FwResult<SyncValidationReport> {
    validate_verified_sync_with_after_run_ids(db_path, jsonl_dir, || Ok(()))
}

fn validate_verified_sync_with_after_run_ids(
    db_path: &Path,
    jsonl_dir: &Path,
    after_run_ids: impl FnOnce() -> FwResult<()>,
) -> FwResult<SyncValidationReport> {
    // A typo in a verification command must not initialize an empty database.
    let metadata = fs::metadata(db_path).map_err(|error| {
        FwError::Storage(format!(
            "sync validation requires an existing database: {error}"
        ))
    })?;
    if !metadata.is_file() {
        return Err(FwError::Storage(
            "sync validation requires a regular database file".to_owned(),
        ));
    }
    let restored = restore_archive_for_verification(jsonl_dir)?;
    // Do not use RunStore::open on the target: schema migration can rebuild
    // indexes and would erase the very inconsistency we need to diagnose.
    let connection = Connection::open(db_path.display().to_string())
        .map_err(|error| FwError::Storage(error.to_string()))?;
    with_read_snapshot(&connection, || {
        let mut report = validate_sync_snapshot(
            &connection,
            restored.pinned._directory.path(),
            after_run_ids,
        )?;
        for (table, columns, has_index) in VERIFICATION_INDEXES {
            let actual = verification_index_rows(&connection, table, columns, has_index)?;
            let expected = verification_index_rows(
                restored.store.connection(),
                table,
                columns,
                has_index,
            )?;
            collect_verification_index_mismatches(&actual, &expected, table, &mut report);
        }
        Ok(report)
    })
}

fn verification_index_rows(
    connection: &Connection,
    table: &str,
    columns: &str,
    has_index: bool,
) -> FwResult<VerificationIndexRows> {
    let rows = connection
        .query(&format!("SELECT {columns} FROM {table}"))
        .map_err(|error| {
            FwError::Storage(format!("cannot verify derived index {table}: {error}"))
        })?;
    let width = columns.split(',').count();
    let mut values = VerificationIndexRows::new();
    for row in &rows {
        let run_id = sqlite_required_text(row.get(0), &format!("{table}.run_id"))?;
        let index = if has_index {
            sqlite_required_i64(row.get(1), &format!("{table}.idx"))?
        } else {
            0
        };
        let columns = (0..width)
            .map(|column| {
                row.get(column).cloned().ok_or_else(|| {
                    FwError::Storage(format!("missing column {column} in derived index {table}"))
                })
            })
            .collect::<FwResult<Vec<_>>>()?;
        if values.insert((run_id, index), columns).is_some() {
            return Err(FwError::Storage(format!(
                "duplicate key in derived index {table}"
            )));
        }
    }
    Ok(values)
}

fn collect_verification_index_mismatches(
    actual: &VerificationIndexRows,
    expected: &VerificationIndexRows,
    table: &str,
    report: &mut SyncValidationReport,
) {
    let keys: std::collections::BTreeSet<_> = actual.keys().chain(expected.keys()).collect();
    for key in keys {
        let reason = match (actual.get(key), expected.get(key)) {
            (Some(left), Some(right)) if left == right => continue,
            (Some(_), Some(_)) => "derived payload differs from canonical reconstruction",
            (Some(_), None) => "unexpected derived record absent from canonical reconstruction",
            (None, Some(_)) => "missing derived record required by canonical reconstruction",
            (None, None) => continue,
        };
        report.is_valid = false;
        if report.child_mismatches.len() < MAX_SYNC_VALIDATION_CHILD_MISMATCHES {
            report.child_mismatches.push(SyncConflict {
                table: table.to_owned(),
                key: format!("{}/{}", key.0, key.1),
                reason: reason.to_owned(),
            });
        } else {
            report.child_mismatches_truncated = true;
        }
    }
}

#[cfg(test)]
mod archive_verification_tests {
    use super::*;
    use crate::model::{
        BackendKind, BackendParams, DiarizationReport, DiarizationRequest, InputSource,
        ReplayEnvelope, RunReport, TranscribeRequest, TranscriptionResult, TranscriptionSegment,
    };
    use serde_json::json;

    fn fixture(root: &Path, with_run: bool) -> (PathBuf, PathBuf) {
        let db = root.join("source.sqlite3");
        let store = RunStore::open(&db).expect("source store");
        if with_run {
            // Model-free persistence fixture: preserve the default engine's
            // actual contract and learned-activity evidence, never relabel it.
            let diarization: DiarizationReport = serde_json::from_value(json!({
                "implementation": "native-sortformer-v1",
                "contract_version": crate::diarization::SORTFORMER_DIARIZATION_CONTRACT_VERSION,
                "feature_schema": "sortformer-activity-80ms-v1",
                "speaker_evidence_mode": "sortformer_activity",
                "normalized_input_sha256": "a".repeat(64),
                "turns": [{
                    "start_ms": 0, "end_ms": 6000, "speaker_ref": "SPEAKER_00",
                    "speaker_confidence": 0.9, "change_confidence": null,
                    "overlap_suspected": false, "hard_hint_attributed": false
                }],
                "profiles": [{
                    "speaker_ref": "SPEAKER_00", "frame_count": 75,
                    "voiced_duration_ms": 6000, "reliability": 0.9,
                    "voice_profile_count": 1, "channel_profile_count": 0,
                    "training_accepted_count": 1, "training_downweighted_count": 0,
                    "training_quarantined_count": 0, "anchored": false,
                    "soft_hint_contradiction": null
                }],
                "speaker_count": {
                    "request": {"mode": "infer"}, "status": "unresolved",
                    "supported_speaker_count": 1, "active_speaker_refs": ["SPEAKER_00"],
                    "dominant_speaker_share": 1.0, "unknown_voiced_share": 0.0,
                    "reasons": ["speaker_count_evidence_unresolved"],
                    "speaker_evidence": [{
                        "speaker_ref": "SPEAKER_00", "assigned_tracklet_count": 1,
                        "independent_tracklet_count": 1, "recurrence_episode_count": 1,
                        "voiced_frame_count": 75, "independent_voiced_frame_count": 75,
                        "voiced_duration_ms": 6000, "mean_assignment_confidence": 0.9,
                        "profile_reliability": 0.9, "hard_anchored": false,
                        "separated_from_supported_speakers": true,
                        "reasons": ["supported_by_learned_model_activity"], "supported": true
                    }]
                },
                "fallback_status": "speaker_count_unresolved"
            })).expect("Sortformer fixture DTO");
            diarization.validate_against_request(&DiarizationRequest::default(), None)
                .expect("Sortformer fixture semantics");
            let report = RunReport {
                run_id: "verify-sortformer".to_owned(),
                trace_id: "0".repeat(32),
                started_at_rfc3339: "2026-10-15T00:00:00Z".to_owned(),
                finished_at_rfc3339: "2026-10-15T00:00:06Z".to_owned(),
                input_path: "fixture.wav".to_owned(),
                normalized_wav_path: "normalized.wav".to_owned(),
                request: TranscribeRequest {
                    input: InputSource::File { path: PathBuf::from("fixture.wav") },
                    backend: BackendKind::Auto,
                    model: None,
                    language: None,
                    translate: false,
                    diarize: true,
                    persist: true,
                    db_path: db.clone(),
                    timeout_ms: None,
                    backend_params: BackendParams::default(),
                },
                result: TranscriptionResult {
                    backend: BackendKind::WhisperCpp,
                    transcript: "verification fixture".to_owned(),
                    language: None,
                    segments: vec![TranscriptionSegment {
                        start_sec: Some(0.0), end_sec: Some(6.0),
                        text: "verification fixture".to_owned(),
                        speaker: Some("SPEAKER_00".to_owned()), confidence: Some(0.9),
                    }],
                    acceleration: None,
                    diarization: Some(diarization),
                    raw_output: json!({}),
                    artifact_paths: Vec::new(),
                },
                events: Vec::new(),
                warnings: Vec::new(),
                evidence: Vec::new(),
                replay: ReplayEnvelope::default(),
            };
            store.persist_report(&report).expect("persist source");
        }
        drop(store);
        let archive = root.join("snapshot");
        export(&db, &archive, &root.join("export-state")).expect("export");
        (db, archive)
    }

    #[test]
    fn verify_archive_restores_default_sortformer_plain_and_gzip() {
        for compression in [CompressionMode::None, CompressionMode::Gzip] {
            let dir = tempfile::tempdir().expect("tempdir");
            let (db, archive) = fixture(dir.path(), true);
            export_with_compression(&db, &archive, &dir.path().join("state"), compression)
                .expect("export selected encoding");
            let verified = verify_archive(&archive).expect("verify restorable archive");
            assert!(verified.is_valid);
            assert_eq!(verified.row_counts.runs, 1);
            assert_eq!(verified.row_counts.segments, 1);
            assert_eq!(verified.declared_deletions, 0);
            assert_eq!(verified.diarization_contracts.len(), 1);
            assert_eq!(verified.diarization_contracts[
                crate::diarization::SORTFORMER_DIARIZATION_CONTRACT_VERSION
            ], 1);
            assert_eq!(verified.derived_row_counts["diarization_turns"], 1);
            assert_eq!(verified.derived_row_counts["speaker_profile_summaries"], 1);
            assert_eq!(verified.derived_row_counts["speaker_hints"], 0);
            assert!(crate::sync::validate_sync(&db, &archive).expect("compare indexes").is_valid);
            assert!(!archive.join("sync_conflicts.jsonl").exists());
        }
    }

    #[test]
    fn verify_archive_accepts_empty_snapshot_without_creating_missing_target() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (db, archive) = fixture(dir.path(), false);
        let verified = verify_archive(&archive).expect("empty archive");
        assert_eq!(verified.row_counts.runs, 0);
        assert!(verified.diarization_contracts.is_empty());
        assert!(verified.derived_row_counts.values().all(|count| *count == 0));
        assert!(crate::sync::validate_sync(&db, &archive).expect("empty target").is_valid);
        let absent = dir.path().join("not-a-database.sqlite3");
        let error = crate::sync::validate_sync(&absent, &archive).expect_err("no implicit DB");
        assert!(error.to_string().contains("existing database"), "{error}");
        assert!(!absent.exists());
    }

    #[test]
    fn verified_sync_rejects_bad_manifest_even_when_canonical_rows_match() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (db, archive) = fixture(dir.path(), true);
        let path = archive.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).expect("manifest")).expect("JSON");
        manifest["checksums"]["runs_jsonl_sha256"] = json!("f".repeat(64));
        fs::write(path, manifest.to_string()).expect("corrupt only manifest");
        // Mutation control: the old row-only comparator misses this damage.
        assert!(validate_sync(&db, &archive).expect("row-only control").is_valid);
        let error = crate::sync::validate_sync(&db, &archive).expect_err("checksum rejection");
        assert!(error.to_string().contains("checksum mismatch"), "{error}");
    }

    #[test]
    fn verify_archive_rejects_correctly_checksummed_unknown_diarization_contract() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (_, archive) = fixture(dir.path(), true);
        let runs = archive.join("runs.jsonl");
        let mut row: serde_json::Value =
            serde_json::from_slice(&fs::read(&runs).expect("runs")).expect("one row");
        let mut result: serde_json::Value =
            serde_json::from_str(row["result_json"].as_str().expect("result")).expect("JSON");
        result["diarization"]["contract_version"] = json!("sortformer-diarization-v999");
        row["result_json"] = json!(result.to_string());
        let bytes = format!("{row}\n");
        fs::write(&runs, &bytes).expect("corrupt typed contract");
        let path = archive.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).expect("manifest")).expect("JSON");
        manifest["checksums"]["runs_jsonl_sha256"] =
            json!(format!("{:x}", Sha256::digest(bytes.as_bytes())));
        fs::write(path, manifest.to_string()).expect("valid checksum");
        assert!(verify_archive(&archive).is_err());
        assert!(!archive.join("sync_conflicts.jsonl").exists());
    }

    #[test]
    fn verified_sync_detects_damage_in_each_derived_table_without_repairing_it() {
        for (table, mutation) in [
            ("diarization_reports", "UPDATE diarization_reports SET contract_version = 'damaged-index'"),
            ("diarization_turns", "DELETE FROM diarization_turns"),
            ("speaker_profile_summaries", "UPDATE speaker_profile_summaries SET reliability = 0.123"),
            ("speaker_hints", "INSERT INTO speaker_hints (run_id, idx, speaker_ref, start_ms, end_ms, confidence, policy, provenance) VALUES ('verify-sortformer', 0, 'invented', 0, 50, 1.0, 'hard_must_link', 'forged-provenance')"),
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let (db, archive) = fixture(dir.path(), true);
            let connection = Connection::open(db.display().to_string()).expect("target");
            connection.execute(mutation).expect("damage only derived state");
            let (_, columns, has_index) = VERIFICATION_INDEXES.into_iter()
                .find(|(name, _, _)| *name == table).expect("known table");
            let before = verification_index_rows(&connection, table, columns, has_index)
                .expect("damaged state");
            assert!(validate_sync(&db, &archive).expect("old comparator control").is_valid);
            let report = crate::sync::validate_sync(&db, &archive).expect("verify all tables");
            assert!(!report.is_valid, "{table} damage must invalidate verification");
            assert!(report.child_mismatches.iter().any(|mismatch| mismatch.table == table));
            assert_eq!(verification_index_rows(&connection, table, columns, has_index)
                .expect("state after verification"), before, "verification must not repair {table}");
        }
    }

    #[test]
    fn verified_sync_reads_pinned_archive_after_original_files_are_replaced() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (db, archive) = fixture(dir.path(), true);
        let report = validate_verified_sync_with_after_run_ids(&db, &archive, || {
            for name in ["manifest.json", "runs.jsonl", "segments.jsonl", "events.jsonl"] {
                fs::write(archive.join(name), b"replaced after pinning\n")?;
            }
            Ok(())
        }).expect("compare the verified bytes");
        assert!(report.is_valid);
        assert!(crate::sync::validate_sync(&db, &archive).is_err());
    }

    #[test]
    fn verified_sync_uses_one_target_snapshot_for_canonical_and_derived_reads() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (db, archive) = fixture(dir.path(), true);
        let writer = Connection::open(db.display().to_string()).expect("writer");
        let report = validate_verified_sync_with_after_run_ids(&db, &archive, || {
            writer.execute("UPDATE speaker_profile_summaries SET reliability = 0.123")
                .map_err(|error| FwError::Storage(error.to_string()))?;
            Ok(())
        }).expect("stable target snapshot");
        assert!(report.is_valid, "the earlier read snapshot still has valid indexes");
        assert!(!crate::sync::validate_sync(&db, &archive).expect("new snapshot").is_valid);
    }

    #[test]
    fn derived_verification_diagnostics_are_sorted_bounded_and_payload_free() {
        let mut report = SyncValidationReport {
            db_run_count: 0, jsonl_run_count: 0,
            db_segment_count: 0, jsonl_segment_count: 0,
            db_event_count: 0, jsonl_event_count: 0,
            missing_from_jsonl: Vec::new(), missing_from_db: Vec::new(),
            mismatched_records: Vec::new(), child_mismatches: Vec::new(),
            child_mismatches_truncated: false, is_valid: true,
        };
        let actual = (0..=MAX_SYNC_VALIDATION_CHILD_MISMATCHES).rev().map(|index| {
            (("run".to_owned(), index as i64), vec![SqliteValue::Text("never-emit-payload".into())])
        }).collect();
        collect_verification_index_mismatches(
            &actual, &VerificationIndexRows::new(), "speaker_hints", &mut report,
        );
        assert!(!report.is_valid);
        assert!(report.child_mismatches_truncated);
        assert_eq!(report.child_mismatches.len(), MAX_SYNC_VALIDATION_CHILD_MISMATCHES);
        assert_eq!(report.child_mismatches[0].key, "run/0");
        assert_eq!(report.child_mismatches[1].key, "run/1");
        assert!(!serde_json::to_string(&report).expect("report").contains("never-emit-payload"));
    }
}
