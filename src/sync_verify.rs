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
    /// Read from the same manifest that authorized the pinned channel bytes.
    pub archive_mode: ArchiveMode,
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
        schema_version: "franken-whisper-archive-verification-v2".to_owned(),
        archive_mode: restored.pinned.archive_mode,
        row_counts: restored.pinned.manifest.row_counts.clone(),
        declared_deletions: restored.pinned.deleted_runs.len() as u64,
        diarization_contracts,
        derived_row_counts,
        is_valid: true,
    })
}

/// Full snapshots compare the entire database. Incremental snapshots compare
/// only supplied run aggregates plus declared deletions; `db_*_count` fields
/// count rows in that scope. Success does not certify an entire backup chain.
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
    // Scope is bound to the pinned manifest, never a second read of a mutable
    // archive. Include tombstones so orphan child/index rows for deleted runs
    // are checked even when their canonical parent is already absent.
    let scope = (restored.pinned.archive_mode == ArchiveMode::Incremental).then(|| {
        let mut ids = restored.pinned.run_ids.clone();
        ids.extend(restored.pinned.deleted_runs.iter().map(|row| row.run_id.clone()));
        ids.sort();
        ids.dedup();
        ids
    });
    // Do not use RunStore::open on the target: schema migration can rebuild
    // indexes and would erase the very inconsistency we need to diagnose.
    let connection = Connection::open(db_path.display().to_string())
        .map_err(|error| FwError::Storage(error.to_string()))?;
    with_read_snapshot(&connection, || {
        let mut report = match scope.as_deref() {
            None => validate_sync_snapshot(
                &connection,
                restored.pinned._directory.path(),
                after_run_ids,
            )?,
            Some(ids) => compare_incremental_canonical_rows(
                &connection,
                restored.store.connection(),
                ids,
                after_run_ids,
            )?,
        };
        for (table, columns, has_index) in VERIFICATION_INDEXES {
            let actual = verification_rows_in_scope(
                &connection, table, columns, has_index, scope.as_deref(),
            )?;
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
    verification_rows_in_scope(connection, table, columns, has_index, None)
}

/// Select scope before applying diagnostic limits. Filtering a truncated full
/// comparison afterward could hide real damage behind unrelated extra rows.
/// Table/column identifiers are internal constants; run IDs are bound values.
fn verification_rows_in_scope(
    connection: &Connection,
    table: &str,
    columns: &str,
    has_index: bool,
    scope: Option<&[String]>,
) -> FwResult<VerificationIndexRows> {
    let query_error = |error| {
        FwError::Storage(format!("cannot verify table {table}: {error}"))
    };
    let rows = match scope {
        None => connection
            .query(&format!("SELECT {columns} FROM {table}"))
            .map_err(query_error)?,
        Some(ids) => {
            let identity_column = if table == "runs" { "id" } else { "run_id" };
            let mut rows = Vec::new();
            // An empty delta has no authority over existing rows, but still
            // check the table schema and establish the target read snapshot.
            if ids.is_empty() {
                connection
                    .query(&format!("SELECT {columns} FROM {table} WHERE 0"))
                    .map_err(query_error)?;
            }
            for chunk in ids.chunks(sync_query_batch_size().clamp(1, 512)) {
                let placeholders = (1..=chunk.len())
                    .map(|index| format!("?{index}"))
                    .collect::<Vec<_>>()
                    .join(",");
                let params: Vec<SqliteValue> = chunk
                    .iter()
                    .map(|id| SqliteValue::Text(id.clone().into()))
                    .collect();
                rows.extend(connection.query_with_params(
                    &format!(
                        "SELECT {columns} FROM {table} WHERE {identity_column} IN ({placeholders})"
                    ),
                    &params,
                ).map_err(query_error)?);
            }
            rows
        }
    };
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

/// The expected database has already passed exact pinned-channel comparison.
/// Reuse that independently reconstructed state rather than reinterpreting
/// result_json.segments (native-listen and edited child rows have their own
/// table authority). All target reads remain inside the caller's snapshot.
fn compare_incremental_canonical_rows(
    actual: &Connection,
    expected: &Connection,
    scope: &[String],
    after_run_ids: impl FnOnce() -> FwResult<()>,
) -> FwResult<SyncValidationReport> {
    verify_schema_exists(actual)?;
    const RUN_COLUMNS: &str = "id, started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json";
    let actual_runs = verification_rows_in_scope(actual, "runs", RUN_COLUMNS, false, Some(scope))?;
    after_run_ids()?;
    let expected_runs = verification_index_rows(expected, "runs", RUN_COLUMNS, false)?;
    let mut report = SyncValidationReport {
        db_run_count: actual_runs.len() as u64,
        jsonl_run_count: expected_runs.len() as u64,
        db_segment_count: 0,
        jsonl_segment_count: 0,
        db_event_count: 0,
        jsonl_event_count: 0,
        missing_from_jsonl: Vec::new(),
        missing_from_db: Vec::new(),
        mismatched_records: Vec::new(),
        child_mismatches: Vec::new(),
        child_mismatches_truncated: false,
        is_valid: true,
    };
    for (key, row) in &actual_runs {
        match expected_runs.get(key) {
            None => report.missing_from_jsonl.push(key.0.clone()),
            Some(expected_row) if row != expected_row => {
                report.mismatched_records.push(key.0.clone());
            }
            Some(_) => {}
        }
    }
    for key in expected_runs.keys() {
        if !actual_runs.contains_key(key) {
            report.missing_from_db.push(key.0.clone());
        }
    }

    const SEGMENT_COLUMNS: &str = "run_id, idx, start_sec, end_sec, speaker, text, confidence";
    let actual_segments = verification_segment_values(verification_rows_in_scope(
        actual, "segments", SEGMENT_COLUMNS, true, Some(scope),
    )?)?;
    let expected_segments = verification_segment_values(verification_index_rows(
        expected, "segments", SEGMENT_COLUMNS, true,
    )?)?;
    report.db_segment_count = actual_segments.len() as u64;
    report.jsonl_segment_count = expected_segments.len() as u64;
    collect_child_mismatches(
        &actual_segments, &expected_segments, "segments",
        &mut report.child_mismatches, &mut report.child_mismatches_truncated,
    );

    const EVENT_COLUMNS: &str = "run_id, seq, ts_rfc3339, stage, code, message, payload_json";
    let actual_events: HashMap<_, _> = verification_rows_in_scope(
        actual, "events", EVENT_COLUMNS, true, Some(scope),
    )?.into_iter().collect();
    let expected_events: HashMap<_, _> = verification_index_rows(
        expected, "events", EVENT_COLUMNS, true,
    )?.into_iter().collect();
    report.db_event_count = actual_events.len() as u64;
    report.jsonl_event_count = expected_events.len() as u64;
    collect_child_mismatches(
        &actual_events, &expected_events, "events",
        &mut report.child_mismatches, &mut report.child_mismatches_truncated,
    );
    report.is_valid = report.missing_from_jsonl.is_empty()
        && report.missing_from_db.is_empty()
        && report.mismatched_records.is_empty()
        && report.child_mismatches.is_empty()
        && !report.child_mismatches_truncated;
    Ok(report)
}

fn verification_segment_values(
    rows: VerificationIndexRows,
) -> FwResult<HashMap<ChildValidationKey, SegmentValidationRow>> {
    rows.into_iter().map(|(key, row)| {
        // Match the full comparator's nullable/numeric semantics: SQLite
        // INTEGER 1 and REAL 1.0 represent the same segment time.
        let segment = SegmentValidationRow {
            start_sec: sqlite_nullable_f64(row.get(2), "segments.start_sec")?,
            end_sec: sqlite_nullable_f64(row.get(3), "segments.end_sec")?,
            speaker: sqlite_nullable_text(row.get(4), "segments.speaker")?,
            text: sqlite_required_text(row.get(5), "segments.text")?,
            confidence: sqlite_nullable_f64(row.get(6), "segments.confidence")?,
        };
        Ok((key, segment))
    }).collect()
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

    fn set_archive_mode(archive: &Path, mode: serde_json::Value) {
        let path = archive.join("manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).expect("manifest")).expect("JSON");
        manifest["export_mode"] = mode;
        fs::write(path, manifest.to_string()).expect("set mode");
    }

    // An independent aggregate uses the same run-local speaker reference. Its
    // many events exercise scoping *before* the bounded mismatch accumulator.
    fn add_unrelated_aggregate(db: &Path) {
        let store = RunStore::open(db).expect("store");
        let connection = store.connection();
        connection.execute("BEGIN;").expect("begin fixture");
        connection.execute(
            "INSERT INTO runs (id, started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json) \
             SELECT 'unrelated', started_at, finished_at, backend, input_path, normalized_wav_path, request_json, result_json, warnings_json, transcript, replay_json, acceleration_json FROM runs WHERE id = 'verify-sortformer'"
        ).expect("independent parent");
        connection.execute(
            "INSERT INTO segments (run_id, idx, start_sec, end_sec, speaker, text, confidence) \
             SELECT 'unrelated', idx, start_sec, end_sec, speaker, text, confidence FROM segments WHERE run_id = 'verify-sortformer'"
        ).expect("independent segments");
        for index in 0..=MAX_SYNC_VALIDATION_CHILD_MISMATCHES {
            connection.execute_with_params(
                "INSERT INTO events (run_id, seq, ts_rfc3339, stage, code, message, payload_json) \
                 VALUES ('unrelated', ?1, '2026-10-15T00:00:01Z', 'backend', 'backend.ok', 'fixture', '{}')",
                &[SqliteValue::Integer(index as i64)],
            ).expect("unrelated events");
        }
        store.rebuild_diarization_index().expect("fixture indexes");
        connection.execute("COMMIT;").expect("commit fixture");
    }

    fn delta_fixture(root: &Path, compression: CompressionMode) -> (PathBuf, PathBuf) {
        let (source, _) = fixture(root, true);
        add_unrelated_aggregate(&source);
        let cursor = root.join("incremental-state");
        let baseline = root.join("baseline");
        export_incremental(&source, &baseline, &cursor).expect("baseline and cursor");
        let target = root.join("target.sqlite3");
        crate::sync::import(&target, &baseline, &root.join("import-state"), ConflictPolicy::Reject)
            .expect("restore baseline");
        let writer = Connection::open(source.display().to_string()).expect("source writer");
        writer.execute(
            "UPDATE segments SET text = 'edited child' WHERE run_id = 'verify-sortformer'"
        ).expect("child-only mutation");
        let delta = root.join("delta");
        let (manifest, _) = export_incremental_with_compression(
            &source, &delta, &cursor, compression,
        ).expect("real delta export");
        assert_eq!(manifest.row_counts.runs, 1);
        assert_eq!(manifest.row_counts.events, 0);
        crate::sync::import(
            &target, &delta, &root.join("import-state"), ConflictPolicy::OverwriteStrict,
        ).expect("apply child-only delta");
        (target, delta)
    }

    #[test]
    fn incremental_verification_scopes_before_diagnostic_truncation_plain_and_gzip() {
        for compression in [CompressionMode::None, CompressionMode::Gzip] {
            let dir = tempfile::tempdir().expect("tempdir");
            let (target, delta) = delta_fixture(dir.path(), compression);
            let verified = verify_archive(&delta).expect("verified delta");
            assert_eq!(verified.archive_mode, ArchiveMode::Incremental);
            assert_eq!(verified.schema_version, "franken-whisper-archive-verification-v2");
            let old = validate_sync(&target, &delta).expect("old full-comparison control");
            assert!(!old.is_valid);
            assert!(old.child_mismatches_truncated);
            let report = crate::sync::validate_sync(&target, &delta).expect("scoped validation");
            assert!(report.is_valid, "{report:?}");
            assert_eq!(report.db_run_count, 1);
            assert_eq!(report.db_segment_count, 1);
            assert_eq!(report.db_event_count, 0);
            let writer = Connection::open(target.display().to_string()).expect("writer");
            writer.execute(
                "UPDATE segments SET speaker = 'corrupt' WHERE run_id = 'verify-sortformer'"
            ).expect("damage selected child");
            let report = crate::sync::validate_sync(&target, &delta).expect("detect selected damage");
            assert!(!report.is_valid);
            assert_eq!(report.child_mismatches.len(), 1);
            assert_eq!(report.child_mismatches[0].table, "segments");
            assert_eq!(report.child_mismatches[0].key, "verify-sortformer/0");
            assert!(!report.child_mismatches_truncated);
        }
    }

    #[test]
    fn incremental_verification_checks_selected_indexes_and_retains_one_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (target, delta) = delta_fixture(dir.path(), CompressionMode::None);
        let writer = Connection::open(target.display().to_string()).expect("writer");
        let report = validate_verified_sync_with_after_run_ids(&target, &delta, || {
            writer.execute(
                "UPDATE speaker_profile_summaries SET reliability = 0.123 WHERE run_id = 'verify-sortformer'"
            ).map_err(|error| FwError::Storage(error.to_string()))?;
            Ok(())
        }).expect("stable incremental snapshot");
        assert!(report.is_valid);
        let report = crate::sync::validate_sync(&target, &delta).expect("new snapshot");
        assert!(!report.is_valid);
        assert_eq!(report.child_mismatches.len(), 1);
        assert_eq!(report.child_mismatches[0].table, "speaker_profile_summaries");
    }

    #[test]
    fn incremental_scope_is_pinned_and_full_snapshots_still_compare_every_run() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (target, delta) = delta_fixture(dir.path(), CompressionMode::None);
        let report = validate_verified_sync_with_after_run_ids(&target, &delta, || {
            set_archive_mode(&delta, json!("full"));
            Ok(())
        }).expect("pinned mode");
        assert!(report.is_valid);
        let full = crate::sync::validate_sync(&target, &delta).expect("explicit full mode");
        assert!(!full.is_valid);
        assert_eq!(full.missing_from_jsonl, vec!["unrelated"]);
        let path = delta.join("manifest.json");
        let mut legacy: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).expect("manifest")).expect("JSON");
        legacy.as_object_mut().expect("object").remove("export_mode");
        fs::write(path, legacy.to_string()).expect("legacy full mode");
        assert_eq!(pin_import(&delta).expect("legacy mode").archive_mode, ArchiveMode::Full);
        assert!(!crate::sync::validate_sync(&target, &delta).expect("legacy comparison").is_valid);
    }

    #[test]
    fn malformed_archive_modes_fail_before_any_destination_access() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (_, archive) = fixture(dir.path(), false);
        for mode in [json!(null), json!(7), json!({"incremental": null}), json!("incrementall")] {
            set_archive_mode(&archive, mode);
            let target = dir.path().join("absent.sqlite3");
            let error = import_pinned_inner(&target, &archive, ConflictPolicy::Skip)
                .expect_err("invalid manifest mode");
            assert!(error.to_string().contains("export_mode"), "{error}");
            assert!(!target.exists());
            assert!(verify_archive(&archive).is_err());
        }
    }

    #[test]
    fn empty_incremental_validation_does_not_claim_the_database_is_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (source, _) = fixture(dir.path(), true);
        let cursor = dir.path().join("cursor");
        export_incremental(&source, &dir.path().join("baseline"), &cursor).expect("baseline");
        let delta = dir.path().join("empty-delta");
        let manifest = export_incremental(&source, &delta, &cursor).expect("empty delta");
        assert_eq!(manifest.row_counts.runs, 0);
        let report = crate::sync::validate_sync(&source, &delta).expect("empty scope");
        assert!(report.is_valid);
        assert_eq!(report.db_run_count, 0, "only scoped rows are counted");
        assert_eq!(RunStore::open(&source).expect("store").list_recent_runs(10).expect("runs").len(), 1);
    }

    #[test]
    fn incremental_deletion_verification_rejects_preserved_parents_and_orphan_rows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (source, _) = fixture(dir.path(), true);
        add_unrelated_aggregate(&source);
        let cursor = dir.path().join("cursor");
        let baseline = dir.path().join("baseline");
        export_incremental(&source, &baseline, &cursor).expect("baseline");
        let target = dir.path().join("target.sqlite3");
        let import_state = dir.path().join("import-state");
        crate::sync::import(&target, &baseline, &import_state, ConflictPolicy::Reject).expect("restore");
        let writer = Connection::open(source.display().to_string()).expect("writer");
        writer.execute("DELETE FROM segments WHERE run_id = 'verify-sortformer'").expect("delete child");
        writer.execute("DELETE FROM runs WHERE id = 'verify-sortformer'").expect("delete parent");
        let delta = dir.path().join("deletion");
        let manifest = export_incremental(&source, &delta, &cursor).expect("deletion export");
        assert_eq!(manifest.row_counts.runs, 0);
        assert_eq!(manifest.deleted_runs.expect("declared deletions").count, 1);
        crate::sync::import(&target, &delta, &import_state, ConflictPolicy::Skip).expect("skip deletion");
        let report = crate::sync::validate_sync(&target, &delta).expect("check skipped deletion");
        assert!(!report.is_valid);
        assert_eq!(report.missing_from_jsonl, vec!["verify-sortformer"]);
        crate::sync::import(&target, &delta, &import_state, ConflictPolicy::OverwriteStrict).expect("apply deletion");
        assert!(crate::sync::validate_sync(&target, &delta).expect("deleted aggregate").is_valid);

        let connection = Connection::open(target.display().to_string()).expect("target");
        connection.execute("PRAGMA foreign_keys=OFF").expect("permit deliberate orphan fixture");
        let mut tables = VERIFICATION_INDEXES.to_vec();
        tables.extend([
            ("segments", "run_id, idx, start_sec, end_sec, speaker, text, confidence", true),
            ("events", "run_id, seq, ts_rfc3339, stage, code, message, payload_json", true),
        ]);
        for (table, columns, _) in tables {
            if table == "speaker_hints" {
                connection.execute(
                    "INSERT INTO speaker_hints (run_id, idx, speaker_ref, start_ms, end_ms, confidence, policy, provenance) VALUES ('verify-sortformer', 0, 'orphan', 0, 50, 1.0, 'hard_must_link', 'fixture')"
                ).expect("orphan hint");
            } else {
                let (_, payload_columns) = columns.split_once(',').expect("column list");
                connection.execute(&format!(
                    "INSERT INTO {table} ({columns}) SELECT 'verify-sortformer', {payload_columns} FROM {table} WHERE run_id = 'unrelated'"
                )).expect("orphan rows copied from unrelated run");
            }
            let report = crate::sync::validate_sync(&target, &delta).expect("check orphan");
            assert!(!report.is_valid, "orphan {table} must fail even without its parent");
            assert!(report.child_mismatches.iter().any(|row| row.table == table));
            connection.execute(&format!("DELETE FROM {table} WHERE run_id = 'verify-sortformer'"))
                .expect("remove only deliberate orphan fixture");
        }
        assert!(crate::sync::validate_sync(&target, &delta).expect("cleaned fixture").is_valid);
    }

    #[test]
    fn scoped_queries_bind_ids_across_parameter_chunks_without_matching_other_runs() {
        let connection = Connection::open(":memory:").expect("connection");
        connection.execute("CREATE TABLE runs (id TEXT PRIMARY KEY, value INTEGER)").expect("table");
        let tricky = "quoted' OR 1=1 --";
        for id in ["selected", "outside", tricky] {
            connection.execute_with_params("INSERT INTO runs VALUES (?1, 42)", &[SqliteValue::Text(id.into())])
                .expect("row");
        }
        let mut scope = vec!["selected".to_owned()];
        scope.extend((0..513).map(|index| format!("absent-{index}")));
        scope.push(tricky.to_owned());
        let rows = verification_rows_in_scope(&connection, "runs", "id, value", false, Some(&scope))
            .expect("bound chunked query");
        assert_eq!(rows.len(), 2);
        assert!(rows.contains_key(&(tricky.to_owned(), 0)));
        assert!(!rows.contains_key(&("outside".to_owned(), 0)));
        assert!(verification_rows_in_scope(&connection, "runs", "id, value", false, Some(&[]))
            .expect("empty scope").is_empty());
    }
}
