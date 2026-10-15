use std::path::{Path, PathBuf};
use std::{fs, io::Write};

use fsqlite_types::value::SqliteValue;
use serde_json::{Value, json};
use tempfile::tempdir;

use franken_whisper::diarization::{
    ACOUSTIC_DIARIZATION_CONTRACT_VERSION, AcousticBoundaryHints, AcousticDiarizationInput,
    SORTFORMER_DIARIZATION_CONTRACT_VERSION, diarize_acoustic_pcm,
};
use franken_whisper::model::{
    BackendKind, BackendParams, DiarizationEngine, DiarizationFallbackStatus, DiarizationReport,
    DiarizationRequest, DiarizationSpeakerEvidenceMode, DiarizationTurn, InputSource,
    KnownSpeakerInterval, KnownSpeakerPolicy, RunEvent, RunReport, SpeakerCountOutcome,
    SpeakerCountOutcomeReason, SpeakerCountOutcomeStatus, SpeakerCountRequest,
    SpeakerEvidenceReason, SpeakerEvidenceSummary, SpeakerProfileSummary, TranscribeRequest,
    TranscriptionResult, TranscriptionSegment,
};
use franken_whisper::storage::{BlockingConnection, RunStore};
use franken_whisper::sync::{self, ConflictPolicy};

fn fixture_report(id: &str, db_path: &Path) -> RunReport {
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
            pcm_sha256: None,
            live_note: None,
        },
    }
}

fn mutate_export_runs_transcript(export_dir: &Path, transcript: &str) {
    let runs_path = export_dir.join("runs.jsonl");
    let first_line = fs::read_to_string(&runs_path)
        .expect("runs.jsonl should be readable")
        .lines()
        .next()
        .expect("runs snapshot should include at least one row")
        .to_owned();

    let mut mutated: Value = serde_json::from_str(&first_line).expect("valid jsonl row");
    mutated["transcript"] = json!(transcript);
    if let Some(result_json) = mutated.get("result_json").and_then(|v| v.as_str()) {
        let mut parsed_result: Value =
            serde_json::from_str(result_json).expect("result_json should parse");
        parsed_result["transcript"] = json!(transcript);
        mutated["result_json"] =
            json!(serde_json::to_string(&parsed_result).expect("serialize result_json"));
    }
    fs::write(
        &runs_path,
        format!("{}\n", serde_json::to_string(&mutated).expect("serialize row")),
    )
    .expect("write runs snapshot");
    refresh_runs_checksum(export_dir);
}

fn refresh_runs_checksum(export_dir: &Path) {
    let manifest_path = export_dir.join("manifest.json");
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).expect("manifest"))
            .expect("manifest json");
    manifest["checksums"]["runs_jsonl_sha256"] =
        json!(sha256_file(&export_dir.join("runs.jsonl")));
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).expect("manifest serialize"),
    )
    .expect("manifest rewrite");
}

fn sha256_file(path: &Path) -> String {
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
    fs::write(export_dir.join("events.jsonl"), "tampered\n").expect("tamper");

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
    assert!(conflicts_path.exists(), "conflict artifact should be written");
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
    let import_result =
        sync::import(&rebuilt_db, &export_dir, &state_root, ConflictPolicy::Reject)
            .expect("import");

    assert_eq!(import_result.runs_imported, manifest.row_counts.runs);
    assert_eq!(import_result.segments_imported, manifest.row_counts.segments);
    assert_eq!(import_result.events_imported, manifest.row_counts.events);

    let rebuilt_store = RunStore::open(&rebuilt_db).expect("rebuilt store");
    let runs = rebuilt_store.list_recent_runs(10).expect("list rebuilt runs");
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

// Issue #19 concerns persistence/recovery of a runtime output contract, not
// neural inference. This model-free fixture uses the same valid Sortformer
// DTO as the storage rebuild regression, with the compact default request
// (diarize=true, no explicitly configured engine). No schema is relabeled as
// acoustic to make the import pass.
fn sortformer_report(id: &str, db_path: &Path) -> RunReport {
    let mut report = fixture_report(id, db_path);
    report.request.diarize = true;
    report.result.transcript = "hello".to_owned();
    report.result.segments[0].end_sec = Some(6.0);
    report.result.segments[0].speaker = Some("SPEAKER_00".to_owned());
    report.result.diarization = Some(DiarizationReport {
        implementation: "native-sortformer-v1".to_owned(),
        contract_version: SORTFORMER_DIARIZATION_CONTRACT_VERSION.to_owned(),
        feature_schema: "sortformer-activity-80ms-v1".to_owned(),
        speaker_evidence_mode: DiarizationSpeakerEvidenceMode::SortformerActivity,
        normalized_input_sha256:
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_owned(),
        hint_document_sha256: None,
        turns: vec![DiarizationTurn {
            start_ms: 0,
            end_ms: 6_000,
            speaker_ref: Some("SPEAKER_00".to_owned()),
            speaker_confidence: Some(0.9),
            change_confidence: None,
            overlap_suspected: false,
            hard_hint_attributed: false,
        }],
        profiles: vec![SpeakerProfileSummary {
            speaker_ref: "SPEAKER_00".to_owned(),
            frame_count: 75,
            voiced_duration_ms: 6_000,
            reliability: 0.9,
            voice_profile_count: 1,
            channel_profile_count: 0,
            training_accepted_count: 1,
            training_downweighted_count: 0,
            training_quarantined_count: 0,
            anchored: false,
            soft_hint_contradiction: None,
        }],
        hint_evidence: Vec::new(),
        speaker_queries: Vec::new(),
        speaker_count: SpeakerCountOutcome {
            request: SpeakerCountRequest::Infer,
            estimate: None,
            status: SpeakerCountOutcomeStatus::Unresolved,
            supported_speaker_count: 1,
            active_speaker_refs: vec!["SPEAKER_00".to_owned()],
            dominant_speaker_share: 1.0,
            unknown_voiced_share: 0.0,
            reasons: vec![SpeakerCountOutcomeReason::SpeakerCountEvidenceUnresolved],
            speaker_evidence: vec![SpeakerEvidenceSummary {
                speaker_ref: "SPEAKER_00".to_owned(),
                assigned_tracklet_count: 1,
                independent_tracklet_count: 1,
                recurrence_episode_count: 1,
                voiced_frame_count: 75,
                independent_voiced_frame_count: 75,
                voiced_duration_ms: 6_000,
                mean_assignment_confidence: 0.9,
                profile_reliability: 0.9,
                hard_anchored: false,
                separated_from_supported_speakers: true,
                reasons: vec![SpeakerEvidenceReason::SupportedByLearnedModelActivity],
                supported: true,
            }],
        },
        fallback_status: DiarizationFallbackStatus::SpeakerCountUnresolved,
        operational_partition: None,
        neural_representation: None,
        speaker_segments: Vec::new(),
        diagnostics: Vec::new(),
    });
    report
}

fn acoustic_report(id: &str, db_path: &Path) -> RunReport {
    use sha2::{Digest, Sha256};

    let mut report = fixture_report(id, db_path);
    report.request.diarize = true;
    report.result.transcript = "hello".to_owned();
    let request = DiarizationRequest {
        engine: DiarizationEngine::Acoustic,
        known_intervals: vec![KnownSpeakerInterval {
            speaker_ref: "enrolled_speaker".to_owned(),
            start_ms: 0,
            end_ms: 1_000,
            confidence: 0.95,
            policy: KnownSpeakerPolicy::HardMustLink,
            provenance: Some("public-synthetic-recovery-fixture".to_owned()),
        }],
        ..DiarizationRequest::default()
    };
    // Exercise the real acoustic producer, including its versioned count
    // estimate and enrollment digest, rather than inventing those contracts.
    let samples: Vec<f32> = (0..16_000)
        .map(|index| {
            let phase = std::f32::consts::TAU * 220.0 * index as f32 / 16_000.0;
            0.4 * phase.sin() + 0.2 * (2.0 * phase).sin()
        })
        .collect();
    let mut hasher = Sha256::new();
    for sample in &samples {
        hasher.update(sample.to_le_bytes());
    }
    let input_digest = format!("{:x}", hasher.finalize());
    let boundaries = AcousticBoundaryHints::default();
    let (diarization, projection) = diarize_acoustic_pcm(
        AcousticDiarizationInput {
            samples: &samples,
            normalized_input_sha256: &input_digest,
            segments: &report.result.segments,
            word_aligned: false,
            request: &request,
            boundary_hints: &boundaries,
        },
        || false,
    )
    .expect("produce the acoustic recovery fixture");
    assert_eq!(diarization.contract_version, ACOUSTIC_DIARIZATION_CONTRACT_VERSION);
    assert!(diarization.speaker_count.estimate.is_some());
    assert!(diarization.hint_document_sha256.is_some());
    report.request.backend_params.acoustic_diarization = Some(request);
    report.result.segments = projection.segments;
    report.result.diarization = Some(diarization);
    report
}

const RECOVERY_TABLES: [(&str, &str); 7] = [
    ("runs", "id"),
    ("segments", "run_id, idx"),
    ("events", "run_id, seq"),
    ("diarization_reports", "run_id"),
    ("diarization_turns", "run_id, idx"),
    ("speaker_hints", "run_id, idx"),
    ("speaker_profile_summaries", "run_id, idx"),
];

// Compare values, not database file bytes (fresh databases have independent
// local sync identities and page layouts). Include every column, so loss of a
// provenance field or a stale derived row cannot hide behind row-count parity.
fn table_snapshot(db_path: &Path, tables: &[(&str, &str)]) -> Vec<Vec<Vec<SqliteValue>>> {
    let connection = BlockingConnection::open(db_path.display().to_string()).expect("snapshot DB");
    tables
        .iter()
        .map(|(table, order)| {
            let width = connection
                .query(&format!("PRAGMA table_info({table});"))
                .expect("table columns")
                .len();
            assert!(width > 0, "missing recovery table {table}");
            connection
                .query(&format!("SELECT * FROM {table} ORDER BY {order};"))
                .expect("snapshot table")
                .iter()
                .map(|row| {
                    (0..width)
                        .map(|column| row.get(column).expect("selected column").clone())
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn assert_diarized_round_trip(make_report: fn(&str, &Path) -> RunReport) {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let restored_db = dir.path().join("fresh.sqlite3");
    let export_dir = dir.path().join("snapshot");
    let reexport_dir = dir.path().join("restored-snapshot");
    let state_root = dir.path().join("state");
    let report = make_report("contract-round-trip", &source_db);
    let request = report
        .request
        .backend_params
        .acoustic_diarization
        .clone()
        .unwrap_or_default();
    let expected_diarization = report.result.diarization.as_ref().expect("fixture report");
    expected_diarization
        .validate_against_request(&request, None)
        .expect("fixture must satisfy its actual request, not just deserialize");
    let source = RunStore::open(&source_db).expect("source");
    source.persist_report(&report).expect("persist canonical report");
    drop(source);
    let source_tables = table_snapshot(&source_db, &RECOVERY_TABLES);
    assert_eq!(source_tables[3].len(), 1, "a derived report must exist");
    assert_eq!(source_tables[4].len(), expected_diarization.turns.len());
    assert_eq!(source_tables[5].len(), request.known_intervals.len());
    assert_eq!(source_tables[6].len(), expected_diarization.profiles.len());

    let manifest = sync::export(&source_db, &export_dir, &state_root).expect("export");
    assert!(!restored_db.exists(), "exercise creation of a fresh database");
    let imported = sync::import(
        &restored_db,
        &export_dir,
        &state_root,
        ConflictPolicy::Reject,
    )
    .expect("same-version canonical diarization must round-trip");
    assert_eq!(imported.runs_imported, manifest.row_counts.runs);
    assert_eq!(imported.segments_imported, manifest.row_counts.segments);
    assert_eq!(imported.events_imported, manifest.row_counts.events);
    assert_eq!(imported.runs_deleted, 0);
    assert!(imported.conflicts.is_empty());
    assert_eq!(table_snapshot(&restored_db, &RECOVERY_TABLES), source_tables);

    let restored = RunStore::open(&restored_db).expect("restored store");
    assert_eq!(restored.query_integrity_check(), "ok");
    let details = restored
        .load_run_details(&report.run_id)
        .expect("load restored details")
        .expect("original run identity must survive");
    assert_eq!(details.run_id, report.run_id);
    assert_eq!(details.diarization.as_ref(), Some(expected_diarization));
    assert_eq!(
        serde_json::to_value(&details.segments).expect("restored segments"),
        serde_json::to_value(&report.result.segments).expect("source segments")
    );
    assert_eq!(
        serde_json::to_value(&details.events).expect("restored events"),
        serde_json::to_value(&report.events).expect("source events")
    );
    assert_eq!(
        serde_json::to_value(&details.replay).expect("restored replay"),
        serde_json::to_value(&report.replay).expect("source replay")
    );
    assert_eq!(
        restored.stored_request_json(&report.run_id).expect("request"),
        Some(serde_json::to_string(&report.request).expect("canonical request"))
    );
    drop(restored);

    // A second import is idempotent, including the reconstructed indexes.
    sync::import(&restored_db, &export_dir, &state_root, ConflictPolicy::Reject)
        .expect("idempotent re-import");
    assert_eq!(table_snapshot(&restored_db, &RECOVERY_TABLES), source_tables);
    let reexport = sync::export(&restored_db, &reexport_dir, &state_root).expect("re-export");
    assert_eq!(reexport.row_counts, manifest.row_counts);
    assert_eq!(reexport.checksums, manifest.checksums);
    for channel in ["runs.jsonl", "segments.jsonl", "events.jsonl"] {
        assert_eq!(
            fs::read(export_dir.join(channel)).expect("source channel"),
            fs::read(reexport_dir.join(channel)).expect("restored channel"),
            "canonical {channel} bytes changed during recovery"
        );
    }
    for (table, _) in &RECOVERY_TABLES[3..] {
        assert!(
            !export_dir.join(format!("{table}.jsonl")).exists(),
            "derived indexes must not become a second export authority"
        );
    }
}

#[test]
fn recovery_smoke_default_sortformer_round_trip_preserves_canonical_and_derived_state() {
    let default = DiarizationRequest::default();
    assert_eq!(default.engine, DiarizationEngine::Auto);
    assert_diarized_round_trip(sortformer_report);
}

#[test]
fn recovery_smoke_acoustic_round_trip_preserves_enrollment_provenance() {
    assert_diarized_round_trip(acoustic_report);
}

type CanonicalMutation = fn(&mut Value, &mut Value);

fn mutate_canonical_run(export_dir: &Path, run_id: &str, mutate: CanonicalMutation) {
    let path = export_dir.join("runs.jsonl");
    let mut rows: Vec<Value> = fs::read_to_string(&path)
        .expect("runs channel")
        .lines()
        .map(|line| serde_json::from_str(line).expect("run row"))
        .collect();
    let row = rows
        .iter_mut()
        .find(|row| row["id"].as_str() == Some(run_id))
        .expect("selected canonical run");
    let mut request: Value =
        serde_json::from_str(row["request_json"].as_str().expect("request string"))
            .expect("request JSON");
    let mut result: Value =
        serde_json::from_str(row["result_json"].as_str().expect("result string"))
            .expect("result JSON");
    mutate(&mut request, &mut result);
    row["request_json"] = json!(serde_json::to_string(&request).expect("serialize request"));
    row["result_json"] = json!(serde_json::to_string(&result).expect("serialize result"));
    let mut bytes = Vec::new();
    for row in rows {
        writeln!(&mut bytes, "{}", serde_json::to_string(&row).expect("serialize row"))
            .expect("write row");
    }
    fs::write(path, bytes).expect("write mutated canonical channel");
    // These are semantically invalid, not transport-corrupt, snapshots.
    refresh_runs_checksum(export_dir);
}

#[test]
fn recovery_smoke_malformed_diarization_contracts_roll_back_fresh_and_existing_databases() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("existing.sqlite3");
    let export_dir = dir.path().join("snapshot");
    let state_root = dir.path().join("state");
    let source = RunStore::open(&source_db).expect("source");
    source
        .persist_report(&sortformer_report("a-valid-new", &source_db))
        .expect("valid earlier run");
    source
        .persist_report(&sortformer_report("z-invalid", &source_db))
        .expect("later run before mutation");
    drop(source);
    sync::export(&source_db, &export_dir, &state_root).expect("export");
    let canonical_runs = fs::read(export_dir.join("runs.jsonl")).expect("baseline channel");

    let target = RunStore::open(&target_db).expect("existing target");
    let mut existing = sortformer_report("z-invalid", &target_db);
    existing.result.transcript = "preserve the previous version".to_owned();
    existing.result.segments[0].text = existing.result.transcript.clone();
    target.persist_report(&existing).expect("existing conflicting run");
    target
        .persist_report(&acoustic_report("retained-acoustic", &target_db))
        .expect("unrelated run and enrollment provenance");
    drop(target);
    let before = table_snapshot(&target_db, &RECOVERY_TABLES);
    let local_tables = [("_meta", "key"), ("sync_run_mutations", "run_id")];
    let local_before = table_snapshot(&target_db, &local_tables);

    let mutations: &[(&str, CanonicalMutation)] = &[
        ("unknown contract", |_, result| {
            result["diarization"]["contract_version"] = json!("sortformer-diarization-v999");
        }),
        ("missing count contract", |_, result| {
            result["diarization"].as_object_mut().expect("report object").remove("speaker_count");
        }),
        ("wrong feature schema", |_, result| {
            result["diarization"]["feature_schema"] = json!("acoustic-feature-v2");
        }),
        ("wrong evidence mode", |_, result| {
            result["diarization"]["speaker_evidence_mode"] =
                serde_json::to_value(DiarizationSpeakerEvidenceMode::AcousticV2).expect("mode");
        }),
        ("invalid input digest", |_, result| {
            result["diarization"]["normalized_input_sha256"] = json!("not-a-sha256");
        }),
        ("reversed turn", |_, result| {
            result["diarization"]["turns"][0]["start_ms"] = json!(7_000);
        }),
        ("invalid confidence", |_, result| {
            result["diarization"]["turns"][0]["speaker_confidence"] = json!(1.1);
        }),
        ("unknown report field", |_, result| {
            result["diarization"]["unrecognized_contract_field"] = json!(true);
        }),
        ("disabled canonical request", |request, _| {
            request["diarize"] = json!(false);
        }),
        ("incompatible requested engine", |request, _| {
            request["backend_params"]["acoustic_diarization"] =
                serde_json::to_value(DiarizationRequest {
                    engine: DiarizationEngine::Acoustic,
                    ..DiarizationRequest::default()
                })
                .expect("explicit acoustic request");
        }),
    ];
    for (index, (name, mutate)) in mutations.iter().enumerate() {
        fs::write(export_dir.join("runs.jsonl"), &canonical_runs).expect("reset baseline");
        mutate_canonical_run(&export_dir, "z-invalid", *mutate);
        let manifest: Value =
            serde_json::from_slice(&fs::read(export_dir.join("manifest.json")).expect("manifest"))
                .expect("manifest JSON");
        assert_eq!(
            manifest["checksums"]["runs_jsonl_sha256"],
            json!(sha256_file(&export_dir.join("runs.jsonl")))
        );
        let fresh_db = dir.path().join(format!("fresh-invalid-{index}.sqlite3"));
        for (db, policy) in [
            (&fresh_db, ConflictPolicy::Reject),
            (&target_db, ConflictPolicy::OverwriteStrict),
        ] {
            let error = sync::import(db, &export_dir, &state_root, policy)
                .expect_err(name)
                .to_string();
            assert!(error.contains("z-invalid"), "{name}: {error}");
            assert!(!error.contains("checksum mismatch"), "{name}: {error}");
            assert_eq!(RunStore::open(db).expect("reopen after rejection").query_integrity_check(), "ok");
        }
        assert!(
            table_snapshot(&fresh_db, &RECOVERY_TABLES).iter().all(Vec::is_empty),
            "{name}: a failed fresh import committed partial canonical or derived rows"
        );
        assert_eq!(table_snapshot(&target_db, &RECOVERY_TABLES), before, "{name}");
        assert_eq!(table_snapshot(&target_db, &local_tables), local_before, "{name}");
        assert!(!state_root.join("locks/sync.lock").exists(), "{name}: lock leaked");
    }

    // Rejection must not poison the connection state, lock, or recovery path.
    fs::write(export_dir.join("runs.jsonl"), canonical_runs).expect("restore valid snapshot");
    refresh_runs_checksum(&export_dir);
    sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::OverwriteStrict)
        .expect("valid retry after contract rejection");
    assert!(
        RunStore::open(&target_db)
            .expect("target after retry")
            .load_run_details("a-valid-new")
            .expect("lookup new run")
            .is_some()
    );
}

#[test]
fn recovery_smoke_checksums_protect_every_sortformer_snapshot_channel() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("source.sqlite3");
    let target_db = dir.path().join("existing.sqlite3");
    let export_dir = dir.path().join("snapshot");
    let state_root = dir.path().join("state");
    let source = RunStore::open(&source_db).expect("source");
    source.persist_report(&sortformer_report("checksum-source", &source_db)).expect("persist");
    drop(source);
    let target = RunStore::open(&target_db).expect("target");
    target.persist_report(&sortformer_report("checksum-retained", &target_db)).expect("retain");
    drop(target);
    let before = table_snapshot(&target_db, &RECOVERY_TABLES);
    sync::export(&source_db, &export_dir, &state_root).expect("export");

    for (index, channel) in ["runs.jsonl", "segments.jsonl", "events.jsonl"].iter().enumerate() {
        let path = export_dir.join(channel);
        let original = fs::read(&path).expect("channel");
        let mut altered = original.clone();
        // Keep the JSONL syntactically valid; only its authenticated bytes change.
        altered.extend_from_slice(b"\n");
        fs::write(&path, altered).expect("tamper channel");
        let fresh_db = dir.path().join(format!("checksum-fresh-{index}.sqlite3"));
        for db in [&fresh_db, &target_db] {
            let error = sync::import(db, &export_dir, &state_root, ConflictPolicy::Reject)
                .expect_err("checksum failure must precede import")
                .to_string();
            assert!(error.contains("checksum mismatch"), "{channel}: {error}");
            assert!(error.contains(channel), "{channel}: {error}");
        }
        assert!(!fresh_db.exists(), "{channel}: checksum failure created a target DB");
        assert_eq!(table_snapshot(&target_db, &RECOVERY_TABLES), before);
        assert!(!state_root.join("locks/sync.lock").exists());
        fs::write(path, original).expect("restore channel");
    }
    sync::import(&target_db, &export_dir, &state_root, ConflictPolicy::Reject)
        .expect("valid snapshot after checksum failures");
}

#[test]
fn recovery_smoke_empty_snapshot_restores_a_fresh_database_without_deleting_existing_runs() {
    let dir = tempdir().expect("tempdir");
    let source_db = dir.path().join("empty-source.sqlite3");
    let fresh_db = dir.path().join("empty-restored.sqlite3");
    let existing_db = dir.path().join("existing.sqlite3");
    let export_dir = dir.path().join("empty-snapshot");
    let state_root = dir.path().join("state");
    drop(RunStore::open(&source_db).expect("empty source"));
    let manifest = sync::export(&source_db, &export_dir, &state_root).expect("empty export");
    assert_eq!(manifest.row_counts.runs, 0);
    assert_eq!(manifest.row_counts.segments, 0);
    assert_eq!(manifest.row_counts.events, 0);
    for channel in ["runs.jsonl", "segments.jsonl", "events.jsonl"] {
        assert!(fs::read(export_dir.join(channel)).expect("empty channel").is_empty());
    }
    let empty_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    assert_eq!(manifest.checksums.runs_jsonl_sha256, empty_sha256);
    assert_eq!(manifest.checksums.segments_jsonl_sha256, empty_sha256);
    assert_eq!(manifest.checksums.events_jsonl_sha256, empty_sha256);
    assert!(!fresh_db.exists());
    let imported = sync::import(&fresh_db, &export_dir, &state_root, ConflictPolicy::Reject)
        .expect("empty fresh restore");
    assert_eq!(imported.runs_imported, 0);
    assert_eq!(imported.segments_imported, 0);
    assert_eq!(imported.events_imported, 0);
    assert_eq!(imported.runs_deleted, 0);
    assert!(imported.conflicts.is_empty());
    assert!(table_snapshot(&fresh_db, &RECOVERY_TABLES).iter().all(Vec::is_empty));
    assert_eq!(RunStore::open(&fresh_db).expect("fresh store").query_integrity_check(), "ok");

    let existing = RunStore::open(&existing_db).expect("existing store");
    existing.persist_report(&sortformer_report("retain-on-empty", &existing_db)).expect("persist");
    drop(existing);
    let before = table_snapshot(&existing_db, &RECOVERY_TABLES);
    sync::import(&existing_db, &export_dir, &state_root, ConflictPolicy::OverwriteStrict)
        .expect("an empty snapshot does not imply deletion");
    assert_eq!(table_snapshot(&existing_db, &RECOVERY_TABLES), before);
}
