//! Deterministic replay artifact pack (K13.6).
//!
//! Generates a self-contained directory of artifacts that enable reproducible
//! analysis of a pipeline run:
//!
//! - `env.json` — runtime environment snapshot (OS, arch, backend identity/version)
//! - `manifest.json` — file inventory with content hashes, timestamps, and trace
//! - `repro.lock` — frozen decision state: routing parameters, loss matrix hash,
//!   fallback policy configuration
//! - `tolerance_manifest.json` — canonical conformance tolerance and rollout mode

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::conformance::{NativeEngineRolloutStage, SegmentCompatibilityTolerance};
use crate::error::FwResult;
use crate::model::RunReport;

/// Environment snapshot captured at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvSnapshot {
    pub os: String,
    pub arch: String,
    pub backend_identity: Option<String>,
    pub backend_version: Option<String>,
    pub franken_whisper_version: String,
}

/// Manifest listing all artifact files and content hashes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackManifest {
    pub trace_id: String,
    pub run_id: String,
    pub started_at: String,
    pub finished_at: String,
    pub input_content_hash: Option<String>,
    pub output_payload_hash: Option<String>,
    pub segment_count: usize,
    pub event_count: usize,
    pub evidence_count: usize,
}

/// Frozen decision state for routing reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproLock {
    pub routing_evidence: Vec<Value>,
    pub replay_envelope: Value,
    pub backend_requested: String,
    pub diarize: bool,
}

/// Frozen conformance and rollout parameters needed to replay compatibility
/// comparisons with the same tolerance contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceManifest {
    pub schema_version: String,
    pub timestamp_tolerance_sec: f64,
    pub require_text_exact: bool,
    pub require_speaker_exact: bool,
    pub native_rollout_stage: String,
    pub segment_count: usize,
    pub event_count: usize,
}

/// Build the env.json snapshot from a completed report.
pub fn build_env_snapshot(report: &RunReport) -> EnvSnapshot {
    EnvSnapshot {
        os: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
        backend_identity: report.replay.backend_identity.clone(),
        backend_version: report.replay.backend_version.clone(),
        franken_whisper_version: env!("CARGO_PKG_VERSION").to_owned(),
    }
}

/// Build the manifest.json from a completed report.
pub fn build_manifest(report: &RunReport) -> PackManifest {
    PackManifest {
        trace_id: report.trace_id.clone(),
        run_id: report.run_id.clone(),
        started_at: report.started_at_rfc3339.clone(),
        finished_at: report.finished_at_rfc3339.clone(),
        input_content_hash: report.replay.input_content_hash.clone(),
        output_payload_hash: report.replay.output_payload_hash.clone(),
        segment_count: report.result.segments.len(),
        event_count: report.events.len(),
        evidence_count: report.evidence.len(),
    }
}

/// Build the repro.lock from a completed report.
pub fn build_repro_lock(report: &RunReport) -> ReproLock {
    ReproLock {
        routing_evidence: report.evidence.clone(),
        replay_envelope: serde_json::to_value(&report.replay).unwrap_or(json!(null)),
        backend_requested: report.request.backend.as_str().to_owned(),
        diarize: report.request.diarize,
    }
}

fn latest_event_payload_value<'a>(
    report: &'a RunReport,
    code: &str,
    key: &str,
) -> Option<&'a Value> {
    report
        .events
        .iter()
        .filter(|event| event.code == code)
        .max_by_key(|event| event.seq)
        .and_then(|event| event.payload.get(key))
}

fn rollout_stage_from_report(report: &RunReport) -> String {
    latest_event_payload_value(report, "backend.ok", "native_rollout_stage")
        .or_else(|| {
            latest_event_payload_value(
                report,
                "backend.routing.decision_contract",
                "native_rollout_stage",
            )
        })
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| NativeEngineRolloutStage::Primary.as_str().to_owned())
}

/// Build the tolerance manifest from a completed report.
pub fn build_tolerance_manifest(report: &RunReport) -> ToleranceManifest {
    let tolerance = SegmentCompatibilityTolerance::default();
    ToleranceManifest {
        schema_version: "tolerance-manifest-v1".to_owned(),
        timestamp_tolerance_sec: tolerance.timestamp_tolerance_sec,
        require_text_exact: tolerance.require_text_exact,
        require_speaker_exact: tolerance.require_speaker_exact,
        native_rollout_stage: rollout_stage_from_report(report),
        segment_count: report.result.segments.len(),
        event_count: report.events.len(),
    }
}

/// Write a complete replay artifact pack to disk.
pub fn write_replay_pack(report: &RunReport, output_dir: &Path) -> FwResult<()> {
    std::fs::create_dir_all(output_dir)?;

    let env_snapshot = build_env_snapshot(report);
    let manifest = build_manifest(report);
    let repro_lock = build_repro_lock(report);
    let tolerance_manifest = build_tolerance_manifest(report);

    std::fs::write(
        output_dir.join("env.json"),
        serde_json::to_string_pretty(&env_snapshot)?,
    )?;
    std::fs::write(
        output_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    std::fs::write(
        output_dir.join("repro.lock"),
        serde_json::to_string_pretty(&repro_lock)?,
    )?;
    std::fs::write(
        output_dir.join("tolerance_manifest.json"),
        serde_json::to_string_pretty(&tolerance_manifest)?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use crate::model::{
        BackendKind, BackendParams, InputSource, ReplayEnvelope, RunReport, TranscriptionResult,
        TranscriptionSegment,
    };

    use super::{
        build_env_snapshot, build_manifest, build_repro_lock, build_tolerance_manifest,
        write_replay_pack,
    };

    fn fixture_report() -> RunReport {
        RunReport {
            run_id: "pack-run-1".to_owned(),
            trace_id: "aabbccddaabbccddaabbccddaabbccdd".to_owned(),
            started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-22T00:00:05Z".to_owned(),
            input_path: "input.wav".to_owned(),
            normalized_wav_path: "normalized.wav".to_owned(),
            request: crate::model::TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("input.wav"),
                },
                backend: BackendKind::Auto,
                model: None,
                language: None,
                translate: false,
                diarize: true,
                persist: false,
                db_path: PathBuf::from("db.sqlite3"),
                timeout_ms: None,
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "hello replay".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![
                    TranscriptionSegment {
                        start_sec: Some(0.0),
                        end_sec: Some(1.0),
                        text: "hello".to_owned(),
                        speaker: None,
                        confidence: Some(0.9),
                    },
                    TranscriptionSegment {
                        start_sec: Some(1.0),
                        end_sec: Some(2.0),
                        text: "replay".to_owned(),
                        speaker: None,
                        confidence: Some(0.85),
                    },
                ],
                acceleration: None,
                diarization: None,
                raw_output: json!({}),
                artifact_paths: vec![],
            },
            events: vec![],
            warnings: vec![],
            evidence: vec![json!({"routing": "test"})],
            replay: ReplayEnvelope {
                input_content_hash: Some("input-sha256".to_owned()),
                backend_identity: Some("whisper-cli".to_owned()),
                backend_version: Some("whisper 1.7.2".to_owned()),
                output_payload_hash: Some("output-sha256".to_owned()),
            },
        }
    }

    #[test]
    fn env_snapshot_captures_platform() {
        let report = fixture_report();
        let env = build_env_snapshot(&report);

        assert!(!env.os.is_empty());
        assert!(!env.arch.is_empty());
        assert_eq!(env.backend_identity.as_deref(), Some("whisper-cli"));
        assert_eq!(env.backend_version.as_deref(), Some("whisper 1.7.2"));
        assert!(!env.franken_whisper_version.is_empty());
    }

    #[test]
    fn manifest_captures_hashes_and_counts() {
        let report = fixture_report();
        let manifest = build_manifest(&report);

        assert_eq!(manifest.trace_id, "aabbccddaabbccddaabbccddaabbccdd");
        assert_eq!(manifest.run_id, "pack-run-1");
        assert_eq!(manifest.input_content_hash.as_deref(), Some("input-sha256"));
        assert_eq!(
            manifest.output_payload_hash.as_deref(),
            Some("output-sha256")
        );
        assert_eq!(manifest.segment_count, 2);
        assert_eq!(manifest.evidence_count, 1);
    }

    #[test]
    fn repro_lock_captures_decision_state() {
        let report = fixture_report();
        let lock = build_repro_lock(&report);

        assert_eq!(lock.backend_requested, "auto");
        assert!(lock.diarize);
        assert_eq!(lock.routing_evidence.len(), 1);
        assert!(
            lock.replay_envelope["input_content_hash"]
                .as_str()
                .is_some()
        );
    }

    #[test]
    fn write_replay_pack_creates_all_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack_dir = dir.path().join("replay_pack");
        let report = fixture_report();

        write_replay_pack(&report, &pack_dir).expect("write pack");

        assert!(pack_dir.join("env.json").exists());
        assert!(pack_dir.join("manifest.json").exists());
        assert!(pack_dir.join("repro.lock").exists());
        assert!(pack_dir.join("tolerance_manifest.json").exists());

        // Verify round-trip parseable
        let env_json = std::fs::read_to_string(pack_dir.join("env.json")).unwrap();
        let _env: super::EnvSnapshot = serde_json::from_str(&env_json).expect("valid env.json");

        let manifest_json = std::fs::read_to_string(pack_dir.join("manifest.json")).unwrap();
        let _manifest: super::PackManifest =
            serde_json::from_str(&manifest_json).expect("valid manifest.json");

        let repro_json = std::fs::read_to_string(pack_dir.join("repro.lock")).unwrap();
        let _repro: super::ReproLock = serde_json::from_str(&repro_json).expect("valid repro.lock");

        let tolerance_json =
            std::fs::read_to_string(pack_dir.join("tolerance_manifest.json")).unwrap();
        let _tolerance: super::ToleranceManifest =
            serde_json::from_str(&tolerance_json).expect("valid tolerance_manifest.json");
    }

    #[test]
    fn replay_pack_with_empty_replay_envelope() {
        let mut report = fixture_report();
        report.replay = ReplayEnvelope::default();
        report.evidence.clear();

        let env = build_env_snapshot(&report);
        assert!(env.backend_identity.is_none());

        let manifest = build_manifest(&report);
        assert!(manifest.input_content_hash.is_none());

        let lock = build_repro_lock(&report);
        assert!(lock.routing_evidence.is_empty());
    }

    #[test]
    fn write_replay_pack_overwrites_existing_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack_dir = dir.path().join("overwrite_test");

        let report = fixture_report();
        write_replay_pack(&report, &pack_dir).expect("first write");

        // Write again — should succeed (overwrite, not error).
        write_replay_pack(&report, &pack_dir).expect("second write should overwrite");

        // Files are still valid.
        let env_json = std::fs::read_to_string(pack_dir.join("env.json")).unwrap();
        let _: super::EnvSnapshot = serde_json::from_str(&env_json).expect("valid");
    }

    #[test]
    fn manifest_with_zero_segments_events_evidence() {
        let mut report = fixture_report();
        report.result.segments.clear();
        report.events.clear();
        report.evidence.clear();

        let manifest = build_manifest(&report);
        assert_eq!(manifest.segment_count, 0);
        assert_eq!(manifest.event_count, 0);
        assert_eq!(manifest.evidence_count, 0);
    }

    #[test]
    fn env_snapshot_serde_round_trip() {
        let report = fixture_report();
        let env = build_env_snapshot(&report);

        let json = serde_json::to_string(&env).expect("serialize");
        let parsed: super::EnvSnapshot = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.os, env.os);
        assert_eq!(parsed.arch, env.arch);
        assert_eq!(parsed.backend_identity, env.backend_identity);
        assert_eq!(parsed.backend_version, env.backend_version);
        assert_eq!(parsed.franken_whisper_version, env.franken_whisper_version);
    }

    #[test]
    fn pack_manifest_serde_round_trip() {
        let report = fixture_report();
        let manifest = build_manifest(&report);

        let json = serde_json::to_string(&manifest).expect("serialize");
        let parsed: super::PackManifest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.trace_id, manifest.trace_id);
        assert_eq!(parsed.run_id, manifest.run_id);
        assert_eq!(parsed.started_at, manifest.started_at);
        assert_eq!(parsed.finished_at, manifest.finished_at);
        assert_eq!(parsed.input_content_hash, manifest.input_content_hash);
        assert_eq!(parsed.output_payload_hash, manifest.output_payload_hash);
        assert_eq!(parsed.segment_count, manifest.segment_count);
        assert_eq!(parsed.event_count, manifest.event_count);
        assert_eq!(parsed.evidence_count, manifest.evidence_count);
    }

    #[test]
    fn repro_lock_serde_round_trip() {
        let report = fixture_report();
        let lock = build_repro_lock(&report);

        let json = serde_json::to_string(&lock).expect("serialize");
        let parsed: super::ReproLock = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.backend_requested, lock.backend_requested);
        assert_eq!(parsed.diarize, lock.diarize);
        assert_eq!(parsed.routing_evidence.len(), lock.routing_evidence.len());
        assert_eq!(parsed.replay_envelope, lock.replay_envelope);
    }

    #[test]
    fn write_and_reread_pack_integrity() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack_dir = dir.path().join("integrity_check");
        let report = fixture_report();

        write_replay_pack(&report, &pack_dir).expect("write pack");

        // Re-read each file and compare field-by-field with the builders.
        let env_json = std::fs::read_to_string(pack_dir.join("env.json")).unwrap();
        let env: super::EnvSnapshot = serde_json::from_str(&env_json).expect("parse env");
        let expected_env = build_env_snapshot(&report);
        assert_eq!(env.os, expected_env.os);
        assert_eq!(env.arch, expected_env.arch);
        assert_eq!(env.backend_identity, expected_env.backend_identity);
        assert_eq!(env.backend_version, expected_env.backend_version);

        let manifest_json = std::fs::read_to_string(pack_dir.join("manifest.json")).unwrap();
        let manifest: super::PackManifest =
            serde_json::from_str(&manifest_json).expect("parse manifest");
        assert_eq!(manifest.run_id, report.run_id);
        assert_eq!(manifest.trace_id, report.trace_id);
        assert_eq!(manifest.segment_count, report.result.segments.len());
        assert_eq!(manifest.event_count, report.events.len());
        assert_eq!(manifest.evidence_count, report.evidence.len());

        let repro_json = std::fs::read_to_string(pack_dir.join("repro.lock")).unwrap();
        let repro: super::ReproLock = serde_json::from_str(&repro_json).expect("parse repro");
        assert_eq!(repro.backend_requested, "auto");
        assert!(repro.diarize);
        assert_eq!(repro.routing_evidence.len(), 1);
    }

    #[test]
    fn repro_lock_with_explicit_backend() {
        let mut report = fixture_report();
        report.request.backend = BackendKind::InsanelyFast;
        report.request.diarize = false;

        let lock = build_repro_lock(&report);
        assert_eq!(lock.backend_requested, "insanely_fast");
        assert!(!lock.diarize);
    }

    #[test]
    fn env_snapshot_without_backend_info() {
        let mut report = fixture_report();
        report.replay.backend_identity = None;
        report.replay.backend_version = None;

        let env = build_env_snapshot(&report);
        assert!(env.backend_identity.is_none());
        assert!(env.backend_version.is_none());
        // Platform fields still populated.
        assert!(!env.os.is_empty());
        assert!(!env.arch.is_empty());
    }

    #[test]
    fn manifest_with_many_segments() {
        let mut report = fixture_report();
        report.result.segments = (0..1000)
            .map(|i| TranscriptionSegment {
                start_sec: Some(i as f64),
                end_sec: Some((i + 1) as f64),
                text: format!("seg-{i}"),
                speaker: None,
                confidence: None,
            })
            .collect();
        report.evidence = (0..50).map(|i| json!({"idx": i})).collect();

        let manifest = build_manifest(&report);
        assert_eq!(manifest.segment_count, 1000);
        assert_eq!(manifest.evidence_count, 50);
    }

    #[test]
    fn write_replay_pack_to_nested_nonexistent_dirs() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack_dir = dir.path().join("a").join("b").join("c").join("replay");
        let report = fixture_report();

        write_replay_pack(&report, &pack_dir).expect("should create nested dirs");
        assert!(pack_dir.join("env.json").exists());
        assert!(pack_dir.join("manifest.json").exists());
        assert!(pack_dir.join("repro.lock").exists());
    }

    #[test]
    fn repro_lock_with_all_backend_kinds() {
        let mut report = fixture_report();
        for (kind, expected_str) in [
            (BackendKind::Auto, "auto"),
            (BackendKind::WhisperCpp, "whisper_cpp"),
            (BackendKind::InsanelyFast, "insanely_fast"),
            (BackendKind::WhisperDiarization, "whisper_diarization"),
        ] {
            report.request.backend = kind;
            let lock = build_repro_lock(&report);
            assert_eq!(
                lock.backend_requested, expected_str,
                "mismatch for {kind:?}"
            );
        }
    }

    #[test]
    fn manifest_timestamps_preserved_exactly() {
        let mut report = fixture_report();
        report.started_at_rfc3339 = "2025-06-15T09:30:45.123Z".to_owned();
        report.finished_at_rfc3339 = "2025-06-15T09:31:12.789Z".to_owned();

        let manifest = build_manifest(&report);
        assert_eq!(manifest.started_at, "2025-06-15T09:30:45.123Z");
        assert_eq!(manifest.finished_at, "2025-06-15T09:31:12.789Z");
    }

    #[test]
    fn repro_lock_diarize_false() {
        let mut report = fixture_report();
        report.request.diarize = false;
        let lock = build_repro_lock(&report);
        assert!(!lock.diarize);
    }

    #[test]
    fn repro_lock_with_many_evidence_entries() {
        let mut report = fixture_report();
        report.evidence = (0..100)
            .map(|i| json!({"step": i, "action": "retry"}))
            .collect();
        let lock = build_repro_lock(&report);
        assert_eq!(lock.routing_evidence.len(), 100);
        assert_eq!(lock.routing_evidence[99]["step"], 99);
    }

    #[test]
    fn write_replay_pack_files_are_pretty_printed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack_dir = dir.path().join("pretty");
        let report = fixture_report();

        write_replay_pack(&report, &pack_dir).expect("write");

        // Pretty-printed JSON contains newlines.
        let env_json = std::fs::read_to_string(pack_dir.join("env.json")).unwrap();
        assert!(env_json.contains('\n'), "env.json should be pretty-printed");
        let manifest_json = std::fs::read_to_string(pack_dir.join("manifest.json")).unwrap();
        assert!(
            manifest_json.contains('\n'),
            "manifest.json should be pretty-printed"
        );
        let repro_json = std::fs::read_to_string(pack_dir.join("repro.lock")).unwrap();
        assert!(
            repro_json.contains('\n'),
            "repro.lock should be pretty-printed"
        );
    }

    #[test]
    fn env_snapshot_franken_whisper_version_matches_cargo_pkg() {
        let report = fixture_report();
        let env = build_env_snapshot(&report);
        assert_eq!(env.franken_whisper_version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn manifest_counts_events_when_present() {
        use crate::model::RunEvent;

        let mut report = fixture_report();
        report.events = (0..7)
            .map(|i| RunEvent {
                seq: i,
                ts_rfc3339: format!("2026-01-01T00:00:{i:02}Z"),
                stage: "ingest".to_owned(),
                code: "ok".to_owned(),
                message: format!("event {i}"),
                payload: json!({}),
            })
            .collect();

        let manifest = build_manifest(&report);
        assert_eq!(manifest.event_count, 7);
    }

    #[test]
    fn repro_lock_replay_envelope_contains_all_expected_keys() {
        let report = fixture_report();
        let lock = build_repro_lock(&report);

        let envelope = lock.replay_envelope.as_object().expect("should be object");
        assert!(envelope.contains_key("input_content_hash"));
        assert!(envelope.contains_key("backend_identity"));
        assert!(envelope.contains_key("backend_version"));
        assert!(envelope.contains_key("output_payload_hash"));
    }

    #[test]
    fn env_snapshot_with_unicode_backend_identity() {
        let mut report = fixture_report();
        report.replay.backend_identity = Some("whisper-ü∂ƒ".to_owned());
        report.replay.backend_version = Some("v1.0-日本語".to_owned());

        let env = build_env_snapshot(&report);
        assert_eq!(env.backend_identity.as_deref(), Some("whisper-ü∂ƒ"));
        assert_eq!(env.backend_version.as_deref(), Some("v1.0-日本語"));

        // Round-trip through JSON preserves unicode.
        let json = serde_json::to_string(&env).expect("serialize");
        let parsed: super::EnvSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.backend_identity, env.backend_identity);
        assert_eq!(parsed.backend_version, env.backend_version);
    }

    #[test]
    fn manifest_with_both_hashes_none() {
        let mut report = fixture_report();
        report.replay.input_content_hash = None;
        report.replay.output_payload_hash = None;

        let manifest = build_manifest(&report);
        assert!(manifest.input_content_hash.is_none());
        assert!(manifest.output_payload_hash.is_none());

        // Ensure JSON round-trip preserves None as null.
        let json = serde_json::to_string(&manifest).expect("serialize");
        assert!(json.contains("null"), "None should serialize as null");
        let parsed: super::PackManifest = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.input_content_hash.is_none());
        assert!(parsed.output_payload_hash.is_none());
    }

    #[test]
    fn repro_lock_evidence_preserves_nested_json_structures() {
        let mut report = fixture_report();
        report.evidence = vec![
            json!({"a": {"b": {"c": [1, 2, 3]}}}),
            json!(null),
            json!([true, false, null, 42, "text"]),
        ];

        let lock = build_repro_lock(&report);
        assert_eq!(lock.routing_evidence.len(), 3);
        assert_eq!(lock.routing_evidence[0]["a"]["b"]["c"][2], 3);
        assert!(lock.routing_evidence[1].is_null());
        assert_eq!(lock.routing_evidence[2][4], "text");
    }

    #[test]
    fn write_replay_pack_to_read_only_dir_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let locked_dir = dir.path().join("locked");
        std::fs::create_dir(&locked_dir).expect("mkdir");
        let blocker_file = locked_dir.join("blocker");
        std::fs::write(&blocker_file, "not-a-directory").expect("write blocker");

        let report = fixture_report();
        let result = write_replay_pack(&report, &blocker_file.join("subdir"));
        assert!(result.is_err(), "should fail when parent path is a file");
    }

    // ── Second-pass edge case tests ──

    #[test]
    fn repro_lock_default_replay_envelope_produces_empty_object() {
        // ReplayEnvelope uses skip_serializing_if on all fields, so default
        // (all None) should serialize to `{}` via serde_json::to_value.
        let mut report = fixture_report();
        report.replay = ReplayEnvelope::default();

        let lock = build_repro_lock(&report);
        assert!(
            lock.replay_envelope.is_object(),
            "default replay should be object"
        );
        let obj = lock.replay_envelope.as_object().unwrap();
        assert!(
            obj.is_empty(),
            "default ReplayEnvelope should serialize to empty object, got: {obj:?}"
        );
    }

    #[test]
    fn write_replay_pack_content_is_deterministic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let report = fixture_report();

        let pack_a = dir.path().join("pack_a");
        let pack_b = dir.path().join("pack_b");
        write_replay_pack(&report, &pack_a).expect("write a");
        write_replay_pack(&report, &pack_b).expect("write b");

        for filename in [
            "env.json",
            "manifest.json",
            "repro.lock",
            "tolerance_manifest.json",
        ] {
            let content_a = std::fs::read_to_string(pack_a.join(filename)).unwrap();
            let content_b = std::fs::read_to_string(pack_b.join(filename)).unwrap();
            assert_eq!(
                content_a, content_b,
                "{filename} should be byte-identical for same report"
            );
        }
    }

    #[test]
    fn manifest_json_has_exactly_expected_keys() {
        let report = fixture_report();
        let manifest = build_manifest(&report);
        let value = serde_json::to_value(&manifest).expect("to_value");
        let obj = value.as_object().expect("object");

        let expected_keys = [
            "trace_id",
            "run_id",
            "started_at",
            "finished_at",
            "input_content_hash",
            "output_payload_hash",
            "segment_count",
            "event_count",
            "evidence_count",
        ];
        assert_eq!(
            obj.len(),
            expected_keys.len(),
            "manifest should have exactly {} keys, got {}",
            expected_keys.len(),
            obj.len()
        );
        for key in expected_keys {
            assert!(obj.contains_key(key), "manifest missing key `{key}`");
        }
    }

    #[test]
    fn structs_implement_debug_and_clone() {
        fn assert_debug<T: std::fmt::Debug>() {}
        fn assert_clone<T: Clone>() {}

        assert_debug::<super::EnvSnapshot>();
        assert_clone::<super::EnvSnapshot>();
        assert_debug::<super::PackManifest>();
        assert_clone::<super::PackManifest>();
        assert_debug::<super::ReproLock>();
        assert_clone::<super::ReproLock>();
        assert_debug::<super::ToleranceManifest>();
        assert_clone::<super::ToleranceManifest>();
    }

    #[test]
    fn tolerance_manifest_defaults_to_primary_rollout_stage_when_absent() {
        let report = fixture_report();
        let tolerance = build_tolerance_manifest(&report);
        assert_eq!(tolerance.native_rollout_stage, "primary");
        assert_eq!(tolerance.schema_version, "tolerance-manifest-v1");
        assert!(tolerance.timestamp_tolerance_sec > 0.0);
        assert!(tolerance.require_text_exact);
    }

    #[test]
    fn tolerance_manifest_uses_routing_event_rollout_stage_when_present() {
        use crate::model::RunEvent;

        let mut report = fixture_report();
        report.events.push(RunEvent {
            seq: 1,
            ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            stage: "backend_routing".to_owned(),
            code: "backend.routing.decision_contract".to_owned(),
            message: "routing decision".to_owned(),
            payload: json!({"native_rollout_stage": "shadow"}),
        });

        let tolerance = build_tolerance_manifest(&report);
        assert_eq!(tolerance.native_rollout_stage, "shadow");
    }

    #[test]
    fn tolerance_manifest_prefers_backend_ok_rollout_stage_over_routing_decision() {
        use crate::model::RunEvent;

        let mut report = fixture_report();
        report.events.extend([
            RunEvent {
                seq: 1,
                ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                stage: "backend_routing".to_owned(),
                code: "backend.routing.decision_contract".to_owned(),
                message: "routing decision".to_owned(),
                payload: json!({"native_rollout_stage": "shadow"}),
            },
            RunEvent {
                seq: 2,
                ts_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
                stage: "backend".to_owned(),
                code: "backend.ok".to_owned(),
                message: "backend completed".to_owned(),
                payload: json!({"native_rollout_stage": "primary"}),
            },
        ]);

        let tolerance = build_tolerance_manifest(&report);
        assert_eq!(tolerance.native_rollout_stage, "primary");
    }

    #[test]
    fn tolerance_manifest_uses_highest_seq_backend_ok_when_events_out_of_order() {
        use crate::model::RunEvent;

        let mut report = fixture_report();
        report.events.extend([
            RunEvent {
                seq: 9,
                ts_rfc3339: "2026-01-01T00:00:09Z".to_owned(),
                stage: "backend".to_owned(),
                code: "backend.ok".to_owned(),
                message: "older inserted later".to_owned(),
                payload: json!({"native_rollout_stage": "validated"}),
            },
            RunEvent {
                seq: 3,
                ts_rfc3339: "2026-01-01T00:00:03Z".to_owned(),
                stage: "backend".to_owned(),
                code: "backend.ok".to_owned(),
                message: "newer inserted earlier in vec order".to_owned(),
                payload: json!({"native_rollout_stage": "shadow"}),
            },
        ]);
        report.events.swap(0, 1);

        let tolerance = build_tolerance_manifest(&report);
        assert_eq!(
            tolerance.native_rollout_stage, "validated",
            "latest backend.ok by seq should win even when vec order is scrambled"
        );
    }

    #[test]
    fn write_replay_pack_with_all_empty_strings() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack_dir = dir.path().join("empty_strings");

        let mut report = fixture_report();
        report.run_id = String::new();
        report.trace_id = String::new();
        report.started_at_rfc3339 = String::new();
        report.finished_at_rfc3339 = String::new();
        report.replay = ReplayEnvelope::default();
        report.result.segments.clear();
        report.events.clear();
        report.evidence.clear();

        write_replay_pack(&report, &pack_dir).expect("write with empty strings");

        // All files should be parseable.
        let manifest_json = std::fs::read_to_string(pack_dir.join("manifest.json")).unwrap();
        let manifest: super::PackManifest =
            serde_json::from_str(&manifest_json).expect("valid manifest");
        assert_eq!(manifest.run_id, "");
        assert_eq!(manifest.trace_id, "");
        assert_eq!(manifest.started_at, "");
        assert_eq!(manifest.finished_at, "");
        assert_eq!(manifest.segment_count, 0);
        assert_eq!(manifest.event_count, 0);
        assert_eq!(manifest.evidence_count, 0);

        let repro_json = std::fs::read_to_string(pack_dir.join("repro.lock")).unwrap();
        let repro: super::ReproLock = serde_json::from_str(&repro_json).expect("valid repro");
        assert!(repro.routing_evidence.is_empty());
    }

    #[test]
    fn tolerance_manifest_serde_round_trip() {
        let report = fixture_report();
        let tolerance = build_tolerance_manifest(&report);

        let json = serde_json::to_string(&tolerance).expect("serialize");
        let parsed: super::ToleranceManifest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.schema_version, tolerance.schema_version);
        assert_eq!(
            parsed.timestamp_tolerance_sec,
            tolerance.timestamp_tolerance_sec
        );
        assert_eq!(parsed.require_text_exact, tolerance.require_text_exact);
        assert_eq!(
            parsed.require_speaker_exact,
            tolerance.require_speaker_exact
        );
        assert_eq!(parsed.native_rollout_stage, tolerance.native_rollout_stage);
        assert_eq!(parsed.segment_count, tolerance.segment_count);
        assert_eq!(parsed.event_count, tolerance.event_count);
    }

    #[test]
    fn rollout_stage_non_string_value_falls_back_to_primary() {
        use crate::model::RunEvent;

        let mut report = fixture_report();
        report.events.push(RunEvent {
            seq: 1,
            ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            stage: "backend_routing".to_owned(),
            code: "backend.routing.decision_contract".to_owned(),
            message: "routing".to_owned(),
            payload: json!({"native_rollout_stage": 42}),
        });

        let tolerance = build_tolerance_manifest(&report);
        assert_eq!(
            tolerance.native_rollout_stage, "primary",
            "non-string rollout stage should fall back to primary"
        );
    }

    #[test]
    fn tolerance_manifest_counts_match_report() {
        use crate::model::RunEvent;

        let mut report = fixture_report();
        report.result.segments = (0..5)
            .map(|i| TranscriptionSegment {
                start_sec: Some(i as f64),
                end_sec: Some((i + 1) as f64),
                text: format!("s{i}"),
                speaker: None,
                confidence: None,
            })
            .collect();
        report.events = (0..3)
            .map(|i| RunEvent {
                seq: i,
                ts_rfc3339: format!("2026-01-01T00:00:{i:02}Z"),
                stage: "test".to_owned(),
                code: "ok".to_owned(),
                message: String::new(),
                payload: json!({}),
            })
            .collect();

        let tolerance = build_tolerance_manifest(&report);
        assert_eq!(tolerance.segment_count, 5);
        assert_eq!(tolerance.event_count, 3);
    }

    #[test]
    fn tolerance_manifest_require_speaker_exact_from_default() {
        let report = fixture_report();
        let tolerance = build_tolerance_manifest(&report);
        // SegmentCompatibilityTolerance::default() sets require_speaker_exact to false.
        assert!(
            !tolerance.require_speaker_exact,
            "default require_speaker_exact should be false"
        );
        // Round-trip preserves the value
        let json = serde_json::to_string(&tolerance).expect("serialize");
        let parsed: super::ToleranceManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            parsed.require_speaker_exact,
            tolerance.require_speaker_exact
        );
    }

    #[test]
    fn tolerance_manifest_json_has_expected_keys() {
        let report = fixture_report();
        let tolerance = build_tolerance_manifest(&report);
        let value = serde_json::to_value(&tolerance).expect("to_value");
        let obj = value.as_object().expect("object");

        let expected_keys = [
            "schema_version",
            "timestamp_tolerance_sec",
            "require_text_exact",
            "require_speaker_exact",
            "native_rollout_stage",
            "segment_count",
            "event_count",
        ];
        assert_eq!(obj.len(), expected_keys.len());
        for key in expected_keys {
            assert!(
                obj.contains_key(key),
                "tolerance_manifest missing key `{key}`"
            );
        }
    }
}
