use std::path::PathBuf;

use serde_json::json;
use tempfile::tempdir;

use franken_whisper::conformance::compare_replay_envelopes;
use franken_whisper::model::{
    BackendKind, BackendParams, InputSource, ReplayEnvelope, RunEvent, RunReport,
    TranscribeRequest, TranscriptionResult, TranscriptionSegment,
};
use franken_whisper::storage::RunStore;

fn fixture_report(id: &str, db_path: &std::path::Path, replay: ReplayEnvelope) -> RunReport {
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
            transcript: "replay envelope test".to_owned(),
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
            raw_output: json!({"text":"hello"}),
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
        replay,
    }
}

#[test]
fn replay_envelope_persists_and_round_trips() {
    let dir = tempdir().expect("tempdir");
    let db_path = dir.path().join("storage.sqlite3");
    let store = RunStore::open(&db_path).expect("store open");

    let expected = ReplayEnvelope {
        input_content_hash: Some("input-sha-256".to_owned()),
        backend_identity: Some("whisper-cli".to_owned()),
        backend_version: Some("whisper 1.2.3".to_owned()),
        output_payload_hash: Some("output-sha-256".to_owned()),
    };

    store
        .persist_report(&fixture_report(
            "replay-roundtrip",
            &db_path,
            expected.clone(),
        ))
        .expect("persist");

    let details = store
        .load_run_details("replay-roundtrip")
        .expect("query")
        .expect("row");

    let comparison = compare_replay_envelopes(&expected, &details.replay);
    assert!(
        comparison.within_tolerance(),
        "expected replay metadata to round-trip exactly, got {comparison:?}"
    );
}

#[test]
fn replay_comparator_flags_semantic_drift() {
    let expected = ReplayEnvelope {
        input_content_hash: Some("input-a".to_owned()),
        backend_identity: Some("whisper-cli".to_owned()),
        backend_version: Some("whisper 1.2.3".to_owned()),
        output_payload_hash: Some("output-a".to_owned()),
    };
    let observed = ReplayEnvelope {
        input_content_hash: Some("input-b".to_owned()),
        backend_identity: Some("whisper-cli".to_owned()),
        backend_version: None,
        output_payload_hash: Some("output-b".to_owned()),
    };

    let comparison = compare_replay_envelopes(&expected, &observed);
    assert!(!comparison.within_tolerance());
    assert!(!comparison.input_hash_match);
    assert!(comparison.backend_identity_match);
    assert!(!comparison.backend_version_match);
    assert!(!comparison.output_hash_match);
    assert_eq!(comparison.missing_observed_fields, vec!["backend_version"]);
}
