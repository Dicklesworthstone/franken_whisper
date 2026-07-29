//! bd-3pf.5.1: Benchmark-style tests (timed assertions, not criterion).
//!
//! Covers:
//! - TTY frame encode/decode round-trip performance
//! - Sync export/import throughput
//! - Storage persist latency with varying segment counts
//! - Pipeline stage dispatch overhead
//!
//! Each test measures wall-clock time and asserts it stays below a generous
//! upper bound to catch catastrophic regressions, not micro-benchmark precision.

#![forbid(unsafe_code)]

mod helpers;

use std::io::Cursor;
use std::time::Instant;

use base64::Engine;
use base64::engine::general_purpose::STANDARD_NO_PAD;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use std::io::Write;

use franken_whisper::error::FwError;
use franken_whisper::model::*;
use franken_whisper::orchestrator::{PipelineBuilder, PipelineConfig, PipelineStage};
use franken_whisper::storage::RunStore;
use franken_whisper::tty_audio::{
    DecodeRecoveryPolicy, TtyAudioFrame, decode_frames_to_raw_with_policy,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compress raw bytes with zlib.
fn compress(data: &[u8]) -> Vec<u8> {
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::fast());
    enc.write_all(data).expect("zlib write");
    enc.finish().expect("zlib finish")
}

/// CRC32 of raw bytes.
fn crc32_of(data: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(data);
    h.finalize()
}

/// SHA-256 hex digest.
fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(data);
    format!("{d:x}")
}

/// Build a valid TtyAudioFrame from raw audio bytes.
fn build_frame(seq: u64, raw_audio: &[u8]) -> TtyAudioFrame {
    let crc = crc32_of(raw_audio);
    let sha = sha256_hex(raw_audio);
    let compressed = compress(raw_audio);
    let payload_b64 = STANDARD_NO_PAD.encode(&compressed);
    TtyAudioFrame {
        protocol_version: 1,
        seq,
        codec: "mulaw+zlib+b64".to_owned(),
        sample_rate_hz: 8_000,
        channels: 1,
        payload_b64,
        crc32: Some(crc),
        payload_sha256: Some(sha),
    }
}

/// Serialize frames into NDJSON bytes (as if they came over a pipe).
fn frames_to_ndjson(frames: &[TtyAudioFrame]) -> Vec<u8> {
    let mut buf = Vec::new();
    for frame in frames {
        let line = serde_json::to_string(frame).expect("serialize frame");
        buf.extend_from_slice(line.as_bytes());
        buf.push(b'\n');
    }
    buf
}

/// Build a test RunReport with `n_segments` segments and `n_events` events.
fn build_report(run_id: &str, n_segments: usize, n_events: usize) -> RunReport {
    let segments: Vec<TranscriptionSegment> = (0..n_segments)
        .map(|i| TranscriptionSegment {
            start_sec: Some(i as f64),
            end_sec: Some((i + 1) as f64),
            text: format!("segment number {i}"),
            speaker: if i % 3 == 0 {
                Some(format!("SPEAKER_{:02}", i % 5))
            } else {
                None
            },
            confidence: Some(0.9),
        })
        .collect();

    let events: Vec<RunEvent> = (0..n_events)
        .map(|i| RunEvent {
            seq: (i + 1) as u64,
            ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: format!("event {i}"),
            payload: serde_json::json!({"idx": i}),
        })
        .collect();

    let transcript = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    RunReport {
        run_id: run_id.to_owned(),
        trace_id: "00000000000000000000000000000001".to_owned(),
        started_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
        finished_at_rfc3339: "2026-01-01T00:01:00Z".to_owned(),
        input_path: "/tmp/test.wav".to_owned(),
        normalized_wav_path: "/tmp/normalized.wav".to_owned(),
        request: TranscribeRequest {
            input: InputSource::File {
                path: std::path::PathBuf::from("test.wav"),
            },
            backend: BackendKind::WhisperCpp,
            model: None,
            language: Some("en".to_owned()),
            translate: false,
            diarize: false,
            persist: true,
            db_path: std::path::PathBuf::from("/tmp/bench.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        },
        result: TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript,
            language: Some("en".to_owned()),
            segments,
            acceleration: None,
            diarization: None,
            raw_output: serde_json::json!({"bench": true}),
            artifact_paths: vec![],
        },
        events,
        warnings: vec![],
        evidence: vec![],
        replay: ReplayEnvelope::default(),
    }
}

// ---------------------------------------------------------------------------
// 1. TTY frame encode/decode round-trip performance
// ---------------------------------------------------------------------------

#[test]
fn benchmark_tty_frame_round_trip_small() {
    // 10 frames of 160 bytes each (20ms at 8kHz mulaw).
    let frame_count = 10;
    let raw_chunk = vec![0x80u8; 160]; // 160 bytes of mulaw silence

    let frames: Vec<TtyAudioFrame> = (0..frame_count)
        .map(|seq| build_frame(seq, &raw_chunk))
        .collect();

    let ndjson = frames_to_ndjson(&frames);

    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        let mut cursor = Cursor::new(&ndjson);
        let (report, raw) =
            decode_frames_to_raw_with_policy(&mut cursor, DecodeRecoveryPolicy::SkipMissing)
                .expect("decode should succeed");
        assert_eq!(report.frames_decoded, frame_count);
        assert_eq!(raw.len(), 160 * frame_count as usize);
    }
    let elapsed = start.elapsed();

    eprintln!(
        "benchmark_tty_frame_round_trip_small: {iterations} iterations of {frame_count} frames in {elapsed:?} ({:.2} us/iter)",
        elapsed.as_micros() as f64 / iterations as f64
    );

    // Generous upper bound: 10 seconds for 100 iterations.
    assert!(
        elapsed.as_secs() < 10,
        "TTY frame round-trip took too long: {elapsed:?}"
    );
}

#[test]
fn benchmark_tty_frame_round_trip_large() {
    // 100 frames of 800 bytes each (100ms at 8kHz mulaw).
    let frame_count = 100u64;
    let raw_chunk = vec![0x7Fu8; 800];

    let frames: Vec<TtyAudioFrame> = (0..frame_count)
        .map(|seq| build_frame(seq, &raw_chunk))
        .collect();

    let ndjson = frames_to_ndjson(&frames);

    let start = Instant::now();
    let iterations = 20;
    for _ in 0..iterations {
        let mut cursor = Cursor::new(&ndjson);
        let (report, raw) =
            decode_frames_to_raw_with_policy(&mut cursor, DecodeRecoveryPolicy::SkipMissing)
                .expect("decode should succeed");
        assert_eq!(report.frames_decoded, frame_count);
        assert_eq!(raw.len(), 800 * frame_count as usize);
    }
    let elapsed = start.elapsed();

    eprintln!(
        "benchmark_tty_frame_round_trip_large: {iterations} iterations of {frame_count} frames in {elapsed:?} ({:.2} us/iter)",
        elapsed.as_micros() as f64 / iterations as f64
    );

    assert!(
        elapsed.as_secs() < 30,
        "TTY frame large round-trip took too long: {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// 2. Sync export/import throughput
// ---------------------------------------------------------------------------

#[test]
fn benchmark_sync_export_import_throughput() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let db_path = tmp.path().join("bench_sync.sqlite3");
    let state_root = tmp.path().join("state");
    let export_dir = tmp.path().join("export");

    // Populate the DB with some runs.
    let store = RunStore::open(&db_path).expect("open store");
    let run_count: u64 = 5;
    for i in 0..run_count {
        let report = build_report(&format!("run-sync-{i:04}"), 10, 5);
        store.persist_report(&report).expect("persist report");
    }
    drop(store);

    // Benchmark export.
    let start_export = Instant::now();
    let iterations = 5;
    for _ in 0..iterations {
        let iter_export_dir = tempfile::tempdir().expect("create iter export dir");
        let manifest = franken_whisper::sync::export(&db_path, iter_export_dir.path(), &state_root)
            .expect("export should succeed");
        assert_eq!(manifest.row_counts.runs, run_count);
    }
    let export_elapsed = start_export.elapsed();
    eprintln!(
        "benchmark_sync_export: {iterations} exports of {run_count} runs in {export_elapsed:?} ({:.2} ms/iter)",
        export_elapsed.as_millis() as f64 / iterations as f64
    );

    // Perform one export for the import benchmark.
    let _manifest = franken_whisper::sync::export(&db_path, &export_dir, &state_root)
        .expect("export for import");

    // Benchmark import.
    let start_import = Instant::now();
    for _ in 0..iterations {
        let import_db_dir = tempfile::tempdir().expect("create import temp dir");
        let import_db_path = import_db_dir.path().join("imported.sqlite3");
        let import_state = import_db_dir.path().join("state");
        let result = franken_whisper::sync::import(
            &import_db_path,
            &export_dir,
            &import_state,
            franken_whisper::sync::ConflictPolicy::Overwrite,
        )
        .expect("import should succeed");
        assert_eq!(result.runs_imported, run_count);
    }
    let import_elapsed = start_import.elapsed();
    eprintln!(
        "benchmark_sync_import: {iterations} imports of {run_count} runs in {import_elapsed:?} ({:.2} ms/iter)",
        import_elapsed.as_millis() as f64 / iterations as f64
    );

    assert!(
        export_elapsed.as_secs() < 30,
        "sync export too slow: {export_elapsed:?}"
    );
    assert!(
        import_elapsed.as_secs() < 30,
        "sync import too slow: {import_elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// 3. Storage persist latency with varying segment counts
// ---------------------------------------------------------------------------

#[test]
fn benchmark_storage_persist_latency_small() {
    benchmark_storage_persist_latency(1, 1, "small");
}

#[test]
fn benchmark_storage_persist_latency_medium() {
    benchmark_storage_persist_latency(10, 8, "medium");
}

#[test]
fn benchmark_storage_persist_latency_large() {
    benchmark_storage_persist_latency(15, 10, "large");
}

fn benchmark_storage_persist_latency(n_segments: usize, n_events: usize, label: &str) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let db_path = tmp.path().join(format!("bench_persist_{label}.sqlite3"));
    let store = RunStore::open(&db_path).expect("open store");

    let iterations = 10;
    let start = Instant::now();
    for i in 0..iterations {
        let report = build_report(&format!("run-{label}-{i:04}"), n_segments, n_events);
        store.persist_report(&report).expect("persist report");
    }
    let elapsed = start.elapsed();

    eprintln!(
        "benchmark_storage_persist_{label}: {iterations} persists of {n_segments} segments + {n_events} events in {elapsed:?} ({:.2} ms/iter)",
        elapsed.as_millis() as f64 / iterations as f64
    );

    // Verify data was actually persisted.
    let runs = store.list_recent_runs(iterations + 1).expect("list runs");
    assert_eq!(
        runs.len(),
        iterations,
        "should have {iterations} persisted runs"
    );

    assert!(
        elapsed.as_secs() < 30,
        "storage persist ({label}) too slow: {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// 4. Pipeline stage dispatch overhead
// ---------------------------------------------------------------------------

#[test]
fn benchmark_pipeline_config_construction() {
    let iterations = 10_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let config = PipelineConfig::default();
        assert_eq!(config.stages().len(), 10);
    }
    let elapsed = start.elapsed();

    eprintln!(
        "benchmark_pipeline_config_construction: {iterations} iterations in {elapsed:?} ({:.2} ns/iter)",
        elapsed.as_nanos() as f64 / iterations as f64
    );

    assert!(
        elapsed.as_secs() < 5,
        "PipelineConfig::default() construction too slow: {elapsed:?}"
    );
}

#[test]
fn benchmark_pipeline_config_validation() {
    let iterations = 10_000;
    let config = PipelineConfig::default();

    let start = Instant::now();
    for _ in 0..iterations {
        config.validate().expect("validate should succeed");
    }
    let elapsed = start.elapsed();

    eprintln!(
        "benchmark_pipeline_config_validation: {iterations} iterations in {elapsed:?} ({:.2} ns/iter)",
        elapsed.as_nanos() as f64 / iterations as f64
    );

    assert!(
        elapsed.as_secs() < 5,
        "PipelineConfig validation too slow: {elapsed:?}"
    );
}

#[test]
fn benchmark_pipeline_builder_with_skips() {
    let iterations = 10_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let config = PipelineBuilder::default_stages()
            .without(PipelineStage::Vad)
            .without(PipelineStage::Separate)
            .without(PipelineStage::Diarize)
            .build()
            .expect("build should succeed");
        assert_eq!(config.stages().len(), 7);
    }
    let elapsed = start.elapsed();

    eprintln!(
        "benchmark_pipeline_builder_with_skips: {iterations} iterations in {elapsed:?} ({:.2} ns/iter)",
        elapsed.as_nanos() as f64 / iterations as f64
    );

    assert!(
        elapsed.as_secs() < 5,
        "PipelineBuilder with skips too slow: {elapsed:?}"
    );
}

#[test]
fn benchmark_pipeline_has_stage_lookup() {
    let config = PipelineConfig::default();
    let iterations = 100_000;
    let start = Instant::now();
    for _ in 0..iterations {
        assert!(config.has_stage(PipelineStage::Backend));
        assert!(config.has_stage(PipelineStage::Ingest));
        assert!(config.has_stage(PipelineStage::Persist));
    }
    let elapsed = start.elapsed();

    eprintln!(
        "benchmark_pipeline_has_stage_lookup: {iterations} iterations (3 lookups each) in {elapsed:?} ({:.2} ns/iter)",
        elapsed.as_nanos() as f64 / iterations as f64
    );

    assert!(
        elapsed.as_secs() < 5,
        "has_stage lookup too slow: {elapsed:?}"
    );
}

#[test]
fn benchmark_pipeline_stage_label() {
    let iterations = 100_000;
    let start = Instant::now();
    for _ in 0..iterations {
        for stage in [
            PipelineStage::Ingest,
            PipelineStage::Normalize,
            PipelineStage::Vad,
            PipelineStage::Separate,
            PipelineStage::Backend,
            PipelineStage::Accelerate,
            PipelineStage::Align,
            PipelineStage::Punctuate,
            PipelineStage::Diarize,
            PipelineStage::Persist,
        ] {
            let label = stage.label();
            assert!(!label.is_empty());
        }
    }
    let elapsed = start.elapsed();

    eprintln!(
        "benchmark_pipeline_stage_label: {iterations} iterations (10 labels each) in {elapsed:?} ({:.2} ns/iter)",
        elapsed.as_nanos() as f64 / iterations as f64
    );

    assert!(elapsed.as_secs() < 5, "stage label() too slow: {elapsed:?}");
}

// ---------------------------------------------------------------------------
// 5. Error code dispatch overhead
// ---------------------------------------------------------------------------

#[test]
fn benchmark_error_code_dispatch() {
    let errors: Vec<FwError> = vec![
        FwError::Io(std::io::Error::other("x")),
        FwError::BackendUnavailable("x".to_owned()),
        FwError::InvalidRequest("x".to_owned()),
        FwError::Storage("x".to_owned()),
        FwError::StageTimeout {
            stage: "x".to_owned(),
            budget_ms: 1,
        },
    ];

    let iterations = 100_000;
    let start = Instant::now();
    for _ in 0..iterations {
        for err in &errors {
            let code = err.error_code();
            assert!(code.starts_with("FW-"));
            let robot = err.robot_error_code();
            assert!(robot.starts_with("FW-ROBOT-"));
        }
    }
    let elapsed = start.elapsed();

    eprintln!(
        "benchmark_error_code_dispatch: {iterations} iterations (5 errors each) in {elapsed:?} ({:.2} ns/iter)",
        elapsed.as_nanos() as f64 / iterations as f64
    );

    assert!(
        elapsed.as_secs() < 5,
        "error code dispatch too slow: {elapsed:?}"
    );
}
