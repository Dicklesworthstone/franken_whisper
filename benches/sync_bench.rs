//! Criterion benches for SQLite <-> JSONL sync paths.
//!
//! Covers:
//! - `sync::export` throughput from a seeded SQLite store
//! - `sync::import` throughput from a deterministic JSONL snapshot

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use serde_json::json;
use tempfile::tempdir;

use franken_whisper::model::{
    BackendKind, BackendParams, InputSource, ReplayEnvelope, RunEvent, RunReport,
    TranscribeRequest, TranscriptionResult, TranscriptionSegment,
};
use franken_whisper::storage::RunStore;
use franken_whisper::sync::{self, ConflictPolicy};

fn make_report(run_id: &str, db_path: &std::path::Path) -> RunReport {
    let segments = vec![
        TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(0.4),
            text: "hello".to_owned(),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: Some(0.9),
        },
        TranscriptionSegment {
            start_sec: Some(0.4),
            end_sec: Some(0.8),
            text: "world".to_owned(),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: Some(0.9),
        },
    ];

    let events = vec![
        RunEvent {
            seq: 1,
            ts_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
            stage: "ingest".to_owned(),
            code: "ingest.ok".to_owned(),
            message: "ok".to_owned(),
            payload: json!({}),
        },
        RunEvent {
            seq: 2,
            ts_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.ok".to_owned(),
            message: "ok".to_owned(),
            payload: json!({"segments": 2}),
        },
    ];

    RunReport {
        run_id: run_id.to_owned(),
        trace_id: "bench-trace-id".to_owned(),
        started_at_rfc3339: "2026-02-22T00:00:00Z".to_owned(),
        finished_at_rfc3339: "2026-02-22T00:00:01Z".to_owned(),
        input_path: "bench.wav".to_owned(),
        normalized_wav_path: "bench.norm.wav".to_owned(),
        request: TranscribeRequest {
            input: InputSource::File {
                path: std::path::PathBuf::from("bench.wav"),
            },
            backend: BackendKind::WhisperCpp,
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
            transcript: "hello world".to_owned(),
            language: Some("en".to_owned()),
            segments,
            acceleration: None,
            raw_output: json!({"text":"hello world"}),
            artifact_paths: vec![],
        },
        events,
        warnings: vec![],
        evidence: vec![],
        replay: ReplayEnvelope::default(),
    }
}

fn seed_db(db_path: &std::path::Path, run_count: usize) {
    let store = RunStore::open(db_path).expect("store should open");
    for i in 0..run_count {
        let report = make_report(&format!("sync-bench-run-{i:04}"), db_path);
        store
            .persist_report(&report)
            .expect("seed persist should succeed");
    }
}

fn directory_size_bytes(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let entries = std::fs::read_dir(path).expect("read_dir should succeed");
    for entry in entries {
        let entry = entry.expect("dir entry should be readable");
        if let Ok(metadata) = entry.metadata()
            && metadata.is_file()
        {
            total += metadata.len();
        }
    }
    total
}

fn bench_sync_export(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/export");
    for run_count in [10usize, 50usize] {
        group.bench_with_input(BenchmarkId::new("runs", run_count), &run_count, |b, &n| {
            b.iter_batched(
                || {
                    let dir = tempdir().expect("tempdir");
                    let db_path = dir.path().join("storage.sqlite3");
                    let state_root = dir.path().join("state");
                    let output_dir = dir.path().join("snapshot");
                    seed_db(&db_path, n);
                    (dir, db_path, state_root, output_dir)
                },
                |(_dir, db_path, state_root, output_dir)| {
                    let manifest = sync::export(&db_path, &output_dir, &state_root)
                        .expect("export should succeed");
                    assert_eq!(manifest.row_counts.runs, n as u64);
                    let bytes = directory_size_bytes(&output_dir);
                    assert!(bytes > 0, "export should write non-empty snapshot");
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_sync_import(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/import");

    for run_count in [10usize, 50usize] {
        let fixture_dir = tempdir().expect("tempdir");
        let source_db = fixture_dir
            .path()
            .join(format!("source-{run_count}.sqlite3"));
        let source_state = fixture_dir.path().join("source-state");
        let export_dir = fixture_dir.path().join(format!("snapshot-{run_count}"));
        seed_db(&source_db, run_count);
        let manifest = sync::export(&source_db, &export_dir, &source_state)
            .expect("fixture export should succeed");
        let snapshot_size = directory_size_bytes(&export_dir);
        group.throughput(Throughput::Bytes(snapshot_size.max(1)));

        group.bench_with_input(BenchmarkId::new("runs", run_count), &run_count, |b, &n| {
            b.iter_batched(
                || {
                    let iter_dir = tempdir().expect("tempdir");
                    let target_db = iter_dir.path().join("target.sqlite3");
                    let target_state = iter_dir.path().join("target-state");
                    (iter_dir, target_db, target_state)
                },
                |(_iter_dir, target_db, target_state)| {
                    let import_result = sync::import(
                        &target_db,
                        &export_dir,
                        &target_state,
                        ConflictPolicy::Reject,
                    )
                    .expect("import should succeed");
                    assert_eq!(import_result.runs_imported, n as u64);
                    assert_eq!(import_result.conflicts.len(), 0);
                },
                BatchSize::SmallInput,
            );
        });

        assert_eq!(manifest.row_counts.runs, run_count as u64);
    }

    group.finish();
}

/// First-time incremental export of a seeded store (exports all runs, so the
/// `export_table_{segments,events}_for_runs` writers run over every run_id). A/B the
/// per-run N+1 query vs the batched `WHERE run_id IN (…)` query via external env:
/// `FW_SYNC_BATCH_QUERY=0 cargo bench ...` (legacy per-run) vs unset/`1` (batched).
fn bench_sync_export_incremental(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync/export_incremental");

    for run_count in [50usize] {
        let fixture = tempdir().expect("tempdir");
        let src_db = fixture.path().join(format!("src-{run_count}.sqlite3"));
        seed_db(&src_db, run_count);

        group.bench_with_input(BenchmarkId::new("runs", run_count), &run_count, |b, &n| {
            b.iter_batched(
                || {
                    let iter_dir = tempdir().expect("tempdir");
                    let out_dir = iter_dir.path().join("out");
                    let state_root = iter_dir.path().join("state");
                    (iter_dir, out_dir, state_root)
                },
                |(_iter_dir, out_dir, state_root)| {
                    let manifest = sync::export_incremental(&src_db, &out_dir, &state_root)
                        .expect("incremental export should succeed");
                    assert_eq!(manifest.row_counts.runs, n as u64);
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sync_export,
    bench_sync_import,
    bench_sync_export_incremental,
);
criterion_main!(benches);
