//! Performance benchmarks for the `RunStore` persistence layer.
//!
//! Exercises the SQLite-backed storage hot paths: persist, load, list, and
//! schema migration.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde_json::json;
use tempfile::tempdir;

use franken_whisper::model::{
    BackendKind, BackendParams, InputSource, ReplayEnvelope, RunEvent, RunReport,
    TranscribeRequest, TranscriptionResult, TranscriptionSegment,
};
use franken_whisper::storage::RunStore;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Build a synthetic `RunReport` with a configurable number of segments and
/// events to exercise different throughput scenarios.
fn make_report(run_id: &str, num_segments: usize, num_events: usize) -> RunReport {
    let segments: Vec<TranscriptionSegment> = (0..num_segments)
        .map(|i| TranscriptionSegment {
            start_sec: Some(i as f64),
            end_sec: Some(i as f64 + 0.5),
            text: format!("segment {i}"),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: Some(0.95),
        })
        .collect();

    let events: Vec<RunEvent> = (0..num_events)
        .map(|i| RunEvent {
            seq: i as u64,
            ts_rfc3339: "2025-01-01T00:00:00Z".to_owned(),
            stage: "backend".to_owned(),
            code: "progress".to_owned(),
            message: format!("event {i}"),
            payload: json!({"step": i}),
        })
        .collect();

    let transcript: String = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    RunReport {
        run_id: run_id.to_owned(),
        trace_id: "bench-trace-id".to_owned(),
        started_at_rfc3339: "2025-01-01T00:00:00Z".to_owned(),
        finished_at_rfc3339: "2025-01-01T00:01:00Z".to_owned(),
        input_path: "/tmp/bench_input.wav".to_owned(),
        normalized_wav_path: "/tmp/bench_normalized.wav".to_owned(),
        request: TranscribeRequest {
            input: InputSource::File {
                path: "/tmp/bench_input.wav".into(),
            },
            backend: BackendKind::WhisperCpp,
            model: Some("base.en".to_owned()),
            language: Some("en".to_owned()),
            translate: false,
            diarize: false,
            persist: true,
            db_path: "/tmp/bench.sqlite3".into(),
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
            raw_output: json!({"text": "bench output"}),
            artifact_paths: vec![],
        },
        events,
        warnings: vec![],
        evidence: vec![],
        replay: ReplayEnvelope::default(),
    }
}

/// Open a fresh in-tempdir `RunStore` for isolation between benchmark
/// iterations.
fn open_temp_store() -> (tempfile::TempDir, RunStore) {
    let dir = tempdir().expect("tempdir creation should succeed");
    let db_path = dir.path().join("bench.sqlite3");
    let store = RunStore::open(&db_path).expect("store should open");
    (dir, store)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_persist_report(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/persist_report");

    for num_segments in [0, 10, 100] {
        group.bench_with_input(
            BenchmarkId::new("segments", num_segments),
            &num_segments,
            |b, &n| {
                let (_dir, store) = open_temp_store();
                let mut counter = 0u64;

                b.iter(|| {
                    counter += 1;
                    let report = make_report(&format!("run-{counter}"), n, 5);
                    store
                        .persist_report(&report)
                        .expect("persist should succeed");
                });
            },
        );
    }

    group.finish();
}

fn bench_load_run_details(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/load_run_details");

    for num_segments in [10, 50] {
        group.bench_with_input(
            BenchmarkId::new("segments", num_segments),
            &num_segments,
            |b, &n| {
                let (_dir, store) = open_temp_store();
                let report = make_report("load-bench-run", n, 10);
                store
                    .persist_report(&report)
                    .expect("seed persist should succeed");

                b.iter(|| {
                    let details = store
                        .load_run_details("load-bench-run")
                        .expect("load should succeed");
                    assert!(details.is_some());
                });
            },
        );
    }

    group.finish();
}

/// A/B for the routing-history app-level N+1: loading full details for many runs via
/// the batched two-query `load_run_details_batch` vs the per-run path. Toggle with
/// external env: `FW_STORAGE_BATCH_HISTORY=0 cargo bench …` (per-run N+1) vs unset/`1`
/// (batched). Same cached binary; output is byte-identical (asserted in unit tests).
fn bench_load_run_details_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/load_history_batch");

    for run_count in [50usize] {
        let (_dir, store) = open_temp_store();
        let mut run_ids: Vec<String> = Vec::with_capacity(run_count);
        for i in 0..run_count {
            let id = format!("hist-{i:04}");
            let report = make_report(&id, 5, 5);
            store.persist_report(&report).expect("seed persist should succeed");
            run_ids.push(id);
        }

        group.bench_with_input(BenchmarkId::new("runs", run_count), &run_ids, |b, ids| {
            b.iter(|| {
                let details = store
                    .load_run_details_batch(ids)
                    .expect("batch load should succeed");
                assert_eq!(details.len(), ids.len());
            });
        });
    }

    group.finish();
}

fn bench_list_recent_runs(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage/list_recent_runs");

    // Seed the database with a fixed number of runs, then benchmark listing
    // with varying limits.
    let (_dir, store) = open_temp_store();
    for i in 0..100 {
        let report = make_report(&format!("list-run-{i}"), 5, 3);
        store
            .persist_report(&report)
            .expect("seed persist should succeed");
    }

    for limit in [5, 25, 100] {
        group.bench_with_input(BenchmarkId::new("limit", limit), &limit, |b, &lim| {
            b.iter(|| {
                let runs = store.list_recent_runs(lim).expect("list should succeed");
                assert!(!runs.is_empty());
            });
        });
    }

    group.finish();
}

fn bench_schema_migration(c: &mut Criterion) {
    c.bench_function("storage/schema_migration_open", |b| {
        b.iter(|| {
            // Each iteration opens a fresh database, which triggers full
            // schema creation and migration to the current version.
            let dir = tempdir().expect("tempdir creation should succeed");
            let db_path = dir.path().join("migration_bench.sqlite3");
            let _store = RunStore::open(&db_path).expect("store should open");
        });
    });
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_persist_report,
    bench_load_run_details,
    bench_load_run_details_batch,
    bench_list_recent_runs,
    bench_schema_migration,
);
criterion_main!(benches);
