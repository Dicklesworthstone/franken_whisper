//! Performance benchmarks for pipeline-adjacent hot paths.
//!
//! Covers event logging throughput, SHA-256 hashing performance (the same
//! primitives used by the orchestrator's replay envelope), and stage budget
//! calculation via `PipelineConfig` construction and validation.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use franken_whisper::model::{RunEvent, StreamedRunEvent};
use franken_whisper::orchestrator::{PipelineBuilder, PipelineConfig, PipelineStage};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reproduce the SHA-256 hex-encoding pattern used inside the orchestrator
/// (`sha256_bytes_hex`).  This is the function under benchmark.
fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Build a synthetic `RunEvent` (mirrors `EventLog::push` output shape).
fn make_event(seq: u64, stage: &str, code: &str) -> RunEvent {
    RunEvent {
        seq,
        ts_rfc3339: "2025-01-01T00:00:00Z".to_owned(),
        stage: stage.to_owned(),
        code: code.to_owned(),
        message: format!("event {seq}"),
        payload: json!({
            "trace_id": "bench-trace",
            "elapsed_ms": 42,
        }),
    }
}

// ---------------------------------------------------------------------------
// Benchmarks: event logging throughput
// ---------------------------------------------------------------------------

fn bench_event_logging_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/event_logging");

    // Benchmark the construction and serialization of RunEvent + StreamedRunEvent,
    // which mirrors the EventLog::push hot path (create event, serialize for
    // channel send, push to vec).
    for batch_size in [1, 10, 100] {
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            &batch_size,
            |b, &n| {
                b.iter(|| {
                    let mut events: Vec<RunEvent> = Vec::with_capacity(n);
                    for i in 0..n {
                        let event = make_event(i as u64, "backend", "progress");
                        // Simulate the channel serialization path: wrap in
                        // StreamedRunEvent and serialize to JSON (the NDJSON
                        // emitter does this on the streaming side).
                        let streamed = StreamedRunEvent {
                            run_id: "bench-run-id".to_owned(),
                            event: event.clone(),
                        };
                        let _ = serde_json::to_string(&streamed);
                        events.push(event);
                    }
                    events
                });
            },
        );
    }

    group.finish();
}

/// Benchmark just the event serialization (JSON encoding) independent of
/// construction, to isolate serde overhead.
fn bench_event_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/event_serialization");

    let event = make_event(1, "backend", "progress");
    let streamed = StreamedRunEvent {
        run_id: "bench-run-id".to_owned(),
        event: event.clone(),
    };

    group.bench_function("single_event_to_json", |b| {
        b.iter(|| serde_json::to_string(&streamed).expect("serialization should succeed"));
    });

    // Larger payload to stress serde
    let heavy_event = RunEvent {
        seq: 1,
        ts_rfc3339: "2025-01-01T00:00:00Z".to_owned(),
        stage: "backend".to_owned(),
        code: "output".to_owned(),
        message: "heavy payload".to_owned(),
        payload: json!({
            "trace_id": "bench-trace",
            "raw_output": {
                "text": "a]".repeat(500),
                "segments": (0..50).map(|i| json!({
                    "start": i as f64,
                    "end": i as f64 + 0.5,
                    "text": format!("word {i}"),
                })).collect::<Vec<_>>(),
            },
        }),
    };

    group.bench_function("heavy_payload_to_json", |b| {
        b.iter(|| serde_json::to_string(&heavy_event).expect("serialization should succeed"));
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: SHA-256 hashing
// ---------------------------------------------------------------------------

fn bench_sha256_hashing(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/sha256");

    // Benchmark various input sizes representative of real orchestrator usage:
    // - small: JSON payload hash (~1 KB)
    // - medium: normalized WAV header + small audio (~64 KB)
    // - large: full audio file hash (~1 MB)
    for (label, size) in [("1KB", 1024), ("64KB", 65_536), ("1MB", 1_048_576)] {
        let data = vec![0xABu8; size];
        group.bench_with_input(BenchmarkId::new("input_size", label), &data, |b, bytes| {
            b.iter(|| sha256_hex(bytes));
        });
    }

    group.finish();
}

/// Benchmark SHA-256 of a JSON value (the `sha256_json_value` pattern used
/// for output payload hashing in replay envelopes).
fn bench_sha256_json_value(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/sha256_json");

    for n in [1, 50, 200] {
        let segments: Vec<Value> = (0..n)
            .map(|i| {
                json!({
                    "start": i as f64 * 0.5,
                    "end": i as f64 * 0.5 + 0.5,
                    "text": format!("segment {i}"),
                })
            })
            .collect();

        let payload = json!({
            "text": "benchmark transcript",
            "segments": segments,
            "language": "en",
        });

        group.bench_with_input(BenchmarkId::new("segments", n), &payload, |b, value| {
            b.iter(|| {
                let encoded = serde_json::to_vec(value).expect("serialization");
                sha256_hex(&encoded)
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: stage budget calculation / pipeline config
// ---------------------------------------------------------------------------

fn bench_pipeline_config_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/config_validation");

    // Default pipeline (all stages)
    group.bench_function("default_pipeline", |b| {
        b.iter(|| {
            let config = PipelineConfig::default();
            config.validate().expect("default config should be valid");
        });
    });

    // Minimal pipeline (Ingest + Normalize + Backend)
    group.bench_function("minimal_pipeline", |b| {
        b.iter(|| {
            PipelineBuilder::new()
                .stage(PipelineStage::Ingest)
                .stage(PipelineStage::Normalize)
                .stage(PipelineStage::Backend)
                .build()
                .expect("minimal config should be valid")
        });
    });

    // Full pipeline through builder with skip
    group.bench_function("builder_without_accelerate", |b| {
        b.iter(|| {
            PipelineBuilder::default_stages()
                .without(PipelineStage::Accelerate)
                .build()
                .expect("config without accelerate should be valid")
        });
    });

    group.finish();
}

/// Benchmark `PipelineConfig::has_stage` lookups, which are used during
/// pipeline execution to decide whether to run each stage.
fn bench_pipeline_has_stage(c: &mut Criterion) {
    let config = PipelineConfig::default();

    c.bench_function("pipeline/has_stage_lookup", |b| {
        b.iter(|| {
            let config = black_box(&config);
            black_box([
                config.has_stage(black_box(PipelineStage::Ingest)),
                config.has_stage(black_box(PipelineStage::Normalize)),
                config.has_stage(black_box(PipelineStage::Backend)),
                config.has_stage(black_box(PipelineStage::Accelerate)),
                config.has_stage(black_box(PipelineStage::Align)),
                config.has_stage(black_box(PipelineStage::Persist)),
            ]);
        });
    });
}

// ---------------------------------------------------------------------------
// Criterion harness
// ---------------------------------------------------------------------------

/// Isolated same-binary A/B for the export-writer `BufWriter` change (see
/// `src/export.rs`): writing an SRT-shaped transcript to a real file per-segment
/// through a raw `File` (one `write()` syscall per line) vs a `BufWriter`
/// (batched ~8 KiB writes). Both arms emit byte-identical file content; only the
/// syscall count differs. Interleaved unbuffered/buffered/unbuffered/buffered.
fn bench_export_srt_buffering(c: &mut Criterion) {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let dir = tempfile::tempdir().expect("tempdir");
    let n = 5_000usize;
    let segs: Vec<(f64, f64, String)> = (0..n)
        .map(|i| {
            (
                i as f64,
                i as f64 + 0.9,
                format!("segment number {i} with a handful of transcribed words"),
            )
        })
        .collect();

    let mut group = c.benchmark_group("export/srt_write");

    for arm in ["unbuffered_r1", "buffered_r1", "unbuffered_r2", "buffered_r2"] {
        let buffered = arm.starts_with("buffered");
        let path = dir.path().join(format!("{arm}.srt"));
        group.bench_function(arm, |b| {
            if buffered {
                b.iter(|| {
                    let mut file = BufWriter::new(File::create(&path).expect("create"));
                    for (i, (start, end, text)) in segs.iter().enumerate() {
                        writeln!(file, "{}", i + 1).unwrap();
                        writeln!(file, "{start:.3} --> {end:.3}").unwrap();
                        writeln!(file, "{text}\n").unwrap();
                    }
                    file.flush().unwrap();
                });
            } else {
                b.iter(|| {
                    let mut file = File::create(&path).expect("create");
                    for (i, (start, end, text)) in segs.iter().enumerate() {
                        writeln!(file, "{}", i + 1).unwrap();
                        writeln!(file, "{start:.3} --> {end:.3}").unwrap();
                        writeln!(file, "{text}\n").unwrap();
                    }
                });
            }
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_event_logging_throughput,
    bench_event_serialization,
    bench_sha256_hashing,
    bench_sha256_json_value,
    bench_pipeline_config_validation,
    bench_pipeline_has_stage,
    bench_export_srt_buffering,
);
criterion_main!(benches);
