//! Performance benchmarks for pipeline-adjacent hot paths.
//!
//! Covers event logging throughput, SHA-256 hashing performance (the same
//! primitives used by the orchestrator's replay envelope), and stage budget
//! calculation via `PipelineConfig` construction and validation.

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use franken_whisper::backend::{BackendMetrics, RouterState, RoutingOutcomeRecord};
use franken_whisper::model::{BackendKind, RunEvent, StreamedRunEvent};
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

/// Profile the adaptive router's per-backend aggregate over its full retained
/// history window. `evaluate_backend_selection` invokes this aggregate for
/// every backend while building the loss matrix and again while constructing
/// the evidence snapshot.
fn bench_router_metrics(c: &mut Criterion) {
    fn historical_metrics(history: &[RoutingOutcomeRecord]) -> BackendMetrics {
        let sample_count = history.len();
        let success_count = history.iter().filter(|record| record.success).count();
        let success_rate = success_count as f64 / sample_count as f64;
        let successful_latencies: Vec<f64> = history
            .iter()
            .filter(|record| record.success)
            .map(|record| record.latency_ms as f64)
            .collect();
        let avg_latency_ms = if successful_latencies.is_empty() {
            0.0
        } else {
            successful_latencies.iter().sum::<f64>() / successful_latencies.len() as f64
        };
        let last_error = history
            .iter()
            .rev()
            .find_map(|record| record.error_message.clone());
        BackendMetrics {
            success_rate,
            avg_latency_ms,
            last_error,
            sample_count,
            success_count,
        }
    }

    fn measure_metrics<F>(inner_steps: usize, mut metrics: F) -> Duration
    where
        F: FnMut() -> BackendMetrics,
    {
        let started = Instant::now();
        let mut checksum = 0_u64;
        for _ in 0..inner_steps {
            let result = black_box(metrics());
            checksum ^= result.avg_latency_ms.to_bits();
            checksum = checksum.rotate_left(1) ^ result.success_rate.to_bits();
            black_box(&result);
        }
        black_box(checksum);
        started.elapsed()
    }

    fn percentile(sorted: &[f64], percentile: f64) -> f64 {
        let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
        sorted[index]
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        percentile(&sorted, 0.5)
    }

    fn cv(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / values.len() as f64;
        variance.sqrt() / mean
    }

    let mut state = RouterState::new();
    let mut history = Vec::with_capacity(50);
    for i in 0..50_u64 {
        let record = RoutingOutcomeRecord {
            backend: BackendKind::WhisperCpp,
            success: i % 4 != 0,
            latency_ms: 350 + i * 17,
            error_message: (i % 4 == 0).then(|| format!("backend failure {i}")),
            recorded_at_rfc3339: format!("2026-07-22T12:{i:02}:00Z"),
        };
        state.record_outcome(record.clone());
        history.push(record);
    }

    let historical = historical_metrics(&history);
    let streamed = state.metrics_for(BackendKind::WhisperCpp);
    assert_eq!(
        serde_json::to_vec(&streamed).expect("serialize streamed metrics"),
        serde_json::to_vec(&historical).expect("serialize historical metrics")
    );
    assert_eq!(
        streamed.avg_latency_ms.to_bits(),
        historical.avg_latency_ms.to_bits()
    );

    // One remote executable performs the null and candidate comparisons in an
    // order-alternated sequence. Each historical arm runs for about 200 ms at
    // the profiled 200 ns/call baseline, smoothing shared-worker scheduling.
    let inner_steps = 1_000_000;
    let mut null_ratios = Vec::with_capacity(21);
    let mut speedups = Vec::with_capacity(21);
    let mut candidate_ns = Vec::with_capacity(21);
    for pair in 0..21 {
        if pair % 2 == 0 {
            let null_a = measure_metrics(inner_steps, || historical_metrics(&history));
            let null_b = measure_metrics(inner_steps, || historical_metrics(&history));
            let baseline = measure_metrics(inner_steps, || historical_metrics(&history));
            let candidate =
                measure_metrics(inner_steps, || state.metrics_for(BackendKind::WhisperCpp));
            null_ratios.push(null_a.as_secs_f64() / null_b.as_secs_f64());
            speedups.push(baseline.as_secs_f64() / candidate.as_secs_f64());
            candidate_ns.push(candidate.as_secs_f64() * 1e9 / inner_steps as f64);
        } else {
            let candidate =
                measure_metrics(inner_steps, || state.metrics_for(BackendKind::WhisperCpp));
            let baseline = measure_metrics(inner_steps, || historical_metrics(&history));
            let null_b = measure_metrics(inner_steps, || historical_metrics(&history));
            let null_a = measure_metrics(inner_steps, || historical_metrics(&history));
            null_ratios.push(null_a.as_secs_f64() / null_b.as_secs_f64());
            speedups.push(baseline.as_secs_f64() / candidate.as_secs_f64());
            candidate_ns.push(candidate.as_secs_f64() * 1e9 / inner_steps as f64);
        }
    }

    let mut sorted_null = null_ratios.clone();
    sorted_null.sort_by(f64::total_cmp);
    let mut sorted_speedups = speedups.clone();
    sorted_speedups.sort_by(f64::total_cmp);
    let wins = speedups.iter().filter(|speedup| **speedup > 1.0).count();
    eprintln!(
        "ROUTER_METRICS_AB inner_steps={inner_steps} null={null_ratios:?} speedup={speedups:?} candidate_ns={candidate_ns:?} null_p10={:.6} null_median={:.6} null_p90={:.6} speedup_p10={:.6} speedup_median={:.6} speedup_p90={:.6} candidate_cv={:.6} wins={wins}/21",
        percentile(&sorted_null, 0.1),
        median(&null_ratios),
        percentile(&sorted_null, 0.9),
        percentile(&sorted_speedups, 0.1),
        median(&speedups),
        percentile(&sorted_speedups, 0.9),
        cv(&candidate_ns),
    );

    c.bench_function("pipeline/router_metrics/history_50", |b| {
        b.iter(|| state.metrics_for(black_box(BackendKind::WhisperCpp)));
    });
}

/// Same-worker A/B for hoisting the three state-invariant backend aggregates
/// out of the adaptive loss matrix's three availability rows.
fn bench_router_loss_hoist(c: &mut Criterion) {
    const BACKENDS: [BackendKind; 3] = [
        BackendKind::WhisperCpp,
        BackendKind::InsanelyFast,
        BackendKind::WhisperDiarization,
    ];

    fn checksum(metrics: &BackendMetrics) -> u64 {
        metrics.success_rate.to_bits()
            ^ metrics.avg_latency_ms.to_bits().rotate_left(7)
            ^ (metrics.sample_count as u64).rotate_left(13)
            ^ (metrics.success_count as u64).rotate_left(19)
            ^ metrics
                .last_error
                .as_ref()
                .map_or(0, |error| error.len() as u64)
    }

    fn historical_loss_inputs(state: &RouterState) -> u64 {
        let mut result = 0_u64;
        for state_idx in 0..3_u32 {
            for backend in BACKENDS {
                let metrics = black_box(state.metrics_for(backend));
                result = result.rotate_left(state_idx + 1) ^ checksum(&metrics);
            }
        }
        result
    }

    fn hoisted_loss_inputs(state: &RouterState) -> u64 {
        let metrics: [BackendMetrics; 3] =
            std::array::from_fn(|index| black_box(state.metrics_for(BACKENDS[index])));
        let mut result = 0_u64;
        for state_idx in 0..3_u32 {
            for backend_metrics in &metrics {
                result = result.rotate_left(state_idx + 1) ^ checksum(backend_metrics);
            }
        }
        result
    }

    fn measure<F>(inner_steps: usize, mut operation: F) -> Duration
    where
        F: FnMut() -> u64,
    {
        let started = Instant::now();
        let mut aggregate = 0_u64;
        for _ in 0..inner_steps {
            aggregate ^= black_box(operation());
        }
        black_box(aggregate);
        started.elapsed()
    }

    fn percentile(sorted: &[f64], percentile: f64) -> f64 {
        let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
        sorted[index]
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        percentile(&sorted, 0.5)
    }

    fn cv(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / values.len() as f64;
        variance.sqrt() / mean
    }

    let mut state = RouterState::new();
    for backend in BACKENDS {
        for sample in 0..50_u64 {
            state.record_outcome(RoutingOutcomeRecord {
                backend,
                success: sample % 4 != 0,
                latency_ms: 350 + sample * 17,
                error_message: (sample % 4 == 0).then(|| format!("{backend:?} failure {sample}")),
                recorded_at_rfc3339: format!("2026-07-22T12:{sample:02}:00Z"),
            });
        }
    }

    let historical_metrics: Vec<BackendMetrics> = (0..3)
        .flat_map(|_| BACKENDS.map(|backend| state.metrics_for(backend)))
        .collect();
    let cached_metrics: [BackendMetrics; 3] =
        std::array::from_fn(|index| state.metrics_for(BACKENDS[index]));
    let mut candidate_metrics = Vec::with_capacity(9);
    for _ in 0..3 {
        candidate_metrics.extend(cached_metrics.iter());
    }
    assert_eq!(
        serde_json::to_vec(&historical_metrics).expect("serialize historical loss inputs"),
        serde_json::to_vec(&candidate_metrics).expect("serialize hoisted loss inputs")
    );
    assert_eq!(historical_loss_inputs(&state), hoisted_loss_inputs(&state));

    let inner_steps = 200_000;
    let mut null_ratios = Vec::with_capacity(21);
    let mut speedups = Vec::with_capacity(21);
    let mut candidate_ns = Vec::with_capacity(21);
    for pair in 0..21 {
        if pair % 2 == 0 {
            let null_a = measure(inner_steps, || historical_loss_inputs(&state));
            let null_b = measure(inner_steps, || historical_loss_inputs(&state));
            let baseline = measure(inner_steps, || historical_loss_inputs(&state));
            let candidate = measure(inner_steps, || hoisted_loss_inputs(&state));
            null_ratios.push(null_a.as_secs_f64() / null_b.as_secs_f64());
            speedups.push(baseline.as_secs_f64() / candidate.as_secs_f64());
            candidate_ns.push(candidate.as_secs_f64() * 1e9 / inner_steps as f64);
        } else {
            let candidate = measure(inner_steps, || hoisted_loss_inputs(&state));
            let baseline = measure(inner_steps, || historical_loss_inputs(&state));
            let null_b = measure(inner_steps, || historical_loss_inputs(&state));
            let null_a = measure(inner_steps, || historical_loss_inputs(&state));
            null_ratios.push(null_a.as_secs_f64() / null_b.as_secs_f64());
            speedups.push(baseline.as_secs_f64() / candidate.as_secs_f64());
            candidate_ns.push(candidate.as_secs_f64() * 1e9 / inner_steps as f64);
        }
    }

    let mut sorted_null = null_ratios.clone();
    sorted_null.sort_by(f64::total_cmp);
    let mut sorted_speedups = speedups.clone();
    sorted_speedups.sort_by(f64::total_cmp);
    let null_median = median(&null_ratios);
    let null_p90 = percentile(&sorted_null, 0.9);
    let speedup_p10 = percentile(&sorted_speedups, 0.1);
    let candidate_cv = cv(&candidate_ns);
    let wins = speedups.iter().filter(|speedup| **speedup > 1.0).count();
    eprintln!(
        "ROUTER_LOSS_HOIST_AB inner_steps={inner_steps} null={null_ratios:?} speedup={speedups:?} candidate_ns={candidate_ns:?} null_p10={:.6} null_median={null_median:.6} null_p90={null_p90:.6} speedup_p10={speedup_p10:.6} speedup_median={:.6} speedup_p90={:.6} candidate_cv={candidate_cv:.6} wins={wins}/21",
        percentile(&sorted_null, 0.1),
        median(&speedups),
        percentile(&sorted_speedups, 0.9),
    );

    assert!((0.95..=1.05).contains(&null_median));
    assert!(candidate_cv < 0.05);
    assert!(wins >= 18);
    assert!(speedup_p10 > null_p90.max(1.10));

    c.bench_function("pipeline/router_loss_hoist/history_50", |b| {
        b.iter(|| hoisted_loss_inputs(black_box(&state)));
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
    bench_router_metrics,
    bench_router_loss_hoist,
    bench_export_srt_buffering,
);
criterion_main!(benches);
