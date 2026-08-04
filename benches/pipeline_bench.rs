//! Performance benchmarks for pipeline-adjacent hot paths.
//!
//! Covers event logging throughput, SHA-256 hashing performance (the same
//! primitives used by the orchestrator's replay envelope), and stage budget
//! calculation via `PipelineConfig` construction and validation.

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use franken_whisper::backend::{
    BackendMetrics, RouterState, RoutingEvidenceLedger, RoutingEvidenceLedgerEntry,
    RoutingOutcomeRecord,
};
use franken_whisper::model::{BackendKind, RunEvent, StreamedRunEvent};
use franken_whisper::orchestrator::{PipelineBuilder, PipelineConfig, PipelineStage};
use franken_whisper::speculation::{
    CorrectionDecision, CorrectionDrift, CorrectionEvent, CorrectionEvidenceEntry,
    CorrectionEvidenceLedger, SpeculationWindowController,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const HARNESS_PAIRS: usize = 41;
const HARNESS_MIN_OF: usize = 3;
const HARNESS_TARGET_NS: u128 = 2_000_000;
const HARNESS_BOOTSTRAP_RESAMPLES: usize = 20_000;

fn self_identity() -> String {
    let Ok(path) = std::env::current_exe() else {
        return "unavailable".to_owned();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return "unavailable".to_owned();
    };
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    format!(
        "{:x} ({} bytes) {}",
        hasher.finalize(),
        bytes.len(),
        path.display()
    )
}

fn harness_percentile(values: &[f64], percentile: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn harness_cv(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean
}

fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
    let mut state = 0x510e_527f_ade6_82d1_u64 ^ values.len() as u64;
    let mut sample = Vec::with_capacity(values.len());
    let mut medians = Vec::with_capacity(HARNESS_BOOTSTRAP_RESAMPLES);
    for _ in 0..HARNESS_BOOTSTRAP_RESAMPLES {
        sample.clear();
        for _ in 0..values.len() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            sample.push(values[state as usize % values.len()]);
        }
        sample.sort_by(f64::total_cmp);
        medians.push(sample[sample.len() / 2]);
    }
    medians.sort_by(f64::total_cmp);
    (
        medians[HARNESS_BOOTSTRAP_RESAMPLES * 25 / 1_000],
        medians[HARNESS_BOOTSTRAP_RESAMPLES * 975 / 1_000],
    )
}

fn harness_calibrated_repetitions<F, R>(operation: &mut F) -> usize
where
    F: FnMut() -> R,
{
    let probe_repetitions = 64;
    let started = Instant::now();
    for _ in 0..probe_repetitions {
        black_box(operation());
    }
    let probe_ns = started.elapsed().as_nanos().max(1);
    usize::try_from((HARNESS_TARGET_NS * probe_repetitions as u128 / probe_ns).clamp(1, 5_000_000))
        .expect("bounded repetitions")
}

fn harness_min_ns_per_call<F, R>(operation: &mut F, repetitions: usize) -> f64
where
    F: FnMut() -> R,
{
    (0..HARNESS_MIN_OF)
        .map(|_| {
            let started = Instant::now();
            for _ in 0..repetitions {
                black_box(operation());
            }
            started.elapsed().as_nanos() as f64
        })
        .fold(f64::INFINITY, f64::min)
        / repetitions as f64
}

fn harness_paired_ratios<FB, FC, RB, RC>(
    baseline: &mut FB,
    contender: &mut FC,
    baseline_repetitions: usize,
    contender_repetitions: usize,
) -> Vec<f64>
where
    FB: FnMut() -> RB,
    FC: FnMut() -> RC,
{
    let mut ratios = Vec::with_capacity(HARNESS_PAIRS);
    for pair in 0..HARNESS_PAIRS {
        let (baseline_ns, contender_ns) = if pair.is_multiple_of(2) {
            (
                harness_min_ns_per_call(baseline, baseline_repetitions),
                harness_min_ns_per_call(contender, contender_repetitions),
            )
        } else {
            let contender_ns = harness_min_ns_per_call(contender, contender_repetitions);
            let baseline_ns = harness_min_ns_per_call(baseline, baseline_repetitions);
            (baseline_ns, contender_ns)
        };
        ratios.push(baseline_ns / contender_ns.max(f64::MIN_POSITIVE));
    }
    ratios
}

fn report_harness_gate(label: &str, null_ratios: &[f64], candidate_ratios: &[f64]) -> bool {
    let (null_ci_low, null_ci_high) = bootstrap_median_ci(null_ratios);
    let (candidate_ci_low, candidate_ci_high) = bootstrap_median_ci(candidate_ratios);
    let null_half_width = (1.0 - null_ci_low).abs().max((null_ci_high - 1.0).abs());
    let required_speedup = 1.0 + 2.0 * null_half_width;
    let candidate_median = harness_percentile(candidate_ratios, 50);
    let keep = candidate_median >= required_speedup;
    println!(
        "{label}_NULL ratios={null_ratios:?} median={:.6} median_ci95=[{null_ci_low:.6},{null_ci_high:.6}] cv={:.6}",
        harness_percentile(null_ratios, 50),
        harness_cv(null_ratios)
    );
    println!(
        "{label}_CANDIDATE ratios={candidate_ratios:?} median={candidate_median:.6} median_ci95=[{candidate_ci_low:.6},{candidate_ci_high:.6}] cv={:.6} wins={}/{}",
        harness_cv(candidate_ratios),
        candidate_ratios
            .iter()
            .filter(|ratio| **ratio > 1.0)
            .count(),
        candidate_ratios.len()
    );
    println!(
        "{label}_GATE method=median_vs_null_ci95_2x_margin null_half_width={null_half_width:.6} required_speedup={required_speedup:.6} candidate_median={candidate_median:.6} cv_is_provenance_only=true verdict={}",
        if keep { "KEEP" } else { "REJECT" }
    );
    keep
}

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
    if std::env::args()
        .nth(1)
        .is_some_and(|filter| !filter.starts_with('-') && !filter.contains("router_metrics"))
    {
        return;
    }

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
    for pair in 0..21 {
        if pair % 2 == 0 {
            let null_a = measure_metrics(inner_steps, || historical_metrics(&history));
            let null_b = measure_metrics(inner_steps, || historical_metrics(&history));
            null_ratios.push(null_a.as_secs_f64() / null_b.as_secs_f64());
        } else {
            let null_b = measure_metrics(inner_steps, || historical_metrics(&history));
            let null_a = measure_metrics(inner_steps, || historical_metrics(&history));
            null_ratios.push(null_a.as_secs_f64() / null_b.as_secs_f64());
        }
    }
    let mut speedups = Vec::with_capacity(21);
    let mut candidate_ns = Vec::with_capacity(21);
    for pair in 0..21 {
        if pair % 2 == 0 {
            let baseline = measure_metrics(inner_steps, || historical_metrics(&history));
            let candidate =
                measure_metrics(inner_steps, || state.metrics_for(BackendKind::WhisperCpp));
            speedups.push(baseline.as_secs_f64() / candidate.as_secs_f64());
            candidate_ns.push(candidate.as_secs_f64() * 1e9 / inner_steps as f64);
        } else {
            let candidate =
                measure_metrics(inner_steps, || state.metrics_for(BackendKind::WhisperCpp));
            let baseline = measure_metrics(inner_steps, || historical_metrics(&history));
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
    report_harness_gate("ROUTER_METRICS", &null_ratios, &speedups);

    c.bench_function("pipeline/router_metrics/history_50", |b| {
        b.iter(|| state.metrics_for(black_box(BackendKind::WhisperCpp)));
    });
}

/// Same-worker A/B for hoisting the three state-invariant backend aggregates
/// out of the adaptive loss matrix's three availability rows.
fn bench_router_loss_hoist(c: &mut Criterion) {
    if std::env::args()
        .nth(1)
        .is_some_and(|filter| !filter.starts_with('-') && !filter.contains("router_loss_hoist"))
    {
        return;
    }

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
    for pair in 0..21 {
        if pair % 2 == 0 {
            let null_a = measure(inner_steps, || historical_loss_inputs(&state));
            let null_b = measure(inner_steps, || historical_loss_inputs(&state));
            null_ratios.push(null_a.as_secs_f64() / null_b.as_secs_f64());
        } else {
            let null_b = measure(inner_steps, || historical_loss_inputs(&state));
            let null_a = measure(inner_steps, || historical_loss_inputs(&state));
            null_ratios.push(null_a.as_secs_f64() / null_b.as_secs_f64());
        }
    }
    let mut speedups = Vec::with_capacity(21);
    let mut candidate_ns = Vec::with_capacity(21);
    for pair in 0..21 {
        if pair % 2 == 0 {
            let baseline = measure(inner_steps, || historical_loss_inputs(&state));
            let candidate = measure(inner_steps, || hoisted_loss_inputs(&state));
            speedups.push(baseline.as_secs_f64() / candidate.as_secs_f64());
            candidate_ns.push(candidate.as_secs_f64() * 1e9 / inner_steps as f64);
        } else {
            let candidate = measure(inner_steps, || hoisted_loss_inputs(&state));
            let baseline = measure(inner_steps, || historical_loss_inputs(&state));
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
    report_harness_gate("ROUTER_LOSS_HOIST", &null_ratios, &speedups);

    c.bench_function("pipeline/router_loss_hoist/history_50", |b| {
        b.iter(|| hoisted_loss_inputs(black_box(&state)));
    });
}

/// Same-worker A/B for streaming the optional-Brier sum and count instead of
/// materializing a temporary `Vec` during every diagnostics snapshot.
fn bench_router_diagnostics_profile(c: &mut Criterion) {
    if std::env::args().nth(1).is_some_and(|filter| {
        !filter.starts_with('-') && !filter.contains("router_diagnostics_profile")
    }) {
        return;
    }

    fn historical_brier_average(ledger: &RoutingEvidenceLedger) -> Option<f64> {
        let brier_values: Vec<f64> = ledger
            .entries()
            .iter()
            .filter_map(|entry| entry.brier_score)
            .collect();
        if brier_values.is_empty() {
            None
        } else {
            Some(brier_values.iter().sum::<f64>() / brier_values.len() as f64)
        }
    }

    fn streamed_brier_average(ledger: &RoutingEvidenceLedger) -> Option<f64> {
        let (sum, count) = ledger
            .entries()
            .iter()
            .filter_map(|entry| entry.brier_score)
            .fold((0.0, 0_usize), |(sum, count), score| {
                (sum + score, count + 1)
            });
        (count != 0).then(|| sum / count as f64)
    }

    fn measure<F>(inner_steps: usize, mut operation: F) -> Duration
    where
        F: FnMut() -> Option<f64>,
    {
        let started = Instant::now();
        let mut aggregate = 0_u64;
        for _ in 0..inner_steps {
            aggregate = aggregate.rotate_left(1) ^ black_box(operation()).map_or(0, f64::to_bits);
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

    let mut ledger = RoutingEvidenceLedger::new(200);
    for sample in 0..200_u64 {
        ledger.record(RoutingEvidenceLedgerEntry {
            decision_id: format!("decision-{sample}"),
            trace_id: format!("trace-{}", sample / 4),
            timestamp_rfc3339: format!("2026-07-22T12:{:02}:{:02}Z", sample / 60, sample % 60),
            observed_state: "all_available".to_owned(),
            chosen_action: "whisper_cpp".to_owned(),
            recommended_order: vec![
                "whisper_cpp".to_owned(),
                "insanely_fast".to_owned(),
                "whisper_diarization".to_owned(),
            ],
            fallback_active: sample % 10 == 0,
            fallback_reason: (sample % 10 == 0).then(|| "calibration".to_owned()),
            posterior_snapshot: vec![0.6, 0.3, 0.1],
            calibration_score: 0.75 + (sample % 20) as f64 * 0.005,
            brier_score: (sample % 5 != 0).then_some(0.08 + (sample % 17) as f64 * 0.01),
            e_process: 1.0 + sample as f64 * 0.01,
            ci_width: 0.2,
            adaptive_mode: true,
            policy_id: "router-policy-v1".to_owned(),
            loss_matrix_hash: format!("loss-{sample:016x}"),
            availability: vec![
                ("whisper_cpp".to_owned(), true),
                ("insanely_fast".to_owned(), true),
                ("whisper_diarization".to_owned(), true),
            ],
            duration_bucket: "medium".to_owned(),
            diarize: sample % 3 == 0,
            actual_outcome: None,
        });
    }

    let expected = historical_brier_average(&ledger).expect("profile has Brier values");
    assert_eq!(
        streamed_brier_average(&ledger).map(f64::to_bits),
        Some(expected.to_bits())
    );
    let diagnostics = ledger.diagnostics();
    assert_eq!(
        diagnostics["avg_brier_score"]
            .as_f64()
            .expect("diagnostics Brier average")
            .to_bits(),
        expected.to_bits()
    );

    let mut historical_diagnostics = diagnostics.clone();
    historical_diagnostics["avg_brier_score"] = serde_json::json!(expected);
    assert_eq!(
        serde_json::to_vec(&diagnostics).expect("serialize streamed diagnostics"),
        serde_json::to_vec(&historical_diagnostics).expect("serialize historical diagnostics")
    );

    let inner_steps = 250_000;
    let mut null_ratios = Vec::with_capacity(21);
    for pair in 0..21 {
        if pair % 2 == 0 {
            let null_a = measure(inner_steps, || historical_brier_average(&ledger));
            let null_b = measure(inner_steps, || historical_brier_average(&ledger));
            null_ratios.push(null_a.as_secs_f64() / null_b.as_secs_f64());
        } else {
            let null_b = measure(inner_steps, || historical_brier_average(&ledger));
            let null_a = measure(inner_steps, || historical_brier_average(&ledger));
            null_ratios.push(null_a.as_secs_f64() / null_b.as_secs_f64());
        }
    }
    let mut speedups = Vec::with_capacity(21);
    let mut candidate_ns = Vec::with_capacity(21);
    for pair in 0..21 {
        if pair % 2 == 0 {
            let baseline = measure(inner_steps, || historical_brier_average(&ledger));
            let candidate = measure(inner_steps, || streamed_brier_average(&ledger));
            speedups.push(baseline.as_secs_f64() / candidate.as_secs_f64());
            candidate_ns.push(candidate.as_secs_f64() * 1e9 / inner_steps as f64);
        } else {
            let candidate = measure(inner_steps, || streamed_brier_average(&ledger));
            let baseline = measure(inner_steps, || historical_brier_average(&ledger));
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
        "ROUTER_BRIER_STREAM_AB inner_steps={inner_steps} null={null_ratios:?} speedup={speedups:?} candidate_ns={candidate_ns:?} null_p10={:.6} null_median={null_median:.6} null_p90={null_p90:.6} speedup_p10={speedup_p10:.6} speedup_median={:.6} speedup_p90={:.6} candidate_cv={candidate_cv:.6} wins={wins}/21",
        percentile(&sorted_null, 0.1),
        median(&speedups),
        percentile(&sorted_speedups, 0.9),
    );
    report_harness_gate("ROUTER_BRIER_STREAM", &null_ratios, &speedups);

    let mut group = c.benchmark_group("pipeline/router_diagnostics_profile");
    group.bench_function("full_200", |b| {
        b.iter(|| black_box(&ledger).diagnostics());
    });
    group.bench_function("brier_stream_200", |b| {
        b.iter(|| streamed_brier_average(black_box(&ledger)));
    });
    group.finish();
}

/// Profile the four independent count/calibration passes in a realistic
/// routing-diagnostics snapshot before considering a fused traversal.
fn bench_router_diagnostics_counts_profile(c: &mut Criterion) {
    if std::env::args().nth(1).is_some_and(|filter| {
        !filter.starts_with('-') && !filter.contains("router_diagnostics_counts")
    }) {
        return;
    }

    fn historical_counts(ledger: &RoutingEvidenceLedger) -> (usize, usize, usize, f64) {
        let fallback_count = ledger
            .entries()
            .iter()
            .filter(|entry| entry.fallback_active)
            .count();
        let resolved_count = ledger
            .entries()
            .iter()
            .filter(|entry| entry.actual_outcome.is_some())
            .count();
        let resolved_success_count = ledger
            .entries()
            .iter()
            .filter_map(|entry| entry.actual_outcome.as_ref())
            .filter(|outcome| outcome.success)
            .count();
        let calibration_sum = ledger
            .entries()
            .iter()
            .map(|entry| entry.calibration_score)
            .sum::<f64>();
        (
            fallback_count,
            resolved_count,
            resolved_success_count,
            calibration_sum,
        )
    }

    fn historical_diagnostics(ledger: &RoutingEvidenceLedger) -> Value {
        let total = ledger.entries().len();
        let (fallback_count, resolved, resolved_success, calibration_sum) =
            historical_counts(ledger);
        let avg_calibration = if total > 0 {
            calibration_sum / total as f64
        } else {
            0.0
        };
        let (brier_sum, brier_count) = ledger
            .entries()
            .iter()
            .filter_map(|entry| entry.brier_score)
            .fold((0.0, 0_usize), |(sum, count), score| {
                (sum + score, count + 1)
            });
        let avg_brier = (brier_count != 0).then(|| brier_sum / brier_count as f64);

        serde_json::json!({
            "total_entries": total,
            "total_ever_recorded": ledger.total_recorded(),
            "capacity": ledger.capacity(),
            "fallback_count": fallback_count,
            "fallback_rate": if total > 0 { fallback_count as f64 / total as f64 } else { 0.0 },
            "resolved_count": resolved,
            "resolved_success_count": resolved_success,
            "resolved_success_rate": if resolved > 0 { resolved_success as f64 / resolved as f64 } else { 0.0 },
            "avg_calibration_score": avg_calibration,
            "avg_brier_score": avg_brier,
        })
    }

    let mut ledger = RoutingEvidenceLedger::new(200);
    for sample in 0..200_u64 {
        ledger.record(RoutingEvidenceLedgerEntry {
            decision_id: format!("count-decision-{sample}"),
            trace_id: format!("count-trace-{}", sample / 4),
            timestamp_rfc3339: format!("2026-07-22T13:{:02}:{:02}Z", sample / 60, sample % 60),
            observed_state: "all_available".to_owned(),
            chosen_action: "whisper_cpp".to_owned(),
            recommended_order: vec!["whisper_cpp".to_owned(), "insanely_fast".to_owned()],
            fallback_active: sample % 10 == 0,
            fallback_reason: (sample % 10 == 0).then(|| "calibration".to_owned()),
            posterior_snapshot: vec![0.6, 0.3, 0.1],
            calibration_score: 0.75 + (sample % 20) as f64 * 0.005,
            brier_score: (sample % 5 != 0).then_some(0.08 + (sample % 17) as f64 * 0.01),
            e_process: 1.0 + sample as f64 * 0.01,
            ci_width: 0.2,
            adaptive_mode: true,
            policy_id: "router-policy-v1".to_owned(),
            loss_matrix_hash: format!("count-loss-{sample:016x}"),
            availability: vec![("whisper_cpp".to_owned(), true)],
            duration_bucket: "medium".to_owned(),
            diarize: sample % 3 == 0,
            actual_outcome: (sample % 4 != 0).then(|| RoutingOutcomeRecord {
                backend: BackendKind::WhisperCpp,
                success: sample % 5 != 0,
                latency_ms: 300 + sample,
                error_message: (sample % 5 == 0).then(|| format!("failure-{sample}")),
                recorded_at_rfc3339: format!("2026-07-22T13:{:02}:30Z", sample / 60),
            }),
        });
    }

    let diagnostics = ledger.diagnostics();
    let historical = historical_counts(&ledger);
    assert_eq!(diagnostics["fallback_count"], historical.0);
    assert_eq!(diagnostics["resolved_count"], historical.1);
    assert_eq!(diagnostics["resolved_success_count"], historical.2);
    assert_eq!(
        diagnostics["avg_calibration_score"]
            .as_f64()
            .expect("calibration average")
            .to_bits(),
        (historical.3 / ledger.entries().len() as f64).to_bits()
    );

    let historical_json = historical_diagnostics(&ledger);
    let candidate_json = ledger.diagnostics();
    assert_eq!(
        serde_json::to_vec(&candidate_json).expect("serialize candidate diagnostics"),
        serde_json::to_vec(&historical_json).expect("serialize historical diagnostics")
    );

    let mut historical = || historical_diagnostics(&ledger);
    let historical_repetitions = harness_calibrated_repetitions(&mut historical);
    let mut candidate = || ledger.diagnostics();
    let candidate_repetitions = harness_calibrated_repetitions(&mut candidate);
    let mut null_left = || historical_diagnostics(&ledger);
    let mut null_right = || historical_diagnostics(&ledger);
    let null_ratios = harness_paired_ratios(
        &mut null_left,
        &mut null_right,
        historical_repetitions,
        historical_repetitions,
    );
    let candidate_ratios = harness_paired_ratios(
        &mut historical,
        &mut candidate,
        historical_repetitions,
        candidate_repetitions,
    );
    println!(
        "ROUTER_COUNTS_FUSION_AB historical_repetitions={historical_repetitions} candidate_repetitions={candidate_repetitions} pairs={HARNESS_PAIRS} min_of={HARNESS_MIN_OF}"
    );
    report_harness_gate("ROUTER_COUNTS_FUSION", &null_ratios, &candidate_ratios);

    let mut group = c.benchmark_group("pipeline/router_diagnostics_counts_profile");
    group.bench_function("candidate_full_200", |b| {
        b.iter(|| black_box(&ledger).diagnostics());
    });
    group.bench_function("historical_full_200", |b| {
        b.iter(|| historical_diagnostics(black_box(&ledger)));
    });
    group.bench_function("historical_four_passes_200", |b| {
        b.iter(|| historical_counts(black_box(&ledger)));
    });
    group.finish();
}

/// Profile the six historical scans performed by correction diagnostics before
/// considering a fused traversal.
fn bench_speculation_diagnostics_profile(c: &mut Criterion) {
    if std::env::args().nth(1).is_some_and(|filter| {
        !filter.starts_with('-') && !filter.contains("speculation_diagnostics_profile")
    }) {
        return;
    }

    fn historical_six_passes(ledger: &CorrectionEvidenceLedger) -> [f64; 5] {
        [
            ledger.correction_rate(),
            ledger.mean_fast_latency(),
            ledger.mean_quality_latency(),
            ledger.mean_wer(),
            ledger.latency_savings_pct(),
        ]
    }

    fn historical_diagnostics(ledger: &CorrectionEvidenceLedger) -> Value {
        let values = historical_six_passes(ledger);
        serde_json::json!({
            "correction_rate": values[0],
            "mean_fast_latency_ms": values[1],
            "mean_quality_latency_ms": values[2],
            "mean_wer": values[3],
            "latency_savings_pct": values[4],
        })
    }

    fn measure(
        ledger: &CorrectionEvidenceLedger,
        historical: bool,
        repetitions: usize,
    ) -> Duration {
        let started = Instant::now();
        let mut checksum = 0_u64;
        for index in 0..repetitions {
            let value = if historical {
                historical_diagnostics(ledger)
            } else {
                ledger.diagnostics()
            };
            checksum ^= value["mean_wer"]
                .as_f64()
                .unwrap_or_default()
                .to_bits()
                .rotate_left((index & 63) as u32);
            black_box(value);
        }
        black_box(checksum);
        started.elapsed()
    }

    fn calibrated_repetitions(ledger: &CorrectionEvidenceLedger, historical: bool) -> usize {
        let probe_repetitions = 64;
        let probe_ns = measure(ledger, historical, probe_repetitions)
            .as_nanos()
            .max(1);
        usize::try_from(
            (HARNESS_TARGET_NS * probe_repetitions as u128 / probe_ns).clamp(1, 5_000_000),
        )
        .expect("bounded repetitions")
    }

    fn min_ns_per_call(
        ledger: &CorrectionEvidenceLedger,
        historical: bool,
        repetitions: usize,
    ) -> f64 {
        (0..HARNESS_MIN_OF)
            .map(|_| measure(ledger, historical, repetitions).as_nanos() as f64)
            .fold(f64::INFINITY, f64::min)
            / repetitions as f64
    }

    fn paired_ratios(
        ledger: &CorrectionEvidenceLedger,
        contender_historical: bool,
        historical_repetitions: usize,
        contender_repetitions: usize,
    ) -> Vec<f64> {
        let mut ratios = Vec::with_capacity(HARNESS_PAIRS);
        for pair in 0..HARNESS_PAIRS {
            let (historical_ns, contender_ns) = if pair.is_multiple_of(2) {
                (
                    min_ns_per_call(ledger, true, historical_repetitions),
                    min_ns_per_call(ledger, contender_historical, contender_repetitions),
                )
            } else {
                let contender_ns =
                    min_ns_per_call(ledger, contender_historical, contender_repetitions);
                let historical_ns = min_ns_per_call(ledger, true, historical_repetitions);
                (historical_ns, contender_ns)
            };
            ratios.push(historical_ns / contender_ns.max(f64::MIN_POSITIVE));
        }
        ratios
    }

    let mut ledger = CorrectionEvidenceLedger::new(200);
    for sample in 0..200_u64 {
        ledger.record(CorrectionEvidenceEntry {
            entry_id: sample,
            window_id: sample,
            run_id: format!("speculation-run-{}", sample / 20),
            timestamp_rfc3339: format!("2026-07-23T14:{:02}:{:02}Z", sample / 60, sample % 60),
            fast_model_id: "tiny.en".to_owned(),
            fast_latency_ms: 80 + sample % 41,
            fast_confidence_mean: 0.77 + (sample % 13) as f64 * 0.01,
            fast_segment_count: 1 + sample as usize % 4,
            quality_model_id: "large-v3-turbo".to_owned(),
            quality_latency_ms: 360 + sample % 97,
            quality_confidence_mean: 0.88 + (sample % 9) as f64 * 0.01,
            quality_segment_count: 1 + sample as usize % 5,
            drift: CorrectionDrift {
                wer_approx: (sample % 17) as f64 * 0.01,
                confidence_delta: (sample % 11) as f64 * 0.005,
                segment_count_delta: sample as i32 % 5 - 2,
                text_edit_distance: sample as usize % 12,
            },
            decision: match sample % 8 {
                0 => "correct".to_owned(),
                1 => " corrected ".to_owned(),
                2 => "CORRECTION".to_owned(),
                _ => "confirm".to_owned(),
            },
            window_size_ms: 2_000 + sample % 5 * 500,
            correction_rate_at_decision: (sample % 20) as f64 / 20.0,
            controller_confidence: (sample.min(20)) as f64 / 20.0,
            fallback_active: sample % 19 == 0,
            fallback_reason: (sample % 19 == 0).then(|| "calibration".to_owned()),
        });
    }

    let diagnostics = ledger.diagnostics();
    let historical = historical_six_passes(&ledger);
    for (field, expected) in [
        ("correction_rate", historical[0]),
        ("mean_fast_latency_ms", historical[1]),
        ("mean_quality_latency_ms", historical[2]),
        ("mean_wer", historical[3]),
        ("latency_savings_pct", historical[4]),
    ] {
        assert_eq!(
            diagnostics[field]
                .as_f64()
                .expect("diagnostic field")
                .to_bits(),
            expected.to_bits(),
            "field={field}"
        );
    }
    assert_eq!(
        serde_json::to_vec(&diagnostics).expect("serialize candidate diagnostics"),
        serde_json::to_vec(&historical_diagnostics(&ledger))
            .expect("serialize historical diagnostics")
    );

    let historical_repetitions = calibrated_repetitions(&ledger, true);
    let candidate_repetitions = calibrated_repetitions(&ledger, false);
    let null_ratios = paired_ratios(
        &ledger,
        true,
        historical_repetitions,
        historical_repetitions,
    );
    let candidate_ratios = paired_ratios(
        &ledger,
        false,
        historical_repetitions,
        candidate_repetitions,
    );
    let (null_ci_low, null_ci_high) = bootstrap_median_ci(&null_ratios);
    let (candidate_ci_low, candidate_ci_high) = bootstrap_median_ci(&candidate_ratios);
    let null_half_width = (1.0 - null_ci_low).abs().max((null_ci_high - 1.0).abs());
    let required_speedup = 1.0 + 2.0 * null_half_width;
    let candidate_median = harness_percentile(&candidate_ratios, 50);
    let keep = candidate_median >= required_speedup;
    println!(
        "CORRECTION_DIAGNOSTICS_AB historical_repetitions={historical_repetitions} candidate_repetitions={candidate_repetitions} pairs={HARNESS_PAIRS} min_of={HARNESS_MIN_OF}"
    );
    println!(
        "CORRECTION_DIAGNOSTICS_NULL ratios={null_ratios:?} median={:.6} median_ci95=[{null_ci_low:.6},{null_ci_high:.6}] cv={:.6}",
        harness_percentile(&null_ratios, 50),
        harness_cv(&null_ratios)
    );
    println!(
        "CORRECTION_DIAGNOSTICS_CANDIDATE ratios={candidate_ratios:?} median={candidate_median:.6} median_ci95=[{candidate_ci_low:.6},{candidate_ci_high:.6}] cv={:.6} wins={}/{}",
        harness_cv(&candidate_ratios),
        candidate_ratios
            .iter()
            .filter(|ratio| **ratio > 1.0)
            .count(),
        HARNESS_PAIRS
    );
    println!(
        "CORRECTION_DIAGNOSTICS_GATE method=median_vs_null_ci95_2x_margin null_half_width={null_half_width:.6} required_speedup={required_speedup:.6} candidate_median={candidate_median:.6} cv_is_provenance_only=true verdict={}",
        if keep { "KEEP" } else { "REJECT" }
    );

    let mut group = c.benchmark_group("pipeline/speculation_diagnostics_profile");
    group.bench_function("full_200", |b| {
        b.iter(|| black_box(&ledger).diagnostics());
    });
    group.bench_function("historical_six_passes_200", |b| {
        b.iter(|| historical_six_passes(black_box(&ledger)));
    });
    group.finish();
}

/// Retry the previously blocked controller Brier-score reuse with one pinned
/// binary, exact evidence parity, alternating A/B order, and an identity null.
fn bench_speculation_controller_brier_reuse(c: &mut Criterion) {
    if std::env::args().nth(1).is_some_and(|filter| {
        !filter.starts_with('-') && !filter.contains("speculation_controller_brier_reuse")
    }) {
        return;
    }

    fn controller_fixture() -> SpeculationWindowController {
        let mut controller = SpeculationWindowController::new(5_000, 1_000, 30_000, 500);
        let drift = CorrectionDrift {
            wer_approx: 0.05,
            confidence_delta: 0.01,
            segment_count_delta: 0,
            text_edit_distance: 1,
        };
        for index in 0..15_u64 {
            let correction = CorrectionEvent::new(
                index,
                index,
                index,
                "quality".to_owned(),
                vec![],
                100,
                "2026-07-24T00:00:00Z".to_owned(),
                &[],
            );
            controller.observe(&CorrectionDecision::Correct { correction }, &drift);
        }
        for index in 0..10_u64 {
            controller.observe(
                &CorrectionDecision::Confirm {
                    seq: index,
                    drift: drift.clone(),
                },
                &drift,
            );
        }
        controller
    }

    fn measure<const HISTORICAL: bool>(inner_steps: usize) -> Duration {
        let mut controller = controller_fixture();
        controller.set_historical_double_brier_fold(HISTORICAL);
        let started = Instant::now();
        let mut checksum = 0_u64;
        for _ in 0..inner_steps {
            checksum ^= black_box(controller.apply());
        }
        black_box((checksum, controller.evidence().len()));
        started.elapsed()
    }

    fn calibrated_steps<const HISTORICAL: bool>() -> usize {
        let probe_steps = 64;
        let probe_ns = measure::<HISTORICAL>(probe_steps).as_nanos().max(1);
        usize::try_from((HARNESS_TARGET_NS * probe_steps as u128 / probe_ns).clamp(1, 5_000_000))
            .expect("bounded controller steps")
    }

    fn min_ns_per_step<const HISTORICAL: bool>(steps: usize) -> f64 {
        (0..HARNESS_MIN_OF)
            .map(|_| measure::<HISTORICAL>(steps).as_nanos() as f64)
            .fold(f64::INFINITY, f64::min)
            / steps as f64
    }

    let mut historical_oracle = controller_fixture();
    let mut candidate_oracle = controller_fixture();
    historical_oracle.set_historical_double_brier_fold(true);
    let historical_size = historical_oracle.apply();
    let candidate_size = candidate_oracle.apply();
    assert_eq!(candidate_size, historical_size);
    assert_eq!(
        candidate_oracle.is_fallback_active(),
        historical_oracle.is_fallback_active()
    );
    assert_eq!(
        candidate_oracle
            .evidence()
            .last()
            .map(|entry| &entry.action_taken),
        historical_oracle
            .evidence()
            .last()
            .map(|entry| &entry.action_taken)
    );
    assert_eq!(
        serde_json::to_vec(
            candidate_oracle
                .evidence()
                .last()
                .expect("candidate evidence"),
        )
        .expect("serialize candidate evidence"),
        serde_json::to_vec(
            historical_oracle
                .evidence()
                .last()
                .expect("historical evidence"),
        )
        .expect("serialize historical evidence")
    );

    let historical_steps = calibrated_steps::<true>();
    let candidate_steps = calibrated_steps::<false>();
    let mut null_ratios = Vec::with_capacity(HARNESS_PAIRS);
    for pair in 0..HARNESS_PAIRS {
        if pair.is_multiple_of(2) {
            let null_a = min_ns_per_step::<true>(historical_steps);
            let null_b = min_ns_per_step::<true>(historical_steps);
            null_ratios.push(null_a / null_b.max(f64::MIN_POSITIVE));
        } else {
            let null_b = min_ns_per_step::<true>(historical_steps);
            let null_a = min_ns_per_step::<true>(historical_steps);
            null_ratios.push(null_a / null_b.max(f64::MIN_POSITIVE));
        }
    }
    let mut candidate_ratios = Vec::with_capacity(HARNESS_PAIRS);
    for pair in 0..HARNESS_PAIRS {
        if pair.is_multiple_of(2) {
            let baseline = min_ns_per_step::<true>(historical_steps);
            let candidate = min_ns_per_step::<false>(candidate_steps);
            candidate_ratios.push(baseline / candidate.max(f64::MIN_POSITIVE));
        } else {
            let candidate = min_ns_per_step::<false>(candidate_steps);
            let baseline = min_ns_per_step::<true>(historical_steps);
            candidate_ratios.push(baseline / candidate.max(f64::MIN_POSITIVE));
        }
    }

    println!(
        "SPECULATION_CONTROLLER_BRIER_REUSE_AB historical_steps={historical_steps} candidate_steps={candidate_steps} pairs={HARNESS_PAIRS} min_of={HARNESS_MIN_OF}"
    );
    report_harness_gate(
        "SPECULATION_CONTROLLER_BRIER_REUSE",
        &null_ratios,
        &candidate_ratios,
    );

    let mut group = c.benchmark_group("pipeline/speculation_controller_brier_reuse");
    group.bench_function("candidate_apply_window_20", |b| {
        b.iter_batched(
            controller_fixture,
            |mut controller| black_box(controller.apply()),
            BatchSize::SmallInput,
        );
    });
    group.bench_function("historical_apply_window_20", |b| {
        b.iter_batched(
            controller_fixture,
            |mut controller| {
                controller.set_historical_double_brier_fold(true);
                black_box(controller.apply())
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Retry direct transcript concatenation after the historical row lost its
/// null-control tail. Both arms build the exact same bytes; the candidate
/// writes directly into one capacity-calibrated `String`.
fn bench_transcript_concat_resurrection(c: &mut Criterion) {
    if std::env::args()
        .nth(1)
        .is_some_and(|filter| !filter.starts_with('-') && !filter.contains("transcript_concat"))
    {
        return;
    }

    fn historical(segments: &[String]) -> String {
        segments
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn candidate(segments: &[String]) -> String {
        let separator_bytes = segments.len().saturating_sub(1);
        let capacity = segments.iter().fold(separator_bytes, |bytes, segment| {
            bytes.saturating_add(segment.len())
        });
        let mut text = String::with_capacity(capacity);
        for (index, segment) in segments.iter().enumerate() {
            if index != 0 {
                text.push(' ');
            }
            text.push_str(segment);
        }
        text
    }

    let segments: Vec<String> = (0..200)
        .map(|index| match index % 4 {
            0 => format!("segment-{index}"),
            1 => format!("two words {index}"),
            2 => format!("naïve café {index}"),
            _ => String::new(),
        })
        .collect();
    assert_eq!(
        candidate(&segments).as_bytes(),
        historical(&segments).as_bytes()
    );
    for edge in [
        Vec::<String>::new(),
        vec![String::new()],
        vec!["solo".to_owned()],
        vec!["".to_owned(), "".to_owned()],
        vec!["alpha".to_owned(), "".to_owned(), "omega".to_owned()],
    ] {
        assert_eq!(candidate(&edge).as_bytes(), historical(&edge).as_bytes());
    }

    let mut historical_arm = || historical(&segments);
    let historical_repetitions = harness_calibrated_repetitions(&mut historical_arm);
    let mut candidate_arm = || candidate(&segments);
    let candidate_repetitions = harness_calibrated_repetitions(&mut candidate_arm);
    let mut null_left = || historical(&segments);
    let mut null_right = || historical(&segments);
    let null_ratios = harness_paired_ratios(
        &mut null_left,
        &mut null_right,
        historical_repetitions,
        historical_repetitions,
    );
    let candidate_ratios = harness_paired_ratios(
        &mut historical_arm,
        &mut candidate_arm,
        historical_repetitions,
        candidate_repetitions,
    );
    println!(
        "TRANSCRIPT_CONCAT_AB segments={} historical_repetitions={historical_repetitions} candidate_repetitions={candidate_repetitions} pairs={HARNESS_PAIRS} min_of={HARNESS_MIN_OF}",
        segments.len()
    );
    report_harness_gate("TRANSCRIPT_CONCAT", &null_ratios, &candidate_ratios);

    let mut group = c.benchmark_group("pipeline/transcript_concat_resurrection");
    group.bench_function("candidate_200", |b| {
        b.iter(|| candidate(black_box(&segments)));
    });
    group.bench_function("historical_200", |b| {
        b.iter(|| historical(black_box(&segments)));
    });
    group.finish();
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

    for arm in [
        "unbuffered_r1",
        "buffered_r1",
        "unbuffered_r2",
        "buffered_r2",
    ] {
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
    bench_router_diagnostics_profile,
    bench_router_diagnostics_counts_profile,
    bench_speculation_diagnostics_profile,
    bench_speculation_controller_brier_reuse,
    bench_transcript_concat_resurrection,
    bench_export_srt_buffering,
);

fn main() {
    println!("bench_elf_sha256={}", self_identity());
    benches();
    Criterion::default().configure_from_args().final_summary();
}
