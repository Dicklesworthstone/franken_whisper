//! bd-3pf.17: GPU cancellation and stream-ownership telemetry tests.
//!
//! Deterministic tests validating acceleration telemetry fields for:
//! - Stream owner identity tracking
//! - Cancellation fence payload round-trips
//! - Test-local fallback trigger scenarios for future GPU orchestration
//! - AccelerationContext telemetry field persistence
//! - Evidence artifact structure validation
//! - Integration between cancellation tokens and acceleration context

#![forbid(unsafe_code)]

mod helpers;

use franken_whisper::accelerate::{
    AccelRecommendation, AttentionKind, BenchmarkHarness, BenchmarkReport, EmbeddingKind,
    benchmark_result_from_timings, compute_attention, compute_embedding, compute_layer_norm,
    compute_vad_scores,
};
use franken_whisper::model::{
    AccelerationBackend, AccelerationReport, BackendKind, TranscriptionResult, TranscriptionSegment,
};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Test-local domain types (bd-3pf.17)
//
// These types model the GPU cancellation and stream-ownership telemetry
// concepts described in the bead specification.  They are defined here rather
// than in production code because the production types have not yet been
// promoted from the design phase.
// ---------------------------------------------------------------------------

/// Identifies the owner of an acceleration stream (GPU context, CPU thread,
/// or a named external process).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum StreamOwner {
    /// GPU device with a numeric index.
    Gpu { device_index: u32 },
    /// CPU thread pool identified by name.
    Cpu { pool_name: String },
    /// External process (e.g. a subprocess running frankentorch).
    External { pid: u32, label: String },
}

/// Payload carried by a cancellation fence, capturing the reason and context
/// at the moment cancellation was requested.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct FencePayload {
    /// Unique fence identifier.
    fence_id: String,
    /// Why the fence was triggered.
    reason: String,
    /// Which pipeline stage was active when the fence fired.
    active_stage: String,
    /// Monotonic sequence number for ordering.
    seq: u64,
    /// Arbitrary metadata.
    metadata: Value,
}

/// Reasons that trigger a fallback from GPU to CPU acceleration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum FallbackTrigger {
    /// GPU feature flag not enabled at compile time.
    GpuFeatureDisabled,
    /// GPU runtime error (driver, OOM, etc.).
    GpuRuntimeError,
    /// Cancellation fence fired before GPU could complete.
    CancellationFence,
    /// Benchmark indicated CPU is faster.
    BenchmarkPrefersCpu,
    /// Input size below threshold where GPU overhead dominates.
    InputTooSmall,
}

/// A cancellation fence that pairs a trigger reason with a payload.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct CancellationFence {
    trigger: FallbackTrigger,
    payload: FencePayload,
    /// Whether the fence has been acknowledged/consumed.
    acknowledged: bool,
}

/// Aggregated telemetry context for an acceleration pass, combining stream
/// ownership, cancellation fences, fallback triggers, and evidence artifacts.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct AccelerationContext {
    /// Who owns the acceleration stream.
    owner: StreamOwner,
    /// The acceleration report from the production pipeline.
    report: AccelerationReport,
    /// Cancellation fences that fired during this context's lifetime.
    fences: Vec<CancellationFence>,
    /// Fallback triggers that were active.
    active_triggers: Vec<FallbackTrigger>,
    /// Evidence artifacts (serialized JSON values).
    evidence: Vec<Value>,
    /// Recommendation from the benchmark harness.
    recommendation: AccelRecommendation,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_segment(text: &str, confidence: Option<f64>) -> TranscriptionSegment {
    TranscriptionSegment {
        start_sec: Some(0.0),
        end_sec: Some(1.0),
        text: text.to_owned(),
        speaker: None,
        confidence,
    }
}

fn make_result(segments: Vec<TranscriptionSegment>) -> TranscriptionResult {
    TranscriptionResult {
        backend: BackendKind::WhisperCpp,
        transcript: "test".to_owned(),
        language: Some("en".to_owned()),
        segments,
        acceleration: None,
        diarization: None,
        raw_output: json!({}),
        artifact_paths: vec![],
    }
}

fn make_fence_payload(fence_id: &str, reason: &str, stage: &str, seq: u64) -> FencePayload {
    FencePayload {
        fence_id: fence_id.to_owned(),
        reason: reason.to_owned(),
        active_stage: stage.to_owned(),
        seq,
        metadata: json!({}),
    }
}

fn make_cancellation_fence(
    trigger: FallbackTrigger,
    fence_id: &str,
    stage: &str,
    seq: u64,
) -> CancellationFence {
    CancellationFence {
        trigger,
        payload: make_fence_payload(fence_id, &format!("{trigger:?}"), stage, seq),
        acknowledged: false,
    }
}

/// Build an AccelerationContext from a production AccelerationReport.
fn build_context(
    owner: StreamOwner,
    report: AccelerationReport,
    recommendation: AccelRecommendation,
) -> AccelerationContext {
    AccelerationContext {
        owner,
        report,
        fences: Vec::new(),
        active_triggers: Vec::new(),
        evidence: Vec::new(),
        recommendation,
    }
}

// ===========================================================================
// 1. Stream owner identity tracking and telemetry
// ===========================================================================

#[test]
fn stream_owner_gpu_identity_is_deterministic() {
    let owner = StreamOwner::Gpu { device_index: 0 };
    let json_str = serde_json::to_string(&owner).expect("serialize StreamOwner::Gpu");
    let roundtripped: StreamOwner =
        serde_json::from_str(&json_str).expect("deserialize StreamOwner::Gpu");
    assert_eq!(owner, roundtripped);
    assert!(json_str.contains("Gpu"));
    assert!(json_str.contains("device_index"));
}

#[test]
fn stream_owner_cpu_identity_is_deterministic() {
    let owner = StreamOwner::Cpu {
        pool_name: "accel-workers".to_owned(),
    };
    let json_str = serde_json::to_string(&owner).expect("serialize StreamOwner::Cpu");
    let roundtripped: StreamOwner =
        serde_json::from_str(&json_str).expect("deserialize StreamOwner::Cpu");
    assert_eq!(owner, roundtripped);
    assert!(json_str.contains("accel-workers"));
}

#[test]
fn stream_owner_external_identity_roundtrip() {
    let owner = StreamOwner::External {
        pid: 42,
        label: "frankentorch-subprocess".to_owned(),
    };
    let json_str = serde_json::to_string(&owner).expect("serialize StreamOwner::External");
    let roundtripped: StreamOwner =
        serde_json::from_str(&json_str).expect("deserialize StreamOwner::External");
    assert_eq!(owner, roundtripped);
}

#[test]
fn stream_owner_equality_distinguishes_variants() {
    let gpu0 = StreamOwner::Gpu { device_index: 0 };
    let gpu1 = StreamOwner::Gpu { device_index: 1 };
    let cpu = StreamOwner::Cpu {
        pool_name: "default".to_owned(),
    };

    assert_ne!(gpu0, gpu1, "different GPU indices should not be equal");
    assert_ne!(gpu0, cpu, "GPU and CPU owners should not be equal");
}

#[test]
fn stream_owner_in_context_persists_through_report() {
    let mut result = make_result(vec![
        make_segment("hello", Some(0.6)),
        make_segment("world", Some(0.4)),
    ]);
    let report = franken_whisper::accelerate::apply(&mut result);

    let ctx = build_context(
        StreamOwner::Cpu {
            pool_name: "test-pool".to_owned(),
        },
        report,
        AccelRecommendation::UseCpu,
    );

    // Verify the context preserves stream owner through serialization.
    let json_str = serde_json::to_string(&ctx).expect("serialize context");
    let roundtripped: AccelerationContext =
        serde_json::from_str(&json_str).expect("deserialize context");

    assert_eq!(
        roundtripped.owner,
        StreamOwner::Cpu {
            pool_name: "test-pool".to_owned()
        }
    );
    assert!(roundtripped.report.normalized_confidences);
}

// ===========================================================================
// 2. Cancellation fence payload round-trips
// ===========================================================================

#[test]
fn fence_payload_serde_roundtrip() {
    let payload = FencePayload {
        fence_id: "fence-001".to_owned(),
        reason: "deadline exceeded".to_owned(),
        active_stage: "acceleration".to_owned(),
        seq: 42,
        metadata: json!({"budget_ms": 5000, "elapsed_ms": 5100}),
    };

    let json_str = serde_json::to_string(&payload).expect("serialize");
    let roundtripped: FencePayload = serde_json::from_str(&json_str).expect("deserialize");

    assert_eq!(payload, roundtripped);
    assert_eq!(roundtripped.fence_id, "fence-001");
    assert_eq!(roundtripped.seq, 42);
    assert_eq!(roundtripped.metadata["budget_ms"], 5000);
}

#[test]
fn fence_payload_metadata_preserves_nested_structure() {
    let payload = FencePayload {
        fence_id: "fence-nested".to_owned(),
        reason: "test".to_owned(),
        active_stage: "backend".to_owned(),
        seq: 1,
        metadata: json!({
            "context": {
                "trace_id": "abc123",
                "evidence_count": 3,
                "stages_completed": ["ingest", "normalize", "backend"]
            }
        }),
    };

    let json_str = serde_json::to_string(&payload).expect("serialize");
    let roundtripped: FencePayload = serde_json::from_str(&json_str).expect("deserialize");

    let ctx = &roundtripped.metadata["context"];
    assert_eq!(ctx["trace_id"], "abc123");
    assert_eq!(ctx["evidence_count"], 3);
    assert_eq!(ctx["stages_completed"].as_array().unwrap().len(), 3);
}

#[test]
fn cancellation_fence_roundtrip_preserves_trigger_and_acknowledged() {
    let fence = CancellationFence {
        trigger: FallbackTrigger::CancellationFence,
        payload: make_fence_payload("f-1", "deadline", "accelerate", 10),
        acknowledged: false,
    };

    let json_str = serde_json::to_string(&fence).expect("serialize");
    let roundtripped: CancellationFence = serde_json::from_str(&json_str).expect("deserialize");

    assert_eq!(roundtripped.trigger, FallbackTrigger::CancellationFence);
    assert!(!roundtripped.acknowledged);
    assert_eq!(roundtripped.payload.fence_id, "f-1");
}

#[test]
fn cancellation_fence_acknowledged_state_roundtrips() {
    let mut fence =
        make_cancellation_fence(FallbackTrigger::GpuRuntimeError, "f-ack", "backend", 5);
    fence.acknowledged = true;

    let json_str = serde_json::to_string(&fence).expect("serialize");
    let roundtripped: CancellationFence = serde_json::from_str(&json_str).expect("deserialize");
    assert!(roundtripped.acknowledged);
}

#[test]
fn fence_payload_seq_ordering_is_monotonic() {
    let payloads: Vec<FencePayload> = (0..5)
        .map(|i| make_fence_payload(&format!("f-{i}"), "test", "backend", i))
        .collect();

    for window in payloads.windows(2) {
        assert!(
            window[1].seq > window[0].seq,
            "fence payload seq must be monotonically increasing: {} vs {}",
            window[0].seq,
            window[1].seq
        );
    }
}

// ===========================================================================
// 3. Test-local fallback trigger scenarios for future GPU orchestration
// ===========================================================================

#[test]
fn production_normalization_uses_cpu_backend_identity() {
    let mut result = make_result(vec![
        make_segment("alpha", Some(0.5)),
        make_segment("beta", Some(0.3)),
    ]);

    let report = franken_whisper::accelerate::apply(&mut result);

    assert_eq!(
        report.backend,
        AccelerationBackend::None,
        "CPU confidence normalization should not claim a GPU backend"
    );
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.contains("CPU") || n.contains("cpu")),
        "notes should identify CPU normalization: {:?}",
        report.notes
    );
}

#[test]
fn fallback_trigger_serde_roundtrip_all_variants() {
    let triggers = vec![
        FallbackTrigger::GpuFeatureDisabled,
        FallbackTrigger::GpuRuntimeError,
        FallbackTrigger::CancellationFence,
        FallbackTrigger::BenchmarkPrefersCpu,
        FallbackTrigger::InputTooSmall,
    ];

    for trigger in &triggers {
        let json_str = serde_json::to_string(trigger).expect("serialize trigger");
        let roundtripped: FallbackTrigger =
            serde_json::from_str(&json_str).expect("deserialize trigger");
        assert_eq!(
            *trigger, roundtripped,
            "trigger {trigger:?} failed roundtrip"
        );
    }
}

#[test]
fn fallback_trigger_benchmark_prefers_cpu_when_no_gpu() {
    let result = benchmark_result_from_timings("softmax".to_owned(), 100, None);
    assert_eq!(
        result.recommendation,
        AccelRecommendation::UseCpu,
        "with no GPU timing, recommendation should be UseCpu"
    );
    assert!(result.gpu_time_us.is_none());
}

#[test]
fn fallback_trigger_benchmark_prefers_gpu_when_faster() {
    let result = benchmark_result_from_timings("softmax".to_owned(), 1000, Some(100));
    assert_eq!(
        result.recommendation,
        AccelRecommendation::UseGpu,
        "GPU 10x faster should recommend UseGpu"
    );
    assert!((result.speedup_ratio - 10.0).abs() < f64::EPSILON);
}

#[test]
fn fallback_trigger_either_fine_within_threshold() {
    // 5% difference: within the 10% threshold for EitherFine.
    let result = benchmark_result_from_timings("attention".to_owned(), 1000, Some(950));
    assert_eq!(
        result.recommendation,
        AccelRecommendation::EitherFine,
        "5% speedup should be EitherFine"
    );
}

#[test]
fn fallback_context_records_multiple_triggers() {
    let mut result = make_result(vec![make_segment("test", Some(0.5))]);
    let report = franken_whisper::accelerate::apply(&mut result);

    let mut ctx = build_context(
        StreamOwner::Cpu {
            pool_name: "fallback".to_owned(),
        },
        report,
        AccelRecommendation::UseCpu,
    );

    // Record multiple fallback triggers.
    ctx.active_triggers
        .push(FallbackTrigger::GpuFeatureDisabled);
    ctx.active_triggers.push(FallbackTrigger::InputTooSmall);

    assert_eq!(ctx.active_triggers.len(), 2);
    assert!(
        ctx.active_triggers
            .contains(&FallbackTrigger::GpuFeatureDisabled)
    );
    assert!(
        ctx.active_triggers
            .contains(&FallbackTrigger::InputTooSmall)
    );

    // Round-trip the context.
    let json_str = serde_json::to_string(&ctx).expect("serialize");
    let roundtripped: AccelerationContext = serde_json::from_str(&json_str).expect("deserialize");
    assert_eq!(roundtripped.active_triggers.len(), 2);
}

#[test]
fn compute_attention_uses_packaged_cpu_implementation() {
    let query = vec![1.0, 2.0, 3.0, 4.0];
    let key = vec![0.5, 1.5, 2.5, 3.5];

    let result = compute_attention(&query, &key, AttentionKind::SelfAttention);
    assert!(
        !result.gpu_accelerated,
        "the packaged helper should use CPU"
    );
    assert_eq!(result.kind, AttentionKind::SelfAttention);
    assert_eq!(result.scores.len(), 4);

    // Scores should be a valid probability distribution (softmax output).
    let sum: f64 = result.scores.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "attention scores should sum to 1.0, got {sum}"
    );
}

#[test]
fn compute_vad_uses_packaged_cpu_implementation() {
    let energy = vec![0.0, 1.0, -1.0, 2.0, -2.0];
    let result = compute_vad_scores(&energy);
    assert!(!result.gpu_accelerated);
    assert_eq!(result.frame_scores.len(), 5);
    for score in &result.frame_scores {
        assert!(
            (0.0..=1.0).contains(score),
            "VAD score should be in [0, 1], got {score}"
        );
    }
}

#[test]
fn compute_layer_norm_uses_packaged_cpu_implementation() {
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let result = compute_layer_norm(&values, &gamma, &beta, 1e-5);
    assert!(!result.gpu_accelerated);
    assert_eq!(result.normalized.len(), 4);

    // Layer-normed values should have mean ~0.
    let mean: f64 = result.normalized.iter().sum::<f64>() / result.normalized.len() as f64;
    assert!(
        mean.abs() < 1e-6,
        "layer norm output mean should be ~0.0, got {mean}"
    );
}

#[test]
fn compute_embedding_uses_packaged_cpu_implementation() {
    let table = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let result = compute_embedding(&[0, 2, 1], &table, EmbeddingKind::Token);
    assert!(!result.gpu_accelerated);
    assert_eq!(result.kind, EmbeddingKind::Token);
    assert_eq!(result.embeddings.len(), 3);
    assert_eq!(result.embeddings[0], vec![1.0, 0.0, 0.0]);
    assert_eq!(result.embeddings[1], vec![0.0, 0.0, 1.0]);
    assert_eq!(result.embeddings[2], vec![0.0, 1.0, 0.0]);
}

// ===========================================================================
// 4. AccelerationContext telemetry field persistence
// ===========================================================================

#[test]
fn acceleration_context_full_roundtrip() {
    let mut result = make_result(vec![
        make_segment("hello", Some(0.7)),
        make_segment("world", Some(0.3)),
    ]);
    let report = franken_whisper::accelerate::apply(&mut result);

    let ctx = AccelerationContext {
        owner: StreamOwner::Gpu { device_index: 0 },
        report: report.clone(),
        fences: vec![make_cancellation_fence(
            FallbackTrigger::GpuRuntimeError,
            "fence-ctx-1",
            "accelerate",
            1,
        )],
        active_triggers: vec![FallbackTrigger::GpuRuntimeError],
        evidence: vec![json!({"type": "benchmark", "cpu_time_us": 120})],
        recommendation: AccelRecommendation::UseCpu,
    };

    let json_str = serde_json::to_string(&ctx).expect("serialize");
    let roundtripped: AccelerationContext = serde_json::from_str(&json_str).expect("deserialize");

    assert_eq!(roundtripped.owner, ctx.owner);
    assert_eq!(roundtripped.fences.len(), 1);
    assert_eq!(roundtripped.active_triggers.len(), 1);
    assert_eq!(roundtripped.evidence.len(), 1);
    assert_eq!(roundtripped.recommendation, AccelRecommendation::UseCpu);
    assert_eq!(roundtripped.report.backend, AccelerationBackend::None);
    assert!(roundtripped.report.normalized_confidences);
}

#[test]
fn acceleration_context_preserves_empty_fences() {
    let mut result = make_result(vec![make_segment("solo", Some(1.0))]);
    let report = franken_whisper::accelerate::apply(&mut result);

    let ctx = build_context(
        StreamOwner::Cpu {
            pool_name: "main".to_owned(),
        },
        report,
        AccelRecommendation::UseCpu,
    );

    let json_str = serde_json::to_string(&ctx).expect("serialize");
    let roundtripped: AccelerationContext = serde_json::from_str(&json_str).expect("deserialize");

    assert!(roundtripped.fences.is_empty());
    assert!(roundtripped.active_triggers.is_empty());
    assert!(roundtripped.evidence.is_empty());
}

#[test]
fn acceleration_context_report_fields_match_production_apply() {
    let mut result = make_result(vec![
        make_segment("a", Some(0.5)),
        make_segment("b", Some(0.3)),
        make_segment("c", Some(0.2)),
    ]);
    let report = franken_whisper::accelerate::apply(&mut result);

    let ctx = build_context(
        StreamOwner::Cpu {
            pool_name: "test".to_owned(),
        },
        report.clone(),
        AccelRecommendation::UseCpu,
    );

    // Validate report fields match what apply() produces.
    assert_eq!(ctx.report.input_values, 3);
    assert!(ctx.report.normalized_confidences);
    assert!(ctx.report.pre_mass.is_some());
    assert!(ctx.report.post_mass.is_some());

    let post_mass = ctx.report.post_mass.unwrap();
    assert!(
        (post_mass - 1.0).abs() < 1e-6,
        "post_mass should be ~1.0, got {post_mass}"
    );
}

#[test]
fn acceleration_context_pre_mass_equals_input_sum() {
    let mut result = make_result(vec![
        make_segment("x", Some(0.4)),
        make_segment("y", Some(0.6)),
    ]);
    let report = franken_whisper::accelerate::apply(&mut result);

    assert!(report.pre_mass.is_some());
    let pre = report.pre_mass.unwrap();
    assert!(
        (pre - 1.0).abs() < 1e-6,
        "pre_mass for [0.4, 0.6] should be 1.0, got {pre}"
    );
}

#[test]
fn acceleration_context_no_segments_produces_zero_input_values() {
    let mut result = make_result(vec![]);
    let report = franken_whisper::accelerate::apply(&mut result);

    assert_eq!(report.input_values, 0);
    assert!(!report.normalized_confidences);
    assert_eq!(report.backend, AccelerationBackend::None);
}

// ===========================================================================
// 5. Evidence artifact structure validation
// ===========================================================================

#[test]
fn evidence_artifact_json_structure_is_valid() {
    let evidence = json!({
        "type": "cancellation_fence",
        "fence_id": "ev-001",
        "trigger": "GpuRuntimeError",
        "stage": "accelerate",
        "timestamp_ms": 1700000000000_u64,
        "details": {
            "error_message": "GPU memory exhausted",
            "gpu_memory_used_mb": 8192,
            "fallback": "cpu"
        }
    });

    // Validate structure.
    assert!(evidence.is_object());
    let obj = evidence.as_object().unwrap();
    assert!(obj.contains_key("type"));
    assert!(obj.contains_key("fence_id"));
    assert!(obj.contains_key("trigger"));
    assert!(obj.contains_key("stage"));
    assert!(obj.contains_key("timestamp_ms"));
    assert!(obj.contains_key("details"));

    // Nested details must also be an object.
    assert!(obj["details"].is_object());
    assert_eq!(obj["details"]["fallback"], "cpu");
}

#[test]
fn evidence_ledger_builder_produces_valid_acceleration_entry() {
    use franken_evidence::EvidenceLedgerBuilder;

    let entry = EvidenceLedgerBuilder::new()
        .ts_unix_ms(1_700_000_000_000)
        .component("accelerate")
        .action("fallback_to_cpu")
        .posterior(vec![0.9, 0.1]) // 90% confidence CPU is better
        .expected_loss("use_gpu", 0.8)
        .expected_loss("use_cpu", 0.1)
        .expected_loss("fallback_to_cpu", 0.1)
        .chosen_expected_loss(0.1)
        .calibration_score(0.85)
        .fallback_active(true)
        .top_feature("gpu_unavailable", 0.95)
        .top_feature("input_size", 0.05)
        .build()
        .expect("evidence entry should be valid");

    assert!(entry.is_valid());
    assert_eq!(entry.component, "accelerate");
    assert_eq!(entry.action, "fallback_to_cpu");
    assert!(entry.fallback_active);
    assert_eq!(entry.posterior, vec![0.9, 0.1]);
}

#[test]
fn evidence_ledger_roundtrips_through_json() {
    use franken_evidence::EvidenceLedgerBuilder;

    let entry = EvidenceLedgerBuilder::new()
        .ts_unix_ms(1_700_000_000_000)
        .component("gpu_cancellation")
        .action("cancel_and_fallback")
        .posterior(vec![1.0])
        .expected_loss("cancel", 0.0)
        .expected_loss("cancel_and_fallback", 0.0)
        .chosen_expected_loss(0.0)
        .calibration_score(1.0)
        .fallback_active(true)
        .build()
        .expect("valid");

    let json_str = serde_json::to_string(&entry).expect("serialize");
    let roundtripped: franken_evidence::EvidenceLedger =
        serde_json::from_str(&json_str).expect("deserialize");

    assert_eq!(entry.ts_unix_ms, roundtripped.ts_unix_ms);
    assert_eq!(entry.component, roundtripped.component);
    assert_eq!(entry.action, roundtripped.action);
    assert_eq!(entry.fallback_active, roundtripped.fallback_active);
}

#[test]
fn evidence_artifacts_accumulate_in_context() {
    let mut result = make_result(vec![make_segment("a", Some(0.5))]);
    let report = franken_whisper::accelerate::apply(&mut result);

    let mut ctx = build_context(
        StreamOwner::Cpu {
            pool_name: "evidence-test".to_owned(),
        },
        report,
        AccelRecommendation::UseCpu,
    );

    // Accumulate multiple evidence entries.
    ctx.evidence.push(json!({"step": 1, "action": "init"}));
    ctx.evidence
        .push(json!({"step": 2, "action": "gpu_check_failed"}));
    ctx.evidence
        .push(json!({"step": 3, "action": "cpu_fallback"}));

    assert_eq!(ctx.evidence.len(), 3);

    // Verify ordering is preserved through serialization.
    let json_str = serde_json::to_string(&ctx).expect("serialize");
    let roundtripped: AccelerationContext = serde_json::from_str(&json_str).expect("deserialize");

    assert_eq!(roundtripped.evidence.len(), 3);
    assert_eq!(roundtripped.evidence[0]["step"], 1);
    assert_eq!(roundtripped.evidence[1]["step"], 2);
    assert_eq!(roundtripped.evidence[2]["step"], 3);
}

#[test]
fn evidence_artifact_preserves_benchmark_report() {
    let mut harness = BenchmarkHarness::new();
    harness.benchmark_softmax(64);
    harness.benchmark_layer_norm(64);
    let bench_report = harness.report();

    let evidence_value = serde_json::to_value(&bench_report).expect("serialize report to Value");

    // Validate the evidence artifact has expected structure.
    assert!(evidence_value.is_object());
    assert!(evidence_value.get("results").is_some());
    assert!(evidence_value.get("overall").is_some());
    assert!(evidence_value.get("notes").is_some());

    let results = evidence_value["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);

    // Round-trip back to BenchmarkReport.
    let roundtripped: BenchmarkReport =
        serde_json::from_value(evidence_value).expect("deserialize from Value");
    assert_eq!(roundtripped.results.len(), 2);
    assert_eq!(roundtripped.overall, AccelRecommendation::UseCpu);
}

// ===========================================================================
// 6. Integration between cancellation tokens and acceleration context
// ===========================================================================

#[test]
fn cancellation_fence_integrates_with_acceleration_report() {
    let mut result = make_result(vec![
        make_segment("cancelled", Some(0.5)),
        make_segment("test", Some(0.5)),
    ]);

    // Run acceleration normally to get a report.
    let report = franken_whisper::accelerate::apply(&mut result);

    // Build a context that simulates cancellation during GPU acceleration.
    let fence = CancellationFence {
        trigger: FallbackTrigger::CancellationFence,
        payload: FencePayload {
            fence_id: "integration-fence-1".to_owned(),
            reason: "pipeline deadline exceeded".to_owned(),
            active_stage: "accelerate".to_owned(),
            seq: 1,
            metadata: json!({
                "budget_ms": 5000,
                "elapsed_ms": 5100,
                "overdue_ms": 100,
            }),
        },
        acknowledged: false,
    };

    let mut ctx = AccelerationContext {
        owner: StreamOwner::Gpu { device_index: 0 },
        report,
        fences: vec![fence],
        active_triggers: vec![FallbackTrigger::CancellationFence],
        evidence: vec![json!({
            "type": "cancellation",
            "reason": "deadline exceeded during GPU acceleration",
        })],
        recommendation: AccelRecommendation::UseCpu,
    };

    // Acknowledge the fence.
    ctx.fences[0].acknowledged = true;

    // Full round-trip validation.
    let json_str = serde_json::to_string(&ctx).expect("serialize");
    let roundtripped: AccelerationContext = serde_json::from_str(&json_str).expect("deserialize");

    assert!(roundtripped.fences[0].acknowledged);
    assert_eq!(
        roundtripped.fences[0].trigger,
        FallbackTrigger::CancellationFence
    );
    assert_eq!(roundtripped.fences[0].payload.metadata["overdue_ms"], 100);
    assert_eq!(roundtripped.evidence.len(), 1);
    assert_eq!(roundtripped.evidence[0]["type"], "cancellation");
}

#[test]
fn multiple_fences_in_context_maintain_ordering() {
    let mut result = make_result(vec![make_segment("multi", Some(1.0))]);
    let report = franken_whisper::accelerate::apply(&mut result);

    let fences: Vec<CancellationFence> = vec![
        make_cancellation_fence(FallbackTrigger::GpuRuntimeError, "f-1", "backend", 1),
        make_cancellation_fence(FallbackTrigger::CancellationFence, "f-2", "accelerate", 2),
        make_cancellation_fence(FallbackTrigger::InputTooSmall, "f-3", "accelerate", 3),
    ];

    let ctx = AccelerationContext {
        owner: StreamOwner::Cpu {
            pool_name: "multi-fence".to_owned(),
        },
        report,
        fences: fences.clone(),
        active_triggers: vec![
            FallbackTrigger::GpuRuntimeError,
            FallbackTrigger::CancellationFence,
            FallbackTrigger::InputTooSmall,
        ],
        evidence: vec![],
        recommendation: AccelRecommendation::UseCpu,
    };

    let json_str = serde_json::to_string(&ctx).expect("serialize");
    let roundtripped: AccelerationContext = serde_json::from_str(&json_str).expect("deserialize");

    assert_eq!(roundtripped.fences.len(), 3);

    // Verify ordering by sequence number.
    for (i, fence) in roundtripped.fences.iter().enumerate() {
        assert_eq!(fence.payload.seq, (i + 1) as u64);
    }

    // Verify trigger types match.
    assert_eq!(
        roundtripped.fences[0].trigger,
        FallbackTrigger::GpuRuntimeError
    );
    assert_eq!(
        roundtripped.fences[1].trigger,
        FallbackTrigger::CancellationFence
    );
    assert_eq!(
        roundtripped.fences[2].trigger,
        FallbackTrigger::InputTooSmall
    );
}

#[test]
fn acceleration_report_notes_capture_cancellation_reason() {
    // When acceleration is applied with no segments, the notes should explain why.
    let mut result = make_result(vec![]);
    let report = franken_whisper::accelerate::apply(&mut result);

    assert!(
        report.notes.iter().any(|n| n.contains("no segments")),
        "notes should explain no-segment short-circuit: {:?}",
        report.notes
    );
    assert_eq!(report.backend, AccelerationBackend::None);
    assert_eq!(report.input_values, 0);
}

#[test]
fn benchmark_harness_recommendation_informs_acceleration_context() {
    let mut harness = BenchmarkHarness::new();
    let _ = harness.run_all();
    let recommendation = harness.recommend_backend();
    let bench_report = harness.report();

    // Without GPU, recommendation should always be UseCpu.
    assert_eq!(recommendation, AccelRecommendation::UseCpu);

    // Build a context that uses the recommendation.
    let mut result = make_result(vec![make_segment("bench", Some(0.5))]);
    let accel_report = franken_whisper::accelerate::apply(&mut result);

    let ctx = AccelerationContext {
        owner: StreamOwner::Cpu {
            pool_name: "benchmark-informed".to_owned(),
        },
        report: accel_report,
        fences: vec![],
        active_triggers: if recommendation == AccelRecommendation::UseCpu {
            vec![FallbackTrigger::BenchmarkPrefersCpu]
        } else {
            vec![]
        },
        evidence: vec![serde_json::to_value(&bench_report).expect("serialize bench report")],
        recommendation,
    };

    assert_eq!(ctx.recommendation, AccelRecommendation::UseCpu);
    assert!(
        ctx.active_triggers
            .contains(&FallbackTrigger::BenchmarkPrefersCpu)
    );

    // Evidence should contain the full benchmark report.
    let evidence_report: BenchmarkReport =
        serde_json::from_value(ctx.evidence[0].clone()).expect("deserialize");
    assert_eq!(evidence_report.results.len(), 8); // run_all produces 8 results
}

#[test]
fn accel_recommendation_display_is_deterministic() {
    assert_eq!(AccelRecommendation::UseCpu.to_string(), "use_cpu");
    assert_eq!(AccelRecommendation::UseGpu.to_string(), "use_gpu");
    assert_eq!(AccelRecommendation::EitherFine.to_string(), "either_fine");
}

#[test]
fn acceleration_context_with_all_fields_populated() {
    use franken_evidence::EvidenceLedgerBuilder;

    let mut result = make_result(vec![
        make_segment("comprehensive", Some(0.4)),
        make_segment("test", Some(0.3)),
        make_segment("case", Some(0.3)),
    ]);
    let report = franken_whisper::accelerate::apply(&mut result);

    let evidence_entry = EvidenceLedgerBuilder::new()
        .ts_unix_ms(1_700_000_000_000)
        .component("accelerate")
        .action("fallback_to_cpu")
        .posterior(vec![0.85, 0.15])
        .expected_loss("use_gpu", 0.7)
        .expected_loss("use_cpu", 0.05)
        .expected_loss("fallback_to_cpu", 0.05)
        .chosen_expected_loss(0.05)
        .calibration_score(0.9)
        .fallback_active(true)
        .top_feature("gpu_unavailable", 1.0)
        .build()
        .expect("valid");

    let ctx = AccelerationContext {
        owner: StreamOwner::Gpu { device_index: 0 },
        report,
        fences: vec![make_cancellation_fence(
            FallbackTrigger::GpuFeatureDisabled,
            "f-full-1",
            "accelerate",
            1,
        )],
        active_triggers: vec![FallbackTrigger::GpuFeatureDisabled],
        evidence: vec![
            serde_json::to_value(&evidence_entry).expect("serialize evidence"),
            json!({"benchmark_cpu_time_us": 150}),
        ],
        recommendation: AccelRecommendation::UseCpu,
    };

    // Validate all fields survive full serialization round-trip.
    let json_str = serde_json::to_string_pretty(&ctx).expect("serialize");
    let roundtripped: AccelerationContext = serde_json::from_str(&json_str).expect("deserialize");

    assert_eq!(roundtripped.owner, StreamOwner::Gpu { device_index: 0 });
    assert_eq!(roundtripped.report.input_values, 3);
    assert!(roundtripped.report.normalized_confidences);
    assert_eq!(roundtripped.fences.len(), 1);
    assert_eq!(roundtripped.active_triggers.len(), 1);
    assert_eq!(roundtripped.evidence.len(), 2);
    assert_eq!(roundtripped.recommendation, AccelRecommendation::UseCpu);

    // Validate the evidence entry round-trips correctly.
    let recovered_evidence: franken_evidence::EvidenceLedger =
        serde_json::from_value(roundtripped.evidence[0].clone()).expect("deserialize evidence");
    assert_eq!(recovered_evidence.component, "accelerate");
    assert!(recovered_evidence.fallback_active);
}

// ===========================================================================
// 7. Cross-attention and self-attention fallback telemetry
// ===========================================================================

#[test]
fn cross_attention_fallback_to_cpu_is_deterministic() {
    let query = vec![0.1, 0.2, 0.3];
    let key = vec![0.3, 0.2, 0.1];

    let result_self = compute_attention(&query, &key, AttentionKind::SelfAttention);
    let result_cross = compute_attention(&query, &key, AttentionKind::CrossAttention);

    // Both should use CPU without GPU feature.
    assert!(!result_self.gpu_accelerated);
    assert!(!result_cross.gpu_accelerated);

    // Same inputs should produce same scores regardless of kind.
    assert_eq!(result_self.scores, result_cross.scores);

    // But the kind field should differ.
    assert_eq!(result_self.kind, AttentionKind::SelfAttention);
    assert_eq!(result_cross.kind, AttentionKind::CrossAttention);
}

#[test]
fn empty_input_attention_produces_empty_scores() {
    let result = compute_attention(&[], &[], AttentionKind::SelfAttention);
    assert!(result.scores.is_empty());
    assert!(!result.gpu_accelerated);
}

#[test]
fn empty_input_vad_produces_zero_activity() {
    let result = compute_vad_scores(&[]);
    assert!(result.frame_scores.is_empty());
    assert_eq!(result.activity_ratio, 0.0);
    assert!(!result.gpu_accelerated);
}

#[test]
fn empty_input_layer_norm_produces_empty_output() {
    let result = compute_layer_norm(&[], &[], &[], 1e-5);
    assert!(result.normalized.is_empty());
    assert!(!result.gpu_accelerated);
}

// ===========================================================================
// 8. Determinism validation (run same computation twice, compare)
// ===========================================================================

#[test]
fn acceleration_apply_is_deterministic_across_runs() {
    let segments = vec![
        make_segment("determinism", Some(0.4)),
        make_segment("test", Some(0.3)),
        make_segment("run", Some(0.3)),
    ];

    let mut result1 = make_result(segments.clone());
    let report1 = franken_whisper::accelerate::apply(&mut result1);

    let mut result2 = make_result(segments);
    let report2 = franken_whisper::accelerate::apply(&mut result2);

    // Reports should be identical.
    assert_eq!(report1.backend, report2.backend);
    assert_eq!(report1.input_values, report2.input_values);
    assert_eq!(
        report1.normalized_confidences,
        report2.normalized_confidences
    );
    assert_eq!(report1.pre_mass, report2.pre_mass);
    assert_eq!(report1.post_mass, report2.post_mass);
    assert_eq!(report1.notes, report2.notes);

    // Confidence values should be identical.
    for (s1, s2) in result1.segments.iter().zip(result2.segments.iter()) {
        assert_eq!(
            s1.confidence, s2.confidence,
            "confidence values should be identical across runs"
        );
    }
}

#[test]
fn benchmark_result_from_timings_is_deterministic() {
    let r1 = benchmark_result_from_timings("op".to_owned(), 500, Some(250));
    let r2 = benchmark_result_from_timings("op".to_owned(), 500, Some(250));

    assert_eq!(r1.recommendation, r2.recommendation);
    assert_eq!(r1.speedup_ratio, r2.speedup_ratio);
    assert_eq!(r1.cpu_time_us, r2.cpu_time_us);
    assert_eq!(r1.gpu_time_us, r2.gpu_time_us);
}
