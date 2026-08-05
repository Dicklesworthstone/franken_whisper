//! Bounded safe-Rust forward inference for the frozen ECAPA-TDNN speaker model.
//!
//! This module consumes only the exact authenticated safetensors package from
//! [`crate::ecapa_conformance`]. It implements the pinned SpeechBrain forward
//! semantics with explicit reflection padding, checked allocation planning,
//! bounded CPU matmul chunks, cooperative cancellation, and content-redacted
//! diagnostics. It does not segment audio, assign speaker identities, or own
//! any clustering/output policy.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::ecapa_conformance::{
    ECAPA_BATCH_NORM_EPSILON, ECAPA_CONTRACT_SHA256, ECAPA_EMBEDDING_DIMENSIONS,
    ECAPA_EXPORTED_TENSOR_COUNT, ECAPA_MAXIMUM_INFERENCE_FRAMES, ECAPA_MEL_BANDS,
    ECAPA_MINIMUM_INFERENCE_FRAMES, ECAPA_PACKAGE_SHA256, ECAPA_POOLING_VARIANCE_FLOOR,
    EcapaFrontendOutput, ecapa_load_checkpoint, expected_ecapa_tensors,
    load_verified_ecapa_weight_package_with_checkpoint, normalize_ecapa_embedding,
};
use crate::error::{FwError, FwResult};
use crate::native_engine::Mat;
use crate::native_engine::nn;
use crate::native_engine::weights::SafetensorsFile;
use crate::orchestrator::CancellationToken;

pub const ECAPA_INFERENCE_SCHEMA: &str = "franken-whisper-ecapa-inference-v3";
pub use crate::ecapa_conformance::ECAPA_MAXIMUM_ABSOLUTE_INPUT_FEATURE;
pub const ECAPA_SCRATCH_PROOF_REVIEWED_FRANKENTORCH_REVISION: &str =
    "523aaf827faf538aa541126ee222fcd7af348410";
/// Upper reserve for packed f32 panels created below the CPU-kernel boundary.
///
/// With the pinned 32-row ECAPA bands, 3,072-column maximum, FrankenTorch's
/// 128-column minimum tiles, and matrixmultiply 0.3's 256-deep panels, at most
/// 24 concurrent combined A+B packing allocations are possible. The exact
/// reviewed maximum is 3,956,736 bytes at 22 allocations. Eight MiB leaves
/// more than 2x headroom for alignment and kernel scheduling while remaining
/// inside a practical process budget.
pub const ECAPA_CONSERVATIVE_KERNEL_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;
pub const ECAPA_DEFAULT_MAXIMUM_PEAK_BUFFER_BYTES: u64 = 20 * 1024 * 1024;
pub const ECAPA_MODEL_RESIDENT_BYTES: u64 = 83_070_208;
/// Logical bytes retained or materialized together at the typed load boundary.
///
/// This accounts for the authenticated package, resident model `f32` payload,
/// and largest decoded source tensor. It is not allocator size or process RSS.
pub const ECAPA_CONSERVATIVE_LOAD_ACCOUNTED_PAYLOAD_BYTES: u64 = 204_065_488;

const MATMUL_ROW_CHUNK_FRAMES: usize = 32;
const NETWORK_CHANNELS: usize = 1_024;
const RES2_SCALE: usize = 8;
const RES2_CHANNELS: usize = NETWORK_CHANNELS / RES2_SCALE;
const MFA_CHANNELS: usize = 3_072;
const ATTENTION_CHANNELS: usize = 128;
const ASP_CONTEXT_CHANNELS: usize = MFA_CHANNELS * 3;
const POOLED_CHANNELS: usize = MFA_CHANNELS * 2;
const EXPECTED_CONVOLUTION_COUNT: usize = 38;
const EXPECTED_BATCH_NORM_COUNT: usize = 31;
const MAXIMUM_RECORDED_STAGES: usize = 8;
const WEIGHT_FINITE_SCAN_CHUNK_VALUES: usize = 64 * 1024;

// The scratch proof below is derived from matrixmultiply 0.3's default f32
// panel geometry. Reject its compile-time tuning escape hatches so the compiled
// kernel cannot exceed the published reserve without an explicit contract
// revision and a new memory proof.
const _: () = {
    assert!(option_env!("MATMUL_SGEMM_NC").is_none());
    assert!(option_env!("MATMUL_SGEMM_KC").is_none());
    assert!(option_env!("MATMUL_SGEMM_MC").is_none());
};

#[derive(Clone)]
struct ScratchMeter {
    state: Arc<ScratchMeterState>,
}

struct ScratchMeterState {
    limit_bytes: u64,
    current_bytes: AtomicU64,
    peak_bytes: AtomicU64,
}

struct ScratchLease {
    state: Arc<ScratchMeterState>,
    bytes: u64,
}

struct Accounted<T> {
    value: T,
    _lease: ScratchLease,
}

type AccountedStatistics = (Accounted<Vec<f32>>, Accounted<Vec<f32>>);

impl<T> fmt::Debug for Accounted<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Accounted")
            .field("value", &"<redacted>")
            .field("bytes", &self._lease.bytes)
            .finish()
    }
}

impl ScratchMeter {
    fn new(limit_bytes: u64) -> FwResult<Self> {
        if limit_bytes == 0 {
            return Err(ecapa_error(
                "inference_resource",
                "logical scratch limit must be nonzero",
            ));
        }
        Ok(Self {
            state: Arc::new(ScratchMeterState {
                limit_bytes,
                current_bytes: AtomicU64::new(0),
                peak_bytes: AtomicU64::new(0),
            }),
        })
    }

    fn reserve_f32(&self, elements: usize) -> FwResult<ScratchLease> {
        let bytes = checked_f32_bytes(elements)?;
        let mut current = self.state.current_bytes.load(Ordering::Relaxed);
        let next = loop {
            let next = current.checked_add(bytes).ok_or_else(|| {
                ecapa_error("inference_resource", "logical scratch accounting overflows")
            })?;
            if next > self.state.limit_bytes {
                return Err(ecapa_error(
                    "inference_resource",
                    "live logical scratch exceeds the admitted bound",
                ));
            }
            match self.state.current_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break next,
                Err(observed) => current = observed,
            }
        };
        self.state.peak_bytes.fetch_max(next, Ordering::Relaxed);
        Ok(ScratchLease {
            state: Arc::clone(&self.state),
            bytes,
        })
    }

    fn current_bytes(&self) -> u64 {
        self.state.current_bytes.load(Ordering::Relaxed)
    }

    fn peak_bytes(&self) -> u64 {
        self.state.peak_bytes.load(Ordering::Relaxed)
    }
}

impl Drop for ScratchLease {
    fn drop(&mut self) {
        let previous = self
            .state
            .current_bytes
            .fetch_sub(self.bytes, Ordering::Relaxed);
        debug_assert!(previous >= self.bytes);
    }
}

impl<T> Accounted<T> {
    fn new(value: T, lease: ScratchLease) -> Self {
        Self {
            value,
            _lease: lease,
        }
    }

    fn map<U>(self, transform: impl FnOnce(T) -> U) -> Accounted<U> {
        Accounted {
            value: transform(self.value),
            _lease: self._lease,
        }
    }
}

impl<T> Deref for Accounted<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for Accounted<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EcapaComputePath {
    FrankenTorchCpuF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EcapaInferenceStage {
    InputValidation,
    InitialTdnn,
    SeRes2BlockOne,
    SeRes2BlockTwo,
    SeRes2BlockThree,
    MultiFeatureAggregation,
    AttentivePooling,
    ProjectionAndNormalization,
}

impl EcapaInferenceStage {
    const fn as_str(self) -> &'static str {
        match self {
            Self::InputValidation => "input_validation",
            Self::InitialTdnn => "initial_tdnn",
            Self::SeRes2BlockOne => "se_res2_block_one",
            Self::SeRes2BlockTwo => "se_res2_block_two",
            Self::SeRes2BlockThree => "se_res2_block_three",
            Self::MultiFeatureAggregation => "multi_feature_aggregation",
            Self::AttentivePooling => "attentive_pooling",
            Self::ProjectionAndNormalization => "projection_and_normalization",
        }
    }
}

const ECAPA_STAGE_ORDER: [EcapaInferenceStage; MAXIMUM_RECORDED_STAGES] = [
    EcapaInferenceStage::InputValidation,
    EcapaInferenceStage::InitialTdnn,
    EcapaInferenceStage::SeRes2BlockOne,
    EcapaInferenceStage::SeRes2BlockTwo,
    EcapaInferenceStage::SeRes2BlockThree,
    EcapaInferenceStage::MultiFeatureAggregation,
    EcapaInferenceStage::AttentivePooling,
    EcapaInferenceStage::ProjectionAndNormalization,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EcapaFallbackReason {
    InvalidInput,
    ResourceLimit,
    Cancelled,
    CheckpointFailure,
    InternalContractFailure,
    NumericalFailure,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(try_from = "EcapaScratchPlanWire")]
pub struct EcapaScratchPlan {
    pub frame_count: usize,
    pub row_chunk_frames: usize,
    pub input_feature_bytes: u64,
    pub owned_peak_buffer_bytes: u64,
    pub kernel_scratch_reserve_bytes: u64,
    pub estimated_peak_buffer_bytes: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EcapaScratchPlanWire {
    frame_count: usize,
    row_chunk_frames: usize,
    input_feature_bytes: u64,
    owned_peak_buffer_bytes: u64,
    kernel_scratch_reserve_bytes: u64,
    estimated_peak_buffer_bytes: u64,
}

impl Serialize for EcapaScratchPlan {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let expected = plan_ecapa_scratch(self.frame_count)
            .map_err(|_| serde::ser::Error::custom("ecapa.trace_plan: invalid frame count"))?;
        if self != &expected {
            return Err(serde::ser::Error::custom(
                "ecapa.trace_plan: scratch-plan arithmetic is invalid",
            ));
        }
        EcapaScratchPlanWire {
            frame_count: self.frame_count,
            row_chunk_frames: self.row_chunk_frames,
            input_feature_bytes: self.input_feature_bytes,
            owned_peak_buffer_bytes: self.owned_peak_buffer_bytes,
            kernel_scratch_reserve_bytes: self.kernel_scratch_reserve_bytes,
            estimated_peak_buffer_bytes: self.estimated_peak_buffer_bytes,
        }
        .serialize(serializer)
    }
}

impl TryFrom<EcapaScratchPlanWire> for EcapaScratchPlan {
    type Error = String;

    fn try_from(wire: EcapaScratchPlanWire) -> Result<Self, Self::Error> {
        let candidate = Self {
            frame_count: wire.frame_count,
            row_chunk_frames: wire.row_chunk_frames,
            input_feature_bytes: wire.input_feature_bytes,
            owned_peak_buffer_bytes: wire.owned_peak_buffer_bytes,
            kernel_scratch_reserve_bytes: wire.kernel_scratch_reserve_bytes,
            estimated_peak_buffer_bytes: wire.estimated_peak_buffer_bytes,
        };
        let expected = plan_ecapa_scratch(candidate.frame_count)
            .map_err(|_| "ecapa.trace_plan: scratch-plan frame count is invalid".to_owned())?;
        if candidate != expected {
            return Err("ecapa.trace_plan: scratch-plan arithmetic is invalid".to_owned());
        }
        Ok(candidate)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EcapaStageTiming {
    pub stage: EcapaInferenceStage,
    pub elapsed_micros: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(try_from = "EcapaInferenceTraceWire")]
pub struct EcapaInferenceTrace {
    pub schema_version: String,
    pub contract_sha256: String,
    pub package_sha256: String,
    pub scratch_proof_reviewed_frankentorch_revision: String,
    pub compute_path: EcapaComputePath,
    pub frame_count: usize,
    pub input_features: usize,
    pub embedding_dimensions: usize,
    pub scratch_plan: Option<EcapaScratchPlan>,
    pub observed_owned_peak_buffer_bytes: Option<u64>,
    pub stages: Vec<EcapaStageTiming>,
    pub last_stage: Option<EcapaInferenceStage>,
    pub cancellation_stage: Option<EcapaInferenceStage>,
    pub fallback_reason: Option<EcapaFallbackReason>,
    pub maximum_attention_sum_error: Option<f32>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EcapaInferenceTraceWire {
    schema_version: String,
    contract_sha256: String,
    package_sha256: String,
    scratch_proof_reviewed_frankentorch_revision: String,
    compute_path: EcapaComputePath,
    frame_count: usize,
    input_features: usize,
    embedding_dimensions: usize,
    scratch_plan: Option<EcapaScratchPlan>,
    observed_owned_peak_buffer_bytes: Option<u64>,
    stages: BoundedStageTimings,
    last_stage: Option<EcapaInferenceStage>,
    cancellation_stage: Option<EcapaInferenceStage>,
    fallback_reason: Option<EcapaFallbackReason>,
    maximum_attention_sum_error: Option<f32>,
}

#[derive(Serialize)]
struct BoundedStageTimings(Vec<EcapaStageTiming>);

impl<'de> Deserialize<'de> for BoundedStageTimings {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct StageVisitor;

        impl<'de> serde::de::Visitor<'de> for StageVisitor {
            type Value = BoundedStageTimings;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    "at most {MAXIMUM_RECORDED_STAGES} ECAPA stage timings"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut stages = Vec::with_capacity(MAXIMUM_RECORDED_STAGES);
                while let Some(stage) = sequence.next_element()? {
                    if stages.len() == MAXIMUM_RECORDED_STAGES {
                        return Err(serde::de::Error::custom(
                            "ecapa.trace_stages: too many stage timings",
                        ));
                    }
                    stages.push(stage);
                }
                Ok(BoundedStageTimings(stages))
            }
        }

        deserializer.deserialize_seq(StageVisitor)
    }
}

impl TryFrom<EcapaInferenceTraceWire> for EcapaInferenceTrace {
    type Error = String;

    fn try_from(wire: EcapaInferenceTraceWire) -> Result<Self, Self::Error> {
        let trace = Self {
            schema_version: wire.schema_version,
            contract_sha256: wire.contract_sha256,
            package_sha256: wire.package_sha256,
            scratch_proof_reviewed_frankentorch_revision: wire
                .scratch_proof_reviewed_frankentorch_revision,
            compute_path: wire.compute_path,
            frame_count: wire.frame_count,
            input_features: wire.input_features,
            embedding_dimensions: wire.embedding_dimensions,
            scratch_plan: wire.scratch_plan,
            observed_owned_peak_buffer_bytes: wire.observed_owned_peak_buffer_bytes,
            stages: wire.stages.0,
            last_stage: wire.last_stage,
            cancellation_stage: wire.cancellation_stage,
            fallback_reason: wire.fallback_reason,
            maximum_attention_sum_error: wire.maximum_attention_sum_error,
        };
        trace.validate_serialized_contract()?;
        Ok(trace)
    }
}

impl Serialize for EcapaInferenceTrace {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.validate_serialized_contract()
            .map_err(serde::ser::Error::custom)?;
        EcapaInferenceTraceWire {
            schema_version: self.schema_version.clone(),
            contract_sha256: self.contract_sha256.clone(),
            package_sha256: self.package_sha256.clone(),
            scratch_proof_reviewed_frankentorch_revision: self
                .scratch_proof_reviewed_frankentorch_revision
                .clone(),
            compute_path: self.compute_path,
            frame_count: self.frame_count,
            input_features: self.input_features,
            embedding_dimensions: self.embedding_dimensions,
            scratch_plan: self.scratch_plan.clone(),
            observed_owned_peak_buffer_bytes: self.observed_owned_peak_buffer_bytes,
            stages: BoundedStageTimings(self.stages.clone()),
            last_stage: self.last_stage,
            cancellation_stage: self.cancellation_stage,
            fallback_reason: self.fallback_reason,
            maximum_attention_sum_error: self.maximum_attention_sum_error,
        }
        .serialize(serializer)
    }
}

impl Default for EcapaInferenceTrace {
    fn default() -> Self {
        Self {
            schema_version: ECAPA_INFERENCE_SCHEMA.to_owned(),
            contract_sha256: ECAPA_CONTRACT_SHA256.to_owned(),
            package_sha256: ECAPA_PACKAGE_SHA256.to_owned(),
            scratch_proof_reviewed_frankentorch_revision:
                ECAPA_SCRATCH_PROOF_REVIEWED_FRANKENTORCH_REVISION.to_owned(),
            compute_path: EcapaComputePath::FrankenTorchCpuF32,
            frame_count: 0,
            input_features: ECAPA_MEL_BANDS,
            embedding_dimensions: ECAPA_EMBEDDING_DIMENSIONS,
            scratch_plan: None,
            observed_owned_peak_buffer_bytes: None,
            stages: Vec::with_capacity(MAXIMUM_RECORDED_STAGES),
            last_stage: None,
            cancellation_stage: None,
            fallback_reason: None,
            maximum_attention_sum_error: None,
        }
    }
}

impl EcapaInferenceTrace {
    fn validate_serialized_contract(&self) -> Result<(), String> {
        let identity_is_valid = self.schema_version == ECAPA_INFERENCE_SCHEMA
            && self.contract_sha256 == ECAPA_CONTRACT_SHA256
            && self.package_sha256 == ECAPA_PACKAGE_SHA256
            && self.scratch_proof_reviewed_frankentorch_revision
                == ECAPA_SCRATCH_PROOF_REVIEWED_FRANKENTORCH_REVISION
            && self.compute_path == EcapaComputePath::FrankenTorchCpuF32
            && self.input_features == ECAPA_MEL_BANDS
            && self.embedding_dimensions == ECAPA_EMBEDDING_DIMENSIONS;
        if !identity_is_valid {
            return Err("ecapa.trace_identity: inference trace identity is invalid".to_owned());
        }
        if self.stages.len() > MAXIMUM_RECORDED_STAGES
            || self
                .stages
                .iter()
                .zip(ECAPA_STAGE_ORDER)
                .any(|(timing, expected)| timing.stage != expected)
        {
            return Err("ecapa.trace_stages: inference stage sequence is invalid".to_owned());
        }
        if let Some(plan) = &self.scratch_plan {
            let expected_plan = plan_ecapa_scratch(self.frame_count).map_err(|_| {
                "ecapa.trace_plan: scratch plan has an invalid frame count".to_owned()
            })?;
            if plan.frame_count != self.frame_count || plan != &expected_plan {
                return Err(
                    "ecapa.trace_plan: scratch plan does not match the trace contract".to_owned(),
                );
            }
        } else if (ECAPA_MINIMUM_INFERENCE_FRAMES..=ECAPA_MAXIMUM_INFERENCE_FRAMES)
            .contains(&self.frame_count)
        {
            return Err(
                "ecapa.trace_plan: admitted frame count requires a scratch plan".to_owned(),
            );
        }
        if self.scratch_plan.is_none()
            && (self.stages.len() > 1
                || self
                    .last_stage
                    .is_some_and(|stage| stage != EcapaInferenceStage::InputValidation))
        {
            return Err("ecapa.trace_plan: unplanned trace progressed past validation".to_owned());
        }
        match (&self.scratch_plan, self.observed_owned_peak_buffer_bytes) {
            (Some(plan), Some(observed))
                if observed > 0 && observed <= plan.owned_peak_buffer_bytes => {}
            (_, None) => {}
            _ => {
                return Err("ecapa.trace_plan: observed logical scratch is inconsistent".to_owned());
            }
        }
        match self.last_stage {
            None if !self.stages.is_empty() || self.fallback_reason.is_some() => {
                return Err("ecapa.trace_stages: last stage is missing".to_owned());
            }
            Some(last_stage) => {
                let last_index = ECAPA_STAGE_ORDER
                    .iter()
                    .position(|stage| *stage == last_stage)
                    .ok_or_else(|| "ecapa.trace_stages: last stage is invalid".to_owned())?;
                let matches_timed_stage = self
                    .stages
                    .len()
                    .checked_sub(1)
                    .is_some_and(|index| last_index == index);
                let matches_untimed_checkpoint =
                    self.stages.len() < MAXIMUM_RECORDED_STAGES && last_index == self.stages.len();
                if !matches_timed_stage && !matches_untimed_checkpoint {
                    return Err(
                        "ecapa.trace_stages: last stage is inconsistent with timings".to_owned(),
                    );
                }
            }
            None => {}
        }
        match (self.cancellation_stage, self.fallback_reason) {
            (Some(stage), Some(EcapaFallbackReason::Cancelled))
                if self.last_stage == Some(stage) => {}
            (None, Some(EcapaFallbackReason::Cancelled)) | (Some(_), _) => {
                return Err(
                    "ecapa.trace_cancellation: cancellation fields are inconsistent".to_owned(),
                );
            }
            (None, _) => {}
        }
        if self.fallback_reason.is_none() {
            if self.frame_count == 0 {
                if self.scratch_plan.is_some()
                    || self.observed_owned_peak_buffer_bytes.is_some()
                    || !self.stages.is_empty()
                    || self.last_stage.is_some()
                    || self.maximum_attention_sum_error.is_some()
                {
                    return Err("ecapa.trace_stages: empty trace is inconsistent".to_owned());
                }
            } else if !(ECAPA_MINIMUM_INFERENCE_FRAMES..=ECAPA_MAXIMUM_INFERENCE_FRAMES)
                .contains(&self.frame_count)
                || self.scratch_plan.is_none()
                || self.observed_owned_peak_buffer_bytes.is_none()
                || self.stages.len() != MAXIMUM_RECORDED_STAGES
            {
                return Err("ecapa.trace_stages: successful trace is incomplete".to_owned());
            }
        }
        let attention_diagnostic_required = self.fallback_reason.is_none() && self.frame_count != 0
            || self.last_stage == Some(EcapaInferenceStage::ProjectionAndNormalization);
        let attention_diagnostic_valid = match self.maximum_attention_sum_error {
            Some(value) => {
                value.is_finite()
                    && value >= 0.0
                    && self.stages.len() >= 7
                    && self.last_stage == Some(EcapaInferenceStage::ProjectionAndNormalization)
            }
            None => !attention_diagnostic_required,
        };
        if !attention_diagnostic_valid {
            return Err("ecapa.trace_attention: attention diagnostic is inconsistent".to_owned());
        }
        Ok(())
    }

    fn begin(&mut self, frame_count: usize) {
        self.schema_version = ECAPA_INFERENCE_SCHEMA.to_owned();
        self.contract_sha256 = ECAPA_CONTRACT_SHA256.to_owned();
        self.package_sha256 = ECAPA_PACKAGE_SHA256.to_owned();
        self.scratch_proof_reviewed_frankentorch_revision =
            ECAPA_SCRATCH_PROOF_REVIEWED_FRANKENTORCH_REVISION.to_owned();
        self.compute_path = EcapaComputePath::FrankenTorchCpuF32;
        self.frame_count = frame_count;
        self.input_features = ECAPA_MEL_BANDS;
        self.embedding_dimensions = ECAPA_EMBEDDING_DIMENSIONS;
        self.scratch_plan = None;
        self.observed_owned_peak_buffer_bytes = None;
        self.stages.clear();
        self.last_stage = None;
        self.cancellation_stage = None;
        self.fallback_reason = None;
        self.maximum_attention_sum_error = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EcapaInferenceConfig {
    pub maximum_peak_buffer_bytes: u64,
}

impl Default for EcapaInferenceConfig {
    fn default() -> Self {
        Self {
            maximum_peak_buffer_bytes: ECAPA_DEFAULT_MAXIMUM_PEAK_BUFFER_BYTES,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct EcapaEmbedding([f32; ECAPA_EMBEDDING_DIMENSIONS]);

impl EcapaEmbedding {
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    #[must_use]
    pub fn into_array(self) -> [f32; ECAPA_EMBEDDING_DIMENSIONS] {
        self.0
    }
}

impl fmt::Debug for EcapaEmbedding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EcapaEmbedding")
            .field("dimensions", &ECAPA_EMBEDDING_DIMENSIONS)
            .field("values", &"<redacted>")
            .finish()
    }
}

#[derive(Clone)]
pub struct EcapaInferenceOutput {
    pub embedding: EcapaEmbedding,
    pub diagnostics: EcapaInferenceTrace,
}

impl fmt::Debug for EcapaInferenceOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EcapaInferenceOutput")
            .field("embedding", &self.embedding)
            .field("diagnostics", &self.diagnostics)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EcapaModelInfo {
    pub contract_sha256: String,
    pub package_sha256: String,
    pub scratch_proof_reviewed_frankentorch_revision: String,
    pub tensor_count: usize,
    pub resident_weight_bytes: u64,
    pub conservative_load_accounted_payload_bytes: u64,
}

pub struct EcapaModel {
    weights: EcapaWeights,
    info: EcapaModelInfo,
}

impl fmt::Debug for EcapaModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EcapaModel")
            .field("info", &self.info)
            .field("weights", &"<redacted>")
            .finish()
    }
}

impl EcapaModel {
    /// Load the exact pinned ECAPA package and materialize typed inference weights.
    pub fn load(package_path: &Path) -> FwResult<Self> {
        let token = CancellationToken::unbounded(); // ubs:ignore — cancellation token is not a secret
        Self::load_with_checkpoint(package_path, &|| token.checkpoint())
    }

    /// Load with a caller-owned cooperative cancellation/deadline checkpoint.
    ///
    /// Package reads are checked every 64 KiB, typed loading checks before
    /// every tensor, and weight transposition checks every 32 output channels.
    /// Decoding one already bounded safetensors tensor is atomic because the
    /// shared loader has no callback surface; the largest such tensor contains
    /// 9,437,184 public model values. Its subsequent finite-value scan checks
    /// cancellation every 65,536 values.
    pub fn load_with_checkpoint(
        package_path: &Path,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let load_checkpoint = || ecapa_load_checkpoint(checkpoint);
        let package =
            load_verified_ecapa_weight_package_with_checkpoint(package_path, &load_checkpoint)?;
        Self::from_verified_package(&package, &load_checkpoint)
    }

    #[allow(dead_code)]
    pub(crate) fn load_with_token(
        package_path: &Path,
        token: &CancellationToken,
    ) -> FwResult<Self> {
        Self::load_with_checkpoint(package_path, &|| token.checkpoint())
    }

    fn from_verified_package(
        package: &SafetensorsFile,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let weights = EcapaWeights::load(package, checkpoint)?;
        let resident_weight_bytes = checked_f32_bytes(weights.resident_f32_elements())?;
        if resident_weight_bytes != ECAPA_MODEL_RESIDENT_BYTES {
            return Err(ecapa_error(
                "model_census",
                "typed model resident byte census does not match the frozen architecture",
            ));
        }
        Ok(Self {
            weights,
            info: EcapaModelInfo {
                contract_sha256: ECAPA_CONTRACT_SHA256.to_owned(),
                package_sha256: ECAPA_PACKAGE_SHA256.to_owned(),
                scratch_proof_reviewed_frankentorch_revision:
                    ECAPA_SCRATCH_PROOF_REVIEWED_FRANKENTORCH_REVISION.to_owned(),
                tensor_count: ECAPA_EXPORTED_TENSOR_COUNT,
                resident_weight_bytes,
                conservative_load_accounted_payload_bytes:
                    ECAPA_CONSERVATIVE_LOAD_ACCOUNTED_PAYLOAD_BYTES,
            },
        })
    }

    #[must_use]
    pub fn info(&self) -> &EcapaModelInfo {
        &self.info
    }

    /// Infer one normalized speaker embedding from frame-major `[T, 80]` features.
    pub fn infer(
        &self,
        features: &[f32],
        frame_count: usize,
        config: EcapaInferenceConfig,
    ) -> FwResult<EcapaInferenceOutput> {
        let token = CancellationToken::unbounded(); // ubs:ignore — cancellation token is not a secret
        let checkpoint = || token.checkpoint();
        let mut trace = EcapaInferenceTrace::default();
        self.infer_with_checkpoint(features, frame_count, config, &checkpoint, &mut trace)
    }

    /// Infer with a caller-owned cooperative cancellation/deadline checkpoint.
    ///
    /// `trace` is reset before work and remains inspectable on failure. It never
    /// contains feature values, embeddings, source paths, or inferred identities.
    pub fn infer_with_checkpoint(
        &self,
        features: &[f32],
        frame_count: usize,
        config: EcapaInferenceConfig,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
        trace: &mut EcapaInferenceTrace,
    ) -> FwResult<EcapaInferenceOutput> {
        self.infer_internal(features, frame_count, config, checkpoint, trace, false)
            .map(|(output, _)| output)
    }

    /// Convenience boundary for the frozen frontend output.
    pub fn infer_frontend(
        &self,
        frontend: &EcapaFrontendOutput,
        config: EcapaInferenceConfig,
    ) -> FwResult<EcapaInferenceOutput> {
        if frontend.mel_band_count != ECAPA_MEL_BANDS {
            return Err(ecapa_error(
                "input_shape",
                "frontend mel-band count does not match the model",
            ));
        }
        self.infer(
            &frontend.sentence_mean_normalized,
            frontend.frame_count,
            config,
        )
    }

    /// Compute the production frontend and infer one speaker embedding from an
    /// admitted 16 kHz mono PCM window under one cancellation boundary.
    pub fn infer_pcm_with_checkpoint(
        &self,
        samples: &[f32],
        config: EcapaInferenceConfig,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
        trace: &mut EcapaInferenceTrace,
    ) -> FwResult<EcapaInferenceOutput> {
        let frontend = crate::ecapa_conformance::ecapa_frontend_runtime(samples, checkpoint)?;
        self.infer_with_checkpoint(
            &frontend.sentence_mean_normalized,
            frontend.frame_count,
            config,
            checkpoint,
            trace,
        )
    }

    fn infer_internal(
        &self,
        features: &[f32],
        frame_count: usize,
        config: EcapaInferenceConfig,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
        trace: &mut EcapaInferenceTrace,
        capture_reference_stages: bool,
    ) -> FwResult<(EcapaInferenceOutput, Option<InternalReferenceTrace>)> {
        trace.begin(frame_count);
        // Preserve a valid allocation plan even when a later shape, value, or
        // caller resource-limit check rejects the request. This is diagnostic
        // metadata only: the estimate covers owned f32 payloads, not allocator
        // bookkeeping, stack use, model weights, or process RSS.
        trace.scratch_plan = plan_ecapa_scratch(frame_count).ok();
        let (input, plan, scratch_meter) = run_stage(
            trace,
            EcapaInferenceStage::InputValidation,
            checkpoint,
            |_| {
                let plan = validate_inference_request(features, frame_count, config)?;
                let scratch_meter = ScratchMeter::new(plan.owned_peak_buffer_bytes)?;
                matrix_from_slice(frame_count, ECAPA_MEL_BANDS, features, &scratch_meter)
                    .map(|matrix| (matrix, plan, scratch_meter))
            },
        )?;
        trace.scratch_plan = Some(plan);

        let initial = run_stage(
            trace,
            EcapaInferenceStage::InitialTdnn,
            checkpoint,
            |stage_checkpoint| {
                self.weights
                    .initial
                    .forward(&input, stage_checkpoint, &scratch_meter)
            },
        )?;
        let mut reference = capture_reference_stages.then(InternalReferenceTrace::default);
        if let Some(reference) = &mut reference {
            reference.initial_tdnn = Some(initial.value.clone());
        }
        drop(input);

        let block_one = run_stage(
            trace,
            EcapaInferenceStage::SeRes2BlockOne,
            checkpoint,
            |stage_checkpoint| {
                self.weights.se_res2[0].forward(&initial, stage_checkpoint, &scratch_meter)
            },
        )?;
        if let Some(reference) = &mut reference {
            reference.first_se_res2 = Some(block_one.value.clone());
        }
        drop(initial);
        let block_two = run_stage(
            trace,
            EcapaInferenceStage::SeRes2BlockTwo,
            checkpoint,
            |stage_checkpoint| {
                self.weights.se_res2[1].forward(&block_one, stage_checkpoint, &scratch_meter)
            },
        )?;
        let block_three = run_stage(
            trace,
            EcapaInferenceStage::SeRes2BlockThree,
            checkpoint,
            |stage_checkpoint| {
                self.weights.se_res2[2].forward(&block_two, stage_checkpoint, &scratch_meter)
            },
        )?;

        let mfa = run_stage(
            trace,
            EcapaInferenceStage::MultiFeatureAggregation,
            checkpoint,
            |stage_checkpoint| {
                let aggregation = concatenate_mfa(
                    &block_one,
                    &block_two,
                    &block_three,
                    stage_checkpoint,
                    &scratch_meter,
                )?;
                drop(block_one);
                drop(block_two);
                drop(block_three);
                self.weights
                    .mfa
                    .forward(&aggregation, stage_checkpoint, &scratch_meter)
            },
        )?;
        if let Some(reference) = &mut reference {
            reference.multi_feature_aggregation = Some(mfa.value.clone());
        }
        let (mut pooled, attention_sum_error) = run_stage(
            trace,
            EcapaInferenceStage::AttentivePooling,
            checkpoint,
            |stage_checkpoint| {
                self.weights
                    .attentive_pool(&mfa, stage_checkpoint, &scratch_meter)
            },
        )?;
        trace.maximum_attention_sum_error = Some(attention_sum_error);
        if let Some(reference) = &mut reference {
            reference.attentive_pooling = Some(pooled.value.clone());
        }
        drop(mfa);

        let raw_embedding = run_stage(
            trace,
            EcapaInferenceStage::ProjectionAndNormalization,
            checkpoint,
            |stage_checkpoint| {
                self.weights
                    .asp_bn
                    .apply(&mut pooled, false, stage_checkpoint)?;
                let raw = self
                    .weights
                    .fc
                    .forward(&pooled, stage_checkpoint, &scratch_meter)?;
                ensure_finite(&raw, "embedding", stage_checkpoint)?;
                let mut values: [f32; ECAPA_EMBEDDING_DIMENSIONS] =
                    raw.data.as_slice().try_into().map_err(|_| {
                        ecapa_error("embedding_shape", "raw embedding shape is invalid")
                    })?;
                let unnormalized = values;
                normalize_ecapa_embedding(&mut values)?;
                Ok((unnormalized, values))
            },
        )?;
        if let Some(reference) = &mut reference {
            reference.raw_embedding = Some(raw_embedding.0);
        }
        drop(pooled);
        if scratch_meter.current_bytes() != 0 {
            return Err(ecapa_error(
                "inference_resource",
                "logical scratch was retained after inference",
            ));
        }
        trace.observed_owned_peak_buffer_bytes = Some(scratch_meter.peak_bytes());
        let output = EcapaInferenceOutput {
            embedding: EcapaEmbedding(raw_embedding.1),
            diagnostics: trace.clone(),
        };
        Ok((output, reference))
    }
}

/// Compute a checked conservative numeric-buffer ceiling for one request.
///
/// The plan separately reports ECAPA-owned f32 payloads and a conservative
/// reserve for the reviewed FrankenTorch/matrixmultiply packing buffers. It
/// excludes allocator bookkeeping, stack use, resident model weights,
/// optional test-only golden captures, and process RSS.
pub fn plan_ecapa_scratch(frame_count: usize) -> FwResult<EcapaScratchPlan> {
    if !(ECAPA_MINIMUM_INFERENCE_FRAMES..=ECAPA_MAXIMUM_INFERENCE_FRAMES).contains(&frame_count) {
        return Err(ecapa_error(
            "input_frames",
            "frame count is outside the frozen 51 through 301 range",
        ));
    }
    let batch = frame_count.min(MATMUL_ROW_CHUNK_FRAMES);
    let input_feature_elements = checked_product(frame_count, ECAPA_MEL_BANDS)?;
    let phases = [
        checked_linear_phase(1_104, frame_count, 1_424, batch, 0)?,
        checked_linear_phase(5_120, frame_count, 1_024, batch, 0)?,
        checked_linear_phase(5_120, frame_count, 512, batch, 0)?,
        checked_linear_phase(6_144, frame_count, 1_024, batch, 0)?,
        checked_linear_phase(6_144, frame_count, 3_072, batch, 0)?,
        // Context construction retains the 3,072-channel global mean and
        // standard-deviation vectors until its final chunk completes.
        checked_linear_phase(3_200, frame_count, 9_344, batch, 6_144)?,
        checked_linear_phase(6_272, frame_count, 3_072, batch, 0)?,
        checked_linear_phase(6_144, frame_count, 0, batch, 6_144)?,
    ];
    let peak_elements = phases
        .into_iter()
        .max()
        .ok_or_else(|| ecapa_error("inference_resource", "scratch plan has no phases"))?;
    let owned_peak_buffer_bytes = checked_f32_bytes(peak_elements)?;
    let estimated_peak_buffer_bytes = owned_peak_buffer_bytes
        .checked_add(ECAPA_CONSERVATIVE_KERNEL_SCRATCH_BYTES)
        .ok_or_else(|| ecapa_error("inference_resource", "scratch plan byte count overflows"))?;
    Ok(EcapaScratchPlan {
        frame_count,
        row_chunk_frames: batch,
        input_feature_bytes: checked_f32_bytes(input_feature_elements)?,
        owned_peak_buffer_bytes,
        kernel_scratch_reserve_bytes: ECAPA_CONSERVATIVE_KERNEL_SCRATCH_BYTES,
        estimated_peak_buffer_bytes,
    })
}

fn validate_inference_request(
    features: &[f32],
    frame_count: usize,
    config: EcapaInferenceConfig,
) -> FwResult<EcapaScratchPlan> {
    let plan = plan_ecapa_scratch(frame_count)?;
    let expected = checked_product(frame_count, ECAPA_MEL_BANDS)?;
    if features.len() != expected {
        return Err(ecapa_error(
            "input_shape",
            "feature length does not match frame_count times 80",
        ));
    }
    if features
        .iter()
        .any(|value| !value.is_finite() || value.abs() > ECAPA_MAXIMUM_ABSOLUTE_INPUT_FEATURE)
    {
        return Err(ecapa_error(
            "input_value",
            "features must be finite and within the frozen magnitude bound",
        ));
    }
    if config.maximum_peak_buffer_bytes == 0
        || plan.estimated_peak_buffer_bytes > config.maximum_peak_buffer_bytes
    {
        return Err(ecapa_error(
            "inference_resource",
            "planned inference buffers exceed the caller resource limit",
        ));
    }
    Ok(plan)
}

fn checked_linear_phase(
    frames_coefficient: usize,
    frames: usize,
    batch_coefficient: usize,
    batch: usize,
    constant: usize,
) -> FwResult<usize> {
    let frame_elements = checked_product(frames_coefficient, frames)?;
    let batch_elements = checked_product(batch_coefficient, batch)?;
    frame_elements
        .checked_add(batch_elements)
        .and_then(|value| value.checked_add(constant))
        .ok_or_else(|| ecapa_error("inference_resource", "scratch plan size overflows"))
}

fn checked_product(left: usize, right: usize) -> FwResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| ecapa_error("inference_resource", "matrix size overflows"))
}

fn checked_f32_bytes(elements: usize) -> FwResult<u64> {
    let elements = u64::try_from(elements)
        .map_err(|_| ecapa_error("inference_resource", "element count does not fit u64"))?;
    elements
        .checked_mul(4)
        .ok_or_else(|| ecapa_error("inference_resource", "buffer byte count overflows"))
}

fn run_stage<T>(
    trace: &mut EcapaInferenceTrace,
    stage: EcapaInferenceStage,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    operation: impl FnOnce(&(dyn Fn() -> FwResult<()> + Sync)) -> FwResult<T>,
) -> FwResult<T> {
    trace.last_stage = Some(stage);
    let stage_checkpoint = || match checkpoint() {
        Ok(()) => Ok(()),
        Err(error @ FwError::Cancelled(_)) => Err(error),
        Err(_) => Err(ecapa_error(
            "checkpoint_failure",
            "caller checkpoint returned a non-cancellation failure",
        )),
    };
    if let Err(error) = stage_checkpoint() {
        return Err(record_stage_failure(trace, stage, error));
    }
    let started = Instant::now();
    let result = operation(&stage_checkpoint);
    let elapsed_micros = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX);
    if trace.stages.len() < MAXIMUM_RECORDED_STAGES {
        trace.stages.push(EcapaStageTiming {
            stage,
            elapsed_micros,
        });
    }
    match result {
        Ok(value) => Ok(value),
        Err(error) => Err(record_stage_failure(trace, stage, error)),
    }
}

fn record_stage_failure(
    trace: &mut EcapaInferenceTrace,
    stage: EcapaInferenceStage,
    error: FwError,
) -> FwError {
    if matches!(error, FwError::Cancelled(_)) {
        trace.cancellation_stage = Some(stage);
        trace.fallback_reason = Some(EcapaFallbackReason::Cancelled);
        return FwError::Cancelled(format!(
            "ecapa.inference_cancelled: stage={}",
            stage.as_str()
        ));
    }
    trace.fallback_reason = Some(classify_ecapa_fallback_reason(&error));
    error
}

pub(crate) fn classify_ecapa_fallback_reason(error: &FwError) -> EcapaFallbackReason {
    match error {
        FwError::InvalidRequest(message) if message.contains("ecapa.inference_resource") => {
            EcapaFallbackReason::ResourceLimit
        }
        FwError::InvalidRequest(message) if message.contains("ecapa.checkpoint_failure") => {
            EcapaFallbackReason::CheckpointFailure
        }
        FwError::InvalidRequest(message)
            if message.contains("ecapa.input_") || message.contains("ecapa.frontend_") =>
        {
            EcapaFallbackReason::InvalidInput
        }
        FwError::InvalidRequest(message)
            if message.contains("ecapa.kernel_failure")
                || message.contains("ecapa.") && message.contains("_shape") =>
        {
            EcapaFallbackReason::InternalContractFailure
        }
        FwError::InvalidRequest(_) => EcapaFallbackReason::NumericalFailure,
        _ => EcapaFallbackReason::InternalContractFailure,
    }
}

#[derive(Default)]
struct InternalReferenceTrace {
    initial_tdnn: Option<Mat>,
    first_se_res2: Option<Mat>,
    multi_feature_aggregation: Option<Mat>,
    attentive_pooling: Option<Mat>,
    raw_embedding: Option<[f32; ECAPA_EMBEDDING_DIMENSIONS]>,
}

struct EcapaWeights {
    initial: TdnnLayer,
    se_res2: [SeRes2Block; 3],
    mfa: TdnnLayer,
    asp_tdnn: TdnnLayer,
    asp_conv: AffineConv1d,
    asp_bn: BatchNormAffine,
    fc: AffineConv1d,
}

impl EcapaWeights {
    fn load(
        package: &SafetensorsFile,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let mut loader = WeightLoader::new(package, checkpoint);

        // Load the largest tensor first. This keeps the conservative peak well
        // bounded even though each natural-layout weight is transposed once.
        let mfa = loader.load_tdnn("mfa", MFA_CHANNELS, MFA_CHANNELS, 1, 1)?;
        let asp_tdnn =
            loader.load_tdnn("asp.tdnn", ASP_CONTEXT_CHANNELS, ATTENTION_CHANNELS, 1, 1)?;
        let asp_conv =
            loader.load_wrapped_conv("asp.conv", ATTENTION_CHANNELS, MFA_CHANNELS, 1, 1)?;
        let asp_bn = loader.load_batch_norm("asp_bn", POOLED_CHANNELS)?;
        let fc =
            loader.load_wrapped_conv("fc", POOLED_CHANNELS, ECAPA_EMBEDDING_DIMENSIONS, 1, 1)?;
        let initial = loader.load_tdnn("blocks.0", ECAPA_MEL_BANDS, NETWORK_CHANNELS, 5, 1)?;

        checkpoint()?;
        let block_one = loader.load_se_res2(1, 2)?;
        checkpoint()?;
        let block_two = loader.load_se_res2(2, 3)?;
        checkpoint()?;
        let block_three = loader.load_se_res2(3, 4)?;
        let se_res2 = [block_one, block_two, block_three];
        checkpoint()?;
        loader.finish()?;
        Ok(Self {
            initial,
            se_res2,
            mfa,
            asp_tdnn,
            asp_conv,
            asp_bn,
            fc,
        })
    }

    fn resident_f32_elements(&self) -> usize {
        self.initial.resident_f32_elements()
            + self
                .se_res2
                .iter()
                .map(SeRes2Block::resident_f32_elements)
                .sum::<usize>()
            + self.mfa.resident_f32_elements()
            + self.asp_tdnn.resident_f32_elements()
            + self.asp_conv.resident_f32_elements()
            + self.asp_bn.resident_f32_elements()
            + self.fc.resident_f32_elements()
    }

    fn attentive_pool(
        &self,
        input: &Mat,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
        scratch_meter: &ScratchMeter,
    ) -> FwResult<(Accounted<Mat>, f32)> {
        validate_matrix(input, input.rows, MFA_CHANNELS, "asp_input")?;
        let (mean, standard_deviation) = unweighted_statistics(input, checkpoint, scratch_meter)?;
        let mut hidden = context_tdnn_forward(
            input,
            &mean,
            &standard_deviation,
            &self.asp_tdnn,
            checkpoint,
            scratch_meter,
        )?;
        drop(mean);
        drop(standard_deviation);
        for value in &mut hidden.data {
            *value = value.tanh();
        }
        ensure_finite(&hidden, "asp_tanh", checkpoint)?;
        let mut attention = self.asp_conv.forward(&hidden, checkpoint, scratch_meter)?;
        drop(hidden);
        let maximum_sum_error = softmax_over_time(&mut attention, checkpoint)?;
        let pooled = weighted_statistics(input, &attention, checkpoint, scratch_meter)?;
        Ok((pooled, maximum_sum_error))
    }
}

struct WeightLoader<'a> {
    package: &'a SafetensorsFile,
    checkpoint: &'a (dyn Fn() -> FwResult<()> + Sync),
    consumed: BTreeSet<String>,
    convolution_count: usize,
    batch_norm_count: usize,
}

impl<'a> WeightLoader<'a> {
    fn new(
        package: &'a SafetensorsFile,
        checkpoint: &'a (dyn Fn() -> FwResult<()> + Sync),
    ) -> Self {
        Self {
            package,
            checkpoint,
            consumed: BTreeSet::new(),
            convolution_count: 0,
            batch_norm_count: 0,
        }
    }

    fn tensor(&mut self, name: String, expected_shape: &[usize]) -> FwResult<Vec<f32>> {
        (self.checkpoint)()?;
        if !self.consumed.insert(name.clone()) {
            return Err(ecapa_error(
                "tensor_mapping",
                "typed loader attempted to consume one tensor more than once",
            ));
        }
        let (shape, values) = self.package.tensor_f32(&name).map_err(|_| {
            ecapa_error(
                "tensor_decode",
                "authenticated model tensor could not be decoded",
            )
        })?;
        if shape != expected_shape {
            return Err(ecapa_error(
                "tensor_shape",
                "typed model tensor does not match its frozen shape",
            ));
        }
        for values_chunk in values.chunks(WEIGHT_FINITE_SCAN_CHUNK_VALUES) {
            (self.checkpoint)()?;
            if values_chunk.iter().any(|value| !value.is_finite()) {
                return Err(ecapa_error(
                    "tensor_value",
                    "typed model tensor contains a non-finite value",
                ));
            }
        }
        Ok(values)
    }

    fn load_wrapped_conv(
        &mut self,
        path: &str,
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> FwResult<AffineConv1d> {
        self.convolution_count = self
            .convolution_count
            .checked_add(1)
            .ok_or_else(|| ecapa_error("model_census", "convolution count overflows"))?;
        let weights = self.tensor(
            format!("{path}.conv.weight"),
            &[output_channels, input_channels, kernel_size],
        )?;
        let bias = self.tensor(format!("{path}.conv.bias"), &[output_channels])?;
        AffineConv1d::from_natural(
            weights,
            bias,
            input_channels,
            output_channels,
            kernel_size,
            dilation,
            self.checkpoint,
        )
    }

    fn load_batch_norm(&mut self, path: &str, channels: usize) -> FwResult<BatchNormAffine> {
        self.batch_norm_count = self
            .batch_norm_count
            .checked_add(1)
            .ok_or_else(|| ecapa_error("model_census", "batch norm count overflows"))?;
        let gamma = self.tensor(format!("{path}.norm.weight"), &[channels])?;
        let beta = self.tensor(format!("{path}.norm.bias"), &[channels])?;
        let running_mean = self.tensor(format!("{path}.norm.running_mean"), &[channels])?;
        let running_variance = self.tensor(format!("{path}.norm.running_var"), &[channels])?;
        BatchNormAffine::from_source(gamma, beta, running_mean, running_variance)
    }

    fn load_tdnn(
        &mut self,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> FwResult<TdnnLayer> {
        let convolution = self.load_wrapped_conv(
            &format!("{prefix}.conv"),
            input_channels,
            output_channels,
            kernel_size,
            dilation,
        )?;
        let normalization = self.load_batch_norm(&format!("{prefix}.norm"), output_channels)?;
        Ok(TdnnLayer {
            convolution,
            normalization,
        })
    }

    fn load_se_res2(&mut self, block: usize, dilation: usize) -> FwResult<SeRes2Block> {
        let prefix = format!("blocks.{block}");
        let tdnn1 = self.load_tdnn(
            &format!("{prefix}.tdnn1"),
            NETWORK_CHANNELS,
            NETWORK_CHANNELS,
            1,
            1,
        )?;
        let mut res2 = Vec::with_capacity(RES2_SCALE - 1);
        for inner in 0..RES2_SCALE - 1 {
            res2.push(self.load_tdnn(
                &format!("{prefix}.res2net_block.blocks.{inner}"),
                RES2_CHANNELS,
                RES2_CHANNELS,
                3,
                dilation,
            )?);
        }
        let tdnn2 = self.load_tdnn(
            &format!("{prefix}.tdnn2"),
            NETWORK_CHANNELS,
            NETWORK_CHANNELS,
            1,
            1,
        )?;
        let se_conv1 = self.load_wrapped_conv(
            &format!("{prefix}.se_block.conv1"),
            NETWORK_CHANNELS,
            ATTENTION_CHANNELS,
            1,
            1,
        )?;
        let se_conv2 = self.load_wrapped_conv(
            &format!("{prefix}.se_block.conv2"),
            ATTENTION_CHANNELS,
            NETWORK_CHANNELS,
            1,
            1,
        )?;
        Ok(SeRes2Block {
            tdnn1,
            res2,
            tdnn2,
            se_conv1,
            se_conv2,
        })
    }

    fn finish(self) -> FwResult<()> {
        if self.convolution_count != EXPECTED_CONVOLUTION_COUNT
            || self.batch_norm_count != EXPECTED_BATCH_NORM_COUNT
            || self.consumed.len() != ECAPA_EXPORTED_TENSOR_COUNT
        {
            return Err(ecapa_error(
                "model_census",
                "typed layer census does not consume exactly 38 convolutions, 31 batch norms, and 200 tensors",
            ));
        }
        let expected = expected_ecapa_tensors()
            .into_iter()
            .map(|tensor| tensor.name)
            .collect::<BTreeSet<_>>();
        if self.consumed != expected {
            return Err(ecapa_error(
                "tensor_mapping",
                "typed layer mapping does not consume the exact frozen tensor set",
            ));
        }
        Ok(())
    }
}

struct AffineConv1d {
    weight_transposed: Mat,
    bias: Vec<f32>,
    input_channels: usize,
    output_channels: usize,
    kernel_size: usize,
    dilation: usize,
}

impl AffineConv1d {
    fn from_natural(
        weights: Vec<f32>,
        bias: Vec<f32>,
        input_channels: usize,
        output_channels: usize,
        kernel_size: usize,
        dilation: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        if kernel_size == 0 || kernel_size.is_multiple_of(2) || dilation == 0 {
            return Err(ecapa_error(
                "tensor_shape",
                "convolution kernel and dilation are invalid",
            ));
        }
        let patch = checked_product(input_channels, kernel_size)?;
        let expected_weights = checked_product(output_channels, patch)?;
        if weights.len() != expected_weights || bias.len() != output_channels {
            return Err(ecapa_error(
                "tensor_shape",
                "convolution parameters do not match their declared dimensions",
            ));
        }
        let mut transposed = zeroed_f32(expected_weights)?;
        for output in 0..output_channels {
            if output % MATMUL_ROW_CHUNK_FRAMES == 0 {
                checkpoint()?;
            }
            for input in 0..input_channels {
                for kernel in 0..kernel_size {
                    let natural = (output * input_channels + input) * kernel_size + kernel;
                    let target = (input * kernel_size + kernel) * output_channels + output;
                    transposed[target] = weights[natural];
                }
            }
        }
        Ok(Self {
            weight_transposed: Mat::from_vec(patch, output_channels, transposed),
            bias,
            input_channels,
            output_channels,
            kernel_size,
            dilation,
        })
    }

    fn forward(
        &self,
        input: &Mat,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
        scratch_meter: &ScratchMeter,
    ) -> FwResult<Accounted<Mat>> {
        reflect_conv1d(input, self, checkpoint, scratch_meter)
    }

    fn resident_f32_elements(&self) -> usize {
        self.weight_transposed.data.len() + self.bias.len()
    }
}

struct BatchNormAffine {
    scale: Vec<f32>,
    shift: Vec<f32>,
}

impl BatchNormAffine {
    fn from_source(
        gamma: Vec<f32>,
        beta: Vec<f32>,
        running_mean: Vec<f32>,
        running_variance: Vec<f32>,
    ) -> FwResult<Self> {
        let channels = gamma.len();
        if channels == 0
            || beta.len() != channels
            || running_mean.len() != channels
            || running_variance.len() != channels
            || running_variance.iter().any(|variance| *variance < 0.0)
        {
            return Err(ecapa_error(
                "batch_norm_shape",
                "batch norm parameters are inconsistent",
            ));
        }
        let mut scale = zeroed_f32(channels)?;
        let mut shift = zeroed_f32(channels)?;
        for channel in 0..channels {
            scale[channel] =
                gamma[channel] / (running_variance[channel] + ECAPA_BATCH_NORM_EPSILON).sqrt();
            shift[channel] = beta[channel] - running_mean[channel] * scale[channel];
        }
        if scale.iter().chain(&shift).any(|value| !value.is_finite()) {
            return Err(ecapa_error(
                "batch_norm_value",
                "derived batch norm coefficients are non-finite",
            ));
        }
        Ok(Self { scale, shift })
    }

    fn apply(
        &self,
        matrix: &mut Mat,
        relu_first: bool,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<()> {
        if matrix.cols != self.scale.len() || self.shift.len() != self.scale.len() {
            return Err(ecapa_error(
                "batch_norm_shape",
                "activation columns do not match batch norm channels",
            ));
        }
        for (row_index, row) in matrix.data.chunks_mut(matrix.cols).enumerate() {
            if row_index % MATMUL_ROW_CHUNK_FRAMES == 0 {
                checkpoint()?;
            }
            self.apply_row(row, relu_first)?;
        }
        ensure_finite(matrix, "batch_norm", checkpoint)
    }

    fn apply_row(&self, row: &mut [f32], relu_first: bool) -> FwResult<()> {
        if row.len() != self.scale.len() || self.shift.len() != self.scale.len() {
            return Err(ecapa_error(
                "batch_norm_shape",
                "activation row does not match batch norm channels",
            ));
        }
        for (channel, value) in row.iter_mut().enumerate() {
            if !value.is_finite() {
                return Err(ecapa_error(
                    "batch_norm_value",
                    "batch norm input contains a non-finite value",
                ));
            }
            let activated = if relu_first {
                (*value).max(0.0)
            } else {
                *value
            };
            let normalized = activated * self.scale[channel] + self.shift[channel];
            if !normalized.is_finite() {
                return Err(ecapa_error(
                    "batch_norm_value",
                    "batch norm output contains a non-finite value",
                ));
            }
            *value = normalized;
        }
        Ok(())
    }

    fn resident_f32_elements(&self) -> usize {
        self.scale.len() + self.shift.len()
    }
}

struct TdnnLayer {
    convolution: AffineConv1d,
    normalization: BatchNormAffine,
}

impl TdnnLayer {
    fn forward(
        &self,
        input: &Mat,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
        scratch_meter: &ScratchMeter,
    ) -> FwResult<Accounted<Mat>> {
        let mut output = self.convolution.forward(input, checkpoint, scratch_meter)?;
        self.normalization.apply(&mut output, true, checkpoint)?;
        Ok(output)
    }

    fn resident_f32_elements(&self) -> usize {
        self.convolution.resident_f32_elements() + self.normalization.resident_f32_elements()
    }
}

struct SeRes2Block {
    tdnn1: TdnnLayer,
    res2: Vec<TdnnLayer>,
    tdnn2: TdnnLayer,
    se_conv1: AffineConv1d,
    se_conv2: AffineConv1d,
}

impl SeRes2Block {
    fn forward(
        &self,
        residual: &Mat,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
        scratch_meter: &ScratchMeter,
    ) -> FwResult<Accounted<Mat>> {
        validate_matrix(residual, residual.rows, NETWORK_CHANNELS, "se_res2_input")?;
        let projected = self.tdnn1.forward(residual, checkpoint, scratch_meter)?;
        let mixed = res2_forward(&projected, &self.res2, checkpoint, scratch_meter)?;
        let mut output = self.tdnn2.forward(&mixed, checkpoint, scratch_meter)?;
        let mean = time_mean(&output, checkpoint, scratch_meter)?;
        let mean_matrix = mean.map(|values| Mat::from_vec(1, NETWORK_CHANNELS, values));
        let mut squeeze = self
            .se_conv1
            .forward(&mean_matrix, checkpoint, scratch_meter)?;
        for value in &mut squeeze.data {
            *value = value.max(0.0);
        }
        let mut gates = self.se_conv2.forward(&squeeze, checkpoint, scratch_meter)?;
        for gate in &mut gates.data {
            *gate = 1.0 / (1.0 + (-*gate).exp());
        }
        ensure_finite(&gates, "se_gate", checkpoint)?;
        apply_se_gate_and_residual(&mut output, &gates.data, residual, checkpoint)?;
        ensure_finite(&output, "se_res2_output", checkpoint)?;
        Ok(output)
    }

    fn resident_f32_elements(&self) -> usize {
        self.tdnn1.resident_f32_elements()
            + self
                .res2
                .iter()
                .map(TdnnLayer::resident_f32_elements)
                .sum::<usize>()
            + self.tdnn2.resident_f32_elements()
            + self.se_conv1.resident_f32_elements()
            + self.se_conv2.resident_f32_elements()
    }
}

fn res2_forward(
    input: &Mat,
    blocks: &[TdnnLayer],
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    scratch_meter: &ScratchMeter,
) -> FwResult<Accounted<Mat>> {
    if !input.cols.is_multiple_of(RES2_SCALE) || blocks.len() != RES2_SCALE - 1 {
        return Err(ecapa_error(
            "res2_shape",
            "Res2Net input or block count does not match scale eight",
        ));
    }
    let chunk_channels = input.cols / RES2_SCALE;
    let mut output = zeroed_matrix(input.rows, input.cols, scratch_meter)?;
    copy_channel_chunk(input, 0, chunk_channels, &mut output, scratch_meter)?;
    let mut previous: Option<Accounted<Mat>> = None;
    for chunk in 1..RES2_SCALE {
        checkpoint()?;
        let mut current_input = extract_channel_chunk(input, chunk, chunk_channels, scratch_meter)?;
        if let Some(previous) = &previous {
            add_in_place(&mut current_input, previous)?;
        }
        let current = blocks[chunk - 1].forward(&current_input, checkpoint, scratch_meter)?;
        write_channel_chunk(&current, chunk, chunk_channels, &mut output)?;
        previous = Some(current);
    }
    ensure_finite(&output, "res2_output", checkpoint)?;
    Ok(output)
}

fn apply_se_gate_and_residual(
    output: &mut Mat,
    gates: &[f32],
    residual: &Mat,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    if output.rows != residual.rows
        || output.cols != residual.cols
        || output.data.len() != residual.data.len()
        || gates.len() != output.cols
    {
        return Err(ecapa_error(
            "se_shape",
            "SE gate, activation, and residual dimensions disagree",
        ));
    }
    for (row_index, (row, residual_row)) in output
        .data
        .chunks_mut(output.cols)
        .zip(residual.data.chunks(residual.cols))
        .enumerate()
    {
        if row_index % MATMUL_ROW_CHUNK_FRAMES == 0 {
            checkpoint()?;
        }
        for channel in 0..output.cols {
            row[channel] = row[channel] * gates[channel] + residual_row[channel];
        }
    }
    Ok(())
}

fn reflect_conv1d(
    input: &Mat,
    convolution: &AffineConv1d,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    scratch_meter: &ScratchMeter,
) -> FwResult<Accounted<Mat>> {
    validate_matrix(
        input,
        input.rows,
        convolution.input_channels,
        "convolution_input",
    )?;
    let effective_kernel = convolution
        .dilation
        .checked_mul(convolution.kernel_size - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| ecapa_error("convolution_shape", "effective kernel overflows"))?;
    let padding = effective_kernel / 2;
    if effective_kernel % 2 == 0 || padding >= input.rows {
        return Err(ecapa_error(
            "convolution_shape",
            "reflection padding is invalid for the input length",
        ));
    }
    if convolution.kernel_size == 1 {
        return chunked_matmul_bias(
            input,
            &convolution.weight_transposed,
            &convolution.bias,
            checkpoint,
            scratch_meter,
        );
    }

    let patch_channels = checked_product(convolution.input_channels, convolution.kernel_size)?;
    let mut output = zeroed_matrix(input.rows, convolution.output_channels, scratch_meter)?;
    for row_start in (0..input.rows).step_by(MATMUL_ROW_CHUNK_FRAMES) {
        checkpoint()?;
        let row_end = (row_start + MATMUL_ROW_CHUNK_FRAMES).min(input.rows);
        let rows = row_end - row_start;
        let mut patches = zeroed_matrix(rows, patch_channels, scratch_meter)?;
        for local_row in 0..rows {
            let output_time = row_start + local_row;
            for kernel in 0..convolution.kernel_size {
                let padded_index = output_time
                    .checked_add(checked_product(kernel, convolution.dilation)?)
                    .ok_or_else(|| {
                        ecapa_error("convolution_shape", "padded sample index overflows")
                    })?;
                let source_time = reflect_padded_index(padded_index, padding, input.rows)?;
                let source = input.row(source_time);
                let patch_row =
                    &mut patches.data[local_row * patch_channels..(local_row + 1) * patch_channels];
                for input_channel in 0..convolution.input_channels {
                    patch_row[input_channel * convolution.kernel_size + kernel] =
                        source[input_channel];
                }
            }
        }
        let band = cpu_matmul_raw_lhs(
            &patches.data,
            rows,
            &convolution.weight_transposed,
            scratch_meter,
        )?;
        copy_affine_band(&band, &convolution.bias, row_start, &mut output)?;
        checkpoint()?;
    }
    ensure_finite(&output, "convolution_output", checkpoint)?;
    Ok(output)
}

fn reflect_padded_index(
    padded_index: usize,
    padding: usize,
    input_length: usize,
) -> FwResult<usize> {
    if input_length < 2 || padding >= input_length {
        return Err(ecapa_error(
            "convolution_shape",
            "reflection requires padding smaller than a nontrivial input",
        ));
    }
    if padded_index < padding {
        return Ok(padding - padded_index);
    }
    let shifted = padded_index - padding;
    if shifted < input_length {
        return Ok(shifted);
    }
    input_length
        .checked_mul(2)
        .and_then(|value| value.checked_sub(2))
        .and_then(|last| last.checked_sub(shifted))
        .ok_or_else(|| ecapa_error("convolution_shape", "reflected sample index is invalid"))
}

fn chunked_matmul_bias(
    input: &Mat,
    weight_transposed: &Mat,
    bias: &[f32],
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    scratch_meter: &ScratchMeter,
) -> FwResult<Accounted<Mat>> {
    validate_matrix(input, input.rows, weight_transposed.rows, "matmul_input")?;
    validate_matrix(
        weight_transposed,
        weight_transposed.rows,
        weight_transposed.cols,
        "matmul_weight",
    )?;
    if bias.len() != weight_transposed.cols {
        return Err(ecapa_error(
            "matmul_shape",
            "affine bias length does not match output channels",
        ));
    }
    let mut output = zeroed_matrix(input.rows, weight_transposed.cols, scratch_meter)?;
    for row_start in (0..input.rows).step_by(MATMUL_ROW_CHUNK_FRAMES) {
        checkpoint()?;
        let row_end = (row_start + MATMUL_ROW_CHUNK_FRAMES).min(input.rows);
        let rows = row_end - row_start;
        let source_start = checked_product(row_start, input.cols)?;
        let source_end = checked_product(row_end, input.cols)?;
        let band = cpu_matmul_raw_lhs(
            input
                .data
                .get(source_start..source_end)
                .ok_or_else(|| ecapa_error("matmul_shape", "matmul input band is out of bounds"))?,
            rows,
            weight_transposed,
            scratch_meter,
        )?;
        copy_affine_band(&band, bias, row_start, &mut output)?;
        checkpoint()?;
    }
    ensure_finite(&output, "matmul_output", checkpoint)?;
    Ok(output)
}

fn cpu_matmul_raw_lhs(
    lhs: &[f32],
    rows: usize,
    rhs: &Mat,
    scratch_meter: &ScratchMeter,
) -> FwResult<Accounted<Mat>> {
    let output_elements = checked_product(rows, rhs.cols)?;
    let lease = scratch_meter.reserve_f32(output_elements)?;
    nn::matmul_raw_lhs_cpu(lhs, rows, rhs)
        .map(|matrix| Accounted::new(matrix, lease))
        .map_err(|_| ecapa_error("kernel_failure", "CPU matmul rejected a frozen ECAPA shape"))
}

fn copy_affine_band(band: &Mat, bias: &[f32], row_start: usize, output: &mut Mat) -> FwResult<()> {
    if band.cols != output.cols
        || bias.len() != output.cols
        || row_start
            .checked_add(band.rows)
            .is_none_or(|row_end| row_end > output.rows)
    {
        return Err(ecapa_error(
            "matmul_shape",
            "matmul output band dimensions are invalid",
        ));
    }
    for local_row in 0..band.rows {
        let target_row = row_start + local_row;
        let target = &mut output.data[target_row * output.cols..(target_row + 1) * output.cols];
        let source = band.row(local_row);
        for channel in 0..output.cols {
            target[channel] = source[channel] + bias[channel];
        }
    }
    Ok(())
}

fn concatenate_mfa(
    first: &Mat,
    second: &Mat,
    third: &Mat,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    scratch_meter: &ScratchMeter,
) -> FwResult<Accounted<Mat>> {
    if first.rows == 0
        || first.rows != second.rows
        || first.rows != third.rows
        || first.cols != NETWORK_CHANNELS
        || second.cols != NETWORK_CHANNELS
        || third.cols != NETWORK_CHANNELS
    {
        return Err(ecapa_error(
            "mfa_shape",
            "multi-feature aggregation inputs do not match",
        ));
    }
    let mut output = zeroed_matrix(first.rows, MFA_CHANNELS, scratch_meter)?;
    for row in 0..first.rows {
        if row % MATMUL_ROW_CHUNK_FRAMES == 0 {
            checkpoint()?;
        }
        let target = &mut output.data[row * MFA_CHANNELS..(row + 1) * MFA_CHANNELS];
        target[..NETWORK_CHANNELS].copy_from_slice(first.row(row));
        target[NETWORK_CHANNELS..NETWORK_CHANNELS * 2].copy_from_slice(second.row(row));
        target[NETWORK_CHANNELS * 2..].copy_from_slice(third.row(row));
    }
    Ok(output)
}

fn context_tdnn_forward(
    input: &Mat,
    mean: &[f32],
    standard_deviation: &[f32],
    tdnn: &TdnnLayer,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    scratch_meter: &ScratchMeter,
) -> FwResult<Accounted<Mat>> {
    if input.cols != MFA_CHANNELS
        || mean.len() != MFA_CHANNELS
        || standard_deviation.len() != MFA_CHANNELS
        || tdnn.convolution.input_channels != ASP_CONTEXT_CHANNELS
        || tdnn.convolution.output_channels != ATTENTION_CHANNELS
        || tdnn.convolution.kernel_size != 1
    {
        return Err(ecapa_error(
            "asp_shape",
            "attentive pooling context dimensions are invalid",
        ));
    }
    let mut output = zeroed_matrix(input.rows, ATTENTION_CHANNELS, scratch_meter)?;
    for row_start in (0..input.rows).step_by(MATMUL_ROW_CHUNK_FRAMES) {
        checkpoint()?;
        let row_end = (row_start + MATMUL_ROW_CHUNK_FRAMES).min(input.rows);
        let rows = row_end - row_start;
        let mut context = zeroed_matrix(rows, ASP_CONTEXT_CHANNELS, scratch_meter)?;
        for local_row in 0..rows {
            let target = &mut context.data
                [local_row * ASP_CONTEXT_CHANNELS..(local_row + 1) * ASP_CONTEXT_CHANNELS];
            target[..MFA_CHANNELS].copy_from_slice(input.row(row_start + local_row));
            target[MFA_CHANNELS..MFA_CHANNELS * 2].copy_from_slice(mean);
            target[MFA_CHANNELS * 2..].copy_from_slice(standard_deviation);
        }
        let band = cpu_matmul_raw_lhs(
            &context.data,
            rows,
            &tdnn.convolution.weight_transposed,
            scratch_meter,
        )?;
        copy_affine_band(&band, &tdnn.convolution.bias, row_start, &mut output)?;
        let target = &mut output.data[row_start * ATTENTION_CHANNELS..row_end * ATTENTION_CHANNELS];
        for row in target.chunks_mut(ATTENTION_CHANNELS) {
            tdnn.normalization.apply_row(row, true)?;
        }
        checkpoint()?;
    }
    ensure_finite(&output, "asp_tdnn", checkpoint)?;
    Ok(output)
}

fn unweighted_statistics(
    input: &Mat,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    scratch_meter: &ScratchMeter,
) -> FwResult<AccountedStatistics> {
    if input.rows == 0 || input.cols == 0 {
        return Err(ecapa_error("statistics_shape", "statistics input is empty"));
    }
    let mut mean = accounted_zeroed_f32(input.cols, scratch_meter)?;
    let mut standard_deviation = accounted_zeroed_f32(input.cols, scratch_meter)?;
    let weight = 1.0 / input.rows as f32;
    for channel in 0..input.cols {
        if channel % MATMUL_ROW_CHUNK_FRAMES == 0 {
            checkpoint()?;
        }
        let mut channel_mean = 0.0f32;
        for row in 0..input.rows {
            channel_mean += weight * input.data[row * input.cols + channel];
        }
        mean[channel] = channel_mean;
        let mut variance = 0.0f32;
        for row in 0..input.rows {
            let difference = input.data[row * input.cols + channel] - channel_mean;
            variance += weight * difference * difference;
        }
        if !variance.is_finite() {
            return Err(ecapa_error(
                "statistics_value",
                "global variance is non-finite",
            ));
        }
        standard_deviation[channel] = variance.max(ECAPA_POOLING_VARIANCE_FLOOR).sqrt();
    }
    if mean
        .iter()
        .chain(standard_deviation.iter())
        .any(|value| !value.is_finite())
    {
        return Err(ecapa_error(
            "statistics_value",
            "global statistics contain a non-finite value",
        ));
    }
    Ok((mean, standard_deviation))
}

fn softmax_over_time(
    attention: &mut Mat,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<f32> {
    if attention.rows == 0 || attention.cols != MFA_CHANNELS {
        return Err(ecapa_error(
            "attention_shape",
            "attention logits have invalid dimensions",
        ));
    }
    let mut maximum_sum_error = 0.0f32;
    for channel in 0..attention.cols {
        if channel % MATMUL_ROW_CHUNK_FRAMES == 0 {
            checkpoint()?;
        }
        let mut maximum = f32::NEG_INFINITY;
        for row in 0..attention.rows {
            maximum = maximum.max(attention.data[row * attention.cols + channel]);
        }
        let mut total = 0.0f32;
        for row in 0..attention.rows {
            let index = row * attention.cols + channel;
            let value = (attention.data[index] - maximum).exp();
            attention.data[index] = value;
            total += value;
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(ecapa_error(
                "attention_value",
                "attention normalization denominator is invalid",
            ));
        }
        let inverse = total.recip();
        let mut observed_sum = 0.0f32;
        for row in 0..attention.rows {
            let index = row * attention.cols + channel;
            attention.data[index] *= inverse;
            observed_sum += attention.data[index];
        }
        maximum_sum_error = maximum_sum_error.max((observed_sum - 1.0).abs());
    }
    ensure_finite(attention, "attention_weights", checkpoint)?;
    Ok(maximum_sum_error)
}

fn weighted_statistics(
    input: &Mat,
    attention: &Mat,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    scratch_meter: &ScratchMeter,
) -> FwResult<Accounted<Mat>> {
    if input.rows != attention.rows || input.cols != attention.cols || input.cols != MFA_CHANNELS {
        return Err(ecapa_error(
            "pooling_shape",
            "attention weights do not match pooling input",
        ));
    }
    let mut pooled = accounted_zeroed_f32(POOLED_CHANNELS, scratch_meter)?;
    for channel in 0..input.cols {
        if channel % MATMUL_ROW_CHUNK_FRAMES == 0 {
            checkpoint()?;
        }
        let mut mean = 0.0f32;
        for row in 0..input.rows {
            let index = row * input.cols + channel;
            mean += attention.data[index] * input.data[index];
        }
        let mut variance = 0.0f32;
        for row in 0..input.rows {
            let index = row * input.cols + channel;
            let difference = input.data[index] - mean;
            let squared_difference = difference * difference;
            let contribution = attention.data[index] * squared_difference;
            if !difference.is_finite()
                || !squared_difference.is_finite()
                || !contribution.is_finite()
            {
                return Err(ecapa_error(
                    "pooling_value",
                    "attentive statistics contain a non-finite value",
                ));
            }
            variance += contribution;
        }
        if !mean.is_finite() || !variance.is_finite() {
            return Err(ecapa_error(
                "pooling_value",
                "attentive statistics contain a non-finite value",
            ));
        }
        pooled[channel] = mean;
        pooled[MFA_CHANNELS + channel] = variance.max(ECAPA_POOLING_VARIANCE_FLOOR).sqrt();
    }
    let output = pooled.map(|values| Mat::from_vec(1, POOLED_CHANNELS, values));
    ensure_finite(&output, "attentive_pooling", checkpoint)?;
    Ok(output)
}

fn time_mean(
    input: &Mat,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    scratch_meter: &ScratchMeter,
) -> FwResult<Accounted<Vec<f32>>> {
    if input.rows == 0 || input.cols == 0 {
        return Err(ecapa_error("statistics_shape", "time mean input is empty"));
    }
    let mut mean = accounted_zeroed_f32(input.cols, scratch_meter)?;
    let inverse = 1.0 / input.rows as f32;
    for channel in 0..input.cols {
        if channel % MATMUL_ROW_CHUNK_FRAMES == 0 {
            checkpoint()?;
        }
        let mut sum = 0.0f32;
        for row in 0..input.rows {
            sum += input.data[row * input.cols + channel];
        }
        mean[channel] = sum * inverse;
    }
    Ok(mean)
}

fn extract_channel_chunk(
    input: &Mat,
    chunk: usize,
    chunk_channels: usize,
    scratch_meter: &ScratchMeter,
) -> FwResult<Accounted<Mat>> {
    let channel_start = checked_product(chunk, chunk_channels)?;
    let channel_end = channel_start
        .checked_add(chunk_channels)
        .ok_or_else(|| ecapa_error("res2_shape", "channel chunk overflows"))?;
    if channel_end > input.cols {
        return Err(ecapa_error(
            "res2_shape",
            "channel chunk exceeds input dimensions",
        ));
    }
    let mut output = zeroed_matrix(input.rows, chunk_channels, scratch_meter)?;
    for row in 0..input.rows {
        output.data[row * chunk_channels..(row + 1) * chunk_channels]
            .copy_from_slice(&input.row(row)[channel_start..channel_end]);
    }
    Ok(output)
}

fn copy_channel_chunk(
    input: &Mat,
    chunk: usize,
    chunk_channels: usize,
    output: &mut Mat,
    scratch_meter: &ScratchMeter,
) -> FwResult<()> {
    let input_chunk = extract_channel_chunk(input, chunk, chunk_channels, scratch_meter)?;
    write_channel_chunk(&input_chunk, chunk, chunk_channels, output)
}

fn write_channel_chunk(
    input: &Mat,
    chunk: usize,
    chunk_channels: usize,
    output: &mut Mat,
) -> FwResult<()> {
    let channel_start = checked_product(chunk, chunk_channels)?;
    let channel_end = channel_start
        .checked_add(chunk_channels)
        .ok_or_else(|| ecapa_error("res2_shape", "channel chunk overflows"))?;
    if input.rows != output.rows || input.cols != chunk_channels || channel_end > output.cols {
        return Err(ecapa_error(
            "res2_shape",
            "channel chunk write dimensions are invalid",
        ));
    }
    for row in 0..output.rows {
        output.data[row * output.cols + channel_start..row * output.cols + channel_end]
            .copy_from_slice(input.row(row));
    }
    Ok(())
}

fn add_in_place(left: &mut Mat, right: &Mat) -> FwResult<()> {
    if left.rows != right.rows || left.cols != right.cols || left.data.len() != right.data.len() {
        return Err(ecapa_error(
            "res2_shape",
            "recursive Res2Net add dimensions disagree",
        ));
    }
    for (left, right) in left.data.iter_mut().zip(&right.data) {
        *left += *right;
    }
    Ok(())
}

fn matrix_from_slice(
    rows: usize,
    cols: usize,
    source: &[f32],
    scratch_meter: &ScratchMeter,
) -> FwResult<Accounted<Mat>> {
    let len = checked_product(rows, cols)?;
    if source.len() != len {
        return Err(ecapa_error(
            "input_shape",
            "matrix source does not match its declared shape",
        ));
    }
    let lease = scratch_meter.reserve_f32(len)?;
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| ecapa_error("inference_resource", "matrix allocation was refused"))?;
    data.extend_from_slice(source);
    Ok(Accounted::new(Mat::from_vec(rows, cols, data), lease))
}

fn zeroed_matrix(
    rows: usize,
    cols: usize,
    scratch_meter: &ScratchMeter,
) -> FwResult<Accounted<Mat>> {
    let len = checked_product(rows, cols)?;
    accounted_zeroed_f32(len, scratch_meter)
        .map(|values| values.map(|data| Mat::from_vec(rows, cols, data)))
}

fn accounted_zeroed_f32(len: usize, scratch_meter: &ScratchMeter) -> FwResult<Accounted<Vec<f32>>> {
    let lease = scratch_meter.reserve_f32(len)?;
    zeroed_f32(len).map(|values| Accounted::new(values, lease))
}

fn zeroed_f32(len: usize) -> FwResult<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| ecapa_error("inference_resource", "buffer allocation was refused"))?;
    values.resize(len, 0.0);
    Ok(values)
}

fn validate_matrix(matrix: &Mat, rows: usize, cols: usize, stage: &str) -> FwResult<()> {
    let expected = checked_product(rows, cols)?;
    if matrix.rows != rows || matrix.cols != cols || matrix.data.len() != expected {
        return Err(ecapa_error(
            "matrix_shape",
            &format!("{stage} matrix dimensions are invalid"),
        ));
    }
    Ok(())
}

fn ensure_finite(
    matrix: &Mat,
    stage: &str,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    if matrix.rows == 0 || matrix.cols == 0 {
        return Err(ecapa_error(
            "matrix_shape",
            &format!("{stage} matrix dimensions are empty"),
        ));
    }
    validate_matrix(matrix, matrix.rows, matrix.cols, stage)?;
    let band_elements = checked_product(MATMUL_ROW_CHUNK_FRAMES, matrix.cols)?;
    for band in matrix.data.chunks(band_elements) {
        checkpoint()?;
        if band.iter().any(|value| !value.is_finite()) {
            return Err(ecapa_error(
                "numerical_value",
                &format!("{stage} produced a non-finite value"),
            ));
        }
    }
    Ok(())
}

fn ecapa_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("ecapa.{code}: {message}"))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::ecapa_conformance::{
        ECAPA_FULL_ORACLE_F32_ELEMENTS, ECAPA_FULL_ORACLE_FRONTEND_F32_ELEMENTS,
        ECAPA_FULL_ORACLE_NEURAL_F32_ELEMENTS, EcapaConformanceStage, EcapaFullOracleTensorSpec,
        compare_ecapa_values, ecapa_analytic_fixture, ecapa_frontend_conformance,
        ecapa_reference_f32_sha256, expected_ecapa_full_oracle_tensors,
        load_verified_ecapa_full_oracle,
    };

    fn no_cancel() -> FwResult<()> {
        Ok(())
    }

    fn test_scratch_meter() -> ScratchMeter {
        ScratchMeter::new(ECAPA_DEFAULT_MAXIMUM_PEAK_BUFFER_BYTES).expect("scratch meter")
    }

    fn identity_tdnn(channels: usize) -> TdnnLayer {
        let mut weights = vec![0.0; channels * channels];
        for channel in 0..channels {
            weights[channel * channels + channel] = 1.0;
        }
        TdnnLayer {
            convolution: AffineConv1d::from_natural(
                weights,
                vec![0.0; channels],
                channels,
                channels,
                1,
                1,
                &no_cancel,
            )
            .expect("identity convolution"),
            normalization: BatchNormAffine {
                scale: vec![1.0; channels],
                shift: vec![0.0; channels],
            },
        }
    }

    #[test]
    fn scratch_plan_is_exact_bounded_and_checked() {
        let minimum = plan_ecapa_scratch(ECAPA_MINIMUM_INFERENCE_FRAMES).expect("minimum plan");
        assert_eq!(minimum.row_chunk_frames, MATMUL_ROW_CHUNK_FRAMES);
        assert_eq!(minimum.input_feature_bytes, 16_320);
        assert_eq!(minimum.owned_peak_buffer_bytes, 1_873_408);
        assert_eq!(
            minimum.kernel_scratch_reserve_bytes,
            ECAPA_CONSERVATIVE_KERNEL_SCRATCH_BYTES
        );
        assert_eq!(minimum.estimated_peak_buffer_bytes, 10_262_016);

        let maximum = plan_ecapa_scratch(ECAPA_MAXIMUM_INFERENCE_FRAMES).expect("maximum plan");
        assert_eq!(maximum.input_feature_bytes, 96_320);
        assert_eq!(maximum.owned_peak_buffer_bytes, 7_944_704);
        assert_eq!(maximum.estimated_peak_buffer_bytes, 16_333_312);
        assert!(maximum.estimated_peak_buffer_bytes < ECAPA_DEFAULT_MAXIMUM_PEAK_BUFFER_BYTES);

        assert!(plan_ecapa_scratch(0).is_err());
        assert!(plan_ecapa_scratch(ECAPA_MINIMUM_INFERENCE_FRAMES - 1).is_err());
        assert!(plan_ecapa_scratch(ECAPA_MAXIMUM_INFERENCE_FRAMES + 1).is_err());
        assert!(checked_product(usize::MAX, 2).is_err());
    }

    #[test]
    fn logical_scratch_meter_enforces_live_bytes_and_releases_with_raii() {
        let meter = ScratchMeter::new(16).expect("meter");
        let first = accounted_zeroed_f32(2, &meter).expect("first lease");
        assert_eq!(meter.current_bytes(), 8);
        assert_eq!(meter.peak_bytes(), 8);
        let second = accounted_zeroed_f32(2, &meter).expect("second lease");
        assert_eq!(meter.current_bytes(), 16);
        assert_eq!(meter.peak_bytes(), 16);
        assert!(accounted_zeroed_f32(1, &meter).is_err());
        drop(first);
        assert_eq!(meter.current_bytes(), 8);
        drop(second);
        assert_eq!(meter.current_bytes(), 0);
        assert_eq!(meter.peak_bytes(), 16);
    }

    #[test]
    fn input_validation_rejects_shapes_values_and_resource_overrun() {
        let frames = ECAPA_MINIMUM_INFERENCE_FRAMES;
        let mut features = vec![0.0; frames * ECAPA_MEL_BANDS];
        features[0] = -ECAPA_MAXIMUM_ABSOLUTE_INPUT_FEATURE;
        features[1] = ECAPA_MAXIMUM_ABSOLUTE_INPUT_FEATURE;
        assert!(
            validate_inference_request(&features, frames, EcapaInferenceConfig::default()).is_ok()
        );

        assert!(
            validate_inference_request(
                &features[..features.len() - 1],
                frames,
                EcapaInferenceConfig::default(),
            )
            .is_err()
        );
        features[2] = f32::NAN;
        assert!(
            validate_inference_request(&features, frames, EcapaInferenceConfig::default()).is_err()
        );
        features[2] = f32::INFINITY;
        assert!(
            validate_inference_request(&features, frames, EcapaInferenceConfig::default()).is_err()
        );
        features[2] = ECAPA_MAXIMUM_ABSOLUTE_INPUT_FEATURE + 0.01;
        assert!(
            validate_inference_request(&features, frames, EcapaInferenceConfig::default()).is_err()
        );

        let maximum_features = vec![0.0; ECAPA_MAXIMUM_INFERENCE_FRAMES * ECAPA_MEL_BANDS];
        let plan = plan_ecapa_scratch(ECAPA_MAXIMUM_INFERENCE_FRAMES).expect("plan");
        let error = validate_inference_request(
            &maximum_features,
            ECAPA_MAXIMUM_INFERENCE_FRAMES,
            EcapaInferenceConfig {
                maximum_peak_buffer_bytes: plan.estimated_peak_buffer_bytes - 1,
            },
        )
        .expect_err("resource cap must fail");
        assert!(error.to_string().contains("ecapa.inference_resource"));
    }

    #[test]
    fn reflection_excludes_edges_and_dilated_conv_uses_cross_correlation() {
        let expected_indices = [2, 1, 0, 1, 2, 3, 2, 1];
        for (padded, expected) in expected_indices.into_iter().enumerate() {
            assert_eq!(reflect_padded_index(padded, 2, 4).expect("index"), expected);
        }
        let wide_expected = [4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0];
        for (padded, expected) in wide_expected.into_iter().enumerate() {
            assert_eq!(reflect_padded_index(padded, 4, 5).expect("index"), expected);
        }

        let convolution =
            AffineConv1d::from_natural(vec![1.0, 10.0, 100.0], vec![0.0], 1, 1, 3, 2, &no_cancel)
                .expect("convolution");
        let input = Mat::from_vec(5, 1, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let meter = test_scratch_meter();
        let output = convolution
            .forward(&input, &no_cancel, &meter)
            .expect("forward");
        assert_eq!(output.data, vec![202.0, 311.0, 420.0, 331.0, 242.0]);
    }

    #[test]
    fn tdnn_applies_relu_before_batch_norm_even_for_negative_gamma() {
        let convolution = AffineConv1d::from_natural(vec![1.0], vec![0.0], 1, 1, 1, 1, &no_cancel)
            .expect("convolution");
        let normalization =
            BatchNormAffine::from_source(vec![-2.0], vec![1.0], vec![0.0], vec![1.0])
                .expect("batch norm");
        let expected_scale = normalization.scale[0];
        let tdnn = TdnnLayer {
            convolution,
            normalization,
        };
        let input = Mat::from_vec(2, 1, vec![-2.0, 3.0]);
        let meter = test_scratch_meter();
        let output = tdnn.forward(&input, &no_cancel, &meter).expect("forward");
        assert_eq!(output.data[0], 1.0);
        assert!((output.data[1] - (3.0 * expected_scale + 1.0)).abs() < 1.0e-6);
        assert!(output.data[1] < 0.0);
    }

    #[test]
    fn res2_chunks_accumulate_the_previous_output_recursively() {
        let blocks = (0..RES2_SCALE - 1)
            .map(|_| identity_tdnn(1))
            .collect::<Vec<_>>();
        let input = Mat::from_vec(1, RES2_SCALE, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let meter = test_scratch_meter();
        let output = res2_forward(&input, &blocks, &no_cancel, &meter).expect("res2");
        assert_eq!(
            output.data,
            vec![1.0, 2.0, 5.0, 9.0, 14.0, 20.0, 27.0, 35.0]
        );
    }

    #[test]
    fn se_gate_is_broadcast_before_outer_residual_add() {
        let mut output = Mat::from_vec(2, 2, vec![2.0, 4.0, 6.0, 8.0]);
        let residual = Mat::from_vec(2, 2, vec![1.0, 10.0, 100.0, 1_000.0]);
        apply_se_gate_and_residual(&mut output, &[0.5, 0.25], &residual, &no_cancel).expect("gate");
        assert_eq!(output.data, vec![2.0, 11.0, 103.0, 1_002.0]);
    }

    #[test]
    fn chunked_frankentorch_affine_matches_scalar_and_is_deterministic() {
        let rows = 35;
        let input = Mat::from_vec(
            rows,
            3,
            (0..rows * 3)
                .map(|index| index as f32 * 0.03125 - 1.0)
                .collect(),
        );
        let weight = Mat::from_vec(3, 2, vec![0.5, -0.25, 2.0, 1.0, -1.5, 0.75]);
        let bias = [0.125, -0.5];
        let meter = test_scratch_meter();
        let first = chunked_matmul_bias(&input, &weight, &bias, &no_cancel, &meter).expect("first");
        let second =
            chunked_matmul_bias(&input, &weight, &bias, &no_cancel, &meter).expect("second");
        assert_eq!(first.value, second.value);
        for row in 0..rows {
            for (output, output_bias) in bias.iter().copied().enumerate() {
                let mut expected = output_bias;
                for input_channel in 0..3 {
                    expected += input.data[row * 3 + input_channel]
                        * weight.data[input_channel * 2 + output];
                }
                assert!((first.data[row * 2 + output] - expected).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn temporal_attention_is_channelwise_normalized_and_cancel_aware() {
        let meter = test_scratch_meter();
        let mut logits = zeroed_matrix(3, MFA_CHANNELS, &meter).expect("logits");
        for row in 0..3 {
            for channel in 0..MFA_CHANNELS {
                logits.data[row * MFA_CHANNELS + channel] = row as f32 - channel as f32 * 0.001;
            }
        }
        let maximum_error = softmax_over_time(&mut logits, &no_cancel).expect("softmax");
        assert!(maximum_error <= 2.0e-6);
        for channel in [0, 17, MFA_CHANNELS - 1] {
            let sum = (0..3)
                .map(|row| logits.data[row * MFA_CHANNELS + channel])
                .sum::<f32>();
            assert!((sum - 1.0).abs() <= 2.0e-6);
        }

        let calls = AtomicUsize::new(0);
        let cancel = || {
            if calls.fetch_add(1, Ordering::SeqCst) >= 1 {
                Err(FwError::Cancelled("test".to_owned()))
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            softmax_over_time(&mut logits, &cancel),
            Err(FwError::Cancelled(_))
        ));
    }

    #[test]
    fn pooling_clamps_variance_and_rejects_malformed_shapes() {
        let meter = test_scratch_meter();
        let input = zeroed_matrix(2, MFA_CHANNELS, &meter).expect("input");
        let attention = Mat::from_vec(2, MFA_CHANNELS, vec![0.5; 2 * MFA_CHANNELS]);
        let pooled = weighted_statistics(&input, &attention, &no_cancel, &meter).expect("pool");
        assert!(
            pooled.data[..MFA_CHANNELS]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            pooled.data[MFA_CHANNELS..]
                .iter()
                .all(|value| (*value - 1.0e-6).abs() < 1.0e-9)
        );
        let malformed = Mat::from_vec(1, MFA_CHANNELS, vec![1.0; MFA_CHANNELS]);
        assert!(weighted_statistics(&input, &malformed, &no_cancel, &meter).is_err());

        let mut extreme_input = zeroed_matrix(2, MFA_CHANNELS, &meter).expect("extreme input");
        extreme_input.data[MFA_CHANNELS] = f32::MAX;
        let mut one_hot_attention =
            zeroed_matrix(2, MFA_CHANNELS, &meter).expect("one-hot attention");
        one_hot_attention.data[..MFA_CHANNELS].fill(1.0);
        assert!(
            weighted_statistics(&extreme_input, &one_hot_attention, &no_cancel, &meter)
                .expect_err("zero times infinity must not be hidden by variance clamping")
                .to_string()
                .contains("ecapa.pooling_value")
        );
    }

    #[test]
    fn malformed_batch_norm_and_mid_chunk_cancellation_fail_closed() {
        assert!(
            BatchNormAffine::from_source(vec![1.0], vec![0.0], vec![0.0], vec![-1.0],).is_err()
        );

        let normalization = BatchNormAffine {
            scale: vec![1.0],
            shift: vec![0.0],
        };
        let mut non_finite = Mat::from_vec(1, 1, vec![f32::NAN]);
        assert!(
            normalization
                .apply(&mut non_finite, true, &no_cancel)
                .expect_err("ReLU must not suppress NaN")
                .to_string()
                .contains("ecapa.batch_norm_value")
        );

        let input = Mat::from_vec(65, 1, vec![1.0; 65]);
        let weight = Mat::from_vec(1, 1, vec![1.0]);
        let calls = AtomicUsize::new(0);
        let cancel = || {
            if calls.fetch_add(1, Ordering::SeqCst) >= 2 {
                Err(FwError::Cancelled("test".to_owned()))
            } else {
                Ok(())
            }
        };
        let meter = test_scratch_meter();
        assert!(matches!(
            chunked_matmul_bias(&input, &weight, &[0.0], &cancel, &meter),
            Err(FwError::Cancelled(_))
        ));
    }

    #[test]
    fn typed_weight_finiteness_scan_observes_chunked_cancellation() {
        let value_count = WEIGHT_FINITE_SCAN_CHUNK_VALUES + 1;
        let payload_bytes = value_count * std::mem::size_of::<f32>();
        let header = serde_json::json!({
            "large": {
                "dtype": "F32",
                "shape": [value_count],
                "data_offsets": [0, payload_bytes]
            }
        });
        let header_bytes = serde_json::to_vec(&header).expect("serialize safetensors header");
        let mut package_bytes = Vec::new();
        package_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        package_bytes.extend_from_slice(&header_bytes);
        package_bytes.resize(package_bytes.len() + payload_bytes, 0);
        let package =
            SafetensorsFile::from_owned_bytes(package_bytes).expect("parse finite package");

        let calls = AtomicUsize::new(0);
        let cancel = || {
            if calls.fetch_add(1, Ordering::SeqCst) >= 2 {
                Err(FwError::Cancelled("test".to_owned()))
            } else {
                Ok(())
            }
        };
        let mut loader = WeightLoader::new(&package, &cancel);
        assert!(matches!(
            loader.tensor("large".to_owned(), &[value_count]),
            Err(FwError::Cancelled(_))
        ));
    }

    #[test]
    fn bounded_row_passes_observe_mid_operation_cancellation() {
        let normalization = BatchNormAffine {
            scale: vec![1.0],
            shift: vec![0.0],
        };
        let mut matrix = Mat::from_vec(65, 1, vec![1.0; 65]);
        let bn_calls = AtomicUsize::new(0);
        let cancel_bn = || {
            if bn_calls.fetch_add(1, Ordering::SeqCst) >= 1 {
                Err(FwError::Cancelled("test".to_owned()))
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            normalization.apply(&mut matrix, false, &cancel_bn),
            Err(FwError::Cancelled(_))
        ));

        let meter = test_scratch_meter();
        let activation = zeroed_matrix(65, NETWORK_CHANNELS, &meter).expect("activation");
        let concat_calls = AtomicUsize::new(0);
        let cancel_concat = || {
            if concat_calls.fetch_add(1, Ordering::SeqCst) >= 1 {
                Err(FwError::Cancelled("test".to_owned()))
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            concatenate_mfa(
                &activation,
                &activation,
                &activation,
                &cancel_concat,
                &meter,
            ),
            Err(FwError::Cancelled(_))
        ));
    }

    #[test]
    fn trace_preserves_fixed_cancellation_stage_and_round_trips() {
        let mut trace = EcapaInferenceTrace::default();
        trace.begin(51);
        trace.scratch_plan = Some(plan_ecapa_scratch(51).expect("scratch plan"));
        let cancel = || Err(FwError::Cancelled("private detail".to_owned()));
        let error = run_stage(
            &mut trace,
            EcapaInferenceStage::InputValidation,
            &cancel,
            |_| Ok(()),
        )
        .expect_err("cancel");
        assert!(error.to_string().contains("stage=input_validation"));
        assert_eq!(
            trace.cancellation_stage,
            Some(EcapaInferenceStage::InputValidation)
        );
        assert_eq!(trace.fallback_reason, Some(EcapaFallbackReason::Cancelled));
        let encoded = serde_json::to_vec(&trace).expect("encode");
        let decoded: EcapaInferenceTrace = serde_json::from_slice(&encoded).expect("decode");
        assert_eq!(decoded, trace);
        assert!(trace.stages.len() <= MAXIMUM_RECORDED_STAGES);
        let json = String::from_utf8(encoded).expect("utf8");
        assert!(!json.contains("private detail"));

        let mut wrong_schema = serde_json::to_value(&trace).expect("trace value");
        wrong_schema["schema_version"] = serde_json::json!("wrong");
        assert!(serde_json::from_value::<EcapaInferenceTrace>(wrong_schema).is_err());

        let mut wrong_plan = serde_json::to_value(&trace).expect("trace value");
        wrong_plan["scratch_plan"]["estimated_peak_buffer_bytes"] = serde_json::json!(1);
        assert!(serde_json::from_value::<EcapaInferenceTrace>(wrong_plan).is_err());

        let mut invalid_for_serialization = trace.clone();
        invalid_for_serialization
            .scratch_plan
            .as_mut()
            .expect("scratch plan")
            .estimated_peak_buffer_bytes = 1;
        assert!(serde_json::to_value(&invalid_for_serialization).is_err());

        let mut unknown_field = serde_json::to_value(&trace).expect("trace value");
        unknown_field["private_payload"] = serde_json::json!("must be rejected");
        assert!(serde_json::from_value::<EcapaInferenceTrace>(unknown_field).is_err());

        let mut too_many_stages = serde_json::to_value(&trace).expect("trace value");
        too_many_stages["stages"] = serde_json::Value::Array(
            (0..=MAXIMUM_RECORDED_STAGES)
                .map(|_| {
                    serde_json::json!({
                        "stage": "input_validation",
                        "elapsed_micros": 0
                    })
                })
                .collect(),
        );
        assert!(serde_json::from_value::<EcapaInferenceTrace>(too_many_stages).is_err());

        let mut incomplete_success = EcapaInferenceTrace::default();
        incomplete_success.begin(51);
        incomplete_success.scratch_plan = Some(plan_ecapa_scratch(51).expect("scratch plan"));
        incomplete_success.stages = ECAPA_STAGE_ORDER
            .iter()
            .map(|stage| EcapaStageTiming {
                stage: *stage,
                elapsed_micros: 0,
            })
            .collect();
        incomplete_success.last_stage = Some(EcapaInferenceStage::ProjectionAndNormalization);
        assert!(incomplete_success.validate_serialized_contract().is_err());

        let mut fabricated_empty_success = EcapaInferenceTrace::default();
        fabricated_empty_success.stages.push(EcapaStageTiming {
            stage: EcapaInferenceStage::InputValidation,
            elapsed_micros: 0,
        });
        fabricated_empty_success.last_stage = Some(EcapaInferenceStage::InputValidation);
        assert!(
            fabricated_empty_success
                .validate_serialized_contract()
                .is_err()
        );
    }

    #[test]
    fn trace_classifies_and_redacts_non_cancellation_checkpoint_failures() {
        let calls = AtomicUsize::new(0);
        let fail_inside_stage = || {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(())
            } else {
                Err(FwError::InvalidRequest(
                    "caller-private checkpoint detail".to_owned(),
                ))
            }
        };
        let mut trace = EcapaInferenceTrace::default();
        trace.begin(ECAPA_MINIMUM_INFERENCE_FRAMES);
        let error = run_stage(
            &mut trace,
            EcapaInferenceStage::InitialTdnn,
            &fail_inside_stage,
            |stage_checkpoint| stage_checkpoint(),
        )
        .expect_err("checkpoint failure");
        assert_eq!(
            trace.fallback_reason,
            Some(EcapaFallbackReason::CheckpointFailure)
        );
        assert!(error.to_string().contains("ecapa.checkpoint_failure"));
        assert!(!error.to_string().contains("caller-private"));

        let mut internal_trace = EcapaInferenceTrace::default();
        internal_trace.begin(ECAPA_MINIMUM_INFERENCE_FRAMES);
        let error = run_stage(
            &mut internal_trace,
            EcapaInferenceStage::MultiFeatureAggregation,
            &no_cancel,
            |_| {
                Err::<(), _>(ecapa_error(
                    "kernel_failure",
                    "CPU matmul rejected a frozen ECAPA shape",
                ))
            },
        )
        .expect_err("internal kernel failure");
        assert_eq!(
            internal_trace.fallback_reason,
            Some(EcapaFallbackReason::InternalContractFailure)
        );
        assert!(error.to_string().contains("ecapa.kernel_failure"));
    }

    #[test]
    fn cpu_kernel_errors_are_namespaced_and_content_free() {
        let rhs = Mat::from_vec(3, 2, vec![0.0; 6]);
        let meter = test_scratch_meter();
        let error = cpu_matmul_raw_lhs(&[1.0, 2.0], 1, &rhs, &meter)
            .expect_err("malformed kernel input must fail");
        let rendered = error.to_string();
        assert!(rendered.contains("ecapa.kernel_failure"));
        assert!(!rendered.contains("matmul_raw_lhs_cpu"));
        assert!(!rendered.contains("ft-kernel-cpu"));
    }

    #[test]
    fn reference_hash_rule_and_redacted_debug_are_content_safe() {
        let fixture = ecapa_analytic_fixture();
        assert_eq!(
            ecapa_reference_f32_sha256(&fixture, &[fixture.len()]).expect("hash"),
            "acc240c07370020bbd1b3aaf9b8b81be43ef053b8da950969e86f62b6f1dba2f"
        );
        assert!(ecapa_reference_f32_sha256(&[f32::NAN], &[1]).is_err());
        assert!(ecapa_reference_f32_sha256(&[0.0], &[2]).is_err());

        let mut values = [0.0; ECAPA_EMBEDDING_DIMENSIONS];
        values[0] = 12_345.625;
        let embedding = EcapaEmbedding(values);
        let debug = format!("{embedding:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("12345.625"));
    }

    #[test]
    fn module_forbids_unsafe_code_at_compile_time() {
        let source = include_str!("ecapa_inference.rs");
        assert!(source.contains("#![forbid(unsafe_code)]"));
    }

    #[test]
    #[ignore = "requires externally converted public ECAPA weights"]
    fn external_package_runtime_pcm_smoke() {
        let weight_path = std::env::var_os("FRANKEN_WHISPER_ECAPA_TEST_WEIGHTS")
            .map(std::path::PathBuf::from)
            .expect("set FRANKEN_WHISPER_ECAPA_TEST_WEIGHTS");
        let model = EcapaModel::load(&weight_path).expect("load model");
        let mut trace = EcapaInferenceTrace::default();
        let output = model
            .infer_pcm_with_checkpoint(
                &ecapa_analytic_fixture(),
                EcapaInferenceConfig::default(),
                &no_cancel,
                &mut trace,
            )
            .expect("runtime frontend and native ECAPA inference");
        assert_eq!(
            output.embedding.as_slice().len(),
            ECAPA_EMBEDDING_DIMENSIONS
        );
        let norm = output
            .embedding
            .as_slice()
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1.0e-5, "embedding norm={norm}");
        assert_eq!(trace.frame_count, 101);
        assert_eq!(
            trace.last_stage,
            Some(EcapaInferenceStage::ProjectionAndNormalization)
        );
    }

    #[test]
    #[ignore = "requires externally generated public ECAPA weights and full oracle"]
    fn external_package_forward_matches_every_frozen_oracle_value() {
        let weight_path = std::env::var_os("FRANKEN_WHISPER_ECAPA_TEST_WEIGHTS")
            .map(std::path::PathBuf::from)
            .expect("set FRANKEN_WHISPER_ECAPA_TEST_WEIGHTS");
        let oracle_path = std::env::var_os("FRANKEN_WHISPER_ECAPA_TEST_ORACLE")
            .map(std::path::PathBuf::from)
            .expect("set FRANKEN_WHISPER_ECAPA_TEST_ORACLE");
        let model = EcapaModel::load(&weight_path).expect("load model");
        let oracle = load_verified_ecapa_full_oracle(&oracle_path).expect("load full oracle");
        assert_eq!(
            model.info().resident_weight_bytes,
            ECAPA_MODEL_RESIDENT_BYTES
        );
        assert_eq!(
            model.info().conservative_load_accounted_payload_bytes,
            ECAPA_CONSERVATIVE_LOAD_ACCOUNTED_PAYLOAD_BYTES
        );
        let oracle_specs = expected_ecapa_full_oracle_tensors();
        let frontend = ecapa_frontend_conformance(&ecapa_analytic_fixture()).expect("frontend");

        let frontend_observed = [
            frontend.log_fbank_db.as_slice(),
            frontend.sentence_mean_normalized.as_slice(),
        ];
        let mut compared_values = 0u64;
        for (spec, observed) in oracle_specs.iter().take(2).zip(frontend_observed) {
            let (shape, expected) = oracle.tensor_f32(&spec.name).expect("frontend oracle");
            assert_eq!(shape, spec.shape);
            let comparison = compare_ecapa_values(spec.stage, &expected, observed)
                .expect("compare complete frontend stage");
            assert!(
                comparison.passes,
                "{:?} compared={} max_abs={} max_rel={}",
                spec.stage,
                comparison.compared_values,
                comparison.maximum_absolute_error,
                comparison.maximum_relative_error
            );
            compared_values = compared_values
                .checked_add(
                    u64::try_from(comparison.compared_values).expect("comparison count fits u64"),
                )
                .expect("comparison count remains bounded");
        }
        assert_eq!(compared_values, ECAPA_FULL_ORACLE_FRONTEND_F32_ELEMENTS);

        let normalized_spec = &oracle_specs[1];
        let (normalized_shape, normalized) = oracle
            .tensor_f32(&normalized_spec.name)
            .expect("normalized frontend oracle");
        assert_eq!(normalized_shape, normalized_spec.shape);
        let frame_count = normalized_shape[1];
        assert_eq!(normalized_shape, vec![1, frame_count, ECAPA_MEL_BANDS]);
        let mut isolated_trace = EcapaInferenceTrace::default();
        let (isolated, isolated_reference) = model
            .infer_internal(
                &normalized,
                frame_count,
                EcapaInferenceConfig::default(),
                &no_cancel,
                &mut isolated_trace,
                true,
            )
            .expect("isolated neural forward");
        let isolated_reference = isolated_reference.expect("isolated reference stages");
        let isolated_neural_values = compare_neural_oracle_stages(
            &oracle,
            &oracle_specs,
            &isolated_reference,
            "oracle-normalized",
        );
        assert_eq!(
            isolated_neural_values,
            ECAPA_FULL_ORACLE_NEURAL_F32_ELEMENTS
        );
        compared_values = compared_values
            .checked_add(isolated_neural_values)
            .expect("full comparison count remains bounded");
        assert_eq!(compared_values, ECAPA_FULL_ORACLE_F32_ELEMENTS);
        assert_trace_scratch_is_observed_and_bounded(&isolated_trace);
        assert!(
            isolated_trace
                .maximum_attention_sum_error
                .is_some_and(|error| error <= 2.0e-6)
        );
        let isolated_norm = isolated
            .embedding
            .as_slice()
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((isolated_norm - 1.0).abs() < 1.0e-5);

        let isolated_repeat = model
            .infer(&normalized, frame_count, EcapaInferenceConfig::default())
            .expect("isolated repeat");
        assert_eq!(isolated.embedding, isolated_repeat.embedding);

        let mut composed_trace = EcapaInferenceTrace::default();
        let (composed, composed_reference) = model
            .infer_internal(
                &frontend.sentence_mean_normalized,
                frontend.frame_count,
                EcapaInferenceConfig::default(),
                &no_cancel,
                &mut composed_trace,
                true,
            )
            .expect("composed native frontend and neural forward");
        let composed_reference = composed_reference.expect("composed reference stages");
        assert_eq!(
            compare_neural_oracle_stages(
                &oracle,
                &oracle_specs,
                &composed_reference,
                "native-frontend-composed",
            ),
            ECAPA_FULL_ORACLE_NEURAL_F32_ELEMENTS
        );
        assert_trace_scratch_is_observed_and_bounded(&composed_trace);
        assert!(
            composed_trace
                .maximum_attention_sum_error
                .is_some_and(|error| error <= 2.0e-6)
        );
        let composed_repeat = model
            .infer_frontend(&frontend, EcapaInferenceConfig::default())
            .expect("composed repeat");
        assert_eq!(composed.embedding, composed_repeat.embedding);

        let maximum_features = vec![0.0; ECAPA_MAXIMUM_INFERENCE_FRAMES * ECAPA_MEL_BANDS];
        let maximum = model
            .infer(
                &maximum_features,
                ECAPA_MAXIMUM_INFERENCE_FRAMES,
                EcapaInferenceConfig::default(),
            )
            .expect("maximum-frame scratch proof");
        assert_trace_scratch_is_observed_and_bounded(&maximum.diagnostics);
    }

    fn compare_neural_oracle_stages(
        oracle: &SafetensorsFile,
        oracle_specs: &[EcapaFullOracleTensorSpec],
        reference: &InternalReferenceTrace,
        proof_path: &str,
    ) -> u64 {
        let mut compared_values = 0u64;
        for spec in oracle_specs.iter().skip(2) {
            let (shape, expected) = oracle.tensor_f32(&spec.name).expect("neural oracle");
            assert_eq!(shape, spec.shape);
            let observed = captured_neural_stage_values(reference, spec);
            let comparison = compare_ecapa_values(spec.stage, &expected, &observed)
                .expect("compare complete neural stage");
            assert!(
                comparison.passes,
                "path={} stage={:?} compared={} max_abs={} max_rel={}",
                proof_path,
                spec.stage,
                comparison.compared_values,
                comparison.maximum_absolute_error,
                comparison.maximum_relative_error
            );
            compared_values = compared_values
                .checked_add(
                    u64::try_from(comparison.compared_values).expect("comparison count fits u64"),
                )
                .expect("neural comparison count remains bounded");
        }
        compared_values
    }

    fn captured_neural_stage_values(
        reference: &InternalReferenceTrace,
        spec: &EcapaFullOracleTensorSpec,
    ) -> Vec<f32> {
        match spec.stage {
            EcapaConformanceStage::InitialTdnn => channel_first_values(
                reference
                    .initial_tdnn
                    .as_ref()
                    .expect("initial TDNN capture"),
                &spec.shape,
            ),
            EcapaConformanceStage::FirstSeRes2 => channel_first_values(
                reference
                    .first_se_res2
                    .as_ref()
                    .expect("first SE-Res2 capture"),
                &spec.shape,
            ),
            EcapaConformanceStage::MultiFeatureAggregation => channel_first_values(
                reference
                    .multi_feature_aggregation
                    .as_ref()
                    .expect("MFA capture"),
                &spec.shape,
            ),
            EcapaConformanceStage::AttentivePooling => {
                let pooling = reference
                    .attentive_pooling
                    .as_ref()
                    .expect("pooling capture");
                assert_eq!(spec.shape, vec![1, pooling.cols, pooling.rows]);
                pooling.data.clone()
            }
            EcapaConformanceStage::Embedding => {
                let embedding = reference
                    .raw_embedding
                    .as_ref()
                    .expect("raw embedding capture");
                assert_eq!(spec.shape, vec![1, 1, embedding.len()]);
                embedding.to_vec()
            }
            _ => unreachable!("frontend stages were compared separately"),
        }
    }

    fn assert_trace_scratch_is_observed_and_bounded(trace: &EcapaInferenceTrace) {
        let plan = trace.scratch_plan.as_ref().expect("scratch plan");
        let observed = trace
            .observed_owned_peak_buffer_bytes
            .expect("observed logical scratch peak");
        assert!(observed > 0);
        assert!(observed <= plan.owned_peak_buffer_bytes);
        assert!(trace.validate_serialized_contract().is_ok());
    }

    fn channel_first_values(matrix: &Mat, expected_shape: &[usize]) -> Vec<f32> {
        assert_eq!(expected_shape, &[1, matrix.cols, matrix.rows]);
        let mut values = Vec::with_capacity(matrix.data.len());
        for channel in 0..matrix.cols {
            for frame in 0..matrix.rows {
                values.push(matrix.data[frame * matrix.cols + channel]);
            }
        }
        values
    }
}
