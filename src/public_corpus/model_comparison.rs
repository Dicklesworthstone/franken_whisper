//! Aggregate-only, same-invocation learned-versus-native diarization comparison.
//!
//! The retained contract deliberately contains no recording identifiers, paths,
//! turns, labels, embeddings, logits, transcript text, or raw error strings.
//! Per-recording hypotheses and authoritative scores exist only long enough to
//! update these aggregate sufficient statistics and ordered digests.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::diarization::{
    AcousticBoundaryHints, AcousticDiarizationInput, AuthoritativeDiarizationScore,
    DIARIZATION_HYPOTHESIS_SCHEMA_VERSION, DIARIZATION_SCORER_VERSION,
    DiarizationHypothesisDocument, DiarizationScorerConfig, EvaluationOverlapPolicy,
    EvaluationSplit, EvaluationTurn, score_diarization_documents,
};
use crate::differential_oracle::{
    DifferentialAuthority, DifferentialExecutionStage, DifferentialOracleFamily,
    DifferentialOracleTool, DifferentialSkipReason, SORTFORMER_ORACLE_ADAPTER_SHA256,
    SORTFORMER_ORACLE_ADAPTER_VERSION, SORTFORMER_ORACLE_CONTRACT_SHA256,
    SORTFORMER_ORACLE_MAX_SPEAKERS, SORTFORMER_ORACLE_OUTPUT_FRAME_MS,
    SORTFORMER_ORACLE_TOOL_VERSION, SortformerObservationOutcome, SortformerObservationProvenance,
    SortformerObservationRequest, run_sortformer_observation_with_cancel,
    sortformer_oracle_contract,
};
use crate::ecapa_conformance::{
    ECAPA_CONTRACT_SHA256, ECAPA_PACKAGE_FILENAME, ECAPA_PACKAGE_SHA256,
};
use crate::ecapa_inference::{EcapaFallbackReason, EcapaModel, classify_ecapa_fallback_reason};
use crate::error::{FwError, FwResult};
use crate::model::{
    DiarizationEngine, DiarizationFallbackPolicy, DiarizationRequest, SpeakerCountEstimate,
    SpeakerCountRequest,
};
use crate::model_distribution::resolve_cached_sortformer_with_cancel;
use crate::orchestrator::CancellationToken;
use crate::sortformer_conformance::load_verified_sortformer_package_with_checkpoint;
use crate::sortformer_inference::{
    SORTFORMER_SAMPLE_RATE_HZ, SORTFORMER_SPEAKER_LANES, SortformerPcm, SortformerSession,
    SortformerSpeakerTurn,
};

pub const PUBLIC_MODEL_COMPARISON_SCHEMA_VERSION: &str = "public-diarization-model-comparison-v5";
pub const PUBLIC_MODEL_COMPARISON_RUNNER_VERSION: &str =
    "public-diarization-model-comparison-runner-v5";
pub const PUBLIC_MODEL_COMPARISON_PROTOCOL_VERSION: &str =
    "public-diarization-model-comparison-protocol-v5";
pub const PUBLIC_MODEL_COMPARISON_OUTCOME_TAXONOMY_VERSION: &str =
    "public-diarization-model-comparison-outcomes-v3";
pub const PUBLIC_MODEL_COMPARISON_PROTOCOL_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";
pub const PUBLIC_MODEL_COMPARISON_SCHEDULE_VERSION: &str = "five-lane-balanced-williams-v1";
pub const PUBLIC_MODEL_COMPARISON_ATTEMPT_TIMEOUT_SECONDS: u64 = 1_800;
const MODEL_COMPARISON_WORKER_SCHEMA_VERSION: &str = "public-model-comparison-worker-v1";
const MAX_MODEL_COMPARISON_WORKER_REQUEST_BYTES: u64 = 32 * 1024 * 1024;
const MAX_MODEL_COMPARISON_EXECUTABLE_BYTES: u64 = 1024 * 1024 * 1024;
const MODEL_COMPARISON_CANCEL_PROBE_DELAY_MS: u64 = 150;
const MODEL_COMPARISON_CANCEL_PROBE_TIMEOUT_SECONDS: u64 = 5;
const MODEL_COMPARISON_CANCEL_PROBE_SELF_LIMIT_SECONDS: u64 = 10;
const MODEL_COMPARISON_NATIVE_RTF_CAP_MILLIONTHS: u64 = 500_000;
const MODEL_COMPARISON_EXTERNAL_RTF_CAP_MILLIONTHS: u64 = 10_000_000;
const MODEL_COMPARISON_PEAK_RSS_CAP_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MODEL_COMPARISON_CANCELLATION_LATENCY_CAP_MS: u64 = 500;
const MODEL_COMPARISON_PROCESS_TREE_RSS_MINIMUM_SAMPLE_INTERVAL_MS: u64 = 50;
const MODEL_COMPARISON_PROCESS_TREE_RSS_PROBE_TIMEOUT_MS: u64 = 250;

/// Canonical comparison lanes. The order is part of the retained contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelComparisonLane {
    NativeAcoustic,
    NativeEcapa,
    NativeEcapaFused,
    NativeSortformer,
    ExternalSortformer,
}

impl ModelComparisonLane {
    pub const ALL: [Self; 5] = [
        Self::NativeAcoustic,
        Self::NativeEcapa,
        Self::NativeEcapaFused,
        Self::NativeSortformer,
        Self::ExternalSortformer,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeAcoustic => "native_acoustic",
            Self::NativeEcapa => "native_ecapa",
            Self::NativeEcapaFused => "native_ecapa_fused",
            Self::NativeSortformer => "native_sortformer",
            Self::ExternalSortformer => "external_sortformer",
        }
    }
}

/// Frozen balanced order used across consecutive sorted observations.
pub const MODEL_COMPARISON_WILLIAMS_SCHEDULE: [[ModelComparisonLane; 5]; 10] = [
    [
        ModelComparisonLane::NativeAcoustic,
        ModelComparisonLane::NativeEcapa,
        ModelComparisonLane::ExternalSortformer,
        ModelComparisonLane::NativeEcapaFused,
        ModelComparisonLane::NativeSortformer,
    ],
    [
        ModelComparisonLane::NativeEcapa,
        ModelComparisonLane::NativeEcapaFused,
        ModelComparisonLane::NativeAcoustic,
        ModelComparisonLane::NativeSortformer,
        ModelComparisonLane::ExternalSortformer,
    ],
    [
        ModelComparisonLane::NativeEcapaFused,
        ModelComparisonLane::NativeSortformer,
        ModelComparisonLane::NativeEcapa,
        ModelComparisonLane::ExternalSortformer,
        ModelComparisonLane::NativeAcoustic,
    ],
    [
        ModelComparisonLane::NativeSortformer,
        ModelComparisonLane::ExternalSortformer,
        ModelComparisonLane::NativeEcapaFused,
        ModelComparisonLane::NativeAcoustic,
        ModelComparisonLane::NativeEcapa,
    ],
    [
        ModelComparisonLane::ExternalSortformer,
        ModelComparisonLane::NativeAcoustic,
        ModelComparisonLane::NativeSortformer,
        ModelComparisonLane::NativeEcapa,
        ModelComparisonLane::NativeEcapaFused,
    ],
    [
        ModelComparisonLane::NativeSortformer,
        ModelComparisonLane::NativeEcapaFused,
        ModelComparisonLane::ExternalSortformer,
        ModelComparisonLane::NativeEcapa,
        ModelComparisonLane::NativeAcoustic,
    ],
    [
        ModelComparisonLane::ExternalSortformer,
        ModelComparisonLane::NativeSortformer,
        ModelComparisonLane::NativeAcoustic,
        ModelComparisonLane::NativeEcapaFused,
        ModelComparisonLane::NativeEcapa,
    ],
    [
        ModelComparisonLane::NativeAcoustic,
        ModelComparisonLane::ExternalSortformer,
        ModelComparisonLane::NativeEcapa,
        ModelComparisonLane::NativeSortformer,
        ModelComparisonLane::NativeEcapaFused,
    ],
    [
        ModelComparisonLane::NativeEcapa,
        ModelComparisonLane::NativeAcoustic,
        ModelComparisonLane::NativeEcapaFused,
        ModelComparisonLane::ExternalSortformer,
        ModelComparisonLane::NativeSortformer,
    ],
    [
        ModelComparisonLane::NativeEcapaFused,
        ModelComparisonLane::NativeEcapa,
        ModelComparisonLane::NativeSortformer,
        ModelComparisonLane::NativeAcoustic,
        ModelComparisonLane::ExternalSortformer,
    ],
];

#[must_use]
pub const fn model_comparison_schedule_row(observation_index: usize) -> [ModelComparisonLane; 5] {
    MODEL_COMPARISON_WILLIAMS_SCHEDULE[observation_index % MODEL_COMPARISON_WILLIAMS_SCHEDULE.len()]
}

/// Whether a declared lane produced an authoritative score for one observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelComparisonOutcomeStatus {
    Completed,
    Skipped,
    Failed,
}

impl ModelComparisonOutcomeStatus {
    #[must_use]
    const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Skipped => "skipped",
            Self::Failed => "failed",
        }
    }
}

/// Stable payload-free reason for a non-completed declared lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelComparisonOutcomeCode {
    EcapaModelUnavailable,
    EcapaModelInvalid,
    NativeSortformerModelUnavailable,
    NativeSortformerModelInvalid,
    SortformerModelCapacityExceeded,
    SortformerAdapterUnavailable,
    SortformerRuntimeIneligible,
    NativeExecutionFailed,
    EcapaInvalidInput,
    EcapaResourceLimit,
    EcapaCheckpointFailure,
    EcapaInternalContractFailure,
    EcapaNumericalFailure,
    EcapaPipelineRejected,
    EcapaContractViolation,
    EcapaStageTimedOut,
    EcapaExecutionFailed,
    NativeSortformerExecutionFailed,
    SortformerExecutionFailed,
    ScoringFailed,
    WorkerTimedOut,
    WorkerExecutionFailed,
    WorkerMalformedOutput,
    WorkerResourceProbeFailed,
}

impl ModelComparisonOutcomeCode {
    const ALL: [Self; 24] = [
        Self::EcapaModelUnavailable,
        Self::EcapaModelInvalid,
        Self::NativeSortformerModelUnavailable,
        Self::NativeSortformerModelInvalid,
        Self::SortformerModelCapacityExceeded,
        Self::SortformerAdapterUnavailable,
        Self::SortformerRuntimeIneligible,
        Self::NativeExecutionFailed,
        Self::EcapaInvalidInput,
        Self::EcapaResourceLimit,
        Self::EcapaCheckpointFailure,
        Self::EcapaInternalContractFailure,
        Self::EcapaNumericalFailure,
        Self::EcapaPipelineRejected,
        Self::EcapaContractViolation,
        Self::EcapaStageTimedOut,
        Self::EcapaExecutionFailed,
        Self::NativeSortformerExecutionFailed,
        Self::SortformerExecutionFailed,
        Self::ScoringFailed,
        Self::WorkerTimedOut,
        Self::WorkerExecutionFailed,
        Self::WorkerMalformedOutput,
        Self::WorkerResourceProbeFailed,
    ];

    #[must_use]
    pub const fn is_skip(self) -> bool {
        matches!(
            self,
            Self::EcapaModelUnavailable
                | Self::NativeSortformerModelUnavailable
                | Self::SortformerModelCapacityExceeded
                | Self::SortformerAdapterUnavailable
                | Self::SortformerRuntimeIneligible
        )
    }

    #[must_use]
    const fn as_str(self) -> &'static str {
        match self {
            Self::EcapaModelUnavailable => "ecapa_model_unavailable",
            Self::EcapaModelInvalid => "ecapa_model_invalid",
            Self::NativeSortformerModelUnavailable => "native_sortformer_model_unavailable",
            Self::NativeSortformerModelInvalid => "native_sortformer_model_invalid",
            Self::SortformerModelCapacityExceeded => "sortformer_model_capacity_exceeded",
            Self::SortformerAdapterUnavailable => "sortformer_adapter_unavailable",
            Self::SortformerRuntimeIneligible => "sortformer_runtime_ineligible",
            Self::NativeExecutionFailed => "native_execution_failed",
            Self::EcapaInvalidInput => "ecapa_invalid_input",
            Self::EcapaResourceLimit => "ecapa_resource_limit",
            Self::EcapaCheckpointFailure => "ecapa_checkpoint_failure",
            Self::EcapaInternalContractFailure => "ecapa_internal_contract_failure",
            Self::EcapaNumericalFailure => "ecapa_numerical_failure",
            Self::EcapaPipelineRejected => "ecapa_pipeline_rejected",
            Self::EcapaContractViolation => "ecapa_contract_violation",
            Self::EcapaStageTimedOut => "ecapa_stage_timed_out",
            Self::EcapaExecutionFailed => "ecapa_execution_failed",
            Self::NativeSortformerExecutionFailed => "native_sortformer_execution_failed",
            Self::SortformerExecutionFailed => "sortformer_execution_failed",
            Self::ScoringFailed => "scoring_failed",
            Self::WorkerTimedOut => "worker_timed_out",
            Self::WorkerExecutionFailed => "worker_execution_failed",
            Self::WorkerMalformedOutput => "worker_malformed_output",
            Self::WorkerResourceProbeFailed => "worker_resource_probe_failed",
        }
    }
}

/// Authority of one resource measurement. Zero is never used as a sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelComparisonResourceAuthority {
    Measured,
    UnavailableNoProbe,
}

/// What work is included in each lane's measured elapsed time.
///
/// Every lane uses this same scope so retained wall times are cross-lane
/// comparable without relying on shared parent-process setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelComparisonWallTimeScope {
    FreshProcessIdentityValidationModelLoadInferenceAndScorer,
}

/// Process scope covered by the native platform peak-RSS probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelComparisonPeakRssScope {
    /// Maximum concurrently observed RSS sum across the fresh worker process
    /// group, including every inherited subprocess adapter.
    WholeProcessTree,
}

/// Counts for every declared outcome, including stable reasons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ModelComparisonOutcomeCounts {
    pub declared: u64,
    pub completed: u64,
    pub skipped: u64,
    pub failed: u64,
    pub skipped_by_code: BTreeMap<ModelComparisonOutcomeCode, u64>,
    pub failed_by_code: BTreeMap<ModelComparisonOutcomeCode, u64>,
}

impl ModelComparisonOutcomeCounts {
    pub(crate) fn observe(
        &mut self,
        status: ModelComparisonOutcomeStatus,
        code: Option<ModelComparisonOutcomeCode>,
    ) -> FwResult<()> {
        match (status, code) {
            (ModelComparisonOutcomeStatus::Completed, None) => {
                let declared = self.declared.checked_add(1).ok_or_else(|| {
                    model_comparison_error("outcome_overflow", "declared outcome count overflowed")
                })?;
                let completed = self.completed.checked_add(1).ok_or_else(|| {
                    model_comparison_error("outcome_overflow", "completed outcome count overflowed")
                })?;
                self.declared = declared;
                self.completed = completed;
            }
            (ModelComparisonOutcomeStatus::Skipped, Some(code)) if code.is_skip() => {
                let declared = self.declared.checked_add(1).ok_or_else(|| {
                    model_comparison_error("outcome_overflow", "declared outcome count overflowed")
                })?;
                let skipped = self.skipped.checked_add(1).ok_or_else(|| {
                    model_comparison_error("outcome_overflow", "skipped outcome count overflowed")
                })?;
                let by_code = self
                    .skipped_by_code
                    .get(&code)
                    .copied()
                    .unwrap_or(0)
                    .checked_add(1)
                    .ok_or_else(|| {
                        model_comparison_error(
                            "outcome_overflow",
                            "skipped reason count overflowed",
                        )
                    })?;
                self.declared = declared;
                self.skipped = skipped;
                self.skipped_by_code.insert(code, by_code);
            }
            (ModelComparisonOutcomeStatus::Failed, Some(code)) if !code.is_skip() => {
                let declared = self.declared.checked_add(1).ok_or_else(|| {
                    model_comparison_error("outcome_overflow", "declared outcome count overflowed")
                })?;
                let failed = self.failed.checked_add(1).ok_or_else(|| {
                    model_comparison_error("outcome_overflow", "failed outcome count overflowed")
                })?;
                let by_code = self
                    .failed_by_code
                    .get(&code)
                    .copied()
                    .unwrap_or(0)
                    .checked_add(1)
                    .ok_or_else(|| {
                        model_comparison_error("outcome_overflow", "failed reason count overflowed")
                    })?;
                self.declared = declared;
                self.failed = failed;
                self.failed_by_code.insert(code, by_code);
            }
            _ => {
                return Err(model_comparison_error(
                    "outcome_state",
                    "comparison outcome status and reason code disagree",
                ));
            }
        }
        Ok(())
    }

    fn validate(&self, expected_declared: u64) -> bool {
        let Some(total_statuses) = self
            .completed
            .checked_add(self.skipped)
            .and_then(|value| value.checked_add(self.failed))
        else {
            return false;
        };
        let skipped_reasons = self
            .skipped_by_code
            .values()
            .try_fold(0u64, |sum, value| sum.checked_add(*value));
        let failed_reasons = self
            .failed_by_code
            .values()
            .try_fold(0u64, |sum, value| sum.checked_add(*value));
        self.declared == expected_declared
            && total_statuses == self.declared
            && skipped_reasons == Some(self.skipped)
            && failed_reasons == Some(self.failed)
            && self.skipped_by_code.values().all(|value| *value > 0)
            && self.failed_by_code.values().all(|value| *value > 0)
            && self.skipped_by_code.keys().all(|code| code.is_skip())
            && self.failed_by_code.keys().all(|code| !code.is_skip())
    }
}

/// Complete aggregate accuracy and elapsed-time metrics from the frozen scorer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ModelComparisonAggregateMetrics {
    pub recording_count: u64,
    pub audio_duration_sec: f64,
    pub reference_speaker_time_sec: f64,
    pub missed_speech_sec: f64,
    pub false_alarm_sec: f64,
    pub speaker_confusion_sec: f64,
    pub micro_der: Option<f64>,
    pub macro_der: Option<f64>,
    pub macro_jer: Option<f64>,
    pub scored_region_total_absolute_speaker_count_error: u64,
    pub scored_region_mean_absolute_speaker_count_error: Option<f64>,
    pub scored_region_exact_speaker_count: u64,
    pub scored_region_exact_speaker_count_rate: Option<f64>,
    pub full_timeline_total_absolute_speaker_count_error: u64,
    pub full_timeline_mean_absolute_speaker_count_error: Option<f64>,
    pub full_timeline_exact_speaker_count: u64,
    pub full_timeline_exact_speaker_count_rate: Option<f64>,
    pub count_estimate_resolved: u64,
    pub count_estimate_unresolved: u64,
    pub count_estimate_total_absolute_error: u64,
    pub count_estimate_mean_absolute_error: Option<f64>,
    pub count_estimate_exact: u64,
    pub count_estimate_exact_rate: Option<f64>,
    pub overlap_reference_sec: f64,
    pub overlap_hypothesis_sec: f64,
    pub overlap_true_positive_sec: f64,
    pub overlap_false_positive_sec: f64,
    pub overlap_false_negative_sec: f64,
    pub overlap_precision: Option<f64>,
    pub overlap_recall: Option<f64>,
    pub overlap_f1: Option<f64>,
    pub change_reference_count: u64,
    pub change_hypothesis_count: u64,
    pub change_matched_count: u64,
    pub change_mean_absolute_error_sec: Option<f64>,
    pub selective_reference_speaker_time_sec: f64,
    pub selective_covered_speaker_time_sec: f64,
    pub selective_error_covered_speaker_time_sec: f64,
    pub selective_coverage: Option<f64>,
    pub selective_risk: Option<f64>,
    pub labeled_speaker_time_sec: f64,
    pub unknown_speaker_time_sec: f64,
    pub unknown_speaker_share: Option<f64>,
    pub wall_time_sec: f64,
    pub real_time_factor: Option<f64>,
}

/// Honest resource statement kept separate from accuracy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelComparisonResourceEvidence {
    pub wall_time_authority: ModelComparisonResourceAuthority,
    pub wall_time_scope: ModelComparisonWallTimeScope,
    pub wall_time_cross_lane_comparable: bool,
    pub timed_attempt_count: u64,
    pub attempted_wall_time_ms: u64,
    pub completed_wall_time_ms: u64,
    pub peak_rss_authority: ModelComparisonResourceAuthority,
    pub peak_rss_scope: ModelComparisonPeakRssScope,
    pub peak_rss_minimum_sampling_interval_ms: Option<u64>,
    pub sampled_peak_rss_bytes: Option<u64>,
    pub cancellation_latency_authority: ModelComparisonResourceAuthority,
    pub maximum_cancellation_latency_ms: Option<u64>,
    pub hard_timeout_seconds: Option<u64>,
    pub maximum_completed_real_time_factor_millionths: Option<u64>,
    pub real_time_factor_cap_millionths: u64,
    pub real_time_factor_within_cap: Option<bool>,
    pub peak_rss_cap_bytes: u64,
    pub peak_rss_within_cap: Option<bool>,
    pub cancellation_latency_cap_ms: u64,
    pub cancellation_latency_within_cap: Option<bool>,
}

/// One path-free external runtime identity shared by every Sortformer row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelComparisonExternalRuntimeIdentity {
    pub protocol_version: String,
    pub authority: String,
    pub tool_version: String,
    pub adapter_version: String,
    pub model_id: String,
    pub model_revision: String,
    pub upstream_license: String,
    pub model_contract_sha256: String,
    pub model_artifact_sha256: String,
    pub model_artifact_bytes: u64,
    pub runtime_fingerprint_sha256: String,
    pub executable_sha256: String,
    pub version_stdout_sha256: String,
}

/// One lane's available-case and all-five-common-case reductions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelComparisonLaneAggregate {
    pub lane: ModelComparisonLane,
    pub outcomes: ModelComparisonOutcomeCounts,
    pub available_case: ModelComparisonAggregateMetrics,
    pub common_complete_case: ModelComparisonAggregateMetrics,
    pub resources: ModelComparisonResourceEvidence,
}

/// Frozen protocol binding every consequential comparison choice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicModelComparisonProtocol {
    pub schema_version: String,
    pub runner_version: String,
    pub schedule_version: String,
    pub lanes: Vec<ModelComparisonLane>,
    pub schedule: Vec<[ModelComparisonLane; 5]>,
    pub full_recordings_only: bool,
    pub input_sample_rate_hz: u32,
    pub input_channels: u16,
    pub input_bits_per_sample: u16,
    pub input_sample_format: String,
    pub normalized_observation_binding: String,
    pub speaker_count_policy: String,
    pub oracle_count_diagnostic_present: bool,
    pub sortformer_capacity_eligibility: String,
    pub speech_activity_authority: String,
    pub overlap_policy: String,
    pub speaker_boundary_collar_ms: u64,
    pub change_boundary_collar_ms: u64,
    pub scorer_config_sha256: String,
    pub outcome_taxonomy_version: String,
    pub outcome_codes: Vec<ModelComparisonOutcomeCode>,
    pub native_acoustic_request_sha256: String,
    pub native_ecapa_request_sha256: String,
    pub native_ecapa_fused_request_sha256: String,
    pub native_sortformer_package_sha256: String,
    pub native_sortformer_receipt_sha256: String,
    pub native_rayon_threads: u16,
    pub sortformer_intraop_threads: u16,
    pub sortformer_interop_threads: u16,
    pub sortformer_max_speakers: u16,
    pub sortformer_output_frame_ms: u32,
    pub attempt_hard_timeout_seconds: u64,
    pub worker_schema_version: String,
    pub worker_process_policy: String,
    pub worker_stdout_limit_bytes: u64,
    pub cancellation_probe_delay_ms: u64,
    pub cancellation_probe_timeout_seconds: u64,
    pub cancellation_probe_self_limit_seconds: u64,
    pub native_real_time_factor_cap_millionths: u64,
    pub external_real_time_factor_cap_millionths: u64,
    pub peak_rss_cap_bytes: u64,
    pub process_tree_rss_minimum_sample_interval_ms: u64,
    pub cancellation_latency_cap_ms: u64,
    pub ecapa_contract_sha256: String,
    pub ecapa_package_sha256: String,
    pub sortformer_contract_sha256: String,
    pub native_acoustic_postprocessing: String,
    pub native_ecapa_postprocessing: String,
    pub native_ecapa_fused_postprocessing: String,
    pub sortformer_postprocessing: String,
    pub wall_time_policy: String,
    pub aggregate_only: bool,
    pub production_route_changed: bool,
}

/// Aggregate path-free evidence. Per-recording rows are intentionally absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicModelComparisonEvidence {
    pub schema_version: String,
    pub runner_version: String,
    pub scorer_version: String,
    pub corpus_key: String,
    pub source_version: String,
    pub evaluation_split: EvaluationSplit,
    pub descriptor_sha256: String,
    pub bundle_sha256: String,
    pub comparison_executable_sha256: String,
    pub observation_count: u64,
    pub observation_set_sha256: String,
    pub execution_order_sha256: String,
    pub outcome_sequence_sha256: String,
    pub external_runtime_observation_set_sha256: String,
    pub sortformer_runtime_identity: Option<ModelComparisonExternalRuntimeIdentity>,
    pub common_complete_recording_count: u64,
    pub common_complete_observation_set_sha256: String,
    pub order_balance_complete: bool,
    pub protocol: PublicModelComparisonProtocol,
    pub protocol_sha256: String,
    pub lanes: Vec<ModelComparisonLaneAggregate>,
    pub development_uncertified: bool,
    pub comparison_authority: String,
    pub superiority_claim_permitted: bool,
    pub production_route_changed: bool,
    /// Hash after runtime identity and all timing/resource values are normalized.
    pub deterministic_accuracy_sha256: String,
    /// Hash of this document with this field empty.
    pub result_sha256: String,
}

/// External-only paths and bounded execution settings for one comparison.
///
/// Deliberately neither `Debug` nor serializable: source and output paths can
/// never become part of retained evidence or an error payload.
pub struct PublicModelComparisonRequest<'a> {
    pub project_root: &'a Path,
    pub input_root: &'a Path,
    pub descriptor_path: &'a Path,
    pub bundle_output_path: &'a Path,
    pub evidence_output_path: &'a Path,
    pub license_acknowledgement_id: &'a str,
    pub evaluation_split: EvaluationSplit,
    pub attempt_hard_timeout: Duration,
}

enum EcapaAvailability {
    Ready {
        model: Box<EcapaModel>,
        package_path: PathBuf,
    },
    Unavailable,
    Invalid,
}

enum NativeSortformerAvailability {
    Ready {
        session: Box<SortformerSession>,
        package_path: PathBuf,
        receipt_path: PathBuf,
    },
    Unavailable,
    Invalid,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InternalLaneOutcome {
    lane: ModelComparisonLane,
    status: ModelComparisonOutcomeStatus,
    code: Option<ModelComparisonOutcomeCode>,
    score: Option<AuthoritativeDiarizationScore>,
    external_provenance_sha256: Option<String>,
    external_runtime_identity: Option<ModelComparisonExternalRuntimeIdentity>,
    attempt_wall_time_ms: u64,
    count: Option<ModelComparisonCountObservation>,
    worker_peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelComparisonCountObservation {
    reference_full_timeline: u64,
    hypothesis_full_timeline: u64,
    selected_count: Option<u64>,
}

impl InternalLaneOutcome {
    fn completed(
        lane: ModelComparisonLane,
        score: AuthoritativeDiarizationScore,
        external_provenance_sha256: Option<String>,
        count: ModelComparisonCountObservation,
    ) -> Self {
        Self {
            lane,
            status: ModelComparisonOutcomeStatus::Completed,
            code: None,
            score: Some(score),
            external_provenance_sha256,
            external_runtime_identity: None,
            attempt_wall_time_ms: 0,
            count: Some(count),
            worker_peak_rss_bytes: None,
        }
    }

    fn skipped(lane: ModelComparisonLane, code: ModelComparisonOutcomeCode) -> Self {
        debug_assert!(code.is_skip());
        Self {
            lane,
            status: ModelComparisonOutcomeStatus::Skipped,
            code: Some(code),
            score: None,
            external_provenance_sha256: None,
            external_runtime_identity: None,
            attempt_wall_time_ms: 0,
            count: None,
            worker_peak_rss_bytes: None,
        }
    }

    fn skipped_with_provenance(
        lane: ModelComparisonLane,
        code: ModelComparisonOutcomeCode,
        external_provenance_sha256: String,
    ) -> Self {
        let mut outcome = Self::skipped(lane, code);
        outcome.external_provenance_sha256 = Some(external_provenance_sha256);
        outcome
    }

    fn failed(lane: ModelComparisonLane, code: ModelComparisonOutcomeCode) -> Self {
        debug_assert!(!code.is_skip());
        Self {
            lane,
            status: ModelComparisonOutcomeStatus::Failed,
            code: Some(code),
            score: None,
            external_provenance_sha256: None,
            external_runtime_identity: None,
            attempt_wall_time_ms: 0,
            count: None,
            worker_peak_rss_bytes: None,
        }
    }

    fn failed_with_provenance(
        lane: ModelComparisonLane,
        code: ModelComparisonOutcomeCode,
        external_provenance_sha256: String,
    ) -> Self {
        let mut outcome = Self::failed(lane, code);
        outcome.external_provenance_sha256 = Some(external_provenance_sha256);
        outcome
    }

    fn with_external_runtime_identity(
        mut self,
        identity: Option<ModelComparisonExternalRuntimeIdentity>,
    ) -> Self {
        self.external_runtime_identity = identity;
        self
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelComparisonWorkerRequest {
    schema_version: String,
    lane: ModelComparisonLane,
    audio_path: PathBuf,
    expected_audio_sha256: String,
    expected_normalized_input_sha256: String,
    reference: crate::diarization::DiarizationReferenceDocument,
    expected_reference_sha256: String,
    reference_speaker_count: u64,
    protocol_sha256: String,
    scorer_config_sha256: String,
    executable_sha256: String,
    hard_timeout_seconds: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelComparisonWorkerResponse {
    schema_version: String,
    lane: ModelComparisonLane,
    executable_sha256_before: String,
    executable_sha256_after: String,
    audio_sha256_before: String,
    audio_sha256_after: String,
    normalized_input_sha256: String,
    reference_sha256: String,
    protocol_sha256: String,
    scorer_config_sha256: String,
    ecapa_package_sha256_after: Option<String>,
    native_sortformer_package_sha256_after: Option<String>,
    native_sortformer_receipt_sha256_after: Option<String>,
    worker_wall_time_ms: u64,
    outcome: InternalLaneOutcome,
}

#[derive(Default)]
struct LaneReduction {
    outcomes: ModelComparisonOutcomeCounts,
    available: ModelComparisonAccumulator,
    common: ModelComparisonAccumulator,
    timed_attempt_count: u64,
    attempted_wall_time_ms: u64,
    completed_wall_time_ms: u64,
    maximum_peak_rss_bytes: Option<u64>,
    peak_rss_measured_attempt_count: u64,
    maximum_completed_real_time_factor_millionths: Option<u64>,
}

#[derive(Default)]
struct ModelComparisonAccumulator {
    recording_count: u64,
    audio_duration_sec: f64,
    reference_speaker_time_sec: f64,
    missed_speech_sec: f64,
    false_alarm_sec: f64,
    speaker_confusion_sec: f64,
    macro_der_sum: f64,
    macro_der_count: u64,
    macro_jer_sum: f64,
    macro_jer_count: u64,
    scored_region_total_absolute_speaker_count_error: u64,
    scored_region_exact_speaker_count: u64,
    full_timeline_total_absolute_speaker_count_error: u64,
    full_timeline_exact_speaker_count: u64,
    count_estimate_resolved: u64,
    count_estimate_unresolved: u64,
    count_estimate_total_absolute_error: u64,
    count_estimate_exact: u64,
    overlap_reference_sec: f64,
    overlap_hypothesis_sec: f64,
    overlap_true_positive_sec: f64,
    overlap_false_positive_sec: f64,
    overlap_false_negative_sec: f64,
    change_reference_count: u64,
    change_hypothesis_count: u64,
    change_matched_count: u64,
    change_absolute_error_sec: f64,
    selective_reference_speaker_time_sec: f64,
    selective_covered_speaker_time_sec: f64,
    selective_error_covered_speaker_time_sec: f64,
    labeled_speaker_time_sec: f64,
    unknown_speaker_time_sec: f64,
    wall_time_ms: u64,
}

impl ModelComparisonAccumulator {
    fn push(
        &mut self,
        score: &AuthoritativeDiarizationScore,
        count: ModelComparisonCountObservation,
        completed_wall_time_ms: u64,
    ) {
        self.recording_count = self.recording_count.saturating_add(1);
        self.audio_duration_sec += score.scored_duration_sec + score.ignored_duration_sec;
        self.reference_speaker_time_sec += score.diarization.reference_speaker_time_sec;
        self.missed_speech_sec += score.diarization.missed_speech_sec;
        self.false_alarm_sec += score.diarization.false_alarm_sec;
        self.speaker_confusion_sec += score.diarization.speaker_confusion_sec;
        if let Some(value) = score.diarization.der {
            self.macro_der_sum += value;
            self.macro_der_count = self.macro_der_count.saturating_add(1);
        }
        if let Some(value) = score.diarization.jer {
            self.macro_jer_sum += value;
            self.macro_jer_count = self.macro_jer_count.saturating_add(1);
        }
        self.scored_region_total_absolute_speaker_count_error = self
            .scored_region_total_absolute_speaker_count_error
            .saturating_add(score.speaker_count.absolute_error);
        self.scored_region_exact_speaker_count = self
            .scored_region_exact_speaker_count
            .saturating_add(u64::from(score.speaker_count.absolute_error == 0));
        let full_timeline_error = count
            .reference_full_timeline
            .abs_diff(count.hypothesis_full_timeline);
        self.full_timeline_total_absolute_speaker_count_error = self
            .full_timeline_total_absolute_speaker_count_error
            .saturating_add(full_timeline_error);
        self.full_timeline_exact_speaker_count = self
            .full_timeline_exact_speaker_count
            .saturating_add(u64::from(full_timeline_error == 0));
        if let Some(selected_count) = count.selected_count {
            let count_estimate_error = count.reference_full_timeline.abs_diff(selected_count);
            self.count_estimate_resolved = self.count_estimate_resolved.saturating_add(1);
            self.count_estimate_total_absolute_error = self
                .count_estimate_total_absolute_error
                .saturating_add(count_estimate_error);
            self.count_estimate_exact = self
                .count_estimate_exact
                .saturating_add(u64::from(count_estimate_error == 0));
        } else {
            self.count_estimate_unresolved = self.count_estimate_unresolved.saturating_add(1);
        }
        self.overlap_reference_sec += score.overlap.reference_overlap_sec;
        self.overlap_hypothesis_sec += score.overlap.hypothesis_overlap_sec;
        self.overlap_true_positive_sec += score.overlap.true_positive_sec;
        self.overlap_false_positive_sec += score.overlap.false_positive_sec;
        self.overlap_false_negative_sec += score.overlap.false_negative_sec;
        self.change_reference_count = self
            .change_reference_count
            .saturating_add(score.change_points.reference_count as u64);
        self.change_hypothesis_count = self
            .change_hypothesis_count
            .saturating_add(score.change_points.hypothesis_count as u64);
        self.change_matched_count = self
            .change_matched_count
            .saturating_add(score.change_points.matched_count as u64);
        self.change_absolute_error_sec +=
            score.change_points.mean_absolute_error_sec.unwrap_or(0.0)
                * score.change_points.matched_count as f64;
        self.selective_reference_speaker_time_sec +=
            score.selective_attribution.reference_speaker_time_sec;
        self.selective_covered_speaker_time_sec +=
            score.selective_attribution.covered_speaker_time_sec;
        self.selective_error_covered_speaker_time_sec +=
            score.selective_attribution.error_covered_speaker_time_sec;
        self.labeled_speaker_time_sec += score.speaker_occupancy.labeled_speaker_time_sec;
        self.unknown_speaker_time_sec += score.speaker_occupancy.unknown_speaker_time_sec;
        self.wall_time_ms = self.wall_time_ms.saturating_add(completed_wall_time_ms);
    }

    fn finish(self) -> ModelComparisonAggregateMetrics {
        let wall_time_sec = self.wall_time_ms as f64 / 1_000.0;
        let diarization_error =
            self.missed_speech_sec + self.false_alarm_sec + self.speaker_confusion_sec;
        let overlap_precision = ratio_f64(
            self.overlap_true_positive_sec,
            self.overlap_true_positive_sec + self.overlap_false_positive_sec,
        );
        let overlap_recall = ratio_f64(
            self.overlap_true_positive_sec,
            self.overlap_true_positive_sec + self.overlap_false_negative_sec,
        );
        let overlap_f1 = harmonic_mean(overlap_precision, overlap_recall);
        ModelComparisonAggregateMetrics {
            recording_count: self.recording_count,
            audio_duration_sec: canonical(self.audio_duration_sec),
            reference_speaker_time_sec: canonical(self.reference_speaker_time_sec),
            missed_speech_sec: canonical(self.missed_speech_sec),
            false_alarm_sec: canonical(self.false_alarm_sec),
            speaker_confusion_sec: canonical(self.speaker_confusion_sec),
            micro_der: ratio_f64(diarization_error, self.reference_speaker_time_sec),
            macro_der: ratio_f64(self.macro_der_sum, self.macro_der_count as f64),
            macro_jer: ratio_f64(self.macro_jer_sum, self.macro_jer_count as f64),
            scored_region_total_absolute_speaker_count_error: self
                .scored_region_total_absolute_speaker_count_error,
            scored_region_mean_absolute_speaker_count_error: ratio_f64(
                self.scored_region_total_absolute_speaker_count_error as f64,
                self.recording_count as f64,
            ),
            scored_region_exact_speaker_count: self.scored_region_exact_speaker_count,
            scored_region_exact_speaker_count_rate: ratio_f64(
                self.scored_region_exact_speaker_count as f64,
                self.recording_count as f64,
            ),
            full_timeline_total_absolute_speaker_count_error: self
                .full_timeline_total_absolute_speaker_count_error,
            full_timeline_mean_absolute_speaker_count_error: ratio_f64(
                self.full_timeline_total_absolute_speaker_count_error as f64,
                self.recording_count as f64,
            ),
            full_timeline_exact_speaker_count: self.full_timeline_exact_speaker_count,
            full_timeline_exact_speaker_count_rate: ratio_f64(
                self.full_timeline_exact_speaker_count as f64,
                self.recording_count as f64,
            ),
            count_estimate_resolved: self.count_estimate_resolved,
            count_estimate_unresolved: self.count_estimate_unresolved,
            count_estimate_total_absolute_error: self.count_estimate_total_absolute_error,
            count_estimate_mean_absolute_error: ratio_f64(
                self.count_estimate_total_absolute_error as f64,
                self.count_estimate_resolved as f64,
            ),
            count_estimate_exact: self.count_estimate_exact,
            count_estimate_exact_rate: ratio_f64(
                self.count_estimate_exact as f64,
                self.count_estimate_resolved as f64,
            ),
            overlap_reference_sec: canonical(self.overlap_reference_sec),
            overlap_hypothesis_sec: canonical(self.overlap_hypothesis_sec),
            overlap_true_positive_sec: canonical(self.overlap_true_positive_sec),
            overlap_false_positive_sec: canonical(self.overlap_false_positive_sec),
            overlap_false_negative_sec: canonical(self.overlap_false_negative_sec),
            overlap_precision,
            overlap_recall,
            overlap_f1,
            change_reference_count: self.change_reference_count,
            change_hypothesis_count: self.change_hypothesis_count,
            change_matched_count: self.change_matched_count,
            change_mean_absolute_error_sec: ratio_f64(
                self.change_absolute_error_sec,
                self.change_matched_count as f64,
            ),
            selective_reference_speaker_time_sec: canonical(
                self.selective_reference_speaker_time_sec,
            ),
            selective_covered_speaker_time_sec: canonical(self.selective_covered_speaker_time_sec),
            selective_error_covered_speaker_time_sec: canonical(
                self.selective_error_covered_speaker_time_sec,
            ),
            selective_coverage: ratio_f64(
                self.selective_covered_speaker_time_sec,
                self.selective_reference_speaker_time_sec,
            ),
            selective_risk: ratio_f64(
                self.selective_error_covered_speaker_time_sec,
                self.selective_covered_speaker_time_sec,
            ),
            labeled_speaker_time_sec: canonical(self.labeled_speaker_time_sec),
            unknown_speaker_time_sec: canonical(self.unknown_speaker_time_sec),
            unknown_speaker_share: ratio_f64(
                self.unknown_speaker_time_sec,
                self.labeled_speaker_time_sec + self.unknown_speaker_time_sec,
            ),
            wall_time_sec: canonical(wall_time_sec),
            real_time_factor: ratio_f64(wall_time_sec, self.audio_duration_sec),
        }
    }
}

/// Execute the frozen five-lane comparison and publish a validated public
/// bundle plus aggregate comparison evidence, each with an atomic no-clobber
/// commit. The two independent files are not a transactional pair.
pub fn run_public_model_comparison_with_cancel<F>(
    request: PublicModelComparisonRequest<'_>,
    is_cancelled: F,
) -> FwResult<PublicModelComparisonEvidence>
where
    F: Fn() -> bool + Sync,
{
    let PublicModelComparisonRequest {
        project_root,
        input_root,
        descriptor_path,
        bundle_output_path,
        evidence_output_path,
        license_acknowledgement_id,
        evaluation_split,
        attempt_hard_timeout,
    } = request;
    if evaluation_split != EvaluationSplit::Development {
        return Err(model_comparison_error(
            "split",
            "the uncertified v4 comparison is restricted to the development split",
        ));
    }
    if attempt_hard_timeout != Duration::from_secs(PUBLIC_MODEL_COMPARISON_ATTEMPT_TIMEOUT_SECONDS)
    {
        return Err(model_comparison_error(
            "attempt_timeout",
            "attempt timeout must equal the frozen 1800-second comparison timeout",
        ));
    }
    cancellation_checkpoint(&is_cancelled)?;

    let canonical_project = super::canonical_directory(project_root, "project_root")?;
    let canonical_input = super::canonical_directory(input_root, "input_root")?;
    if super::paths_overlap(&canonical_project, &canonical_input) {
        return Err(model_comparison_error(
            "input_root_overlap",
            "input root must be disjoint from the project checkout",
        ));
    }
    let bundle_output_parent =
        super::validate_new_output(&canonical_project, &canonical_input, bundle_output_path)?;
    let evidence_output_parent =
        super::validate_new_output(&canonical_project, &canonical_input, evidence_output_path)?;
    if bundle_output_parent.same_output_target(
        bundle_output_path,
        &evidence_output_parent,
        evidence_output_path,
        "model_comparison_output",
    )? {
        return Err(model_comparison_error(
            "output",
            "bundle and comparison evidence outputs must be distinct",
        ));
    }

    let canonical_descriptor =
        super::canonical_input_file(&canonical_input, descriptor_path, "descriptor")?;
    let descriptor_bytes = super::read_bounded(
        &canonical_descriptor,
        super::MAX_DESCRIPTOR_BYTES,
        "descriptor",
    )?;
    let descriptor_sha256 = format!("{:x}", Sha256::digest(&descriptor_bytes));
    let descriptor: super::PublicCorpusInput =
        serde_json::from_slice(&descriptor_bytes).map_err(|_| {
            model_comparison_error(
                "descriptor_json",
                "descriptor must be valid public-corpus input JSON",
            )
        })?;
    if descriptor.schema_version != super::PUBLIC_CORPUS_INPUT_SCHEMA_VERSION {
        return Err(model_comparison_error(
            "descriptor_schema",
            "descriptor schema is unsupported",
        ));
    }
    super::validate_public_id(&descriptor.corpus_key, "corpus_key")?;
    super::validate_public_id(&descriptor.source_version, "source_version")?;

    let sortformer_contract = sortformer_oracle_contract();
    let native_rayon_threads = u16::try_from(rayon::current_num_threads()).map_err(|_| {
        model_comparison_error(
            "thread_policy",
            "native Rayon thread count exceeds the retained range",
        )
    })?;
    if native_rayon_threads != sortformer_contract.torch_intraop_threads {
        return Err(model_comparison_error(
            "thread_policy",
            "launch with RAYON_NUM_THREADS=8 so native and Sortformer CPU lanes have matched worker counts",
        ));
    }
    let protocol = frozen_model_comparison_protocol()?;
    if protocol.native_rayon_threads != native_rayon_threads {
        return Err(model_comparison_error(
            "thread_policy",
            "runtime thread count disagrees with the frozen comparison protocol",
        ));
    }
    let protocol_sha256 = super::canonical_sha256(&protocol)?;
    if protocol_sha256 != PUBLIC_MODEL_COMPARISON_PROTOCOL_SHA256 {
        return Err(model_comparison_error(
            "protocol_drift",
            "model-comparison protocol changed without a versioned digest update",
        ));
    }
    let worker_executable = std::env::current_exe()
        .and_then(std::fs::canonicalize)
        .map_err(|_| {
            model_comparison_error(
                "worker_executable",
                "the current comparison executable could not be resolved",
            )
        })?;
    let worker_executable_sha256 = hash_bounded_file_with_cancel(
        &worker_executable,
        MAX_MODEL_COMPARISON_EXECUTABLE_BYTES,
        &is_cancelled,
        "worker_executable",
    )?;
    let cancellation_latency_ms = measure_worker_cancellation_latency(
        &worker_executable,
        &worker_executable_sha256,
        &is_cancelled,
    )?;
    cancellation_checkpoint(&is_cancelled)?;

    let mut cancellation_adapter = || is_cancelled();
    let bundle = super::materialize_public_corpus_bundle_for_split_with_cancel(
        project_root,
        input_root,
        descriptor_path,
        license_acknowledgement_id,
        Some(evaluation_split),
        Some(&descriptor_sha256),
        &mut cancellation_adapter,
    )?;
    if bundle.descriptor_sha256 != descriptor_sha256
        || bundle.corpus_key != descriptor.corpus_key
        || bundle.source_version != descriptor.source_version
    {
        return Err(model_comparison_error(
            "descriptor_changed",
            "descriptor identity changed between preflight and materialization",
        ));
    }
    let input_recordings = descriptor
        .recordings
        .into_iter()
        .filter(|recording| recording.split == evaluation_split)
        .map(|recording| (recording.recording_id.clone(), recording))
        .collect::<BTreeMap<_, _>>();
    if input_recordings.len() != bundle.references.len() || bundle.references.is_empty() {
        return Err(model_comparison_error(
            "alignment",
            "selected descriptor and bundle recording counts differ or are empty",
        ));
    }

    let mut reductions = ModelComparisonLane::ALL
        .into_iter()
        .map(|lane| (lane, LaneReduction::default()))
        .collect::<BTreeMap<_, _>>();
    let mut observation_hasher =
        domain_hasher(b"franken-whisper-model-comparison-observations-v1\0");
    let mut order_hasher = domain_hasher(b"franken-whisper-model-comparison-orders-v1\0");
    let mut outcome_hasher = domain_hasher(b"franken-whisper-model-comparison-outcomes-v1\0");
    let mut external_runtime_hasher =
        domain_hasher(b"franken-whisper-model-comparison-external-runtime-v1\0");
    let mut common_complete_hasher =
        domain_hasher(b"franken-whisper-model-comparison-common-complete-v1\0");
    let mut sortformer_runtime_identity_state =
        None::<Option<ModelComparisonExternalRuntimeIdentity>>;
    hash_token(&mut observation_hasher, bundle.bundle_sha256.as_bytes());
    hash_token(&mut observation_hasher, protocol_sha256.as_bytes());
    let mut common_complete_recording_count = 0u64;

    for (observation_index, ((reference, recording_evidence), manifest_recording)) in bundle
        .references
        .iter()
        .zip(&bundle.recordings)
        .zip(&bundle.manifest.recordings)
        .enumerate()
    {
        cancellation_checkpoint(&is_cancelled)?;
        if reference.recording_id != recording_evidence.recording_id
            || reference.recording_id != manifest_recording.recording_id
            || manifest_recording.split != evaluation_split
        {
            return Err(model_comparison_error(
                "alignment",
                "bundle reference and recording evidence are misaligned",
            ));
        }
        let input_recording = input_recordings
            .get(&reference.recording_id)
            .ok_or_else(|| {
                model_comparison_error(
                    "alignment",
                    "validated recording is absent from the selected descriptor",
                )
            })?;
        if recording_evidence.sample_rate_hz != protocol.input_sample_rate_hz
            || recording_evidence.channel_count != protocol.input_channels
            || recording_evidence.selected_channel != 1
        {
            return Err(model_comparison_error(
                "audio_contract",
                "comparison requires already-normalized 16 kHz mono WAV input",
            ));
        }
        let audio_path =
            super::canonical_relative_file(&canonical_input, &input_recording.audio_path, "audio")?;
        let audio_bytes = super::read_bounded(
            &audio_path,
            super::MAX_EVALUATION_AUDIO_BYTES,
            "model_comparison_audio",
        )?;
        if format!("{:x}", Sha256::digest(&audio_bytes)) != recording_evidence.audio_sha256 {
            return Err(model_comparison_error(
                "audio_changed",
                "audio changed after bundle validation and before comparison execution",
            ));
        }
        let samples = decode_pcm16_wave(&audio_bytes, &is_cancelled)?;
        drop(audio_bytes);
        let duration_ms = u64::try_from(samples.len()).unwrap_or(u64::MAX) / 16;
        if duration_ms == 0 {
            return Err(model_comparison_error(
                "audio_duration",
                "comparison audio must contain at least one full millisecond",
            ));
        }
        let clipped_reference = super::clipped_reference(reference, Some(duration_ms))?;
        let clipped_reference_sha256 = super::canonical_sha256(&clipped_reference)?;
        let normalized_input_sha256 = super::hash_pcm_prefix(&samples);
        observation_hasher.update((observation_index as u64).to_le_bytes());
        hash_token(
            &mut observation_hasher,
            recording_evidence.audio_sha256.as_bytes(),
        );
        hash_token(&mut observation_hasher, normalized_input_sha256.as_bytes());
        hash_token(
            &mut observation_hasher,
            recording_evidence.reference_sha256.as_bytes(),
        );
        observation_hasher.update(duration_ms.to_le_bytes());

        let schedule = model_comparison_schedule_row(observation_index);
        order_hasher.update((observation_index as u64).to_le_bytes());
        for lane in schedule {
            hash_token(&mut order_hasher, lane.as_str().as_bytes());
        }
        let reference_speaker_count = count_labeled_speakers(&clipped_reference.turns)?;
        let mut row = Vec::with_capacity(ModelComparisonLane::ALL.len());
        for lane in schedule {
            cancellation_checkpoint(&is_cancelled)?;
            let outcome = execute_lane_worker(
                lane,
                &worker_executable,
                &worker_executable_sha256,
                &audio_path,
                &recording_evidence.audio_sha256,
                &normalized_input_sha256,
                &clipped_reference,
                &clipped_reference_sha256,
                reference_speaker_count,
                &protocol_sha256,
                &protocol.scorer_config_sha256,
                attempt_hard_timeout,
                &is_cancelled,
            )?;
            row.push(outcome);
        }
        let common_complete = row
            .iter()
            .all(|outcome| outcome.status == ModelComparisonOutcomeStatus::Completed);
        common_complete_recording_count =
            common_complete_recording_count.saturating_add(u64::from(common_complete));
        if common_complete {
            common_complete_hasher.update((observation_index as u64).to_le_bytes());
            hash_token(
                &mut common_complete_hasher,
                recording_evidence.audio_sha256.as_bytes(),
            );
            hash_token(
                &mut common_complete_hasher,
                recording_evidence.reference_sha256.as_bytes(),
            );
        }
        for outcome in &row {
            let reduction = reductions.get_mut(&outcome.lane).ok_or_else(|| {
                model_comparison_error("lane", "comparison returned an undeclared lane")
            })?;
            reduction.outcomes.observe(outcome.status, outcome.code)?;
            reduction.timed_attempt_count = reduction.timed_attempt_count.saturating_add(1);
            reduction.attempted_wall_time_ms = reduction
                .attempted_wall_time_ms
                .saturating_add(outcome.attempt_wall_time_ms);
            if let Some(peak_rss_bytes) = outcome.worker_peak_rss_bytes {
                reduction.peak_rss_measured_attempt_count =
                    reduction.peak_rss_measured_attempt_count.saturating_add(1);
                reduction.maximum_peak_rss_bytes = Some(
                    reduction
                        .maximum_peak_rss_bytes
                        .unwrap_or(0)
                        .max(peak_rss_bytes),
                );
            }
            if outcome.status == ModelComparisonOutcomeStatus::Completed {
                reduction.completed_wall_time_ms = reduction
                    .completed_wall_time_ms
                    .saturating_add(outcome.attempt_wall_time_ms);
                let rtf_millionths =
                    ratio_millionths_ceil(outcome.attempt_wall_time_ms, duration_ms, "worker_rtf")?;
                reduction.maximum_completed_real_time_factor_millionths = Some(
                    reduction
                        .maximum_completed_real_time_factor_millionths
                        .unwrap_or(0)
                        .max(rtf_millionths),
                );
            }
            outcome_hasher.update((observation_index as u64).to_le_bytes());
            hash_token(&mut outcome_hasher, outcome.lane.as_str().as_bytes());
            hash_token(&mut outcome_hasher, outcome.status.as_str().as_bytes());
            hash_token(
                &mut outcome_hasher,
                outcome
                    .code
                    .map_or("none", ModelComparisonOutcomeCode::as_str)
                    .as_bytes(),
            );
            if let Some(score) = &outcome.score {
                let score_sha256 = deterministic_score_sha256(score)?;
                hash_token(&mut outcome_hasher, score_sha256.as_bytes());
                let count = outcome.count.ok_or_else(|| {
                    model_comparison_error(
                        "count_observation",
                        "completed lane omitted its full-timeline count observation",
                    )
                })?;
                reduction
                    .available
                    .push(score, count, outcome.attempt_wall_time_ms);
                if common_complete {
                    reduction
                        .common
                        .push(score, count, outcome.attempt_wall_time_ms);
                }
            } else {
                hash_token(&mut outcome_hasher, b"no_score");
            }
            if let Some(provenance_sha256) = &outcome.external_provenance_sha256 {
                external_runtime_hasher.update((observation_index as u64).to_le_bytes());
                hash_token(&mut external_runtime_hasher, provenance_sha256.as_bytes());
                match &sortformer_runtime_identity_state {
                    None => {
                        sortformer_runtime_identity_state =
                            Some(outcome.external_runtime_identity.clone());
                    }
                    Some(expected)
                        if expected.as_ref() != outcome.external_runtime_identity.as_ref() =>
                    {
                        return Err(model_comparison_error(
                            "sortformer_runtime_changed",
                            "Sortformer executable or runtime identity changed within the comparison invocation",
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
    }
    cancellation_checkpoint(&is_cancelled)?;
    let final_worker_executable_sha256 = hash_bounded_file_with_cancel(
        &worker_executable,
        MAX_MODEL_COMPARISON_EXECUTABLE_BYTES,
        &is_cancelled,
        "worker_executable",
    )?;
    if final_worker_executable_sha256 != worker_executable_sha256 {
        return Err(model_comparison_error(
            "worker_executable_changed",
            "the comparison executable changed during the invocation",
        ));
    }

    let timeout_seconds = attempt_hard_timeout.as_secs();
    let lanes = ModelComparisonLane::ALL
        .into_iter()
        .map(|lane| {
            let reduction = reductions.remove(&lane).ok_or_else(|| {
                model_comparison_error("lane", "comparison omitted a declared lane reduction")
            })?;
            Ok(ModelComparisonLaneAggregate {
                lane,
                outcomes: reduction.outcomes,
                available_case: reduction.available.finish(),
                common_complete_case: reduction.common.finish(),
                resources: lane_resource_evidence(
                    lane,
                    timeout_seconds,
                    reduction.timed_attempt_count,
                    reduction.attempted_wall_time_ms,
                    reduction.completed_wall_time_ms,
                    reduction.peak_rss_measured_attempt_count,
                    reduction.maximum_peak_rss_bytes,
                    cancellation_latency_ms,
                    reduction.maximum_completed_real_time_factor_millionths,
                ),
            })
        })
        .collect::<FwResult<Vec<_>>>()?;
    let mut evidence = PublicModelComparisonEvidence {
        schema_version: PUBLIC_MODEL_COMPARISON_SCHEMA_VERSION.to_owned(),
        runner_version: PUBLIC_MODEL_COMPARISON_RUNNER_VERSION.to_owned(),
        scorer_version: DIARIZATION_SCORER_VERSION.to_owned(),
        corpus_key: bundle.corpus_key.clone(),
        source_version: bundle.source_version.clone(),
        evaluation_split,
        descriptor_sha256,
        bundle_sha256: bundle.bundle_sha256.clone(),
        comparison_executable_sha256: worker_executable_sha256,
        observation_count: u64::try_from(bundle.references.len()).unwrap_or(u64::MAX),
        observation_set_sha256: format!("{:x}", observation_hasher.finalize()),
        execution_order_sha256: format!("{:x}", order_hasher.finalize()),
        outcome_sequence_sha256: format!("{:x}", outcome_hasher.finalize()),
        external_runtime_observation_set_sha256: format!(
            "{:x}",
            external_runtime_hasher.finalize()
        ),
        sortformer_runtime_identity: sortformer_runtime_identity_state.flatten(),
        common_complete_recording_count,
        common_complete_observation_set_sha256: format!("{:x}", common_complete_hasher.finalize()),
        order_balance_complete: bundle.references.len() % MODEL_COMPARISON_WILLIAMS_SCHEDULE.len()
            == 0,
        protocol,
        protocol_sha256,
        lanes,
        development_uncertified: true,
        comparison_authority: "diagnostic_only".to_owned(),
        superiority_claim_permitted: false,
        production_route_changed: false,
        deterministic_accuracy_sha256: String::new(),
        result_sha256: String::new(),
    };
    evidence.deterministic_accuracy_sha256 = deterministic_accuracy_sha256(&evidence)?;
    evidence.result_sha256 = super::canonical_sha256(&evidence)?;
    verify_public_model_comparison_bundle_identity_pair(&bundle, &evidence)?;

    let mut cancellation_adapter = || is_cancelled();
    let staged_bundle = super::stage_new_json(
        bundle_output_path,
        &bundle_output_parent,
        &bundle,
        "public-corpus bundle",
        &mut cancellation_adapter,
    )?;
    let staged_evidence = match super::stage_new_json(
        evidence_output_path,
        &evidence_output_parent,
        &evidence,
        "model-comparison evidence",
        &mut cancellation_adapter,
    ) {
        Ok(staged) => staged,
        Err(error) => {
            return Err(super::staged_scrubbed_error(
                error,
                &[(&staged_bundle, "public-corpus bundle")],
            ));
        }
    };
    if let Err(error) = super::checkpoint_cancelled(&mut cancellation_adapter) {
        return Err(super::staged_scrubbed_error(
            error,
            &[
                (&staged_bundle, "public-corpus bundle"),
                (&staged_evidence, "model-comparison evidence"),
            ],
        ));
    }
    if let Err(error) = super::publish_staged_json(staged_bundle, "public-corpus bundle") {
        return Err(super::staged_scrubbed_error(
            error,
            &[(&staged_evidence, "model-comparison evidence")],
        ));
    }
    super::publish_staged_json(staged_evidence, "model-comparison evidence")?;
    Ok(evidence)
}

fn comparison_scorer_config() -> DiarizationScorerConfig {
    DiarizationScorerConfig {
        schema_version: crate::diarization::DIARIZATION_SCORER_CONFIG_SCHEMA_VERSION.to_owned(),
        speaker_boundary_collar_ms: 250,
        change_boundary_collar_ms: 250,
        overlap_policy: EvaluationOverlapPolicy::Exclude,
        calibration_bins: 10,
        count_top_k: 3,
        count_credible_mass_millionths: 900_000,
        dominant_speaker_collapse_share_millionths: 990_000,
        minimum_reference_speaker_recall_millionths: 100_000,
        minimum_effective_occupancy_ms: 250,
    }
}

fn frozen_model_comparison_protocol() -> FwResult<PublicModelComparisonProtocol> {
    let scorer_config = comparison_scorer_config();
    let native_acoustic_request = comparison_diarization_request(DiarizationEngine::Acoustic);
    let native_ecapa_request = comparison_diarization_request(DiarizationEngine::Ecapa);
    let native_ecapa_fused_request = comparison_diarization_request(DiarizationEngine::EcapaFused);
    let sortformer_contract = sortformer_oracle_contract();
    let sortformer_max_speakers = u16::try_from(SORTFORMER_ORACLE_MAX_SPEAKERS).map_err(|_| {
        model_comparison_error(
            "sortformer_contract",
            "Sortformer speaker capacity exceeds the retained protocol range",
        )
    })?;
    Ok(PublicModelComparisonProtocol {
        schema_version: PUBLIC_MODEL_COMPARISON_PROTOCOL_VERSION.to_owned(),
        runner_version: PUBLIC_MODEL_COMPARISON_RUNNER_VERSION.to_owned(),
        schedule_version: PUBLIC_MODEL_COMPARISON_SCHEDULE_VERSION.to_owned(),
        lanes: ModelComparisonLane::ALL.to_vec(),
        schedule: MODEL_COMPARISON_WILLIAMS_SCHEDULE.to_vec(),
        full_recordings_only: true,
        input_sample_rate_hz: 16_000,
        input_channels: 1,
        input_bits_per_sample: 16,
        input_sample_format: "signed_integer_pcm_little_endian".to_owned(),
        normalized_observation_binding:
            "same_validated_wav_bytes+fresh_worker_decode_f32+audio_and_executable_hash_before_and_after+full_duration_floor_ms".to_owned(),
        speaker_count_policy: "infer_without_oracle_count_input".to_owned(),
        oracle_count_diagnostic_present: false,
        sortformer_capacity_eligibility:
            "reference_count_over_4_is_declared_ineligible_and_never_passed_to_model".to_owned(),
        speech_activity_authority: "end_to_end_no_reference_sad_no_asr_boundaries".to_owned(),
        overlap_policy: "exclude".to_owned(),
        speaker_boundary_collar_ms: scorer_config.speaker_boundary_collar_ms,
        change_boundary_collar_ms: scorer_config.change_boundary_collar_ms,
        scorer_config_sha256: super::canonical_sha256(&scorer_config)?,
        outcome_taxonomy_version: PUBLIC_MODEL_COMPARISON_OUTCOME_TAXONOMY_VERSION.to_owned(),
        outcome_codes: ModelComparisonOutcomeCode::ALL.to_vec(),
        native_acoustic_request_sha256: super::canonical_sha256(&native_acoustic_request)?,
        native_ecapa_request_sha256: super::canonical_sha256(&native_ecapa_request)?,
        native_ecapa_fused_request_sha256: super::canonical_sha256(
            &native_ecapa_fused_request,
        )?,
        native_sortformer_package_sha256:
            crate::sortformer_conformance::SORTFORMER_PACKAGE_SHA256.to_owned(),
        native_sortformer_receipt_sha256:
            crate::sortformer_conformance::SORTFORMER_CONVERSION_RECEIPT_SHA256.to_owned(),
        native_rayon_threads: sortformer_contract.torch_intraop_threads,
        sortformer_intraop_threads: sortformer_contract.torch_intraop_threads,
        sortformer_interop_threads: sortformer_contract.torch_interop_threads,
        sortformer_max_speakers,
        sortformer_output_frame_ms: SORTFORMER_ORACLE_OUTPUT_FRAME_MS,
        attempt_hard_timeout_seconds: PUBLIC_MODEL_COMPARISON_ATTEMPT_TIMEOUT_SECONDS,
        worker_schema_version: MODEL_COMPARISON_WORKER_SCHEMA_VERSION.to_owned(),
        worker_process_policy: "one_fresh_process_per_lane_observation+attempt_deadline_covers_parent_request_worker_and_post_identity+outer_process_group_descendant_termination+nested_adapter_group_inheritance+live_size_bounded_nonblocking_pipe_capture_on_linux_android_apple+live_size_bounded_anonymous_file_capture_on_other_supported_platforms+fail_closed_without_recursive_termination+executable_audio_normalized_pcm_reference_protocol_scorer_identity_binding"
            .to_owned(),
        worker_stdout_limit_bytes: crate::process::MAX_CAPTURED_OUTPUT_BYTES as u64,
        cancellation_probe_delay_ms: MODEL_COMPARISON_CANCEL_PROBE_DELAY_MS,
        cancellation_probe_timeout_seconds: MODEL_COMPARISON_CANCEL_PROBE_TIMEOUT_SECONDS,
        cancellation_probe_self_limit_seconds: MODEL_COMPARISON_CANCEL_PROBE_SELF_LIMIT_SECONDS,
        native_real_time_factor_cap_millionths:
            MODEL_COMPARISON_NATIVE_RTF_CAP_MILLIONTHS,
        external_real_time_factor_cap_millionths:
            MODEL_COMPARISON_EXTERNAL_RTF_CAP_MILLIONTHS,
        peak_rss_cap_bytes: MODEL_COMPARISON_PEAK_RSS_CAP_BYTES,
        process_tree_rss_minimum_sample_interval_ms:
            MODEL_COMPARISON_PROCESS_TREE_RSS_MINIMUM_SAMPLE_INTERVAL_MS,
        cancellation_latency_cap_ms: MODEL_COMPARISON_CANCELLATION_LATENCY_CAP_MS,
        ecapa_contract_sha256: ECAPA_CONTRACT_SHA256.to_owned(),
        ecapa_package_sha256: ECAPA_PACKAGE_SHA256.to_owned(),
        sortformer_contract_sha256: SORTFORMER_ORACLE_CONTRACT_SHA256.to_owned(),
        native_acoustic_postprocessing:
            "fixed_safe_v1_change+fixed_safe_v1_clustering+unknown_fallback".to_owned(),
        native_ecapa_postprocessing:
            "fixed_safe_v1_change+probabilistic_v1_clustering+unknown_fallback".to_owned(),
        native_ecapa_fused_postprocessing:
            "fixed_safe_v1_change+probabilistic_v1_clustering+unknown_fallback".to_owned(),
        sortformer_postprocessing: "native_l8_and_pinned_external_sortformer_oracle_contract_v2"
            .to_owned(),
        wall_time_policy: "fresh_process_per_lane_and_observation;process_launch+bounded_ipc+identity_validation+audio_decode+model_load+inference+output_validation+scorer+parent_post_identity+resource_probe;matched_timeout_and_thread_policy;cross_lane_comparable;peak_rss_is_maximum_concurrently_sampled_sum_across_worker_process_group_including_descendant_adapters".to_owned(),
        aggregate_only: true,
        production_route_changed: false,
    })
}

fn measure_worker_cancellation_latency<F>(
    executable: &Path,
    expected_executable_sha256: &str,
    is_cancelled: &F,
) -> FwResult<u64>
where
    F: Fn() -> bool + Sync,
{
    cancellation_checkpoint(is_cancelled)?;
    let executable_text = executable.to_str().ok_or_else(|| {
        model_comparison_error(
            "worker_executable",
            "the current executable path is not valid UTF-8",
        )
    })?;
    let cancel_after = Duration::from_millis(MODEL_COMPARISON_CANCEL_PROBE_DELAY_MS);
    let pre_spawn_probe_seen = std::sync::atomic::AtomicBool::new(false);
    let post_spawn_started = std::sync::OnceLock::new();
    let cancel_probe = || {
        if !pre_spawn_probe_seen.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return is_cancelled();
        }
        if is_cancelled() {
            return true;
        }
        post_spawn_started.get_or_init(Instant::now).elapsed() >= cancel_after
    };
    let cancellation = CancellationToken::unbounded();
    let mut rss_sampler = ProcessTreeRssSampler::new();
    let result = {
        let mut observer = |root_pid| rss_sampler.observe(root_pid, &cancel_probe);
        crate::process::run_command_cancellable_with_input_probe_and_observer(
            executable_text,
            &["__comparison-cancel-probe".to_owned()],
            &cancellation,
            Some(Duration::from_secs(
                MODEL_COMPARISON_CANCEL_PROBE_TIMEOUT_SECONDS,
            )),
            Some(&cancel_probe),
            &[],
            &mut observer,
        )
    };
    if is_cancelled() {
        return Err(FwError::Cancelled(
            "public model comparison cancelled".to_owned(),
        ));
    }
    if !matches!(result, Err(FwError::Cancelled(_))) {
        return Err(model_comparison_error(
            "worker_cancellation_probe",
            "the fresh-process cancellation probe did not terminate by cancellation",
        ));
    }
    if rss_sampler.finish()?.is_none() {
        return Err(model_comparison_error(
            "worker_cancellation_probe",
            "the live cancellation path did not retain a whole-process-tree RSS sample",
        ));
    }
    let total_ms = u64::try_from(
        post_spawn_started
            .get()
            .ok_or_else(|| {
                model_comparison_error(
                    "worker_cancellation_probe",
                    "the cancellation timer did not start after the probe process launched",
                )
            })?
            .elapsed()
            .as_millis(),
    )
    .unwrap_or(u64::MAX);
    let executable_sha256_after = hash_bounded_file_with_cancel(
        executable,
        MAX_MODEL_COMPARISON_EXECUTABLE_BYTES,
        is_cancelled,
        "worker_executable",
    )?;
    if executable_sha256_after != expected_executable_sha256 {
        return Err(model_comparison_error(
            "worker_executable_changed",
            "the comparison executable changed during the cancellation probe",
        ));
    }
    Ok(total_ms
        .saturating_sub(MODEL_COMPARISON_CANCEL_PROBE_DELAY_MS)
        .max(1))
}

/// Run a process-tree probe until the comparison parent kills the worker group.
/// The root probe launches one copy of itself so the measured path proves that
/// inherited descendants do not keep pipes or work alive after cancellation.
pub fn run_model_comparison_cancel_probe(descendant_mode: bool) -> FwResult<()> {
    let started = Instant::now();
    let self_limit = Duration::from_secs(MODEL_COMPARISON_CANCEL_PROBE_SELF_LIMIT_SECONDS);
    if descendant_mode {
        while started.elapsed() < self_limit {
            std::thread::sleep(Duration::from_secs(1));
        }
        return Err(model_comparison_error(
            "worker_cancellation_probe",
            "the cancellation-probe descendant reached its standalone safety limit",
        ));
    }
    let executable = std::env::current_exe().map_err(|_| {
        model_comparison_error(
            "worker_cancellation_probe",
            "the cancellation-probe executable could not be resolved",
        )
    })?;
    let mut descendant = std::process::Command::new(executable)
        .arg("__comparison-cancel-probe")
        .arg("--descendant")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|_| {
            model_comparison_error(
                "worker_cancellation_probe",
                "the cancellation-probe descendant could not be launched",
            )
        })?;
    loop {
        if started.elapsed() >= self_limit {
            let _ = descendant.kill();
            let _ = descendant.wait();
            return Err(model_comparison_error(
                "worker_cancellation_probe",
                "the cancellation-probe root reached its standalone safety limit",
            ));
        }
        match descendant.try_wait() {
            Ok(Some(_)) => {
                return Err(model_comparison_error(
                    "worker_cancellation_probe",
                    "the cancellation-probe descendant exited before cancellation",
                ));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(_) => {
                let _ = descendant.kill();
                let _ = descendant.wait();
                return Err(model_comparison_error(
                    "worker_cancellation_probe",
                    "the cancellation-probe descendant could not be monitored",
                ));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_lane_worker<F>(
    lane: ModelComparisonLane,
    executable: &Path,
    executable_sha256: &str,
    audio_path: &Path,
    audio_sha256: &str,
    normalized_input_sha256: &str,
    reference: &crate::diarization::DiarizationReferenceDocument,
    reference_sha256: &str,
    reference_speaker_count: u64,
    protocol_sha256: &str,
    scorer_config_sha256: &str,
    hard_timeout: Duration,
    is_cancelled: &F,
) -> FwResult<InternalLaneOutcome>
where
    F: Fn() -> bool + Sync,
{
    let attempt_started = Instant::now();
    let request = ModelComparisonWorkerRequest {
        schema_version: MODEL_COMPARISON_WORKER_SCHEMA_VERSION.to_owned(),
        lane,
        audio_path: audio_path.to_path_buf(),
        expected_audio_sha256: audio_sha256.to_owned(),
        expected_normalized_input_sha256: normalized_input_sha256.to_owned(),
        reference: reference.clone(),
        expected_reference_sha256: reference_sha256.to_owned(),
        reference_speaker_count,
        protocol_sha256: protocol_sha256.to_owned(),
        scorer_config_sha256: scorer_config_sha256.to_owned(),
        executable_sha256: executable_sha256.to_owned(),
        hard_timeout_seconds: hard_timeout.as_secs(),
    };
    let payload = serde_json::to_vec(&request).map_err(|_| {
        model_comparison_error(
            "worker_request",
            "the fresh-process worker request could not be serialized",
        )
    })?;
    if payload.len() as u64 > MAX_MODEL_COMPARISON_WORKER_REQUEST_BYTES {
        return Err(model_comparison_error(
            "worker_request",
            "the fresh-process worker request exceeds its fixed byte bound",
        ));
    }
    cancellation_checkpoint(is_cancelled)?;
    let Some(worker_timeout) = remaining_attempt_budget(hard_timeout, attempt_started.elapsed())
    else {
        return Ok(worker_timeout_outcome(lane, attempt_started));
    };

    let executable_text = executable.to_str().ok_or_else(|| {
        model_comparison_error(
            "worker_executable",
            "the current executable path is not valid UTF-8",
        )
    })?;
    let program = executable_text.to_owned();
    let args = vec!["__comparison-worker".to_owned()];
    let cancellation = CancellationToken::unbounded();
    let mut rss_sampler = ProcessTreeRssSampler::new();
    let command_result = {
        let mut observer = |root_pid| rss_sampler.observe(root_pid, is_cancelled);
        crate::process::run_command_cancellable_with_input_probe_and_observer(
            &program,
            &args,
            &cancellation,
            Some(worker_timeout),
            Some(is_cancelled),
            &payload,
            &mut observer,
        )
    };
    let output = match command_result {
        Ok(output) => output,
        Err(error @ FwError::Cancelled(_)) => return Err(error),
        Err(FwError::CommandTimedOut { .. }) => {
            let _timed_out = verify_parent_worker_inputs_for_attempt(
                executable,
                executable_sha256,
                audio_path,
                audio_sha256,
                attempt_started,
                hard_timeout,
                is_cancelled,
            )?;
            return Ok(worker_timeout_outcome(lane, attempt_started));
        }
        Err(_) => {
            let timed_out = verify_parent_worker_inputs_for_attempt(
                executable,
                executable_sha256,
                audio_path,
                audio_sha256,
                attempt_started,
                hard_timeout,
                is_cancelled,
            )?;
            if timed_out {
                return Ok(worker_timeout_outcome(lane, attempt_started));
            }
            if rss_sampler.failed() {
                let mut outcome = InternalLaneOutcome::failed(
                    lane,
                    ModelComparisonOutcomeCode::WorkerResourceProbeFailed,
                );
                outcome.attempt_wall_time_ms = elapsed_millis(attempt_started);
                return Ok(outcome);
            }
            let mut outcome = InternalLaneOutcome::failed(
                lane,
                ModelComparisonOutcomeCode::WorkerExecutionFailed,
            );
            outcome.attempt_wall_time_ms = elapsed_millis(attempt_started);
            return Ok(outcome);
        }
    };
    let response: ModelComparisonWorkerResponse = match serde_json::from_slice(&output.stdout) {
        Ok(response) => response,
        Err(_) => {
            let timed_out = verify_parent_worker_inputs_for_attempt(
                executable,
                executable_sha256,
                audio_path,
                audio_sha256,
                attempt_started,
                hard_timeout,
                is_cancelled,
            )?;
            if timed_out {
                return Ok(worker_timeout_outcome(lane, attempt_started));
            }
            let mut outcome = InternalLaneOutcome::failed(
                lane,
                ModelComparisonOutcomeCode::WorkerMalformedOutput,
            );
            outcome.attempt_wall_time_ms = elapsed_millis(attempt_started);
            return Ok(outcome);
        }
    };
    let timed_out = verify_parent_worker_inputs_for_attempt(
        executable,
        executable_sha256,
        audio_path,
        audio_sha256,
        attempt_started,
        hard_timeout,
        is_cancelled,
    )?;
    if timed_out {
        return Ok(worker_timeout_outcome(lane, attempt_started));
    }
    let parent_elapsed_after_identity_ms = elapsed_millis(attempt_started);
    let ecapa_identity_expected = matches!(
        lane,
        ModelComparisonLane::NativeEcapa | ModelComparisonLane::NativeEcapaFused
    ) && !matches!(
        response.outcome.code,
        Some(
            ModelComparisonOutcomeCode::EcapaModelUnavailable
                | ModelComparisonOutcomeCode::EcapaModelInvalid
        )
    );
    let native_sortformer_identity_expected = lane == ModelComparisonLane::NativeSortformer
        && !matches!(
            response.outcome.code,
            Some(
                ModelComparisonOutcomeCode::NativeSortformerModelUnavailable
                    | ModelComparisonOutcomeCode::NativeSortformerModelInvalid
                    | ModelComparisonOutcomeCode::SortformerModelCapacityExceeded
            )
        );
    if response.schema_version != MODEL_COMPARISON_WORKER_SCHEMA_VERSION
        || response.lane != lane
        || response.outcome.lane != lane
        || response.executable_sha256_before != executable_sha256
        || response.executable_sha256_after != executable_sha256
        || response.audio_sha256_before != audio_sha256
        || response.audio_sha256_after != audio_sha256
        || response.normalized_input_sha256 != normalized_input_sha256
        || response.reference_sha256 != reference_sha256
        || response.protocol_sha256 != protocol_sha256
        || response.scorer_config_sha256 != scorer_config_sha256
        || response.worker_wall_time_ms == 0
        || response.worker_wall_time_ms > parent_elapsed_after_identity_ms
        || response.outcome.attempt_wall_time_ms != 0
        || response.outcome.worker_peak_rss_bytes.is_some()
        || response.ecapa_package_sha256_after.as_deref()
            != ecapa_identity_expected.then_some(ECAPA_PACKAGE_SHA256)
        || response.native_sortformer_package_sha256_after.as_deref()
            != native_sortformer_identity_expected
                .then_some(crate::sortformer_conformance::SORTFORMER_PACKAGE_SHA256)
        || response.native_sortformer_receipt_sha256_after.as_deref()
            != native_sortformer_identity_expected
                .then_some(crate::sortformer_conformance::SORTFORMER_CONVERSION_RECEIPT_SHA256)
    {
        return Err(model_comparison_error(
            "worker_identity",
            "a fresh-process comparison worker contradicted its bound identities",
        ));
    }
    let peak_rss_bytes = match rss_sampler.finish() {
        Ok(value) => value,
        Err(_) => {
            if remaining_attempt_budget(hard_timeout, attempt_started.elapsed()).is_none() {
                return Ok(worker_timeout_outcome(lane, attempt_started));
            }
            let mut outcome = InternalLaneOutcome::failed(
                lane,
                ModelComparisonOutcomeCode::WorkerResourceProbeFailed,
            );
            outcome.attempt_wall_time_ms = elapsed_millis(attempt_started);
            return Ok(outcome);
        }
    };
    if remaining_attempt_budget(hard_timeout, attempt_started.elapsed()).is_none() {
        return Ok(worker_timeout_outcome(lane, attempt_started));
    }
    let mut outcome = response.outcome;
    // The authoritative lane time is measured by the parent and therefore
    // includes process launch, bounded IPC, the complete worker, response
    // validation, post-run identity hashing, and the resource probe.
    outcome.attempt_wall_time_ms = elapsed_millis(attempt_started);
    outcome.worker_peak_rss_bytes = peak_rss_bytes;
    Ok(outcome)
}

fn remaining_attempt_budget(limit: Duration, elapsed: Duration) -> Option<Duration> {
    limit
        .checked_sub(elapsed)
        .filter(|remaining| !remaining.is_zero())
}

fn worker_timeout_outcome(
    lane: ModelComparisonLane,
    attempt_started: Instant,
) -> InternalLaneOutcome {
    let mut outcome = InternalLaneOutcome::failed(lane, ModelComparisonOutcomeCode::WorkerTimedOut);
    outcome.attempt_wall_time_ms = elapsed_millis(attempt_started);
    outcome
}

struct ProcessTreeRssSampler {
    supported: bool,
    failed: bool,
    last_sample: Option<Instant>,
    consecutive_missing_samples: u8,
    maximum_bytes: Option<u64>,
}

impl ProcessTreeRssSampler {
    fn new() -> Self {
        #[cfg(target_vendor = "apple")]
        let supported = Path::new("/bin/ps").is_file();
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let supported = Path::new("/proc").is_dir();
        #[cfg(not(any(target_vendor = "apple", target_os = "linux", target_os = "android")))]
        let supported = false;
        Self {
            supported,
            failed: false,
            last_sample: None,
            consecutive_missing_samples: 0,
            maximum_bytes: None,
        }
    }

    fn observe(&mut self, root_pid: u32, is_cancelled: &(dyn Fn() -> bool + Sync)) -> FwResult<()> {
        if !self.supported
            || self.last_sample.is_some_and(|sampled_at| {
                sampled_at.elapsed()
                    < Duration::from_millis(
                        MODEL_COMPARISON_PROCESS_TREE_RSS_MINIMUM_SAMPLE_INTERVAL_MS,
                    )
            })
        {
            return Ok(());
        }
        self.last_sample = Some(Instant::now());
        let sample = sample_process_group_rss_bytes(root_pid, is_cancelled);
        self.retain_observation(sample)
    }

    fn retain_observation(&mut self, sample: FwResult<Option<u64>>) -> FwResult<()> {
        match sample {
            Ok(Some(bytes)) if bytes > 0 => {
                self.consecutive_missing_samples = 0;
                self.maximum_bytes = Some(self.maximum_bytes.unwrap_or(0).max(bytes));
                Ok(())
            }
            Ok(None) => {
                self.consecutive_missing_samples =
                    self.consecutive_missing_samples.saturating_add(1);
                if self.consecutive_missing_samples < 2 {
                    return Ok(());
                }
                self.failed = true;
                Err(model_comparison_error(
                    "worker_resource",
                    "the whole-process-tree RSS probe repeatedly lost the live worker group",
                ))
            }
            Err(error @ FwError::Cancelled(_)) => Err(error),
            Ok(Some(_)) | Err(_) => {
                self.failed = true;
                Err(model_comparison_error(
                    "worker_resource",
                    "the whole-process-tree RSS probe could not produce a complete positive sample",
                ))
            }
        }
    }

    const fn failed(&self) -> bool {
        self.failed
    }

    fn finish(&self) -> FwResult<Option<u64>> {
        if self.failed {
            return Err(model_comparison_error(
                "worker_resource",
                "the whole-process-tree RSS probe did not retain a complete sample",
            ));
        }
        Ok(self.maximum_bytes)
    }
}

#[cfg(target_vendor = "apple")]
fn sample_process_group_rss_bytes(
    root_pid: u32,
    is_cancelled: &(dyn Fn() -> bool + Sync),
) -> FwResult<Option<u64>> {
    let cancellation = CancellationToken::unbounded();
    let output = crate::process::run_command_cancellable_with_probe(
        "/bin/ps",
        &[
            "-o".to_owned(),
            "rss=".to_owned(),
            "-g".to_owned(),
            root_pid.to_string(),
        ],
        None,
        &cancellation,
        Some(Duration::from_millis(
            MODEL_COMPARISON_PROCESS_TREE_RSS_PROBE_TIMEOUT_MS,
        )),
        Some(is_cancelled),
    )
    .map_err(|error| match error {
        error @ FwError::Cancelled(_) => error,
        _ => model_comparison_error(
            "worker_resource",
            "the bounded macOS RSS probe could not complete",
        ),
    })?;
    parse_process_group_rss_kib(&output.stdout)?.map_or(Ok(None), |kib| {
        kib.checked_mul(1_024).map(Some).ok_or_else(|| {
            model_comparison_error("worker_resource", "the macOS RSS sample overflowed bytes")
        })
    })
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn sample_process_group_rss_bytes(
    root_pid: u32,
    is_cancelled: &(dyn Fn() -> bool + Sync),
) -> FwResult<Option<u64>> {
    let started = Instant::now();
    let mut total_kib = 0u64;
    let mut matched = false;
    let entries = std::fs::read_dir("/proc").map_err(|_| {
        model_comparison_error(
            "worker_resource",
            "the procfs RSS probe could not enumerate",
        )
    })?;
    for entry in entries {
        cancellation_checkpoint(is_cancelled)?;
        if started.elapsed()
            >= Duration::from_millis(MODEL_COMPARISON_PROCESS_TREE_RSS_PROBE_TIMEOUT_MS)
        {
            return Err(model_comparison_error(
                "worker_resource",
                "the procfs RSS probe exceeded its fixed observation budget",
            ));
        }
        let entry = entry.map_err(|_| {
            model_comparison_error(
                "worker_resource",
                "the procfs RSS probe encountered an unreadable directory entry",
            )
        })?;
        let Some(_pid) = entry
            .file_name()
            .to_str()
            .and_then(|value| value.parse::<u32>().ok())
        else {
            continue;
        };
        let process_root = entry.path();
        let Ok(stat) = std::fs::read_to_string(process_root.join("stat")) else {
            continue;
        };
        if linux_stat_process_group(&stat) != Some(root_pid) {
            continue;
        }
        let status = std::fs::read_to_string(process_root.join("status")).map_err(|_| {
            model_comparison_error(
                "worker_resource",
                "a matched procfs process could not provide status",
            )
        })?;
        let rss_kib = linux_status_rss_kib(&status).ok_or_else(|| {
            model_comparison_error(
                "worker_resource",
                "a matched procfs process did not provide positive VmRSS",
            )
        })?;
        total_kib = total_kib.checked_add(rss_kib).ok_or_else(|| {
            model_comparison_error("worker_resource", "the procfs RSS sum overflowed")
        })?;
        matched = true;
    }
    if !matched || total_kib == 0 {
        return Ok(None);
    }
    total_kib.checked_mul(1_024).map(Some).ok_or_else(|| {
        model_comparison_error("worker_resource", "the procfs RSS sample overflowed bytes")
    })
}

#[cfg(not(any(target_vendor = "apple", target_os = "linux", target_os = "android")))]
fn sample_process_group_rss_bytes(
    _root_pid: u32,
    _is_cancelled: &(dyn Fn() -> bool + Sync),
) -> FwResult<Option<u64>> {
    Ok(None)
}

#[cfg(any(test, target_vendor = "apple"))]
fn parse_process_group_rss_kib(stdout: &[u8]) -> FwResult<Option<u64>> {
    let text = std::str::from_utf8(stdout).map_err(|_| {
        model_comparison_error("worker_resource", "the RSS probe emitted non-UTF-8 output")
    })?;
    let mut total = 0u64;
    let mut observed = false;
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let value = line.parse::<u64>().map_err(|_| {
            model_comparison_error(
                "worker_resource",
                "the RSS probe emitted a non-integer process sample",
            )
        })?;
        total = total.checked_add(value).ok_or_else(|| {
            model_comparison_error("worker_resource", "the RSS probe sum overflowed")
        })?;
        observed = true;
    }
    Ok((observed && total > 0).then_some(total))
}

#[cfg(any(test, target_os = "linux", target_os = "android"))]
fn linux_stat_process_group(stat: &str) -> Option<u32> {
    let after_command = stat.get(stat.rfind(") ")? + 2..)?;
    after_command.split_whitespace().nth(2)?.parse().ok()
}

#[cfg(any(test, target_os = "linux", target_os = "android"))]
fn linux_status_rss_kib(status: &str) -> Option<u64> {
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let mut fields = line.split_whitespace();
    if fields.next()? != "VmRSS:" {
        return None;
    }
    let value = fields.next()?.parse::<u64>().ok()?;
    (fields.next()? == "kB" && fields.next().is_none() && value > 0).then_some(value)
}

fn verify_parent_worker_inputs<F>(
    executable: &Path,
    expected_executable_sha256: &str,
    audio_path: &Path,
    expected_audio_sha256: &str,
    is_cancelled: &F,
) -> FwResult<()>
where
    F: Fn() -> bool + Sync,
{
    let executable_sha256 = hash_bounded_file_with_cancel(
        executable,
        MAX_MODEL_COMPARISON_EXECUTABLE_BYTES,
        is_cancelled,
        "worker_executable",
    )?;
    let audio_sha256 = hash_bounded_file_with_cancel(
        audio_path,
        super::MAX_EVALUATION_AUDIO_BYTES,
        is_cancelled,
        "worker_audio",
    )?;
    if executable_sha256 != expected_executable_sha256 || audio_sha256 != expected_audio_sha256 {
        return Err(model_comparison_error(
            "worker_identity_changed",
            "a bound worker input changed during an attempt",
        ));
    }
    Ok(())
}

fn verify_parent_worker_inputs_for_attempt<F>(
    executable: &Path,
    expected_executable_sha256: &str,
    audio_path: &Path,
    expected_audio_sha256: &str,
    attempt_started: Instant,
    hard_timeout: Duration,
    is_cancelled: &F,
) -> FwResult<bool>
where
    F: Fn() -> bool + Sync,
{
    let stop = || {
        is_cancelled()
            || remaining_attempt_budget(hard_timeout, attempt_started.elapsed()).is_none()
    };
    match verify_parent_worker_inputs(
        executable,
        expected_executable_sha256,
        audio_path,
        expected_audio_sha256,
        &stop,
    ) {
        Ok(()) => Ok(remaining_attempt_budget(hard_timeout, attempt_started.elapsed()).is_none()),
        Err(FwError::Cancelled(_)) if is_cancelled() => Err(FwError::Cancelled(
            "public model comparison cancelled".to_owned(),
        )),
        Err(FwError::Cancelled(_))
            if remaining_attempt_budget(hard_timeout, attempt_started.elapsed()).is_none() =>
        {
            Ok(true)
        }
        Err(error) => Err(error),
    }
}

/// Execute exactly one hidden comparison lane from a bounded stdin request.
/// The response contains no filesystem paths, recording identifier, turns, or
/// speaker labels and is intended only for the parent comparison invocation.
pub fn run_model_comparison_worker_from_stdio() -> FwResult<String> {
    let mut request_bytes = Vec::new();
    std::io::stdin()
        .take(MAX_MODEL_COMPARISON_WORKER_REQUEST_BYTES.saturating_add(1))
        .read_to_end(&mut request_bytes)
        .map_err(|_| {
            model_comparison_error(
                "worker_request",
                "the fresh-process worker could not read its request",
            )
        })?;
    if request_bytes.len() as u64 > MAX_MODEL_COMPARISON_WORKER_REQUEST_BYTES {
        return Err(model_comparison_error(
            "worker_request",
            "the fresh-process worker request exceeds its fixed byte bound",
        ));
    }
    let request: ModelComparisonWorkerRequest =
        serde_json::from_slice(&request_bytes).map_err(|_| {
            model_comparison_error(
                "worker_request",
                "the fresh-process worker request is malformed",
            )
        })?;
    if request.schema_version != MODEL_COMPARISON_WORKER_SCHEMA_VERSION
        || !super::is_sha256_hex(&request.expected_audio_sha256)
        || !super::is_sha256_hex(&request.expected_normalized_input_sha256)
        || !super::is_sha256_hex(&request.expected_reference_sha256)
        || !super::is_sha256_hex(&request.protocol_sha256)
        || !super::is_sha256_hex(&request.scorer_config_sha256)
        || !super::is_sha256_hex(&request.executable_sha256)
        || request.hard_timeout_seconds != PUBLIC_MODEL_COMPARISON_ATTEMPT_TIMEOUT_SECONDS
        || !request.audio_path.is_absolute()
    {
        return Err(model_comparison_error(
            "worker_request",
            "the fresh-process worker request violates its frozen contract",
        ));
    }

    let started = Instant::now();
    let is_cancelled = crate::cli::ShutdownController::is_shutting_down;
    let executable = std::env::current_exe()
        .and_then(std::fs::canonicalize)
        .map_err(|_| {
            model_comparison_error(
                "worker_executable",
                "the fresh-process worker executable could not be resolved",
            )
        })?;
    let executable_sha256_before = hash_bounded_file_with_cancel(
        &executable,
        MAX_MODEL_COMPARISON_EXECUTABLE_BYTES,
        &is_cancelled,
        "worker_executable",
    )?;
    if executable_sha256_before != request.executable_sha256 {
        return Err(model_comparison_error(
            "worker_executable_changed",
            "the fresh-process worker executable identity changed before execution",
        ));
    }
    let canonical_audio = std::fs::canonicalize(&request.audio_path).map_err(|_| {
        model_comparison_error(
            "worker_audio",
            "the fresh-process worker audio input could not be resolved",
        )
    })?;
    if canonical_audio != request.audio_path {
        return Err(model_comparison_error(
            "worker_audio",
            "the fresh-process worker audio input was not canonical",
        ));
    }
    let audio_bytes = super::read_bounded(
        &canonical_audio,
        super::MAX_EVALUATION_AUDIO_BYTES,
        "model_comparison_worker_audio",
    )?;
    let audio_sha256_before = format!("{:x}", Sha256::digest(&audio_bytes));
    if audio_sha256_before != request.expected_audio_sha256 {
        return Err(model_comparison_error(
            "worker_audio_changed",
            "the fresh-process worker audio identity changed before execution",
        ));
    }
    let samples = decode_pcm16_wave(&audio_bytes, &is_cancelled)?;
    drop(audio_bytes);
    let normalized_input_sha256 = super::hash_pcm_prefix(&samples);
    if normalized_input_sha256 != request.expected_normalized_input_sha256 {
        return Err(model_comparison_error(
            "worker_audio_changed",
            "the fresh-process worker normalized audio identity disagrees",
        ));
    }
    let reference_sha256 = super::canonical_sha256(&request.reference)?;
    let reference_speaker_count = count_labeled_speakers(&request.reference.turns)?;
    if reference_sha256 != request.expected_reference_sha256
        || reference_speaker_count != request.reference_speaker_count
    {
        return Err(model_comparison_error(
            "worker_reference",
            "the fresh-process worker reference identity disagrees",
        ));
    }
    let protocol = frozen_model_comparison_protocol()?;
    let protocol_sha256 = super::canonical_sha256(&protocol)?;
    let scorer_config = comparison_scorer_config();
    let scorer_config_sha256 = super::canonical_sha256(&scorer_config)?;
    if protocol_sha256 != request.protocol_sha256
        || scorer_config_sha256 != request.scorer_config_sha256
        || u16::try_from(rayon::current_num_threads()).ok() != Some(protocol.native_rayon_threads)
    {
        return Err(model_comparison_error(
            "worker_protocol",
            "the fresh-process worker runtime disagrees with the frozen protocol",
        ));
    }
    // The comparison parent launched this authenticated worker inside one
    // dedicated process group. Nested external adapters must inherit that
    // group so an outer timeout or cancellation cannot orphan them.
    crate::process::mark_process_tree_externally_owned()?;

    let ecapa = if matches!(
        request.lane,
        ModelComparisonLane::NativeEcapa | ModelComparisonLane::NativeEcapaFused
    ) {
        load_ecapa_availability(&is_cancelled)?
    } else {
        EcapaAvailability::Unavailable
    };
    let native_sortformer_capacity = u64::try_from(SORTFORMER_SPEAKER_LANES).map_err(|_| {
        model_comparison_error(
            "worker_protocol",
            "native Sortformer capacity exceeds the retained range",
        )
    })?;
    let native_sortformer = if request.lane == ModelComparisonLane::NativeSortformer
        && reference_speaker_count <= native_sortformer_capacity
    {
        load_native_sortformer_availability(&is_cancelled)?
    } else {
        NativeSortformerAvailability::Unavailable
    };
    let mut outcome = execute_lane(
        request.lane,
        &canonical_audio,
        &request.expected_audio_sha256,
        &samples,
        &normalized_input_sha256,
        &request.reference,
        reference_speaker_count,
        &scorer_config,
        &ecapa,
        &native_sortformer,
        Duration::from_secs(request.hard_timeout_seconds),
        &is_cancelled,
    )?;
    sanitize_worker_outcome(&mut outcome);

    let ecapa_package_sha256_after = match &ecapa {
        EcapaAvailability::Ready { package_path, .. } => Some(hash_bounded_file_with_cancel(
            package_path,
            super::MAX_EVALUATION_AUDIO_BYTES,
            &is_cancelled,
            "worker_ecapa_package",
        )?),
        EcapaAvailability::Unavailable | EcapaAvailability::Invalid => None,
    };
    let (native_sortformer_package_sha256_after, native_sortformer_receipt_sha256_after) =
        match &native_sortformer {
            NativeSortformerAvailability::Ready {
                package_path,
                receipt_path,
                ..
            } => (
                Some(hash_bounded_file_with_cancel(
                    package_path,
                    crate::sortformer_conformance::SORTFORMER_PACKAGE_BYTES,
                    &is_cancelled,
                    "worker_native_sortformer_package",
                )?),
                Some(hash_bounded_file_with_cancel(
                    receipt_path,
                    MAX_MODEL_COMPARISON_WORKER_REQUEST_BYTES,
                    &is_cancelled,
                    "worker_native_sortformer_receipt",
                )?),
            ),
            NativeSortformerAvailability::Unavailable | NativeSortformerAvailability::Invalid => {
                (None, None)
            }
        };
    if ecapa_package_sha256_after
        .as_deref()
        .is_some_and(|value| value != ECAPA_PACKAGE_SHA256)
        || native_sortformer_package_sha256_after
            .as_deref()
            .is_some_and(|value| value != crate::sortformer_conformance::SORTFORMER_PACKAGE_SHA256)
        || native_sortformer_receipt_sha256_after
            .as_deref()
            .is_some_and(|value| {
                value != crate::sortformer_conformance::SORTFORMER_CONVERSION_RECEIPT_SHA256
            })
    {
        return Err(model_comparison_error(
            "worker_model_identity_changed",
            "a fresh-process worker model artifact changed during execution",
        ));
    }

    let audio_sha256_after = hash_bounded_file_with_cancel(
        &canonical_audio,
        super::MAX_EVALUATION_AUDIO_BYTES,
        &is_cancelled,
        "worker_audio",
    )?;
    let executable_sha256_after = hash_bounded_file_with_cancel(
        &executable,
        MAX_MODEL_COMPARISON_EXECUTABLE_BYTES,
        &is_cancelled,
        "worker_executable",
    )?;
    if audio_sha256_after != request.expected_audio_sha256
        || executable_sha256_after != request.executable_sha256
    {
        return Err(model_comparison_error(
            "worker_identity_changed",
            "a fresh-process worker input identity changed during execution",
        ));
    }
    let response = ModelComparisonWorkerResponse {
        schema_version: MODEL_COMPARISON_WORKER_SCHEMA_VERSION.to_owned(),
        lane: request.lane,
        executable_sha256_before,
        executable_sha256_after,
        audio_sha256_before,
        audio_sha256_after,
        normalized_input_sha256,
        reference_sha256,
        protocol_sha256,
        scorer_config_sha256,
        ecapa_package_sha256_after,
        native_sortformer_package_sha256_after,
        native_sortformer_receipt_sha256_after,
        worker_wall_time_ms: elapsed_millis(started),
        outcome,
    };
    serde_json::to_string(&response).map_err(|_| {
        model_comparison_error(
            "worker_output",
            "the fresh-process worker response could not be serialized",
        )
    })
}

fn sanitize_worker_outcome(outcome: &mut InternalLaneOutcome) {
    let Some(score) = outcome.score.as_mut() else {
        return;
    };
    score.recording_id.clear();
    score.reference_sha256.clear();
    score.hypothesis_sha256.clear();
    score.result_sha256.clear();
    score.performance = None;
    score.diarization.speaker_mapping.clear();
    score.speaker_occupancy.speakers.clear();
}

fn hash_bounded_file_with_cancel<F>(
    path: &Path,
    maximum_bytes: u64,
    is_cancelled: &F,
    field: &str,
) -> FwResult<String>
where
    F: Fn() -> bool + Sync,
{
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|_| model_comparison_error(field, "a bound file could not be inspected"))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() > maximum_bytes {
        return Err(model_comparison_error(
            field,
            "a bound file is not a regular non-symlink file within its byte limit",
        ));
    }
    #[cfg(target_family = "unix")]
    let mut file = {
        use rustix::fs::{Mode, OFlags, open};

        let descriptor = open(
            path,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(|_| model_comparison_error(field, "a bound file could not be opened"))?;
        std::fs::File::from(descriptor)
    };
    #[cfg(windows)]
    let mut file = {
        use std::os::windows::fs::OpenOptionsExt as _;

        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)
            .map_err(|_| model_comparison_error(field, "a bound file could not be opened"))?
    };
    #[cfg(not(any(target_family = "unix", windows)))]
    let mut file = std::fs::File::open(path)
        .map_err(|_| model_comparison_error(field, "a bound file could not be opened"))?;
    let opened_metadata = file.metadata().map_err(|_| {
        model_comparison_error(field, "a bound file descriptor could not be inspected")
    })?;
    if !opened_metadata.is_file() || opened_metadata.len() != metadata.len() {
        return Err(model_comparison_error(
            field,
            "a bound file identity changed while opening",
        ));
    }
    #[cfg(target_family = "unix")]
    {
        use std::os::unix::fs::MetadataExt as _;

        if opened_metadata.dev() != metadata.dev() || opened_metadata.ino() != metadata.ino() {
            return Err(model_comparison_error(
                field,
                "a bound file identity changed while opening",
            ));
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt as _;

        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        if opened_metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(model_comparison_error(
                field,
                "a bound file descriptor resolves through a reparse point",
            ));
        }
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut observed = 0u64;
    loop {
        cancellation_checkpoint(is_cancelled)?;
        let count = file
            .read(&mut buffer)
            .map_err(|_| model_comparison_error(field, "a bound file could not be read"))?;
        if count == 0 {
            break;
        }
        observed = observed.saturating_add(count as u64);
        if observed > maximum_bytes {
            return Err(model_comparison_error(
                field,
                "a bound file grew beyond its byte limit while hashing",
            ));
        }
        hasher.update(&buffer[..count]);
    }
    let final_metadata = file.metadata().map_err(|_| {
        model_comparison_error(field, "a bound file descriptor could not be re-inspected")
    })?;
    if observed != metadata.len() || final_metadata.len() != metadata.len() {
        return Err(model_comparison_error(
            field,
            "a bound file changed length while hashing",
        ));
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn load_ecapa_availability<F>(is_cancelled: &F) -> FwResult<EcapaAvailability>
where
    F: Fn() -> bool + Sync,
{
    cancellation_checkpoint(is_cancelled)?;
    let model_path = match crate::native_engine::weights::resolve_aux_model(ECAPA_PACKAGE_FILENAME)
    {
        Ok(path) => path,
        Err(_) => return Ok(EcapaAvailability::Unavailable),
    };
    let checkpoint = || cancellation_checkpoint(is_cancelled);
    match EcapaModel::load_with_checkpoint(&model_path, &checkpoint) {
        Ok(model) => Ok(EcapaAvailability::Ready {
            model: Box::new(model),
            package_path: model_path,
        }),
        Err(error @ FwError::Cancelled(_)) => Err(error),
        Err(_) => Ok(EcapaAvailability::Invalid),
    }
}

fn load_native_sortformer_availability<F>(
    is_cancelled: &F,
) -> FwResult<NativeSortformerAvailability>
where
    F: Fn() -> bool + Sync,
{
    cancellation_checkpoint(is_cancelled)?;
    let cached = match resolve_cached_sortformer_with_cancel(|| is_cancelled()) {
        Ok(cached) => cached,
        Err(error @ FwError::Cancelled(_)) => return Err(error),
        Err(FwError::MissingArtifact(_)) => return Ok(NativeSortformerAvailability::Unavailable),
        Err(_) => return Ok(NativeSortformerAvailability::Invalid),
    };
    let checkpoint = || cancellation_checkpoint(is_cancelled);
    let package = match load_verified_sortformer_package_with_checkpoint(
        &cached.receipt_path,
        &cached.package_path,
        &checkpoint,
    ) {
        Ok(package) => package,
        Err(error @ FwError::Cancelled(_)) => return Err(error),
        Err(_) => return Ok(NativeSortformerAvailability::Invalid),
    };
    match SortformerSession::from_verified_package_with_checkpoint(&package, &checkpoint) {
        Ok(session) => Ok(NativeSortformerAvailability::Ready {
            session: Box::new(session),
            package_path: cached.package_path,
            receipt_path: cached.receipt_path,
        }),
        Err(error @ FwError::Cancelled(_)) => Err(error),
        Err(_) => Ok(NativeSortformerAvailability::Invalid),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_lane<F>(
    lane: ModelComparisonLane,
    audio_path: &Path,
    audio_sha256: &str,
    samples: &[f32],
    normalized_input_sha256: &str,
    reference: &crate::diarization::DiarizationReferenceDocument,
    reference_speaker_count: u64,
    scorer_config: &DiarizationScorerConfig,
    ecapa: &EcapaAvailability,
    native_sortformer: &NativeSortformerAvailability,
    attempt_hard_timeout: Duration,
    is_cancelled: &F,
) -> FwResult<InternalLaneOutcome>
where
    F: Fn() -> bool + Sync,
{
    let boundary_hints = AcousticBoundaryHints::default();
    match lane {
        ModelComparisonLane::NativeAcoustic => {
            let diarization_request = comparison_diarization_request(DiarizationEngine::Acoustic);
            let result = crate::diarization::diarize_acoustic_pcm(
                AcousticDiarizationInput {
                    samples,
                    normalized_input_sha256,
                    segments: &[],
                    word_aligned: false,
                    request: &diarization_request,
                    boundary_hints: &boundary_hints,
                },
                is_cancelled,
            );
            let (report, _) = match result {
                Ok(value) => value,
                Err(error @ FwError::Cancelled(_)) => return Err(error),
                Err(_) => {
                    return Ok(InternalLaneOutcome::failed(
                        lane,
                        ModelComparisonOutcomeCode::NativeExecutionFailed,
                    ));
                }
            };
            score_native_report(lane, report, reference, scorer_config)
        }
        ModelComparisonLane::NativeEcapa | ModelComparisonLane::NativeEcapaFused => {
            let model = match ecapa {
                EcapaAvailability::Ready { model, .. } => model,
                EcapaAvailability::Unavailable => {
                    return Ok(InternalLaneOutcome::skipped(
                        lane,
                        ModelComparisonOutcomeCode::EcapaModelUnavailable,
                    ));
                }
                EcapaAvailability::Invalid => {
                    return Ok(InternalLaneOutcome::failed(
                        lane,
                        ModelComparisonOutcomeCode::EcapaModelInvalid,
                    ));
                }
            };
            let engine = if lane == ModelComparisonLane::NativeEcapa {
                DiarizationEngine::Ecapa
            } else {
                DiarizationEngine::EcapaFused
            };
            let diarization_request = comparison_diarization_request(engine);
            let checkpoint = || cancellation_checkpoint(is_cancelled);
            let result = crate::diarization::diarize_ecapa_pcm(
                AcousticDiarizationInput {
                    samples,
                    normalized_input_sha256,
                    segments: &[],
                    word_aligned: false,
                    request: &diarization_request,
                    boundary_hints: &boundary_hints,
                },
                model,
                &checkpoint,
            );
            let (report, _) = match result {
                Ok(value) => value,
                Err(error @ FwError::Cancelled(_)) => return Err(error),
                Err(error) => {
                    let Some(code) = classify_ecapa_failure(&error) else {
                        return Err(error);
                    };
                    return Ok(InternalLaneOutcome::failed(lane, code));
                }
            };
            score_native_report(lane, report, reference, scorer_config)
        }
        ModelComparisonLane::NativeSortformer => {
            let capacity = u64::try_from(SORTFORMER_SPEAKER_LANES).map_err(|_| {
                model_comparison_error(
                    "native_sortformer_contract",
                    "native Sortformer speaker capacity exceeds the retained count range",
                )
            })?;
            if reference_speaker_count > capacity {
                return Ok(InternalLaneOutcome::skipped(
                    lane,
                    ModelComparisonOutcomeCode::SortformerModelCapacityExceeded,
                ));
            }
            let session = match native_sortformer {
                NativeSortformerAvailability::Ready { session, .. } => session,
                NativeSortformerAvailability::Unavailable => {
                    return Ok(InternalLaneOutcome::skipped(
                        lane,
                        ModelComparisonOutcomeCode::NativeSortformerModelUnavailable,
                    ));
                }
                NativeSortformerAvailability::Invalid => {
                    return Ok(InternalLaneOutcome::failed(
                        lane,
                        ModelComparisonOutcomeCode::NativeSortformerModelInvalid,
                    ));
                }
            };
            if SORTFORMER_SAMPLE_RATE_HZ != 16_000 {
                return Err(model_comparison_error(
                    "native_sortformer_contract",
                    "native Sortformer sample rate disagrees with the comparison protocol",
                ));
            }
            let checkpoint = || cancellation_checkpoint(is_cancelled);
            let output = match session
                .diarize_with_checkpoint(SortformerPcm::mono_16khz(samples), &checkpoint)
            {
                Ok(output) => output,
                Err(error @ FwError::Cancelled(_)) => return Err(error),
                Err(_) => {
                    return Ok(InternalLaneOutcome::failed(
                        lane,
                        ModelComparisonOutcomeCode::NativeSortformerExecutionFailed,
                    ));
                }
            };
            let (turns, selected_count) =
                sortformer_evaluation_turns(output.turns, reference.duration_ms)?;
            let count = count_observation(reference, &turns, Some(selected_count))?;
            match score_hypothesis(reference, turns, None, scorer_config) {
                Ok(score) => Ok(InternalLaneOutcome::completed(lane, score, None, count)),
                Err(_) => Ok(InternalLaneOutcome::failed(
                    lane,
                    ModelComparisonOutcomeCode::ScoringFailed,
                )),
            }
        }
        ModelComparisonLane::ExternalSortformer => {
            let sortformer_capacity =
                u64::try_from(SORTFORMER_ORACLE_MAX_SPEAKERS).map_err(|_| {
                    model_comparison_error(
                        "sortformer_contract",
                        "Sortformer speaker capacity exceeds the retained count range",
                    )
                })?;
            if reference_speaker_count > sortformer_capacity {
                return Ok(InternalLaneOutcome::skipped(
                    lane,
                    ModelComparisonOutcomeCode::SortformerModelCapacityExceeded,
                ));
            }
            cancellation_checkpoint(is_cancelled)?;
            let cancellation = CancellationToken::unbounded();
            let observation = run_sortformer_observation_with_cancel(
                SortformerObservationRequest {
                    audio_path,
                    expected_audio_sha256: audio_sha256,
                    expected_duration_ms: reference.duration_ms,
                    recording_key: audio_sha256,
                    hard_timeout: attempt_hard_timeout,
                },
                &cancellation,
                is_cancelled,
            );
            cancellation_checkpoint(is_cancelled)?;
            match observation {
                Ok(SortformerObservationOutcome::Completed {
                    document,
                    provenance,
                }) => {
                    let runtime_identity = sortformer_runtime_identity(&provenance)?;
                    let provenance_sha256 = super::canonical_sha256(&provenance)?;
                    let Some(turns) = document.final_projection else {
                        return Ok(InternalLaneOutcome::failed_with_provenance(
                            lane,
                            ModelComparisonOutcomeCode::SortformerExecutionFailed,
                            provenance_sha256,
                        )
                        .with_external_runtime_identity(runtime_identity));
                    };
                    let count = count_observation(reference, &turns, None)?;
                    match score_hypothesis(reference, turns, None, scorer_config) {
                        Ok(score) => Ok(InternalLaneOutcome::completed(
                            lane,
                            score,
                            Some(provenance_sha256),
                            count,
                        )
                        .with_external_runtime_identity(runtime_identity)),
                        Err(_) => Ok(InternalLaneOutcome::failed_with_provenance(
                            lane,
                            ModelComparisonOutcomeCode::ScoringFailed,
                            provenance_sha256,
                        )
                        .with_external_runtime_identity(runtime_identity)),
                    }
                }
                Ok(SortformerObservationOutcome::Skipped {
                    reason,
                    stage,
                    provenance,
                }) => {
                    if matches!(
                        reason,
                        DifferentialSkipReason::InputContractMismatch
                            | DifferentialSkipReason::InputIdentityMismatch
                            | DifferentialSkipReason::ExecutableIdentityMismatch
                    ) {
                        return Err(model_comparison_error(
                            "input_identity",
                            "Sortformer validation detected an input or executable identity change",
                        ));
                    }
                    let runtime_identity = sortformer_runtime_identity(&provenance)?;
                    let provenance_sha256 = super::canonical_sha256(&provenance)?;
                    let (status, code) = classify_sortformer_skip(reason, stage)?;
                    match status {
                        ModelComparisonOutcomeStatus::Skipped => {
                            Ok(InternalLaneOutcome::skipped_with_provenance(
                                lane,
                                code,
                                provenance_sha256,
                            )
                            .with_external_runtime_identity(runtime_identity))
                        }
                        ModelComparisonOutcomeStatus::Failed => {
                            Ok(InternalLaneOutcome::failed_with_provenance(
                                lane,
                                code,
                                provenance_sha256,
                            )
                            .with_external_runtime_identity(runtime_identity))
                        }
                        ModelComparisonOutcomeStatus::Completed => Err(model_comparison_error(
                            "sortformer_classification",
                            "Sortformer skip classification produced an impossible completed status",
                        )),
                    }
                }
                Err(error @ FwError::Cancelled(_)) => Err(error),
                Err(_) => Ok(InternalLaneOutcome::failed(
                    lane,
                    ModelComparisonOutcomeCode::SortformerExecutionFailed,
                )),
            }
        }
    }
}

fn checked_sortformer_timestamp_ms(seconds: f32, duration_ms: u64) -> FwResult<u64> {
    let milliseconds = f64::from(seconds) * 1_000.0;
    if !milliseconds.is_finite() || milliseconds < 0.0 || milliseconds > u64::MAX as f64 {
        return Err(model_comparison_error(
            "native_sortformer_timestamp",
            "native Sortformer emitted an invalid timestamp",
        ));
    }
    Ok((milliseconds.round() as u64).min(duration_ms))
}

fn sortformer_evaluation_turns(
    source: Vec<SortformerSpeakerTurn>,
    duration_ms: u64,
) -> FwResult<(Vec<EvaluationTurn>, u64)> {
    let mut active_lanes = BTreeSet::new();
    let mut turns = Vec::with_capacity(source.len());
    for turn in source {
        if turn.speaker >= SORTFORMER_SPEAKER_LANES {
            return Err(model_comparison_error(
                "native_sortformer_speaker_lane",
                "native Sortformer emitted an out-of-range speaker lane",
            ));
        }
        let start_ms = checked_sortformer_timestamp_ms(turn.start_seconds, duration_ms)?;
        let end_ms = checked_sortformer_timestamp_ms(turn.end_seconds, duration_ms)?;
        if end_ms <= start_ms {
            continue;
        }
        active_lanes.insert(turn.speaker);
        turns.push(EvaluationTurn {
            start_ms,
            end_ms,
            speaker: Some(format!("speaker_{:02}", turn.speaker)),
            speaker_confidence: None,
            overlap_suspected: false,
        });
    }
    let selected_count = u64::try_from(active_lanes.len()).map_err(|_| {
        model_comparison_error(
            "native_sortformer_count",
            "native Sortformer active lane count exceeds the retained range",
        )
    })?;
    Ok((turns, selected_count))
}

fn comparison_diarization_request(engine: DiarizationEngine) -> DiarizationRequest {
    DiarizationRequest {
        engine,
        fallback: DiarizationFallbackPolicy::Unknown,
        speaker_count: SpeakerCountRequest::Infer,
        known_intervals: Vec::new(),
        persist_profiles: false,
        ..DiarizationRequest::default()
    }
}

fn score_native_report(
    lane: ModelComparisonLane,
    report: crate::model::DiarizationReport,
    reference: &crate::diarization::DiarizationReferenceDocument,
    scorer_config: &DiarizationScorerConfig,
) -> FwResult<InternalLaneOutcome> {
    let selected_count = report
        .speaker_count
        .estimate
        .as_ref()
        .and_then(|estimate| estimate.selected_count)
        .map(u64::from);
    let speaker_count_estimate = report.speaker_count.estimate.clone();
    let turns = report
        .turns
        .into_iter()
        .map(|turn| EvaluationTurn {
            start_ms: turn.start_ms,
            end_ms: turn.end_ms.min(reference.duration_ms),
            speaker: turn.speaker_ref,
            speaker_confidence: turn.speaker_confidence,
            overlap_suspected: turn.overlap_suspected,
        })
        .filter(|turn| turn.end_ms > turn.start_ms)
        .collect::<Vec<_>>();
    let count = count_observation(reference, &turns, selected_count)?;
    match score_hypothesis(reference, turns, speaker_count_estimate, scorer_config) {
        Ok(score) => Ok(InternalLaneOutcome::completed(lane, score, None, count)),
        Err(_) => Ok(InternalLaneOutcome::failed(
            lane,
            ModelComparisonOutcomeCode::ScoringFailed,
        )),
    }
}

fn score_hypothesis(
    reference: &crate::diarization::DiarizationReferenceDocument,
    turns: Vec<EvaluationTurn>,
    speaker_count_estimate: Option<SpeakerCountEstimate>,
    scorer_config: &DiarizationScorerConfig,
) -> FwResult<AuthoritativeDiarizationScore> {
    let hypothesis = DiarizationHypothesisDocument {
        schema_version: DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
        recording_id: reference.recording_id.clone(),
        duration_ms: reference.duration_ms,
        turns,
        speaker_count_estimate,
        performance: None,
    };
    score_diarization_documents(reference, &hypothesis, scorer_config)
}

fn count_observation(
    reference: &crate::diarization::DiarizationReferenceDocument,
    hypothesis_turns: &[EvaluationTurn],
    selected_count: Option<u64>,
) -> FwResult<ModelComparisonCountObservation> {
    Ok(ModelComparisonCountObservation {
        reference_full_timeline: count_labeled_speakers(&reference.turns)?,
        hypothesis_full_timeline: count_labeled_speakers(hypothesis_turns)?,
        selected_count,
    })
}

fn count_labeled_speakers(turns: &[EvaluationTurn]) -> FwResult<u64> {
    u64::try_from(
        turns
            .iter()
            .filter_map(|turn| turn.speaker.as_deref())
            .collect::<BTreeSet<_>>()
            .len(),
    )
    .map_err(|_| {
        model_comparison_error("speaker_count", "speaker count exceeds the retained range")
    })
}

fn sortformer_runtime_identity(
    provenance: &SortformerObservationProvenance,
) -> FwResult<Option<ModelComparisonExternalRuntimeIdentity>> {
    if provenance.tool != DifferentialOracleTool::Sortformer
        || provenance.family != DifferentialOracleFamily::EndToEndAttractor
        || provenance.authority != DifferentialAuthority::DiagnosticOnly
        || provenance.expected_model_contract_sha256 != SORTFORMER_ORACLE_CONTRACT_SHA256
        || provenance
            .model_contract_sha256
            .as_deref()
            .is_some_and(|value| value != SORTFORMER_ORACLE_CONTRACT_SHA256)
        || provenance
            .executable_sha256
            .as_deref()
            .is_some_and(|value| value != SORTFORMER_ORACLE_ADAPTER_SHA256)
    {
        return Err(model_comparison_error(
            "sortformer_runtime_identity",
            "Sortformer provenance contradicts the frozen external runtime contract",
        ));
    }
    let (
        Some(tool_version),
        Some(adapter_version),
        Some(model_contract_sha256),
        Some(model_artifact_sha256),
        Some(model_artifact_bytes),
        Some(runtime_fingerprint_sha256),
        Some(executable_sha256),
        Some(version_stdout_sha256),
    ) = (
        provenance.tool_version.clone(),
        provenance.adapter_version.clone(),
        provenance.model_contract_sha256.clone(),
        provenance.model_artifact_sha256.clone(),
        provenance.model_artifact_bytes,
        provenance.runtime_fingerprint_sha256.clone(),
        provenance.executable_sha256.clone(),
        provenance.version_stdout_sha256.clone(),
    )
    else {
        return Ok(None);
    };
    let contract = sortformer_oracle_contract();
    Ok(Some(ModelComparisonExternalRuntimeIdentity {
        protocol_version: provenance.protocol_version.clone(),
        authority: "diagnostic_only".to_owned(),
        tool_version,
        adapter_version,
        model_id: contract.model_id,
        model_revision: contract.model_revision,
        upstream_license: contract.upstream_license,
        model_contract_sha256,
        model_artifact_sha256,
        model_artifact_bytes,
        runtime_fingerprint_sha256,
        executable_sha256,
        version_stdout_sha256,
    }))
}

fn classify_sortformer_skip(
    reason: DifferentialSkipReason,
    _stage: DifferentialExecutionStage,
) -> FwResult<(ModelComparisonOutcomeStatus, ModelComparisonOutcomeCode)> {
    Ok(match reason {
        DifferentialSkipReason::MissingExecutable
        | DifferentialSkipReason::UnreadableExecutable => (
            ModelComparisonOutcomeStatus::Skipped,
            ModelComparisonOutcomeCode::SortformerAdapterUnavailable,
        ),
        DifferentialSkipReason::ProtocolVersionMismatch
        | DifferentialSkipReason::ToolIdentityMismatch
        | DifferentialSkipReason::ModelContractMismatch => (
            ModelComparisonOutcomeStatus::Skipped,
            ModelComparisonOutcomeCode::SortformerRuntimeIneligible,
        ),
        DifferentialSkipReason::ReferenceModelCapacityExceeded => (
            ModelComparisonOutcomeStatus::Skipped,
            ModelComparisonOutcomeCode::SortformerModelCapacityExceeded,
        ),
        DifferentialSkipReason::InputContractMismatch
        | DifferentialSkipReason::InputIdentityMismatch
        | DifferentialSkipReason::ExecutableIdentityMismatch => {
            return Err(model_comparison_error(
                "input_identity",
                "input and executable identity failures cannot become lane outcomes",
            ));
        }
        DifferentialSkipReason::ModelCapacityExceeded
        | DifferentialSkipReason::VersionProbeFailed
        | DifferentialSkipReason::VersionProbeTimedOut
        | DifferentialSkipReason::InvalidVersionOutput
        | DifferentialSkipReason::OracleRunFailed
        | DifferentialSkipReason::OracleRunTimedOut
        | DifferentialSkipReason::InvalidOracleOutput
        | DifferentialSkipReason::OracleIdentityMismatch => (
            ModelComparisonOutcomeStatus::Failed,
            ModelComparisonOutcomeCode::SortformerExecutionFailed,
        ),
    })
}

fn classify_ecapa_failure(error: &FwError) -> Option<ModelComparisonOutcomeCode> {
    match error {
        FwError::InvalidRequest(message) if message.starts_with("ecapa.") => {
            Some(match classify_ecapa_fallback_reason(error) {
                EcapaFallbackReason::InvalidInput => ModelComparisonOutcomeCode::EcapaInvalidInput,
                EcapaFallbackReason::ResourceLimit => {
                    ModelComparisonOutcomeCode::EcapaResourceLimit
                }
                EcapaFallbackReason::CheckpointFailure => {
                    ModelComparisonOutcomeCode::EcapaCheckpointFailure
                }
                EcapaFallbackReason::InternalContractFailure => {
                    ModelComparisonOutcomeCode::EcapaInternalContractFailure
                }
                EcapaFallbackReason::NumericalFailure => {
                    ModelComparisonOutcomeCode::EcapaNumericalFailure
                }
                EcapaFallbackReason::Cancelled => return None,
            })
        }
        FwError::InvalidRequest(_) => Some(ModelComparisonOutcomeCode::EcapaPipelineRejected),
        FwError::ContractViolation(_) => Some(ModelComparisonOutcomeCode::EcapaContractViolation),
        FwError::StageTimeout { .. } | FwError::CommandTimedOut { .. } => {
            Some(ModelComparisonOutcomeCode::EcapaStageTimedOut)
        }
        FwError::Cancelled(_) => None,
        FwError::Io(_)
        | FwError::Json(_)
        | FwError::CommandMissing { .. }
        | FwError::CommandFailed { .. }
        | FwError::BackendUnavailable(_)
        | FwError::Storage(_)
        | FwError::Unsupported(_)
        | FwError::MissingArtifact(_) => Some(ModelComparisonOutcomeCode::EcapaExecutionFailed),
    }
}

fn lane_resource_evidence(
    lane: ModelComparisonLane,
    sortformer_timeout_seconds: u64,
    timed_attempt_count: u64,
    attempted_wall_time_ms: u64,
    completed_wall_time_ms: u64,
    peak_rss_measured_attempt_count: u64,
    maximum_peak_rss_bytes: Option<u64>,
    maximum_cancellation_latency_ms: u64,
    maximum_completed_real_time_factor_millionths: Option<u64>,
) -> ModelComparisonResourceEvidence {
    let peak_rss_complete = timed_attempt_count > 0
        && peak_rss_measured_attempt_count == timed_attempt_count
        && maximum_peak_rss_bytes.is_some();
    let rtf_cap_millionths = if lane == ModelComparisonLane::ExternalSortformer {
        MODEL_COMPARISON_EXTERNAL_RTF_CAP_MILLIONTHS
    } else {
        MODEL_COMPARISON_NATIVE_RTF_CAP_MILLIONTHS
    };
    ModelComparisonResourceEvidence {
        wall_time_authority: ModelComparisonResourceAuthority::Measured,
        wall_time_scope:
            ModelComparisonWallTimeScope::FreshProcessIdentityValidationModelLoadInferenceAndScorer,
        wall_time_cross_lane_comparable: true,
        timed_attempt_count,
        attempted_wall_time_ms,
        completed_wall_time_ms,
        peak_rss_authority: if peak_rss_complete {
            ModelComparisonResourceAuthority::Measured
        } else {
            ModelComparisonResourceAuthority::UnavailableNoProbe
        },
        peak_rss_scope: ModelComparisonPeakRssScope::WholeProcessTree,
        peak_rss_minimum_sampling_interval_ms: peak_rss_complete
            .then_some(MODEL_COMPARISON_PROCESS_TREE_RSS_MINIMUM_SAMPLE_INTERVAL_MS),
        sampled_peak_rss_bytes: peak_rss_complete
            .then_some(maximum_peak_rss_bytes)
            .flatten(),
        cancellation_latency_authority: ModelComparisonResourceAuthority::Measured,
        maximum_cancellation_latency_ms: Some(maximum_cancellation_latency_ms),
        hard_timeout_seconds: Some(sortformer_timeout_seconds),
        maximum_completed_real_time_factor_millionths,
        real_time_factor_cap_millionths: rtf_cap_millionths,
        real_time_factor_within_cap: maximum_completed_real_time_factor_millionths
            .map(|value| value <= rtf_cap_millionths),
        peak_rss_cap_bytes: MODEL_COMPARISON_PEAK_RSS_CAP_BYTES,
        peak_rss_within_cap: maximum_peak_rss_bytes
            .filter(|_| peak_rss_complete)
            .map(|value| value <= MODEL_COMPARISON_PEAK_RSS_CAP_BYTES),
        cancellation_latency_cap_ms: MODEL_COMPARISON_CANCELLATION_LATENCY_CAP_MS,
        cancellation_latency_within_cap: Some(
            maximum_cancellation_latency_ms <= MODEL_COMPARISON_CANCELLATION_LATENCY_CAP_MS,
        ),
    }
}

fn decode_pcm16_wave<F>(bytes: &[u8], is_cancelled: &F) -> FwResult<Vec<f32>>
where
    F: Fn() -> bool + Sync,
{
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).map_err(|_| {
        model_comparison_error("audio_contract", "comparison input must be a readable WAV")
    })?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.sample_rate != 16_000
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        return Err(model_comparison_error(
            "audio_contract",
            "comparison input must be mono 16 kHz signed 16-bit PCM WAV",
        ));
    }
    let declared_samples = u64::from(reader.duration());
    let maximum_samples_from_file = u64::try_from(bytes.len() / 2).unwrap_or(u64::MAX);
    if declared_samples > maximum_samples_from_file {
        return Err(model_comparison_error(
            "audio_contract",
            "comparison WAV declares more PCM samples than its bounded file can contain",
        ));
    }
    let capacity = usize::try_from(declared_samples).map_err(|_| {
        model_comparison_error(
            "audio_contract",
            "comparison WAV sample count exceeds the platform range",
        )
    })?;
    let mut samples = Vec::new();
    samples.try_reserve_exact(capacity).map_err(|_| {
        model_comparison_error(
            "audio_contract",
            "comparison WAV sample buffer cannot be allocated within the platform limits",
        )
    })?;
    for (index, sample) in reader.samples::<i16>().enumerate() {
        if index.is_multiple_of(64 * 1024) {
            cancellation_checkpoint(is_cancelled)?;
        }
        let sample = sample.map_err(|_| {
            model_comparison_error(
                "audio_contract",
                "comparison WAV contains truncated or malformed PCM data",
            )
        })?;
        samples.push(f32::from(sample) / 32_768.0);
    }
    cancellation_checkpoint(is_cancelled)?;
    if u64::try_from(samples.len()).unwrap_or(u64::MAX) != declared_samples {
        return Err(model_comparison_error(
            "audio_contract",
            "comparison WAV decoded sample count disagrees with its data declaration",
        ));
    }
    Ok(samples)
}

fn cancellation_checkpoint<F>(is_cancelled: &F) -> FwResult<()>
where
    F: Fn() -> bool + Sync + ?Sized,
{
    if is_cancelled() {
        Err(FwError::Cancelled(
            "public model comparison cancelled".to_owned(),
        ))
    } else {
        Ok(())
    }
}

fn elapsed_millis(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis())
        .unwrap_or(u64::MAX)
        .max(1)
}

fn ratio_millionths_ceil(numerator: u64, denominator: u64, field: &str) -> FwResult<u64> {
    if denominator == 0 {
        return Err(model_comparison_error(
            field,
            "a resource ratio denominator was zero",
        ));
    }
    let scaled = u128::from(numerator)
        .checked_mul(1_000_000)
        .ok_or_else(|| model_comparison_error(field, "a resource ratio overflowed"))?;
    let rounded = scaled
        .checked_add(u128::from(denominator) - 1)
        .ok_or_else(|| model_comparison_error(field, "a resource ratio overflowed"))?
        / u128::from(denominator);
    u64::try_from(rounded)
        .map_err(|_| model_comparison_error(field, "a resource ratio exceeds u64"))
}

fn deterministic_score_sha256(score: &AuthoritativeDiarizationScore) -> FwResult<String> {
    let mut deterministic = score.clone();
    deterministic.recording_id.clear();
    deterministic.reference_sha256.clear();
    deterministic.hypothesis_sha256.clear();
    deterministic.result_sha256.clear();
    deterministic.performance = None;
    deterministic.diarization.speaker_mapping.clear();
    deterministic.speaker_occupancy.speakers.clear();
    super::canonical_sha256(&deterministic)
}

fn domain_hasher(domain: &[u8]) -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher
}

fn hash_token(hasher: &mut Sha256, token: &[u8]) {
    hasher.update((token.len() as u64).to_le_bytes());
    hasher.update(token);
}

fn expected_execution_order_sha256(observation_count: u64) -> FwResult<String> {
    let count = usize::try_from(observation_count).map_err(|_| {
        model_comparison_error(
            "observation_count",
            "observation count exceeds the platform range",
        )
    })?;
    let mut hasher = domain_hasher(b"franken-whisper-model-comparison-orders-v1\0");
    for observation_index in 0..count {
        hasher.update((observation_index as u64).to_le_bytes());
        for lane in model_comparison_schedule_row(observation_index) {
            hash_token(&mut hasher, lane.as_str().as_bytes());
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn empty_common_complete_sha256() -> String {
    format!(
        "{:x}",
        domain_hasher(b"franken-whisper-model-comparison-common-complete-v1\0").finalize()
    )
}

fn validate_sortformer_runtime_identity(
    identity: &ModelComparisonExternalRuntimeIdentity,
) -> FwResult<()> {
    let contract = sortformer_oracle_contract();
    let hash_fields = [
        identity.model_contract_sha256.as_str(),
        identity.model_artifact_sha256.as_str(),
        identity.runtime_fingerprint_sha256.as_str(),
        identity.executable_sha256.as_str(),
        identity.version_stdout_sha256.as_str(),
    ];
    if identity.protocol_version != crate::differential_oracle::DIFFERENTIAL_ORACLE_PROTOCOL_VERSION
        || identity.authority != "diagnostic_only"
        || identity.tool_version != SORTFORMER_ORACLE_TOOL_VERSION
        || identity.adapter_version != SORTFORMER_ORACLE_ADAPTER_VERSION
        || identity.model_id != contract.model_id
        || identity.model_revision != contract.model_revision
        || identity.upstream_license != contract.upstream_license
        || identity.model_contract_sha256 != SORTFORMER_ORACLE_CONTRACT_SHA256
        || identity.model_artifact_sha256 != contract.upstream_artifact_sha256
        || identity.model_artifact_bytes != contract.upstream_artifact_bytes
        || identity.executable_sha256 != SORTFORMER_ORACLE_ADAPTER_SHA256
        || hash_fields
            .into_iter()
            .any(|value| !super::is_sha256_hex(value))
    {
        return Err(model_comparison_error(
            "sortformer_runtime_identity",
            "retained Sortformer runtime identity violates the pinned contract",
        ));
    }
    Ok(())
}

fn lane_outcome_codes_valid(
    lane: ModelComparisonLane,
    outcomes: &ModelComparisonOutcomeCounts,
) -> bool {
    let skips_valid = outcomes.skipped_by_code.keys().all(|code| match lane {
        ModelComparisonLane::NativeAcoustic => false,
        ModelComparisonLane::NativeEcapa | ModelComparisonLane::NativeEcapaFused => {
            *code == ModelComparisonOutcomeCode::EcapaModelUnavailable
        }
        ModelComparisonLane::NativeSortformer => matches!(
            code,
            ModelComparisonOutcomeCode::NativeSortformerModelUnavailable
                | ModelComparisonOutcomeCode::SortformerModelCapacityExceeded
        ),
        ModelComparisonLane::ExternalSortformer => matches!(
            code,
            ModelComparisonOutcomeCode::SortformerModelCapacityExceeded
                | ModelComparisonOutcomeCode::SortformerAdapterUnavailable
                | ModelComparisonOutcomeCode::SortformerRuntimeIneligible
        ),
    });
    let failures_valid = outcomes.failed_by_code.keys().all(|code| {
        if matches!(
            code,
            ModelComparisonOutcomeCode::WorkerTimedOut
                | ModelComparisonOutcomeCode::WorkerExecutionFailed
                | ModelComparisonOutcomeCode::WorkerMalformedOutput
                | ModelComparisonOutcomeCode::WorkerResourceProbeFailed
        ) {
            return true;
        }
        match lane {
            ModelComparisonLane::NativeAcoustic => matches!(
                code,
                ModelComparisonOutcomeCode::NativeExecutionFailed
                    | ModelComparisonOutcomeCode::ScoringFailed
            ),
            ModelComparisonLane::NativeEcapa | ModelComparisonLane::NativeEcapaFused => matches!(
                code,
                ModelComparisonOutcomeCode::EcapaModelInvalid
                    | ModelComparisonOutcomeCode::EcapaInvalidInput
                    | ModelComparisonOutcomeCode::EcapaResourceLimit
                    | ModelComparisonOutcomeCode::EcapaCheckpointFailure
                    | ModelComparisonOutcomeCode::EcapaInternalContractFailure
                    | ModelComparisonOutcomeCode::EcapaNumericalFailure
                    | ModelComparisonOutcomeCode::EcapaPipelineRejected
                    | ModelComparisonOutcomeCode::EcapaContractViolation
                    | ModelComparisonOutcomeCode::EcapaStageTimedOut
                    | ModelComparisonOutcomeCode::EcapaExecutionFailed
                    | ModelComparisonOutcomeCode::ScoringFailed
            ),
            ModelComparisonLane::NativeSortformer => matches!(
                code,
                ModelComparisonOutcomeCode::NativeSortformerModelInvalid
                    | ModelComparisonOutcomeCode::NativeSortformerExecutionFailed
                    | ModelComparisonOutcomeCode::ScoringFailed
            ),
            ModelComparisonLane::ExternalSortformer => matches!(
                code,
                ModelComparisonOutcomeCode::SortformerExecutionFailed
                    | ModelComparisonOutcomeCode::ScoringFailed
            ),
        }
    });
    skips_valid && failures_valid
}

fn validate_matched_ecapa_model_availability(
    ecapa: &ModelComparisonOutcomeCounts,
    ecapa_fused: &ModelComparisonOutcomeCounts,
    observation_count: u64,
) -> FwResult<()> {
    let unavailable = |outcomes: &ModelComparisonOutcomeCounts| {
        outcomes
            .skipped_by_code
            .get(&ModelComparisonOutcomeCode::EcapaModelUnavailable)
            .copied()
            .unwrap_or(0)
    };
    let invalid = |outcomes: &ModelComparisonOutcomeCounts| {
        outcomes
            .failed_by_code
            .get(&ModelComparisonOutcomeCode::EcapaModelInvalid)
            .copied()
            .unwrap_or(0)
    };
    let unavailable_pair = (unavailable(ecapa), unavailable(ecapa_fused));
    let invalid_pair = (invalid(ecapa), invalid(ecapa_fused));
    let state_valid = if unavailable_pair != (0, 0) {
        unavailable_pair == (observation_count, observation_count) && invalid_pair == (0, 0)
    } else if invalid_pair != (0, 0) {
        invalid_pair == (observation_count, observation_count)
    } else {
        true
    };
    if !state_valid {
        return Err(model_comparison_error(
            "ecapa_shared_state",
            "fresh-process ECAPA and fused ECAPA lanes disagree about bound model availability",
        ));
    }
    Ok(())
}

/// Verify the retained artifact's frozen schema, self-hashes, lane accounting,
/// and aggregate invariants.
///
/// This is intentionally a structural verifier. Aggregate-only evidence does
/// not retain the per-recording outcomes needed to recompute observation-set
/// membership or metrics; source reconstruction requires the separately
/// validated public bundle and a fresh comparison run.
pub fn verify_public_model_comparison_evidence(
    evidence: &PublicModelComparisonEvidence,
) -> FwResult<()> {
    let expected_protocol = frozen_model_comparison_protocol()?;
    let maximum_recordings = u64::try_from(super::MAX_RECORDINGS).unwrap_or(u64::MAX);
    let schedule_period =
        u64::try_from(MODEL_COMPARISON_WILLIAMS_SCHEDULE.len()).unwrap_or(u64::MAX);
    if evidence.schema_version != PUBLIC_MODEL_COMPARISON_SCHEMA_VERSION
        || evidence.runner_version != PUBLIC_MODEL_COMPARISON_RUNNER_VERSION
        || evidence.scorer_version != DIARIZATION_SCORER_VERSION
        || evidence.protocol != expected_protocol
        || evidence.evaluation_split != EvaluationSplit::Development
        || evidence.observation_count == 0
        || evidence.observation_count > maximum_recordings
        || evidence.production_route_changed
        || !evidence.development_uncertified
        || evidence.comparison_authority != "diagnostic_only"
        || evidence.superiority_claim_permitted
    {
        return Err(model_comparison_error(
            "evidence_contract",
            "model-comparison evidence violates the frozen protocol",
        ));
    }
    super::validate_public_id(&evidence.corpus_key, "corpus_key")?;
    super::validate_public_id(&evidence.source_version, "source_version")?;
    if !super::public_corpus_registry()
        .entries
        .iter()
        .any(|entry| entry.corpus_key == evidence.corpus_key)
    {
        return Err(model_comparison_error(
            "corpus_key",
            "model-comparison corpus is absent from the frozen public registry",
        ));
    }
    for value in [
        &evidence.descriptor_sha256,
        &evidence.bundle_sha256,
        &evidence.comparison_executable_sha256,
        &evidence.observation_set_sha256,
        &evidence.execution_order_sha256,
        &evidence.outcome_sequence_sha256,
        &evidence.external_runtime_observation_set_sha256,
        &evidence.common_complete_observation_set_sha256,
        &evidence.protocol_sha256,
        &evidence.deterministic_accuracy_sha256,
        &evidence.result_sha256,
    ] {
        if !super::is_sha256_hex(value) {
            return Err(model_comparison_error(
                "hash_format",
                "model-comparison evidence contains an invalid digest",
            ));
        }
    }
    if evidence.protocol_sha256 != PUBLIC_MODEL_COMPARISON_PROTOCOL_SHA256
        || super::canonical_sha256(&evidence.protocol)? != evidence.protocol_sha256
        || evidence.lanes.len() != ModelComparisonLane::ALL.len()
        || evidence.common_complete_recording_count > evidence.observation_count
        || evidence.execution_order_sha256
            != expected_execution_order_sha256(evidence.observation_count)?
        || evidence.order_balance_complete
            != evidence.observation_count.is_multiple_of(schedule_period)
        || (evidence.common_complete_recording_count == 0
            && evidence.common_complete_observation_set_sha256 != empty_common_complete_sha256())
    {
        return Err(model_comparison_error(
            "evidence_integrity",
            "model-comparison protocol or lane cardinality is invalid",
        ));
    }
    if let Some(identity) = &evidence.sortformer_runtime_identity {
        validate_sortformer_runtime_identity(identity)?;
    }
    let mut completed_sum = 0u64;
    for (aggregate, expected_lane) in evidence.lanes.iter().zip(ModelComparisonLane::ALL) {
        if aggregate.lane != expected_lane
            || !aggregate.outcomes.validate(evidence.observation_count)
            || !lane_outcome_codes_valid(expected_lane, &aggregate.outcomes)
            || aggregate.available_case.recording_count != aggregate.outcomes.completed
            || aggregate.common_complete_case.recording_count
                != evidence.common_complete_recording_count
            || !common_complete_metrics_are_subset(
                &aggregate.available_case,
                &aggregate.common_complete_case,
            )
        {
            return Err(model_comparison_error(
                "aggregate_integrity",
                "model-comparison aggregate counts or common-complete subset metrics are inconsistent",
            ));
        }
        validate_metric_numbers(&aggregate.available_case)?;
        validate_metric_numbers(&aggregate.common_complete_case)?;
        validate_resource_evidence(
            expected_lane,
            &aggregate.resources,
            &aggregate.outcomes,
            &aggregate.available_case,
        )?;
        completed_sum = completed_sum
            .checked_add(aggregate.outcomes.completed)
            .ok_or_else(|| {
                model_comparison_error(
                    "aggregate_overflow",
                    "lane completion counts overflow the retained range",
                )
            })?;
    }
    let minimum_intersection = completed_sum.saturating_sub(
        evidence.observation_count.checked_mul(4).ok_or_else(|| {
            model_comparison_error(
                "aggregate_overflow",
                "observation intersection bound overflows the retained range",
            )
        })?,
    );
    if evidence.common_complete_recording_count < minimum_intersection {
        return Err(model_comparison_error(
            "common_complete",
            "common-complete count violates the five-set intersection lower bound",
        ));
    }
    let common_reference = &evidence.lanes[0].common_complete_case;
    if evidence
        .lanes
        .iter()
        .skip(1)
        .any(|lane| !common_reference_metrics_equal(common_reference, &lane.common_complete_case))
    {
        return Err(model_comparison_error(
            "common_complete",
            "common-complete lanes disagree on reference-side sufficient statistics",
        ));
    }
    if evidence
        .lanes
        .iter()
        .any(|lane| !lane.resources.wall_time_cross_lane_comparable)
    {
        return Err(model_comparison_error(
            "resource_setup",
            "every lane must retain the frozen fresh-process cross-lane-comparable scope",
        ));
    }
    validate_matched_ecapa_model_availability(
        &evidence.lanes[1].outcomes,
        &evidence.lanes[2].outcomes,
        evidence.observation_count,
    )?;
    let external_sortformer = &evidence.lanes[4];
    if sortformer_runtime_identity_required(&external_sortformer.outcomes)
        && evidence.sortformer_runtime_identity.is_none()
    {
        return Err(model_comparison_error(
            "sortformer_runtime_identity",
            "model-produced Sortformer outcomes require a retained runtime identity",
        ));
    }
    if deterministic_accuracy_sha256(evidence)? != evidence.deterministic_accuracy_sha256 {
        return Err(model_comparison_error(
            "accuracy_hash",
            "model-comparison deterministic accuracy digest does not match",
        ));
    }
    let mut unhashed = evidence.clone();
    unhashed.result_sha256.clear();
    if super::canonical_sha256(&unhashed)? != evidence.result_sha256 {
        return Err(model_comparison_error(
            "result_hash",
            "model-comparison result digest does not match",
        ));
    }
    Ok(())
}

/// Structurally validate a model-comparison evidence file and pair it with one
/// exact path-free public-bundle identity.
///
/// Aggregate-only comparison evidence cannot reconstruct per-recording
/// observations, the observation-set commitment, or aggregate metrics. This
/// identity verifier therefore establishes only that both self-validating
/// artifacts assert the same corpus, source, descriptor, bundle, split, and
/// recording count. Derivation proof still requires source reconstruction and
/// a fresh comparison run.
pub fn verify_public_model_comparison_bundle_identity_pair(
    bundle: &super::PublicCorpusBundle,
    evidence: &PublicModelComparisonEvidence,
) -> FwResult<()> {
    super::verify_public_corpus_bundle(bundle)?;
    verify_public_model_comparison_evidence(evidence)?;
    let bundle_identity = model_comparison_bundle_identity(bundle).ok_or_else(|| {
        model_comparison_error(
            "bundle_pair",
            "the public bundle cannot produce one uniform comparison identity",
        )
    })?;
    if bundle_identity != model_comparison_evidence_identity(evidence) {
        return Err(model_comparison_error(
            "bundle_pair",
            "the public bundle and model-comparison evidence identities do not match",
        ));
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct ModelComparisonArtifactIdentity<'a> {
    corpus_key: &'a str,
    source_version: &'a str,
    descriptor_sha256: &'a str,
    bundle_sha256: &'a str,
    evaluation_split: EvaluationSplit,
    observation_count: u64,
}

fn model_comparison_bundle_identity(
    bundle: &super::PublicCorpusBundle,
) -> Option<ModelComparisonArtifactIdentity<'_>> {
    let observation_count = u64::try_from(bundle.references.len()).ok()?;
    let evaluation_split = bundle.recordings.first()?.split;
    if bundle.references.len() != bundle.recordings.len()
        || bundle.references.len() != bundle.manifest.recordings.len()
        || !bundle
            .recordings
            .iter()
            .all(|recording| recording.split == evaluation_split)
        || !bundle
            .manifest
            .recordings
            .iter()
            .all(|recording| recording.split == evaluation_split)
    {
        return None;
    }
    Some(ModelComparisonArtifactIdentity {
        corpus_key: &bundle.corpus_key,
        source_version: &bundle.source_version,
        descriptor_sha256: &bundle.descriptor_sha256,
        bundle_sha256: &bundle.bundle_sha256,
        evaluation_split,
        observation_count,
    })
}

fn model_comparison_evidence_identity(
    evidence: &PublicModelComparisonEvidence,
) -> ModelComparisonArtifactIdentity<'_> {
    ModelComparisonArtifactIdentity {
        corpus_key: &evidence.corpus_key,
        source_version: &evidence.source_version,
        descriptor_sha256: &evidence.descriptor_sha256,
        bundle_sha256: &evidence.bundle_sha256,
        evaluation_split: evidence.evaluation_split,
        observation_count: evidence.observation_count,
    }
}

fn sortformer_runtime_identity_required(outcomes: &ModelComparisonOutcomeCounts) -> bool {
    outcomes.completed > 0
        || outcomes
            .failed_by_code
            .get(&ModelComparisonOutcomeCode::ScoringFailed)
            .is_some_and(|count| *count > 0)
}

fn common_complete_metrics_are_subset(
    available: &ModelComparisonAggregateMetrics,
    common: &ModelComparisonAggregateMetrics,
) -> bool {
    if common.recording_count > available.recording_count {
        return false;
    }
    if common.recording_count == available.recording_count {
        return common == available;
    }
    let Some(extra_recordings) = available
        .recording_count
        .checked_sub(common.recording_count)
    else {
        return false;
    };
    let minimum_extra_duration_sec = extra_recordings as f64 / 1_000.0;
    if exceeds(
        minimum_extra_duration_sec,
        available.audio_duration_sec - common.audio_duration_sec,
    ) || exceeds(
        minimum_extra_duration_sec,
        available.wall_time_sec - common.wall_time_sec,
    ) {
        return false;
    }

    // Only retained additive sufficient statistics are ordered for a strict
    // subset. Means and rates need not be monotone, and the rounded change
    // MAE cannot safely reconstruct its hidden sum for large match counts.
    let common_float_sums = [
        common.audio_duration_sec,
        common.reference_speaker_time_sec,
        common.missed_speech_sec,
        common.false_alarm_sec,
        common.speaker_confusion_sec,
        common.overlap_reference_sec,
        common.overlap_hypothesis_sec,
        common.overlap_true_positive_sec,
        common.overlap_false_positive_sec,
        common.overlap_false_negative_sec,
        common.selective_reference_speaker_time_sec,
        common.selective_covered_speaker_time_sec,
        common.selective_error_covered_speaker_time_sec,
        common.labeled_speaker_time_sec,
        common.unknown_speaker_time_sec,
        common.wall_time_sec,
    ];
    let available_float_sums = [
        available.audio_duration_sec,
        available.reference_speaker_time_sec,
        available.missed_speech_sec,
        available.false_alarm_sec,
        available.speaker_confusion_sec,
        available.overlap_reference_sec,
        available.overlap_hypothesis_sec,
        available.overlap_true_positive_sec,
        available.overlap_false_positive_sec,
        available.overlap_false_negative_sec,
        available.selective_reference_speaker_time_sec,
        available.selective_covered_speaker_time_sec,
        available.selective_error_covered_speaker_time_sec,
        available.labeled_speaker_time_sec,
        available.unknown_speaker_time_sec,
        available.wall_time_sec,
    ];
    let common_integer_sums = [
        common.scored_region_total_absolute_speaker_count_error,
        common.scored_region_exact_speaker_count,
        common.full_timeline_total_absolute_speaker_count_error,
        common.full_timeline_exact_speaker_count,
        common.count_estimate_resolved,
        common.count_estimate_unresolved,
        common.count_estimate_total_absolute_error,
        common.count_estimate_exact,
        common.change_reference_count,
        common.change_hypothesis_count,
        common.change_matched_count,
    ];
    let available_integer_sums = [
        available.scored_region_total_absolute_speaker_count_error,
        available.scored_region_exact_speaker_count,
        available.full_timeline_total_absolute_speaker_count_error,
        available.full_timeline_exact_speaker_count,
        available.count_estimate_resolved,
        available.count_estimate_unresolved,
        available.count_estimate_total_absolute_error,
        available.count_estimate_exact,
        available.change_reference_count,
        available.change_hypothesis_count,
        available.change_matched_count,
    ];
    let macro_support_is_monotone = (common.macro_der.is_none() || available.macro_der.is_some())
        && (common.macro_jer.is_none() || available.macro_jer.is_some());
    if !macro_support_is_monotone
        || !common_float_sums
            .into_iter()
            .zip(available_float_sums)
            .all(|(common_value, available_value)| common_value <= available_value)
        || !common_integer_sums
            .into_iter()
            .zip(available_integer_sums)
            .all(|(common_value, available_value)| common_value <= available_value)
    {
        return false;
    }
    if common.change_matched_count == available.change_matched_count
        && common.change_mean_absolute_error_sec != available.change_mean_absolute_error_sec
    {
        return false;
    }

    let Some(scored_error_delta) = available
        .scored_region_total_absolute_speaker_count_error
        .checked_sub(common.scored_region_total_absolute_speaker_count_error)
    else {
        return false;
    };
    let Some(scored_exact_delta) = available
        .scored_region_exact_speaker_count
        .checked_sub(common.scored_region_exact_speaker_count)
    else {
        return false;
    };
    let Some(full_timeline_error_delta) = available
        .full_timeline_total_absolute_speaker_count_error
        .checked_sub(common.full_timeline_total_absolute_speaker_count_error)
    else {
        return false;
    };
    let Some(full_timeline_exact_delta) = available
        .full_timeline_exact_speaker_count
        .checked_sub(common.full_timeline_exact_speaker_count)
    else {
        return false;
    };
    let Some(resolved_delta) = available
        .count_estimate_resolved
        .checked_sub(common.count_estimate_resolved)
    else {
        return false;
    };
    let Some(count_error_delta) = available
        .count_estimate_total_absolute_error
        .checked_sub(common.count_estimate_total_absolute_error)
    else {
        return false;
    };
    let Some(count_exact_delta) = available
        .count_estimate_exact
        .checked_sub(common.count_estimate_exact)
    else {
        return false;
    };
    let Some(change_reference_delta) = available
        .change_reference_count
        .checked_sub(common.change_reference_count)
    else {
        return false;
    };
    let Some(change_hypothesis_delta) = available
        .change_hypothesis_count
        .checked_sub(common.change_hypothesis_count)
    else {
        return false;
    };
    let Some(change_matched_delta) = available
        .change_matched_count
        .checked_sub(common.change_matched_count)
    else {
        return false;
    };
    let scored_count_delta_is_possible = scored_error_delta
        .checked_add(scored_exact_delta)
        .is_some_and(|accounted| accounted >= extra_recordings)
        && scored_exact_delta <= extra_recordings;
    let full_timeline_count_delta_is_possible = full_timeline_error_delta
        .checked_add(full_timeline_exact_delta)
        .is_some_and(|accounted| accounted >= extra_recordings)
        && full_timeline_exact_delta <= extra_recordings;
    let resolved_count_delta_is_possible = count_error_delta
        .checked_add(count_exact_delta)
        .is_some_and(|accounted| accounted >= resolved_delta)
        && count_exact_delta <= resolved_delta;
    let change_delta_is_possible = change_matched_delta <= change_reference_delta
        && change_matched_delta <= change_hypothesis_delta;
    let selective_reference_delta = available.selective_reference_speaker_time_sec
        - common.selective_reference_speaker_time_sec;
    let selective_covered_delta =
        available.selective_covered_speaker_time_sec - common.selective_covered_speaker_time_sec;
    let selective_error_delta = available.selective_error_covered_speaker_time_sec
        - common.selective_error_covered_speaker_time_sec;
    let selective_delta_is_possible = !exceeds(selective_covered_delta, selective_reference_delta)
        && !exceeds(selective_error_delta, selective_covered_delta);
    let reference_speaker_time_delta =
        available.reference_speaker_time_sec - common.reference_speaker_time_sec;
    let missed_speech_delta = available.missed_speech_sec - common.missed_speech_sec;
    let speaker_confusion_delta = available.speaker_confusion_sec - common.speaker_confusion_sec;
    let diarization_delta_is_possible = !exceeds(
        missed_speech_delta + speaker_confusion_delta,
        reference_speaker_time_delta,
    );

    scored_count_delta_is_possible
        && full_timeline_count_delta_is_possible
        && resolved_count_delta_is_possible
        && change_delta_is_possible
        && selective_delta_is_possible
        && diarization_delta_is_possible
}

fn common_reference_metrics_equal(
    left: &ModelComparisonAggregateMetrics,
    right: &ModelComparisonAggregateMetrics,
) -> bool {
    left.recording_count == right.recording_count
        && option_close(
            Some(left.audio_duration_sec),
            Some(right.audio_duration_sec),
        )
        && option_close(
            Some(left.reference_speaker_time_sec),
            Some(right.reference_speaker_time_sec),
        )
        && option_close(
            Some(left.overlap_reference_sec),
            Some(right.overlap_reference_sec),
        )
        && left.change_reference_count == right.change_reference_count
        && option_close(
            Some(left.selective_reference_speaker_time_sec),
            Some(right.selective_reference_speaker_time_sec),
        )
}

pub(crate) fn deterministic_accuracy_sha256(
    evidence: &PublicModelComparisonEvidence,
) -> FwResult<String> {
    let mut deterministic = evidence.clone();
    deterministic.deterministic_accuracy_sha256.clear();
    deterministic.result_sha256.clear();
    deterministic
        .external_runtime_observation_set_sha256
        .clear();
    deterministic.sortformer_runtime_identity = None;
    for lane in &mut deterministic.lanes {
        lane.available_case.wall_time_sec = 0.0;
        lane.available_case.real_time_factor = None;
        lane.common_complete_case.wall_time_sec = 0.0;
        lane.common_complete_case.real_time_factor = None;
        lane.resources.peak_rss_authority = ModelComparisonResourceAuthority::UnavailableNoProbe;
        lane.resources.peak_rss_minimum_sampling_interval_ms = None;
        lane.resources.sampled_peak_rss_bytes = None;
        lane.resources.maximum_cancellation_latency_ms = None;
        lane.resources.attempted_wall_time_ms = 0;
        lane.resources.completed_wall_time_ms = 0;
        lane.resources.hard_timeout_seconds = None;
        lane.resources.maximum_completed_real_time_factor_millionths = None;
        lane.resources.real_time_factor_within_cap = None;
        lane.resources.peak_rss_within_cap = None;
        lane.resources.cancellation_latency_within_cap = None;
    }
    super::canonical_sha256(&deterministic)
}

fn validate_metric_numbers(metrics: &ModelComparisonAggregateMetrics) -> FwResult<()> {
    let required = [
        metrics.audio_duration_sec,
        metrics.reference_speaker_time_sec,
        metrics.missed_speech_sec,
        metrics.false_alarm_sec,
        metrics.speaker_confusion_sec,
        metrics.overlap_reference_sec,
        metrics.overlap_hypothesis_sec,
        metrics.overlap_true_positive_sec,
        metrics.overlap_false_positive_sec,
        metrics.overlap_false_negative_sec,
        metrics.selective_reference_speaker_time_sec,
        metrics.selective_covered_speaker_time_sec,
        metrics.selective_error_covered_speaker_time_sec,
        metrics.labeled_speaker_time_sec,
        metrics.unknown_speaker_time_sec,
        metrics.wall_time_sec,
    ];
    let optional = [
        metrics.micro_der,
        metrics.macro_der,
        metrics.macro_jer,
        metrics.scored_region_mean_absolute_speaker_count_error,
        metrics.scored_region_exact_speaker_count_rate,
        metrics.full_timeline_mean_absolute_speaker_count_error,
        metrics.full_timeline_exact_speaker_count_rate,
        metrics.count_estimate_mean_absolute_error,
        metrics.count_estimate_exact_rate,
        metrics.overlap_precision,
        metrics.overlap_recall,
        metrics.overlap_f1,
        metrics.change_mean_absolute_error_sec,
        metrics.selective_coverage,
        metrics.selective_risk,
        metrics.unknown_speaker_share,
        metrics.real_time_factor,
    ];
    if required
        .into_iter()
        .any(|value| !value.is_finite() || value < 0.0 || canonical(value) != value)
        || optional
            .into_iter()
            .flatten()
            .any(|value| !value.is_finite() || value < 0.0 || canonical(value) != value)
    {
        return Err(model_comparison_error(
            "aggregate_number",
            "model-comparison aggregates must contain canonical finite non-negative numbers",
        ));
    }
    let resolved_total = metrics
        .count_estimate_resolved
        .checked_add(metrics.count_estimate_unresolved);
    if metrics.scored_region_exact_speaker_count > metrics.recording_count
        || metrics.full_timeline_exact_speaker_count > metrics.recording_count
        || resolved_total != Some(metrics.recording_count)
        || metrics.count_estimate_exact > metrics.count_estimate_resolved
        || metrics.scored_region_total_absolute_speaker_count_error
            < metrics
                .recording_count
                .saturating_sub(metrics.scored_region_exact_speaker_count)
        || (metrics.scored_region_exact_speaker_count == metrics.recording_count
            && metrics.scored_region_total_absolute_speaker_count_error != 0)
        || metrics.full_timeline_total_absolute_speaker_count_error
            < metrics
                .recording_count
                .saturating_sub(metrics.full_timeline_exact_speaker_count)
        || (metrics.full_timeline_exact_speaker_count == metrics.recording_count
            && metrics.full_timeline_total_absolute_speaker_count_error != 0)
        || metrics.count_estimate_total_absolute_error
            < metrics
                .count_estimate_resolved
                .saturating_sub(metrics.count_estimate_exact)
        || (metrics.count_estimate_exact == metrics.count_estimate_resolved
            && metrics.count_estimate_total_absolute_error != 0)
        || (metrics.count_estimate_resolved == 0
            && metrics.count_estimate_total_absolute_error != 0)
        || metrics.change_matched_count > metrics.change_reference_count
        || metrics.change_matched_count > metrics.change_hypothesis_count
        || exceeds(
            metrics.overlap_true_positive_sec,
            metrics.overlap_reference_sec,
        )
        || exceeds(
            metrics.overlap_true_positive_sec,
            metrics.overlap_hypothesis_sec,
        )
        || exceeds(
            metrics.selective_covered_speaker_time_sec,
            metrics.selective_reference_speaker_time_sec,
        )
        || exceeds(
            metrics.selective_error_covered_speaker_time_sec,
            metrics.selective_covered_speaker_time_sec,
        )
        || exceeds(
            metrics.missed_speech_sec + metrics.speaker_confusion_sec,
            metrics.reference_speaker_time_sec,
        )
        || !option_close(
            Some(metrics.selective_reference_speaker_time_sec),
            Some(metrics.reference_speaker_time_sec),
        )
        || !option_close(
            Some(
                metrics.labeled_speaker_time_sec
                    + metrics.unknown_speaker_time_sec
                    + metrics.missed_speech_sec,
            ),
            Some(metrics.reference_speaker_time_sec + metrics.false_alarm_sec),
        )
        || !option_close(
            Some(metrics.overlap_reference_sec),
            Some(metrics.overlap_true_positive_sec + metrics.overlap_false_negative_sec),
        )
        || !option_close(
            Some(metrics.overlap_hypothesis_sec),
            Some(metrics.overlap_true_positive_sec + metrics.overlap_false_positive_sec),
        )
    {
        return Err(model_comparison_error(
            "aggregate_invariant",
            "model-comparison aggregate counts or sufficient statistics are impossible",
        ));
    }
    let bounded_rates = [
        metrics.macro_jer,
        metrics.scored_region_exact_speaker_count_rate,
        metrics.full_timeline_exact_speaker_count_rate,
        metrics.count_estimate_exact_rate,
        metrics.overlap_precision,
        metrics.overlap_recall,
        metrics.overlap_f1,
        metrics.selective_coverage,
        metrics.selective_risk,
        metrics.unknown_speaker_share,
    ];
    if bounded_rates.into_iter().flatten().any(|value| value > 1.0) {
        return Err(model_comparison_error(
            "aggregate_rate",
            "model-comparison probability or rate exceeds one",
        ));
    }
    if metrics.recording_count == 0 {
        let integer_sum = [
            metrics.scored_region_total_absolute_speaker_count_error,
            metrics.scored_region_exact_speaker_count,
            metrics.full_timeline_total_absolute_speaker_count_error,
            metrics.full_timeline_exact_speaker_count,
            metrics.count_estimate_resolved,
            metrics.count_estimate_unresolved,
            metrics.count_estimate_total_absolute_error,
            metrics.count_estimate_exact,
            metrics.change_reference_count,
            metrics.change_hypothesis_count,
            metrics.change_matched_count,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add);
        if required.into_iter().any(|value| value != 0.0)
            || integer_sum != Some(0)
            || optional.into_iter().any(|value| value.is_some())
        {
            return Err(model_comparison_error(
                "aggregate_empty",
                "zero-recording aggregate must use the canonical empty shape",
            ));
        }
        return Ok(());
    }
    if metrics.audio_duration_sec <= 0.0
        || metrics.wall_time_sec <= 0.0
        || exceeds(
            metrics.recording_count as f64 / 1_000.0,
            metrics.audio_duration_sec,
        )
        || exceeds(
            metrics.recording_count as f64 / 1_000.0,
            metrics.wall_time_sec,
        )
        || metrics.macro_der.is_some() != (metrics.reference_speaker_time_sec > 0.0)
        || metrics.macro_jer.is_some() != (metrics.reference_speaker_time_sec > 0.0)
        || !option_close(
            metrics.micro_der,
            ratio_f64(
                metrics.missed_speech_sec + metrics.false_alarm_sec + metrics.speaker_confusion_sec,
                metrics.reference_speaker_time_sec,
            ),
        )
        || !option_close(
            metrics.scored_region_mean_absolute_speaker_count_error,
            ratio_f64(
                metrics.scored_region_total_absolute_speaker_count_error as f64,
                metrics.recording_count as f64,
            ),
        )
        || !option_close(
            metrics.scored_region_exact_speaker_count_rate,
            ratio_f64(
                metrics.scored_region_exact_speaker_count as f64,
                metrics.recording_count as f64,
            ),
        )
        || !option_close(
            metrics.full_timeline_mean_absolute_speaker_count_error,
            ratio_f64(
                metrics.full_timeline_total_absolute_speaker_count_error as f64,
                metrics.recording_count as f64,
            ),
        )
        || !option_close(
            metrics.full_timeline_exact_speaker_count_rate,
            ratio_f64(
                metrics.full_timeline_exact_speaker_count as f64,
                metrics.recording_count as f64,
            ),
        )
        || !option_close(
            metrics.count_estimate_mean_absolute_error,
            ratio_f64(
                metrics.count_estimate_total_absolute_error as f64,
                metrics.count_estimate_resolved as f64,
            ),
        )
        || !option_close(
            metrics.count_estimate_exact_rate,
            ratio_f64(
                metrics.count_estimate_exact as f64,
                metrics.count_estimate_resolved as f64,
            ),
        )
        || !option_close(
            metrics.overlap_precision,
            ratio_f64(
                metrics.overlap_true_positive_sec,
                metrics.overlap_true_positive_sec + metrics.overlap_false_positive_sec,
            ),
        )
        || !option_close(
            metrics.overlap_recall,
            ratio_f64(
                metrics.overlap_true_positive_sec,
                metrics.overlap_true_positive_sec + metrics.overlap_false_negative_sec,
            ),
        )
        || !option_close(
            metrics.overlap_f1,
            harmonic_mean(metrics.overlap_precision, metrics.overlap_recall),
        )
        || !option_close(
            metrics.selective_coverage,
            ratio_f64(
                metrics.selective_covered_speaker_time_sec,
                metrics.selective_reference_speaker_time_sec,
            ),
        )
        || !option_close(
            metrics.selective_risk,
            ratio_f64(
                metrics.selective_error_covered_speaker_time_sec,
                metrics.selective_covered_speaker_time_sec,
            ),
        )
        || !option_close(
            metrics.unknown_speaker_share,
            ratio_f64(
                metrics.unknown_speaker_time_sec,
                metrics.labeled_speaker_time_sec + metrics.unknown_speaker_time_sec,
            ),
        )
        || !option_close(
            metrics.real_time_factor,
            ratio_f64(metrics.wall_time_sec, metrics.audio_duration_sec),
        )
        || metrics.change_mean_absolute_error_sec.is_some_and(|error| {
            exceeds(
                error,
                comparison_scorer_config().change_boundary_collar_ms as f64 / 1_000.0,
            )
        })
        || (metrics.change_matched_count == 0 && metrics.change_mean_absolute_error_sec.is_some())
        || (metrics.change_matched_count > 0 && metrics.change_mean_absolute_error_sec.is_none())
    {
        return Err(model_comparison_error(
            "aggregate_identity",
            "model-comparison derived metrics disagree with their sufficient statistics",
        ));
    }
    Ok(())
}

fn validate_resource_evidence(
    lane: ModelComparisonLane,
    resources: &ModelComparisonResourceEvidence,
    outcomes: &ModelComparisonOutcomeCounts,
    available: &ModelComparisonAggregateMetrics,
) -> FwResult<()> {
    let peak_rss_valid = match resources.peak_rss_authority {
        ModelComparisonResourceAuthority::Measured => resources
            .sampled_peak_rss_bytes
            .is_some_and(|value| value > 0),
        ModelComparisonResourceAuthority::UnavailableNoProbe => {
            resources.sampled_peak_rss_bytes.is_none()
        }
    };
    let expected_rtf_cap = if lane == ModelComparisonLane::ExternalSortformer {
        MODEL_COMPARISON_EXTERNAL_RTF_CAP_MILLIONTHS
    } else {
        MODEL_COMPARISON_NATIVE_RTF_CAP_MILLIONTHS
    };
    let completed_rtf_valid = match resources.maximum_completed_real_time_factor_millionths {
        Some(value) => {
            outcomes.completed > 0
                && resources.real_time_factor_within_cap
                    == Some(value <= resources.real_time_factor_cap_millionths)
        }
        None => outcomes.completed == 0 && resources.real_time_factor_within_cap.is_none(),
    };
    let peak_cap_valid = resources.peak_rss_within_cap
        == resources
            .sampled_peak_rss_bytes
            .map(|value| value <= resources.peak_rss_cap_bytes);
    let cancellation_cap_valid = resources.cancellation_latency_within_cap
        == resources
            .maximum_cancellation_latency_ms
            .map(|value| value <= resources.cancellation_latency_cap_ms);
    if resources.wall_time_authority != ModelComparisonResourceAuthority::Measured
        || resources.wall_time_scope
            != ModelComparisonWallTimeScope::FreshProcessIdentityValidationModelLoadInferenceAndScorer
        || !resources.wall_time_cross_lane_comparable
        || resources.timed_attempt_count != outcomes.declared
        || resources.attempted_wall_time_ms < resources.completed_wall_time_ms
        || resources.attempted_wall_time_ms < outcomes.declared
        || (outcomes.completed == 0 && resources.completed_wall_time_ms != 0)
        || resources.completed_wall_time_ms < outcomes.completed
        || canonical(resources.completed_wall_time_ms as f64 / 1_000.0) != available.wall_time_sec
        || !peak_rss_valid
        || resources.peak_rss_scope != ModelComparisonPeakRssScope::WholeProcessTree
        || resources.peak_rss_minimum_sampling_interval_ms
            != (resources.peak_rss_authority == ModelComparisonResourceAuthority::Measured)
                .then_some(MODEL_COMPARISON_PROCESS_TREE_RSS_MINIMUM_SAMPLE_INTERVAL_MS)
        || resources.cancellation_latency_authority != ModelComparisonResourceAuthority::Measured
        || !resources
            .maximum_cancellation_latency_ms
            .is_some_and(|value| value > 0)
        || resources.hard_timeout_seconds
            != Some(PUBLIC_MODEL_COMPARISON_ATTEMPT_TIMEOUT_SECONDS)
        || resources.real_time_factor_cap_millionths != expected_rtf_cap
        || !completed_rtf_valid
        || resources.peak_rss_cap_bytes != MODEL_COMPARISON_PEAK_RSS_CAP_BYTES
        || !peak_cap_valid
        || resources.cancellation_latency_cap_ms
            != MODEL_COMPARISON_CANCELLATION_LATENCY_CAP_MS
        || !cancellation_cap_valid
    {
        return Err(model_comparison_error(
            "resource_authority",
            "lane resource values disagree with their scope or measurement authority",
        ));
    }
    Ok(())
}

fn exceeds(value: f64, upper_bound: f64) -> bool {
    value - upper_bound > 1.0e-9
}

fn option_close(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => {
            (left - right).abs() <= 1.0e-9 * left.abs().max(right.abs()).max(1.0)
        }
        (None, None) => true,
        _ => false,
    }
}

fn ratio_f64(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then(|| canonical(numerator / denominator))
}

fn harmonic_mean(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) if left + right > 0.0 => {
            Some(canonical(2.0 * left * right / (left + right)))
        }
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    }
}

fn canonical(value: f64) -> f64 {
    super::canonical_evidence_number(value)
}

fn model_comparison_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("model_comparison.{code}: {message}"))
}

#[cfg(test)]
mod tests {
    use crate::diarization::{
        DIARIZATION_HYPOTHESIS_SCHEMA_VERSION, DIARIZATION_REFERENCE_SCHEMA_VERSION,
        DiarizationHypothesisDocument, DiarizationReferenceDocument, DiarizationScorerConfig,
        EvaluationPerformanceObservation, EvaluationTurn, score_diarization_documents,
    };

    use super::*;

    fn score(reference_speaker: &str, hypothesis_speaker: &str) -> AuthoritativeDiarizationScore {
        let reference = DiarizationReferenceDocument {
            schema_version: DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
            recording_id: "public-model-comparison-fixture".to_owned(),
            duration_ms: 1_000,
            turns: vec![EvaluationTurn::labeled(0, 1_000, reference_speaker)],
            ignored_regions: Vec::new(),
            speaker_hints: Vec::new(),
            words: Vec::new(),
        };
        let hypothesis = DiarizationHypothesisDocument {
            schema_version: DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: reference.recording_id.clone(),
            duration_ms: 1_000,
            turns: vec![EvaluationTurn::labeled(0, 1_000, hypothesis_speaker)],
            speaker_count_estimate: None,
            performance: Some(EvaluationPerformanceObservation {
                audio_duration_ms: 1_000,
                wall_time_ms: 250,
                peak_rss_bytes: 0,
            }),
        };
        score_diarization_documents(&reference, &hypothesis, &DiarizationScorerConfig::default())
            .expect("fixture score")
    }

    fn aggregate(recording_count: usize) -> ModelComparisonAggregateMetrics {
        let count = ModelComparisonCountObservation {
            reference_full_timeline: 1,
            hypothesis_full_timeline: 1,
            selected_count: Some(1),
        };
        let mut accumulator = ModelComparisonAccumulator::default();
        for _ in 0..recording_count {
            accumulator.push(&score("reference-a", "hypothesis-x"), count, 250);
        }
        accumulator.finish()
    }

    fn set_scored_count_metrics(
        metrics: &mut ModelComparisonAggregateMetrics,
        total_absolute_error: u64,
        exact: u64,
    ) {
        metrics.scored_region_total_absolute_speaker_count_error = total_absolute_error;
        metrics.scored_region_mean_absolute_speaker_count_error =
            ratio_f64(total_absolute_error as f64, metrics.recording_count as f64);
        metrics.scored_region_exact_speaker_count = exact;
        metrics.scored_region_exact_speaker_count_rate =
            ratio_f64(exact as f64, metrics.recording_count as f64);
    }

    fn set_full_timeline_count_metrics(
        metrics: &mut ModelComparisonAggregateMetrics,
        total_absolute_error: u64,
        exact: u64,
    ) {
        metrics.full_timeline_total_absolute_speaker_count_error = total_absolute_error;
        metrics.full_timeline_mean_absolute_speaker_count_error =
            ratio_f64(total_absolute_error as f64, metrics.recording_count as f64);
        metrics.full_timeline_exact_speaker_count = exact;
        metrics.full_timeline_exact_speaker_count_rate =
            ratio_f64(exact as f64, metrics.recording_count as f64);
    }

    fn set_resolved_count_metrics(
        metrics: &mut ModelComparisonAggregateMetrics,
        total_absolute_error: u64,
        exact: u64,
    ) {
        metrics.count_estimate_total_absolute_error = total_absolute_error;
        metrics.count_estimate_mean_absolute_error = ratio_f64(
            total_absolute_error as f64,
            metrics.count_estimate_resolved as f64,
        );
        metrics.count_estimate_exact = exact;
        metrics.count_estimate_exact_rate =
            ratio_f64(exact as f64, metrics.count_estimate_resolved as f64);
    }

    fn assert_valid_metrics(metrics: &ModelComparisonAggregateMetrics) {
        validate_metric_numbers(metrics).expect("fixture aggregate must be internally valid");
    }

    #[test]
    #[ignore = "set FRANKEN_WHISPER_MODEL_COMPARISON_TEST_BUNDLE and _EVIDENCE to public artifacts"]
    fn external_public_model_comparison_pair_rejects_self_consistent_identity_tampering() {
        let bundle_path = std::env::var_os("FRANKEN_WHISPER_MODEL_COMPARISON_TEST_BUNDLE")
            .map(PathBuf::from)
            .expect("set FRANKEN_WHISPER_MODEL_COMPARISON_TEST_BUNDLE");
        let evidence_path = std::env::var_os("FRANKEN_WHISPER_MODEL_COMPARISON_TEST_EVIDENCE")
            .map(PathBuf::from)
            .expect("set FRANKEN_WHISPER_MODEL_COMPARISON_TEST_EVIDENCE");
        let bundle_bytes = std::fs::read(bundle_path).expect("read public bundle");
        let evidence_bytes = std::fs::read(evidence_path).expect("read public evidence");
        let bundle = super::super::parse_public_corpus_bundle(&bundle_bytes)
            .expect("independently valid public bundle");
        let mut evidence: PublicModelComparisonEvidence =
            serde_json::from_slice(&evidence_bytes).expect("parse public comparison evidence");
        verify_public_model_comparison_bundle_identity_pair(&bundle, &evidence)
            .expect("matching public artifact pair");

        evidence.bundle_sha256 = "00".repeat(32);
        evidence.deterministic_accuracy_sha256 =
            deterministic_accuracy_sha256(&evidence).expect("rehash tampered accuracy evidence");
        evidence.result_sha256.clear();
        evidence.result_sha256 =
            super::super::canonical_sha256(&evidence).expect("rehash tampered evidence");
        let error = verify_public_model_comparison_bundle_identity_pair(&bundle, &evidence)
            .expect_err("a self-consistent evidence document cannot switch bundle identity");
        assert!(error.to_string().contains("bundle_pair"));
    }

    #[test]
    fn williams_schedule_balances_every_lane_and_position() {
        for lane in ModelComparisonLane::ALL {
            let mut positions = [0u32; 5];
            for row in MODEL_COMPARISON_WILLIAMS_SCHEDULE {
                positions[row
                    .iter()
                    .position(|candidate| *candidate == lane)
                    .expect("lane")] += 1;
            }
            assert_eq!(positions, [2, 2, 2, 2, 2]);
        }
        assert_eq!(
            model_comparison_schedule_row(10),
            MODEL_COMPARISON_WILLIAMS_SCHEDULE[0]
        );
        let mut directed_adjacencies = BTreeMap::new();
        for row in MODEL_COMPARISON_WILLIAMS_SCHEDULE {
            for pair in row.windows(2) {
                *directed_adjacencies.entry((pair[0], pair[1])).or_insert(0) += 1;
            }
        }
        assert_eq!(directed_adjacencies.len(), 20);
        assert!(directed_adjacencies.values().all(|count| *count == 2));
    }

    #[test]
    fn frozen_protocol_version_and_digest_detect_contract_drift() {
        let protocol = frozen_model_comparison_protocol().expect("frozen protocol");
        assert_eq!(
            protocol.schema_version,
            PUBLIC_MODEL_COMPARISON_PROTOCOL_VERSION
        );
        assert_eq!(
            protocol.outcome_taxonomy_version,
            PUBLIC_MODEL_COMPARISON_OUTCOME_TAXONOMY_VERSION
        );
        assert_eq!(protocol.outcome_codes, ModelComparisonOutcomeCode::ALL);
        assert_eq!(
            super::super::canonical_sha256(&protocol).expect("canonical protocol digest"),
            PUBLIC_MODEL_COMPARISON_PROTOCOL_SHA256
        );
    }

    #[test]
    fn native_sortformer_timestamps_fail_closed_before_clamping() {
        assert_eq!(checked_sortformer_timestamp_ms(0.000_5, 10).unwrap(), 1);
        assert_eq!(checked_sortformer_timestamp_ms(2.0, 10).unwrap(), 10);
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.001] {
            let error = checked_sortformer_timestamp_ms(invalid, 10)
                .expect_err("invalid model timestamps must fail closed");
            assert!(error.to_string().contains("native_sortformer_timestamp"));
        }
    }

    #[test]
    fn native_sortformer_count_excludes_turns_removed_by_duration_clamping() {
        let source = vec![
            SortformerSpeakerTurn {
                start_seconds: 0.001,
                end_seconds: 0.001,
                speaker: 0,
            },
            SortformerSpeakerTurn {
                start_seconds: 0.002,
                end_seconds: 0.008,
                speaker: 1,
            },
            SortformerSpeakerTurn {
                start_seconds: 0.020,
                end_seconds: 0.030,
                speaker: 2,
            },
        ];
        let (turns, selected_count) =
            sortformer_evaluation_turns(source, 10).expect("valid Sortformer turns");
        assert_eq!(selected_count, 1);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker.as_deref(), Some("speaker_01"));
        let error = sortformer_evaluation_turns(
            vec![SortformerSpeakerTurn {
                start_seconds: 0.0,
                end_seconds: 0.005,
                speaker: SORTFORMER_SPEAKER_LANES,
            }],
            10,
        )
        .expect_err("out-of-range speaker lanes must fail closed");
        assert!(error.to_string().contains("native_sortformer_speaker_lane"));
    }

    #[test]
    fn process_tree_rss_parsers_require_positive_bounded_values() {
        assert_eq!(
            parse_process_group_rss_kib(b"1024\n2048\n").unwrap(),
            Some(3_072)
        );
        assert_eq!(parse_process_group_rss_kib(b"\n0\n").unwrap(), None);
        assert!(parse_process_group_rss_kib(b"not-a-number\n").is_err());
        assert_eq!(
            linux_stat_process_group("123 (worker with spaces) S 7 123 123 0"),
            Some(123)
        );
        assert_eq!(linux_stat_process_group("malformed"), None);
        assert_eq!(
            linux_status_rss_kib("Name:\tx\nVmRSS:\t4096 kB\n"),
            Some(4_096)
        );
        assert_eq!(linux_status_rss_kib("VmRSS:\t0 kB\n"), None);
    }

    #[test]
    fn process_tree_rss_sampler_allows_fast_exit_without_counterfeit_measurement() {
        let sampler = ProcessTreeRssSampler::new();
        assert_eq!(sampler.finish().unwrap(), None);
    }

    #[test]
    fn process_tree_rss_sampler_fails_after_a_live_group_repeatedly_disappears() {
        let mut sampler = ProcessTreeRssSampler::new();
        sampler
            .retain_observation(Ok(Some(4_096)))
            .expect("positive sample");
        sampler
            .retain_observation(Ok(None))
            .expect("one terminal race is allowed");
        assert!(sampler.retain_observation(Ok(None)).is_err());
        assert!(sampler.finish().is_err());
    }

    #[cfg(any(target_vendor = "apple", target_os = "linux", target_os = "android"))]
    #[test]
    fn process_tree_rss_sampler_observes_a_live_bounded_group() {
        let cancellation = CancellationToken::no_deadline();
        let mut sampler = ProcessTreeRssSampler::new();
        let never_cancel = || false;
        let mut observer = |root_pid| sampler.observe(root_pid, &never_cancel);
        let output = crate::process::run_command_cancellable_with_input_probe_and_observer(
            "sh",
            &["-c".to_owned(), "sleep 0.15".to_owned()],
            &cancellation,
            Some(Duration::from_secs(5)),
            None,
            &[],
            &mut observer,
        )
        .expect("bounded process group must be observable");
        assert!(output.status.success());
        assert!(sampler.finish().unwrap().is_some_and(|bytes| bytes > 0));
    }

    #[test]
    fn attempt_budget_covers_parent_work_before_and_after_the_worker() {
        assert_eq!(
            remaining_attempt_budget(Duration::from_secs(10), Duration::from_secs(3)),
            Some(Duration::from_secs(7))
        );
        assert_eq!(
            remaining_attempt_budget(Duration::from_secs(10), Duration::from_secs(10)),
            None
        );
        assert_eq!(
            remaining_attempt_budget(Duration::from_secs(10), Duration::from_secs(11)),
            None
        );
    }

    #[cfg(unix)]
    #[test]
    fn worker_identity_hash_rejects_a_symlink_even_when_bytes_match() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("temporary identity directory");
        let target = directory.path().join("target");
        let link = directory.path().join("link");
        std::fs::write(&target, b"identity-bound bytes").expect("identity target");
        symlink(&target, &link).expect("identity symlink");
        let error = hash_bounded_file_with_cancel(&link, 1024, &|| false, "fixture")
            .expect_err("worker identity hashes must not follow symlinks");
        assert!(error.to_string().contains("non-symlink"));
    }

    #[test]
    fn worker_outcome_serialization_removes_recording_and_speaker_identifiers() {
        let count = ModelComparisonCountObservation {
            reference_full_timeline: 1,
            hypothesis_full_timeline: 1,
            selected_count: Some(1),
        };
        let mut outcome = InternalLaneOutcome::completed(
            ModelComparisonLane::NativeAcoustic,
            score("reference-private-label", "hypothesis-private-label"),
            None,
            count,
        );
        sanitize_worker_outcome(&mut outcome);
        let json = serde_json::to_string(&outcome).expect("sanitized worker outcome");
        assert!(!json.contains("public-model-comparison-fixture"));
        assert!(!json.contains("reference-private-label"));
        assert!(!json.contains("hypothesis-private-label"));
    }

    #[test]
    fn installed_broken_sortformer_is_a_failure_not_a_capability_skip() {
        for reason in [
            DifferentialSkipReason::VersionProbeFailed,
            DifferentialSkipReason::VersionProbeTimedOut,
            DifferentialSkipReason::InvalidVersionOutput,
        ] {
            assert_eq!(
                classify_sortformer_skip(reason, DifferentialExecutionStage::VersionProbe)
                    .expect("classification"),
                (
                    ModelComparisonOutcomeStatus::Failed,
                    ModelComparisonOutcomeCode::SortformerExecutionFailed,
                )
            );
        }
    }

    #[test]
    fn sortformer_model_produced_outcomes_require_runtime_identity() {
        assert!(!sortformer_runtime_identity_required(
            &ModelComparisonOutcomeCounts::default()
        ));

        let completed = ModelComparisonOutcomeCounts {
            completed: 1,
            ..ModelComparisonOutcomeCounts::default()
        };
        assert!(sortformer_runtime_identity_required(&completed));

        let mut scoring_failed = ModelComparisonOutcomeCounts::default();
        scoring_failed
            .failed_by_code
            .insert(ModelComparisonOutcomeCode::ScoringFailed, 1);
        assert!(sortformer_runtime_identity_required(&scoring_failed));

        let mut execution_failed = ModelComparisonOutcomeCounts::default();
        execution_failed
            .failed_by_code
            .insert(ModelComparisonOutcomeCode::SortformerExecutionFailed, 1);
        assert!(!sortformer_runtime_identity_required(&execution_failed));
    }

    #[test]
    fn retained_sortformer_identity_requires_the_frozen_adapter_executable() {
        let contract = sortformer_oracle_contract();
        let mut identity = ModelComparisonExternalRuntimeIdentity {
            protocol_version: crate::differential_oracle::DIFFERENTIAL_ORACLE_PROTOCOL_VERSION
                .to_owned(),
            authority: "diagnostic_only".to_owned(),
            tool_version: SORTFORMER_ORACLE_TOOL_VERSION.to_owned(),
            adapter_version: SORTFORMER_ORACLE_ADAPTER_VERSION.to_owned(),
            model_id: contract.model_id,
            model_revision: contract.model_revision,
            upstream_license: contract.upstream_license,
            model_contract_sha256: SORTFORMER_ORACLE_CONTRACT_SHA256.to_owned(),
            model_artifact_sha256: contract.upstream_artifact_sha256,
            model_artifact_bytes: contract.upstream_artifact_bytes,
            runtime_fingerprint_sha256: "11".repeat(32),
            executable_sha256: SORTFORMER_ORACLE_ADAPTER_SHA256.to_owned(),
            version_stdout_sha256: "22".repeat(32),
        };
        validate_sortformer_runtime_identity(&identity).expect("pinned runtime identity");

        identity.executable_sha256 = "33".repeat(32);
        let error = validate_sortformer_runtime_identity(&identity)
            .expect_err("syntactically valid but unfrozen adapter digest must fail");
        assert!(error.to_string().contains("sortformer_runtime_identity"));
    }

    #[test]
    fn ecapa_failures_retain_stable_payload_free_error_classes() {
        let cases = [
            (
                FwError::InvalidRequest("ecapa.input_shape: private detail".to_owned()),
                ModelComparisonOutcomeCode::EcapaInvalidInput,
                "\"ecapa_invalid_input\"",
            ),
            (
                FwError::InvalidRequest("ecapa.inference_resource: private detail".to_owned()),
                ModelComparisonOutcomeCode::EcapaResourceLimit,
                "\"ecapa_resource_limit\"",
            ),
            (
                FwError::InvalidRequest("ecapa.checkpoint_failure: private detail".to_owned()),
                ModelComparisonOutcomeCode::EcapaCheckpointFailure,
                "\"ecapa_checkpoint_failure\"",
            ),
            (
                FwError::InvalidRequest("ecapa.kernel_failure: private detail".to_owned()),
                ModelComparisonOutcomeCode::EcapaInternalContractFailure,
                "\"ecapa_internal_contract_failure\"",
            ),
            (
                FwError::InvalidRequest("ecapa.numerical_value: private detail".to_owned()),
                ModelComparisonOutcomeCode::EcapaNumericalFailure,
                "\"ecapa_numerical_failure\"",
            ),
            (
                FwError::InvalidRequest("private detail".to_owned()),
                ModelComparisonOutcomeCode::EcapaPipelineRejected,
                "\"ecapa_pipeline_rejected\"",
            ),
            (
                FwError::ContractViolation("private detail".to_owned()),
                ModelComparisonOutcomeCode::EcapaContractViolation,
                "\"ecapa_contract_violation\"",
            ),
            (
                FwError::StageTimeout {
                    stage: "ecapa".to_owned(),
                    budget_ms: 1,
                },
                ModelComparisonOutcomeCode::EcapaStageTimedOut,
                "\"ecapa_stage_timed_out\"",
            ),
        ];
        for (error, expected, expected_json) in cases {
            let classified = classify_ecapa_failure(&error).expect("non-cancellation class");
            assert_eq!(classified, expected);
            let serialized = serde_json::to_string(&classified).expect("serialize class");
            assert_eq!(serialized, expected_json);
            assert!(!serialized.contains("private detail"));
        }
        assert_eq!(
            classify_ecapa_failure(&FwError::Cancelled("private detail".to_owned())),
            None
        );
    }

    #[test]
    fn outcome_counts_reject_status_reason_mismatches() {
        let mut counts = ModelComparisonOutcomeCounts::default();
        counts
            .observe(ModelComparisonOutcomeStatus::Completed, None)
            .expect("completed");
        counts
            .observe(
                ModelComparisonOutcomeStatus::Skipped,
                Some(ModelComparisonOutcomeCode::EcapaModelUnavailable),
            )
            .expect("skipped");
        counts
            .observe(
                ModelComparisonOutcomeStatus::Failed,
                Some(ModelComparisonOutcomeCode::ScoringFailed),
            )
            .expect("failed");
        assert!(counts.validate(3));
        let before_invalid = counts.clone();
        assert!(
            counts
                .observe(
                    ModelComparisonOutcomeStatus::Skipped,
                    Some(ModelComparisonOutcomeCode::NativeExecutionFailed),
                )
                .is_err()
        );
        assert_eq!(counts, before_invalid);
        let overflow = ModelComparisonOutcomeCounts {
            declared: u64::MAX,
            completed: u64::MAX,
            skipped: 1,
            ..ModelComparisonOutcomeCounts::default()
        };
        assert!(!overflow.validate(u64::MAX));
    }

    #[test]
    fn accumulator_reduces_authoritative_scores_without_rows() {
        let mut accumulator = ModelComparisonAccumulator::default();
        let count = ModelComparisonCountObservation {
            reference_full_timeline: 1,
            hypothesis_full_timeline: 1,
            selected_count: Some(1),
        };
        accumulator.push(&score("reference-a", "hypothesis-x"), count, 250);
        accumulator.push(&score("reference-b", "hypothesis-y"), count, 250);
        let aggregate = accumulator.finish();
        assert_eq!(aggregate.recording_count, 2);
        assert_eq!(aggregate.micro_der, Some(0.0));
        assert_eq!(aggregate.macro_jer, Some(0.0));
        assert_eq!(aggregate.scored_region_exact_speaker_count_rate, Some(1.0));
        assert_eq!(aggregate.full_timeline_exact_speaker_count_rate, Some(1.0));
        assert_eq!(aggregate.count_estimate_exact_rate, Some(1.0));
        assert_eq!(aggregate.real_time_factor, Some(0.25));
        let json = serde_json::to_string(&aggregate).expect("aggregate JSON");
        assert!(!json.contains("public-model-comparison-fixture"));
        assert!(!json.contains("reference-a"));
        assert!(!json.contains("hypothesis-x"));
    }

    #[test]
    fn common_complete_subset_requires_equal_metrics_at_equal_cardinality() {
        let mut accumulator = ModelComparisonAccumulator::default();
        let count = ModelComparisonCountObservation {
            reference_full_timeline: 1,
            hypothesis_full_timeline: 1,
            selected_count: Some(1),
        };
        accumulator.push(&score("reference-a", "hypothesis-x"), count, 250);
        let available = accumulator.finish();
        let mut forged_common = available.clone();
        forged_common.wall_time_sec += 0.001;
        forged_common.real_time_factor = ratio_f64(
            forged_common.wall_time_sec,
            forged_common.audio_duration_sec,
        );

        assert!(common_complete_metrics_are_subset(&available, &available));
        assert!(!common_complete_metrics_are_subset(
            &available,
            &forged_common,
        ));
    }

    #[test]
    fn common_complete_subset_preserves_macro_metric_support() {
        let count = ModelComparisonCountObservation {
            reference_full_timeline: 1,
            hypothesis_full_timeline: 1,
            selected_count: Some(1),
        };
        let mut available_accumulator = ModelComparisonAccumulator::default();
        available_accumulator.push(&score("reference-a", "hypothesis-x"), count, 250);
        available_accumulator.push(&score("reference-b", "hypothesis-y"), count, 250);
        let available = available_accumulator.finish();
        let mut common_accumulator = ModelComparisonAccumulator::default();
        common_accumulator.push(&score("reference-a", "hypothesis-x"), count, 250);
        let common = common_accumulator.finish();

        let mut missing_available_der = available.clone();
        missing_available_der.macro_der = None;
        assert!(!common_complete_metrics_are_subset(
            &missing_available_der,
            &common,
        ));
        let mut missing_available_jer = available.clone();
        missing_available_jer.macro_jer = None;
        assert!(!common_complete_metrics_are_subset(
            &missing_available_jer,
            &common,
        ));
        let mut missing_common_macro_metrics = common;
        missing_common_macro_metrics.reference_speaker_time_sec = 0.0;
        missing_common_macro_metrics.micro_der = None;
        missing_common_macro_metrics.macro_der = None;
        missing_common_macro_metrics.macro_jer = None;
        missing_common_macro_metrics.selective_reference_speaker_time_sec = 0.0;
        missing_common_macro_metrics.selective_covered_speaker_time_sec = 0.0;
        missing_common_macro_metrics.selective_error_covered_speaker_time_sec = 0.0;
        missing_common_macro_metrics.selective_coverage = None;
        missing_common_macro_metrics.selective_risk = None;
        missing_common_macro_metrics.labeled_speaker_time_sec = 0.0;
        missing_common_macro_metrics.unknown_speaker_time_sec = 0.0;
        missing_common_macro_metrics.unknown_speaker_share = None;
        assert_valid_metrics(&missing_common_macro_metrics);
        assert!(common_complete_metrics_are_subset(
            &available,
            &missing_common_macro_metrics,
        ));
    }

    #[test]
    fn common_complete_subset_rejects_non_monotone_sufficient_statistics() {
        let count = ModelComparisonCountObservation {
            reference_full_timeline: 1,
            hypothesis_full_timeline: 1,
            selected_count: Some(1),
        };
        let mut available_accumulator = ModelComparisonAccumulator::default();
        available_accumulator.push(&score("reference-a", "hypothesis-x"), count, 250);
        available_accumulator.push(&score("reference-b", "hypothesis-y"), count, 250);
        let available = available_accumulator.finish();
        let mut common_accumulator = ModelComparisonAccumulator::default();
        common_accumulator.push(&score("reference-a", "hypothesis-x"), count, 250);
        let common = common_accumulator.finish();

        assert!(common_complete_metrics_are_subset(&available, &common));
        macro_rules! assert_float_sum_rejected {
            ($($field:ident),+ $(,)?) => {
                $(
                    let mut forged_common = common.clone();
                    forged_common.$field = available.$field + 0.001;
                    assert!(
                        !common_complete_metrics_are_subset(&available, &forged_common),
                        "accepted non-monotone {}",
                        stringify!($field),
                    );
                )+
            };
        }
        macro_rules! assert_integer_sum_rejected {
            ($($field:ident),+ $(,)?) => {
                $(
                    let mut forged_common = common.clone();
                    forged_common.$field = available.$field.saturating_add(1);
                    assert!(
                        !common_complete_metrics_are_subset(&available, &forged_common),
                        "accepted non-monotone {}",
                        stringify!($field),
                    );
                )+
            };
        }
        assert_float_sum_rejected!(
            audio_duration_sec,
            reference_speaker_time_sec,
            missed_speech_sec,
            false_alarm_sec,
            speaker_confusion_sec,
            overlap_reference_sec,
            overlap_hypothesis_sec,
            overlap_true_positive_sec,
            overlap_false_positive_sec,
            overlap_false_negative_sec,
            selective_reference_speaker_time_sec,
            selective_covered_speaker_time_sec,
            selective_error_covered_speaker_time_sec,
            labeled_speaker_time_sec,
            unknown_speaker_time_sec,
            wall_time_sec,
        );
        assert_integer_sum_rejected!(
            scored_region_total_absolute_speaker_count_error,
            scored_region_exact_speaker_count,
            full_timeline_total_absolute_speaker_count_error,
            full_timeline_exact_speaker_count,
            count_estimate_resolved,
            count_estimate_unresolved,
            count_estimate_total_absolute_error,
            count_estimate_exact,
            change_reference_count,
            change_hypothesis_count,
            change_matched_count,
        );

        let mut available_with_matched_change = available;
        available_with_matched_change.change_reference_count = 1;
        available_with_matched_change.change_hypothesis_count = 1;
        available_with_matched_change.change_matched_count = 1;
        available_with_matched_change.change_mean_absolute_error_sec = Some(0.1);
        assert!(common_complete_metrics_are_subset(
            &available_with_matched_change,
            &common,
        ));
    }

    #[test]
    fn common_complete_strict_subset_requires_strict_extent_and_stable_change_mae() {
        let available = aggregate(2);
        let common = aggregate(1);
        assert_valid_metrics(&available);
        assert_valid_metrics(&common);
        assert!(common_complete_metrics_are_subset(&available, &common));

        let mut forged_audio = common.clone();
        forged_audio.audio_duration_sec = available.audio_duration_sec;
        forged_audio.real_time_factor =
            ratio_f64(forged_audio.wall_time_sec, forged_audio.audio_duration_sec);
        assert_valid_metrics(&forged_audio);
        assert!(!common_complete_metrics_are_subset(
            &available,
            &forged_audio,
        ));

        let mut forged_wall_time = common.clone();
        forged_wall_time.wall_time_sec = available.wall_time_sec;
        forged_wall_time.real_time_factor = ratio_f64(
            forged_wall_time.wall_time_sec,
            forged_wall_time.audio_duration_sec,
        );
        assert_valid_metrics(&forged_wall_time);
        assert!(!common_complete_metrics_are_subset(
            &available,
            &forged_wall_time,
        ));

        let mut sub_millisecond_audio = common.clone();
        sub_millisecond_audio.audio_duration_sec = available.audio_duration_sec - 0.0005;
        sub_millisecond_audio.real_time_factor = ratio_f64(
            sub_millisecond_audio.wall_time_sec,
            sub_millisecond_audio.audio_duration_sec,
        );
        assert_valid_metrics(&sub_millisecond_audio);
        assert!(!common_complete_metrics_are_subset(
            &available,
            &sub_millisecond_audio,
        ));

        let mut sub_millisecond_wall_time = common.clone();
        sub_millisecond_wall_time.wall_time_sec = available.wall_time_sec - 0.0005;
        sub_millisecond_wall_time.real_time_factor = ratio_f64(
            sub_millisecond_wall_time.wall_time_sec,
            sub_millisecond_wall_time.audio_duration_sec,
        );
        assert_valid_metrics(&sub_millisecond_wall_time);
        assert!(!common_complete_metrics_are_subset(
            &available,
            &sub_millisecond_wall_time,
        ));

        let mut available_with_change = available;
        available_with_change.change_reference_count = 1;
        available_with_change.change_hypothesis_count = 1;
        available_with_change.change_matched_count = 1;
        available_with_change.change_mean_absolute_error_sec = Some(0.1);
        let mut forged_change_mae = common;
        forged_change_mae.change_reference_count = 1;
        forged_change_mae.change_hypothesis_count = 1;
        forged_change_mae.change_matched_count = 1;
        forged_change_mae.change_mean_absolute_error_sec = Some(0.2);
        assert_valid_metrics(&available_with_change);
        assert_valid_metrics(&forged_change_mae);
        assert!(!common_complete_metrics_are_subset(
            &available_with_change,
            &forged_change_mae,
        ));

        let available_without_change = aggregate(2);
        let common_without_change = aggregate(1);
        assert_eq!(available_without_change.change_matched_count, 0);
        assert_eq!(common_without_change.change_mean_absolute_error_sec, None);
        assert!(common_complete_metrics_are_subset(
            &available_without_change,
            &common_without_change,
        ));
    }

    #[test]
    fn common_complete_subset_rejects_impossible_speaker_count_deltas() {
        let available = aggregate(2);
        let common = aggregate(1);

        let mut scored_available = available.clone();
        let mut scored_common = common.clone();
        set_scored_count_metrics(&mut scored_available, 2, 0);
        set_scored_count_metrics(&mut scored_common, 2, 0);
        assert_valid_metrics(&scored_available);
        assert_valid_metrics(&scored_common);
        assert!(!common_complete_metrics_are_subset(
            &scored_available,
            &scored_common,
        ));

        let mut full_available = available.clone();
        let mut full_common = common.clone();
        set_full_timeline_count_metrics(&mut full_available, 2, 0);
        set_full_timeline_count_metrics(&mut full_common, 2, 0);
        assert_valid_metrics(&full_available);
        assert_valid_metrics(&full_common);
        assert!(!common_complete_metrics_are_subset(
            &full_available,
            &full_common,
        ));

        let mut resolved_available = available;
        let mut resolved_common = common;
        set_resolved_count_metrics(&mut resolved_available, 2, 0);
        set_resolved_count_metrics(&mut resolved_common, 2, 0);
        assert_valid_metrics(&resolved_available);
        assert_valid_metrics(&resolved_common);
        assert!(!common_complete_metrics_are_subset(
            &resolved_available,
            &resolved_common,
        ));

        let available = aggregate(3);
        let common = aggregate(2);

        let mut scored_available = available.clone();
        let mut scored_common = common.clone();
        set_scored_count_metrics(&mut scored_available, 2, 2);
        set_scored_count_metrics(&mut scored_common, 2, 0);
        assert_valid_metrics(&scored_available);
        assert_valid_metrics(&scored_common);
        assert!(!common_complete_metrics_are_subset(
            &scored_available,
            &scored_common,
        ));

        let mut full_available = available.clone();
        let mut full_common = common.clone();
        set_full_timeline_count_metrics(&mut full_available, 2, 2);
        set_full_timeline_count_metrics(&mut full_common, 2, 0);
        assert_valid_metrics(&full_available);
        assert_valid_metrics(&full_common);
        assert!(!common_complete_metrics_are_subset(
            &full_available,
            &full_common,
        ));

        let mut resolved_available = available;
        let mut resolved_common = common;
        set_resolved_count_metrics(&mut resolved_available, 2, 2);
        set_resolved_count_metrics(&mut resolved_common, 2, 0);
        assert_valid_metrics(&resolved_available);
        assert_valid_metrics(&resolved_common);
        assert!(!common_complete_metrics_are_subset(
            &resolved_available,
            &resolved_common,
        ));
    }

    #[test]
    fn common_complete_subset_rejects_impossible_change_and_time_deltas() {
        let available = aggregate(2);
        let common = aggregate(1);

        let mut missing_reference_change = available.clone();
        missing_reference_change.change_reference_count = 1;
        missing_reference_change.change_hypothesis_count = 2;
        missing_reference_change.change_matched_count = 1;
        missing_reference_change.change_mean_absolute_error_sec = Some(0.1);
        let mut common_change = common.clone();
        common_change.change_reference_count = 1;
        common_change.change_hypothesis_count = 1;
        assert_valid_metrics(&missing_reference_change);
        assert_valid_metrics(&common_change);
        assert!(!common_complete_metrics_are_subset(
            &missing_reference_change,
            &common_change,
        ));

        let mut missing_hypothesis_change = available.clone();
        missing_hypothesis_change.change_reference_count = 2;
        missing_hypothesis_change.change_hypothesis_count = 1;
        missing_hypothesis_change.change_matched_count = 1;
        missing_hypothesis_change.change_mean_absolute_error_sec = Some(0.1);
        assert_valid_metrics(&missing_hypothesis_change);
        assert!(!common_complete_metrics_are_subset(
            &missing_hypothesis_change,
            &common_change,
        ));

        let mut excessive_coverage = common.clone();
        excessive_coverage.selective_covered_speaker_time_sec = 0.0;
        excessive_coverage.selective_error_covered_speaker_time_sec = 0.0;
        excessive_coverage.selective_coverage = Some(0.0);
        excessive_coverage.selective_risk = None;
        assert_valid_metrics(&excessive_coverage);
        assert!(!common_complete_metrics_are_subset(
            &available,
            &excessive_coverage,
        ));

        let mut excessive_selective_error = available.clone();
        excessive_selective_error.selective_covered_speaker_time_sec = 1.0;
        excessive_selective_error.selective_error_covered_speaker_time_sec = 1.0;
        excessive_selective_error.selective_coverage = Some(0.5);
        excessive_selective_error.selective_risk = Some(1.0);
        assert_valid_metrics(&excessive_selective_error);
        assert!(!common_complete_metrics_are_subset(
            &excessive_selective_error,
            &common,
        ));

        let mut excessive_diarization_error = available;
        excessive_diarization_error.missed_speech_sec = 2.0;
        excessive_diarization_error.false_alarm_sec = 1.0;
        excessive_diarization_error.micro_der = Some(1.5);
        excessive_diarization_error.selective_covered_speaker_time_sec = 0.0;
        excessive_diarization_error.selective_error_covered_speaker_time_sec = 0.0;
        excessive_diarization_error.selective_coverage = Some(0.0);
        excessive_diarization_error.selective_risk = None;
        excessive_diarization_error.labeled_speaker_time_sec = 0.0;
        excessive_diarization_error.unknown_speaker_time_sec = 1.0;
        excessive_diarization_error.unknown_speaker_share = Some(1.0);
        let mut diarization_common = common;
        diarization_common.selective_covered_speaker_time_sec = 0.0;
        diarization_common.selective_error_covered_speaker_time_sec = 0.0;
        diarization_common.selective_coverage = Some(0.0);
        diarization_common.selective_risk = None;
        diarization_common.labeled_speaker_time_sec = 0.0;
        diarization_common.unknown_speaker_time_sec = 1.0;
        diarization_common.unknown_speaker_share = Some(1.0);
        assert_valid_metrics(&excessive_diarization_error);
        assert_valid_metrics(&diarization_common);
        assert!(!common_complete_metrics_are_subset(
            &excessive_diarization_error,
            &diarization_common,
        ));
    }

    #[test]
    fn aggregate_validation_enforces_frozen_scorer_identities() {
        let aggregate = aggregate(2);
        assert_valid_metrics(&aggregate);

        let mut excessive_change_error = aggregate.clone();
        excessive_change_error.change_reference_count = 1;
        excessive_change_error.change_hypothesis_count = 1;
        excessive_change_error.change_matched_count = 1;
        excessive_change_error.change_mean_absolute_error_sec = Some(0.251);
        assert!(validate_metric_numbers(&excessive_change_error).is_err());

        let mut missing_macro_der = aggregate.clone();
        missing_macro_der.macro_der = None;
        assert!(validate_metric_numbers(&missing_macro_der).is_err());
        let mut missing_macro_jer = aggregate.clone();
        missing_macro_jer.macro_jer = None;
        assert!(validate_metric_numbers(&missing_macro_jer).is_err());

        let mut excessive_reference_error = aggregate.clone();
        excessive_reference_error.missed_speech_sec = 2.1;
        excessive_reference_error.false_alarm_sec = 2.1;
        excessive_reference_error.micro_der = Some(2.1);
        assert!(validate_metric_numbers(&excessive_reference_error).is_err());

        let mut mismatched_selective_reference = aggregate.clone();
        mismatched_selective_reference.selective_reference_speaker_time_sec = 2.1;
        mismatched_selective_reference.selective_coverage = ratio_f64(2.0, 2.1);
        assert!(validate_metric_numbers(&mismatched_selective_reference).is_err());

        let mut mismatched_occupancy = aggregate.clone();
        mismatched_occupancy.labeled_speaker_time_sec = 2.1;
        assert!(validate_metric_numbers(&mismatched_occupancy).is_err());

        let mut sub_millisecond_audio_per_recording = aggregate.clone();
        sub_millisecond_audio_per_recording.audio_duration_sec = 0.001;
        sub_millisecond_audio_per_recording.real_time_factor = ratio_f64(
            sub_millisecond_audio_per_recording.wall_time_sec,
            sub_millisecond_audio_per_recording.audio_duration_sec,
        );
        assert!(validate_metric_numbers(&sub_millisecond_audio_per_recording).is_err());

        let mut sub_millisecond_wall_time_per_recording = aggregate.clone();
        sub_millisecond_wall_time_per_recording.wall_time_sec = 0.001;
        sub_millisecond_wall_time_per_recording.real_time_factor = ratio_f64(
            sub_millisecond_wall_time_per_recording.wall_time_sec,
            sub_millisecond_wall_time_per_recording.audio_duration_sec,
        );
        assert!(validate_metric_numbers(&sub_millisecond_wall_time_per_recording).is_err());

        let mut impossible_scored_exact = aggregate.clone();
        set_scored_count_metrics(&mut impossible_scored_exact, 1, 2);
        assert!(validate_metric_numbers(&impossible_scored_exact).is_err());
        let mut impossible_full_exact = aggregate.clone();
        set_full_timeline_count_metrics(&mut impossible_full_exact, 1, 2);
        assert!(validate_metric_numbers(&impossible_full_exact).is_err());
        let mut impossible_resolved_exact = aggregate;
        set_resolved_count_metrics(&mut impossible_resolved_exact, 1, 2);
        assert!(validate_metric_numbers(&impossible_resolved_exact).is_err());
    }

    #[test]
    fn resource_authority_never_uses_zero_as_unavailable() {
        let mut resources = ModelComparisonResourceEvidence {
            wall_time_authority: ModelComparisonResourceAuthority::Measured,
            wall_time_scope:
                ModelComparisonWallTimeScope::FreshProcessIdentityValidationModelLoadInferenceAndScorer,
            wall_time_cross_lane_comparable: true,
            timed_attempt_count: 1,
            attempted_wall_time_ms: 250,
            completed_wall_time_ms: 250,
            peak_rss_authority: ModelComparisonResourceAuthority::Measured,
            peak_rss_scope: ModelComparisonPeakRssScope::WholeProcessTree,
            peak_rss_minimum_sampling_interval_ms: Some(
                MODEL_COMPARISON_PROCESS_TREE_RSS_MINIMUM_SAMPLE_INTERVAL_MS,
            ),
            sampled_peak_rss_bytes: Some(1024),
            cancellation_latency_authority: ModelComparisonResourceAuthority::Measured,
            maximum_cancellation_latency_ms: Some(50),
            hard_timeout_seconds: Some(PUBLIC_MODEL_COMPARISON_ATTEMPT_TIMEOUT_SECONDS),
            maximum_completed_real_time_factor_millionths: Some(250_000),
            real_time_factor_cap_millionths: MODEL_COMPARISON_NATIVE_RTF_CAP_MILLIONTHS,
            real_time_factor_within_cap: Some(true),
            peak_rss_cap_bytes: MODEL_COMPARISON_PEAK_RSS_CAP_BYTES,
            peak_rss_within_cap: Some(true),
            cancellation_latency_cap_ms: MODEL_COMPARISON_CANCELLATION_LATENCY_CAP_MS,
            cancellation_latency_within_cap: Some(true),
        };
        let outcomes = ModelComparisonOutcomeCounts {
            declared: 1,
            completed: 1,
            ..ModelComparisonOutcomeCounts::default()
        };
        let available = ModelComparisonAggregateMetrics {
            recording_count: 1,
            audio_duration_sec: 1.0,
            wall_time_sec: 0.25,
            ..ModelComparisonAggregateMetrics::default()
        };
        assert!(
            validate_resource_evidence(
                ModelComparisonLane::NativeAcoustic,
                &resources,
                &outcomes,
                &available,
            )
            .is_ok()
        );
        resources.sampled_peak_rss_bytes = Some(0);
        assert!(
            validate_resource_evidence(
                ModelComparisonLane::NativeAcoustic,
                &resources,
                &outcomes,
                &available,
            )
            .is_err()
        );
    }

    #[test]
    fn lane_outcome_codes_are_lane_specific() {
        let native_with_ecapa_skip = ModelComparisonOutcomeCounts {
            declared: 1,
            skipped: 1,
            skipped_by_code: BTreeMap::from([(
                ModelComparisonOutcomeCode::EcapaModelUnavailable,
                1,
            )]),
            ..ModelComparisonOutcomeCounts::default()
        };
        assert!(!lane_outcome_codes_valid(
            ModelComparisonLane::NativeAcoustic,
            &native_with_ecapa_skip,
        ));
        assert!(lane_outcome_codes_valid(
            ModelComparisonLane::NativeEcapa,
            &native_with_ecapa_skip,
        ));
    }

    #[test]
    fn matched_ecapa_availability_must_cover_both_fresh_process_lanes() {
        let unavailable = ModelComparisonOutcomeCounts {
            declared: 2,
            skipped: 2,
            skipped_by_code: BTreeMap::from([(
                ModelComparisonOutcomeCode::EcapaModelUnavailable,
                2,
            )]),
            ..ModelComparisonOutcomeCounts::default()
        };
        assert!(validate_matched_ecapa_model_availability(&unavailable, &unavailable, 2).is_ok());
        let ready = ModelComparisonOutcomeCounts {
            declared: 2,
            completed: 2,
            ..ModelComparisonOutcomeCounts::default()
        };
        assert!(validate_matched_ecapa_model_availability(&unavailable, &ready, 2).is_err());
    }

    fn pcm16_wave_bytes(sample_count: usize) -> Vec<u8> {
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("WAV writer");
            for index in 0..sample_count {
                writer
                    .write_sample(i16::try_from(index).unwrap_or(i16::MAX))
                    .expect("PCM sample");
            }
            writer.finalize().expect("finalize WAV");
        }
        cursor.into_inner()
    }

    #[test]
    fn strict_wave_decode_rejects_truncated_and_odd_pcm_payloads() {
        let valid = pcm16_wave_bytes(16);
        assert_eq!(
            decode_pcm16_wave(&valid, &|| false)
                .expect("valid PCM")
                .len(),
            16
        );

        let mut truncated = valid.clone();
        assert_eq!(truncated.pop(), Some(0));
        assert!(decode_pcm16_wave(&truncated, &|| false).is_err());

        let mut odd = valid;
        assert_eq!(odd.pop(), Some(0));
        odd[4..8].copy_from_slice(&67_u32.to_le_bytes());
        odd[40..44].copy_from_slice(&31_u32.to_le_bytes());
        assert!(decode_pcm16_wave(&odd, &|| false).is_err());

        let mut impossible_declaration = pcm16_wave_bytes(1);
        impossible_declaration[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_pcm16_wave(&impossible_declaration, &|| false).is_err());
    }
}
