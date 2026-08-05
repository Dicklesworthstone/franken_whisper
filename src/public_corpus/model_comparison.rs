//! Aggregate-only, same-invocation learned-versus-native diarization comparison.
//!
//! The retained contract deliberately contains no recording identifiers, paths,
//! turns, labels, embeddings, logits, transcript text, or raw error strings.
//! Per-recording hypotheses and authoritative scores exist only long enough to
//! update these aggregate sufficient statistics and ordered digests.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
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
    DifferentialOracleTool, DifferentialSkipReason, SORTFORMER_ORACLE_ADAPTER_VERSION,
    SORTFORMER_ORACLE_CONTRACT_SHA256, SORTFORMER_ORACLE_MAX_SPEAKERS,
    SORTFORMER_ORACLE_OUTPUT_FRAME_MS, SORTFORMER_ORACLE_TOOL_VERSION,
    SortformerObservationOutcome, SortformerObservationProvenance, SortformerObservationRequest,
    run_sortformer_observation_with_cancel, sortformer_oracle_contract,
};
use crate::ecapa_conformance::{
    ECAPA_CONTRACT_SHA256, ECAPA_PACKAGE_FILENAME, ECAPA_PACKAGE_SHA256,
};
use crate::ecapa_inference::EcapaModel;
use crate::error::{FwError, FwResult};
use crate::model::{
    DiarizationEngine, DiarizationFallbackPolicy, DiarizationRequest, SpeakerCountEstimate,
    SpeakerCountRequest,
};
use crate::orchestrator::CancellationToken;

pub const PUBLIC_MODEL_COMPARISON_SCHEMA_VERSION: &str = "public-diarization-model-comparison-v1";
pub const PUBLIC_MODEL_COMPARISON_RUNNER_VERSION: &str =
    "public-diarization-model-comparison-runner-v1";
pub const PUBLIC_MODEL_COMPARISON_PROTOCOL_VERSION: &str =
    "public-diarization-model-comparison-protocol-v1";
pub const PUBLIC_MODEL_COMPARISON_SCHEDULE_VERSION: &str = "four-lane-balanced-williams-v1";
pub const PUBLIC_MODEL_COMPARISON_SORTFORMER_TIMEOUT_SECONDS: u64 = 1_800;

/// Canonical comparison lanes. The order is part of the retained contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelComparisonLane {
    NativeAcoustic,
    NativeEcapa,
    NativeEcapaFused,
    ExternalSortformer,
}

impl ModelComparisonLane {
    pub const ALL: [Self; 4] = [
        Self::NativeAcoustic,
        Self::NativeEcapa,
        Self::NativeEcapaFused,
        Self::ExternalSortformer,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeAcoustic => "native_acoustic",
            Self::NativeEcapa => "native_ecapa",
            Self::NativeEcapaFused => "native_ecapa_fused",
            Self::ExternalSortformer => "external_sortformer",
        }
    }
}

/// Frozen balanced order used across consecutive sorted observations.
pub const MODEL_COMPARISON_WILLIAMS_SCHEDULE: [[ModelComparisonLane; 4]; 4] = [
    [
        ModelComparisonLane::NativeAcoustic,
        ModelComparisonLane::NativeEcapa,
        ModelComparisonLane::ExternalSortformer,
        ModelComparisonLane::NativeEcapaFused,
    ],
    [
        ModelComparisonLane::NativeEcapa,
        ModelComparisonLane::NativeEcapaFused,
        ModelComparisonLane::NativeAcoustic,
        ModelComparisonLane::ExternalSortformer,
    ],
    [
        ModelComparisonLane::NativeEcapaFused,
        ModelComparisonLane::ExternalSortformer,
        ModelComparisonLane::NativeEcapa,
        ModelComparisonLane::NativeAcoustic,
    ],
    [
        ModelComparisonLane::ExternalSortformer,
        ModelComparisonLane::NativeAcoustic,
        ModelComparisonLane::NativeEcapaFused,
        ModelComparisonLane::NativeEcapa,
    ],
];

#[must_use]
pub const fn model_comparison_schedule_row(observation_index: usize) -> [ModelComparisonLane; 4] {
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
    SortformerModelCapacityExceeded,
    SortformerAdapterUnavailable,
    SortformerRuntimeIneligible,
    NativeExecutionFailed,
    EcapaExecutionFailed,
    SortformerExecutionFailed,
    ScoringFailed,
}

impl ModelComparisonOutcomeCode {
    #[must_use]
    pub const fn is_skip(self) -> bool {
        matches!(
            self,
            Self::EcapaModelUnavailable
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
            Self::SortformerModelCapacityExceeded => "sortformer_model_capacity_exceeded",
            Self::SortformerAdapterUnavailable => "sortformer_adapter_unavailable",
            Self::SortformerRuntimeIneligible => "sortformer_runtime_ineligible",
            Self::NativeExecutionFailed => "native_execution_failed",
            Self::EcapaExecutionFailed => "ecapa_execution_failed",
            Self::SortformerExecutionFailed => "sortformer_execution_failed",
            Self::ScoringFailed => "scoring_failed",
        }
    }
}

/// Authority of one resource measurement. Zero is never used as a sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelComparisonResourceAuthority {
    Measured,
    NotComparableSharedProcess,
    UnavailableChildProcess,
    UnavailableNoProbe,
}

/// What work is included in each lane's measured elapsed time.
///
/// These scopes are intentionally different, so retained wall times are useful
/// deployment observations but are not an inference-throughput leaderboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelComparisonWallTimeScope {
    NativeAcousticPipelineAndScorer,
    NativeEcapaPipelineAndScorerSharedModelLoadExcluded,
    ExternalSortformerColdProcessAndScorerPerObservation,
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
    pub shared_setup_wall_time_ms: Option<u64>,
    pub peak_rss_authority: ModelComparisonResourceAuthority,
    pub authoritative_peak_rss_bytes: Option<u64>,
    pub cancellation_latency_authority: ModelComparisonResourceAuthority,
    pub maximum_cancellation_latency_ms: Option<u64>,
    pub hard_timeout_seconds: Option<u64>,
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

/// One lane's available-case and all-four-common-case reductions.
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
    pub schedule: Vec<[ModelComparisonLane; 4]>,
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
    pub native_rayon_threads: u16,
    pub sortformer_intraop_threads: u16,
    pub sortformer_interop_threads: u16,
    pub sortformer_max_speakers: u16,
    pub sortformer_output_frame_ms: u32,
    pub sortformer_hard_timeout_seconds: u64,
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
    pub sortformer_hard_timeout: Duration,
}

enum EcapaAvailability {
    Ready(EcapaModel),
    Unavailable,
    Invalid,
}

struct InternalLaneOutcome {
    lane: ModelComparisonLane,
    status: ModelComparisonOutcomeStatus,
    code: Option<ModelComparisonOutcomeCode>,
    score: Option<AuthoritativeDiarizationScore>,
    external_provenance_sha256: Option<String>,
    external_runtime_identity: Option<ModelComparisonExternalRuntimeIdentity>,
    attempt_wall_time_ms: u64,
    count: Option<ModelComparisonCountObservation>,
}

#[derive(Debug, Clone, Copy)]
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

#[derive(Default)]
struct LaneReduction {
    outcomes: ModelComparisonOutcomeCounts,
    available: ModelComparisonAccumulator,
    common: ModelComparisonAccumulator,
    timed_attempt_count: u64,
    attempted_wall_time_ms: u64,
    completed_wall_time_ms: u64,
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

/// Execute the frozen four-lane comparison and publish a validated public
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
        sortformer_hard_timeout,
    } = request;
    if evaluation_split != EvaluationSplit::Development {
        return Err(model_comparison_error(
            "split",
            "the uncertified v1 comparison is restricted to the development split",
        ));
    }
    if sortformer_hard_timeout
        != Duration::from_secs(PUBLIC_MODEL_COMPARISON_SORTFORMER_TIMEOUT_SECONDS)
    {
        return Err(model_comparison_error(
            "sortformer_timeout",
            "Sortformer timeout must equal the frozen 1800-second comparison timeout",
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

    let scorer_config = comparison_scorer_config();
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

    let ecapa_setup_started = Instant::now();
    let ecapa = load_ecapa_availability(&is_cancelled)?;
    let ecapa_shared_setup_wall_time_ms = elapsed_millis(ecapa_setup_started);
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
            let attempt_started = Instant::now();
            let mut outcome = execute_lane(
                lane,
                &audio_path,
                &recording_evidence.audio_sha256,
                &samples,
                &normalized_input_sha256,
                &clipped_reference,
                reference_speaker_count,
                &scorer_config,
                &ecapa,
                sortformer_hard_timeout,
                &is_cancelled,
            )?;
            outcome.attempt_wall_time_ms = elapsed_millis(attempt_started);
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
            if outcome.status == ModelComparisonOutcomeStatus::Completed {
                reduction.completed_wall_time_ms = reduction
                    .completed_wall_time_ms
                    .saturating_add(outcome.attempt_wall_time_ms);
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

    let timeout_seconds = sortformer_hard_timeout.as_secs();
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
                    ecapa_shared_setup_wall_time_ms,
                    reduction.timed_attempt_count,
                    reduction.attempted_wall_time_ms,
                    reduction.completed_wall_time_ms,
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
    verify_public_model_comparison_evidence(&evidence)?;

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
            "same_validated_wav_bytes+native_single_decode_f32+full_duration_floor_ms".to_owned(),
        speaker_count_policy: "infer_without_oracle_count_input".to_owned(),
        oracle_count_diagnostic_present: false,
        sortformer_capacity_eligibility:
            "reference_count_over_4_is_declared_ineligible_and_never_passed_to_model".to_owned(),
        speech_activity_authority: "end_to_end_no_reference_sad_no_asr_boundaries".to_owned(),
        overlap_policy: "exclude".to_owned(),
        speaker_boundary_collar_ms: scorer_config.speaker_boundary_collar_ms,
        change_boundary_collar_ms: scorer_config.change_boundary_collar_ms,
        scorer_config_sha256: super::canonical_sha256(&scorer_config)?,
        native_rayon_threads: sortformer_contract.torch_intraop_threads,
        sortformer_intraop_threads: sortformer_contract.torch_intraop_threads,
        sortformer_interop_threads: sortformer_contract.torch_interop_threads,
        sortformer_max_speakers,
        sortformer_output_frame_ms: SORTFORMER_ORACLE_OUTPUT_FRAME_MS,
        sortformer_hard_timeout_seconds: PUBLIC_MODEL_COMPARISON_SORTFORMER_TIMEOUT_SECONDS,
        ecapa_contract_sha256: ECAPA_CONTRACT_SHA256.to_owned(),
        ecapa_package_sha256: ECAPA_PACKAGE_SHA256.to_owned(),
        sortformer_contract_sha256: SORTFORMER_ORACLE_CONTRACT_SHA256.to_owned(),
        native_acoustic_postprocessing:
            "fixed_safe_v1_change+fixed_safe_v1_clustering+unknown_fallback".to_owned(),
        native_ecapa_postprocessing:
            "fixed_safe_v1_change+probabilistic_v1_clustering+unknown_fallback".to_owned(),
        native_ecapa_fused_postprocessing:
            "fixed_safe_v1_change+probabilistic_v1_clustering+unknown_fallback".to_owned(),
        sortformer_postprocessing: "pinned_sortformer_oracle_contract_v2".to_owned(),
        wall_time_policy: "measured_per_observation;native_ecapa_shared_model_load_separate;sortformer_cold_process_per_observation;not_cross_lane_comparable".to_owned(),
        aggregate_only: true,
        production_route_changed: false,
    })
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
        Ok(model) => Ok(EcapaAvailability::Ready(model)),
        Err(error @ FwError::Cancelled(_)) => Err(error),
        Err(_) => Ok(EcapaAvailability::Invalid),
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
    sortformer_hard_timeout: Duration,
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
                || is_cancelled(),
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
                EcapaAvailability::Ready(model) => model,
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
                Err(_) => {
                    return Ok(InternalLaneOutcome::failed(
                        lane,
                        ModelComparisonOutcomeCode::EcapaExecutionFailed,
                    ));
                }
            };
            score_native_report(lane, report, reference, scorer_config)
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
            let token = CancellationToken::unbounded();
            let observation = run_sortformer_observation_with_cancel(
                SortformerObservationRequest {
                    audio_path,
                    expected_audio_sha256: audio_sha256,
                    expected_duration_ms: reference.duration_ms,
                    recording_key: audio_sha256,
                    hard_timeout: sortformer_hard_timeout,
                },
                &token,
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
        | DifferentialSkipReason::ModelContractMismatch
        | DifferentialSkipReason::VersionProbeFailed
        | DifferentialSkipReason::VersionProbeTimedOut
        | DifferentialSkipReason::InvalidVersionOutput => (
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
        | DifferentialSkipReason::OracleRunFailed
        | DifferentialSkipReason::OracleRunTimedOut
        | DifferentialSkipReason::InvalidOracleOutput
        | DifferentialSkipReason::OracleIdentityMismatch => (
            ModelComparisonOutcomeStatus::Failed,
            ModelComparisonOutcomeCode::SortformerExecutionFailed,
        ),
    })
}

fn lane_resource_evidence(
    lane: ModelComparisonLane,
    sortformer_timeout_seconds: u64,
    ecapa_shared_setup_wall_time_ms: u64,
    timed_attempt_count: u64,
    attempted_wall_time_ms: u64,
    completed_wall_time_ms: u64,
) -> ModelComparisonResourceEvidence {
    ModelComparisonResourceEvidence {
        wall_time_authority: ModelComparisonResourceAuthority::Measured,
        wall_time_scope: match lane {
            ModelComparisonLane::NativeAcoustic => {
                ModelComparisonWallTimeScope::NativeAcousticPipelineAndScorer
            }
            ModelComparisonLane::NativeEcapa | ModelComparisonLane::NativeEcapaFused => {
                ModelComparisonWallTimeScope::NativeEcapaPipelineAndScorerSharedModelLoadExcluded
            }
            ModelComparisonLane::ExternalSortformer => {
                ModelComparisonWallTimeScope::ExternalSortformerColdProcessAndScorerPerObservation
            }
        },
        wall_time_cross_lane_comparable: false,
        timed_attempt_count,
        attempted_wall_time_ms,
        completed_wall_time_ms,
        shared_setup_wall_time_ms: matches!(
            lane,
            ModelComparisonLane::NativeEcapa | ModelComparisonLane::NativeEcapaFused
        )
        .then_some(ecapa_shared_setup_wall_time_ms),
        peak_rss_authority: if lane == ModelComparisonLane::ExternalSortformer {
            ModelComparisonResourceAuthority::UnavailableChildProcess
        } else {
            ModelComparisonResourceAuthority::NotComparableSharedProcess
        },
        authoritative_peak_rss_bytes: None,
        cancellation_latency_authority: ModelComparisonResourceAuthority::UnavailableNoProbe,
        maximum_cancellation_latency_ms: None,
        hard_timeout_seconds: (lane == ModelComparisonLane::ExternalSortformer)
            .then_some(sortformer_timeout_seconds),
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
    F: Fn() -> bool + Sync,
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
        ModelComparisonLane::ExternalSortformer => matches!(
            code,
            ModelComparisonOutcomeCode::SortformerModelCapacityExceeded
                | ModelComparisonOutcomeCode::SortformerAdapterUnavailable
                | ModelComparisonOutcomeCode::SortformerRuntimeIneligible
        ),
    });
    let failures_valid = outcomes.failed_by_code.keys().all(|code| match lane {
        ModelComparisonLane::NativeAcoustic => matches!(
            code,
            ModelComparisonOutcomeCode::NativeExecutionFailed
                | ModelComparisonOutcomeCode::ScoringFailed
        ),
        ModelComparisonLane::NativeEcapa | ModelComparisonLane::NativeEcapaFused => matches!(
            code,
            ModelComparisonOutcomeCode::EcapaModelInvalid
                | ModelComparisonOutcomeCode::EcapaExecutionFailed
                | ModelComparisonOutcomeCode::ScoringFailed
        ),
        ModelComparisonLane::ExternalSortformer => matches!(
            code,
            ModelComparisonOutcomeCode::SortformerExecutionFailed
                | ModelComparisonOutcomeCode::ScoringFailed
        ),
    });
    skips_valid && failures_valid
}

fn validate_shared_ecapa_outcomes(
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
            "ECAPA and fused ECAPA lanes disagree with the one shared model-load state",
        ));
    }
    Ok(())
}

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
    if super::canonical_sha256(&evidence.protocol)? != evidence.protocol_sha256
        || evidence.lanes.len() != ModelComparisonLane::ALL.len()
        || evidence.common_complete_recording_count > evidence.observation_count
        || evidence.execution_order_sha256
            != expected_execution_order_sha256(evidence.observation_count)?
        || evidence.order_balance_complete != (evidence.observation_count % schedule_period == 0)
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
            || aggregate.common_complete_case.recording_count
                > aggregate.available_case.recording_count
        {
            return Err(model_comparison_error(
                "aggregate_integrity",
                "model-comparison aggregate counts are inconsistent",
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
        evidence.observation_count.checked_mul(3).ok_or_else(|| {
            model_comparison_error(
                "aggregate_overflow",
                "observation intersection bound overflows the retained range",
            )
        })?,
    );
    if evidence.common_complete_recording_count < minimum_intersection {
        return Err(model_comparison_error(
            "common_complete",
            "common-complete count violates the four-set intersection lower bound",
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
    let ecapa_setup = evidence.lanes[1].resources.shared_setup_wall_time_ms;
    if ecapa_setup.is_none() || ecapa_setup != evidence.lanes[2].resources.shared_setup_wall_time_ms
    {
        return Err(model_comparison_error(
            "resource_setup",
            "ECAPA lanes must share one retained model-load observation",
        ));
    }
    validate_shared_ecapa_outcomes(
        &evidence.lanes[1].outcomes,
        &evidence.lanes[2].outcomes,
        evidence.observation_count,
    )?;
    let sortformer = &evidence.lanes[3];
    if sortformer.outcomes.completed > 0 && evidence.sortformer_runtime_identity.is_none() {
        return Err(model_comparison_error(
            "sortformer_runtime_identity",
            "completed Sortformer outcomes require a retained runtime identity",
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
        lane.resources.authoritative_peak_rss_bytes = None;
        lane.resources.maximum_cancellation_latency_ms = None;
        lane.resources.attempted_wall_time_ms = 0;
        lane.resources.completed_wall_time_ms = 0;
        lane.resources.shared_setup_wall_time_ms = None;
        lane.resources.hard_timeout_seconds = None;
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
        || metrics.full_timeline_total_absolute_speaker_count_error
            < metrics
                .recording_count
                .saturating_sub(metrics.full_timeline_exact_speaker_count)
        || metrics.count_estimate_total_absolute_error
            < metrics
                .count_estimate_resolved
                .saturating_sub(metrics.count_estimate_exact)
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
    let expected_scope = match lane {
        ModelComparisonLane::NativeAcoustic => {
            ModelComparisonWallTimeScope::NativeAcousticPipelineAndScorer
        }
        ModelComparisonLane::NativeEcapa | ModelComparisonLane::NativeEcapaFused => {
            ModelComparisonWallTimeScope::NativeEcapaPipelineAndScorerSharedModelLoadExcluded
        }
        ModelComparisonLane::ExternalSortformer => {
            ModelComparisonWallTimeScope::ExternalSortformerColdProcessAndScorerPerObservation
        }
    };
    let expected_peak_authority = if lane == ModelComparisonLane::ExternalSortformer {
        ModelComparisonResourceAuthority::UnavailableChildProcess
    } else {
        ModelComparisonResourceAuthority::NotComparableSharedProcess
    };
    let setup_valid = if matches!(
        lane,
        ModelComparisonLane::NativeEcapa | ModelComparisonLane::NativeEcapaFused
    ) {
        resources
            .shared_setup_wall_time_ms
            .is_some_and(|value| value > 0)
    } else {
        resources.shared_setup_wall_time_ms.is_none()
    };
    if resources.wall_time_authority != ModelComparisonResourceAuthority::Measured
        || resources.wall_time_scope != expected_scope
        || resources.wall_time_cross_lane_comparable
        || resources.timed_attempt_count != outcomes.declared
        || resources.attempted_wall_time_ms < resources.completed_wall_time_ms
        || resources.attempted_wall_time_ms < outcomes.declared
        || (outcomes.completed == 0 && resources.completed_wall_time_ms != 0)
        || resources.completed_wall_time_ms < outcomes.completed
        || canonical(resources.completed_wall_time_ms as f64 / 1_000.0) != available.wall_time_sec
        || !setup_valid
        || resources.peak_rss_authority != expected_peak_authority
        || resources.authoritative_peak_rss_bytes.is_some()
        || resources.cancellation_latency_authority
            != ModelComparisonResourceAuthority::UnavailableNoProbe
        || resources.maximum_cancellation_latency_ms.is_some()
        || resources.hard_timeout_seconds
            != (lane == ModelComparisonLane::ExternalSortformer)
                .then_some(PUBLIC_MODEL_COMPARISON_SORTFORMER_TIMEOUT_SECONDS)
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

    #[test]
    fn williams_schedule_balances_every_lane_and_position() {
        for lane in ModelComparisonLane::ALL {
            let mut positions = [0u32; 4];
            for row in MODEL_COMPARISON_WILLIAMS_SCHEDULE {
                positions[row
                    .iter()
                    .position(|candidate| *candidate == lane)
                    .expect("lane")] += 1;
            }
            assert_eq!(positions, [1, 1, 1, 1]);
        }
        assert_eq!(
            model_comparison_schedule_row(4),
            MODEL_COMPARISON_WILLIAMS_SCHEDULE[0]
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
    fn resource_authority_never_uses_zero_as_unavailable() {
        let mut resources = ModelComparisonResourceEvidence {
            wall_time_authority: ModelComparisonResourceAuthority::Measured,
            wall_time_scope: ModelComparisonWallTimeScope::NativeAcousticPipelineAndScorer,
            wall_time_cross_lane_comparable: false,
            timed_attempt_count: 1,
            attempted_wall_time_ms: 250,
            completed_wall_time_ms: 250,
            shared_setup_wall_time_ms: None,
            peak_rss_authority: ModelComparisonResourceAuthority::NotComparableSharedProcess,
            authoritative_peak_rss_bytes: None,
            cancellation_latency_authority: ModelComparisonResourceAuthority::UnavailableNoProbe,
            maximum_cancellation_latency_ms: None,
            hard_timeout_seconds: None,
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
        resources.authoritative_peak_rss_bytes = Some(0);
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
    fn shared_ecapa_load_state_must_cover_both_lanes() {
        let unavailable = ModelComparisonOutcomeCounts {
            declared: 2,
            skipped: 2,
            skipped_by_code: BTreeMap::from([(
                ModelComparisonOutcomeCode::EcapaModelUnavailable,
                2,
            )]),
            ..ModelComparisonOutcomeCounts::default()
        };
        assert!(validate_shared_ecapa_outcomes(&unavailable, &unavailable, 2).is_ok());
        let ready = ModelComparisonOutcomeCounts {
            declared: 2,
            completed: 2,
            ..ModelComparisonOutcomeCounts::default()
        };
        assert!(validate_shared_ecapa_outcomes(&unavailable, &ready, 2).is_err());
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
