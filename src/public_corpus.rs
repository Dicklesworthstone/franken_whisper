//! Reproducible adapters for public or user-licensed diarization corpora.
//!
//! This module deliberately separates path-bearing local preparation inputs
//! from the path-free corpus, reference, leakage, and integrity evidence that
//! can be retained externally. It never copies source media and refuses to
//! write generated annotations inside the project checkout.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::diarization::{
    ACOUSTIC_CHANGE_CALIBRATION_FIT_VERSION, ACOUSTIC_CHANGE_CALIBRATION_VERSION,
    ACOUSTIC_CLUSTERING_PROBABILISTIC_VERSION, ACOUSTIC_SIDECAR_FUSION_VERSION,
    ACOUSTIC_SIDECAR_STUDY_SCHEMA_VERSION, AcousticBoundaryHints, AcousticChangeDetectorMode,
    AcousticClusteringEvaluationEvidence, AcousticClusteringFallbackReason, AcousticClusteringMode,
    AcousticDiarizationInput, AcousticFeatureAblation, AcousticScatteringMode,
    AcousticSidecarEvaluationRequest, AcousticSidecarFusionCalibration,
    AcousticSidecarFusionEvaluationEvidence, AcousticSidecarStudy, AcousticSidecarStudyConfig,
    AcousticSidecarStudyMode, AcousticSidecarStudyObservation, AcousticTrajectoryWaveletMode,
    ChangePointScore,
    DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION, DIARIZATION_HYPOTHESIS_SCHEMA_VERSION,
    DIARIZATION_REFERENCE_SCHEMA_VERSION, DIARIZATION_SCORER_VERSION, DiarizationCorpusManifest,
    DiarizationHypothesisDocument, DiarizationLeakageAudit, DiarizationReferenceDocument,
    DiarizationScorerConfig, EvaluationOverlapPolicy, EvaluationPerformanceObservation,
    EvaluationRegion, EvaluationSplit, EvaluationTurn, EvaluationWord, acoustic_change_calibration,
    acoustic_change_calibration_sha256, acoustic_feature_schema_sha256,
    acoustic_sidecar_fusion_configuration_sha256,
    acoustic_sidecar_observation_owner_contrast, acoustic_sidecar_study_config_sha256,
    acoustic_speaker_pair_calibration_sha256, audit_diarization_manifest,
    diarize_acoustic_pcm_with_modes_evidence, diarize_acoustic_pcm_with_sidecar_evidence,
    extract_acoustic_features, parse_diarization_corpus_manifest, parse_diarization_reference,
    score_change_points, score_diarization_documents,
    select_acoustic_change_evidence_at_threshold, speaker_change_points_ms,
    verify_leakage_audit_hash,
};
use crate::error::{FwError, FwResult};
use crate::model::{DiarizationEngine, DiarizationRequest, SpeakerCountRequest};

/// Schema identity for the path-bearing, external-only adapter input.
pub const PUBLIC_CORPUS_INPUT_SCHEMA_VERSION: &str = "public-diarization-corpus-input-v2";
/// Schema identity for the path-free generated bundle.
pub const PUBLIC_CORPUS_BUNDLE_SCHEMA_VERSION: &str = "public-diarization-corpus-bundle-v2";
/// Frozen implementation identity for this adapter.
pub const PUBLIC_CORPUS_ADAPTER_VERSION: &str = "public-diarization-corpus-adapter-v2";
/// Schema identity for the built-in public-corpus registry.
pub const PUBLIC_CORPUS_REGISTRY_SCHEMA_VERSION: &str = "public-diarization-corpus-registry-v1";
/// Schema for optional transcript-free aligned-word annotation documents.
pub const PUBLIC_CORPUS_WORD_ANNOTATION_SCHEMA_VERSION: &str =
    "public-diarization-word-annotation-v1";
/// Schema identity for path-free public representation-ablation evidence.
pub const PUBLIC_CORPUS_ABLATION_SCHEMA_VERSION: &str = "public-diarization-acoustic-ablation-v8";
/// Frozen public ablation implementation identity.
pub const PUBLIC_CORPUS_ABLATION_RUNNER_VERSION: &str =
    "public-diarization-acoustic-ablation-runner-v8";
/// Schema identity for the separate aggregate-only acoustic sidecar study.
pub const PUBLIC_CORPUS_SIDECAR_STUDY_SCHEMA_VERSION: &str =
    "public-diarization-acoustic-sidecar-study-v1";
/// Frozen implementation identity for the aggregate-only sidecar runner.
pub const PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION: &str =
    "public-diarization-acoustic-sidecar-study-runner-v1";
/// Identity of the bounded development calibration fit.
pub const PUBLIC_CORPUS_SIDECAR_CALIBRATION_FIT_VERSION: &str =
    "public-sidecar-boundary-calibration-grid-v1";
/// Identity of the deterministic conditional-pair scorer.
pub const PUBLIC_CORPUS_SIDECAR_PAIR_SCORER_VERSION: &str =
    "public-sidecar-conditional-pair-scorer-v1";
/// Identity of the deterministic paired-recording uncertainty calculation.
pub const PUBLIC_CORPUS_SIDECAR_UNCERTAINTY_VERSION: &str =
    "public-sidecar-paired-bootstrap-sha-counter-v1";
/// Identity of the fail-closed development selector and held-out gate.
pub const PUBLIC_CORPUS_SIDECAR_SELECTION_POLICY_VERSION: &str =
    "public-sidecar-selection-policy-v1";
/// A candidate must reduce development micro-DER by at least one percent.
pub const PUBLIC_CORPUS_SIDECAR_MIN_DEVELOPMENT_DER_IMPROVEMENT: f64 = 0.01;
/// A candidate may regress macro-JER by no more than one absolute point.
pub const PUBLIC_CORPUS_SIDECAR_MAX_MACRO_JER_REGRESSION: f64 = 0.01;
/// At least one quarter of submitted frames must yield comparable evidence.
pub const PUBLIC_CORPUS_SIDECAR_MIN_COMPARABLE_FRAME_COVERAGE: f64 = 0.25;
/// Minimum conditional same/different ROC AUC admitted on development.
pub const PUBLIC_CORPUS_SIDECAR_MIN_PAIR_ROC_AUC: f64 = 0.55;
/// Maximum Brier score admitted for conditional different-speaker pairs.
pub const PUBLIC_CORPUS_SIDECAR_MAX_PAIR_BRIER: f64 = 0.25;
/// Maximum expected calibration error admitted for conditional pairs.
pub const PUBLIC_CORPUS_SIDECAR_MAX_PAIR_ECE: f64 = 0.10;
/// Maximum rate at which channel contrast dominates voice contrast on
/// comparable different-speaker pairs.
pub const PUBLIC_CORPUS_SIDECAR_MAX_CHANNEL_CONFOUND_RATE: f64 = 0.50;

const PUBLIC_SIDECAR_BOUNDARY_COLLAR_MS: u64 = 250;
const PUBLIC_SIDECAR_RELIABILITY_BINS: usize = 10;
const PUBLIC_SIDECAR_FIT_BINS: usize = 256;
const PUBLIC_SIDECAR_PAIR_SCORE_BINS: usize = 256;
const PUBLIC_SIDECAR_PAIR_LAGS_FRAMES: [usize; 4] = [25, 50, 100, 200];
const PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING: usize = 4_096;
const PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS: usize = 1;
const PUBLIC_SIDECAR_BOOTSTRAP_REPLICATES: usize = 2_000;
/// Predeclared minimum relative micro-DER reduction required on development.
pub const PUBLIC_CORPUS_MIN_DEVELOPMENT_DER_IMPROVEMENT: f64 = 0.05;
/// Predeclared relative change-F1 gain for the calibrated detector.
pub const PUBLIC_CORPUS_MIN_CHANGE_F1_IMPROVEMENT: f64 = 0.20;
/// Maximum absolute DER or JER regression admitted during detector tuning.
pub const PUBLIC_CORPUS_MAX_CHANGE_DER_JER_REGRESSION: f64 = 0.01;
/// Maximum event-level Brier score admitted for a calibrated detector.
pub const PUBLIC_CORPUS_MAX_CHANGE_BRIER: f64 = 0.25;
/// Maximum event-level expected calibration error admitted for promotion.
pub const PUBLIC_CORPUS_MAX_CHANGE_ECE: f64 = 0.10;
/// Minimum relative micro-DER gain required for probabilistic clustering.
pub const PUBLIC_CORPUS_MIN_CLUSTERING_DER_IMPROVEMENT: f64 = 0.05;
/// Maximum macro-JER regression admitted while tuning clustering.
pub const PUBLIC_CORPUS_MAX_CLUSTERING_JER_REGRESSION: f64 = 0.01;
/// Maximum assignment-confidence ECE admitted for clustering promotion.
pub const PUBLIC_CORPUS_MAX_CLUSTERING_ECE: f64 = 0.10;
/// Minimum deterministic perturbation agreement admitted for count selection.
pub const PUBLIC_CORPUS_MIN_CLUSTERING_COUNT_STABILITY: f64 = 2.0 / 3.0;
/// Maximum absolute loss of reference-time coverage allowed for a candidate.
pub const PUBLIC_CORPUS_MAX_CLUSTERING_COVERAGE_REGRESSION: f64 = 0.01;
/// Maximum selective-risk regression allowed on development or held-out data.
pub const PUBLIC_CORPUS_MAX_CLUSTERING_SELECTIVE_RISK_REGRESSION: f64 = 0.0;
/// Identity for the fail-closed public development selector.
pub const PUBLIC_CORPUS_CHANGE_SELECTION_POLICY_VERSION: &str =
    "public-change-detector-selection-v1";
const PUBLIC_CORPUS_CHANGE_CANDIDATE_ORDER: [AcousticChangeDetectorMode; 3] = [
    AcousticChangeDetectorMode::CalibratedPosterior,
    AcousticChangeDetectorMode::PageHinkleyV1,
    AcousticChangeDetectorMode::BayesianTwoRegimeV1,
];
const PUBLIC_CHANGE_DIAGNOSTIC_COLLARS_MS: [u64; 3] = [100, 250, 500];
const PUBLIC_CHANGE_THRESHOLD_SWEEP: [f64; 19] = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95,
];

const MAX_DESCRIPTOR_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ANNOTATION_BYTES: u64 = 64 * 1024 * 1024;
const MAX_RECORDINGS: usize = 100_000;
const MAX_TURNS_PER_RECORDING: usize = 1_000_000;
const MAX_TOTAL_TURNS: usize = 2_000_000;
const MAX_WORDS_PER_RECORDING: usize = 2_000_000;
const MAX_TOTAL_WORDS: usize = 4_000_000;
const HASH_HEX_LEN: usize = 64;
// The current decoder necessarily holds both source bytes and f32 samples.
// Keep the fail-closed cap well below multi-gigabyte allocations; longer
// corpora must be partitioned into independently checksummed recordings.
const MAX_EVALUATION_AUDIO_BYTES: u64 = 256 * 1024 * 1024;

/// How a registry entry freezes its train/development/test assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicCorpusSplitPolicy {
    /// The official AMI scenario-only family split is checked in code.
    AmiScenarioOfficialV1,
    /// The external descriptor is frozen by its SHA-256 and then leakage-audited.
    ExternalDescriptorV1,
}

/// Explicit authority boundary for tuning versus held-out certification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicCorpusEvaluationStage {
    Development,
    Certification,
}

impl PublicCorpusEvaluationStage {
    #[must_use]
    const fn selected_split(self) -> EvaluationSplit {
        match self {
            Self::Development => EvaluationSplit::Development,
            Self::Certification => EvaluationSplit::Test,
        }
    }
}

/// One built-in corpus source and its reproducibility/licensing contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusRegistryEntry {
    pub corpus_key: String,
    pub description: String,
    pub authoritative_url: String,
    pub license_id: String,
    pub license_url: String,
    /// Exact CLI value required by `--license-ack`.
    pub license_acknowledgement_id: String,
    pub split_policy: PublicCorpusSplitPolicy,
    pub expected_local_layout: String,
    pub conversion_contract: String,
    pub upstream_integrity_note: String,
    pub condition_tags: Vec<String>,
}

/// Complete built-in registry emitted by robot-safe CLI output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusRegistry {
    pub schema_version: String,
    pub adapter_version: String,
    pub entries: Vec<PublicCorpusRegistryEntry>,
}

/// Path-free integrity and media-layout evidence for one recording.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusRecordingEvidence {
    pub recording_id: String,
    pub split: EvaluationSplit,
    pub audio_sha256: String,
    pub annotation_sha256: String,
    pub word_annotation_sha256: Option<String>,
    pub reference_sha256: String,
    pub sample_rate_hz: u32,
    pub channel_count: u16,
    pub selected_channel: u16,
    pub turn_count: usize,
    pub word_count: usize,
    pub overlap_turn_count: usize,
    pub ignored_region_count: usize,
}

/// Generated public-corpus evidence. This contains no paths, URIs, or text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusBundle {
    pub schema_version: String,
    pub adapter_version: String,
    pub corpus_key: String,
    pub source_version: String,
    pub license_id: String,
    pub license_acknowledgement_id: String,
    pub descriptor_sha256: String,
    pub manifest: DiarizationCorpusManifest,
    pub leakage_audit: DiarizationLeakageAudit,
    pub references: Vec<DiarizationReferenceDocument>,
    pub recordings: Vec<PublicCorpusRecordingEvidence>,
    /// Hash of the complete bundle with this field temporarily empty.
    pub bundle_sha256: String,
}

/// Frozen evaluation protocol attached to every public ablation artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusAblationProtocol {
    pub oracle_vad: bool,
    pub oracle_speaker_count: bool,
    pub maximum_recording_duration_ms: Option<u64>,
    pub prefix_selection: String,
    pub rss_observation: String,
    pub diarization_request: DiarizationRequest,
    pub diarization_request_sha256: String,
    pub change_calibration_id: String,
    pub change_calibration_fit_id: String,
    pub change_calibration_sha256: String,
    pub change_decision_probability: f64,
    pub change_calibration_bins: usize,
    pub change_selection_policy_id: String,
    pub change_selection_policy_sha256: String,
    pub speaker_pair_calibration_id: String,
    pub speaker_pair_calibration_sha256: String,
}

/// One aggregate event-confidence reliability bin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicChangeReliabilityBin {
    pub index: usize,
    pub lower_probability: f64,
    pub upper_probability: f64,
    pub observation_count: u64,
    pub positive_count: u64,
    pub mean_probability: Option<f64>,
    pub empirical_frequency: Option<f64>,
}

/// Aggregate operating-point metrics at one declared boundary collar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicChangeCollarMetrics {
    pub collar_ms: u64,
    pub reference_count: u64,
    pub hypothesis_count: u64,
    pub matched_count: u64,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub mean_absolute_error_sec: Option<f64>,
    pub p50_absolute_error_sec: Option<f64>,
    pub p90_absolute_error_sec: Option<f64>,
    pub p95_absolute_error_sec: Option<f64>,
}

/// Development diagnostic for one predeclared probability threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicChangeThresholdSweepPoint {
    pub threshold: f64,
    pub collar_ms: u64,
    pub reference_count: u64,
    pub hypothesis_count: u64,
    pub matched_count: u64,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub mean_absolute_error_sec: Option<f64>,
    pub p50_absolute_error_sec: Option<f64>,
    pub p90_absolute_error_sec: Option<f64>,
    pub p95_absolute_error_sec: Option<f64>,
}

/// One permutation-invariant reference/hypothesis count confusion cell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicSpeakerCountConfusionCell {
    pub reference_speakers: u32,
    pub hypothesis_speakers: u32,
    pub recording_count: u64,
}

/// Count-posterior quality stratified by the reference speaker count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicSpeakerCountStratum {
    pub reference_speakers: u32,
    pub recording_count: u64,
    pub posterior_recording_count: u64,
    pub unresolved_recording_count: u64,
    pub zero_reference_probability_count: u64,
    pub exact_speaker_count_rate: Option<f64>,
    pub mean_negative_log_likelihood: Option<f64>,
    pub mean_brier_score: Option<f64>,
    pub top_k_coverage: Option<f64>,
    pub credible_set_coverage: Option<f64>,
}

/// Stable duration bucket for count-posterior calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicSpeakerCountDurationBucket {
    UpToThirtySeconds,
    UpToTwoMinutes,
    UpToTenMinutes,
    LongerThanTenMinutes,
}

/// Count-posterior quality stratified by scored recording duration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicSpeakerCountDurationStratum {
    pub duration_bucket: PublicSpeakerCountDurationBucket,
    pub recording_count: u64,
    pub posterior_recording_count: u64,
    pub unresolved_recording_count: u64,
    pub zero_reference_probability_count: u64,
    pub exact_speaker_count_rate: Option<f64>,
    pub mean_negative_log_likelihood: Option<f64>,
    pub mean_brier_score: Option<f64>,
    pub top_k_coverage: Option<f64>,
    pub credible_set_coverage: Option<f64>,
}

/// Aggregate metrics for one feature ablation and one frozen corpus split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusAblationSplit {
    pub split: EvaluationSplit,
    pub recording_count: u64,
    pub reference_speaker_time_sec: f64,
    pub micro_der: Option<f64>,
    pub macro_der: Option<f64>,
    pub macro_jer: Option<f64>,
    pub speaker_confusion_sec: f64,
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
    pub change_precision: Option<f64>,
    pub change_recall: Option<f64>,
    pub change_f1: Option<f64>,
    pub change_mean_absolute_error_sec: Option<f64>,
    pub change_event_observation_count: u64,
    pub change_event_positive_count: u64,
    pub change_brier_score: Option<f64>,
    pub change_expected_calibration_error: Option<f64>,
    pub change_reliability: Vec<PublicChangeReliabilityBin>,
    pub change_collar_metrics: Vec<PublicChangeCollarMetrics>,
    pub change_threshold_sweep: Vec<PublicChangeThresholdSweepPoint>,
    pub exact_speaker_count_rate: Option<f64>,
    pub mean_signed_speaker_count_error: Option<f64>,
    pub mean_absolute_speaker_count_error: Option<f64>,
    pub p50_absolute_speaker_count_error: Option<f64>,
    pub p90_absolute_speaker_count_error: Option<f64>,
    pub p95_absolute_speaker_count_error: Option<f64>,
    pub maximum_absolute_speaker_count_error: Option<u64>,
    pub speaker_count_confusion: Vec<PublicSpeakerCountConfusionCell>,
    pub speaker_count_strata: Vec<PublicSpeakerCountStratum>,
    pub speaker_count_duration_strata: Vec<PublicSpeakerCountDurationStratum>,
    pub count_posterior_recording_count: u64,
    pub count_posterior_unavailable_count: u64,
    pub count_unresolved_recording_count: u64,
    pub count_zero_reference_probability_count: u64,
    pub count_mean_negative_log_likelihood: Option<f64>,
    pub count_mean_brier_score: Option<f64>,
    pub count_top_k_coverage: Option<f64>,
    pub count_credible_set_coverage: Option<f64>,
    pub count_mean_entropy_bits: Option<f64>,
    pub dominant_collapse_recording_count: u64,
    pub reference_collapse_recording_count: u64,
    pub phantom_speaker_count: u64,
    pub collapsed_reference_speaker_count: u64,
    pub mean_effective_speaker_count: Option<f64>,
    pub mean_dominant_speaker_share: Option<f64>,
    pub p90_dominant_speaker_share: Option<f64>,
    pub p99_dominant_speaker_share: Option<f64>,
    pub maximum_dominant_speaker_share: Option<f64>,
    pub mean_unknown_speaker_share: Option<f64>,
    pub maximum_unknown_speaker_share: Option<f64>,
    pub mean_minority_reference_recall: Option<f64>,
    pub reference_word_count: u64,
    pub scored_word_count: u64,
    pub correct_word_count: u64,
    pub incorrect_word_count: u64,
    pub unknown_word_count: u64,
    pub excluded_word_count: u64,
    pub micro_word_diarization_error_rate: Option<f64>,
    pub macro_word_diarization_error_rate: Option<f64>,
    pub selective_reference_speaker_time_sec: f64,
    pub selective_covered_speaker_time_sec: f64,
    pub selective_correct_covered_speaker_time_sec: f64,
    pub selective_error_covered_speaker_time_sec: f64,
    pub selective_unknown_speaker_time_sec: f64,
    pub selective_coverage: Option<f64>,
    pub selective_risk: Option<f64>,
    pub assignment_observed_duration_sec: f64,
    pub assignment_opportunity_duration_sec: f64,
    pub assignment_coverage: Option<f64>,
    pub assignment_brier_score: Option<f64>,
    pub assignment_expected_calibration_error: Option<f64>,
    pub mean_speaker_count_stability: Option<f64>,
    pub clustering_fallback_count: u64,
    pub clustering_insufficient_voice_fallback_count: u64,
    pub clustering_invalid_posterior_fallback_count: u64,
    pub clustering_unstable_count_fallback_count: u64,
    pub audio_duration_sec: f64,
    pub wall_time_sec: f64,
    pub real_time_factor: Option<f64>,
    pub sampled_peak_rss_bytes: u64,
}

/// Aggregate public evidence for one frozen representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusAblationVariant {
    pub ablation: AcousticFeatureAblation,
    pub feature_schema: String,
    pub feature_schema_sha256: String,
    pub feature_configuration_sha256: String,
    pub splits: Vec<PublicCorpusAblationSplit>,
}

/// Aggregate public evidence for one frozen speaker-change detector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusChangeDetectorVariant {
    pub detector_mode: AcousticChangeDetectorMode,
    pub feature_ablation: AcousticFeatureAblation,
    pub feature_schema_sha256: String,
    pub configuration_sha256: String,
    pub splits: Vec<PublicCorpusAblationSplit>,
}

/// Aggregate public evidence for one frozen acoustic clustering strategy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusClusteringVariant {
    pub clustering_mode: AcousticClusteringMode,
    pub detector_mode: AcousticChangeDetectorMode,
    pub feature_ablation: AcousticFeatureAblation,
    pub configuration_sha256: String,
    pub splits: Vec<PublicCorpusAblationSplit>,
}

/// Development decision for probabilistic clustering versus fixed-safe v1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusClusteringDevelopmentGate {
    pub split: EvaluationSplit,
    pub candidate: AcousticClusteringMode,
    pub baseline: AcousticClusteringMode,
    pub minimum_relative_micro_der_improvement: f64,
    pub maximum_macro_jer_regression: f64,
    pub maximum_assignment_expected_calibration_error: f64,
    pub minimum_mean_speaker_count_stability: f64,
    pub maximum_selective_coverage_regression: f64,
    pub maximum_selective_risk_regression: f64,
    pub relative_micro_der_improvement: Option<f64>,
    pub macro_jer_delta: Option<f64>,
    pub speaker_confusion_delta_sec: Option<f64>,
    pub overlap_f1_delta: Option<f64>,
    pub mean_absolute_speaker_count_error_delta: Option<f64>,
    pub selective_coverage_regression: Option<f64>,
    pub selective_risk_delta: Option<f64>,
    pub candidate_assignment_expected_calibration_error: Option<f64>,
    pub candidate_mean_speaker_count_stability: Option<f64>,
    pub candidate_fallback_count: u64,
    pub passed: bool,
}

/// Held-out non-regression decision for a locked clustering candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusClusteringHeldOutGate {
    pub split: EvaluationSplit,
    pub candidate: AcousticClusteringMode,
    pub baseline: AcousticClusteringMode,
    pub micro_der_delta: Option<f64>,
    pub macro_jer_delta: Option<f64>,
    pub overlap_f1_delta: Option<f64>,
    pub mean_absolute_speaker_count_error_delta: Option<f64>,
    pub selective_coverage_regression: Option<f64>,
    pub selective_risk_delta: Option<f64>,
    pub candidate_assignment_expected_calibration_error: Option<f64>,
    pub candidate_fallback_count: u64,
    pub passed: bool,
}

/// Development decision for the calibrated detector versus fixed-safe v1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusChangeDevelopmentGate {
    pub split: EvaluationSplit,
    pub candidate: AcousticChangeDetectorMode,
    pub baseline: AcousticChangeDetectorMode,
    pub minimum_relative_change_f1_improvement: f64,
    pub maximum_der_jer_regression: f64,
    pub maximum_brier_score: f64,
    pub maximum_expected_calibration_error: f64,
    pub relative_change_f1_improvement: Option<f64>,
    pub mean_absolute_timing_error_delta_sec: Option<f64>,
    pub micro_der_delta: Option<f64>,
    pub macro_jer_delta: Option<f64>,
    pub candidate_brier_score: Option<f64>,
    pub candidate_expected_calibration_error: Option<f64>,
    pub passed: bool,
}

/// Held-out non-regression decision for the locked calibrated detector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusChangeHeldOutGate {
    pub split: EvaluationSplit,
    pub candidate: AcousticChangeDetectorMode,
    pub baseline: AcousticChangeDetectorMode,
    pub change_f1_delta: Option<f64>,
    pub timing_error_delta_sec: Option<f64>,
    pub micro_der_delta: Option<f64>,
    pub macro_jer_delta: Option<f64>,
    pub candidate_brier_score: Option<f64>,
    pub candidate_expected_calibration_error: Option<f64>,
    pub passed: bool,
}

/// Predeclared development-set improvement decision for full v2 versus v1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusDevelopmentGate {
    pub split: EvaluationSplit,
    pub candidate: AcousticFeatureAblation,
    pub baseline: AcousticFeatureAblation,
    pub minimum_relative_micro_der_improvement: f64,
    pub relative_micro_der_improvement: Option<f64>,
    pub macro_jer_delta: Option<f64>,
    pub change_f1_delta: Option<f64>,
    pub passed: bool,
}

/// Held-out non-regression decision comparing full v2 with the original v1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusHeldOutGate {
    pub split: EvaluationSplit,
    pub candidate: AcousticFeatureAblation,
    pub baseline: AcousticFeatureAblation,
    pub micro_der_delta: Option<f64>,
    pub macro_jer_delta: Option<f64>,
    pub passed: bool,
}

/// Path-free, transcript-free public accuracy/performance evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusAblationEvidence {
    pub schema_version: String,
    pub runner_version: String,
    pub scorer_version: String,
    pub corpus_key: String,
    pub source_version: String,
    pub bundle_sha256: String,
    pub descriptor_sha256: String,
    pub scorer_config: DiarizationScorerConfig,
    pub scorer_config_sha256: String,
    pub evaluation_stage: PublicCorpusEvaluationStage,
    pub locked_development_result_sha256: Option<String>,
    pub locked_development_accuracy_sha256: Option<String>,
    pub selected_change_detector_mode: AcousticChangeDetectorMode,
    pub selected_clustering_mode: AcousticClusteringMode,
    pub protocol: PublicCorpusAblationProtocol,
    pub variants: Vec<PublicCorpusAblationVariant>,
    pub change_detector_variants: Vec<PublicCorpusChangeDetectorVariant>,
    pub clustering_variants: Vec<PublicCorpusClusteringVariant>,
    pub development_gate: Option<PublicCorpusDevelopmentGate>,
    pub held_out_gate: Option<PublicCorpusHeldOutGate>,
    pub change_development_gate: Option<PublicCorpusChangeDevelopmentGate>,
    pub change_held_out_gate: Option<PublicCorpusChangeHeldOutGate>,
    pub clustering_development_gate: Option<PublicCorpusClusteringDevelopmentGate>,
    pub clustering_held_out_gate: Option<PublicCorpusClusteringHeldOutGate>,
    /// Hash with wall-time, RSS, and RTF observations normalized away.
    pub deterministic_accuracy_sha256: String,
    /// Hash of this evidence with this field temporarily empty.
    pub result_sha256: String,
}

/// External-only inputs and outputs for one frozen public-corpus ablation.
///
/// This request deliberately has no `Debug` or serialization implementation:
/// its paths are operator-local and must never enter diagnostics or retained
/// evidence.
pub struct PublicCorpusAblationRequest<'a> {
    pub project_root: &'a Path,
    pub input_root: &'a Path,
    pub descriptor_path: &'a Path,
    pub bundle_output_path: &'a Path,
    pub evidence_output_path: &'a Path,
    pub license_acknowledgement_id: &'a str,
    pub maximum_recording_duration_ms: Option<u64>,
    pub evaluation_stage: PublicCorpusEvaluationStage,
    pub locked_development_evidence_path: Option<&'a Path>,
}

/// Frozen lane identity for the aggregate-only acoustic sidecar study.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicCorpusSidecarLane {
    FullV2Baseline,
    FrameHaarL4,
    FrameD4L4,
    Modulation,
    FrameHaarL4AndModulation,
    FrameD4L4AndModulation,
    TrajectoryHaarL4,
    TrajectoryD4L4,
    ScatteringFirstOrder,
    ScatteringSecondOrder,
    ScatteringFirstAndSecondOrder,
    AllHaarL4,
    AllD4L4,
}

impl PublicCorpusSidecarLane {
    /// Canonical lane order. This order is part of the retained artifact.
    pub const ALL: [Self; 13] = [
        Self::FullV2Baseline,
        Self::FrameHaarL4,
        Self::FrameD4L4,
        Self::Modulation,
        Self::FrameHaarL4AndModulation,
        Self::FrameD4L4AndModulation,
        Self::TrajectoryHaarL4,
        Self::TrajectoryD4L4,
        Self::ScatteringFirstOrder,
        Self::ScatteringSecondOrder,
        Self::ScatteringFirstAndSecondOrder,
        Self::AllHaarL4,
        Self::AllD4L4,
    ];

    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::FullV2Baseline => "full_v2_baseline",
            Self::FrameHaarL4 => "frame_haar_l4",
            Self::FrameD4L4 => "frame_d4_l4",
            Self::Modulation => "modulation",
            Self::FrameHaarL4AndModulation => "frame_haar_l4_and_modulation",
            Self::FrameD4L4AndModulation => "frame_d4_l4_and_modulation",
            Self::TrajectoryHaarL4 => "trajectory_haar_l4",
            Self::TrajectoryD4L4 => "trajectory_d4_l4",
            Self::ScatteringFirstOrder => "scattering_first_order",
            Self::ScatteringSecondOrder => "scattering_second_order",
            Self::ScatteringFirstAndSecondOrder => "scattering_first_and_second_order",
            Self::AllHaarL4 => "all_haar_l4",
            Self::AllD4L4 => "all_d4_l4",
        }
    }

    const fn study_config(self) -> AcousticSidecarStudyConfig {
        match self {
            Self::FullV2Baseline => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Off,
                frame_wavelet_levels: 0,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Off,
                trajectory_wavelet_levels: 0,
                scattering_mode: AcousticScatteringMode::Off,
            },
            Self::FrameHaarL4 => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Haar,
                frame_wavelet_levels: 4,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Off,
                trajectory_wavelet_levels: 0,
                scattering_mode: AcousticScatteringMode::Off,
            },
            Self::FrameD4L4 => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::DaubechiesFourTap,
                frame_wavelet_levels: 4,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Off,
                trajectory_wavelet_levels: 0,
                scattering_mode: AcousticScatteringMode::Off,
            },
            Self::Modulation => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Modulation,
                frame_wavelet_levels: 0,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Off,
                trajectory_wavelet_levels: 0,
                scattering_mode: AcousticScatteringMode::Off,
            },
            Self::FrameHaarL4AndModulation => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::HaarAndModulation,
                frame_wavelet_levels: 4,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Off,
                trajectory_wavelet_levels: 0,
                scattering_mode: AcousticScatteringMode::Off,
            },
            Self::FrameD4L4AndModulation => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::DaubechiesFourTapAndModulation,
                frame_wavelet_levels: 4,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Off,
                trajectory_wavelet_levels: 0,
                scattering_mode: AcousticScatteringMode::Off,
            },
            Self::TrajectoryHaarL4 => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Off,
                frame_wavelet_levels: 0,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Haar,
                trajectory_wavelet_levels: 4,
                scattering_mode: AcousticScatteringMode::Off,
            },
            Self::TrajectoryD4L4 => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Off,
                frame_wavelet_levels: 0,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::DaubechiesFourTap,
                trajectory_wavelet_levels: 4,
                scattering_mode: AcousticScatteringMode::Off,
            },
            Self::ScatteringFirstOrder => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Off,
                frame_wavelet_levels: 0,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Off,
                trajectory_wavelet_levels: 0,
                scattering_mode: AcousticScatteringMode::FirstOrder,
            },
            Self::ScatteringSecondOrder => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Off,
                frame_wavelet_levels: 0,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Off,
                trajectory_wavelet_levels: 0,
                scattering_mode: AcousticScatteringMode::SecondOrder,
            },
            Self::ScatteringFirstAndSecondOrder => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Off,
                frame_wavelet_levels: 0,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Off,
                trajectory_wavelet_levels: 0,
                scattering_mode: AcousticScatteringMode::FirstAndSecondOrder,
            },
            Self::AllHaarL4 => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::HaarAndModulation,
                frame_wavelet_levels: 4,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Haar,
                trajectory_wavelet_levels: 4,
                scattering_mode: AcousticScatteringMode::FirstAndSecondOrder,
            },
            Self::AllD4L4 => AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::DaubechiesFourTapAndModulation,
                frame_wavelet_levels: 4,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::DaubechiesFourTap,
                trajectory_wavelet_levels: 4,
                scattering_mode: AcousticScatteringMode::FirstAndSecondOrder,
            },
        }
    }
}

/// Whether a lane is the unfused control or an opt-in boundary-fusion pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicCorpusSidecarFusionScope {
    BaselineUnfused,
    BoundaryFusionV1,
}

/// Promotion authority attached to one lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicCorpusSidecarDisposition {
    Baseline,
    Rejected,
    AdvanceToCertification,
    Adopted,
}

/// Stable, machine-readable reasons that a promotion gate failed closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicCorpusSidecarGateFailure {
    FusionNotExecuted,
    MissingAccuracy,
    InsufficientDerImprovement,
    MacroJerRegression,
    BoundaryF1Regression,
    InsufficientComparableCoverage,
    MissingConditionalPairs,
    PairDiscrimination,
    PairBrier,
    PairCalibration,
    ChannelConfound,
    PairedDerUncertainty,
}

/// Frozen aggregate-only scoring and promotion policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarGatePolicy {
    pub minimum_relative_development_micro_der_improvement: f64,
    pub maximum_macro_jer_regression: f64,
    pub minimum_comparable_frame_coverage: f64,
    pub minimum_pair_roc_auc: f64,
    pub maximum_pair_brier_score: f64,
    pub maximum_pair_expected_calibration_error: f64,
    pub maximum_channel_confound_rate: f64,
    pub require_boundary_f1_non_regression: bool,
    pub require_held_out_micro_der_non_regression: bool,
    pub require_nonpositive_paired_der_upper_bound: bool,
}

/// Frozen protocol attached to every sidecar-study artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarStudyProtocol {
    pub oracle_vad: bool,
    pub oracle_speaker_count: bool,
    pub maximum_recording_duration_ms: Option<u64>,
    pub prefix_selection: String,
    pub rss_observation: String,
    pub diarization_request: DiarizationRequest,
    pub diarization_request_sha256: String,
    pub feature_ablation: AcousticFeatureAblation,
    pub detector_mode: AcousticChangeDetectorMode,
    pub clustering_mode: AcousticClusteringMode,
    pub sidecar_schema_id: String,
    pub fusion_id: String,
    pub calibration_fit_id: String,
    pub pair_scorer_id: String,
    pub uncertainty_id: String,
    pub selection_policy_id: String,
    pub selection_policy_sha256: String,
    pub boundary_collar_ms: u64,
    pub reliability_bins: usize,
    pub pair_lags_frames: [usize; 4],
    pub maximum_pairs_per_recording: usize,
    pub paired_bootstrap_replicates: usize,
    pub lane_order: Vec<PublicCorpusSidecarLane>,
    pub gate_policy: PublicCorpusSidecarGatePolicy,
}

/// Development-fitted calibration that contains no raw frame observations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarCalibration {
    pub fit_id: String,
    pub logit_intercept: f64,
    pub contrast_weight: f64,
    pub minimum_comparable_components: usize,
    pub fit_observation_count: u64,
    pub fit_positive_count: u64,
    pub fit_brier_score: Option<f64>,
    pub calibration_sha256: String,
}

/// One fixed probability-reliability bin. Its count is aggregate-only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarReliabilityBin {
    pub index: usize,
    pub lower_probability: f64,
    pub upper_probability: f64,
    pub observation_count: u64,
    pub positive_count: u64,
    pub mean_probability: Option<f64>,
    pub empirical_frequency: Option<f64>,
}

/// Aggregate boundary metrics for one lane and split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarBoundaryMetrics {
    pub reference_count: u64,
    pub hypothesis_count: u64,
    pub matched_count: u64,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub mean_absolute_error_sec: Option<f64>,
    pub probability_observation_count: u64,
    pub probability_positive_count: u64,
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub reliability: Vec<PublicCorpusSidecarReliabilityBin>,
}

/// Aggregate conditional same/different-speaker discrimination metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarPairMetrics {
    pub comparison_count: u64,
    pub same_speaker_count: u64,
    pub different_speaker_count: u64,
    pub mean_same_speaker_probability: Option<f64>,
    pub mean_different_speaker_probability: Option<f64>,
    pub roc_auc: Option<f64>,
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub reliability: Vec<PublicCorpusSidecarReliabilityBin>,
}

/// Aggregate availability and channel-confound accounting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarCoverage {
    pub submitted_frame_count: u64,
    pub comparable_frame_count: u64,
    pub comparable_frame_coverage: Option<f64>,
    pub component_comparison_count: u64,
    /// Canonical Voice, Channel, MixedAuxiliary order.
    pub owner_available_frame_counts: [u64; 3],
    pub channel_confound_opportunity_count: u64,
    pub channel_confound_count: u64,
    pub channel_confound_rate: Option<f64>,
    pub maximum_retained_signal_count: u64,
}

/// Exact operation counts and bounded memory payloads reported by the kernels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarOperations {
    pub frame_wavelet_filter_tap_terms: u64,
    pub trajectory_wavelet_filter_tap_terms: u64,
    pub trajectory_validity_sample_visits: u64,
    pub scattering_filter_sample_terms: u64,
    pub scattering_validity_sample_visits: u64,
    pub modulation_projection_sample_frequency_visits: u64,
    pub peak_scratch_buffer_payload_bytes: u64,
    pub peak_retained_state_bytes_on_target: u64,
    pub cached_twiddle_payload_bytes: u64,
}

/// Runtime observations retained even when candidate accuracy is withheld.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarPerformance {
    pub audio_duration_sec: f64,
    pub wall_time_sec: f64,
    pub real_time_factor: Option<f64>,
    pub sampled_peak_rss_bytes: u64,
}

/// Paired per-recording uncertainty versus the unfused baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarPairedUncertainty {
    pub paired_recording_count: u64,
    pub bootstrap_replicates: usize,
    pub bootstrap_seed_sha256: String,
    pub mean_der_delta: Option<f64>,
    pub der_delta_ci95_lower: Option<f64>,
    pub der_delta_ci95_upper: Option<f64>,
    pub mean_jer_delta: Option<f64>,
    pub jer_delta_ci95_lower: Option<f64>,
    pub jer_delta_ci95_upper: Option<f64>,
}

/// Aggregate evidence for one lane and one selected split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarStudySplit {
    pub split: EvaluationSplit,
    /// Baseline metrics are always present. Candidate metrics are present only
    /// when a calibrated sidecar probability reached the boundary selector.
    pub pipeline: Option<PublicCorpusAblationSplit>,
    pub fusion_executed: bool,
    pub boundary: PublicCorpusSidecarBoundaryMetrics,
    pub conditional_pairs: PublicCorpusSidecarPairMetrics,
    pub coverage: PublicCorpusSidecarCoverage,
    pub operations: PublicCorpusSidecarOperations,
    pub performance: PublicCorpusSidecarPerformance,
    pub paired_uncertainty: Option<PublicCorpusSidecarPairedUncertainty>,
}

/// Recomputed fail-closed decision for one candidate versus the baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarPromotionGate {
    pub split: EvaluationSplit,
    pub candidate: PublicCorpusSidecarLane,
    pub baseline: PublicCorpusSidecarLane,
    pub relative_micro_der_improvement: Option<f64>,
    pub macro_jer_delta: Option<f64>,
    pub boundary_f1_delta: Option<f64>,
    pub comparable_frame_coverage: Option<f64>,
    pub pair_roc_auc: Option<f64>,
    pub pair_brier_score: Option<f64>,
    pub pair_expected_calibration_error: Option<f64>,
    pub channel_confound_rate: Option<f64>,
    pub paired_der_ci95_upper: Option<f64>,
    pub failures: Vec<PublicCorpusSidecarGateFailure>,
    pub passed: bool,
}

/// Aggregate-only retained evidence for one frozen lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarStudyVariant {
    pub lane: PublicCorpusSidecarLane,
    pub fusion_scope: PublicCorpusSidecarFusionScope,
    pub study_configuration_sha256: String,
    pub fusion_configuration_sha256: Option<String>,
    pub lane_configuration_sha256: String,
    pub calibration: Option<PublicCorpusSidecarCalibration>,
    pub disposition: PublicCorpusSidecarDisposition,
    pub splits: Vec<PublicCorpusSidecarStudySplit>,
    pub gate: Option<PublicCorpusSidecarPromotionGate>,
}

/// Path-free, transcript-free public acoustic-sidecar evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarStudyEvidence {
    pub schema_version: String,
    pub runner_version: String,
    pub scorer_version: String,
    pub corpus_key: String,
    pub source_version: String,
    pub bundle_sha256: String,
    pub descriptor_sha256: String,
    pub scorer_config: DiarizationScorerConfig,
    pub scorer_config_sha256: String,
    pub evaluation_stage: PublicCorpusEvaluationStage,
    pub locked_development_result_sha256: Option<String>,
    pub locked_development_accuracy_sha256: Option<String>,
    pub protocol: PublicCorpusSidecarStudyProtocol,
    pub protocol_sha256: String,
    pub selected_candidate_lane: Option<PublicCorpusSidecarLane>,
    pub adopted_candidate_lane: Option<PublicCorpusSidecarLane>,
    pub variants: Vec<PublicCorpusSidecarStudyVariant>,
    /// Hash with wall-time, RSS, and RTF observations normalized away.
    pub deterministic_accuracy_sha256: String,
    /// Hash of this evidence with this field temporarily empty.
    pub result_sha256: String,
}

/// External-only inputs and outputs for one frozen sidecar study.
///
/// This request deliberately has no `Debug` or serialization implementation:
/// local paths cannot enter diagnostics or retained evidence.
pub struct PublicCorpusSidecarStudyRequest<'a> {
    pub project_root: &'a Path,
    pub input_root: &'a Path,
    pub descriptor_path: &'a Path,
    pub bundle_output_path: &'a Path,
    pub evidence_output_path: &'a Path,
    pub license_acknowledgement_id: &'a str,
    pub maximum_recording_duration_ms: Option<u64>,
    pub evaluation_stage: PublicCorpusEvaluationStage,
    pub locked_development_evidence_path: Option<&'a Path>,
}

#[derive(Serialize)]
struct PublicCorpusSidecarCalibrationFingerprint<'a> {
    fit_id: &'a str,
    logit_intercept: f64,
    contrast_weight: f64,
    minimum_comparable_components: usize,
    fit_observation_count: u64,
    fit_positive_count: u64,
    fit_brier_score: Option<f64>,
}

#[derive(Serialize)]
struct PublicCorpusSidecarLaneFingerprint<'a> {
    runner_version: &'a str,
    lane: PublicCorpusSidecarLane,
    fusion_scope: PublicCorpusSidecarFusionScope,
    study_configuration_sha256: &'a str,
    fusion_configuration_sha256: Option<&'a str>,
    protocol_sha256: &'a str,
}

#[derive(Serialize)]
struct PublicCorpusSidecarSelectionFingerprint<'a> {
    policy_id: &'a str,
    lane_order: &'a [PublicCorpusSidecarLane],
    gate_policy: &'a PublicCorpusSidecarGatePolicy,
}

#[derive(Serialize)]
struct PublicFeatureConfigurationFingerprint<'a> {
    runner_version: &'a str,
    ablation: AcousticFeatureAblation,
    feature_schema_sha256: &'a str,
    diarization_request_sha256: &'a str,
    change_calibration_sha256: &'a str,
}

#[derive(Serialize)]
struct PublicChangeConfigurationFingerprint<'a> {
    runner_version: &'a str,
    detector_mode: AcousticChangeDetectorMode,
    feature_ablation: AcousticFeatureAblation,
    feature_schema_sha256: &'a str,
    diarization_request_sha256: &'a str,
    change_calibration_sha256: &'a str,
}

#[derive(Serialize)]
struct PublicClusteringConfigurationFingerprint<'a> {
    runner_version: &'a str,
    clustering_mode: AcousticClusteringMode,
    detector_mode: AcousticChangeDetectorMode,
    feature_ablation: AcousticFeatureAblation,
    feature_schema_sha256: &'a str,
    diarization_request_sha256: &'a str,
    speaker_pair_calibration_sha256: &'a str,
}

#[derive(Serialize)]
struct PublicChangeSelectionPolicyFingerprint {
    policy_id: &'static str,
    candidate_order: [AcousticChangeDetectorMode; 3],
    baseline: AcousticChangeDetectorMode,
    minimum_relative_change_f1_improvement: f64,
    maximum_der_jer_regression: f64,
    maximum_brier_score: f64,
    maximum_expected_calibration_error: f64,
    require_timing_non_regression: bool,
    fail_closed_default: AcousticChangeDetectorMode,
}

/// External-only local descriptor. It intentionally has neither `Debug` nor
/// `Serialize`, preventing accidental logging or retention of source paths.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicCorpusInput {
    schema_version: String,
    corpus_key: String,
    source_version: String,
    recordings: Vec<PublicCorpusInputRecording>,
}

/// One path-bearing external source row.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicCorpusInputRecording {
    recording_id: String,
    split: EvaluationSplit,
    origin_recording_id: String,
    audio_path: PathBuf,
    audio_sha256: String,
    expected_sample_rate_hz: u32,
    expected_channel_count: u16,
    selected_channel: u16,
    annotation_path: PathBuf,
    annotation_sha256: String,
    annotation_recording_id: String,
    annotation_channel: String,
    speaker_map: BTreeMap<String, String>,
    #[serde(default)]
    word_annotation_path: Option<PathBuf>,
    #[serde(default)]
    word_annotation_sha256: Option<String>,
    #[serde(default)]
    ignored_regions: Vec<EvaluationRegion>,
    #[serde(default)]
    derived_from_recording_ids: Vec<String>,
    #[serde(default)]
    augmentation_group_id: Option<String>,
    #[serde(default)]
    enrollment_recording_ids: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PublicCorpusWordAnnotation {
    schema_version: String,
    recording_id: String,
    words: Vec<EvaluationWord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WaveMetadata {
    sample_rate_hz: u32,
    channel_count: u16,
    duration_ms: u64,
}

/// Return the frozen registry of supported corpus sources.
#[must_use]
pub fn public_corpus_registry() -> PublicCorpusRegistry {
    let mut entries = vec![
        PublicCorpusRegistryEntry {
            corpus_key: "aishell-4-openslr111-v1".to_owned(),
            description:
                "Mandarin multi-channel meetings with short turns, noise, and overlap".to_owned(),
            authoritative_url: "https://www.openslr.org/111/".to_owned(),
            license_id: "CC-BY-SA-4.0".to_owned(),
            license_url:
                "https://creativecommons.org/licenses/by-sa/4.0/legalcode".to_owned(),
            license_acknowledgement_id: "accept-aishell-4-cc-by-sa-4.0".to_owned(),
            split_policy: PublicCorpusSplitPolicy::ExternalDescriptorV1,
            expected_local_layout:
                "<external-root>/audio/**/*.wav and <external-root>/annotations/**/*.rttm"
                    .to_owned(),
            conversion_contract:
                "Select one immutable WAV channel per recording and convert official speaker activity to ten-field RTTM without changing time geometry"
                    .to_owned(),
            upstream_integrity_note:
                "OpenSLR publishes archive names and sizes; this adapter requires SHA-256 for every selected WAV and RTTM after extraction"
                    .to_owned(),
            condition_tags: vec![
                "far-field".to_owned(),
                "meeting".to_owned(),
                "multichannel".to_owned(),
                "overlap".to_owned(),
                "short-turn".to_owned(),
            ],
        },
        PublicCorpusRegistryEntry {
            corpus_key: "ami-scenario-v1".to_owned(),
            description:
                "English meetings with synchronized close-talk and far-field microphones"
                    .to_owned(),
            authoritative_url: "https://groups.inf.ed.ac.uk/ami/corpus/".to_owned(),
            license_id: "CC-BY-4.0".to_owned(),
            license_url: "https://creativecommons.org/licenses/by/4.0/legalcode".to_owned(),
            license_acknowledgement_id: "accept-ami-cc-by-4.0".to_owned(),
            split_policy: PublicCorpusSplitPolicy::AmiScenarioOfficialV1,
            expected_local_layout:
                "<external-root>/audio/<meeting>.wav and <external-root>/annotations/<meeting>.rttm"
                    .to_owned(),
            conversion_contract:
                "Use one official scenario meeting and one named microphone view per recording; convert NXT speaker segments to ten-field RTTM and preserve the official scenario-only family split"
                    .to_owned(),
            upstream_integrity_note:
                "The adapter requires SHA-256 for every selected WAV and converted RTTM because the corpus site does not publish a complete SHA-256 manifest"
                    .to_owned(),
            condition_tags: vec![
                "close-talk".to_owned(),
                "far-field".to_owned(),
                "meeting".to_owned(),
                "overlap".to_owned(),
                "same-speaker-multi-device".to_owned(),
            ],
        },
        PublicCorpusRegistryEntry {
            corpus_key: "callhome-american-english-2e-v1".to_owned(),
            description:
                "Licensed English two-channel 8 kHz conversational telephone speech".to_owned(),
            authoritative_url: "https://catalog.ldc.upenn.edu/LDC2026S08".to_owned(),
            license_id: "LDC-USER-AGREEMENT".to_owned(),
            license_url: "https://catalog.ldc.upenn.edu/LDC2026S08".to_owned(),
            license_acknowledgement_id: "accept-ldc2026s08-user-agreement".to_owned(),
            split_policy: PublicCorpusSplitPolicy::ExternalDescriptorV1,
            expected_local_layout:
                "<external-root>/audio/**/*.wav and <external-root>/annotations/**/*.rttm"
                    .to_owned(),
            conversion_contract:
                "Under an operator-held LDC license, decode selected 8 kHz FLAC channels to immutable PCM WAV and convert licensed speaker turns to ten-field RTTM; no LDC material may enter Git"
                    .to_owned(),
            upstream_integrity_note:
                "LDC access is user-licensed; record SHA-256 for every locally decoded WAV and RTTM and retain the descriptor outside the checkout"
                    .to_owned(),
            condition_tags: vec![
                "channel-mismatch".to_owned(),
                "dyadic".to_owned(),
                "telephone".to_owned(),
                "two-channel".to_owned(),
            ],
        },
        PublicCorpusRegistryEntry {
            corpus_key: "voxconverse-v1".to_owned(),
            description:
                "In-the-wild multi-speaker clips with overlap and challenging backgrounds"
                    .to_owned(),
            authoritative_url: "https://mm.kaist.ac.kr/datasets/voxconverse/".to_owned(),
            license_id: "CC-BY-4.0-ORIGINAL-COPYRIGHT".to_owned(),
            license_url: "https://mm.kaist.ac.kr/datasets/voxconverse/".to_owned(),
            license_acknowledgement_id:
                "accept-voxconverse-cc-by-4.0-and-original-copyright".to_owned(),
            split_policy: PublicCorpusSplitPolicy::ExternalDescriptorV1,
            expected_local_layout:
                "<external-root>/audio/{dev,test}/*.wav and <external-root>/annotations/{dev,test}/*.rttm"
                    .to_owned(),
            conversion_contract:
                "Use the upstream WAV and RTTM pairing without transcript material; keep development and test identities disjoint and freeze every selected file by SHA-256"
                    .to_owned(),
            upstream_integrity_note:
                "The corpus page publishes archive MD5 values; this adapter additionally requires SHA-256 for every selected extracted WAV and RTTM"
                    .to_owned(),
            condition_tags: vec![
                "background-noise".to_owned(),
                "in-the-wild".to_owned(),
                "overlap".to_owned(),
                "same-gender".to_owned(),
                "short-turn".to_owned(),
            ],
        },
    ];
    entries.sort_by(|left, right| left.corpus_key.cmp(&right.corpus_key));
    for entry in &mut entries {
        entry.condition_tags.sort();
    }
    PublicCorpusRegistry {
        schema_version: PUBLIC_CORPUS_REGISTRY_SCHEMA_VERSION.to_owned(),
        adapter_version: PUBLIC_CORPUS_ADAPTER_VERSION.to_owned(),
        entries,
    }
}

/// Parse and fully validate one generated bundle.
pub fn parse_public_corpus_bundle(bytes: &[u8]) -> FwResult<PublicCorpusBundle> {
    let bundle = serde_json::from_slice(bytes).map_err(|_| {
        public_corpus_error(
            "bundle_json",
            "bundle must be valid public-corpus JSON without trailing data",
        )
    })?;
    verify_public_corpus_bundle(&bundle)?;
    Ok(bundle)
}

/// Parse and fully validate aggregate public ablation evidence.
pub fn parse_public_corpus_ablation_evidence(
    bytes: &[u8],
) -> FwResult<PublicCorpusAblationEvidence> {
    let evidence = serde_json::from_slice(bytes).map_err(|_| {
        public_corpus_error(
            "ablation_json",
            "ablation evidence must be valid JSON without trailing data",
        )
    })?;
    verify_public_corpus_ablation_evidence(&evidence)?;
    Ok(evidence)
}

/// Verify schemas, hashes, scorer documents, ordering, and leakage evidence.
pub fn verify_public_corpus_bundle(bundle: &PublicCorpusBundle) -> FwResult<()> {
    if bundle.schema_version != PUBLIC_CORPUS_BUNDLE_SCHEMA_VERSION {
        return Err(public_corpus_error(
            "bundle_schema_version",
            "unsupported public-corpus bundle schema version",
        ));
    }
    if bundle.adapter_version != PUBLIC_CORPUS_ADAPTER_VERSION {
        return Err(public_corpus_error(
            "adapter_version",
            "unsupported public-corpus adapter version",
        ));
    }
    let registry = public_corpus_registry();
    let entry = registry
        .entries
        .iter()
        .find(|candidate| candidate.corpus_key == bundle.corpus_key)
        .ok_or_else(|| {
            public_corpus_error(
                "corpus_key",
                "bundle corpus key is not in the frozen registry",
            )
        })?;
    if bundle.license_id != entry.license_id
        || bundle.license_acknowledgement_id != entry.license_acknowledgement_id
    {
        return Err(public_corpus_error(
            "license_contract",
            "bundle license identity does not match the frozen registry",
        ));
    }
    for (field, value) in [
        ("descriptor_sha256", &bundle.descriptor_sha256),
        ("bundle_sha256", &bundle.bundle_sha256),
    ] {
        if !is_sha256_hex(value) {
            return Err(public_corpus_error(
                "hash_format",
                &format!("{field} must be 64 lowercase hexadecimal characters"),
            ));
        }
    }
    validate_public_id(&bundle.source_version, "source_version")?;

    let manifest_bytes = serde_json::to_vec(&bundle.manifest)?;
    parse_diarization_corpus_manifest(&manifest_bytes)?;
    if bundle.manifest.corpus_id != bundle.corpus_key
        || bundle.manifest.license_id != bundle.license_id
    {
        return Err(public_corpus_error(
            "manifest_identity",
            "embedded manifest identity differs from the bundle",
        ));
    }
    verify_leakage_audit_hash(&bundle.leakage_audit)?;
    if !bundle.leakage_audit.passed {
        return Err(public_corpus_error(
            "leakage_audit",
            "generated public-corpus bundle must have a passing leakage audit",
        ));
    }
    let regenerated_audit = audit_diarization_manifest(&bundle.manifest)?;
    if regenerated_audit != bundle.leakage_audit {
        return Err(public_corpus_error(
            "leakage_audit_mismatch",
            "embedded leakage audit does not match the embedded manifest",
        ));
    }
    if bundle.references.len() != bundle.recordings.len()
        || bundle.references.len() != bundle.manifest.recordings.len()
    {
        return Err(public_corpus_error(
            "recording_cardinality",
            "reference, evidence, and manifest recording counts differ",
        ));
    }
    if !bundle
        .references
        .windows(2)
        .all(|window| window[0].recording_id < window[1].recording_id)
        || !bundle
            .recordings
            .windows(2)
            .all(|window| window[0].recording_id < window[1].recording_id)
    {
        return Err(public_corpus_error(
            "recording_order",
            "bundle recordings must be strictly ordered by recording_id",
        ));
    }
    for ((reference, evidence), manifest_recording) in bundle
        .references
        .iter()
        .zip(&bundle.recordings)
        .zip(&bundle.manifest.recordings)
    {
        parse_diarization_reference(&serde_json::to_vec(reference)?)?;
        if reference.recording_id != evidence.recording_id
            || reference.recording_id != manifest_recording.recording_id
            || evidence.split != manifest_recording.split
        {
            return Err(public_corpus_error(
                "recording_alignment",
                "reference, evidence, and manifest recording identities differ",
            ));
        }
        if evidence.reference_sha256 != canonical_sha256(reference)?
            || evidence.turn_count != reference.turns.len()
            || evidence.word_count != reference.words.len()
            || (!reference.words.is_empty() && evidence.word_annotation_sha256.is_none())
            || evidence.overlap_turn_count
                != reference
                    .turns
                    .iter()
                    .filter(|turn| turn.overlap_suspected)
                    .count()
            || evidence.ignored_region_count != reference.ignored_regions.len()
        {
            return Err(public_corpus_error(
                "recording_evidence",
                "recording evidence does not match the embedded reference",
            ));
        }
        if !is_sha256_hex(&evidence.audio_sha256)
            || !is_sha256_hex(&evidence.annotation_sha256)
            || evidence
                .word_annotation_sha256
                .as_ref()
                .is_some_and(|hash| !is_sha256_hex(hash))
            || !is_sha256_hex(&evidence.reference_sha256)
        {
            return Err(public_corpus_error(
                "recording_hash_format",
                "recording evidence hashes must be lowercase SHA-256",
            ));
        }
        if evidence.sample_rate_hz == 0
            || evidence.channel_count == 0
            || evidence.selected_channel == 0
            || evidence.selected_channel > evidence.channel_count
        {
            return Err(public_corpus_error(
                "recording_audio_contract",
                "recording evidence has invalid sample-rate or channel geometry",
            ));
        }
    }
    let mut unhashed = bundle.clone();
    let expected = unhashed.bundle_sha256.clone();
    unhashed.bundle_sha256.clear();
    if canonical_sha256(&unhashed)? != expected {
        return Err(public_corpus_error(
            "bundle_hash_mismatch",
            "bundle_sha256 does not match canonical bundle content",
        ));
    }
    Ok(())
}

/// Build one path-free bundle from external WAV and RTTM inputs.
///
/// The output is opened with `create_new`, and all source/output roots must be
/// absolute, canonical, and disjoint from the project checkout.
pub fn build_public_corpus_bundle(
    project_root: &Path,
    input_root: &Path,
    descriptor_path: &Path,
    output_path: &Path,
    license_acknowledgement_id: &str,
) -> FwResult<PublicCorpusBundle> {
    build_public_corpus_bundle_with_cancel(
        project_root,
        input_root,
        descriptor_path,
        output_path,
        license_acknowledgement_id,
        || false,
    )
}

/// Cancellation-aware form of [`build_public_corpus_bundle`].
pub fn build_public_corpus_bundle_with_cancel(
    project_root: &Path,
    input_root: &Path,
    descriptor_path: &Path,
    output_path: &Path,
    license_acknowledgement_id: &str,
    mut is_cancelled: impl FnMut() -> bool,
) -> FwResult<PublicCorpusBundle> {
    checkpoint_cancelled(&mut is_cancelled)?;
    let canonical_project = canonical_directory(project_root, "project_root")?;
    let canonical_input = canonical_directory(input_root, "input_root")?;
    if paths_overlap(&canonical_project, &canonical_input) {
        return Err(public_corpus_error(
            "input_root_overlap",
            "input root must be disjoint from the project checkout",
        ));
    }
    let canonical_descriptor =
        canonical_input_file(&canonical_input, descriptor_path, "descriptor")?;
    let output_parent = validate_new_output(&canonical_project, &canonical_input, output_path)?;
    checkpoint_cancelled(&mut is_cancelled)?;

    let descriptor_bytes = read_bounded(&canonical_descriptor, MAX_DESCRIPTOR_BYTES, "descriptor")?;
    let descriptor_sha256 = format!("{:x}", Sha256::digest(&descriptor_bytes));
    let mut descriptor: PublicCorpusInput =
        serde_json::from_slice(&descriptor_bytes).map_err(|_| {
            public_corpus_error(
                "descriptor_json",
                "descriptor must be valid public-corpus input JSON without trailing data",
            )
        })?;
    if descriptor.schema_version != PUBLIC_CORPUS_INPUT_SCHEMA_VERSION {
        return Err(public_corpus_error(
            "descriptor_schema_version",
            "unsupported public-corpus input schema version",
        ));
    }
    validate_public_id(&descriptor.corpus_key, "corpus_key")?;
    validate_public_id(&descriptor.source_version, "source_version")?;
    if descriptor.recordings.is_empty() || descriptor.recordings.len() > MAX_RECORDINGS {
        return Err(public_corpus_error(
            "recording_count",
            "descriptor recording count is outside the supported range",
        ));
    }
    let registry = public_corpus_registry();
    let registry_entry = registry
        .entries
        .iter()
        .find(|entry| entry.corpus_key == descriptor.corpus_key)
        .ok_or_else(|| {
            public_corpus_error(
                "corpus_key",
                "descriptor corpus key is not in the frozen registry",
            )
        })?;
    if license_acknowledgement_id != registry_entry.license_acknowledgement_id {
        return Err(public_corpus_error(
            "license_acknowledgement",
            "the exact registry license acknowledgement is required",
        ));
    }
    descriptor
        .recordings
        .sort_by(|left, right| left.recording_id.cmp(&right.recording_id));

    let mut recording_ids = BTreeSet::new();
    let mut references = Vec::with_capacity(descriptor.recordings.len());
    let mut manifest_recordings = Vec::with_capacity(descriptor.recordings.len());
    let mut evidence = Vec::with_capacity(descriptor.recordings.len());
    let mut total_turn_count = 0_usize;
    let mut total_word_count = 0_usize;
    for recording in descriptor.recordings {
        checkpoint_cancelled(&mut is_cancelled)?;
        validate_public_id(&recording.recording_id, "recording_id")?;
        if !recording_ids.insert(recording.recording_id.clone()) {
            return Err(public_corpus_error(
                "duplicate_recording",
                "descriptor recording IDs must be unique",
            ));
        }
        validate_public_id(&recording.origin_recording_id, "origin_recording_id")?;
        validate_split(
            registry_entry.split_policy,
            &recording.recording_id,
            recording.split,
        )?;
        validate_sha256(&recording.audio_sha256, "audio_sha256")?;
        validate_sha256(&recording.annotation_sha256, "annotation_sha256")?;
        match (
            recording.word_annotation_path.as_ref(),
            recording.word_annotation_sha256.as_ref(),
        ) {
            (Some(_), Some(hash)) => validate_sha256(hash, "word_annotation_sha256")?,
            (None, None) => {}
            _ => {
                return Err(public_corpus_error(
                    "word_annotation_pair",
                    "word_annotation_path and word_annotation_sha256 must be supplied together",
                ));
            }
        }
        if recording.expected_sample_rate_hz == 0
            || recording.expected_channel_count == 0
            || recording.selected_channel == 0
            || recording.selected_channel > recording.expected_channel_count
        {
            return Err(public_corpus_error(
                "audio_contract",
                "expected sample-rate and channel geometry is invalid",
            ));
        }
        validate_public_id(
            &recording.annotation_recording_id,
            "annotation_recording_id",
        )?;
        validate_rttm_channel(&recording.annotation_channel)?;
        validate_speaker_map(&recording.speaker_map)?;
        let audio_path = canonical_relative_file(&canonical_input, &recording.audio_path, "audio")?;
        let annotation_path =
            canonical_relative_file(&canonical_input, &recording.annotation_path, "annotation")?;
        let (actual_audio_sha256, wave) = hash_and_inspect_wave(&audio_path, &mut is_cancelled)?;
        if actual_audio_sha256 != recording.audio_sha256 {
            return Err(public_corpus_error(
                "audio_checksum_mismatch",
                "audio SHA-256 does not match the descriptor",
            ));
        }
        if wave.sample_rate_hz != recording.expected_sample_rate_hz
            || wave.channel_count != recording.expected_channel_count
        {
            return Err(public_corpus_error(
                "audio_metadata_mismatch",
                "WAV sample rate or channel count does not match the descriptor",
            ));
        }
        let annotation_bytes = read_bounded(&annotation_path, MAX_ANNOTATION_BYTES, "annotation")?;
        let actual_annotation_sha256 = format!("{:x}", Sha256::digest(&annotation_bytes));
        if actual_annotation_sha256 != recording.annotation_sha256 {
            return Err(public_corpus_error(
                "annotation_checksum_mismatch",
                "annotation SHA-256 does not match the descriptor",
            ));
        }
        let turns = parse_rttm(
            &annotation_bytes,
            &recording.annotation_recording_id,
            &recording.annotation_channel,
            &recording.speaker_map,
            wave.duration_ms,
        )?;
        let (mut words, actual_word_annotation_sha256) =
            if let (Some(relative_path), Some(expected_sha256)) = (
                recording.word_annotation_path.as_ref(),
                recording.word_annotation_sha256.as_ref(),
            ) {
                let word_path =
                    canonical_relative_file(&canonical_input, relative_path, "word_annotation")?;
                let word_bytes = read_bounded(&word_path, MAX_ANNOTATION_BYTES, "word_annotation")?;
                let actual_sha256 = format!("{:x}", Sha256::digest(&word_bytes));
                if &actual_sha256 != expected_sha256 {
                    return Err(public_corpus_error(
                        "word_annotation_checksum_mismatch",
                        "word-annotation SHA-256 does not match the descriptor",
                    ));
                }
                let annotation: PublicCorpusWordAnnotation = serde_json::from_slice(&word_bytes)
                    .map_err(|_| {
                        public_corpus_error(
                            "word_annotation_json",
                            "word annotation must be valid schema-bound JSON",
                        )
                    })?;
                if annotation.schema_version != PUBLIC_CORPUS_WORD_ANNOTATION_SCHEMA_VERSION
                    || annotation.recording_id != recording.recording_id
                {
                    return Err(public_corpus_error(
                        "word_annotation_identity",
                        "word annotation schema or recording identity does not match",
                    ));
                }
                if annotation.words.len() > MAX_WORDS_PER_RECORDING {
                    return Err(public_corpus_error(
                        "word_annotation_count",
                        "word annotation exceeds the per-recording memory-safety limit",
                    ));
                }
                (annotation.words, Some(actual_sha256))
            } else {
                (Vec::new(), None)
            };
        words.sort_by(|left, right| {
            (
                left.start_ms,
                left.end_ms,
                left.word_id.as_str(),
                left.speaker_ref.as_str(),
            )
                .cmp(&(
                    right.start_ms,
                    right.end_ms,
                    right.word_id.as_str(),
                    right.speaker_ref.as_str(),
                ))
        });
        total_turn_count = total_turn_count
            .checked_add(turns.len())
            .filter(|count| *count <= MAX_TOTAL_TURNS)
            .ok_or_else(|| {
                public_corpus_error(
                    "total_turn_count",
                    "corpus turn count exceeds the supported memory-safety limit",
                )
            })?;
        total_word_count = total_word_count
            .checked_add(words.len())
            .filter(|count| *count <= MAX_TOTAL_WORDS)
            .ok_or_else(|| {
                public_corpus_error(
                    "total_word_count",
                    "corpus word count exceeds the supported memory-safety limit",
                )
            })?;
        let mut ignored_regions = recording.ignored_regions;
        ignored_regions.sort_by(|left, right| {
            (left.start_ms, left.end_ms, left.reason_code.as_str()).cmp(&(
                right.start_ms,
                right.end_ms,
                right.reason_code.as_str(),
            ))
        });
        let reference = DiarizationReferenceDocument {
            schema_version: DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
            recording_id: recording.recording_id.clone(),
            duration_ms: wave.duration_ms,
            turns,
            ignored_regions,
            speaker_hints: Vec::new(),
            words,
        };
        parse_diarization_reference(&serde_json::to_vec(&reference)?)?;
        let reference_sha256 = canonical_sha256(&reference)?;

        let speaker_refs = reference
            .turns
            .iter()
            .filter_map(|turn| turn.speaker.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut derived_from_recording_ids = recording.derived_from_recording_ids;
        let mut enrollment_recording_ids = recording.enrollment_recording_ids;
        derived_from_recording_ids.sort();
        enrollment_recording_ids.sort();
        manifest_recordings.push(crate::diarization::CorpusRecordingManifest {
            recording_id: recording.recording_id.clone(),
            split: recording.split,
            origin_recording_id: recording.origin_recording_id,
            speaker_refs,
            derived_from_recording_ids,
            augmentation_group_id: recording.augmentation_group_id,
            enrollment_recording_ids,
        });
        evidence.push(PublicCorpusRecordingEvidence {
            recording_id: recording.recording_id,
            split: recording.split,
            audio_sha256: actual_audio_sha256,
            annotation_sha256: actual_annotation_sha256,
            word_annotation_sha256: actual_word_annotation_sha256,
            reference_sha256,
            sample_rate_hz: wave.sample_rate_hz,
            channel_count: wave.channel_count,
            selected_channel: recording.selected_channel,
            turn_count: reference.turns.len(),
            word_count: reference.words.len(),
            overlap_turn_count: reference
                .turns
                .iter()
                .filter(|turn| turn.overlap_suspected)
                .count(),
            ignored_region_count: reference.ignored_regions.len(),
        });
        references.push(reference);
    }

    manifest_recordings.sort_by(|left, right| {
        (left.recording_id.as_str(), left.split).cmp(&(right.recording_id.as_str(), right.split))
    });
    references.sort_by(|left, right| left.recording_id.cmp(&right.recording_id));
    evidence.sort_by(|left, right| left.recording_id.cmp(&right.recording_id));
    let manifest = DiarizationCorpusManifest {
        schema_version: DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION.to_owned(),
        corpus_id: descriptor.corpus_key.clone(),
        license_id: registry_entry.license_id.clone(),
        recordings: manifest_recordings,
    };
    parse_diarization_corpus_manifest(&serde_json::to_vec(&manifest)?)?;
    let leakage_audit = audit_diarization_manifest(&manifest)?;
    if !leakage_audit.passed {
        return Err(public_corpus_error(
            "split_leakage",
            "descriptor violates the frozen cross-split leakage contract",
        ));
    }
    let mut bundle = PublicCorpusBundle {
        schema_version: PUBLIC_CORPUS_BUNDLE_SCHEMA_VERSION.to_owned(),
        adapter_version: PUBLIC_CORPUS_ADAPTER_VERSION.to_owned(),
        corpus_key: descriptor.corpus_key,
        source_version: descriptor.source_version,
        license_id: registry_entry.license_id.clone(),
        license_acknowledgement_id: registry_entry.license_acknowledgement_id.clone(),
        descriptor_sha256,
        manifest,
        leakage_audit,
        references,
        recordings: evidence,
        bundle_sha256: String::new(),
    };
    bundle.bundle_sha256 = canonical_sha256(&bundle)?;
    verify_public_corpus_bundle(&bundle)?;
    checkpoint_cancelled(&mut is_cancelled)?;
    write_new_json(output_path, &output_parent, &bundle, "public-corpus bundle")?;
    Ok(bundle)
}

/// Build, execute, score, and retain one aggregate-only public feature
/// ablation. Source media and per-recording hypotheses never leave
/// `input_root` or process memory.
pub fn run_public_corpus_ablation_with_cancel(
    request: PublicCorpusAblationRequest<'_>,
    mut is_cancelled: impl FnMut() -> bool,
) -> FwResult<PublicCorpusAblationEvidence> {
    let PublicCorpusAblationRequest {
        project_root,
        input_root,
        descriptor_path,
        bundle_output_path,
        evidence_output_path,
        license_acknowledgement_id,
        maximum_recording_duration_ms,
        evaluation_stage,
        locked_development_evidence_path,
    } = request;
    if maximum_recording_duration_ms == Some(0) {
        return Err(public_corpus_error(
            "ablation_duration",
            "maximum recording duration must be positive when supplied",
        ));
    }
    match (evaluation_stage, locked_development_evidence_path) {
        (PublicCorpusEvaluationStage::Development, None)
        | (PublicCorpusEvaluationStage::Certification, Some(_)) => {}
        (PublicCorpusEvaluationStage::Development, Some(_)) => {
            return Err(public_corpus_error(
                "ablation_stage_lock",
                "development evaluation must not receive held-out lock evidence",
            ));
        }
        (PublicCorpusEvaluationStage::Certification, None) => {
            return Err(public_corpus_error(
                "ablation_stage_lock",
                "certification requires locked development evidence",
            ));
        }
    }
    let canonical_project = canonical_directory(project_root, "project_root")?;
    let canonical_input = canonical_directory(input_root, "input_root")?;
    let bundle_output_parent =
        validate_new_output(&canonical_project, &canonical_input, bundle_output_path)?;
    let evidence_output_parent =
        validate_new_output(&canonical_project, &canonical_input, evidence_output_path)?;
    let normalized_output = |path: &Path, parent: &Path| {
        path.file_name()
            .map(|file_name| parent.join(file_name))
            .ok_or_else(|| {
                public_corpus_error("ablation_output", "output must have a terminal file name")
            })
    };
    if normalized_output(bundle_output_path, &bundle_output_parent)?
        == normalized_output(evidence_output_path, &evidence_output_parent)?
    {
        return Err(public_corpus_error(
            "ablation_output",
            "bundle and ablation evidence outputs must be distinct",
        ));
    }
    let locked_development = if let Some(path) = locked_development_evidence_path {
        let canonical_locked = canonical_external_file(
            &canonical_project,
            &canonical_input,
            path,
            "development_lock",
        )?;
        let bytes = read_bounded(&canonical_locked, MAX_DESCRIPTOR_BYTES, "development_lock")?;
        let evidence = parse_public_corpus_ablation_evidence(&bytes)?;
        let all_development_gates_passed = evidence
            .development_gate
            .as_ref()
            .is_some_and(|gate| gate.passed)
            && evidence
                .change_development_gate
                .as_ref()
                .is_some_and(|gate| gate.passed)
            && evidence
                .clustering_development_gate
                .as_ref()
                .is_some_and(|gate| gate.passed);
        if evidence.evaluation_stage != PublicCorpusEvaluationStage::Development
            || !all_development_gates_passed
        {
            return Err(public_corpus_error(
                "ablation_stage_lock",
                "held-out certification requires locked development evidence whose representation, change-detector, and clustering gates all passed",
            ));
        }
        Some(evidence)
    } else {
        None
    };
    let bundle = build_public_corpus_bundle_with_cancel(
        project_root,
        input_root,
        descriptor_path,
        bundle_output_path,
        license_acknowledgement_id,
        &mut is_cancelled,
    )?;
    checkpoint_cancelled(&mut is_cancelled)?;
    if let Some(evidence) = &locked_development
        && (evidence.evaluation_stage != PublicCorpusEvaluationStage::Development
            || evidence.bundle_sha256 != bundle.bundle_sha256
            || evidence.descriptor_sha256 != bundle.descriptor_sha256
            || evidence.protocol.maximum_recording_duration_ms != maximum_recording_duration_ms)
    {
        return Err(public_corpus_error(
            "ablation_stage_lock",
            "locked development evidence does not bind this exact corpus and duration protocol",
        ));
    }

    let canonical_descriptor =
        canonical_input_file(&canonical_input, descriptor_path, "descriptor")?;
    let descriptor_bytes = read_bounded(&canonical_descriptor, MAX_DESCRIPTOR_BYTES, "descriptor")?;
    if format!("{:x}", Sha256::digest(&descriptor_bytes)) != bundle.descriptor_sha256 {
        return Err(public_corpus_error(
            "descriptor_changed",
            "descriptor changed after bundle validation and before ablation execution",
        ));
    }
    let descriptor: PublicCorpusInput =
        serde_json::from_slice(&descriptor_bytes).map_err(|_| {
            public_corpus_error(
                "descriptor_json",
                "descriptor must remain valid during ablation execution",
            )
        })?;
    let input_recordings = descriptor
        .recordings
        .into_iter()
        .map(|recording| (recording.recording_id.clone(), recording))
        .collect::<BTreeMap<_, _>>();
    if input_recordings.len() != bundle.references.len() {
        return Err(public_corpus_error(
            "ablation_alignment",
            "descriptor and validated bundle recording counts differ",
        ));
    }

    let scorer_config = DiarizationScorerConfig {
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
    };
    let scorer_config_sha256 = canonical_sha256(&scorer_config)?;
    let diarization_request = DiarizationRequest {
        engine: DiarizationEngine::Acoustic,
        speaker_count: SpeakerCountRequest::Infer,
        ..DiarizationRequest::default()
    };
    let diarization_request_sha256 = canonical_sha256(&diarization_request)?;
    let protocol = PublicCorpusAblationProtocol {
        oracle_vad: true,
        oracle_speaker_count: false,
        maximum_recording_duration_ms,
        prefix_selection: "deterministic-prefix-v1".to_owned(),
        rss_observation: "linux-vmhwm-otherwise-sampled-process-rss-v1".to_owned(),
        diarization_request: diarization_request.clone(),
        diarization_request_sha256: diarization_request_sha256.clone(),
        change_calibration_id: ACOUSTIC_CHANGE_CALIBRATION_VERSION.to_owned(),
        change_calibration_fit_id: ACOUSTIC_CHANGE_CALIBRATION_FIT_VERSION.to_owned(),
        change_calibration_sha256: acoustic_change_calibration_sha256(),
        change_decision_probability: canonical_evidence_number(f64::from(
            acoustic_change_calibration().decision_probability,
        )),
        change_calibration_bins: scorer_config.calibration_bins,
        change_selection_policy_id: PUBLIC_CORPUS_CHANGE_SELECTION_POLICY_VERSION.to_owned(),
        change_selection_policy_sha256: change_selection_policy_sha256()?,
        speaker_pair_calibration_id: ACOUSTIC_CLUSTERING_PROBABILISTIC_VERSION.to_owned(),
        speaker_pair_calibration_sha256: acoustic_speaker_pair_calibration_sha256(),
    };
    let mut variants = Vec::with_capacity(AcousticFeatureAblation::ALL.len());
    for feature_ablation in AcousticFeatureAblation::ALL {
        let splits = evaluate_public_variant(
            &bundle,
            &input_recordings,
            &canonical_input,
            maximum_recording_duration_ms,
            &diarization_request,
            &scorer_config,
            feature_ablation,
            AcousticChangeDetectorMode::CalibratedPosterior,
            AcousticClusteringMode::FixedSafeV1,
            Some(evaluation_stage.selected_split()),
            &mut is_cancelled,
        )?;
        let feature_schema_sha256 =
            acoustic_feature_schema_sha256(feature_ablation.schema_version());
        let feature_configuration_sha256 =
            canonical_sha256(&PublicFeatureConfigurationFingerprint {
                runner_version: PUBLIC_CORPUS_ABLATION_RUNNER_VERSION,
                ablation: feature_ablation,
                feature_schema_sha256: &feature_schema_sha256,
                diarization_request_sha256: &diarization_request_sha256,
                change_calibration_sha256: &protocol.change_calibration_sha256,
            })?;
        variants.push(PublicCorpusAblationVariant {
            ablation: feature_ablation,
            feature_schema: feature_ablation.schema_version().id().to_owned(),
            feature_schema_sha256,
            feature_configuration_sha256,
            splits,
        });
    }
    let mut change_detector_variants = Vec::with_capacity(AcousticChangeDetectorMode::ALL.len());
    for detector_mode in AcousticChangeDetectorMode::ALL {
        let splits = if detector_mode == AcousticChangeDetectorMode::CalibratedPosterior {
            variants
                .iter()
                .find(|variant| variant.ablation == AcousticFeatureAblation::FullV2)
                .map(|variant| variant.splits.clone())
                .ok_or_else(|| {
                    public_corpus_error(
                        "change_detector_alignment",
                        "full-v2 representation evidence is unavailable",
                    )
                })?
        } else {
            evaluate_public_variant(
                &bundle,
                &input_recordings,
                &canonical_input,
                maximum_recording_duration_ms,
                &diarization_request,
                &scorer_config,
                AcousticFeatureAblation::FullV2,
                detector_mode,
                AcousticClusteringMode::FixedSafeV1,
                Some(evaluation_stage.selected_split()),
                &mut is_cancelled,
            )?
        };
        let feature_schema_sha256 =
            acoustic_feature_schema_sha256(AcousticFeatureAblation::FullV2.schema_version());
        let configuration_sha256 = canonical_sha256(&PublicChangeConfigurationFingerprint {
            runner_version: PUBLIC_CORPUS_ABLATION_RUNNER_VERSION,
            detector_mode,
            feature_ablation: AcousticFeatureAblation::FullV2,
            feature_schema_sha256: &feature_schema_sha256,
            diarization_request_sha256: &diarization_request_sha256,
            change_calibration_sha256: &protocol.change_calibration_sha256,
        })?;
        change_detector_variants.push(PublicCorpusChangeDetectorVariant {
            detector_mode,
            feature_ablation: AcousticFeatureAblation::FullV2,
            feature_schema_sha256,
            configuration_sha256,
            splits,
        });
    }
    let mut clustering_variants = Vec::with_capacity(AcousticClusteringMode::ALL.len());
    for clustering_mode in AcousticClusteringMode::ALL {
        let detector_mode = AcousticChangeDetectorMode::FixedSafeV1;
        let splits = if clustering_mode == AcousticClusteringMode::FixedSafeV1 {
            change_detector_variants
                .iter()
                .find(|variant| variant.detector_mode == detector_mode)
                .map(|variant| variant.splits.clone())
                .ok_or_else(|| {
                    public_corpus_error(
                        "clustering_alignment",
                        "fixed-safe detector evidence is unavailable",
                    )
                })?
        } else {
            evaluate_public_variant(
                &bundle,
                &input_recordings,
                &canonical_input,
                maximum_recording_duration_ms,
                &diarization_request,
                &scorer_config,
                AcousticFeatureAblation::FullV2,
                detector_mode,
                clustering_mode,
                Some(evaluation_stage.selected_split()),
                &mut is_cancelled,
            )?
        };
        let feature_schema_sha256 =
            acoustic_feature_schema_sha256(AcousticFeatureAblation::FullV2.schema_version());
        let configuration_sha256 = canonical_sha256(&PublicClusteringConfigurationFingerprint {
            runner_version: PUBLIC_CORPUS_ABLATION_RUNNER_VERSION,
            clustering_mode,
            detector_mode,
            feature_ablation: AcousticFeatureAblation::FullV2,
            feature_schema_sha256: &feature_schema_sha256,
            diarization_request_sha256: &diarization_request_sha256,
            speaker_pair_calibration_sha256: &protocol.speaker_pair_calibration_sha256,
        })?;
        clustering_variants.push(PublicCorpusClusteringVariant {
            clustering_mode,
            detector_mode,
            feature_ablation: AcousticFeatureAblation::FullV2,
            configuration_sha256,
            splits,
        });
    }
    let (
        development_gate,
        held_out_gate,
        change_development_gate,
        change_held_out_gate,
        selected_change_detector_mode,
        clustering_development_gate,
        clustering_held_out_gate,
        selected_clustering_mode,
    ) = match evaluation_stage {
        PublicCorpusEvaluationStage::Development => {
            let change_gate = change_development_gate(&change_detector_variants)?;
            let selected_change = if change_gate.passed {
                change_gate.candidate
            } else {
                change_gate.baseline
            };
            let clustering_gate = clustering_development_gate(&clustering_variants)?;
            let selected_clustering = if clustering_gate.passed {
                clustering_gate.candidate
            } else {
                clustering_gate.baseline
            };
            (
                Some(development_improvement_gate(&variants)?),
                None,
                Some(change_gate),
                None,
                selected_change,
                Some(clustering_gate),
                None,
                selected_clustering,
            )
        }
        PublicCorpusEvaluationStage::Certification => {
            let locked = locked_development.as_ref().ok_or_else(|| {
                public_corpus_error(
                    "ablation_stage_lock",
                    "certification requires locked development evidence",
                )
            })?;
            let change_gate = locked.change_development_gate.as_ref().ok_or_else(|| {
                public_corpus_error(
                    "ablation_stage_lock",
                    "locked development evidence has no change-detector gate",
                )
            })?;
            let clustering_gate = locked.clustering_development_gate.as_ref().ok_or_else(|| {
                public_corpus_error(
                    "ablation_stage_lock",
                    "locked development evidence has no clustering gate",
                )
            })?;
            let representation_gate = locked.development_gate.as_ref().ok_or_else(|| {
                public_corpus_error(
                    "ablation_stage_lock",
                    "locked development evidence has no representation gate",
                )
            })?;
            if !representation_gate.passed || !change_gate.passed || !clustering_gate.passed {
                return Err(public_corpus_error(
                    "ablation_stage_lock",
                    "held-out certification is forbidden until every development candidate passes its promotion gate",
                ));
            }
            let selected_change = locked.selected_change_detector_mode;
            let selected_clustering = locked.selected_clustering_mode;
            (
                None,
                Some(held_out_non_regression_gate(&variants)?),
                None,
                Some(change_held_out_gate(
                    &change_detector_variants,
                    selected_change,
                )?),
                selected_change,
                None,
                Some(clustering_held_out_gate(
                    &clustering_variants,
                    selected_clustering,
                )?),
                selected_clustering,
            )
        }
    };
    let locked_development_result_sha256 = locked_development
        .as_ref()
        .map(|evidence| evidence.result_sha256.clone());
    let locked_development_accuracy_sha256 = locked_development
        .as_ref()
        .map(|evidence| evidence.deterministic_accuracy_sha256.clone());
    let mut result = PublicCorpusAblationEvidence {
        schema_version: PUBLIC_CORPUS_ABLATION_SCHEMA_VERSION.to_owned(),
        runner_version: PUBLIC_CORPUS_ABLATION_RUNNER_VERSION.to_owned(),
        scorer_version: DIARIZATION_SCORER_VERSION.to_owned(),
        corpus_key: bundle.corpus_key,
        source_version: bundle.source_version,
        bundle_sha256: bundle.bundle_sha256,
        descriptor_sha256: bundle.descriptor_sha256,
        scorer_config,
        scorer_config_sha256,
        evaluation_stage,
        locked_development_result_sha256,
        locked_development_accuracy_sha256,
        selected_change_detector_mode,
        selected_clustering_mode,
        protocol,
        variants,
        change_detector_variants,
        clustering_variants,
        development_gate,
        held_out_gate,
        change_development_gate,
        change_held_out_gate,
        clustering_development_gate,
        clustering_held_out_gate,
        deterministic_accuracy_sha256: String::new(),
        result_sha256: String::new(),
    };
    result.deterministic_accuracy_sha256 = deterministic_accuracy_sha256(&result)?;
    result.result_sha256 = canonical_sha256(&result)?;
    verify_public_corpus_ablation_evidence(&result)?;
    checkpoint_cancelled(&mut is_cancelled)?;
    write_new_json(
        evidence_output_path,
        &evidence_output_parent,
        &result,
        "ablation evidence",
    )?;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_public_variant(
    bundle: &PublicCorpusBundle,
    input_recordings: &BTreeMap<String, PublicCorpusInputRecording>,
    canonical_input: &Path,
    maximum_recording_duration_ms: Option<u64>,
    diarization_request: &DiarizationRequest,
    scorer_config: &DiarizationScorerConfig,
    feature_ablation: AcousticFeatureAblation,
    detector_mode: AcousticChangeDetectorMode,
    clustering_mode: AcousticClusteringMode,
    target_split: Option<EvaluationSplit>,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<Vec<PublicCorpusAblationSplit>> {
    let mut by_split = BTreeMap::<EvaluationSplit, PublicAblationAccumulator>::new();
    for ((reference, recording_evidence), manifest_recording) in bundle
        .references
        .iter()
        .zip(&bundle.recordings)
        .zip(&bundle.manifest.recordings)
    {
        if target_split.is_some_and(|target| manifest_recording.split != target) {
            continue;
        }
        checkpoint_cancelled(is_cancelled)?;
        let input_recording = input_recordings
            .get(&reference.recording_id)
            .ok_or_else(|| {
                public_corpus_error(
                    "ablation_alignment",
                    "validated recording is absent from the descriptor",
                )
            })?;
        if recording_evidence.sample_rate_hz != 16_000
            || recording_evidence.channel_count != 1
            || recording_evidence.selected_channel != 1
        {
            return Err(public_corpus_error(
                "ablation_audio_contract",
                "the current acoustic ablation runner requires 16 kHz mono PCM WAV input",
            ));
        }
        let audio_path =
            canonical_relative_file(canonical_input, &input_recording.audio_path, "audio")?;
        let audio_bytes = read_bounded(&audio_path, MAX_EVALUATION_AUDIO_BYTES, "ablation_audio")?;
        checkpoint_cancelled(is_cancelled)?;
        if format!("{:x}", Sha256::digest(&audio_bytes)) != recording_evidence.audio_sha256 {
            return Err(public_corpus_error(
                "audio_changed",
                "audio changed after bundle validation and before ablation execution",
            ));
        }
        let mut samples =
            crate::native_engine::decode::read_wav_16k_mono(&audio_bytes).map_err(|_| {
                public_corpus_error(
                    "ablation_audio_decode",
                    "one validated WAV could not be decoded as 16 kHz mono PCM",
                )
            })?;
        checkpoint_cancelled(is_cancelled)?;
        let available_duration_ms = u64::try_from(samples.len()).unwrap_or(u64::MAX) / 16;
        let evaluation_duration_ms = maximum_recording_duration_ms
            .map_or(available_duration_ms, |maximum| {
                maximum.min(available_duration_ms)
            });
        let clipped_reference = clipped_reference(reference, Some(evaluation_duration_ms))?;
        let maximum_samples = usize::try_from(clipped_reference.duration_ms)
            .ok()
            .and_then(|duration_ms| duration_ms.checked_mul(16))
            .ok_or_else(|| {
                public_corpus_error(
                    "ablation_duration",
                    "clipped recording duration exceeds the supported sample range",
                )
            })?;
        samples.truncate(maximum_samples);
        let boundary_hints = AcousticBoundaryHints {
            speech_regions_ms: merged_scored_speech_regions(
                &clipped_reference.turns,
                &clipped_reference.ignored_regions,
            ),
            ..AcousticBoundaryHints::default()
        };
        let started = Instant::now();
        let (
            report_turns,
            speaker_count_estimate,
            detector_changes,
            evaluated_changes,
            clustering_evidence,
        ) = if boundary_hints.speech_regions_ms.is_empty() {
            checkpoint_cancelled(is_cancelled)?;
            (
                Vec::new(),
                None,
                Vec::new(),
                Vec::new(),
                AcousticClusteringEvaluationEvidence {
                    requested_mode: clustering_mode,
                    executed_mode: AcousticClusteringMode::FixedSafeV1,
                    fallback_reason: None,
                    speaker_count_stability: 0.0,
                },
            )
        } else {
            let input_sha256 = hash_pcm_prefix(&samples);
            let (report, _, change_evidence, clustering_evidence) =
                diarize_acoustic_pcm_with_modes_evidence(
                    AcousticDiarizationInput {
                        samples: &samples,
                        normalized_input_sha256: &input_sha256,
                        segments: &[],
                        word_aligned: false,
                        request: diarization_request,
                        boundary_hints: &boundary_hints,
                    },
                    feature_ablation,
                    detector_mode,
                    clustering_mode,
                    &mut *is_cancelled,
                )?;
            let detector_changes = change_evidence
                .emitted
                .into_iter()
                .filter(|evidence| {
                    !evidence.vad_boundary
                        && !evidence.supervised_boundary
                        && evidence.boundary_ms > 0
                        && evidence.boundary_ms < clipped_reference.duration_ms
                })
                .map(|evidence| ChangeProbabilityObservation {
                    boundary_ms: evidence.boundary_ms,
                    probability: f64::from(evidence.change_probability),
                })
                .collect();
            (
                report.turns,
                report.speaker_count.estimate,
                detector_changes,
                change_evidence.evaluated,
                clustering_evidence,
            )
        };
        let wall_time_ms = u64::try_from(started.elapsed().as_millis())
            .unwrap_or(u64::MAX)
            .max(1);
        let peak_rss_bytes = sampled_process_rss_bytes();
        let hypothesis = DiarizationHypothesisDocument {
            schema_version: DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: clipped_reference.recording_id.clone(),
            duration_ms: clipped_reference.duration_ms,
            turns: report_turns
                .into_iter()
                .map(|turn| EvaluationTurn {
                    start_ms: turn.start_ms,
                    end_ms: turn.end_ms.min(clipped_reference.duration_ms),
                    speaker: turn.speaker_ref,
                    speaker_confidence: turn.speaker_confidence,
                    overlap_suspected: turn.overlap_suspected,
                })
                .filter(|turn| turn.end_ms > turn.start_ms)
                .collect(),
            speaker_count_estimate,
            performance: Some(EvaluationPerformanceObservation {
                audio_duration_ms: clipped_reference.duration_ms,
                wall_time_ms,
                peak_rss_bytes,
            }),
        };
        let score = score_diarization_documents(&clipped_reference, &hypothesis, scorer_config)?;
        let reference_changes = speaker_change_points_ms(
            &clipped_reference.turns,
            clipped_reference.duration_ms,
            true,
        )?;
        let detector_change_score = score_change_points(
            &reference_changes
                .iter()
                .map(|timestamp| *timestamp as f64 / 1_000.0)
                .collect::<Vec<_>>(),
            &detector_changes
                .iter()
                .map(|observation| observation.boundary_ms as f64 / 1_000.0)
                .collect::<Vec<_>>(),
            scorer_config.change_boundary_collar_ms as f64 / 1_000.0,
        )?;
        let detector_change_ms = detector_changes
            .iter()
            .map(|observation| observation.boundary_ms)
            .collect::<Vec<_>>();
        let change_collar_metrics = PUBLIC_CHANGE_DIAGNOSTIC_COLLARS_MS
            .iter()
            .map(|&collar_ms| {
                score_change_metric_observation(
                    &reference_changes,
                    &detector_change_ms,
                    collar_ms,
                    collar_ms as f64,
                )
            })
            .collect::<FwResult<Vec<_>>>()?;
        let change_threshold_sweep = PUBLIC_CHANGE_THRESHOLD_SWEEP
            .iter()
            .map(|&threshold| {
                let selected = select_acoustic_change_evidence_at_threshold(
                    &evaluated_changes,
                    threshold as f32,
                )?;
                let hypothesis_ms = selected
                    .into_iter()
                    .filter_map(|evidence| {
                        (evidence.boundary_ms > 0
                            && evidence.boundary_ms < clipped_reference.duration_ms)
                            .then_some(evidence.boundary_ms)
                    })
                    .collect::<Vec<_>>();
                score_change_metric_observation(
                    &reference_changes,
                    &hypothesis_ms,
                    scorer_config.change_boundary_collar_ms,
                    threshold,
                )
            })
            .collect::<FwResult<Vec<_>>>()?;
        let change_calibration = score_change_event_calibration(
            &reference_changes,
            &detector_changes,
            scorer_config.change_boundary_collar_ms,
            scorer_config.calibration_bins,
        )?;
        by_split.entry(manifest_recording.split).or_default().push(
            &score,
            &detector_change_score,
            &change_calibration,
            &change_collar_metrics,
            &change_threshold_sweep,
            &clustering_evidence,
        )?;
    }
    let splits = [
        EvaluationSplit::Train,
        EvaluationSplit::Development,
        EvaluationSplit::Test,
    ]
    .into_iter()
    .filter_map(|split| {
        by_split
            .remove(&split)
            .map(|aggregate| aggregate.finish(split))
    })
    .collect::<Vec<_>>();
    if let Some(target_split) = target_split
        && (splits.len() != 1 || splits[0].split != target_split)
    {
        return Err(public_corpus_error(
            "ablation_split_missing",
            "the selected evaluation stage has no recording in the validated bundle",
        ));
    }
    Ok(splits)
}

#[derive(Debug, Clone, Copy)]
struct ChangeProbabilityObservation {
    boundary_ms: u64,
    probability: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ChangeReliabilityAccumulator {
    observation_count: u64,
    positive_count: u64,
    probability_sum: f64,
}

#[derive(Debug, Clone)]
struct ChangeEventCalibrationAggregate {
    observation_count: u64,
    positive_count: u64,
    brier_sum: f64,
    bins: Vec<ChangeReliabilityAccumulator>,
}

#[derive(Debug)]
struct ChangeMetricObservation {
    key: f64,
    collar_ms: u64,
    score: ChangePointScore,
    absolute_errors_sec: Vec<f64>,
}

#[derive(Debug, Default)]
struct ChangeMetricAccumulator {
    key: f64,
    collar_ms: u64,
    reference_count: u64,
    hypothesis_count: u64,
    matched_count: u64,
    absolute_errors_sec: Vec<f64>,
}

#[derive(Debug)]
struct ChangeMetricValues {
    reference_count: u64,
    hypothesis_count: u64,
    matched_count: u64,
    precision: Option<f64>,
    recall: Option<f64>,
    f1: Option<f64>,
    mean_absolute_error_sec: Option<f64>,
    p50_absolute_error_sec: Option<f64>,
    p90_absolute_error_sec: Option<f64>,
    p95_absolute_error_sec: Option<f64>,
}

impl ChangeMetricAccumulator {
    fn push(&mut self, observation: &ChangeMetricObservation) -> FwResult<()> {
        if self.reference_count == 0 && self.hypothesis_count == 0 && self.matched_count == 0 {
            self.key = observation.key;
            self.collar_ms = observation.collar_ms;
        } else if self.key.to_bits() != observation.key.to_bits()
            || self.collar_ms != observation.collar_ms
        {
            return Err(public_corpus_error(
                "change_diagnostic_grid",
                "all recordings must use the same change diagnostic grid",
            ));
        }
        if observation.absolute_errors_sec.len() != observation.score.matched_count {
            return Err(public_corpus_error(
                "change_diagnostic_match",
                "change timing-error count does not match the scored boundary matches",
            ));
        }
        self.reference_count = self
            .reference_count
            .saturating_add(u64::try_from(observation.score.reference_count).unwrap_or(u64::MAX));
        self.hypothesis_count = self
            .hypothesis_count
            .saturating_add(u64::try_from(observation.score.hypothesis_count).unwrap_or(u64::MAX));
        self.matched_count = self
            .matched_count
            .saturating_add(u64::try_from(observation.score.matched_count).unwrap_or(u64::MAX));
        self.absolute_errors_sec
            .extend_from_slice(&observation.absolute_errors_sec);
        Ok(())
    }

    fn finish_values(mut self) -> ChangeMetricValues {
        self.absolute_errors_sec.sort_by(f64::total_cmp);
        let precision = ratio(self.matched_count, self.hypothesis_count);
        let recall = ratio(self.matched_count, self.reference_count);
        let f1 = precision.zip(recall).map(|(precision, recall)| {
            let denominator = precision + recall;
            if denominator > 0.0 {
                2.0 * precision * recall / denominator
            } else {
                0.0
            }
        });
        let mean = (!self.absolute_errors_sec.is_empty()).then(|| {
            canonical_evidence_number(
                self.absolute_errors_sec.iter().sum::<f64>()
                    / self.absolute_errors_sec.len() as f64,
            )
        });
        let quantile = |probability: f64| {
            (!self.absolute_errors_sec.is_empty()).then(|| {
                let index = ((self.absolute_errors_sec.len() - 1) as f64 * probability)
                    .round()
                    .clamp(0.0, (self.absolute_errors_sec.len() - 1) as f64)
                    as usize;
                canonical_evidence_number(self.absolute_errors_sec[index])
            })
        };
        ChangeMetricValues {
            reference_count: self.reference_count,
            hypothesis_count: self.hypothesis_count,
            matched_count: self.matched_count,
            precision,
            recall,
            f1,
            mean_absolute_error_sec: mean,
            p50_absolute_error_sec: quantile(0.50),
            p90_absolute_error_sec: quantile(0.90),
            p95_absolute_error_sec: quantile(0.95),
        }
    }
}

fn score_change_metric_observation(
    reference_ms: &[u64],
    hypothesis_ms: &[u64],
    collar_ms: u64,
    key: f64,
) -> FwResult<ChangeMetricObservation> {
    let reference_sec = reference_ms
        .iter()
        .map(|timestamp| *timestamp as f64 / 1_000.0)
        .collect::<Vec<_>>();
    let hypothesis_sec = hypothesis_ms
        .iter()
        .map(|timestamp| *timestamp as f64 / 1_000.0)
        .collect::<Vec<_>>();
    let collar_sec = collar_ms as f64 / 1_000.0;
    let score = score_change_points(&reference_sec, &hypothesis_sec, collar_sec)?;
    let absolute_errors_sec =
        minimum_error_change_match_errors(&reference_sec, &hypothesis_sec, collar_sec);
    Ok(ChangeMetricObservation {
        key,
        collar_ms,
        score,
        absolute_errors_sec,
    })
}

fn minimum_error_change_match_errors(
    reference: &[f64],
    hypothesis: &[f64],
    collar_sec: f64,
) -> Vec<f64> {
    #[derive(Clone, Copy, Default)]
    struct Match {
        count: usize,
        total_error: f64,
    }
    let better = |left: Match, right: Match| {
        if right.count > left.count
            || (right.count == left.count && right.total_error < left.total_error)
        {
            right
        } else {
            left
        }
    };
    let columns = hypothesis.len() + 1;
    let mut previous = vec![Match::default(); columns];
    let mut current = vec![Match::default(); columns];
    let mut decisions = vec![0_u8; (reference.len() + 1).saturating_mul(columns)];
    for reference_index in 1..=reference.len() {
        decisions[reference_index * columns] = 1;
        current[0] = Match::default();
        for hypothesis_index in 1..=hypothesis.len() {
            let up = previous[hypothesis_index];
            let left = current[hypothesis_index - 1];
            let mut best = better(up, left);
            let mut decision = if best.count == left.count
                && best.total_error.to_bits() == left.total_error.to_bits()
                && (up.count != left.count
                    || up.total_error.to_bits() != left.total_error.to_bits())
            {
                2
            } else {
                1
            };
            let error = (reference[reference_index - 1] - hypothesis[hypothesis_index - 1]).abs();
            if error <= collar_sec {
                let matched = Match {
                    count: previous[hypothesis_index - 1].count + 1,
                    total_error: previous[hypothesis_index - 1].total_error + error,
                };
                let selected = better(best, matched);
                if selected.count == matched.count
                    && selected.total_error.to_bits() == matched.total_error.to_bits()
                    && (best.count != matched.count
                        || best.total_error.to_bits() != matched.total_error.to_bits())
                {
                    best = selected;
                    decision = 3;
                }
            }
            current[hypothesis_index] = best;
            decisions[reference_index * columns + hypothesis_index] = decision;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    let mut errors = Vec::with_capacity(previous[hypothesis.len()].count);
    let (mut reference_index, mut hypothesis_index) = (reference.len(), hypothesis.len());
    while reference_index > 0 && hypothesis_index > 0 {
        match decisions[reference_index * columns + hypothesis_index] {
            3 => {
                errors.push(
                    (reference[reference_index - 1] - hypothesis[hypothesis_index - 1]).abs(),
                );
                reference_index -= 1;
                hypothesis_index -= 1;
            }
            2 => hypothesis_index -= 1,
            _ => reference_index -= 1,
        }
    }
    errors
}

fn accumulate_change_metric_grid(
    aggregate: &mut Vec<ChangeMetricAccumulator>,
    recording: &[ChangeMetricObservation],
    label: &str,
) -> FwResult<()> {
    if aggregate.is_empty() {
        aggregate.resize_with(recording.len(), ChangeMetricAccumulator::default);
    } else if aggregate.len() != recording.len() {
        return Err(public_corpus_error(
            "change_diagnostic_grid",
            &format!("{label} diagnostic grid changed between recordings"),
        ));
    }
    for (aggregate, observation) in aggregate.iter_mut().zip(recording) {
        aggregate.push(observation)?;
    }
    Ok(())
}

#[cfg(test)]
fn report_change_observations(
    turns: &[crate::model::DiarizationTurn],
) -> Vec<ChangeProbabilityObservation> {
    turns
        .windows(2)
        .filter_map(|window| {
            let previous = &window[0];
            let next = &window[1];
            (previous.end_ms == next.start_ms && previous.speaker_ref != next.speaker_ref).then(
                || ChangeProbabilityObservation {
                    boundary_ms: next.start_ms,
                    probability: previous.change_confidence.unwrap_or(0.0),
                },
            )
        })
        .collect()
}

fn score_change_event_calibration(
    reference_ms: &[u64],
    hypothesis: &[ChangeProbabilityObservation],
    collar_ms: u64,
    bins: usize,
) -> FwResult<ChangeEventCalibrationAggregate> {
    if bins == 0 || bins > 100 {
        return Err(public_corpus_error(
            "change_calibration_bins",
            "change-event calibration bins must be within 1..=100",
        ));
    }
    if reference_ms.windows(2).any(|window| window[0] >= window[1])
        || hypothesis
            .windows(2)
            .any(|window| window[0].boundary_ms >= window[1].boundary_ms)
    {
        return Err(public_corpus_error(
            "change_calibration_order",
            "change-event inputs must be strictly increasing",
        ));
    }
    if hypothesis.iter().any(|observation| {
        !observation.probability.is_finite() || !(0.0..=1.0).contains(&observation.probability)
    }) {
        return Err(public_corpus_error(
            "change_calibration_probability",
            "change-event probabilities must be finite and within [0, 1]",
        ));
    }

    let mut aggregate = ChangeEventCalibrationAggregate {
        observation_count: 0,
        positive_count: 0,
        brier_sum: 0.0,
        bins: vec![ChangeReliabilityAccumulator::default(); bins],
    };
    let mut observe = |probability: f64, positive: bool| {
        let outcome = f64::from(positive);
        let bin = ((probability * bins as f64).floor() as usize).min(bins - 1);
        aggregate.observation_count = aggregate.observation_count.saturating_add(1);
        aggregate.positive_count = aggregate.positive_count.saturating_add(u64::from(positive));
        aggregate.brier_sum += (probability - outcome).powi(2);
        aggregate.bins[bin].observation_count =
            aggregate.bins[bin].observation_count.saturating_add(1);
        aggregate.bins[bin].positive_count = aggregate.bins[bin]
            .positive_count
            .saturating_add(u64::from(positive));
        aggregate.bins[bin].probability_sum += probability;
    };

    let mut reference_index = 0usize;
    let mut hypothesis_index = 0usize;
    while reference_index < reference_ms.len() && hypothesis_index < hypothesis.len() {
        let reference = reference_ms[reference_index];
        let prediction = hypothesis[hypothesis_index];
        if prediction.boundary_ms.saturating_add(collar_ms) < reference {
            observe(prediction.probability, false);
            hypothesis_index += 1;
        } else if reference.saturating_add(collar_ms) < prediction.boundary_ms {
            observe(0.0, true);
            reference_index += 1;
        } else {
            observe(prediction.probability, true);
            reference_index += 1;
            hypothesis_index += 1;
        }
    }
    for prediction in &hypothesis[hypothesis_index..] {
        observe(prediction.probability, false);
    }
    for _ in &reference_ms[reference_index..] {
        observe(0.0, true);
    }
    Ok(aggregate)
}

#[derive(Default)]
struct SpeakerCountStratumAccumulator {
    recording_count: u64,
    posterior_recording_count: u64,
    unresolved_recording_count: u64,
    zero_reference_probability_count: u64,
    exact_speaker_count_count: u64,
    negative_log_likelihood_sum: f64,
    negative_log_likelihood_count: u64,
    brier_sum: f64,
    brier_count: u64,
    top_k_hit_count: u64,
    top_k_observation_count: u64,
    credible_set_hit_count: u64,
    credible_set_observation_count: u64,
}

#[derive(Default)]
struct PublicAblationAccumulator {
    recording_count: u64,
    reference_speaker_time_sec: f64,
    missed_speech_sec: f64,
    false_alarm_sec: f64,
    speaker_confusion_sec: f64,
    overlap_reference_sec: f64,
    overlap_hypothesis_sec: f64,
    overlap_true_positive_sec: f64,
    overlap_false_positive_sec: f64,
    overlap_false_negative_sec: f64,
    macro_der_sum: f64,
    macro_der_count: u64,
    macro_jer_sum: f64,
    macro_jer_count: u64,
    change_reference_count: u64,
    change_hypothesis_count: u64,
    change_matched_count: u64,
    change_absolute_error_sec: f64,
    change_event_observation_count: u64,
    change_event_positive_count: u64,
    change_event_brier_sum: f64,
    change_reliability: Vec<ChangeReliabilityAccumulator>,
    change_collar_metrics: Vec<ChangeMetricAccumulator>,
    change_threshold_sweep: Vec<ChangeMetricAccumulator>,
    exact_speaker_count: u64,
    signed_speaker_count_error: i64,
    absolute_speaker_count_error: u64,
    absolute_speaker_count_errors: Vec<u64>,
    speaker_count_confusion: BTreeMap<(u32, u32), u64>,
    speaker_count_strata: BTreeMap<u32, SpeakerCountStratumAccumulator>,
    speaker_count_duration_strata:
        BTreeMap<PublicSpeakerCountDurationBucket, SpeakerCountStratumAccumulator>,
    count_posterior_recording_count: u64,
    count_posterior_unavailable_count: u64,
    count_unresolved_recording_count: u64,
    count_zero_reference_probability_count: u64,
    count_negative_log_likelihood_sum: f64,
    count_negative_log_likelihood_count: u64,
    count_brier_sum: f64,
    count_brier_count: u64,
    count_top_k_hit_count: u64,
    count_top_k_observation_count: u64,
    count_credible_set_hit_count: u64,
    count_credible_set_observation_count: u64,
    count_entropy_sum: f64,
    count_entropy_count: u64,
    dominant_collapse_recording_count: u64,
    reference_collapse_recording_count: u64,
    phantom_speaker_count: u64,
    collapsed_reference_speaker_count: u64,
    effective_speaker_count_sum: u64,
    effective_speaker_count_count: u64,
    dominant_speaker_shares: Vec<f64>,
    unknown_speaker_shares: Vec<f64>,
    minority_reference_recall_sum: f64,
    minority_reference_recall_count: u64,
    reference_word_count: u64,
    scored_word_count: u64,
    correct_word_count: u64,
    incorrect_word_count: u64,
    unknown_word_count: u64,
    excluded_word_count: u64,
    macro_word_diarization_error_sum: f64,
    macro_word_diarization_error_count: u64,
    selective_reference_speaker_time_sec: f64,
    selective_covered_speaker_time_sec: f64,
    selective_correct_covered_speaker_time_sec: f64,
    selective_error_covered_speaker_time_sec: f64,
    selective_unknown_speaker_time_sec: f64,
    assignment_observed_duration_sec: f64,
    assignment_opportunity_duration_sec: f64,
    assignment_brier_weighted_sum: f64,
    assignment_brier_weight: f64,
    assignment_ece_weighted_sum: f64,
    assignment_ece_weight: f64,
    speaker_count_stability_sum: f64,
    speaker_count_stability_count: u64,
    clustering_fallback_count: u64,
    clustering_insufficient_voice_fallback_count: u64,
    clustering_invalid_posterior_fallback_count: u64,
    clustering_unstable_count_fallback_count: u64,
    audio_duration_sec: f64,
    wall_time_sec: f64,
    sampled_peak_rss_bytes: u64,
}

impl PublicAblationAccumulator {
    fn push(
        &mut self,
        score: &crate::diarization::AuthoritativeDiarizationScore,
        detector_change_score: &ChangePointScore,
        change_calibration: &ChangeEventCalibrationAggregate,
        change_collar_metrics: &[ChangeMetricObservation],
        change_threshold_sweep: &[ChangeMetricObservation],
        clustering: &AcousticClusteringEvaluationEvidence,
    ) -> FwResult<()> {
        self.recording_count = self.recording_count.checked_add(1).ok_or_else(|| {
            public_corpus_error(
                "ablation_aggregate_overflow",
                "recording count exceeds the supported range",
            )
        })?;
        self.reference_speaker_time_sec += score.diarization.reference_speaker_time_sec;
        self.missed_speech_sec += score.diarization.missed_speech_sec;
        self.false_alarm_sec += score.diarization.false_alarm_sec;
        self.speaker_confusion_sec += score.diarization.speaker_confusion_sec;
        self.overlap_reference_sec += score.overlap.reference_overlap_sec;
        self.overlap_hypothesis_sec += score.overlap.hypothesis_overlap_sec;
        self.overlap_true_positive_sec += score.overlap.true_positive_sec;
        self.overlap_false_positive_sec += score.overlap.false_positive_sec;
        self.overlap_false_negative_sec += score.overlap.false_negative_sec;
        if let Some(value) = score.diarization.der {
            self.macro_der_sum += value;
            self.macro_der_count += 1;
        }
        if let Some(value) = score.diarization.jer {
            self.macro_jer_sum += value;
            self.macro_jer_count += 1;
        }
        self.change_reference_count = self.change_reference_count.saturating_add(
            u64::try_from(detector_change_score.reference_count).unwrap_or(u64::MAX),
        );
        self.change_hypothesis_count = self.change_hypothesis_count.saturating_add(
            u64::try_from(detector_change_score.hypothesis_count).unwrap_or(u64::MAX),
        );
        self.change_matched_count = self
            .change_matched_count
            .saturating_add(u64::try_from(detector_change_score.matched_count).unwrap_or(u64::MAX));
        if let Some(mean_absolute_error_sec) = detector_change_score.mean_absolute_error_sec {
            self.change_absolute_error_sec +=
                mean_absolute_error_sec * detector_change_score.matched_count as f64;
        }
        if self.change_reliability.is_empty() {
            self.change_reliability.resize(
                change_calibration.bins.len(),
                ChangeReliabilityAccumulator::default(),
            );
        } else if self.change_reliability.len() != change_calibration.bins.len() {
            return Err(public_corpus_error(
                "change_calibration_bins",
                "all recordings in an ablation split must use the same reliability bins",
            ));
        }
        self.change_event_observation_count = self
            .change_event_observation_count
            .saturating_add(change_calibration.observation_count);
        self.change_event_positive_count = self
            .change_event_positive_count
            .saturating_add(change_calibration.positive_count);
        self.change_event_brier_sum += change_calibration.brier_sum;
        for (aggregate, recording) in self
            .change_reliability
            .iter_mut()
            .zip(&change_calibration.bins)
        {
            aggregate.observation_count = aggregate
                .observation_count
                .saturating_add(recording.observation_count);
            aggregate.positive_count = aggregate
                .positive_count
                .saturating_add(recording.positive_count);
            aggregate.probability_sum += recording.probability_sum;
        }
        accumulate_change_metric_grid(
            &mut self.change_collar_metrics,
            change_collar_metrics,
            "change collar",
        )?;
        accumulate_change_metric_grid(
            &mut self.change_threshold_sweep,
            change_threshold_sweep,
            "change threshold",
        )?;
        self.exact_speaker_count += u64::from(score.speaker_count.absolute_error == 0);
        self.signed_speaker_count_error = self
            .signed_speaker_count_error
            .saturating_add(score.speaker_count.signed_error);
        self.absolute_speaker_count_error = self
            .absolute_speaker_count_error
            .saturating_add(score.speaker_count.absolute_error);
        self.absolute_speaker_count_errors
            .push(score.speaker_count.absolute_error);
        let reference_speakers =
            u32::try_from(score.speaker_count.reference_speakers).unwrap_or(u32::MAX);
        let hypothesis_speakers =
            u32::try_from(score.speaker_count.hypothesis_speakers).unwrap_or(u32::MAX);
        let confusion_count = self
            .speaker_count_confusion
            .entry((reference_speakers, hypothesis_speakers))
            .or_default();
        *confusion_count = confusion_count.saturating_add(1);
        let stratum = self
            .speaker_count_strata
            .entry(reference_speakers)
            .or_default();
        stratum.recording_count = stratum.recording_count.saturating_add(1);
        stratum.exact_speaker_count_count = stratum
            .exact_speaker_count_count
            .saturating_add(u64::from(score.speaker_count.absolute_error == 0));
        if score.speaker_count_posterior.posterior_available {
            self.count_posterior_recording_count =
                self.count_posterior_recording_count.saturating_add(1);
            stratum.posterior_recording_count = stratum.posterior_recording_count.saturating_add(1);
        } else {
            self.count_posterior_unavailable_count =
                self.count_posterior_unavailable_count.saturating_add(1);
        }
        if score.speaker_count_posterior.unresolved {
            self.count_unresolved_recording_count =
                self.count_unresolved_recording_count.saturating_add(1);
            stratum.unresolved_recording_count =
                stratum.unresolved_recording_count.saturating_add(1);
        }
        if score
            .speaker_count_posterior
            .infinite_negative_log_likelihood
        {
            self.count_zero_reference_probability_count = self
                .count_zero_reference_probability_count
                .saturating_add(1);
            stratum.zero_reference_probability_count =
                stratum.zero_reference_probability_count.saturating_add(1);
        }
        if let Some(value) = score.speaker_count_posterior.negative_log_likelihood {
            self.count_negative_log_likelihood_sum += value;
            self.count_negative_log_likelihood_count =
                self.count_negative_log_likelihood_count.saturating_add(1);
            stratum.negative_log_likelihood_sum += value;
            stratum.negative_log_likelihood_count =
                stratum.negative_log_likelihood_count.saturating_add(1);
        }
        if let Some(value) = score.speaker_count_posterior.brier_score {
            self.count_brier_sum += value;
            self.count_brier_count = self.count_brier_count.saturating_add(1);
            stratum.brier_sum += value;
            stratum.brier_count = stratum.brier_count.saturating_add(1);
        }
        if let Some(hit) = score.speaker_count_posterior.top_k_hit {
            self.count_top_k_observation_count =
                self.count_top_k_observation_count.saturating_add(1);
            self.count_top_k_hit_count = self.count_top_k_hit_count.saturating_add(u64::from(hit));
            stratum.top_k_observation_count = stratum.top_k_observation_count.saturating_add(1);
            stratum.top_k_hit_count = stratum.top_k_hit_count.saturating_add(u64::from(hit));
        }
        if let Some(hit) = score.speaker_count_posterior.credible_set_hit {
            self.count_credible_set_observation_count =
                self.count_credible_set_observation_count.saturating_add(1);
            self.count_credible_set_hit_count = self
                .count_credible_set_hit_count
                .saturating_add(u64::from(hit));
            stratum.credible_set_observation_count =
                stratum.credible_set_observation_count.saturating_add(1);
            stratum.credible_set_hit_count = stratum
                .credible_set_hit_count
                .saturating_add(u64::from(hit));
        }
        let duration_bucket =
            speaker_count_duration_bucket(score.scored_duration_sec + score.ignored_duration_sec);
        let duration_stratum = self
            .speaker_count_duration_strata
            .entry(duration_bucket)
            .or_default();
        duration_stratum.recording_count = duration_stratum.recording_count.saturating_add(1);
        duration_stratum.exact_speaker_count_count = duration_stratum
            .exact_speaker_count_count
            .saturating_add(u64::from(score.speaker_count.absolute_error == 0));
        if score.speaker_count_posterior.posterior_available {
            duration_stratum.posterior_recording_count =
                duration_stratum.posterior_recording_count.saturating_add(1);
        }
        if score.speaker_count_posterior.unresolved {
            duration_stratum.unresolved_recording_count = duration_stratum
                .unresolved_recording_count
                .saturating_add(1);
        }
        if score
            .speaker_count_posterior
            .infinite_negative_log_likelihood
        {
            duration_stratum.zero_reference_probability_count = duration_stratum
                .zero_reference_probability_count
                .saturating_add(1);
        }
        if let Some(value) = score.speaker_count_posterior.negative_log_likelihood {
            duration_stratum.negative_log_likelihood_sum += value;
            duration_stratum.negative_log_likelihood_count = duration_stratum
                .negative_log_likelihood_count
                .saturating_add(1);
        }
        if let Some(value) = score.speaker_count_posterior.brier_score {
            duration_stratum.brier_sum += value;
            duration_stratum.brier_count = duration_stratum.brier_count.saturating_add(1);
        }
        if let Some(hit) = score.speaker_count_posterior.top_k_hit {
            duration_stratum.top_k_observation_count =
                duration_stratum.top_k_observation_count.saturating_add(1);
            duration_stratum.top_k_hit_count = duration_stratum
                .top_k_hit_count
                .saturating_add(u64::from(hit));
        }
        if let Some(hit) = score.speaker_count_posterior.credible_set_hit {
            duration_stratum.credible_set_observation_count = duration_stratum
                .credible_set_observation_count
                .saturating_add(1);
            duration_stratum.credible_set_hit_count = duration_stratum
                .credible_set_hit_count
                .saturating_add(u64::from(hit));
        }
        if score.speaker_count_posterior.posterior_available
            && let Some(entropy_bits) = score.speaker_count_posterior.entropy_bits
        {
            self.count_entropy_sum += entropy_bits;
            self.count_entropy_count = self.count_entropy_count.saturating_add(1);
        }
        self.dominant_collapse_recording_count = self
            .dominant_collapse_recording_count
            .saturating_add(u64::from(
                score.speaker_occupancy.dominant_collapse_detected,
            ));
        self.reference_collapse_recording_count = self
            .reference_collapse_recording_count
            .saturating_add(u64::from(
                score.speaker_occupancy.any_reference_collapse_detected,
            ));
        self.phantom_speaker_count = self.phantom_speaker_count.saturating_add(
            u64::try_from(score.speaker_occupancy.phantom_speaker_count).unwrap_or(u64::MAX),
        );
        self.collapsed_reference_speaker_count =
            self.collapsed_reference_speaker_count.saturating_add(
                u64::try_from(score.speaker_occupancy.collapsed_reference_speaker_count)
                    .unwrap_or(u64::MAX),
            );
        self.effective_speaker_count_sum = self.effective_speaker_count_sum.saturating_add(
            u64::try_from(score.speaker_occupancy.effective_speaker_count).unwrap_or(u64::MAX),
        );
        self.effective_speaker_count_count = self.effective_speaker_count_count.saturating_add(1);
        if let Some(value) = score.speaker_occupancy.dominant_speaker_share {
            self.dominant_speaker_shares.push(value);
        }
        if let Some(value) = score.speaker_occupancy.unknown_speaker_share {
            self.unknown_speaker_shares.push(value);
        }
        if let Some(value) = score.speaker_occupancy.minority_reference_recall {
            self.minority_reference_recall_sum += value;
            self.minority_reference_recall_count =
                self.minority_reference_recall_count.saturating_add(1);
        }
        self.reference_word_count = self
            .reference_word_count
            .saturating_add(score.word_attribution.reference_word_count);
        self.scored_word_count = self
            .scored_word_count
            .saturating_add(score.word_attribution.scored_word_count);
        self.correct_word_count = self
            .correct_word_count
            .saturating_add(score.word_attribution.correct_word_count);
        self.incorrect_word_count = self
            .incorrect_word_count
            .saturating_add(score.word_attribution.incorrect_word_count);
        self.unknown_word_count = self
            .unknown_word_count
            .saturating_add(score.word_attribution.unknown_word_count);
        self.excluded_word_count = self
            .excluded_word_count
            .saturating_add(score.word_attribution.excluded_word_count);
        if let Some(value) = score.word_attribution.word_diarization_error_rate {
            self.macro_word_diarization_error_sum += value;
            self.macro_word_diarization_error_count =
                self.macro_word_diarization_error_count.saturating_add(1);
        }
        self.selective_reference_speaker_time_sec +=
            score.selective_attribution.reference_speaker_time_sec;
        self.selective_covered_speaker_time_sec +=
            score.selective_attribution.covered_speaker_time_sec;
        self.selective_correct_covered_speaker_time_sec +=
            score.selective_attribution.correct_covered_speaker_time_sec;
        self.selective_error_covered_speaker_time_sec +=
            score.selective_attribution.error_covered_speaker_time_sec;
        self.selective_unknown_speaker_time_sec +=
            score.selective_attribution.unknown_speaker_time_sec;
        let assignment_observed = score.calibration.observed_duration_sec;
        let assignment_opportunities = score.calibration.opportunity_duration_sec;
        self.assignment_observed_duration_sec += assignment_observed;
        self.assignment_opportunity_duration_sec += assignment_opportunities;
        if let Some(brier) = score.calibration.brier_score {
            self.assignment_brier_weighted_sum += brier * assignment_observed;
            self.assignment_brier_weight += assignment_observed;
        }
        if let Some(ece) = score.calibration.expected_calibration_error {
            self.assignment_ece_weighted_sum += ece * assignment_observed;
            self.assignment_ece_weight += assignment_observed;
        }
        if clustering.requested_mode == AcousticClusteringMode::ProbabilisticV1 {
            self.speaker_count_stability_sum += f64::from(clustering.speaker_count_stability);
            self.speaker_count_stability_count =
                self.speaker_count_stability_count.saturating_add(1);
        }
        self.clustering_fallback_count = self
            .clustering_fallback_count
            .saturating_add(u64::from(clustering.fallback_reason.is_some()));
        match clustering.fallback_reason {
            Some(AcousticClusteringFallbackReason::InsufficientSharedVoiceDimensions) => {
                self.clustering_insufficient_voice_fallback_count = self
                    .clustering_insufficient_voice_fallback_count
                    .saturating_add(1);
            }
            Some(AcousticClusteringFallbackReason::InvalidPosterior) => {
                self.clustering_invalid_posterior_fallback_count = self
                    .clustering_invalid_posterior_fallback_count
                    .saturating_add(1);
            }
            Some(AcousticClusteringFallbackReason::UnstableSpeakerCount) => {
                self.clustering_unstable_count_fallback_count = self
                    .clustering_unstable_count_fallback_count
                    .saturating_add(1);
            }
            Some(AcousticClusteringFallbackReason::SpeakerCountPriorUnresolved) => {
                return Err(public_corpus_error(
                    "ablation_speaker_count_prior_unresolved",
                    "public-corpus ablation evidence cannot admit an unresolved speaker-count prior",
                ));
            }
            None => {}
        }
        let performance = score.performance.as_ref().ok_or_else(|| {
            public_corpus_error(
                "ablation_performance",
                "public ablation score is missing its performance observation",
            )
        })?;
        self.audio_duration_sec += performance.audio_duration_sec;
        self.wall_time_sec += performance.wall_time_sec;
        self.sampled_peak_rss_bytes = self.sampled_peak_rss_bytes.max(performance.peak_rss_bytes);
        Ok(())
    }

    fn finish(self, split: EvaluationSplit) -> PublicCorpusAblationSplit {
        let precision = ratio(self.change_matched_count, self.change_hypothesis_count);
        let recall = ratio(self.change_matched_count, self.change_reference_count);
        let change_f1 = precision.zip(recall).map(|(precision, recall)| {
            let denominator = precision + recall;
            if denominator > 0.0 {
                2.0 * precision * recall / denominator
            } else {
                0.0
            }
        });
        let diarization_error =
            self.missed_speech_sec + self.false_alarm_sec + self.speaker_confusion_sec;
        let overlap_precision = positive_ratio(
            self.overlap_true_positive_sec,
            self.overlap_true_positive_sec + self.overlap_false_positive_sec,
        );
        let overlap_recall = positive_ratio(
            self.overlap_true_positive_sec,
            self.overlap_true_positive_sec + self.overlap_false_negative_sec,
        );
        let overlap_f1 = positive_ratio(
            2.0 * self.overlap_true_positive_sec,
            2.0 * self.overlap_true_positive_sec
                + self.overlap_false_positive_sec
                + self.overlap_false_negative_sec,
        );
        let change_brier_score = positive_ratio(
            self.change_event_brier_sum,
            self.change_event_observation_count as f64,
        );
        let speaker_count_confusion = self
            .speaker_count_confusion
            .iter()
            .map(
                |(&(reference_speakers, hypothesis_speakers), &recording_count)| {
                    PublicSpeakerCountConfusionCell {
                        reference_speakers,
                        hypothesis_speakers,
                        recording_count,
                    }
                },
            )
            .collect::<Vec<_>>();
        let speaker_count_strata = self
            .speaker_count_strata
            .iter()
            .map(|(&reference_speakers, stratum)| PublicSpeakerCountStratum {
                reference_speakers,
                recording_count: stratum.recording_count,
                posterior_recording_count: stratum.posterior_recording_count,
                unresolved_recording_count: stratum.unresolved_recording_count,
                zero_reference_probability_count: stratum.zero_reference_probability_count,
                exact_speaker_count_rate: ratio(
                    stratum.exact_speaker_count_count,
                    stratum.recording_count,
                ),
                mean_negative_log_likelihood: positive_ratio(
                    stratum.negative_log_likelihood_sum,
                    stratum.negative_log_likelihood_count as f64,
                ),
                mean_brier_score: positive_ratio(stratum.brier_sum, stratum.brier_count as f64),
                top_k_coverage: ratio(stratum.top_k_hit_count, stratum.top_k_observation_count),
                credible_set_coverage: ratio(
                    stratum.credible_set_hit_count,
                    stratum.credible_set_observation_count,
                ),
            })
            .collect::<Vec<_>>();
        let speaker_count_duration_strata = self
            .speaker_count_duration_strata
            .iter()
            .map(
                |(&duration_bucket, stratum)| PublicSpeakerCountDurationStratum {
                    duration_bucket,
                    recording_count: stratum.recording_count,
                    posterior_recording_count: stratum.posterior_recording_count,
                    unresolved_recording_count: stratum.unresolved_recording_count,
                    zero_reference_probability_count: stratum.zero_reference_probability_count,
                    exact_speaker_count_rate: ratio(
                        stratum.exact_speaker_count_count,
                        stratum.recording_count,
                    ),
                    mean_negative_log_likelihood: positive_ratio(
                        stratum.negative_log_likelihood_sum,
                        stratum.negative_log_likelihood_count as f64,
                    ),
                    mean_brier_score: positive_ratio(stratum.brier_sum, stratum.brier_count as f64),
                    top_k_coverage: ratio(stratum.top_k_hit_count, stratum.top_k_observation_count),
                    credible_set_coverage: ratio(
                        stratum.credible_set_hit_count,
                        stratum.credible_set_observation_count,
                    ),
                },
            )
            .collect::<Vec<_>>();
        let maximum_absolute_speaker_count_error =
            self.absolute_speaker_count_errors.iter().copied().max();
        let dominant_share_sum = self.dominant_speaker_shares.iter().sum::<f64>();
        let unknown_share_sum = self.unknown_speaker_shares.iter().sum::<f64>();
        let maximum_dominant_speaker_share = self
            .dominant_speaker_shares
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .map(canonical_evidence_number);
        let maximum_unknown_speaker_share = self
            .unknown_speaker_shares
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .map(canonical_evidence_number);
        let mut change_expected_calibration_error = 0.0;
        let change_reliability = self
            .change_reliability
            .iter()
            .enumerate()
            .map(|(index, bin)| {
                let mean_probability =
                    positive_ratio(bin.probability_sum, bin.observation_count as f64);
                let empirical_frequency = ratio(bin.positive_count, bin.observation_count);
                if let (Some(mean), Some(empirical)) = (mean_probability, empirical_frequency) {
                    change_expected_calibration_error += bin.observation_count as f64
                        / self.change_event_observation_count.max(1) as f64
                        * (mean - empirical).abs();
                }
                PublicChangeReliabilityBin {
                    index,
                    lower_probability: canonical_evidence_number(
                        index as f64 / self.change_reliability.len() as f64,
                    ),
                    upper_probability: canonical_evidence_number(
                        (index + 1) as f64 / self.change_reliability.len() as f64,
                    ),
                    observation_count: bin.observation_count,
                    positive_count: bin.positive_count,
                    mean_probability,
                    empirical_frequency,
                }
            })
            .collect::<Vec<_>>();
        let change_collar_metrics = self
            .change_collar_metrics
            .into_iter()
            .map(|metric| {
                let collar_ms = metric.collar_ms;
                let values = metric.finish_values();
                PublicChangeCollarMetrics {
                    collar_ms,
                    reference_count: values.reference_count,
                    hypothesis_count: values.hypothesis_count,
                    matched_count: values.matched_count,
                    precision: values.precision,
                    recall: values.recall,
                    f1: values.f1,
                    mean_absolute_error_sec: values.mean_absolute_error_sec,
                    p50_absolute_error_sec: values.p50_absolute_error_sec,
                    p90_absolute_error_sec: values.p90_absolute_error_sec,
                    p95_absolute_error_sec: values.p95_absolute_error_sec,
                }
            })
            .collect();
        let change_threshold_sweep = self
            .change_threshold_sweep
            .into_iter()
            .map(|metric| {
                let threshold = canonical_evidence_number(metric.key);
                let collar_ms = metric.collar_ms;
                let values = metric.finish_values();
                PublicChangeThresholdSweepPoint {
                    threshold,
                    collar_ms,
                    reference_count: values.reference_count,
                    hypothesis_count: values.hypothesis_count,
                    matched_count: values.matched_count,
                    precision: values.precision,
                    recall: values.recall,
                    f1: values.f1,
                    mean_absolute_error_sec: values.mean_absolute_error_sec,
                    p50_absolute_error_sec: values.p50_absolute_error_sec,
                    p90_absolute_error_sec: values.p90_absolute_error_sec,
                    p95_absolute_error_sec: values.p95_absolute_error_sec,
                }
            })
            .collect();
        PublicCorpusAblationSplit {
            split,
            recording_count: self.recording_count,
            reference_speaker_time_sec: canonical_evidence_number(self.reference_speaker_time_sec),
            micro_der: positive_ratio(diarization_error, self.reference_speaker_time_sec),
            macro_der: positive_ratio(self.macro_der_sum, self.macro_der_count as f64),
            macro_jer: positive_ratio(self.macro_jer_sum, self.macro_jer_count as f64),
            speaker_confusion_sec: canonical_evidence_number(self.speaker_confusion_sec),
            overlap_reference_sec: canonical_evidence_number(self.overlap_reference_sec),
            overlap_hypothesis_sec: canonical_evidence_number(self.overlap_hypothesis_sec),
            overlap_true_positive_sec: canonical_evidence_number(self.overlap_true_positive_sec),
            overlap_false_positive_sec: canonical_evidence_number(self.overlap_false_positive_sec),
            overlap_false_negative_sec: canonical_evidence_number(self.overlap_false_negative_sec),
            overlap_precision,
            overlap_recall,
            overlap_f1,
            change_reference_count: self.change_reference_count,
            change_hypothesis_count: self.change_hypothesis_count,
            change_matched_count: self.change_matched_count,
            change_precision: precision,
            change_recall: recall,
            change_f1,
            change_mean_absolute_error_sec: positive_ratio(
                self.change_absolute_error_sec,
                self.change_matched_count as f64,
            ),
            change_event_observation_count: self.change_event_observation_count,
            change_event_positive_count: self.change_event_positive_count,
            change_brier_score,
            change_expected_calibration_error: (self.change_event_observation_count > 0)
                .then(|| canonical_evidence_number(change_expected_calibration_error)),
            change_reliability,
            change_collar_metrics,
            change_threshold_sweep,
            exact_speaker_count_rate: ratio(self.exact_speaker_count, self.recording_count),
            mean_signed_speaker_count_error: signed_ratio(
                self.signed_speaker_count_error as f64,
                self.recording_count as f64,
            ),
            mean_absolute_speaker_count_error: positive_ratio(
                self.absolute_speaker_count_error as f64,
                self.recording_count as f64,
            ),
            p50_absolute_speaker_count_error: quantile_nearest_rank_u64(
                &self.absolute_speaker_count_errors,
                500_000,
            ),
            p90_absolute_speaker_count_error: quantile_nearest_rank_u64(
                &self.absolute_speaker_count_errors,
                900_000,
            ),
            p95_absolute_speaker_count_error: quantile_nearest_rank_u64(
                &self.absolute_speaker_count_errors,
                950_000,
            ),
            maximum_absolute_speaker_count_error,
            speaker_count_confusion,
            speaker_count_strata,
            speaker_count_duration_strata,
            count_posterior_recording_count: self.count_posterior_recording_count,
            count_posterior_unavailable_count: self.count_posterior_unavailable_count,
            count_unresolved_recording_count: self.count_unresolved_recording_count,
            count_zero_reference_probability_count: self.count_zero_reference_probability_count,
            count_mean_negative_log_likelihood: positive_ratio(
                self.count_negative_log_likelihood_sum,
                self.count_negative_log_likelihood_count as f64,
            ),
            count_mean_brier_score: positive_ratio(
                self.count_brier_sum,
                self.count_brier_count as f64,
            ),
            count_top_k_coverage: ratio(
                self.count_top_k_hit_count,
                self.count_top_k_observation_count,
            ),
            count_credible_set_coverage: ratio(
                self.count_credible_set_hit_count,
                self.count_credible_set_observation_count,
            ),
            count_mean_entropy_bits: positive_ratio(
                self.count_entropy_sum,
                self.count_entropy_count as f64,
            ),
            dominant_collapse_recording_count: self.dominant_collapse_recording_count,
            reference_collapse_recording_count: self.reference_collapse_recording_count,
            phantom_speaker_count: self.phantom_speaker_count,
            collapsed_reference_speaker_count: self.collapsed_reference_speaker_count,
            mean_effective_speaker_count: positive_ratio(
                self.effective_speaker_count_sum as f64,
                self.effective_speaker_count_count as f64,
            ),
            mean_dominant_speaker_share: positive_ratio(
                dominant_share_sum,
                self.dominant_speaker_shares.len() as f64,
            ),
            p90_dominant_speaker_share: quantile_nearest_rank_f64(
                &self.dominant_speaker_shares,
                900_000,
            ),
            p99_dominant_speaker_share: quantile_nearest_rank_f64(
                &self.dominant_speaker_shares,
                990_000,
            ),
            maximum_dominant_speaker_share,
            mean_unknown_speaker_share: positive_ratio(
                unknown_share_sum,
                self.unknown_speaker_shares.len() as f64,
            ),
            maximum_unknown_speaker_share,
            mean_minority_reference_recall: positive_ratio(
                self.minority_reference_recall_sum,
                self.minority_reference_recall_count as f64,
            ),
            reference_word_count: self.reference_word_count,
            scored_word_count: self.scored_word_count,
            correct_word_count: self.correct_word_count,
            incorrect_word_count: self.incorrect_word_count,
            unknown_word_count: self.unknown_word_count,
            excluded_word_count: self.excluded_word_count,
            micro_word_diarization_error_rate: ratio(
                self.incorrect_word_count
                    .saturating_add(self.unknown_word_count),
                self.scored_word_count,
            ),
            macro_word_diarization_error_rate: positive_ratio(
                self.macro_word_diarization_error_sum,
                self.macro_word_diarization_error_count as f64,
            ),
            selective_reference_speaker_time_sec: canonical_evidence_number(
                self.selective_reference_speaker_time_sec,
            ),
            selective_covered_speaker_time_sec: canonical_evidence_number(
                self.selective_covered_speaker_time_sec,
            ),
            selective_correct_covered_speaker_time_sec: canonical_evidence_number(
                self.selective_correct_covered_speaker_time_sec,
            ),
            selective_error_covered_speaker_time_sec: canonical_evidence_number(
                self.selective_error_covered_speaker_time_sec,
            ),
            selective_unknown_speaker_time_sec: canonical_evidence_number(
                self.selective_unknown_speaker_time_sec,
            ),
            selective_coverage: positive_ratio(
                self.selective_covered_speaker_time_sec,
                self.selective_reference_speaker_time_sec,
            ),
            selective_risk: positive_ratio(
                self.selective_error_covered_speaker_time_sec,
                self.selective_covered_speaker_time_sec,
            ),
            assignment_observed_duration_sec: canonical_evidence_number(
                self.assignment_observed_duration_sec,
            ),
            assignment_opportunity_duration_sec: canonical_evidence_number(
                self.assignment_opportunity_duration_sec,
            ),
            assignment_coverage: positive_ratio(
                self.assignment_observed_duration_sec,
                self.assignment_opportunity_duration_sec,
            ),
            assignment_brier_score: positive_ratio(
                self.assignment_brier_weighted_sum,
                self.assignment_brier_weight,
            ),
            assignment_expected_calibration_error: positive_ratio(
                self.assignment_ece_weighted_sum,
                self.assignment_ece_weight,
            ),
            mean_speaker_count_stability: positive_ratio(
                self.speaker_count_stability_sum,
                self.speaker_count_stability_count as f64,
            ),
            clustering_fallback_count: self.clustering_fallback_count,
            clustering_insufficient_voice_fallback_count: self
                .clustering_insufficient_voice_fallback_count,
            clustering_invalid_posterior_fallback_count: self
                .clustering_invalid_posterior_fallback_count,
            clustering_unstable_count_fallback_count: self.clustering_unstable_count_fallback_count,
            audio_duration_sec: canonical_evidence_number(self.audio_duration_sec),
            wall_time_sec: canonical_evidence_number(self.wall_time_sec),
            real_time_factor: positive_ratio(self.wall_time_sec, self.audio_duration_sec),
            sampled_peak_rss_bytes: self.sampled_peak_rss_bytes,
        }
    }
}

fn clipped_reference(
    reference: &DiarizationReferenceDocument,
    maximum_recording_duration_ms: Option<u64>,
) -> FwResult<DiarizationReferenceDocument> {
    let duration_ms = maximum_recording_duration_ms.map_or(reference.duration_ms, |maximum| {
        reference.duration_ms.min(maximum)
    });
    let clip_turn = |turn: &EvaluationTurn| {
        (turn.start_ms < duration_ms).then(|| {
            let mut clipped = turn.clone();
            clipped.end_ms = clipped.end_ms.min(duration_ms);
            clipped
        })
    };
    let clip_region = |region: &EvaluationRegion| {
        (region.start_ms < duration_ms).then(|| {
            let mut clipped = region.clone();
            clipped.end_ms = clipped.end_ms.min(duration_ms);
            clipped
        })
    };
    let mut clipped = reference.clone();
    clipped.duration_ms = duration_ms;
    clipped.turns = reference
        .turns
        .iter()
        .filter_map(clip_turn)
        .filter(|turn| turn.end_ms > turn.start_ms)
        .collect();
    clipped.ignored_regions = reference
        .ignored_regions
        .iter()
        .filter_map(clip_region)
        .filter(|region| region.end_ms > region.start_ms)
        .collect();
    clipped.speaker_hints = reference
        .speaker_hints
        .iter()
        .filter(|hint| hint.start_ms < duration_ms)
        .cloned()
        .map(|mut hint| {
            hint.end_ms = hint.end_ms.min(duration_ms);
            hint
        })
        .filter(|hint| hint.end_ms > hint.start_ms)
        .collect();
    clipped.words = reference
        .words
        .iter()
        .filter(|word| word.start_ms < duration_ms)
        .cloned()
        .map(|mut word| {
            word.end_ms = word.end_ms.min(duration_ms);
            word
        })
        .filter(|word| word.end_ms > word.start_ms)
        .collect();
    parse_diarization_reference(&serde_json::to_vec(&clipped)?)?;
    Ok(clipped)
}

fn merged_scored_speech_regions(
    turns: &[EvaluationTurn],
    ignored_regions: &[EvaluationRegion],
) -> Vec<(u64, u64)> {
    let mut regions = turns
        .iter()
        .map(|turn| (turn.start_ms, turn.end_ms))
        .collect::<Vec<_>>();
    regions.sort_unstable();
    let mut merged = Vec::<(u64, u64)>::new();
    for (start_ms, end_ms) in regions {
        if let Some(previous) = merged.last_mut()
            && start_ms <= previous.1
        {
            previous.1 = previous.1.max(end_ms);
        } else {
            merged.push((start_ms, end_ms));
        }
    }
    let mut scored = Vec::new();
    let mut ignored_index = 0usize;
    for (start_ms, end_ms) in merged {
        while ignored_index < ignored_regions.len()
            && ignored_regions[ignored_index].end_ms <= start_ms
        {
            ignored_index += 1;
        }
        let mut cursor = start_ms;
        let mut candidate_index = ignored_index;
        while candidate_index < ignored_regions.len()
            && ignored_regions[candidate_index].start_ms < end_ms
        {
            let ignored = &ignored_regions[candidate_index];
            if ignored.start_ms > cursor {
                scored.push((cursor, ignored.start_ms.min(end_ms)));
            }
            cursor = cursor.max(ignored.end_ms);
            if cursor >= end_ms {
                break;
            }
            candidate_index += 1;
        }
        if cursor < end_ms {
            scored.push((cursor, end_ms));
        }
    }
    scored
}

fn hash_pcm_prefix(samples: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"franken-whisper-public-ablation-pcm-v1\0");
    hasher.update((samples.len() as u64).to_le_bytes());
    for sample in samples {
        hasher.update(sample.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn sampled_process_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for field in ["VmHWM:", "VmRSS:"] {
                if let Some(kibibytes) = status.lines().find_map(|line| {
                    line.strip_prefix(field)?
                        .split_ascii_whitespace()
                        .next()?
                        .parse::<u64>()
                        .ok()
                }) {
                    return kibibytes.saturating_mul(1_024);
                }
            }
        }
    }
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output();
    output
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|text| text.trim().parse::<u64>().ok())
        .unwrap_or(0)
        .saturating_mul(1_024)
}

fn change_selection_policy_fingerprint() -> PublicChangeSelectionPolicyFingerprint {
    PublicChangeSelectionPolicyFingerprint {
        policy_id: PUBLIC_CORPUS_CHANGE_SELECTION_POLICY_VERSION,
        candidate_order: PUBLIC_CORPUS_CHANGE_CANDIDATE_ORDER,
        baseline: AcousticChangeDetectorMode::FixedSafeV1,
        minimum_relative_change_f1_improvement: canonical_evidence_number(
            PUBLIC_CORPUS_MIN_CHANGE_F1_IMPROVEMENT,
        ),
        maximum_der_jer_regression: canonical_evidence_number(
            PUBLIC_CORPUS_MAX_CHANGE_DER_JER_REGRESSION,
        ),
        maximum_brier_score: canonical_evidence_number(PUBLIC_CORPUS_MAX_CHANGE_BRIER),
        maximum_expected_calibration_error: canonical_evidence_number(PUBLIC_CORPUS_MAX_CHANGE_ECE),
        require_timing_non_regression: true,
        fail_closed_default: AcousticChangeDetectorMode::FixedSafeV1,
    }
}

fn change_selection_policy_sha256() -> FwResult<String> {
    canonical_sha256(&change_selection_policy_fingerprint())
}

fn development_improvement_gate(
    variants: &[PublicCorpusAblationVariant],
) -> FwResult<PublicCorpusDevelopmentGate> {
    let find = |ablation| {
        variants
            .iter()
            .find(|variant| variant.ablation == ablation)
            .and_then(|variant| {
                variant
                    .splits
                    .iter()
                    .find(|split| split.split == EvaluationSplit::Development)
            })
    };
    let candidate = find(AcousticFeatureAblation::FullV2).ok_or_else(|| {
        public_corpus_error(
            "ablation_development",
            "full v2 evidence is missing the frozen development split",
        )
    })?;
    let baseline = find(AcousticFeatureAblation::V1).ok_or_else(|| {
        public_corpus_error(
            "ablation_development",
            "v1 evidence is missing the frozen development split",
        )
    })?;
    let relative_micro_der_improvement =
        candidate
            .micro_der
            .zip(baseline.micro_der)
            .and_then(|(candidate, baseline)| {
                (baseline > 0.0)
                    .then(|| canonical_evidence_number((baseline - candidate) / baseline))
            });
    let macro_jer_delta = candidate
        .macro_jer
        .zip(baseline.macro_jer)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let change_f1_delta = candidate
        .change_f1
        .zip(baseline.change_f1)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let passed = relative_micro_der_improvement
        .is_some_and(|improvement| improvement >= PUBLIC_CORPUS_MIN_DEVELOPMENT_DER_IMPROVEMENT)
        && macro_jer_delta.is_some_and(|delta| delta <= 0.0)
        && change_f1_delta.is_some_and(|delta| delta >= 0.0);
    Ok(PublicCorpusDevelopmentGate {
        split: EvaluationSplit::Development,
        candidate: AcousticFeatureAblation::FullV2,
        baseline: AcousticFeatureAblation::V1,
        minimum_relative_micro_der_improvement: canonical_evidence_number(
            PUBLIC_CORPUS_MIN_DEVELOPMENT_DER_IMPROVEMENT,
        ),
        relative_micro_der_improvement,
        macro_jer_delta,
        change_f1_delta,
        passed,
    })
}

fn held_out_non_regression_gate(
    variants: &[PublicCorpusAblationVariant],
) -> FwResult<PublicCorpusHeldOutGate> {
    let find = |ablation| {
        variants
            .iter()
            .find(|variant| variant.ablation == ablation)
            .and_then(|variant| {
                variant
                    .splits
                    .iter()
                    .find(|split| split.split == EvaluationSplit::Test)
            })
    };
    let candidate = find(AcousticFeatureAblation::FullV2).ok_or_else(|| {
        public_corpus_error(
            "ablation_held_out",
            "full v2 evidence is missing the frozen test split",
        )
    })?;
    let baseline = find(AcousticFeatureAblation::V1).ok_or_else(|| {
        public_corpus_error(
            "ablation_held_out",
            "v1 evidence is missing the frozen test split",
        )
    })?;
    let micro_der_delta = candidate
        .micro_der
        .zip(baseline.micro_der)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let macro_jer_delta = candidate
        .macro_jer
        .zip(baseline.macro_jer)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let passed = micro_der_delta.is_some_and(|delta| delta <= 0.0)
        && macro_jer_delta.is_some_and(|delta| delta <= 0.0);
    Ok(PublicCorpusHeldOutGate {
        split: EvaluationSplit::Test,
        candidate: AcousticFeatureAblation::FullV2,
        baseline: AcousticFeatureAblation::V1,
        micro_der_delta,
        macro_jer_delta,
        passed,
    })
}

fn change_detector_split(
    variants: &[PublicCorpusChangeDetectorVariant],
    detector_mode: AcousticChangeDetectorMode,
    split: EvaluationSplit,
) -> Option<&PublicCorpusAblationSplit> {
    variants
        .iter()
        .find(|variant| variant.detector_mode == detector_mode)
        .and_then(|variant| {
            variant
                .splits
                .iter()
                .find(|candidate| candidate.split == split)
        })
}

fn change_timing_requirement_passed(candidate: Option<f64>, baseline: Option<f64>) -> bool {
    match (candidate, baseline) {
        (Some(candidate), Some(baseline)) => candidate <= baseline,
        (Some(_), None) => true,
        (None, Some(_)) | (None, None) => false,
    }
}

fn change_candidate_development_gate(
    variants: &[PublicCorpusChangeDetectorVariant],
    candidate_mode: AcousticChangeDetectorMode,
) -> FwResult<PublicCorpusChangeDevelopmentGate> {
    let candidate = change_detector_split(variants, candidate_mode, EvaluationSplit::Development)
        .ok_or_else(|| {
        public_corpus_error(
            "change_development",
            "candidate detector evidence is missing the development split",
        )
    })?;
    let baseline = change_detector_split(
        variants,
        AcousticChangeDetectorMode::FixedSafeV1,
        EvaluationSplit::Development,
    )
    .ok_or_else(|| {
        public_corpus_error(
            "change_development",
            "fixed-safe detector evidence is missing the development split",
        )
    })?;
    let relative_change_f1_improvement =
        candidate
            .change_f1
            .zip(baseline.change_f1)
            .and_then(|(candidate, baseline)| {
                if baseline > 0.0 {
                    Some(canonical_evidence_number((candidate - baseline) / baseline))
                } else if candidate > 0.0 {
                    Some(1.0)
                } else {
                    None
                }
            });
    let mean_absolute_timing_error_delta_sec = candidate
        .change_mean_absolute_error_sec
        .zip(baseline.change_mean_absolute_error_sec)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let micro_der_delta = candidate
        .micro_der
        .zip(baseline.micro_der)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let macro_jer_delta = candidate
        .macro_jer
        .zip(baseline.macro_jer)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let candidate_brier_score = candidate.change_brier_score;
    let candidate_expected_calibration_error = candidate.change_expected_calibration_error;
    let passed = relative_change_f1_improvement
        .is_some_and(|gain| gain >= PUBLIC_CORPUS_MIN_CHANGE_F1_IMPROVEMENT)
        && change_timing_requirement_passed(
            candidate.change_mean_absolute_error_sec,
            baseline.change_mean_absolute_error_sec,
        )
        && micro_der_delta
            .is_some_and(|delta| delta <= PUBLIC_CORPUS_MAX_CHANGE_DER_JER_REGRESSION)
        && macro_jer_delta
            .is_some_and(|delta| delta <= PUBLIC_CORPUS_MAX_CHANGE_DER_JER_REGRESSION)
        && candidate_brier_score.is_some_and(|score| score <= PUBLIC_CORPUS_MAX_CHANGE_BRIER)
        && candidate_expected_calibration_error
            .is_some_and(|error| error <= PUBLIC_CORPUS_MAX_CHANGE_ECE);
    Ok(PublicCorpusChangeDevelopmentGate {
        split: EvaluationSplit::Development,
        candidate: candidate_mode,
        baseline: AcousticChangeDetectorMode::FixedSafeV1,
        minimum_relative_change_f1_improvement: canonical_evidence_number(
            PUBLIC_CORPUS_MIN_CHANGE_F1_IMPROVEMENT,
        ),
        maximum_der_jer_regression: canonical_evidence_number(
            PUBLIC_CORPUS_MAX_CHANGE_DER_JER_REGRESSION,
        ),
        maximum_brier_score: canonical_evidence_number(PUBLIC_CORPUS_MAX_CHANGE_BRIER),
        maximum_expected_calibration_error: canonical_evidence_number(PUBLIC_CORPUS_MAX_CHANGE_ECE),
        relative_change_f1_improvement,
        mean_absolute_timing_error_delta_sec,
        micro_der_delta,
        macro_jer_delta,
        candidate_brier_score,
        candidate_expected_calibration_error,
        passed,
    })
}

fn change_development_gate(
    variants: &[PublicCorpusChangeDetectorVariant],
) -> FwResult<PublicCorpusChangeDevelopmentGate> {
    let fail_closed = change_candidate_development_gate(
        variants,
        AcousticChangeDetectorMode::CalibratedPosterior,
    )?;
    let mut selected: Option<PublicCorpusChangeDevelopmentGate> = None;
    for candidate_mode in PUBLIC_CORPUS_CHANGE_CANDIDATE_ORDER {
        let candidate_gate = change_candidate_development_gate(variants, candidate_mode)?;
        if !candidate_gate.passed {
            continue;
        }
        let candidate_split =
            change_detector_split(variants, candidate_mode, EvaluationSplit::Development)
                .ok_or_else(|| {
                    public_corpus_error(
                        "change_development",
                        "candidate detector evidence disappeared during selection",
                    )
                })?;
        let replace = selected.as_ref().is_none_or(|incumbent| {
            let Some(incumbent_split) =
                change_detector_split(variants, incumbent.candidate, EvaluationSplit::Development)
            else {
                return true;
            };
            let descending = |candidate: Option<f64>, incumbent: Option<f64>| {
                candidate
                    .unwrap_or(f64::NEG_INFINITY)
                    .total_cmp(&incumbent.unwrap_or(f64::NEG_INFINITY))
            };
            let ascending = |candidate: Option<f64>, incumbent: Option<f64>| {
                incumbent
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&candidate.unwrap_or(f64::INFINITY))
            };
            descending(candidate_split.change_f1, incumbent_split.change_f1)
                .then_with(|| {
                    ascending(
                        candidate_split.change_mean_absolute_error_sec,
                        incumbent_split.change_mean_absolute_error_sec,
                    )
                })
                .then_with(|| {
                    ascending(
                        candidate_split.change_brier_score,
                        incumbent_split.change_brier_score,
                    )
                })
                .then_with(|| {
                    ascending(
                        candidate_split.change_expected_calibration_error,
                        incumbent_split.change_expected_calibration_error,
                    )
                })
                .then_with(|| ascending(candidate_split.micro_der, incumbent_split.micro_der))
                .then_with(|| ascending(candidate_split.macro_jer, incumbent_split.macro_jer))
                .is_gt()
        });
        if replace {
            selected = Some(candidate_gate);
        }
    }
    Ok(selected.unwrap_or(fail_closed))
}

fn change_held_out_gate(
    variants: &[PublicCorpusChangeDetectorVariant],
    selected_mode: AcousticChangeDetectorMode,
) -> FwResult<PublicCorpusChangeHeldOutGate> {
    if selected_mode == AcousticChangeDetectorMode::FixedSafeV1 {
        return Err(public_corpus_error(
            "change_held_out",
            "the fixed-safe baseline cannot be promoted as its own candidate",
        ));
    }
    let candidate = change_detector_split(variants, selected_mode, EvaluationSplit::Test)
        .ok_or_else(|| {
            public_corpus_error(
                "change_held_out",
                "selected detector evidence is missing the held-out split",
            )
        })?;
    let baseline = change_detector_split(
        variants,
        AcousticChangeDetectorMode::FixedSafeV1,
        EvaluationSplit::Test,
    )
    .ok_or_else(|| {
        public_corpus_error(
            "change_held_out",
            "fixed-safe detector evidence is missing the held-out split",
        )
    })?;
    let change_f1_delta = candidate
        .change_f1
        .zip(baseline.change_f1)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let timing_error_delta_sec = candidate
        .change_mean_absolute_error_sec
        .zip(baseline.change_mean_absolute_error_sec)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let micro_der_delta = candidate
        .micro_der
        .zip(baseline.micro_der)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let macro_jer_delta = candidate
        .macro_jer
        .zip(baseline.macro_jer)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let candidate_brier_score = candidate.change_brier_score;
    let candidate_expected_calibration_error = candidate.change_expected_calibration_error;
    let passed = change_f1_delta.is_some_and(|delta| delta >= 0.0)
        && change_timing_requirement_passed(
            candidate.change_mean_absolute_error_sec,
            baseline.change_mean_absolute_error_sec,
        )
        && micro_der_delta
            .is_some_and(|delta| delta <= PUBLIC_CORPUS_MAX_CHANGE_DER_JER_REGRESSION)
        && macro_jer_delta
            .is_some_and(|delta| delta <= PUBLIC_CORPUS_MAX_CHANGE_DER_JER_REGRESSION)
        && candidate_brier_score.is_some_and(|score| score <= PUBLIC_CORPUS_MAX_CHANGE_BRIER)
        && candidate_expected_calibration_error
            .is_some_and(|error| error <= PUBLIC_CORPUS_MAX_CHANGE_ECE);
    Ok(PublicCorpusChangeHeldOutGate {
        split: EvaluationSplit::Test,
        candidate: selected_mode,
        baseline: AcousticChangeDetectorMode::FixedSafeV1,
        change_f1_delta,
        timing_error_delta_sec,
        micro_der_delta,
        macro_jer_delta,
        candidate_brier_score,
        candidate_expected_calibration_error,
        passed,
    })
}

fn clustering_split(
    variants: &[PublicCorpusClusteringVariant],
    clustering_mode: AcousticClusteringMode,
    split: EvaluationSplit,
) -> Option<&PublicCorpusAblationSplit> {
    variants
        .iter()
        .find(|variant| variant.clustering_mode == clustering_mode)
        .and_then(|variant| {
            variant
                .splits
                .iter()
                .find(|candidate| candidate.split == split)
        })
}

fn clustering_development_gate(
    variants: &[PublicCorpusClusteringVariant],
) -> FwResult<PublicCorpusClusteringDevelopmentGate> {
    let candidate = clustering_split(
        variants,
        AcousticClusteringMode::ProbabilisticV1,
        EvaluationSplit::Development,
    )
    .ok_or_else(|| {
        public_corpus_error(
            "clustering_development",
            "probabilistic clustering evidence is missing the development split",
        )
    })?;
    let baseline = clustering_split(
        variants,
        AcousticClusteringMode::FixedSafeV1,
        EvaluationSplit::Development,
    )
    .ok_or_else(|| {
        public_corpus_error(
            "clustering_development",
            "fixed-safe clustering evidence is missing the development split",
        )
    })?;
    let relative_micro_der_improvement =
        candidate
            .micro_der
            .zip(baseline.micro_der)
            .and_then(|(candidate, baseline)| {
                (baseline > 0.0)
                    .then(|| canonical_evidence_number((baseline - candidate) / baseline))
            });
    let macro_jer_delta = candidate
        .macro_jer
        .zip(baseline.macro_jer)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let speaker_confusion_delta_sec = Some(canonical_evidence_number(
        candidate.speaker_confusion_sec - baseline.speaker_confusion_sec,
    ));
    let overlap_f1_delta = candidate
        .overlap_f1
        .zip(baseline.overlap_f1)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let mean_absolute_speaker_count_error_delta = candidate
        .mean_absolute_speaker_count_error
        .zip(baseline.mean_absolute_speaker_count_error)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let selective_coverage_regression = candidate
        .selective_coverage
        .zip(baseline.selective_coverage)
        .map(|(candidate, baseline)| canonical_evidence_number(baseline - candidate));
    let selective_risk_delta = candidate
        .selective_risk
        .zip(baseline.selective_risk)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let candidate_assignment_expected_calibration_error =
        candidate.assignment_expected_calibration_error;
    let candidate_mean_speaker_count_stability = candidate.mean_speaker_count_stability;
    let candidate_fallback_count = candidate.clustering_fallback_count;
    let passed = relative_micro_der_improvement
        .is_some_and(|gain| gain >= PUBLIC_CORPUS_MIN_CLUSTERING_DER_IMPROVEMENT)
        && macro_jer_delta
            .is_some_and(|delta| delta <= PUBLIC_CORPUS_MAX_CLUSTERING_JER_REGRESSION)
        && speaker_confusion_delta_sec.is_some_and(|delta| delta <= 0.0)
        && overlap_f1_delta.is_some_and(|delta| delta >= 0.0)
        && mean_absolute_speaker_count_error_delta.is_some_and(|delta| delta <= 0.0)
        && selective_coverage_regression
            .is_some_and(|delta| delta <= PUBLIC_CORPUS_MAX_CLUSTERING_COVERAGE_REGRESSION)
        && selective_risk_delta
            .is_some_and(|delta| delta <= PUBLIC_CORPUS_MAX_CLUSTERING_SELECTIVE_RISK_REGRESSION)
        && candidate_assignment_expected_calibration_error
            .is_some_and(|error| error <= PUBLIC_CORPUS_MAX_CLUSTERING_ECE)
        && candidate_mean_speaker_count_stability
            .is_some_and(|stability| stability >= PUBLIC_CORPUS_MIN_CLUSTERING_COUNT_STABILITY)
        && candidate_fallback_count == 0;
    Ok(PublicCorpusClusteringDevelopmentGate {
        split: EvaluationSplit::Development,
        candidate: AcousticClusteringMode::ProbabilisticV1,
        baseline: AcousticClusteringMode::FixedSafeV1,
        minimum_relative_micro_der_improvement: canonical_evidence_number(
            PUBLIC_CORPUS_MIN_CLUSTERING_DER_IMPROVEMENT,
        ),
        maximum_macro_jer_regression: canonical_evidence_number(
            PUBLIC_CORPUS_MAX_CLUSTERING_JER_REGRESSION,
        ),
        maximum_assignment_expected_calibration_error: canonical_evidence_number(
            PUBLIC_CORPUS_MAX_CLUSTERING_ECE,
        ),
        minimum_mean_speaker_count_stability: canonical_evidence_number(
            PUBLIC_CORPUS_MIN_CLUSTERING_COUNT_STABILITY,
        ),
        maximum_selective_coverage_regression: canonical_evidence_number(
            PUBLIC_CORPUS_MAX_CLUSTERING_COVERAGE_REGRESSION,
        ),
        maximum_selective_risk_regression: canonical_evidence_number(
            PUBLIC_CORPUS_MAX_CLUSTERING_SELECTIVE_RISK_REGRESSION,
        ),
        relative_micro_der_improvement,
        macro_jer_delta,
        speaker_confusion_delta_sec,
        overlap_f1_delta,
        mean_absolute_speaker_count_error_delta,
        selective_coverage_regression,
        selective_risk_delta,
        candidate_assignment_expected_calibration_error,
        candidate_mean_speaker_count_stability,
        candidate_fallback_count,
        passed,
    })
}

fn clustering_held_out_gate(
    variants: &[PublicCorpusClusteringVariant],
    selected_mode: AcousticClusteringMode,
) -> FwResult<PublicCorpusClusteringHeldOutGate> {
    if selected_mode == AcousticClusteringMode::FixedSafeV1 {
        return Err(public_corpus_error(
            "clustering_held_out",
            "the fixed-safe clustering baseline cannot be promoted as its own candidate",
        ));
    }
    let candidate =
        clustering_split(variants, selected_mode, EvaluationSplit::Test).ok_or_else(|| {
            public_corpus_error(
                "clustering_held_out",
                "selected clustering evidence is missing the held-out split",
            )
        })?;
    let baseline = clustering_split(
        variants,
        AcousticClusteringMode::FixedSafeV1,
        EvaluationSplit::Test,
    )
    .ok_or_else(|| {
        public_corpus_error(
            "clustering_held_out",
            "fixed-safe clustering evidence is missing the held-out split",
        )
    })?;
    let micro_der_delta = candidate
        .micro_der
        .zip(baseline.micro_der)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let macro_jer_delta = candidate
        .macro_jer
        .zip(baseline.macro_jer)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let overlap_f1_delta = candidate
        .overlap_f1
        .zip(baseline.overlap_f1)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let mean_absolute_speaker_count_error_delta = candidate
        .mean_absolute_speaker_count_error
        .zip(baseline.mean_absolute_speaker_count_error)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let selective_coverage_regression = candidate
        .selective_coverage
        .zip(baseline.selective_coverage)
        .map(|(candidate, baseline)| canonical_evidence_number(baseline - candidate));
    let selective_risk_delta = candidate
        .selective_risk
        .zip(baseline.selective_risk)
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let candidate_assignment_expected_calibration_error =
        candidate.assignment_expected_calibration_error;
    let candidate_fallback_count = candidate.clustering_fallback_count;
    let passed = micro_der_delta.is_some_and(|delta| delta <= 0.0)
        && macro_jer_delta.is_some_and(|delta| delta <= 0.0)
        && overlap_f1_delta.is_some_and(|delta| delta >= 0.0)
        && mean_absolute_speaker_count_error_delta.is_some_and(|delta| delta <= 0.0)
        && selective_coverage_regression
            .is_some_and(|delta| delta <= PUBLIC_CORPUS_MAX_CLUSTERING_COVERAGE_REGRESSION)
        && selective_risk_delta
            .is_some_and(|delta| delta <= PUBLIC_CORPUS_MAX_CLUSTERING_SELECTIVE_RISK_REGRESSION)
        && candidate_assignment_expected_calibration_error
            .is_some_and(|error| error <= PUBLIC_CORPUS_MAX_CLUSTERING_ECE)
        && candidate_fallback_count == 0;
    Ok(PublicCorpusClusteringHeldOutGate {
        split: EvaluationSplit::Test,
        candidate: selected_mode,
        baseline: AcousticClusteringMode::FixedSafeV1,
        micro_der_delta,
        macro_jer_delta,
        overlap_f1_delta,
        mean_absolute_speaker_count_error_delta,
        selective_coverage_regression,
        selective_risk_delta,
        candidate_assignment_expected_calibration_error,
        candidate_fallback_count,
        passed,
    })
}

/// Verify every frozen identity, hash, aggregate bound, and held-out decision.
pub fn verify_public_corpus_ablation_evidence(
    evidence: &PublicCorpusAblationEvidence,
) -> FwResult<()> {
    if evidence.schema_version != PUBLIC_CORPUS_ABLATION_SCHEMA_VERSION
        || evidence.runner_version != PUBLIC_CORPUS_ABLATION_RUNNER_VERSION
        || evidence.scorer_version != DIARIZATION_SCORER_VERSION
    {
        return Err(public_corpus_error(
            "ablation_version",
            "ablation schema, runner, or scorer version is unsupported",
        ));
    }
    validate_public_id(&evidence.corpus_key, "corpus_key")?;
    validate_public_id(&evidence.source_version, "source_version")?;
    if !public_corpus_registry()
        .entries
        .iter()
        .any(|entry| entry.corpus_key == evidence.corpus_key)
    {
        return Err(public_corpus_error(
            "ablation_corpus_key",
            "ablation corpus key is not in the frozen public registry",
        ));
    }
    for (field, value) in [
        ("bundle_sha256", &evidence.bundle_sha256),
        ("descriptor_sha256", &evidence.descriptor_sha256),
        ("scorer_config_sha256", &evidence.scorer_config_sha256),
        (
            "diarization_request_sha256",
            &evidence.protocol.diarization_request_sha256,
        ),
        (
            "change_calibration_sha256",
            &evidence.protocol.change_calibration_sha256,
        ),
        (
            "change_selection_policy_sha256",
            &evidence.protocol.change_selection_policy_sha256,
        ),
        (
            "speaker_pair_calibration_sha256",
            &evidence.protocol.speaker_pair_calibration_sha256,
        ),
        (
            "deterministic_accuracy_sha256",
            &evidence.deterministic_accuracy_sha256,
        ),
        ("result_sha256", &evidence.result_sha256),
    ] {
        if !is_sha256_hex(value) {
            return Err(public_corpus_error(
                "ablation_hash_format",
                &format!("{field} must be 64 lowercase hexadecimal characters"),
            ));
        }
    }
    for (field, value) in [
        (
            "locked_development_result_sha256",
            evidence.locked_development_result_sha256.as_deref(),
        ),
        (
            "locked_development_accuracy_sha256",
            evidence.locked_development_accuracy_sha256.as_deref(),
        ),
    ] {
        if value.is_some_and(|value| !is_sha256_hex(value)) {
            return Err(public_corpus_error(
                "ablation_hash_format",
                &format!("{field} must be absent or 64 lowercase hexadecimal characters"),
            ));
        }
    }
    let stage_contract_valid = match evidence.evaluation_stage {
        PublicCorpusEvaluationStage::Development => {
            evidence.locked_development_result_sha256.is_none()
                && evidence.locked_development_accuracy_sha256.is_none()
                && evidence.development_gate.is_some()
                && evidence.held_out_gate.is_none()
                && evidence
                    .change_development_gate
                    .as_ref()
                    .is_some_and(|gate| {
                        evidence.selected_change_detector_mode
                            == if gate.passed {
                                gate.candidate
                            } else {
                                gate.baseline
                            }
                    })
                && evidence.change_held_out_gate.is_none()
                && evidence
                    .clustering_development_gate
                    .as_ref()
                    .is_some_and(|gate| {
                        evidence.selected_clustering_mode
                            == if gate.passed {
                                gate.candidate
                            } else {
                                gate.baseline
                            }
                    })
                && evidence.clustering_held_out_gate.is_none()
        }
        PublicCorpusEvaluationStage::Certification => {
            evidence.locked_development_result_sha256.is_some()
                && evidence.locked_development_accuracy_sha256.is_some()
                && evidence.development_gate.is_none()
                && evidence.held_out_gate.is_some()
                && evidence.change_development_gate.is_none()
                && evidence
                    .change_held_out_gate
                    .as_ref()
                    .is_some_and(|gate| gate.candidate == evidence.selected_change_detector_mode)
                && evidence.clustering_development_gate.is_none()
                && evidence
                    .clustering_held_out_gate
                    .as_ref()
                    .is_some_and(|gate| gate.candidate == evidence.selected_clustering_mode)
        }
    };
    if !stage_contract_valid {
        return Err(public_corpus_error(
            "ablation_stage_contract",
            "evaluation stage, development lock, and gate authority are inconsistent",
        ));
    }
    if canonical_sha256(&evidence.scorer_config)? != evidence.scorer_config_sha256 {
        return Err(public_corpus_error(
            "ablation_scorer_hash",
            "scorer configuration hash does not match its canonical content",
        ));
    }
    let expected_scorer_config = DiarizationScorerConfig {
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
    };
    if evidence.scorer_config != expected_scorer_config {
        return Err(public_corpus_error(
            "ablation_scorer_contract",
            "scorer configuration differs from the frozen public ablation protocol",
        ));
    }
    let expected_request = DiarizationRequest {
        engine: DiarizationEngine::Acoustic,
        speaker_count: SpeakerCountRequest::Infer,
        ..DiarizationRequest::default()
    };
    if !evidence.protocol.oracle_vad
        || evidence.protocol.oracle_speaker_count
        || evidence.protocol.maximum_recording_duration_ms == Some(0)
        || evidence.protocol.prefix_selection != "deterministic-prefix-v1"
        || evidence.protocol.rss_observation != "linux-vmhwm-otherwise-sampled-process-rss-v1"
        || evidence.protocol.diarization_request != expected_request
        || canonical_sha256(&evidence.protocol.diarization_request)?
            != evidence.protocol.diarization_request_sha256
        || evidence.protocol.change_calibration_id != ACOUSTIC_CHANGE_CALIBRATION_VERSION
        || evidence.protocol.change_calibration_fit_id != ACOUSTIC_CHANGE_CALIBRATION_FIT_VERSION
        || evidence.protocol.change_calibration_sha256 != acoustic_change_calibration_sha256()
        || evidence.protocol.change_decision_probability
            != canonical_evidence_number(f64::from(
                acoustic_change_calibration().decision_probability,
            ))
        || evidence.protocol.change_calibration_bins != evidence.scorer_config.calibration_bins
        || evidence.protocol.change_selection_policy_id
            != PUBLIC_CORPUS_CHANGE_SELECTION_POLICY_VERSION
        || evidence.protocol.change_selection_policy_sha256 != change_selection_policy_sha256()?
        || evidence.protocol.speaker_pair_calibration_id
            != ACOUSTIC_CLUSTERING_PROBABILISTIC_VERSION
        || evidence.protocol.speaker_pair_calibration_sha256
            != acoustic_speaker_pair_calibration_sha256()
    {
        return Err(public_corpus_error(
            "ablation_protocol",
            "ablation protocol differs from the frozen oracle-VAD request contract",
        ));
    }
    if evidence.variants.len() != AcousticFeatureAblation::ALL.len() {
        return Err(public_corpus_error(
            "ablation_variants",
            "ablation evidence must contain every frozen feature variant exactly once",
        ));
    }
    for (variant, expected_ablation) in evidence.variants.iter().zip(AcousticFeatureAblation::ALL) {
        let expected_schema_sha256 =
            acoustic_feature_schema_sha256(expected_ablation.schema_version());
        let expected_configuration_sha256 =
            canonical_sha256(&PublicFeatureConfigurationFingerprint {
                runner_version: PUBLIC_CORPUS_ABLATION_RUNNER_VERSION,
                ablation: expected_ablation,
                feature_schema_sha256: &expected_schema_sha256,
                diarization_request_sha256: &evidence.protocol.diarization_request_sha256,
                change_calibration_sha256: &evidence.protocol.change_calibration_sha256,
            })?;
        if variant.ablation != expected_ablation
            || variant.feature_schema != expected_ablation.schema_version().id()
            || variant.feature_schema_sha256 != expected_schema_sha256
            || variant.feature_configuration_sha256 != expected_configuration_sha256
            || !variant_splits_are_valid(&variant.splits, evidence.protocol.change_calibration_bins)
            || variant.splits.len() != 1
            || variant.splits[0].split != evidence.evaluation_stage.selected_split()
        {
            return Err(public_corpus_error(
                "ablation_variant_contract",
                "ablation variant identity, hashes, ordering, or aggregate bounds are invalid",
            ));
        }
    }
    if evidence.change_detector_variants.len() != AcousticChangeDetectorMode::ALL.len() {
        return Err(public_corpus_error(
            "change_detector_variants",
            "change-detector evidence must contain every frozen detector exactly once",
        ));
    }
    for (variant, expected_mode) in evidence
        .change_detector_variants
        .iter()
        .zip(AcousticChangeDetectorMode::ALL)
    {
        let expected_schema_sha256 =
            acoustic_feature_schema_sha256(AcousticFeatureAblation::FullV2.schema_version());
        let expected_configuration_sha256 =
            canonical_sha256(&PublicChangeConfigurationFingerprint {
                runner_version: PUBLIC_CORPUS_ABLATION_RUNNER_VERSION,
                detector_mode: expected_mode,
                feature_ablation: AcousticFeatureAblation::FullV2,
                feature_schema_sha256: &expected_schema_sha256,
                diarization_request_sha256: &evidence.protocol.diarization_request_sha256,
                change_calibration_sha256: &evidence.protocol.change_calibration_sha256,
            })?;
        if variant.detector_mode != expected_mode
            || variant.feature_ablation != AcousticFeatureAblation::FullV2
            || variant.feature_schema_sha256 != expected_schema_sha256
            || variant.configuration_sha256 != expected_configuration_sha256
            || !variant_splits_are_valid(&variant.splits, evidence.protocol.change_calibration_bins)
            || variant.splits.len() != 1
            || variant.splits[0].split != evidence.evaluation_stage.selected_split()
        {
            return Err(public_corpus_error(
                "change_detector_variant_contract",
                "change-detector identity, hashes, ordering, or aggregates are invalid",
            ));
        }
    }
    if evidence.clustering_variants.len() != AcousticClusteringMode::ALL.len() {
        return Err(public_corpus_error(
            "clustering_variants",
            "clustering evidence must contain every frozen mode exactly once",
        ));
    }
    for (variant, expected_mode) in evidence
        .clustering_variants
        .iter()
        .zip(AcousticClusteringMode::ALL)
    {
        let expected_schema_sha256 =
            acoustic_feature_schema_sha256(AcousticFeatureAblation::FullV2.schema_version());
        let expected_configuration_sha256 =
            canonical_sha256(&PublicClusteringConfigurationFingerprint {
                runner_version: PUBLIC_CORPUS_ABLATION_RUNNER_VERSION,
                clustering_mode: expected_mode,
                detector_mode: AcousticChangeDetectorMode::FixedSafeV1,
                feature_ablation: AcousticFeatureAblation::FullV2,
                feature_schema_sha256: &expected_schema_sha256,
                diarization_request_sha256: &evidence.protocol.diarization_request_sha256,
                speaker_pair_calibration_sha256: &evidence.protocol.speaker_pair_calibration_sha256,
            })?;
        if variant.clustering_mode != expected_mode
            || variant.detector_mode != AcousticChangeDetectorMode::FixedSafeV1
            || variant.feature_ablation != AcousticFeatureAblation::FullV2
            || variant.configuration_sha256 != expected_configuration_sha256
            || !variant_splits_are_valid(&variant.splits, evidence.protocol.change_calibration_bins)
            || variant.splits.len() != 1
            || variant.splits[0].split != evidence.evaluation_stage.selected_split()
        {
            return Err(public_corpus_error(
                "clustering_variant_contract",
                "clustering identity, hashes, ordering, or aggregates are invalid",
            ));
        }
    }
    let full_v2_splits = evidence
        .variants
        .iter()
        .find(|variant| variant.ablation == AcousticFeatureAblation::FullV2)
        .map(|variant| &variant.splits)
        .ok_or_else(|| {
            public_corpus_error(
                "change_detector_alignment",
                "full-v2 representation evidence is unavailable",
            )
        })?;
    let calibrated_splits = evidence
        .change_detector_variants
        .iter()
        .find(|variant| variant.detector_mode == AcousticChangeDetectorMode::CalibratedPosterior)
        .map(|variant| &variant.splits)
        .ok_or_else(|| {
            public_corpus_error(
                "change_detector_alignment",
                "calibrated detector evidence is unavailable",
            )
        })?;
    if calibrated_splits != full_v2_splits {
        return Err(public_corpus_error(
            "change_detector_alignment",
            "calibrated detector and full-v2 representation aggregates differ",
        ));
    }
    let fixed_detector_splits = evidence
        .change_detector_variants
        .iter()
        .find(|variant| variant.detector_mode == AcousticChangeDetectorMode::FixedSafeV1)
        .map(|variant| &variant.splits)
        .ok_or_else(|| {
            public_corpus_error(
                "clustering_alignment",
                "fixed-safe detector evidence is unavailable",
            )
        })?;
    let fixed_clustering_splits = evidence
        .clustering_variants
        .iter()
        .find(|variant| variant.clustering_mode == AcousticClusteringMode::FixedSafeV1)
        .map(|variant| &variant.splits)
        .ok_or_else(|| {
            public_corpus_error(
                "clustering_alignment",
                "fixed-safe clustering evidence is unavailable",
            )
        })?;
    if fixed_clustering_splits != fixed_detector_splits {
        return Err(public_corpus_error(
            "clustering_alignment",
            "fixed-safe clustering and detector aggregates differ",
        ));
    }
    let gates_match = match evidence.evaluation_stage {
        PublicCorpusEvaluationStage::Development => {
            evidence.development_gate.as_ref()
                == Some(&development_improvement_gate(&evidence.variants)?)
                && evidence.change_development_gate.as_ref()
                    == Some(&change_development_gate(
                        &evidence.change_detector_variants,
                    )?)
                && evidence.clustering_development_gate.as_ref()
                    == Some(&clustering_development_gate(&evidence.clustering_variants)?)
        }
        PublicCorpusEvaluationStage::Certification => {
            evidence.held_out_gate.as_ref()
                == Some(&held_out_non_regression_gate(&evidence.variants)?)
                && evidence.change_held_out_gate.as_ref()
                    == Some(&change_held_out_gate(
                        &evidence.change_detector_variants,
                        evidence.selected_change_detector_mode,
                    )?)
                && evidence.clustering_held_out_gate.as_ref()
                    == Some(&clustering_held_out_gate(
                        &evidence.clustering_variants,
                        evidence.selected_clustering_mode,
                    )?)
        }
    };
    if !gates_match {
        return Err(public_corpus_error(
            "ablation_gate_mismatch",
            "development or held-out decision does not match the retained aggregate metrics",
        ));
    }
    if deterministic_accuracy_sha256(evidence)? != evidence.deterministic_accuracy_sha256 {
        return Err(public_corpus_error(
            "ablation_accuracy_hash_mismatch",
            "deterministic accuracy hash does not match normalized evidence",
        ));
    }
    let mut unhashed = evidence.clone();
    let expected_result_sha256 = unhashed.result_sha256.clone();
    unhashed.result_sha256.clear();
    if canonical_sha256(&unhashed)? != expected_result_sha256 {
        return Err(public_corpus_error(
            "ablation_hash_mismatch",
            "result_sha256 does not match canonical ablation evidence",
        ));
    }
    Ok(())
}

fn deterministic_accuracy_sha256(evidence: &PublicCorpusAblationEvidence) -> FwResult<String> {
    let mut normalized = evidence.clone();
    normalized.deterministic_accuracy_sha256.clear();
    normalized.result_sha256.clear();
    for variant in &mut normalized.variants {
        for split in &mut variant.splits {
            split.wall_time_sec = 0.0;
            split.real_time_factor = None;
            split.sampled_peak_rss_bytes = 0;
        }
    }
    for variant in &mut normalized.change_detector_variants {
        for split in &mut variant.splits {
            split.wall_time_sec = 0.0;
            split.real_time_factor = None;
            split.sampled_peak_rss_bytes = 0;
        }
    }
    for variant in &mut normalized.clustering_variants {
        for split in &mut variant.splits {
            split.wall_time_sec = 0.0;
            split.real_time_factor = None;
            split.sampled_peak_rss_bytes = 0;
        }
    }
    canonical_sha256(&normalized)
}

#[derive(Debug, Clone, Copy)]
struct ChangeMetricValidation {
    reference_count: u64,
    hypothesis_count: u64,
    matched_count: u64,
    precision: Option<f64>,
    recall: Option<f64>,
    f1: Option<f64>,
    mean: Option<f64>,
    p50: Option<f64>,
    p90: Option<f64>,
    p95: Option<f64>,
    collar_ms: u64,
}

impl ChangeMetricValidation {
    fn from_collar(metric: &PublicChangeCollarMetrics) -> Self {
        Self {
            reference_count: metric.reference_count,
            hypothesis_count: metric.hypothesis_count,
            matched_count: metric.matched_count,
            precision: metric.precision,
            recall: metric.recall,
            f1: metric.f1,
            mean: metric.mean_absolute_error_sec,
            p50: metric.p50_absolute_error_sec,
            p90: metric.p90_absolute_error_sec,
            p95: metric.p95_absolute_error_sec,
            collar_ms: metric.collar_ms,
        }
    }

    fn from_threshold(metric: &PublicChangeThresholdSweepPoint) -> Self {
        Self {
            reference_count: metric.reference_count,
            hypothesis_count: metric.hypothesis_count,
            matched_count: metric.matched_count,
            precision: metric.precision,
            recall: metric.recall,
            f1: metric.f1,
            mean: metric.mean_absolute_error_sec,
            p50: metric.p50_absolute_error_sec,
            p90: metric.p90_absolute_error_sec,
            p95: metric.p95_absolute_error_sec,
            collar_ms: metric.collar_ms,
        }
    }
}

fn change_metric_is_valid(metric: ChangeMetricValidation) -> bool {
    let ChangeMetricValidation {
        reference_count,
        hypothesis_count,
        matched_count,
        precision,
        recall,
        f1,
        mean,
        p50,
        p90,
        p95,
        collar_ms,
    } = metric;
    let expected_precision = ratio(matched_count, hypothesis_count);
    let expected_recall = ratio(matched_count, reference_count);
    let expected_f1 = expected_precision
        .zip(expected_recall)
        .map(|(precision, recall)| {
            let denominator = precision + recall;
            if denominator > 0.0 {
                2.0 * precision * recall / denominator
            } else {
                0.0
            }
        });
    let finite_nonnegative =
        |value: Option<f64>| value.is_none_or(|value| value.is_finite() && value >= 0.0);
    let timing_shape_valid = if matched_count == 0 {
        mean.is_none() && p50.is_none() && p90.is_none() && p95.is_none()
    } else {
        mean.is_some()
            && p50.is_some()
            && p90.is_some()
            && p95.is_some()
            && p50 <= p90
            && p90 <= p95
            && p95.is_some_and(|value| value <= collar_ms as f64 / 1_000.0 + 1e-12)
    };
    matched_count <= reference_count
        && matched_count <= hypothesis_count
        && precision == expected_precision
        && recall == expected_recall
        && f1 == expected_f1
        && finite_nonnegative(mean)
        && finite_nonnegative(p50)
        && finite_nonnegative(p90)
        && finite_nonnegative(p95)
        && timing_shape_valid
}

fn change_diagnostic_grids_are_valid(split: &PublicCorpusAblationSplit) -> bool {
    let collars_valid = split.change_collar_metrics.len()
        == PUBLIC_CHANGE_DIAGNOSTIC_COLLARS_MS.len()
        && split
            .change_collar_metrics
            .iter()
            .zip(PUBLIC_CHANGE_DIAGNOSTIC_COLLARS_MS)
            .all(|(metric, expected_collar)| {
                metric.collar_ms == expected_collar
                    && metric.reference_count == split.change_reference_count
                    && change_metric_is_valid(ChangeMetricValidation::from_collar(metric))
            });
    let operating_point_valid = split
        .change_collar_metrics
        .iter()
        .find(|metric| metric.collar_ms == 250)
        .is_some_and(|metric| {
            metric.hypothesis_count == split.change_hypothesis_count
                && metric.matched_count == split.change_matched_count
                && metric.precision == split.change_precision
                && metric.recall == split.change_recall
                && metric.f1 == split.change_f1
                && metric.mean_absolute_error_sec == split.change_mean_absolute_error_sec
        });
    let threshold_sweep_valid = split.change_threshold_sweep.len()
        == PUBLIC_CHANGE_THRESHOLD_SWEEP.len()
        && split
            .change_threshold_sweep
            .iter()
            .zip(PUBLIC_CHANGE_THRESHOLD_SWEEP)
            .all(|(metric, expected_threshold)| {
                metric.threshold == canonical_evidence_number(expected_threshold)
                    && metric.collar_ms == 250
                    && metric.reference_count == split.change_reference_count
                    && change_metric_is_valid(ChangeMetricValidation::from_threshold(metric))
            });
    collars_valid && operating_point_valid && threshold_sweep_valid
}

fn variant_splits_are_valid(splits: &[PublicCorpusAblationSplit], calibration_bins: usize) -> bool {
    if splits.is_empty()
        || !splits
            .windows(2)
            .all(|window| window[0].split < window[1].split)
    {
        return false;
    }
    splits.iter().all(|split| {
        let finite_nonnegative = |value: f64| value.is_finite() && value >= 0.0;
        let bounded = |value: Option<f64>| {
            value.is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        };
        let reliability_valid = split.change_reliability.len() == calibration_bins
            && split
                .change_reliability
                .iter()
                .enumerate()
                .all(|(index, bin)| {
                    bin.index == index
                        && bin.lower_probability
                            == canonical_evidence_number(index as f64 / calibration_bins as f64)
                        && bin.upper_probability
                            == canonical_evidence_number(
                                (index + 1) as f64 / calibration_bins as f64,
                            )
                        && bin.positive_count <= bin.observation_count
                        && match (bin.mean_probability, bin.empirical_frequency) {
                            (None, None) => bin.observation_count == 0,
                            (Some(mean), Some(empirical)) => {
                                bin.observation_count > 0
                                    && (0.0..=1.0).contains(&mean)
                                    && (0.0..=1.0).contains(&empirical)
                            }
                            _ => false,
                        }
                });
        let reliability_observations = split
            .change_reliability
            .iter()
            .map(|bin| bin.observation_count)
            .sum::<u64>();
        let reliability_positives = split
            .change_reliability
            .iter()
            .map(|bin| bin.positive_count)
            .sum::<u64>();
        let expected_ece = canonical_evidence_number(
            split
                .change_reliability
                .iter()
                .filter_map(|bin| {
                    bin.mean_probability
                        .zip(bin.empirical_frequency)
                        .map(|(mean, empirical)| {
                            bin.observation_count as f64
                                / split.change_event_observation_count.max(1) as f64
                                * (mean - empirical).abs()
                        })
                })
                .sum::<f64>(),
        );
        let calibration_valid = reliability_valid
            && reliability_observations == split.change_event_observation_count
            && reliability_positives == split.change_event_positive_count
            && split.change_event_positive_count == split.change_reference_count
            && split.change_event_observation_count
                == split
                    .change_reference_count
                    .saturating_add(split.change_hypothesis_count)
                    .saturating_sub(split.change_matched_count)
            && bounded(split.change_brier_score)
            && bounded(split.change_expected_calibration_error)
            && if split.change_event_observation_count == 0 {
                split.change_brier_score.is_none()
                    && split.change_expected_calibration_error.is_none()
            } else {
                split.change_brier_score.is_some()
                    && split
                        .change_expected_calibration_error
                        .is_some_and(|ece| (ece - expected_ece).abs() <= 1e-12)
            };
        let assignment_calibration_valid = split.assignment_observed_duration_sec
            <= split.assignment_opportunity_duration_sec
            && finite_nonnegative(split.assignment_observed_duration_sec)
            && finite_nonnegative(split.assignment_opportunity_duration_sec)
            && split.assignment_coverage
                == positive_ratio(
                    split.assignment_observed_duration_sec,
                    split.assignment_opportunity_duration_sec,
                )
            && bounded(split.assignment_brier_score)
            && bounded(split.assignment_expected_calibration_error)
            && if split.assignment_observed_duration_sec == 0.0 {
                split.assignment_brier_score.is_none()
                    && split.assignment_expected_calibration_error.is_none()
            } else {
                split.assignment_brier_score.is_some()
                    && split.assignment_expected_calibration_error.is_some()
            };
        let selective_valid = [
            split.selective_reference_speaker_time_sec,
            split.selective_covered_speaker_time_sec,
            split.selective_correct_covered_speaker_time_sec,
            split.selective_error_covered_speaker_time_sec,
            split.selective_unknown_speaker_time_sec,
        ]
        .into_iter()
        .all(finite_nonnegative)
            && split.selective_covered_speaker_time_sec
                <= split.selective_reference_speaker_time_sec + 1e-9
            && (split.selective_correct_covered_speaker_time_sec
                + split.selective_error_covered_speaker_time_sec
                - split.selective_covered_speaker_time_sec)
                .abs()
                <= 1e-9
            && (split.selective_covered_speaker_time_sec
                + split.selective_unknown_speaker_time_sec
                - split.selective_reference_speaker_time_sec)
                .abs()
                <= 1e-9
            && split.selective_coverage
                == positive_ratio(
                    split.selective_covered_speaker_time_sec,
                    split.selective_reference_speaker_time_sec,
                )
            && split.selective_risk
                == positive_ratio(
                    split.selective_error_covered_speaker_time_sec,
                    split.selective_covered_speaker_time_sec,
                )
            && bounded(split.selective_coverage)
            && bounded(split.selective_risk);
        let confusion_recording_count = split
            .speaker_count_confusion
            .iter()
            .map(|cell| cell.recording_count)
            .sum::<u64>();
        let confusion_exact_count = split
            .speaker_count_confusion
            .iter()
            .filter(|cell| cell.reference_speakers == cell.hypothesis_speakers)
            .map(|cell| cell.recording_count)
            .sum::<u64>();
        let confusion_signed_error = split
            .speaker_count_confusion
            .iter()
            .map(|cell| {
                i128::from(cell.hypothesis_speakers)
                    .saturating_sub(i128::from(cell.reference_speakers))
                    .saturating_mul(i128::from(cell.recording_count))
            })
            .sum::<i128>();
        let confusion_absolute_error = split
            .speaker_count_confusion
            .iter()
            .map(|cell| {
                u128::from(cell.hypothesis_speakers.abs_diff(cell.reference_speakers))
                    .saturating_mul(u128::from(cell.recording_count))
            })
            .sum::<u128>();
        let confusion_valid = !split.speaker_count_confusion.is_empty()
            && split.speaker_count_confusion.windows(2).all(|window| {
                (window[0].reference_speakers, window[0].hypothesis_speakers)
                    < (window[1].reference_speakers, window[1].hypothesis_speakers)
            })
            && split
                .speaker_count_confusion
                .iter()
                .all(|cell| cell.recording_count > 0)
            && confusion_recording_count == split.recording_count
            && split.exact_speaker_count_rate
                == ratio(confusion_exact_count, split.recording_count)
            && split.mean_signed_speaker_count_error
                == signed_ratio(confusion_signed_error as f64, split.recording_count as f64)
            && split.mean_absolute_speaker_count_error
                == positive_ratio(
                    confusion_absolute_error as f64,
                    split.recording_count as f64,
                );
        let stratum_recording_count = split
            .speaker_count_strata
            .iter()
            .map(|stratum| stratum.recording_count)
            .sum::<u64>();
        let stratum_posterior_count = split
            .speaker_count_strata
            .iter()
            .map(|stratum| stratum.posterior_recording_count)
            .sum::<u64>();
        let stratum_unresolved_count = split
            .speaker_count_strata
            .iter()
            .map(|stratum| stratum.unresolved_recording_count)
            .sum::<u64>();
        let stratum_zero_probability_count = split
            .speaker_count_strata
            .iter()
            .map(|stratum| stratum.zero_reference_probability_count)
            .sum::<u64>();
        let count_strata_valid = !split.speaker_count_strata.is_empty()
            && split
                .speaker_count_strata
                .windows(2)
                .all(|window| window[0].reference_speakers < window[1].reference_speakers)
            && split.speaker_count_strata.iter().all(|stratum| {
                stratum.recording_count > 0
                    && stratum.posterior_recording_count <= stratum.recording_count
                    && stratum.unresolved_recording_count <= stratum.recording_count
                    && stratum.zero_reference_probability_count <= stratum.posterior_recording_count
                    && bounded(stratum.exact_speaker_count_rate)
                    && stratum
                        .mean_negative_log_likelihood
                        .is_none_or(finite_nonnegative)
                    && stratum
                        .mean_brier_score
                        .is_none_or(|value| value.is_finite() && (0.0..=2.0).contains(&value))
                    && bounded(stratum.top_k_coverage)
                    && bounded(stratum.credible_set_coverage)
            })
            && stratum_recording_count == split.recording_count
            && stratum_posterior_count == split.count_posterior_recording_count
            && stratum_unresolved_count == split.count_unresolved_recording_count
            && stratum_zero_probability_count == split.count_zero_reference_probability_count;
        let duration_stratum_recording_count = split
            .speaker_count_duration_strata
            .iter()
            .map(|stratum| stratum.recording_count)
            .sum::<u64>();
        let duration_stratum_posterior_count = split
            .speaker_count_duration_strata
            .iter()
            .map(|stratum| stratum.posterior_recording_count)
            .sum::<u64>();
        let duration_stratum_unresolved_count = split
            .speaker_count_duration_strata
            .iter()
            .map(|stratum| stratum.unresolved_recording_count)
            .sum::<u64>();
        let duration_stratum_zero_probability_count = split
            .speaker_count_duration_strata
            .iter()
            .map(|stratum| stratum.zero_reference_probability_count)
            .sum::<u64>();
        let count_duration_strata_valid = !split.speaker_count_duration_strata.is_empty()
            && split
                .speaker_count_duration_strata
                .windows(2)
                .all(|window| window[0].duration_bucket < window[1].duration_bucket)
            && split.speaker_count_duration_strata.iter().all(|stratum| {
                stratum.recording_count > 0
                    && stratum.posterior_recording_count <= stratum.recording_count
                    && stratum.unresolved_recording_count <= stratum.recording_count
                    && stratum.zero_reference_probability_count <= stratum.posterior_recording_count
                    && bounded(stratum.exact_speaker_count_rate)
                    && stratum
                        .mean_negative_log_likelihood
                        .is_none_or(finite_nonnegative)
                    && stratum
                        .mean_brier_score
                        .is_none_or(|value| value.is_finite() && (0.0..=2.0).contains(&value))
                    && bounded(stratum.top_k_coverage)
                    && bounded(stratum.credible_set_coverage)
            })
            && duration_stratum_recording_count == split.recording_count
            && duration_stratum_posterior_count == split.count_posterior_recording_count
            && duration_stratum_unresolved_count == split.count_unresolved_recording_count
            && duration_stratum_zero_probability_count
                == split.count_zero_reference_probability_count;
        let count_posterior_valid = split
            .count_posterior_recording_count
            .saturating_add(split.count_posterior_unavailable_count)
            == split.recording_count
            && split.count_unresolved_recording_count <= split.recording_count
            && split.count_zero_reference_probability_count
                <= split.count_posterior_recording_count
            && split
                .count_mean_negative_log_likelihood
                .is_none_or(finite_nonnegative)
            && split
                .count_mean_brier_score
                .is_none_or(|value| value.is_finite() && (0.0..=2.0).contains(&value))
            && bounded(split.count_top_k_coverage)
            && bounded(split.count_credible_set_coverage)
            && split.count_mean_entropy_bits.is_none_or(finite_nonnegative)
            && if split.count_posterior_recording_count == 0 {
                split.count_mean_negative_log_likelihood.is_none()
                    && split.count_mean_brier_score.is_none()
                    && split.count_top_k_coverage.is_none()
                    && split.count_credible_set_coverage.is_none()
                    && split.count_mean_entropy_bits.is_none()
            } else {
                split.count_mean_brier_score.is_some()
                    && split.count_top_k_coverage.is_some()
                    && split.count_credible_set_coverage.is_some()
                    && split.count_mean_entropy_bits.is_some()
                    && (split.count_zero_reference_probability_count
                        == split.count_posterior_recording_count
                        || split.count_mean_negative_log_likelihood.is_some())
            };
        let count_quantiles_valid = [
            split.p50_absolute_speaker_count_error,
            split.p90_absolute_speaker_count_error,
            split.p95_absolute_speaker_count_error,
        ]
        .into_iter()
        .all(|value| value.is_some_and(finite_nonnegative))
            && split.p50_absolute_speaker_count_error <= split.p90_absolute_speaker_count_error
            && split.p90_absolute_speaker_count_error <= split.p95_absolute_speaker_count_error
            && split.p95_absolute_speaker_count_error
                <= split
                    .maximum_absolute_speaker_count_error
                    .map(|value| value as f64);
        let occupancy_valid = split.dominant_collapse_recording_count <= split.recording_count
            && split.reference_collapse_recording_count <= split.recording_count
            && split.mean_effective_speaker_count.is_none_or(|value| {
                value.is_finite()
                    && (0.0..=f64::from(crate::model::MAX_SPEAKER_COUNT)).contains(&value)
            })
            && bounded(split.mean_dominant_speaker_share)
            && bounded(split.p90_dominant_speaker_share)
            && bounded(split.p99_dominant_speaker_share)
            && bounded(split.maximum_dominant_speaker_share)
            && split.mean_dominant_speaker_share <= split.maximum_dominant_speaker_share
            && split.p90_dominant_speaker_share <= split.p99_dominant_speaker_share
            && split.p99_dominant_speaker_share <= split.maximum_dominant_speaker_share
            && bounded(split.mean_unknown_speaker_share)
            && bounded(split.maximum_unknown_speaker_share)
            && split.mean_unknown_speaker_share <= split.maximum_unknown_speaker_share
            && bounded(split.mean_minority_reference_recall);
        let word_metrics_valid = split
            .scored_word_count
            .saturating_add(split.excluded_word_count)
            == split.reference_word_count
            && split
                .correct_word_count
                .saturating_add(split.incorrect_word_count)
                .saturating_add(split.unknown_word_count)
                == split.scored_word_count
            && split.micro_word_diarization_error_rate
                == ratio(
                    split
                        .incorrect_word_count
                        .saturating_add(split.unknown_word_count),
                    split.scored_word_count,
                )
            && bounded(split.micro_word_diarization_error_rate)
            && bounded(split.macro_word_diarization_error_rate)
            && (split.scored_word_count > 0 || split.macro_word_diarization_error_rate.is_none());
        split.recording_count > 0
            && finite_nonnegative(split.reference_speaker_time_sec)
            && split.micro_der.is_none_or(finite_nonnegative)
            && split.macro_der.is_none_or(finite_nonnegative)
            && bounded(split.macro_jer)
            && finite_nonnegative(split.speaker_confusion_sec)
            && [
                split.overlap_reference_sec,
                split.overlap_hypothesis_sec,
                split.overlap_true_positive_sec,
                split.overlap_false_positive_sec,
                split.overlap_false_negative_sec,
            ]
            .into_iter()
            .all(finite_nonnegative)
            && (split.overlap_true_positive_sec + split.overlap_false_negative_sec
                - split.overlap_reference_sec)
                .abs()
                <= 1e-9
            && (split.overlap_true_positive_sec + split.overlap_false_positive_sec
                - split.overlap_hypothesis_sec)
                .abs()
                <= 1e-9
            && bounded(split.overlap_precision)
            && bounded(split.overlap_recall)
            && bounded(split.overlap_f1)
            && split.change_matched_count <= split.change_reference_count
            && split.change_matched_count <= split.change_hypothesis_count
            && bounded(split.change_precision)
            && bounded(split.change_recall)
            && bounded(split.change_f1)
            && split
                .change_mean_absolute_error_sec
                .is_none_or(finite_nonnegative)
            && change_diagnostic_grids_are_valid(split)
            && calibration_valid
            && bounded(split.exact_speaker_count_rate)
            && split
                .mean_signed_speaker_count_error
                .is_none_or(f64::is_finite)
            && split
                .mean_absolute_speaker_count_error
                .is_none_or(finite_nonnegative)
            && confusion_valid
            && count_strata_valid
            && count_duration_strata_valid
            && count_posterior_valid
            && count_quantiles_valid
            && occupancy_valid
            && word_metrics_valid
            && selective_valid
            && assignment_calibration_valid
            && bounded(split.mean_speaker_count_stability)
            && split.clustering_fallback_count <= split.recording_count
            && split.clustering_fallback_count
                == split
                    .clustering_insufficient_voice_fallback_count
                    .saturating_add(split.clustering_invalid_posterior_fallback_count)
                    .saturating_add(split.clustering_unstable_count_fallback_count)
            && finite_nonnegative(split.audio_duration_sec)
            && finite_nonnegative(split.wall_time_sec)
            && split.real_time_factor.is_none_or(finite_nonnegative)
    })
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then(|| canonical_evidence_number(numerator as f64 / denominator as f64))
}

fn positive_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then(|| canonical_evidence_number(numerator / denominator))
}

fn signed_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0 && numerator.is_finite())
        .then(|| canonical_evidence_number(numerator / denominator))
}

fn speaker_count_duration_bucket(duration_sec: f64) -> PublicSpeakerCountDurationBucket {
    if duration_sec <= 30.0 {
        PublicSpeakerCountDurationBucket::UpToThirtySeconds
    } else if duration_sec <= 120.0 {
        PublicSpeakerCountDurationBucket::UpToTwoMinutes
    } else if duration_sec <= 600.0 {
        PublicSpeakerCountDurationBucket::UpToTenMinutes
    } else {
        PublicSpeakerCountDurationBucket::LongerThanTenMinutes
    }
}

fn quantile_nearest_rank_u64(values: &[u64], probability_millionths: u32) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    quantile_nearest_rank_index(sorted.len(), probability_millionths)
        .map(|index| canonical_evidence_number(sorted[index] as f64))
}

fn quantile_nearest_rank_f64(values: &[f64], probability_millionths: u32) -> Option<f64> {
    if values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    quantile_nearest_rank_index(sorted.len(), probability_millionths)
        .map(|index| canonical_evidence_number(sorted[index]))
}

fn quantile_nearest_rank_index(length: usize, probability_millionths: u32) -> Option<usize> {
    if length == 0 || !(1..=1_000_000).contains(&probability_millionths) {
        return None;
    }
    let rank = (u128::from(probability_millionths)
        .saturating_mul(length as u128)
        .saturating_add(999_999))
        / 1_000_000;
    usize::try_from(rank.saturating_sub(1))
        .ok()
        .map(|index| index.min(length - 1))
}

/// Quantize retained aggregate evidence without changing inference precision.
///
/// JSON is the lock artifact for the two-stage public evaluation. Constraining
/// its floating-point fields to twelve decimal places prevents adjacent binary
/// representations of the same aggregate from invalidating their own hash
/// after a serialize/parse cycle. Twelve places are substantially finer than
/// the scorer's millisecond timing resolution and predeclared gate margins.
fn canonical_evidence_number(value: f64) -> f64 {
    const SCALE: f64 = 1_000_000_000_000.0;
    if !value.is_finite() || value.abs() > f64::MAX / SCALE {
        return value;
    }
    let rounded = (value * SCALE).round() / SCALE;
    if rounded == 0.0 { 0.0 } else { rounded }
}

fn parse_rttm(
    bytes: &[u8],
    recording_id: &str,
    channel: &str,
    speaker_map: &BTreeMap<String, String>,
    duration_ms: u64,
) -> FwResult<Vec<EvaluationTurn>> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| public_corpus_error("rttm_utf8", "RTTM annotation must be valid UTF-8"))?;
    let mut turns = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = trimmed.split_ascii_whitespace().collect::<Vec<_>>();
        if fields.len() != 10 || fields[0] != "SPEAKER" {
            return Err(public_corpus_error(
                "rttm_shape",
                &format!(
                    "RTTM line {} must contain exactly ten SPEAKER fields",
                    line_index + 1
                ),
            ));
        }
        if fields[1] != recording_id || fields[2] != channel {
            continue;
        }
        let start_ms = parse_rttm_milliseconds(fields[3], line_index)?;
        let duration = parse_rttm_milliseconds(fields[4], line_index)?;
        if duration == 0 {
            return Err(public_corpus_error(
                "rttm_duration",
                &format!("RTTM line {} has zero duration", line_index + 1),
            ));
        }
        let end_ms = start_ms.checked_add(duration).ok_or_else(|| {
            public_corpus_error(
                "rttm_time_overflow",
                &format!("RTTM line {} exceeds supported time range", line_index + 1),
            )
        })?;
        if end_ms > duration_ms {
            return Err(public_corpus_error(
                "rttm_bounds",
                &format!("RTTM line {} exceeds WAV duration", line_index + 1),
            ));
        }
        let speaker = speaker_map.get(fields[7]).ok_or_else(|| {
            public_corpus_error(
                "rttm_speaker_map",
                &format!(
                    "RTTM line {} speaker is absent from speaker_map",
                    line_index + 1
                ),
            )
        })?;
        turns.push(EvaluationTurn {
            start_ms,
            end_ms,
            speaker: Some(speaker.clone()),
            speaker_confidence: None,
            overlap_suspected: false,
        });
        if turns.len() > MAX_TURNS_PER_RECORDING {
            return Err(public_corpus_error(
                "rttm_turn_count",
                "RTTM turn count exceeds the supported limit",
            ));
        }
    }
    if turns.is_empty() {
        return Err(public_corpus_error(
            "rttm_no_matching_turns",
            "RTTM contains no turns for the selected recording and channel",
        ));
    }
    turns.sort_by(|left, right| {
        (
            left.start_ms,
            left.end_ms,
            left.speaker.as_deref().unwrap_or_default(),
        )
            .cmp(&(
                right.start_ms,
                right.end_ms,
                right.speaker.as_deref().unwrap_or_default(),
            ))
    });
    mark_overlapping_turns(&mut turns);
    Ok(turns)
}

fn parse_rttm_milliseconds(value: &str, line_index: usize) -> FwResult<u64> {
    if value.is_empty() || value.starts_with('-') || value.starts_with('+') {
        return Err(public_corpus_error(
            "rttm_time",
            &format!(
                "RTTM line {} time must be a non-negative decimal",
                line_index + 1
            ),
        ));
    }
    let mut parts = value.split('.');
    let whole = parts.next().unwrap_or_default();
    let fraction = parts.next();
    if parts.next().is_some()
        || whole.is_empty()
        || !whole.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.is_some_and(|digits| {
            digits.is_empty()
                || digits.len() > 9
                || !digits.bytes().all(|byte| byte.is_ascii_digit())
        })
    {
        return Err(public_corpus_error(
            "rttm_time",
            &format!(
                "RTTM line {} time must be a plain decimal with at most nine fractional digits",
                line_index + 1
            ),
        ));
    }
    let whole_seconds = whole.parse::<u64>().map_err(|_| {
        public_corpus_error(
            "rttm_time",
            &format!("RTTM line {} time is out of range", line_index + 1),
        )
    })?;
    let mut milliseconds = whole_seconds.checked_mul(1_000).ok_or_else(|| {
        public_corpus_error(
            "rttm_time",
            &format!("RTTM line {} time is out of range", line_index + 1),
        )
    })?;
    if let Some(digits) = fraction {
        let bytes = digits.as_bytes();
        let hundreds = u64::from(bytes.first().copied().unwrap_or(b'0') - b'0');
        let tens = u64::from(bytes.get(1).copied().unwrap_or(b'0') - b'0');
        let ones = u64::from(bytes.get(2).copied().unwrap_or(b'0') - b'0');
        milliseconds = milliseconds
            .checked_add(hundreds * 100 + tens * 10 + ones)
            .ok_or_else(|| {
                public_corpus_error(
                    "rttm_time",
                    &format!("RTTM line {} time is out of range", line_index + 1),
                )
            })?;
        if bytes.get(3).is_some_and(|digit| *digit >= b'5') {
            milliseconds = milliseconds.checked_add(1).ok_or_else(|| {
                public_corpus_error(
                    "rttm_time",
                    &format!("RTTM line {} time is out of range", line_index + 1),
                )
            })?;
        }
    }
    Ok(milliseconds)
}

fn mark_overlapping_turns(turns: &mut [EvaluationTurn]) {
    let mut overlaps = vec![false; turns.len()];
    {
        let mut maximum_end_by_speaker = BTreeMap::<Option<&str>, u64>::new();
        let mut ranked_maximum_ends = BTreeSet::<(u64, Option<&str>)>::new();
        for (index, turn) in turns.iter().enumerate() {
            let speaker = turn.speaker.as_deref();
            if ranked_maximum_ends
                .iter()
                .rev()
                .find(|(_, candidate)| *candidate != speaker)
                .is_some_and(|(end_ms, _)| *end_ms > turn.start_ms)
            {
                overlaps[index] = true;
            }
            let prior_end = maximum_end_by_speaker.get(&speaker).copied();
            if prior_end.is_none_or(|end_ms| turn.end_ms > end_ms) {
                if let Some(end_ms) = prior_end {
                    ranked_maximum_ends.remove(&(end_ms, speaker));
                }
                maximum_end_by_speaker.insert(speaker, turn.end_ms);
                ranked_maximum_ends.insert((turn.end_ms, speaker));
            }
        }
    }
    {
        let mut minimum_start_by_speaker = BTreeMap::<Option<&str>, u64>::new();
        let mut ranked_minimum_starts = BTreeSet::<(u64, Option<&str>)>::new();
        for (index, turn) in turns.iter().enumerate().rev() {
            let speaker = turn.speaker.as_deref();
            if ranked_minimum_starts
                .iter()
                .find(|(_, candidate)| *candidate != speaker)
                .is_some_and(|(start_ms, _)| *start_ms < turn.end_ms)
            {
                overlaps[index] = true;
            }
            let prior_start = minimum_start_by_speaker.get(&speaker).copied();
            if prior_start.is_none_or(|start_ms| turn.start_ms < start_ms) {
                if let Some(start_ms) = prior_start {
                    ranked_minimum_starts.remove(&(start_ms, speaker));
                }
                minimum_start_by_speaker.insert(speaker, turn.start_ms);
                ranked_minimum_starts.insert((turn.start_ms, speaker));
            }
        }
    }
    for (turn, overlap) in turns.iter_mut().zip(overlaps) {
        turn.overlap_suspected = overlap;
    }
}

fn hash_and_inspect_wave(
    path: &Path,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<(String, WaveMetadata)> {
    let mut file = File::open(path)
        .map_err(|_| public_corpus_error("audio_read", "audio input could not be opened"))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 128 * 1024];
    loop {
        checkpoint_cancelled(is_cancelled)?;
        let read = file
            .read(&mut buffer)
            .map_err(|_| public_corpus_error("audio_read", "audio input could not be read"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|_| public_corpus_error("audio_read", "audio input could not be rewound"))?;
    let reader = hound::WavReader::new(file).map_err(|_| {
        public_corpus_error(
            "wave_parse",
            "audio input must be a readable finite PCM or IEEE-float WAV",
        )
    })?;
    let spec = reader.spec();
    if spec.sample_rate == 0 || spec.channels == 0 {
        return Err(public_corpus_error(
            "wave_metadata",
            "WAV sample rate and channel count must be non-zero",
        ));
    }
    let frames = u64::from(reader.duration());
    let duration_ms = frames
        .checked_mul(1_000)
        .and_then(|scaled| scaled.checked_add(u64::from(spec.sample_rate) - 1))
        .map(|scaled| scaled / u64::from(spec.sample_rate))
        .ok_or_else(|| {
            public_corpus_error("wave_duration", "WAV duration exceeds the supported range")
        })?;
    if duration_ms == 0 {
        return Err(public_corpus_error(
            "wave_duration",
            "WAV must contain at least one millisecond of audio",
        ));
    }
    Ok((
        format!("{:x}", hasher.finalize()),
        WaveMetadata {
            sample_rate_hz: spec.sample_rate,
            channel_count: spec.channels,
            duration_ms,
        },
    ))
}

fn validate_split(
    policy: PublicCorpusSplitPolicy,
    recording_id: &str,
    actual: EvaluationSplit,
) -> FwResult<()> {
    if policy == PublicCorpusSplitPolicy::ExternalDescriptorV1 {
        return Ok(());
    }
    let meeting = recording_id
        .strip_prefix("ami-")
        .or_else(|| recording_id.strip_prefix("AMI-"))
        .ok_or_else(|| {
            public_corpus_error(
                "ami_recording_id",
                "AMI recording IDs must start with the ami- namespace",
            )
        })?;
    let family = meeting.get(..6).ok_or_else(|| {
        public_corpus_error(
            "ami_recording_id",
            "AMI recording ID does not contain an ASCII scenario meeting family",
        )
    })?;
    let expected = if AMI_SCENARIO_TRAIN.contains(&family) {
        EvaluationSplit::Train
    } else if AMI_SCENARIO_DEVELOPMENT.contains(&family) {
        EvaluationSplit::Development
    } else if AMI_SCENARIO_TEST.contains(&family) {
        EvaluationSplit::Test
    } else {
        return Err(public_corpus_error(
            "ami_split_unknown",
            "AMI recording is outside the frozen scenario-only family split",
        ));
    };
    if actual == expected {
        Ok(())
    } else {
        Err(public_corpus_error(
            "ami_split_mismatch",
            "AMI recording split differs from the frozen official scenario-only split",
        ))
    }
}

const AMI_SCENARIO_TRAIN: [&str; 25] = [
    "ES2002", "ES2005", "ES2006", "ES2007", "ES2008", "ES2009", "ES2010", "ES2012", "ES2013",
    "ES2015", "ES2016", "IS1000", "IS1001", "IS1002", "IS1003", "IS1004", "IS1005", "IS1006",
    "IS1007", "TS3005", "TS3008", "TS3009", "TS3010", "TS3011", "TS3012",
];
const AMI_SCENARIO_DEVELOPMENT: [&str; 5] = ["ES2003", "ES2011", "IS1008", "TS3004", "TS3006"];
const AMI_SCENARIO_TEST: [&str; 5] = ["ES2004", "ES2014", "IS1009", "TS3003", "TS3007"];

fn validate_speaker_map(speaker_map: &BTreeMap<String, String>) -> FwResult<()> {
    if speaker_map.is_empty() {
        return Err(public_corpus_error(
            "speaker_map",
            "speaker_map must contain at least one source-to-opaque identity",
        ));
    }
    let mut targets = BTreeSet::new();
    for (source, target) in speaker_map {
        if source.is_empty()
            || source.len() > 160
            || source.trim() != source
            || source.chars().any(char::is_control)
            || source.bytes().any(|byte| byte.is_ascii_whitespace())
        {
            return Err(public_corpus_error(
                "speaker_map_source",
                "speaker_map source labels must be bounded non-whitespace tokens",
            ));
        }
        validate_public_id(target, "speaker_map target")?;
        if !targets.insert(target) {
            return Err(public_corpus_error(
                "speaker_map_target",
                "speaker_map target identities must be unique",
            ));
        }
    }
    Ok(())
}

fn validate_rttm_channel(channel: &str) -> FwResult<()> {
    if channel.is_empty()
        || channel.len() > 32
        || channel.trim() != channel
        || channel
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte.is_ascii_control())
    {
        Err(public_corpus_error(
            "annotation_channel",
            "RTTM channel must be one bounded non-whitespace token",
        ))
    } else {
        Ok(())
    }
}

fn validate_public_id(value: &str, field: &str) -> FwResult<()> {
    if value.is_empty()
        || value.len() > 160
        || value.trim() != value
        || value.chars().any(char::is_control)
        || value.contains('/')
        || value.contains('\\')
        || value.contains("..")
    {
        return Err(public_corpus_error(
            "opaque_id",
            &format!("{field} must be a bounded path-free opaque identifier"),
        ));
    }
    let lower = value.to_ascii_lowercase();
    for forbidden in [
        "downloads",
        "transcript",
        ".m4a",
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".aac",
        ".wma",
        ".mp4",
        ".srt",
        ".md",
    ] {
        if lower.contains(forbidden) {
            return Err(public_corpus_error(
                "opaque_id_sensitive",
                &format!("{field} contains a forbidden path or media marker"),
            ));
        }
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> FwResult<()> {
    if is_sha256_hex(value) {
        Ok(())
    } else {
        Err(public_corpus_error(
            "hash_format",
            &format!("{field} must be 64 lowercase hexadecimal characters"),
        ))
    }
}

fn canonical_directory(path: &Path, field: &str) -> FwResult<PathBuf> {
    if !path.is_absolute() {
        return Err(public_corpus_error(
            "absolute_path",
            &format!("{field} must be absolute"),
        ));
    }
    let canonical = path.canonicalize().map_err(|_| {
        public_corpus_error(
            "directory",
            &format!("{field} must be an existing readable directory"),
        )
    })?;
    if !canonical.is_dir() {
        return Err(public_corpus_error(
            "directory",
            &format!("{field} must resolve to a directory"),
        ));
    }
    Ok(canonical)
}

fn canonical_input_file(root: &Path, path: &Path, field: &str) -> FwResult<PathBuf> {
    if !path.is_absolute() {
        return Err(public_corpus_error(
            "absolute_path",
            &format!("{field} must be absolute"),
        ));
    }
    let canonical = path.canonicalize().map_err(|_| {
        public_corpus_error(
            "input_file",
            &format!("{field} must resolve to a readable file"),
        )
    })?;
    if !canonical.starts_with(root) || !canonical.is_file() {
        return Err(public_corpus_error(
            "input_escape",
            &format!("{field} must resolve beneath input_root"),
        ));
    }
    Ok(canonical)
}

fn canonical_external_file(
    project: &Path,
    input: &Path,
    path: &Path,
    field: &str,
) -> FwResult<PathBuf> {
    if !path.is_absolute() {
        return Err(public_corpus_error(
            "absolute_path",
            &format!("{field} must be absolute"),
        ));
    }
    let canonical = path.canonicalize().map_err(|_| {
        public_corpus_error(
            "external_file",
            &format!("{field} must resolve to a readable file"),
        )
    })?;
    if !canonical.is_file()
        || canonical.extension().and_then(|value| value.to_str()) != Some("json")
        || canonical.starts_with(project)
        || canonical.starts_with(input)
    {
        return Err(public_corpus_error(
            "external_file",
            &format!("{field} must be an external JSON file"),
        ));
    }
    Ok(canonical)
}

fn canonical_relative_file(root: &Path, relative: &Path, field: &str) -> FwResult<PathBuf> {
    if relative.is_absolute()
        || relative.as_os_str().is_empty()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(public_corpus_error(
            "relative_path",
            &format!("{field} path must be a non-empty relative path without traversal"),
        ));
    }
    let canonical = root.join(relative).canonicalize().map_err(|_| {
        public_corpus_error(
            "input_file",
            &format!("{field} input must resolve to a readable file"),
        )
    })?;
    if !canonical.starts_with(root) || !canonical.is_file() {
        return Err(public_corpus_error(
            "input_escape",
            &format!("{field} input must resolve beneath input_root"),
        ));
    }
    Ok(canonical)
}

fn validate_new_output(project: &Path, input: &Path, output: &Path) -> FwResult<PathBuf> {
    if !output.is_absolute() || output.extension().and_then(|value| value.to_str()) != Some("json")
    {
        return Err(public_corpus_error(
            "output_path",
            "output must be an absolute path with a .json extension",
        ));
    }
    if output.exists() {
        return Err(public_corpus_error(
            "output_exists",
            "output must not already exist",
        ));
    }
    let parent = output.parent().ok_or_else(|| {
        public_corpus_error(
            "output_parent",
            "output must have an existing parent directory",
        )
    })?;
    let canonical_parent = parent.canonicalize().map_err(|_| {
        public_corpus_error(
            "output_parent",
            "output parent must be an existing directory",
        )
    })?;
    if !canonical_parent.is_dir()
        || paths_overlap(project, &canonical_parent)
        || paths_overlap(input, &canonical_parent)
    {
        return Err(public_corpus_error(
            "output_overlap",
            "output parent must be disjoint from the project and input roots",
        ));
    }
    Ok(canonical_parent)
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left.starts_with(right) || right.starts_with(left)
}

fn read_bounded(path: &Path, limit: u64, field: &str) -> FwResult<Vec<u8>> {
    let file = File::open(path).map_err(|_| {
        public_corpus_error("input_read", &format!("{field} input could not be opened"))
    })?;
    let metadata = file.metadata().map_err(|_| {
        public_corpus_error("input_read", &format!("{field} metadata could not be read"))
    })?;
    if metadata.len() > limit {
        return Err(public_corpus_error(
            "input_size",
            &format!("{field} input exceeds its safety limit"),
        ));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        public_corpus_error(
            "input_size",
            &format!("{field} input length is unsupported on this platform"),
        )
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(limit + 1).read_to_end(&mut bytes).map_err(|_| {
        public_corpus_error("input_read", &format!("{field} input could not be read"))
    })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > limit {
        return Err(public_corpus_error(
            "input_size",
            &format!("{field} input exceeds its safety limit"),
        ));
    }
    Ok(bytes)
}

fn write_new_json<T: Serialize>(
    output_path: &Path,
    canonical_parent: &Path,
    value: &T,
    artifact: &str,
) -> FwResult<()> {
    let output_name = output_path
        .file_name()
        .ok_or_else(|| public_corpus_error("output_path", "output must include a file name"))?;
    let canonical_target = canonical_parent.join(output_name);
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&canonical_target)
        .map_err(|_| {
            public_corpus_error(
                "output_create",
                &format!("new {artifact} output could not be created"),
            )
        })?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value).map_err(|_| {
        public_corpus_error(
            "output_write",
            &format!("{artifact} output could not be serialized"),
        )
    })?;
    writer.write_all(b"\n").map_err(|_| {
        public_corpus_error(
            "output_write",
            &format!("{artifact} output could not be written"),
        )
    })?;
    writer.flush().map_err(|_| {
        public_corpus_error(
            "output_write",
            &format!("{artifact} output could not be flushed"),
        )
    })?;
    writer.get_ref().sync_all().map_err(|_| {
        public_corpus_error(
            "output_write",
            &format!("{artifact} output could not be durably synchronized"),
        )
    })
}

fn canonical_sha256<T: Serialize>(value: &T) -> FwResult<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == HASH_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn checkpoint_cancelled(is_cancelled: &mut impl FnMut() -> bool) -> FwResult<()> {
    if is_cancelled() {
        Err(FwError::Cancelled(
            "public_corpus.cancelled: public corpus preparation cancelled".to_owned(),
        ))
    } else {
        Ok(())
    }
}

fn public_corpus_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("public_corpus.{code}: {message}"))
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use serde_json::json;
    use sha2::Digest as _;
    use tempfile::tempdir;

    use super::{
        PUBLIC_CORPUS_INPUT_SCHEMA_VERSION, PublicAblationAccumulator, PublicCorpusAblationSplit,
        PublicCorpusAblationVariant, build_public_corpus_bundle,
        build_public_corpus_bundle_with_cancel, clipped_reference, development_improvement_gate,
        held_out_non_regression_gate, merged_scored_speech_regions, parse_public_corpus_bundle,
        public_corpus_registry, validate_split,
    };
    use crate::FwResult;
    use crate::diarization::{
        AcousticFeatureAblation, DIARIZATION_REFERENCE_SCHEMA_VERSION,
        DiarizationReferenceDocument, EvaluationRegion, EvaluationSplit, EvaluationTurn,
    };

    fn write_wave(path: &Path, sample_rate: u32, channels: u16, frames: u32) {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("WAV");
        for _ in 0..frames * u32::from(channels) {
            writer.write_sample(0_i16).expect("sample");
        }
        writer.finalize().expect("finalize WAV");
    }

    fn sha256(path: &Path) -> String {
        format!(
            "{:x}",
            sha2::Sha256::digest(std::fs::read(path).expect("fixture"))
        )
    }

    fn descriptor(
        corpus_key: &str,
        recording_id: &str,
        split: &str,
        audio_sha256: &str,
        annotation_sha256: &str,
        sample_rate: u32,
        channels: u16,
    ) -> serde_json::Value {
        json!({
            "schema_version": PUBLIC_CORPUS_INPUT_SCHEMA_VERSION,
            "corpus_key": corpus_key,
            "source_version": "fixture-v1",
            "recordings": [{
                "recording_id": recording_id,
                "split": split,
                "origin_recording_id": format!("{recording_id}-origin"),
                "audio_path": "audio.wav",
                "audio_sha256": audio_sha256,
                "expected_sample_rate_hz": sample_rate,
                "expected_channel_count": channels,
                "selected_channel": 1,
                "annotation_path": "annotation.rttm",
                "annotation_sha256": annotation_sha256,
                "annotation_recording_id": "source-call",
                "annotation_channel": "1",
                "speaker_map": {
                    "source-a": format!("{recording_id}-speaker-a"),
                    "source-b": format!("{recording_id}-speaker-b")
                },
                "ignored_regions": [{
                    "start_ms": 900,
                    "end_ms": 950,
                    "reason_code": "annotation_uncertain"
                }]
            }]
        })
    }

    struct Fixture {
        project: tempfile::TempDir,
        input: tempfile::TempDir,
        output: tempfile::TempDir,
        descriptor_path: PathBuf,
        output_path: PathBuf,
    }

    impl Fixture {
        fn new(corpus_key: &str, recording_id: &str, split: &str) -> Self {
            let project = tempdir().expect("project");
            let input = tempdir().expect("input");
            let output = tempdir().expect("output");
            write_wave(&input.path().join("audio.wav"), 8_000, 2, 8_000);
            std::fs::write(
                input.path().join("annotation.rttm"),
                concat!(
                    "SPEAKER source-call 1 0.000 0.600 <NA> <NA> source-a <NA> <NA>\n",
                    "SPEAKER source-call 1 0.400 0.500 <NA> <NA> source-b <NA> <NA>\n",
                ),
            )
            .expect("RTTM");
            let audio_hash = sha256(&input.path().join("audio.wav"));
            let annotation_hash = sha256(&input.path().join("annotation.rttm"));
            let descriptor_path = input.path().join("descriptor.json");
            std::fs::write(
                &descriptor_path,
                serde_json::to_vec_pretty(&descriptor(
                    corpus_key,
                    recording_id,
                    split,
                    &audio_hash,
                    &annotation_hash,
                    8_000,
                    2,
                ))
                .expect("descriptor JSON"),
            )
            .expect("descriptor");
            let output_path = output.path().join("bundle.json");
            Self {
                project,
                input,
                output,
                descriptor_path,
                output_path,
            }
        }

        fn build(&self, acknowledgement: &str) -> FwResult<super::PublicCorpusBundle> {
            build_public_corpus_bundle(
                self.project.path(),
                self.input.path(),
                &self.descriptor_path,
                &self.output_path,
                acknowledgement,
            )
        }
    }

    #[test]
    fn registry_is_sorted_complete_and_path_free() {
        let registry = public_corpus_registry();
        assert_eq!(registry.entries.len(), 4);
        assert!(
            registry
                .entries
                .windows(2)
                .all(|window| window[0].corpus_key < window[1].corpus_key)
        );
        for entry in &registry.entries {
            assert!(entry.authoritative_url.starts_with("https://"));
            assert!(entry.license_url.starts_with("https://"));
            assert!(!entry.license_acknowledgement_id.is_empty());
            assert!(!entry.condition_tags.is_empty());
            assert!(
                entry
                    .condition_tags
                    .windows(2)
                    .all(|window| window[0] < window[1])
            );
        }
    }

    #[test]
    fn build_requires_exact_license_acknowledgement() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let error = fixture.build("yes").expect_err("missing acknowledgement");
        assert!(error.to_string().contains("license_acknowledgement"));
        assert!(!fixture.output_path.exists());
    }

    #[test]
    fn checksum_mismatch_fails_before_output() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&fixture.descriptor_path).expect("descriptor"))
                .expect("JSON");
        value["recordings"][0]["audio_sha256"] = json!("0".repeat(64));
        std::fs::write(
            &fixture.descriptor_path,
            serde_json::to_vec_pretty(&value).expect("JSON"),
        )
        .expect("descriptor");
        let error = fixture
            .build("accept-aishell-4-cc-by-sa-4.0")
            .expect_err("checksum mismatch");
        assert!(error.to_string().contains("audio_checksum_mismatch"));
        assert!(!fixture.output_path.exists());
    }

    #[test]
    fn malformed_rttm_fails_with_stable_code() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        std::fs::write(
            fixture.input.path().join("annotation.rttm"),
            "SPEAKER too few fields\n",
        )
        .expect("malformed RTTM");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&fixture.descriptor_path).expect("descriptor"))
                .expect("JSON");
        value["recordings"][0]["annotation_sha256"] =
            json!(sha256(&fixture.input.path().join("annotation.rttm")));
        std::fs::write(
            &fixture.descriptor_path,
            serde_json::to_vec_pretty(&value).expect("JSON"),
        )
        .expect("descriptor");
        let error = fixture
            .build("accept-aishell-4-cc-by-sa-4.0")
            .expect_err("malformed RTTM");
        assert!(error.to_string().contains("rttm_shape"));
    }

    #[test]
    fn wave_channel_and_sample_rate_contracts_are_checked() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&fixture.descriptor_path).expect("descriptor"))
                .expect("JSON");
        value["recordings"][0]["expected_sample_rate_hz"] = json!(16_000);
        std::fs::write(
            &fixture.descriptor_path,
            serde_json::to_vec_pretty(&value).expect("JSON"),
        )
        .expect("descriptor");
        let error = fixture
            .build("accept-aishell-4-cc-by-sa-4.0")
            .expect_err("sample-rate mismatch");
        assert!(error.to_string().contains("audio_metadata_mismatch"));
    }

    #[test]
    fn overlap_ignored_regions_and_determinism_survive_round_trip() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let bundle = fixture
            .build("accept-aishell-4-cc-by-sa-4.0")
            .expect("bundle");
        assert_eq!(bundle.references[0].turns.len(), 2);
        assert!(
            bundle.references[0]
                .turns
                .iter()
                .all(|turn| turn.overlap_suspected)
        );
        assert_eq!(bundle.references[0].ignored_regions.len(), 1);
        let retained = std::fs::read(&fixture.output_path).expect("retained bundle");
        assert_eq!(
            parse_public_corpus_bundle(&retained).expect("parse bundle"),
            bundle
        );
        let hypothesis = crate::diarization::DiarizationHypothesisDocument {
            schema_version: crate::diarization::DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: bundle.references[0].recording_id.clone(),
            duration_ms: bundle.references[0].duration_ms,
            turns: bundle.references[0].turns.clone(),
            speaker_count_estimate: None,
            performance: None,
        };
        let score = crate::diarization::score_diarization_documents(
            &bundle.references[0],
            &hypothesis,
            &crate::diarization::DiarizationScorerConfig::default(),
        )
        .expect("generated reference must run through the frozen scorer");
        assert_eq!(score.diarization.der, Some(0.0));
        assert_eq!(score.diarization.jer, Some(0.0));

        let second_output = fixture.output.path().join("bundle-second.json");
        let second = build_public_corpus_bundle(
            fixture.project.path(),
            fixture.input.path(),
            &fixture.descriptor_path,
            &second_output,
            "accept-aishell-4-cc-by-sa-4.0",
        )
        .expect("second bundle");
        assert_eq!(second, bundle);
        assert_eq!(
            std::fs::read_to_string(&second_output).expect("second output"),
            std::fs::read_to_string(&fixture.output_path).expect("first output")
        );
    }

    #[test]
    fn optional_word_annotations_are_checksum_bound_and_transcript_free() {
        let fixture = Fixture::new(
            "aishell-4-openslr111-v1",
            "aishell-word-fixture",
            "development",
        );
        let word_path = fixture.input.path().join("words.json");
        std::fs::write(
            &word_path,
            serde_json::to_vec_pretty(&json!({
                "schema_version": super::PUBLIC_CORPUS_WORD_ANNOTATION_SCHEMA_VERSION,
                "recording_id": "aishell-word-fixture",
                "words": [
                    {
                        "word_id": "word-001",
                        "start_ms": 100,
                        "end_ms": 200,
                        "speaker_ref": "aishell-word-fixture-speaker-a"
                    },
                    {
                        "word_id": "word-002",
                        "start_ms": 450,
                        "end_ms": 500,
                        "speaker_ref": "aishell-word-fixture-speaker-b"
                    }
                ]
            }))
            .expect("word annotation JSON"),
        )
        .expect("word annotation");
        let mut descriptor: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&fixture.descriptor_path).expect("descriptor"))
                .expect("descriptor JSON");
        descriptor["recordings"][0]["word_annotation_path"] = json!("words.json");
        descriptor["recordings"][0]["word_annotation_sha256"] = json!(sha256(&word_path));
        std::fs::write(
            &fixture.descriptor_path,
            serde_json::to_vec_pretty(&descriptor).expect("descriptor JSON"),
        )
        .expect("descriptor");

        let bundle = fixture
            .build("accept-aishell-4-cc-by-sa-4.0")
            .expect("word-bound bundle");
        assert_eq!(bundle.references[0].words.len(), 2);
        assert_eq!(bundle.recordings[0].word_count, 2);
        assert_eq!(
            bundle.recordings[0].word_annotation_sha256,
            Some(sha256(&word_path))
        );
        let hypothesis = crate::diarization::DiarizationHypothesisDocument {
            schema_version: crate::diarization::DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: bundle.references[0].recording_id.clone(),
            duration_ms: bundle.references[0].duration_ms,
            turns: bundle.references[0].turns.clone(),
            speaker_count_estimate: None,
            performance: None,
        };
        let score = crate::diarization::score_diarization_documents(
            &bundle.references[0],
            &hypothesis,
            &crate::diarization::DiarizationScorerConfig::default(),
        )
        .expect("word score");
        assert_eq!(score.word_attribution.correct_word_count, 2);
        assert_eq!(
            score.word_attribution.word_diarization_error_rate,
            Some(0.0)
        );
    }

    #[test]
    fn official_ami_split_is_enforced() {
        validate_split(
            super::PublicCorpusSplitPolicy::AmiScenarioOfficialV1,
            "ami-ES2003a-array",
            EvaluationSplit::Development,
        )
        .expect("official dev split");
        let error = validate_split(
            super::PublicCorpusSplitPolicy::AmiScenarioOfficialV1,
            "ami-ES2003a-array",
            EvaluationSplit::Test,
        )
        .expect_err("wrong split");
        assert!(error.to_string().contains("ami_split_mismatch"));
    }

    #[test]
    fn malformed_unicode_ami_id_returns_error_instead_of_panicking() {
        let error = validate_split(
            super::PublicCorpusSplitPolicy::AmiScenarioOfficialV1,
            "ami-aééé",
            EvaluationSplit::Development,
        )
        .expect_err("non-ASCII family");
        assert!(error.to_string().contains("ami_recording_id"));
    }

    #[test]
    fn cancellation_leaves_no_output() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let mut checks = 0_u8;
        let error = build_public_corpus_bundle_with_cancel(
            fixture.project.path(),
            fixture.input.path(),
            &fixture.descriptor_path,
            &fixture.output_path,
            "accept-aishell-4-cc-by-sa-4.0",
            || {
                checks = checks.saturating_add(1);
                checks >= 2
            },
        )
        .expect_err("cancelled");
        assert!(matches!(error, crate::FwError::Cancelled(_)));
        assert!(!fixture.output_path.exists());
    }

    #[test]
    fn output_must_remain_outside_project_and_inputs() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let unsafe_output = fixture.project.path().join("bundle.json");
        let error = build_public_corpus_bundle(
            fixture.project.path(),
            fixture.input.path(),
            &fixture.descriptor_path,
            &unsafe_output,
            "accept-aishell-4-cc-by-sa-4.0",
        )
        .expect_err("project output");
        assert!(error.to_string().contains("output_overlap"));
    }

    #[test]
    fn rttm_time_rounding_is_decimal_and_deterministic() {
        assert_eq!(
            super::parse_rttm_milliseconds("1.2344", 0).expect("time"),
            1_234
        );
        assert_eq!(
            super::parse_rttm_milliseconds("1.2345", 0).expect("time"),
            1_235
        );
        assert!(super::parse_rttm_milliseconds("1e3", 0).is_err());
    }

    #[test]
    fn overlap_marking_distinguishes_same_and_different_speakers() {
        let mut turns = vec![
            crate::diarization::EvaluationTurn::labeled(0, 100, "speaker-a"),
            crate::diarization::EvaluationTurn::labeled(10, 90, "speaker-a"),
            crate::diarization::EvaluationTurn::labeled(95, 110, "speaker-b"),
            crate::diarization::EvaluationTurn::labeled(200, 300, "speaker-c"),
        ];
        super::mark_overlapping_turns(&mut turns);
        assert!(turns[0].overlap_suspected);
        assert!(!turns[1].overlap_suspected);
        assert!(turns[2].overlap_suspected);
        assert!(!turns[3].overlap_suspected);
    }

    #[test]
    fn ablation_reference_clipping_and_oracle_vad_are_bounded() {
        let reference = DiarizationReferenceDocument {
            schema_version: DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
            recording_id: "fixture".to_owned(),
            duration_ms: 1_000,
            turns: vec![
                EvaluationTurn::labeled(100, 300, "speaker-a"),
                EvaluationTurn::labeled(250, 700, "speaker-b"),
                EvaluationTurn::labeled(800, 950, "speaker-a"),
            ],
            ignored_regions: vec![EvaluationRegion {
                start_ms: 600,
                end_ms: 900,
                reason_code: "fixture".to_owned(),
            }],
            speaker_hints: Vec::new(),
            words: Vec::new(),
        };
        let clipped = clipped_reference(&reference, Some(650)).expect("clipped reference");
        assert_eq!(clipped.duration_ms, 650);
        assert_eq!(clipped.turns.len(), 2);
        assert_eq!(clipped.turns[1].end_ms, 650);
        assert_eq!(clipped.ignored_regions[0].end_ms, 650);
        assert_eq!(
            merged_scored_speech_regions(&clipped.turns, &clipped.ignored_regions),
            vec![(100, 600)],
            "overlapping turns merge while ignored scoring regions stay outside oracle VAD"
        );
    }

    #[test]
    fn change_event_calibration_counts_matches_false_alarms_and_misses_once() {
        let hypothesis = [
            super::ChangeProbabilityObservation {
                boundary_ms: 900,
                probability: 0.8,
            },
            super::ChangeProbabilityObservation {
                boundary_ms: 1_500,
                probability: 0.9,
            },
            super::ChangeProbabilityObservation {
                boundary_ms: 2_050,
                probability: 0.6,
            },
        ];
        let aggregate =
            super::score_change_event_calibration(&[1_000, 2_000], &hypothesis, 200, 10)
                .expect("event calibration");
        assert_eq!(aggregate.observation_count, 3);
        assert_eq!(aggregate.positive_count, 2);
        assert!((aggregate.brier_sum - 1.01).abs() < 1e-12);
        assert_eq!(aggregate.bins[6].observation_count, 1);
        assert_eq!(aggregate.bins[8].observation_count, 1);
        assert_eq!(aggregate.bins[9].observation_count, 1);

        let missed =
            super::score_change_event_calibration(&[1_000], &[], 200, 10).expect("missed event");
        assert_eq!(missed.observation_count, 1);
        assert_eq!(missed.positive_count, 1);
        assert_eq!(missed.brier_sum, 1.0);
        assert_eq!(missed.bins[0].positive_count, 1);
    }

    #[test]
    fn change_timing_traceback_matches_the_max_cardinality_minimum_error_score() {
        let reference = [0.0, 1.0, 2.0];
        let hypothesis = [0.20, 0.85, 2.40];
        let errors = super::minimum_error_change_match_errors(&reference, &hypothesis, 0.30);
        let score =
            crate::diarization::score_change_points(&reference, &hypothesis, 0.30).expect("score");
        assert_eq!(errors.len(), score.matched_count);
        assert!(
            (errors.iter().sum::<f64>()
                - score.mean_absolute_error_sec.expect("matched") * score.matched_count as f64)
                .abs()
                < 1e-12
        );
        assert!((errors[0] - 0.15).abs() < 1e-12);
        assert!((errors[1] - 0.20).abs() < 1e-12);
    }

    #[test]
    fn zero_match_aggregate_reports_defined_zero_f1_and_timing_authority_is_asymmetric() {
        let aggregate = PublicAblationAccumulator {
            recording_count: 1,
            change_reference_count: 1,
            change_hypothesis_count: 1,
            overlap_reference_sec: 1.0,
            overlap_false_negative_sec: 1.0,
            ..PublicAblationAccumulator::default()
        }
        .finish(EvaluationSplit::Development);
        assert_eq!(aggregate.change_precision, Some(0.0));
        assert_eq!(aggregate.change_recall, Some(0.0));
        assert_eq!(aggregate.change_f1, Some(0.0));
        assert_eq!(aggregate.overlap_precision, None);
        assert_eq!(aggregate.overlap_recall, Some(0.0));
        assert_eq!(aggregate.overlap_f1, Some(0.0));
        assert!(super::change_timing_requirement_passed(Some(0.2), None));
        assert!(!super::change_timing_requirement_passed(None, Some(0.2)));
        assert!(!super::change_timing_requirement_passed(None, None));
    }

    #[test]
    fn report_change_observations_ignore_silence_gaps_and_use_boundary_confidence() {
        let turn = |start_ms, end_ms, speaker: Option<&str>, change_confidence| {
            crate::model::DiarizationTurn {
                start_ms,
                end_ms,
                speaker_ref: speaker.map(str::to_owned),
                speaker_confidence: speaker.map(|_| 0.8),
                change_confidence,
                overlap_suspected: false,
                hard_hint_attributed: false,
            }
        };
        let observations = super::report_change_observations(&[
            turn(0, 1_000, Some("a"), Some(0.75)),
            turn(1_000, 2_000, Some("b"), Some(0.25)),
            turn(2_100, 3_000, Some("a"), Some(0.9)),
        ]);
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].boundary_ms, 1_000);
        assert_eq!(observations[0].probability, 0.75);
    }

    #[test]
    fn ablation_aggregate_uses_micro_and_macro_denominators_exactly() {
        let aggregate = PublicAblationAccumulator {
            recording_count: 2,
            reference_speaker_time_sec: 10.0,
            missed_speech_sec: 1.0,
            false_alarm_sec: 0.5,
            speaker_confusion_sec: 0.5,
            macro_der_sum: 0.6,
            macro_der_count: 2,
            macro_jer_sum: 0.8,
            macro_jer_count: 2,
            change_reference_count: 4,
            change_hypothesis_count: 5,
            change_matched_count: 3,
            exact_speaker_count: 1,
            absolute_speaker_count_error: 2,
            audio_duration_sec: 20.0,
            wall_time_sec: 5.0,
            sampled_peak_rss_bytes: 123,
            ..PublicAblationAccumulator::default()
        }
        .finish(EvaluationSplit::Development);
        assert_eq!(aggregate.micro_der, Some(0.2));
        assert_eq!(aggregate.macro_der, Some(0.3));
        assert_eq!(aggregate.macro_jer, Some(0.4));
        assert!(
            aggregate
                .change_f1
                .is_some_and(|value| (value - 2.0 / 3.0).abs() < 1e-12)
        );
        assert_eq!(aggregate.exact_speaker_count_rate, Some(0.5));
        assert_eq!(aggregate.mean_absolute_speaker_count_error, Some(1.0));
        assert_eq!(aggregate.real_time_factor, Some(0.25));
        assert_eq!(aggregate.sampled_peak_rss_bytes, 123);
    }

    fn ablation_variant(
        ablation: AcousticFeatureAblation,
        micro_der: Option<f64>,
        macro_jer: Option<f64>,
    ) -> PublicCorpusAblationVariant {
        let split = |split| PublicCorpusAblationSplit {
            split,
            recording_count: 1,
            reference_speaker_time_sec: 1.0,
            micro_der,
            macro_der: micro_der,
            macro_jer,
            speaker_confusion_sec: 0.0,
            overlap_reference_sec: 1.0,
            overlap_hypothesis_sec: 1.0,
            overlap_true_positive_sec: 1.0,
            overlap_false_positive_sec: 0.0,
            overlap_false_negative_sec: 0.0,
            overlap_precision: Some(1.0),
            overlap_recall: Some(1.0),
            overlap_f1: Some(1.0),
            change_reference_count: 1,
            change_hypothesis_count: 1,
            change_matched_count: 1,
            change_precision: Some(1.0),
            change_recall: Some(1.0),
            change_f1: Some(1.0),
            change_mean_absolute_error_sec: Some(0.0),
            change_event_observation_count: 1,
            change_event_positive_count: 1,
            change_brier_score: Some(0.0),
            change_expected_calibration_error: Some(0.0),
            change_reliability: (0..10)
                .map(|index| super::PublicChangeReliabilityBin {
                    index,
                    lower_probability: index as f64 / 10.0,
                    upper_probability: (index + 1) as f64 / 10.0,
                    observation_count: u64::from(index == 9),
                    positive_count: u64::from(index == 9),
                    mean_probability: (index == 9).then_some(1.0),
                    empirical_frequency: (index == 9).then_some(1.0),
                })
                .collect(),
            change_collar_metrics: super::PUBLIC_CHANGE_DIAGNOSTIC_COLLARS_MS
                .into_iter()
                .map(|collar_ms| super::PublicChangeCollarMetrics {
                    collar_ms,
                    reference_count: 1,
                    hypothesis_count: 1,
                    matched_count: 1,
                    precision: Some(1.0),
                    recall: Some(1.0),
                    f1: Some(1.0),
                    mean_absolute_error_sec: Some(0.0),
                    p50_absolute_error_sec: Some(0.0),
                    p90_absolute_error_sec: Some(0.0),
                    p95_absolute_error_sec: Some(0.0),
                })
                .collect(),
            change_threshold_sweep: super::PUBLIC_CHANGE_THRESHOLD_SWEEP
                .into_iter()
                .map(|threshold| super::PublicChangeThresholdSweepPoint {
                    threshold,
                    collar_ms: 250,
                    reference_count: 1,
                    hypothesis_count: 1,
                    matched_count: 1,
                    precision: Some(1.0),
                    recall: Some(1.0),
                    f1: Some(1.0),
                    mean_absolute_error_sec: Some(0.0),
                    p50_absolute_error_sec: Some(0.0),
                    p90_absolute_error_sec: Some(0.0),
                    p95_absolute_error_sec: Some(0.0),
                })
                .collect(),
            exact_speaker_count_rate: Some(1.0),
            mean_signed_speaker_count_error: Some(0.0),
            mean_absolute_speaker_count_error: Some(0.0),
            p50_absolute_speaker_count_error: Some(0.0),
            p90_absolute_speaker_count_error: Some(0.0),
            p95_absolute_speaker_count_error: Some(0.0),
            maximum_absolute_speaker_count_error: Some(0),
            speaker_count_confusion: vec![super::PublicSpeakerCountConfusionCell {
                reference_speakers: 1,
                hypothesis_speakers: 1,
                recording_count: 1,
            }],
            speaker_count_strata: vec![super::PublicSpeakerCountStratum {
                reference_speakers: 1,
                recording_count: 1,
                posterior_recording_count: 0,
                unresolved_recording_count: 1,
                zero_reference_probability_count: 0,
                exact_speaker_count_rate: Some(1.0),
                mean_negative_log_likelihood: None,
                mean_brier_score: None,
                top_k_coverage: None,
                credible_set_coverage: None,
            }],
            speaker_count_duration_strata: vec![super::PublicSpeakerCountDurationStratum {
                duration_bucket: super::PublicSpeakerCountDurationBucket::UpToThirtySeconds,
                recording_count: 1,
                posterior_recording_count: 0,
                unresolved_recording_count: 1,
                zero_reference_probability_count: 0,
                exact_speaker_count_rate: Some(1.0),
                mean_negative_log_likelihood: None,
                mean_brier_score: None,
                top_k_coverage: None,
                credible_set_coverage: None,
            }],
            count_posterior_recording_count: 0,
            count_posterior_unavailable_count: 1,
            count_unresolved_recording_count: 1,
            count_zero_reference_probability_count: 0,
            count_mean_negative_log_likelihood: None,
            count_mean_brier_score: None,
            count_top_k_coverage: None,
            count_credible_set_coverage: None,
            count_mean_entropy_bits: None,
            dominant_collapse_recording_count: 0,
            reference_collapse_recording_count: 0,
            phantom_speaker_count: 0,
            collapsed_reference_speaker_count: 0,
            mean_effective_speaker_count: Some(1.0),
            mean_dominant_speaker_share: Some(1.0),
            p90_dominant_speaker_share: Some(1.0),
            p99_dominant_speaker_share: Some(1.0),
            maximum_dominant_speaker_share: Some(1.0),
            mean_unknown_speaker_share: Some(0.0),
            maximum_unknown_speaker_share: Some(0.0),
            mean_minority_reference_recall: Some(1.0),
            reference_word_count: 0,
            scored_word_count: 0,
            correct_word_count: 0,
            incorrect_word_count: 0,
            unknown_word_count: 0,
            excluded_word_count: 0,
            micro_word_diarization_error_rate: None,
            macro_word_diarization_error_rate: None,
            selective_reference_speaker_time_sec: 1.0,
            selective_covered_speaker_time_sec: 1.0,
            selective_correct_covered_speaker_time_sec: 1.0,
            selective_error_covered_speaker_time_sec: 0.0,
            selective_unknown_speaker_time_sec: 0.0,
            selective_coverage: Some(1.0),
            selective_risk: Some(0.0),
            assignment_observed_duration_sec: 1.0,
            assignment_opportunity_duration_sec: 1.0,
            assignment_coverage: Some(1.0),
            assignment_brier_score: Some(0.0),
            assignment_expected_calibration_error: Some(0.0),
            mean_speaker_count_stability: Some(1.0),
            clustering_fallback_count: 0,
            clustering_insufficient_voice_fallback_count: 0,
            clustering_invalid_posterior_fallback_count: 0,
            clustering_unstable_count_fallback_count: 0,
            audio_duration_sec: 1.0,
            wall_time_sec: 0.1,
            real_time_factor: Some(0.1),
            sampled_peak_rss_bytes: 1,
        };
        PublicCorpusAblationVariant {
            ablation,
            feature_schema: ablation.schema_version().id().to_owned(),
            feature_schema_sha256: "0".repeat(64),
            feature_configuration_sha256: "1".repeat(64),
            splits: vec![
                split(EvaluationSplit::Development),
                split(EvaluationSplit::Test),
            ],
        }
    }

    #[test]
    fn development_gate_requires_material_der_gain_without_secondary_regression() {
        let passed = development_improvement_gate(&[
            ablation_variant(AcousticFeatureAblation::FullV2, Some(0.19), Some(0.3)),
            ablation_variant(AcousticFeatureAblation::V1, Some(0.2), Some(0.3)),
        ])
        .expect("complete evidence");
        assert!(passed.passed);
        assert!(
            passed
                .relative_micro_der_improvement
                .is_some_and(|improvement| (improvement - 0.05).abs() < 1e-12)
        );
        assert_eq!(passed.macro_jer_delta, Some(0.0));
        assert_eq!(passed.change_f1_delta, Some(0.0));

        let insufficient = development_improvement_gate(&[
            ablation_variant(AcousticFeatureAblation::FullV2, Some(0.195), Some(0.3)),
            ablation_variant(AcousticFeatureAblation::V1, Some(0.2), Some(0.3)),
        ])
        .expect("complete evidence");
        assert!(!insufficient.passed);
    }

    #[test]
    fn held_out_gate_requires_both_der_and_jer_non_regression() {
        let passed = held_out_non_regression_gate(&[
            ablation_variant(AcousticFeatureAblation::FullV2, Some(0.2), Some(0.3)),
            ablation_variant(AcousticFeatureAblation::V1, Some(0.25), Some(0.3)),
        ])
        .expect("complete evidence");
        assert!(passed.passed);
        assert!(
            passed
                .micro_der_delta
                .is_some_and(|delta| (delta + 0.05).abs() < 1e-12)
        );
        assert_eq!(passed.macro_jer_delta, Some(0.0));

        let failed = held_out_non_regression_gate(&[
            ablation_variant(AcousticFeatureAblation::FullV2, Some(0.2), Some(0.31)),
            ablation_variant(AcousticFeatureAblation::V1, Some(0.25), Some(0.3)),
        ])
        .expect("complete evidence");
        assert!(!failed.passed);
        assert!(failed.macro_jer_delta.is_some_and(|delta| delta > 0.0));
    }

    #[test]
    fn ablation_split_verifier_recomputes_count_and_word_conservation() {
        let valid = ablation_variant(AcousticFeatureAblation::FullV2, Some(0.2), Some(0.3));
        assert!(super::variant_splits_are_valid(&valid.splits, 10));

        let mut forged_count = valid.clone();
        forged_count.splits[0].exact_speaker_count_rate = Some(0.0);
        assert!(!super::variant_splits_are_valid(&forged_count.splits, 10));

        let mut forged_words = valid;
        forged_words.splits[0].reference_word_count = 1;
        assert!(!super::variant_splits_are_valid(&forged_words.splits, 10));
    }

    #[allow(clippy::too_many_arguments)]
    fn clustering_variant(
        clustering_mode: super::AcousticClusteringMode,
        micro_der: f64,
        macro_jer: f64,
        speaker_confusion_sec: f64,
        count_error: f64,
        assignment_ece: f64,
        count_stability: f64,
        fallback_count: u64,
    ) -> super::PublicCorpusClusteringVariant {
        let mut splits = ablation_variant(
            AcousticFeatureAblation::FullV2,
            Some(micro_der),
            Some(macro_jer),
        )
        .splits;
        for split in &mut splits {
            split.speaker_confusion_sec = speaker_confusion_sec;
            split.mean_absolute_speaker_count_error = Some(count_error);
            split.assignment_expected_calibration_error = Some(assignment_ece);
            split.mean_speaker_count_stability = Some(count_stability);
            split.clustering_fallback_count = fallback_count;
        }
        super::PublicCorpusClusteringVariant {
            clustering_mode,
            detector_mode: super::AcousticChangeDetectorMode::FixedSafeV1,
            feature_ablation: AcousticFeatureAblation::FullV2,
            configuration_sha256: "1".repeat(64),
            splits,
        }
    }

    #[test]
    fn clustering_development_gate_requires_accuracy_calibration_and_stability() {
        let passing_candidate = clustering_variant(
            super::AcousticClusteringMode::ProbabilisticV1,
            0.18,
            0.30,
            0.8,
            0.5,
            0.05,
            1.0,
            0,
        );
        let baseline = clustering_variant(
            super::AcousticClusteringMode::FixedSafeV1,
            0.20,
            0.30,
            1.0,
            1.0,
            0.0,
            0.0,
            0,
        );
        let passed =
            super::clustering_development_gate(&[baseline.clone(), passing_candidate.clone()])
                .expect("complete clustering evidence");
        assert!(passed.passed);
        assert_eq!(passed.candidate_fallback_count, 0);
        assert_eq!(passed.selective_coverage_regression, Some(0.0));
        assert_eq!(passed.selective_risk_delta, Some(0.0));

        let mut uncalibrated = passing_candidate.clone();
        for split in &mut uncalibrated.splits {
            split.assignment_expected_calibration_error = Some(0.11);
        }
        assert!(
            !super::clustering_development_gate(&[baseline.clone(), uncalibrated])
                .expect("uncalibrated evidence")
                .passed
        );

        let mut unstable = passing_candidate.clone();
        for split in &mut unstable.splits {
            split.mean_speaker_count_stability = Some(1.0 / 3.0);
        }
        assert!(
            !super::clustering_development_gate(&[baseline.clone(), unstable])
                .expect("unstable evidence")
                .passed
        );

        let mut coverage_regression = passing_candidate.clone();
        for split in &mut coverage_regression.splits {
            split.selective_covered_speaker_time_sec *= 0.98;
            split.selective_correct_covered_speaker_time_sec *= 0.98;
            split.selective_coverage = Some(0.98);
        }
        assert!(
            !super::clustering_development_gate(&[baseline.clone(), coverage_regression])
                .expect("coverage-regressing evidence")
                .passed
        );

        let mut risk_regression = passing_candidate.clone();
        for split in &mut risk_regression.splits {
            split.selective_error_covered_speaker_time_sec = 0.01;
            split.selective_correct_covered_speaker_time_sec =
                split.selective_covered_speaker_time_sec - 0.01;
            split.selective_risk = Some(0.01 / split.selective_covered_speaker_time_sec);
        }
        assert!(
            !super::clustering_development_gate(&[baseline.clone(), risk_regression])
                .expect("risk-regressing evidence")
                .passed
        );

        let mut fallback = passing_candidate;
        for split in &mut fallback.splits {
            split.clustering_fallback_count = 1;
        }
        assert!(
            !super::clustering_development_gate(&[baseline, fallback])
                .expect("fallback evidence")
                .passed
        );
    }

    #[test]
    fn clustering_held_out_gate_requires_non_regression() {
        let baseline = clustering_variant(
            super::AcousticClusteringMode::FixedSafeV1,
            0.20,
            0.30,
            1.0,
            1.0,
            0.0,
            0.0,
            0,
        );
        let passing = clustering_variant(
            super::AcousticClusteringMode::ProbabilisticV1,
            0.19,
            0.30,
            0.8,
            0.5,
            0.05,
            1.0,
            0,
        );
        assert!(
            super::clustering_held_out_gate(
                &[baseline.clone(), passing],
                super::AcousticClusteringMode::ProbabilisticV1,
            )
            .expect("complete held-out evidence")
            .passed
        );

        let regressing = clustering_variant(
            super::AcousticClusteringMode::ProbabilisticV1,
            0.21,
            0.30,
            0.8,
            0.5,
            0.05,
            1.0,
            0,
        );
        assert!(
            !super::clustering_held_out_gate(
                &[baseline, regressing],
                super::AcousticClusteringMode::ProbabilisticV1,
            )
            .expect("regressing held-out evidence")
            .passed
        );
    }

    fn change_detector_variant(
        detector_mode: super::AcousticChangeDetectorMode,
        change_f1: f64,
        timing_error_sec: f64,
        micro_der: f64,
        macro_jer: f64,
        brier: f64,
        ece: f64,
    ) -> super::PublicCorpusChangeDetectorVariant {
        let mut splits = ablation_variant(
            AcousticFeatureAblation::FullV2,
            Some(micro_der),
            Some(macro_jer),
        )
        .splits;
        for split in &mut splits {
            split.change_f1 = Some(change_f1);
            split.change_mean_absolute_error_sec = Some(timing_error_sec);
            split.change_brier_score = Some(brier);
            split.change_expected_calibration_error = Some(ece);
        }
        super::PublicCorpusChangeDetectorVariant {
            detector_mode,
            feature_ablation: AcousticFeatureAblation::FullV2,
            feature_schema_sha256: "0".repeat(64),
            configuration_sha256: "1".repeat(64),
            splits,
        }
    }

    #[test]
    fn development_selector_chooses_only_the_best_fully_eligible_candidate() {
        let variants = vec![
            change_detector_variant(
                super::AcousticChangeDetectorMode::CalibratedPosterior,
                0.55,
                0.10,
                0.20,
                0.30,
                0.10,
                0.05,
            ),
            change_detector_variant(
                super::AcousticChangeDetectorMode::PageHinkleyV1,
                0.65,
                0.09,
                0.20,
                0.30,
                0.10,
                0.05,
            ),
            change_detector_variant(
                super::AcousticChangeDetectorMode::BayesianTwoRegimeV1,
                0.75,
                0.08,
                0.20,
                0.30,
                0.40,
                0.05,
            ),
            change_detector_variant(
                super::AcousticChangeDetectorMode::FixedSafeV1,
                0.50,
                0.10,
                0.20,
                0.30,
                0.20,
                0.05,
            ),
        ];
        let selected = super::change_development_gate(&variants).expect("development selection");
        assert!(selected.passed);
        assert_eq!(
            selected.candidate,
            super::AcousticChangeDetectorMode::PageHinkleyV1
        );
        let held_out = super::change_held_out_gate(
            &variants,
            super::AcousticChangeDetectorMode::PageHinkleyV1,
        )
        .expect("held-out decision");
        assert!(held_out.passed);
        assert_eq!(held_out.candidate, selected.candidate);
    }

    #[test]
    fn development_selector_fails_closed_when_no_candidate_is_eligible() {
        let variants = super::AcousticChangeDetectorMode::ALL
            .into_iter()
            .map(|mode| change_detector_variant(mode, 0.50, 0.10, 0.20, 0.30, 0.20, 0.05))
            .collect::<Vec<_>>();
        let selected = super::change_development_gate(&variants).expect("development selection");
        assert!(!selected.passed);
        assert_eq!(
            selected.candidate,
            super::AcousticChangeDetectorMode::CalibratedPosterior
        );
    }

    #[test]
    fn public_ablation_runner_completes_with_path_free_aggregate_evidence() {
        let project = tempdir().expect("project");
        let input = tempdir().expect("input");
        let output = tempdir().expect("output");
        let mut recordings = Vec::new();
        for (recording_id, split, wave_frames, turn_start) in [
            ("aishell-fixture-train", "train", 32_000, "1.200"),
            (
                "aishell-fixture-development",
                "development",
                16_000,
                "0.000",
            ),
            ("aishell-fixture-test", "test", 16_000, "0.000"),
        ] {
            let audio_name = format!("{recording_id}.wav");
            let annotation_name = format!("{recording_id}.rttm");
            let audio_path = input.path().join(&audio_name);
            let annotation_path = input.path().join(&annotation_name);
            write_wave(&audio_path, 16_000, 1, wave_frames);
            std::fs::write(
                &annotation_path,
                format!(
                    "SPEAKER source-{split} 1 {turn_start} 0.600 <NA> <NA> source-speaker <NA> <NA>\n"
                ),
            )
            .expect("RTTM");
            recordings.push(json!({
                "recording_id": recording_id,
                "split": split,
                "origin_recording_id": format!("{recording_id}-origin"),
                "audio_path": audio_name,
                "audio_sha256": sha256(&audio_path),
                "expected_sample_rate_hz": 16_000,
                "expected_channel_count": 1,
                "selected_channel": 1,
                "annotation_path": annotation_name,
                "annotation_sha256": sha256(&annotation_path),
                "annotation_recording_id": format!("source-{split}"),
                "annotation_channel": "1",
                "speaker_map": {
                    "source-speaker": format!("{recording_id}-speaker")
                },
                "ignored_regions": []
            }));
        }
        let descriptor_path = input.path().join("descriptor.json");
        std::fs::write(
            &descriptor_path,
            serde_json::to_vec_pretty(&json!({
                "schema_version": PUBLIC_CORPUS_INPUT_SCHEMA_VERSION,
                "corpus_key": "aishell-4-openslr111-v1",
                "source_version": "fixture-v1",
                "recordings": recordings
            }))
            .expect("descriptor JSON"),
        )
        .expect("descriptor");
        let bundle_path = output.path().join("bundle.json");
        let evidence_path = output.path().join("evidence.json");

        let evidence = super::run_public_corpus_ablation_with_cancel(
            super::PublicCorpusAblationRequest {
                project_root: project.path(),
                input_root: input.path(),
                descriptor_path: &descriptor_path,
                bundle_output_path: &bundle_path,
                evidence_output_path: &evidence_path,
                license_acknowledgement_id: "accept-aishell-4-cc-by-sa-4.0",
                maximum_recording_duration_ms: Some(1_000),
                evaluation_stage: super::PublicCorpusEvaluationStage::Development,
                locked_development_evidence_path: None,
            },
            || false,
        )
        .expect("complete public ablation");

        assert_eq!(evidence.variants.len(), AcousticFeatureAblation::ALL.len());
        assert!(evidence.variants.iter().all(|variant| {
            variant.splits.len() == 1 && variant.splits[0].split == EvaluationSplit::Development
        }));
        assert!(
            !evidence
                .development_gate
                .as_ref()
                .expect("development gate")
                .passed
        );
        assert!(evidence.held_out_gate.is_none());
        assert_eq!(
            evidence.change_detector_variants.len(),
            super::AcousticChangeDetectorMode::ALL.len()
        );
        assert!(bundle_path.is_file());
        assert!(evidence_path.is_file());
        super::verify_public_corpus_ablation_evidence(&evidence).expect("verified evidence");
        let encoded = serde_json::to_string(&evidence).expect("aggregate evidence JSON");
        assert!(!encoded.contains(input.path().to_string_lossy().as_ref()));
        assert!(!encoded.contains(output.path().to_string_lossy().as_ref()));
        assert!(!encoded.contains("aishell-fixture-train"));
        assert!(!encoded.contains("aishell-fixture-development"));
        assert!(!encoded.contains("aishell-fixture-test"));
        let retained_development: super::PublicCorpusAblationEvidence =
            serde_json::from_slice(&std::fs::read(&evidence_path).expect("development evidence"))
                .expect("retained development JSON");
        assert!(
            retained_development == evidence,
            "retained evidence must round-trip exactly"
        );
        assert_eq!(
            super::deterministic_accuracy_sha256(&retained_development)
                .expect("retained accuracy hash"),
            retained_development.deterministic_accuracy_sha256
        );

        let certification_bundle_path = output.path().join("certification-bundle.json");
        let certification_evidence_path = output.path().join("certification-evidence.json");
        let certification_error = super::run_public_corpus_ablation_with_cancel(
            super::PublicCorpusAblationRequest {
                project_root: project.path(),
                input_root: input.path(),
                descriptor_path: &descriptor_path,
                bundle_output_path: &certification_bundle_path,
                evidence_output_path: &certification_evidence_path,
                license_acknowledgement_id: "accept-aishell-4-cc-by-sa-4.0",
                maximum_recording_duration_ms: Some(1_000),
                evaluation_stage: super::PublicCorpusEvaluationStage::Certification,
                locked_development_evidence_path: Some(&evidence_path),
            },
            || false,
        )
        .expect_err("failed development gates must block certification before corpus access");
        assert!(
            certification_error
                .to_string()
                .contains("ablation_stage_lock")
        );
        assert!(!certification_bundle_path.exists());
        assert!(!certification_evidence_path.exists());
    }

    #[test]
    fn ablation_evidence_round_trips_and_detects_metric_tampering() {
        let scorer_config = crate::diarization::DiarizationScorerConfig {
            schema_version: crate::diarization::DIARIZATION_SCORER_CONFIG_SCHEMA_VERSION.to_owned(),
            speaker_boundary_collar_ms: 250,
            change_boundary_collar_ms: 250,
            overlap_policy: crate::diarization::EvaluationOverlapPolicy::Exclude,
            calibration_bins: 10,
            count_top_k: 3,
            count_credible_mass_millionths: 900_000,
            dominant_speaker_collapse_share_millionths: 990_000,
            minimum_reference_speaker_recall_millionths: 100_000,
            minimum_effective_occupancy_ms: 250,
        };
        let diarization_request = crate::model::DiarizationRequest {
            engine: crate::model::DiarizationEngine::Acoustic,
            speaker_count: crate::model::SpeakerCountRequest::Infer,
            ..crate::model::DiarizationRequest::default()
        };
        let diarization_request_sha256 =
            super::canonical_sha256(&diarization_request).expect("request hash");
        let variants = AcousticFeatureAblation::ALL
            .into_iter()
            .map(|ablation| {
                let mut variant = ablation_variant(ablation, Some(0.2), Some(0.3));
                variant
                    .splits
                    .retain(|split| split.split == EvaluationSplit::Development);
                let schema_hash =
                    crate::diarization::acoustic_feature_schema_sha256(ablation.schema_version());
                variant.feature_schema_sha256 = schema_hash.clone();
                variant.feature_configuration_sha256 =
                    super::canonical_sha256(&super::PublicFeatureConfigurationFingerprint {
                        runner_version: super::PUBLIC_CORPUS_ABLATION_RUNNER_VERSION,
                        ablation,
                        feature_schema_sha256: &schema_hash,
                        diarization_request_sha256: &diarization_request_sha256,
                        change_calibration_sha256: &super::acoustic_change_calibration_sha256(),
                    })
                    .expect("feature configuration hash");
                variant
            })
            .collect::<Vec<_>>();
        let change_calibration_sha256 = super::acoustic_change_calibration_sha256();
        let full_v2_splits = variants
            .iter()
            .find(|variant| variant.ablation == AcousticFeatureAblation::FullV2)
            .expect("full v2")
            .splits
            .clone();
        let change_detector_variants = super::AcousticChangeDetectorMode::ALL
            .into_iter()
            .map(|detector_mode| {
                let schema_hash = crate::diarization::acoustic_feature_schema_sha256(
                    AcousticFeatureAblation::FullV2.schema_version(),
                );
                let configuration_sha256 =
                    super::canonical_sha256(&super::PublicChangeConfigurationFingerprint {
                        runner_version: super::PUBLIC_CORPUS_ABLATION_RUNNER_VERSION,
                        detector_mode,
                        feature_ablation: AcousticFeatureAblation::FullV2,
                        feature_schema_sha256: &schema_hash,
                        diarization_request_sha256: &diarization_request_sha256,
                        change_calibration_sha256: &change_calibration_sha256,
                    })
                    .expect("change configuration hash");
                super::PublicCorpusChangeDetectorVariant {
                    detector_mode,
                    feature_ablation: AcousticFeatureAblation::FullV2,
                    feature_schema_sha256: schema_hash,
                    configuration_sha256,
                    splits: full_v2_splits.clone(),
                }
            })
            .collect::<Vec<_>>();
        let development_gate = development_improvement_gate(&variants)
            .expect("development gate from complete variants");
        let change_development_gate = super::change_development_gate(&change_detector_variants)
            .expect("change development gate");
        let speaker_pair_calibration_sha256 =
            crate::diarization::acoustic_speaker_pair_calibration_sha256();
        let clustering_variants = super::AcousticClusteringMode::ALL
            .into_iter()
            .map(|clustering_mode| {
                let schema_hash = crate::diarization::acoustic_feature_schema_sha256(
                    AcousticFeatureAblation::FullV2.schema_version(),
                );
                let configuration_sha256 =
                    super::canonical_sha256(&super::PublicClusteringConfigurationFingerprint {
                        runner_version: super::PUBLIC_CORPUS_ABLATION_RUNNER_VERSION,
                        clustering_mode,
                        detector_mode: super::AcousticChangeDetectorMode::FixedSafeV1,
                        feature_ablation: AcousticFeatureAblation::FullV2,
                        feature_schema_sha256: &schema_hash,
                        diarization_request_sha256: &diarization_request_sha256,
                        speaker_pair_calibration_sha256: &speaker_pair_calibration_sha256,
                    })
                    .expect("clustering configuration hash");
                super::PublicCorpusClusteringVariant {
                    clustering_mode,
                    detector_mode: super::AcousticChangeDetectorMode::FixedSafeV1,
                    feature_ablation: AcousticFeatureAblation::FullV2,
                    configuration_sha256,
                    splits: full_v2_splits.clone(),
                }
            })
            .collect::<Vec<_>>();
        let clustering_development_gate = super::clustering_development_gate(&clustering_variants)
            .expect("clustering development gate");
        let mut evidence = super::PublicCorpusAblationEvidence {
            schema_version: super::PUBLIC_CORPUS_ABLATION_SCHEMA_VERSION.to_owned(),
            runner_version: super::PUBLIC_CORPUS_ABLATION_RUNNER_VERSION.to_owned(),
            scorer_version: crate::diarization::DIARIZATION_SCORER_VERSION.to_owned(),
            corpus_key: "ami-scenario-v1".to_owned(),
            source_version: "fixture-v1".to_owned(),
            bundle_sha256: "2".repeat(64),
            descriptor_sha256: "3".repeat(64),
            scorer_config_sha256: super::canonical_sha256(&scorer_config).expect("scorer hash"),
            evaluation_stage: super::PublicCorpusEvaluationStage::Development,
            locked_development_result_sha256: None,
            locked_development_accuracy_sha256: None,
            selected_change_detector_mode: if change_development_gate.passed {
                change_development_gate.candidate
            } else {
                super::AcousticChangeDetectorMode::FixedSafeV1
            },
            selected_clustering_mode: if clustering_development_gate.passed {
                clustering_development_gate.candidate
            } else {
                super::AcousticClusteringMode::FixedSafeV1
            },
            scorer_config,
            protocol: super::PublicCorpusAblationProtocol {
                oracle_vad: true,
                oracle_speaker_count: false,
                maximum_recording_duration_ms: Some(1_000),
                prefix_selection: "deterministic-prefix-v1".to_owned(),
                rss_observation: "linux-vmhwm-otherwise-sampled-process-rss-v1".to_owned(),
                diarization_request,
                diarization_request_sha256,
                change_calibration_id: super::ACOUSTIC_CHANGE_CALIBRATION_VERSION.to_owned(),
                change_calibration_fit_id: super::ACOUSTIC_CHANGE_CALIBRATION_FIT_VERSION
                    .to_owned(),
                change_calibration_sha256,
                change_decision_probability: super::canonical_evidence_number(f64::from(
                    super::acoustic_change_calibration().decision_probability,
                )),
                change_calibration_bins: 10,
                change_selection_policy_id: super::PUBLIC_CORPUS_CHANGE_SELECTION_POLICY_VERSION
                    .to_owned(),
                change_selection_policy_sha256: super::change_selection_policy_sha256()
                    .expect("selection policy hash"),
                speaker_pair_calibration_id:
                    crate::diarization::ACOUSTIC_CLUSTERING_PROBABILISTIC_VERSION.to_owned(),
                speaker_pair_calibration_sha256,
            },
            variants,
            change_detector_variants,
            clustering_variants,
            development_gate: Some(development_gate),
            held_out_gate: None,
            change_development_gate: Some(change_development_gate),
            change_held_out_gate: None,
            clustering_development_gate: Some(clustering_development_gate),
            clustering_held_out_gate: None,
            deterministic_accuracy_sha256: String::new(),
            result_sha256: String::new(),
        };
        evidence.deterministic_accuracy_sha256 =
            super::deterministic_accuracy_sha256(&evidence).expect("accuracy hash");
        evidence.result_sha256 = super::canonical_sha256(&evidence).expect("result hash");
        let bytes = serde_json::to_vec(&evidence).expect("evidence JSON");
        assert_eq!(
            super::parse_public_corpus_ablation_evidence(&bytes).expect("valid evidence"),
            evidence
        );
        let mut different_host = evidence.clone();
        different_host.variants[0].splits[0].wall_time_sec = 999.0;
        different_host.variants[0].splits[0].real_time_factor = Some(999.0);
        different_host.variants[0].splits[0].sampled_peak_rss_bytes = u64::MAX;
        assert_eq!(
            super::deterministic_accuracy_sha256(&different_host)
                .expect("host-independent accuracy hash"),
            evidence.deterministic_accuracy_sha256
        );

        let mut tampered: serde_json::Value =
            serde_json::from_slice(&bytes).expect("evidence JSON value");
        tampered["variants"][0]["splits"][0]["micro_der"] = json!(0.9);
        let error = super::parse_public_corpus_ablation_evidence(
            &serde_json::to_vec(&tampered).expect("tampered JSON"),
        )
        .expect_err("tampered metric");
        assert!(
            error.to_string().contains("ablation_gate_mismatch")
                || error.to_string().contains("ablation_variant_contract")
                || error.to_string().contains("change_detector_alignment")
                || error
                    .to_string()
                    .contains("ablation_accuracy_hash_mismatch")
                || error.to_string().contains("ablation_hash_mismatch")
        );
    }
}
