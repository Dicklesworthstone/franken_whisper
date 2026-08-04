//! Reproducible adapters for public or user-licensed diarization corpora.
//!
//! This module deliberately separates path-bearing local preparation inputs
//! from the path-free corpus, reference, leakage, and integrity evidence that
//! can be retained externally. It never copies source media and refuses to
//! write generated annotations inside the project checkout.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
use std::ffi::OsString;
#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
use std::io::{BufWriter, Write};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
use uuid::Uuid;

use crate::diarization::{
    ACOUSTIC_CHANGE_CALIBRATION_FIT_VERSION, ACOUSTIC_CHANGE_CALIBRATION_VERSION,
    ACOUSTIC_CLUSTERING_PROBABILISTIC_VERSION, ACOUSTIC_SIDECAR_FUSION_VERSION,
    ACOUSTIC_SIDECAR_STUDY_SCHEMA_VERSION, AcousticBoundaryHints, AcousticChangeDetectorMode,
    AcousticClusteringEvaluationEvidence, AcousticClusteringFallbackReason, AcousticClusteringMode,
    AcousticCountMergeStepEvidence, AcousticDiarizationInput, AcousticFeatureAblation,
    AcousticScatteringMode, AcousticSidecarEvaluationRequest, AcousticSidecarFusionCalibration,
    AcousticSidecarFusionEvaluationEvidence, AcousticSidecarStudy, AcousticSidecarStudyConfig,
    AcousticSidecarStudyMode, AcousticSidecarStudyObservation, AcousticTrajectoryWaveletMode,
    ChangePointScore, DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION,
    DIARIZATION_HYPOTHESIS_SCHEMA_VERSION, DIARIZATION_REFERENCE_SCHEMA_VERSION,
    DIARIZATION_SCORER_VERSION, DiarizationCorpusManifest, DiarizationHypothesisDocument,
    DiarizationLeakageAudit, DiarizationReferenceDocument, DiarizationScorerConfig,
    EvaluationOverlapPolicy, EvaluationPerformanceObservation, EvaluationRegion, EvaluationSplit,
    EvaluationTurn, EvaluationWord, acoustic_change_calibration,
    acoustic_change_calibration_sha256, acoustic_feature_schema_sha256,
    acoustic_sidecar_calibrate_owner_contrast, acoustic_sidecar_fusion_configuration_sha256,
    acoustic_sidecar_observation_owner_contrast_from_study, acoustic_sidecar_study_config_sha256,
    acoustic_speaker_pair_calibration_sha256, audit_diarization_manifest,
    diarize_acoustic_pcm_with_modes_evidence, diarize_acoustic_pcm_with_sidecar_evidence,
    extract_acoustic_features_with_frames, parse_diarization_corpus_manifest,
    parse_diarization_reference, score_change_points, score_diarization_documents,
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
pub const PUBLIC_CORPUS_ABLATION_SCHEMA_VERSION: &str = "public-diarization-acoustic-ablation-v10";
/// Frozen public ablation implementation identity.
pub const PUBLIC_CORPUS_ABLATION_RUNNER_VERSION: &str =
    "public-diarization-acoustic-ablation-runner-v10";
/// Schema identity for the separate aggregate-only acoustic sidecar study.
pub const PUBLIC_CORPUS_SIDECAR_STUDY_SCHEMA_VERSION: &str =
    "public-diarization-acoustic-sidecar-study-v3";
/// Frozen implementation identity for the aggregate-only sidecar runner.
pub const PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION: &str =
    "public-diarization-acoustic-sidecar-study-runner-v3";
/// Identity of the bounded development calibration fit.
pub const PUBLIC_CORPUS_SIDECAR_CALIBRATION_FIT_VERSION: &str =
    "public-sidecar-boundary-calibration-empirical-grid-v2";
/// Identity of the separately fitted lagged-pair calibration.
pub const PUBLIC_CORPUS_SIDECAR_PAIR_CALIBRATION_FIT_VERSION: &str =
    "public-sidecar-pair-calibration-empirical-grid-v1";
/// Target represented by the separately calibrated lagged-pair probability.
pub const PUBLIC_CORPUS_SIDECAR_PAIR_PROBABILITY_TARGET_VERSION: &str =
    "public-sidecar-different-speaker-given-selected-comparable-frozen-lag-pair-v1";
/// Lane-independent reference-labeled universe sampled before feature availability.
pub const PUBLIC_CORPUS_SIDECAR_PAIR_POPULATION_VERSION: &str =
    "public-sidecar-reference-labeled-frozen-lag-pair-population-v1";
/// Identity of the score-independent deterministic bottom-k selection key.
pub const PUBLIC_CORPUS_SIDECAR_PAIR_SELECTION_KEY_VERSION: &str =
    "public-sidecar-conditional-pair-bottom-k-normalized-pcm-sha256-v3";
/// Identity of the lane-independent selected-pair sequence digest.
pub const PUBLIC_CORPUS_SIDECAR_PAIR_SELECTION_DIGEST_VERSION: &str =
    "public-sidecar-reference-labeled-selected-pair-sequence-sha256-v2";
/// Identity of the deterministic conditional-pair scorer.
pub const PUBLIC_CORPUS_SIDECAR_PAIR_SCORER_VERSION: &str =
    "public-sidecar-conditional-pair-calibrated-v3";
/// Identity of the deterministic paired-recording uncertainty calculation.
pub const PUBLIC_CORPUS_SIDECAR_UNCERTAINTY_VERSION: &str =
    "public-sidecar-paired-bootstrap-splitmix64-v2";
/// Identity of the fail-closed development selector and held-out gate.
pub const PUBLIC_CORPUS_SIDECAR_SELECTION_POLICY_VERSION: &str =
    "public-sidecar-selection-policy-v3";
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
/// Maximum rate at which an auxiliary contrast dominates Voice on selected
/// same-speaker pairs where both owners are available.
pub const PUBLIC_CORPUS_SIDECAR_MAX_SAME_SPEAKER_AUXILIARY_DOMINANCE_RATE: f64 = 0.50;
/// Every expected auxiliary owner needs substantial same-speaker support.
pub const PUBLIC_CORPUS_SIDECAR_MIN_AUXILIARY_DOMINANCE_OPPORTUNITIES: u64 = 100;
/// At least one quarter of the frozen reference-pair sample must be scored.
pub const PUBLIC_CORPUS_SIDECAR_MIN_PAIR_SCORE_COVERAGE: f64 = 0.25;
/// Each conditional class needs substantial aggregate support before promotion.
pub const PUBLIC_CORPUS_SIDECAR_MIN_PAIRS_PER_CLASS: u64 = 100;
/// Conditional-pair evidence must span at least five recordings.
pub const PUBLIC_CORPUS_SIDECAR_MIN_PAIR_RECORDINGS: u64 = 5;
/// Maximum relative candidate RTF regression admitted for promotion.
pub const PUBLIC_CORPUS_SIDECAR_MAX_RELATIVE_RTF_REGRESSION: f64 = 0.25;
/// Maximum relative sampled peak-RSS regression admitted for promotion.
pub const PUBLIC_CORPUS_SIDECAR_MAX_RELATIVE_RSS_REGRESSION: f64 = 0.25;

const PUBLIC_SIDECAR_BOUNDARY_COLLAR_MS: u64 = 250;
const PUBLIC_SIDECAR_RELIABILITY_BINS: usize = 10;
const PUBLIC_SIDECAR_FIT_BINS: usize = 256;
const PUBLIC_SIDECAR_PAIR_SCORE_BINS: usize = 100;
const PUBLIC_SIDECAR_PAIR_LAGS_FRAMES: [usize; 4] = [25, 50, 100, 200];
const PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING: usize = 4_096;
const PUBLIC_SIDECAR_MAX_RETAINED_PAIR_SAMPLE_CAPACITY: usize = 8_192;
const PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS: usize = 1;
const PUBLIC_SIDECAR_BOOTSTRAP_REPLICATES: usize = 2_000;
const PUBLIC_SIDECAR_BOOTSTRAP_SEED_POLICY: &str = "fixed-lane-split-bootstrap-seed-v2";
const PUBLIC_SIDECAR_BOOTSTRAP_SAMPLER: &str = "splitmix64-per-replicate-stream-v1";
const PUBLIC_SIDECAR_MAX_RETAINED_SIGNALS: u64 = 401;
const PUBLIC_SIDECAR_MAX_RETAINED_SIGNAL_CAPACITY: u64 = 1_024;
const PUBLIC_SIDECAR_MAX_REPORTED_PAYLOAD_BYTES: u64 = 16 * 1024 * 1024;
const PUBLIC_SIDECAR_MINIMUM_PAIRED_RECORDINGS: u64 = 5;
const PUBLIC_OUTPUT_CANCELLATION_GRANULARITY_BYTES: usize = 64 * 1024;
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

/// Aggregate diagnostic at the reference-count frontier of the full-feature
/// probabilistic agglomeration lane.
///
/// A well-ordered hierarchy should assign a higher same-speaker probability to
/// the last merge reaching the reference count than to the first merge below
/// it. This is diagnostic evidence only; the reference count is never passed
/// into the diarizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicSpeakerCountMergeFrontier {
    pub recording_count: u64,
    pub to_reference_count_observation_count: u64,
    pub mean_probability_to_reference_count: Option<f64>,
    pub below_reference_count_observation_count: u64,
    pub mean_probability_below_reference_count: Option<f64>,
    pub paired_frontier_observation_count: u64,
    pub mean_probability_margin: Option<f64>,
    pub correctly_ordered_frontier_count: u64,
    pub correctly_ordered_frontier_rate: Option<f64>,
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
    /// Reference count versus the concrete posterior MAP before abstention.
    pub speaker_count_posterior_map_confusion: Vec<PublicSpeakerCountConfusionCell>,
    pub speaker_count_merge_frontier: PublicSpeakerCountMergeFrontier,
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

    const fn uses_frame_wavelet(self) -> bool {
        matches!(
            self,
            Self::FrameHaarL4
                | Self::FrameD4L4
                | Self::FrameHaarL4AndModulation
                | Self::FrameD4L4AndModulation
                | Self::AllHaarL4
                | Self::AllD4L4
        )
    }

    const fn uses_modulation(self) -> bool {
        matches!(
            self,
            Self::Modulation
                | Self::FrameHaarL4AndModulation
                | Self::FrameD4L4AndModulation
                | Self::AllHaarL4
                | Self::AllD4L4
        )
    }

    const fn uses_trajectory_wavelet(self) -> bool {
        matches!(
            self,
            Self::TrajectoryHaarL4 | Self::TrajectoryD4L4 | Self::AllHaarL4 | Self::AllD4L4
        )
    }

    const fn uses_scattering(self) -> bool {
        matches!(
            self,
            Self::ScatteringFirstOrder
                | Self::ScatteringSecondOrder
                | Self::ScatteringFirstAndSecondOrder
                | Self::AllHaarL4
                | Self::AllD4L4
        )
    }

    /// Expected Voice-versus-Channel and Voice-versus-MixedAuxiliary checks.
    const fn auxiliary_dominance_expectations(self) -> [bool; 2] {
        match self {
            Self::FullV2Baseline | Self::FrameHaarL4 | Self::FrameD4L4 => [false, false],
            Self::Modulation => [true, false],
            Self::FrameHaarL4AndModulation
            | Self::FrameD4L4AndModulation
            | Self::TrajectoryHaarL4
            | Self::TrajectoryD4L4
            | Self::ScatteringFirstOrder
            | Self::ScatteringSecondOrder
            | Self::ScatteringFirstAndSecondOrder
            | Self::AllHaarL4
            | Self::AllD4L4 => [true, true],
        }
    }
}

/// Whether a lane is the unfused control or an opt-in boundary-fusion pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublicCorpusSidecarFusionScope {
    BaselineUnfused,
    BoundaryFusionV2,
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
    InsufficientPairCoverage,
    InsufficientPairSupport,
    MissingConditionalPairs,
    PairDiscrimination,
    PairBrier,
    PairCalibration,
    AuxiliaryConfound,
    SpeakerCountRegression,
    PerformanceRegression,
    PairedDerUncertainty,
    NotSelectedByRanking,
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
    pub maximum_same_speaker_auxiliary_dominance_rate: f64,
    pub minimum_same_speaker_auxiliary_dominance_opportunities: u64,
    pub minimum_pair_score_coverage: f64,
    pub minimum_conditional_pairs_per_class: u64,
    pub minimum_conditional_pair_recording_count: u64,
    pub minimum_paired_recording_count: u64,
    pub maximum_relative_rtf_regression: f64,
    pub maximum_relative_rss_regression: f64,
    pub require_boundary_f1_non_regression: bool,
    pub require_speaker_count_non_regression: bool,
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
    pub feature_schema_sha256: String,
    pub detector_mode: AcousticChangeDetectorMode,
    pub clustering_mode: AcousticClusteringMode,
    pub change_calibration_id: String,
    pub change_calibration_fit_id: String,
    pub change_calibration_sha256: String,
    pub change_decision_probability: f64,
    pub speaker_pair_calibration_id: String,
    pub speaker_pair_calibration_sha256: String,
    pub sidecar_schema_id: String,
    pub fusion_id: String,
    pub calibration_fit_id: String,
    pub pair_calibration_fit_id: String,
    pub pair_probability_target_id: String,
    pub pair_population_id: String,
    pub pair_selection_key_id: String,
    pub pair_selection_digest_id: String,
    pub pair_scorer_id: String,
    pub uncertainty_id: String,
    pub uncertainty_seed_policy_id: String,
    pub uncertainty_sampler_id: String,
    pub selection_policy_id: String,
    pub selection_policy_sha256: String,
    pub boundary_collar_ms: u64,
    pub reliability_bins: usize,
    pub pair_score_bins: usize,
    pub pair_lags_frames: [usize; 4],
    pub maximum_pairs_per_recording: usize,
    pub maximum_retained_pair_sample_capacity: usize,
    pub paired_bootstrap_replicates: usize,
    pub maximum_retained_signal_capacity: usize,
    pub maximum_reported_payload_bytes: u64,
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

/// Development-fitted mapping from a frozen lagged-pair contrast to the
/// empirical probability that the pair belongs to different speakers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarPairCalibration {
    pub fit_id: String,
    pub target_id: String,
    pub logit_intercept: f64,
    pub contrast_weight: f64,
    pub minimum_comparable_components: usize,
    pub fit_observation_count: u64,
    pub fit_positive_count: u64,
    pub fit_brier_score: Option<f64>,
    /// SHA-256 of the calibration fingerprint with this field excluded.
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
    pub probability_sum: f64,
    pub positive_probability_sum: f64,
    pub squared_probability_sum: f64,
    pub positive_squared_probability_sum: f64,
    pub squared_error_sum: f64,
    pub mean_probability: Option<f64>,
    pub empirical_frequency: Option<f64>,
}

/// One fixed score-order bin used to recompute conditional-pair ROC AUC.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarPairScoreBin {
    pub index: usize,
    pub lower_probability: f64,
    pub upper_probability: f64,
    pub same_speaker_count: u64,
    pub different_speaker_count: u64,
    pub probability_sum: f64,
    pub different_speaker_probability_sum: f64,
    pub squared_probability_sum: f64,
    pub different_speaker_squared_probability_sum: f64,
    pub squared_error_sum: f64,
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
    pub mean_different_probability_given_same_speaker: Option<f64>,
    pub mean_different_probability_given_different_speaker: Option<f64>,
    pub roc_auc: Option<f64>,
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub reliability: Vec<PublicCorpusSidecarReliabilityBin>,
    pub score_histogram: Vec<PublicCorpusSidecarPairScoreBin>,
}

/// Aggregate dominance of one auxiliary owner over Voice on selected pairs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarAuxiliaryDominanceMetrics {
    pub same_speaker_opportunity_count: u64,
    pub same_speaker_dominance_count: u64,
    pub same_speaker_dominance_rate: Option<f64>,
    pub different_speaker_opportunity_count: u64,
    pub different_speaker_dominance_count: u64,
    pub different_speaker_dominance_rate: Option<f64>,
}

/// Aggregate availability, sampled-pair coverage, and auxiliary dominance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicCorpusSidecarCoverage {
    pub fusion_requested: bool,
    pub evaluated_recording_count: u64,
    pub fusion_requested_recording_count: u64,
    pub fusion_executed_recording_count: u64,
    pub submitted_frame_count: u64,
    pub comparable_frame_count: u64,
    pub calibrated_signal_count: u64,
    pub consumed_probability_count: u64,
    pub changed_boundary_probability_count: u64,
    pub comparable_frame_coverage: Option<f64>,
    pub component_comparison_count: u64,
    /// Canonical Voice, Channel, MixedAuxiliary order.
    pub owner_available_frame_counts: [u64; 3],
    pub channel_dominance: PublicCorpusSidecarAuxiliaryDominanceMetrics,
    pub mixed_auxiliary_dominance: PublicCorpusSidecarAuxiliaryDominanceMetrics,
    pub eligible_pair_count: u64,
    pub retained_pair_sample_count: u64,
    pub retained_same_speaker_pair_count: u64,
    pub retained_different_speaker_pair_count: u64,
    pub pair_selection_sha256: Option<String>,
    pub pair_score_coverage: Option<f64>,
    pub same_speaker_pair_score_coverage: Option<f64>,
    pub different_speaker_pair_score_coverage: Option<f64>,
    pub pair_scored_recording_count: u64,
    pub same_speaker_pair_recording_count: u64,
    pub different_speaker_pair_recording_count: u64,
    pub maximum_retained_pair_sample_count: u64,
    pub retained_pair_sample_capacity: u64,
    pub maximum_retained_signal_count: u64,
    pub retained_signal_capacity: u64,
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
    pub paired_der_recording_count: u64,
    pub paired_jer_recording_count: u64,
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
    pub pair_score_coverage: Option<f64>,
    pub same_speaker_pair_score_coverage: Option<f64>,
    pub different_speaker_pair_score_coverage: Option<f64>,
    pub pair_scored_recording_count: u64,
    pub pair_roc_auc: Option<f64>,
    pub pair_brier_score: Option<f64>,
    pub pair_expected_calibration_error: Option<f64>,
    pub channel_same_speaker_dominance_rate: Option<f64>,
    pub mixed_auxiliary_same_speaker_dominance_rate: Option<f64>,
    pub exact_speaker_count_rate_delta: Option<f64>,
    pub mean_absolute_speaker_count_error_delta: Option<f64>,
    pub dominant_collapse_count_delta: Option<i64>,
    pub relative_rtf_regression: Option<f64>,
    pub relative_rss_regression: Option<f64>,
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
    pub pair_calibration: Option<PublicCorpusSidecarPairCalibration>,
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
    /// Hash with host timing, RSS, allocator-capacity, and target-sized
    /// retained-state diagnostics normalized away. Accuracy-derived gate and
    /// selection decisions remain bound; performance-only consequences do not.
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
struct PublicCorpusSidecarPairCalibrationFingerprint<'a> {
    fit_id: &'a str,
    target_id: &'a str,
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
    pair_calibration_sha256: Option<&'a str>,
    protocol_sha256: &'a str,
}

#[derive(Serialize)]
struct PublicCorpusSidecarSelectionFingerprint<'a> {
    policy_id: &'a str,
    accuracy_ranking: &'a str,
    operational_gate_behavior: &'a str,
    lane_order: &'a [PublicCorpusSidecarLane],
    gate_policy: &'a PublicCorpusSidecarGatePolicy,
}

#[derive(Serialize)]
struct PublicCorpusSidecarBootstrapSeedFingerprint<'a> {
    uncertainty_id: &'a str,
    seed_policy_id: &'a str,
    sampler_id: &'a str,
    lane: PublicCorpusSidecarLane,
    split: EvaluationSplit,
    replicates: usize,
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

/// Parse and fully validate aggregate-only acoustic-sidecar evidence.
pub fn parse_public_corpus_sidecar_study_evidence(
    bytes: &[u8],
) -> FwResult<PublicCorpusSidecarStudyEvidence> {
    let evidence = serde_json::from_slice(bytes).map_err(|_| {
        public_corpus_error(
            "sidecar_study_json",
            "sidecar study evidence must be valid JSON without trailing data",
        )
    })?;
    verify_public_corpus_sidecar_study_evidence(&evidence)?;
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

fn validate_public_corpus_descriptor_metadata(
    descriptor: &PublicCorpusInput,
    registry_entry: &PublicCorpusRegistryEntry,
) -> FwResult<()> {
    let mut recording_ids = BTreeSet::new();
    let mut manifest_recordings = Vec::with_capacity(descriptor.recordings.len());
    for recording in &descriptor.recordings {
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
        validate_relative_path_syntax(&recording.audio_path, "audio")?;
        validate_relative_path_syntax(&recording.annotation_path, "annotation")?;
        match (
            recording.word_annotation_path.as_ref(),
            recording.word_annotation_sha256.as_ref(),
        ) {
            (Some(path), Some(hash)) => {
                validate_relative_path_syntax(path, "word_annotation")?;
                validate_sha256(hash, "word_annotation_sha256")?;
            }
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
        let speaker_refs = recording
            .speaker_map
            .values()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut derived_from_recording_ids = recording.derived_from_recording_ids.clone();
        let mut enrollment_recording_ids = recording.enrollment_recording_ids.clone();
        derived_from_recording_ids.sort();
        enrollment_recording_ids.sort();
        manifest_recordings.push(crate::diarization::CorpusRecordingManifest {
            recording_id: recording.recording_id.clone(),
            split: recording.split,
            origin_recording_id: recording.origin_recording_id.clone(),
            speaker_refs,
            derived_from_recording_ids,
            augmentation_group_id: recording.augmentation_group_id.clone(),
            enrollment_recording_ids,
        });
    }
    let manifest = DiarizationCorpusManifest {
        schema_version: DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION.to_owned(),
        corpus_id: descriptor.corpus_key.clone(),
        license_id: registry_entry.license_id.clone(),
        recordings: manifest_recordings,
    };
    parse_diarization_corpus_manifest(&serde_json::to_vec(&manifest)?)?;
    if !audit_diarization_manifest(&manifest)?.passed {
        return Err(public_corpus_error(
            "split_leakage",
            "descriptor metadata violates the frozen cross-split leakage contract",
        ));
    }
    Ok(())
}

/// Build one path-free bundle from external WAV and RTTM inputs.
///
/// On Linux, Android, and Apple platforms, the complete output is privately
/// staged and published with an identity-bound no-clobber rename. Its parent
/// must be owned by the effective user and not group/world writable. Other
/// targets return `FwError::Unsupported` with `public_corpus.output_platform`.
/// All source/output roots must be absolute, canonical, and disjoint from the
/// project checkout.
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
    is_cancelled: impl FnMut() -> bool,
) -> FwResult<PublicCorpusBundle> {
    build_public_corpus_bundle_for_split_with_cancel(
        project_root,
        input_root,
        descriptor_path,
        output_path,
        license_acknowledgement_id,
        None,
        None,
        is_cancelled,
    )
}

/// Sidecar-only materialization seam. Descriptor metadata for every split is
/// validated up front, but only `selected_split` media and annotations are
/// opened. An expected descriptor digest closes the preflight/materialization
/// race before any selected media is accessed. A `None` split and digest
/// preserve the public bundle adapter's original all-split behavior and bytes.
#[allow(clippy::too_many_arguments)]
fn build_public_corpus_bundle_for_split_with_cancel(
    project_root: &Path,
    input_root: &Path,
    descriptor_path: &Path,
    output_path: &Path,
    license_acknowledgement_id: &str,
    selected_split: Option<EvaluationSplit>,
    expected_descriptor_sha256: Option<&str>,
    mut is_cancelled: impl FnMut() -> bool,
) -> FwResult<PublicCorpusBundle> {
    let canonical_project = canonical_directory(project_root, "project_root")?;
    let canonical_input = canonical_directory(input_root, "input_root")?;
    if paths_overlap(&canonical_project, &canonical_input) {
        return Err(public_corpus_error(
            "input_root_overlap",
            "input root must be disjoint from the project checkout",
        ));
    }
    let output_parent = validate_new_output(&canonical_project, &canonical_input, output_path)?;
    let bundle = materialize_public_corpus_bundle_for_split_with_cancel(
        project_root,
        input_root,
        descriptor_path,
        license_acknowledgement_id,
        selected_split,
        expected_descriptor_sha256,
        &mut is_cancelled,
    )?;
    write_new_json(
        output_path,
        &output_parent,
        &bundle,
        "public-corpus bundle",
        &mut is_cancelled,
    )?;
    Ok(bundle)
}

#[allow(clippy::too_many_arguments)]
fn materialize_public_corpus_bundle_for_split_with_cancel(
    project_root: &Path,
    input_root: &Path,
    descriptor_path: &Path,
    license_acknowledgement_id: &str,
    selected_split: Option<EvaluationSplit>,
    expected_descriptor_sha256: Option<&str>,
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
    checkpoint_cancelled(&mut is_cancelled)?;

    let descriptor_bytes = read_bounded(&canonical_descriptor, MAX_DESCRIPTOR_BYTES, "descriptor")?;
    let descriptor_sha256 = format!("{:x}", Sha256::digest(&descriptor_bytes));
    if expected_descriptor_sha256.is_some_and(|expected| expected != descriptor_sha256) {
        return Err(public_corpus_error(
            "sidecar_descriptor_changed",
            "descriptor changed after the stage lock preflight",
        ));
    }
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
    if selected_split.is_some() {
        validate_public_corpus_descriptor_metadata(&descriptor, registry_entry)?;
    }
    if let Some(selected_split) = selected_split {
        descriptor
            .recordings
            .retain(|recording| recording.split == selected_split);
        if descriptor.recordings.is_empty() {
            return Err(public_corpus_error(
                "sidecar_split_missing",
                "the selected evaluation stage has no recording in the descriptor",
            ));
        }
    }

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
    Ok(bundle)
}

/// Build, execute, score, and retain one aggregate-only public feature
/// ablation. Source media and per-recording hypotheses never leave
/// `input_root` or process memory.
///
/// Bundle/evidence artifact publication requires Linux, Android, or an Apple
/// platform and an effective-user-owned parent that is not group/world
/// writable. Other targets fail before corpus materialization with
/// `public_corpus.output_platform`.
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
    if bundle_output_parent.same_output_target(
        bundle_output_path,
        &evidence_output_parent,
        evidence_output_path,
        "ablation_output",
    )? {
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
    let bundle = materialize_public_corpus_bundle_for_split_with_cancel(
        project_root,
        input_root,
        descriptor_path,
        license_acknowledgement_id,
        None,
        None,
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
        corpus_key: bundle.corpus_key.clone(),
        source_version: bundle.source_version.clone(),
        bundle_sha256: bundle.bundle_sha256.clone(),
        descriptor_sha256: bundle.descriptor_sha256.clone(),
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
    let staged_bundle = stage_new_json(
        bundle_output_path,
        &bundle_output_parent,
        &bundle,
        "public-corpus bundle",
        &mut is_cancelled,
    )?;
    let staged_evidence = match stage_new_json(
        evidence_output_path,
        &evidence_output_parent,
        &result,
        "ablation evidence",
        &mut is_cancelled,
    ) {
        Ok(staged) => staged,
        Err(error) => {
            return Err(staged_scrubbed_error(
                error,
                &[(&staged_bundle, "public-corpus bundle")],
            ));
        }
    };
    if let Err(error) = checkpoint_cancelled(&mut is_cancelled) {
        return Err(staged_scrubbed_error(
            error,
            &[
                (&staged_bundle, "public-corpus bundle"),
                (&staged_evidence, "ablation evidence"),
            ],
        ));
    }
    if let Err(error) = publish_staged_json(staged_bundle, "public-corpus bundle") {
        return Err(staged_scrubbed_error(
            error,
            &[(&staged_evidence, "ablation evidence")],
        ));
    }
    publish_staged_json(staged_evidence, "ablation evidence")?;
    Ok(result)
}

/// Run one sealed, aggregate-only public acoustic-sidecar study.
///
/// Development can inspect descriptor metadata for every split, but opens
/// only development media and annotations. Certification validates the exact
/// development lock and selected lane before opening any held-out bytes.
/// Bundle/evidence artifact publication requires Linux, Android, or an Apple
/// platform and an effective-user-owned parent that is not group/world
/// writable. Other targets fail before corpus materialization with
/// `public_corpus.output_platform`.
pub fn run_public_corpus_sidecar_study_with_cancel(
    request: PublicCorpusSidecarStudyRequest<'_>,
    mut is_cancelled: impl FnMut() -> bool,
) -> FwResult<PublicCorpusSidecarStudyEvidence> {
    let PublicCorpusSidecarStudyRequest {
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
            "sidecar_duration",
            "maximum recording duration must be positive when supplied",
        ));
    }
    match (evaluation_stage, locked_development_evidence_path) {
        (PublicCorpusEvaluationStage::Development, None)
        | (PublicCorpusEvaluationStage::Certification, Some(_)) => {}
        (PublicCorpusEvaluationStage::Development, Some(_)) => {
            return Err(public_corpus_error(
                "sidecar_stage_lock",
                "development evaluation must not receive held-out lock evidence",
            ));
        }
        (PublicCorpusEvaluationStage::Certification, None) => {
            return Err(public_corpus_error(
                "sidecar_stage_lock",
                "certification requires locked development evidence",
            ));
        }
    }
    checkpoint_cancelled(&mut is_cancelled)?;
    let canonical_project = canonical_directory(project_root, "project_root")?;
    let canonical_input = canonical_directory(input_root, "input_root")?;
    if paths_overlap(&canonical_project, &canonical_input) {
        return Err(public_corpus_error(
            "input_root_overlap",
            "input root must be disjoint from the project checkout",
        ));
    }
    let bundle_output_parent =
        validate_new_output(&canonical_project, &canonical_input, bundle_output_path)?;
    let evidence_output_parent =
        validate_new_output(&canonical_project, &canonical_input, evidence_output_path)?;
    if bundle_output_parent.same_output_target(
        bundle_output_path,
        &evidence_output_parent,
        evidence_output_path,
        "sidecar_output",
    )? {
        return Err(public_corpus_error(
            "sidecar_output",
            "bundle and sidecar evidence outputs must be distinct",
        ));
    }

    // Lock verification deliberately precedes descriptor-driven media access.
    let locked_development = if let Some(path) = locked_development_evidence_path {
        let canonical_locked = canonical_external_file(
            &canonical_project,
            &canonical_input,
            path,
            "sidecar_development_lock",
        )?;
        let bytes = read_bounded(
            &canonical_locked,
            MAX_DESCRIPTOR_BYTES,
            "sidecar_development_lock",
        )?;
        let evidence = parse_public_corpus_sidecar_study_evidence(&bytes)?;
        let selected = evidence.selected_candidate_lane.ok_or_else(|| {
            public_corpus_error(
                "sidecar_stage_lock",
                "certification requires one development-selected candidate",
            )
        })?;
        let selected_variant = evidence
            .variants
            .iter()
            .find(|variant| variant.lane == selected)
            .ok_or_else(|| {
                public_corpus_error(
                    "sidecar_stage_lock",
                    "development-selected candidate is absent from its artifact",
                )
            })?;
        if evidence.evaluation_stage != PublicCorpusEvaluationStage::Development
            || evidence.adopted_candidate_lane.is_some()
            || selected == PublicCorpusSidecarLane::FullV2Baseline
            || selected_variant.disposition
                != PublicCorpusSidecarDisposition::AdvanceToCertification
            || !selected_variant
                .gate
                .as_ref()
                .is_some_and(|gate| gate.passed)
            || selected_variant.calibration.is_none()
            || selected_variant.pair_calibration.is_none()
        {
            return Err(public_corpus_error(
                "sidecar_stage_lock",
                "certification lock has no valid development-selected candidate",
            ));
        }
        Some(evidence)
    } else {
        None
    };

    let canonical_descriptor =
        canonical_input_file(&canonical_input, descriptor_path, "descriptor")?;
    let descriptor_bytes = read_bounded(&canonical_descriptor, MAX_DESCRIPTOR_BYTES, "descriptor")?;
    let descriptor_sha256 = format!("{:x}", Sha256::digest(&descriptor_bytes));
    let descriptor: PublicCorpusInput =
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
    let protocol = public_sidecar_protocol(
        maximum_recording_duration_ms,
        &diarization_request,
        diarization_request_sha256,
    )?;
    let protocol_sha256 = canonical_sha256(&protocol)?;
    if let Some(locked) = &locked_development
        && (locked.descriptor_sha256 != descriptor_sha256
            || locked.corpus_key != descriptor.corpus_key
            || locked.source_version != descriptor.source_version
            || locked.scorer_config != scorer_config
            || locked.scorer_config_sha256 != scorer_config_sha256
            || locked.protocol != protocol
            || locked.protocol_sha256 != protocol_sha256)
    {
        return Err(public_corpus_error(
            "sidecar_stage_lock",
            "development lock does not bind the current descriptor, scorer, and protocol",
        ));
    }
    checkpoint_cancelled(&mut is_cancelled)?;

    let target_split = evaluation_stage.selected_split();
    let bundle = materialize_public_corpus_bundle_for_split_with_cancel(
        project_root,
        input_root,
        descriptor_path,
        license_acknowledgement_id,
        Some(target_split),
        Some(&descriptor_sha256),
        &mut is_cancelled,
    )?;
    if bundle.descriptor_sha256 != descriptor_sha256
        || bundle.corpus_key != descriptor.corpus_key
        || bundle.source_version != descriptor.source_version
    {
        return Err(public_corpus_error(
            "sidecar_descriptor_changed",
            "descriptor changed between lock preflight and selected-split materialization",
        ));
    }
    let input_recordings = descriptor
        .recordings
        .into_iter()
        .filter(|recording| recording.split == target_split)
        .map(|recording| (recording.recording_id.clone(), recording))
        .collect::<BTreeMap<_, _>>();
    if input_recordings.len() != bundle.references.len() {
        return Err(public_corpus_error(
            "sidecar_alignment",
            "selected descriptor and materialized bundle recording counts differ",
        ));
    }

    let mut baseline = evaluate_public_sidecar_lane(
        &bundle,
        &input_recordings,
        &canonical_input,
        maximum_recording_duration_ms,
        &diarization_request,
        &scorer_config,
        PublicCorpusSidecarLane::FullV2Baseline,
        None,
        None,
        protocol.detector_mode,
        protocol.clustering_mode,
        target_split,
        &mut is_cancelled,
    )?;
    let baseline_study_sha256 = acoustic_sidecar_study_config_sha256(
        PublicCorpusSidecarLane::FullV2Baseline.study_config(),
    )?;
    let baseline_lane_sha256 = canonical_sha256(&PublicCorpusSidecarLaneFingerprint {
        runner_version: PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION,
        lane: PublicCorpusSidecarLane::FullV2Baseline,
        fusion_scope: PublicCorpusSidecarFusionScope::BaselineUnfused,
        study_configuration_sha256: &baseline_study_sha256,
        fusion_configuration_sha256: None,
        pair_calibration_sha256: None,
        protocol_sha256: &protocol_sha256,
    })?;
    baseline.split.paired_uncertainty = None;
    let mut variants = vec![PublicCorpusSidecarStudyVariant {
        lane: PublicCorpusSidecarLane::FullV2Baseline,
        fusion_scope: PublicCorpusSidecarFusionScope::BaselineUnfused,
        study_configuration_sha256: baseline_study_sha256,
        fusion_configuration_sha256: None,
        lane_configuration_sha256: baseline_lane_sha256,
        calibration: None,
        pair_calibration: None,
        disposition: PublicCorpusSidecarDisposition::Baseline,
        splits: vec![baseline.split.clone()],
        gate: None,
    }];

    match evaluation_stage {
        PublicCorpusEvaluationStage::Development => {
            for lane in PublicCorpusSidecarLane::ALL.into_iter().skip(1) {
                checkpoint_cancelled(&mut is_cancelled)?;
                let calibrations = fit_public_sidecar_lane(
                    &bundle,
                    &input_recordings,
                    &canonical_input,
                    maximum_recording_duration_ms,
                    lane,
                    target_split,
                    &mut is_cancelled,
                )?;
                let mut evaluation = if let Some(calibrations) = calibrations.as_ref() {
                    evaluate_public_sidecar_lane(
                        &bundle,
                        &input_recordings,
                        &canonical_input,
                        maximum_recording_duration_ms,
                        &diarization_request,
                        &scorer_config,
                        lane,
                        Some(&calibrations.boundary),
                        calibrations.pair.as_ref(),
                        protocol.detector_mode,
                        protocol.clustering_mode,
                        target_split,
                        &mut is_cancelled,
                    )?
                } else {
                    SidecarLaneEvaluation {
                        split: unavailable_public_sidecar_split(target_split)?,
                        recording_accuracy: Vec::new(),
                    }
                };
                if evaluation.split.fusion_executed {
                    evaluation.split.paired_uncertainty = Some(paired_sidecar_uncertainty(
                        &baseline.recording_accuracy,
                        &evaluation.recording_accuracy,
                        lane,
                        target_split,
                        &mut is_cancelled,
                    )?);
                }
                let gate = public_sidecar_promotion_gate(
                    evaluation_stage,
                    &protocol.gate_policy,
                    &baseline.split,
                    lane,
                    &evaluation.split,
                );
                let study_configuration_sha256 =
                    acoustic_sidecar_study_config_sha256(lane.study_config())?;
                let fusion_configuration_sha256 = calibrations
                    .as_ref()
                    .map(|calibrations| sidecar_evaluation_request(lane, &calibrations.boundary))
                    .transpose()?
                    .map(|request| {
                        acoustic_sidecar_fusion_configuration_sha256(
                            request,
                            protocol.detector_mode,
                        )
                    })
                    .transpose()?;
                let pair_calibration_sha256 = calibrations
                    .as_ref()
                    .and_then(|calibrations| calibrations.pair.as_ref())
                    .map(|calibration| calibration.calibration_sha256.as_str());
                let lane_configuration_sha256 =
                    canonical_sha256(&PublicCorpusSidecarLaneFingerprint {
                        runner_version: PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION,
                        lane,
                        fusion_scope: PublicCorpusSidecarFusionScope::BoundaryFusionV2,
                        study_configuration_sha256: &study_configuration_sha256,
                        fusion_configuration_sha256: fusion_configuration_sha256.as_deref(),
                        pair_calibration_sha256,
                        protocol_sha256: &protocol_sha256,
                    })?;
                variants.push(PublicCorpusSidecarStudyVariant {
                    lane,
                    fusion_scope: PublicCorpusSidecarFusionScope::BoundaryFusionV2,
                    study_configuration_sha256,
                    fusion_configuration_sha256,
                    lane_configuration_sha256,
                    calibration: calibrations
                        .as_ref()
                        .map(|calibrations| calibrations.boundary.clone()),
                    pair_calibration: calibrations
                        .as_ref()
                        .and_then(|calibrations| calibrations.pair.clone()),
                    disposition: PublicCorpusSidecarDisposition::Rejected,
                    splits: vec![evaluation.split],
                    gate: Some(gate),
                });
            }
            apply_public_sidecar_development_selection(&mut variants);
        }
        PublicCorpusEvaluationStage::Certification => {
            let locked = locked_development.as_ref().ok_or_else(|| {
                public_corpus_error(
                    "sidecar_stage_lock",
                    "certification requires locked development evidence",
                )
            })?;
            let selected = locked.selected_candidate_lane.ok_or_else(|| {
                public_corpus_error(
                    "sidecar_stage_lock",
                    "development lock has no selected sidecar candidate",
                )
            })?;
            for lane in PublicCorpusSidecarLane::ALL.into_iter().skip(1) {
                let locked_variant = locked
                    .variants
                    .iter()
                    .find(|variant| variant.lane == lane)
                    .ok_or_else(|| {
                        public_corpus_error(
                            "sidecar_stage_lock",
                            "development lock is missing a frozen sidecar lane",
                        )
                    })?;
                let boundary_calibration = locked_variant.calibration.clone();
                let pair_calibration = locked_variant.pair_calibration.clone();
                let study_configuration_sha256 =
                    acoustic_sidecar_study_config_sha256(lane.study_config())?;
                let fusion_configuration_sha256 = boundary_calibration
                    .as_ref()
                    .map(|calibration| sidecar_evaluation_request(lane, calibration))
                    .transpose()?
                    .map(|request| {
                        acoustic_sidecar_fusion_configuration_sha256(
                            request,
                            protocol.detector_mode,
                        )
                    })
                    .transpose()?;
                if let Some(pair_calibration) = pair_calibration.as_ref() {
                    validate_public_sidecar_pair_calibration(pair_calibration)?;
                }
                let lane_configuration_sha256 =
                    canonical_sha256(&PublicCorpusSidecarLaneFingerprint {
                        runner_version: PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION,
                        lane,
                        fusion_scope: PublicCorpusSidecarFusionScope::BoundaryFusionV2,
                        study_configuration_sha256: &study_configuration_sha256,
                        fusion_configuration_sha256: fusion_configuration_sha256.as_deref(),
                        pair_calibration_sha256: pair_calibration
                            .as_ref()
                            .map(|calibration| calibration.calibration_sha256.as_str()),
                        protocol_sha256: &protocol_sha256,
                    })?;
                if lane == selected {
                    let boundary_calibration = boundary_calibration.as_ref().ok_or_else(|| {
                        public_corpus_error(
                            "sidecar_stage_lock",
                            "selected development lane has no locked boundary calibration",
                        )
                    })?;
                    let pair_calibration = pair_calibration.as_ref().ok_or_else(|| {
                        public_corpus_error(
                            "sidecar_stage_lock",
                            "selected development lane has no locked pair calibration",
                        )
                    })?;
                    let mut evaluation = evaluate_public_sidecar_lane(
                        &bundle,
                        &input_recordings,
                        &canonical_input,
                        maximum_recording_duration_ms,
                        &diarization_request,
                        &scorer_config,
                        lane,
                        Some(boundary_calibration),
                        Some(pair_calibration),
                        protocol.detector_mode,
                        protocol.clustering_mode,
                        target_split,
                        &mut is_cancelled,
                    )?;
                    if evaluation.split.fusion_executed {
                        evaluation.split.paired_uncertainty = Some(paired_sidecar_uncertainty(
                            &baseline.recording_accuracy,
                            &evaluation.recording_accuracy,
                            lane,
                            target_split,
                            &mut is_cancelled,
                        )?);
                    }
                    let gate = public_sidecar_promotion_gate(
                        evaluation_stage,
                        &protocol.gate_policy,
                        &baseline.split,
                        lane,
                        &evaluation.split,
                    );
                    let disposition = if gate.passed {
                        PublicCorpusSidecarDisposition::Adopted
                    } else {
                        PublicCorpusSidecarDisposition::Rejected
                    };
                    variants.push(PublicCorpusSidecarStudyVariant {
                        lane,
                        fusion_scope: PublicCorpusSidecarFusionScope::BoundaryFusionV2,
                        study_configuration_sha256,
                        fusion_configuration_sha256,
                        lane_configuration_sha256,
                        calibration: Some(boundary_calibration.clone()),
                        pair_calibration: Some(pair_calibration.clone()),
                        disposition,
                        splits: vec![evaluation.split],
                        gate: Some(gate),
                    });
                } else {
                    variants.push(PublicCorpusSidecarStudyVariant {
                        lane,
                        fusion_scope: PublicCorpusSidecarFusionScope::BoundaryFusionV2,
                        study_configuration_sha256,
                        fusion_configuration_sha256,
                        lane_configuration_sha256,
                        calibration: boundary_calibration,
                        pair_calibration,
                        disposition: PublicCorpusSidecarDisposition::Rejected,
                        splits: vec![unavailable_public_sidecar_split(target_split)?],
                        gate: None,
                    });
                }
            }
        }
    }
    let selected_candidate_lane = match evaluation_stage {
        PublicCorpusEvaluationStage::Development => variants
            .iter()
            .find(|variant| {
                variant.disposition == PublicCorpusSidecarDisposition::AdvanceToCertification
            })
            .map(|variant| variant.lane),
        PublicCorpusEvaluationStage::Certification => locked_development
            .as_ref()
            .and_then(|evidence| evidence.selected_candidate_lane),
    };
    let adopted_candidate_lane = variants
        .iter()
        .find(|variant| variant.disposition == PublicCorpusSidecarDisposition::Adopted)
        .map(|variant| variant.lane);
    let locked_development_result_sha256 = locked_development
        .as_ref()
        .map(|evidence| evidence.result_sha256.clone());
    let locked_development_accuracy_sha256 = locked_development
        .as_ref()
        .map(|evidence| evidence.deterministic_accuracy_sha256.clone());
    let mut result = PublicCorpusSidecarStudyEvidence {
        schema_version: PUBLIC_CORPUS_SIDECAR_STUDY_SCHEMA_VERSION.to_owned(),
        runner_version: PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION.to_owned(),
        scorer_version: DIARIZATION_SCORER_VERSION.to_owned(),
        corpus_key: bundle.corpus_key.clone(),
        source_version: bundle.source_version.clone(),
        bundle_sha256: bundle.bundle_sha256.clone(),
        descriptor_sha256,
        scorer_config,
        scorer_config_sha256,
        evaluation_stage,
        locked_development_result_sha256,
        locked_development_accuracy_sha256,
        protocol,
        protocol_sha256,
        selected_candidate_lane,
        adopted_candidate_lane,
        variants,
        deterministic_accuracy_sha256: String::new(),
        result_sha256: String::new(),
    };
    result.deterministic_accuracy_sha256 = deterministic_sidecar_accuracy_sha256(&result)?;
    result.result_sha256 = canonical_sha256(&result)?;
    verify_public_corpus_sidecar_study_evidence(&result)?;
    checkpoint_cancelled(&mut is_cancelled)?;
    let staged_bundle = stage_new_json(
        bundle_output_path,
        &bundle_output_parent,
        &bundle,
        "public-corpus bundle",
        &mut is_cancelled,
    )?;
    let staged_evidence = match stage_new_json(
        evidence_output_path,
        &evidence_output_parent,
        &result,
        "sidecar study evidence",
        &mut is_cancelled,
    ) {
        Ok(staged) => staged,
        Err(error) => {
            return Err(staged_scrubbed_error(
                error,
                &[(&staged_bundle, "public-corpus bundle")],
            ));
        }
    };
    if let Err(error) = checkpoint_cancelled(&mut is_cancelled) {
        return Err(staged_scrubbed_error(
            error,
            &[
                (&staged_bundle, "public-corpus bundle"),
                (&staged_evidence, "sidecar study evidence"),
            ],
        ));
    }
    if let Err(error) = publish_staged_json(staged_bundle, "public-corpus bundle") {
        return Err(staged_scrubbed_error(
            error,
            &[(&staged_evidence, "sidecar study evidence")],
        ));
    }
    publish_staged_json(staged_evidence, "sidecar study evidence")?;
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
                    count_merge_steps: Vec::new(),
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

#[derive(Clone, Copy, Default)]
struct SidecarReliabilityAccumulator {
    observation_count: u64,
    positive_count: u64,
    probability_sum: f64,
    probability_sum_correction: f64,
    positive_probability_sum: f64,
    positive_probability_sum_correction: f64,
    squared_probability_sum: f64,
    squared_probability_sum_correction: f64,
    positive_squared_probability_sum: f64,
    positive_squared_probability_sum_correction: f64,
    squared_error_sum: f64,
    squared_error_sum_correction: f64,
}

fn add_sidecar_compensated(total: &mut f64, correction: &mut f64, value: f64) {
    let next = *total + value;
    if total.abs() >= value.abs() {
        *correction += (*total - next) + value;
    } else {
        *correction += (value - next) + *total;
    }
    *total = next;
}

fn sidecar_compensated_total(total: f64, correction: f64) -> f64 {
    total + correction
}

struct FinishedSidecarReliability {
    brier_score: Option<f64>,
    expected_calibration_error: Option<f64>,
    probability_sum: f64,
    positive_probability_sum: f64,
    reliability: Vec<PublicCorpusSidecarReliabilityBin>,
}

struct SidecarProbabilityAccumulator {
    observation_count: u64,
    positive_count: u64,
    reliability: Vec<SidecarReliabilityAccumulator>,
}

impl SidecarProbabilityAccumulator {
    fn new() -> Self {
        Self {
            observation_count: 0,
            positive_count: 0,
            reliability: vec![
                SidecarReliabilityAccumulator::default();
                PUBLIC_SIDECAR_RELIABILITY_BINS
            ],
        }
    }

    fn push(&mut self, probability: f64, positive: bool) -> FwResult<()> {
        if !probability.is_finite()
            || !(f64::from(f32::EPSILON)..=f64::from(1.0_f32 - f32::EPSILON)).contains(&probability)
            || f64::from(probability as f32) != probability
        {
            return Err(public_corpus_error(
                "sidecar_probability",
                "sidecar probability must be a finite f32 value within the calibrated open-unit clamp",
            ));
        }
        self.observation_count = self.observation_count.checked_add(1).ok_or_else(|| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "sidecar probability observation count overflowed",
            )
        })?;
        self.positive_count = self
            .positive_count
            .checked_add(u64::from(positive))
            .ok_or_else(|| {
                public_corpus_error(
                    "sidecar_aggregate_overflow",
                    "sidecar positive observation count overflowed",
                )
            })?;
        let bin =
            sidecar_f32_probability_bin_index(probability as f32, PUBLIC_SIDECAR_RELIABILITY_BINS);
        let reliability = &mut self.reliability[bin];
        reliability.observation_count =
            reliability
                .observation_count
                .checked_add(1)
                .ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_aggregate_overflow",
                        "sidecar reliability observation count overflowed",
                    )
                })?;
        reliability.positive_count = reliability
            .positive_count
            .checked_add(u64::from(positive))
            .ok_or_else(|| {
                public_corpus_error(
                    "sidecar_aggregate_overflow",
                    "sidecar reliability positive count overflowed",
                )
            })?;
        add_sidecar_compensated(
            &mut reliability.probability_sum,
            &mut reliability.probability_sum_correction,
            probability,
        );
        add_sidecar_compensated(
            &mut reliability.positive_probability_sum,
            &mut reliability.positive_probability_sum_correction,
            probability * f64::from(positive),
        );
        add_sidecar_compensated(
            &mut reliability.squared_probability_sum,
            &mut reliability.squared_probability_sum_correction,
            probability * probability,
        );
        add_sidecar_compensated(
            &mut reliability.positive_squared_probability_sum,
            &mut reliability.positive_squared_probability_sum_correction,
            probability * probability * f64::from(positive),
        );
        let error = probability - f64::from(positive);
        add_sidecar_compensated(
            &mut reliability.squared_error_sum,
            &mut reliability.squared_error_sum_correction,
            error * error,
        );
        Ok(())
    }

    fn finish_reliability(&self) -> FinishedSidecarReliability {
        let mut ece = 0.0;
        let reliability = self
            .reliability
            .iter()
            .enumerate()
            .map(|(index, bin)| {
                let probability_sum = canonical_evidence_number(sidecar_compensated_total(
                    bin.probability_sum,
                    bin.probability_sum_correction,
                ));
                let positive_probability_sum =
                    canonical_evidence_number(sidecar_compensated_total(
                        bin.positive_probability_sum,
                        bin.positive_probability_sum_correction,
                    ));
                let squared_probability_sum = canonical_evidence_number(sidecar_compensated_total(
                    bin.squared_probability_sum,
                    bin.squared_probability_sum_correction,
                ));
                let positive_squared_probability_sum =
                    canonical_evidence_number(sidecar_compensated_total(
                        bin.positive_squared_probability_sum,
                        bin.positive_squared_probability_sum_correction,
                    ));
                let squared_error_sum = canonical_evidence_number(sidecar_compensated_total(
                    bin.squared_error_sum,
                    bin.squared_error_sum_correction,
                ));
                let mean_probability =
                    positive_ratio(probability_sum, bin.observation_count as f64);
                let empirical_frequency = ratio(bin.positive_count, bin.observation_count);
                if let (Some(mean), Some(empirical)) = (mean_probability, empirical_frequency) {
                    ece += bin.observation_count as f64 / self.observation_count.max(1) as f64
                        * (mean - empirical).abs();
                }
                PublicCorpusSidecarReliabilityBin {
                    index,
                    lower_probability: canonical_evidence_number(
                        index as f64 / PUBLIC_SIDECAR_RELIABILITY_BINS as f64,
                    ),
                    upper_probability: canonical_evidence_number(
                        (index + 1) as f64 / PUBLIC_SIDECAR_RELIABILITY_BINS as f64,
                    ),
                    observation_count: bin.observation_count,
                    positive_count: bin.positive_count,
                    probability_sum,
                    positive_probability_sum,
                    squared_probability_sum,
                    positive_squared_probability_sum,
                    squared_error_sum,
                    mean_probability,
                    empirical_frequency,
                }
            })
            .collect::<Vec<_>>();
        let probability_sum =
            canonical_evidence_number(reliability.iter().map(|bin| bin.probability_sum).sum());
        let positive_probability_sum = canonical_evidence_number(
            reliability
                .iter()
                .map(|bin| bin.positive_probability_sum)
                .sum(),
        );
        let brier_sum =
            canonical_evidence_number(reliability.iter().map(|bin| bin.squared_error_sum).sum());
        FinishedSidecarReliability {
            brier_score: positive_ratio(brier_sum, self.observation_count as f64),
            expected_calibration_error: (self.observation_count > 0)
                .then(|| canonical_evidence_number(ece)),
            probability_sum,
            positive_probability_sum,
            reliability,
        }
    }
}

struct SidecarPairAccumulator {
    probabilities: SidecarProbabilityAccumulator,
    score_bins: Vec<SidecarReliabilityAccumulator>,
}

impl SidecarPairAccumulator {
    fn new() -> Self {
        Self {
            probabilities: SidecarProbabilityAccumulator::new(),
            score_bins: vec![
                SidecarReliabilityAccumulator::default();
                PUBLIC_SIDECAR_PAIR_SCORE_BINS
            ],
        }
    }

    fn push(&mut self, probability: f64, different_speaker: bool) -> FwResult<()> {
        self.probabilities.push(probability, different_speaker)?;
        let bin =
            sidecar_f32_probability_bin_index(probability as f32, PUBLIC_SIDECAR_PAIR_SCORE_BINS);
        let score_bin = &mut self.score_bins[bin];
        score_bin.observation_count =
            score_bin.observation_count.checked_add(1).ok_or_else(|| {
                public_corpus_error(
                    "sidecar_aggregate_overflow",
                    "pair score histogram count overflowed",
                )
            })?;
        score_bin.positive_count = score_bin
            .positive_count
            .checked_add(u64::from(different_speaker))
            .ok_or_else(|| {
                public_corpus_error(
                    "sidecar_aggregate_overflow",
                    "different-speaker score histogram count overflowed",
                )
            })?;
        add_sidecar_compensated(
            &mut score_bin.probability_sum,
            &mut score_bin.probability_sum_correction,
            probability,
        );
        add_sidecar_compensated(
            &mut score_bin.positive_probability_sum,
            &mut score_bin.positive_probability_sum_correction,
            probability * f64::from(different_speaker),
        );
        add_sidecar_compensated(
            &mut score_bin.squared_probability_sum,
            &mut score_bin.squared_probability_sum_correction,
            probability * probability,
        );
        add_sidecar_compensated(
            &mut score_bin.positive_squared_probability_sum,
            &mut score_bin.positive_squared_probability_sum_correction,
            probability * probability * f64::from(different_speaker),
        );
        let error = probability - f64::from(different_speaker);
        add_sidecar_compensated(
            &mut score_bin.squared_error_sum,
            &mut score_bin.squared_error_sum_correction,
            error * error,
        );
        Ok(())
    }

    fn finish(&self) -> FwResult<PublicCorpusSidecarPairMetrics> {
        let same_speaker_count = self
            .probabilities
            .observation_count
            .checked_sub(self.probabilities.positive_count)
            .ok_or_else(|| {
                public_corpus_error(
                    "sidecar_aggregate_overflow",
                    "different-speaker count exceeds pair observation count",
                )
            })?;
        let different_speaker_count = self.probabilities.positive_count;
        let roc_auc = if same_speaker_count > 0 && different_speaker_count > 0 {
            let mut lower_same = 0_u64;
            let mut concordance = 0.0;
            for bin in &self.score_bins {
                let same = bin
                    .observation_count
                    .checked_sub(bin.positive_count)
                    .ok_or_else(|| {
                        public_corpus_error(
                            "sidecar_aggregate_overflow",
                            "different-speaker score count exceeds the bin total",
                        )
                    })?;
                let different = bin.positive_count;
                concordance += different as f64 * (lower_same as f64 + 0.5 * same as f64);
                lower_same = lower_same.checked_add(same).ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_aggregate_overflow",
                        "same-speaker AUC accumulator overflowed",
                    )
                })?;
            }
            Some(canonical_evidence_number(
                concordance / (same_speaker_count as f64 * different_speaker_count as f64),
            ))
        } else {
            None
        };
        let score_histogram = self
            .score_bins
            .iter()
            .enumerate()
            .map(|(index, bin)| {
                let same_speaker_count = bin
                    .observation_count
                    .checked_sub(bin.positive_count)
                    .ok_or_else(|| {
                        public_corpus_error(
                            "sidecar_aggregate_overflow",
                            "different-speaker score count exceeds the bin total",
                        )
                    })?;
                Ok(PublicCorpusSidecarPairScoreBin {
                    index,
                    lower_probability: canonical_evidence_number(
                        index as f64 / PUBLIC_SIDECAR_PAIR_SCORE_BINS as f64,
                    ),
                    upper_probability: canonical_evidence_number(
                        (index + 1) as f64 / PUBLIC_SIDECAR_PAIR_SCORE_BINS as f64,
                    ),
                    same_speaker_count,
                    different_speaker_count: bin.positive_count,
                    probability_sum: canonical_evidence_number(sidecar_compensated_total(
                        bin.probability_sum,
                        bin.probability_sum_correction,
                    )),
                    different_speaker_probability_sum: canonical_evidence_number(
                        sidecar_compensated_total(
                            bin.positive_probability_sum,
                            bin.positive_probability_sum_correction,
                        ),
                    ),
                    squared_probability_sum: canonical_evidence_number(sidecar_compensated_total(
                        bin.squared_probability_sum,
                        bin.squared_probability_sum_correction,
                    )),
                    different_speaker_squared_probability_sum: canonical_evidence_number(
                        sidecar_compensated_total(
                            bin.positive_squared_probability_sum,
                            bin.positive_squared_probability_sum_correction,
                        ),
                    ),
                    squared_error_sum: canonical_evidence_number(sidecar_compensated_total(
                        bin.squared_error_sum,
                        bin.squared_error_sum_correction,
                    )),
                })
            })
            .collect::<FwResult<Vec<_>>>()?;
        let score_bins_per_reliability_bin =
            PUBLIC_SIDECAR_PAIR_SCORE_BINS / PUBLIC_SIDECAR_RELIABILITY_BINS;
        let mut linked_reliability =
            vec![SidecarReliabilityAccumulator::default(); PUBLIC_SIDECAR_RELIABILITY_BINS];
        for (index, bin) in score_histogram.iter().enumerate() {
            let linked = &mut linked_reliability[index / score_bins_per_reliability_bin];
            let observation_count = bin
                .same_speaker_count
                .checked_add(bin.different_speaker_count)
                .ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_aggregate_overflow",
                        "linked pair reliability observation count overflowed",
                    )
                })?;
            linked.observation_count = linked
                .observation_count
                .checked_add(observation_count)
                .ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_aggregate_overflow",
                        "linked pair reliability observation count overflowed",
                    )
                })?;
            linked.positive_count = linked
                .positive_count
                .checked_add(bin.different_speaker_count)
                .ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_aggregate_overflow",
                        "linked pair reliability positive count overflowed",
                    )
                })?;
            add_sidecar_compensated(
                &mut linked.probability_sum,
                &mut linked.probability_sum_correction,
                bin.probability_sum,
            );
            add_sidecar_compensated(
                &mut linked.positive_probability_sum,
                &mut linked.positive_probability_sum_correction,
                bin.different_speaker_probability_sum,
            );
            add_sidecar_compensated(
                &mut linked.squared_probability_sum,
                &mut linked.squared_probability_sum_correction,
                bin.squared_probability_sum,
            );
            add_sidecar_compensated(
                &mut linked.positive_squared_probability_sum,
                &mut linked.positive_squared_probability_sum_correction,
                bin.different_speaker_squared_probability_sum,
            );
            add_sidecar_compensated(
                &mut linked.squared_error_sum,
                &mut linked.squared_error_sum_correction,
                bin.squared_error_sum,
            );
        }
        let linked_probabilities = SidecarProbabilityAccumulator {
            observation_count: self.probabilities.observation_count,
            positive_count: self.probabilities.positive_count,
            reliability: linked_reliability,
        };
        let finished = linked_probabilities.finish_reliability();
        let same_probability_sum =
            canonical_evidence_number(finished.probability_sum - finished.positive_probability_sum);
        Ok(PublicCorpusSidecarPairMetrics {
            comparison_count: self.probabilities.observation_count,
            same_speaker_count,
            different_speaker_count,
            mean_different_probability_given_same_speaker: positive_ratio(
                same_probability_sum,
                same_speaker_count as f64,
            ),
            mean_different_probability_given_different_speaker: positive_ratio(
                finished.positive_probability_sum,
                different_speaker_count as f64,
            ),
            roc_auc,
            brier_score: finished.brier_score,
            expected_calibration_error: finished.expected_calibration_error,
            reliability: finished.reliability,
            score_histogram,
        })
    }
}

#[derive(Default)]
struct SidecarAuxiliaryDominanceAccumulator {
    same_speaker_opportunity_count: u64,
    same_speaker_dominance_count: u64,
    different_speaker_opportunity_count: u64,
    different_speaker_dominance_count: u64,
}

impl SidecarAuxiliaryDominanceAccumulator {
    fn push(
        &mut self,
        different_speaker: bool,
        opportunity: bool,
        dominance: bool,
    ) -> FwResult<()> {
        if dominance && !opportunity {
            return Err(public_corpus_error(
                "sidecar_auxiliary_dominance",
                "auxiliary dominance cannot be present without a comparison opportunity",
            ));
        }
        let (opportunity_count, dominance_count) = if different_speaker {
            (
                &mut self.different_speaker_opportunity_count,
                &mut self.different_speaker_dominance_count,
            )
        } else {
            (
                &mut self.same_speaker_opportunity_count,
                &mut self.same_speaker_dominance_count,
            )
        };
        *opportunity_count = opportunity_count
            .checked_add(u64::from(opportunity))
            .ok_or_else(|| {
                public_corpus_error(
                    "sidecar_aggregate_overflow",
                    "auxiliary-dominance opportunity count overflowed",
                )
            })?;
        *dominance_count = dominance_count
            .checked_add(u64::from(dominance))
            .ok_or_else(|| {
                public_corpus_error(
                    "sidecar_aggregate_overflow",
                    "auxiliary-dominance count overflowed",
                )
            })?;
        Ok(())
    }

    fn finish(&self) -> PublicCorpusSidecarAuxiliaryDominanceMetrics {
        PublicCorpusSidecarAuxiliaryDominanceMetrics {
            same_speaker_opportunity_count: self.same_speaker_opportunity_count,
            same_speaker_dominance_count: self.same_speaker_dominance_count,
            same_speaker_dominance_rate: ratio(
                self.same_speaker_dominance_count,
                self.same_speaker_opportunity_count,
            ),
            different_speaker_opportunity_count: self.different_speaker_opportunity_count,
            different_speaker_dominance_count: self.different_speaker_dominance_count,
            different_speaker_dominance_rate: ratio(
                self.different_speaker_dominance_count,
                self.different_speaker_opportunity_count,
            ),
        }
    }
}

struct SidecarObservationAccumulator {
    boundary_probabilities: SidecarProbabilityAccumulator,
    pairs: SidecarPairAccumulator,
    submitted_frame_count: u64,
    comparable_frame_count: u64,
    component_comparison_count: u64,
    owner_available_frame_counts: [u64; 3],
    channel_dominance: SidecarAuxiliaryDominanceAccumulator,
    mixed_auxiliary_dominance: SidecarAuxiliaryDominanceAccumulator,
    eligible_pair_count: u64,
    retained_pair_sample_count: u64,
    retained_same_speaker_pair_count: u64,
    retained_different_speaker_pair_count: u64,
    pair_selection_hasher: Sha256,
    pair_scored_recording_count: u64,
    same_speaker_pair_recording_count: u64,
    different_speaker_pair_recording_count: u64,
    maximum_retained_pair_sample_count: u64,
    retained_pair_sample_capacity: u64,
}

impl SidecarObservationAccumulator {
    fn new() -> Self {
        let mut pair_selection_hasher = Sha256::new();
        pair_selection_hasher
            .update(PUBLIC_CORPUS_SIDECAR_PAIR_SELECTION_DIGEST_VERSION.as_bytes());
        Self {
            boundary_probabilities: SidecarProbabilityAccumulator::new(),
            pairs: SidecarPairAccumulator::new(),
            submitted_frame_count: 0,
            comparable_frame_count: 0,
            component_comparison_count: 0,
            owner_available_frame_counts: [0; 3],
            channel_dominance: SidecarAuxiliaryDominanceAccumulator::default(),
            mixed_auxiliary_dominance: SidecarAuxiliaryDominanceAccumulator::default(),
            eligible_pair_count: 0,
            retained_pair_sample_count: 0,
            retained_same_speaker_pair_count: 0,
            retained_different_speaker_pair_count: 0,
            pair_selection_hasher,
            pair_scored_recording_count: 0,
            same_speaker_pair_recording_count: 0,
            different_speaker_pair_recording_count: 0,
            maximum_retained_pair_sample_count: 0,
            retained_pair_sample_capacity: 0,
        }
    }
}

fn update_sidecar_pair_selection_digest(
    hasher: &mut Sha256,
    normalized_pcm_sha256: &[u8; 32],
    selected_pairs: &[SidecarPairSample],
) -> FwResult<()> {
    let retained_pair_count = u64::try_from(selected_pairs.len()).map_err(|_| {
        public_corpus_error(
            "sidecar_aggregate_overflow",
            "retained pair sample count exceeds u64",
        )
    })?;
    hasher.update(normalized_pcm_sha256);
    hasher.update(retained_pair_count.to_le_bytes());
    for pair in selected_pairs {
        hasher.update(pair.key.digest);
        hasher.update(pair.key.left_frame_index.to_le_bytes());
        hasher.update(pair.key.right_frame_index.to_le_bytes());
        hasher.update(pair.key.lag_frames.to_le_bytes());
        hasher.update([u8::from(pair.different_speaker)]);
    }
    Ok(())
}

#[derive(Default)]
struct SidecarOperationsAccumulator {
    fusion_requested: bool,
    fusion_executed: bool,
    evaluated_recording_count: u64,
    fusion_requested_recording_count: u64,
    fusion_executed_recording_count: u64,
    submitted_frame_count: u64,
    comparable_frame_count: u64,
    calibrated_signal_count: u64,
    consumed_probability_count: u64,
    changed_boundary_probability_count: u64,
    maximum_retained_signal_count: u64,
    retained_signal_capacity: u64,
    frame_wavelet_filter_tap_terms: u64,
    trajectory_wavelet_filter_tap_terms: u64,
    trajectory_validity_sample_visits: u64,
    scattering_filter_sample_terms: u64,
    scattering_validity_sample_visits: u64,
    modulation_projection_sample_frequency_visits: u64,
    peak_scratch_buffer_payload_bytes: u64,
    peak_retained_state_bytes_on_target: u64,
    cached_twiddle_payload_bytes: u64,
    expected_sidecar_configuration_sha256: Option<String>,
    expected_fusion_configuration_sha256: Option<String>,
}

impl SidecarOperationsAccumulator {
    fn push(&mut self, evidence: &AcousticSidecarFusionEvaluationEvidence) -> FwResult<()> {
        if evidence.fusion_executed != (evidence.consumed_probability_count > 0)
            || evidence.comparable_frame_count > evidence.submitted_frame_count
            || evidence.calibrated_signal_count > evidence.comparable_frame_count
            || evidence.consumed_probability_count > evidence.calibrated_signal_count
            || evidence.changed_boundary_probability_count > evidence.consumed_probability_count
        {
            return Err(public_corpus_error(
                "sidecar_fusion_evidence",
                "per-recording fusion consumption counters are inconsistent",
            ));
        }
        if evidence.fusion_requested
            != (evidence.sidecar_configuration_sha256.is_some()
                && evidence.fusion_configuration_sha256.is_some())
        {
            return Err(public_corpus_error(
                "sidecar_fusion_evidence",
                "fusion request and configuration identities are inconsistent",
            ));
        }
        let checked_usize = |value: usize, field: &str| {
            u64::try_from(value).map_err(|_| {
                public_corpus_error(
                    "sidecar_aggregate_overflow",
                    &format!("{field} exceeds the retained u64 range"),
                )
            })
        };
        let checked_add = |total: &mut u64, value: u64, field: &str| -> FwResult<()> {
            *total = total.checked_add(value).ok_or_else(|| {
                public_corpus_error(
                    "sidecar_aggregate_overflow",
                    &format!("{field} aggregate overflowed"),
                )
            })?;
            Ok(())
        };
        self.fusion_requested |= evidence.fusion_requested;
        self.fusion_executed |= evidence.fusion_executed;
        checked_add(
            &mut self.evaluated_recording_count,
            1,
            "evaluated recording count",
        )?;
        checked_add(
            &mut self.fusion_requested_recording_count,
            u64::from(evidence.fusion_requested),
            "fusion-requested recording count",
        )?;
        checked_add(
            &mut self.fusion_executed_recording_count,
            u64::from(evidence.fusion_executed),
            "fusion-executed recording count",
        )?;
        checked_add(
            &mut self.submitted_frame_count,
            checked_usize(evidence.submitted_frame_count, "submitted frame count")?,
            "submitted frame count",
        )?;
        checked_add(
            &mut self.comparable_frame_count,
            checked_usize(evidence.comparable_frame_count, "comparable frame count")?,
            "comparable frame count",
        )?;
        checked_add(
            &mut self.calibrated_signal_count,
            checked_usize(evidence.calibrated_signal_count, "calibrated signal count")?,
            "calibrated signal count",
        )?;
        checked_add(
            &mut self.consumed_probability_count,
            checked_usize(
                evidence.consumed_probability_count,
                "consumed probability count",
            )?,
            "consumed probability count",
        )?;
        checked_add(
            &mut self.changed_boundary_probability_count,
            checked_usize(
                evidence.changed_boundary_probability_count,
                "changed boundary probability count",
            )?,
            "changed boundary probability count",
        )?;
        self.maximum_retained_signal_count = self.maximum_retained_signal_count.max(checked_usize(
            evidence.maximum_retained_signals,
            "maximum retained signal count",
        )?);
        self.retained_signal_capacity = self.retained_signal_capacity.max(checked_usize(
            evidence.retained_signal_capacity,
            "retained signal capacity",
        )?);
        for (total, value, field) in [
            (
                &mut self.frame_wavelet_filter_tap_terms,
                evidence.frame_wavelet_filter_tap_terms,
                "frame wavelet terms",
            ),
            (
                &mut self.trajectory_wavelet_filter_tap_terms,
                evidence.trajectory_wavelet_filter_tap_terms,
                "trajectory wavelet terms",
            ),
            (
                &mut self.trajectory_validity_sample_visits,
                evidence.trajectory_validity_sample_visits,
                "trajectory validity visits",
            ),
            (
                &mut self.scattering_filter_sample_terms,
                evidence.scattering_filter_sample_terms,
                "scattering filter terms",
            ),
            (
                &mut self.scattering_validity_sample_visits,
                evidence.scattering_validity_sample_visits,
                "scattering validity visits",
            ),
            (
                &mut self.modulation_projection_sample_frequency_visits,
                evidence.modulation_projection_sample_frequency_visits,
                "modulation projection visits",
            ),
        ] {
            checked_add(total, value, field)?;
        }
        self.peak_scratch_buffer_payload_bytes =
            self.peak_scratch_buffer_payload_bytes.max(checked_usize(
                evidence.peak_scratch_buffer_payload_bytes,
                "peak scratch payload",
            )?);
        self.peak_retained_state_bytes_on_target =
            self.peak_retained_state_bytes_on_target.max(checked_usize(
                evidence.peak_retained_state_bytes_on_target,
                "peak retained payload",
            )?);
        self.cached_twiddle_payload_bytes = self.cached_twiddle_payload_bytes.max(checked_usize(
            evidence.cached_twiddle_payload_bytes,
            "cached twiddle payload",
        )?);
        for (expected, actual, field) in [
            (
                &mut self.expected_sidecar_configuration_sha256,
                evidence.sidecar_configuration_sha256.as_ref(),
                "sidecar configuration",
            ),
            (
                &mut self.expected_fusion_configuration_sha256,
                evidence.fusion_configuration_sha256.as_ref(),
                "fusion configuration",
            ),
        ] {
            if let Some(actual) = actual {
                if expected.as_ref().is_some_and(|expected| expected != actual) {
                    return Err(public_corpus_error(
                        "sidecar_configuration_drift",
                        &format!("{field} changed across recordings"),
                    ));
                }
                *expected = Some(actual.clone());
            }
        }
        Ok(())
    }

    fn operations(&self) -> PublicCorpusSidecarOperations {
        PublicCorpusSidecarOperations {
            frame_wavelet_filter_tap_terms: self.frame_wavelet_filter_tap_terms,
            trajectory_wavelet_filter_tap_terms: self.trajectory_wavelet_filter_tap_terms,
            trajectory_validity_sample_visits: self.trajectory_validity_sample_visits,
            scattering_filter_sample_terms: self.scattering_filter_sample_terms,
            scattering_validity_sample_visits: self.scattering_validity_sample_visits,
            modulation_projection_sample_frequency_visits: self
                .modulation_projection_sample_frequency_visits,
            peak_scratch_buffer_payload_bytes: self.peak_scratch_buffer_payload_bytes,
            peak_retained_state_bytes_on_target: self.peak_retained_state_bytes_on_target,
            cached_twiddle_payload_bytes: self.cached_twiddle_payload_bytes,
        }
    }
}

#[derive(Clone, Copy)]
struct SidecarRecordingAccuracy {
    recording_audio_sha256: [u8; 32],
    reference_sha256: [u8; 32],
    der: Option<f64>,
    jer: Option<f64>,
}

struct SidecarLaneEvaluation {
    split: PublicCorpusSidecarStudySplit,
    recording_accuracy: Vec<SidecarRecordingAccuracy>,
}

struct SidecarRingEntry {
    frame_index: usize,
    speaker_token: Option<usize>,
    observation: AcousticSidecarStudyObservation,
}

fn push_bounded_sidecar_ring_entry<T>(
    ring: &mut VecDeque<T>,
    current_frame_index: usize,
    maximum_lag: usize,
    entry: T,
    mut frame_index: impl FnMut(&T) -> usize,
) -> FwResult<()> {
    let maximum_retained = maximum_lag.checked_add(1).ok_or_else(|| {
        public_corpus_error(
            "sidecar_pair_resource_bound",
            "sidecar ring length bound overflowed",
        )
    })?;
    if frame_index(&entry) != current_frame_index {
        return Err(public_corpus_error(
            "sidecar_pair_alignment",
            "sidecar ring entry does not match the current frame",
        ));
    }
    while let Some(front) = ring.front() {
        let age = current_frame_index
            .checked_sub(frame_index(front))
            .ok_or_else(|| {
                public_corpus_error(
                    "sidecar_pair_alignment",
                    "sidecar ring contains an entry from a future frame",
                )
            })?;
        if age <= maximum_lag {
            break;
        }
        ring.pop_front();
    }
    if ring.len() > maximum_lag || ring.len() >= ring.capacity() {
        return Err(public_corpus_error(
            "sidecar_pair_resource_bound",
            "sidecar ring cannot accept the current frame without growing",
        ));
    }
    let retained_capacity = ring.capacity();
    ring.push_back(entry);
    if ring.len() > maximum_retained || ring.capacity() != retained_capacity {
        return Err(public_corpus_error(
            "sidecar_pair_resource_bound",
            "sidecar ring exceeded its frozen length or capacity bound",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SidecarPairSelectionKey {
    digest: [u8; 32],
    left_frame_index: u64,
    right_frame_index: u64,
    lag_frames: u64,
}

struct SidecarPairSample {
    key: SidecarPairSelectionKey,
    maximum_contrast: Option<f64>,
    different_speaker: bool,
    channel_dominance_opportunity: bool,
    channel_dominance: bool,
    mixed_auxiliary_dominance_opportunity: bool,
    mixed_auxiliary_dominance: bool,
}

impl PartialEq for SidecarPairSample {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for SidecarPairSample {}

impl PartialOrd for SidecarPairSample {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SidecarPairSample {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

struct SidecarPairBottomKSampler {
    retained_limit: usize,
    eligible_pair_count: u64,
    retained: BinaryHeap<SidecarPairSample>,
}

impl SidecarPairBottomKSampler {
    fn new(retained_limit: usize) -> FwResult<Self> {
        if retained_limit == 0 || retained_limit > PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING {
            return Err(public_corpus_error(
                "sidecar_pair_sampler",
                "pair sampler retained limit is outside the frozen bound",
            ));
        }
        let retained = BinaryHeap::with_capacity(retained_limit);
        if retained.capacity() > PUBLIC_SIDECAR_MAX_RETAINED_PAIR_SAMPLE_CAPACITY {
            return Err(public_corpus_error(
                "sidecar_pair_sampler",
                "pair sampler allocation exceeds the frozen capacity bound",
            ));
        }
        Ok(Self {
            retained_limit,
            eligible_pair_count: 0,
            retained,
        })
    }

    fn consider(&mut self, sample: SidecarPairSample) -> FwResult<()> {
        self.eligible_pair_count = self.eligible_pair_count.checked_add(1).ok_or_else(|| {
            public_corpus_error("sidecar_pair_sampler", "eligible pair count overflowed")
        })?;
        if self.retained.len() < self.retained_limit {
            self.retained.push(sample);
        } else if self
            .retained
            .peek()
            .is_some_and(|maximum| sample.key < maximum.key)
        {
            self.retained.pop();
            self.retained.push(sample);
        }
        Ok(())
    }

    fn finish(self) -> FwResult<(u64, usize, usize, Vec<SidecarPairSample>)> {
        let mut retained = self.retained.into_vec();
        let retained_capacity = retained.capacity();
        if retained_capacity > PUBLIC_SIDECAR_MAX_RETAINED_PAIR_SAMPLE_CAPACITY {
            return Err(public_corpus_error(
                "sidecar_pair_sampler",
                "retained pair vector exceeds the frozen capacity bound",
            ));
        }
        retained.sort_unstable_by_key(|sample| sample.key);
        Ok((
            self.eligible_pair_count,
            retained.len(),
            retained_capacity,
            retained,
        ))
    }
}

fn sidecar_pair_selection_key(
    normalized_pcm_sha256: &[u8; 32],
    left_frame_index: usize,
    right_frame_index: usize,
    lag_frames: usize,
) -> FwResult<SidecarPairSelectionKey> {
    let left_frame_index = u64::try_from(left_frame_index).map_err(|_| {
        public_corpus_error(
            "sidecar_pair_sampler",
            "left pair frame exceeds the retained u64 range",
        )
    })?;
    let right_frame_index = u64::try_from(right_frame_index).map_err(|_| {
        public_corpus_error(
            "sidecar_pair_sampler",
            "right pair frame exceeds the retained u64 range",
        )
    })?;
    let lag_frames = u64::try_from(lag_frames).map_err(|_| {
        public_corpus_error(
            "sidecar_pair_sampler",
            "pair lag exceeds the retained u64 range",
        )
    })?;
    let mut hasher = Sha256::new();
    hasher.update(PUBLIC_CORPUS_SIDECAR_PAIR_SELECTION_KEY_VERSION.as_bytes());
    hasher.update(normalized_pcm_sha256);
    hasher.update(left_frame_index.to_le_bytes());
    hasher.update(right_frame_index.to_le_bytes());
    hasher.update(lag_frames.to_le_bytes());
    Ok(SidecarPairSelectionKey {
        digest: hasher.finalize().into(),
        left_frame_index,
        right_frame_index,
        lag_frames,
    })
}

fn public_sidecar_probability(
    calibration: &PublicCorpusSidecarCalibration,
    contrast: crate::diarization::AcousticSidecarOwnerContrast,
) -> FwResult<Option<f64>> {
    acoustic_sidecar_calibrate_owner_contrast(
        contrast,
        AcousticSidecarFusionCalibration {
            logit_intercept: calibration.logit_intercept as f32,
            contrast_weight: calibration.contrast_weight as f32,
            minimum_comparable_components: calibration.minimum_comparable_components,
        },
    )
    .map(|evidence| evidence.map(|evidence| f64::from(evidence.probability)))
}

fn maximum_available_sidecar_contrast(
    contrast: &crate::diarization::AcousticSidecarOwnerContrast,
) -> Option<f64> {
    contrast
        .owner_contrast
        .iter()
        .zip(contrast.owner_available)
        .filter_map(|(&value, available)| available.then_some(f64::from(value)))
        .max_by(f64::total_cmp)
}

fn sidecar_auxiliary_dominance(
    contrast: &crate::diarization::AcousticSidecarOwnerContrast,
    auxiliary_owner_index: usize,
) -> (bool, bool) {
    let opportunity = contrast.owner_available[0]
        && contrast
            .owner_available
            .get(auxiliary_owner_index)
            .copied()
            .unwrap_or(false);
    let dominance =
        opportunity && contrast.owner_contrast[auxiliary_owner_index] > contrast.owner_contrast[0];
    (opportunity, dominance)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SidecarReferenceFrameLabel<'a> {
    speaker: Option<&'a str>,
    boundary: Option<bool>,
}

/// Monotonic, overlap-aware reference lookup for the 10 ms acoustic cadence.
///
/// Every turn and ignored interval is admitted at most once. Active turns use
/// a heap ordered by end time, so per-frame work depends on overlap depth rather
/// than rescanning the complete annotation document.
struct SidecarReferenceSweep<'a> {
    reference: &'a DiarizationReferenceDocument,
    reference_changes: &'a [u64],
    next_turn_index: usize,
    active_turn_ends: BinaryHeap<Reverse<(u64, usize)>>,
    active_speaker_counts: BTreeMap<&'a str, usize>,
    active_unknown_speaker_count: usize,
    next_ignored_region_index: usize,
    active_ignored_end_ms: u64,
    next_change_index: usize,
    previous_timestamp_ms: Option<u64>,
}

impl<'a> SidecarReferenceSweep<'a> {
    fn new(
        reference: &'a DiarizationReferenceDocument,
        reference_changes: &'a [u64],
    ) -> FwResult<Self> {
        if !reference
            .turns
            .windows(2)
            .all(|window| window[0].start_ms <= window[1].start_ms)
            || !reference
                .ignored_regions
                .windows(2)
                .all(|window| window[0].start_ms <= window[1].start_ms)
            || !reference_changes
                .windows(2)
                .all(|window| window[0] <= window[1])
        {
            return Err(public_corpus_error(
                "sidecar_reference_order",
                "reference turns, ignored regions, and change points must be monotonically ordered",
            ));
        }
        Ok(Self {
            reference,
            reference_changes,
            next_turn_index: 0,
            active_turn_ends: BinaryHeap::new(),
            active_speaker_counts: BTreeMap::new(),
            active_unknown_speaker_count: 0,
            next_ignored_region_index: 0,
            active_ignored_end_ms: 0,
            next_change_index: 0,
            previous_timestamp_ms: None,
        })
    }

    fn add_active_turn(&mut self, turn_index: usize) -> FwResult<()> {
        if let Some(speaker) = self.reference.turns[turn_index].speaker.as_deref() {
            let count = self.active_speaker_counts.entry(speaker).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                public_corpus_error(
                    "sidecar_reference_bound",
                    "active reference speaker count overflowed",
                )
            })?;
        } else {
            self.active_unknown_speaker_count = self
                .active_unknown_speaker_count
                .checked_add(1)
                .ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_reference_bound",
                        "active unknown-speaker count overflowed",
                    )
                })?;
        }
        Ok(())
    }

    fn remove_active_turn(&mut self, turn_index: usize) -> FwResult<()> {
        if let Some(speaker) = self.reference.turns[turn_index].speaker.as_deref() {
            let remove_speaker = {
                let count = self.active_speaker_counts.get_mut(speaker).ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_reference_state",
                        "ended reference turn was absent from the active speaker set",
                    )
                })?;
                *count = count.checked_sub(1).ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_reference_state",
                        "active reference speaker count underflowed",
                    )
                })?;
                *count == 0
            };
            if remove_speaker {
                self.active_speaker_counts.remove(speaker);
            }
        } else {
            self.active_unknown_speaker_count = self
                .active_unknown_speaker_count
                .checked_sub(1)
                .ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_reference_state",
                        "active unknown-speaker count underflowed",
                    )
                })?;
        }
        Ok(())
    }

    fn label_at_ms(&mut self, timestamp_ms: u64) -> FwResult<SidecarReferenceFrameLabel<'a>> {
        if self
            .previous_timestamp_ms
            .is_some_and(|previous| timestamp_ms < previous)
        {
            return Err(public_corpus_error(
                "sidecar_reference_order",
                "sidecar reference timestamps must be monotonically ordered",
            ));
        }
        self.previous_timestamp_ms = Some(timestamp_ms);

        while self
            .active_turn_ends
            .peek()
            .is_some_and(|Reverse((end_ms, _))| *end_ms <= timestamp_ms)
        {
            let Reverse((_, turn_index)) = self.active_turn_ends.pop().ok_or_else(|| {
                public_corpus_error(
                    "sidecar_reference_state",
                    "active reference heap unexpectedly became empty",
                )
            })?;
            self.remove_active_turn(turn_index)?;
        }
        while self
            .reference
            .turns
            .get(self.next_turn_index)
            .is_some_and(|turn| turn.start_ms <= timestamp_ms)
        {
            let turn_index = self.next_turn_index;
            self.next_turn_index = self.next_turn_index.checked_add(1).ok_or_else(|| {
                public_corpus_error(
                    "sidecar_reference_bound",
                    "reference turn cursor overflowed",
                )
            })?;
            let turn = &self.reference.turns[turn_index];
            if turn.end_ms > timestamp_ms {
                self.active_turn_ends
                    .push(Reverse((turn.end_ms, turn_index)));
                self.add_active_turn(turn_index)?;
            }
        }

        while self
            .reference
            .ignored_regions
            .get(self.next_ignored_region_index)
            .is_some_and(|region| region.start_ms <= timestamp_ms)
        {
            let region = &self.reference.ignored_regions[self.next_ignored_region_index];
            self.active_ignored_end_ms = self.active_ignored_end_ms.max(region.end_ms);
            self.next_ignored_region_index = self
                .next_ignored_region_index
                .checked_add(1)
                .ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_reference_bound",
                        "ignored-region cursor overflowed",
                    )
                })?;
        }
        let ignored = timestamp_ms < self.active_ignored_end_ms;

        while self
            .reference_changes
            .get(self.next_change_index)
            .is_some_and(|change| {
                change.saturating_add(PUBLIC_SIDECAR_BOUNDARY_COLLAR_MS) < timestamp_ms
            })
        {
            self.next_change_index = self.next_change_index.checked_add(1).ok_or_else(|| {
                public_corpus_error(
                    "sidecar_reference_bound",
                    "reference change cursor overflowed",
                )
            })?;
        }
        let near_boundary = self
            .reference_changes
            .get(self.next_change_index)
            .is_some_and(|change| {
                change.abs_diff(timestamp_ms) <= PUBLIC_SIDECAR_BOUNDARY_COLLAR_MS
            });
        let speaker =
            if self.active_unknown_speaker_count == 0 && self.active_speaker_counts.len() == 1 {
                self.active_speaker_counts.keys().next().copied()
            } else {
                None
            };
        Ok(SidecarReferenceFrameLabel {
            speaker: if ignored { None } else { speaker },
            boundary: if ignored {
                None
            } else if near_boundary {
                Some(true)
            } else {
                speaker.map(|_| false)
            },
        })
    }
}

fn sidecar_frame_timestamp_ms(frame_index: usize) -> FwResult<u64> {
    u64::try_from(frame_index)
        .ok()
        .and_then(|index| index.checked_mul(10))
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_frame_index",
                "sidecar frame timestamp exceeds the retained u64 range",
            )
        })
}

struct SidecarObservationAnalysis<'a> {
    reference: &'a DiarizationReferenceDocument,
    normalized_pcm_sha256: &'a str,
    study_config: AcousticSidecarStudyConfig,
    boundary_calibration: &'a PublicCorpusSidecarCalibration,
    pair_calibration: Option<&'a PublicCorpusSidecarPairCalibration>,
}

fn analyze_sidecar_observations(
    samples: &[f32],
    analysis: SidecarObservationAnalysis<'_>,
    aggregate: &mut SidecarObservationAccumulator,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<()> {
    let SidecarObservationAnalysis {
        reference,
        normalized_pcm_sha256,
        study_config,
        boundary_calibration,
        pair_calibration,
    } = analysis;
    if let Some(pair_calibration) = pair_calibration {
        validate_public_sidecar_pair_calibration(pair_calibration)?;
    }
    let mut study = AcousticSidecarStudy::new(study_config)?;
    let study_configuration_sha256_digest = study.configuration_sha256_digest();
    let mut previous = None;
    let mut ring = VecDeque::<SidecarRingEntry>::with_capacity(
        PUBLIC_SIDECAR_PAIR_LAGS_FRAMES[PUBLIC_SIDECAR_PAIR_LAGS_FRAMES.len() - 1] + 1,
    );
    let reference_changes =
        speaker_change_points_ms(&reference.turns, reference.duration_ms, true)?;
    let mut reference_sweep = SidecarReferenceSweep::new(reference, &reference_changes)?;
    let speaker_tokens = reference
        .turns
        .iter()
        .filter_map(|turn| turn.speaker.as_deref())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(token, speaker)| (speaker, token))
        .collect::<BTreeMap<_, _>>();
    let normalized_pcm_sha256 = hex_sha256_bytes(
        normalized_pcm_sha256,
        "sidecar_pair_sampler",
        "normalized PCM digest",
    )?;
    let mut pair_sampler = SidecarPairBottomKSampler::new(PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING)?;
    extract_acoustic_features_with_frames(
        samples,
        &mut *is_cancelled,
        |frame_samples, frame, is_cancelled| {
            let observation =
                study.observe_normalized_16khz_frame(frame_samples, &frame, is_cancelled)?;
            let timestamp_ms = sidecar_frame_timestamp_ms(frame.frame_index)?;
            let reference_label = reference_sweep.label_at_ms(timestamp_ms)?;
            let speaker_token = reference_label
                .speaker
                .and_then(|speaker| speaker_tokens.get(speaker).copied());
            aggregate.submitted_frame_count = aggregate
                .submitted_frame_count
                .checked_add(1)
                .ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_aggregate_overflow",
                        "sidecar analysis frame count overflowed",
                    )
                })?;
            if let Some(previous) = previous.as_ref() {
                let contrast = acoustic_sidecar_observation_owner_contrast_from_study(
                    previous,
                    &observation,
                    study_config,
                    study_configuration_sha256_digest,
                )?;
                let component_comparisons =
                    u64::try_from(contrast.component_comparisons).map_err(|_| {
                        public_corpus_error(
                            "sidecar_aggregate_overflow",
                            "sidecar component comparison count exceeds u64",
                        )
                    })?;
                aggregate.component_comparison_count = aggregate
                    .component_comparison_count
                    .checked_add(component_comparisons)
                    .ok_or_else(|| {
                        public_corpus_error(
                            "sidecar_aggregate_overflow",
                            "sidecar component comparison count overflowed",
                        )
                    })?;
                if contrast.comparable_components > 0 {
                    aggregate.comparable_frame_count = aggregate
                        .comparable_frame_count
                        .checked_add(1)
                        .ok_or_else(|| {
                            public_corpus_error(
                                "sidecar_aggregate_overflow",
                                "sidecar comparable frame count overflowed",
                            )
                        })?;
                }
                for (count, available) in aggregate
                    .owner_available_frame_counts
                    .iter_mut()
                    .zip(contrast.owner_available)
                {
                    *count = count.checked_add(u64::from(available)).ok_or_else(|| {
                        public_corpus_error(
                            "sidecar_aggregate_overflow",
                            "sidecar owner-availability count overflowed",
                        )
                    })?;
                }
                if contrast.comparable_components
                    >= boundary_calibration.minimum_comparable_components
                    && let Some(probability) =
                        public_sidecar_probability(boundary_calibration, contrast)?
                    && let Some(positive) = reference_label.boundary
                {
                    aggregate
                        .boundary_probabilities
                        .push(probability, positive)?;
                }
            }

            for lag in PUBLIC_SIDECAR_PAIR_LAGS_FRAMES {
                let Some(expected_frame) = frame.frame_index.checked_sub(lag) else {
                    continue;
                };
                let Some(position) = ring.len().checked_sub(lag) else {
                    continue;
                };
                let left = ring.get(position).ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_pair_alignment",
                        "ordered sidecar ring position is unavailable",
                    )
                })?;
                if left.frame_index != expected_frame {
                    return Err(public_corpus_error(
                        "sidecar_pair_alignment",
                        "sidecar extractor frames are not contiguous in the ordered ring",
                    ));
                }
                let (Some(left_speaker), Some(right_speaker)) = (left.speaker_token, speaker_token)
                else {
                    continue;
                };
                let contrast = acoustic_sidecar_observation_owner_contrast_from_study(
                    &left.observation,
                    &observation,
                    study_config,
                    study_configuration_sha256_digest,
                )?;
                let maximum_contrast = (contrast.comparable_components
                    >= PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS)
                    .then(|| maximum_available_sidecar_contrast(&contrast))
                    .flatten();
                let different_speaker = left_speaker != right_speaker;
                let (channel_dominance_opportunity, channel_dominance) =
                    sidecar_auxiliary_dominance(&contrast, 1);
                let (mixed_auxiliary_dominance_opportunity, mixed_auxiliary_dominance) =
                    sidecar_auxiliary_dominance(&contrast, 2);
                pair_sampler.consider(SidecarPairSample {
                    key: sidecar_pair_selection_key(
                        &normalized_pcm_sha256,
                        left.frame_index,
                        frame.frame_index,
                        lag,
                    )?,
                    maximum_contrast,
                    different_speaker,
                    channel_dominance_opportunity,
                    channel_dominance,
                    mixed_auxiliary_dominance_opportunity,
                    mixed_auxiliary_dominance,
                })?;
            }
            let maximum_lag =
                PUBLIC_SIDECAR_PAIR_LAGS_FRAMES[PUBLIC_SIDECAR_PAIR_LAGS_FRAMES.len() - 1];
            previous = Some(observation);
            push_bounded_sidecar_ring_entry(
                &mut ring,
                frame.frame_index,
                maximum_lag,
                SidecarRingEntry {
                    frame_index: frame.frame_index,
                    speaker_token,
                    observation,
                },
                |entry| entry.frame_index,
            )?;
            Ok(())
        },
    )?;
    let (eligible_pair_count, retained_pair_count, retained_pair_capacity, selected_pairs) =
        pair_sampler.finish()?;
    update_sidecar_pair_selection_digest(
        &mut aggregate.pair_selection_hasher,
        &normalized_pcm_sha256,
        &selected_pairs,
    )?;
    aggregate.eligible_pair_count = aggregate
        .eligible_pair_count
        .checked_add(eligible_pair_count)
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "eligible pair count overflowed",
            )
        })?;
    aggregate.maximum_retained_pair_sample_count = aggregate
        .maximum_retained_pair_sample_count
        .max(u64::try_from(retained_pair_count).map_err(|_| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "retained pair sample count exceeds u64",
            )
        })?);
    let retained_pair_count = u64::try_from(retained_pair_count).map_err(|_| {
        public_corpus_error(
            "sidecar_aggregate_overflow",
            "retained pair sample count exceeds u64",
        )
    })?;
    aggregate.retained_pair_sample_count = aggregate
        .retained_pair_sample_count
        .checked_add(retained_pair_count)
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "retained pair sample count overflowed",
            )
        })?;
    let retained_different_speaker_pair_count = u64::try_from(
        selected_pairs
            .iter()
            .filter(|pair| pair.different_speaker)
            .count(),
    )
    .map_err(|_| {
        public_corpus_error(
            "sidecar_aggregate_overflow",
            "retained different-speaker pair count exceeds u64",
        )
    })?;
    let retained_same_speaker_pair_count = retained_pair_count
        .checked_sub(retained_different_speaker_pair_count)
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "retained different-speaker count exceeds the retained pair count",
            )
        })?;
    aggregate.retained_same_speaker_pair_count = aggregate
        .retained_same_speaker_pair_count
        .checked_add(retained_same_speaker_pair_count)
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "retained same-speaker pair count overflowed",
            )
        })?;
    aggregate.retained_different_speaker_pair_count = aggregate
        .retained_different_speaker_pair_count
        .checked_add(retained_different_speaker_pair_count)
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "retained different-speaker pair count overflowed",
            )
        })?;
    aggregate.retained_pair_sample_capacity = aggregate.retained_pair_sample_capacity.max(
        u64::try_from(retained_pair_capacity).map_err(|_| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "retained pair sample capacity exceeds u64",
            )
        })?,
    );
    let mut scored_pair = false;
    let mut scored_same_speaker = false;
    let mut scored_different_speaker = false;
    for pair in selected_pairs {
        if let (Some(pair_calibration), Some(maximum_contrast)) =
            (pair_calibration, pair.maximum_contrast)
        {
            let probability =
                public_sidecar_pair_probability_from_validated(pair_calibration, maximum_contrast)?;
            aggregate.pairs.push(probability, pair.different_speaker)?;
            scored_pair = true;
            scored_different_speaker |= pair.different_speaker;
            scored_same_speaker |= !pair.different_speaker;
        }
        aggregate.channel_dominance.push(
            pair.different_speaker,
            pair.channel_dominance_opportunity,
            pair.channel_dominance,
        )?;
        aggregate.mixed_auxiliary_dominance.push(
            pair.different_speaker,
            pair.mixed_auxiliary_dominance_opportunity,
            pair.mixed_auxiliary_dominance,
        )?;
    }
    aggregate.pair_scored_recording_count = aggregate
        .pair_scored_recording_count
        .checked_add(u64::from(scored_pair))
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "pair-scored recording count overflowed",
            )
        })?;
    aggregate.same_speaker_pair_recording_count = aggregate
        .same_speaker_pair_recording_count
        .checked_add(u64::from(scored_same_speaker))
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "same-speaker pair recording count overflowed",
            )
        })?;
    aggregate.different_speaker_pair_recording_count = aggregate
        .different_speaker_pair_recording_count
        .checked_add(u64::from(scored_different_speaker))
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "different-speaker pair recording count overflowed",
            )
        })?;
    Ok(())
}

fn finish_sidecar_boundary_metrics(
    probabilities: &SidecarProbabilityAccumulator,
    pipeline: &PublicCorpusAblationSplit,
) -> PublicCorpusSidecarBoundaryMetrics {
    let finished = probabilities.finish_reliability();
    PublicCorpusSidecarBoundaryMetrics {
        reference_count: pipeline.change_reference_count,
        hypothesis_count: pipeline.change_hypothesis_count,
        matched_count: pipeline.change_matched_count,
        precision: pipeline.change_precision,
        recall: pipeline.change_recall,
        f1: pipeline.change_f1,
        mean_absolute_error_sec: pipeline.change_mean_absolute_error_sec,
        probability_observation_count: probabilities.observation_count,
        probability_positive_count: probabilities.positive_count,
        brier_score: finished.brier_score,
        expected_calibration_error: finished.expected_calibration_error,
        reliability: finished.reliability,
    }
}

fn empty_sidecar_boundary_metrics(
    pipeline: &PublicCorpusAblationSplit,
) -> PublicCorpusSidecarBoundaryMetrics {
    finish_sidecar_boundary_metrics(&SidecarProbabilityAccumulator::new(), pipeline)
}

struct SidecarCalibrationFitHistogram {
    negative_counts: Vec<u64>,
    positive_counts: Vec<u64>,
}

impl SidecarCalibrationFitHistogram {
    fn new() -> Self {
        Self {
            negative_counts: vec![0; PUBLIC_SIDECAR_FIT_BINS],
            positive_counts: vec![0; PUBLIC_SIDECAR_FIT_BINS],
        }
    }

    fn push(&mut self, contrast: f64, positive: bool) -> FwResult<()> {
        if !contrast.is_finite() || !(0.0..=1.0).contains(&contrast) {
            return Err(public_corpus_error(
                "sidecar_fit_contrast",
                "calibration contrast must be finite and within [0, 1]",
            ));
        }
        let bin = ((contrast * PUBLIC_SIDECAR_FIT_BINS as f64).floor() as usize)
            .min(PUBLIC_SIDECAR_FIT_BINS - 1);
        let target = if positive {
            &mut self.positive_counts[bin]
        } else {
            &mut self.negative_counts[bin]
        };
        *target = target.checked_add(1).ok_or_else(|| {
            public_corpus_error(
                "sidecar_fit_overflow",
                "calibration histogram count overflowed",
            )
        })?;
        Ok(())
    }

    fn observation_count(&self) -> FwResult<u64> {
        self.negative_counts
            .iter()
            .chain(&self.positive_counts)
            .copied()
            .try_fold(0_u64, |total, count| {
                total.checked_add(count).ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_fit_overflow",
                        "calibration observation count overflowed",
                    )
                })
            })
    }

    fn positive_count(&self) -> FwResult<u64> {
        self.positive_counts
            .iter()
            .copied()
            .try_fold(0_u64, |total, count| {
                total.checked_add(count).ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_fit_overflow",
                        "calibration positive count overflowed",
                    )
                })
            })
    }
}

fn fit_public_sidecar_calibration(
    histogram: &SidecarCalibrationFitHistogram,
) -> FwResult<Option<PublicCorpusSidecarCalibration>> {
    let observation_count = histogram.observation_count()?;
    let positive_count = histogram.positive_count()?;
    let negative_count = observation_count
        .checked_sub(positive_count)
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_fit_overflow",
                "calibration positive count exceeds its observation count",
            )
        })?;
    if positive_count == 0 || negative_count == 0 {
        return Ok(None);
    }
    let mut best = None::<(f64, f32, f32)>;
    for intercept_step in -32_i32..=32 {
        let intercept = intercept_step as f32 * 0.25;
        for weight_step in 0_u32..=64 {
            let weight = weight_step as f32 * 0.25;
            let calibration = AcousticSidecarFusionCalibration {
                logit_intercept: intercept,
                contrast_weight: weight,
                minimum_comparable_components: PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS,
            };
            let mut positive_loss = 0.0;
            let mut negative_loss = 0.0;
            for index in 0..PUBLIC_SIDECAR_FIT_BINS {
                let contrast_value = ((index as f64 + 0.5) / PUBLIC_SIDECAR_FIT_BINS as f64) as f32;
                let contrast = crate::diarization::AcousticSidecarOwnerContrast {
                    owner_contrast: [contrast_value, 0.0, 0.0],
                    owner_available: [true, false, false],
                    comparable_components: PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS,
                    component_comparisons: PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS,
                };
                let probability = f64::from(
                    acoustic_sidecar_calibrate_owner_contrast(contrast, calibration)?
                        .ok_or_else(|| {
                            public_corpus_error(
                                "sidecar_fit_calibration",
                                "valid fit contrast unexpectedly produced no probability",
                            )
                        })?
                        .probability,
                );
                positive_loss -= histogram.positive_counts[index] as f64 * probability.ln();
                negative_loss -= histogram.negative_counts[index] as f64 * (1.0 - probability).ln();
            }
            // Ordinary empirical NLL retains the observed class prior. Equal
            // class weighting would produce a discrimination score centered
            // near 0.5 rather than a calibrated probability when boundaries
            // or speaker changes are rare.
            let empirical_loss = (positive_loss + negative_loss) / observation_count as f64;
            if !empirical_loss.is_finite() {
                continue;
            }
            let candidate = (empirical_loss, intercept, weight);
            if best.is_none_or(|best| {
                candidate.0 < best.0
                    || (candidate.0 == best.0 && (candidate.2, candidate.1) < (best.2, best.1))
            }) {
                best = Some(candidate);
            }
        }
    }
    let (_, intercept, weight) = best.ok_or_else(|| {
        public_corpus_error(
            "sidecar_fit_calibration",
            "calibration grid produced no finite candidate",
        )
    })?;
    let fitted = AcousticSidecarFusionCalibration {
        logit_intercept: intercept,
        contrast_weight: weight,
        minimum_comparable_components: PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS,
    };
    let mut brier_sum = 0.0;
    for index in 0..PUBLIC_SIDECAR_FIT_BINS {
        let contrast_value = ((index as f64 + 0.5) / PUBLIC_SIDECAR_FIT_BINS as f64) as f32;
        let contrast = crate::diarization::AcousticSidecarOwnerContrast {
            owner_contrast: [contrast_value, 0.0, 0.0],
            owner_available: [true, false, false],
            comparable_components: PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS,
            component_comparisons: PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS,
        };
        let probability = f64::from(
            acoustic_sidecar_calibrate_owner_contrast(contrast, fitted)?
                .ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_fit_calibration",
                        "valid fitted contrast unexpectedly produced no probability",
                    )
                })?
                .probability,
        );
        brier_sum += histogram.positive_counts[index] as f64 * (1.0 - probability).powi(2)
            + histogram.negative_counts[index] as f64 * probability.powi(2);
    }
    let mut calibration = PublicCorpusSidecarCalibration {
        fit_id: PUBLIC_CORPUS_SIDECAR_CALIBRATION_FIT_VERSION.to_owned(),
        logit_intercept: f64::from(intercept),
        contrast_weight: f64::from(weight),
        minimum_comparable_components: PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS,
        fit_observation_count: observation_count,
        fit_positive_count: positive_count,
        fit_brier_score: positive_ratio(brier_sum, observation_count as f64),
        calibration_sha256: String::new(),
    };
    calibration.calibration_sha256 = public_sidecar_calibration_sha256(&calibration)?;
    Ok(Some(calibration))
}

fn public_sidecar_calibration_sha256(
    calibration: &PublicCorpusSidecarCalibration,
) -> FwResult<String> {
    canonical_sha256(&PublicCorpusSidecarCalibrationFingerprint {
        fit_id: &calibration.fit_id,
        logit_intercept: calibration.logit_intercept,
        contrast_weight: calibration.contrast_weight,
        minimum_comparable_components: calibration.minimum_comparable_components,
        fit_observation_count: calibration.fit_observation_count,
        fit_positive_count: calibration.fit_positive_count,
        fit_brier_score: calibration.fit_brier_score,
    })
}

fn fit_public_sidecar_pair_calibration(
    histogram: &SidecarCalibrationFitHistogram,
) -> FwResult<Option<PublicCorpusSidecarPairCalibration>> {
    let Some(boundary_shape) = fit_public_sidecar_calibration(histogram)? else {
        return Ok(None);
    };
    let mut calibration = PublicCorpusSidecarPairCalibration {
        fit_id: PUBLIC_CORPUS_SIDECAR_PAIR_CALIBRATION_FIT_VERSION.to_owned(),
        target_id: PUBLIC_CORPUS_SIDECAR_PAIR_PROBABILITY_TARGET_VERSION.to_owned(),
        logit_intercept: boundary_shape.logit_intercept,
        contrast_weight: boundary_shape.contrast_weight,
        minimum_comparable_components: boundary_shape.minimum_comparable_components,
        fit_observation_count: boundary_shape.fit_observation_count,
        fit_positive_count: boundary_shape.fit_positive_count,
        fit_brier_score: boundary_shape.fit_brier_score,
        calibration_sha256: String::new(),
    };
    calibration.calibration_sha256 = public_sidecar_pair_calibration_sha256(&calibration)?;
    Ok(Some(calibration))
}

fn public_sidecar_pair_calibration_sha256(
    calibration: &PublicCorpusSidecarPairCalibration,
) -> FwResult<String> {
    canonical_sha256(&PublicCorpusSidecarPairCalibrationFingerprint {
        fit_id: &calibration.fit_id,
        target_id: &calibration.target_id,
        logit_intercept: calibration.logit_intercept,
        contrast_weight: calibration.contrast_weight,
        minimum_comparable_components: calibration.minimum_comparable_components,
        fit_observation_count: calibration.fit_observation_count,
        fit_positive_count: calibration.fit_positive_count,
        fit_brier_score: calibration.fit_brier_score,
    })
}

fn validate_public_sidecar_pair_calibration(
    calibration: &PublicCorpusSidecarPairCalibration,
) -> FwResult<()> {
    if calibration.fit_id != PUBLIC_CORPUS_SIDECAR_PAIR_CALIBRATION_FIT_VERSION
        || calibration.target_id != PUBLIC_CORPUS_SIDECAR_PAIR_PROBABILITY_TARGET_VERSION
        || calibration.minimum_comparable_components != PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS
        || calibration.fit_observation_count == 0
        || calibration.fit_positive_count == 0
        || calibration.fit_positive_count >= calibration.fit_observation_count
        || !calibration.logit_intercept.is_finite()
        || !calibration.contrast_weight.is_finite()
        || !(-8.0..=8.0).contains(&calibration.logit_intercept)
        || !(0.0..=16.0).contains(&calibration.contrast_weight)
        || (calibration.logit_intercept * 4.0).fract() != 0.0
        || (calibration.contrast_weight * 4.0).fract() != 0.0
        || f64::from(calibration.logit_intercept as f32) != calibration.logit_intercept
        || f64::from(calibration.contrast_weight as f32) != calibration.contrast_weight
        || !calibration
            .fit_brier_score
            .is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        || !is_sha256_hex(&calibration.calibration_sha256)
        || public_sidecar_pair_calibration_sha256(calibration)? != calibration.calibration_sha256
    {
        return Err(public_corpus_error(
            "sidecar_pair_calibration",
            "pair calibration identity, target, bounds, class support, or hash is invalid",
        ));
    }
    Ok(())
}

fn public_sidecar_pair_probability_from_validated(
    calibration: &PublicCorpusSidecarPairCalibration,
    maximum_contrast: f64,
) -> FwResult<f64> {
    if !maximum_contrast.is_finite() || !(0.0..=1.0).contains(&maximum_contrast) {
        return Err(public_corpus_error(
            "sidecar_pair_calibration",
            "pair contrast must be finite and within [0, 1]",
        ));
    }
    let contrast = crate::diarization::AcousticSidecarOwnerContrast {
        owner_contrast: [maximum_contrast as f32, 0.0, 0.0],
        owner_available: [true, false, false],
        comparable_components: calibration.minimum_comparable_components,
        component_comparisons: calibration.minimum_comparable_components,
    };
    acoustic_sidecar_calibrate_owner_contrast(
        contrast,
        AcousticSidecarFusionCalibration {
            logit_intercept: calibration.logit_intercept as f32,
            contrast_weight: calibration.contrast_weight as f32,
            minimum_comparable_components: calibration.minimum_comparable_components,
        },
    )?
    .map(|evidence| f64::from(evidence.probability))
    .ok_or_else(|| {
        public_corpus_error(
            "sidecar_pair_calibration",
            "valid pair contrast unexpectedly produced no probability",
        )
    })
}

fn sidecar_evaluation_request(
    lane: PublicCorpusSidecarLane,
    calibration: &PublicCorpusSidecarCalibration,
) -> FwResult<AcousticSidecarEvaluationRequest> {
    if calibration.fit_id != PUBLIC_CORPUS_SIDECAR_CALIBRATION_FIT_VERSION
        || calibration.minimum_comparable_components != PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS
        || !calibration.logit_intercept.is_finite()
        || !calibration.contrast_weight.is_finite()
        || !(-8.0..=8.0).contains(&calibration.logit_intercept)
        || !(0.0..=16.0).contains(&calibration.contrast_weight)
        || (calibration.logit_intercept * 4.0).fract() != 0.0
        || (calibration.contrast_weight * 4.0).fract() != 0.0
        || f64::from(calibration.logit_intercept as f32) != calibration.logit_intercept
        || f64::from(calibration.contrast_weight as f32) != calibration.contrast_weight
        || public_sidecar_calibration_sha256(calibration)? != calibration.calibration_sha256
    {
        return Err(public_corpus_error(
            "sidecar_calibration",
            "sidecar calibration identity, bounds, or hash is invalid",
        ));
    }
    if lane == PublicCorpusSidecarLane::FullV2Baseline {
        return Err(public_corpus_error(
            "sidecar_calibration",
            "the unfused baseline cannot receive a sidecar calibration",
        ));
    }
    Ok(AcousticSidecarEvaluationRequest {
        study_config: lane.study_config(),
        calibration: AcousticSidecarFusionCalibration {
            logit_intercept: calibration.logit_intercept as f32,
            contrast_weight: calibration.contrast_weight as f32,
            minimum_comparable_components: calibration.minimum_comparable_components,
        },
    })
}

struct LoadedPublicSidecarRecording {
    samples: Vec<f32>,
    reference: DiarizationReferenceDocument,
    boundary_hints: AcousticBoundaryHints,
}

struct FittedPublicSidecarCalibrations {
    boundary: PublicCorpusSidecarCalibration,
    pair: Option<PublicCorpusSidecarPairCalibration>,
}

fn load_public_sidecar_recording(
    reference: &DiarizationReferenceDocument,
    recording_evidence: &PublicCorpusRecordingEvidence,
    input_recording: &PublicCorpusInputRecording,
    canonical_input: &Path,
    maximum_recording_duration_ms: Option<u64>,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<LoadedPublicSidecarRecording> {
    if recording_evidence.sample_rate_hz != 16_000
        || recording_evidence.channel_count != 1
        || recording_evidence.selected_channel != 1
    {
        return Err(public_corpus_error(
            "sidecar_audio_contract",
            "the acoustic sidecar runner requires 16 kHz mono PCM WAV input",
        ));
    }
    let audio_path =
        canonical_relative_file(canonical_input, &input_recording.audio_path, "audio")?;
    let audio_bytes = read_bounded(&audio_path, MAX_EVALUATION_AUDIO_BYTES, "sidecar_audio")?;
    checkpoint_cancelled(is_cancelled)?;
    if format!("{:x}", Sha256::digest(&audio_bytes)) != recording_evidence.audio_sha256 {
        return Err(public_corpus_error(
            "audio_changed",
            "audio changed after bundle validation and before sidecar execution",
        ));
    }
    let mut samples =
        crate::native_engine::decode::read_wav_16k_mono(&audio_bytes).map_err(|_| {
            public_corpus_error(
                "sidecar_audio_decode",
                "one validated WAV could not be decoded as 16 kHz mono PCM",
            )
        })?;
    checkpoint_cancelled(is_cancelled)?;
    let available_duration_ms = u64::try_from(samples.len()).unwrap_or(u64::MAX) / 16;
    let evaluation_duration_ms = maximum_recording_duration_ms
        .map_or(available_duration_ms, |max| max.min(available_duration_ms));
    let clipped_reference = clipped_reference(reference, Some(evaluation_duration_ms))?;
    let maximum_samples = usize::try_from(clipped_reference.duration_ms)
        .ok()
        .and_then(|duration_ms| duration_ms.checked_mul(16))
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_duration",
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
    Ok(LoadedPublicSidecarRecording {
        samples,
        reference: clipped_reference,
        boundary_hints,
    })
}

#[allow(clippy::too_many_arguments)]
fn fit_public_sidecar_lane(
    bundle: &PublicCorpusBundle,
    input_recordings: &BTreeMap<String, PublicCorpusInputRecording>,
    canonical_input: &Path,
    maximum_recording_duration_ms: Option<u64>,
    lane: PublicCorpusSidecarLane,
    target_split: EvaluationSplit,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<Option<FittedPublicSidecarCalibrations>> {
    let mut boundary_histogram = SidecarCalibrationFitHistogram::new();
    let mut pair_histogram = SidecarCalibrationFitHistogram::new();
    for ((reference, recording_evidence), manifest_recording) in bundle
        .references
        .iter()
        .zip(&bundle.recordings)
        .zip(&bundle.manifest.recordings)
    {
        if manifest_recording.split != target_split {
            continue;
        }
        checkpoint_cancelled(is_cancelled)?;
        let input_recording = input_recordings
            .get(&reference.recording_id)
            .ok_or_else(|| {
                public_corpus_error(
                    "sidecar_alignment",
                    "validated recording is absent from the descriptor",
                )
            })?;
        let loaded = load_public_sidecar_recording(
            reference,
            recording_evidence,
            input_recording,
            canonical_input,
            maximum_recording_duration_ms,
            is_cancelled,
        )?;
        let reference_changes =
            speaker_change_points_ms(&loaded.reference.turns, loaded.reference.duration_ms, true)?;
        let mut reference_sweep =
            SidecarReferenceSweep::new(&loaded.reference, &reference_changes)?;
        let speaker_tokens = loaded
            .reference
            .turns
            .iter()
            .filter_map(|turn| turn.speaker.as_deref())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .enumerate()
            .map(|(token, speaker)| (speaker, token))
            .collect::<BTreeMap<_, _>>();
        let normalized_pcm_sha256 = hex_sha256_bytes(
            &hash_pcm_prefix(&loaded.samples),
            "sidecar_pair_sampler",
            "normalized PCM digest",
        )?;
        let study_config = lane.study_config();
        let mut study = AcousticSidecarStudy::new(study_config)?;
        let study_configuration_sha256_digest = study.configuration_sha256_digest();
        let mut previous = None;
        let mut ring = VecDeque::<SidecarRingEntry>::with_capacity(
            PUBLIC_SIDECAR_PAIR_LAGS_FRAMES[PUBLIC_SIDECAR_PAIR_LAGS_FRAMES.len() - 1] + 1,
        );
        let mut pair_sampler =
            SidecarPairBottomKSampler::new(PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING)?;
        extract_acoustic_features_with_frames(
            &loaded.samples,
            &mut *is_cancelled,
            |frame_samples, frame, is_cancelled| {
                let observation =
                    study.observe_normalized_16khz_frame(frame_samples, &frame, is_cancelled)?;
                let timestamp_ms = sidecar_frame_timestamp_ms(frame.frame_index)?;
                let reference_label = reference_sweep.label_at_ms(timestamp_ms)?;
                let speaker_token = reference_label
                    .speaker
                    .and_then(|speaker| speaker_tokens.get(speaker).copied());
                if let Some(previous) = previous.as_ref() {
                    let contrast = acoustic_sidecar_observation_owner_contrast_from_study(
                        previous,
                        &observation,
                        study_config,
                        study_configuration_sha256_digest,
                    )?;
                    if contrast.comparable_components
                        >= PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS
                        && let Some(maximum_contrast) =
                            maximum_available_sidecar_contrast(&contrast)
                        && let Some(positive) = reference_label.boundary
                    {
                        boundary_histogram.push(maximum_contrast, positive)?;
                    }
                }
                for lag in PUBLIC_SIDECAR_PAIR_LAGS_FRAMES {
                    let Some(expected_frame) = frame.frame_index.checked_sub(lag) else {
                        continue;
                    };
                    let Some(position) = ring.len().checked_sub(lag) else {
                        continue;
                    };
                    let left = ring.get(position).ok_or_else(|| {
                        public_corpus_error(
                            "sidecar_pair_alignment",
                            "ordered calibration ring position is unavailable",
                        )
                    })?;
                    if left.frame_index != expected_frame {
                        return Err(public_corpus_error(
                            "sidecar_pair_alignment",
                            "calibration extractor frames are not contiguous",
                        ));
                    }
                    let (Some(left_speaker), Some(right_speaker)) =
                        (left.speaker_token, speaker_token)
                    else {
                        continue;
                    };
                    let contrast = acoustic_sidecar_observation_owner_contrast_from_study(
                        &left.observation,
                        &observation,
                        study_config,
                        study_configuration_sha256_digest,
                    )?;
                    let maximum_contrast = (contrast.comparable_components
                        >= PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS)
                        .then(|| maximum_available_sidecar_contrast(&contrast))
                        .flatten();
                    let different_speaker = left_speaker != right_speaker;
                    let (channel_dominance_opportunity, channel_dominance) =
                        sidecar_auxiliary_dominance(&contrast, 1);
                    let (mixed_auxiliary_dominance_opportunity, mixed_auxiliary_dominance) =
                        sidecar_auxiliary_dominance(&contrast, 2);
                    pair_sampler.consider(SidecarPairSample {
                        key: sidecar_pair_selection_key(
                            &normalized_pcm_sha256,
                            left.frame_index,
                            frame.frame_index,
                            lag,
                        )?,
                        maximum_contrast,
                        different_speaker,
                        channel_dominance_opportunity,
                        channel_dominance,
                        mixed_auxiliary_dominance_opportunity,
                        mixed_auxiliary_dominance,
                    })?;
                }
                previous = Some(observation);
                let maximum_lag =
                    PUBLIC_SIDECAR_PAIR_LAGS_FRAMES[PUBLIC_SIDECAR_PAIR_LAGS_FRAMES.len() - 1];
                push_bounded_sidecar_ring_entry(
                    &mut ring,
                    frame.frame_index,
                    maximum_lag,
                    SidecarRingEntry {
                        frame_index: frame.frame_index,
                        speaker_token,
                        observation,
                    },
                    |entry| entry.frame_index,
                )?;
                Ok(())
            },
        )?;
        let (_, _, _, selected_pairs) = pair_sampler.finish()?;
        for pair in selected_pairs {
            if let Some(maximum_contrast) = pair.maximum_contrast {
                pair_histogram.push(maximum_contrast, pair.different_speaker)?;
            }
        }
    }
    let boundary = fit_public_sidecar_calibration(&boundary_histogram)?;
    let pair = fit_public_sidecar_pair_calibration(&pair_histogram)?;
    Ok(boundary.map(|boundary| FittedPublicSidecarCalibrations { boundary, pair }))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_public_sidecar_lane(
    bundle: &PublicCorpusBundle,
    input_recordings: &BTreeMap<String, PublicCorpusInputRecording>,
    canonical_input: &Path,
    maximum_recording_duration_ms: Option<u64>,
    diarization_request: &DiarizationRequest,
    scorer_config: &DiarizationScorerConfig,
    lane: PublicCorpusSidecarLane,
    boundary_calibration: Option<&PublicCorpusSidecarCalibration>,
    pair_calibration: Option<&PublicCorpusSidecarPairCalibration>,
    detector_mode: AcousticChangeDetectorMode,
    clustering_mode: AcousticClusteringMode,
    target_split: EvaluationSplit,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<SidecarLaneEvaluation> {
    let sidecar_request = match (lane, boundary_calibration, pair_calibration) {
        (PublicCorpusSidecarLane::FullV2Baseline, None, None) => None,
        (PublicCorpusSidecarLane::FullV2Baseline, _, _) => {
            return Err(public_corpus_error(
                "sidecar_lane_configuration",
                "the unfused baseline cannot receive boundary or pair calibration",
            ));
        }
        (_, Some(boundary), pair) => {
            if let Some(pair) = pair {
                validate_public_sidecar_pair_calibration(pair)?;
            }
            Some(sidecar_evaluation_request(lane, boundary)?)
        }
        (_, _, _) => {
            return Err(public_corpus_error(
                "sidecar_lane_configuration",
                "every fused sidecar lane requires a locked boundary calibration and cannot retain a pair calibration alone",
            ));
        }
    };
    let expected_sidecar_configuration_sha256 = sidecar_request
        .map(|request| acoustic_sidecar_study_config_sha256(request.study_config))
        .transpose()?;
    let expected_fusion_configuration_sha256 = sidecar_request
        .map(|request| acoustic_sidecar_fusion_configuration_sha256(request, detector_mode))
        .transpose()?;
    let mut pipeline = PublicAblationAccumulator::default();
    let mut observations = SidecarObservationAccumulator::new();
    let mut operations = SidecarOperationsAccumulator::default();
    let mut recording_accuracy = Vec::new();
    let mut selected_recording_count = 0_u64;

    for ((reference, recording_evidence), manifest_recording) in bundle
        .references
        .iter()
        .zip(&bundle.recordings)
        .zip(&bundle.manifest.recordings)
    {
        if manifest_recording.split != target_split {
            continue;
        }
        selected_recording_count = selected_recording_count.checked_add(1).ok_or_else(|| {
            public_corpus_error(
                "sidecar_aggregate_overflow",
                "selected recording count overflowed",
            )
        })?;
        checkpoint_cancelled(is_cancelled)?;
        let input_recording = input_recordings
            .get(&reference.recording_id)
            .ok_or_else(|| {
                public_corpus_error(
                    "sidecar_alignment",
                    "validated recording is absent from the descriptor",
                )
            })?;
        let loaded = load_public_sidecar_recording(
            reference,
            recording_evidence,
            input_recording,
            canonical_input,
            maximum_recording_duration_ms,
            is_cancelled,
        )?;
        let normalized_pcm_sha256 = hash_pcm_prefix(&loaded.samples);
        let started = Instant::now();
        let (
            report_turns,
            speaker_count_estimate,
            detector_changes,
            evaluated_changes,
            clustering_evidence,
            sidecar_evidence,
        ) = if loaded.boundary_hints.speech_regions_ms.is_empty() {
            checkpoint_cancelled(is_cancelled)?;
            let sidecar_evidence = sidecar_request.map_or_else(
                AcousticSidecarFusionEvaluationEvidence::default,
                |_| AcousticSidecarFusionEvaluationEvidence {
                    fusion_requested: true,
                    fusion_configuration_sha256: expected_fusion_configuration_sha256.clone(),
                    sidecar_configuration_sha256: expected_sidecar_configuration_sha256.clone(),
                    ..AcousticSidecarFusionEvaluationEvidence::default()
                },
            );
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
                    count_merge_steps: Vec::new(),
                },
                sidecar_evidence,
            )
        } else {
            let (report, _, change_evidence, clustering_evidence, sidecar_evidence) =
                diarize_acoustic_pcm_with_sidecar_evidence(
                    AcousticDiarizationInput {
                        samples: &loaded.samples,
                        normalized_input_sha256: &normalized_pcm_sha256,
                        segments: &[],
                        word_aligned: false,
                        request: diarization_request,
                        boundary_hints: &loaded.boundary_hints,
                    },
                    detector_mode,
                    clustering_mode,
                    sidecar_request,
                    &mut *is_cancelled,
                )?;
            let detector_changes = change_evidence
                .emitted
                .iter()
                .filter(|evidence| {
                    !evidence.vad_boundary
                        && !evidence.supervised_boundary
                        && evidence.boundary_ms > 0
                        && evidence.boundary_ms < loaded.reference.duration_ms
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
                sidecar_evidence,
            )
        };
        let wall_time_ms = u64::try_from(started.elapsed().as_millis())
            .unwrap_or(u64::MAX)
            .max(1);
        let peak_rss_bytes = sampled_process_rss_bytes();
        if let Some(boundary_calibration) = boundary_calibration {
            if loaded.boundary_hints.speech_regions_ms.is_empty() {
                // The selected-pair digest is record-framed, so a candidate
                // recording with no scoreable speech must still contribute
                // its normalized PCM identity and an explicit zero-pair
                // count. Otherwise an empty/fully ignored recording could be
                // changed or omitted without changing the evidence identity.
                let normalized_pcm_sha256 = hex_sha256_bytes(
                    &normalized_pcm_sha256,
                    "sidecar_pair_sampler",
                    "normalized PCM digest",
                )?;
                update_sidecar_pair_selection_digest(
                    &mut observations.pair_selection_hasher,
                    &normalized_pcm_sha256,
                    &[],
                )?;
            } else {
                analyze_sidecar_observations(
                    &loaded.samples,
                    SidecarObservationAnalysis {
                        reference: &loaded.reference,
                        normalized_pcm_sha256: &normalized_pcm_sha256,
                        study_config: lane.study_config(),
                        boundary_calibration,
                        pair_calibration,
                    },
                    &mut observations,
                    is_cancelled,
                )?;
            }
        }
        operations.push(&sidecar_evidence)?;
        let hypothesis = DiarizationHypothesisDocument {
            schema_version: DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: loaded.reference.recording_id.clone(),
            duration_ms: loaded.reference.duration_ms,
            turns: report_turns
                .into_iter()
                .map(|turn| EvaluationTurn {
                    start_ms: turn.start_ms,
                    end_ms: turn.end_ms.min(loaded.reference.duration_ms),
                    speaker: turn.speaker_ref,
                    speaker_confidence: turn.speaker_confidence,
                    overlap_suspected: turn.overlap_suspected,
                })
                .filter(|turn| turn.end_ms > turn.start_ms)
                .collect(),
            speaker_count_estimate,
            performance: Some(EvaluationPerformanceObservation {
                audio_duration_ms: loaded.reference.duration_ms,
                wall_time_ms,
                peak_rss_bytes,
            }),
        };
        let score = score_diarization_documents(&loaded.reference, &hypothesis, scorer_config)?;
        let reference_changes =
            speaker_change_points_ms(&loaded.reference.turns, loaded.reference.duration_ms, true)?;
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
                            && evidence.boundary_ms < loaded.reference.duration_ms)
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
        recording_accuracy.push(SidecarRecordingAccuracy {
            recording_audio_sha256: hex_sha256_bytes(
                &recording_evidence.audio_sha256,
                "sidecar_pair_alignment",
                "recording audio digest",
            )?,
            reference_sha256: hex_sha256_bytes(
                &recording_evidence.reference_sha256,
                "sidecar_pair_alignment",
                "reference digest",
            )?,
            der: score.diarization.der,
            jer: score.diarization.jer,
        });
        pipeline.push(
            &score,
            &detector_change_score,
            &change_calibration,
            &change_collar_metrics,
            &change_threshold_sweep,
            &clustering_evidence,
        )?;
    }
    if selected_recording_count == 0 {
        return Err(public_corpus_error(
            "sidecar_split_missing",
            "the selected evaluation stage has no recording in the validated bundle",
        ));
    }
    if operations.evaluated_recording_count != selected_recording_count {
        return Err(public_corpus_error(
            "sidecar_recording_alignment",
            "the sidecar reducer did not account for every selected recording",
        ));
    }
    let aggregate_pipeline = pipeline.finish(target_split);
    let is_baseline = lane == PublicCorpusSidecarLane::FullV2Baseline;
    if is_baseline {
        if operations.fusion_requested
            || operations.fusion_executed
            || operations.fusion_requested_recording_count != 0
            || operations.fusion_executed_recording_count != 0
            || operations.expected_sidecar_configuration_sha256.is_some()
            || operations.expected_fusion_configuration_sha256.is_some()
        {
            return Err(public_corpus_error(
                "sidecar_baseline_fusion",
                "the unfused baseline unexpectedly reported sidecar activity",
            ));
        }
    } else if !operations.fusion_requested
        || operations.fusion_requested_recording_count != selected_recording_count
        || operations.expected_sidecar_configuration_sha256 != expected_sidecar_configuration_sha256
        || operations.expected_fusion_configuration_sha256 != expected_fusion_configuration_sha256
        || operations.fusion_executed
            != (operations.consumed_probability_count > 0
                && operations.fusion_executed_recording_count > 0)
    {
        return Err(public_corpus_error(
            "sidecar_fusion_evidence",
            "candidate fusion request, hashes, or downstream consumption evidence is inconsistent",
        ));
    }
    if !is_baseline
        && (observations.submitted_frame_count != operations.submitted_frame_count
            || observations.comparable_frame_count != operations.comparable_frame_count)
    {
        return Err(public_corpus_error(
            "sidecar_observation_alignment",
            "bounded diagnostic observations do not match fused frame accounting",
        ));
    }
    let boundary = if is_baseline {
        empty_sidecar_boundary_metrics(&aggregate_pipeline)
    } else {
        finish_sidecar_boundary_metrics(&observations.boundary_probabilities, &aggregate_pipeline)
    };
    let conditional_pairs = observations.pairs.finish()?;
    let conditional_pair_count = conditional_pairs.comparison_count;
    let conditional_same_speaker_count = conditional_pairs.same_speaker_count;
    let conditional_different_speaker_count = conditional_pairs.different_speaker_count;
    let performance = PublicCorpusSidecarPerformance {
        audio_duration_sec: aggregate_pipeline.audio_duration_sec,
        wall_time_sec: aggregate_pipeline.wall_time_sec,
        real_time_factor: aggregate_pipeline.real_time_factor,
        sampled_peak_rss_bytes: aggregate_pipeline.sampled_peak_rss_bytes,
    };
    let pipeline_is_publishable = is_baseline || operations.fusion_executed;
    if !pipeline_is_publishable {
        recording_accuracy.clear();
    }
    let split = PublicCorpusSidecarStudySplit {
        split: target_split,
        pipeline: pipeline_is_publishable.then_some(aggregate_pipeline),
        fusion_executed: operations.fusion_executed,
        boundary,
        conditional_pairs,
        coverage: PublicCorpusSidecarCoverage {
            fusion_requested: operations.fusion_requested,
            evaluated_recording_count: operations.evaluated_recording_count,
            fusion_requested_recording_count: operations.fusion_requested_recording_count,
            fusion_executed_recording_count: operations.fusion_executed_recording_count,
            submitted_frame_count: operations.submitted_frame_count,
            comparable_frame_count: operations.comparable_frame_count,
            calibrated_signal_count: operations.calibrated_signal_count,
            consumed_probability_count: operations.consumed_probability_count,
            changed_boundary_probability_count: operations.changed_boundary_probability_count,
            comparable_frame_coverage: ratio(
                operations.comparable_frame_count,
                operations.submitted_frame_count,
            ),
            component_comparison_count: observations.component_comparison_count,
            owner_available_frame_counts: observations.owner_available_frame_counts,
            channel_dominance: observations.channel_dominance.finish(),
            mixed_auxiliary_dominance: observations.mixed_auxiliary_dominance.finish(),
            eligible_pair_count: observations.eligible_pair_count,
            retained_pair_sample_count: observations.retained_pair_sample_count,
            retained_same_speaker_pair_count: observations.retained_same_speaker_pair_count,
            retained_different_speaker_pair_count: observations
                .retained_different_speaker_pair_count,
            pair_selection_sha256: (!is_baseline && operations.evaluated_recording_count > 0).then(
                || {
                    format!(
                        "{:x}",
                        observations.pair_selection_hasher.clone().finalize()
                    )
                },
            ),
            pair_score_coverage: ratio(
                conditional_pair_count,
                observations.retained_pair_sample_count,
            ),
            same_speaker_pair_score_coverage: ratio(
                conditional_same_speaker_count,
                observations.retained_same_speaker_pair_count,
            ),
            different_speaker_pair_score_coverage: ratio(
                conditional_different_speaker_count,
                observations.retained_different_speaker_pair_count,
            ),
            pair_scored_recording_count: observations.pair_scored_recording_count,
            same_speaker_pair_recording_count: observations.same_speaker_pair_recording_count,
            different_speaker_pair_recording_count: observations
                .different_speaker_pair_recording_count,
            maximum_retained_pair_sample_count: observations.maximum_retained_pair_sample_count,
            retained_pair_sample_capacity: observations.retained_pair_sample_capacity,
            maximum_retained_signal_count: operations.maximum_retained_signal_count,
            retained_signal_capacity: operations.retained_signal_capacity,
        },
        operations: operations.operations(),
        performance,
        paired_uncertainty: None,
    };
    Ok(SidecarLaneEvaluation {
        split,
        recording_accuracy,
    })
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn bootstrap_mean_interval(
    deltas: &[f64],
    seed: &[u8; 32],
    stream: u64,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<(Option<f64>, Option<f64>, Option<f64>)> {
    checkpoint_cancelled(is_cancelled)?;
    if deltas.is_empty() {
        return Ok((None, None, None));
    }
    let delta_count = u64::try_from(deltas.len()).map_err(|_| {
        public_corpus_error(
            "sidecar_bootstrap_bound",
            "paired uncertainty row count exceeds the retained u64 range",
        )
    })?;
    let mut observed_sum = 0.0;
    for (index, delta) in deltas.iter().enumerate() {
        if index % 4_096 == 0 {
            checkpoint_cancelled(is_cancelled)?;
        }
        observed_sum += delta;
    }
    let mean = canonical_evidence_number(observed_sum / deltas.len() as f64);
    let mut replicates = Vec::with_capacity(PUBLIC_SIDECAR_BOOTSTRAP_REPLICATES);
    for replicate in 0..PUBLIC_SIDECAR_BOOTSTRAP_REPLICATES {
        if replicate % 16 == 0 {
            checkpoint_cancelled(is_cancelled)?;
        }
        let replicate = u64::try_from(replicate).map_err(|_| {
            public_corpus_error(
                "sidecar_bootstrap_bound",
                "bootstrap replicate index exceeds the retained u64 range",
            )
        })?;
        let mut hasher = Sha256::new();
        hasher.update(seed);
        hasher.update(stream.to_le_bytes());
        hasher.update(replicate.to_le_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        let mut state = u64::from_le_bytes(digest[..8].try_into().map_err(|_| {
            public_corpus_error(
                "sidecar_bootstrap_seed",
                "bootstrap stream digest did not contain eight seed bytes",
            )
        })?);
        let mut sum = 0.0;
        for draw in 0..deltas.len() {
            if draw % 4_096 == 0 {
                checkpoint_cancelled(is_cancelled)?;
            }
            let index = splitmix64_next(&mut state) % delta_count;
            sum += deltas[index as usize];
        }
        replicates.push(sum / deltas.len() as f64);
    }
    replicates.sort_by(f64::total_cmp);
    let last = replicates.len() - 1;
    let lower = replicates[last * 25 / 1_000];
    let upper = replicates[last * 975 / 1_000];
    Ok((
        Some(mean),
        Some(canonical_evidence_number(lower)),
        Some(canonical_evidence_number(upper)),
    ))
}

fn hex_sha256_bytes(value: &str, code: &str, field: &str) -> FwResult<[u8; 32]> {
    if !is_sha256_hex(value) {
        return Err(public_corpus_error(
            code,
            &format!("{field} must be lowercase SHA-256"),
        ));
    }
    let mut output = [0_u8; 32];
    let bytes = value.as_bytes();
    let nibble = |byte: u8| -> u8 {
        match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => 0,
        }
    };
    for (index, output) in output.iter_mut().enumerate() {
        *output = (nibble(bytes[index * 2]) << 4) | nibble(bytes[index * 2 + 1]);
    }
    Ok(output)
}

fn paired_sidecar_uncertainty(
    baseline: &[SidecarRecordingAccuracy],
    candidate: &[SidecarRecordingAccuracy],
    lane: PublicCorpusSidecarLane,
    split: EvaluationSplit,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<PublicCorpusSidecarPairedUncertainty> {
    if baseline.len() != candidate.len() {
        return Err(public_corpus_error(
            "sidecar_pair_alignment",
            "baseline and candidate recording orders have different cardinality",
        ));
    }
    let seed_fingerprint = PublicCorpusSidecarBootstrapSeedFingerprint {
        uncertainty_id: PUBLIC_CORPUS_SIDECAR_UNCERTAINTY_VERSION,
        seed_policy_id: PUBLIC_SIDECAR_BOOTSTRAP_SEED_POLICY,
        sampler_id: PUBLIC_SIDECAR_BOOTSTRAP_SAMPLER,
        lane,
        split,
        replicates: PUBLIC_SIDECAR_BOOTSTRAP_REPLICATES,
    };
    let bootstrap_seed_sha256 = canonical_sha256(&seed_fingerprint)?;
    let seed: [u8; 32] = hex_sha256_bytes(
        &bootstrap_seed_sha256,
        "sidecar_bootstrap_seed",
        "bootstrap seed",
    )?;
    let mut der_deltas = Vec::new();
    let mut jer_deltas = Vec::new();
    for index in 0..baseline.len() {
        if index % 4_096 == 0 {
            checkpoint_cancelled(is_cancelled)?;
        }
        if baseline[index].recording_audio_sha256 != candidate[index].recording_audio_sha256
            || baseline[index].reference_sha256 != candidate[index].reference_sha256
        {
            return Err(public_corpus_error(
                "sidecar_pair_alignment",
                "baseline and candidate recording identities differ",
            ));
        }
        if let (Some(baseline), Some(candidate)) = (baseline[index].der, candidate[index].der) {
            der_deltas.push(candidate - baseline);
        }
        if let (Some(baseline), Some(candidate)) = (baseline[index].jer, candidate[index].jer) {
            jer_deltas.push(candidate - baseline);
        }
    }
    let (mean_der_delta, der_delta_ci95_lower, der_delta_ci95_upper) =
        bootstrap_mean_interval(&der_deltas, &seed, 0, is_cancelled)?;
    let (mean_jer_delta, jer_delta_ci95_lower, jer_delta_ci95_upper) =
        bootstrap_mean_interval(&jer_deltas, &seed, 1, is_cancelled)?;
    let paired_der_recording_count = u64::try_from(der_deltas.len()).map_err(|_| {
        public_corpus_error(
            "sidecar_pair_alignment",
            "paired DER recording count exceeds u64",
        )
    })?;
    let paired_jer_recording_count = u64::try_from(jer_deltas.len()).map_err(|_| {
        public_corpus_error(
            "sidecar_pair_alignment",
            "paired JER recording count exceeds u64",
        )
    })?;
    Ok(PublicCorpusSidecarPairedUncertainty {
        paired_der_recording_count,
        paired_jer_recording_count,
        bootstrap_replicates: PUBLIC_SIDECAR_BOOTSTRAP_REPLICATES,
        bootstrap_seed_sha256,
        mean_der_delta,
        der_delta_ci95_lower,
        der_delta_ci95_upper,
        mean_jer_delta,
        jer_delta_ci95_lower,
        jer_delta_ci95_upper,
    })
}

fn public_sidecar_gate_policy() -> PublicCorpusSidecarGatePolicy {
    PublicCorpusSidecarGatePolicy {
        minimum_relative_development_micro_der_improvement: canonical_evidence_number(
            PUBLIC_CORPUS_SIDECAR_MIN_DEVELOPMENT_DER_IMPROVEMENT,
        ),
        maximum_macro_jer_regression: canonical_evidence_number(
            PUBLIC_CORPUS_SIDECAR_MAX_MACRO_JER_REGRESSION,
        ),
        minimum_comparable_frame_coverage: canonical_evidence_number(
            PUBLIC_CORPUS_SIDECAR_MIN_COMPARABLE_FRAME_COVERAGE,
        ),
        minimum_pair_roc_auc: canonical_evidence_number(PUBLIC_CORPUS_SIDECAR_MIN_PAIR_ROC_AUC),
        maximum_pair_brier_score: canonical_evidence_number(PUBLIC_CORPUS_SIDECAR_MAX_PAIR_BRIER),
        maximum_pair_expected_calibration_error: canonical_evidence_number(
            PUBLIC_CORPUS_SIDECAR_MAX_PAIR_ECE,
        ),
        maximum_same_speaker_auxiliary_dominance_rate: canonical_evidence_number(
            PUBLIC_CORPUS_SIDECAR_MAX_SAME_SPEAKER_AUXILIARY_DOMINANCE_RATE,
        ),
        minimum_same_speaker_auxiliary_dominance_opportunities:
            PUBLIC_CORPUS_SIDECAR_MIN_AUXILIARY_DOMINANCE_OPPORTUNITIES,
        minimum_pair_score_coverage: canonical_evidence_number(
            PUBLIC_CORPUS_SIDECAR_MIN_PAIR_SCORE_COVERAGE,
        ),
        minimum_conditional_pairs_per_class: PUBLIC_CORPUS_SIDECAR_MIN_PAIRS_PER_CLASS,
        minimum_conditional_pair_recording_count: PUBLIC_CORPUS_SIDECAR_MIN_PAIR_RECORDINGS,
        minimum_paired_recording_count: PUBLIC_SIDECAR_MINIMUM_PAIRED_RECORDINGS,
        maximum_relative_rtf_regression: canonical_evidence_number(
            PUBLIC_CORPUS_SIDECAR_MAX_RELATIVE_RTF_REGRESSION,
        ),
        maximum_relative_rss_regression: canonical_evidence_number(
            PUBLIC_CORPUS_SIDECAR_MAX_RELATIVE_RSS_REGRESSION,
        ),
        require_boundary_f1_non_regression: true,
        require_speaker_count_non_regression: true,
        require_held_out_micro_der_non_regression: true,
        require_nonpositive_paired_der_upper_bound: true,
    }
}

fn public_sidecar_selection_policy_sha256(
    gate_policy: &PublicCorpusSidecarGatePolicy,
) -> FwResult<String> {
    canonical_sha256(&PublicCorpusSidecarSelectionFingerprint {
        policy_id: PUBLIC_CORPUS_SIDECAR_SELECTION_POLICY_VERSION,
        accuracy_ranking: "accuracy-eligible-minimum-micro-der-then-lane-v2",
        operational_gate_behavior: "unique-accuracy-winner-must-pass-full-gate-no-runner-up-v2",
        lane_order: &PublicCorpusSidecarLane::ALL,
        gate_policy,
    })
}

fn public_sidecar_protocol(
    maximum_recording_duration_ms: Option<u64>,
    diarization_request: &DiarizationRequest,
    diarization_request_sha256: String,
) -> FwResult<PublicCorpusSidecarStudyProtocol> {
    if maximum_recording_duration_ms == Some(0) {
        return Err(public_corpus_error(
            "sidecar_protocol",
            "maximum recording duration must be positive when supplied",
        ));
    }
    let gate_policy = public_sidecar_gate_policy();
    Ok(PublicCorpusSidecarStudyProtocol {
        oracle_vad: true,
        oracle_speaker_count: false,
        maximum_recording_duration_ms,
        prefix_selection: "deterministic-prefix-v1".to_owned(),
        rss_observation: "linux-vmhwm-otherwise-sampled-process-rss-v1".to_owned(),
        diarization_request: diarization_request.clone(),
        diarization_request_sha256,
        feature_ablation: AcousticFeatureAblation::FullV2,
        feature_schema_sha256: acoustic_feature_schema_sha256(
            AcousticFeatureAblation::FullV2.schema_version(),
        ),
        detector_mode: AcousticChangeDetectorMode::CalibratedPosterior,
        clustering_mode: AcousticClusteringMode::ProbabilisticV1,
        change_calibration_id: ACOUSTIC_CHANGE_CALIBRATION_VERSION.to_owned(),
        change_calibration_fit_id: ACOUSTIC_CHANGE_CALIBRATION_FIT_VERSION.to_owned(),
        change_calibration_sha256: acoustic_change_calibration_sha256(),
        change_decision_probability: canonical_evidence_number(f64::from(
            acoustic_change_calibration().decision_probability,
        )),
        speaker_pair_calibration_id: ACOUSTIC_CLUSTERING_PROBABILISTIC_VERSION.to_owned(),
        speaker_pair_calibration_sha256: acoustic_speaker_pair_calibration_sha256(),
        sidecar_schema_id: ACOUSTIC_SIDECAR_STUDY_SCHEMA_VERSION.to_owned(),
        fusion_id: ACOUSTIC_SIDECAR_FUSION_VERSION.to_owned(),
        calibration_fit_id: PUBLIC_CORPUS_SIDECAR_CALIBRATION_FIT_VERSION.to_owned(),
        pair_calibration_fit_id: PUBLIC_CORPUS_SIDECAR_PAIR_CALIBRATION_FIT_VERSION.to_owned(),
        pair_probability_target_id: PUBLIC_CORPUS_SIDECAR_PAIR_PROBABILITY_TARGET_VERSION
            .to_owned(),
        pair_population_id: PUBLIC_CORPUS_SIDECAR_PAIR_POPULATION_VERSION.to_owned(),
        pair_selection_key_id: PUBLIC_CORPUS_SIDECAR_PAIR_SELECTION_KEY_VERSION.to_owned(),
        pair_selection_digest_id: PUBLIC_CORPUS_SIDECAR_PAIR_SELECTION_DIGEST_VERSION.to_owned(),
        pair_scorer_id: PUBLIC_CORPUS_SIDECAR_PAIR_SCORER_VERSION.to_owned(),
        uncertainty_id: PUBLIC_CORPUS_SIDECAR_UNCERTAINTY_VERSION.to_owned(),
        uncertainty_seed_policy_id: PUBLIC_SIDECAR_BOOTSTRAP_SEED_POLICY.to_owned(),
        uncertainty_sampler_id: PUBLIC_SIDECAR_BOOTSTRAP_SAMPLER.to_owned(),
        selection_policy_id: PUBLIC_CORPUS_SIDECAR_SELECTION_POLICY_VERSION.to_owned(),
        selection_policy_sha256: public_sidecar_selection_policy_sha256(&gate_policy)?,
        boundary_collar_ms: PUBLIC_SIDECAR_BOUNDARY_COLLAR_MS,
        reliability_bins: PUBLIC_SIDECAR_RELIABILITY_BINS,
        pair_score_bins: PUBLIC_SIDECAR_PAIR_SCORE_BINS,
        pair_lags_frames: PUBLIC_SIDECAR_PAIR_LAGS_FRAMES,
        maximum_pairs_per_recording: PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING,
        maximum_retained_pair_sample_capacity: PUBLIC_SIDECAR_MAX_RETAINED_PAIR_SAMPLE_CAPACITY,
        paired_bootstrap_replicates: PUBLIC_SIDECAR_BOOTSTRAP_REPLICATES,
        maximum_retained_signal_capacity: PUBLIC_SIDECAR_MAX_RETAINED_SIGNAL_CAPACITY as usize,
        maximum_reported_payload_bytes: PUBLIC_SIDECAR_MAX_REPORTED_PAYLOAD_BYTES,
        lane_order: PublicCorpusSidecarLane::ALL.to_vec(),
        gate_policy,
    })
}

fn unavailable_public_sidecar_split(
    split: EvaluationSplit,
) -> FwResult<PublicCorpusSidecarStudySplit> {
    let probabilities = SidecarProbabilityAccumulator::new();
    let finished = probabilities.finish_reliability();
    Ok(PublicCorpusSidecarStudySplit {
        split,
        pipeline: None,
        fusion_executed: false,
        boundary: PublicCorpusSidecarBoundaryMetrics {
            reference_count: 0,
            hypothesis_count: 0,
            matched_count: 0,
            precision: None,
            recall: None,
            f1: None,
            mean_absolute_error_sec: None,
            probability_observation_count: 0,
            probability_positive_count: 0,
            brier_score: finished.brier_score,
            expected_calibration_error: finished.expected_calibration_error,
            reliability: finished.reliability,
        },
        conditional_pairs: SidecarPairAccumulator::new().finish()?,
        coverage: PublicCorpusSidecarCoverage {
            fusion_requested: false,
            evaluated_recording_count: 0,
            fusion_requested_recording_count: 0,
            fusion_executed_recording_count: 0,
            submitted_frame_count: 0,
            comparable_frame_count: 0,
            calibrated_signal_count: 0,
            consumed_probability_count: 0,
            changed_boundary_probability_count: 0,
            comparable_frame_coverage: None,
            component_comparison_count: 0,
            owner_available_frame_counts: [0; 3],
            channel_dominance: SidecarAuxiliaryDominanceAccumulator::default().finish(),
            mixed_auxiliary_dominance: SidecarAuxiliaryDominanceAccumulator::default().finish(),
            eligible_pair_count: 0,
            retained_pair_sample_count: 0,
            retained_same_speaker_pair_count: 0,
            retained_different_speaker_pair_count: 0,
            pair_selection_sha256: None,
            pair_score_coverage: None,
            same_speaker_pair_score_coverage: None,
            different_speaker_pair_score_coverage: None,
            pair_scored_recording_count: 0,
            same_speaker_pair_recording_count: 0,
            different_speaker_pair_recording_count: 0,
            maximum_retained_pair_sample_count: 0,
            retained_pair_sample_capacity: 0,
            maximum_retained_signal_count: 0,
            retained_signal_capacity: 0,
        },
        operations: PublicCorpusSidecarOperations {
            frame_wavelet_filter_tap_terms: 0,
            trajectory_wavelet_filter_tap_terms: 0,
            trajectory_validity_sample_visits: 0,
            scattering_filter_sample_terms: 0,
            scattering_validity_sample_visits: 0,
            modulation_projection_sample_frequency_visits: 0,
            peak_scratch_buffer_payload_bytes: 0,
            peak_retained_state_bytes_on_target: 0,
            cached_twiddle_payload_bytes: 0,
        },
        performance: PublicCorpusSidecarPerformance {
            audio_duration_sec: 0.0,
            wall_time_sec: 0.0,
            real_time_factor: None,
            sampled_peak_rss_bytes: 0,
        },
        paired_uncertainty: None,
    })
}

fn public_sidecar_promotion_gate(
    stage: PublicCorpusEvaluationStage,
    policy: &PublicCorpusSidecarGatePolicy,
    baseline: &PublicCorpusSidecarStudySplit,
    candidate_lane: PublicCorpusSidecarLane,
    candidate: &PublicCorpusSidecarStudySplit,
) -> PublicCorpusSidecarPromotionGate {
    let baseline_pipeline = baseline.pipeline.as_ref();
    let candidate_pipeline = candidate.pipeline.as_ref();
    let relative_micro_der_improvement = candidate_pipeline
        .and_then(|candidate| candidate.micro_der)
        .zip(baseline_pipeline.and_then(|baseline| baseline.micro_der))
        .and_then(|(candidate, baseline)| {
            if baseline > 0.0 {
                Some(canonical_evidence_number((baseline - candidate) / baseline))
            } else if baseline == 0.0 && candidate == 0.0 {
                Some(0.0)
            } else {
                None
            }
        });
    let macro_jer_delta = candidate_pipeline
        .and_then(|candidate| candidate.macro_jer)
        .zip(baseline_pipeline.and_then(|baseline| baseline.macro_jer))
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let boundary_f1_delta = candidate_pipeline
        .and_then(|candidate| candidate.change_f1)
        .zip(baseline_pipeline.and_then(|baseline| baseline.change_f1))
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let paired_der_ci95_upper = candidate
        .paired_uncertainty
        .as_ref()
        .and_then(|uncertainty| uncertainty.der_delta_ci95_upper);
    let exact_speaker_count_rate_delta = candidate_pipeline
        .and_then(|candidate| candidate.exact_speaker_count_rate)
        .zip(baseline_pipeline.and_then(|baseline| baseline.exact_speaker_count_rate))
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let mean_absolute_speaker_count_error_delta = candidate_pipeline
        .and_then(|candidate| candidate.mean_absolute_speaker_count_error)
        .zip(baseline_pipeline.and_then(|baseline| baseline.mean_absolute_speaker_count_error))
        .map(|(candidate, baseline)| canonical_evidence_number(candidate - baseline));
    let dominant_collapse_count_delta =
        candidate_pipeline
            .zip(baseline_pipeline)
            .and_then(|(candidate, baseline)| {
                let candidate = i64::try_from(candidate.dominant_collapse_recording_count).ok()?;
                let baseline = i64::try_from(baseline.dominant_collapse_recording_count).ok()?;
                candidate.checked_sub(baseline)
            });
    let relative_rtf_regression = candidate
        .performance
        .real_time_factor
        .zip(baseline.performance.real_time_factor)
        .and_then(|(candidate, baseline)| {
            (baseline > 0.0).then(|| canonical_evidence_number((candidate - baseline) / baseline))
        });
    let relative_rss_regression = (baseline.performance.sampled_peak_rss_bytes > 0).then(|| {
        canonical_evidence_number(
            (candidate.performance.sampled_peak_rss_bytes as f64
                - baseline.performance.sampled_peak_rss_bytes as f64)
                / baseline.performance.sampled_peak_rss_bytes as f64,
        )
    });
    let mut failures = Vec::new();
    if !candidate.fusion_executed || candidate.coverage.consumed_probability_count == 0 {
        failures.push(PublicCorpusSidecarGateFailure::FusionNotExecuted);
    }
    if candidate_pipeline.is_none()
        || relative_micro_der_improvement.is_none()
        || macro_jer_delta.is_none()
    {
        failures.push(PublicCorpusSidecarGateFailure::MissingAccuracy);
    }
    let der_passed = match stage {
        PublicCorpusEvaluationStage::Development => {
            relative_micro_der_improvement.is_some_and(|improvement| {
                improvement >= policy.minimum_relative_development_micro_der_improvement
            })
        }
        PublicCorpusEvaluationStage::Certification => {
            relative_micro_der_improvement.is_some_and(|improvement| {
                !policy.require_held_out_micro_der_non_regression || improvement >= 0.0
            })
        }
    };
    if !der_passed {
        failures.push(PublicCorpusSidecarGateFailure::InsufficientDerImprovement);
    }
    if !macro_jer_delta.is_some_and(|delta| delta <= policy.maximum_macro_jer_regression) {
        failures.push(PublicCorpusSidecarGateFailure::MacroJerRegression);
    }
    if policy.require_boundary_f1_non_regression
        && !boundary_f1_delta.is_some_and(|delta| delta >= 0.0)
    {
        failures.push(PublicCorpusSidecarGateFailure::BoundaryF1Regression);
    }
    if !candidate
        .coverage
        .comparable_frame_coverage
        .is_some_and(|coverage| coverage >= policy.minimum_comparable_frame_coverage)
    {
        failures.push(PublicCorpusSidecarGateFailure::InsufficientComparableCoverage);
    }
    if !candidate
        .coverage
        .pair_score_coverage
        .is_some_and(|coverage| coverage >= policy.minimum_pair_score_coverage)
        || !candidate
            .coverage
            .same_speaker_pair_score_coverage
            .is_some_and(|coverage| coverage >= policy.minimum_pair_score_coverage)
        || !candidate
            .coverage
            .different_speaker_pair_score_coverage
            .is_some_and(|coverage| coverage >= policy.minimum_pair_score_coverage)
    {
        failures.push(PublicCorpusSidecarGateFailure::InsufficientPairCoverage);
    }
    if candidate.conditional_pairs.same_speaker_count == 0
        || candidate.conditional_pairs.different_speaker_count == 0
    {
        failures.push(PublicCorpusSidecarGateFailure::MissingConditionalPairs);
    }
    if candidate.conditional_pairs.same_speaker_count < policy.minimum_conditional_pairs_per_class
        || candidate.conditional_pairs.different_speaker_count
            < policy.minimum_conditional_pairs_per_class
        || candidate.coverage.pair_scored_recording_count
            < policy.minimum_conditional_pair_recording_count
        || candidate.coverage.same_speaker_pair_recording_count
            < policy.minimum_conditional_pair_recording_count
        || candidate.coverage.different_speaker_pair_recording_count
            < policy.minimum_conditional_pair_recording_count
    {
        failures.push(PublicCorpusSidecarGateFailure::InsufficientPairSupport);
    }
    if !candidate
        .conditional_pairs
        .roc_auc
        .is_some_and(|auc| auc >= policy.minimum_pair_roc_auc)
    {
        failures.push(PublicCorpusSidecarGateFailure::PairDiscrimination);
    }
    if !candidate
        .conditional_pairs
        .brier_score
        .is_some_and(|brier| brier <= policy.maximum_pair_brier_score)
    {
        failures.push(PublicCorpusSidecarGateFailure::PairBrier);
    }
    if !candidate
        .conditional_pairs
        .expected_calibration_error
        .is_some_and(|ece| ece <= policy.maximum_pair_expected_calibration_error)
    {
        failures.push(PublicCorpusSidecarGateFailure::PairCalibration);
    }
    let auxiliary_dominance_failed = candidate_lane
        .auxiliary_dominance_expectations()
        .into_iter()
        .zip([
            &candidate.coverage.channel_dominance,
            &candidate.coverage.mixed_auxiliary_dominance,
        ])
        .any(|(expected, diagnostics)| {
            expected
                && (diagnostics.same_speaker_opportunity_count
                    < policy.minimum_same_speaker_auxiliary_dominance_opportunities
                    || !diagnostics.same_speaker_dominance_rate.is_some_and(|rate| {
                        rate <= policy.maximum_same_speaker_auxiliary_dominance_rate
                    }))
        });
    if auxiliary_dominance_failed {
        failures.push(PublicCorpusSidecarGateFailure::AuxiliaryConfound);
    }
    if policy.require_speaker_count_non_regression
        && (!exact_speaker_count_rate_delta.is_some_and(|delta| delta >= 0.0)
            || !mean_absolute_speaker_count_error_delta.is_some_and(|delta| delta <= 0.0)
            || !dominant_collapse_count_delta.is_some_and(|delta| delta <= 0))
    {
        failures.push(PublicCorpusSidecarGateFailure::SpeakerCountRegression);
    }
    if !relative_rtf_regression
        .is_some_and(|regression| regression <= policy.maximum_relative_rtf_regression)
        || !relative_rss_regression
            .is_some_and(|regression| regression <= policy.maximum_relative_rss_regression)
    {
        failures.push(PublicCorpusSidecarGateFailure::PerformanceRegression);
    }
    if policy.require_nonpositive_paired_der_upper_bound
        && (candidate
            .paired_uncertainty
            .as_ref()
            .is_none_or(|uncertainty| {
                uncertainty.paired_der_recording_count < policy.minimum_paired_recording_count
            })
            || !paired_der_ci95_upper.is_some_and(|upper| upper <= 0.0))
    {
        failures.push(PublicCorpusSidecarGateFailure::PairedDerUncertainty);
    }
    failures.sort();
    failures.dedup();
    PublicCorpusSidecarPromotionGate {
        split: candidate.split,
        candidate: candidate_lane,
        baseline: PublicCorpusSidecarLane::FullV2Baseline,
        relative_micro_der_improvement,
        macro_jer_delta,
        boundary_f1_delta,
        comparable_frame_coverage: candidate.coverage.comparable_frame_coverage,
        pair_score_coverage: candidate.coverage.pair_score_coverage,
        same_speaker_pair_score_coverage: candidate.coverage.same_speaker_pair_score_coverage,
        different_speaker_pair_score_coverage: candidate
            .coverage
            .different_speaker_pair_score_coverage,
        pair_scored_recording_count: candidate.coverage.pair_scored_recording_count,
        pair_roc_auc: candidate.conditional_pairs.roc_auc,
        pair_brier_score: candidate.conditional_pairs.brier_score,
        pair_expected_calibration_error: candidate.conditional_pairs.expected_calibration_error,
        channel_same_speaker_dominance_rate: candidate
            .coverage
            .channel_dominance
            .same_speaker_dominance_rate,
        mixed_auxiliary_same_speaker_dominance_rate: candidate
            .coverage
            .mixed_auxiliary_dominance
            .same_speaker_dominance_rate,
        exact_speaker_count_rate_delta,
        mean_absolute_speaker_count_error_delta,
        dominant_collapse_count_delta,
        relative_rtf_regression,
        relative_rss_regression,
        paired_der_ci95_upper,
        passed: failures.is_empty(),
        failures,
    }
}

fn sidecar_gate_is_accuracy_eligible(gate: &PublicCorpusSidecarPromotionGate) -> bool {
    gate.failures.iter().all(|failure| {
        matches!(
            failure,
            PublicCorpusSidecarGateFailure::PerformanceRegression
                | PublicCorpusSidecarGateFailure::NotSelectedByRanking
        )
    })
}

/// Return the index in the full lane vector (including the baseline at index
/// zero) of the unique accuracy-ranked candidate.
fn sidecar_accuracy_ranked_variant_index(
    variants: &[PublicCorpusSidecarStudyVariant],
) -> Option<usize> {
    variants
        .iter()
        .enumerate()
        .skip(1)
        .filter(|(_, variant)| {
            variant
                .gate
                .as_ref()
                .is_some_and(sidecar_gate_is_accuracy_eligible)
        })
        .min_by(|(_, left), (_, right)| {
            let left_der = left
                .splits
                .first()
                .and_then(|split| split.pipeline.as_ref())
                .and_then(|pipeline| pipeline.micro_der)
                .unwrap_or(f64::INFINITY);
            let right_der = right
                .splits
                .first()
                .and_then(|split| split.pipeline.as_ref())
                .and_then(|pipeline| pipeline.micro_der)
                .unwrap_or(f64::INFINITY);
            left_der
                .total_cmp(&right_der)
                .then_with(|| left.lane.cmp(&right.lane))
        })
        .map(|(index, _)| index)
}

fn apply_public_sidecar_development_selection(
    variants: &mut [PublicCorpusSidecarStudyVariant],
) -> Option<PublicCorpusSidecarLane> {
    let selected_index = sidecar_accuracy_ranked_variant_index(variants);
    for (index, variant) in variants.iter_mut().enumerate().skip(1) {
        if Some(index) == selected_index && variant.gate.as_ref().is_some_and(|gate| gate.passed) {
            variant.disposition = PublicCorpusSidecarDisposition::AdvanceToCertification;
        } else {
            variant.disposition = PublicCorpusSidecarDisposition::Rejected;
            if Some(index) != selected_index
                && let Some(gate) = variant.gate.as_mut()
                && sidecar_gate_is_accuracy_eligible(gate)
            {
                gate.passed = false;
                if !gate
                    .failures
                    .contains(&PublicCorpusSidecarGateFailure::NotSelectedByRanking)
                {
                    gate.failures
                        .push(PublicCorpusSidecarGateFailure::NotSelectedByRanking);
                    gate.failures.sort();
                }
            }
        }
    }
    selected_index.and_then(|index| {
        (variants[index].disposition == PublicCorpusSidecarDisposition::AdvanceToCertification)
            .then_some(variants[index].lane)
    })
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

fn count_merge_frontier_observation(
    steps: &[AcousticCountMergeStepEvidence],
    reference_count: usize,
) -> FwResult<Option<(Option<f64>, Option<f64>)>> {
    if steps.is_empty() {
        return Ok(None);
    }
    if steps.iter().any(|step| {
        !step.same_speaker_probability.is_finite()
            || !(0.0..=1.0).contains(&step.same_speaker_probability)
    }) || !steps.windows(2).all(|window| {
        window[0].remaining_clusters.checked_sub(1) == Some(window[1].remaining_clusters)
    }) {
        return Err(public_corpus_error(
            "count_merge_frontier",
            "probabilistic count merge steps must be finite probabilities with consecutive descending cluster counts",
        ));
    }
    let probability_at = |count| {
        steps
            .iter()
            .find(|step| step.remaining_clusters == count)
            .map(|step| f64::from(step.same_speaker_probability))
    };
    Ok(Some((
        probability_at(reference_count),
        reference_count.checked_sub(1).and_then(probability_at),
    )))
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
    speaker_count_posterior_map_confusion: BTreeMap<(u32, u32), u64>,
    count_merge_frontier_recording_count: u64,
    count_merge_to_reference_probability_sum: f64,
    count_merge_to_reference_probability_count: u64,
    count_merge_below_reference_probability_sum: f64,
    count_merge_below_reference_probability_count: u64,
    count_merge_frontier_margin_sum: f64,
    count_merge_frontier_margin_count: u64,
    count_merge_frontier_correctly_ordered_count: u64,
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
        if let Some(map_count) = score.speaker_count_posterior.map_count {
            let map_confusion_count = self
                .speaker_count_posterior_map_confusion
                .entry((reference_speakers, map_count))
                .or_default();
            *map_confusion_count = map_confusion_count.saturating_add(1);
        }
        let reference_count = usize::try_from(reference_speakers).map_err(|_| {
            public_corpus_error(
                "count_merge_frontier",
                "reference speaker count exceeds the native count domain",
            )
        })?;
        if let Some((to_reference, below_reference)) =
            count_merge_frontier_observation(&clustering.count_merge_steps, reference_count)?
        {
            self.count_merge_frontier_recording_count = self
                .count_merge_frontier_recording_count
                .checked_add(1)
                .ok_or_else(|| {
                    public_corpus_error(
                        "ablation_aggregate_overflow",
                        "count merge-frontier recording count overflowed",
                    )
                })?;
            if let Some(probability) = to_reference {
                self.count_merge_to_reference_probability_sum += probability;
                self.count_merge_to_reference_probability_count = self
                    .count_merge_to_reference_probability_count
                    .checked_add(1)
                    .ok_or_else(|| {
                        public_corpus_error(
                            "ablation_aggregate_overflow",
                            "to-reference count merge observation count overflowed",
                        )
                    })?;
            }
            if let Some(probability) = below_reference {
                self.count_merge_below_reference_probability_sum += probability;
                self.count_merge_below_reference_probability_count = self
                    .count_merge_below_reference_probability_count
                    .checked_add(1)
                    .ok_or_else(|| {
                        public_corpus_error(
                            "ablation_aggregate_overflow",
                            "below-reference count merge observation count overflowed",
                        )
                    })?;
            }
            if let (Some(to_reference), Some(below_reference)) = (to_reference, below_reference) {
                self.count_merge_frontier_margin_sum += to_reference - below_reference;
                self.count_merge_frontier_margin_count = self
                    .count_merge_frontier_margin_count
                    .checked_add(1)
                    .ok_or_else(|| {
                        public_corpus_error(
                            "ablation_aggregate_overflow",
                            "paired count merge-frontier observation count overflowed",
                        )
                    })?;
                self.count_merge_frontier_correctly_ordered_count = self
                    .count_merge_frontier_correctly_ordered_count
                    .checked_add(u64::from(to_reference > below_reference))
                    .ok_or_else(|| {
                        public_corpus_error(
                            "ablation_aggregate_overflow",
                            "ordered count merge-frontier count overflowed",
                        )
                    })?;
            }
        }
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
        let speaker_count_posterior_map_confusion = self
            .speaker_count_posterior_map_confusion
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
        let speaker_count_merge_frontier = PublicSpeakerCountMergeFrontier {
            recording_count: self.count_merge_frontier_recording_count,
            to_reference_count_observation_count: self.count_merge_to_reference_probability_count,
            mean_probability_to_reference_count: positive_ratio(
                self.count_merge_to_reference_probability_sum,
                self.count_merge_to_reference_probability_count as f64,
            ),
            below_reference_count_observation_count: self
                .count_merge_below_reference_probability_count,
            mean_probability_below_reference_count: positive_ratio(
                self.count_merge_below_reference_probability_sum,
                self.count_merge_below_reference_probability_count as f64,
            ),
            paired_frontier_observation_count: self.count_merge_frontier_margin_count,
            mean_probability_margin: signed_ratio(
                self.count_merge_frontier_margin_sum,
                self.count_merge_frontier_margin_count as f64,
            ),
            correctly_ordered_frontier_count: self.count_merge_frontier_correctly_ordered_count,
            correctly_ordered_frontier_rate: ratio(
                self.count_merge_frontier_correctly_ordered_count,
                self.count_merge_frontier_margin_count,
            ),
        };
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
            speaker_count_posterior_map_confusion,
            speaker_count_merge_frontier,
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

fn sidecar_f32_probability_bin_index(probability: f32, bin_count: usize) -> usize {
    ((f64::from(probability) * bin_count as f64).floor() as usize).min(bin_count - 1)
}

fn sidecar_f32_probability_bin_support(index: usize, bin_count: usize) -> Option<(f64, f64)> {
    if bin_count == 0 || index >= bin_count {
        return None;
    }
    let minimum_bits = f32::EPSILON.to_bits() as u64;
    let maximum_bits = (1.0_f32 - f32::EPSILON).to_bits() as u64;
    let end_bits = maximum_bits + 1;
    let lower_bound = |target_index: usize| {
        let mut lower = minimum_bits;
        let mut upper = end_bits;
        while lower < upper {
            let midpoint = lower + (upper - lower) / 2;
            let is_before = midpoint <= maximum_bits
                && sidecar_f32_probability_bin_index(f32::from_bits(midpoint as u32), bin_count)
                    < target_index;
            if is_before {
                lower = midpoint + 1;
            } else {
                upper = midpoint;
            }
        }
        lower
    };
    let first_bits = lower_bound(index);
    let after_last_bits = lower_bound(index + 1);
    if first_bits >= after_last_bits || first_bits > maximum_bits {
        return None;
    }
    let last_bits = after_last_bits - 1;
    Some((
        f64::from(f32::from_bits(first_bits as u32)),
        f64::from(f32::from_bits(last_bits as u32)),
    ))
}

fn maximum_bounded_second_moment(
    lower_probability: f64,
    upper_probability: f64,
    observation_count: u64,
    probability_sum: f64,
) -> f64 {
    let count = observation_count as f64;
    let width = upper_probability - lower_probability;
    if observation_count == 0 || width == 0.0 {
        return count * lower_probability.powi(2);
    }
    let normalized_excess =
        ((probability_sum - count * lower_probability) / width).clamp(0.0, count);
    let full_upper_count = normalized_excess.floor();
    let fractional_upper = normalized_excess - full_upper_count;
    count * lower_probability.powi(2)
        + 2.0 * lower_probability * width * normalized_excess
        + width.powi(2) * (full_upper_count + fractional_upper.powi(2))
}

#[allow(clippy::too_many_arguments)]
fn sidecar_probability_moments_are_feasible(
    lower_probability: f64,
    upper_probability: f64,
    observation_count: u64,
    positive_count: u64,
    probability_sum: f64,
    positive_probability_sum: f64,
    squared_probability_sum: f64,
    positive_squared_probability_sum: f64,
    squared_error_sum: f64,
) -> bool {
    if !lower_probability.is_finite()
        || !upper_probability.is_finite()
        || !(0.0..=1.0).contains(&lower_probability)
        || !(0.0..=1.0).contains(&upper_probability)
        || lower_probability > upper_probability
        || positive_count > observation_count
    {
        return false;
    }
    if observation_count == 0 {
        return positive_count == 0
            && probability_sum == 0.0
            && positive_probability_sum == 0.0
            && squared_probability_sum == 0.0
            && positive_squared_probability_sum == 0.0
            && squared_error_sum == 0.0;
    }
    let count = observation_count as f64;
    let positives = positive_count as f64;
    let negatives = count - positives;
    let negative_probability_sum = probability_sum - positive_probability_sum;
    let negative_squared_probability_sum =
        squared_probability_sum - positive_squared_probability_sum;
    // Direct first-moment sums and their bounds share the canonical
    // 12-decimal evidence lattice, whose rounding is monotone. Compare those
    // canonical values directly so the aggregate support check does not grow
    // looser merely because a bin contains more observations.
    let canonical_lower_sum = canonical_evidence_number(lower_probability * count);
    let canonical_upper_sum = canonical_evidence_number(upper_probability * count);
    let canonical_positive_lower_sum = canonical_evidence_number(lower_probability * positives);
    let canonical_positive_upper_sum = canonical_evidence_number(upper_probability * positives);
    let canonical_negative_probability_sum = canonical_evidence_number(negative_probability_sum);
    let canonical_negative_lower_sum = canonical_evidence_number(lower_probability * negatives);
    let canonical_negative_upper_sum = canonical_evidence_number(upper_probability * negatives);
    // The negative-class sum is derived by subtracting two independently
    // canonicalized aggregates. Its absolute binary subtraction error grows
    // with the magnitudes of those totals even when the derived result is
    // small, so budget against the operands rather than an adjacent score gap.
    let subtraction_scale = probability_sum
        .abs()
        .max(positive_probability_sum.abs())
        .max(canonical_negative_probability_sum.abs())
        .max(1.0);
    let negative_support_tolerance = 2e-12 + 8.0 * f64::EPSILON * subtraction_scale;
    let tolerance = 1e-9 * count.max(1.0);
    let maximum_squared_probability_sum = maximum_bounded_second_moment(
        lower_probability,
        upper_probability,
        observation_count,
        probability_sum,
    );
    let maximum_positive_squared_probability_sum = maximum_bounded_second_moment(
        lower_probability,
        upper_probability,
        positive_count,
        positive_probability_sum,
    );
    let negative_count = observation_count - positive_count;
    let maximum_negative_squared_probability_sum = maximum_bounded_second_moment(
        lower_probability,
        upper_probability,
        negative_count,
        negative_probability_sum,
    );
    let moment_scale = squared_probability_sum
        .abs()
        .max(positive_squared_probability_sum.abs())
        .max(negative_squared_probability_sum.abs())
        .max(squared_error_sum.abs())
        .max(maximum_squared_probability_sum.abs())
        .max(1.0);
    // These sufficient statistics can establish aggregate feasibility, not
    // the membership of every omitted raw score. Keep the allowances tied
    // only to canonicalization and floating arithmetic; an adjacency-derived
    // cap both overclaimed that proof boundary and rejected valid large bins.
    let moment_tolerance = 2e-12 + 8.0 * f64::EPSILON * moment_scale;
    // Negative-class moments are differences of two independently rounded
    // fixed-12 aggregates. Their Jensen/secant checks can accumulate one
    // decimal half-unit from each input plus binary arithmetic error. Four
    // decimal units and 32 ulps cover that explicit error budget without
    // scaling the allowance by observation count.
    let derived_moment_tolerance = 4e-12 + 32.0 * f64::EPSILON * moment_scale;
    probability_sum.is_finite()
        && positive_probability_sum.is_finite()
        && squared_probability_sum.is_finite()
        && positive_squared_probability_sum.is_finite()
        && squared_error_sum.is_finite()
        && canonical_evidence_number(probability_sum) == probability_sum
        && canonical_evidence_number(positive_probability_sum) == positive_probability_sum
        && canonical_evidence_number(squared_probability_sum) == squared_probability_sum
        && canonical_evidence_number(positive_squared_probability_sum)
            == positive_squared_probability_sum
        && canonical_evidence_number(squared_error_sum) == squared_error_sum
        && (positive_count > 0
            || (positive_probability_sum == 0.0 && positive_squared_probability_sum == 0.0))
        && (positive_count < observation_count
            || (negative_probability_sum == 0.0 && negative_squared_probability_sum == 0.0))
        && probability_sum >= canonical_lower_sum
        && probability_sum <= canonical_upper_sum
        && positive_probability_sum >= canonical_positive_lower_sum
        && positive_probability_sum <= canonical_positive_upper_sum
        && canonical_negative_probability_sum + negative_support_tolerance
            >= canonical_negative_lower_sum
        && canonical_negative_probability_sum - negative_support_tolerance
            <= canonical_negative_upper_sum
        && squared_probability_sum + moment_tolerance >= lower_probability.powi(2) * count
        && squared_probability_sum - moment_tolerance <= upper_probability.powi(2) * count
        && positive_squared_probability_sum + moment_tolerance
            >= lower_probability.powi(2) * positives
        && positive_squared_probability_sum - moment_tolerance
            <= upper_probability.powi(2) * positives
        && negative_squared_probability_sum + derived_moment_tolerance
            >= lower_probability.powi(2) * negatives
        && negative_squared_probability_sum - derived_moment_tolerance
            <= upper_probability.powi(2) * negatives
        && (count == 0.0
            || squared_probability_sum + moment_tolerance >= probability_sum.powi(2) / count)
        && (positives == 0.0
            || positive_squared_probability_sum + moment_tolerance
                >= positive_probability_sum.powi(2) / positives)
        && (negatives == 0.0
            || negative_squared_probability_sum + derived_moment_tolerance
                >= negative_probability_sum.powi(2) / negatives)
        && squared_probability_sum - moment_tolerance
            <= (lower_probability + upper_probability) * probability_sum
                - lower_probability * upper_probability * count
        && positive_squared_probability_sum - moment_tolerance
            <= (lower_probability + upper_probability) * positive_probability_sum
                - lower_probability * upper_probability * positives
        && negative_squared_probability_sum - derived_moment_tolerance
            <= (lower_probability + upper_probability) * negative_probability_sum
                - lower_probability * upper_probability * negatives
        && squared_probability_sum - moment_tolerance <= maximum_squared_probability_sum
        && positive_squared_probability_sum - moment_tolerance
            <= maximum_positive_squared_probability_sum
        && negative_squared_probability_sum - derived_moment_tolerance
            <= maximum_negative_squared_probability_sum
        && squared_error_sum >= 0.0
        && squared_error_sum - moment_tolerance <= count
        && (squared_error_sum
            - canonical_evidence_number(
                squared_probability_sum - 2.0 * positive_probability_sum + positives,
            ))
        .abs()
            <= derived_moment_tolerance
        && negative_probability_sum >= -tolerance
        && positive_probability_sum - tolerance <= probability_sum
        && positive_squared_probability_sum - derived_moment_tolerance <= squared_probability_sum
        && negative_squared_probability_sum >= -derived_moment_tolerance
        && positive_squared_probability_sum - moment_tolerance <= positive_probability_sum
        && negative_squared_probability_sum - derived_moment_tolerance <= negative_probability_sum
        && squared_probability_sum - moment_tolerance <= probability_sum
}

struct VerifiedSidecarReliability {
    probability_sum: f64,
    positive_probability_sum: f64,
}

fn verified_sidecar_reliability(
    observation_count: u64,
    positive_count: u64,
    brier_score: Option<f64>,
    expected_calibration_error: Option<f64>,
    reliability: &[PublicCorpusSidecarReliabilityBin],
) -> Option<VerifiedSidecarReliability> {
    if positive_count > observation_count || reliability.len() != PUBLIC_SIDECAR_RELIABILITY_BINS {
        return None;
    }
    let mut retained_observations = 0_u64;
    let mut retained_positives = 0_u64;
    let mut probability_sum = 0.0;
    let mut positive_probability_sum = 0.0;
    let mut squared_error_sum = 0.0;
    let mut expected_ece = 0.0;
    for (index, bin) in reliability.iter().enumerate() {
        if bin.index != index
            || bin.lower_probability
                != canonical_evidence_number(index as f64 / PUBLIC_SIDECAR_RELIABILITY_BINS as f64)
            || bin.upper_probability
                != canonical_evidence_number(
                    (index + 1) as f64 / PUBLIC_SIDECAR_RELIABILITY_BINS as f64,
                )
            || bin.positive_count > bin.observation_count
        {
            return None;
        }
        let next_observations = retained_observations.checked_add(bin.observation_count)?;
        retained_observations = next_observations;
        let next_positives = retained_positives.checked_add(bin.positive_count)?;
        retained_positives = next_positives;
        let (support_lower, support_upper) =
            sidecar_f32_probability_bin_support(index, PUBLIC_SIDECAR_RELIABILITY_BINS)?;
        if !sidecar_probability_moments_are_feasible(
            support_lower,
            support_upper,
            bin.observation_count,
            bin.positive_count,
            bin.probability_sum,
            bin.positive_probability_sum,
            bin.squared_probability_sum,
            bin.positive_squared_probability_sum,
            bin.squared_error_sum,
        ) {
            return None;
        }
        probability_sum += bin.probability_sum;
        positive_probability_sum += bin.positive_probability_sum;
        squared_error_sum += bin.squared_error_sum;
        match (bin.mean_probability, bin.empirical_frequency) {
            (None, None)
                if bin.observation_count == 0
                    && bin.probability_sum == 0.0
                    && bin.positive_probability_sum == 0.0
                    && bin.squared_probability_sum == 0.0
                    && bin.positive_squared_probability_sum == 0.0
                    && bin.squared_error_sum == 0.0 => {}
            (Some(mean), Some(empirical))
                if bin.observation_count > 0
                    && mean.is_finite()
                    && empirical.is_finite()
                    && (0.0..=1.0).contains(&mean)
                    && Some(mean)
                        == positive_ratio(bin.probability_sum, bin.observation_count as f64)
                    && mean >= bin.lower_probability
                    && if index + 1 == PUBLIC_SIDECAR_RELIABILITY_BINS {
                        mean <= bin.upper_probability
                    } else {
                        mean < bin.upper_probability
                    }
                    && (0.0..=1.0).contains(&empirical)
                    && empirical
                        == ratio(bin.positive_count, bin.observation_count).unwrap_or(0.0) =>
            {
                expected_ece += bin.observation_count as f64 / observation_count.max(1) as f64
                    * (mean - empirical).abs();
            }
            _ => return None,
        }
    }
    let bounded = |value: Option<f64>| {
        value.is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    };
    let expected_brier = positive_ratio(
        canonical_evidence_number(squared_error_sum),
        observation_count as f64,
    );
    let valid = retained_observations == observation_count
        && retained_positives == positive_count
        && bounded(brier_score)
        && bounded(expected_calibration_error)
        && if observation_count == 0 {
            brier_score.is_none() && expected_calibration_error.is_none()
        } else {
            brier_score == expected_brier
                && expected_calibration_error == Some(canonical_evidence_number(expected_ece))
        };
    valid.then_some(VerifiedSidecarReliability {
        probability_sum: canonical_evidence_number(probability_sum),
        positive_probability_sum: canonical_evidence_number(positive_probability_sum),
    })
}

fn sidecar_reliability_is_valid(
    observation_count: u64,
    positive_count: u64,
    brier_score: Option<f64>,
    expected_calibration_error: Option<f64>,
    reliability: &[PublicCorpusSidecarReliabilityBin],
) -> bool {
    verified_sidecar_reliability(
        observation_count,
        positive_count,
        brier_score,
        expected_calibration_error,
        reliability,
    )
    .is_some()
}

fn sidecar_boundary_metrics_are_valid(
    boundary: &PublicCorpusSidecarBoundaryMetrics,
    pipeline: Option<&PublicCorpusAblationSplit>,
) -> bool {
    let precision = ratio(boundary.matched_count, boundary.hypothesis_count);
    let recall = ratio(boundary.matched_count, boundary.reference_count);
    let f1 = precision.zip(recall).map(|(precision, recall)| {
        let denominator = precision + recall;
        if denominator > 0.0 {
            2.0 * precision * recall / denominator
        } else {
            0.0
        }
    });
    let aligned = pipeline.is_none_or(|pipeline| {
        boundary.reference_count == pipeline.change_reference_count
            && boundary.hypothesis_count == pipeline.change_hypothesis_count
            && boundary.matched_count == pipeline.change_matched_count
            && boundary.precision == pipeline.change_precision
            && boundary.recall == pipeline.change_recall
            && boundary.f1 == pipeline.change_f1
            && boundary.mean_absolute_error_sec == pipeline.change_mean_absolute_error_sec
    });
    boundary.matched_count <= boundary.reference_count
        && boundary.matched_count <= boundary.hypothesis_count
        && boundary.precision == precision
        && boundary.recall == recall
        && boundary.f1 == f1
        && boundary.mean_absolute_error_sec.is_none_or(|value| {
            value.is_finite()
                && value >= 0.0
                && value <= PUBLIC_SIDECAR_BOUNDARY_COLLAR_MS as f64 / 1_000.0
        })
        && (boundary.matched_count > 0) == boundary.mean_absolute_error_sec.is_some()
        && aligned
        && sidecar_reliability_is_valid(
            boundary.probability_observation_count,
            boundary.probability_positive_count,
            boundary.brier_score,
            boundary.expected_calibration_error,
            &boundary.reliability,
        )
}

fn sidecar_pair_metrics_are_valid(metrics: &PublicCorpusSidecarPairMetrics) -> bool {
    let Some(expected_count) = metrics
        .same_speaker_count
        .checked_add(metrics.different_speaker_count)
    else {
        return false;
    };
    let Some(reliability) = verified_sidecar_reliability(
        metrics.comparison_count,
        metrics.different_speaker_count,
        metrics.brier_score,
        metrics.expected_calibration_error,
        &metrics.reliability,
    ) else {
        return false;
    };
    if metrics.score_histogram.len() != PUBLIC_SIDECAR_PAIR_SCORE_BINS
        || !PUBLIC_SIDECAR_PAIR_SCORE_BINS.is_multiple_of(PUBLIC_SIDECAR_RELIABILITY_BINS)
    {
        return false;
    }
    let score_bins_per_reliability_bin =
        PUBLIC_SIDECAR_PAIR_SCORE_BINS / PUBLIC_SIDECAR_RELIABILITY_BINS;
    let mut grouped_score_bins =
        [SidecarReliabilityAccumulator::default(); PUBLIC_SIDECAR_RELIABILITY_BINS];
    let mut retained_same = 0_u64;
    let mut retained_different = 0_u64;
    let mut lower_same = 0_u64;
    let mut concordance = 0.0;
    for (index, bin) in metrics.score_histogram.iter().enumerate() {
        if bin.index != index
            || bin.lower_probability
                != canonical_evidence_number(index as f64 / PUBLIC_SIDECAR_PAIR_SCORE_BINS as f64)
            || bin.upper_probability
                != canonical_evidence_number(
                    (index + 1) as f64 / PUBLIC_SIDECAR_PAIR_SCORE_BINS as f64,
                )
        {
            return false;
        }
        let Some(bin_observation_count) = bin
            .same_speaker_count
            .checked_add(bin.different_speaker_count)
        else {
            return false;
        };
        let Some((support_lower, support_upper)) =
            sidecar_f32_probability_bin_support(index, PUBLIC_SIDECAR_PAIR_SCORE_BINS)
        else {
            return false;
        };
        if !sidecar_probability_moments_are_feasible(
            support_lower,
            support_upper,
            bin_observation_count,
            bin.different_speaker_count,
            bin.probability_sum,
            bin.different_speaker_probability_sum,
            bin.squared_probability_sum,
            bin.different_speaker_squared_probability_sum,
            bin.squared_error_sum,
        ) {
            return false;
        }
        let grouped = &mut grouped_score_bins[index / score_bins_per_reliability_bin];
        let Some(grouped_observation_count) =
            grouped.observation_count.checked_add(bin_observation_count)
        else {
            return false;
        };
        let Some(grouped_positive_count) = grouped
            .positive_count
            .checked_add(bin.different_speaker_count)
        else {
            return false;
        };
        grouped.observation_count = grouped_observation_count;
        grouped.positive_count = grouped_positive_count;
        add_sidecar_compensated(
            &mut grouped.probability_sum,
            &mut grouped.probability_sum_correction,
            bin.probability_sum,
        );
        add_sidecar_compensated(
            &mut grouped.positive_probability_sum,
            &mut grouped.positive_probability_sum_correction,
            bin.different_speaker_probability_sum,
        );
        add_sidecar_compensated(
            &mut grouped.squared_probability_sum,
            &mut grouped.squared_probability_sum_correction,
            bin.squared_probability_sum,
        );
        add_sidecar_compensated(
            &mut grouped.positive_squared_probability_sum,
            &mut grouped.positive_squared_probability_sum_correction,
            bin.different_speaker_squared_probability_sum,
        );
        add_sidecar_compensated(
            &mut grouped.squared_error_sum,
            &mut grouped.squared_error_sum_correction,
            bin.squared_error_sum,
        );
        let Some(next_same) = retained_same.checked_add(bin.same_speaker_count) else {
            return false;
        };
        let Some(next_different) = retained_different.checked_add(bin.different_speaker_count)
        else {
            return false;
        };
        concordance += bin.different_speaker_count as f64
            * (lower_same as f64 + 0.5 * bin.same_speaker_count as f64);
        lower_same = next_same;
        retained_same = next_same;
        retained_different = next_different;
    }
    for (grouped, reliability_bin) in grouped_score_bins.iter().zip(&metrics.reliability) {
        if grouped.observation_count != reliability_bin.observation_count
            || grouped.positive_count != reliability_bin.positive_count
            || canonical_evidence_number(sidecar_compensated_total(
                grouped.probability_sum,
                grouped.probability_sum_correction,
            )) != reliability_bin.probability_sum
            || canonical_evidence_number(sidecar_compensated_total(
                grouped.positive_probability_sum,
                grouped.positive_probability_sum_correction,
            )) != reliability_bin.positive_probability_sum
            || canonical_evidence_number(sidecar_compensated_total(
                grouped.squared_probability_sum,
                grouped.squared_probability_sum_correction,
            )) != reliability_bin.squared_probability_sum
            || canonical_evidence_number(sidecar_compensated_total(
                grouped.positive_squared_probability_sum,
                grouped.positive_squared_probability_sum_correction,
            )) != reliability_bin.positive_squared_probability_sum
            || canonical_evidence_number(sidecar_compensated_total(
                grouped.squared_error_sum,
                grouped.squared_error_sum_correction,
            )) != reliability_bin.squared_error_sum
        {
            return false;
        }
    }
    let expected_auc = if metrics.same_speaker_count > 0 && metrics.different_speaker_count > 0 {
        Some(canonical_evidence_number(
            concordance
                / (metrics.same_speaker_count as f64 * metrics.different_speaker_count as f64),
        ))
    } else {
        None
    };
    let same_probability_sum = canonical_evidence_number(
        reliability.probability_sum - reliability.positive_probability_sum,
    );
    let expected_same_mean =
        positive_ratio(same_probability_sum, metrics.same_speaker_count as f64);
    let expected_different_mean = positive_ratio(
        reliability.positive_probability_sum,
        metrics.different_speaker_count as f64,
    );
    let mean_shape_is_valid = |mean: Option<f64>, count: u64| {
        if count == 0 {
            mean.is_none()
        } else {
            mean.is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        }
    };
    expected_count == metrics.comparison_count
        && retained_same == metrics.same_speaker_count
        && retained_different == metrics.different_speaker_count
        && mean_shape_is_valid(
            metrics.mean_different_probability_given_same_speaker,
            metrics.same_speaker_count,
        )
        && mean_shape_is_valid(
            metrics.mean_different_probability_given_different_speaker,
            metrics.different_speaker_count,
        )
        && metrics.mean_different_probability_given_same_speaker == expected_same_mean
        && metrics.mean_different_probability_given_different_speaker == expected_different_mean
        && metrics.roc_auc == expected_auc
}

fn sidecar_auxiliary_dominance_is_valid(
    diagnostics: &PublicCorpusSidecarAuxiliaryDominanceMetrics,
    retained_same_speaker_pair_count: u64,
    retained_different_speaker_pair_count: u64,
) -> bool {
    diagnostics.same_speaker_dominance_count <= diagnostics.same_speaker_opportunity_count
        && diagnostics.different_speaker_dominance_count
            <= diagnostics.different_speaker_opportunity_count
        && diagnostics.same_speaker_opportunity_count <= retained_same_speaker_pair_count
        && diagnostics.different_speaker_opportunity_count <= retained_different_speaker_pair_count
        && diagnostics.same_speaker_dominance_rate
            == ratio(
                diagnostics.same_speaker_dominance_count,
                diagnostics.same_speaker_opportunity_count,
            )
        && diagnostics.different_speaker_dominance_rate
            == ratio(
                diagnostics.different_speaker_dominance_count,
                diagnostics.different_speaker_opportunity_count,
            )
}

fn sidecar_auxiliary_dominance_is_zero(
    diagnostics: &PublicCorpusSidecarAuxiliaryDominanceMetrics,
) -> bool {
    diagnostics.same_speaker_opportunity_count == 0
        && diagnostics.same_speaker_dominance_count == 0
        && diagnostics.same_speaker_dominance_rate.is_none()
        && diagnostics.different_speaker_opportunity_count == 0
        && diagnostics.different_speaker_dominance_count == 0
        && diagnostics.different_speaker_dominance_rate.is_none()
}

fn sidecar_uncertainty_is_valid(
    uncertainty: &PublicCorpusSidecarPairedUncertainty,
    lane: PublicCorpusSidecarLane,
    split: EvaluationSplit,
    baseline_recording_count: usize,
    candidate_recording_count: usize,
) -> bool {
    let seed_fingerprint = PublicCorpusSidecarBootstrapSeedFingerprint {
        uncertainty_id: PUBLIC_CORPUS_SIDECAR_UNCERTAINTY_VERSION,
        seed_policy_id: PUBLIC_SIDECAR_BOOTSTRAP_SEED_POLICY,
        sampler_id: PUBLIC_SIDECAR_BOOTSTRAP_SAMPLER,
        lane,
        split,
        replicates: PUBLIC_SIDECAR_BOOTSTRAP_REPLICATES,
    };
    let seed_valid = canonical_sha256(&seed_fingerprint)
        .is_ok_and(|expected| expected == uncertainty.bootstrap_seed_sha256);
    let interval_valid = |count: u64, mean: Option<f64>, lower: Option<f64>, upper: Option<f64>| {
        if count == 0 {
            mean.is_none() && lower.is_none() && upper.is_none()
        } else {
            mean.zip(lower)
                .zip(upper)
                .is_some_and(|((mean, lower), upper)| {
                    mean.is_finite() && lower.is_finite() && upper.is_finite() && lower <= upper
                })
        }
    };
    uncertainty.bootstrap_replicates == PUBLIC_SIDECAR_BOOTSTRAP_REPLICATES
        && seed_valid
        && usize::try_from(uncertainty.paired_der_recording_count)
            .is_ok_and(|count| count <= candidate_recording_count)
        && usize::try_from(uncertainty.paired_jer_recording_count)
            .is_ok_and(|count| count <= candidate_recording_count)
        && baseline_recording_count == candidate_recording_count
        && interval_valid(
            uncertainty.paired_der_recording_count,
            uncertainty.mean_der_delta,
            uncertainty.der_delta_ci95_lower,
            uncertainty.der_delta_ci95_upper,
        )
        && interval_valid(
            uncertainty.paired_jer_recording_count,
            uncertainty.mean_jer_delta,
            uncertainty.jer_delta_ci95_lower,
            uncertainty.jer_delta_ci95_upper,
        )
}

fn sidecar_split_is_valid(
    split: &PublicCorpusSidecarStudySplit,
    lane: PublicCorpusSidecarLane,
    evidence: &PublicCorpusSidecarStudyEvidence,
    baseline_recording_count: usize,
    baseline_audio_duration_sec: f64,
) -> bool {
    let is_baseline = lane == PublicCorpusSidecarLane::FullV2Baseline;
    let pipeline_valid = split.pipeline.as_ref().is_none_or(|pipeline| {
        pipeline.split == split.split
            && variant_splits_are_valid(
                std::slice::from_ref(pipeline),
                evidence.scorer_config.calibration_bins,
            )
            && pipeline.audio_duration_sec == split.performance.audio_duration_sec
            && pipeline.wall_time_sec == split.performance.wall_time_sec
            && pipeline.real_time_factor == split.performance.real_time_factor
            && pipeline.sampled_peak_rss_bytes == split.performance.sampled_peak_rss_bytes
    });
    let performance_valid = split.performance.audio_duration_sec.is_finite()
        && split.performance.audio_duration_sec >= 0.0
        && split.performance.wall_time_sec.is_finite()
        && split.performance.wall_time_sec >= 0.0
        && split.performance.real_time_factor
            == positive_ratio(
                split.performance.wall_time_sec,
                split.performance.audio_duration_sec,
            );
    let coverage = &split.coverage;
    let maximum_pair_count = coverage
        .evaluated_recording_count
        .checked_mul(PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING as u64);
    let maximum_eligible_pair_count = PUBLIC_SIDECAR_PAIR_LAGS_FRAMES
        .into_iter()
        .try_fold(0_u64, |total, lag| {
            total.checked_add(coverage.submitted_frame_count.saturating_sub(lag as u64))
        });
    let owner_available_frame_count = coverage
        .owner_available_frame_counts
        .into_iter()
        .try_fold(0_u64, |total, count| total.checked_add(count));
    let retained_class_pair_count = coverage
        .retained_same_speaker_pair_count
        .checked_add(coverage.retained_different_speaker_pair_count);
    let pair_class_recording_count_sum = coverage
        .same_speaker_pair_recording_count
        .checked_add(coverage.different_speaker_pair_recording_count);
    let [expects_channel_dominance, expects_mixed_dominance] =
        lane.auxiliary_dominance_expectations();
    let auxiliary_dominance_owner_shape_valid = (expects_channel_dominance
        || sidecar_auxiliary_dominance_is_zero(&coverage.channel_dominance))
        && (expects_mixed_dominance
            || sidecar_auxiliary_dominance_is_zero(&coverage.mixed_auxiliary_dominance));
    let minimum_per_recording_maximum = |total: u64| -> Option<u64> {
        if total == 0 {
            Some(0)
        } else {
            let quotient = total.checked_div(coverage.evaluated_recording_count)?;
            quotient.checked_add(u64::from(
                !total.is_multiple_of(coverage.evaluated_recording_count),
            ))
        }
    };
    let minimum_maximum_retained_pair_count =
        minimum_per_recording_maximum(coverage.retained_pair_sample_count);
    let minimum_maximum_retained_signal_count =
        minimum_per_recording_maximum(coverage.submitted_frame_count)
            .map(|minimum| minimum.min(PUBLIC_SIDECAR_MAX_RETAINED_SIGNALS));
    let retained_signal_shape_valid = coverage.maximum_retained_signal_count
        <= coverage.submitted_frame_count
        && (coverage.submitted_frame_count == 0
            || (coverage.maximum_retained_signal_count > 0
                && coverage.retained_signal_capacity >= PUBLIC_SIDECAR_MAX_RETAINED_SIGNALS
                && split.operations.peak_retained_state_bytes_on_target > 0));
    let minimum_retained_pair_count = coverage
        .eligible_pair_count
        .min(PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING as u64);
    let submitted_sidecar_shape_valid = coverage.submitted_frame_count == 0
        || (coverage.retained_pair_sample_capacity
            >= PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING as u64
            && (!lane.uses_frame_wavelet()
                || split.operations.peak_scratch_buffer_payload_bytes > 0));
    let zero_submitted_computation_shape_valid = coverage.submitted_frame_count > 0
        || (coverage.comparable_frame_count == 0
            && coverage.calibrated_signal_count == 0
            && coverage.consumed_probability_count == 0
            && coverage.changed_boundary_probability_count == 0
            && coverage.component_comparison_count == 0
            && coverage.owner_available_frame_counts == [0; 3]
            && sidecar_auxiliary_dominance_is_zero(&coverage.channel_dominance)
            && sidecar_auxiliary_dominance_is_zero(&coverage.mixed_auxiliary_dominance)
            && coverage.eligible_pair_count == 0
            && coverage.retained_pair_sample_count == 0
            && coverage.retained_same_speaker_pair_count == 0
            && coverage.retained_different_speaker_pair_count == 0
            && coverage.pair_score_coverage.is_none()
            && coverage.same_speaker_pair_score_coverage.is_none()
            && coverage.different_speaker_pair_score_coverage.is_none()
            && coverage.pair_scored_recording_count == 0
            && coverage.same_speaker_pair_recording_count == 0
            && coverage.different_speaker_pair_recording_count == 0
            && coverage.maximum_retained_pair_sample_count == 0
            && coverage.maximum_retained_signal_count == 0
            && split.conditional_pairs.comparison_count == 0
            && split.operations.frame_wavelet_filter_tap_terms == 0
            && split.operations.trajectory_wavelet_filter_tap_terms == 0
            && split.operations.trajectory_validity_sample_visits == 0
            && split.operations.scattering_filter_sample_terms == 0
            && split.operations.scattering_validity_sample_visits == 0
            && split
                .operations
                .modulation_projection_sample_frequency_visits
                == 0
            && split.operations.peak_scratch_buffer_payload_bytes == 0
            && split.operations.cached_twiddle_payload_bytes == 0);
    let coverage_valid = coverage.fusion_requested_recording_count
        <= coverage.evaluated_recording_count
        && coverage
            .pair_selection_sha256
            .as_ref()
            .is_none_or(|digest| is_sha256_hex(digest))
        && coverage.fusion_executed_recording_count <= coverage.fusion_requested_recording_count
        && coverage.fusion_executed_recording_count <= coverage.consumed_probability_count
        && coverage.comparable_frame_count <= coverage.submitted_frame_count
        && coverage.calibrated_signal_count == coverage.comparable_frame_count
        && coverage.consumed_probability_count <= coverage.calibrated_signal_count
        && coverage.changed_boundary_probability_count <= coverage.consumed_probability_count
        && split.boundary.probability_observation_count <= coverage.calibrated_signal_count
        && coverage.comparable_frame_coverage
            == ratio(
                coverage.comparable_frame_count,
                coverage.submitted_frame_count,
            )
        && coverage
            .owner_available_frame_counts
            .iter()
            .all(|count| *count <= coverage.comparable_frame_count)
        && owner_available_frame_count
            .is_some_and(|count| count >= coverage.comparable_frame_count)
        && coverage.component_comparison_count >= coverage.comparable_frame_count
        && sidecar_auxiliary_dominance_is_valid(
            &coverage.channel_dominance,
            coverage.retained_same_speaker_pair_count,
            coverage.retained_different_speaker_pair_count,
        )
        && sidecar_auxiliary_dominance_is_valid(
            &coverage.mixed_auxiliary_dominance,
            coverage.retained_same_speaker_pair_count,
            coverage.retained_different_speaker_pair_count,
        )
        && auxiliary_dominance_owner_shape_valid
        && coverage.retained_pair_sample_count <= coverage.eligible_pair_count
        && retained_class_pair_count == Some(coverage.retained_pair_sample_count)
        && coverage.retained_pair_sample_count >= minimum_retained_pair_count
        && split.conditional_pairs.comparison_count <= coverage.retained_pair_sample_count
        && split.conditional_pairs.same_speaker_count <= coverage.retained_same_speaker_pair_count
        && split.conditional_pairs.different_speaker_count
            <= coverage.retained_different_speaker_pair_count
        && coverage.pair_score_coverage
            == ratio(
                split.conditional_pairs.comparison_count,
                coverage.retained_pair_sample_count,
            )
        && coverage.same_speaker_pair_score_coverage
            == ratio(
                split.conditional_pairs.same_speaker_count,
                coverage.retained_same_speaker_pair_count,
            )
        && coverage.different_speaker_pair_score_coverage
            == ratio(
                split.conditional_pairs.different_speaker_count,
                coverage.retained_different_speaker_pair_count,
            )
        && coverage.pair_scored_recording_count <= coverage.evaluated_recording_count
        && coverage.same_speaker_pair_recording_count <= coverage.pair_scored_recording_count
        && coverage.different_speaker_pair_recording_count <= coverage.pair_scored_recording_count
        && pair_class_recording_count_sum
            .is_some_and(|sum| coverage.pair_scored_recording_count <= sum)
        && coverage.pair_scored_recording_count <= split.conditional_pairs.comparison_count
        && coverage.same_speaker_pair_recording_count <= split.conditional_pairs.same_speaker_count
        && coverage.different_speaker_pair_recording_count
            <= split.conditional_pairs.different_speaker_count
        && (split.conditional_pairs.comparison_count > 0)
            == (coverage.pair_scored_recording_count > 0)
        && (split.conditional_pairs.same_speaker_count > 0)
            == (coverage.same_speaker_pair_recording_count > 0)
        && (split.conditional_pairs.different_speaker_count > 0)
            == (coverage.different_speaker_pair_recording_count > 0)
        && maximum_eligible_pair_count
            .is_some_and(|maximum| coverage.eligible_pair_count <= maximum)
        && coverage.maximum_retained_pair_sample_count
            <= PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING as u64
        && coverage.maximum_retained_pair_sample_count <= coverage.retained_pair_sample_count
        && (coverage.eligible_pair_count > 0) == (coverage.maximum_retained_pair_sample_count > 0)
        && minimum_maximum_retained_pair_count
            .is_some_and(|minimum| coverage.maximum_retained_pair_sample_count >= minimum)
        && coverage.maximum_retained_pair_sample_count <= coverage.retained_pair_sample_capacity
        && coverage.retained_pair_sample_capacity
            <= evidence.protocol.maximum_retained_pair_sample_capacity as u64
        && coverage.maximum_retained_signal_count <= PUBLIC_SIDECAR_MAX_RETAINED_SIGNALS
        && minimum_maximum_retained_signal_count
            .is_some_and(|minimum| coverage.maximum_retained_signal_count >= minimum)
        && coverage.maximum_retained_signal_count <= coverage.retained_signal_capacity
        && coverage.retained_signal_capacity
            <= evidence.protocol.maximum_retained_signal_capacity as u64
        && retained_signal_shape_valid
        && submitted_sidecar_shape_valid
        && zero_submitted_computation_shape_valid
        && maximum_pair_count.is_some_and(|maximum| coverage.retained_pair_sample_count <= maximum);
    let execution_valid = if is_baseline {
        split.pipeline.is_some()
            && split.pipeline.as_ref().is_some_and(|pipeline| {
                coverage.evaluated_recording_count == pipeline.recording_count
            })
            && !split.fusion_executed
            && !coverage.fusion_requested
            && coverage.pair_selection_sha256.is_none()
            && coverage.fusion_requested_recording_count == 0
            && coverage.fusion_executed_recording_count == 0
            && coverage.submitted_frame_count == 0
            && coverage.comparable_frame_count == 0
            && coverage.calibrated_signal_count == 0
            && coverage.consumed_probability_count == 0
            && coverage.changed_boundary_probability_count == 0
            && coverage.component_comparison_count == 0
            && coverage.owner_available_frame_counts == [0; 3]
            && sidecar_auxiliary_dominance_is_zero(&coverage.channel_dominance)
            && sidecar_auxiliary_dominance_is_zero(&coverage.mixed_auxiliary_dominance)
            && coverage.eligible_pair_count == 0
            && coverage.retained_pair_sample_count == 0
            && coverage.retained_same_speaker_pair_count == 0
            && coverage.retained_different_speaker_pair_count == 0
            && coverage.pair_score_coverage.is_none()
            && coverage.same_speaker_pair_score_coverage.is_none()
            && coverage.different_speaker_pair_score_coverage.is_none()
            && coverage.pair_scored_recording_count == 0
            && coverage.same_speaker_pair_recording_count == 0
            && coverage.different_speaker_pair_recording_count == 0
            && coverage.maximum_retained_pair_sample_count == 0
            && coverage.retained_pair_sample_capacity == 0
            && coverage.maximum_retained_signal_count == 0
            && coverage.retained_signal_capacity == 0
            && split.boundary.probability_observation_count == 0
            && split.boundary.probability_positive_count == 0
            && split.boundary.brier_score.is_none()
            && split.boundary.expected_calibration_error.is_none()
            && split.conditional_pairs.comparison_count == 0
            && split.operations.frame_wavelet_filter_tap_terms == 0
            && split.operations.trajectory_wavelet_filter_tap_terms == 0
            && split.operations.trajectory_validity_sample_visits == 0
            && split.operations.scattering_filter_sample_terms == 0
            && split.operations.scattering_validity_sample_visits == 0
            && split
                .operations
                .modulation_projection_sample_frequency_visits
                == 0
            && split.operations.peak_scratch_buffer_payload_bytes == 0
            && split.operations.peak_retained_state_bytes_on_target == 0
            && split.operations.cached_twiddle_payload_bytes == 0
            && split.paired_uncertainty.is_none()
    } else {
        split.fusion_executed == (coverage.consumed_probability_count > 0)
            && (split.pipeline.is_some() == split.fusion_executed)
            && if coverage.evaluated_recording_count == 0 {
                !coverage.fusion_requested
                    && coverage.fusion_requested_recording_count == 0
                    && coverage.fusion_executed_recording_count == 0
                    && coverage.pair_selection_sha256.is_none()
                    && !split.fusion_executed
            } else {
                usize::try_from(coverage.evaluated_recording_count)
                    .is_ok_and(|count| count == baseline_recording_count)
                    && split.performance.audio_duration_sec == baseline_audio_duration_sec
                    && coverage.fusion_requested
                    && coverage.pair_selection_sha256.is_some()
                    && coverage.fusion_requested_recording_count
                        == coverage.evaluated_recording_count
                    && split.pipeline.as_ref().is_none_or(|pipeline| {
                        coverage.evaluated_recording_count == pipeline.recording_count
                    })
                    && (coverage.fusion_executed_recording_count > 0) == split.fusion_executed
            }
    };
    let candidate_recording_count = split
        .pipeline
        .as_ref()
        .and_then(|pipeline| usize::try_from(pipeline.recording_count).ok())
        .unwrap_or(0);
    let uncertainty_valid = match (&split.paired_uncertainty, split.fusion_executed) {
        (Some(uncertainty), true) => sidecar_uncertainty_is_valid(
            uncertainty,
            lane,
            split.split,
            baseline_recording_count,
            candidate_recording_count,
        ),
        (None, false) => true,
        _ => false,
    };
    let disabled_operations_valid = (lane.uses_frame_wavelet()
        || split.operations.frame_wavelet_filter_tap_terms == 0)
        && (lane.uses_trajectory_wavelet()
            || (split.operations.trajectory_wavelet_filter_tap_terms == 0
                && split.operations.trajectory_validity_sample_visits == 0))
        && (lane.uses_scattering()
            || (split.operations.scattering_filter_sample_terms == 0
                && split.operations.scattering_validity_sample_visits == 0))
        && (lane.uses_modulation()
            || (split
                .operations
                .modulation_projection_sample_frequency_visits
                == 0
                && split.operations.cached_twiddle_payload_bytes == 0));
    let memory_accounting_valid = split.operations.peak_scratch_buffer_payload_bytes
        <= evidence.protocol.maximum_reported_payload_bytes
        && split.operations.peak_retained_state_bytes_on_target
            <= evidence.protocol.maximum_reported_payload_bytes
        && split.operations.cached_twiddle_payload_bytes
            <= evidence.protocol.maximum_reported_payload_bytes;
    pipeline_valid
        && performance_valid
        && coverage_valid
        && execution_valid
        && uncertainty_valid
        && disabled_operations_valid
        && memory_accounting_valid
        && sidecar_boundary_metrics_are_valid(&split.boundary, split.pipeline.as_ref())
        && sidecar_pair_metrics_are_valid(&split.conditional_pairs)
}

fn deterministic_sidecar_accuracy_sha256(
    evidence: &PublicCorpusSidecarStudyEvidence,
) -> FwResult<String> {
    let mut normalized = evidence.clone();
    normalized.deterministic_accuracy_sha256.clear();
    normalized.result_sha256.clear();
    // The certification accuracy identity binds the deterministic development
    // accuracy identity, not its result hash (which intentionally includes
    // timing and target memory observations).
    normalized.locked_development_result_sha256 = None;
    for variant in &mut normalized.variants {
        if let Some(gate) = variant.gate.as_mut() {
            gate.relative_rtf_regression = None;
            gate.relative_rss_regression = None;
            gate.failures.retain(|failure| {
                !matches!(
                    failure,
                    PublicCorpusSidecarGateFailure::PerformanceRegression
                        | PublicCorpusSidecarGateFailure::NotSelectedByRanking
                )
            });
            gate.passed = gate.failures.is_empty();
        }
        for split in &mut variant.splits {
            split.performance.wall_time_sec = 0.0;
            split.performance.real_time_factor = None;
            split.performance.sampled_peak_rss_bytes = 0;
            // VecDeque capacity and target-sized retained-state accounting are
            // allocator/target diagnostics, not accuracy evidence.
            split.coverage.retained_signal_capacity = 0;
            split.coverage.retained_pair_sample_capacity = 0;
            split.operations.peak_retained_state_bytes_on_target = 0;
            if let Some(pipeline) = split.pipeline.as_mut() {
                pipeline.wall_time_sec = 0.0;
                pipeline.real_time_factor = None;
                pipeline.sampled_peak_rss_bytes = 0;
            }
        }
    }
    match normalized.evaluation_stage {
        PublicCorpusEvaluationStage::Development => {
            normalized.adopted_candidate_lane = None;
            normalized.selected_candidate_lane =
                apply_public_sidecar_development_selection(&mut normalized.variants);
        }
        PublicCorpusEvaluationStage::Certification => {
            let selected = normalized.selected_candidate_lane;
            normalized.adopted_candidate_lane = None;
            for variant in normalized.variants.iter_mut().skip(1) {
                if Some(variant.lane) == selected {
                    variant.disposition = if variant.gate.as_ref().is_some_and(|gate| gate.passed) {
                        normalized.adopted_candidate_lane = Some(variant.lane);
                        PublicCorpusSidecarDisposition::Adopted
                    } else {
                        PublicCorpusSidecarDisposition::Rejected
                    };
                } else {
                    variant.disposition = PublicCorpusSidecarDisposition::Rejected;
                }
            }
        }
    }
    canonical_sha256(&normalized)
}

/// Verify every frozen sidecar identity, aggregate bound, lock, and decision.
pub fn verify_public_corpus_sidecar_study_evidence(
    evidence: &PublicCorpusSidecarStudyEvidence,
) -> FwResult<()> {
    if evidence.schema_version != PUBLIC_CORPUS_SIDECAR_STUDY_SCHEMA_VERSION
        || evidence.runner_version != PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION
        || evidence.scorer_version != DIARIZATION_SCORER_VERSION
    {
        return Err(public_corpus_error(
            "sidecar_study_version",
            "sidecar study schema, runner, or scorer version is unsupported",
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
            "sidecar_corpus_key",
            "sidecar corpus key is not in the frozen public registry",
        ));
    }
    for (field, value) in [
        ("bundle_sha256", &evidence.bundle_sha256),
        ("descriptor_sha256", &evidence.descriptor_sha256),
        ("scorer_config_sha256", &evidence.scorer_config_sha256),
        ("protocol_sha256", &evidence.protocol_sha256),
        (
            "deterministic_accuracy_sha256",
            &evidence.deterministic_accuracy_sha256,
        ),
        ("result_sha256", &evidence.result_sha256),
    ] {
        if !is_sha256_hex(value) {
            return Err(public_corpus_error(
                "sidecar_hash_format",
                &format!("{field} must be 64 lowercase hexadecimal characters"),
            ));
        }
    }
    for value in [
        evidence.locked_development_result_sha256.as_deref(),
        evidence.locked_development_accuracy_sha256.as_deref(),
    ] {
        if value.is_some_and(|value| !is_sha256_hex(value)) {
            return Err(public_corpus_error(
                "sidecar_hash_format",
                "locked development hashes must be absent or lowercase SHA-256",
            ));
        }
    }
    let expected_scorer = DiarizationScorerConfig {
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
    let expected_request = DiarizationRequest {
        engine: DiarizationEngine::Acoustic,
        speaker_count: SpeakerCountRequest::Infer,
        ..DiarizationRequest::default()
    };
    let expected_protocol = public_sidecar_protocol(
        evidence.protocol.maximum_recording_duration_ms,
        &expected_request,
        canonical_sha256(&expected_request)?,
    )?;
    if evidence.scorer_config != expected_scorer
        || canonical_sha256(&evidence.scorer_config)? != evidence.scorer_config_sha256
        || evidence.protocol != expected_protocol
        || canonical_sha256(&evidence.protocol)? != evidence.protocol_sha256
    {
        return Err(public_corpus_error(
            "sidecar_protocol",
            "sidecar scorer or protocol differs from the frozen contract",
        ));
    }
    if evidence.variants.len() != PublicCorpusSidecarLane::ALL.len() {
        return Err(public_corpus_error(
            "sidecar_variants",
            "sidecar evidence must contain exactly thirteen frozen lanes",
        ));
    }
    let baseline_validation_split = evidence
        .variants
        .first()
        .and_then(|variant| variant.splits.first())
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_baseline",
                "sidecar evidence has no scored unfused baseline",
            )
        })?;
    let baseline_recording_count = baseline_validation_split
        .pipeline
        .as_ref()
        .and_then(|pipeline| usize::try_from(pipeline.recording_count).ok())
        .ok_or_else(|| {
            public_corpus_error(
                "sidecar_baseline",
                "sidecar evidence has no scored unfused baseline",
            )
        })?;
    for (variant, expected_lane) in evidence.variants.iter().zip(PublicCorpusSidecarLane::ALL) {
        if variant.pair_calibration.is_some() && variant.calibration.is_none() {
            return Err(public_corpus_error(
                "sidecar_calibration_pair",
                "a lagged-pair calibration cannot be retained without its boundary calibration",
            ));
        }
        let expected_scope = if expected_lane == PublicCorpusSidecarLane::FullV2Baseline {
            PublicCorpusSidecarFusionScope::BaselineUnfused
        } else {
            PublicCorpusSidecarFusionScope::BoundaryFusionV2
        };
        let expected_study_sha256 =
            acoustic_sidecar_study_config_sha256(expected_lane.study_config())?;
        let expected_fusion_sha256 = match variant.calibration.as_ref() {
            Some(calibration) => {
                if expected_lane == PublicCorpusSidecarLane::FullV2Baseline
                    || calibration.fit_observation_count == 0
                    || calibration.fit_positive_count == 0
                    || calibration.fit_positive_count >= calibration.fit_observation_count
                    || !is_sha256_hex(&calibration.calibration_sha256)
                    || !calibration
                        .fit_brier_score
                        .is_some_and(|value| value.is_finite() && (0.0..=1.0).contains(&value))
                    || public_sidecar_calibration_sha256(calibration)?
                        != calibration.calibration_sha256
                {
                    return Err(public_corpus_error(
                        "sidecar_calibration",
                        "retained sidecar calibration is invalid or unfitted",
                    ));
                }
                Some(acoustic_sidecar_fusion_configuration_sha256(
                    sidecar_evaluation_request(expected_lane, calibration)?,
                    evidence.protocol.detector_mode,
                )?)
            }
            None => None,
        };
        let expected_pair_calibration_sha256 = match variant.pair_calibration.as_ref() {
            Some(calibration) => {
                if expected_lane == PublicCorpusSidecarLane::FullV2Baseline {
                    return Err(public_corpus_error(
                        "sidecar_pair_calibration",
                        "the unfused baseline cannot retain a pair calibration",
                    ));
                }
                validate_public_sidecar_pair_calibration(calibration)?;
                Some(calibration.calibration_sha256.as_str())
            }
            None => None,
        };
        let expected_lane_sha256 = canonical_sha256(&PublicCorpusSidecarLaneFingerprint {
            runner_version: PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION,
            lane: expected_lane,
            fusion_scope: expected_scope,
            study_configuration_sha256: &expected_study_sha256,
            fusion_configuration_sha256: expected_fusion_sha256.as_deref(),
            pair_calibration_sha256: expected_pair_calibration_sha256,
            protocol_sha256: &evidence.protocol_sha256,
        })?;
        if variant.lane != expected_lane
            || variant.fusion_scope != expected_scope
            || variant.study_configuration_sha256 != expected_study_sha256
            || variant.fusion_configuration_sha256 != expected_fusion_sha256
            || variant.lane_configuration_sha256 != expected_lane_sha256
            || !is_sha256_hex(&variant.study_configuration_sha256)
            || !is_sha256_hex(&variant.lane_configuration_sha256)
            || variant
                .fusion_configuration_sha256
                .as_ref()
                .is_some_and(|hash| !is_sha256_hex(hash))
            || variant.splits.len() != 1
            || variant.splits[0].split != evidence.evaluation_stage.selected_split()
            || !sidecar_split_is_valid(
                &variant.splits[0],
                expected_lane,
                evidence,
                baseline_recording_count,
                baseline_validation_split.performance.audio_duration_sec,
            )
        {
            return Err(public_corpus_error(
                "sidecar_variant_contract",
                "sidecar lane identity, hashes, split, or aggregates are invalid",
            ));
        }
        if expected_lane != PublicCorpusSidecarLane::FullV2Baseline {
            let split = &variant.splits[0];
            if split.pipeline.is_none()
                && split.coverage.evaluated_recording_count == 0
                && split != &unavailable_public_sidecar_split(split.split)?
            {
                return Err(public_corpus_error(
                    "sidecar_unavailable_shape",
                    "an unavailable sidecar lane must use the canonical all-zero split",
                ));
            }
            if split.coverage.evaluated_recording_count > 0
                && (variant.calibration.is_none() || variant.fusion_configuration_sha256.is_none())
            {
                return Err(public_corpus_error(
                    "sidecar_fusion_evidence",
                    "an evaluated sidecar lane requires calibration and fusion identity",
                ));
            }
            if evidence.evaluation_stage == PublicCorpusEvaluationStage::Development
                && split.coverage.evaluated_recording_count > 0
            {
                let calibration = variant.calibration.as_ref().ok_or_else(|| {
                    public_corpus_error(
                        "sidecar_calibration_provenance",
                        "development boundary calibration is missing",
                    )
                })?;
                if calibration.fit_observation_count != split.boundary.probability_observation_count
                    || calibration.fit_positive_count != split.boundary.probability_positive_count
                {
                    return Err(public_corpus_error(
                        "sidecar_calibration_provenance",
                        "development boundary fit counts do not match the measured rows",
                    ));
                }
                match variant.pair_calibration.as_ref() {
                    Some(pair_calibration)
                        if pair_calibration.fit_observation_count
                            == split.conditional_pairs.comparison_count
                            && pair_calibration.fit_positive_count
                                == split.conditional_pairs.different_speaker_count => {}
                    None if split.conditional_pairs.comparison_count == 0 => {}
                    _ => {
                        return Err(public_corpus_error(
                            "sidecar_pair_calibration_provenance",
                            "development pair fit counts do not match the measured rows",
                        ));
                    }
                }
            }
        }
    }
    let mut common_pair_population = None;
    for variant in evidence.variants.iter().skip(1) {
        let coverage = &variant.splits[0].coverage;
        if coverage.evaluated_recording_count == 0 {
            continue;
        }
        let fingerprint = (
            coverage.eligible_pair_count,
            coverage.retained_pair_sample_count,
            coverage.retained_same_speaker_pair_count,
            coverage.retained_different_speaker_pair_count,
            coverage.maximum_retained_pair_sample_count,
            coverage.pair_selection_sha256.as_deref(),
        );
        if common_pair_population.is_some_and(|expected| expected != fingerprint) {
            return Err(public_corpus_error(
                "sidecar_pair_population",
                "evaluated lanes do not share one reference-labeled selected-pair universe",
            ));
        }
        common_pair_population = Some(fingerprint);
    }
    let baseline = &evidence.variants[0];
    if baseline.disposition != PublicCorpusSidecarDisposition::Baseline
        || baseline.calibration.is_some()
        || baseline.pair_calibration.is_some()
        || baseline.fusion_configuration_sha256.is_some()
        || baseline.gate.is_some()
    {
        return Err(public_corpus_error(
            "sidecar_baseline",
            "unfused baseline has an invalid calibration, gate, or disposition",
        ));
    }
    let baseline_split = &baseline.splits[0];
    let stage_valid = match evidence.evaluation_stage {
        PublicCorpusEvaluationStage::Development => {
            if evidence.locked_development_result_sha256.is_some()
                || evidence.locked_development_accuracy_sha256.is_some()
                || evidence.adopted_candidate_lane.is_some()
                || evidence.variants.iter().skip(1).any(|variant| {
                    variant.calibration.is_some()
                        != (variant.splits[0].coverage.evaluated_recording_count > 0)
                })
            {
                false
            } else {
                let mut expected_variants = evidence.variants.clone();
                for variant in expected_variants.iter_mut().skip(1) {
                    variant.gate = Some(public_sidecar_promotion_gate(
                        evidence.evaluation_stage,
                        &evidence.protocol.gate_policy,
                        baseline_split,
                        variant.lane,
                        &variant.splits[0],
                    ));
                    variant.disposition = PublicCorpusSidecarDisposition::Rejected;
                }
                let expected_selected =
                    apply_public_sidecar_development_selection(&mut expected_variants);
                evidence.selected_candidate_lane == expected_selected
                    && evidence
                        .variants
                        .iter()
                        .skip(1)
                        .zip(expected_variants.iter().skip(1))
                        .all(|(variant, expected)| {
                            variant.gate == expected.gate
                                && variant.disposition == expected.disposition
                        })
            }
        }
        PublicCorpusEvaluationStage::Certification => {
            let Some(selected) = evidence.selected_candidate_lane else {
                return Err(public_corpus_error(
                    "sidecar_stage_contract",
                    "certification evidence has no locked selected candidate",
                ));
            };
            if evidence.locked_development_result_sha256.is_none()
                || evidence.locked_development_accuracy_sha256.is_none()
                || selected == PublicCorpusSidecarLane::FullV2Baseline
            {
                false
            } else {
                evidence.variants.iter().skip(1).all(|variant| {
                    if variant.lane == selected {
                        let expected_gate = public_sidecar_promotion_gate(
                            evidence.evaluation_stage,
                            &evidence.protocol.gate_policy,
                            baseline_split,
                            variant.lane,
                            &variant.splits[0],
                        );
                        variant.calibration.is_some()
                            && variant.pair_calibration.is_some()
                            && variant.splits[0].coverage.evaluated_recording_count > 0
                            && variant.gate.as_ref() == Some(&expected_gate)
                            && variant.disposition
                                == if expected_gate.passed {
                                    PublicCorpusSidecarDisposition::Adopted
                                } else {
                                    PublicCorpusSidecarDisposition::Rejected
                                }
                    } else {
                        variant.gate.is_none()
                            && variant.disposition == PublicCorpusSidecarDisposition::Rejected
                            && variant.splits[0].pipeline.is_none()
                            && variant.splits[0].coverage.evaluated_recording_count == 0
                            && !variant.splits[0].fusion_executed
                    }
                }) && evidence.adopted_candidate_lane
                    == evidence
                        .variants
                        .iter()
                        .find(|variant| {
                            variant.disposition == PublicCorpusSidecarDisposition::Adopted
                        })
                        .map(|variant| variant.lane)
            }
        }
    };
    if !stage_valid {
        return Err(public_corpus_error(
            "sidecar_stage_contract",
            "sidecar stage, locks, gates, selection, or dispositions are inconsistent",
        ));
    }
    if deterministic_sidecar_accuracy_sha256(evidence)? != evidence.deterministic_accuracy_sha256 {
        return Err(public_corpus_error(
            "sidecar_accuracy_hash_mismatch",
            "sidecar deterministic accuracy hash does not match normalized evidence",
        ));
    }
    let mut unhashed = evidence.clone();
    let expected_result_sha256 = unhashed.result_sha256.clone();
    unhashed.result_sha256.clear();
    if canonical_sha256(&unhashed)? != expected_result_sha256 {
        return Err(public_corpus_error(
            "sidecar_hash_mismatch",
            "result_sha256 does not match canonical sidecar evidence",
        ));
    }
    Ok(())
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
        let posterior_map_confusion_count = split
            .speaker_count_posterior_map_confusion
            .iter()
            .map(|cell| cell.recording_count)
            .sum::<u64>();
        let posterior_map_confusion_valid = split
            .speaker_count_posterior_map_confusion
            .windows(2)
            .all(|window| {
                (window[0].reference_speakers, window[0].hypothesis_speakers)
                    < (window[1].reference_speakers, window[1].hypothesis_speakers)
            })
            && split
                .speaker_count_posterior_map_confusion
                .iter()
                .all(|cell| cell.recording_count > 0)
            && posterior_map_confusion_count == split.count_posterior_recording_count;
        let merge_frontier = &split.speaker_count_merge_frontier;
        let merge_frontier_valid = merge_frontier.recording_count <= split.recording_count
            && merge_frontier.to_reference_count_observation_count
                <= merge_frontier.recording_count
            && merge_frontier.below_reference_count_observation_count
                <= merge_frontier.recording_count
            && merge_frontier.paired_frontier_observation_count
                <= merge_frontier
                    .to_reference_count_observation_count
                    .min(merge_frontier.below_reference_count_observation_count)
            && merge_frontier.correctly_ordered_frontier_count
                <= merge_frontier.paired_frontier_observation_count
            && bounded(merge_frontier.mean_probability_to_reference_count)
            && bounded(merge_frontier.mean_probability_below_reference_count)
            && merge_frontier
                .mean_probability_margin
                .is_none_or(|margin| margin.is_finite() && (-1.0..=1.0).contains(&margin))
            && bounded(merge_frontier.correctly_ordered_frontier_rate)
            && (merge_frontier.to_reference_count_observation_count > 0)
                == merge_frontier.mean_probability_to_reference_count.is_some()
            && (merge_frontier.below_reference_count_observation_count > 0)
                == merge_frontier
                    .mean_probability_below_reference_count
                    .is_some()
            && (merge_frontier.paired_frontier_observation_count > 0)
                == merge_frontier.mean_probability_margin.is_some()
            && merge_frontier.correctly_ordered_frontier_rate
                == ratio(
                    merge_frontier.correctly_ordered_frontier_count,
                    merge_frontier.paired_frontier_observation_count,
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
            && posterior_map_confusion_valid
            && merge_frontier_valid
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
    if !value.is_finite() {
        return value;
    }
    // Decimal formatting performs the rounding in decimal space. In
    // contrast, multiply/round/divide by 1e12 is not idempotent once the
    // scaled value approaches f64's integer-precision limit: a producer's
    // once-canonical value can move again when the verifier canonicalizes it.
    // Parsing a fixed-12 decimal representation yields a stable lattice point
    // while retaining resolution far below the smallest frozen f32 score-bin
    // gap.
    let rounded = format!("{value:.12}").parse::<f64>().unwrap_or(value);
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
    validate_relative_path_syntax(relative, field)?;
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

fn validate_relative_path_syntax(relative: &Path, field: &str) -> FwResult<()> {
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
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
struct ValidatedOutputParent {
    canonical_path: PathBuf,
    requested_path: PathBuf,
    directory: File,
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
struct ValidatedOutputParent;

impl ValidatedOutputParent {
    fn same_output_target(
        &self,
        output: &Path,
        other: &Self,
        other_output: &Path,
        field: &str,
    ) -> FwResult<bool> {
        let file_name = output
            .file_name()
            .ok_or_else(|| public_corpus_error(field, "output must have a terminal file name"))?;
        let other_file_name = other_output
            .file_name()
            .ok_or_else(|| public_corpus_error(field, "output must have a terminal file name"))?;
        let file_name = file_name
            .to_str()
            .ok_or_else(|| public_corpus_error(field, "output file names must be ASCII"))?;
        let other_file_name = other_file_name
            .to_str()
            .ok_or_else(|| public_corpus_error(field, "output file names must be ASCII"))?;
        #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
        {
            use std::os::unix::fs::MetadataExt as _;

            verify_output_parent_identity(self)?;
            verify_output_parent_identity(other)?;
            let left = self.directory.metadata().map_err(|_| {
                public_corpus_error(field, "output parent identity could not be read")
            })?;
            let right = other.directory.metadata().map_err(|_| {
                public_corpus_error(field, "output parent identity could not be read")
            })?;
            Ok(file_name.eq_ignore_ascii_case(other_file_name)
                && left.dev() == right.dev()
                && left.ino() == right.ino())
        }
        #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
        {
            let _ = (self, other, file_name, other_file_name);
            Err(FwError::Unsupported(
                "public_corpus.output_platform: race-safe public artifact publication requires Linux, Android, or an Apple platform"
                    .to_owned(),
            ))
        }
    }
}

fn validate_new_output(
    project: &Path,
    input: &Path,
    output: &Path,
) -> FwResult<ValidatedOutputParent> {
    if !output.is_absolute() || output.extension().and_then(|value| value.to_str()) != Some("json")
    {
        return Err(public_corpus_error(
            "output_path",
            "output must be an absolute path with a .json extension",
        ));
    }
    let parent = output.parent().ok_or_else(|| {
        public_corpus_error(
            "output_parent",
            "output must have an existing parent directory",
        )
    })?;
    let output_name = output
        .file_name()
        .ok_or_else(|| public_corpus_error("output_path", "output must include a file name"))?;
    let output_name_text = output_name
        .to_str()
        .ok_or_else(|| public_corpus_error("output_path", "output file names must be ASCII"))?;
    if !output_name_text.is_ascii()
        || output_name_text
            .bytes()
            .any(|byte| byte.is_ascii_uppercase())
        || !output_name_text
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(public_corpus_error(
            "output_path",
            "output file names may contain only lowercase ASCII letters, digits, period, underscore, and hyphen",
        ));
    }
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        use rustix::fs::{Mode, OFlags, open};

        // Open first, then canonicalize and compare identity. Canonicalizing
        // before opening would leave a swap window between the overlap check
        // and the directory handle that later owns openat publication.
        // DIRECTORY and NOFOLLOW also reject a FIFO/non-directory or a
        // terminal symlink without a potentially blocking std::fs::File open.
        let directory = File::from(
            open(
                parent,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(|_| {
                public_corpus_error("output_parent", "output parent could not be identity-bound")
            })?,
        );
        let canonical_parent = parent.canonicalize().map_err(|_| {
            public_corpus_error(
                "output_parent",
                "output parent must be an existing directory",
            )
        })?;
        let validated = ValidatedOutputParent {
            canonical_path: canonical_parent,
            requested_path: parent.to_owned(),
            directory,
        };
        verify_output_parent_identity(&validated)?;
        use rustix::fs::{AtFlags, Dir, statat};

        // Enumerate through the already identity-bound directory handle, not
        // through a path that could resolve to a replacement directory. This
        // makes an existing differently-cased sibling fail consistently on
        // case-sensitive and case-insensitive filesystems. Exact `statat` and
        // no-clobber publication remain the atomic guards; the effective user
        // is already the trust boundary for concurrent directory mutation.
        let entries = Dir::read_from(&validated.directory).map_err(|_| {
            public_corpus_error(
                "output_target",
                "output parent entries could not be checked for case-fold collisions",
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|_| {
                public_corpus_error(
                    "output_target",
                    "output parent entries could not be checked for case-fold collisions",
                )
            })?;
            if entry
                .file_name()
                .to_bytes()
                .eq_ignore_ascii_case(output_name_text.as_bytes())
            {
                return Err(public_corpus_error(
                    "output_exists",
                    "output or an ASCII-case-fold sibling already exists",
                ));
            }
        }

        match statat(&validated.directory, output_name, AtFlags::SYMLINK_NOFOLLOW) {
            Ok(_) => {
                return Err(public_corpus_error(
                    "output_exists",
                    "output must not already exist",
                ));
            }
            Err(error) if error == rustix::io::Errno::NOENT => {}
            Err(_) => {
                return Err(public_corpus_error(
                    "output_target",
                    "output target could not be checked relative to its validated parent",
                ));
            }
        }
        if paths_overlap(project, &validated.canonical_path)
            || paths_overlap(input, &validated.canonical_path)
        {
            return Err(public_corpus_error(
                "output_overlap",
                "output parent must be disjoint from the project and input roots",
            ));
        }
        Ok(validated)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    {
        let _ = (project, input, parent);
        Err(FwError::Unsupported(
            "public_corpus.output_platform: race-safe public artifact publication requires Linux, Android, or an Apple platform"
                .to_owned(),
        ))
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn verify_output_parent_identity(parent: &ValidatedOutputParent) -> FwResult<()> {
    use std::os::unix::fs::MetadataExt as _;

    let opened = parent.directory.metadata().map_err(|_| {
        public_corpus_error(
            "output_parent_changed",
            "validated output parent identity could not be read",
        )
    })?;
    let current = parent.canonical_path.symlink_metadata().map_err(|_| {
        public_corpus_error(
            "output_parent_changed",
            "output parent changed after validation",
        )
    })?;
    let requested = parent.requested_path.symlink_metadata().map_err(|_| {
        public_corpus_error(
            "output_parent_changed",
            "requested output parent changed after validation",
        )
    })?;
    if !opened.is_dir()
        || !current.is_dir()
        || !requested.is_dir()
        || opened.dev() != current.dev()
        || opened.ino() != current.ino()
        || opened.dev() != requested.dev()
        || opened.ino() != requested.ino()
    {
        return Err(public_corpus_error(
            "output_parent_changed",
            "output parent changed after validation",
        ));
    }
    if opened.uid() != rustix::process::geteuid().as_raw() || opened.mode() & 0o022 != 0 {
        return Err(public_corpus_error(
            "output_parent_permissions",
            "output parent must be owned by the effective user and not group/world writable",
        ));
    }
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputLeafState {
    Missing,
    Expected,
    Other,
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn output_leaf_state(
    parent: &ValidatedOutputParent,
    name: &std::ffi::OsStr,
    expected: &File,
) -> FwResult<OutputLeafState> {
    use std::os::unix::fs::MetadataExt as _;

    use rustix::fs::{AtFlags, FileType, statat};

    let expected = expected.metadata().map_err(|_| {
        public_corpus_error(
            "output_target_changed",
            "staged output identity could not be verified",
        )
    })?;
    let current = match statat(&parent.directory, name, AtFlags::SYMLINK_NOFOLLOW) {
        Ok(current) => current,
        Err(error) if error == rustix::io::Errno::NOENT => return Ok(OutputLeafState::Missing),
        Err(_) => {
            return Err(public_corpus_error(
                "output_target_changed",
                "staged output identity could not be inspected during publication",
            ));
        }
    };
    #[cfg(target_vendor = "apple")]
    let device_matches = expected.dev() == current.st_dev as u64;
    #[cfg(not(target_vendor = "apple"))]
    let device_matches = expected.dev() == current.st_dev;
    if expected.is_file()
        && FileType::from_raw_mode(current.st_mode) == FileType::RegularFile
        && device_matches
        && expected.ino() == current.st_ino
    {
        Ok(OutputLeafState::Expected)
    } else {
        Ok(OutputLeafState::Other)
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn verify_output_leaf_identity(
    parent: &ValidatedOutputParent,
    name: &std::ffi::OsStr,
    expected: &File,
) -> FwResult<()> {
    if output_leaf_state(parent, name, expected)? != OutputLeafState::Expected {
        return Err(public_corpus_error(
            "output_target_changed",
            "staged output identity changed during publication",
        ));
    }
    Ok(())
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    if left.starts_with(right) || right.starts_with(left) {
        return true;
    }
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        use std::os::unix::fs::MetadataExt as _;

        let ancestry = |path: &Path| -> std::io::Result<Vec<(u64, u64)>> {
            let mut identities = Vec::new();
            let mut current = path;
            loop {
                let metadata = current.metadata()?;
                if !metadata.is_dir() {
                    return Err(std::io::Error::other("path ancestor is not a directory"));
                }
                identities.push((metadata.dev(), metadata.ino()));
                let Some(parent) = current.parent() else {
                    break;
                };
                if parent == current {
                    break;
                }
                current = parent;
            }
            Ok(identities)
        };
        // Canonical path strings do not collapse every filesystem alias. APFS
        // firmlinks and aliases of either root can expose the same directory
        // inode under disjoint spellings, so compare each terminal directory
        // identity against the other path's ancestor chain. An arbitrary mount
        // alias of a strict descendant is not discoverable from these two
        // ancestry walks and remains outside this check's authority. Metadata
        // failure is treated as overlap to keep publication fail-closed.
        match (ancestry(left), ancestry(right)) {
            (Ok(left_ancestry), Ok(right_ancestry)) => {
                let left_identity = left_ancestry[0];
                let right_identity = right_ancestry[0];
                right_ancestry.contains(&left_identity) || left_ancestry.contains(&right_identity)
            }
            _ => true,
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    {
        false
    }
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

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
struct StagedJsonOutput<'a> {
    output_parent: &'a ValidatedOutputParent,
    staging_name: OsString,
    output_name: OsString,
    file: File,
    preserve_payload_on_drop: bool,
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
struct StagedJsonOutput<'a>(std::marker::PhantomData<&'a ()>);

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
impl StagedJsonOutput<'_> {
    fn scrub_payload(&self) -> std::io::Result<()> {
        self.file.set_len(0)?;
        self.file.sync_all()
    }

    fn scrub_payload_or_error(&self, artifact: &str) -> FwResult<()> {
        self.scrub_payload().map_err(|_| {
            public_corpus_error(
                "output_cleanup_uncertain",
                &format!(
                    "{artifact} staging payload could not be explicitly truncated and synchronized"
                ),
            )
        })
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
impl StagedJsonOutput<'_> {
    fn scrub_payload_or_error(&self, _artifact: &str) -> FwResult<()> {
        Ok(())
    }
}

fn staged_scrubbed_error(
    primary: FwError,
    staged_outputs: &[(&StagedJsonOutput<'_>, &str)],
) -> FwError {
    let mut uncertain_artifacts = Vec::new();
    for (staged, artifact) in staged_outputs {
        if staged.scrub_payload_or_error(artifact).is_err() {
            uncertain_artifacts.push(*artifact);
        }
    }
    if uncertain_artifacts.is_empty() {
        primary
    } else {
        public_corpus_error(
            "output_cleanup_uncertain",
            &format!(
                "staging payload cleanup is uncertain for {}; preceding failure: {primary}",
                uncertain_artifacts.join(", ")
            ),
        )
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
impl Drop for StagedJsonOutput<'_> {
    fn drop(&mut self) {
        if !self.preserve_payload_on_drop {
            // Panic/unwind fallback. Ordinary errors explicitly scrub and
            // report any cleanup failure before returning from staging or
            // publication orchestration.
            let _ = self.scrub_payload();
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
struct CancellingWriter<'a, W, C> {
    inner: W,
    is_cancelled: &'a mut C,
    cancelled: bool,
    bytes_until_cancel_check: usize,
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
impl<W: Write, C: FnMut() -> bool> Write for CancellingWriter<'_, W, C> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.cancelled || (self.bytes_until_cancel_check == 0 && (self.is_cancelled)()) {
            self.cancelled = true;
            return Err(std::io::Error::other(
                "public corpus output serialization cancelled",
            ));
        }
        if self.bytes_until_cancel_check == 0 {
            self.bytes_until_cancel_check = PUBLIC_OUTPUT_CANCELLATION_GRANULARITY_BYTES;
        }
        let maximum_write = bytes.len().min(self.bytes_until_cancel_check);
        let written = self.inner.write(&bytes[..maximum_write])?;
        self.bytes_until_cancel_check = self.bytes_until_cancel_check.saturating_sub(written);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.cancelled || (self.is_cancelled)() {
            self.cancelled = true;
            return Err(std::io::Error::other(
                "public corpus output serialization cancelled",
            ));
        }
        self.inner.flush()
    }
}

fn stage_new_json<'a, T: Serialize>(
    output_path: &Path,
    output_parent: &'a ValidatedOutputParent,
    value: &T,
    artifact: &str,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<StagedJsonOutput<'a>> {
    checkpoint_cancelled(is_cancelled)?;
    let output_name = output_path
        .file_name()
        .ok_or_else(|| public_corpus_error("output_path", "output must include a file name"))?
        .to_owned();
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        use std::os::unix::fs::MetadataExt as _;

        use rustix::fs::{Mode, OFlags, fchmod, openat};

        verify_output_parent_identity(output_parent)?;
        let mut staging = None;
        for _ in 0..8 {
            let staging_name = OsString::from(format!(
                ".franken-whisper-output-{}.tmp",
                Uuid::new_v4().simple()
            ));
            match openat(
                &output_parent.directory,
                &staging_name,
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::RUSR | Mode::WUSR,
            ) {
                Ok(descriptor) => {
                    staging = Some((staging_name, File::from(descriptor)));
                    break;
                }
                Err(error) if error == rustix::io::Errno::EXIST => {}
                Err(_) => {
                    return Err(public_corpus_error(
                        "output_create",
                        &format!("new {artifact} staging output could not be created"),
                    ));
                }
            }
        }
        let (staging_name, file) = staging.ok_or_else(|| {
            public_corpus_error(
                "output_create",
                &format!("new {artifact} staging output could not be created"),
            )
        })?;
        let staged = StagedJsonOutput {
            output_parent,
            staging_name,
            output_name,
            file,
            preserve_payload_on_drop: false,
        };
        if fchmod(&staged.file, Mode::RUSR | Mode::WUSR).is_err() {
            staged.scrub_payload_or_error(artifact)?;
            return Err(public_corpus_error(
                "output_target_permissions",
                "new staging output permissions could not be restricted to mode 0600",
            ));
        }
        if let Err(error) = verify_output_parent_identity(output_parent) {
            staged.scrub_payload_or_error(artifact)?;
            return Err(error);
        }
        let staging_metadata = match staged.file.metadata() {
            Ok(metadata) => metadata,
            Err(_) => {
                staged.scrub_payload_or_error(artifact)?;
                return Err(public_corpus_error(
                    "output_target_changed",
                    "new staging output identity could not be verified",
                ));
            }
        };
        if !staging_metadata.is_file()
            || staging_metadata.uid() != rustix::process::geteuid().as_raw()
            || staging_metadata.mode() & 0o777 != 0o600
        {
            staged.scrub_payload_or_error(artifact)?;
            return Err(public_corpus_error(
                "output_target_permissions",
                "new staging output must remain a mode-0600 regular file owned by the effective user",
            ));
        }
        let mut writer = CancellingWriter {
            inner: BufWriter::new(&staged.file),
            is_cancelled,
            cancelled: false,
            bytes_until_cancel_check: PUBLIC_OUTPUT_CANCELLATION_GRANULARITY_BYTES,
        };
        let serialized = serde_json::to_writer_pretty(&mut writer, value);
        if writer.cancelled || serialized.is_err() {
            let cancelled = writer.cancelled;
            let CancellingWriter { inner, .. } = writer;
            let (_, buffered) = inner.into_parts();
            drop(buffered);
            staged.scrub_payload_or_error(artifact)?;
            return if cancelled {
                Err(public_corpus_cancelled_error())
            } else {
                Err(public_corpus_error(
                    "output_write",
                    &format!("{artifact} output could not be serialized"),
                ))
            };
        }
        let newline = writer.write_all(b"\n");
        if writer.cancelled || newline.is_err() {
            let cancelled = writer.cancelled;
            let CancellingWriter { inner, .. } = writer;
            let (_, buffered) = inner.into_parts();
            drop(buffered);
            staged.scrub_payload_or_error(artifact)?;
            return if cancelled {
                Err(public_corpus_cancelled_error())
            } else {
                Err(public_corpus_error(
                    "output_write",
                    &format!("{artifact} output could not be written"),
                ))
            };
        }
        let flushed = writer.flush();
        if writer.cancelled || flushed.is_err() {
            let cancelled = writer.cancelled;
            let CancellingWriter { inner, .. } = writer;
            let (_, buffered) = inner.into_parts();
            drop(buffered);
            staged.scrub_payload_or_error(artifact)?;
            return if cancelled {
                Err(public_corpus_cancelled_error())
            } else {
                Err(public_corpus_error(
                    "output_write",
                    &format!("{artifact} output could not be flushed"),
                ))
            };
        }
        drop(writer);
        if staged.file.sync_all().is_err() {
            staged.scrub_payload_or_error(artifact)?;
            return Err(public_corpus_error(
                "output_write",
                &format!("{artifact} staging output could not be durably synchronized"),
            ));
        }
        if is_cancelled() {
            staged.scrub_payload_or_error(artifact)?;
            return Err(public_corpus_cancelled_error());
        }
        if let Err(error) = verify_output_parent_identity(output_parent) {
            staged.scrub_payload_or_error(artifact)?;
            return Err(error);
        }
        Ok(staged)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    {
        let _ = (output_parent, value, artifact, is_cancelled, output_name);
        Err(FwError::Unsupported(
            "public_corpus.output_platform: atomic public artifact publication requires Linux, Android, or an Apple platform"
                .to_owned(),
        ))
    }
}

fn publish_staged_json(staged: StagedJsonOutput<'_>, artifact: &str) -> FwResult<()> {
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        use rustix::fs::{RenameFlags, renameat_with};

        let mut staged = staged;
        if let Err(error) = verify_output_parent_identity(staged.output_parent) {
            return Err(staged_scrubbed_error(error, &[(&staged, artifact)]));
        }
        if let Err(error) =
            verify_output_leaf_identity(staged.output_parent, &staged.staging_name, &staged.file)
        {
            return Err(staged_scrubbed_error(error, &[(&staged, artifact)]));
        }
        let renamed = renameat_with(
            &staged.output_parent.directory,
            &staged.staging_name,
            &staged.output_parent.directory,
            &staged.output_name,
            RenameFlags::NOREPLACE,
        );
        if let Err(rename_error) = renamed {
            let final_state =
                output_leaf_state(staged.output_parent, &staged.output_name, &staged.file);
            let staging_state =
                output_leaf_state(staged.output_parent, &staged.staging_name, &staged.file);
            match (final_state, staging_state) {
                (Ok(OutputLeafState::Expected), _) => {
                    staged.preserve_payload_on_drop = true;
                    return Err(public_corpus_error(
                        "output_commit_uncertain",
                        &format!(
                            "{artifact} output names the staged inode after an ambiguous publication error"
                        ),
                    ));
                }
                (
                    Ok(OutputLeafState::Missing | OutputLeafState::Other),
                    Ok(OutputLeafState::Expected),
                ) if rename_error == rustix::io::Errno::EXIST => {
                    let error = public_corpus_error(
                        "output_create",
                        &format!("new {artifact} output could not be atomically published"),
                    );
                    return Err(staged_scrubbed_error(error, &[(&staged, artifact)]));
                }
                _ => {
                    // Only a no-clobber EXIST result plus an intact source is
                    // authoritative pre-commit evidence. Other filesystem
                    // errors may be ambiguous, so do not truncate the held
                    // inode even when a follow-up lookup appears unchanged.
                    staged.preserve_payload_on_drop = true;
                    return Err(public_corpus_error(
                        "output_commit_uncertain",
                        &format!(
                            "{artifact} publication state could not be resolved after a rename error"
                        ),
                    ));
                }
            }
        }
        // The no-clobber rename is the publication commit point. From here on,
        // an error means the committed file could not be durably confirmed;
        // Drop must never truncate that final-name inode.
        staged.preserve_payload_on_drop = true;
        verify_output_parent_identity(staged.output_parent).map_err(|_| {
            public_corpus_error(
                "output_commit_uncertain",
                &format!(
                    "{artifact} output was published but its parent identity could not be confirmed"
                ),
            )
        })?;
        verify_output_leaf_identity(staged.output_parent, &staged.output_name, &staged.file)
            .map_err(|_| {
                public_corpus_error(
                    "output_commit_uncertain",
                    &format!(
                        "{artifact} output was published but its final identity could not be confirmed"
                    ),
                )
            })?;
        staged.output_parent.directory.sync_all().map_err(|_| {
            public_corpus_error(
                "output_commit_uncertain",
                &format!(
                    "{artifact} output was published but its directory could not be durably synchronized"
                ),
            )
        })?;
        Ok(())
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    {
        let _ = (staged, artifact);
        Err(FwError::Unsupported(
            "public_corpus.output_platform: atomic public artifact publication requires Linux, Android, or an Apple platform"
                .to_owned(),
        ))
    }
}

fn write_new_json<T: Serialize>(
    output_path: &Path,
    output_parent: &ValidatedOutputParent,
    value: &T,
    artifact: &str,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<()> {
    let staged = stage_new_json(output_path, output_parent, value, artifact, is_cancelled)?;
    publish_staged_json(staged, artifact)
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
        Err(public_corpus_cancelled_error())
    } else {
        Ok(())
    }
}

fn public_corpus_cancelled_error() -> FwError {
    FwError::Cancelled("public_corpus.cancelled: public corpus preparation cancelled".to_owned())
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

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    use serde::Serialize;
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    use serde::ser::SerializeStruct as _;

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    use super::write_new_json;
    use super::{
        PUBLIC_CORPUS_INPUT_SCHEMA_VERSION, PublicAblationAccumulator, PublicCorpusAblationSplit,
        PublicCorpusAblationVariant, build_public_corpus_bundle,
        build_public_corpus_bundle_with_cancel, clipped_reference, development_improvement_gate,
        held_out_non_regression_gate, merged_scored_speech_regions, parse_public_corpus_bundle,
        public_corpus_registry, validate_new_output, validate_split,
    };
    use crate::FwResult;
    use crate::diarization::{
        AcousticFeatureAblation, DIARIZATION_REFERENCE_SCHEMA_VERSION,
        DiarizationReferenceDocument, EvaluationRegion, EvaluationSplit, EvaluationTurn,
    };

    fn private_tempdir(label: &str) -> tempfile::TempDir {
        let directory = tempdir().unwrap_or_else(|error| panic!("{label}: {error}"));
        #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
        {
            use std::os::unix::fs::PermissionsExt as _;

            std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap_or_else(|error| panic!("private {label}: {error}"));
        }
        directory
    }

    fn create_private_directory(path: &Path) {
        std::fs::create_dir(path).expect("fixture directory");
        #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
        {
            use std::os::unix::fs::PermissionsExt as _;

            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
                .expect("private fixture directory");
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    struct AlwaysFailsSerialization;

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    impl Serialize for AlwaysFailsSerialization {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            let mut state = serializer.serialize_struct("SensitiveFixture", 2)?;
            state.serialize_field("sensitive", "must-not-survive")?;
            Err(<S::Error as serde::ser::Error>::custom(
                "intentional failure after a serialized field",
            ))
        }
    }

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

    fn external_recording(
        recording_id: &str,
        split: &str,
        audio_path: &str,
        audio_sha256: &str,
        annotation_path: &str,
        annotation_sha256: &str,
    ) -> serde_json::Value {
        json!({
            "recording_id": recording_id,
            "split": split,
            "origin_recording_id": format!("{recording_id}-origin"),
            "audio_path": audio_path,
            "audio_sha256": audio_sha256,
            "expected_sample_rate_hz": 16_000,
            "expected_channel_count": 1,
            "selected_channel": 1,
            "annotation_path": annotation_path,
            "annotation_sha256": annotation_sha256,
            "annotation_recording_id": format!("source-{recording_id}"),
            "annotation_channel": "1",
            "speaker_map": {
                "source-speaker": format!("{recording_id}-speaker")
            },
            "ignored_regions": []
        })
    }

    fn sidecar_hash_fixture() -> super::PublicCorpusSidecarStudyEvidence {
        let scorer_config = crate::diarization::DiarizationScorerConfig::default();
        let diarization_request = crate::model::DiarizationRequest {
            engine: crate::model::DiarizationEngine::Acoustic,
            speaker_count: crate::model::SpeakerCountRequest::Infer,
            ..crate::model::DiarizationRequest::default()
        };
        let request_sha256 =
            super::canonical_sha256(&diarization_request).expect("diarization request hash");
        let protocol = super::public_sidecar_protocol(None, &diarization_request, request_sha256)
            .expect("sidecar protocol");
        let protocol_sha256 = super::canonical_sha256(&protocol).expect("protocol hash");
        super::PublicCorpusSidecarStudyEvidence {
            schema_version: super::PUBLIC_CORPUS_SIDECAR_STUDY_SCHEMA_VERSION.to_owned(),
            runner_version: super::PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION.to_owned(),
            scorer_version: crate::diarization::DIARIZATION_SCORER_VERSION.to_owned(),
            corpus_key: "aishell-4-openslr111-v1".to_owned(),
            source_version: "fixture-v1".to_owned(),
            bundle_sha256: "1".repeat(64),
            descriptor_sha256: "2".repeat(64),
            scorer_config_sha256: super::canonical_sha256(&scorer_config).expect("scorer hash"),
            scorer_config,
            evaluation_stage: super::PublicCorpusEvaluationStage::Development,
            locked_development_result_sha256: None,
            locked_development_accuracy_sha256: None,
            protocol,
            protocol_sha256,
            selected_candidate_lane: None,
            adopted_candidate_lane: None,
            variants: vec![
                super::PublicCorpusSidecarStudyVariant {
                    lane: super::PublicCorpusSidecarLane::FullV2Baseline,
                    fusion_scope: super::PublicCorpusSidecarFusionScope::BaselineUnfused,
                    study_configuration_sha256: "3".repeat(64),
                    fusion_configuration_sha256: None,
                    lane_configuration_sha256: "4".repeat(64),
                    calibration: None,
                    pair_calibration: None,
                    disposition: super::PublicCorpusSidecarDisposition::Baseline,
                    splits: vec![
                        super::unavailable_public_sidecar_split(EvaluationSplit::Development)
                            .expect("baseline shell"),
                    ],
                    gate: None,
                },
                super::PublicCorpusSidecarStudyVariant {
                    lane: super::PublicCorpusSidecarLane::FrameHaarL4,
                    fusion_scope: super::PublicCorpusSidecarFusionScope::BoundaryFusionV2,
                    study_configuration_sha256: "5".repeat(64),
                    fusion_configuration_sha256: Some("6".repeat(64)),
                    lane_configuration_sha256: "7".repeat(64),
                    calibration: None,
                    pair_calibration: None,
                    disposition: super::PublicCorpusSidecarDisposition::Rejected,
                    splits: vec![
                        super::unavailable_public_sidecar_split(EvaluationSplit::Development)
                            .expect("unavailable split"),
                    ],
                    gate: Some(super::PublicCorpusSidecarPromotionGate {
                        split: EvaluationSplit::Development,
                        candidate: super::PublicCorpusSidecarLane::FrameHaarL4,
                        baseline: super::PublicCorpusSidecarLane::FullV2Baseline,
                        relative_micro_der_improvement: None,
                        macro_jer_delta: None,
                        boundary_f1_delta: None,
                        comparable_frame_coverage: None,
                        pair_score_coverage: None,
                        same_speaker_pair_score_coverage: None,
                        different_speaker_pair_score_coverage: None,
                        pair_scored_recording_count: 0,
                        pair_roc_auc: None,
                        pair_brier_score: None,
                        pair_expected_calibration_error: None,
                        channel_same_speaker_dominance_rate: None,
                        mixed_auxiliary_same_speaker_dominance_rate: None,
                        exact_speaker_count_rate_delta: None,
                        mean_absolute_speaker_count_error_delta: None,
                        dominant_collapse_count_delta: None,
                        relative_rtf_regression: Some(0.30),
                        relative_rss_regression: Some(0.30),
                        paired_der_ci95_upper: None,
                        failures: vec![
                            super::PublicCorpusSidecarGateFailure::PerformanceRegression,
                        ],
                        passed: false,
                    }),
                },
            ],
            deterministic_accuracy_sha256: String::new(),
            result_sha256: String::new(),
        }
    }

    /// Gate-only aggregate input. Tests that need verifier coverage must use
    /// a runner-produced artifact; this helper intentionally does not model
    /// every per-recording pipeline and sampler invariant.
    fn sidecar_gate_split(
        split: EvaluationSplit,
        micro_der: f64,
        real_time_factor: f64,
        sampled_peak_rss_bytes: u64,
    ) -> super::PublicCorpusSidecarStudySplit {
        let mut pipeline =
            ablation_variant(AcousticFeatureAblation::FullV2, Some(micro_der), Some(0.30))
                .splits
                .into_iter()
                .find(|candidate| candidate.split == split)
                .expect("requested ablation split");
        pipeline.audio_duration_sec = 10.0;
        pipeline.wall_time_sec = real_time_factor * pipeline.audio_duration_sec;
        pipeline.real_time_factor = Some(real_time_factor);
        pipeline.sampled_peak_rss_bytes = sampled_peak_rss_bytes;
        let mut result =
            super::unavailable_public_sidecar_split(split).expect("sidecar split shell");
        result.pipeline = Some(pipeline.clone());
        result.fusion_executed = true;
        result.boundary = super::empty_sidecar_boundary_metrics(&pipeline);
        result.conditional_pairs.comparison_count = 200;
        result.conditional_pairs.same_speaker_count = 100;
        result.conditional_pairs.different_speaker_count = 100;
        result
            .conditional_pairs
            .mean_different_probability_given_same_speaker = Some(0.20);
        result
            .conditional_pairs
            .mean_different_probability_given_different_speaker = Some(0.80);
        result.conditional_pairs.roc_auc = Some(0.60);
        result.conditional_pairs.brier_score = Some(0.20);
        result.conditional_pairs.expected_calibration_error = Some(0.05);
        result.coverage.fusion_requested = true;
        result.coverage.evaluated_recording_count = 1;
        result.coverage.fusion_requested_recording_count = 1;
        result.coverage.fusion_executed_recording_count = 1;
        result.coverage.submitted_frame_count = 2;
        result.coverage.comparable_frame_count = 2;
        result.coverage.calibrated_signal_count = 2;
        result.coverage.consumed_probability_count = 1;
        result.coverage.comparable_frame_coverage = Some(1.0);
        result.coverage.component_comparison_count = 2;
        result.coverage.owner_available_frame_counts = [2, 0, 0];
        result.coverage.eligible_pair_count = 200;
        result.coverage.retained_pair_sample_count = 200;
        result.coverage.retained_same_speaker_pair_count = 100;
        result.coverage.retained_different_speaker_pair_count = 100;
        result.coverage.pair_selection_sha256 = Some("9".repeat(64));
        result.coverage.pair_score_coverage = Some(1.0);
        result.coverage.same_speaker_pair_score_coverage = Some(1.0);
        result.coverage.different_speaker_pair_score_coverage = Some(1.0);
        result.coverage.pair_scored_recording_count = 5;
        result.coverage.same_speaker_pair_recording_count = 5;
        result.coverage.different_speaker_pair_recording_count = 5;
        result.coverage.maximum_retained_pair_sample_count = 200;
        result.coverage.retained_pair_sample_capacity = 4_096;
        result.performance.audio_duration_sec = 10.0;
        result.performance.wall_time_sec = real_time_factor * 10.0;
        result.performance.real_time_factor = Some(real_time_factor);
        result.performance.sampled_peak_rss_bytes = sampled_peak_rss_bytes;
        result.paired_uncertainty = Some(super::PublicCorpusSidecarPairedUncertainty {
            paired_der_recording_count: 5,
            paired_jer_recording_count: 5,
            bootstrap_replicates: super::PUBLIC_SIDECAR_BOOTSTRAP_REPLICATES,
            bootstrap_seed_sha256: "8".repeat(64),
            mean_der_delta: Some(-0.01),
            der_delta_ci95_lower: Some(-0.02),
            der_delta_ci95_upper: Some(-0.001),
            mean_jer_delta: Some(0.0),
            jer_delta_ci95_lower: Some(-0.01),
            jer_delta_ci95_upper: Some(0.01),
        });
        result
    }

    fn sidecar_gate_scenario_evidence(
        stage: super::PublicCorpusEvaluationStage,
        candidate_real_time_factor: f64,
        candidate_peak_rss_bytes: u64,
    ) -> super::PublicCorpusSidecarStudyEvidence {
        let split = stage.selected_split();
        let mut evidence = sidecar_hash_fixture();
        evidence.evaluation_stage = stage;
        let mut baseline = sidecar_gate_split(split, 0.20, 0.10, 100);
        baseline.fusion_executed = false;
        baseline.coverage = super::unavailable_public_sidecar_split(split)
            .expect("baseline coverage shell")
            .coverage;
        baseline.conditional_pairs = super::SidecarPairAccumulator::new()
            .finish()
            .expect("baseline pairs");
        baseline.paired_uncertainty = None;
        let candidate = sidecar_gate_split(
            split,
            0.19,
            candidate_real_time_factor,
            candidate_peak_rss_bytes,
        );
        let gate = super::public_sidecar_promotion_gate(
            stage,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &candidate,
        );
        evidence.variants[0].splits = vec![baseline];
        evidence.variants[1].splits = vec![candidate];
        evidence.variants[1].gate = Some(gate.clone());
        match stage {
            super::PublicCorpusEvaluationStage::Development => {
                evidence.locked_development_result_sha256 = None;
                evidence.locked_development_accuracy_sha256 = None;
                evidence.selected_candidate_lane = gate
                    .passed
                    .then_some(super::PublicCorpusSidecarLane::FrameHaarL4);
                evidence.adopted_candidate_lane = None;
                evidence.variants[1].disposition = if gate.passed {
                    super::PublicCorpusSidecarDisposition::AdvanceToCertification
                } else {
                    super::PublicCorpusSidecarDisposition::Rejected
                };
            }
            super::PublicCorpusEvaluationStage::Certification => {
                evidence.locked_development_result_sha256 = Some("a".repeat(64));
                evidence.locked_development_accuracy_sha256 = Some("b".repeat(64));
                evidence.selected_candidate_lane =
                    Some(super::PublicCorpusSidecarLane::FrameHaarL4);
                evidence.adopted_candidate_lane = gate
                    .passed
                    .then_some(super::PublicCorpusSidecarLane::FrameHaarL4);
                evidence.variants[1].disposition = if gate.passed {
                    super::PublicCorpusSidecarDisposition::Adopted
                } else {
                    super::PublicCorpusSidecarDisposition::Rejected
                };
            }
        }
        evidence
    }

    fn valid_unavailable_sidecar_evidence() -> super::PublicCorpusSidecarStudyEvidence {
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
        let request = crate::model::DiarizationRequest {
            engine: crate::model::DiarizationEngine::Acoustic,
            speaker_count: crate::model::SpeakerCountRequest::Infer,
            ..crate::model::DiarizationRequest::default()
        };
        let request_sha256 = super::canonical_sha256(&request).expect("request hash");
        let protocol = super::public_sidecar_protocol(None, &request, request_sha256)
            .expect("sidecar protocol");
        let protocol_sha256 = super::canonical_sha256(&protocol).expect("protocol hash");
        let mut baseline_split =
            super::unavailable_public_sidecar_split(EvaluationSplit::Development)
                .expect("baseline split shell");
        let baseline_pipeline =
            ablation_variant(AcousticFeatureAblation::FullV2, Some(0.20), Some(0.30))
                .splits
                .into_iter()
                .find(|split| split.split == EvaluationSplit::Development)
                .expect("baseline pipeline");
        baseline_split.boundary = super::empty_sidecar_boundary_metrics(&baseline_pipeline);
        baseline_split.performance = super::PublicCorpusSidecarPerformance {
            audio_duration_sec: baseline_pipeline.audio_duration_sec,
            wall_time_sec: baseline_pipeline.wall_time_sec,
            real_time_factor: baseline_pipeline.real_time_factor,
            sampled_peak_rss_bytes: baseline_pipeline.sampled_peak_rss_bytes,
        };
        baseline_split.coverage.evaluated_recording_count = baseline_pipeline.recording_count;
        baseline_split.pipeline = Some(baseline_pipeline);

        let baseline_study_sha256 = crate::diarization::acoustic_sidecar_study_config_sha256(
            super::PublicCorpusSidecarLane::FullV2Baseline.study_config(),
        )
        .expect("baseline study hash");
        let baseline_lane_sha256 =
            super::canonical_sha256(&super::PublicCorpusSidecarLaneFingerprint {
                runner_version: super::PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION,
                lane: super::PublicCorpusSidecarLane::FullV2Baseline,
                fusion_scope: super::PublicCorpusSidecarFusionScope::BaselineUnfused,
                study_configuration_sha256: &baseline_study_sha256,
                fusion_configuration_sha256: None,
                pair_calibration_sha256: None,
                protocol_sha256: &protocol_sha256,
            })
            .expect("baseline lane hash");
        let mut variants = vec![super::PublicCorpusSidecarStudyVariant {
            lane: super::PublicCorpusSidecarLane::FullV2Baseline,
            fusion_scope: super::PublicCorpusSidecarFusionScope::BaselineUnfused,
            study_configuration_sha256: baseline_study_sha256,
            fusion_configuration_sha256: None,
            lane_configuration_sha256: baseline_lane_sha256,
            calibration: None,
            pair_calibration: None,
            disposition: super::PublicCorpusSidecarDisposition::Baseline,
            splits: vec![baseline_split.clone()],
            gate: None,
        }];
        for lane in super::PublicCorpusSidecarLane::ALL.into_iter().skip(1) {
            let split = super::unavailable_public_sidecar_split(EvaluationSplit::Development)
                .expect("candidate unavailable split");
            let gate = super::public_sidecar_promotion_gate(
                super::PublicCorpusEvaluationStage::Development,
                &protocol.gate_policy,
                &baseline_split,
                lane,
                &split,
            );
            let study_configuration_sha256 =
                crate::diarization::acoustic_sidecar_study_config_sha256(lane.study_config())
                    .expect("candidate study hash");
            let lane_configuration_sha256 =
                super::canonical_sha256(&super::PublicCorpusSidecarLaneFingerprint {
                    runner_version: super::PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION,
                    lane,
                    fusion_scope: super::PublicCorpusSidecarFusionScope::BoundaryFusionV2,
                    study_configuration_sha256: &study_configuration_sha256,
                    fusion_configuration_sha256: None,
                    pair_calibration_sha256: None,
                    protocol_sha256: &protocol_sha256,
                })
                .expect("candidate lane hash");
            variants.push(super::PublicCorpusSidecarStudyVariant {
                lane,
                fusion_scope: super::PublicCorpusSidecarFusionScope::BoundaryFusionV2,
                study_configuration_sha256,
                fusion_configuration_sha256: None,
                lane_configuration_sha256,
                calibration: None,
                pair_calibration: None,
                disposition: super::PublicCorpusSidecarDisposition::Rejected,
                splits: vec![split],
                gate: Some(gate),
            });
        }
        assert_eq!(
            super::apply_public_sidecar_development_selection(&mut variants),
            None
        );
        let scorer_config_sha256 = super::canonical_sha256(&scorer_config).expect("scorer hash");
        let mut evidence = super::PublicCorpusSidecarStudyEvidence {
            schema_version: super::PUBLIC_CORPUS_SIDECAR_STUDY_SCHEMA_VERSION.to_owned(),
            runner_version: super::PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION.to_owned(),
            scorer_version: crate::diarization::DIARIZATION_SCORER_VERSION.to_owned(),
            corpus_key: "aishell-4-openslr111-v1".to_owned(),
            source_version: "unavailable-fixture-v1".to_owned(),
            bundle_sha256: "1".repeat(64),
            descriptor_sha256: "2".repeat(64),
            scorer_config,
            scorer_config_sha256,
            evaluation_stage: super::PublicCorpusEvaluationStage::Development,
            locked_development_result_sha256: None,
            locked_development_accuracy_sha256: None,
            protocol,
            protocol_sha256,
            selected_candidate_lane: None,
            adopted_candidate_lane: None,
            variants,
            deterministic_accuracy_sha256: String::new(),
            result_sha256: String::new(),
        };
        evidence.deterministic_accuracy_sha256 =
            super::deterministic_sidecar_accuracy_sha256(&evidence).expect("accuracy hash");
        evidence.result_sha256 = super::canonical_sha256(&evidence).expect("result hash");
        evidence
    }

    fn rehash_sidecar_evidence(evidence: &mut super::PublicCorpusSidecarStudyEvidence) {
        evidence.deterministic_accuracy_sha256 =
            super::deterministic_sidecar_accuracy_sha256(evidence).expect("accuracy hash");
        evidence.result_sha256.clear();
        evidence.result_sha256 = super::canonical_sha256(evidence).expect("result hash");
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
            let project = private_tempdir("project");
            let input = private_tempdir("input");
            let output = private_tempdir("output");
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

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
    #[test]
    fn selected_split_materialization_never_opens_held_out_media() {
        let project = private_tempdir("project");
        let input = private_tempdir("input");
        let output = private_tempdir("output");
        let development_audio = input.path().join("development.wav");
        let development_annotation = input.path().join("development.rttm");
        write_wave(&development_audio, 16_000, 1, 16_000);
        std::fs::write(
            &development_annotation,
            "SPEAKER source-development-recording 1 0.000 0.500 <NA> <NA> source-speaker <NA> <NA>\n",
        )
        .expect("development RTTM");
        let descriptor_path = input.path().join("descriptor.json");
        std::fs::write(
            &descriptor_path,
            serde_json::to_vec_pretty(&json!({
                "schema_version": PUBLIC_CORPUS_INPUT_SCHEMA_VERSION,
                "corpus_key": "aishell-4-openslr111-v1",
                "source_version": "stage-selective-fixture-v1",
                "recordings": [
                    external_recording(
                        "development-recording",
                        "development",
                        "development.wav",
                        &sha256(&development_audio),
                        "development.rttm",
                        &sha256(&development_annotation),
                    ),
                    external_recording(
                        "held-out-recording",
                        "test",
                        "held-out-does-not-exist.wav",
                        &"0".repeat(64),
                        "held-out-does-not-exist.rttm",
                        &"1".repeat(64),
                    ),
                ]
            }))
            .expect("descriptor JSON"),
        )
        .expect("descriptor");
        let descriptor_sha256 = sha256(&descriptor_path);
        let bundle_path = output.path().join("development-bundle.json");

        let bundle = super::build_public_corpus_bundle_for_split_with_cancel(
            project.path(),
            input.path(),
            &descriptor_path,
            &bundle_path,
            "accept-aishell-4-cc-by-sa-4.0",
            Some(EvaluationSplit::Development),
            Some(&descriptor_sha256),
            || false,
        )
        .expect("development-only materialization");

        assert_eq!(bundle.references.len(), 1);
        assert_eq!(bundle.references[0].recording_id, "development-recording");
        assert!(bundle_path.is_file());
        assert!(!input.path().join("held-out-does-not-exist.wav").exists());
    }

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
    #[test]
    fn descriptor_metadata_rejects_held_out_traversal_before_selected_media_access() {
        let project = private_tempdir("project");
        let input = private_tempdir("input");
        let output = private_tempdir("output");
        let descriptor_path = input.path().join("descriptor.json");
        std::fs::write(
            &descriptor_path,
            serde_json::to_vec_pretty(&json!({
                "schema_version": PUBLIC_CORPUS_INPUT_SCHEMA_VERSION,
                "corpus_key": "aishell-4-openslr111-v1",
                "source_version": "metadata-preflight-fixture-v1",
                "recordings": [
                    external_recording(
                        "development-missing",
                        "development",
                        "development-missing.wav",
                        &"0".repeat(64),
                        "development-missing.rttm",
                        &"1".repeat(64),
                    ),
                    external_recording(
                        "held-out-traversal",
                        "test",
                        "../held-out.wav",
                        &"2".repeat(64),
                        "held-out.rttm",
                        &"3".repeat(64),
                    ),
                ]
            }))
            .expect("descriptor JSON"),
        )
        .expect("descriptor");
        let bundle_path = output.path().join("bundle.json");
        let error = super::build_public_corpus_bundle_for_split_with_cancel(
            project.path(),
            input.path(),
            &descriptor_path,
            &bundle_path,
            "accept-aishell-4-cc-by-sa-4.0",
            Some(EvaluationSplit::Development),
            None,
            || false,
        )
        .expect_err("held-out path traversal must fail metadata preflight");
        assert!(error.to_string().contains("relative_path"));
        assert!(!error.to_string().contains("input_file"));
        assert!(!bundle_path.exists());
    }

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
    #[test]
    fn descriptor_metadata_rejects_cross_split_identity_leakage_before_media_access() {
        let project = private_tempdir("project");
        let input = private_tempdir("input");
        let output = private_tempdir("output");
        let development = external_recording(
            "development-missing",
            "development",
            "development-missing.wav",
            &"0".repeat(64),
            "development-missing.rttm",
            &"1".repeat(64),
        );
        let mut held_out = external_recording(
            "held-out-missing",
            "test",
            "held-out-missing.wav",
            &"2".repeat(64),
            "held-out-missing.rttm",
            &"3".repeat(64),
        );
        held_out["origin_recording_id"] = development["origin_recording_id"].clone();
        let descriptor_path = input.path().join("descriptor.json");
        std::fs::write(
            &descriptor_path,
            serde_json::to_vec_pretty(&json!({
                "schema_version": PUBLIC_CORPUS_INPUT_SCHEMA_VERSION,
                "corpus_key": "aishell-4-openslr111-v1",
                "source_version": "metadata-leakage-fixture-v1",
                "recordings": [development, held_out]
            }))
            .expect("descriptor JSON"),
        )
        .expect("descriptor");
        let bundle_path = output.path().join("bundle.json");
        let error = super::build_public_corpus_bundle_for_split_with_cancel(
            project.path(),
            input.path(),
            &descriptor_path,
            &bundle_path,
            "accept-aishell-4-cc-by-sa-4.0",
            Some(EvaluationSplit::Development),
            None,
            || false,
        )
        .expect_err("cross-split origin reuse must fail metadata preflight");
        assert!(error.to_string().contains("split_leakage"));
        assert!(!error.to_string().contains("input_file"));
        assert!(!bundle_path.exists());
    }

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
    #[test]
    fn descriptor_digest_mismatch_fails_before_selected_media_access() {
        let project = private_tempdir("project");
        let input = private_tempdir("input");
        let output = private_tempdir("output");
        let descriptor_path = input.path().join("descriptor.json");
        std::fs::write(
            &descriptor_path,
            serde_json::to_vec_pretty(&json!({
                "schema_version": PUBLIC_CORPUS_INPUT_SCHEMA_VERSION,
                "corpus_key": "aishell-4-openslr111-v1",
                "source_version": "descriptor-race-fixture-v1",
                "recordings": [external_recording(
                    "development-missing",
                    "development",
                    "development-missing.wav",
                    &"0".repeat(64),
                    "development-missing.rttm",
                    &"1".repeat(64),
                )]
            }))
            .expect("descriptor JSON"),
        )
        .expect("descriptor");
        let bundle_path = output.path().join("bundle.json");
        let error = super::build_public_corpus_bundle_for_split_with_cancel(
            project.path(),
            input.path(),
            &descriptor_path,
            &bundle_path,
            "accept-aishell-4-cc-by-sa-4.0",
            Some(EvaluationSplit::Development),
            Some(&"f".repeat(64)),
            || false,
        )
        .expect_err("descriptor mismatch must precede media access");
        assert!(error.to_string().contains("sidecar_descriptor_changed"));
        assert!(!error.to_string().contains("input_file"));
        assert!(!bundle_path.exists());
    }

    #[test]
    fn sidecar_lane_order_and_configuration_identities_are_frozen() {
        let expected_ids = [
            "full_v2_baseline",
            "frame_haar_l4",
            "frame_d4_l4",
            "modulation",
            "frame_haar_l4_and_modulation",
            "frame_d4_l4_and_modulation",
            "trajectory_haar_l4",
            "trajectory_d4_l4",
            "scattering_first_order",
            "scattering_second_order",
            "scattering_first_and_second_order",
            "all_haar_l4",
            "all_d4_l4",
        ];
        assert_eq!(
            super::PublicCorpusSidecarLane::ALL.map(super::PublicCorpusSidecarLane::id),
            expected_ids
        );
        let configuration_hashes = super::PublicCorpusSidecarLane::ALL
            .into_iter()
            .map(|lane| {
                crate::diarization::acoustic_sidecar_study_config_sha256(lane.study_config())
                    .expect("lane configuration hash")
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(configuration_hashes.len(), expected_ids.len());

        let expected_configurations = [
            ("off", 0, "off", 0, "off"),
            ("haar", 4, "off", 0, "off"),
            ("daubechies_four_tap", 4, "off", 0, "off"),
            ("modulation", 0, "off", 0, "off"),
            ("haar_modulation", 4, "off", 0, "off"),
            ("daubechies_four_tap_modulation", 4, "off", 0, "off"),
            ("off", 0, "haar", 4, "off"),
            ("off", 0, "daubechies_four_tap", 4, "off"),
            ("off", 0, "off", 0, "first_order"),
            ("off", 0, "off", 0, "second_order"),
            ("off", 0, "off", 0, "first_and_second_order"),
            ("haar_modulation", 4, "haar", 4, "first_and_second_order"),
            (
                "daubechies_four_tap_modulation",
                4,
                "daubechies_four_tap",
                4,
                "first_and_second_order",
            ),
        ];
        let actual_configurations = super::PublicCorpusSidecarLane::ALL.map(|lane| {
            let config = lane.study_config();
            (
                config.mode.id(),
                config.frame_wavelet_levels,
                config.trajectory_wavelet_mode.id(),
                config.trajectory_wavelet_levels,
                config.scattering_mode.id(),
            )
        });
        assert_eq!(actual_configurations, expected_configurations);
    }

    #[test]
    fn sidecar_pair_bottom_k_matches_full_sort_and_samples_late_audio() {
        let audio_sha256 = [0x5a; 32];
        let mut sampler =
            super::SidecarPairBottomKSampler::new(super::PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING)
                .expect("bounded sampler");
        let mut lane_peer =
            super::SidecarPairBottomKSampler::new(super::PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING)
                .expect("peer sampler");
        let mut all_keys = Vec::new();
        let mut prefix_keys = Vec::new();
        for right_frame_index in 1_usize..=10_000 {
            let left_frame_index = right_frame_index - 1;
            let key = super::sidecar_pair_selection_key(
                &audio_sha256,
                left_frame_index,
                right_frame_index,
                1,
            )
            .expect("selection key");
            all_keys.push(key);
            if prefix_keys.len() < super::PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING {
                prefix_keys.push(key);
            }
            sampler
                .consider(super::SidecarPairSample {
                    key,
                    maximum_contrast: Some(0.25),
                    different_speaker: false,
                    channel_dominance_opportunity: false,
                    channel_dominance: false,
                    mixed_auxiliary_dominance_opportunity: false,
                    mixed_auxiliary_dominance: false,
                })
                .expect("sample candidate");
            lane_peer
                .consider(super::SidecarPairSample {
                    key,
                    maximum_contrast: (right_frame_index % 2 == 0).then_some(0.75),
                    different_speaker: true,
                    channel_dominance_opportunity: true,
                    channel_dominance: true,
                    mixed_auxiliary_dominance_opportunity: true,
                    mixed_auxiliary_dominance: true,
                })
                .expect("peer candidate");
        }
        let mut reverse_sampler =
            super::SidecarPairBottomKSampler::new(super::PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING)
                .expect("reverse sampler");
        for right_frame_index in (1_usize..=10_000).rev() {
            let key = super::sidecar_pair_selection_key(
                &audio_sha256,
                right_frame_index - 1,
                right_frame_index,
                1,
            )
            .expect("reverse selection key");
            reverse_sampler
                .consider(super::SidecarPairSample {
                    key,
                    maximum_contrast: Some(0.50),
                    different_speaker: false,
                    channel_dominance_opportunity: false,
                    channel_dominance: false,
                    mixed_auxiliary_dominance_opportunity: false,
                    mixed_auxiliary_dominance: false,
                })
                .expect("reverse candidate");
        }

        all_keys.sort_unstable();
        all_keys.truncate(super::PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING);
        prefix_keys.sort_unstable();
        let (eligible_count, retained_count, retained_capacity, selected) =
            sampler.finish().expect("selected samples");
        let (_, peer_retained_count, _, peer_selected) =
            lane_peer.finish().expect("peer selected samples");
        let (_, reverse_retained_count, _, reverse_selected) =
            reverse_sampler.finish().expect("reverse selected samples");
        let selected_keys = selected.iter().map(|sample| sample.key).collect::<Vec<_>>();
        let peer_keys = peer_selected
            .iter()
            .map(|sample| sample.key)
            .collect::<Vec<_>>();
        let reverse_keys = reverse_selected
            .iter()
            .map(|sample| sample.key)
            .collect::<Vec<_>>();

        assert_eq!(eligible_count, 10_000);
        assert_eq!(
            retained_count,
            super::PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING
        );
        assert_eq!(peer_retained_count, retained_count);
        assert_eq!(reverse_retained_count, retained_count);
        assert!(retained_capacity <= super::PUBLIC_SIDECAR_MAX_RETAINED_PAIR_SAMPLE_CAPACITY);
        assert!(
            selected_keys == all_keys,
            "bounded sampler must match an exact full-sort bottom-k"
        );
        assert!(
            peer_keys == selected_keys,
            "feature availability, lane score, and class label must not alter positions"
        );
        assert!(
            peer_selected
                .iter()
                .any(|sample| sample.maximum_contrast.is_none()),
            "the peer sample must exercise lane-specific score missingness"
        );
        assert!(
            reverse_keys == selected_keys,
            "input order must not alter positions"
        );
        assert!(
            selected_keys != prefix_keys,
            "bottom-k must not be prefix sampling"
        );
        assert!(
            selected_keys
                .iter()
                .any(|key| key.right_frame_index > 9_000),
            "the bounded sample must retain evidence from late audio"
        );
    }

    #[test]
    fn sidecar_pair_selection_digest_binds_labels_but_not_lane_scores() {
        fn digest(different_speaker: bool, maximum_contrast: Option<f64>) -> Vec<u8> {
            let normalized_pcm_sha256 = [0x6b; 32];
            let key = super::sidecar_pair_selection_key(&normalized_pcm_sha256, 10, 35, 25)
                .expect("selection key");
            let selected = [super::SidecarPairSample {
                key,
                maximum_contrast,
                different_speaker,
                channel_dominance_opportunity: false,
                channel_dominance: false,
                mixed_auxiliary_dominance_opportunity: false,
                mixed_auxiliary_dominance: false,
            }];
            let mut hasher = sha2::Sha256::new();
            hasher.update(super::PUBLIC_CORPUS_SIDECAR_PAIR_SELECTION_DIGEST_VERSION.as_bytes());
            super::update_sidecar_pair_selection_digest(
                &mut hasher,
                &normalized_pcm_sha256,
                &selected,
            )
            .expect("selection digest");
            hasher.finalize().to_vec()
        }

        assert_ne!(
            digest(false, Some(0.25)),
            digest(true, Some(0.25)),
            "a same/different reference-label swap must change the digest"
        );
        assert_eq!(
            digest(false, Some(0.25)),
            digest(false, None),
            "lane-specific score availability must not change the selected universe"
        );
    }

    #[test]
    fn sidecar_pair_selection_digest_binds_empty_recordings_and_their_order() {
        fn digest(recordings: &[[u8; 32]]) -> Vec<u8> {
            let mut hasher = sha2::Sha256::new();
            hasher.update(super::PUBLIC_CORPUS_SIDECAR_PAIR_SELECTION_DIGEST_VERSION.as_bytes());
            for normalized_pcm_sha256 in recordings {
                super::update_sidecar_pair_selection_digest(
                    &mut hasher,
                    normalized_pcm_sha256,
                    &[],
                )
                .expect("empty-recording selection digest");
            }
            hasher.finalize().to_vec()
        }

        let first = [0x17; 32];
        let second = [0x93; 32];
        assert_ne!(
            digest(&[first]),
            digest(&[second]),
            "a zero-pair recording's normalized PCM identity must remain bound"
        );
        assert_ne!(
            digest(&[first, second]),
            digest(&[second, first]),
            "record-framed zero-pair blocks must retain deterministic recording order"
        );
        assert_ne!(
            digest(&[first]),
            digest(&[first, second]),
            "adding an empty recording must change the selected-universe identity"
        );
    }

    #[test]
    fn sidecar_pair_ring_never_grows_past_its_initial_bound() {
        let maximum_lag = super::PUBLIC_SIDECAR_PAIR_LAGS_FRAMES
            [super::PUBLIC_SIDECAR_PAIR_LAGS_FRAMES.len() - 1];
        let requested_capacity = maximum_lag + 1;
        let mut ring = std::collections::VecDeque::with_capacity(requested_capacity);
        let initial_capacity = ring.capacity();
        let mut peak_len = 0;
        let mut peak_capacity = initial_capacity;

        for frame_index in 0_usize..10_000 {
            super::push_bounded_sidecar_ring_entry(
                &mut ring,
                frame_index,
                maximum_lag,
                frame_index,
                |entry| *entry,
            )
            .expect("bounded ring push");
            peak_len = peak_len.max(ring.len());
            peak_capacity = peak_capacity.max(ring.capacity());
            assert_eq!(ring.back().copied(), Some(frame_index));
            assert_eq!(
                ring.front().copied(),
                Some(frame_index.saturating_sub(maximum_lag))
            );
        }

        assert_eq!(peak_len, requested_capacity);
        assert_eq!(peak_capacity, initial_capacity);
        assert!(initial_capacity >= requested_capacity);
    }

    #[test]
    fn sidecar_reference_sweep_matches_overlap_ignore_and_boundary_semantics() {
        let reference = DiarizationReferenceDocument {
            schema_version: DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
            recording_id: "sidecar-reference-sweep".to_owned(),
            duration_ms: 2_000,
            turns: vec![
                EvaluationTurn::labeled(0, 300, "speaker-a"),
                EvaluationTurn::labeled(100, 400, "speaker-a"),
                EvaluationTurn::labeled(250, 500, "speaker-b"),
                EvaluationTurn::unknown(600, 700),
                EvaluationTurn::labeled(800, 900, "speaker-a"),
            ],
            ignored_regions: vec![
                EvaluationRegion {
                    start_ms: 50,
                    end_ms: 150,
                    reason_code: "first-ignore".to_owned(),
                },
                EvaluationRegion {
                    start_ms: 120,
                    end_ms: 200,
                    reason_code: "overlapping-ignore".to_owned(),
                },
                EvaluationRegion {
                    start_ms: 450,
                    end_ms: 460,
                    reason_code: "boundary-ignore".to_owned(),
                },
            ],
            speaker_hints: Vec::new(),
            words: Vec::new(),
        };
        let reference_changes = [500, 1_500];
        let mut sweep =
            super::SidecarReferenceSweep::new(&reference, &reference_changes).expect("sweep");
        for (timestamp_ms, expected_speaker, expected_boundary) in [
            (0, Some("speaker-a"), Some(false)),
            (50, None, None),
            (149, None, None),
            (150, None, None),
            (200, Some("speaker-a"), Some(false)),
            (249, Some("speaker-a"), Some(false)),
            (250, None, Some(true)),
            (300, None, Some(true)),
            (400, Some("speaker-b"), Some(true)),
            (450, None, None),
            (460, Some("speaker-b"), Some(true)),
            (500, None, Some(true)),
            (600, None, Some(true)),
            (700, None, Some(true)),
            (751, None, None),
            (800, Some("speaker-a"), Some(false)),
            (900, None, None),
            (1_250, None, Some(true)),
            (1_500, None, Some(true)),
            (1_751, None, None),
        ] {
            let label = sweep.label_at_ms(timestamp_ms).expect("ordered label");
            assert_eq!(label.speaker, expected_speaker, "speaker at {timestamp_ms}");
            assert_eq!(
                label.boundary, expected_boundary,
                "boundary at {timestamp_ms}"
            );
        }

        let mut decreasing =
            super::SidecarReferenceSweep::new(&reference, &reference_changes).expect("sweep");
        decreasing.label_at_ms(10).expect("first timestamp");
        assert!(decreasing.label_at_ms(9).is_err());

        let mut exhaustive =
            super::SidecarReferenceSweep::new(&reference, &reference_changes).expect("sweep");
        for timestamp_ms in 0..reference.duration_ms {
            let ignored = reference
                .ignored_regions
                .iter()
                .any(|region| region.start_ms <= timestamp_ms && timestamp_ms < region.end_ms);
            let mut expected_speaker = None;
            let mut speaker_conflict = false;
            for turn in reference
                .turns
                .iter()
                .filter(|turn| turn.start_ms <= timestamp_ms && timestamp_ms < turn.end_ms)
            {
                let Some(speaker) = turn.speaker.as_deref() else {
                    speaker_conflict = true;
                    continue;
                };
                if expected_speaker.is_some_and(|expected| expected != speaker) {
                    speaker_conflict = true;
                } else {
                    expected_speaker = Some(speaker);
                }
            }
            if speaker_conflict {
                expected_speaker = None;
            }
            let expected_boundary = if ignored {
                None
            } else if reference_changes.iter().any(|change| {
                change.abs_diff(timestamp_ms) <= super::PUBLIC_SIDECAR_BOUNDARY_COLLAR_MS
            }) {
                Some(true)
            } else {
                expected_speaker.map(|_| false)
            };
            let expected_speaker = (!ignored).then_some(expected_speaker).flatten();
            assert_eq!(
                exhaustive.label_at_ms(timestamp_ms).expect("ordered label"),
                super::SidecarReferenceFrameLabel {
                    speaker: expected_speaker,
                    boundary: expected_boundary,
                },
                "oracle mismatch at {timestamp_ms}",
            );
        }

        let mut unsorted_turns = reference.clone();
        unsorted_turns.turns.swap(0, 1);
        assert!(super::SidecarReferenceSweep::new(&unsorted_turns, &reference_changes).is_err());
        let mut unsorted_ignored = reference.clone();
        unsorted_ignored.ignored_regions.swap(0, 2);
        assert!(super::SidecarReferenceSweep::new(&unsorted_ignored, &reference_changes).is_err());
    }

    #[test]
    fn sidecar_bootstrap_is_fixed_cancellable_and_identity_aligned() {
        let baseline = (0_u8..5)
            .map(|index| super::SidecarRecordingAccuracy {
                recording_audio_sha256: [index; 32],
                reference_sha256: [index.wrapping_add(16); 32],
                der: Some(0.20 + f64::from(index) * 0.01),
                jer: Some(0.30 + f64::from(index) * 0.01),
            })
            .collect::<Vec<_>>();
        let candidate = (0_u8..5)
            .map(|index| super::SidecarRecordingAccuracy {
                recording_audio_sha256: [index; 32],
                reference_sha256: [index.wrapping_add(16); 32],
                der: Some(0.19 + f64::from(index) * 0.01),
                jer: Some(0.29 + f64::from(index) * 0.01),
            })
            .collect::<Vec<_>>();
        let first = super::paired_sidecar_uncertainty(
            &baseline,
            &candidate,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            EvaluationSplit::Development,
            &mut || false,
        )
        .expect("first fixed bootstrap");
        let repeated = super::paired_sidecar_uncertainty(
            &baseline,
            &candidate,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            EvaluationSplit::Development,
            &mut || false,
        )
        .expect("repeated fixed bootstrap");
        assert_eq!(first, repeated);

        let mut baseline_with_inert_row = baseline.clone();
        baseline_with_inert_row.push(super::SidecarRecordingAccuracy {
            recording_audio_sha256: [0xf0; 32],
            reference_sha256: [0xf1; 32],
            der: None,
            jer: None,
        });
        let mut candidate_with_inert_row = candidate.clone();
        candidate_with_inert_row.push(super::SidecarRecordingAccuracy {
            recording_audio_sha256: [0xf0; 32],
            reference_sha256: [0xf1; 32],
            der: None,
            jer: None,
        });
        let inert_row = super::paired_sidecar_uncertainty(
            &baseline_with_inert_row,
            &candidate_with_inert_row,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            EvaluationSplit::Development,
            &mut || false,
        )
        .expect("inert null row bootstrap");
        assert_eq!(first, inert_row);

        let distinct_lane = super::paired_sidecar_uncertainty(
            &baseline,
            &candidate,
            super::PublicCorpusSidecarLane::FrameD4L4,
            EvaluationSplit::Development,
            &mut || false,
        )
        .expect("distinct-lane bootstrap");
        assert_ne!(
            first.bootstrap_seed_sha256,
            distinct_lane.bootstrap_seed_sha256
        );

        let mut misaligned = candidate.clone();
        misaligned[0].recording_audio_sha256 = [0xff; 32];
        assert!(
            super::paired_sidecar_uncertainty(
                &baseline,
                &misaligned,
                super::PublicCorpusSidecarLane::FrameHaarL4,
                EvaluationSplit::Development,
                &mut || false,
            )
            .is_err()
        );
        let mut reference_misaligned = candidate.clone();
        reference_misaligned[0].reference_sha256 = [0xfe; 32];
        assert!(
            super::paired_sidecar_uncertainty(
                &baseline,
                &reference_misaligned,
                super::PublicCorpusSidecarLane::FrameHaarL4,
                EvaluationSplit::Development,
                &mut || false,
            )
            .is_err()
        );
        assert!(
            super::paired_sidecar_uncertainty(
                &baseline,
                &candidate,
                super::PublicCorpusSidecarLane::FrameHaarL4,
                EvaluationSplit::Development,
                &mut || true,
            )
            .is_err()
        );
    }

    #[test]
    fn sidecar_calibration_fit_requires_both_classes() {
        let mut negative_only = super::SidecarCalibrationFitHistogram::new();
        negative_only.push(0.25, false).expect("negative sample");
        assert!(
            super::fit_public_sidecar_calibration(&negative_only)
                .expect("negative-only fit")
                .is_none()
        );

        let mut positive_only = super::SidecarCalibrationFitHistogram::new();
        positive_only.push(0.75, true).expect("positive sample");
        assert!(
            super::fit_public_sidecar_calibration(&positive_only)
                .expect("positive-only fit")
                .is_none()
        );
    }

    #[test]
    fn sidecar_calibration_fit_preserves_empirical_class_priors() {
        let fitted_probability = |positive_count: usize| {
            let mut histogram = super::SidecarCalibrationFitHistogram::new();
            for index in 0..10 {
                histogram
                    .push(0.50, index < positive_count)
                    .expect("fit observation");
            }
            let calibration = super::fit_public_sidecar_calibration(&histogram)
                .expect("calibration fit")
                .expect("both-class calibration");
            let contrast = crate::diarization::AcousticSidecarOwnerContrast {
                owner_contrast: [0.50, 0.0, 0.0],
                owner_available: [true, false, false],
                comparable_components: super::PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS,
                component_comparisons: super::PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS,
            };
            super::public_sidecar_probability(&calibration, contrast)
                .expect("calibrated probability")
                .expect("comparable probability")
        };

        let rare_probability = fitted_probability(1);
        let common_probability = fitted_probability(9);
        assert!(
            rare_probability < 0.20,
            "a 10% positive prior must not be class-balanced toward 0.5: {rare_probability}"
        );
        assert!(
            common_probability > 0.80,
            "a 90% positive prior must remain representable: {common_probability}"
        );
    }

    #[test]
    fn sidecar_pair_calibration_has_an_independent_target_and_fail_closed_bounds() {
        let mut one_class = super::SidecarCalibrationFitHistogram::new();
        one_class.push(0.25, false).expect("same-speaker row");
        assert!(
            super::fit_public_sidecar_pair_calibration(&one_class)
                .expect("one-class pair fit")
                .is_none()
        );

        let mut histogram = super::SidecarCalibrationFitHistogram::new();
        histogram.push(0.10, false).expect("same-speaker row");
        histogram.push(0.90, true).expect("different-speaker row");
        let fitted = super::fit_public_sidecar_pair_calibration(&histogram)
            .expect("pair calibration fit")
            .expect("both-class pair calibration");
        super::validate_public_sidecar_pair_calibration(&fitted)
            .expect("fitter output must satisfy its public contract");
        assert_eq!(
            fitted.target_id,
            super::PUBLIC_CORPUS_SIDECAR_PAIR_PROBABILITY_TARGET_VERSION
        );

        let mut wrong_target = fitted.clone();
        wrong_target.target_id = "different-target".to_owned();
        wrong_target.calibration_sha256 =
            super::public_sidecar_pair_calibration_sha256(&wrong_target)
                .expect("wrong-target hash");
        assert!(super::validate_public_sidecar_pair_calibration(&wrong_target).is_err());

        let mut off_lattice = fitted.clone();
        off_lattice.logit_intercept = -0.10;
        off_lattice.calibration_sha256 =
            super::public_sidecar_pair_calibration_sha256(&off_lattice).expect("off-lattice hash");
        assert!(super::validate_public_sidecar_pair_calibration(&off_lattice).is_err());

        let mut impossible_counts = fitted;
        impossible_counts.fit_positive_count = impossible_counts.fit_observation_count;
        impossible_counts.calibration_sha256 =
            super::public_sidecar_pair_calibration_sha256(&impossible_counts)
                .expect("impossible-count hash");
        assert!(super::validate_public_sidecar_pair_calibration(&impossible_counts).is_err());
    }

    #[test]
    fn sidecar_reliability_brier_is_stable_near_probability_boundaries() {
        let mut probabilities = super::SidecarProbabilityAccumulator::new();
        probabilities
            .push(f64::from(1.0_f32 - f32::EPSILON), true)
            .expect("near-certain positive");
        let finished = probabilities.finish_reliability();
        assert!(finished.brier_score.is_some_and(|score| score >= 0.0));
        assert!(super::sidecar_reliability_is_valid(
            probabilities.observation_count,
            probabilities.positive_count,
            finished.brier_score,
            finished.expected_calibration_error,
            &finished.reliability,
        ));
    }

    #[test]
    fn sidecar_pair_gate_metrics_are_recomputed_from_retained_aggregates() {
        let mut accumulator = super::SidecarPairAccumulator::new();
        for _ in 0..8 {
            accumulator
                .push(f64::from(0.10_f32), false)
                .expect("same-speaker row");
            accumulator
                .push(f64::from(0.90_f32), true)
                .expect("different-speaker row");
        }
        let metrics = accumulator.finish().expect("pair metrics");
        assert!(super::sidecar_pair_metrics_are_valid(&metrics));

        let changed = |value: f64| if value == 0.0 { 0.125 } else { 0.0 };

        let mut auc = metrics.clone();
        auc.roc_auc = auc.roc_auc.map(changed);
        assert!(!super::sidecar_pair_metrics_are_valid(&auc));

        let mut brier = metrics.clone();
        brier.brier_score = brier.brier_score.map(changed);
        assert!(!super::sidecar_pair_metrics_are_valid(&brier));

        let mut ece = metrics.clone();
        ece.expected_calibration_error = ece.expected_calibration_error.map(changed);
        assert!(!super::sidecar_pair_metrics_are_valid(&ece));

        let mut subpicounit_ece = metrics.clone();
        let canonical_ece = subpicounit_ece
            .expected_calibration_error
            .expect("nonempty pair ECE");
        let noncanonical_ece = f64::from_bits(canonical_ece.to_bits() + 1);
        assert_ne!(
            noncanonical_ece,
            super::canonical_evidence_number(noncanonical_ece)
        );
        subpicounit_ece.expected_calibration_error = Some(noncanonical_ece);
        assert!(
            !super::sidecar_pair_metrics_are_valid(&subpicounit_ece),
            "sub-picounit noncanonical ECE tampering must be rejected exactly"
        );

        let mut same_mean = metrics.clone();
        same_mean.mean_different_probability_given_same_speaker = same_mean
            .mean_different_probability_given_same_speaker
            .map(changed);
        assert!(!super::sidecar_pair_metrics_are_valid(&same_mean));

        let mut reliability_sum = metrics.clone();
        let occupied = reliability_sum
            .reliability
            .iter_mut()
            .find(|bin| bin.observation_count > 0)
            .expect("occupied reliability bin");
        occupied.squared_error_sum =
            super::canonical_evidence_number(occupied.squared_error_sum + 0.01);
        assert!(!super::sidecar_pair_metrics_are_valid(&reliability_sum));

        let mut impossible_negative_subset = metrics.clone();
        let forged = &mut impossible_negative_subset.reliability[5];
        forged.observation_count = 100;
        forged.positive_count = 50;
        forged.probability_sum = 50.0;
        forged.positive_probability_sum = 29.5;
        forged.squared_probability_sum = 25.0;
        forged.positive_squared_probability_sum = 17.405;
        forged.squared_error_sum = 16.0;
        forged.mean_probability = Some(0.5);
        forged.empirical_frequency = Some(0.5);
        for (index, bin) in impossible_negative_subset
            .reliability
            .iter_mut()
            .enumerate()
        {
            if index != 5 {
                bin.observation_count = 0;
                bin.positive_count = 0;
                bin.probability_sum = 0.0;
                bin.positive_probability_sum = 0.0;
                bin.squared_probability_sum = 0.0;
                bin.positive_squared_probability_sum = 0.0;
                bin.squared_error_sum = 0.0;
                bin.mean_probability = None;
                bin.empirical_frequency = None;
            }
        }
        impossible_negative_subset.comparison_count = 100;
        impossible_negative_subset.same_speaker_count = 50;
        impossible_negative_subset.different_speaker_count = 50;
        impossible_negative_subset.mean_different_probability_given_same_speaker = Some(0.41);
        impossible_negative_subset.mean_different_probability_given_different_speaker = Some(0.59);
        impossible_negative_subset.roc_auc = Some(0.5);
        impossible_negative_subset.brier_score = Some(0.16);
        impossible_negative_subset.expected_calibration_error = Some(0.0);
        assert!(
            super::verified_sidecar_reliability(
                100,
                50,
                Some(0.16),
                Some(0.0),
                &impossible_negative_subset.reliability,
            )
            .is_none(),
            "negative-class probability mass below a bin boundary must be rejected"
        );

        assert!(
            !super::sidecar_probability_moments_are_feasible(
                0.5, 0.6, 100, 50, 50.5, 25.25, 26.0, 13.0, 25.5,
            ),
            "second moments above the bounded-interval secant must be rejected"
        );
        assert!(
            !super::sidecar_probability_moments_are_feasible(
                0.5, 0.51, 1, 0, 0.505, 0.0, 0.255_04, 0.0, 0.255_04,
            ),
            "one observation's first moment must determine its second moment"
        );

        let mut impossible_fine_bin = metrics.clone();
        for bin in &mut impossible_fine_bin.reliability {
            bin.observation_count = 0;
            bin.positive_count = 0;
            bin.probability_sum = 0.0;
            bin.positive_probability_sum = 0.0;
            bin.squared_probability_sum = 0.0;
            bin.positive_squared_probability_sum = 0.0;
            bin.squared_error_sum = 0.0;
            bin.mean_probability = None;
            bin.empirical_frequency = None;
        }
        for bin in &mut impossible_fine_bin.score_histogram {
            bin.same_speaker_count = 0;
            bin.different_speaker_count = 0;
            bin.probability_sum = 0.0;
            bin.different_speaker_probability_sum = 0.0;
            bin.squared_probability_sum = 0.0;
            bin.different_speaker_squared_probability_sum = 0.0;
            bin.squared_error_sum = 0.0;
        }
        let mean = super::canonical_evidence_number(0.505);
        let ece = super::canonical_evidence_number((mean - 0.5).abs());
        let brier = super::canonical_evidence_number(25.04 / 100.0);
        let coarse = &mut impossible_fine_bin.reliability[5];
        coarse.observation_count = 100;
        coarse.positive_count = 50;
        coarse.probability_sum = 50.5;
        coarse.positive_probability_sum = 25.25;
        coarse.squared_probability_sum = 25.54;
        coarse.positive_squared_probability_sum = 12.77;
        coarse.squared_error_sum = 25.04;
        coarse.mean_probability = Some(mean);
        coarse.empirical_frequency = Some(0.5);
        let fine = &mut impossible_fine_bin.score_histogram[50];
        fine.same_speaker_count = 50;
        fine.different_speaker_count = 50;
        fine.probability_sum = 50.5;
        fine.different_speaker_probability_sum = 25.25;
        fine.squared_probability_sum = 25.54;
        fine.different_speaker_squared_probability_sum = 12.77;
        fine.squared_error_sum = 25.04;
        impossible_fine_bin.comparison_count = 100;
        impossible_fine_bin.same_speaker_count = 50;
        impossible_fine_bin.different_speaker_count = 50;
        impossible_fine_bin.mean_different_probability_given_same_speaker = Some(mean);
        impossible_fine_bin.mean_different_probability_given_different_speaker = Some(mean);
        impossible_fine_bin.roc_auc = Some(0.5);
        impossible_fine_bin.brier_score = Some(brier);
        impossible_fine_bin.expected_calibration_error = Some(ece);
        assert!(
            super::verified_sidecar_reliability(
                100,
                50,
                Some(brier),
                Some(ece),
                &impossible_fine_bin.reliability,
            )
            .is_some(),
            "the wider coarse bin is deliberately feasible"
        );
        assert!(
            !super::sidecar_pair_metrics_are_valid(&impossible_fine_bin),
            "the tighter fine-bin secant must reject an impossible score histogram"
        );

        let mut contradictory_histogram = metrics.clone();
        let low_index = contradictory_histogram
            .score_histogram
            .iter()
            .position(|bin| bin.same_speaker_count > 0)
            .expect("same-speaker score bin");
        let high_index = contradictory_histogram
            .score_histogram
            .iter()
            .position(|bin| bin.different_speaker_count > 0)
            .expect("different-speaker score bin");
        let low = &mut contradictory_histogram.score_histogram[low_index];
        low.different_speaker_count = low.same_speaker_count;
        low.same_speaker_count = 0;
        low.different_speaker_probability_sum = low.probability_sum;
        low.different_speaker_squared_probability_sum = low.squared_probability_sum;
        low.squared_error_sum = super::canonical_evidence_number(
            low.squared_probability_sum - 2.0 * low.probability_sum
                + low.different_speaker_count as f64,
        );
        let high = &mut contradictory_histogram.score_histogram[high_index];
        high.same_speaker_count = high.different_speaker_count;
        high.different_speaker_count = 0;
        high.different_speaker_probability_sum = 0.0;
        high.different_speaker_squared_probability_sum = 0.0;
        high.squared_error_sum = high.squared_probability_sum;
        contradictory_histogram.roc_auc = Some(0.0);
        assert!(
            !super::sidecar_pair_metrics_are_valid(&contradictory_histogram),
            "fine score ordering must remain linked to coarse reliability evidence"
        );

        let mut score_histogram = metrics;
        score_histogram.score_histogram[0].same_speaker_count += 1;
        assert!(!super::sidecar_pair_metrics_are_valid(&score_histogram));
    }

    #[test]
    fn sidecar_pair_histogram_cannot_split_equal_f32_scores_across_bins() {
        let score = f64::from(0.55_f32);
        let score_index = super::sidecar_f32_probability_bin_index(
            score as f32,
            super::PUBLIC_SIDECAR_PAIR_SCORE_BINS,
        );
        let lower_index = score_index - 1;
        assert_eq!(
            lower_index
                / (super::PUBLIC_SIDECAR_PAIR_SCORE_BINS / super::PUBLIC_SIDECAR_RELIABILITY_BINS),
            score_index
                / (super::PUBLIC_SIDECAR_PAIR_SCORE_BINS / super::PUBLIC_SIDECAR_RELIABILITY_BINS),
            "the forgery must remain inside one linked reliability bin"
        );
        let (_, lower_highest) = super::sidecar_f32_probability_bin_support(
            lower_index,
            super::PUBLIC_SIDECAR_PAIR_SCORE_BINS,
        )
        .expect("lower score-bin support");
        assert!(score > lower_highest);

        let mut accumulator = super::SidecarPairAccumulator::new();
        accumulator.push(score, false).expect("same-speaker score");
        accumulator
            .push(score, true)
            .expect("different-speaker score");
        let mut forged = accumulator.finish().expect("equal-score metrics");
        assert_eq!(forged.roc_auc, Some(0.5));
        assert!(super::sidecar_pair_metrics_are_valid(&forged));

        let probability_sum = super::canonical_evidence_number(score);
        let squared_probability_sum = super::canonical_evidence_number(score * score);
        let positive_squared_error = super::canonical_evidence_number((score - 1.0).powi(2));
        let lower = &mut forged.score_histogram[lower_index];
        lower.same_speaker_count = 1;
        lower.probability_sum = probability_sum;
        lower.squared_probability_sum = squared_probability_sum;
        lower.squared_error_sum = squared_probability_sum;
        let source = &mut forged.score_histogram[score_index];
        source.same_speaker_count = 0;
        source.different_speaker_count = 1;
        source.probability_sum = probability_sum;
        source.different_speaker_probability_sum = probability_sum;
        source.squared_probability_sum = squared_probability_sum;
        source.different_speaker_squared_probability_sum = squared_probability_sum;
        source.squared_error_sum = positive_squared_error;
        forged.roc_auc = Some(1.0);

        assert!(
            !super::sidecar_pair_metrics_are_valid(&forged),
            "one representable score cannot be split to fabricate strict AUC ordering"
        );
    }

    #[test]
    fn sidecar_pair_histogram_rejects_single_cross_boundary_f32_forgery() {
        let (_, lower_highest) =
            super::sidecar_f32_probability_bin_support(0, super::PUBLIC_SIDECAR_PAIR_SCORE_BINS)
                .expect("lowest score-bin support");
        let (upper_lowest, _) =
            super::sidecar_f32_probability_bin_support(1, super::PUBLIC_SIDECAR_PAIR_SCORE_BINS)
                .expect("second score-bin support");
        assert!(upper_lowest > lower_highest);
        assert!(upper_lowest - lower_highest < 1e-9);

        let mut accumulator = super::SidecarPairAccumulator::new();
        for _ in 0..4 {
            accumulator
                .push(lower_highest, false)
                .expect("lower-bin same-speaker score");
        }
        accumulator
            .push(upper_lowest, false)
            .expect("upper-bin same-speaker score");
        accumulator
            .push(upper_lowest, true)
            .expect("upper-bin different-speaker score");
        let mut forged = accumulator.finish().expect("boundary-adjacent metrics");
        assert_eq!(forged.roc_auc, Some(0.9));
        assert!(super::sidecar_pair_metrics_are_valid(&forged));

        let moved_square = upper_lowest * upper_lowest;
        let lower = &mut forged.score_histogram[0];
        lower.same_speaker_count += 1;
        lower.probability_sum =
            super::canonical_evidence_number(lower.probability_sum + upper_lowest);
        lower.squared_probability_sum =
            super::canonical_evidence_number(lower.squared_probability_sum + moved_square);
        lower.squared_error_sum =
            super::canonical_evidence_number(lower.squared_error_sum + moved_square);
        let upper = &mut forged.score_histogram[1];
        upper.same_speaker_count -= 1;
        upper.probability_sum =
            super::canonical_evidence_number(upper.probability_sum - upper_lowest);
        upper.squared_probability_sum =
            super::canonical_evidence_number(upper.squared_probability_sum - moved_square);
        upper.squared_error_sum =
            super::canonical_evidence_number(upper.squared_error_sum - moved_square);
        forged.roc_auc = Some(1.0);

        assert!(
            !super::sidecar_pair_metrics_are_valid(&forged),
            "one adjacent-bin f32 score cannot be moved to fabricate perfect AUC"
        );
    }

    #[test]
    fn sidecar_pair_histogram_rejects_adjacent_bin_class_swap_auc_forgery() {
        let (_, lower_highest) =
            super::sidecar_f32_probability_bin_support(0, super::PUBLIC_SIDECAR_PAIR_SCORE_BINS)
                .expect("lowest score-bin support");
        let (upper_lowest, _) =
            super::sidecar_f32_probability_bin_support(1, super::PUBLIC_SIDECAR_PAIR_SCORE_BINS)
                .expect("second score-bin support");
        assert!(upper_lowest - lower_highest < 1e-9);

        let mut accumulator = super::SidecarPairAccumulator::new();
        accumulator
            .push(lower_highest, true)
            .expect("lower different-speaker score");
        accumulator
            .push(upper_lowest, false)
            .expect("upper same-speaker score");
        let mut forged = accumulator.finish().expect("boundary-adjacent metrics");
        assert_eq!(forged.roc_auc, Some(0.0));
        assert!(super::sidecar_pair_metrics_are_valid(&forged));

        let lower_square = lower_highest * lower_highest;
        let upper_square = upper_lowest * upper_lowest;
        let lower = &mut forged.score_histogram[0];
        lower.same_speaker_count = 1;
        lower.different_speaker_count = 0;
        lower.different_speaker_probability_sum = 0.0;
        lower.different_speaker_squared_probability_sum = 0.0;
        lower.squared_error_sum = super::canonical_evidence_number(lower_square);
        let upper = &mut forged.score_histogram[1];
        upper.same_speaker_count = 0;
        upper.different_speaker_count = 1;
        upper.different_speaker_probability_sum = super::canonical_evidence_number(upper_lowest);
        upper.different_speaker_squared_probability_sum =
            super::canonical_evidence_number(upper_square);
        upper.squared_error_sum = super::canonical_evidence_number((upper_lowest - 1.0).powi(2));
        forged.roc_auc = Some(1.0);

        assert!(
            !super::sidecar_pair_metrics_are_valid(&forged),
            "fine-bin class swaps cannot diverge from coarse reliability to forge AUC"
        );
    }

    #[test]
    fn sidecar_pair_histogram_accepts_large_genuine_f32_bin_aggregate() {
        let score = f64::from(1.0_f32 - f32::EPSILON);
        let mut awkward_accumulator = super::SidecarPairAccumulator::new();
        for _ in 0..4_098 {
            awkward_accumulator
                .push(score, false)
                .expect("genuine awkward-count score");
        }
        let awkward_metrics = awkward_accumulator
            .finish()
            .expect("awkward-count genuine aggregate");
        assert!(super::sidecar_pair_metrics_are_valid(&awkward_metrics));

        let mut accumulator = super::SidecarPairAccumulator::new();
        for index in 0_usize..40_960 {
            accumulator
                .push(score, index.is_multiple_of(2))
                .expect("genuine high-bin score");
        }
        let metrics = accumulator.finish().expect("large genuine aggregate");

        assert!(
            super::sidecar_pair_metrics_are_valid(&metrics),
            "sequential squared-sum roundoff must not reject producer output"
        );

        let (_, lowest_bin_upper) =
            super::sidecar_f32_probability_bin_support(0, super::PUBLIC_SIDECAR_PAIR_SCORE_BINS)
                .expect("lowest score-bin support");
        let mut low_bin_accumulator = super::SidecarPairAccumulator::new();
        for _ in 0..40_960 {
            low_bin_accumulator
                .push(lowest_bin_upper, true)
                .expect("genuine low-bin different-speaker score");
        }
        let low_bin_metrics = low_bin_accumulator
            .finish()
            .expect("large genuine low-bin aggregate");
        assert!(
            super::sidecar_pair_metrics_are_valid(&low_bin_metrics),
            "canonical moment identities must accept a valid low-bin aggregate"
        );

        let derived_negative_score = f64::from(0.99_f32);
        let mut derived_negative_accumulator = super::SidecarPairAccumulator::new();
        for index in 0..64 {
            derived_negative_accumulator
                .push(derived_negative_score, index < 6)
                .expect("genuine mixed-class high-bin score");
        }
        let derived_negative_metrics = derived_negative_accumulator
            .finish()
            .expect("genuine derived-negative aggregate");
        assert!(
            super::sidecar_pair_metrics_are_valid(&derived_negative_metrics),
            "derived negative-class moments must retain their rounding budget"
        );
    }

    #[test]
    fn sidecar_probability_moments_accept_large_rounded_class_difference() {
        let same_probability = f64::from(0.1_f32);
        let different_probability = f64::from(0.199_999_99_f32);
        let same_count = 1_u64;
        let different_count = (super::MAX_RECORDINGS as u64)
            .checked_mul(super::PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING as u64)
            .and_then(|count| count.checked_sub(same_count))
            .expect("maximum retained-pair count");
        let observation_count = same_count + different_count;
        let probability_sum = super::canonical_evidence_number(
            same_probability * same_count as f64 + different_probability * different_count as f64,
        );
        let positive_probability_sum =
            super::canonical_evidence_number(different_probability * different_count as f64);
        let squared_probability_sum = super::canonical_evidence_number(
            same_probability.powi(2) * same_count as f64
                + different_probability.powi(2) * different_count as f64,
        );
        let positive_squared_probability_sum = super::canonical_evidence_number(
            different_probability.powi(2) * different_count as f64,
        );
        let squared_error_sum = super::canonical_evidence_number(
            same_probability.powi(2) * same_count as f64
                + (different_probability - 1.0).powi(2) * different_count as f64,
        );
        let (support_lower, support_upper) =
            super::sidecar_f32_probability_bin_support(1, super::PUBLIC_SIDECAR_RELIABILITY_BINS)
                .expect("second reliability-bin support");

        assert_eq!(support_lower, same_probability);
        assert_eq!(support_upper, different_probability);
        assert!(super::sidecar_probability_moments_are_feasible(
            support_lower,
            support_upper,
            observation_count,
            different_count,
            probability_sum,
            positive_probability_sum,
            squared_probability_sum,
            positive_squared_probability_sum,
            squared_error_sum,
        ));
    }

    #[test]
    fn canonical_evidence_lattice_is_idempotent_at_f32_bin_endpoints() {
        let awkward = 4_097.999_511_480_331_f64;
        let canonical_awkward = super::canonical_evidence_number(awkward);
        assert_eq!(
            super::canonical_evidence_number(canonical_awkward),
            canonical_awkward
        );
        for bin_count in [
            super::PUBLIC_SIDECAR_RELIABILITY_BINS,
            super::PUBLIC_SIDECAR_PAIR_SCORE_BINS,
        ] {
            for index in 0..bin_count {
                let (lower, upper) = super::sidecar_f32_probability_bin_support(index, bin_count)
                    .expect("f32 score-bin support");
                for count in [1_u64, 2, 3, 4_098, 8_192, 40_960] {
                    for value in [
                        lower * count as f64,
                        upper * count as f64,
                        lower.powi(2) * count as f64,
                        upper.powi(2) * count as f64,
                    ] {
                        let canonical = super::canonical_evidence_number(value);
                        assert_eq!(
                            super::canonical_evidence_number(canonical),
                            canonical,
                            "bin_count={bin_count} index={index} count={count} value={value}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sidecar_calibration_rejects_values_outside_the_frozen_fit_lattice() {
        let mut histogram = super::SidecarCalibrationFitHistogram::new();
        histogram.push(0.10, false).expect("negative sample");
        histogram.push(0.90, true).expect("positive sample");
        let fitted = super::fit_public_sidecar_calibration(&histogram)
            .expect("calibration fit")
            .expect("both-class calibration");
        super::sidecar_evaluation_request(super::PublicCorpusSidecarLane::FrameHaarL4, &fitted)
            .expect("fitter output is accepted");

        let mut off_lattice = fitted.clone();
        off_lattice.logit_intercept = -0.10;
        off_lattice.calibration_sha256 =
            super::public_sidecar_calibration_sha256(&off_lattice).expect("tampered hash");
        assert!(
            super::sidecar_evaluation_request(
                super::PublicCorpusSidecarLane::FrameHaarL4,
                &off_lattice,
            )
            .expect_err("off-lattice intercept")
            .to_string()
            .contains("sidecar_calibration")
        );

        let mut wrong_minimum = fitted;
        wrong_minimum.minimum_comparable_components += 1;
        wrong_minimum.calibration_sha256 =
            super::public_sidecar_calibration_sha256(&wrong_minimum).expect("tampered hash");
        assert!(
            super::sidecar_evaluation_request(
                super::PublicCorpusSidecarLane::FrameHaarL4,
                &wrong_minimum,
            )
            .expect_err("wrong comparable-component minimum")
            .to_string()
            .contains("sidecar_calibration")
        );
    }

    #[test]
    fn sidecar_public_verifier_round_trips_and_rejects_rehashed_tampering() {
        let evidence = valid_unavailable_sidecar_evidence();
        super::verify_public_corpus_sidecar_study_evidence(&evidence)
            .expect("valid unavailable evidence");
        let encoded = serde_json::to_vec(&evidence).expect("sidecar JSON");
        assert_eq!(
            super::parse_public_corpus_sidecar_study_evidence(&encoded)
                .expect("verified round trip"),
            evidence
        );

        let mut noncanonical_unavailable = evidence.clone();
        noncanonical_unavailable.variants[1].splits[0]
            .coverage
            .retained_pair_sample_capacity = 1;
        rehash_sidecar_evidence(&mut noncanonical_unavailable);
        let error = super::verify_public_corpus_sidecar_study_evidence(&noncanonical_unavailable)
            .expect_err("rehashed noncanonical unavailable split");
        assert!(error.to_string().contains("sidecar_unavailable_shape"));

        let mut off_lattice = evidence;
        let mut calibration = super::PublicCorpusSidecarCalibration {
            fit_id: super::PUBLIC_CORPUS_SIDECAR_CALIBRATION_FIT_VERSION.to_owned(),
            logit_intercept: -0.10,
            contrast_weight: 1.0,
            minimum_comparable_components: super::PUBLIC_SIDECAR_MINIMUM_COMPARABLE_COMPONENTS,
            fit_observation_count: 2,
            fit_positive_count: 1,
            fit_brier_score: Some(0.20),
            calibration_sha256: String::new(),
        };
        calibration.calibration_sha256 =
            super::public_sidecar_calibration_sha256(&calibration).expect("calibration hash");
        off_lattice.variants[1].calibration = Some(calibration);
        rehash_sidecar_evidence(&mut off_lattice);
        let error = super::verify_public_corpus_sidecar_study_evidence(&off_lattice)
            .expect_err("rehashed off-lattice calibration");
        assert!(error.to_string().contains("sidecar_calibration"));
    }

    #[test]
    fn sidecar_public_verifier_rejects_rehashed_zero_duration_protocol() {
        let mut evidence = valid_unavailable_sidecar_evidence();
        evidence.protocol.maximum_recording_duration_ms = Some(0);
        evidence.protocol_sha256 =
            super::canonical_sha256(&evidence.protocol).expect("zero-duration protocol hash");
        for variant in &mut evidence.variants {
            variant.lane_configuration_sha256 =
                super::canonical_sha256(&super::PublicCorpusSidecarLaneFingerprint {
                    runner_version: super::PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION,
                    lane: variant.lane,
                    fusion_scope: variant.fusion_scope,
                    study_configuration_sha256: &variant.study_configuration_sha256,
                    fusion_configuration_sha256: variant.fusion_configuration_sha256.as_deref(),
                    pair_calibration_sha256: variant
                        .pair_calibration
                        .as_ref()
                        .map(|calibration| calibration.calibration_sha256.as_str()),
                    protocol_sha256: &evidence.protocol_sha256,
                })
                .expect("zero-duration lane hash");
        }
        rehash_sidecar_evidence(&mut evidence);

        let error = super::verify_public_corpus_sidecar_study_evidence(&evidence)
            .expect_err("rehashed zero-duration protocol");
        assert!(error.to_string().contains("sidecar_protocol"));
    }

    #[test]
    fn sidecar_runner_rejects_mismatched_stage_locks_before_path_access() {
        let root = private_tempdir("absent-sidecar-root");
        let absent_root = root.path().join("absent");
        let absent_descriptor = absent_root.join("descriptor.json");
        let absent_bundle = absent_root.join("bundle.json");
        let absent_evidence = absent_root.join("evidence.json");
        let absent_lock = absent_root.join("development-lock.json");

        let development_error = super::run_public_corpus_sidecar_study_with_cancel(
            super::PublicCorpusSidecarStudyRequest {
                project_root: &absent_root,
                input_root: &absent_root,
                descriptor_path: &absent_descriptor,
                bundle_output_path: &absent_bundle,
                evidence_output_path: &absent_evidence,
                license_acknowledgement_id: "unused-stage-lock-test",
                maximum_recording_duration_ms: None,
                evaluation_stage: super::PublicCorpusEvaluationStage::Development,
                locked_development_evidence_path: Some(&absent_lock),
            },
            || false,
        )
        .expect_err("development must reject a held-out lock");
        assert!(development_error.to_string().contains("sidecar_stage_lock"));

        let certification_error = super::run_public_corpus_sidecar_study_with_cancel(
            super::PublicCorpusSidecarStudyRequest {
                project_root: &absent_root,
                input_root: &absent_root,
                descriptor_path: &absent_descriptor,
                bundle_output_path: &absent_bundle,
                evidence_output_path: &absent_evidence,
                license_acknowledgement_id: "unused-stage-lock-test",
                maximum_recording_duration_ms: None,
                evaluation_stage: super::PublicCorpusEvaluationStage::Certification,
                locked_development_evidence_path: None,
            },
            || false,
        )
        .expect_err("certification must require a development lock");
        assert!(
            certification_error
                .to_string()
                .contains("sidecar_stage_lock")
        );

        assert!(!absent_descriptor.exists());
        assert!(!absent_bundle.exists());
        assert!(!absent_evidence.exists());
        assert!(!absent_lock.exists());
        assert!(!absent_root.exists());
    }

    #[test]
    fn sidecar_public_verifier_rejects_calibrated_unmaterialized_development_lane() {
        let mut evidence = valid_unavailable_sidecar_evidence();
        let mut histogram = super::SidecarCalibrationFitHistogram::new();
        histogram.push(0.10, false).expect("negative fit row");
        histogram.push(0.90, true).expect("positive fit row");
        let calibration = super::fit_public_sidecar_calibration(&histogram)
            .expect("calibration fit")
            .expect("both-class calibration");
        let lane = evidence.variants[1].lane;
        let study_configuration_sha256 = evidence.variants[1].study_configuration_sha256.clone();
        let fusion_configuration_sha256 =
            crate::diarization::acoustic_sidecar_fusion_configuration_sha256(
                super::sidecar_evaluation_request(lane, &calibration)
                    .expect("calibrated lane request"),
                evidence.protocol.detector_mode,
            )
            .expect("fusion configuration hash");
        let lane_configuration_sha256 =
            super::canonical_sha256(&super::PublicCorpusSidecarLaneFingerprint {
                runner_version: super::PUBLIC_CORPUS_SIDECAR_STUDY_RUNNER_VERSION,
                lane,
                fusion_scope: super::PublicCorpusSidecarFusionScope::BoundaryFusionV2,
                study_configuration_sha256: &study_configuration_sha256,
                fusion_configuration_sha256: Some(&fusion_configuration_sha256),
                pair_calibration_sha256: None,
                protocol_sha256: &evidence.protocol_sha256,
            })
            .expect("calibrated lane hash");
        evidence.variants[1].calibration = Some(calibration);
        evidence.variants[1].fusion_configuration_sha256 = Some(fusion_configuration_sha256);
        evidence.variants[1].lane_configuration_sha256 = lane_configuration_sha256;
        rehash_sidecar_evidence(&mut evidence);

        let error = super::verify_public_corpus_sidecar_study_evidence(&evidence)
            .expect_err("calibrated development lane without evaluation");
        assert!(error.to_string().contains("sidecar_stage_contract"));
    }

    #[test]
    fn sidecar_public_verifier_rejects_unmaterialized_certification_selection() {
        let mut evidence = valid_unavailable_sidecar_evidence();
        let split = EvaluationSplit::Test;
        let mut baseline_split =
            super::unavailable_public_sidecar_split(split).expect("baseline split shell");
        let baseline_pipeline =
            ablation_variant(AcousticFeatureAblation::FullV2, Some(0.20), Some(0.30))
                .splits
                .into_iter()
                .find(|candidate| candidate.split == split)
                .expect("test baseline pipeline");
        baseline_split.boundary = super::empty_sidecar_boundary_metrics(&baseline_pipeline);
        baseline_split.performance = super::PublicCorpusSidecarPerformance {
            audio_duration_sec: baseline_pipeline.audio_duration_sec,
            wall_time_sec: baseline_pipeline.wall_time_sec,
            real_time_factor: baseline_pipeline.real_time_factor,
            sampled_peak_rss_bytes: baseline_pipeline.sampled_peak_rss_bytes,
        };
        baseline_split.coverage.evaluated_recording_count = baseline_pipeline.recording_count;
        baseline_split.pipeline = Some(baseline_pipeline);

        evidence.evaluation_stage = super::PublicCorpusEvaluationStage::Certification;
        evidence.locked_development_result_sha256 = Some("a".repeat(64));
        evidence.locked_development_accuracy_sha256 = Some("b".repeat(64));
        evidence.selected_candidate_lane = Some(super::PublicCorpusSidecarLane::FrameHaarL4);
        evidence.adopted_candidate_lane = None;
        evidence.variants[0].splits = vec![baseline_split.clone()];
        for variant in evidence.variants.iter_mut().skip(1) {
            variant.disposition = super::PublicCorpusSidecarDisposition::Rejected;
            variant.splits = vec![
                super::unavailable_public_sidecar_split(split)
                    .expect("certification unavailable split"),
            ];
            variant.gate = None;
        }
        let selected_gate = super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Certification,
            &evidence.protocol.gate_policy,
            &baseline_split,
            evidence.variants[1].lane,
            &evidence.variants[1].splits[0],
        );
        evidence.variants[1].gate = Some(selected_gate);
        rehash_sidecar_evidence(&mut evidence);

        let error = super::verify_public_corpus_sidecar_study_evidence(&evidence)
            .expect_err("certification selected lane without calibration or evaluation");
        assert!(error.to_string().contains("sidecar_stage_contract"));
    }

    #[test]
    fn requested_but_unconsumed_sidecar_is_not_canonical_unavailable() {
        let evidence = sidecar_hash_fixture();
        let unavailable = super::unavailable_public_sidecar_split(EvaluationSplit::Development)
            .expect("canonical unavailable split");
        assert!(super::sidecar_split_is_valid(
            &unavailable,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));

        let mut requested = unavailable.clone();
        requested.coverage.fusion_requested = true;
        requested.coverage.evaluated_recording_count = 3;
        requested.coverage.fusion_requested_recording_count = 3;
        requested.coverage.pair_selection_sha256 = Some("9".repeat(64));
        requested.performance.audio_duration_sec = 30.0;
        requested.performance.real_time_factor = Some(0.0);
        assert_ne!(requested, unavailable);
        assert!(super::sidecar_split_is_valid(
            &requested,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));
        let mut impossible_empty_computation = requested.clone();
        impossible_empty_computation
            .operations
            .frame_wavelet_filter_tap_terms = 1;
        assert!(!super::sidecar_split_is_valid(
            &impossible_empty_computation,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));

        let mut comparable = requested.clone();
        comparable.coverage.submitted_frame_count = 1;
        comparable.coverage.comparable_frame_count = 1;
        comparable.coverage.calibrated_signal_count = 1;
        comparable.coverage.comparable_frame_coverage = Some(1.0);
        comparable.coverage.component_comparison_count = 1;
        comparable.coverage.owner_available_frame_counts = [1, 0, 0];
        assert!(!super::sidecar_split_is_valid(
            &comparable,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));
        comparable.coverage.maximum_retained_signal_count = 1;
        comparable.coverage.retained_signal_capacity = 401;
        comparable.operations.peak_retained_state_bytes_on_target = 1;
        assert!(!super::sidecar_split_is_valid(
            &comparable,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));
        comparable.coverage.retained_pair_sample_capacity =
            super::PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING as u64;
        comparable.operations.peak_scratch_buffer_payload_bytes = 1;
        // A constant frame still invokes the wavelet kernel and allocates its
        // fixed scratch payload, but legitimately performs zero nonzero taps.
        assert!(super::sidecar_split_is_valid(
            &comparable,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));

        let mut undersized_signal_capacity = comparable.clone();
        undersized_signal_capacity.coverage.retained_signal_capacity =
            super::PUBLIC_SIDECAR_MAX_RETAINED_SIGNALS - 1;
        assert!(!super::sidecar_split_is_valid(
            &undersized_signal_capacity,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));

        let mut reachable_signal_maximum = comparable.clone();
        reachable_signal_maximum.coverage.submitted_frame_count = 7;
        reachable_signal_maximum.coverage.comparable_frame_coverage = super::ratio(1, 7);
        reachable_signal_maximum
            .coverage
            .maximum_retained_signal_count = 3;
        assert!(super::sidecar_split_is_valid(
            &reachable_signal_maximum,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));
        let mut underreported_signal_maximum = reachable_signal_maximum;
        underreported_signal_maximum
            .coverage
            .maximum_retained_signal_count = 2;
        assert!(!super::sidecar_split_is_valid(
            &underreported_signal_maximum,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));

        let mut reachable_pair_maximum = comparable.clone();
        reachable_pair_maximum.coverage.submitted_frame_count = 79;
        reachable_pair_maximum.coverage.comparable_frame_coverage = super::ratio(1, 79);
        reachable_pair_maximum
            .coverage
            .maximum_retained_signal_count = 27;
        reachable_pair_maximum.coverage.eligible_pair_count = 4;
        reachable_pair_maximum.coverage.retained_pair_sample_count = 4;
        reachable_pair_maximum
            .coverage
            .retained_same_speaker_pair_count = 2;
        reachable_pair_maximum
            .coverage
            .retained_different_speaker_pair_count = 2;
        reachable_pair_maximum.coverage.pair_score_coverage = Some(1.0);
        reachable_pair_maximum
            .coverage
            .same_speaker_pair_score_coverage = Some(1.0);
        reachable_pair_maximum
            .coverage
            .different_speaker_pair_score_coverage = Some(1.0);
        reachable_pair_maximum.coverage.pair_scored_recording_count = 2;
        reachable_pair_maximum
            .coverage
            .same_speaker_pair_recording_count = 2;
        reachable_pair_maximum
            .coverage
            .different_speaker_pair_recording_count = 2;
        let mut retained_pairs = super::SidecarPairAccumulator::new();
        for different_speaker in [false, true, false, true] {
            retained_pairs
                .push(0.5, different_speaker)
                .expect("retained pair");
        }
        reachable_pair_maximum.conditional_pairs =
            retained_pairs.finish().expect("retained pair metrics");
        reachable_pair_maximum
            .coverage
            .maximum_retained_pair_sample_count = 2;
        assert!(super::sidecar_split_is_valid(
            &reachable_pair_maximum,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));
        let mut impossible_pair_recording_union = reachable_pair_maximum.clone();
        impossible_pair_recording_union
            .coverage
            .pair_scored_recording_count = 3;
        impossible_pair_recording_union
            .coverage
            .same_speaker_pair_recording_count = 1;
        impossible_pair_recording_union
            .coverage
            .different_speaker_pair_recording_count = 1;
        assert!(!super::sidecar_split_is_valid(
            &impossible_pair_recording_union,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));
        let mut underreported_pair_maximum = reachable_pair_maximum;
        underreported_pair_maximum
            .coverage
            .maximum_retained_pair_sample_count = 1;
        assert!(!super::sidecar_split_is_valid(
            &underreported_pair_maximum,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));

        let mut under_retained_pairs = comparable.clone();
        under_retained_pairs.coverage.eligible_pair_count = 2;
        under_retained_pairs
            .coverage
            .maximum_retained_pair_sample_count = 1;
        let mut retained_pair = super::SidecarPairAccumulator::new();
        retained_pair.push(0.5, false).expect("retained pair");
        under_retained_pairs.conditional_pairs =
            retained_pair.finish().expect("retained pair metrics");
        assert!(!super::sidecar_split_is_valid(
            &under_retained_pairs,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));

        let mut missing_owner = comparable.clone();
        missing_owner.coverage.owner_available_frame_counts = [0; 3];
        assert!(!super::sidecar_split_is_valid(
            &missing_owner,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));

        let mut uncalibrated_comparable = comparable.clone();
        uncalibrated_comparable.coverage.calibrated_signal_count = 0;
        assert!(!super::sidecar_split_is_valid(
            &uncalibrated_comparable,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));

        let mut too_many_boundary_probabilities = comparable;
        let mut probabilities = super::SidecarProbabilityAccumulator::new();
        probabilities
            .push(f64::from(0.20_f32), false)
            .expect("negative probability");
        probabilities
            .push(f64::from(0.80_f32), true)
            .expect("positive probability");
        let finished = probabilities.finish_reliability();
        too_many_boundary_probabilities
            .boundary
            .probability_observation_count = 2;
        too_many_boundary_probabilities
            .boundary
            .probability_positive_count = 1;
        too_many_boundary_probabilities.boundary.brier_score = finished.brier_score;
        too_many_boundary_probabilities
            .boundary
            .expected_calibration_error = finished.expected_calibration_error;
        too_many_boundary_probabilities.boundary.reliability = finished.reliability;
        assert!(!super::sidecar_split_is_valid(
            &too_many_boundary_probabilities,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));

        let mut wrong_count = requested.clone();
        wrong_count.coverage.evaluated_recording_count = 2;
        wrong_count.coverage.fusion_requested_recording_count = 2;
        assert!(!super::sidecar_split_is_valid(
            &wrong_count,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));

        let mut wrong_duration = requested;
        wrong_duration.performance.audio_duration_sec = 29.0;
        assert!(!super::sidecar_split_is_valid(
            &wrong_duration,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &evidence,
            3,
            30.0,
        ));
    }

    #[test]
    fn sidecar_split_rejects_more_executed_recordings_than_consumed_probabilities() {
        let mut pipeline =
            ablation_variant(AcousticFeatureAblation::FullV2, Some(0.20), Some(0.30))
                .splits
                .into_iter()
                .find(|split| split.split == EvaluationSplit::Development)
                .expect("development pipeline");
        pipeline.recording_count = 2;
        pipeline.speaker_count_confusion[0].recording_count = 2;
        pipeline.speaker_count_strata[0].recording_count = 2;
        pipeline.speaker_count_strata[0].unresolved_recording_count = 2;
        pipeline.speaker_count_duration_strata[0].recording_count = 2;
        pipeline.speaker_count_duration_strata[0].unresolved_recording_count = 2;
        pipeline.count_posterior_unavailable_count = 2;
        pipeline.count_unresolved_recording_count = 2;
        assert!(super::variant_splits_are_valid(
            std::slice::from_ref(&pipeline),
            10,
        ));

        let audio_duration_sec = pipeline.audio_duration_sec;
        let mut executed = super::unavailable_public_sidecar_split(EvaluationSplit::Development)
            .expect("executed split shell");
        executed.boundary = super::empty_sidecar_boundary_metrics(&pipeline);
        executed.performance = super::PublicCorpusSidecarPerformance {
            audio_duration_sec,
            wall_time_sec: pipeline.wall_time_sec,
            real_time_factor: pipeline.real_time_factor,
            sampled_peak_rss_bytes: pipeline.sampled_peak_rss_bytes,
        };
        executed.pipeline = Some(pipeline);
        executed.fusion_executed = true;
        executed.coverage.fusion_requested = true;
        executed.coverage.evaluated_recording_count = 2;
        executed.coverage.fusion_requested_recording_count = 2;
        executed.coverage.fusion_executed_recording_count = 2;
        executed.coverage.submitted_frame_count = 4;
        executed.coverage.comparable_frame_count = 2;
        executed.coverage.calibrated_signal_count = 2;
        executed.coverage.consumed_probability_count = 2;
        executed.coverage.comparable_frame_coverage = super::ratio(2, 4);
        executed.coverage.component_comparison_count = 2;
        executed.coverage.owner_available_frame_counts = [0, 0, 2];
        executed.coverage.pair_selection_sha256 = Some("9".repeat(64));
        executed.coverage.maximum_retained_signal_count = 2;
        executed.coverage.retained_signal_capacity = super::PUBLIC_SIDECAR_MAX_RETAINED_SIGNALS;
        executed.coverage.retained_pair_sample_capacity =
            super::PUBLIC_SIDECAR_MAX_PAIRS_PER_RECORDING as u64;
        executed.operations.frame_wavelet_filter_tap_terms = 1;
        executed.operations.peak_scratch_buffer_payload_bytes = 1;
        executed.operations.peak_retained_state_bytes_on_target = 1;
        let aligned_accuracy = [
            super::SidecarRecordingAccuracy {
                recording_audio_sha256: [1; 32],
                reference_sha256: [11; 32],
                der: None,
                jer: None,
            },
            super::SidecarRecordingAccuracy {
                recording_audio_sha256: [2; 32],
                reference_sha256: [12; 32],
                der: None,
                jer: None,
            },
        ];
        executed.paired_uncertainty = Some(
            super::paired_sidecar_uncertainty(
                &aligned_accuracy,
                &aligned_accuracy,
                super::PublicCorpusSidecarLane::FrameHaarL4,
                EvaluationSplit::Development,
                &mut || false,
            )
            .expect("zero-count paired uncertainty"),
        );
        assert!(super::sidecar_split_is_valid(
            &executed,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &sidecar_hash_fixture(),
            2,
            audio_duration_sec,
        ));

        let mut impossible = executed;
        impossible.coverage.consumed_probability_count = 1;
        assert!(!super::sidecar_split_is_valid(
            &impossible,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &sidecar_hash_fixture(),
            2,
            audio_duration_sec,
        ));
    }

    #[test]
    fn sidecar_pair_gate_rejects_trivial_support_and_class_specific_missingness() {
        let evidence = sidecar_hash_fixture();
        let policy = &evidence.protocol.gate_policy;
        let baseline = sidecar_gate_split(EvaluationSplit::Development, 0.20, 0.10, 100);

        let mut trivial = sidecar_gate_split(EvaluationSplit::Development, 0.18, 0.12, 120);
        trivial.conditional_pairs.comparison_count = 2;
        trivial.conditional_pairs.same_speaker_count = 1;
        trivial.conditional_pairs.different_speaker_count = 1;
        trivial.coverage.eligible_pair_count = 2;
        trivial.coverage.retained_pair_sample_count = 2;
        trivial.coverage.retained_same_speaker_pair_count = 1;
        trivial.coverage.retained_different_speaker_pair_count = 1;
        trivial.coverage.pair_score_coverage = Some(1.0);
        trivial.coverage.same_speaker_pair_score_coverage = Some(1.0);
        trivial.coverage.different_speaker_pair_score_coverage = Some(1.0);
        trivial.coverage.pair_scored_recording_count = 1;
        trivial.coverage.same_speaker_pair_recording_count = 1;
        trivial.coverage.different_speaker_pair_recording_count = 1;
        let trivial_gate = super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            policy,
            &baseline,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &trivial,
        );
        assert!(
            trivial_gate
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::InsufficientPairSupport)
        );
        assert!(
            !trivial_gate
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::InsufficientPairCoverage)
        );

        let mut class_sparse = sidecar_gate_split(EvaluationSplit::Development, 0.18, 0.12, 120);
        class_sparse.conditional_pairs.comparison_count = 300;
        class_sparse.conditional_pairs.same_speaker_count = 200;
        class_sparse.conditional_pairs.different_speaker_count = 100;
        class_sparse.coverage.eligible_pair_count = 1_100;
        class_sparse.coverage.retained_pair_sample_count = 1_100;
        class_sparse.coverage.retained_same_speaker_pair_count = 1_000;
        class_sparse.coverage.retained_different_speaker_pair_count = 100;
        class_sparse.coverage.pair_score_coverage = Some(300.0 / 1_100.0);
        class_sparse.coverage.same_speaker_pair_score_coverage = Some(0.20);
        class_sparse.coverage.different_speaker_pair_score_coverage = Some(1.0);
        let sparse_gate = super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            policy,
            &baseline,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &class_sparse,
        );
        assert!(
            sparse_gate
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::InsufficientPairCoverage),
            "overall coverage must not conceal a sparse same-speaker class"
        );
    }

    #[test]
    fn sidecar_boundary_fusion_survives_missing_pair_calibration() {
        let evidence = sidecar_hash_fixture();
        let baseline = sidecar_gate_split(EvaluationSplit::Development, 0.20, 0.10, 100);
        let mut boundary_only = sidecar_gate_split(EvaluationSplit::Development, 0.18, 0.12, 120);
        boundary_only.conditional_pairs = super::SidecarPairAccumulator::new()
            .finish()
            .expect("empty conditional metrics");
        boundary_only.coverage.eligible_pair_count = 100;
        boundary_only.coverage.retained_pair_sample_count = 100;
        boundary_only.coverage.retained_same_speaker_pair_count = 100;
        boundary_only.coverage.retained_different_speaker_pair_count = 0;
        boundary_only.coverage.pair_score_coverage = Some(0.0);
        boundary_only.coverage.same_speaker_pair_score_coverage = Some(0.0);
        boundary_only.coverage.different_speaker_pair_score_coverage = None;
        boundary_only.coverage.pair_scored_recording_count = 0;
        boundary_only.coverage.same_speaker_pair_recording_count = 0;
        boundary_only
            .coverage
            .different_speaker_pair_recording_count = 0;
        let gate = super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &boundary_only,
        );
        assert!(boundary_only.fusion_executed);
        assert!(boundary_only.pipeline.is_some());
        assert!(
            !gate
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::FusionNotExecuted)
        );
        assert!(
            !gate
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::MissingAccuracy)
        );
        assert!(
            gate.failures
                .contains(&super::PublicCorpusSidecarGateFailure::MissingConditionalPairs)
        );
    }

    #[test]
    fn sidecar_auxiliary_dominance_gate_is_lane_appropriate() {
        let evidence = sidecar_hash_fixture();
        let baseline = sidecar_gate_split(EvaluationSplit::Development, 0.20, 0.10, 100);
        let pure_wavelet = sidecar_gate_split(EvaluationSplit::Development, 0.18, 0.12, 120);
        let pure_wavelet_gate = super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &pure_wavelet,
        );
        assert!(
            !pure_wavelet_gate
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::AuxiliaryConfound),
            "a pure MixedAuxiliary lane has no Voice owner to compare"
        );

        let missing_support_gate = super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::Modulation,
            &pure_wavelet,
        );
        assert!(
            missing_support_gate
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::AuxiliaryConfound)
        );

        let candidate_with_dominance_count = |dominance_count: usize| {
            let mut candidate = sidecar_gate_split(EvaluationSplit::Development, 0.18, 0.12, 120);
            let mut diagnostics = super::SidecarAuxiliaryDominanceAccumulator::default();
            for index in 0..100 {
                diagnostics
                    .push(false, true, index < dominance_count)
                    .expect("same-speaker dominance row");
            }
            candidate.coverage.channel_dominance = diagnostics.finish();
            candidate
        };
        let admitted = candidate_with_dominance_count(50);
        let admitted_gate = super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::Modulation,
            &admitted,
        );
        assert!(
            !admitted_gate
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::AuxiliaryConfound)
        );
        let combined_gate = super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::FrameHaarL4AndModulation,
            &admitted,
        );
        assert!(
            combined_gate
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::AuxiliaryConfound),
            "channel support must not conceal missing MixedAuxiliary support"
        );

        let mut one_opportunity = sidecar_gate_split(EvaluationSplit::Development, 0.18, 0.12, 120);
        let mut one_diagnostic = super::SidecarAuxiliaryDominanceAccumulator::default();
        one_diagnostic
            .push(false, true, false)
            .expect("single same-speaker opportunity");
        one_opportunity.coverage.channel_dominance = one_diagnostic.finish();
        let one_opportunity_gate = super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::Modulation,
            &one_opportunity,
        );
        assert!(
            one_opportunity_gate
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::AuxiliaryConfound),
            "a single opportunity is not substantial confound evidence"
        );

        let rejected = candidate_with_dominance_count(51);
        let rejected_gate = super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::Modulation,
            &rejected,
        );
        assert!(
            rejected_gate
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::AuxiliaryConfound)
        );
    }

    #[test]
    fn sidecar_accuracy_hash_excludes_only_host_resource_diagnostics() {
        let passing_development = sidecar_gate_scenario_evidence(
            super::PublicCorpusEvaluationStage::Development,
            0.12,
            120,
        );
        let mut failing_development = sidecar_gate_scenario_evidence(
            super::PublicCorpusEvaluationStage::Development,
            0.20,
            200,
        );
        assert!(
            passing_development.variants[1]
                .gate
                .as_ref()
                .expect("passing gate")
                .passed
        );
        assert_eq!(
            failing_development.variants[1]
                .gate
                .as_ref()
                .expect("failing gate")
                .failures,
            vec![super::PublicCorpusSidecarGateFailure::PerformanceRegression]
        );
        assert_eq!(
            passing_development.selected_candidate_lane,
            Some(super::PublicCorpusSidecarLane::FrameHaarL4)
        );
        assert_eq!(failing_development.selected_candidate_lane, None);
        failing_development.variants[1].splits[0]
            .coverage
            .retained_signal_capacity = 1_024;
        failing_development.variants[1].splits[0]
            .coverage
            .retained_pair_sample_capacity = 8_192;
        failing_development.variants[1].splits[0]
            .operations
            .peak_retained_state_bytes_on_target = 65_536;
        assert_eq!(
            super::deterministic_sidecar_accuracy_sha256(&failing_development)
                .expect("performance-failing accuracy hash"),
            super::deterministic_sidecar_accuracy_sha256(&passing_development)
                .expect("performance-passing accuracy hash")
        );
        assert_ne!(
            super::canonical_sha256(&failing_development).expect("failing result identity"),
            super::canonical_sha256(&passing_development).expect("passing result identity")
        );

        let passing_certification = sidecar_gate_scenario_evidence(
            super::PublicCorpusEvaluationStage::Certification,
            0.12,
            120,
        );
        let mut failing_certification = sidecar_gate_scenario_evidence(
            super::PublicCorpusEvaluationStage::Certification,
            0.20,
            200,
        );
        failing_certification.locked_development_result_sha256 = Some("c".repeat(64));
        assert_eq!(
            passing_certification.adopted_candidate_lane,
            Some(super::PublicCorpusSidecarLane::FrameHaarL4)
        );
        assert_eq!(failing_certification.adopted_candidate_lane, None);
        assert_eq!(
            super::deterministic_sidecar_accuracy_sha256(&failing_certification)
                .expect("rejected certification accuracy hash"),
            super::deterministic_sidecar_accuracy_sha256(&passing_certification)
                .expect("adopted certification accuracy hash")
        );

        let mut changed_operation_count = passing_development.clone();
        changed_operation_count.variants[1].splits[0]
            .operations
            .frame_wavelet_filter_tap_terms = 1;
        assert_ne!(
            super::deterministic_sidecar_accuracy_sha256(&changed_operation_count)
                .expect("operation-sensitive accuracy hash"),
            super::deterministic_sidecar_accuracy_sha256(&passing_development)
                .expect("unchanged operation hash")
        );
    }

    #[test]
    fn sidecar_accuracy_winner_has_no_operational_runner_up_fallback() {
        let mut evidence = sidecar_hash_fixture();
        let baseline = sidecar_gate_split(EvaluationSplit::Development, 0.20, 0.10, 100);
        let top_accuracy = sidecar_gate_split(EvaluationSplit::Development, 0.18, 0.20, 200);
        let runner_up = sidecar_gate_split(EvaluationSplit::Development, 0.19, 0.12, 120);
        evidence.variants[0].splits = vec![baseline.clone()];
        evidence.variants[1].splits = vec![top_accuracy.clone()];
        evidence.variants[1].gate = Some(super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &top_accuracy,
        ));
        let mut runner_up_variant = evidence.variants[1].clone();
        runner_up_variant.lane = super::PublicCorpusSidecarLane::FrameD4L4;
        runner_up_variant.splits = vec![runner_up.clone()];
        runner_up_variant.gate = Some(super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::FrameD4L4,
            &runner_up,
        ));
        evidence.variants.push(runner_up_variant);

        assert_eq!(
            super::sidecar_accuracy_ranked_variant_index(&evidence.variants),
            Some(1)
        );
        assert_eq!(
            super::apply_public_sidecar_development_selection(&mut evidence.variants),
            None,
            "a performance-failing accuracy winner must not fall through"
        );
        assert_eq!(
            evidence.variants[1].disposition,
            super::PublicCorpusSidecarDisposition::Rejected
        );
        assert_eq!(
            evidence.variants[2].disposition,
            super::PublicCorpusSidecarDisposition::Rejected
        );
        assert!(
            evidence.variants[2]
                .gate
                .as_ref()
                .expect("runner-up gate")
                .failures
                .contains(&super::PublicCorpusSidecarGateFailure::NotSelectedByRanking)
        );

        let passing_top = sidecar_gate_split(EvaluationSplit::Development, 0.18, 0.12, 120);
        evidence.variants[1].splits = vec![passing_top.clone()];
        evidence.variants[1].gate = Some(super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::FrameHaarL4,
            &passing_top,
        ));
        evidence.variants[2].gate = Some(super::public_sidecar_promotion_gate(
            super::PublicCorpusEvaluationStage::Development,
            &evidence.protocol.gate_policy,
            &baseline,
            super::PublicCorpusSidecarLane::FrameD4L4,
            &runner_up,
        ));
        assert_eq!(
            super::apply_public_sidecar_development_selection(&mut evidence.variants),
            Some(super::PublicCorpusSidecarLane::FrameHaarL4)
        );
        assert_eq!(
            evidence.variants[1].disposition,
            super::PublicCorpusSidecarDisposition::AdvanceToCertification
        );
    }

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
    #[test]
    fn build_requires_exact_license_acknowledgement() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        let error = fixture.build("yes").expect_err("missing acknowledgement");
        assert!(error.to_string().contains("license_acknowledgement"));
        assert!(!fixture.output_path.exists());
    }

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
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

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
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

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
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

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
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

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
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

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
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

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
    #[test]
    fn output_must_remain_outside_project_and_inputs() {
        let fixture = Fixture::new("aishell-4-openslr111-v1", "aishell-fixture", "development");
        for (label, unsafe_output) in [
            ("project output", fixture.project.path().join("bundle.json")),
            ("input output", fixture.input.path().join("bundle.json")),
        ] {
            let error = build_public_corpus_bundle(
                fixture.project.path(),
                fixture.input.path(),
                &fixture.descriptor_path,
                &unsafe_output,
                "accept-aishell-4-cc-by-sa-4.0",
            )
            .expect_err(label);
            assert!(error.to_string().contains("output_overlap"));
            assert!(!unsafe_output.exists());
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    #[test]
    fn public_artifact_output_fails_closed_on_unsupported_platforms() {
        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let output_path = output_parent.join("artifact.json");
        let error = match build_public_corpus_bundle(
            &project,
            &input,
            &input.join("descriptor-must-not-be-read.json"),
            &output_path,
            "unused-license-acknowledgement",
        ) {
            Ok(_) => panic!("unsupported public artifact platform was accepted"),
            Err(error) => error,
        };

        assert!(matches!(error, crate::FwError::Unsupported(_)));
        assert!(error.to_string().contains("output_platform"));
        assert!(!output_path.exists());
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn output_parent_swap_is_rejected_before_payload_creation() {
        use std::os::unix::fs::symlink;

        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        let moved_output_parent = root.path().join("moved-output");
        let redirect = root.path().join("redirect");
        for directory in [&project, &input, &output_parent, &redirect] {
            create_private_directory(directory);
        }
        let canonical_project = project.canonicalize().expect("canonical project");
        let canonical_input = input.canonicalize().expect("canonical input");
        let output_path = output_parent.join("artifact.json");
        let canonical_output_parent =
            validate_new_output(&canonical_project, &canonical_input, &output_path)
                .expect("validated output");

        std::fs::rename(&output_parent, &moved_output_parent).expect("move output parent");
        symlink(&redirect, &output_parent).expect("replace output parent with redirect");
        let error = write_new_json(
            &output_path,
            &canonical_output_parent,
            &json!({"sensitive": "must-not-be-written"}),
            "fixture",
            &mut || false,
        )
        .expect_err("redirected output parent");

        assert!(error.to_string().contains("output_parent_changed"));
        assert!(!redirect.join("artifact.json").exists());
        assert!(!moved_output_parent.join("artifact.json").exists());
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn output_parent_symlink_back_to_held_inode_is_rejected() {
        use std::os::unix::fs::symlink;

        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        let moved_output_parent = root.path().join("moved-output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let canonical_project = project.canonicalize().expect("canonical project");
        let canonical_input = input.canonicalize().expect("canonical input");
        let output_path = output_parent.join("artifact.json");
        let validated = validate_new_output(&canonical_project, &canonical_input, &output_path)
            .expect("validated output");

        std::fs::rename(&output_parent, &moved_output_parent).expect("move output parent");
        symlink(&moved_output_parent, &output_parent).expect("symlink back to held parent");
        let error = write_new_json(
            &output_path,
            &validated,
            &json!({"sensitive": "must-not-be-written"}),
            "fixture",
            &mut || false,
        )
        .expect_err("symlink spelling of held output parent");

        assert!(error.to_string().contains("output_parent_changed"));
        assert!(!output_path.exists());
        assert!(!moved_output_parent.join("artifact.json").exists());
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn output_parent_same_path_replacement_is_identity_rejected() {
        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        let moved_output_parent = root.path().join("moved-output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let canonical_project = project.canonicalize().expect("canonical project");
        let canonical_input = input.canonicalize().expect("canonical input");
        let output_path = output_parent.join("artifact.json");
        let canonical_output_parent =
            validate_new_output(&canonical_project, &canonical_input, &output_path)
                .expect("validated output");

        std::fs::rename(&output_parent, &moved_output_parent).expect("move output parent");
        create_private_directory(&output_parent);
        let error = write_new_json(
            &output_path,
            &canonical_output_parent,
            &json!({"sensitive": "must-not-be-written"}),
            "fixture",
            &mut || false,
        )
        .expect_err("replacement output parent");

        assert!(error.to_string().contains("output_parent_changed"));
        assert!(!output_path.exists());
        assert!(!moved_output_parent.join("artifact.json").exists());
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn output_parent_must_be_owner_only_mutable() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        std::fs::set_permissions(&output_parent, std::fs::Permissions::from_mode(0o777))
            .expect("world-writable output parent");
        let output_path = output_parent.join("artifact.json");

        let error = match validate_new_output(
            &project.canonicalize().expect("canonical project"),
            &input.canonicalize().expect("canonical input"),
            &output_path,
        ) {
            Ok(_) => panic!("world-writable output parent was accepted"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("output_parent_permissions"));
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn terminal_output_parent_symlink_is_rejected() {
        use std::os::unix::fs::symlink;

        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let real_output_parent = root.path().join("real-output");
        let output_parent = root.path().join("output-link");
        for directory in [&project, &input, &real_output_parent] {
            create_private_directory(directory);
        }
        symlink(&real_output_parent, &output_parent).expect("output parent symlink");

        let error = match validate_new_output(
            &project.canonicalize().expect("canonical project"),
            &input.canonicalize().expect("canonical input"),
            &output_parent.join("artifact.json"),
        ) {
            Ok(_) => panic!("terminal output parent symlink was accepted"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("output_parent"));
        assert!(!real_output_parent.join("artifact.json").exists());
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn dangling_output_symlink_is_rejected_before_evaluation() {
        use std::os::unix::fs::symlink;

        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let output_path = output_parent.join("artifact.json");
        symlink("missing-target", &output_path).expect("dangling output symlink");

        let error = match validate_new_output(
            &project.canonicalize().expect("canonical project"),
            &input.canonicalize().expect("canonical input"),
            &output_path,
        ) {
            Ok(_) => panic!("dangling output entry was accepted"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("output_exists"));
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    #[test]
    fn fifo_output_leaf_is_rejected_without_opening_it() {
        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let output_path = output_parent.join("artifact.json");
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            use rustix::fs::{Mode, mkfifoat};

            let output_directory =
                std::fs::File::open(&output_parent).expect("output directory handle");
            mkfifoat(&output_directory, "artifact.json", Mode::RUSR | Mode::WUSR)
                .expect("fixture FIFO");
        }
        #[cfg(target_os = "macos")]
        {
            // rustix intentionally does not expose mkfifoat on Apple targets.
            // The system mkfifo utility creates the same private test fixture
            // without introducing unsafe code into this memory-safe crate.
            let status = std::process::Command::new("/usr/bin/mkfifo")
                .args(["-m", "600"])
                .arg(&output_path)
                .status()
                .expect("run system mkfifo for fixture");
            assert!(status.success(), "system mkfifo failed: {status}");
        }

        let error = match validate_new_output(
            &project.canonicalize().expect("canonical project"),
            &input.canonicalize().expect("canonical input"),
            &output_path,
        ) {
            Ok(_) => panic!("FIFO output entry was accepted"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("output_exists"));
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn uppercase_output_file_name_is_rejected_before_creation() {
        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let canonical_project = project.canonicalize().expect("canonical project");
        let canonical_input = input.canonicalize().expect("canonical input");
        let upper_path = output_parent.join("Bundle.json");
        let lower_path = output_parent.join("bundle.json");
        std::fs::write(&lower_path, b"existing\n").expect("existing lowercase output");
        let error = match validate_new_output(&canonical_project, &canonical_input, &upper_path) {
            Ok(_) => panic!("upper-case output name was accepted"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("output_path"));
        assert_eq!(
            std::fs::read(&lower_path).expect("existing lowercase output"),
            b"existing\n"
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn lowercase_output_rejects_existing_ascii_case_fold_sibling() {
        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let canonical_project = project.canonicalize().expect("canonical project");
        let canonical_input = input.canonicalize().expect("canonical input");
        let upper_path = output_parent.join("Bundle.json");
        let lower_path = output_parent.join("bundle.json");
        std::fs::write(&upper_path, b"existing\n").expect("existing uppercase sibling");

        let error = match validate_new_output(&canonical_project, &canonical_input, &lower_path) {
            Ok(_) => panic!("ASCII-case-fold sibling was accepted"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("output_exists"));
        assert_eq!(
            std::fs::read(&upper_path).expect("existing uppercase sibling"),
            b"existing\n"
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn non_ascii_output_file_name_is_rejected_before_creation() {
        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let output_path = output_parent.join("évidence.json");
        let error = match validate_new_output(
            &project.canonicalize().expect("canonical project"),
            &input.canonicalize().expect("canonical input"),
            &output_path,
        ) {
            Ok(_) => panic!("non-ASCII output name was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("output_path"));
        assert!(!output_path.exists());
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn output_file_name_requires_safe_ascii_and_lowercase_json_suffix() {
        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let project = project.canonicalize().expect("canonical project");
        let input = input.canonicalize().expect("canonical input");

        for file_name in ["artifact name.json", "artifact.JSON", "artifact.txt"] {
            let output_path = output_parent.join(file_name);
            let error = match validate_new_output(&project, &input, &output_path) {
                Ok(_) => panic!("unsafe output name was accepted: {file_name}"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("output_path"));
            assert!(!output_path.exists());
        }
    }

    #[cfg(target_vendor = "apple")]
    #[test]
    fn apfs_data_firmlink_alias_cannot_publish_into_the_checkout() {
        use std::os::unix::fs::MetadataExt as _;

        let project = Path::new(env!("CARGO_MANIFEST_DIR"))
            .canonicalize()
            .expect("canonical project");
        let Ok(relative_project) = project.strip_prefix("/") else {
            return;
        };
        let alias = Path::new("/System/Volumes/Data").join(relative_project);
        let (Ok(project_metadata), Ok(alias_metadata)) = (project.metadata(), alias.metadata())
        else {
            return;
        };
        if project_metadata.dev() != alias_metadata.dev()
            || project_metadata.ino() != alias_metadata.ino()
        {
            return;
        }
        assert!(super::paths_overlap(&project, &alias));
        if project_metadata.uid() != rustix::process::geteuid().as_raw()
            || project_metadata.mode() & 0o022 != 0
        {
            return;
        }

        let input = private_tempdir("input");
        let output_path = alias.join(format!(
            "firmlink-overlap-{}.json",
            uuid::Uuid::new_v4().simple()
        ));
        let error = match validate_new_output(&project, input.path(), &output_path) {
            Ok(_) => panic!("firmlink alias into the checkout was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("output_overlap"));
        assert!(!output_path.exists());
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn failed_output_serialization_never_publishes_partial_json() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let output_path = output_parent.join("artifact.json");
        let validated = validate_new_output(
            &project.canonicalize().expect("canonical project"),
            &input.canonicalize().expect("canonical input"),
            &output_path,
        )
        .expect("validated output");

        let error = write_new_json(
            &output_path,
            &validated,
            &AlwaysFailsSerialization,
            "fixture",
            &mut || false,
        )
        .expect_err("serialization failure");

        assert!(error.to_string().contains("output_write"));
        assert!(!output_path.exists());
        let staging = std::fs::read_dir(&output_parent)
            .expect("staging directory")
            .map(|entry| entry.expect("staging entry"))
            .collect::<Vec<_>>();
        assert_eq!(staging.len(), 1);
        assert!(
            staging[0]
                .file_name()
                .to_string_lossy()
                .starts_with(".franken-whisper-output-")
        );
        let metadata = staging[0].metadata().expect("staging metadata");
        assert_eq!(metadata.len(), 0);
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
        assert!(
            std::fs::read(staging[0].path())
                .expect("empty marker")
                .is_empty()
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn output_serialization_cancellation_never_publishes_partial_json() {
        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let output_path = output_parent.join("artifact.json");
        let validated = validate_new_output(
            &project.canonicalize().expect("canonical project"),
            &input.canonicalize().expect("canonical input"),
            &output_path,
        )
        .expect("validated output");
        let value = json!({"payload": "x".repeat(200_000)});
        let mut cancellation_checks = 0_usize;

        let error = write_new_json(&output_path, &validated, &value, "fixture", &mut || {
            cancellation_checks += 1;
            cancellation_checks >= 2
        })
        .expect_err("serialization cancellation");

        assert!(matches!(error, crate::FwError::Cancelled(_)));
        assert!(!output_path.exists());
        let staging = std::fs::read_dir(&output_parent)
            .expect("staging directory")
            .map(|entry| entry.expect("staging entry"))
            .collect::<Vec<_>>();
        assert_eq!(staging.len(), 1);
        assert_eq!(staging[0].metadata().expect("staging metadata").len(), 0);
        assert!(cancellation_checks >= 2);
        assert!(
            std::fs::read(staging[0].path())
                .expect("empty marker")
                .is_empty()
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn no_clobber_publication_preserves_racing_final_output() {
        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let output_path = output_parent.join("artifact.json");
        let validated = validate_new_output(
            &project.canonicalize().expect("canonical project"),
            &input.canonicalize().expect("canonical input"),
            &output_path,
        )
        .expect("validated output");
        let staged = super::stage_new_json(
            &output_path,
            &validated,
            &json!({"sensitive": "must-not-win-the-race"}),
            "fixture",
            &mut || false,
        )
        .expect("staged output");
        std::fs::write(&output_path, b"external-winner\n").expect("racing final output");

        let error = super::publish_staged_json(staged, "fixture")
            .expect_err("no-clobber publication failure");

        assert!(error.to_string().contains("output_create"));
        assert_eq!(
            std::fs::read(&output_path).expect("racing final bytes"),
            b"external-winner\n"
        );
        let staging = std::fs::read_dir(&output_parent)
            .expect("staging directory")
            .map(|entry| entry.expect("staging entry"))
            .filter(|entry| entry.path() != output_path)
            .collect::<Vec<_>>();
        assert_eq!(staging.len(), 1);
        assert_eq!(staging[0].metadata().expect("staging metadata").len(), 0);
        assert!(
            std::fs::read(staging[0].path())
                .expect("empty marker")
                .is_empty()
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn ambiguous_rename_state_never_truncates_final_staged_inode() {
        let root = private_tempdir("root");
        let project = root.path().join("project");
        let input = root.path().join("input");
        let output_parent = root.path().join("output");
        for directory in [&project, &input, &output_parent] {
            create_private_directory(directory);
        }
        let output_path = output_parent.join("artifact.json");
        let validated = validate_new_output(
            &project.canonicalize().expect("canonical project"),
            &input.canonicalize().expect("canonical input"),
            &output_path,
        )
        .expect("validated output");
        let staged = super::stage_new_json(
            &output_path,
            &validated,
            &json!({"payload": "must-remain-intact"}),
            "fixture",
            &mut || false,
        )
        .expect("staged output");
        let staging_path = output_parent.join(&staged.staging_name);
        std::fs::hard_link(&staging_path, &output_path).expect("same-inode final race");

        let error =
            super::publish_staged_json(staged, "fixture").expect_err("ambiguous publication state");

        assert!(error.to_string().contains("output_commit_uncertain"));
        let final_bytes = std::fs::read(&output_path).expect("final bytes");
        assert!(final_bytes.ends_with(b"\n"));
        assert!(
            String::from_utf8(final_bytes)
                .expect("UTF-8 JSON")
                .contains("must-remain-intact")
        );
        assert!(staging_path.metadata().expect("staging marker").len() > 0);
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

    #[test]
    fn count_merge_frontier_extracts_reference_boundary_and_rejects_bad_traces() {
        let steps = [
            super::AcousticCountMergeStepEvidence {
                remaining_clusters: 5,
                same_speaker_probability: 0.9,
            },
            super::AcousticCountMergeStepEvidence {
                remaining_clusters: 4,
                same_speaker_probability: 0.8,
            },
            super::AcousticCountMergeStepEvidence {
                remaining_clusters: 3,
                same_speaker_probability: 0.2,
            },
        ];
        let observation = super::count_merge_frontier_observation(&steps, 4)
            .expect("valid merge trace")
            .expect("non-empty merge trace");
        assert!(
            observation
                .0
                .is_some_and(|value| (value - f64::from(0.8_f32)).abs() < f64::EPSILON)
        );
        assert!(
            observation
                .1
                .is_some_and(|value| (value - f64::from(0.2_f32)).abs() < f64::EPSILON)
        );
        assert_eq!(
            super::count_merge_frontier_observation(&[], 4).expect("empty trace"),
            None
        );

        let mut invalid = steps;
        invalid[1].remaining_clusters = 3;
        assert!(super::count_merge_frontier_observation(&invalid, 4).is_err());
        invalid = steps;
        invalid[1].same_speaker_probability = f32::NAN;
        assert!(super::count_merge_frontier_observation(&invalid, 4).is_err());
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
            speaker_count_posterior_map_confusion: Vec::new(),
            speaker_count_merge_frontier: super::PublicSpeakerCountMergeFrontier {
                recording_count: 0,
                to_reference_count_observation_count: 0,
                mean_probability_to_reference_count: None,
                below_reference_count_observation_count: 0,
                mean_probability_below_reference_count: None,
                paired_frontier_observation_count: 0,
                mean_probability_margin: None,
                correctly_ordered_frontier_count: 0,
                correctly_ordered_frontier_rate: None,
            },
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

    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        ignore = "public artifact publication is unsupported on this platform"
    )]
    #[test]
    fn public_ablation_runner_completes_with_path_free_aggregate_evidence() {
        let project = private_tempdir("project");
        let input = private_tempdir("input");
        let output = private_tempdir("output");
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
