//! Speaker-diarization contracts, scoring, and native acoustic implementation.
//!
//! The first responsibility of this module is to make diarization claims
//! measurable. Speaker identifiers are arbitrary cluster names, so comparing
//! them index-for-index is incorrect: `SPEAKER_00` in one run may be
//! `SPEAKER_01` in another while the underlying partition is identical.
//! [`score_diarization`] therefore finds the maximum-overlap speaker mapping
//! before computing diarization error rate (DER) and Jaccard error rate (JER).
//!
//! The waveform-aware implementation is built behind this contract in the
//! follow-on `bd-odj7` beads. See `docs/acoustic_diarization_contract.md`.

use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::conformance::CANONICAL_PROJECTION_EPSILON_SEC;
use crate::error::{FwError, FwResult};
use crate::model::{
    DiarizationEngine, DiarizationFallbackPolicy, DiarizationFallbackStatus, DiarizationReport,
    DiarizationRequest, DiarizationTurn, KnownSpeakerInterval, KnownSpeakerPolicy,
    MAX_SPEAKER_COUNT, SpeakerAttributionQuery, SpeakerAttributionQueryReason,
    SpeakerCountCalibrationStatus, SpeakerCountEstimate, SpeakerCountEvidenceLane,
    SpeakerCountLaneEvidence, SpeakerCountLaneUnavailableReason, SpeakerCountOutcome,
    SpeakerCountOutcomeReason, SpeakerCountOutcomeStatus, SpeakerCountPosteriorBin,
    SpeakerCountRange, SpeakerCountRequest, SpeakerCountResourceSummary, SpeakerEvidenceReason,
    SpeakerEvidenceSummary, SpeakerHintDisposition, SpeakerHintEvidenceSummary,
    SpeakerProfileSummary, TranscriptionSegment,
};

/// Stable identifier for the native acoustic diarization contract.
pub const ACOUSTIC_DIARIZATION_CONTRACT_VERSION: &str = "acoustic-diarization-v2";
/// Frozen implementation identity for retained diarization evaluation results.
pub const DIARIZATION_SCORER_VERSION: &str = "diarization-scorer-v4";
/// Schema identity for reference annotations accepted by the frozen scorer.
pub const DIARIZATION_REFERENCE_SCHEMA_VERSION: &str = "diarization-reference-v2";
/// Schema identity for system hypotheses accepted by the frozen scorer.
pub const DIARIZATION_HYPOTHESIS_SCHEMA_VERSION: &str = "diarization-hypothesis-v2";
/// Schema identity for scorer configuration.
pub const DIARIZATION_SCORER_CONFIG_SCHEMA_VERSION: &str = "diarization-scorer-config-v2";
/// Schema identity for retained scorer results.
pub const DIARIZATION_SCORE_RESULT_SCHEMA_VERSION: &str = "diarization-score-result-v2";
/// Schema identity for privacy-safe corpus manifests.
pub const DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION: &str = "diarization-corpus-manifest-v1";
/// Schema identity for split-leakage audit results.
pub const DIARIZATION_LEAKAGE_AUDIT_SCHEMA_VERSION: &str = "diarization-leakage-audit-v1";

const SCORE_EPSILON_SEC: f64 = 1e-9;
const SCORE_HASH_HEX_LEN: usize = 64;
const MAX_OPAQUE_ID_LEN: usize = 160;

/// One speech turn used by the permutation-invariant scorer.
///
/// The presence of a turn means speech is active. A missing `speaker` is an
/// honest unknown-speaker hypothesis; reference turns must have a non-empty
/// speaker label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoringTurn {
    pub start_sec: f64,
    pub end_sec: f64,
    pub speaker: Option<String>,
    pub overlap_suspected: bool,
}

impl ScoringTurn {
    /// Construct a labeled scoring turn.
    #[must_use]
    pub fn labeled(start_sec: f64, end_sec: f64, speaker: impl Into<String>) -> Self {
        Self {
            start_sec,
            end_sec,
            speaker: Some(speaker.into()),
            overlap_suspected: false,
        }
    }

    /// Construct an unknown-speaker hypothesis turn.
    #[must_use]
    pub const fn unknown(start_sec: f64, end_sec: f64) -> Self {
        Self {
            start_sec,
            end_sec,
            speaker: None,
            overlap_suspected: false,
        }
    }
}

/// Permutation-invariant diarization metrics in seconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizationScore {
    /// Total reference speaker-time used as the DER denominator.
    pub reference_speaker_time_sec: f64,
    pub missed_speech_sec: f64,
    pub false_alarm_sec: f64,
    pub speaker_confusion_sec: f64,
    /// `(missed + false_alarm + confusion) / reference_speaker_time`.
    pub der: Option<f64>,
    /// Mean per-reference-speaker Jaccard error.
    pub jer: Option<f64>,
    /// Duration for which more than one reference speaker is active.
    pub reference_overlap_sec: f64,
    /// Canonical hypothesis-label to reference-label mapping.
    pub speaker_mapping: BTreeMap<String, String>,
}

/// Change-boundary precision, recall, F1, and timing error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChangePointScore {
    pub reference_count: usize,
    pub hypothesis_count: usize,
    pub matched_count: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub mean_absolute_error_sec: Option<f64>,
    pub collar_sec: f64,
}

/// One confidence observation with known correctness.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationObservation {
    pub confidence: f64,
    pub correct: bool,
}

/// Assignment-confidence calibration and coverage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationScore {
    pub observed: usize,
    pub opportunities: usize,
    pub coverage: f64,
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub bins: usize,
}

/// Whether reference-overlap regions contribute to the headline score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationOverlapPolicy {
    Include,
    Exclude,
}

/// Policy attached to a context-derived speaker hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationHintPolicy {
    Hard,
    Soft,
}

/// Immutable scorer policy. Every retained result embeds this complete value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiarizationScorerConfig {
    pub schema_version: String,
    /// Half-width removed around each reference speaker-change boundary.
    pub speaker_boundary_collar_ms: u64,
    /// One-to-one change-point matching tolerance.
    pub change_boundary_collar_ms: u64,
    pub overlap_policy: EvaluationOverlapPolicy,
    pub calibration_bins: usize,
    /// Number of concrete count bins in the count-posterior top-k diagnostic.
    pub count_top_k: usize,
    /// Target posterior mass for the deterministic count credible set.
    pub count_credible_mass_millionths: u32,
    /// Dominant labeled-speaker share at which a multi-speaker run is collapsed.
    pub dominant_speaker_collapse_share_millionths: u32,
    /// Minimum recall below which one mapped reference speaker is collapsed.
    pub minimum_reference_speaker_recall_millionths: u32,
    /// Minimum scored occupancy for one hypothesis label to be effective.
    pub minimum_effective_occupancy_ms: u64,
}

impl Default for DiarizationScorerConfig {
    fn default() -> Self {
        Self {
            schema_version: DIARIZATION_SCORER_CONFIG_SCHEMA_VERSION.to_owned(),
            speaker_boundary_collar_ms: 0,
            change_boundary_collar_ms: 250,
            overlap_policy: EvaluationOverlapPolicy::Include,
            calibration_bins: 10,
            count_top_k: 3,
            count_credible_mass_millionths: 900_000,
            dominant_speaker_collapse_share_millionths: 990_000,
            minimum_reference_speaker_recall_millionths: 100_000,
            minimum_effective_occupancy_ms: 250,
        }
    }
}

/// Integer-millisecond interval used by versioned evaluation documents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRegion {
    pub start_ms: u64,
    pub end_ms: u64,
    /// Stable machine reason such as `annotation_uncertain`.
    pub reason_code: String,
}

/// One versioned reference or hypothesis turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationTurn {
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker: Option<String>,
    /// Assignment confidence is meaningful only for a labeled hypothesis.
    pub speaker_confidence: Option<f64>,
    pub overlap_suspected: bool,
}

impl EvaluationTurn {
    /// Construct a reference or hypothesis turn without a confidence value.
    #[must_use]
    pub fn labeled(start_ms: u64, end_ms: u64, speaker: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            speaker: Some(speaker.into()),
            speaker_confidence: None,
            overlap_suspected: false,
        }
    }

    /// Construct an unknown-speaker hypothesis turn.
    #[must_use]
    pub const fn unknown(start_ms: u64, end_ms: u64) -> Self {
        Self {
            start_ms,
            end_ms,
            speaker: None,
            speaker_confidence: None,
            overlap_suspected: false,
        }
    }
}

/// Context-derived interval known or believed to belong to one speaker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationSpeakerHint {
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker_ref: String,
    pub policy: EvaluationHintPolicy,
}

/// One transcript-free aligned word used for speaker-attribution scoring.
///
/// `word_id` is an opaque annotation identity, never lexical content. The
/// scorer projects the diarization hypothesis at this interval onto the
/// reference speaker, so no transcript token is retained in the score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationWord {
    pub word_id: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker_ref: String,
}

/// Frozen ground-truth document. It deliberately has no path or transcript field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiarizationReferenceDocument {
    pub schema_version: String,
    pub recording_id: String,
    pub duration_ms: u64,
    pub turns: Vec<EvaluationTurn>,
    pub ignored_regions: Vec<EvaluationRegion>,
    pub speaker_hints: Vec<EvaluationSpeakerHint>,
    pub words: Vec<EvaluationWord>,
}

/// Runtime resource observation associated with one hypothesis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationPerformanceObservation {
    pub audio_duration_ms: u64,
    pub wall_time_ms: u64,
    pub peak_rss_bytes: u64,
}

/// Frozen system-output document. It deliberately has no path or transcript field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiarizationHypothesisDocument {
    pub schema_version: String,
    pub recording_id: String,
    pub duration_ms: u64,
    pub turns: Vec<EvaluationTurn>,
    pub speaker_count_estimate: Option<SpeakerCountEstimate>,
    pub performance: Option<EvaluationPerformanceObservation>,
}

/// Union speech-activity metrics, distinct from speaker-time DER components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechActivityScore {
    pub reference_speech_sec: f64,
    pub hypothesis_speech_sec: f64,
    pub correct_speech_sec: f64,
    pub missed_speech_sec: f64,
    pub false_alarm_sec: f64,
    pub error_rate: Option<f64>,
}

/// Speaker attribution quality after the frozen permutation mapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerAttributionScore {
    pub attributable_reference_speaker_time_sec: f64,
    pub correctly_attributed_speaker_time_sec: f64,
    pub attribution_error_sec: f64,
    pub accuracy: Option<f64>,
}

/// Reference/hypothesis speaker-count comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeakerCountScore {
    pub reference_speakers: usize,
    pub hypothesis_speakers: usize,
    pub signed_error: i64,
    pub absolute_error: u64,
}

/// Proper scores and set diagnostics for one automatic speaker-count posterior.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerCountPosteriorScore {
    pub reference_speakers: usize,
    pub posterior_available: bool,
    pub selected_count: Option<u32>,
    pub unresolved: bool,
    pub reference_probability: Option<f64>,
    pub negative_log_likelihood: Option<f64>,
    pub infinite_negative_log_likelihood: bool,
    pub brier_score: Option<f64>,
    pub top_k_hit: Option<bool>,
    pub credible_set: Vec<u32>,
    pub credible_set_includes_unresolved: bool,
    pub credible_set_hit: Option<bool>,
    pub unresolved_probability: Option<f64>,
    pub entropy_bits: Option<f64>,
    pub calibration_status: Option<SpeakerCountCalibrationStatus>,
}

/// Duration and recurrence evidence for one anonymized hypothesis label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerOccupancyEntry {
    pub hypothesis_speaker: String,
    pub mapped_reference_speaker: Option<String>,
    pub voiced_duration_sec: f64,
    pub labeled_share: f64,
    pub recurrence_episode_count: u64,
    pub effective: bool,
}

/// Run-level collapse and phantom-speaker diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerOccupancyScore {
    pub labeled_speaker_time_sec: f64,
    pub unknown_speaker_time_sec: f64,
    pub dominant_speaker_share: Option<f64>,
    pub unknown_speaker_share: Option<f64>,
    pub effective_speaker_count: usize,
    pub phantom_speaker_count: usize,
    pub collapsed_reference_speaker_count: usize,
    pub minority_reference_recall: Option<f64>,
    pub dominant_collapse_detected: bool,
    pub any_reference_collapse_detected: bool,
    pub speakers: Vec<SpeakerOccupancyEntry>,
}

/// Speaker attribution over aligned reference word intervals without text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordAttributionScore {
    pub reference_word_count: u64,
    pub scored_word_count: u64,
    pub correct_word_count: u64,
    pub incorrect_word_count: u64,
    pub unknown_word_count: u64,
    pub excluded_word_count: u64,
    pub word_diarization_error_rate: Option<f64>,
}

/// Duration-weighted overlap-region detection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlapDetectionScore {
    pub reference_overlap_sec: f64,
    pub hypothesis_overlap_sec: f64,
    pub true_positive_sec: f64,
    pub false_positive_sec: f64,
    pub false_negative_sec: f64,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

/// Adherence to hard and soft context hints after speaker permutation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HintAdherenceScore {
    pub hinted_sec: f64,
    pub adherent_sec: f64,
    pub contradictory_sec: f64,
    pub unknown_sec: f64,
    pub hard_violation_sec: f64,
    pub adherence_rate: Option<f64>,
}

/// Risk conditional on emitting a known speaker, plus abstention coverage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectiveAttributionScore {
    pub reference_speaker_time_sec: f64,
    pub covered_speaker_time_sec: f64,
    pub correct_covered_speaker_time_sec: f64,
    pub error_covered_speaker_time_sec: f64,
    pub unknown_speaker_time_sec: f64,
    pub coverage: Option<f64>,
    pub selective_risk: Option<f64>,
}

/// Duration-weighted assignment-confidence calibration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightedCalibrationScore {
    pub observed_duration_sec: f64,
    pub opportunity_duration_sec: f64,
    pub coverage: Option<f64>,
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub bins: usize,
}

/// Resource score kept separate from accuracy and calibration claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizationPerformanceScore {
    pub audio_duration_sec: f64,
    pub wall_time_sec: f64,
    pub real_time_factor: Option<f64>,
    pub peak_rss_bytes: u64,
}

/// Complete, reproducible result emitted by the authoritative scorer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthoritativeDiarizationScore {
    pub schema_version: String,
    pub scorer_version: String,
    pub recording_id: String,
    pub config: DiarizationScorerConfig,
    pub reference_sha256: String,
    pub hypothesis_sha256: String,
    pub config_sha256: String,
    pub scored_duration_sec: f64,
    pub ignored_duration_sec: f64,
    pub diarization: DiarizationScore,
    pub speech_activity: SpeechActivityScore,
    pub change_points: ChangePointScore,
    pub speaker_attribution: SpeakerAttributionScore,
    pub speaker_count: SpeakerCountScore,
    pub speaker_count_posterior: SpeakerCountPosteriorScore,
    pub speaker_occupancy: SpeakerOccupancyScore,
    pub word_attribution: WordAttributionScore,
    pub overlap: OverlapDetectionScore,
    pub hints: HintAdherenceScore,
    pub selective_attribution: SelectiveAttributionScore,
    pub calibration: WeightedCalibrationScore,
    pub performance: Option<DiarizationPerformanceScore>,
    /// Hash of the result with this field set to the empty string.
    pub result_sha256: String,
}

/// Immutable split identity used by the leakage auditor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSplit {
    Train,
    Development,
    Test,
}

/// One privacy-safe corpus recording descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusRecordingManifest {
    pub recording_id: String,
    pub split: EvaluationSplit,
    /// Stable source-call identity shared by clips from the same recording.
    pub origin_recording_id: String,
    /// Opaque speaker identities when the corpus license permits tracking them.
    pub speaker_refs: Vec<String>,
    /// Opaque ancestor recording IDs for derived clips or mixtures.
    pub derived_from_recording_ids: Vec<String>,
    /// Opaque group shared by all augmentations of the same source example.
    pub augmentation_group_id: Option<String>,
    /// Opaque recordings used to enroll profiles for this example.
    pub enrollment_recording_ids: Vec<String>,
}

/// Versioned, path-free corpus manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiarizationCorpusManifest {
    pub schema_version: String,
    pub corpus_id: String,
    pub license_id: String,
    pub recordings: Vec<CorpusRecordingManifest>,
}

/// Machine-stable split-leakage category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeakageKind {
    DuplicateRecording,
    SharedOrigin,
    SharedSpeaker,
    SharedDerivedSource,
    SharedAugmentation,
    CrossSplitEnrollment,
}

/// Privacy-safe leakage finding. IDs are validated opaque identifiers, not paths.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LeakageFinding {
    pub kind: LeakageKind,
    pub left_split: EvaluationSplit,
    pub right_split: EvaluationSplit,
    pub opaque_id: String,
}

/// Deterministic leakage audit for one corpus manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiarizationLeakageAudit {
    pub schema_version: String,
    pub manifest_sha256: String,
    pub passed: bool,
    pub findings: Vec<LeakageFinding>,
    pub audit_sha256: String,
}

#[derive(Debug, Clone)]
struct AtomicInterval {
    duration: f64,
    reference: Vec<usize>,
    hypothesis: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
struct EvaluationHypothesisState {
    speaker: Option<String>,
    confidence: Option<f64>,
}

#[derive(Debug, Clone)]
struct EvaluationAtomicInterval {
    start_ms: u64,
    end_ms: u64,
    reference: BTreeSet<String>,
    hypothesis: Vec<EvaluationHypothesisState>,
    hypothesis_overlap_suspected: bool,
    excluded: bool,
    overlap_scoring_excluded: bool,
}

impl EvaluationAtomicInterval {
    fn duration_sec(&self) -> f64 {
        (self.end_ms - self.start_ms) as f64 / 1_000.0
    }
}

#[derive(Debug, Clone, Copy)]
struct WeightedCalibrationObservation {
    confidence: f64,
    correct: bool,
    duration_sec: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ChangeMatch {
    count: usize,
    total_error: f64,
}

/// Score a hypothesis turn timeline against labeled reference turns.
///
/// Overlapping reference turns are supported and contribute speaker-time to
/// DER. Hypothesis turns with `speaker == None` count as speech with unknown
/// identity, so they avoid a miss but incur confusion when a reference speaker
/// is active. The scorer does not apply a forgiveness collar; callers must
/// explicitly transform timelines if their evaluation protocol uses one.
pub fn score_diarization(
    reference: &[ScoringTurn],
    hypothesis: &[ScoringTurn],
) -> FwResult<DiarizationScore> {
    validate_turns(reference, true)?;
    validate_turns(hypothesis, false)?;

    let reference_labels = collect_labels(reference);
    let hypothesis_labels = collect_labels(hypothesis);
    let reference_index = label_index(&reference_labels);
    let hypothesis_index = label_index(&hypothesis_labels);
    let intervals = atomic_intervals(reference, hypothesis, &reference_index, &hypothesis_index);

    let mut overlap_weights = vec![vec![0.0_f64; hypothesis_labels.len()]; reference_labels.len()];
    for interval in &intervals {
        for &reference_speaker in &interval.reference {
            for &hypothesis_speaker in interval.hypothesis.iter().flatten() {
                overlap_weights[reference_speaker][hypothesis_speaker] += interval.duration;
            }
        }
    }

    let hypothesis_to_reference =
        maximum_overlap_mapping(&overlap_weights, hypothesis_labels.len());
    let speaker_mapping = hypothesis_to_reference
        .iter()
        .enumerate()
        .filter_map(|(hypothesis_id, reference_id)| {
            reference_id.map(|reference_id| {
                (
                    hypothesis_labels[hypothesis_id].clone(),
                    reference_labels[reference_id].clone(),
                )
            })
        })
        .collect::<BTreeMap<_, _>>();

    let mut reference_speaker_time_sec = 0.0;
    let mut missed_speech_sec = 0.0;
    let mut false_alarm_sec = 0.0;
    let mut speaker_confusion_sec = 0.0;
    let mut reference_overlap_sec = 0.0;

    for interval in &intervals {
        let reference_count = interval.reference.len();
        let hypothesis_count = interval.hypothesis.len();
        reference_speaker_time_sec += interval.duration * reference_count as f64;
        if reference_count > 1 {
            reference_overlap_sec += interval.duration;
        }

        let paired = reference_count.min(hypothesis_count);
        missed_speech_sec += interval.duration * reference_count.saturating_sub(paired) as f64;
        false_alarm_sec += interval.duration * hypothesis_count.saturating_sub(paired) as f64;

        let reference_set = interval.reference.iter().copied().collect::<BTreeSet<_>>();
        let matched = interval
            .hypothesis
            .iter()
            .filter_map(|speaker| speaker.and_then(|id| hypothesis_to_reference[id]))
            .filter(|reference_id| reference_set.contains(reference_id))
            .collect::<BTreeSet<_>>()
            .len()
            .min(paired);
        speaker_confusion_sec += interval.duration * paired.saturating_sub(matched) as f64;
    }

    let der = (reference_speaker_time_sec > SCORE_EPSILON_SEC).then_some(
        (missed_speech_sec + false_alarm_sec + speaker_confusion_sec) / reference_speaker_time_sec,
    );
    let jer = score_jer(
        &intervals,
        reference_labels.len(),
        hypothesis_labels.len(),
        &hypothesis_to_reference,
    );

    Ok(DiarizationScore {
        reference_speaker_time_sec,
        missed_speech_sec,
        false_alarm_sec,
        speaker_confusion_sec,
        der,
        jer,
        reference_overlap_sec,
        speaker_mapping,
    })
}

/// Match hypothesized change points to reference points within `collar_sec`.
///
/// Matching is one-to-one and deterministic. Inputs may be unsorted.
pub fn score_change_points(
    reference_sec: &[f64],
    hypothesis_sec: &[f64],
    collar_sec: f64,
) -> FwResult<ChangePointScore> {
    if !collar_sec.is_finite() || collar_sec < 0.0 {
        return Err(FwError::InvalidRequest(
            "change-point collar must be finite and non-negative".to_owned(),
        ));
    }
    validate_points(reference_sec, "reference")?;
    validate_points(hypothesis_sec, "hypothesis")?;

    let mut reference = reference_sec.to_vec();
    let mut hypothesis = hypothesis_sec.to_vec();
    reference.sort_by(f64::total_cmp);
    hypothesis.sort_by(f64::total_cmp);

    let change_match = minimum_error_change_match(&reference, &hypothesis, collar_sec);
    let matched_count = change_match.count;

    let precision =
        (!hypothesis.is_empty()).then_some(matched_count as f64 / hypothesis.len() as f64);
    let recall = (!reference.is_empty()).then_some(matched_count as f64 / reference.len() as f64);
    let f1 = match (precision, recall) {
        (Some(precision), Some(recall)) if precision + recall > 0.0 => {
            Some(2.0 * precision * recall / (precision + recall))
        }
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };

    Ok(ChangePointScore {
        reference_count: reference.len(),
        hypothesis_count: hypothesis.len(),
        matched_count,
        precision,
        recall,
        f1,
        mean_absolute_error_sec: (matched_count > 0)
            .then_some(change_match.total_error / matched_count as f64),
        collar_sec,
    })
}

/// Score assignment-confidence calibration and labeled coverage.
pub fn score_calibration(
    observations: &[CalibrationObservation],
    total_opportunities: usize,
    bins: usize,
) -> FwResult<CalibrationScore> {
    if bins == 0 {
        return Err(FwError::InvalidRequest(
            "calibration bins must be greater than zero".to_owned(),
        ));
    }
    if observations.len() > total_opportunities {
        return Err(FwError::InvalidRequest(format!(
            "calibration observations ({}) exceed opportunities ({total_opportunities})",
            observations.len()
        )));
    }
    for observation in observations {
        if !observation.confidence.is_finite() || !(0.0..=1.0).contains(&observation.confidence) {
            return Err(FwError::InvalidRequest(
                "calibration confidence must be finite and within [0, 1]".to_owned(),
            ));
        }
    }

    let coverage = if total_opportunities == 0 {
        0.0
    } else {
        observations.len() as f64 / total_opportunities as f64
    };
    if observations.is_empty() {
        return Ok(CalibrationScore {
            observed: 0,
            opportunities: total_opportunities,
            coverage,
            brier_score: None,
            expected_calibration_error: None,
            bins,
        });
    }

    let mut brier_sum = 0.0;
    let mut bin_counts = vec![0usize; bins];
    let mut bin_confidence = vec![0.0_f64; bins];
    let mut bin_correct = vec![0usize; bins];
    for observation in observations {
        let outcome = f64::from(observation.correct);
        brier_sum += (observation.confidence - outcome).powi(2);
        let bin = ((observation.confidence * bins as f64).floor() as usize).min(bins - 1);
        bin_counts[bin] += 1;
        bin_confidence[bin] += observation.confidence;
        bin_correct[bin] += usize::from(observation.correct);
    }

    let mut expected_calibration_error = 0.0;
    for bin in 0..bins {
        if bin_counts[bin] == 0 {
            continue;
        }
        let mean_confidence = bin_confidence[bin] / bin_counts[bin] as f64;
        let accuracy = bin_correct[bin] as f64 / bin_counts[bin] as f64;
        let weight = bin_counts[bin] as f64 / observations.len() as f64;
        expected_calibration_error += weight * (accuracy - mean_confidence).abs();
    }

    Ok(CalibrationScore {
        observed: observations.len(),
        opportunities: total_opportunities,
        coverage,
        brier_score: Some(brier_sum / observations.len() as f64),
        expected_calibration_error: Some(expected_calibration_error),
        bins,
    })
}

/// Parse and validate one frozen reference document.
pub fn parse_diarization_reference(bytes: &[u8]) -> FwResult<DiarizationReferenceDocument> {
    let document = serde_json::from_slice(bytes)?;
    validate_reference_document(&document)?;
    Ok(document)
}

/// Parse and validate one frozen hypothesis document.
pub fn parse_diarization_hypothesis(bytes: &[u8]) -> FwResult<DiarizationHypothesisDocument> {
    let document = serde_json::from_slice(bytes)?;
    validate_hypothesis_document(&document)?;
    Ok(document)
}

/// Parse and validate one frozen corpus manifest.
pub fn parse_diarization_corpus_manifest(bytes: &[u8]) -> FwResult<DiarizationCorpusManifest> {
    let manifest = serde_json::from_slice(bytes)?;
    validate_corpus_manifest(&manifest)?;
    Ok(manifest)
}

/// Score two versioned documents using the complete frozen policy.
pub fn score_diarization_documents(
    reference: &DiarizationReferenceDocument,
    hypothesis: &DiarizationHypothesisDocument,
    config: &DiarizationScorerConfig,
) -> FwResult<AuthoritativeDiarizationScore> {
    validate_reference_document(reference)?;
    validate_hypothesis_document(hypothesis)?;
    validate_scorer_config(config)?;
    if reference.recording_id != hypothesis.recording_id {
        return Err(score_error(
            "recording_id_mismatch",
            "reference and hypothesis recording IDs differ",
        ));
    }
    if reference.duration_ms != hypothesis.duration_ms {
        return Err(score_error(
            "duration_mismatch",
            "reference and hypothesis durations differ",
        ));
    }
    if let Some(performance) = &hypothesis.performance
        && performance.audio_duration_ms != reference.duration_ms
    {
        return Err(score_error(
            "performance_duration_mismatch",
            "performance audio duration differs from the scored recording duration",
        ));
    }

    let reference_changes =
        speaker_change_points_ms(&reference.turns, reference.duration_ms, true)?;
    let hypothesis_changes =
        speaker_change_points_ms(&hypothesis.turns, hypothesis.duration_ms, false)?;
    let atoms = evaluation_atomic_intervals(reference, hypothesis, config, &reference_changes);
    let (scoring_reference, scoring_hypothesis) = scoring_turns_from_atoms(&atoms);
    let diarization = score_diarization(&scoring_reference, &scoring_hypothesis)?;

    let scored_duration_sec = atoms
        .iter()
        .filter(|atom| !atom.excluded)
        .map(EvaluationAtomicInterval::duration_sec)
        .sum();
    let ignored_duration_sec = atoms
        .iter()
        .filter(|atom| atom.excluded)
        .map(EvaluationAtomicInterval::duration_sec)
        .sum();
    let speech_activity = score_speech_activity(&atoms);
    let speaker_attribution = score_speaker_attribution(&diarization);
    let speaker_count = score_speaker_count(&scoring_reference, &scoring_hypothesis);
    let speaker_count_posterior = score_speaker_count_posterior(
        speaker_count.reference_speakers,
        hypothesis.speaker_count_estimate.as_ref(),
        config,
    );
    let speaker_occupancy =
        score_speaker_occupancy(hypothesis, &atoms, &diarization.speaker_mapping, config);
    let word_attribution = score_word_attribution(
        reference,
        hypothesis,
        &diarization.speaker_mapping,
        config,
        &reference_changes,
    );
    let overlap = score_overlap_detection(&atoms);
    let hints = score_hint_adherence(
        &atoms,
        &reference.speaker_hints,
        &diarization.speaker_mapping,
    );
    let selective_attribution = score_selective_attribution(&atoms, &diarization.speaker_mapping);
    let calibration = score_weighted_calibration(
        &atoms,
        &diarization.speaker_mapping,
        config.calibration_bins,
    );
    let change_points = score_change_points(
        &reference_changes
            .iter()
            .filter(|point| !millisecond_is_ignored(**point, &reference.ignored_regions))
            .map(|point| *point as f64 / 1_000.0)
            .collect::<Vec<_>>(),
        &hypothesis_changes
            .iter()
            .filter(|point| !millisecond_is_ignored(**point, &reference.ignored_regions))
            .map(|point| *point as f64 / 1_000.0)
            .collect::<Vec<_>>(),
        config.change_boundary_collar_ms as f64 / 1_000.0,
    )?;
    let performance = hypothesis.performance.as_ref().map(|observation| {
        let audio_duration_sec = observation.audio_duration_ms as f64 / 1_000.0;
        let wall_time_sec = observation.wall_time_ms as f64 / 1_000.0;
        DiarizationPerformanceScore {
            audio_duration_sec,
            wall_time_sec,
            real_time_factor: (audio_duration_sec > 0.0)
                .then_some(wall_time_sec / audio_duration_sec),
            peak_rss_bytes: observation.peak_rss_bytes,
        }
    });

    let mut result = AuthoritativeDiarizationScore {
        schema_version: DIARIZATION_SCORE_RESULT_SCHEMA_VERSION.to_owned(),
        scorer_version: DIARIZATION_SCORER_VERSION.to_owned(),
        recording_id: reference.recording_id.clone(),
        config: config.clone(),
        reference_sha256: canonical_json_sha256(reference)?,
        hypothesis_sha256: canonical_json_sha256(hypothesis)?,
        config_sha256: canonical_json_sha256(config)?,
        scored_duration_sec,
        ignored_duration_sec,
        diarization,
        speech_activity,
        change_points,
        speaker_attribution,
        speaker_count,
        speaker_count_posterior,
        speaker_occupancy,
        word_attribution,
        overlap,
        hints,
        selective_attribution,
        calibration,
        performance,
        result_sha256: String::new(),
    };
    result.result_sha256 = canonical_json_sha256(&result)?;
    Ok(result)
}

/// Verify the self-hash on a retained authoritative score.
pub fn verify_authoritative_score_hash(result: &AuthoritativeDiarizationScore) -> FwResult<()> {
    if result.schema_version != DIARIZATION_SCORE_RESULT_SCHEMA_VERSION {
        return Err(score_error(
            "result_schema_version",
            "unsupported diarization score-result schema version",
        ));
    }
    if result.scorer_version != DIARIZATION_SCORER_VERSION {
        return Err(score_error(
            "scorer_version",
            "unsupported diarization scorer version",
        ));
    }
    validate_scorer_config(&result.config)?;
    for (field, hash) in [
        ("reference_sha256", &result.reference_sha256),
        ("hypothesis_sha256", &result.hypothesis_sha256),
        ("config_sha256", &result.config_sha256),
    ] {
        if !is_sha256_hex(hash) {
            return Err(score_error(
                "result_hash_format",
                &format!("{field} must be 64 lowercase hexadecimal characters"),
            ));
        }
    }
    if canonical_json_sha256(&result.config)? != result.config_sha256 {
        return Err(score_error(
            "config_hash_mismatch",
            "config_sha256 does not match the embedded scorer configuration",
        ));
    }
    if !is_sha256_hex(&result.result_sha256) {
        return Err(score_error(
            "result_hash_format",
            "result_sha256 must be 64 lowercase hexadecimal characters",
        ));
    }
    let mut unhashed = result.clone();
    unhashed.result_sha256.clear();
    let expected = canonical_json_sha256(&unhashed)?;
    if expected != result.result_sha256 {
        return Err(score_error(
            "result_hash_mismatch",
            "result_sha256 does not match the canonical result",
        ));
    }
    Ok(())
}

/// Audit a versioned corpus manifest for cross-split contamination.
pub fn audit_diarization_manifest(
    manifest: &DiarizationCorpusManifest,
) -> FwResult<DiarizationLeakageAudit> {
    validate_corpus_manifest(manifest)?;
    let mut findings = BTreeSet::new();
    for (left_index, left) in manifest.recordings.iter().enumerate() {
        for right in manifest.recordings.iter().skip(left_index + 1) {
            if left.split == right.split {
                continue;
            }
            let (left_split, right_split) = ordered_splits(left.split, right.split);
            if left.recording_id == right.recording_id {
                findings.insert(LeakageFinding {
                    kind: LeakageKind::DuplicateRecording,
                    left_split,
                    right_split,
                    opaque_id: left.recording_id.clone(),
                });
            }
            if left.origin_recording_id == right.origin_recording_id {
                findings.insert(LeakageFinding {
                    kind: LeakageKind::SharedOrigin,
                    left_split,
                    right_split,
                    opaque_id: left.origin_recording_id.clone(),
                });
            }
            for speaker in sorted_intersection(&left.speaker_refs, &right.speaker_refs) {
                findings.insert(LeakageFinding {
                    kind: LeakageKind::SharedSpeaker,
                    left_split,
                    right_split,
                    opaque_id: speaker,
                });
            }
            let right_identity = [
                right.recording_id.clone(),
                right.origin_recording_id.clone(),
            ]
            .into_iter()
            .collect::<BTreeSet<_>>();
            let left_identity = [left.recording_id.clone(), left.origin_recording_id.clone()]
                .into_iter()
                .collect::<BTreeSet<_>>();
            let left_sources = lineage_ids(left);
            let right_sources = lineage_ids(right);
            for source in sorted_set_intersection(&left_sources, &right_sources)
                .into_iter()
                .chain(sorted_set_intersection(&left_sources, &right_identity))
                .chain(sorted_set_intersection(&right_sources, &left_identity))
            {
                findings.insert(LeakageFinding {
                    kind: LeakageKind::SharedDerivedSource,
                    left_split,
                    right_split,
                    opaque_id: source,
                });
            }
            if let (Some(left_group), Some(right_group)) = (
                left.augmentation_group_id.as_ref(),
                right.augmentation_group_id.as_ref(),
            ) && left_group == right_group
            {
                findings.insert(LeakageFinding {
                    kind: LeakageKind::SharedAugmentation,
                    left_split,
                    right_split,
                    opaque_id: left_group.clone(),
                });
            }
            let left_enrollment = left
                .enrollment_recording_ids
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            let right_enrollment = right
                .enrollment_recording_ids
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            for enrollment in sorted_set_intersection(&left_enrollment, &right_enrollment)
                .into_iter()
                .chain(sorted_set_intersection(&left_enrollment, &right_identity))
                .chain(sorted_set_intersection(&right_enrollment, &left_identity))
            {
                findings.insert(LeakageFinding {
                    kind: LeakageKind::CrossSplitEnrollment,
                    left_split,
                    right_split,
                    opaque_id: enrollment,
                });
            }
        }
    }

    let findings = findings.into_iter().collect::<Vec<_>>();
    let mut audit = DiarizationLeakageAudit {
        schema_version: DIARIZATION_LEAKAGE_AUDIT_SCHEMA_VERSION.to_owned(),
        manifest_sha256: canonical_json_sha256(manifest)?,
        passed: findings.is_empty(),
        findings,
        audit_sha256: String::new(),
    };
    audit.audit_sha256 = canonical_json_sha256(&audit)?;
    Ok(audit)
}

/// Verify the self-hash on a retained leakage audit.
pub fn verify_leakage_audit_hash(audit: &DiarizationLeakageAudit) -> FwResult<()> {
    if audit.schema_version != DIARIZATION_LEAKAGE_AUDIT_SCHEMA_VERSION {
        return Err(score_error(
            "leakage_schema_version",
            "unsupported leakage-audit schema version",
        ));
    }
    if !is_sha256_hex(&audit.audit_sha256) {
        return Err(score_error(
            "leakage_hash_format",
            "audit_sha256 must be 64 lowercase hexadecimal characters",
        ));
    }
    if !is_sha256_hex(&audit.manifest_sha256) {
        return Err(score_error(
            "leakage_manifest_hash_format",
            "manifest_sha256 must be 64 lowercase hexadecimal characters",
        ));
    }
    if audit.passed != audit.findings.is_empty() {
        return Err(score_error(
            "leakage_passed_inconsistent",
            "leakage passed flag must equal findings.is_empty()",
        ));
    }
    if !audit
        .findings
        .windows(2)
        .all(|window| window[0] < window[1])
    {
        return Err(score_error(
            "leakage_finding_order",
            "leakage findings must be strictly sorted and unique",
        ));
    }
    let mut unhashed = audit.clone();
    unhashed.audit_sha256.clear();
    if canonical_json_sha256(&unhashed)? != audit.audit_sha256 {
        return Err(score_error(
            "leakage_hash_mismatch",
            "audit_sha256 does not match the canonical leakage audit",
        ));
    }
    Ok(())
}

fn validate_reference_document(document: &DiarizationReferenceDocument) -> FwResult<()> {
    if document.schema_version != DIARIZATION_REFERENCE_SCHEMA_VERSION {
        return Err(score_error(
            "reference_schema_version",
            "unsupported diarization reference schema version",
        ));
    }
    validate_opaque_id(&document.recording_id, "reference recording_id")?;
    if document.duration_ms == 0 {
        return Err(score_error(
            "reference_duration",
            "reference duration_ms must be greater than zero",
        ));
    }
    validate_evaluation_turns(&document.turns, document.duration_ms, true, "reference")?;
    validate_regions(
        &document.ignored_regions,
        document.duration_ms,
        "ignored region",
    )?;
    let reference_speakers = document
        .turns
        .iter()
        .filter_map(|turn| turn.speaker.as_ref())
        .collect::<BTreeSet<_>>();
    validate_evaluation_words(document, &reference_speakers)?;
    for (index, hint) in document.speaker_hints.iter().enumerate() {
        validate_ms_interval(
            hint.start_ms,
            hint.end_ms,
            document.duration_ms,
            &format!("speaker hint {index}"),
        )?;
        validate_opaque_id(
            &hint.speaker_ref,
            &format!("speaker hint {index} speaker_ref"),
        )?;
        if !reference_speakers.contains(&hint.speaker_ref) {
            return Err(score_error(
                "hint_speaker_not_in_reference",
                &format!("speaker hint {index} does not name a reference speaker"),
            ));
        }
    }
    ensure_canonical_hint_order(&document.speaker_hints)?;
    for window in document.speaker_hints.windows(2) {
        if window[0].end_ms > window[1].start_ms {
            return Err(score_error(
                "ambiguous_hint_overlap",
                "overlapping speaker hints are not scoreable",
            ));
        }
    }
    Ok(())
}

fn validate_hypothesis_document(document: &DiarizationHypothesisDocument) -> FwResult<()> {
    if document.schema_version != DIARIZATION_HYPOTHESIS_SCHEMA_VERSION {
        return Err(score_error(
            "hypothesis_schema_version",
            "unsupported diarization hypothesis schema version",
        ));
    }
    validate_opaque_id(&document.recording_id, "hypothesis recording_id")?;
    if document.duration_ms == 0 {
        return Err(score_error(
            "hypothesis_duration",
            "hypothesis duration_ms must be greater than zero",
        ));
    }
    validate_evaluation_turns(&document.turns, document.duration_ms, false, "hypothesis")?;
    if let Some(estimate) = &document.speaker_count_estimate {
        estimate.validate().map_err(|message| {
            score_error(
                "speaker_count_estimate",
                &format!("hypothesis speaker-count estimate is invalid: {message}"),
            )
        })?;
    }
    if let Some(performance) = &document.performance {
        if performance.audio_duration_ms == 0 {
            return Err(score_error(
                "performance_audio_duration",
                "performance audio_duration_ms must be greater than zero",
            ));
        }
        if performance.wall_time_ms == 0 {
            return Err(score_error(
                "performance_wall_time",
                "performance wall_time_ms must be greater than zero",
            ));
        }
    }
    Ok(())
}

fn validate_scorer_config(config: &DiarizationScorerConfig) -> FwResult<()> {
    if config.schema_version != DIARIZATION_SCORER_CONFIG_SCHEMA_VERSION {
        return Err(score_error(
            "config_schema_version",
            "unsupported diarization scorer-config schema version",
        ));
    }
    if config.calibration_bins == 0 || config.calibration_bins > 1_000 {
        return Err(score_error(
            "calibration_bins",
            "calibration_bins must be within 1..=1000",
        ));
    }
    if config.count_top_k == 0 || config.count_top_k > MAX_SPEAKER_COUNT as usize {
        return Err(score_error(
            "count_top_k",
            &format!("count_top_k must be within 1..={}", MAX_SPEAKER_COUNT),
        ));
    }
    if !(1..=1_000_000).contains(&config.count_credible_mass_millionths) {
        return Err(score_error(
            "count_credible_mass",
            "count_credible_mass_millionths must be within 1..=1000000",
        ));
    }
    if !(500_001..=1_000_000).contains(&config.dominant_speaker_collapse_share_millionths) {
        return Err(score_error(
            "dominant_speaker_collapse_share",
            "dominant_speaker_collapse_share_millionths must be within 500001..=1000000",
        ));
    }
    if !(1..=1_000_000).contains(&config.minimum_reference_speaker_recall_millionths) {
        return Err(score_error(
            "minimum_reference_speaker_recall",
            "minimum_reference_speaker_recall_millionths must be within 1..=1000000",
        ));
    }
    if config.minimum_effective_occupancy_ms == 0 {
        return Err(score_error(
            "minimum_effective_occupancy",
            "minimum_effective_occupancy_ms must be greater than zero",
        ));
    }
    Ok(())
}

fn validate_evaluation_words(
    document: &DiarizationReferenceDocument,
    reference_speakers: &BTreeSet<&String>,
) -> FwResult<()> {
    let mut word_ids = BTreeSet::new();
    for (index, word) in document.words.iter().enumerate() {
        validate_opaque_word_id(&word.word_id, index)?;
        validate_opaque_id(
            &word.speaker_ref,
            &format!("reference word {index} speaker_ref"),
        )?;
        validate_ms_interval(
            word.start_ms,
            word.end_ms,
            document.duration_ms,
            &format!("reference word {index}"),
        )?;
        if !word_ids.insert(&word.word_id) {
            return Err(score_error(
                "word_id_duplicate",
                &format!("reference word {index} repeats an opaque word_id"),
            ));
        }
        if !reference_speakers.contains(&word.speaker_ref) {
            return Err(score_error(
                "word_speaker_not_in_reference",
                &format!("reference word {index} does not name a reference speaker"),
            ));
        }
        let point_ms = word.start_ms + (word.end_ms - word.start_ms) / 2;
        if !document.turns.iter().any(|turn| {
            turn.speaker.as_ref() == Some(&word.speaker_ref)
                && turn.start_ms <= point_ms
                && point_ms < turn.end_ms
        }) {
            return Err(score_error(
                "word_speaker_inactive",
                &format!("reference word {index} speaker is not active at the word midpoint"),
            ));
        }
    }
    if !document.words.windows(2).all(|window| {
        (
            window[0].start_ms,
            window[0].end_ms,
            window[0].word_id.as_str(),
            window[0].speaker_ref.as_str(),
        ) < (
            window[1].start_ms,
            window[1].end_ms,
            window[1].word_id.as_str(),
            window[1].speaker_ref.as_str(),
        )
    }) {
        return Err(score_error(
            "word_order",
            "reference words must use strictly increasing start/end/id/speaker order",
        ));
    }
    Ok(())
}

fn validate_evaluation_turns(
    turns: &[EvaluationTurn],
    duration_ms: u64,
    reference: bool,
    kind: &str,
) -> FwResult<()> {
    for (index, turn) in turns.iter().enumerate() {
        validate_ms_interval(
            turn.start_ms,
            turn.end_ms,
            duration_ms,
            &format!("{kind} turn {index}"),
        )?;
        if reference && turn.speaker.is_none() {
            return Err(score_error(
                "reference_speaker_missing",
                &format!("reference turn {index} must have a speaker"),
            ));
        }
        if let Some(speaker) = &turn.speaker {
            validate_opaque_id(speaker, &format!("{kind} turn {index} speaker"))?;
        }
        if reference && turn.speaker_confidence.is_some() {
            return Err(score_error(
                "reference_confidence_forbidden",
                &format!("reference turn {index} must not contain speaker_confidence"),
            ));
        }
        if turn.speaker.is_none() && turn.speaker_confidence.is_some() {
            return Err(score_error(
                "unknown_confidence_forbidden",
                &format!("{kind} turn {index} cannot assign confidence to an unknown speaker"),
            ));
        }
        if let Some(confidence) = turn.speaker_confidence
            && (!confidence.is_finite() || !(0.0..=1.0).contains(&confidence))
        {
            return Err(score_error(
                "speaker_confidence_range",
                &format!("{kind} turn {index} speaker_confidence must be finite and within [0, 1]"),
            ));
        }
    }
    if !turns.windows(2).all(|window| {
        evaluation_turn_order_key(&window[0]) <= evaluation_turn_order_key(&window[1])
    }) {
        return Err(score_error(
            "turn_order",
            &format!("{kind} turns must use canonical start/end/speaker order"),
        ));
    }
    Ok(())
}

fn validate_regions(regions: &[EvaluationRegion], duration_ms: u64, kind: &str) -> FwResult<()> {
    for (index, region) in regions.iter().enumerate() {
        validate_ms_interval(
            region.start_ms,
            region.end_ms,
            duration_ms,
            &format!("{kind} {index}"),
        )?;
        validate_reason_code(&region.reason_code, &format!("{kind} {index} reason_code"))?;
    }
    if !regions.windows(2).all(|window| {
        (
            window[0].start_ms,
            window[0].end_ms,
            window[0].reason_code.as_str(),
        ) <= (
            window[1].start_ms,
            window[1].end_ms,
            window[1].reason_code.as_str(),
        )
    }) {
        return Err(score_error(
            "region_order",
            "ignored regions must use canonical start/end/reason order",
        ));
    }
    Ok(())
}

fn validate_ms_interval(start_ms: u64, end_ms: u64, duration_ms: u64, field: &str) -> FwResult<()> {
    if start_ms >= end_ms {
        return Err(score_error(
            "interval_geometry",
            &format!("{field} must satisfy start_ms < end_ms"),
        ));
    }
    if end_ms > duration_ms {
        return Err(score_error(
            "interval_bounds",
            &format!("{field} exceeds duration_ms"),
        ));
    }
    Ok(())
}

fn ensure_canonical_hint_order(hints: &[EvaluationSpeakerHint]) -> FwResult<()> {
    if hints.windows(2).all(|window| {
        (
            window[0].start_ms,
            window[0].end_ms,
            window[0].speaker_ref.as_str(),
            window[0].policy,
        ) <= (
            window[1].start_ms,
            window[1].end_ms,
            window[1].speaker_ref.as_str(),
            window[1].policy,
        )
    }) {
        Ok(())
    } else {
        Err(score_error(
            "hint_order",
            "speaker hints must use canonical start/end/speaker/policy order",
        ))
    }
}

fn validate_corpus_manifest(manifest: &DiarizationCorpusManifest) -> FwResult<()> {
    if manifest.schema_version != DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION {
        return Err(score_error(
            "manifest_schema_version",
            "unsupported diarization corpus-manifest schema version",
        ));
    }
    validate_opaque_id(&manifest.corpus_id, "manifest corpus_id")?;
    validate_opaque_id(&manifest.license_id, "manifest license_id")?;
    for (index, recording) in manifest.recordings.iter().enumerate() {
        validate_opaque_id(
            &recording.recording_id,
            &format!("manifest recording {index} recording_id"),
        )?;
        validate_opaque_id(
            &recording.origin_recording_id,
            &format!("manifest recording {index} origin_recording_id"),
        )?;
        validate_sorted_opaque_ids(
            &recording.speaker_refs,
            &format!("manifest recording {index} speaker_refs"),
        )?;
        validate_sorted_opaque_ids(
            &recording.derived_from_recording_ids,
            &format!("manifest recording {index} derived_from_recording_ids"),
        )?;
        validate_sorted_opaque_ids(
            &recording.enrollment_recording_ids,
            &format!("manifest recording {index} enrollment_recording_ids"),
        )?;
        if let Some(group) = &recording.augmentation_group_id {
            validate_opaque_id(
                group,
                &format!("manifest recording {index} augmentation_group_id"),
            )?;
        }
    }
    if !manifest.recordings.windows(2).all(|window| {
        (window[0].recording_id.as_str(), window[0].split)
            < (window[1].recording_id.as_str(), window[1].split)
    }) {
        return Err(score_error(
            "manifest_recording_order",
            "manifest recordings must use strictly increasing recording_id/split order",
        ));
    }
    Ok(())
}

fn validate_sorted_opaque_ids(values: &[String], field: &str) -> FwResult<()> {
    for value in values {
        validate_opaque_id(value, field)?;
    }
    if values.windows(2).all(|window| window[0] < window[1]) {
        Ok(())
    } else {
        Err(score_error(
            "opaque_id_order",
            &format!("{field} must be strictly sorted and unique"),
        ))
    }
}

fn validate_opaque_id(value: &str, field: &str) -> FwResult<()> {
    if value.is_empty() || value.len() > MAX_OPAQUE_ID_LEN || value.trim() != value {
        return Err(score_error(
            "opaque_id_shape",
            &format!("{field} must be a non-empty trimmed opaque identifier"),
        ));
    }
    if value.chars().any(char::is_control)
        || value.contains('/')
        || value.contains('\\')
        || value.contains("..")
    {
        return Err(score_error(
            "opaque_id_path",
            &format!("{field} must not contain a path"),
        ));
    }
    let lower = value.to_ascii_lowercase();
    const PRIVATE_MARKERS: [&str; 12] = [
        "transcript",
        "downloads",
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
    ];
    if PRIVATE_MARKERS.iter().any(|marker| lower.contains(marker)) {
        return Err(score_error(
            "opaque_id_sensitive_marker",
            &format!("{field} contains a forbidden media, transcript, or path marker"),
        ));
    }
    Ok(())
}

fn validate_opaque_word_id(value: &str, index: usize) -> FwResult<()> {
    validate_opaque_id(value, &format!("reference word {index} word_id"))?;
    if value
        .strip_prefix("word-")
        .is_none_or(|suffix| suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err(score_error(
            "word_id_shape",
            &format!("reference word {index} word_id must use the non-lexical word-<digits> form"),
        ));
    }
    Ok(())
}

fn validate_reason_code(value: &str, field: &str) -> FwResult<()> {
    validate_opaque_id(value, field)?;
    if value
        .bytes()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte))
    {
        Ok(())
    } else {
        Err(score_error(
            "reason_code_shape",
            &format!("{field} must contain only lowercase ASCII, digits, dot, dash, or underscore"),
        ))
    }
}

fn evaluation_turn_order_key(turn: &EvaluationTurn) -> (u64, u64, Option<&str>, Option<u64>, bool) {
    (
        turn.start_ms,
        turn.end_ms,
        turn.speaker.as_deref(),
        turn.speaker_confidence.map(f64::to_bits),
        turn.overlap_suspected,
    )
}

fn score_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("diarization.scorer.{code}: {message}"))
}

fn canonical_json_sha256<T: Serialize>(value: &T) -> FwResult<String> {
    let encoded = serde_json::to_vec(value)?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == SCORE_HASH_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) fn speaker_change_points_ms(
    turns: &[EvaluationTurn],
    duration_ms: u64,
    reference: bool,
) -> FwResult<Vec<u64>> {
    let mut boundaries = turns
        .iter()
        .flat_map(|turn| [turn.start_ms, turn.end_ms])
        .filter(|point| *point > 0 && *point < duration_ms)
        .collect::<Vec<_>>();
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut changes = Vec::new();
    for boundary in boundaries {
        let before = active_evaluation_labels(turns, boundary - 1, reference)?;
        let after = active_evaluation_labels(turns, boundary, reference)?;
        if !before.is_empty() && !after.is_empty() && before != after {
            changes.push(boundary);
        }
    }
    Ok(changes)
}

fn active_evaluation_labels(
    turns: &[EvaluationTurn],
    point_ms: u64,
    reference: bool,
) -> FwResult<BTreeSet<Option<String>>> {
    turns
        .iter()
        .filter(|turn| turn.start_ms <= point_ms && point_ms < turn.end_ms)
        .map(|turn| {
            if reference && turn.speaker.is_none() {
                Err(score_error(
                    "reference_speaker_missing",
                    "reference turn must have a speaker",
                ))
            } else {
                Ok(turn.speaker.clone())
            }
        })
        .collect()
}

fn evaluation_atomic_intervals(
    reference: &DiarizationReferenceDocument,
    hypothesis: &DiarizationHypothesisDocument,
    config: &DiarizationScorerConfig,
    reference_changes: &[u64],
) -> Vec<EvaluationAtomicInterval> {
    let mut boundaries = vec![0, reference.duration_ms];
    boundaries.extend(
        reference
            .turns
            .iter()
            .chain(&hypothesis.turns)
            .flat_map(|turn| [turn.start_ms, turn.end_ms]),
    );
    boundaries.extend(
        reference
            .ignored_regions
            .iter()
            .flat_map(|region| [region.start_ms, region.end_ms]),
    );
    boundaries.extend(
        reference
            .speaker_hints
            .iter()
            .flat_map(|hint| [hint.start_ms, hint.end_ms]),
    );
    if config.speaker_boundary_collar_ms > 0 {
        for change in reference_changes {
            boundaries.push(change.saturating_sub(config.speaker_boundary_collar_ms));
            boundaries.push(
                change
                    .saturating_add(config.speaker_boundary_collar_ms)
                    .min(reference.duration_ms),
            );
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    boundaries
        .windows(2)
        .filter_map(|window| {
            let start_ms = window[0];
            let end_ms = window[1];
            if start_ms >= end_ms {
                return None;
            }
            let point_ms = start_ms + (end_ms - start_ms) / 2;
            let reference_labels = reference
                .turns
                .iter()
                .filter(|turn| turn.start_ms <= point_ms && point_ms < turn.end_ms)
                .filter_map(|turn| turn.speaker.clone())
                .collect::<BTreeSet<_>>();

            let active_hypothesis = hypothesis
                .turns
                .iter()
                .filter(|turn| turn.start_ms <= point_ms && point_ms < turn.end_ms)
                .collect::<Vec<_>>();
            let hypothesis_overlap_suspected =
                active_hypothesis.iter().any(|turn| turn.overlap_suspected);
            let mut hypothesis_by_speaker = BTreeMap::<Option<String>, Option<f64>>::new();
            for turn in active_hypothesis {
                let entry = hypothesis_by_speaker
                    .entry(turn.speaker.clone())
                    .or_insert(turn.speaker_confidence);
                if let Some(confidence) = turn.speaker_confidence
                    && entry.is_none_or(|current| confidence > current)
                {
                    *entry = Some(confidence);
                }
            }
            let hypothesis_states = hypothesis_by_speaker
                .into_iter()
                .map(|(speaker, confidence)| EvaluationHypothesisState {
                    speaker,
                    confidence,
                })
                .collect::<Vec<_>>();
            let in_ignored_region = millisecond_is_ignored(point_ms, &reference.ignored_regions);
            let in_collar = config.speaker_boundary_collar_ms > 0
                && reference_changes.iter().any(|change| {
                    let lower = change.saturating_sub(config.speaker_boundary_collar_ms);
                    let upper = change
                        .saturating_add(config.speaker_boundary_collar_ms)
                        .min(reference.duration_ms);
                    lower <= point_ms && point_ms < upper
                });
            let excluded_overlap = config.overlap_policy == EvaluationOverlapPolicy::Exclude
                && reference_labels.len() > 1;
            Some(EvaluationAtomicInterval {
                start_ms,
                end_ms,
                reference: reference_labels,
                hypothesis: hypothesis_states,
                hypothesis_overlap_suspected,
                excluded: in_ignored_region || in_collar || excluded_overlap,
                overlap_scoring_excluded: in_ignored_region || in_collar,
            })
        })
        .collect()
}

fn millisecond_is_ignored(point_ms: u64, regions: &[EvaluationRegion]) -> bool {
    regions
        .iter()
        .any(|region| region.start_ms <= point_ms && point_ms < region.end_ms)
}

fn scoring_turns_from_atoms(
    atoms: &[EvaluationAtomicInterval],
) -> (Vec<ScoringTurn>, Vec<ScoringTurn>) {
    let mut reference = Vec::new();
    let mut hypothesis = Vec::new();
    for atom in atoms.iter().filter(|atom| !atom.excluded) {
        let start_sec = atom.start_ms as f64 / 1_000.0;
        let end_sec = atom.end_ms as f64 / 1_000.0;
        reference.extend(
            atom.reference
                .iter()
                .map(|speaker| ScoringTurn::labeled(start_sec, end_sec, speaker)),
        );
        hypothesis.extend(atom.hypothesis.iter().map(|state| ScoringTurn {
            start_sec,
            end_sec,
            speaker: state.speaker.clone(),
            overlap_suspected: atom.hypothesis_overlap_suspected,
        }));
    }
    (reference, hypothesis)
}

fn score_speech_activity(atoms: &[EvaluationAtomicInterval]) -> SpeechActivityScore {
    let mut score = SpeechActivityScore {
        reference_speech_sec: 0.0,
        hypothesis_speech_sec: 0.0,
        correct_speech_sec: 0.0,
        missed_speech_sec: 0.0,
        false_alarm_sec: 0.0,
        error_rate: None,
    };
    for atom in atoms.iter().filter(|atom| !atom.excluded) {
        let duration = atom.duration_sec();
        let reference_active = !atom.reference.is_empty();
        let hypothesis_active = !atom.hypothesis.is_empty();
        if reference_active {
            score.reference_speech_sec += duration;
        }
        if hypothesis_active {
            score.hypothesis_speech_sec += duration;
        }
        match (reference_active, hypothesis_active) {
            (true, true) => score.correct_speech_sec += duration,
            (true, false) => score.missed_speech_sec += duration,
            (false, true) => score.false_alarm_sec += duration,
            (false, false) => {}
        }
    }
    score.error_rate = (score.reference_speech_sec > SCORE_EPSILON_SEC)
        .then_some((score.missed_speech_sec + score.false_alarm_sec) / score.reference_speech_sec);
    score
}

fn score_speaker_attribution(diarization: &DiarizationScore) -> SpeakerAttributionScore {
    let attributable_reference_speaker_time_sec =
        (diarization.reference_speaker_time_sec - diarization.missed_speech_sec).max(0.0);
    let correctly_attributed_speaker_time_sec =
        (attributable_reference_speaker_time_sec - diarization.speaker_confusion_sec).max(0.0);
    SpeakerAttributionScore {
        attributable_reference_speaker_time_sec,
        correctly_attributed_speaker_time_sec,
        attribution_error_sec: diarization.speaker_confusion_sec,
        accuracy: (attributable_reference_speaker_time_sec > SCORE_EPSILON_SEC).then_some(
            correctly_attributed_speaker_time_sec / attributable_reference_speaker_time_sec,
        ),
    }
}

fn score_speaker_count(reference: &[ScoringTurn], hypothesis: &[ScoringTurn]) -> SpeakerCountScore {
    let reference_speakers = collect_labels(reference).len();
    let hypothesis_speakers = collect_labels(hypothesis).len();
    let signed_error = i64::try_from(hypothesis_speakers)
        .unwrap_or(i64::MAX)
        .saturating_sub(i64::try_from(reference_speakers).unwrap_or(i64::MAX));
    SpeakerCountScore {
        reference_speakers,
        hypothesis_speakers,
        signed_error,
        absolute_error: signed_error.unsigned_abs(),
    }
}

fn score_speaker_count_posterior(
    reference_speakers: usize,
    estimate: Option<&SpeakerCountEstimate>,
    config: &DiarizationScorerConfig,
) -> SpeakerCountPosteriorScore {
    let Some(estimate) = estimate else {
        return SpeakerCountPosteriorScore {
            reference_speakers,
            posterior_available: false,
            selected_count: None,
            unresolved: true,
            reference_probability: None,
            negative_log_likelihood: None,
            infinite_negative_log_likelihood: false,
            brier_score: None,
            top_k_hit: None,
            credible_set: Vec::new(),
            credible_set_includes_unresolved: false,
            credible_set_hit: None,
            unresolved_probability: None,
            entropy_bits: None,
            calibration_status: None,
        };
    };

    let posterior_available = !estimate.posterior.is_empty()
        && matches!(
            estimate.calibration_status,
            SpeakerCountCalibrationStatus::Certified
                | SpeakerCountCalibrationStatus::DevelopmentUncertified
        );
    let reference_count = u32::try_from(reference_speakers).unwrap_or(u32::MAX);
    let reference_probability = posterior_available.then(|| {
        estimate
            .posterior
            .iter()
            .find(|bin| bin.count == reference_count)
            .map_or(0.0, |bin| bin.probability)
    });
    let negative_log_likelihood = reference_probability
        .and_then(|probability| (probability > 0.0).then(|| -probability.ln()));
    let infinite_negative_log_likelihood =
        reference_probability.is_some_and(|probability| probability == 0.0);
    let brier_score = posterior_available.then(|| {
        let concrete = estimate
            .posterior
            .iter()
            .map(|bin| {
                let target = f64::from(bin.count == reference_count);
                (bin.probability - target).powi(2)
            })
            .sum::<f64>();
        let unsupported_reference_target = f64::from(
            !estimate
                .posterior
                .iter()
                .any(|bin| bin.count == reference_count),
        );
        concrete + unsupported_reference_target + estimate.unresolved_probability.powi(2)
    });

    let mut ranked_bins = estimate.posterior.clone();
    ranked_bins.sort_by(|left, right| {
        right
            .probability
            .total_cmp(&left.probability)
            .then_with(|| left.count.cmp(&right.count))
    });
    let top_k_hit = posterior_available.then(|| {
        ranked_bins
            .iter()
            .take(config.count_top_k)
            .any(|bin| bin.count == reference_count)
    });

    let target_mass = f64::from(config.count_credible_mass_millionths) / 1_000_000.0;
    let mut ranked_mass = ranked_bins
        .iter()
        .map(|bin| (Some(bin.count), bin.probability))
        .collect::<Vec<_>>();
    ranked_mass.push((None, estimate.unresolved_probability));
    ranked_mass.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.is_none().cmp(&right.0.is_none()))
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut credible_set = Vec::new();
    let mut credible_set_includes_unresolved = false;
    let mut accumulated_mass = 0.0;
    if posterior_available {
        for (count, probability) in ranked_mass {
            if accumulated_mass >= target_mass {
                break;
            }
            accumulated_mass += probability;
            if let Some(count) = count {
                credible_set.push(count);
            } else {
                credible_set_includes_unresolved = true;
            }
        }
        credible_set.sort_unstable();
    }
    let credible_set_hit =
        posterior_available.then(|| credible_set.binary_search(&reference_count).is_ok());

    SpeakerCountPosteriorScore {
        reference_speakers,
        posterior_available,
        selected_count: estimate.selected_count,
        unresolved: estimate.selected_count.is_none(),
        reference_probability,
        negative_log_likelihood,
        infinite_negative_log_likelihood,
        brier_score,
        top_k_hit,
        credible_set,
        credible_set_includes_unresolved,
        credible_set_hit,
        unresolved_probability: Some(estimate.unresolved_probability),
        entropy_bits: Some(estimate.entropy_bits),
        calibration_status: Some(estimate.calibration_status),
    }
}

fn score_speaker_occupancy(
    hypothesis: &DiarizationHypothesisDocument,
    atoms: &[EvaluationAtomicInterval],
    mapping: &BTreeMap<String, String>,
    config: &DiarizationScorerConfig,
) -> SpeakerOccupancyScore {
    let hypothesis_labels = hypothesis
        .turns
        .iter()
        .filter_map(|turn| turn.speaker.clone())
        .collect::<BTreeSet<_>>();
    let mut hypothesis_duration = hypothesis_labels
        .iter()
        .cloned()
        .map(|speaker| (speaker, 0.0_f64))
        .collect::<BTreeMap<_, _>>();
    let reference_labels = atoms
        .iter()
        .filter(|atom| !atom.excluded)
        .flat_map(|atom| atom.reference.iter().cloned())
        .collect::<BTreeSet<_>>();
    let mut reference_duration = reference_labels
        .iter()
        .cloned()
        .map(|speaker| (speaker, 0.0_f64))
        .collect::<BTreeMap<_, _>>();
    let mut correct_by_reference = reference_labels
        .iter()
        .cloned()
        .map(|speaker| (speaker, 0.0_f64))
        .collect::<BTreeMap<_, _>>();
    let mut unknown_speaker_time_sec = 0.0;
    for atom in atoms.iter().filter(|atom| !atom.excluded) {
        let duration = atom.duration_sec();
        for reference_speaker in &atom.reference {
            if let Some(total) = reference_duration.get_mut(reference_speaker) {
                *total += duration;
            }
        }
        for state in &atom.hypothesis {
            let Some(hypothesis_speaker) = state.speaker.as_ref() else {
                unknown_speaker_time_sec += duration;
                continue;
            };
            if let Some(total) = hypothesis_duration.get_mut(hypothesis_speaker) {
                *total += duration;
            }
            if let Some(reference_speaker) = mapping.get(hypothesis_speaker)
                && atom.reference.contains(reference_speaker)
                && let Some(correct) = correct_by_reference.get_mut(reference_speaker)
            {
                *correct += duration;
            }
        }
    }

    let labeled_speaker_time_sec = hypothesis_duration.values().sum::<f64>();
    let minimum_effective_occupancy_sec = config.minimum_effective_occupancy_ms as f64 / 1_000.0;
    let mut speakers = hypothesis_duration
        .iter()
        .map(|(speaker, &duration)| {
            let labeled_share = if labeled_speaker_time_sec > SCORE_EPSILON_SEC {
                duration / labeled_speaker_time_sec
            } else {
                0.0
            };
            SpeakerOccupancyEntry {
                hypothesis_speaker: speaker.clone(),
                mapped_reference_speaker: mapping.get(speaker).cloned(),
                voiced_duration_sec: duration,
                labeled_share,
                recurrence_episode_count: recurrence_episode_count(atoms, speaker),
                effective: duration + SCORE_EPSILON_SEC >= minimum_effective_occupancy_sec,
            }
        })
        .collect::<Vec<_>>();
    speakers.sort_by(|left, right| left.hypothesis_speaker.cmp(&right.hypothesis_speaker));
    let dominant_speaker_share = speakers
        .iter()
        .map(|speaker| speaker.labeled_share)
        .max_by(f64::total_cmp)
        .filter(|_| labeled_speaker_time_sec > SCORE_EPSILON_SEC);
    let dominant_threshold =
        f64::from(config.dominant_speaker_collapse_share_millionths) / 1_000_000.0;
    let dominant_collapse_detected = reference_labels.len() > 1
        && dominant_speaker_share.is_some_and(|share| share >= dominant_threshold);
    let minimum_reference_recall =
        f64::from(config.minimum_reference_speaker_recall_millionths) / 1_000_000.0;
    let reference_recalls = reference_duration
        .iter()
        .map(|(speaker, &duration)| {
            let correct = correct_by_reference.get(speaker).copied().unwrap_or(0.0);
            (
                speaker,
                ratio_or_none(correct, duration).unwrap_or(0.0),
                duration,
            )
        })
        .collect::<Vec<_>>();
    let collapsed_reference_speaker_count = reference_recalls
        .iter()
        .filter(|(_, recall, _)| *recall < minimum_reference_recall)
        .count();
    let minority_reference_recall = reference_recalls
        .iter()
        .min_by(|left, right| {
            left.2
                .total_cmp(&right.2)
                .then_with(|| left.1.total_cmp(&right.1))
                .then_with(|| left.0.cmp(right.0))
        })
        .map(|(_, recall, _)| *recall);

    SpeakerOccupancyScore {
        labeled_speaker_time_sec,
        unknown_speaker_time_sec,
        dominant_speaker_share,
        unknown_speaker_share: ratio_or_none(
            unknown_speaker_time_sec,
            labeled_speaker_time_sec + unknown_speaker_time_sec,
        ),
        effective_speaker_count: speakers.iter().filter(|speaker| speaker.effective).count(),
        phantom_speaker_count: speakers
            .iter()
            .filter(|speaker| speaker.effective && speaker.mapped_reference_speaker.is_none())
            .count(),
        collapsed_reference_speaker_count,
        minority_reference_recall,
        dominant_collapse_detected,
        any_reference_collapse_detected: collapsed_reference_speaker_count > 0,
        speakers,
    }
}

fn recurrence_episode_count(atoms: &[EvaluationAtomicInterval], speaker: &str) -> u64 {
    let mut episodes = 0_u64;
    let mut previously_active = false;
    for atom in atoms {
        let active = !atom.excluded
            && atom
                .hypothesis
                .iter()
                .any(|state| state.speaker.as_deref() == Some(speaker));
        if active && !previously_active {
            episodes = episodes.saturating_add(1);
        }
        previously_active = active;
    }
    episodes
}

fn score_word_attribution(
    reference: &DiarizationReferenceDocument,
    hypothesis: &DiarizationHypothesisDocument,
    mapping: &BTreeMap<String, String>,
    config: &DiarizationScorerConfig,
    reference_changes: &[u64],
) -> WordAttributionScore {
    let mut score = WordAttributionScore {
        reference_word_count: u64::try_from(reference.words.len()).unwrap_or(u64::MAX),
        scored_word_count: 0,
        correct_word_count: 0,
        incorrect_word_count: 0,
        unknown_word_count: 0,
        excluded_word_count: 0,
        word_diarization_error_rate: None,
    };
    for word in &reference.words {
        let point_ms = word.start_ms + (word.end_ms - word.start_ms) / 2;
        let reference_overlap = reference
            .turns
            .iter()
            .filter(|turn| turn.start_ms <= point_ms && point_ms < turn.end_ms)
            .filter_map(|turn| turn.speaker.as_ref())
            .collect::<BTreeSet<_>>()
            .len()
            > 1;
        let in_collar = config.speaker_boundary_collar_ms > 0
            && reference_changes.iter().any(|change| {
                let lower = change.saturating_sub(config.speaker_boundary_collar_ms);
                let upper = change
                    .saturating_add(config.speaker_boundary_collar_ms)
                    .min(reference.duration_ms);
                lower <= point_ms && point_ms < upper
            });
        if millisecond_is_ignored(point_ms, &reference.ignored_regions)
            || in_collar
            || (config.overlap_policy == EvaluationOverlapPolicy::Exclude && reference_overlap)
        {
            score.excluded_word_count = score.excluded_word_count.saturating_add(1);
            continue;
        }
        score.scored_word_count = score.scored_word_count.saturating_add(1);
        let active_mapped = hypothesis
            .turns
            .iter()
            .filter(|turn| turn.start_ms <= point_ms && point_ms < turn.end_ms)
            .filter_map(|turn| turn.speaker.as_ref())
            .filter_map(|speaker| mapping.get(speaker))
            .collect::<BTreeSet<_>>();
        if active_mapped.contains(&word.speaker_ref) {
            score.correct_word_count = score.correct_word_count.saturating_add(1);
        } else if active_mapped.is_empty() {
            score.unknown_word_count = score.unknown_word_count.saturating_add(1);
        } else {
            score.incorrect_word_count = score.incorrect_word_count.saturating_add(1);
        }
    }
    score.word_diarization_error_rate = ratio_or_none(
        (score.incorrect_word_count + score.unknown_word_count) as f64,
        score.scored_word_count as f64,
    );
    score
}

fn score_overlap_detection(atoms: &[EvaluationAtomicInterval]) -> OverlapDetectionScore {
    let mut reference_overlap_sec = 0.0;
    let mut hypothesis_overlap_sec = 0.0;
    let mut true_positive_sec = 0.0;
    let mut false_positive_sec = 0.0;
    let mut false_negative_sec = 0.0;
    for atom in atoms.iter().filter(|atom| !atom.overlap_scoring_excluded) {
        let duration = atom.duration_sec();
        let reference_overlap = atom.reference.len() > 1;
        let hypothesis_overlap = atom.hypothesis.len() > 1 || atom.hypothesis_overlap_suspected;
        if reference_overlap {
            reference_overlap_sec += duration;
        }
        if hypothesis_overlap {
            hypothesis_overlap_sec += duration;
        }
        match (reference_overlap, hypothesis_overlap) {
            (true, true) => true_positive_sec += duration,
            (false, true) => false_positive_sec += duration,
            (true, false) => false_negative_sec += duration,
            (false, false) => {}
        }
    }
    let precision = ratio_or_none(true_positive_sec, true_positive_sec + false_positive_sec);
    let recall = ratio_or_none(true_positive_sec, true_positive_sec + false_negative_sec);
    let f1 = ratio_or_none(
        2.0 * true_positive_sec,
        2.0 * true_positive_sec + false_positive_sec + false_negative_sec,
    );
    OverlapDetectionScore {
        reference_overlap_sec,
        hypothesis_overlap_sec,
        true_positive_sec,
        false_positive_sec,
        false_negative_sec,
        precision,
        recall,
        f1,
    }
}

fn score_hint_adherence(
    atoms: &[EvaluationAtomicInterval],
    hints: &[EvaluationSpeakerHint],
    mapping: &BTreeMap<String, String>,
) -> HintAdherenceScore {
    let mut score = HintAdherenceScore {
        hinted_sec: 0.0,
        adherent_sec: 0.0,
        contradictory_sec: 0.0,
        unknown_sec: 0.0,
        hard_violation_sec: 0.0,
        adherence_rate: None,
    };
    for hint in hints {
        for atom in atoms.iter().filter(|atom| {
            !atom.excluded && atom.start_ms < hint.end_ms && hint.start_ms < atom.end_ms
        }) {
            let overlap_start = atom.start_ms.max(hint.start_ms);
            let overlap_end = atom.end_ms.min(hint.end_ms);
            let duration = (overlap_end - overlap_start) as f64 / 1_000.0;
            score.hinted_sec += duration;
            let mapped = atom
                .hypothesis
                .iter()
                .filter_map(|state| state.speaker.as_ref())
                .filter_map(|speaker| mapping.get(speaker))
                .collect::<BTreeSet<_>>();
            if mapped.contains(&hint.speaker_ref) {
                score.adherent_sec += duration;
            } else if mapped.is_empty() {
                score.unknown_sec += duration;
                if hint.policy == EvaluationHintPolicy::Hard {
                    score.hard_violation_sec += duration;
                }
            } else {
                score.contradictory_sec += duration;
                if hint.policy == EvaluationHintPolicy::Hard {
                    score.hard_violation_sec += duration;
                }
            }
        }
    }
    score.adherence_rate = ratio_or_none(score.adherent_sec, score.hinted_sec);
    score
}

fn score_selective_attribution(
    atoms: &[EvaluationAtomicInterval],
    mapping: &BTreeMap<String, String>,
) -> SelectiveAttributionScore {
    let mut reference_speaker_time_sec = 0.0;
    let mut covered_speaker_time_sec = 0.0;
    let mut correct_covered_speaker_time_sec = 0.0;
    for atom in atoms.iter().filter(|atom| !atom.excluded) {
        let duration = atom.duration_sec();
        let reference_count = atom.reference.len();
        let mapped = atom
            .hypothesis
            .iter()
            .filter_map(|state| state.speaker.as_ref())
            .filter_map(|speaker| mapping.get(speaker))
            .collect::<BTreeSet<_>>();
        let labeled_hypothesis_count = atom
            .hypothesis
            .iter()
            .filter(|state| state.speaker.is_some())
            .count();
        let covered_count = reference_count.min(labeled_hypothesis_count);
        let correct_count = mapped
            .iter()
            .filter(|speaker| atom.reference.contains(**speaker))
            .count()
            .min(covered_count);
        reference_speaker_time_sec += duration * reference_count as f64;
        covered_speaker_time_sec += duration * covered_count as f64;
        correct_covered_speaker_time_sec += duration * correct_count as f64;
    }
    let error_covered_speaker_time_sec =
        (covered_speaker_time_sec - correct_covered_speaker_time_sec).max(0.0);
    SelectiveAttributionScore {
        reference_speaker_time_sec,
        covered_speaker_time_sec,
        correct_covered_speaker_time_sec,
        error_covered_speaker_time_sec,
        unknown_speaker_time_sec: (reference_speaker_time_sec - covered_speaker_time_sec).max(0.0),
        coverage: ratio_or_none(covered_speaker_time_sec, reference_speaker_time_sec),
        selective_risk: ratio_or_none(error_covered_speaker_time_sec, covered_speaker_time_sec),
    }
}

fn score_weighted_calibration(
    atoms: &[EvaluationAtomicInterval],
    mapping: &BTreeMap<String, String>,
    bins: usize,
) -> WeightedCalibrationScore {
    let mut observations = Vec::new();
    let mut opportunity_duration_sec = 0.0;
    for atom in atoms.iter().filter(|atom| !atom.excluded) {
        let duration = atom.duration_sec();
        for state in atom
            .hypothesis
            .iter()
            .filter(|state| state.speaker.is_some())
        {
            opportunity_duration_sec += duration;
            if let (Some(speaker), Some(confidence)) = (&state.speaker, state.confidence) {
                let correct = mapping
                    .get(speaker)
                    .is_some_and(|reference| atom.reference.contains(reference));
                observations.push(WeightedCalibrationObservation {
                    confidence,
                    correct,
                    duration_sec: duration,
                });
            }
        }
    }
    let observed_duration_sec = observations
        .iter()
        .map(|observation| observation.duration_sec)
        .sum::<f64>();
    if observed_duration_sec <= SCORE_EPSILON_SEC {
        return WeightedCalibrationScore {
            observed_duration_sec: 0.0,
            opportunity_duration_sec,
            coverage: ratio_or_none(0.0, opportunity_duration_sec),
            brier_score: None,
            expected_calibration_error: None,
            bins,
        };
    }

    let mut brier_sum = 0.0;
    let mut bin_weights = vec![0.0_f64; bins];
    let mut bin_confidence = vec![0.0_f64; bins];
    let mut bin_correct = vec![0.0_f64; bins];
    for observation in &observations {
        let outcome = f64::from(observation.correct);
        brier_sum += (observation.confidence - outcome).powi(2) * observation.duration_sec;
        let bin = ((observation.confidence * bins as f64).floor() as usize).min(bins - 1);
        bin_weights[bin] += observation.duration_sec;
        bin_confidence[bin] += observation.confidence * observation.duration_sec;
        bin_correct[bin] += outcome * observation.duration_sec;
    }
    let mut expected_calibration_error = 0.0;
    for bin in 0..bins {
        if bin_weights[bin] <= SCORE_EPSILON_SEC {
            continue;
        }
        let mean_confidence = bin_confidence[bin] / bin_weights[bin];
        let accuracy = bin_correct[bin] / bin_weights[bin];
        expected_calibration_error +=
            bin_weights[bin] / observed_duration_sec * (accuracy - mean_confidence).abs();
    }
    WeightedCalibrationScore {
        observed_duration_sec,
        opportunity_duration_sec,
        coverage: ratio_or_none(observed_duration_sec, opportunity_duration_sec),
        brier_score: Some(brier_sum / observed_duration_sec),
        expected_calibration_error: Some(expected_calibration_error),
        bins,
    }
}

fn ratio_or_none(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > SCORE_EPSILON_SEC).then_some(numerator / denominator)
}

fn ordered_splits(
    left: EvaluationSplit,
    right: EvaluationSplit,
) -> (EvaluationSplit, EvaluationSplit) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn sorted_intersection(left: &[String], right: &[String]) -> Vec<String> {
    let left = left.iter().cloned().collect::<BTreeSet<_>>();
    let right = right.iter().cloned().collect::<BTreeSet<_>>();
    sorted_set_intersection(&left, &right)
}

fn sorted_set_intersection(left: &BTreeSet<String>, right: &BTreeSet<String>) -> Vec<String> {
    left.intersection(right).cloned().collect()
}

fn lineage_ids(recording: &CorpusRecordingManifest) -> BTreeSet<String> {
    recording
        .derived_from_recording_ids
        .iter()
        .cloned()
        .collect()
}

fn validate_turns(turns: &[ScoringTurn], reference: bool) -> FwResult<()> {
    for (index, turn) in turns.iter().enumerate() {
        if !turn.start_sec.is_finite() || !turn.end_sec.is_finite() {
            return Err(FwError::InvalidRequest(format!(
                "diarization turn {index} has non-finite timestamp"
            )));
        }
        if turn.start_sec < 0.0 || turn.end_sec <= turn.start_sec {
            return Err(FwError::InvalidRequest(format!(
                "diarization turn {index} must satisfy 0 <= start < end"
            )));
        }
        if let Some(speaker) = &turn.speaker
            && speaker.trim().is_empty()
        {
            return Err(FwError::InvalidRequest(format!(
                "diarization turn {index} has an empty speaker label"
            )));
        }
        if reference && turn.speaker.is_none() {
            return Err(FwError::InvalidRequest(format!(
                "reference diarization turn {index} must have a speaker label"
            )));
        }
    }
    Ok(())
}

fn validate_points(points: &[f64], kind: &str) -> FwResult<()> {
    if points
        .iter()
        .any(|point| !point.is_finite() || *point < 0.0)
    {
        return Err(FwError::InvalidRequest(format!(
            "{kind} change points must be finite and non-negative"
        )));
    }
    Ok(())
}

fn collect_labels(turns: &[ScoringTurn]) -> Vec<String> {
    turns
        .iter()
        .filter_map(|turn| turn.speaker.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn label_index(labels: &[String]) -> BTreeMap<String, usize> {
    labels
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, label)| (label, index))
        .collect()
}

fn atomic_intervals(
    reference: &[ScoringTurn],
    hypothesis: &[ScoringTurn],
    reference_index: &BTreeMap<String, usize>,
    hypothesis_index: &BTreeMap<String, usize>,
) -> Vec<AtomicInterval> {
    let mut boundaries = reference
        .iter()
        .chain(hypothesis)
        .flat_map(|turn| [turn.start_sec, turn.end_sec])
        .collect::<Vec<_>>();
    boundaries.sort_by(f64::total_cmp);
    boundaries.dedup_by(|left, right| (*left - *right).abs() <= SCORE_EPSILON_SEC);

    boundaries
        .windows(2)
        .filter_map(|window| {
            let start = window[0];
            let end = window[1];
            if end - start <= SCORE_EPSILON_SEC {
                return None;
            }
            let midpoint = start + (end - start) * 0.5;
            let active_reference = reference
                .iter()
                .filter(|turn| turn.start_sec <= midpoint && midpoint < turn.end_sec)
                .filter_map(|turn| {
                    turn.speaker
                        .as_ref()
                        .and_then(|label| reference_index.get(label).copied())
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let active_hypothesis = hypothesis
                .iter()
                .filter(|turn| turn.start_sec <= midpoint && midpoint < turn.end_sec)
                .map(|turn| {
                    turn.speaker
                        .as_ref()
                        .and_then(|label| hypothesis_index.get(label).copied())
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            Some(AtomicInterval {
                duration: end - start,
                reference: active_reference,
                hypothesis: active_hypothesis,
            })
        })
        .collect()
}

fn minimum_error_change_match(
    reference: &[f64],
    hypothesis: &[f64],
    collar_sec: f64,
) -> ChangeMatch {
    let mut previous = vec![ChangeMatch::default(); hypothesis.len() + 1];
    let mut current = vec![ChangeMatch::default(); hypothesis.len() + 1];
    for reference_index in 1..=reference.len() {
        current[0] = ChangeMatch::default();
        for hypothesis_index in 1..=hypothesis.len() {
            let mut best =
                better_change_match(previous[hypothesis_index], current[hypothesis_index - 1]);
            let error = (reference[reference_index - 1] - hypothesis[hypothesis_index - 1]).abs();
            if error <= collar_sec {
                let matched = ChangeMatch {
                    count: previous[hypothesis_index - 1].count + 1,
                    total_error: previous[hypothesis_index - 1].total_error + error,
                };
                best = better_change_match(best, matched);
            }
            current[hypothesis_index] = best;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[hypothesis.len()]
}

fn better_change_match(left: ChangeMatch, right: ChangeMatch) -> ChangeMatch {
    if right.count > left.count
        || (right.count == left.count && right.total_error < left.total_error)
    {
        right
    } else {
        left
    }
}

fn maximum_overlap_mapping(weights: &[Vec<f64>], hypothesis_count: usize) -> Vec<Option<usize>> {
    let reference_count = weights.len();
    if hypothesis_count == 0 {
        return Vec::new();
    }
    if reference_count == 0 {
        return vec![None; hypothesis_count];
    }

    let size = reference_count.max(hypothesis_count);
    let max_weight = weights.iter().flatten().copied().fold(0.0_f64, f64::max);
    let mut cost = vec![vec![max_weight; size]; size];
    for (reference_id, row) in weights.iter().enumerate() {
        for (hypothesis_id, weight) in row.iter().enumerate() {
            cost[reference_id][hypothesis_id] = max_weight - weight;
        }
    }

    // Hungarian assignment for a square minimization matrix. `p[column]`
    // stores the assigned row; stable ascending scans provide deterministic
    // tie-breaking for equal-overlap mappings.
    let mut row_potential = vec![0.0_f64; size + 1];
    let mut column_potential = vec![0.0_f64; size + 1];
    let mut assigned_row = vec![0usize; size + 1];
    let mut predecessor = vec![0usize; size + 1];
    for row in 1..=size {
        assigned_row[0] = row;
        let mut column = 0usize;
        let mut minimum = vec![f64::INFINITY; size + 1];
        let mut used = vec![false; size + 1];
        loop {
            used[column] = true;
            let current_row = assigned_row[column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0usize;
            for candidate in 1..=size {
                if used[candidate] {
                    continue;
                }
                let reduced = cost[current_row - 1][candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate];
                if reduced < minimum[candidate] {
                    minimum[candidate] = reduced;
                    predecessor[candidate] = column;
                }
                if minimum[candidate] < delta {
                    delta = minimum[candidate];
                    next_column = candidate;
                }
            }
            for candidate in 0..=size {
                if used[candidate] {
                    row_potential[assigned_row[candidate]] += delta;
                    column_potential[candidate] -= delta;
                } else {
                    minimum[candidate] -= delta;
                }
            }
            column = next_column;
            if assigned_row[column] == 0 {
                break;
            }
        }
        loop {
            let previous = predecessor[column];
            assigned_row[column] = assigned_row[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }

    (1..=hypothesis_count)
        .map(|column| {
            let row = assigned_row[column];
            (row > 0 && row <= reference_count).then_some(row - 1)
        })
        .collect()
}

fn score_jer(
    intervals: &[AtomicInterval],
    reference_count: usize,
    hypothesis_count: usize,
    hypothesis_to_reference: &[Option<usize>],
) -> Option<f64> {
    if reference_count == 0 {
        return None;
    }
    let mut total_error = 0.0;
    for reference_id in 0..reference_count {
        let mapped_hypothesis = (0..hypothesis_count)
            .find(|&hypothesis_id| hypothesis_to_reference[hypothesis_id] == Some(reference_id));
        let mut intersection = 0.0;
        let mut union = 0.0;
        for interval in intervals {
            let reference_active = interval.reference.contains(&reference_id);
            let hypothesis_active = mapped_hypothesis
                .is_some_and(|hypothesis_id| interval.hypothesis.contains(&Some(hypothesis_id)));
            if reference_active && hypothesis_active {
                intersection += interval.duration;
            }
            if reference_active || hypothesis_active {
                union += interval.duration;
            }
        }
        let error = if union > SCORE_EPSILON_SEC {
            1.0 - intersection / union
        } else {
            1.0
        };
        total_error += error;
    }
    Some(total_error / reference_count as f64)
}

/// Stable identity for the default native acoustic feature layout.
pub const ACOUSTIC_FEATURE_SCHEMA_VERSION: &str = "acoustic-feature-v2";
/// Stable identity for the original compact representation.
pub const ACOUSTIC_FEATURE_SCHEMA_V1: &str = "acoustic-feature-v1";
/// Fixed analysis cadence shared with the native Whisper frontend.
pub const ACOUSTIC_FRAME_SAMPLES: usize = crate::native_engine::mel::N_FFT;
/// Fixed frame advance shared with the native Whisper frontend.
pub const ACOUSTIC_HOP_SAMPLES: usize = crate::native_engine::mel::HOP;
/// Maximum number of frames between cancellation checks.
pub const ACOUSTIC_CANCELLATION_INTERVAL_FRAMES: usize = 32;

const ENVELOPE_BANDS: usize = 12;
const CEPSTRAL_COEFFICIENTS: usize = 12;
pub const VOICE_VECTOR_DIMENSIONS: usize = 28;
pub const CHANNEL_VECTOR_DIMENSIONS: usize = 14;
const POWER_EPSILON: f32 = 1e-20;
const PCM_EPSILON: f32 = 1e-12;
const MAX_ABS_ACOUSTIC_FEATURE: f32 = 1_000_000.0;
const MAX_ACOUSTIC_VARIANCE: f32 = MAX_ABS_ACOUSTIC_FEATURE * MAX_ABS_ACOUSTIC_FEATURE;
const MAX_IDENTITY_SUBWINDOWS: usize = 64;

/// Explicit acoustic representation selected for one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticFeatureSchemaVersion {
    /// Original six-cepstrum compact representation. This is fallback-only.
    V1,
    /// Rich representation with validity masks and robust call normalization.
    V2,
}

impl AcousticFeatureSchemaVersion {
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::V1 => ACOUSTIC_FEATURE_SCHEMA_V1,
            Self::V2 => ACOUSTIC_FEATURE_SCHEMA_VERSION,
        }
    }
}

/// One frozen acoustic representation ablation.
///
/// These variants are an evaluation surface, not adaptive runtime choices.
/// Every run records the selected ID so results cannot silently mix feature
/// families.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcousticFeatureAblation {
    #[default]
    FullV2,
    NoPitch,
    NoChannel,
    NoDeltas,
    NoModulation,
    V1,
}

impl AcousticFeatureAblation {
    pub const ALL: [Self; 6] = [
        Self::FullV2,
        Self::NoPitch,
        Self::NoChannel,
        Self::NoDeltas,
        Self::NoModulation,
        Self::V1,
    ];

    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::FullV2 => "full_v2",
            Self::NoPitch => "no_pitch",
            Self::NoChannel => "no_channel",
            Self::NoDeltas => "no_deltas",
            Self::NoModulation => "no_modulation",
            Self::V1 => "v1",
        }
    }

    #[must_use]
    pub const fn schema_version(self) -> AcousticFeatureSchemaVersion {
        match self {
            Self::V1 => AcousticFeatureSchemaVersion::V1,
            Self::FullV2
            | Self::NoPitch
            | Self::NoChannel
            | Self::NoDeltas
            | Self::NoModulation => AcousticFeatureSchemaVersion::V2,
        }
    }
}

/// Ownership of an acoustic coordinate family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticFeatureOwner {
    Voice,
    Channel,
}

/// Declarative coordinate range in a versioned acoustic representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcousticFeatureFamily {
    pub name: &'static str,
    pub owner: AcousticFeatureOwner,
    pub start_dimension: usize,
    pub end_dimension_exclusive: usize,
    pub unit: &'static str,
    pub validity: &'static str,
    pub normalization: &'static str,
}

/// Complete public description of one acoustic representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcousticFeatureSchema {
    pub version: AcousticFeatureSchemaVersion,
    pub voice_dimensions: usize,
    pub channel_dimensions: usize,
    pub frame_samples: usize,
    pub hop_samples: usize,
    pub families: &'static [AcousticFeatureFamily],
}

const V1_FEATURE_FAMILIES: &[AcousticFeatureFamily] = &[
    AcousticFeatureFamily {
        name: "cepstral_envelope",
        owner: AcousticFeatureOwner::Voice,
        start_dimension: 0,
        end_dimension_exclusive: 6,
        unit: "log-energy-dct",
        validity: "non-low-energy",
        normalization: "none",
    },
    AcousticFeatureFamily {
        name: "log_f0",
        owner: AcousticFeatureOwner::Voice,
        start_dimension: 6,
        end_dimension_exclusive: 7,
        unit: "log-hz",
        validity: "reliable-pitch-mask",
        normalization: "divide-by-6",
    },
    AcousticFeatureFamily {
        name: "harmonicity",
        owner: AcousticFeatureOwner::Voice,
        start_dimension: 7,
        end_dimension_exclusive: 8,
        unit: "correlation",
        validity: "non-low-energy",
        normalization: "none",
    },
    AcousticFeatureFamily {
        name: "channel_summary",
        owner: AcousticFeatureOwner::Channel,
        start_dimension: 0,
        end_dimension_exclusive: 8,
        unit: "mixed-declared-v1",
        validity: "non-low-energy",
        normalization: "fixed-physical-range",
    },
];

const V2_FEATURE_FAMILIES: &[AcousticFeatureFamily] = &[
    AcousticFeatureFamily {
        name: "cepstral_envelope",
        owner: AcousticFeatureOwner::Voice,
        start_dimension: 0,
        end_dimension_exclusive: 12,
        unit: "log-filterbank-dct",
        validity: "high-information-voiced",
        normalization: "equal-tracklet-median-mad",
    },
    AcousticFeatureFamily {
        name: "cepstral_delta",
        owner: AcousticFeatureOwner::Voice,
        start_dimension: 12,
        end_dimension_exclusive: 16,
        unit: "log-filterbank-dct-per-hop",
        validity: "high-information-voiced-with-history",
        normalization: "equal-tracklet-median-mad",
    },
    AcousticFeatureFamily {
        name: "cepstral_delta_delta",
        owner: AcousticFeatureOwner::Voice,
        start_dimension: 16,
        end_dimension_exclusive: 20,
        unit: "log-filterbank-dct-per-hop2",
        validity: "high-information-voiced-with-two-frame-history",
        normalization: "equal-tracklet-median-mad",
    },
    AcousticFeatureFamily {
        name: "log_f0",
        owner: AcousticFeatureOwner::Voice,
        start_dimension: 20,
        end_dimension_exclusive: 21,
        unit: "natural-log-hz",
        validity: "reliable-pitch-mask",
        normalization: "equal-tracklet-median-mad",
    },
    AcousticFeatureFamily {
        name: "periodicity_hnr",
        owner: AcousticFeatureOwner::Voice,
        start_dimension: 21,
        end_dimension_exclusive: 23,
        unit: "correlation-and-db",
        validity: "high-information-voiced",
        normalization: "equal-tracklet-median-mad",
    },
    AcousticFeatureFamily {
        name: "formant_proxies",
        owner: AcousticFeatureOwner::Voice,
        start_dimension: 23,
        end_dimension_exclusive: 26,
        unit: "nyquist-fraction",
        validity: "high-information-voiced",
        normalization: "equal-tracklet-median-mad",
    },
    AcousticFeatureFamily {
        name: "pitch_uncertainty",
        owner: AcousticFeatureOwner::Voice,
        start_dimension: 26,
        end_dimension_exclusive: 27,
        unit: "octaves",
        validity: "reliable-pitch-mask",
        normalization: "equal-tracklet-median-mad",
    },
    AcousticFeatureFamily {
        name: "temporal_modulation",
        owner: AcousticFeatureOwner::Voice,
        start_dimension: 27,
        end_dimension_exclusive: 28,
        unit: "mean-absolute-dct-delta",
        validity: "high-information-voiced-with-history",
        normalization: "equal-tracklet-median-mad",
    },
    AcousticFeatureFamily {
        name: "channel_summary",
        owner: AcousticFeatureOwner::Channel,
        start_dimension: 0,
        end_dimension_exclusive: 14,
        unit: "declared-mixed-channel-v2",
        validity: "usable-speech",
        normalization: "equal-tracklet-median-mad",
    },
];

/// Return the exact dimensions, units, validity, normalization, and ownership
/// contract for an acoustic representation.
#[must_use]
pub const fn acoustic_feature_schema(
    version: AcousticFeatureSchemaVersion,
) -> AcousticFeatureSchema {
    match version {
        AcousticFeatureSchemaVersion::V1 => AcousticFeatureSchema {
            version,
            voice_dimensions: 8,
            channel_dimensions: 8,
            frame_samples: ACOUSTIC_FRAME_SAMPLES,
            hop_samples: ACOUSTIC_HOP_SAMPLES,
            families: V1_FEATURE_FAMILIES,
        },
        AcousticFeatureSchemaVersion::V2 => AcousticFeatureSchema {
            version,
            voice_dimensions: VOICE_VECTOR_DIMENSIONS,
            channel_dimensions: CHANNEL_VECTOR_DIMENSIONS,
            frame_samples: ACOUSTIC_FRAME_SAMPLES,
            hop_samples: ACOUSTIC_HOP_SAMPLES,
            families: V2_FEATURE_FAMILIES,
        },
    }
}

/// Canonical SHA-256 of the declarative feature schema.
#[must_use]
pub fn acoustic_feature_schema_sha256(version: AcousticFeatureSchemaVersion) -> String {
    let schema = acoustic_feature_schema(version);
    let mut hasher = Sha256::new();
    hasher.update(b"acoustic-feature-schema\0");
    hasher.update(schema.version.id().as_bytes());
    hasher.update((schema.voice_dimensions as u64).to_le_bytes());
    hasher.update((schema.channel_dimensions as u64).to_le_bytes());
    hasher.update((schema.frame_samples as u64).to_le_bytes());
    hasher.update((schema.hop_samples as u64).to_le_bytes());
    for family in schema.families {
        for value in [
            family.name,
            family.unit,
            family.validity,
            family.normalization,
        ] {
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        hasher.update([match family.owner {
            AcousticFeatureOwner::Voice => 0,
            AcousticFeatureOwner::Channel => 1,
        }]);
        hasher.update((family.start_dimension as u64).to_le_bytes());
        hasher.update((family.end_dimension_exclusive as u64).to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Vocal-source and vocal-tract evidence.
///
/// Pitch is deliberately nullable and is never converted into a demographic
/// label. The cepstral envelope excludes absolute energy so it can remain
/// useful when one person moves relative to the microphone.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceFeatureView {
    pub cepstral_envelope: [f32; CEPSTRAL_COEFFICIENTS],
    pub cepstral_delta: [f32; CEPSTRAL_COEFFICIENTS],
    pub cepstral_delta_delta: [f32; CEPSTRAL_COEFFICIENTS],
    pub f0_hz: Option<f32>,
    pub pitch_uncertainty_octaves: Option<f32>,
    pub voicing_confidence: f32,
    pub harmonicity: f32,
    pub harmonic_to_noise_db: f32,
    pub formant_proxies_hz: [f32; 3],
    pub temporal_modulation: f32,
    pub voiced_fraction: f32,
}

/// Channel, distance, loudness, and degradation evidence.
///
/// This view is kept separate from [`VoiceFeatureView`] so a speakerphone or
/// microphone signature can help within one recording without becoming a
/// reusable voice identity.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelFeatureView {
    pub rms_dbfs: f32,
    pub dynamics_above_noise_db: f32,
    pub spectral_centroid_hz: f32,
    pub spectral_bandwidth_hz: f32,
    pub spectral_rolloff_hz: f32,
    pub spectral_flatness: f32,
    pub spectral_tilt: f32,
    pub low_band_fraction: f32,
    pub mid_band_fraction: f32,
    pub high_band_fraction: f32,
    pub crest_factor: f32,
    pub clipping_fraction: f32,
    pub noise_floor_dbfs: f32,
    pub spectral_flux: f32,
    pub distortion_proxy: f32,
    pub effective_band_limit_hz: f32,
    pub high_frequency_attenuation: f32,
    pub reverberation_proxy: f32,
    pub muffling_proxy: f32,
    pub stationary_coloration: f32,
}

/// Per-frame conditions that downstream stages must account for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcousticQualityMask {
    pub voiced: bool,
    pub reliable_pitch: bool,
    pub low_energy: bool,
    pub clipped: bool,
    pub transient: bool,
}

/// One bounded acoustic observation produced at the shared 10 ms cadence.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticFrameFeatures {
    pub frame_index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub voice: VoiceFeatureView,
    pub channel: ChannelFeatureView,
    /// Conservative dual-periodicity evidence; it is not a speaker label.
    pub overlap_probability: f32,
    pub quality: AcousticQualityMask,
}

/// Resource and quality summary returned after streaming extraction.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureExtractionSummary {
    pub feature_schema: &'static str,
    pub frame_count: usize,
    pub voiced_frame_count: usize,
    pub reliable_pitch_frame_count: usize,
    pub high_information_frame_count: usize,
    pub missing_pitch_frame_count: usize,
    pub low_energy_frame_count: usize,
    /// Fixed upper bound on retained DSP state, independent of call duration.
    pub retained_state_bytes_upper_bound: usize,
}

/// Maximum PCM samples admitted by one bounded wavelet analysis.
pub const ACOUSTIC_WAVELET_MAX_SAMPLES: usize = ACOUSTIC_FRAME_SAMPLES;
/// Maximum decomposition depth admitted by the experimental sidecar.
pub const ACOUSTIC_WAVELET_MAX_LEVELS: usize = 4;
/// Maximum admitted relative Parseval residual for one transform level.
pub const ACOUSTIC_WAVELET_ENERGY_TOLERANCE: f32 = 2e-5;
/// Relative power added to every coefficient when computing wavelet flatness.
///
/// Wavelet coefficients are retained as `f32`, so mathematically zero values can
/// acquire representation noise after a DC offset is added and removed. A
/// scale-relative floor preserves gain invariance without changing acoustic-v2.
pub const ACOUSTIC_WAVELET_FLATNESS_RELATIVE_POWER_FLOOR: f32 = f32::EPSILON;
/// Centered-RMS floor relative to `max(1, abs(input_mean))` before a raw-PCM
/// wavelet window is energy-normalized. This prevents representational jitter
/// in a constant or near-constant frame from becoming unit-energy evidence.
pub const ACOUSTIC_WAVELET_CENTERED_RMS_RELATIVE_FLOOR: f32 = 8.0 * f32::EPSILON;
/// Independent contract identity for the evaluation-only sidecar surface.
pub const ACOUSTIC_SIDECAR_STUDY_SCHEMA_VERSION: &str = "acoustic-multiscale-sidecar-v4";
/// Fixed temporal support of the modulation sidecar at the 10 ms cadence.
pub const ACOUSTIC_MODULATION_HISTORY_FRAMES: usize = 64;
/// Minimum valid trajectory observations required by a modulation regression.
pub const ACOUSTIC_MODULATION_MIN_VALID_FRAMES: usize = 32;
/// Centered-RMS floor relative to `max(1, abs(valid_mean))` before modulation
/// regression. Below this floor, f32 quantization residue is absence rather
/// than evidence with an arbitrary normalized spectrum.
pub const ACOUSTIC_MODULATION_CENTERED_RMS_RELATIVE_FLOOR: f32 = 8.0 * f32::EPSILON;
/// Fixed modulation frequencies represented by harmonics 1, 2, 4, and 8 of a
/// 64-frame window sampled at 100 Hz. These are point frequencies, not bands.
pub const ACOUSTIC_MODULATION_FREQUENCY_HZ: [f32; 4] = [1.5625, 3.125, 6.25, 12.5];
/// Fixed temporal support shared by trajectory-wavelet and scattering candidates.
pub const ACOUSTIC_TRAJECTORY_HISTORY_FRAMES: usize = 64;
/// Minimum observed values required before one trajectory family is analyzed.
pub const ACOUSTIC_TRAJECTORY_MIN_VALID_FRAMES: usize = 32;
/// Centered-RMS floor relative to `max(1, abs(valid_mean))` for trajectory
/// admission. This suppresses representational jitter that unit-energy
/// normalization would otherwise amplify into full-scale evidence.
pub const ACOUSTIC_TRAJECTORY_CENTERED_RMS_RELATIVE_FLOOR: f32 = 8.0 * f32::EPSILON;
/// RMS floor for distribution-shape statistics after a trajectory has been
/// unit-energy normalized. Magnitudes remain observable below this floor, but
/// f32 transform residue cannot create entropy or adjacent-change evidence.
pub const ACOUSTIC_TRAJECTORY_DETAIL_RMS_FLOOR: f32 = 8.0 * f32::EPSILON;
/// Minimum retained detail coefficients required for one masked
/// trajectory-wavelet level.
pub const ACOUSTIC_TRAJECTORY_MIN_VALID_COEFFICIENTS: usize = 2;
/// Number and canonical order of trajectory families in the sidecar ring.
pub const ACOUSTIC_TRAJECTORY_FAMILY_COUNT: usize = 5;
/// Fixed undecimated Haar supports used by the scattering-inspired candidate.
pub const ACOUSTIC_SCATTERING_SCALE_SUPPORTS: [usize; 3] = [2, 4, 8];
/// Canonical `(first_scale, second_scale)` order for second-order outputs.
pub const ACOUSTIC_SCATTERING_SCALE_PAIRS: [[usize; 2]; 3] = [[0, 1], [0, 2], [1, 2]];
/// Minimum valid non-wrapping filter positions required for one scattering output.
pub const ACOUSTIC_SCATTERING_MIN_VALID_OUTPUTS: usize = 8;

// Scratch accounting is expressed in fixed value/mask buffer pairs so the
// implementation, reported payload bytes, and configuration identity share
// one source of truth.
const ACOUSTIC_TRAJECTORY_WAVELET_SCRATCH_PAIR_COUNT: usize = 3;
const ACOUSTIC_SCATTERING_FIRST_ORDER_SCRATCH_PAIR_COUNT: usize = 4;
const ACOUSTIC_SCATTERING_SECOND_ORDER_EXTRA_SCRATCH_PAIR_COUNT: usize = 1;
const ACOUSTIC_TRAJECTORY_SCRATCH_PAIR_PAYLOAD_BYTES: usize =
    std::mem::size_of::<[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]>()
        + std::mem::size_of::<[bool; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]>();
const ACOUSTIC_TRAJECTORY_WAVELET_SCRATCH_PAYLOAD_BYTES: usize =
    ACOUSTIC_TRAJECTORY_WAVELET_SCRATCH_PAIR_COUNT * ACOUSTIC_TRAJECTORY_SCRATCH_PAIR_PAYLOAD_BYTES;
const ACOUSTIC_SCATTERING_FIRST_ORDER_SCRATCH_PAYLOAD_BYTES: usize =
    ACOUSTIC_SCATTERING_FIRST_ORDER_SCRATCH_PAIR_COUNT
        * ACOUSTIC_TRAJECTORY_SCRATCH_PAIR_PAYLOAD_BYTES;
const ACOUSTIC_SCATTERING_SECOND_ORDER_EXTRA_SCRATCH_PAYLOAD_BYTES: usize =
    ACOUSTIC_SCATTERING_SECOND_ORDER_EXTRA_SCRATCH_PAIR_COUNT
        * ACOUSTIC_TRAJECTORY_SCRATCH_PAIR_PAYLOAD_BYTES;

const ACOUSTIC_MODULATION_BIN_INDICES: [usize; 4] = [1, 2, 4, 8];
const ACOUSTIC_TRAJECTORY_FRACTION_DOMAIN_MAX: f32 = 1.001;
const ACOUSTIC_MODULATION_STEP_COMPLEX: [[f64; 2]; 4] = [
    [0.995_184_726_672_196_9, -0.098_017_140_329_560_6],
    [0.980_785_280_403_230_4, -0.195_090_322_016_128_25],
    [0.923_879_532_511_286_7, -0.382_683_432_365_089_8],
    [
        std::f64::consts::FRAC_1_SQRT_2,
        -std::f64::consts::FRAC_1_SQRT_2,
    ],
];
const ACOUSTIC_SCATTERING_SCALE_NORMALIZERS: [f64; 3] = [
    std::f64::consts::FRAC_1_SQRT_2,
    0.5,
    0.353_553_390_593_273_8,
];
const HAAR_LOW: [f64; 2] = [
    std::f64::consts::FRAC_1_SQRT_2,
    std::f64::consts::FRAC_1_SQRT_2,
];
const HAAR_HIGH: [f64; 2] = [
    std::f64::consts::FRAC_1_SQRT_2,
    -std::f64::consts::FRAC_1_SQRT_2,
];
const DAUBECHIES_FOUR_TAP_LOW: [f64; 4] = [
    0.482_962_913_144_534_16,
    0.836_516_303_737_807_9,
    0.224_143_868_042_013_4,
    -0.129_409_522_551_260_37,
];
const DAUBECHIES_FOUR_TAP_HIGH: [f64; 4] = [
    0.129_409_522_551_260_37,
    0.224_143_868_042_013_4,
    -0.836_516_303_737_807_9,
    0.482_962_913_144_534_16,
];

/// Frozen evaluation choices orthogonal to [`AcousticFeatureAblation`].
///
/// The default remains `Off`; none of these variants changes acoustic-v2 or
/// its public-corpus evidence ordering.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AcousticSidecarStudyMode {
    #[default]
    Off,
    Haar,
    DaubechiesFourTap,
    Modulation,
    HaarAndModulation,
    DaubechiesFourTapAndModulation,
}

impl AcousticSidecarStudyMode {
    pub const ALL: [Self; 6] = [
        Self::Off,
        Self::Haar,
        Self::DaubechiesFourTap,
        Self::Modulation,
        Self::HaarAndModulation,
        Self::DaubechiesFourTapAndModulation,
    ];

    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Haar => "haar",
            Self::DaubechiesFourTap => "daubechies_four_tap",
            Self::Modulation => "modulation",
            Self::HaarAndModulation => "haar_modulation",
            Self::DaubechiesFourTapAndModulation => "daubechies_four_tap_modulation",
        }
    }

    const fn wavelet_basis(self) -> Option<AcousticWaveletBasis> {
        match self {
            Self::Haar | Self::HaarAndModulation => Some(AcousticWaveletBasis::Haar),
            Self::DaubechiesFourTap | Self::DaubechiesFourTapAndModulation => {
                Some(AcousticWaveletBasis::DaubechiesFourTap)
            }
            Self::Off | Self::Modulation => None,
        }
    }

    #[must_use]
    pub const fn uses_modulation(self) -> bool {
        matches!(
            self,
            Self::Modulation | Self::HaarAndModulation | Self::DaubechiesFourTapAndModulation
        )
    }
}

/// Independently selectable stationary trajectory-wavelet candidate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AcousticTrajectoryWaveletMode {
    #[default]
    Off,
    Haar,
    DaubechiesFourTap,
}

impl AcousticTrajectoryWaveletMode {
    pub const ALL: [Self; 3] = [Self::Off, Self::Haar, Self::DaubechiesFourTap];

    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Haar => "haar",
            Self::DaubechiesFourTap => "daubechies_four_tap",
        }
    }

    const fn basis(self) -> Option<AcousticWaveletBasis> {
        match self {
            Self::Off => None,
            Self::Haar => Some(AcousticWaveletBasis::Haar),
            Self::DaubechiesFourTap => Some(AcousticWaveletBasis::DaubechiesFourTap),
        }
    }
}

/// Output orders retained by the fixed-filter scattering candidate.
///
/// `SecondOrder` still computes its prerequisite first-order modulus paths,
/// but it does not expose their aggregate values as selected evidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AcousticScatteringMode {
    #[default]
    Off,
    FirstOrder,
    SecondOrder,
    FirstAndSecondOrder,
}

impl AcousticScatteringMode {
    pub const ALL: [Self; 4] = [
        Self::Off,
        Self::FirstOrder,
        Self::SecondOrder,
        Self::FirstAndSecondOrder,
    ];

    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::FirstOrder => "first_order",
            Self::SecondOrder => "second_order",
            Self::FirstAndSecondOrder => "first_and_second_order",
        }
    }

    #[must_use]
    pub const fn emits_first_order(self) -> bool {
        matches!(self, Self::FirstOrder | Self::FirstAndSecondOrder)
    }

    #[must_use]
    pub const fn emits_second_order(self) -> bool {
        matches!(self, Self::SecondOrder | Self::FirstAndSecondOrder)
    }

    #[must_use]
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// Complete configuration of one sidecar-only study lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcousticSidecarStudyConfig {
    pub mode: AcousticSidecarStudyMode,
    /// Must be zero when the selected frame mode has no raw-PCM wavelet.
    pub frame_wavelet_levels: usize,
    pub trajectory_wavelet_mode: AcousticTrajectoryWaveletMode,
    /// Must be zero when `trajectory_wavelet_mode` is `Off`.
    pub trajectory_wavelet_levels: usize,
    pub scattering_mode: AcousticScatteringMode,
}

impl Default for AcousticSidecarStudyConfig {
    fn default() -> Self {
        Self {
            mode: AcousticSidecarStudyMode::Off,
            frame_wavelet_levels: 0,
            trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Off,
            trajectory_wavelet_levels: 0,
            scattering_mode: AcousticScatteringMode::Off,
        }
    }
}

impl AcousticSidecarStudyConfig {
    fn validate(self) -> FwResult<()> {
        if self.mode.wavelet_basis().is_some() {
            if self.frame_wavelet_levels == 0
                || self.frame_wavelet_levels > ACOUSTIC_WAVELET_MAX_LEVELS
            {
                return Err(FwError::InvalidRequest(format!(
                    "frame wavelet sidecar study levels must be within 1..={ACOUSTIC_WAVELET_MAX_LEVELS}"
                )));
            }
        } else if self.frame_wavelet_levels != 0 {
            return Err(FwError::InvalidRequest(
                "non-wavelet frame sidecar study modes require frame_wavelet_levels=0".to_owned(),
            ));
        }
        if self.trajectory_wavelet_mode.basis().is_some() {
            if self.trajectory_wavelet_levels == 0
                || self.trajectory_wavelet_levels > ACOUSTIC_WAVELET_MAX_LEVELS
            {
                return Err(FwError::InvalidRequest(format!(
                    "trajectory wavelet levels must be within 1..={ACOUSTIC_WAVELET_MAX_LEVELS}"
                )));
            }
        } else if self.trajectory_wavelet_levels != 0 {
            return Err(FwError::InvalidRequest(
                "trajectory_wavelet_mode=off requires trajectory_wavelet_levels=0".to_owned(),
            ));
        }
        Ok(())
    }

    const fn uses_trajectory_state(self) -> bool {
        self.trajectory_wavelet_mode.basis().is_some() || self.scattering_mode.is_enabled()
    }
}

/// Canonical configuration fingerprint for an evaluation-only sidecar lane.
///
/// The hash binds the frozen acoustic-v2 base plus the numerical, boundary,
/// validity, ownership, cancellation, and accounting conventions used below.
pub fn acoustic_sidecar_study_config_sha256(
    config: AcousticSidecarStudyConfig,
) -> FwResult<String> {
    Ok(sidecar_sha256_hex(&acoustic_sidecar_study_config_digest(
        config,
    )?))
}

fn acoustic_sidecar_study_config_digest(config: AcousticSidecarStudyConfig) -> FwResult<[u8; 32]> {
    config.validate()?;
    let mut hasher = Sha256::new();
    for field in [
        ACOUSTIC_SIDECAR_STUDY_SCHEMA_VERSION,
        ACOUSTIC_FEATURE_SCHEMA_VERSION,
        config.mode.id(),
        config.trajectory_wavelet_mode.id(),
        config.scattering_mode.id(),
        "configured-runner=exact-400-sample-16khz-frame-at-160-sample-hop",
        "mean-center-energy-normalize",
        "right-half-sample-symmetric-endpoint-duplicate-to-even-then-periodic",
        "approximation-then-detail-level-order",
        "wavelet-analysis-phase=forward-taps-at-two-times-output-index",
        "detail-energy-local-to-level",
        "detail-stats=energy-fraction-ln-mean-square-entropy-flatness-crest-adjacent-change",
        "coefficient-flatness=geometric-mean-of-power-plus-relative-floor-over-mean-power-plus-relative-floor",
        "frame-wavelet-near-constant=centered-rms-at-most-relative-floor-times-max-one-absolute-input-mean-is-unavailable",
        "d4-minimum-support=4-times-two-to-level-minus-one",
        "voice=temporal-modulation-mean-absolute-dct-delta",
        "voice-valid=voiced-and-not-low-energy-clipped-transient-and-frame-index-positive",
        "channel-level=rms-dbfs",
        "channel-coloration=muffling-proxy",
        "channel-valid=not-low-energy-and-not-clipped",
        "minimum-32-of-64-valid-no-zero-fill",
        "modulation-ring=oldest-index-forward-to-newest",
        "twiddles=unit-seed-forward-complex-recurrence-negative-imaginary",
        "intercept-residualized-sine-cosine-regression-r-squared",
        "modulation-near-constant=centered-rms-at-most-relative-floor-times-max-one-absolute-valid-mean-is-unavailable",
        "wavelet-owner=mixed-auxiliary",
        "modulation-owners=voice-channel-channel",
        "trajectory-order=voiced-cepstral-envelope-magnitude-voiced-occupancy-low-mid-high-band-fractions",
        "trajectory-values=frame-local-cepstral-envelope-rms-frame-local-quality-voiced-indicator-low-band-fraction-mid-band-fraction-high-band-fraction",
        "trajectory-owners=voice-mixed-auxiliary-channel-channel-channel",
        "trajectory-ring=oldest-index-forward-to-newest",
        "trajectory-window=exactly-64-contiguous-frame-indices",
        "trajectory-envelope-valid=voiced-and-not-low-energy-clipped-transient",
        "trajectory-occupancy-valid=not-clipped",
        "trajectory-band-valid=not-low-energy-and-not-clipped",
        "trajectory-normalization=valid-mean-center-unit-energy",
        "trajectory-near-constant=centered-rms-at-most-relative-floor-times-max-one-absolute-valid-mean-makes-all-candidate-outputs-unavailable",
        "masked-trajectory-wavelet=undecimated-stationary-all-filter-support-valid-or-coefficient-absent",
        "masked-trajectory-wavelet-invalid-values=omitted-never-zero-imputed",
        "masked-trajectory-wavelet-geometry=nonwrapping-valid-forward-taps-at-output-index-with-dyadic-level-dilation",
        "masked-trajectory-wavelet-output=mean-absolute-rms-entropy-adjacent-change-with-independent-pair-availability",
        "masked-trajectory-wavelet-detail-shape=entropy-and-adjacent-change-unavailable-at-or-below-unit-normalized-detail-rms-floor",
        "masked-trajectory-wavelet-adjacent-change=linear-neighbor-pairs",
        "scattering-filter=undecimated-nonwrapping-valid-forward-haar-unit-l2",
        "scattering-first-order=mean-absolute-filter-response",
        "scattering-second-order=mean-absolute-filtered-first-order-modulus-with-j2-greater-than-j1",
        "scattering-second-only=compute-required-first-scales-zero-and-one-hide-aggregates",
        "scattering-average=arithmetic-mean-over-valid-nonwrapping-positions",
        "runner-pcm-validation=every-submitted-sample-finite-and-within-inclusive-minus-one-plus-one-before-any-enabled-family",
        "trajectory-numerics=f64-valid-sum-mean-centered-energy-filter-adjacent-detail-difference-absolute-value-and-summary-accumulation-f32-normalized-coefficient-and-summary-storage-round-on-cast",
        "scattering-numerics=f64-normalization-filter-response-and-aggregate-accumulation-f32-normalized-modulus-and-summary-storage-round-on-cast",
        "trajectory-wavelet-accounting=one-validity-visit-per-support-tap-and-two-filter-terms-per-tap-only-for-fully-valid-low-high-support",
        "scattering-accounting=one-validity-visit-per-support-tap-and-one-filter-term-per-tap-only-for-fully-valid-support",
        "scratch-accounting=fixed-f32-value-plus-bool-validity-array-pairs-over-64-frame-support",
        "cancellation=entry-frame-wavelet-modulation-family-frequency-trajectory-family-level-scale-pair",
        "mutation=all-enabled-incremental-families-staged-until-frame-success",
        "accounting=filter-terms-validity-visits-projection-visits-and-exact-fixed-payload-bytes",
    ] {
        hasher.update((field.len() as u64).to_le_bytes());
        hasher.update(field.as_bytes());
    }
    let acoustic_v2_sha256 = acoustic_feature_schema_sha256(AcousticFeatureSchemaVersion::V2);
    hasher.update((acoustic_v2_sha256.len() as u64).to_le_bytes());
    hasher.update(acoustic_v2_sha256.as_bytes());
    hasher.update((config.frame_wavelet_levels as u64).to_le_bytes());
    hasher.update((config.trajectory_wavelet_levels as u64).to_le_bytes());
    hasher.update((crate::native_engine::mel::SAMPLE_RATE as u64).to_le_bytes());
    hasher.update((ACOUSTIC_FRAME_SAMPLES as u64).to_le_bytes());
    hasher.update((ACOUSTIC_HOP_SAMPLES as u64).to_le_bytes());
    hasher.update((ACOUSTIC_WAVELET_MAX_SAMPLES as u64).to_le_bytes());
    hasher.update((ACOUSTIC_WAVELET_MAX_LEVELS as u64).to_le_bytes());
    hasher.update((ACOUSTIC_MODULATION_HISTORY_FRAMES as u64).to_le_bytes());
    hasher.update((ACOUSTIC_MODULATION_MIN_VALID_FRAMES as u64).to_le_bytes());
    hasher.update((ACOUSTIC_TRAJECTORY_HISTORY_FRAMES as u64).to_le_bytes());
    hasher.update((ACOUSTIC_TRAJECTORY_MIN_VALID_FRAMES as u64).to_le_bytes());
    hasher.update(
        ACOUSTIC_TRAJECTORY_CENTERED_RMS_RELATIVE_FLOOR
            .to_bits()
            .to_le_bytes(),
    );
    hasher.update(ACOUSTIC_TRAJECTORY_DETAIL_RMS_FLOOR.to_bits().to_le_bytes());
    hasher.update((ACOUSTIC_TRAJECTORY_MIN_VALID_COEFFICIENTS as u64).to_le_bytes());
    hasher.update((ACOUSTIC_TRAJECTORY_FAMILY_COUNT as u64).to_le_bytes());
    hasher.update((CEPSTRAL_COEFFICIENTS as u64).to_le_bytes());
    hasher.update((ACOUSTIC_SCATTERING_MIN_VALID_OUTPUTS as u64).to_le_bytes());
    hasher.update((ACOUSTIC_TRAJECTORY_WAVELET_SCRATCH_PAIR_COUNT as u64).to_le_bytes());
    hasher.update((ACOUSTIC_SCATTERING_FIRST_ORDER_SCRATCH_PAIR_COUNT as u64).to_le_bytes());
    hasher.update((ACOUSTIC_SCATTERING_SECOND_ORDER_EXTRA_SCRATCH_PAIR_COUNT as u64).to_le_bytes());
    hasher.update((ACOUSTIC_TRAJECTORY_SCRATCH_PAIR_PAYLOAD_BYTES as u64).to_le_bytes());
    hasher.update((ACOUSTIC_TRAJECTORY_WAVELET_SCRATCH_PAYLOAD_BYTES as u64).to_le_bytes());
    hasher.update((ACOUSTIC_SCATTERING_FIRST_ORDER_SCRATCH_PAYLOAD_BYTES as u64).to_le_bytes());
    hasher.update(
        (ACOUSTIC_SCATTERING_SECOND_ORDER_EXTRA_SCRATCH_PAYLOAD_BYTES as u64).to_le_bytes(),
    );
    hasher.update(MAX_ABS_ACOUSTIC_FEATURE.to_bits().to_le_bytes());
    hasher.update(
        ACOUSTIC_TRAJECTORY_FRACTION_DOMAIN_MAX
            .to_bits()
            .to_le_bytes(),
    );
    hasher.update(PCM_EPSILON.to_bits().to_le_bytes());
    hasher.update(POWER_EPSILON.to_bits().to_le_bytes());
    hasher.update(ACOUSTIC_WAVELET_ENERGY_TOLERANCE.to_bits().to_le_bytes());
    hasher.update(
        ACOUSTIC_WAVELET_FLATNESS_RELATIVE_POWER_FLOOR
            .to_bits()
            .to_le_bytes(),
    );
    hasher.update(
        ACOUSTIC_WAVELET_CENTERED_RMS_RELATIVE_FLOOR
            .to_bits()
            .to_le_bytes(),
    );
    hasher.update(
        ACOUSTIC_MODULATION_CENTERED_RMS_RELATIVE_FLOOR
            .to_bits()
            .to_le_bytes(),
    );
    for bin in ACOUSTIC_MODULATION_BIN_INDICES {
        hasher.update((bin as u64).to_le_bytes());
    }
    for frequency in ACOUSTIC_MODULATION_FREQUENCY_HZ {
        hasher.update(frequency.to_bits().to_le_bytes());
    }
    for component in ACOUSTIC_MODULATION_STEP_COMPLEX.into_iter().flatten() {
        hasher.update(component.to_bits().to_le_bytes());
    }
    for support in ACOUSTIC_SCATTERING_SCALE_SUPPORTS {
        hasher.update((support as u64).to_le_bytes());
    }
    for pair in ACOUSTIC_SCATTERING_SCALE_PAIRS {
        for scale in pair {
            hasher.update((scale as u64).to_le_bytes());
        }
    }
    for normalizer in ACOUSTIC_SCATTERING_SCALE_NORMALIZERS {
        hasher.update(normalizer.to_bits().to_le_bytes());
    }
    for family in AcousticTrajectoryFamily::ALL {
        hasher.update((family.id().len() as u64).to_le_bytes());
        hasher.update(family.id().as_bytes());
        hasher.update([match family.owner() {
            AcousticSidecarFeatureOwner::Voice => 0,
            AcousticSidecarFeatureOwner::Channel => 1,
            AcousticSidecarFeatureOwner::MixedAuxiliary => 2,
        }]);
    }
    for frequency in &MODULATION_TWIDDLES {
        for twiddle in frequency {
            for component in twiddle {
                hasher.update(component.to_bits().to_le_bytes());
            }
        }
    }
    for coefficient in HAAR_LOW
        .into_iter()
        .chain(HAAR_HIGH)
        .chain(DAUBECHIES_FOUR_TAP_LOW)
        .chain(DAUBECHIES_FOUR_TAP_HIGH)
    {
        hasher.update(coefficient.to_bits().to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut output = [0_u8; 32];
    output.copy_from_slice(&digest);
    Ok(output)
}

fn sidecar_sha256_hex(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

/// One explicitly selected orthogonal wavelet family.
///
/// `DaubechiesFourTap` is the four-coefficient D4 analysis filter, also called
/// `db2` by libraries that number a Daubechies family by vanishing moments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticWaveletBasis {
    Haar,
    DaubechiesFourTap,
}

impl AcousticWaveletBasis {
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Haar => "haar",
            Self::DaubechiesFourTap => "daubechies-four-tap",
        }
    }
}

/// Ownership boundary for an experimental sidecar observation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AcousticSidecarFeatureOwner {
    Voice,
    Channel,
    /// Raw PCM wavelets mix source, vocal-tract, room, device, and codec cues.
    /// They may support a study but may not enter reusable voice identity.
    #[default]
    MixedAuxiliary,
}

/// Bounded configuration for one wavelet sidecar observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcousticWaveletConfig {
    pub basis: AcousticWaveletBasis,
    pub levels: usize,
}

impl Default for AcousticWaveletConfig {
    fn default() -> Self {
        Self {
            basis: AcousticWaveletBasis::Haar,
            levels: ACOUSTIC_WAVELET_MAX_LEVELS,
        }
    }
}

/// Dimensionless statistics for one wavelet detail scale.
///
/// Energy is relative to the symmetrically even-extended input of this level,
/// not to the complete call. Entropy is normalized into `[0, 1]`. These
/// summaries deliberately carry no raw coefficients.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct AcousticWaveletLevelSummary {
    /// Detail energy divided by the even-extended input energy of this level.
    pub detail_energy_fraction: f32,
    /// Natural log of mean squared detail-coefficient energy after the
    /// call-independent input normalization.
    pub detail_log_energy: f32,
    pub normalized_entropy: f32,
    pub coefficient_flatness: f32,
    pub crest_factor: f32,
    pub normalized_detail_change: f32,
    /// Unclamped Parseval residual checked before any normalized fraction.
    pub energy_conservation_relative_error: f32,
}

/// Fixed-size result of one bounded wavelet analysis.
///
/// This experimental result is not part of acoustic feature schema v2 and is
/// never computed by the default diarizer. It remains audio-derived feature
/// data: callers must opt into the sidecar and must not log, serialize, or
/// persist it as a reusable speaker identity.
#[derive(Clone, Copy, PartialEq)]
pub struct AcousticWaveletSummary {
    pub basis: AcousticWaveletBasis,
    pub owner: AcousticSidecarFeatureOwner,
    pub input_samples: usize,
    pub valid_level_count: usize,
    pub input_was_silent_or_near_constant: bool,
    pub levels: [AcousticWaveletLevelSummary; ACOUSTIC_WAVELET_MAX_LEVELS],
    /// Approximation energy relative to the even-extended input of the final
    /// requested level.
    pub final_approximation_energy_fraction: f32,
    /// Number of low/high analysis-filter coefficient applications.
    pub filter_tap_terms: usize,
    pub maximum_energy_conservation_relative_error: f32,
    /// Exact payload bytes of the three fixed scratch buffers. Compiler stack
    /// layout and local scalar metadata are deliberately outside this value.
    pub scratch_buffer_payload_bytes: usize,
}

impl std::fmt::Debug for AcousticWaveletSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcousticWaveletSummary")
            .field("basis", &self.basis)
            .field("owner", &self.owner)
            .field("input_samples", &self.input_samples)
            .field("valid_level_count", &self.valid_level_count)
            .field(
                "input_was_silent_or_near_constant",
                &self.input_was_silent_or_near_constant,
            )
            .field("filter_tap_terms", &self.filter_tap_terms)
            .field(
                "maximum_energy_conservation_relative_error",
                &self.maximum_energy_conservation_relative_error,
            )
            .field(
                "scratch_buffer_payload_bytes",
                &self.scratch_buffer_payload_bytes,
            )
            .finish_non_exhaustive()
    }
}

/// Analyze one normalized PCM window with a bounded orthogonal DWT.
///
/// Input is mean-centered and energy-normalized before analysis. For inputs
/// that remain above the centered-RMS admission gate, this makes the descriptors
/// invariant to DC offset and positive gain up to floating-point roundoff. A
/// centered-RMS gate rejects constant or representationally near-constant
/// windows before unit-energy normalization. Odd widths use a declared right
/// half-sample symmetric extension (duplicate the final sample) to the next
/// even width, followed by periodized filter support. Cancellation is checked
/// before validation and between decomposition levels.
pub fn analyze_acoustic_wavelet<C>(
    samples: &[f32],
    config: AcousticWaveletConfig,
    mut is_cancelled: C,
) -> FwResult<AcousticWaveletSummary>
where
    C: FnMut() -> bool,
{
    if is_cancelled() {
        return Err(FwError::Cancelled(
            "acoustic wavelet sidecar cancelled before validation".to_owned(),
        ));
    }
    if config.levels == 0 || config.levels > ACOUSTIC_WAVELET_MAX_LEVELS {
        return Err(FwError::InvalidRequest(format!(
            "acoustic wavelet levels must be within 1..={ACOUSTIC_WAVELET_MAX_LEVELS}"
        )));
    }
    let minimum_samples = match config.basis {
        AcousticWaveletBasis::Haar => 1usize.checked_shl(config.levels as u32),
        AcousticWaveletBasis::DaubechiesFourTap => {
            4usize.checked_shl(config.levels.saturating_sub(1) as u32)
        }
    }
    .ok_or_else(|| {
        FwError::InvalidRequest("acoustic wavelet level count exceeds size arithmetic".to_owned())
    })?;
    if samples.len() < minimum_samples || samples.len() > ACOUSTIC_WAVELET_MAX_SAMPLES {
        return Err(FwError::InvalidRequest(format!(
            "acoustic wavelet input must contain {minimum_samples}..={ACOUSTIC_WAVELET_MAX_SAMPLES} samples for {} levels",
            config.levels
        )));
    }
    if !normalized_acoustic_pcm_is_valid(samples) {
        return Err(FwError::InvalidRequest(
            "acoustic wavelet input must contain finite normalized PCM within [-1, 1]".to_owned(),
        ));
    }

    let mut current = [0.0_f32; ACOUSTIC_WAVELET_MAX_SAMPLES];
    let mut approximation = [0.0_f32; ACOUSTIC_WAVELET_MAX_SAMPLES];
    let mut detail = [0.0_f32; ACOUSTIC_WAVELET_MAX_SAMPLES];
    let mean = samples.iter().copied().map(f64::from).sum::<f64>() / samples.len() as f64;
    let centered_energy = samples
        .iter()
        .map(|sample| {
            let centered = f64::from(*sample) - mean;
            centered * centered
        })
        .sum::<f64>();
    let scratch_buffer_payload_bytes = std::mem::size_of_val(&current)
        + std::mem::size_of_val(&approximation)
        + std::mem::size_of_val(&detail);
    let centered_rms = (centered_energy / samples.len() as f64).sqrt();
    let centered_rms_floor =
        f64::from(ACOUSTIC_WAVELET_CENTERED_RMS_RELATIVE_FLOOR) * mean.abs().max(1.0);
    if centered_rms <= centered_rms_floor {
        return Ok(AcousticWaveletSummary {
            basis: config.basis,
            owner: AcousticSidecarFeatureOwner::MixedAuxiliary,
            input_samples: samples.len(),
            valid_level_count: 0,
            input_was_silent_or_near_constant: true,
            levels: [AcousticWaveletLevelSummary::default(); ACOUSTIC_WAVELET_MAX_LEVELS],
            final_approximation_energy_fraction: 0.0,
            filter_tap_terms: 0,
            maximum_energy_conservation_relative_error: 0.0,
            scratch_buffer_payload_bytes,
        });
    }
    let inverse_norm = centered_energy.sqrt().recip();
    for (output, sample) in current.iter_mut().zip(samples) {
        *output = ((f64::from(*sample) - mean) * inverse_norm) as f32;
    }

    let mut summaries = [AcousticWaveletLevelSummary::default(); ACOUSTIC_WAVELET_MAX_LEVELS];
    let mut current_len = samples.len();
    let mut filter_tap_terms = 0usize;
    let mut final_approximation_energy_fraction = 0.0_f32;
    let mut maximum_energy_conservation_relative_error = 0.0_f32;
    for (level, summary) in summaries.iter_mut().take(config.levels).enumerate() {
        if is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "acoustic wavelet sidecar cancelled before level {level}"
            )));
        }
        let output_len = current_len.div_ceil(2);
        let extended_energy = wavelet_extended_energy(&current, current_len);
        for output_index in 0..output_len {
            let base = output_index * 2;
            let (low, high, terms) = wavelet_pair(config.basis, &current, current_len, base);
            approximation[output_index] = low;
            detail[output_index] = high;
            filter_tap_terms = filter_tap_terms.checked_add(terms).ok_or_else(|| {
                FwError::InvalidRequest(
                    "acoustic wavelet operation accounting overflowed".to_owned(),
                )
            })?;
        }
        let (mut level_summary, detail_energy) =
            summarize_wavelet_detail(&detail[..output_len], extended_energy);
        let approximation_energy = approximation[..output_len]
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        let energy_conservation_relative_error = if extended_energy > 0.0 {
            ((detail_energy + approximation_energy - extended_energy).abs() / extended_energy)
                as f32
        } else {
            0.0
        };
        if !energy_conservation_relative_error.is_finite()
            || energy_conservation_relative_error > ACOUSTIC_WAVELET_ENERGY_TOLERANCE
        {
            return Err(FwError::InvalidRequest(format!(
                "acoustic wavelet energy invariant failed at level {level}"
            )));
        }
        level_summary.energy_conservation_relative_error = energy_conservation_relative_error;
        maximum_energy_conservation_relative_error =
            maximum_energy_conservation_relative_error.max(energy_conservation_relative_error);
        *summary = level_summary;
        final_approximation_energy_fraction = if extended_energy > 0.0 {
            (approximation_energy / extended_energy).clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
        current[..output_len].copy_from_slice(&approximation[..output_len]);
        approximation[..output_len].fill(0.0);
        detail[..output_len].fill(0.0);
        current_len = output_len;
    }

    Ok(AcousticWaveletSummary {
        basis: config.basis,
        owner: AcousticSidecarFeatureOwner::MixedAuxiliary,
        input_samples: samples.len(),
        valid_level_count: config.levels,
        input_was_silent_or_near_constant: false,
        levels: summaries,
        final_approximation_energy_fraction,
        filter_tap_terms,
        maximum_energy_conservation_relative_error,
        scratch_buffer_payload_bytes,
    })
}

fn normalized_acoustic_pcm_is_valid(samples: &[f32]) -> bool {
    samples
        .iter()
        .all(|sample| sample.is_finite() && (-1.0..=1.0).contains(sample))
}

fn wavelet_extended_energy(input: &[f32; ACOUSTIC_WAVELET_MAX_SAMPLES], input_len: usize) -> f64 {
    let mut energy = input[..input_len]
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    if input_len % 2 == 1 {
        let last = f64::from(input[input_len - 1]);
        energy += last * last;
    }
    energy
}

fn wavelet_extended_sample(
    input: &[f32; ACOUSTIC_WAVELET_MAX_SAMPLES],
    input_len: usize,
    index: usize,
) -> f64 {
    let extended_len = input_len + input_len % 2;
    let wrapped = index % extended_len;
    if wrapped == input_len {
        f64::from(input[input_len - 1])
    } else {
        f64::from(input[wrapped])
    }
}

fn wavelet_pair(
    basis: AcousticWaveletBasis,
    input: &[f32; ACOUSTIC_WAVELET_MAX_SAMPLES],
    input_len: usize,
    base: usize,
) -> (f32, f32, usize) {
    match basis {
        AcousticWaveletBasis::Haar => {
            let left = wavelet_extended_sample(input, input_len, base);
            let right = wavelet_extended_sample(input, input_len, base + 1);
            (
                (HAAR_LOW[0] * left + HAAR_LOW[1] * right) as f32,
                (HAAR_HIGH[0] * left + HAAR_HIGH[1] * right) as f32,
                4,
            )
        }
        AcousticWaveletBasis::DaubechiesFourTap => {
            let mut low = 0.0_f64;
            let mut high = 0.0_f64;
            for tap in 0..4 {
                let sample = wavelet_extended_sample(input, input_len, base + tap);
                low += DAUBECHIES_FOUR_TAP_LOW[tap] * sample;
                high += DAUBECHIES_FOUR_TAP_HIGH[tap] * sample;
            }
            (low as f32, high as f32, 8)
        }
    }
}

fn summarize_wavelet_detail(
    detail: &[f32],
    level_input_energy: f64,
) -> (AcousticWaveletLevelSummary, f64) {
    let detail_energy = detail
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    let detail_energy_fraction = if level_input_energy > 0.0 {
        (detail_energy / level_input_energy).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    if detail_energy <= f64::from(PCM_EPSILON * PCM_EPSILON) {
        return (
            AcousticWaveletLevelSummary {
                detail_energy_fraction,
                detail_log_energy: POWER_EPSILON.ln(),
                ..AcousticWaveletLevelSummary::default()
            },
            detail_energy,
        );
    }
    let mean_squared = detail_energy / detail.len() as f64;
    let rms = mean_squared.sqrt();
    let crest_factor = detail
        .iter()
        .map(|value| f64::from(value.abs()))
        .fold(0.0_f64, f64::max)
        / rms;
    let normalized_entropy = if detail.len() > 1 {
        let entropy = detail
            .iter()
            .map(|value| {
                let probability = f64::from(*value) * f64::from(*value) / detail_energy;
                if probability > 0.0 {
                    -probability * probability.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        (entropy / (detail.len() as f64).ln()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let normalized_detail_change = if detail.len() > 1 {
        let mean_change = detail
            .windows(2)
            .map(|pair| f64::from((pair[1] - pair[0]).abs()))
            .sum::<f64>()
            / (detail.len() - 1) as f64;
        mean_change / rms
    } else {
        0.0
    };
    let log_floor = f64::from(POWER_EPSILON);
    let coefficient_flatness = {
        let relative_floor =
            mean_squared * f64::from(ACOUSTIC_WAVELET_FLATNESS_RELATIVE_POWER_FLOOR);
        let geometric_mean = (detail
            .iter()
            .map(|value| {
                let squared = f64::from(*value) * f64::from(*value);
                (squared + relative_floor).ln()
            })
            .sum::<f64>()
            / detail.len() as f64)
            .exp();
        (geometric_mean / (mean_squared + relative_floor)).clamp(0.0, 1.0)
    };
    (
        AcousticWaveletLevelSummary {
            detail_energy_fraction,
            detail_log_energy: mean_squared.max(log_floor).ln() as f32,
            normalized_entropy: normalized_entropy as f32,
            coefficient_flatness: coefficient_flatness as f32,
            crest_factor: crest_factor as f32,
            normalized_detail_change: normalized_detail_change as f32,
            energy_conservation_relative_error: 0.0,
        },
        detail_energy,
    )
}

/// Canonical order and ownership of the bounded temporal trajectory ring.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AcousticTrajectoryFamily {
    /// Frame-local RMS magnitude of the gain-centered cepstral envelope on
    /// usable voiced frames. This candidate evaluates its trajectory with
    /// scale-local stationary-wavelet and scattering summaries without
    /// inheriting state from a preceding frame.
    #[default]
    VoicedCepstralEnvelopeMagnitude,
    /// Frame-local voiced occupancy. It is activity evidence, not reusable
    /// speaker identity, because turn structure and VAD behavior can dominate.
    VoicedOccupancy,
    LowBandFraction,
    MidBandFraction,
    HighBandFraction,
}

impl AcousticTrajectoryFamily {
    pub const ALL: [Self; ACOUSTIC_TRAJECTORY_FAMILY_COUNT] = [
        Self::VoicedCepstralEnvelopeMagnitude,
        Self::VoicedOccupancy,
        Self::LowBandFraction,
        Self::MidBandFraction,
        Self::HighBandFraction,
    ];

    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::VoicedCepstralEnvelopeMagnitude => "voiced_cepstral_envelope_magnitude",
            Self::VoicedOccupancy => "voiced_occupancy",
            Self::LowBandFraction => "low_band_fraction",
            Self::MidBandFraction => "mid_band_fraction",
            Self::HighBandFraction => "high_band_fraction",
        }
    }

    #[must_use]
    pub const fn owner(self) -> AcousticSidecarFeatureOwner {
        match self {
            Self::VoicedCepstralEnvelopeMagnitude => AcousticSidecarFeatureOwner::Voice,
            Self::VoicedOccupancy => AcousticSidecarFeatureOwner::MixedAuxiliary,
            Self::LowBandFraction | Self::MidBandFraction | Self::HighBandFraction => {
                AcousticSidecarFeatureOwner::Channel
            }
        }
    }

    fn value_and_valid(self, frame: &AcousticFrameFeatures) -> (f32, bool) {
        match self {
            Self::VoicedCepstralEnvelopeMagnitude => {
                let mean_squared = frame
                    .voice
                    .cepstral_envelope
                    .iter()
                    .map(|value| {
                        let value = f64::from(*value);
                        value * value
                    })
                    .sum::<f64>()
                    / CEPSTRAL_COEFFICIENTS as f64;
                (
                    mean_squared.sqrt() as f32,
                    frame.quality.voiced
                        && !frame.quality.low_energy
                        && !frame.quality.clipped
                        && !frame.quality.transient,
                )
            }
            Self::VoicedOccupancy => (
                if frame.quality.voiced { 1.0 } else { 0.0 },
                !frame.quality.clipped,
            ),
            Self::LowBandFraction => (
                frame.channel.low_band_fraction,
                !frame.quality.low_energy && !frame.quality.clipped,
            ),
            Self::MidBandFraction => (
                frame.channel.mid_band_fraction,
                !frame.quality.low_energy && !frame.quality.clipped,
            ),
            Self::HighBandFraction => (
                frame.channel.high_band_fraction,
                !frame.quality.low_energy && !frame.quality.clipped,
            ),
        }
    }

    fn value_is_in_domain(self, value: f32) -> bool {
        match self {
            Self::VoicedCepstralEnvelopeMagnitude => {
                value.is_finite() && (0.0..=MAX_ABS_ACOUSTIC_FEATURE).contains(&value)
            }
            Self::VoicedOccupancy
            | Self::LowBandFraction
            | Self::MidBandFraction
            | Self::HighBandFraction => {
                value.is_finite()
                    && (0.0..=ACOUSTIC_TRAJECTORY_FRACTION_DOMAIN_MAX).contains(&value)
            }
        }
    }
}

/// Missingness-aware statistics for one undecimated stationary trajectory-wavelet level.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct AcousticTrajectoryWaveletLevelSummary {
    pub available: bool,
    pub valid_coefficients: usize,
    pub mean_absolute_detail: f32,
    pub detail_rms: f32,
    pub normalized_entropy_available: bool,
    pub normalized_entropy: f32,
    pub adjacent_valid_pairs: usize,
    pub normalized_detail_change_available: bool,
    pub normalized_detail_change: f32,
}

/// One trajectory family within a masked stationary-wavelet result.
#[derive(Clone, Copy, PartialEq)]
pub struct AcousticTrajectoryWaveletFamilySummary {
    pub family: AcousticTrajectoryFamily,
    pub owner: AcousticSidecarFeatureOwner,
    pub input_valid_frames: usize,
    pub input_was_constant_or_near_constant: bool,
    pub valid_level_count: usize,
    pub levels: [AcousticTrajectoryWaveletLevelSummary; ACOUSTIC_WAVELET_MAX_LEVELS],
}

impl Default for AcousticTrajectoryWaveletFamilySummary {
    fn default() -> Self {
        let family = AcousticTrajectoryFamily::default();
        Self {
            family,
            owner: family.owner(),
            input_valid_frames: 0,
            input_was_constant_or_near_constant: false,
            valid_level_count: 0,
            levels: [AcousticTrajectoryWaveletLevelSummary::default(); ACOUSTIC_WAVELET_MAX_LEVELS],
        }
    }
}

/// Fixed-size result of the stationary trajectory-wavelet candidate.
///
/// Raw trajectory values and coefficients never leave the sidecar state.
#[derive(Clone, Copy, PartialEq)]
pub struct AcousticTrajectoryWaveletSummary {
    pub basis: AcousticWaveletBasis,
    pub requested_levels: usize,
    pub window_start_frame_index: usize,
    pub window_end_frame_index: usize,
    pub families: [AcousticTrajectoryWaveletFamilySummary; ACOUSTIC_TRAJECTORY_FAMILY_COUNT],
    /// Low/high coefficient applications for outputs whose complete filter
    /// support was valid.
    pub filter_tap_terms: usize,
    /// Validity-mask positions inspected before coefficient calculation.
    pub validity_sample_visits: usize,
    /// Exact payload bytes of the fixed value/mask scratch arrays.
    pub scratch_buffer_payload_bytes: usize,
}

impl std::fmt::Debug for AcousticTrajectoryWaveletSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcousticTrajectoryWaveletSummary")
            .field("basis", &self.basis)
            .field("requested_levels", &self.requested_levels)
            .field("window_start_frame_index", &self.window_start_frame_index)
            .field("window_end_frame_index", &self.window_end_frame_index)
            .field(
                "available_family_count",
                &self
                    .families
                    .iter()
                    .filter(|family| family.valid_level_count > 0)
                    .count(),
            )
            .field("filter_tap_terms", &self.filter_tap_terms)
            .field("validity_sample_visits", &self.validity_sample_visits)
            .field(
                "scratch_buffer_payload_bytes",
                &self.scratch_buffer_payload_bytes,
            )
            .finish_non_exhaustive()
    }
}

/// One family of first- and second-order fixed-filter scattering summaries.
#[derive(Clone, Copy, PartialEq)]
pub struct AcousticScatteringFamilySummary {
    pub family: AcousticTrajectoryFamily,
    pub owner: AcousticSidecarFeatureOwner,
    pub input_valid_frames: usize,
    pub input_was_constant_or_near_constant: bool,
    pub first_order_available: [bool; ACOUSTIC_SCATTERING_SCALE_SUPPORTS.len()],
    pub first_order_valid_positions: [usize; ACOUSTIC_SCATTERING_SCALE_SUPPORTS.len()],
    pub first_order_mean_modulus: [f32; ACOUSTIC_SCATTERING_SCALE_SUPPORTS.len()],
    pub second_order_available: [bool; ACOUSTIC_SCATTERING_SCALE_PAIRS.len()],
    pub second_order_valid_positions: [usize; ACOUSTIC_SCATTERING_SCALE_PAIRS.len()],
    pub second_order_mean_modulus: [f32; ACOUSTIC_SCATTERING_SCALE_PAIRS.len()],
}

impl Default for AcousticScatteringFamilySummary {
    fn default() -> Self {
        let family = AcousticTrajectoryFamily::default();
        Self {
            family,
            owner: family.owner(),
            input_valid_frames: 0,
            input_was_constant_or_near_constant: false,
            first_order_available: [false; ACOUSTIC_SCATTERING_SCALE_SUPPORTS.len()],
            first_order_valid_positions: [0; ACOUSTIC_SCATTERING_SCALE_SUPPORTS.len()],
            first_order_mean_modulus: [0.0; ACOUSTIC_SCATTERING_SCALE_SUPPORTS.len()],
            second_order_available: [false; ACOUSTIC_SCATTERING_SCALE_PAIRS.len()],
            second_order_valid_positions: [0; ACOUSTIC_SCATTERING_SCALE_PAIRS.len()],
            second_order_mean_modulus: [0.0; ACOUSTIC_SCATTERING_SCALE_PAIRS.len()],
        }
    }
}

/// Fixed-size result of one configured scattering-order candidate.
#[derive(Clone, Copy, PartialEq)]
pub struct AcousticScatteringSummary {
    pub mode: AcousticScatteringMode,
    pub window_start_frame_index: usize,
    pub window_end_frame_index: usize,
    pub families: [AcousticScatteringFamilySummary; ACOUSTIC_TRAJECTORY_FAMILY_COUNT],
    /// Filter coefficient applications for valid non-wrapping output positions.
    pub filter_sample_terms: usize,
    /// Validity-mask positions inspected across first- and second-order paths.
    pub validity_sample_visits: usize,
    /// Exact payload bytes of fixed local value/mask scratch arrays.
    pub scratch_buffer_payload_bytes: usize,
}

impl std::fmt::Debug for AcousticScatteringSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcousticScatteringSummary")
            .field("mode", &self.mode)
            .field("window_start_frame_index", &self.window_start_frame_index)
            .field("window_end_frame_index", &self.window_end_frame_index)
            .field(
                "available_first_order_count",
                &self
                    .families
                    .iter()
                    .flat_map(|family| family.first_order_available)
                    .filter(|available| *available)
                    .count(),
            )
            .field(
                "available_second_order_count",
                &self
                    .families
                    .iter()
                    .flat_map(|family| family.second_order_available)
                    .filter(|available| *available)
                    .count(),
            )
            .field("filter_sample_terms", &self.filter_sample_terms)
            .field("validity_sample_visits", &self.validity_sample_visits)
            .field(
                "scratch_buffer_payload_bytes",
                &self.scratch_buffer_payload_bytes,
            )
            .finish_non_exhaustive()
    }
}

type AcousticTrajectoryValues =
    [[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
type AcousticTrajectoryValidity =
    [[bool; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; ACOUSTIC_TRAJECTORY_FAMILY_COUNT];

#[derive(Clone, Copy)]
struct AcousticTrajectoryWindow<'a> {
    values: &'a AcousticTrajectoryValues,
    valid: &'a AcousticTrajectoryValidity,
    oldest_index: usize,
    start_frame_index: usize,
    end_frame_index: usize,
}

impl<'a> AcousticTrajectoryWindow<'a> {
    const fn new(
        values: &'a AcousticTrajectoryValues,
        valid: &'a AcousticTrajectoryValidity,
        oldest_index: usize,
        start_frame_index: usize,
        end_frame_index: usize,
    ) -> Self {
        Self {
            values,
            valid,
            oldest_index,
            start_frame_index,
            end_frame_index,
        }
    }

    fn validate(self) -> FwResult<()> {
        if self.oldest_index >= ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
            return Err(FwError::InvalidRequest(
                "acoustic trajectory ring index exceeds its fixed support".to_owned(),
            ));
        }
        if self.end_frame_index.checked_sub(self.start_frame_index)
            != Some(ACOUSTIC_TRAJECTORY_HISTORY_FRAMES - 1)
        {
            return Err(FwError::InvalidRequest(
                "acoustic trajectory window must cover exactly 64 contiguous frame indices"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Default)]
struct AcousticFilterAccounting {
    filter_terms: usize,
    validity_visits: usize,
}

fn normalize_masked_trajectory(
    family: AcousticTrajectoryFamily,
    values: &[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    valid: &[bool; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    oldest_index: usize,
    normalized: &mut [f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    normalized_valid: &mut [bool; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
) -> FwResult<(usize, bool, bool)> {
    if oldest_index >= ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
        return Err(FwError::InvalidRequest(
            "acoustic trajectory ring index exceeds its fixed support".to_owned(),
        ));
    }
    let mut valid_count = 0usize;
    let mut sum = 0.0_f64;
    for (offset, normalized_is_valid) in normalized_valid.iter_mut().enumerate() {
        let ring_index = (oldest_index + offset) % ACOUSTIC_TRAJECTORY_HISTORY_FRAMES;
        *normalized_is_valid = valid[ring_index];
        if *normalized_is_valid {
            let value = values[ring_index];
            if !family.value_is_in_domain(value) {
                return Err(FwError::InvalidRequest(format!(
                    "acoustic trajectory value for {} is outside its finite declared domain",
                    family.id()
                )));
            }
            valid_count += 1;
            sum += f64::from(value);
        }
    }
    if valid_count < ACOUSTIC_TRAJECTORY_MIN_VALID_FRAMES {
        return Ok((valid_count, false, false));
    }
    let mean = sum / valid_count as f64;
    let mut centered_energy = 0.0_f64;
    for (offset, normalized_is_valid) in normalized_valid.iter().copied().enumerate() {
        if !normalized_is_valid {
            continue;
        }
        let ring_index = (oldest_index + offset) % ACOUSTIC_TRAJECTORY_HISTORY_FRAMES;
        let centered = f64::from(values[ring_index]) - mean;
        centered_energy += centered * centered;
    }
    let centered_rms = (centered_energy / valid_count as f64).sqrt();
    let centered_rms_floor =
        f64::from(ACOUSTIC_TRAJECTORY_CENTERED_RMS_RELATIVE_FLOOR) * mean.abs().max(1.0);
    if centered_rms <= centered_rms_floor {
        return Ok((valid_count, false, true));
    }
    let inverse_norm = centered_energy.sqrt().recip();
    for (offset, (normalized_value, normalized_is_valid)) in normalized
        .iter_mut()
        .zip(normalized_valid.iter().copied())
        .enumerate()
    {
        if !normalized_is_valid {
            *normalized_value = 0.0;
            continue;
        }
        let ring_index = (oldest_index + offset) % ACOUSTIC_TRAJECTORY_HISTORY_FRAMES;
        *normalized_value = ((f64::from(values[ring_index]) - mean) * inverse_norm) as f32;
        if !normalized_value.is_finite() {
            return Err(FwError::InvalidRequest(
                "acoustic trajectory normalization produced a non-finite value".to_owned(),
            ));
        }
    }
    Ok((valid_count, true, false))
}

fn trajectory_wavelet_pair(
    basis: AcousticWaveletBasis,
    input: &[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    base: usize,
    dilation: usize,
) -> (f32, f32) {
    match basis {
        AcousticWaveletBasis::Haar => {
            let left = f64::from(input[base]);
            let right = f64::from(input[base + dilation]);
            (
                (HAAR_LOW[0] * left + HAAR_LOW[1] * right) as f32,
                (HAAR_HIGH[0] * left + HAAR_HIGH[1] * right) as f32,
            )
        }
        AcousticWaveletBasis::DaubechiesFourTap => {
            let mut low = 0.0_f64;
            let mut high = 0.0_f64;
            for tap in 0..4 {
                let sample = f64::from(input[base + tap * dilation]);
                low += DAUBECHIES_FOUR_TAP_LOW[tap] * sample;
                high += DAUBECHIES_FOUR_TAP_HIGH[tap] * sample;
            }
            (low as f32, high as f32)
        }
    }
}

fn summarize_masked_trajectory_detail(
    detail: &[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    valid: &[bool; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    detail_len: usize,
) -> AcousticTrajectoryWaveletLevelSummary {
    let valid_coefficients = valid[..detail_len]
        .iter()
        .filter(|is_valid| **is_valid)
        .count();
    if valid_coefficients < ACOUSTIC_TRAJECTORY_MIN_VALID_COEFFICIENTS {
        return AcousticTrajectoryWaveletLevelSummary {
            valid_coefficients,
            ..AcousticTrajectoryWaveletLevelSummary::default()
        };
    }
    let detail_energy = detail[..detail_len]
        .iter()
        .zip(&valid[..detail_len])
        .filter(|(_, is_valid)| **is_valid)
        .map(|(value, _)| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    let mean_absolute_detail = detail[..detail_len]
        .iter()
        .zip(&valid[..detail_len])
        .filter_map(|(value, is_valid)| is_valid.then_some(f64::from(value.abs())))
        .sum::<f64>()
        / valid_coefficients as f64;
    let detail_rms = (detail_energy / valid_coefficients as f64).sqrt();
    let detail_shape_available = detail_rms > f64::from(ACOUSTIC_TRAJECTORY_DETAIL_RMS_FLOOR);
    let normalized_entropy = if detail_shape_available && valid_coefficients > 1 {
        let entropy = detail[..detail_len]
            .iter()
            .zip(&valid[..detail_len])
            .filter(|(_, is_valid)| **is_valid)
            .map(|(value, _)| {
                let probability = f64::from(*value) * f64::from(*value) / detail_energy;
                if probability > 0.0 {
                    -probability * probability.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        (entropy / (valid_coefficients as f64).ln()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut adjacent_pairs = 0usize;
    let mut adjacent_change = 0.0_f64;
    for index in 1..detail_len {
        if valid[index - 1] && valid[index] {
            adjacent_pairs += 1;
            adjacent_change += (f64::from(detail[index]) - f64::from(detail[index - 1])).abs();
        }
    }
    let normalized_detail_change = if adjacent_pairs > 0 && detail_shape_available {
        adjacent_change / adjacent_pairs as f64 / detail_rms
    } else {
        0.0
    };
    AcousticTrajectoryWaveletLevelSummary {
        available: true,
        valid_coefficients,
        mean_absolute_detail: mean_absolute_detail as f32,
        detail_rms: detail_rms as f32,
        normalized_entropy_available: detail_shape_available && valid_coefficients > 1,
        normalized_entropy: normalized_entropy as f32,
        adjacent_valid_pairs: adjacent_pairs,
        normalized_detail_change_available: adjacent_pairs > 0 && detail_shape_available,
        normalized_detail_change: normalized_detail_change as f32,
    }
}

fn analyze_acoustic_trajectory_wavelet<C>(
    window: AcousticTrajectoryWindow<'_>,
    basis: AcousticWaveletBasis,
    requested_levels: usize,
    is_cancelled: &mut C,
) -> FwResult<AcousticTrajectoryWaveletSummary>
where
    C: FnMut() -> bool,
{
    window.validate()?;
    if requested_levels == 0 || requested_levels > ACOUSTIC_WAVELET_MAX_LEVELS {
        return Err(FwError::InvalidRequest(format!(
            "trajectory wavelet levels must be within 1..={ACOUSTIC_WAVELET_MAX_LEVELS}"
        )));
    }
    let mut families = std::array::from_fn(|index| AcousticTrajectoryWaveletFamilySummary {
        family: AcousticTrajectoryFamily::ALL[index],
        owner: AcousticTrajectoryFamily::ALL[index].owner(),
        ..AcousticTrajectoryWaveletFamilySummary::default()
    });
    let mut accounting = AcousticFilterAccounting::default();
    let scratch_buffer_payload_bytes = ACOUSTIC_TRAJECTORY_WAVELET_SCRATCH_PAYLOAD_BYTES;
    for (family_index, family_summary) in families.iter_mut().enumerate() {
        if is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "acoustic trajectory wavelet cancelled before family {}",
                family_summary.family.id()
            )));
        }
        let mut current = [0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        let mut current_valid = [false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        let (input_valid_frames, input_available, input_was_constant_or_near_constant) =
            normalize_masked_trajectory(
                family_summary.family,
                &window.values[family_index],
                &window.valid[family_index],
                window.oldest_index,
                &mut current,
                &mut current_valid,
            )?;
        family_summary.input_valid_frames = input_valid_frames;
        family_summary.input_was_constant_or_near_constant = input_was_constant_or_near_constant;
        if !input_available {
            continue;
        }
        let mut approximation = [0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        let mut approximation_valid = [false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        let mut detail = [0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        let mut detail_valid = [false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        let mut current_len = ACOUSTIC_TRAJECTORY_HISTORY_FRAMES;
        let tap_count: usize = match basis {
            AcousticWaveletBasis::Haar => 2,
            AcousticWaveletBasis::DaubechiesFourTap => 4,
        };
        for (level, level_summary) in family_summary
            .levels
            .iter_mut()
            .take(requested_levels)
            .enumerate()
        {
            if is_cancelled() {
                return Err(FwError::Cancelled(format!(
                    "acoustic trajectory wavelet cancelled before family {} level {level}",
                    family_summary.family.id()
                )));
            }
            let dilation = 1usize.checked_shl(level as u32).ok_or_else(|| {
                FwError::InvalidRequest(
                    "trajectory wavelet dilation exceeds size arithmetic".to_owned(),
                )
            })?;
            let maximum_tap_offset = (tap_count - 1).checked_mul(dilation).ok_or_else(|| {
                FwError::InvalidRequest(
                    "trajectory wavelet support exceeds size arithmetic".to_owned(),
                )
            })?;
            let output_len = current_len.checked_sub(maximum_tap_offset).ok_or_else(|| {
                FwError::InvalidRequest(
                    "trajectory wavelet level is shorter than its analysis filter".to_owned(),
                )
            })?;
            for output_index in 0..output_len {
                let base = output_index;
                let mut support_valid = true;
                for tap in 0..tap_count {
                    accounting.validity_visits =
                        accounting.validity_visits.checked_add(1).ok_or_else(|| {
                            FwError::InvalidRequest(
                                "trajectory wavelet validity accounting overflowed".to_owned(),
                            )
                        })?;
                    support_valid &= current_valid[base + tap * dilation];
                }
                approximation_valid[output_index] = support_valid;
                detail_valid[output_index] = support_valid;
                if !support_valid {
                    approximation[output_index] = 0.0;
                    detail[output_index] = 0.0;
                    continue;
                }
                let (low, high) = trajectory_wavelet_pair(basis, &current, base, dilation);
                approximation[output_index] = low;
                detail[output_index] = high;
                accounting.filter_terms = accounting
                    .filter_terms
                    .checked_add(2 * tap_count)
                    .ok_or_else(|| {
                        FwError::InvalidRequest(
                            "trajectory wavelet filter accounting overflowed".to_owned(),
                        )
                    })?;
            }
            *level_summary = summarize_masked_trajectory_detail(&detail, &detail_valid, output_len);
            if !level_summary.available {
                break;
            }
            family_summary.valid_level_count += 1;
            current[..output_len].copy_from_slice(&approximation[..output_len]);
            current_valid[..output_len].copy_from_slice(&approximation_valid[..output_len]);
            approximation[..output_len].fill(0.0);
            approximation_valid[..output_len].fill(false);
            detail[..output_len].fill(0.0);
            detail_valid[..output_len].fill(false);
            current_len = output_len;
        }
    }
    Ok(AcousticTrajectoryWaveletSummary {
        basis,
        requested_levels,
        window_start_frame_index: window.start_frame_index,
        window_end_frame_index: window.end_frame_index,
        families,
        filter_tap_terms: accounting.filter_terms,
        validity_sample_visits: accounting.validity_visits,
        scratch_buffer_payload_bytes,
    })
}

fn valid_haar_modulus(
    input: &[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    valid: &[bool; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    input_len: usize,
    support: usize,
    normalizer: f64,
    output: (
        &mut [f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
        &mut [bool; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    ),
    accounting: &mut AcousticFilterAccounting,
) -> FwResult<usize> {
    if input_len > ACOUSTIC_TRAJECTORY_HISTORY_FRAMES || support == 0 || support > input_len {
        return Err(FwError::InvalidRequest(
            "scattering filter support exceeds its non-wrapping input geometry".to_owned(),
        ));
    }
    let (output_values, output_valid) = output;
    output_values.fill(0.0);
    output_valid.fill(false);
    let mut valid_positions = 0usize;
    let output_len = input_len - support + 1;
    for position in 0..output_len {
        let mut support_valid = true;
        for tap in 0..support {
            accounting.validity_visits =
                accounting.validity_visits.checked_add(1).ok_or_else(|| {
                    FwError::InvalidRequest("scattering validity accounting overflowed".to_owned())
                })?;
            support_valid &= valid[position + tap];
        }
        output_valid[position] = support_valid;
        if !support_valid {
            output_values[position] = 0.0;
            continue;
        }
        let mut response = 0.0_f64;
        for tap in 0..support {
            let sign = if tap < support / 2 { 1.0 } else { -1.0 };
            response += sign * f64::from(input[position + tap]);
        }
        output_values[position] = (response * normalizer).abs() as f32;
        if !output_values[position].is_finite() {
            return Err(FwError::InvalidRequest(
                "scattering filter produced a non-finite modulus".to_owned(),
            ));
        }
        valid_positions += 1;
        accounting.filter_terms =
            accounting
                .filter_terms
                .checked_add(support)
                .ok_or_else(|| {
                    FwError::InvalidRequest("scattering filter accounting overflowed".to_owned())
                })?;
    }
    Ok(valid_positions)
}

fn mean_valid_modulus(
    values: &[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    valid: &[bool; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    valid_count: usize,
) -> f32 {
    if valid_count == 0 {
        return 0.0;
    }
    (values
        .iter()
        .zip(valid)
        .filter_map(|(value, is_valid)| is_valid.then_some(f64::from(*value)))
        .sum::<f64>()
        / valid_count as f64) as f32
}

fn analyze_acoustic_scattering<C>(
    window: AcousticTrajectoryWindow<'_>,
    mode: AcousticScatteringMode,
    is_cancelled: &mut C,
) -> FwResult<AcousticScatteringSummary>
where
    C: FnMut() -> bool,
{
    window.validate()?;
    if !mode.is_enabled() {
        return Err(FwError::InvalidRequest(
            "scattering analysis requires a selected output order".to_owned(),
        ));
    }
    let mut families = std::array::from_fn(|index| AcousticScatteringFamilySummary {
        family: AcousticTrajectoryFamily::ALL[index],
        owner: AcousticTrajectoryFamily::ALL[index].owner(),
        ..AcousticScatteringFamilySummary::default()
    });
    let mut accounting = AcousticFilterAccounting::default();
    let scratch_buffer_payload_bytes = ACOUSTIC_SCATTERING_FIRST_ORDER_SCRATCH_PAYLOAD_BYTES
        + if mode.emits_second_order() {
            ACOUSTIC_SCATTERING_SECOND_ORDER_EXTRA_SCRATCH_PAYLOAD_BYTES
        } else {
            0
        };
    for (family_index, family_summary) in families.iter_mut().enumerate() {
        if is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "acoustic scattering sidecar cancelled before family {}",
                family_summary.family.id()
            )));
        }
        let mut normalized = [0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        let mut normalized_valid = [false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        let (input_valid_frames, input_available, input_was_constant_or_near_constant) =
            normalize_masked_trajectory(
                family_summary.family,
                &window.values[family_index],
                &window.valid[family_index],
                window.oldest_index,
                &mut normalized,
                &mut normalized_valid,
            )?;
        family_summary.input_valid_frames = input_valid_frames;
        family_summary.input_was_constant_or_near_constant = input_was_constant_or_near_constant;
        if !input_available {
            continue;
        }
        let mut first_modulus = [[0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            ACOUSTIC_SCATTERING_SCALE_SUPPORTS.len()];
        let mut first_valid =
            [[false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; ACOUSTIC_SCATTERING_SCALE_SUPPORTS.len()];
        let first_output_lengths = ACOUSTIC_SCATTERING_SCALE_SUPPORTS
            .map(|support| ACOUSTIC_TRAJECTORY_HISTORY_FRAMES - support + 1);
        let first_scale_count = if mode.emits_first_order() {
            ACOUSTIC_SCATTERING_SCALE_SUPPORTS.len()
        } else {
            2
        };
        for scale_index in 0..first_scale_count {
            if is_cancelled() {
                return Err(FwError::Cancelled(format!(
                    "acoustic scattering sidecar cancelled before family {} first-order scale {scale_index}",
                    family_summary.family.id()
                )));
            }
            let valid_positions = valid_haar_modulus(
                &normalized,
                &normalized_valid,
                ACOUSTIC_TRAJECTORY_HISTORY_FRAMES,
                ACOUSTIC_SCATTERING_SCALE_SUPPORTS[scale_index],
                ACOUSTIC_SCATTERING_SCALE_NORMALIZERS[scale_index],
                (
                    &mut first_modulus[scale_index],
                    &mut first_valid[scale_index],
                ),
                &mut accounting,
            )?;
            if mode.emits_first_order() {
                family_summary.first_order_valid_positions[scale_index] = valid_positions;
                family_summary.first_order_available[scale_index] =
                    valid_positions >= ACOUSTIC_SCATTERING_MIN_VALID_OUTPUTS;
                if family_summary.first_order_available[scale_index] {
                    family_summary.first_order_mean_modulus[scale_index] = mean_valid_modulus(
                        &first_modulus[scale_index],
                        &first_valid[scale_index],
                        valid_positions,
                    );
                }
            }
        }
        if mode.emits_second_order() {
            let mut second_modulus = [0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            let mut second_valid = [false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            for (pair_index, [first_scale, second_scale]) in
                ACOUSTIC_SCATTERING_SCALE_PAIRS.into_iter().enumerate()
            {
                if is_cancelled() {
                    return Err(FwError::Cancelled(format!(
                        "acoustic scattering sidecar cancelled before family {} second-order pair {pair_index}",
                        family_summary.family.id()
                    )));
                }
                let valid_positions = valid_haar_modulus(
                    &first_modulus[first_scale],
                    &first_valid[first_scale],
                    first_output_lengths[first_scale],
                    ACOUSTIC_SCATTERING_SCALE_SUPPORTS[second_scale],
                    ACOUSTIC_SCATTERING_SCALE_NORMALIZERS[second_scale],
                    (&mut second_modulus, &mut second_valid),
                    &mut accounting,
                )?;
                family_summary.second_order_valid_positions[pair_index] = valid_positions;
                family_summary.second_order_available[pair_index] =
                    valid_positions >= ACOUSTIC_SCATTERING_MIN_VALID_OUTPUTS;
                if family_summary.second_order_available[pair_index] {
                    family_summary.second_order_mean_modulus[pair_index] =
                        mean_valid_modulus(&second_modulus, &second_valid, valid_positions);
                }
                second_modulus.fill(0.0);
                second_valid.fill(false);
            }
        }
    }
    Ok(AcousticScatteringSummary {
        mode,
        window_start_frame_index: window.start_frame_index,
        window_end_frame_index: window.end_frame_index,
        families,
        filter_sample_terms: accounting.filter_terms,
        validity_sample_visits: accounting.validity_visits,
        scratch_buffer_payload_bytes,
    })
}

#[derive(Clone)]
struct AcousticTrajectorySidecar {
    values: [[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; ACOUSTIC_TRAJECTORY_FAMILY_COUNT],
    valid: [[bool; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; ACOUSTIC_TRAJECTORY_FAMILY_COUNT],
    next_index: usize,
    buffered_frames: usize,
    expected_next_frame_index: Option<usize>,
}

impl std::fmt::Debug for AcousticTrajectorySidecar {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcousticTrajectorySidecar")
            .field("next_index", &self.next_index)
            .field("buffered_frames", &self.buffered_frames)
            .field("expected_next_frame_index", &self.expected_next_frame_index)
            .field(
                "retained_state_bytes_on_target",
                &Self::retained_state_bytes_on_target(),
            )
            .finish_non_exhaustive()
    }
}

impl AcousticTrajectorySidecar {
    const fn new() -> Self {
        Self {
            values: [[0.0; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; ACOUSTIC_TRAJECTORY_FAMILY_COUNT],
            valid: [[false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; ACOUSTIC_TRAJECTORY_FAMILY_COUNT],
            next_index: 0,
            buffered_frames: 0,
            expected_next_frame_index: None,
        }
    }

    const fn retained_state_bytes_on_target() -> usize {
        std::mem::size_of::<Self>()
    }

    fn push<C>(
        &mut self,
        frame: &AcousticFrameFeatures,
        config: AcousticSidecarStudyConfig,
        is_cancelled: &mut C,
    ) -> FwResult<(
        Option<AcousticTrajectoryWaveletSummary>,
        Option<AcousticScatteringSummary>,
    )>
    where
        C: FnMut() -> bool,
    {
        if is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "acoustic trajectory sidecar cancelled before frame {}",
                frame.frame_index
            )));
        }
        validate_acoustic_frame(frame)?;
        if let Some(expected) = self.expected_next_frame_index
            && frame.frame_index != expected
        {
            return Err(FwError::InvalidRequest(format!(
                "acoustic trajectory sidecar expected frame {expected}, got {}",
                frame.frame_index
            )));
        }
        let next_expected = frame.frame_index.checked_add(1).ok_or_else(|| {
            FwError::InvalidRequest(
                "acoustic trajectory frame index exceeds the supported range".to_owned(),
            )
        })?;
        for (family_index, family) in AcousticTrajectoryFamily::ALL.into_iter().enumerate() {
            let (value, is_valid) = family.value_and_valid(frame);
            self.values[family_index][self.next_index] = value;
            self.valid[family_index][self.next_index] = is_valid;
        }
        self.next_index = (self.next_index + 1) % ACOUSTIC_TRAJECTORY_HISTORY_FRAMES;
        self.buffered_frames = (self.buffered_frames + 1).min(ACOUSTIC_TRAJECTORY_HISTORY_FRAMES);
        self.expected_next_frame_index = Some(next_expected);
        if self.buffered_frames < ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
            return Ok((None, None));
        }
        let window_start_frame_index = frame
            .frame_index
            .checked_add(1)
            .and_then(|end| end.checked_sub(ACOUSTIC_TRAJECTORY_HISTORY_FRAMES))
            .ok_or_else(|| {
                FwError::InvalidRequest(
                    "acoustic trajectory window index exceeds the supported range".to_owned(),
                )
            })?;
        let window = AcousticTrajectoryWindow::new(
            &self.values,
            &self.valid,
            self.next_index,
            window_start_frame_index,
            frame.frame_index,
        );
        let trajectory_wavelet = if let Some(basis) = config.trajectory_wavelet_mode.basis() {
            Some(analyze_acoustic_trajectory_wavelet(
                window,
                basis,
                config.trajectory_wavelet_levels,
                is_cancelled,
            )?)
        } else {
            None
        };
        let scattering = if config.scattering_mode.is_enabled() {
            Some(analyze_acoustic_scattering(
                window,
                config.scattering_mode,
                is_cancelled,
            )?)
        } else {
            None
        };
        Ok((trajectory_wavelet, scattering))
    }
}

/// Fixed-window modulation-spectrum summaries with explicit voice/channel
/// ownership. No raw trajectory or coefficient leaves this type.
#[derive(Clone, Copy, PartialEq)]
pub struct AcousticModulationSummary {
    pub window_start_frame_index: usize,
    pub window_end_frame_index: usize,
    pub voice_available: bool,
    pub channel_level_available: bool,
    pub channel_coloration_available: bool,
    pub voice_valid_frames: usize,
    pub channel_valid_frames: usize,
    pub voice_normalized_power: [f32; 4],
    pub channel_level_normalized_power: [f32; 4],
    pub channel_coloration_normalized_power: [f32; 4],
    pub voice_owner: AcousticSidecarFeatureOwner,
    pub channel_level_owner: AcousticSidecarFeatureOwner,
    pub channel_coloration_owner: AcousticSidecarFeatureOwner,
    /// Number of valid sample-frequency visits across the basis-centering and
    /// regression passes. This is not a floating-point-operation count.
    pub projection_sample_frequency_visits: usize,
    /// Exact in-struct bytes on this compilation target. This is not RSS and
    /// is not a cross-target canonical size.
    pub retained_state_bytes_on_target: usize,
    /// Exact payload bytes of the process-global complex twiddle table.
    pub cached_twiddle_payload_bytes: usize,
}

impl std::fmt::Debug for AcousticModulationSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcousticModulationSummary")
            .field("window_start_frame_index", &self.window_start_frame_index)
            .field("window_end_frame_index", &self.window_end_frame_index)
            .field("voice_available", &self.voice_available)
            .field("channel_level_available", &self.channel_level_available)
            .field(
                "channel_coloration_available",
                &self.channel_coloration_available,
            )
            .field("voice_valid_frames", &self.voice_valid_frames)
            .field("channel_valid_frames", &self.channel_valid_frames)
            .field(
                "projection_sample_frequency_visits",
                &self.projection_sample_frequency_visits,
            )
            .field(
                "retained_state_bytes_on_target",
                &self.retained_state_bytes_on_target,
            )
            .field(
                "cached_twiddle_payload_bytes",
                &self.cached_twiddle_payload_bytes,
            )
            .finish_non_exhaustive()
    }
}

/// Incremental, fixed-memory modulation sidecar over acoustic-v2 frames.
///
/// Voice modulation uses the gain-centered cepstral-delta magnitude already
/// present in [`VoiceFeatureView`]. Level and muffling trajectories remain in
/// separate channel-owned outputs and must not become reusable identity.
#[derive(Clone)]
pub struct AcousticModulationSidecar {
    voice: [f32; ACOUSTIC_MODULATION_HISTORY_FRAMES],
    channel_level: [f32; ACOUSTIC_MODULATION_HISTORY_FRAMES],
    channel_coloration: [f32; ACOUSTIC_MODULATION_HISTORY_FRAMES],
    voice_valid: [bool; ACOUSTIC_MODULATION_HISTORY_FRAMES],
    channel_valid: [bool; ACOUSTIC_MODULATION_HISTORY_FRAMES],
    next_index: usize,
    buffered_frames: usize,
    expected_next_frame_index: Option<usize>,
}

impl std::fmt::Debug for AcousticModulationSidecar {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcousticModulationSidecar")
            .field("next_index", &self.next_index)
            .field("buffered_frames", &self.buffered_frames)
            .field("expected_next_frame_index", &self.expected_next_frame_index)
            .field(
                "retained_state_bytes_on_target",
                &self.retained_state_bytes_on_target(),
            )
            .finish_non_exhaustive()
    }
}

impl Default for AcousticModulationSidecar {
    fn default() -> Self {
        Self::new()
    }
}

impl AcousticModulationSidecar {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            voice: [0.0; ACOUSTIC_MODULATION_HISTORY_FRAMES],
            channel_level: [0.0; ACOUSTIC_MODULATION_HISTORY_FRAMES],
            channel_coloration: [0.0; ACOUSTIC_MODULATION_HISTORY_FRAMES],
            voice_valid: [false; ACOUSTIC_MODULATION_HISTORY_FRAMES],
            channel_valid: [false; ACOUSTIC_MODULATION_HISTORY_FRAMES],
            next_index: 0,
            buffered_frames: 0,
            expected_next_frame_index: None,
        }
    }

    #[must_use]
    pub const fn retained_state_bytes_on_target(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    /// Push one contiguous acoustic frame. The first summary is emitted after
    /// exactly 64 frames; subsequent calls emit a one-frame-sliding result.
    /// Cancellation and validation happen before state mutation.
    pub fn push<C>(
        &mut self,
        frame: &AcousticFrameFeatures,
        is_cancelled: &mut C,
    ) -> FwResult<Option<AcousticModulationSummary>>
    where
        C: FnMut() -> bool,
    {
        if is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "acoustic modulation sidecar cancelled before frame {}",
                frame.frame_index
            )));
        }
        validate_acoustic_frame(frame)?;
        if let Some(expected) = self.expected_next_frame_index
            && frame.frame_index != expected
        {
            return Err(FwError::InvalidRequest(format!(
                "acoustic modulation sidecar expected frame {expected}, got {}",
                frame.frame_index
            )));
        }
        let next_expected = frame.frame_index.checked_add(1).ok_or_else(|| {
            FwError::InvalidRequest(
                "acoustic modulation sidecar frame index exceeds the supported range".to_owned(),
            )
        })?;
        let mut staged = self.clone();
        let voice_valid = frame.quality.voiced
            && !frame.quality.low_energy
            && !frame.quality.clipped
            && !frame.quality.transient
            && frame.frame_index > 0;
        let channel_valid = !frame.quality.low_energy && !frame.quality.clipped;
        staged.voice[staged.next_index] = frame.voice.temporal_modulation;
        staged.channel_level[staged.next_index] = frame.channel.rms_dbfs;
        staged.channel_coloration[staged.next_index] = frame.channel.muffling_proxy;
        staged.voice_valid[staged.next_index] = voice_valid;
        staged.channel_valid[staged.next_index] = channel_valid;
        staged.next_index = (staged.next_index + 1) % ACOUSTIC_MODULATION_HISTORY_FRAMES;
        staged.buffered_frames =
            (staged.buffered_frames + 1).min(ACOUSTIC_MODULATION_HISTORY_FRAMES);
        staged.expected_next_frame_index = Some(next_expected);
        if staged.buffered_frames < ACOUSTIC_MODULATION_HISTORY_FRAMES {
            *self = staged;
            return Ok(None);
        }

        let (voice_available, voice_valid_frames, voice_normalized_power, voice_terms) =
            modulation_spectrum(
                &staged.voice,
                &staged.voice_valid,
                staged.next_index,
                "voice",
                is_cancelled,
            )?;
        let (
            channel_level_available,
            channel_valid_frames,
            channel_level_normalized_power,
            channel_level_terms,
        ) = modulation_spectrum(
            &staged.channel_level,
            &staged.channel_valid,
            staged.next_index,
            "channel-level",
            is_cancelled,
        )?;
        let (
            channel_coloration_available,
            coloration_valid_frames,
            channel_coloration_normalized_power,
            channel_coloration_terms,
        ) = modulation_spectrum(
            &staged.channel_coloration,
            &staged.channel_valid,
            staged.next_index,
            "channel-coloration",
            is_cancelled,
        )?;
        debug_assert_eq!(channel_valid_frames, coloration_valid_frames);
        let window_start_frame_index = frame
            .frame_index
            .checked_add(1)
            .and_then(|end| end.checked_sub(ACOUSTIC_MODULATION_HISTORY_FRAMES))
            .ok_or_else(|| {
                FwError::InvalidRequest(
                    "acoustic modulation sidecar window index exceeds the supported range"
                        .to_owned(),
                )
            })?;
        let projection_sample_frequency_visits = voice_terms
            .checked_add(channel_level_terms)
            .and_then(|terms| terms.checked_add(channel_coloration_terms))
            .ok_or_else(|| {
                FwError::InvalidRequest(
                    "acoustic modulation operation accounting overflowed".to_owned(),
                )
            })?;
        let summary = AcousticModulationSummary {
            window_start_frame_index,
            window_end_frame_index: frame.frame_index,
            voice_available,
            channel_level_available,
            channel_coloration_available,
            voice_valid_frames,
            channel_valid_frames,
            voice_normalized_power,
            channel_level_normalized_power,
            channel_coloration_normalized_power,
            voice_owner: AcousticSidecarFeatureOwner::Voice,
            channel_level_owner: AcousticSidecarFeatureOwner::Channel,
            channel_coloration_owner: AcousticSidecarFeatureOwner::Channel,
            projection_sample_frequency_visits,
            retained_state_bytes_on_target: staged.retained_state_bytes_on_target(),
            cached_twiddle_payload_bytes: std::mem::size_of::<ModulationTwiddles>(),
        };
        *self = staged;
        Ok(Some(summary))
    }
}

type ModulationTwiddles = [[[f64; 2]; ACOUSTIC_MODULATION_HISTORY_FRAMES]; 4];

const fn build_modulation_twiddles() -> ModulationTwiddles {
    let mut twiddles = [[[0.0_f64; 2]; ACOUSTIC_MODULATION_HISTORY_FRAMES]; 4];
    let mut frequency = 0usize;
    while frequency < ACOUSTIC_MODULATION_STEP_COMPLEX.len() {
        let step = ACOUSTIC_MODULATION_STEP_COMPLEX[frequency];
        let mut current = [1.0_f64, 0.0_f64];
        let mut offset = 0usize;
        while offset < ACOUSTIC_MODULATION_HISTORY_FRAMES {
            twiddles[frequency][offset] = current;
            current = [
                current[0] * step[0] - current[1] * step[1],
                current[0] * step[1] + current[1] * step[0],
            ];
            offset += 1;
        }
        frequency += 1;
    }
    twiddles
}

static MODULATION_TWIDDLES: ModulationTwiddles = build_modulation_twiddles();

fn cached_modulation_twiddles() -> &'static ModulationTwiddles {
    &MODULATION_TWIDDLES
}

fn modulation_spectrum<C>(
    values: &[f32; ACOUSTIC_MODULATION_HISTORY_FRAMES],
    valid: &[bool; ACOUSTIC_MODULATION_HISTORY_FRAMES],
    oldest_index: usize,
    family: &str,
    is_cancelled: &mut C,
) -> FwResult<(bool, usize, [f32; 4], usize)>
where
    C: FnMut() -> bool,
{
    if is_cancelled() {
        return Err(FwError::Cancelled(format!(
            "acoustic modulation sidecar cancelled before {family} projection"
        )));
    }
    let valid_count = valid.iter().filter(|is_valid| **is_valid).count();
    if valid_count < ACOUSTIC_MODULATION_MIN_VALID_FRAMES {
        return Ok((false, valid_count, [0.0; 4], 0));
    }
    let mean = (0..ACOUSTIC_MODULATION_HISTORY_FRAMES)
        .filter_map(|offset| {
            let index = (oldest_index + offset) % ACOUSTIC_MODULATION_HISTORY_FRAMES;
            valid[index].then_some(f64::from(values[index]))
        })
        .sum::<f64>()
        / valid_count as f64;
    let centered_energy = (0..ACOUSTIC_MODULATION_HISTORY_FRAMES)
        .filter_map(|offset| {
            let index = (oldest_index + offset) % ACOUSTIC_MODULATION_HISTORY_FRAMES;
            valid[index].then(|| {
                let centered = f64::from(values[index]) - mean;
                centered * centered
            })
        })
        .sum::<f64>();
    let centered_rms = (centered_energy / valid_count as f64).sqrt();
    let centered_rms_floor =
        f64::from(ACOUSTIC_MODULATION_CENTERED_RMS_RELATIVE_FLOOR) * mean.abs().max(1.0);
    if centered_rms <= centered_rms_floor {
        return Ok((false, valid_count, [0.0; 4], 0));
    }
    let mut power = [0.0_f32; 4];
    let mut projection_sample_frequency_visits = 0usize;
    let twiddles = cached_modulation_twiddles();
    for (frequency, output) in power.iter_mut().enumerate() {
        if is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "acoustic modulation sidecar cancelled before {family} frequency {frequency}"
            )));
        }
        let mut basis_real_mean = 0.0_f64;
        let mut basis_imaginary_mean = 0.0_f64;
        let frequency_twiddles = &twiddles[frequency];
        for (offset, twiddle) in frequency_twiddles.iter().enumerate() {
            let index = (oldest_index + offset) % ACOUSTIC_MODULATION_HISTORY_FRAMES;
            if !valid[index] {
                continue;
            }
            basis_real_mean += twiddle[0];
            basis_imaginary_mean += twiddle[1];
            projection_sample_frequency_visits += 1;
        }
        basis_real_mean /= valid_count as f64;
        basis_imaginary_mean /= valid_count as f64;

        let mut gram_real_real = 0.0_f64;
        let mut gram_real_imaginary = 0.0_f64;
        let mut gram_imaginary_imaginary = 0.0_f64;
        let mut projection_real = 0.0_f64;
        let mut projection_imaginary = 0.0_f64;
        for (offset, twiddle) in frequency_twiddles.iter().enumerate() {
            let index = (oldest_index + offset) % ACOUSTIC_MODULATION_HISTORY_FRAMES;
            if !valid[index] {
                continue;
            }
            let centered = f64::from(values[index]) - mean;
            let basis_real = twiddle[0] - basis_real_mean;
            let basis_imaginary = twiddle[1] - basis_imaginary_mean;
            gram_real_real += basis_real * basis_real;
            gram_real_imaginary += basis_real * basis_imaginary;
            gram_imaginary_imaginary += basis_imaginary * basis_imaginary;
            projection_real += centered * basis_real;
            projection_imaginary += centered * basis_imaginary;
            projection_sample_frequency_visits += 1;
        }
        let determinant =
            gram_real_real * gram_imaginary_imaginary - gram_real_imaginary * gram_real_imaginary;
        let determinant_floor =
            f64::from(PCM_EPSILON) * (gram_real_real * gram_imaginary_imaginary).max(1.0);
        if !determinant.is_finite() || determinant <= determinant_floor {
            return Ok((
                false,
                valid_count,
                [0.0; 4],
                projection_sample_frequency_visits,
            ));
        }
        let coefficient_real = (projection_real * gram_imaginary_imaginary
            - projection_imaginary * gram_real_imaginary)
            / determinant;
        let coefficient_imaginary = (projection_imaginary * gram_real_real
            - projection_real * gram_real_imaginary)
            / determinant;
        let explained_energy =
            coefficient_real * projection_real + coefficient_imaginary * projection_imaginary;
        let normalized_power = explained_energy / centered_energy;
        if !normalized_power.is_finite() {
            return Err(FwError::InvalidRequest(format!(
                "acoustic modulation sidecar produced non-finite {family} frequency {frequency}"
            )));
        }
        *output = normalized_power.clamp(0.0, 1.0) as f32;
    }
    Ok((true, valid_count, power, projection_sample_frequency_visits))
}

/// One configuration-bound sidecar observation at the acoustic-v2 cadence.
///
/// The contained feature summaries remain signal-derived and intentionally
/// have no serialization implementation. `Debug` exposes only identities and
/// availability, never feature values.
#[derive(Clone, Copy, PartialEq)]
pub struct AcousticSidecarStudyObservation {
    schema_version: &'static str,
    configuration_sha256_digest: [u8; 32],
    config: AcousticSidecarStudyConfig,
    frame_index: usize,
    wavelet: Option<AcousticWaveletSummary>,
    modulation: Option<AcousticModulationSummary>,
    trajectory_wavelet: Option<AcousticTrajectoryWaveletSummary>,
    scattering: Option<AcousticScatteringSummary>,
}

impl AcousticSidecarStudyObservation {
    #[must_use]
    pub const fn schema_version(&self) -> &'static str {
        self.schema_version
    }

    #[must_use]
    pub const fn configuration_sha256_digest(&self) -> [u8; 32] {
        self.configuration_sha256_digest
    }

    #[must_use]
    pub const fn config(&self) -> AcousticSidecarStudyConfig {
        self.config
    }

    #[must_use]
    pub const fn frame_index(&self) -> usize {
        self.frame_index
    }

    #[must_use]
    pub const fn wavelet(&self) -> Option<AcousticWaveletSummary> {
        self.wavelet
    }

    #[must_use]
    pub const fn modulation(&self) -> Option<AcousticModulationSummary> {
        self.modulation
    }

    #[must_use]
    pub const fn trajectory_wavelet(&self) -> Option<AcousticTrajectoryWaveletSummary> {
        self.trajectory_wavelet
    }

    #[must_use]
    pub const fn scattering(&self) -> Option<AcousticScatteringSummary> {
        self.scattering
    }
}

impl std::fmt::Debug for AcousticSidecarStudyObservation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcousticSidecarStudyObservation")
            .field("schema_version", &self.schema_version)
            .field("config", &self.config)
            .field("frame_index", &self.frame_index)
            .field("wavelet_available", &self.wavelet.is_some())
            .field("modulation_available", &self.modulation.is_some())
            .field(
                "trajectory_wavelet_available",
                &self.trajectory_wavelet.is_some(),
            )
            .field("scattering_available", &self.scattering.is_some())
            .finish_non_exhaustive()
    }
}

/// Authoritative executor that binds one study configuration to its kernels.
///
/// The wavelet input is statically fixed to one normalized 16 kHz, 400-sample
/// acoustic frame. This prevents a single configuration digest from silently
/// describing different physical supports. The executor is never constructed
/// by the default diarization path.
pub struct AcousticSidecarStudy {
    config: AcousticSidecarStudyConfig,
    configuration_sha256_digest: [u8; 32],
    modulation: AcousticModulationSidecar,
    trajectory: AcousticTrajectorySidecar,
}

impl std::fmt::Debug for AcousticSidecarStudy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcousticSidecarStudy")
            .field("mode", &self.config.mode)
            .field("frame_wavelet_levels", &self.config.frame_wavelet_levels)
            .field(
                "trajectory_wavelet_mode",
                &self.config.trajectory_wavelet_mode,
            )
            .field(
                "trajectory_wavelet_levels",
                &self.config.trajectory_wavelet_levels,
            )
            .field("scattering_mode", &self.config.scattering_mode)
            .field(
                "retained_state_bytes_on_target",
                &self.retained_state_bytes_on_target(),
            )
            .finish_non_exhaustive()
    }
}

impl AcousticSidecarStudy {
    pub fn new(config: AcousticSidecarStudyConfig) -> FwResult<Self> {
        let configuration_sha256_digest = acoustic_sidecar_study_config_digest(config)?;
        Ok(Self {
            config,
            configuration_sha256_digest,
            modulation: AcousticModulationSidecar::new(),
            trajectory: AcousticTrajectorySidecar::new(),
        })
    }

    #[must_use]
    pub const fn config(&self) -> AcousticSidecarStudyConfig {
        self.config
    }

    #[must_use]
    pub fn configuration_sha256_hex(&self) -> String {
        sidecar_sha256_hex(&self.configuration_sha256_digest)
    }

    #[must_use]
    pub const fn retained_state_bytes_on_target(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    /// Analyze one exact acoustic-v2 frame with the configured sidecars.
    ///
    /// The caller must pair `frame_samples` with the [`AcousticFrameFeatures`]
    /// derived for that exact PCM frame. Cadence and feature domains are
    /// validated here, but content provenance cannot be inferred from two
    /// independent in-memory values.
    ///
    /// Cancellation is checked before validation and then by each enabled
    /// kernel. Incremental-family mutation remains staged until every enabled
    /// family for the frame completes successfully.
    pub fn observe_normalized_16khz_frame<C>(
        &mut self,
        frame_samples: &[f32; ACOUSTIC_FRAME_SAMPLES],
        frame: &AcousticFrameFeatures,
        is_cancelled: &mut C,
    ) -> FwResult<AcousticSidecarStudyObservation>
    where
        C: FnMut() -> bool,
    {
        if is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "acoustic sidecar study cancelled before frame {}",
                frame.frame_index
            )));
        }
        validate_acoustic_frame(frame)?;
        if !normalized_acoustic_pcm_is_valid(frame_samples) {
            return Err(FwError::InvalidRequest(
                "acoustic sidecar frame input must contain finite normalized PCM within [-1, 1]"
                    .to_owned(),
            ));
        }
        let wavelet = if let Some(basis) = self.config.mode.wavelet_basis() {
            Some(analyze_acoustic_wavelet(
                frame_samples,
                AcousticWaveletConfig {
                    basis,
                    levels: self.config.frame_wavelet_levels,
                },
                &mut *is_cancelled,
            )?)
        } else {
            None
        };
        let mut staged_modulation = self
            .config
            .mode
            .uses_modulation()
            .then(|| self.modulation.clone());
        let mut staged_trajectory = self
            .config
            .uses_trajectory_state()
            .then(|| self.trajectory.clone());
        let modulation = if let Some(staged) = staged_modulation.as_mut() {
            staged.push(frame, is_cancelled)?
        } else {
            None
        };
        let (trajectory_wavelet, scattering) = if let Some(staged) = staged_trajectory.as_mut() {
            staged.push(frame, self.config, is_cancelled)?
        } else {
            (None, None)
        };
        if let Some(staged) = staged_modulation {
            self.modulation = staged;
        }
        if let Some(staged) = staged_trajectory {
            self.trajectory = staged;
        }
        Ok(AcousticSidecarStudyObservation {
            schema_version: ACOUSTIC_SIDECAR_STUDY_SCHEMA_VERSION,
            configuration_sha256_digest: self.configuration_sha256_digest,
            config: self.config,
            frame_index: frame.frame_index,
            wavelet,
            modulation,
            trajectory_wavelet,
            scattering,
        })
    }
}

/// Identity of the evaluation-only boundary fusion seam.
///
/// This lane is never constructed by the production diarization entrypoints.
pub(crate) const ACOUSTIC_SIDECAR_FUSION_VERSION: &str =
    "acoustic-sidecar-boundary-fusion-v1";

/// Development-locked monotone calibration for one sidecar configuration.
///
/// Raw sidecar contrast is deliberately not called a probability.  Only this
/// explicit logistic calibration may turn it into evidence consumed by the
/// existing change selector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AcousticSidecarFusionCalibration {
    pub logit_intercept: f32,
    pub contrast_weight: f32,
    pub minimum_comparable_components: usize,
}

impl AcousticSidecarFusionCalibration {
    fn validate(self) -> FwResult<()> {
        if !self.logit_intercept.is_finite()
            || !self.contrast_weight.is_finite()
            || self.contrast_weight < 0.0
            || self.minimum_comparable_components == 0
        {
            return Err(FwError::InvalidRequest(
                "sidecar fusion calibration requires a finite intercept, a finite nonnegative contrast weight, and at least one comparable component"
                    .to_owned(),
            ));
        }
        Ok(())
    }
}

/// Complete opt-in request for one evaluation-only fused diarization pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AcousticSidecarEvaluationRequest {
    pub study_config: AcousticSidecarStudyConfig,
    pub calibration: AcousticSidecarFusionCalibration,
}

impl AcousticSidecarEvaluationRequest {
    fn validate(self) -> FwResult<()> {
        self.study_config.validate()?;
        self.calibration.validate()
    }
}

/// Missingness-aware contrast in canonical Voice, Channel, MixedAuxiliary order.
#[derive(Clone, Copy, Default, PartialEq)]
pub(crate) struct AcousticSidecarOwnerContrast {
    pub owner_contrast: [f32; 3],
    pub owner_available: [bool; 3],
    pub comparable_components: usize,
    pub component_comparisons: usize,
}

impl std::fmt::Debug for AcousticSidecarOwnerContrast {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcousticSidecarOwnerContrast")
            .field("owner_available", &self.owner_available)
            .field("comparable_components", &self.comparable_components)
            .field("component_comparisons", &self.component_comparisons)
            .finish_non_exhaustive()
    }
}

/// One signal-bearing, non-serializable evaluation observation.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct AcousticSidecarBoundarySignal {
    pub frame_index: usize,
    pub contrast: AcousticSidecarOwnerContrast,
    pub calibrated_probability: Option<f32>,
    pub observation: AcousticSidecarStudyObservation,
}

impl std::fmt::Debug for AcousticSidecarBoundarySignal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcousticSidecarBoundarySignal")
            .field("frame_index", &self.frame_index)
            .field("contrast", &self.contrast)
            .field(
                "calibrated_probability_available",
                &self.calibrated_probability.is_some(),
            )
            .finish_non_exhaustive()
    }
}

/// Aggregate-only evaluation diagnostics.
///
/// Per-frame signals remain confined to the segmenter's fixed 401-frame ring;
/// this evidence never retains or serializes duration-proportional sidecar
/// observations.
#[derive(Clone, Default, PartialEq)]
pub(crate) struct AcousticSidecarFusionEvaluationEvidence {
    pub fusion_requested: bool,
    pub fusion_executed: bool,
    pub fusion_configuration_sha256: Option<String>,
    pub sidecar_configuration_sha256: Option<String>,
    pub submitted_frame_count: usize,
    pub comparable_frame_count: usize,
    pub calibrated_signal_count: usize,
    pub consumed_probability_count: usize,
    pub changed_boundary_probability_count: usize,
    pub maximum_retained_signals: usize,
    pub frame_wavelet_filter_tap_terms: u64,
    pub trajectory_wavelet_filter_tap_terms: u64,
    pub trajectory_validity_sample_visits: u64,
    pub scattering_filter_sample_terms: u64,
    pub scattering_validity_sample_visits: u64,
    pub modulation_projection_sample_frequency_visits: u64,
    pub peak_scratch_buffer_payload_bytes: usize,
    pub peak_retained_state_bytes_on_target: usize,
    pub cached_twiddle_payload_bytes: usize,
}

impl std::fmt::Debug for AcousticSidecarFusionEvaluationEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcousticSidecarFusionEvaluationEvidence")
            .field("fusion_requested", &self.fusion_requested)
            .field("fusion_executed", &self.fusion_executed)
            .field(
                "fusion_configuration_sha256",
                &self.fusion_configuration_sha256,
            )
            .field(
                "sidecar_configuration_sha256",
                &self.sidecar_configuration_sha256,
            )
            .field("submitted_frame_count", &self.submitted_frame_count)
            .field("comparable_frame_count", &self.comparable_frame_count)
            .field("calibrated_signal_count", &self.calibrated_signal_count)
            .field(
                "consumed_probability_count",
                &self.consumed_probability_count,
            )
            .field(
                "changed_boundary_probability_count",
                &self.changed_boundary_probability_count,
            )
            .field("maximum_retained_signals", &self.maximum_retained_signals)
            .finish_non_exhaustive()
    }
}

fn acoustic_sidecar_owner_index(owner: AcousticSidecarFeatureOwner) -> usize {
    match owner {
        AcousticSidecarFeatureOwner::Voice => 0,
        AcousticSidecarFeatureOwner::Channel => 1,
        AcousticSidecarFeatureOwner::MixedAuxiliary => 2,
    }
}

fn bounded_nonnegative_sidecar_value(value: f32) -> FwResult<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(FwError::InvalidRequest(
            "sidecar fusion encountered a non-finite or negative summary component".to_owned(),
        ));
    }
    Ok(value / (1.0 + value))
}

fn bounded_unit_sidecar_value(value: f32) -> FwResult<f32> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(FwError::InvalidRequest(
            "sidecar fusion encountered a summary component outside [0, 1]".to_owned(),
        ));
    }
    Ok(value)
}

/// Compare two fixed-size sidecar observations without filling missing values
/// with zero or allowing channel-owned evidence to enter the Voice lane.
pub(crate) fn acoustic_sidecar_observation_owner_contrast(
    left: &AcousticSidecarStudyObservation,
    right: &AcousticSidecarStudyObservation,
) -> FwResult<AcousticSidecarOwnerContrast> {
    if left.schema_version != right.schema_version
        || left.configuration_sha256_digest != right.configuration_sha256_digest
        || left.config != right.config
    {
        return Err(FwError::InvalidRequest(
            "sidecar fusion observations must share one schema and configuration".to_owned(),
        ));
    }
    let mut output = AcousticSidecarOwnerContrast::default();
    let mut compare = |owner: AcousticSidecarFeatureOwner,
                       left: Option<f32>,
                       right: Option<f32>,
                       unit_interval: bool|
     -> FwResult<()> {
        output.component_comparisons = output.component_comparisons.saturating_add(1);
        let (Some(left), Some(right)) = (left, right) else {
            return Ok(());
        };
        let left = if unit_interval {
            bounded_unit_sidecar_value(left)?
        } else {
            bounded_nonnegative_sidecar_value(left)?
        };
        let right = if unit_interval {
            bounded_unit_sidecar_value(right)?
        } else {
            bounded_nonnegative_sidecar_value(right)?
        };
        let owner_index = acoustic_sidecar_owner_index(owner);
        output.owner_available[owner_index] = true;
        output.owner_contrast[owner_index] =
            output.owner_contrast[owner_index].max((left - right).abs());
        output.comparable_components = output.comparable_components.saturating_add(1);
        Ok(())
    };

    if let (Some(left_wavelet), Some(right_wavelet)) = (left.wavelet, right.wavelet) {
        let levels = left_wavelet
            .valid_level_count
            .min(right_wavelet.valid_level_count)
            .min(left.config.frame_wavelet_levels)
            .min(right.config.frame_wavelet_levels);
        for level_index in 0..levels {
            let left_level = left_wavelet.levels[level_index];
            let right_level = right_wavelet.levels[level_index];
            for (left_value, right_value, unit_interval) in [
                (
                    left_level.detail_energy_fraction,
                    right_level.detail_energy_fraction,
                    true,
                ),
                (
                    left_level.normalized_entropy,
                    right_level.normalized_entropy,
                    true,
                ),
                (
                    left_level.coefficient_flatness,
                    right_level.coefficient_flatness,
                    true,
                ),
                (left_level.crest_factor, right_level.crest_factor, false),
                (
                    left_level.normalized_detail_change,
                    right_level.normalized_detail_change,
                    false,
                ),
            ] {
                compare(
                    left_wavelet.owner,
                    Some(left_value),
                    Some(right_value),
                    unit_interval,
                )?;
            }
        }
        compare(
            left_wavelet.owner,
            (levels > 0).then_some(left_wavelet.final_approximation_energy_fraction),
            (levels > 0).then_some(right_wavelet.final_approximation_energy_fraction),
            true,
        )?;
    }

    if let (Some(left), Some(right)) = (left.modulation, right.modulation) {
        for frequency in 0..ACOUSTIC_MODULATION_FREQUENCY_HZ.len() {
            compare(
                left.voice_owner,
                left.voice_available
                    .then_some(left.voice_normalized_power[frequency]),
                right
                    .voice_available
                    .then_some(right.voice_normalized_power[frequency]),
                true,
            )?;
            compare(
                left.channel_level_owner,
                left.channel_level_available
                    .then_some(left.channel_level_normalized_power[frequency]),
                right
                    .channel_level_available
                    .then_some(right.channel_level_normalized_power[frequency]),
                true,
            )?;
            compare(
                left.channel_coloration_owner,
                left.channel_coloration_available
                    .then_some(left.channel_coloration_normalized_power[frequency]),
                right
                    .channel_coloration_available
                    .then_some(right.channel_coloration_normalized_power[frequency]),
                true,
            )?;
        }
    }

    if let (Some(left), Some(right)) = (left.trajectory_wavelet, right.trajectory_wavelet) {
        for (left_family, right_family) in left.families.iter().zip(right.families) {
            if left_family.family != right_family.family || left_family.owner != right_family.owner {
                return Err(FwError::InvalidRequest(
                    "sidecar trajectory observations use inconsistent family ordering".to_owned(),
                ));
            }
            let levels = left_family
                .valid_level_count
                .min(right_family.valid_level_count)
                .min(left.requested_levels)
                .min(right.requested_levels);
            for level_index in 0..levels {
                let left = left_family.levels[level_index];
                let right = right_family.levels[level_index];
                let available = left.available && right.available;
                compare(
                    left_family.owner,
                    available.then_some(left.mean_absolute_detail),
                    available.then_some(right.mean_absolute_detail),
                    false,
                )?;
                compare(
                    left_family.owner,
                    available.then_some(left.detail_rms),
                    available.then_some(right.detail_rms),
                    false,
                )?;
                compare(
                    left_family.owner,
                    (available && left.normalized_entropy_available)
                        .then_some(left.normalized_entropy),
                    (available && right.normalized_entropy_available)
                        .then_some(right.normalized_entropy),
                    true,
                )?;
                compare(
                    left_family.owner,
                    (available && left.normalized_detail_change_available)
                        .then_some(left.normalized_detail_change),
                    (available && right.normalized_detail_change_available)
                        .then_some(right.normalized_detail_change),
                    false,
                )?;
            }
        }
    }

    if let (Some(left), Some(right)) = (left.scattering, right.scattering) {
        for (left_family, right_family) in left.families.iter().zip(right.families) {
            if left_family.family != right_family.family || left_family.owner != right_family.owner {
                return Err(FwError::InvalidRequest(
                    "sidecar scattering observations use inconsistent family ordering".to_owned(),
                ));
            }
            for scale in 0..ACOUSTIC_SCATTERING_SCALE_SUPPORTS.len() {
                compare(
                    left_family.owner,
                    (left_family.first_order_available[scale]
                        && right_family.first_order_available[scale])
                        .then_some(left_family.first_order_mean_modulus[scale]),
                    (left_family.first_order_available[scale]
                        && right_family.first_order_available[scale])
                        .then_some(right_family.first_order_mean_modulus[scale]),
                    false,
                )?;
            }
            for pair in 0..ACOUSTIC_SCATTERING_SCALE_PAIRS.len() {
                compare(
                    left_family.owner,
                    (left_family.second_order_available[pair]
                        && right_family.second_order_available[pair])
                        .then_some(left_family.second_order_mean_modulus[pair]),
                    (left_family.second_order_available[pair]
                        && right_family.second_order_available[pair])
                        .then_some(right_family.second_order_mean_modulus[pair]),
                    false,
                )?;
            }
        }
    }
    Ok(output)
}

pub(crate) fn acoustic_sidecar_fusion_configuration_sha256(
    request: AcousticSidecarEvaluationRequest,
) -> FwResult<String> {
    request.validate()?;
    let mut hasher = Sha256::new();
    for value in [
        ACOUSTIC_SIDECAR_FUSION_VERSION,
        "contrast=maximum-matching-component-absolute-delta-per-owner",
        "owners=voice-channel-mixed-auxiliary",
        "missingness=both-available-no-zero-imputation",
        "probability=development-locked-monotone-logistic",
        "fusion=max-baseline-sidecar-unless-baseline-fallback",
    ] {
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    hasher.update(acoustic_sidecar_study_config_digest(request.study_config)?);
    hasher.update(request.calibration.logit_intercept.to_bits().to_le_bytes());
    hasher.update(request.calibration.contrast_weight.to_bits().to_le_bytes());
    hasher.update(
        (request.calibration.minimum_comparable_components as u64).to_le_bytes(),
    );
    let digest: [u8; 32] = hasher.finalize().into();
    Ok(sidecar_sha256_hex(&digest))
}

struct AcousticSidecarBoundaryAdapter {
    request: AcousticSidecarEvaluationRequest,
    study: AcousticSidecarStudy,
    previous: Option<AcousticSidecarStudyObservation>,
    evidence: AcousticSidecarFusionEvaluationEvidence,
}

impl AcousticSidecarBoundaryAdapter {
    fn new(request: AcousticSidecarEvaluationRequest) -> FwResult<Self> {
        request.validate()?;
        let study = AcousticSidecarStudy::new(request.study_config)?;
        let evidence = AcousticSidecarFusionEvaluationEvidence {
            fusion_requested: true,
            fusion_configuration_sha256: Some(acoustic_sidecar_fusion_configuration_sha256(
                request,
            )?),
            sidecar_configuration_sha256: Some(study.configuration_sha256_hex()),
            peak_retained_state_bytes_on_target: study.retained_state_bytes_on_target(),
            ..AcousticSidecarFusionEvaluationEvidence::default()
        };
        Ok(Self {
            request,
            study,
            previous: None,
            evidence,
        })
    }

    fn add_operation(total: &mut u64, value: usize, family: &str) -> FwResult<()> {
        let value = u64::try_from(value).map_err(|_| {
            FwError::InvalidRequest(format!(
                "sidecar {family} operation count exceeds u64"
            ))
        })?;
        *total = total.checked_add(value).ok_or_else(|| {
            FwError::InvalidRequest(format!("sidecar {family} operation count overflowed"))
        })?;
        Ok(())
    }

    fn observe<C>(
        &mut self,
        samples: &[f32; ACOUSTIC_FRAME_SAMPLES],
        frame: &AcousticFrameFeatures,
        is_cancelled: &mut C,
    ) -> FwResult<AcousticSidecarBoundarySignal>
    where
        C: FnMut() -> bool,
    {
        if is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "sidecar boundary fusion cancelled before frame {}",
                frame.frame_index
            )));
        }
        let observation =
            self.study
                .observe_normalized_16khz_frame(samples, frame, is_cancelled)?;
        let contrast = self.previous.as_ref().map_or_else(
            || Ok(AcousticSidecarOwnerContrast::default()),
            |previous| acoustic_sidecar_observation_owner_contrast(previous, &observation),
        )?;
        let calibrated_probability =
            (contrast.comparable_components
                >= self.request.calibration.minimum_comparable_components)
                .then(|| {
                    let maximum_contrast = contrast
                        .owner_contrast
                        .iter()
                        .zip(contrast.owner_available)
                        .filter_map(|(contrast, available)| available.then_some(*contrast))
                        .fold(0.0_f32, f32::max);
                    logistic_probability(
                        self.request.calibration.logit_intercept
                            + self.request.calibration.contrast_weight * maximum_contrast,
                    )
                    .clamp(f32::EPSILON, 1.0 - f32::EPSILON)
                });
        if calibrated_probability.is_some_and(|probability| {
            !probability.is_finite() || !(0.0..1.0).contains(&probability)
        }) {
            return Err(FwError::InvalidRequest(
                "sidecar fusion calibration produced an invalid probability".to_owned(),
            ));
        }
        let signal = AcousticSidecarBoundarySignal {
            frame_index: frame.frame_index,
            contrast,
            calibrated_probability,
            observation,
        };

        self.evidence.submitted_frame_count =
            self.evidence.submitted_frame_count.saturating_add(1);
        self.evidence.comparable_frame_count = self
            .evidence
            .comparable_frame_count
            .saturating_add(usize::from(contrast.comparable_components > 0));
        self.evidence.calibrated_signal_count = self
            .evidence
            .calibrated_signal_count
            .saturating_add(usize::from(calibrated_probability.is_some()));
        if let Some(summary) = observation.wavelet {
            Self::add_operation(
                &mut self.evidence.frame_wavelet_filter_tap_terms,
                summary.filter_tap_terms,
                "frame-wavelet filter",
            )?;
            self.evidence.peak_scratch_buffer_payload_bytes = self
                .evidence
                .peak_scratch_buffer_payload_bytes
                .max(summary.scratch_buffer_payload_bytes);
        }
        if let Some(summary) = observation.modulation {
            Self::add_operation(
                &mut self
                    .evidence
                    .modulation_projection_sample_frequency_visits,
                summary.projection_sample_frequency_visits,
                "modulation projection",
            )?;
            self.evidence.peak_retained_state_bytes_on_target = self
                .evidence
                .peak_retained_state_bytes_on_target
                .max(summary.retained_state_bytes_on_target);
            self.evidence.cached_twiddle_payload_bytes = self
                .evidence
                .cached_twiddle_payload_bytes
                .max(summary.cached_twiddle_payload_bytes);
        }
        if let Some(summary) = observation.trajectory_wavelet {
            Self::add_operation(
                &mut self.evidence.trajectory_wavelet_filter_tap_terms,
                summary.filter_tap_terms,
                "trajectory-wavelet filter",
            )?;
            Self::add_operation(
                &mut self.evidence.trajectory_validity_sample_visits,
                summary.validity_sample_visits,
                "trajectory-wavelet validity",
            )?;
            self.evidence.peak_scratch_buffer_payload_bytes = self
                .evidence
                .peak_scratch_buffer_payload_bytes
                .max(summary.scratch_buffer_payload_bytes);
        }
        if let Some(summary) = observation.scattering {
            Self::add_operation(
                &mut self.evidence.scattering_filter_sample_terms,
                summary.filter_sample_terms,
                "scattering filter",
            )?;
            Self::add_operation(
                &mut self.evidence.scattering_validity_sample_visits,
                summary.validity_sample_visits,
                "scattering validity",
            )?;
            self.evidence.peak_scratch_buffer_payload_bytes = self
                .evidence
                .peak_scratch_buffer_payload_bytes
                .max(summary.scratch_buffer_payload_bytes);
        }
        self.previous = Some(observation);
        Ok(signal)
    }

    fn finish(self) -> AcousticSidecarFusionEvaluationEvidence {
        self.evidence
    }
}

/// Stream acoustic-v2 features from normalized 16 kHz mono PCM.
///
/// Frames are emitted to `sink` immediately; this function never retains a
/// whole-call feature matrix. The only spectrum history is one 201-bin frame.
/// `is_cancelled` is sampled before frame zero and at most every 32 frames.
pub fn extract_acoustic_features<S, C>(
    samples: &[f32],
    mut is_cancelled: C,
    mut sink: S,
) -> FwResult<FeatureExtractionSummary>
where
    S: FnMut(AcousticFrameFeatures) -> FwResult<()>,
    C: FnMut() -> bool,
{
    extract_acoustic_features_with_frames(
        samples,
        &mut is_cancelled,
        |_, frame, _| sink(frame),
    )
}

fn extract_acoustic_features_with_frames<S, C>(
    samples: &[f32],
    is_cancelled: &mut C,
    mut sink: S,
) -> FwResult<FeatureExtractionSummary>
where
    S: FnMut(
        &[f32; ACOUSTIC_FRAME_SAMPLES],
        AcousticFrameFeatures,
        &mut C,
    ) -> FwResult<()>,
    C: FnMut() -> bool,
{
    if is_cancelled() {
        return Err(FwError::Cancelled(
            "acoustic feature extraction cancelled before frame zero".to_owned(),
        ));
    }
    for sample in samples {
        if !sample.is_finite() {
            return Err(FwError::InvalidRequest(
                "acoustic feature input contains a non-finite PCM sample".to_owned(),
            ));
        }
        if !(-1.0..=1.0).contains(sample) {
            return Err(FwError::InvalidRequest(
                "acoustic feature input contains a PCM sample outside the normalized [-1, 1] range"
                    .to_owned(),
            ));
        }
    }

    let frame_count = if samples.len() < ACOUSTIC_FRAME_SAMPLES {
        0
    } else {
        1 + (samples.len() - ACOUSTIC_FRAME_SAMPLES) / ACOUSTIC_HOP_SAMPLES
    };
    let mut previous_cepstrum = [0.0_f32; CEPSTRAL_COEFFICIENTS];
    let mut previous_delta = [0.0_f32; CEPSTRAL_COEFFICIENTS];
    let mut previous_normalized_power = [0.0_f32; crate::native_engine::mel::N_FREQ_BINS];
    let mut has_previous = false;
    let mut noise_floor_dbfs = -90.0_f32;
    let mut voiced_fraction = 0.0_f32;
    let mut voiced_frame_count = 0usize;
    let mut reliable_pitch_frame_count = 0usize;
    let mut high_information_frame_count = 0usize;
    let mut low_energy_frame_count = 0usize;

    for frame_index in 0..frame_count {
        if frame_index > 0
            && frame_index % ACOUSTIC_CANCELLATION_INTERVAL_FRAMES == 0
            && is_cancelled()
        {
            return Err(FwError::Cancelled(format!(
                "acoustic feature extraction cancelled at frame {frame_index}"
            )));
        }
        let start = frame_index * ACOUSTIC_HOP_SAMPLES;
        let frame: &[f32; ACOUSTIC_FRAME_SAMPLES] = samples
            [start..start + ACOUSTIC_FRAME_SAMPLES]
            .try_into()
            .map_err(|_| {
                FwError::InvalidRequest(
                    "acoustic feature frame did not match the fixed analysis window".to_owned(),
                )
            })?;
        let mut power = [0.0_f32; crate::native_engine::mel::N_FREQ_BINS];
        crate::native_engine::mel::fixed_frame_power_spectrum(frame, &mut power)?;

        let (rms_dbfs, crest_factor, clipping_fraction) = waveform_descriptors(frame);
        if rms_dbfs < -55.0 {
            low_energy_frame_count += 1;
        }
        noise_floor_dbfs = update_noise_floor(noise_floor_dbfs, rms_dbfs);

        let (f0_hz, voicing_confidence, overlap_probability) = estimate_f0(frame, rms_dbfs);
        let voiced = f0_hz.is_some();
        if voiced {
            voiced_frame_count += 1;
        }
        voiced_fraction = 0.95 * voiced_fraction + 0.05 * if voiced { 1.0 } else { 0.0 };

        let cepstral_envelope = cepstral_envelope(&power);
        let mut cepstral_delta = [0.0_f32; CEPSTRAL_COEFFICIENTS];
        let mut cepstral_delta_delta = [0.0_f32; CEPSTRAL_COEFFICIENTS];
        if has_previous {
            for coefficient in 0..CEPSTRAL_COEFFICIENTS {
                cepstral_delta[coefficient] =
                    cepstral_envelope[coefficient] - previous_cepstrum[coefficient];
                cepstral_delta_delta[coefficient] =
                    cepstral_delta[coefficient] - previous_delta[coefficient];
            }
        }
        previous_cepstrum = cepstral_envelope;
        previous_delta = cepstral_delta;

        let spectral = spectral_descriptors(
            &power,
            &previous_normalized_power,
            has_previous,
            clipping_fraction,
        );
        normalize_power(&power, &mut previous_normalized_power);
        has_previous = true;
        let transient = spectral.flux > 0.35;
        let reliable_pitch = voicing_confidence >= 0.55 && rms_dbfs >= -50.0;
        reliable_pitch_frame_count += usize::from(reliable_pitch);
        let clipped = clipping_fraction > 0.005;
        let high_information =
            voiced && reliable_pitch && rms_dbfs >= -50.0 && !clipped && !transient;
        high_information_frame_count += usize::from(high_information);
        let harmonic_to_noise_db = harmonic_to_noise_db(voicing_confidence);
        let formant_proxies_hz = formant_proxies(&power);
        let temporal_modulation = cepstral_delta.iter().map(|value| value.abs()).sum::<f32>()
            / CEPSTRAL_COEFFICIENTS as f32;

        let features = AcousticFrameFeatures {
            frame_index,
            start_ms: samples_to_ms(start),
            end_ms: samples_to_ms(start + ACOUSTIC_FRAME_SAMPLES),
            voice: VoiceFeatureView {
                cepstral_envelope,
                cepstral_delta,
                cepstral_delta_delta,
                f0_hz,
                pitch_uncertainty_octaves: f0_hz
                    .map(|_| (1.0 - voicing_confidence).clamp(0.0, 1.0) * 2.0),
                voicing_confidence,
                harmonicity: voicing_confidence,
                harmonic_to_noise_db,
                formant_proxies_hz,
                temporal_modulation,
                voiced_fraction,
            },
            channel: ChannelFeatureView {
                rms_dbfs,
                dynamics_above_noise_db: (rms_dbfs - noise_floor_dbfs).max(0.0),
                spectral_centroid_hz: spectral.centroid_hz,
                spectral_bandwidth_hz: spectral.bandwidth_hz,
                spectral_rolloff_hz: spectral.rolloff_hz,
                spectral_flatness: spectral.flatness,
                spectral_tilt: spectral.tilt,
                low_band_fraction: spectral.band_fractions[0],
                mid_band_fraction: spectral.band_fractions[1],
                high_band_fraction: spectral.band_fractions[2],
                crest_factor,
                clipping_fraction,
                noise_floor_dbfs,
                spectral_flux: spectral.flux,
                distortion_proxy: spectral.distortion_proxy,
                effective_band_limit_hz: effective_band_limit_hz(&power),
                high_frequency_attenuation: high_frequency_attenuation(spectral.band_fractions),
                reverberation_proxy: reverberation_proxy(frame),
                muffling_proxy: muffling_proxy(spectral.band_fractions),
                stationary_coloration: (1.0 - spectral.flux).clamp(0.0, 1.0),
            },
            overlap_probability,
            quality: AcousticQualityMask {
                voiced,
                reliable_pitch,
                low_energy: rms_dbfs < -55.0,
                clipped,
                transient,
            },
        };
        sink(frame, features, is_cancelled)?;
    }

    Ok(FeatureExtractionSummary {
        feature_schema: ACOUSTIC_FEATURE_SCHEMA_VERSION,
        frame_count,
        voiced_frame_count,
        reliable_pitch_frame_count,
        high_information_frame_count,
        missing_pitch_frame_count: frame_count.saturating_sub(reliable_pitch_frame_count),
        low_energy_frame_count,
        retained_state_bytes_upper_bound: std::mem::size_of::<
            [f32; crate::native_engine::mel::N_FREQ_BINS],
        >() + 2
            * std::mem::size_of::<[f32; CEPSTRAL_COEFFICIENTS]>()
            + 2 * std::mem::size_of::<[f32; ACOUSTIC_FRAME_SAMPLES]>(),
    })
}

#[derive(Debug, Clone)]
struct AcousticFeatureStreamState {
    previous_cepstrum: [f32; CEPSTRAL_COEFFICIENTS],
    previous_delta: [f32; CEPSTRAL_COEFFICIENTS],
    previous_normalized_power: [f32; crate::native_engine::mel::N_FREQ_BINS],
    has_previous: bool,
    noise_floor_dbfs: f32,
    voiced_fraction: f32,
    voiced_frame_count: usize,
    reliable_pitch_frame_count: usize,
    high_information_frame_count: usize,
    low_energy_frame_count: usize,
}

impl AcousticFeatureStreamState {
    fn new() -> Self {
        Self {
            previous_cepstrum: [0.0; CEPSTRAL_COEFFICIENTS],
            previous_delta: [0.0; CEPSTRAL_COEFFICIENTS],
            previous_normalized_power: [0.0; crate::native_engine::mel::N_FREQ_BINS],
            has_previous: false,
            noise_floor_dbfs: -90.0,
            voiced_fraction: 0.0,
            voiced_frame_count: 0,
            reliable_pitch_frame_count: 0,
            high_information_frame_count: 0,
            low_energy_frame_count: 0,
        }
    }

    fn process_frame(
        &mut self,
        frame_index: usize,
        frame: &[f32; ACOUSTIC_FRAME_SAMPLES],
    ) -> FwResult<AcousticFrameFeatures> {
        let start = frame_index
            .checked_mul(ACOUSTIC_HOP_SAMPLES)
            .ok_or_else(|| {
                FwError::InvalidRequest(
                    "acoustic feature timestamp exceeds the supported range".to_owned(),
                )
            })?;
        let mut power = [0.0_f32; crate::native_engine::mel::N_FREQ_BINS];
        crate::native_engine::mel::fixed_frame_power_spectrum(frame, &mut power)?;
        let (rms_dbfs, crest_factor, clipping_fraction) = waveform_descriptors(frame);
        if rms_dbfs < -55.0 {
            self.low_energy_frame_count += 1;
        }
        self.noise_floor_dbfs = update_noise_floor(self.noise_floor_dbfs, rms_dbfs);
        let (f0_hz, voicing_confidence, overlap_probability) = estimate_f0(frame, rms_dbfs);
        let voiced = f0_hz.is_some();
        self.voiced_frame_count += usize::from(voiced);
        self.voiced_fraction = 0.95 * self.voiced_fraction + 0.05 * if voiced { 1.0 } else { 0.0 };

        let cepstral_envelope = cepstral_envelope(&power);
        let mut cepstral_delta = [0.0_f32; CEPSTRAL_COEFFICIENTS];
        let mut cepstral_delta_delta = [0.0_f32; CEPSTRAL_COEFFICIENTS];
        if self.has_previous {
            for coefficient in 0..CEPSTRAL_COEFFICIENTS {
                cepstral_delta[coefficient] =
                    cepstral_envelope[coefficient] - self.previous_cepstrum[coefficient];
                cepstral_delta_delta[coefficient] =
                    cepstral_delta[coefficient] - self.previous_delta[coefficient];
            }
        }
        self.previous_cepstrum = cepstral_envelope;
        self.previous_delta = cepstral_delta;
        let spectral = spectral_descriptors(
            &power,
            &self.previous_normalized_power,
            self.has_previous,
            clipping_fraction,
        );
        normalize_power(&power, &mut self.previous_normalized_power);
        self.has_previous = true;
        let transient = spectral.flux > 0.35;
        let reliable_pitch = voicing_confidence >= 0.55 && rms_dbfs >= -50.0;
        self.reliable_pitch_frame_count += usize::from(reliable_pitch);
        let clipped = clipping_fraction > 0.005;
        let high_information =
            voiced && reliable_pitch && rms_dbfs >= -50.0 && !clipped && !transient;
        self.high_information_frame_count += usize::from(high_information);
        let harmonic_to_noise_db = harmonic_to_noise_db(voicing_confidence);
        let formant_proxies_hz = formant_proxies(&power);
        let temporal_modulation = cepstral_delta.iter().map(|value| value.abs()).sum::<f32>()
            / CEPSTRAL_COEFFICIENTS as f32;

        Ok(AcousticFrameFeatures {
            frame_index,
            start_ms: samples_to_ms(start),
            end_ms: samples_to_ms(start + ACOUSTIC_FRAME_SAMPLES),
            voice: VoiceFeatureView {
                cepstral_envelope,
                cepstral_delta,
                cepstral_delta_delta,
                f0_hz,
                pitch_uncertainty_octaves: f0_hz
                    .map(|_| (1.0 - voicing_confidence).clamp(0.0, 1.0) * 2.0),
                voicing_confidence,
                harmonicity: voicing_confidence,
                harmonic_to_noise_db,
                formant_proxies_hz,
                temporal_modulation,
                voiced_fraction: self.voiced_fraction,
            },
            channel: ChannelFeatureView {
                rms_dbfs,
                dynamics_above_noise_db: (rms_dbfs - self.noise_floor_dbfs).max(0.0),
                spectral_centroid_hz: spectral.centroid_hz,
                spectral_bandwidth_hz: spectral.bandwidth_hz,
                spectral_rolloff_hz: spectral.rolloff_hz,
                spectral_flatness: spectral.flatness,
                spectral_tilt: spectral.tilt,
                low_band_fraction: spectral.band_fractions[0],
                mid_band_fraction: spectral.band_fractions[1],
                high_band_fraction: spectral.band_fractions[2],
                crest_factor,
                clipping_fraction,
                noise_floor_dbfs: self.noise_floor_dbfs,
                spectral_flux: spectral.flux,
                distortion_proxy: spectral.distortion_proxy,
                effective_band_limit_hz: effective_band_limit_hz(&power),
                high_frequency_attenuation: high_frequency_attenuation(spectral.band_fractions),
                reverberation_proxy: reverberation_proxy(frame),
                muffling_proxy: muffling_proxy(spectral.band_fractions),
                stationary_coloration: (1.0 - spectral.flux).clamp(0.0, 1.0),
            },
            overlap_probability,
            quality: AcousticQualityMask {
                voiced,
                reliable_pitch,
                low_energy: rms_dbfs < -55.0,
                clipped,
                transient,
            },
        })
    }

    fn summary(&self, frame_count: usize) -> FeatureExtractionSummary {
        FeatureExtractionSummary {
            feature_schema: ACOUSTIC_FEATURE_SCHEMA_VERSION,
            frame_count,
            voiced_frame_count: self.voiced_frame_count,
            reliable_pitch_frame_count: self.reliable_pitch_frame_count,
            high_information_frame_count: self.high_information_frame_count,
            missing_pitch_frame_count: frame_count.saturating_sub(self.reliable_pitch_frame_count),
            low_energy_frame_count: self.low_energy_frame_count,
            retained_state_bytes_upper_bound: std::mem::size_of::<Self>()
                + std::mem::size_of::<[f32; ACOUSTIC_FRAME_SAMPLES]>(),
        }
    }
}

/// Incremental feature extractor with fixed retained memory and exact cadence.
///
/// Arbitrary chunk boundaries produce exactly the same observations as
/// [`extract_acoustic_features`]. The caller owns each chunk; this type retains
/// at most one analysis frame plus fixed DSP history.
#[derive(Debug, Clone)]
pub struct AcousticFeatureStream {
    frame_buffer: [f32; ACOUSTIC_FRAME_SAMPLES],
    buffered_samples: usize,
    frame_count: usize,
    cancellation_checked: bool,
    state: AcousticFeatureStreamState,
}

impl Default for AcousticFeatureStream {
    fn default() -> Self {
        Self::new()
    }
}

impl AcousticFeatureStream {
    #[must_use]
    pub fn new() -> Self {
        Self {
            frame_buffer: [0.0; ACOUSTIC_FRAME_SAMPLES],
            buffered_samples: 0,
            frame_count: 0,
            cancellation_checked: false,
            state: AcousticFeatureStreamState::new(),
        }
    }

    pub fn push_chunk<S, C>(
        &mut self,
        samples: &[f32],
        is_cancelled: &mut C,
        sink: &mut S,
    ) -> FwResult<()>
    where
        S: FnMut(AcousticFrameFeatures) -> FwResult<()>,
        C: FnMut() -> bool,
    {
        if !self.cancellation_checked {
            self.cancellation_checked = true;
            if is_cancelled() {
                return Err(FwError::Cancelled(
                    "acoustic feature extraction cancelled before frame zero".to_owned(),
                ));
            }
        }
        for &sample in samples {
            if !sample.is_finite() {
                return Err(FwError::InvalidRequest(
                    "acoustic feature input contains a non-finite PCM sample".to_owned(),
                ));
            }
            if !(-1.0..=1.0).contains(&sample) {
                return Err(FwError::InvalidRequest(
                    "acoustic feature input contains a PCM sample outside the normalized [-1, 1] range"
                        .to_owned(),
                ));
            }
            self.frame_buffer[self.buffered_samples] = sample;
            self.buffered_samples += 1;
            if self.buffered_samples != ACOUSTIC_FRAME_SAMPLES {
                continue;
            }
            if self.frame_count > 0
                && self
                    .frame_count
                    .is_multiple_of(ACOUSTIC_CANCELLATION_INTERVAL_FRAMES)
                && is_cancelled()
            {
                return Err(FwError::Cancelled(format!(
                    "acoustic feature extraction cancelled at frame {}",
                    self.frame_count
                )));
            }
            let features = self
                .state
                .process_frame(self.frame_count, &self.frame_buffer)?;
            sink(features)?;
            self.frame_count += 1;
            self.frame_buffer
                .copy_within(ACOUSTIC_HOP_SAMPLES..ACOUSTIC_FRAME_SAMPLES, 0);
            self.buffered_samples = ACOUSTIC_FRAME_SAMPLES - ACOUSTIC_HOP_SAMPLES;
        }
        Ok(())
    }

    #[must_use]
    pub fn finish(self) -> FeatureExtractionSummary {
        self.state.summary(self.frame_count)
    }
}

/// Chunk-safe feature extraction plus rolling regime segmentation.
///
/// DSP and change-detection state are fixed-size. Returned tracklets are the
/// duration-proportional output needed by the global clustering stage; callers
/// may persist or consume that output after `finish` without retaining PCM.
pub struct AcousticSegmentationStream<'a> {
    feature_stream: AcousticFeatureStream,
    segmenter: AcousticSegmenter<'a>,
}

impl<'a> AcousticSegmentationStream<'a> {
    pub fn new(
        boundary_hints: &'a AcousticBoundaryHints,
        supervised_boundaries_ms: &[u64],
        feature_ablation: AcousticFeatureAblation,
        detector_mode: AcousticChangeDetectorMode,
    ) -> FwResult<Self> {
        Ok(Self {
            feature_stream: AcousticFeatureStream::new(),
            segmenter: AcousticSegmenter::new_with_supervised_boundaries(
                boundary_hints,
                supervised_boundaries_ms,
                feature_ablation,
                detector_mode,
            )?,
        })
    }

    pub fn push_chunk<C>(&mut self, samples: &[f32], is_cancelled: &mut C) -> FwResult<()>
    where
        C: FnMut() -> bool,
    {
        let feature_stream = &mut self.feature_stream;
        let segmenter = &mut self.segmenter;
        feature_stream.push_chunk(samples, is_cancelled, &mut |frame| segmenter.push(frame))
    }

    pub fn finish(
        self,
    ) -> FwResult<(
        FeatureExtractionSummary,
        Vec<AcousticTracklet>,
        AcousticSegmentationSummary,
    )> {
        let feature_summary = self.feature_stream.finish();
        let (tracklets, segmentation_summary, _) = self.segmenter.finish()?;
        Ok((feature_summary, tracklets, segmentation_summary))
    }
}

#[derive(Debug, Clone, Copy)]
struct SpectralDescriptors {
    centroid_hz: f32,
    bandwidth_hz: f32,
    rolloff_hz: f32,
    flatness: f32,
    tilt: f32,
    band_fractions: [f32; 3],
    flux: f32,
    distortion_proxy: f32,
}

fn samples_to_ms(samples: usize) -> u64 {
    (samples as u64 * 1_000) / crate::native_engine::mel::SAMPLE_RATE as u64
}

fn waveform_descriptors(frame: &[f32]) -> (f32, f32, f32) {
    let mut squared_sum = 0.0_f64;
    let mut peak = 0.0_f32;
    let mut clipped = 0usize;
    for &sample in frame {
        squared_sum += f64::from(sample) * f64::from(sample);
        peak = peak.max(sample.abs());
        clipped += usize::from(sample.abs() >= 0.999);
    }
    let rms = (squared_sum / frame.len() as f64).sqrt() as f32;
    let rms_dbfs = 20.0 * rms.max(PCM_EPSILON).log10();
    (
        rms_dbfs,
        peak / rms.max(PCM_EPSILON),
        clipped as f32 / frame.len() as f32,
    )
}

fn update_noise_floor(previous: f32, current: f32) -> f32 {
    if current < previous {
        0.8 * previous + 0.2 * current
    } else {
        0.995 * previous + 0.005 * current
    }
}

fn estimate_f0(frame: &[f32], rms_dbfs: f32) -> (Option<f32>, f32, f32) {
    if rms_dbfs < -60.0 {
        return (None, 0.0, 0.0);
    }
    let mean = frame.iter().copied().sum::<f32>() / frame.len() as f32;
    let min_lag = crate::native_engine::mel::SAMPLE_RATE / 400;
    let max_lag = (crate::native_engine::mel::SAMPLE_RATE / 55).min(frame.len() - 2);
    let mut best_lag = 0usize;
    let mut best_correlation = 0.0_f32;
    let mut correlations = [0.0_f32; ACOUSTIC_FRAME_SAMPLES];
    for lag in min_lag..=max_lag {
        let mut cross = 0.0_f64;
        let mut left_energy = 0.0_f64;
        let mut right_energy = 0.0_f64;
        for index in 0..frame.len() - lag {
            let left = f64::from(frame[index] - mean);
            let right = f64::from(frame[index + lag] - mean);
            cross += left * right;
            left_energy += left * left;
            right_energy += right * right;
        }
        let denominator = (left_energy * right_energy).sqrt();
        let correlation = if denominator > f64::from(PCM_EPSILON) {
            (cross / denominator) as f32
        } else {
            0.0
        };
        correlations[lag] = correlation;
        if correlation > best_correlation {
            best_correlation = correlation;
            best_lag = lag;
        }
    }
    // A periodic waveform also correlates at integer multiples of its true
    // period. Choosing the global maximum alone therefore creates octave and
    // subharmonic errors. Prefer the earliest strong local maximum, which is
    // the first full period for a clean harmonic source.
    let strong_peak = (0.90 * best_correlation).max(0.55);
    if let Some(first_peak) = (min_lag + 1..max_lag).find(|&lag| {
        correlations[lag] >= strong_peak
            && correlations[lag] >= correlations[lag - 1]
            && correlations[lag] > correlations[lag + 1]
    }) {
        best_lag = first_peak;
        best_correlation = correlations[first_peak];
    }
    if best_lag == 0 || best_correlation < 0.30 {
        (None, best_correlation.clamp(0.0, 1.0), 0.0)
    } else {
        let secondary_correlation = (min_lag + 1..max_lag)
            .filter(|&lag| {
                correlations[lag] >= 0.30
                    && correlations[lag] >= correlations[lag - 1]
                    && correlations[lag] > correlations[lag + 1]
                    && !periods_are_harmonically_related(best_lag, lag)
            })
            .map(|lag| correlations[lag])
            .max_by(f32::total_cmp)
            .unwrap_or(0.0);
        let overlap_probability = (best_correlation.clamp(0.0, 1.0)
            * ((secondary_correlation - 0.30) / 0.40).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        (
            Some(crate::native_engine::mel::SAMPLE_RATE as f32 / best_lag as f32),
            best_correlation.clamp(0.0, 1.0),
            overlap_probability,
        )
    }
}

fn periods_are_harmonically_related(left_lag: usize, right_lag: usize) -> bool {
    let minimum = left_lag.min(right_lag).max(1) as f32;
    let maximum = left_lag.max(right_lag) as f32;
    let ratio = maximum / minimum;
    let nearest_integer = ratio.round().max(1.0);
    (ratio - nearest_integer).abs() <= 0.08 * nearest_integer
}

fn cepstral_envelope(
    power: &[f32; crate::native_engine::mel::N_FREQ_BINS],
) -> [f32; CEPSTRAL_COEFFICIENTS] {
    let mut log_bands = [0.0_f32; ENVELOPE_BANDS];
    let usable_bins = crate::native_engine::mel::N_FREQ_BINS / 2;
    for (band, log_energy) in log_bands.iter_mut().enumerate() {
        let start = band * usable_bins / ENVELOPE_BANDS;
        let end = ((band + 1) * usable_bins / ENVELOPE_BANDS).max(start + 1);
        let energy = power[start..end].iter().copied().sum::<f32>();
        *log_energy = (energy + POWER_EPSILON).ln();
    }
    let mean = log_bands.iter().copied().sum::<f32>() / ENVELOPE_BANDS as f32;
    for energy in &mut log_bands {
        *energy -= mean;
    }
    let mut cepstrum = [0.0_f32; CEPSTRAL_COEFFICIENTS];
    for (coefficient, output) in cepstrum.iter_mut().enumerate() {
        let order = coefficient + 1;
        for (band, &energy) in log_bands.iter().enumerate() {
            let angle =
                std::f32::consts::PI * order as f32 * (band as f32 + 0.5) / ENVELOPE_BANDS as f32;
            *output += energy * angle.cos();
        }
        *output /= ENVELOPE_BANDS as f32;
    }
    cepstrum
}

fn harmonic_to_noise_db(periodicity: f32) -> f32 {
    let periodic = periodicity.clamp(0.001, 0.999);
    (10.0 * (periodic / (1.0 - periodic)).log10()).clamp(-20.0, 40.0)
}

fn formant_proxies(power: &[f32; crate::native_engine::mel::N_FREQ_BINS]) -> [f32; 3] {
    let bin_hz =
        crate::native_engine::mel::SAMPLE_RATE as f32 / crate::native_engine::mel::N_FFT as f32;
    [(200.0, 1_000.0), (700.0, 3_000.0), (1_800.0, 4_000.0)].map(|(minimum_hz, maximum_hz)| {
        let start = (minimum_hz / bin_hz) as usize;
        let end = ((maximum_hz / bin_hz) as usize + 1).min(power.len());
        power[start.min(end)..end]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map_or(minimum_hz, |(offset, _)| (start + offset) as f32 * bin_hz)
    })
}

fn effective_band_limit_hz(power: &[f32; crate::native_engine::mel::N_FREQ_BINS]) -> f32 {
    let total = power.iter().copied().sum::<f32>().max(POWER_EPSILON);
    let target = 0.99 * total;
    let bin_hz =
        crate::native_engine::mel::SAMPLE_RATE as f32 / crate::native_engine::mel::N_FFT as f32;
    let mut cumulative = 0.0_f32;
    for (bin, value) in power.iter().copied().enumerate() {
        cumulative += value;
        if cumulative >= target {
            return bin as f32 * bin_hz;
        }
    }
    8_000.0
}

fn high_frequency_attenuation(bands: [f32; 3]) -> f32 {
    ((bands[0] + bands[1]) / (bands[2] + 0.01)).ln_1p() / 5.0
}

fn muffling_proxy(bands: [f32; 3]) -> f32 {
    (bands[0] / (bands[0] + bands[2] + POWER_EPSILON)).clamp(0.0, 1.0)
}

fn reverberation_proxy(frame: &[f32]) -> f32 {
    const BLOCKS: usize = 5;
    let block_len = frame.len() / BLOCKS;
    let mut energy = [0.0_f32; BLOCKS];
    for (block, output) in energy.iter_mut().enumerate() {
        let start = block * block_len;
        let end = if block + 1 == BLOCKS {
            frame.len()
        } else {
            start + block_len
        };
        *output = frame[start..end]
            .iter()
            .map(|sample| sample * sample)
            .sum::<f32>();
    }
    let peak = energy
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(index, _)| index);
    let total = energy.iter().copied().sum::<f32>().max(POWER_EPSILON);
    energy.iter().skip(peak + 1).copied().sum::<f32>() / total
}

fn normalize_power(
    power: &[f32; crate::native_engine::mel::N_FREQ_BINS],
    normalized: &mut [f32; crate::native_engine::mel::N_FREQ_BINS],
) {
    let total = power.iter().copied().sum::<f32>().max(POWER_EPSILON);
    for (output, &input) in normalized.iter_mut().zip(power) {
        *output = input / total;
    }
}

fn spectral_descriptors(
    power: &[f32; crate::native_engine::mel::N_FREQ_BINS],
    previous_normalized: &[f32; crate::native_engine::mel::N_FREQ_BINS],
    has_previous: bool,
    clipping_fraction: f32,
) -> SpectralDescriptors {
    let total = power.iter().copied().sum::<f32>().max(POWER_EPSILON);
    let bin_hz =
        crate::native_engine::mel::SAMPLE_RATE as f32 / crate::native_engine::mel::N_FFT as f32;
    let centroid_hz = power
        .iter()
        .enumerate()
        .map(|(bin, &value)| bin as f32 * bin_hz * value)
        .sum::<f32>()
        / total;
    let bandwidth_hz = (power
        .iter()
        .enumerate()
        .map(|(bin, &value)| {
            let offset = bin as f32 * bin_hz - centroid_hz;
            offset * offset * value
        })
        .sum::<f32>()
        / total)
        .sqrt();

    let mut cumulative = 0.0_f32;
    let rolloff_target = 0.85 * total;
    let mut rolloff_hz = 0.0_f32;
    for (bin, &value) in power.iter().enumerate() {
        cumulative += value;
        if cumulative >= rolloff_target {
            rolloff_hz = bin as f32 * bin_hz;
            break;
        }
    }

    let arithmetic_mean = total / power.len() as f32;
    let log_mean = power
        .iter()
        .map(|value| (value + POWER_EPSILON).ln())
        .sum::<f32>()
        / power.len() as f32;
    let flatness = log_mean.exp() / arithmetic_mean.max(POWER_EPSILON);
    let tilt = spectral_tilt(power, bin_hz);

    let low_end = (500.0 / bin_hz) as usize;
    let mid_end = (2_000.0 / bin_hz) as usize;
    let low = power[..low_end.min(power.len())]
        .iter()
        .copied()
        .sum::<f32>();
    let mid = power[low_end.min(power.len())..mid_end.min(power.len())]
        .iter()
        .copied()
        .sum::<f32>();
    let high = power[mid_end.min(power.len())..]
        .iter()
        .copied()
        .sum::<f32>();

    let flux = if has_previous {
        power
            .iter()
            .zip(previous_normalized)
            .map(|(&current, &previous)| (current / total - previous).max(0.0))
            .sum::<f32>()
    } else {
        0.0
    };
    let distortion_proxy =
        (0.55 * flatness + 0.30 * high / total + 0.15 * clipping_fraction).clamp(0.0, 1.0);

    SpectralDescriptors {
        centroid_hz,
        bandwidth_hz,
        rolloff_hz,
        flatness,
        tilt,
        band_fractions: [low / total, mid / total, high / total],
        flux,
        distortion_proxy,
    }
}

fn spectral_tilt(power: &[f32; crate::native_engine::mel::N_FREQ_BINS], bin_hz: f32) -> f32 {
    let mut count = 0.0_f32;
    let mut sum_x = 0.0_f32;
    let mut sum_y = 0.0_f32;
    let mut sum_xx = 0.0_f32;
    let mut sum_xy = 0.0_f32;
    for (bin, &value) in power.iter().enumerate().skip(1) {
        let x = (bin as f32 * bin_hz).ln();
        let y = (value + POWER_EPSILON).ln();
        count += 1.0;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    let denominator = count * sum_xx - sum_x * sum_x;
    if denominator.abs() <= POWER_EPSILON {
        0.0
    } else {
        (count * sum_xy - sum_x * sum_y) / denominator
    }
}

const CHANGE_SCALES_FRAMES: [usize; 5] = [10, 25, 50, 100, 200];
const CHANGE_RING_FRAMES: usize = 2 * CHANGE_SCALES_FRAMES[4] + 1;
const CHANGE_FALLBACK_DISTANCE_THRESHOLD: f32 = 0.34;
const CHANGE_FIXED_SAFE_SUPPRESSION_FRAMES: usize = 20;
const CHANGE_REFINEMENT_RADIUS_FRAMES: usize = 30;
const CHANGE_HYSTERESIS_MAX_FRAMES: usize = 100;
const CHANGE_HYSTERESIS_REARM_FRAMES: usize = 20;
const CHANGE_HYSTERESIS_RESET_RATIO: f32 = 0.50;
const CHANGE_STRONG_PEAK_PROBABILITY: f32 = 0.50;
const CHANGE_STRONG_PEAK_SUPPRESSION_FRAMES: usize = 20;
const MIN_TRACKLET_FRAMES: usize = 20;
// A speaker candidate is not an output speaker merely because agglomeration
// stopped with a cluster bearing its label. Non-hard candidates need recurring
// attributable speech outside their own enrollment observations. A single
// aggregate interval cannot validate itself against a centroid that contains
// that same observation.
const MIN_SPEAKER_EVIDENCE_VOICED_FRAMES: usize = MIN_TRACKLET_FRAMES * 2;
const MIN_SPEAKER_EVIDENCE_RECURRENCE_EPISODES: usize = 2;
const MIN_SPEAKER_EVIDENCE_CONFIDENCE: f32 = 0.30;
const MIN_SPEAKER_EVIDENCE_RELIABILITY: f32 = 0.30;
const MIN_SPEAKER_SEPARATION_SUPPORT: f32 = 0.40;
const MAX_SAME_SPEAKER_PROBABILITY_FOR_SEPARATION: f32 = 0.45;
const MIN_SPEAKER_SEPARATION_LANES: usize = 3;
const MAX_MULTI_SPEAKER_DOMINANT_SHARE: f32 = 0.98;
// Channel conditions are useful within a recording, but must remain secondary:
// the same vocal source may legitimately appear through both a nearby
// microphone and a distant loudspeaker.
const CHANNEL_DISTANCE_WEIGHT: f32 = 0.08;
const SPEAKER_PAIR_MINIMUM_ACTIVE_DIMENSIONS: usize = 4;
const SPEAKER_COUNT_PERTURBATION_LANES: usize = 5;
const SPEAKER_COUNT_SPARSE_NEIGHBOR_DEGREE: usize = 8;
const SPEAKER_COUNT_EIGENSOLVER_MAX_ITERATIONS: usize = 96;
const SPEAKER_COUNT_EIGENSOLVER_TOLERANCE: f64 = 1.0e-7;
const SPEAKER_COUNT_EIGENSOLVER_DIAGONAL_SHIFT: f64 = 1.01;
const SPEAKER_COUNT_SOFT_PRIOR_MIX_WEIGHT: f64 = 0.15;
const SPEAKER_COUNT_SOFT_PRIOR_STABILITY_ATTENUATION: f64 = 0.50;
const SPEAKER_COUNT_JACKKNIFE_LOG_WEIGHT: f64 = 0.25;
const SPEAKER_COUNT_SPECTRAL_LOG_WEIGHT: f64 = 0.35;
const SPEAKER_COUNT_STABILITY_EVIDENCE_WEIGHT: f64 = 0.55;
const SPEAKER_COUNT_RISK_EVIDENCE_WEIGHT: f64 = 0.30;
const SPEAKER_COUNT_SPECTRAL_EVIDENCE_WEIGHT: f64 = 0.15;
const SPEAKER_COUNT_UNRESOLVED_INTERCEPT: f64 = 0.90;
const SPEAKER_COUNT_UNRESOLVED_EVIDENCE_SLOPE: f64 = 0.75;
const SPEAKER_COUNT_MINIMUM_UNRESOLVED_MASS: f64 = 0.15;
const SPEAKER_COUNT_MAXIMUM_UNRESOLVED_MASS: f64 = 0.90;
const SPEAKER_COUNT_CONCRETE_CREDIBLE_MASS: f64 = 0.90;
const SPEAKER_COUNT_OCCUPANCY_STABILITY_WEIGHT: f64 = 0.30;
/// Versioned monotone map from the candidate's raw attribution likelihood to
/// its reported confidence. Rejection continues to use the raw likelihood so
/// calibration cannot silently increase selective coverage.
pub const ACOUSTIC_ASSIGNMENT_CONFIDENCE_CALIBRATION_VERSION: &str =
    "ami-development-raw-likelihood-v2";
const ACOUSTIC_ASSIGNMENT_CONFIDENCE_FLOOR: f32 = 0.0;
const ACOUSTIC_ASSIGNMENT_CONFIDENCE_SCALE: f32 = 1.0;
/// Frozen identity for the first variance-aware change posterior.
pub const ACOUSTIC_CHANGE_CALIBRATION_VERSION: &str = "acoustic-change-posterior-v2";
/// Public development protocol used to fit the v2 operating point.
pub const ACOUSTIC_CHANGE_CALIBRATION_FIT_VERSION: &str = "ami-scenario-development-prefix-v1";
/// Identity for the bounded deterministic terminal Page-Hinkley candidate.
pub const ACOUSTIC_CHANGE_PAGE_HINKLEY_VERSION: &str = "acoustic-change-page-hinkley-v1";
/// Identity for the bounded diagonal two-regime Bayesian candidate.
pub const ACOUSTIC_CHANGE_BAYESIAN_VERSION: &str = "acoustic-change-bayesian-two-regime-v1";
/// Identity for the frozen pre-posterior fallback detector.
pub const ACOUSTIC_CHANGE_FIXED_SAFE_VERSION: &str = "acoustic-change-fixed-safe-v1";
/// Frozen identity for the existing distance/BIC-like clustering fallback.
pub const ACOUSTIC_CLUSTERING_FIXED_SAFE_VERSION: &str = "acoustic-clustering-fixed-safe-v1";
/// Development identity for probabilistic pair scoring and stable count selection.
pub const ACOUSTIC_CLUSTERING_PROBABILISTIC_VERSION: &str =
    "acoustic-clustering-probabilistic-v3-development";
/// Public schema for bounded count distributions with explicit unresolved mass.
pub const SPEAKER_COUNT_ESTIMATE_SCHEMA_VERSION: &str = "speaker-count-estimate-v2";
const TEMPORAL_KNOWN_SWITCH_BASE: f32 = 0.22;
const TEMPORAL_UNKNOWN_SWITCH_BASE: f32 = 0.10;
const TEMPORAL_KNOWN_BOUNDARY_CREDIT: f32 = 0.18;
const TEMPORAL_UNKNOWN_BOUNDARY_CREDIT: f32 = 0.07;
const TEMPORAL_MAX_GAP_CREDIT: f32 = 0.10;
const TEMPORAL_FULL_GAP_MS: u64 = 500;
const TEMPORAL_SHORT_RUN_MS: u64 = 350;
const TEMPORAL_PREMATURE_SWITCH_PENALTY: f32 = 0.15;
const TEMPORAL_FRAGMENT_MS: u64 = 150;
const TEMPORAL_FRAGMENT_PENALTY: f32 = 0.05;

/// Explicit acoustic clustering selection used by development ablations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcousticClusteringMode {
    FixedSafeV1,
    ProbabilisticV1,
}

impl AcousticClusteringMode {
    pub const ALL: [Self; 2] = [Self::FixedSafeV1, Self::ProbabilisticV1];

    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::FixedSafeV1 => ACOUSTIC_CLUSTERING_FIXED_SAFE_VERSION,
            Self::ProbabilisticV1 => ACOUSTIC_CLUSTERING_PROBABILISTIC_VERSION,
        }
    }
}

/// Loss and posterior parameters for one same-speaker merge decision.
///
/// These values define a development candidate, not a certified calibration.
/// Public-corpus evidence must promote the candidate before production callers
/// may select it by default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticSpeakerPairCalibration {
    pub variance_floor: f32,
    pub different_logit_intercept: f32,
    pub voice_distance_weight: f32,
    pub channel_distance_weight: f32,
    pub full_support_frames: f32,
    pub false_split_loss: f32,
    pub false_merge_loss: f32,
    pub minimum_assignment_probability: f32,
    pub maximum_unknown_prior: f32,
    pub minimum_stable_lane_fraction: f32,
}

/// Return the predeclared development speaker-pair loss contract.
#[must_use]
pub const fn acoustic_speaker_pair_calibration() -> AcousticSpeakerPairCalibration {
    AcousticSpeakerPairCalibration {
        variance_floor: 0.05,
        different_logit_intercept: -3.0,
        voice_distance_weight: 4.0,
        channel_distance_weight: 0.10,
        full_support_frames: 50.0,
        false_split_loss: 1.0,
        false_merge_loss: 12.0,
        minimum_assignment_probability: 0.55,
        maximum_unknown_prior: 0.80,
        minimum_stable_lane_fraction: 3.0 / 5.0,
    }
}

/// Stable SHA-256 of the development speaker-pair and count-selection contract.
#[must_use]
pub fn acoustic_speaker_pair_calibration_sha256() -> String {
    let calibration = acoustic_speaker_pair_calibration();
    let mut hasher = Sha256::new();
    hasher.update(b"acoustic-speaker-pair-calibration\0");
    hasher.update(ACOUSTIC_CLUSTERING_PROBABILISTIC_VERSION.as_bytes());
    for value in [
        calibration.variance_floor,
        calibration.different_logit_intercept,
        calibration.voice_distance_weight,
        calibration.channel_distance_weight,
        calibration.full_support_frames,
        calibration.false_split_loss,
        calibration.false_merge_loss,
        calibration.minimum_assignment_probability,
        calibration.maximum_unknown_prior,
        calibration.minimum_stable_lane_fraction,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.update((SPEAKER_PAIR_MINIMUM_ACTIVE_DIMENSIONS as u64).to_le_bytes());
    hasher.update((SPEAKER_COUNT_PERTURBATION_LANES as u64).to_le_bytes());
    hasher.update((SPEAKER_COUNT_SPARSE_NEIGHBOR_DEGREE as u64).to_le_bytes());
    hasher.update((SPEAKER_COUNT_EIGENSOLVER_MAX_ITERATIONS as u64).to_le_bytes());
    for value in [
        SPEAKER_COUNT_EIGENSOLVER_TOLERANCE,
        SPEAKER_COUNT_EIGENSOLVER_DIAGONAL_SHIFT,
        SPEAKER_COUNT_SOFT_PRIOR_MIX_WEIGHT,
        SPEAKER_COUNT_SOFT_PRIOR_STABILITY_ATTENUATION,
        SPEAKER_COUNT_JACKKNIFE_LOG_WEIGHT,
        SPEAKER_COUNT_SPECTRAL_LOG_WEIGHT,
        SPEAKER_COUNT_STABILITY_EVIDENCE_WEIGHT,
        SPEAKER_COUNT_RISK_EVIDENCE_WEIGHT,
        SPEAKER_COUNT_SPECTRAL_EVIDENCE_WEIGHT,
        SPEAKER_COUNT_UNRESOLVED_INTERCEPT,
        SPEAKER_COUNT_UNRESOLVED_EVIDENCE_SLOPE,
        SPEAKER_COUNT_MINIMUM_UNRESOLVED_MASS,
        SPEAKER_COUNT_MAXIMUM_UNRESOLVED_MASS,
        SPEAKER_COUNT_CONCRETE_CREDIBLE_MASS,
        SPEAKER_COUNT_OCCUPANCY_STABILITY_WEIGHT,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.update(b"bounded-soft-count-prior-linear-pool-v1");
    hasher.update(b"feature-jackknife-full-no-pitch-no-dynamics-no-formants-no-channel-v3");
    hasher.update(ACOUSTIC_ASSIGNMENT_CONFIDENCE_CALIBRATION_VERSION.as_bytes());
    for value in [
        ACOUSTIC_ASSIGNMENT_CONFIDENCE_FLOOR,
        ACOUSTIC_ASSIGNMENT_CONFIDENCE_SCALE,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.update(b"duration-aware-temporal-v1");
    for value in [
        TEMPORAL_KNOWN_SWITCH_BASE,
        TEMPORAL_UNKNOWN_SWITCH_BASE,
        TEMPORAL_KNOWN_BOUNDARY_CREDIT,
        TEMPORAL_UNKNOWN_BOUNDARY_CREDIT,
        TEMPORAL_MAX_GAP_CREDIT,
        TEMPORAL_PREMATURE_SWITCH_PENALTY,
        TEMPORAL_FRAGMENT_PENALTY,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    for value in [
        TEMPORAL_FULL_GAP_MS,
        TEMPORAL_SHORT_RUN_MS,
        TEMPORAL_FRAGMENT_MS,
    ] {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Explicit speaker-change detector selection used by public ablations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcousticChangeDetectorMode {
    CalibratedPosterior,
    PageHinkleyV1,
    BayesianTwoRegimeV1,
    FixedSafeV1,
}

impl AcousticChangeDetectorMode {
    pub const ALL: [Self; 4] = [
        Self::CalibratedPosterior,
        Self::PageHinkleyV1,
        Self::BayesianTwoRegimeV1,
        Self::FixedSafeV1,
    ];

    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::CalibratedPosterior => ACOUSTIC_CHANGE_CALIBRATION_VERSION,
            Self::PageHinkleyV1 => ACOUSTIC_CHANGE_PAGE_HINKLEY_VERSION,
            Self::BayesianTwoRegimeV1 => ACOUSTIC_CHANGE_BAYESIAN_VERSION,
            Self::FixedSafeV1 => ACOUSTIC_CHANGE_FIXED_SAFE_VERSION,
        }
    }

    #[must_use]
    const fn uses_variance_aware_model(self) -> bool {
        !matches!(self, Self::FixedSafeV1)
    }
}

/// Frozen loss and calibration contract for acoustic change decisions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticChangeCalibration {
    pub diagonal_shrinkage: f32,
    pub variance_floor: f32,
    pub logit_intercept: f32,
    pub voice_evidence_weight: f32,
    pub channel_evidence_weight: f32,
    pub page_hinkley_allowance: f32,
    pub page_hinkley_logit_intercept: f32,
    pub page_hinkley_voice_evidence_weight: f32,
    pub bayesian_occam_weight: f32,
    pub bayesian_logit_intercept: f32,
    pub bayesian_voice_evidence_weight: f32,
    pub silence_gap_logit_bonus: f32,
    pub tiny_diarize_logit_bonus: f32,
    pub decision_probability: f32,
    pub hysteresis_reset_probability_ratio: f32,
    pub hysteresis_max_frames: usize,
    pub hysteresis_rearm_frames: usize,
    pub strong_peak_probability: f32,
    pub strong_peak_suppression_frames: usize,
    pub refinement_radius_frames: usize,
    pub minimum_valid_scales: usize,
    pub false_split_loss: f32,
    pub missed_change_loss: f32,
    pub timing_error_loss_per_second: f32,
    pub hint_contradiction_loss: f32,
    pub latency_loss_per_second: f32,
    pub fallback_loss: f32,
}

/// Return the predeclared change-posterior parameters.
#[must_use]
pub const fn acoustic_change_calibration() -> AcousticChangeCalibration {
    AcousticChangeCalibration {
        diagonal_shrinkage: 0.25,
        variance_floor: 0.0025,
        logit_intercept: -4.0,
        voice_evidence_weight: 5.0,
        channel_evidence_weight: 0.50,
        page_hinkley_allowance: 0.20,
        page_hinkley_logit_intercept: -4.0,
        page_hinkley_voice_evidence_weight: 5.0,
        bayesian_occam_weight: 1.0,
        bayesian_logit_intercept: -4.0,
        bayesian_voice_evidence_weight: 5.0,
        silence_gap_logit_bonus: 0.35,
        tiny_diarize_logit_bonus: 0.20,
        decision_probability: 0.10,
        hysteresis_reset_probability_ratio: CHANGE_HYSTERESIS_RESET_RATIO,
        hysteresis_max_frames: CHANGE_HYSTERESIS_MAX_FRAMES,
        hysteresis_rearm_frames: CHANGE_HYSTERESIS_REARM_FRAMES,
        strong_peak_probability: CHANGE_STRONG_PEAK_PROBABILITY,
        strong_peak_suppression_frames: CHANGE_STRONG_PEAK_SUPPRESSION_FRAMES,
        refinement_radius_frames: CHANGE_REFINEMENT_RADIUS_FRAMES,
        minimum_valid_scales: 2,
        false_split_loss: 1.0,
        missed_change_loss: 9.0,
        timing_error_loss_per_second: 0.25,
        hint_contradiction_loss: 4.0,
        latency_loss_per_second: 0.05,
        fallback_loss: 0.10,
    }
}

/// Stable SHA-256 of the frozen change-posterior and loss contract.
#[must_use]
pub fn acoustic_change_calibration_sha256() -> String {
    let calibration = acoustic_change_calibration();
    let mut hasher = Sha256::new();
    hasher.update(b"acoustic-change-calibration\0");
    hasher.update(ACOUSTIC_CHANGE_CALIBRATION_VERSION.as_bytes());
    hasher.update(b"\0");
    hasher.update(ACOUSTIC_CHANGE_CALIBRATION_FIT_VERSION.as_bytes());
    for value in [
        calibration.diagonal_shrinkage,
        calibration.variance_floor,
        calibration.logit_intercept,
        calibration.voice_evidence_weight,
        calibration.channel_evidence_weight,
        calibration.page_hinkley_allowance,
        calibration.page_hinkley_logit_intercept,
        calibration.page_hinkley_voice_evidence_weight,
        calibration.bayesian_occam_weight,
        calibration.bayesian_logit_intercept,
        calibration.bayesian_voice_evidence_weight,
        calibration.silence_gap_logit_bonus,
        calibration.tiny_diarize_logit_bonus,
        calibration.decision_probability,
        calibration.hysteresis_reset_probability_ratio,
        calibration.strong_peak_probability,
        calibration.false_split_loss,
        calibration.missed_change_loss,
        calibration.timing_error_loss_per_second,
        calibration.hint_contradiction_loss,
        calibration.latency_loss_per_second,
        calibration.fallback_loss,
    ] {
        hasher.update(value.to_bits().to_le_bytes());
    }
    hasher.update((calibration.hysteresis_max_frames as u64).to_le_bytes());
    hasher.update((calibration.hysteresis_rearm_frames as u64).to_le_bytes());
    hasher.update((calibration.strong_peak_suppression_frames as u64).to_le_bytes());
    hasher.update((calibration.refinement_radius_frames as u64).to_le_bytes());
    hasher.update((calibration.minimum_valid_scales as u64).to_le_bytes());
    format!("{:x}", hasher.finalize())
}

/// Deterministic action selected by the acoustic change controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticChangeAction {
    NoBoundary,
    Defer,
    EmitBoundary,
    ConservativeFallback,
}

/// Reason the calibrated posterior could not make an authoritative decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticChangeFallbackReason {
    InsufficientVoiceSupport,
    InvalidCovariance,
}

/// Non-lexical timing evidence that may constrain or snap acoustic boundaries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AcousticBoundaryHints {
    /// VAD speech regions. Starts and ends are structural boundaries.
    pub speech_regions_ms: Vec<(u64, u64)>,
    /// Legal DTW word boundaries. These only snap a nearby acoustic candidate.
    pub word_boundaries_ms: Vec<u64>,
    /// Optional turn-token times emitted by a compatible acoustic decoder.
    pub tiny_diarize_boundaries_ms: Vec<u64>,
}

/// Auditable component evidence for a detected acoustic regime change.
#[derive(Debug, Clone, PartialEq)]
pub struct ChangePointEvidence {
    pub boundary_ms: u64,
    pub voice_distance: f32,
    pub channel_distance: f32,
    pub multiscale_scores: [f32; 5],
    pub raw_log_odds: f32,
    pub change_probability: f32,
    pub supporting_scale_mask: u8,
    pub refinement_offset_frames: i16,
    pub action: AcousticChangeAction,
    pub fallback_reason: Option<AcousticChangeFallbackReason>,
    pub detector_mode: AcousticChangeDetectorMode,
    pub calibration_id: &'static str,
    pub silence_gap: bool,
    pub snapped_to_word: bool,
    pub tiny_diarize_support: bool,
    pub vad_boundary: bool,
    /// Boundary forced at the guarded edge of a known-speaker interval.
    pub supervised_boundary: bool,
}

/// Evaluation-only acoustic evidence kept outside the stable diarization report.
///
/// Runtime callers do not populate `evaluated`; the public-corpus evaluator
/// opts in so it can reduce the complete score stream to aggregate threshold
/// and calibration diagnostics. The evaluator's existing audio-byte cap gives
/// this duration-proportional diagnostic state a fixed upper bound.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct AcousticChangeEvaluationEvidence {
    pub emitted: Vec<ChangePointEvidence>,
    pub evaluated: Vec<ChangePointEvidence>,
}

/// Aggregate-safe clustering diagnostics returned only to evaluators.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AcousticClusteringEvaluationEvidence {
    pub requested_mode: AcousticClusteringMode,
    pub executed_mode: AcousticClusteringMode,
    pub fallback_reason: Option<AcousticClusteringFallbackReason>,
    pub speaker_count_stability: f32,
}

/// Compact speaker-homogeneous observation retained after frame segmentation.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticTracklet {
    pub tracklet_index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub frame_count: usize,
    pub voiced_frame_count: usize,
    pub identity_frame_count: usize,
    pub channel_frame_count: usize,
    pub voice_mean: [f32; VOICE_VECTOR_DIMENSIONS],
    pub voice_variance: [f32; VOICE_VECTOR_DIMENSIONS],
    pub voice_valid: [bool; VOICE_VECTOR_DIMENSIONS],
    pub voice_support: [u32; VOICE_VECTOR_DIMENSIONS],
    pub channel_mean: [f32; CHANNEL_VECTOR_DIMENSIONS],
    pub channel_variance: [f32; CHANNEL_VECTOR_DIMENSIONS],
    pub channel_valid: bool,
    /// Active prefix of `channel_mean` and `channel_variance`.
    ///
    /// This is explicit because v1 owns eight channel coordinates, v2 owns
    /// fourteen, and the no-channel ablation owns zero. Inferring that choice
    /// from voice masks makes ablations dependent on which coordinates happen
    /// to be observable in a particular frame.
    pub channel_dimensions: usize,
    pub change_confidence: f32,
    pub overlap_probability: f32,
    pub overlap_suspected: bool,
    pub boundary_evidence: Option<ChangePointEvidence>,
}

/// Summary proving that change detection retained a fixed rolling horizon.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticSegmentationSummary {
    pub input_frame_count: usize,
    pub tracklet_count: usize,
    pub acoustic_change_count: usize,
    pub forced_boundary_count: usize,
    pub maximum_retained_frames: usize,
    pub normalized_voice_dimensions: usize,
    pub normalized_channel_dimensions: usize,
    pub missing_voice_dimensions: usize,
    pub posterior_candidate_count: usize,
    pub page_hinkley_candidate_count: usize,
    pub bayesian_candidate_count: usize,
    pub fixed_candidate_count: usize,
    pub fallback_candidate_count: usize,
}

#[derive(Debug)]
struct ChangePeakSelector {
    detector_mode: AcousticChangeDetectorMode,
    threshold: f32,
    reset_threshold: f32,
    maximum_active_frames: usize,
    rearm_frames: usize,
    strong_probability: f32,
    strong_suppression_frames: usize,
    pending: Option<(usize, usize, ChangePointEvidence)>,
    low_probability_frames: usize,
    armed: bool,
}

impl ChangePeakSelector {
    fn new(detector_mode: AcousticChangeDetectorMode, threshold: f32) -> Self {
        let calibration = acoustic_change_calibration();
        Self {
            detector_mode,
            threshold,
            reset_threshold: threshold * calibration.hysteresis_reset_probability_ratio,
            maximum_active_frames: calibration.hysteresis_max_frames,
            rearm_frames: calibration.hysteresis_rearm_frames,
            strong_probability: calibration.strong_peak_probability,
            strong_suppression_frames: calibration.strong_peak_suppression_frames,
            pending: None,
            low_probability_frames: 0,
            armed: true,
        }
    }

    fn push(
        &mut self,
        frame_index: usize,
        evidence: ChangePointEvidence,
    ) -> Option<ChangePointEvidence> {
        let probability = evidence.change_probability;
        if self.detector_mode == AcousticChangeDetectorMode::FixedSafeV1 {
            if probability >= self.threshold {
                match &mut self.pending {
                    Some((_, peak_index, peak_evidence))
                        if frame_index
                            <= peak_index.saturating_add(CHANGE_FIXED_SAFE_SUPPRESSION_FRAMES) =>
                    {
                        if probability > peak_evidence.change_probability {
                            *peak_index = frame_index;
                            *peak_evidence = evidence;
                        }
                    }
                    Some(_) => {
                        let previous = self.pending.replace((frame_index, frame_index, evidence));
                        return previous.map(|(_, _, peak)| peak);
                    }
                    None => self.pending = Some((frame_index, frame_index, evidence)),
                }
            } else if self.pending.as_ref().is_some_and(|(_, peak_index, _)| {
                frame_index > peak_index.saturating_add(CHANGE_FIXED_SAFE_SUPPRESSION_FRAMES)
            }) {
                return self.pending.take().map(|(_, _, peak)| peak);
            }
            return None;
        }

        if self
            .pending
            .as_ref()
            .is_some_and(|(_, _, peak)| peak.change_probability >= self.strong_probability)
        {
            let within_suppression = self.pending.as_ref().is_some_and(|(_, peak_index, _)| {
                frame_index <= peak_index.saturating_add(self.strong_suppression_frames)
            });
            if within_suppression {
                if let Some((_, peak_index, peak_evidence)) = &mut self.pending
                    && probability > peak_evidence.change_probability
                {
                    *peak_index = frame_index;
                    *peak_evidence = evidence;
                }
                return None;
            }
            let emitted = self.pending.take().map(|(_, _, peak)| peak);
            self.low_probability_frames = 0;
            self.armed = true;
            if probability >= self.threshold {
                self.pending = Some((frame_index, frame_index, evidence));
            }
            return emitted;
        }

        if let Some((start_index, peak_index, peak_evidence)) = &mut self.pending {
            if probability > peak_evidence.change_probability {
                *peak_index = frame_index;
                *peak_evidence = evidence;
            }
            if probability < self.reset_threshold {
                self.low_probability_frames = self.low_probability_frames.saturating_add(1);
            } else {
                self.low_probability_frames = 0;
            }
            let reset = self.low_probability_frames >= self.rearm_frames;
            let latency_cap = frame_index >= start_index.saturating_add(self.maximum_active_frames);
            if (reset || latency_cap)
                && let Some((_, _, emitted)) = self.pending.take()
            {
                self.armed = reset;
                if reset {
                    self.low_probability_frames = 0;
                }
                return Some(emitted);
            }
            return None;
        }

        if !self.armed {
            if probability < self.reset_threshold {
                self.low_probability_frames = self.low_probability_frames.saturating_add(1);
                if self.low_probability_frames >= self.rearm_frames {
                    self.armed = true;
                    self.low_probability_frames = 0;
                }
            } else {
                self.low_probability_frames = 0;
            }
            return None;
        }
        if probability >= self.threshold {
            self.pending = Some((frame_index, frame_index, evidence));
        }
        None
    }

    fn finish(&mut self) -> Option<ChangePointEvidence> {
        self.pending.take().map(|(_, _, evidence)| evidence)
    }
}

struct AcousticSegmenter<'a> {
    hints: &'a AcousticBoundaryHints,
    feature_ablation: AcousticFeatureAblation,
    detector_mode: AcousticChangeDetectorMode,
    forced_boundaries: BTreeMap<usize, ChangePointEvidence>,
    forced_boundary_count: usize,
    detected_boundaries: BTreeMap<usize, ChangePointEvidence>,
    emitted_boundaries: BTreeMap<usize, ChangePointEvidence>,
    capture_evaluation_evidence: bool,
    evaluated_change_evidence: Vec<ChangePointEvidence>,
    peak_selector: ChangePeakSelector,
    ring: VecDeque<AcousticFrameFeatures>,
    sidecar_signal_ring: Option<VecDeque<AcousticSidecarBoundarySignal>>,
    accumulator: TrackletAccumulator,
    tracklets: Vec<AcousticTracklet>,
    input_frame_count: usize,
    last_frame: Option<(usize, u64)>,
    last_evaluated_frame_index: Option<usize>,
    maximum_retained_frames: usize,
    maximum_retained_sidecar_signals: usize,
    consumed_sidecar_probability_count: usize,
    changed_boundary_probability_count: usize,
    posterior_candidate_count: usize,
    page_hinkley_candidate_count: usize,
    bayesian_candidate_count: usize,
    fixed_candidate_count: usize,
    fallback_candidate_count: usize,
}

impl<'a> AcousticSegmenter<'a> {
    #[cfg(test)]
    fn new(hints: &'a AcousticBoundaryHints) -> FwResult<Self> {
        Self::new_with_supervised_boundaries(
            hints,
            &[],
            AcousticFeatureAblation::FullV2,
            AcousticChangeDetectorMode::CalibratedPosterior,
        )
    }

    fn new_with_supervised_boundaries(
        hints: &'a AcousticBoundaryHints,
        supervised_boundaries_ms: &[u64],
        feature_ablation: AcousticFeatureAblation,
        detector_mode: AcousticChangeDetectorMode,
    ) -> FwResult<Self> {
        validate_boundary_hints(hints)?;
        if supervised_boundaries_ms
            .windows(2)
            .any(|window| window[0] > window[1])
        {
            return Err(FwError::InvalidRequest(
                "supervised boundaries must be ordered".to_owned(),
            ));
        }
        let forced_boundaries = forced_boundary_map(hints, supervised_boundaries_ms, detector_mode);
        let forced_boundary_count = forced_boundaries.len();
        Ok(Self {
            hints,
            feature_ablation,
            detector_mode,
            forced_boundaries,
            forced_boundary_count,
            detected_boundaries: BTreeMap::new(),
            emitted_boundaries: BTreeMap::new(),
            capture_evaluation_evidence: false,
            evaluated_change_evidence: Vec::new(),
            peak_selector: ChangePeakSelector::new(
                detector_mode,
                acoustic_change_calibration().decision_probability,
            ),
            ring: VecDeque::with_capacity(CHANGE_RING_FRAMES),
            sidecar_signal_ring: None,
            accumulator: TrackletAccumulator::default(),
            tracklets: Vec::new(),
            input_frame_count: 0,
            last_frame: None,
            last_evaluated_frame_index: None,
            maximum_retained_frames: 0,
            maximum_retained_sidecar_signals: 0,
            consumed_sidecar_probability_count: 0,
            changed_boundary_probability_count: 0,
            posterior_candidate_count: 0,
            page_hinkley_candidate_count: 0,
            bayesian_candidate_count: 0,
            fixed_candidate_count: 0,
            fallback_candidate_count: 0,
        })
    }

    fn push(&mut self, frame: AcousticFrameFeatures) -> FwResult<()> {
        self.push_internal(frame, None)
    }

    fn enable_sidecar_fusion(&mut self) -> FwResult<()> {
        if self.input_frame_count != 0 || !self.ring.is_empty() {
            return Err(FwError::InvalidRequest(
                "sidecar fusion must be enabled before segmentation begins".to_owned(),
            ));
        }
        if self.sidecar_signal_ring.is_some() {
            return Err(FwError::InvalidRequest(
                "sidecar fusion was enabled more than once".to_owned(),
            ));
        }
        self.sidecar_signal_ring = Some(VecDeque::with_capacity(CHANGE_RING_FRAMES));
        Ok(())
    }

    fn push_with_sidecar_signal(
        &mut self,
        frame: AcousticFrameFeatures,
        signal: AcousticSidecarBoundarySignal,
    ) -> FwResult<()> {
        self.push_internal(frame, Some(signal))
    }

    fn push_internal(
        &mut self,
        frame: AcousticFrameFeatures,
        sidecar_signal: Option<AcousticSidecarBoundarySignal>,
    ) -> FwResult<()> {
        validate_acoustic_frame(&frame)?;
        match (&self.sidecar_signal_ring, sidecar_signal.as_ref()) {
            (None, None) => {}
            (Some(_), Some(signal))
                if signal.frame_index == frame.frame_index
                    && signal.observation.frame_index == frame.frame_index =>
            {
            }
            (Some(_), Some(_)) => {
                return Err(FwError::InvalidRequest(
                    "sidecar signal and acoustic frame indices must align".to_owned(),
                ));
            }
            (Some(_), None) => {
                return Err(FwError::InvalidRequest(
                    "sidecar-enabled segmentation requires one signal per frame".to_owned(),
                ));
            }
            (None, Some(_)) => {
                return Err(FwError::InvalidRequest(
                    "sidecar signals require explicitly enabled evaluation fusion".to_owned(),
                ));
            }
        }
        if let Some((previous_index, previous_start_ms)) = self.last_frame
            && (frame.frame_index != previous_index + 1 || frame.start_ms < previous_start_ms)
        {
            return Err(FwError::InvalidRequest(
                "acoustic frames must be contiguous and time-ordered".to_owned(),
            ));
        }
        self.last_frame = Some((frame.frame_index, frame.start_ms));
        self.input_frame_count += 1;
        self.ring.push_back(frame);
        self.maximum_retained_frames = self.maximum_retained_frames.max(self.ring.len());
        if let (Some(signal_ring), Some(signal)) =
            (self.sidecar_signal_ring.as_mut(), sidecar_signal)
        {
            signal_ring.push_back(signal);
            self.maximum_retained_sidecar_signals = self
                .maximum_retained_sidecar_signals
                .max(signal_ring.len());
        }

        if self.ring.len() == CHANGE_RING_FRAMES {
            let latest_center = CHANGE_SCALES_FRAMES[4];
            let earliest_center = self
                .last_evaluated_frame_index
                .map_or(CHANGE_SCALES_FRAMES[0], |_| latest_center);
            for center in earliest_center..=latest_center {
                let center_index = self.ring[center].frame_index;
                if self
                    .last_evaluated_frame_index
                    .is_none_or(|last| center_index > last)
                {
                    self.evaluate_ring_center(center)?;
                }
            }

            if let Some(signal_ring) = self.sidecar_signal_ring.as_ref()
                && signal_ring.front().map(|signal| signal.frame_index)
                    != self.ring.front().map(|frame| frame.frame_index)
            {
                return Err(FwError::InvalidRequest(
                    "sidecar and acoustic segmentation rings lost alignment".to_owned(),
                ));
            }
            let oldest = self.ring.pop_front().ok_or_else(|| {
                FwError::InvalidRequest("segmentation ring unexpectedly empty".to_owned())
            })?;
            if let Some(signal_ring) = self.sidecar_signal_ring.as_mut() {
                signal_ring.pop_front();
            }
            consume_segment_frame(
                oldest,
                &mut self.accumulator,
                &mut self.tracklets,
                &mut self.detected_boundaries,
                &mut self.forced_boundaries,
                self.feature_ablation,
            );
        }
        Ok(())
    }

    fn finish(
        mut self,
    ) -> FwResult<(
        Vec<AcousticTracklet>,
        AcousticSegmentationSummary,
        AcousticChangeEvaluationEvidence,
    )> {
        if self.ring.len() > 2 * CHANGE_SCALES_FRAMES[0] {
            let first_frame_index = self.ring.front().map_or(0, |frame| frame.frame_index);
            let earliest_center = self
                .last_evaluated_frame_index
                .and_then(|last| last.checked_add(1))
                .and_then(|next| next.checked_sub(first_frame_index))
                .unwrap_or(CHANGE_SCALES_FRAMES[0])
                .max(CHANGE_SCALES_FRAMES[0]);
            let latest_center = self.ring.len() - CHANGE_SCALES_FRAMES[0];
            for center in earliest_center..=latest_center {
                self.evaluate_ring_center(center);
            }
        }
        if let Some(peak_evidence) = self.peak_selector.finish() {
            self.emit_detected_boundary(peak_evidence);
        }
        while let Some(frame) = self.ring.pop_front() {
            consume_segment_frame(
                frame,
                &mut self.accumulator,
                &mut self.tracklets,
                &mut self.detected_boundaries,
                &mut self.forced_boundaries,
                self.feature_ablation,
            );
        }
        if let Some(tracklet) = self.accumulator.finish(self.tracklets.len(), None) {
            self.tracklets.push(tracklet);
        }
        merge_compatible_adjacent_tracklets(&mut self.tracklets);
        if !self.hints.speech_regions_ms.is_empty() {
            self.tracklets.retain(|tracklet| {
                let midpoint = tracklet.start_ms + (tracklet.end_ms - tracklet.start_ms) / 2;
                self.hints
                    .speech_regions_ms
                    .iter()
                    .any(|&(start_ms, end_ms)| midpoint >= start_ms && midpoint < end_ms)
            });
        }
        for (index, tracklet) in self.tracklets.iter_mut().enumerate() {
            tracklet.tracklet_index = index;
        }
        let normalization = if self.feature_ablation.schema_version()
            == AcousticFeatureSchemaVersion::V2
        {
            normalize_tracklet_features(&mut self.tracklets)
        } else {
            AcousticNormalizationSummary {
                missing_voice_dimensions: VOICE_VECTOR_DIMENSIONS
                    - acoustic_feature_schema(AcousticFeatureSchemaVersion::V1).voice_dimensions,
                ..AcousticNormalizationSummary::default()
            }
        };

        let acoustic_change_count =
            self.tracklets
                .iter()
                .filter(|tracklet| {
                    tracklet.boundary_evidence.as_ref().is_some_and(|evidence| {
                        !evidence.vad_boundary && !evidence.supervised_boundary
                    })
                })
                .count();
        let summary = AcousticSegmentationSummary {
            input_frame_count: self.input_frame_count,
            tracklet_count: self.tracklets.len(),
            acoustic_change_count,
            forced_boundary_count: self.forced_boundary_count,
            maximum_retained_frames: self.maximum_retained_frames,
            normalized_voice_dimensions: normalization.normalized_voice_dimensions,
            normalized_channel_dimensions: normalization.normalized_channel_dimensions,
            missing_voice_dimensions: normalization.missing_voice_dimensions,
            posterior_candidate_count: self.posterior_candidate_count,
            page_hinkley_candidate_count: self.page_hinkley_candidate_count,
            bayesian_candidate_count: self.bayesian_candidate_count,
            fixed_candidate_count: self.fixed_candidate_count,
            fallback_candidate_count: self.fallback_candidate_count,
        };
        Ok((
            self.tracklets,
            summary,
            AcousticChangeEvaluationEvidence {
                emitted: self.emitted_boundaries.into_values().collect(),
                evaluated: self.evaluated_change_evidence,
            },
        ))
    }

    fn emit_detected_boundary(&mut self, evidence: ChangePointEvidence) {
        insert_detected_boundary(&mut self.detected_boundaries, evidence.clone());
        insert_detected_boundary(&mut self.emitted_boundaries, evidence);
    }

    fn evaluate_ring_center(&mut self, center: usize) {
        let center_index = self.ring[center].frame_index;
        let mut evidence = multiscale_change_evidence_with_detector(
            &self.ring,
            center,
            self.hints,
            self.feature_ablation,
            self.detector_mode,
        );
        if self.capture_evaluation_evidence {
            self.evaluated_change_evidence.push(evidence.clone());
        }
        let decision_threshold = acoustic_change_calibration().decision_probability;
        if evidence.change_probability >= decision_threshold {
            match (self.detector_mode, evidence.fallback_reason) {
                (mode, Some(_)) if mode.uses_variance_aware_model() => {
                    self.fallback_candidate_count += 1;
                    evidence.action = AcousticChangeAction::ConservativeFallback;
                }
                (AcousticChangeDetectorMode::CalibratedPosterior, None) => {
                    self.posterior_candidate_count += 1;
                    evidence.action = AcousticChangeAction::Defer;
                }
                (AcousticChangeDetectorMode::PageHinkleyV1, None) => {
                    self.page_hinkley_candidate_count += 1;
                    evidence.action = AcousticChangeAction::Defer;
                }
                (AcousticChangeDetectorMode::BayesianTwoRegimeV1, None) => {
                    self.bayesian_candidate_count += 1;
                    evidence.action = AcousticChangeAction::Defer;
                }
                (AcousticChangeDetectorMode::FixedSafeV1, _) => {
                    self.fixed_candidate_count += 1;
                    evidence.action = AcousticChangeAction::Defer;
                }
                (_, Some(_)) => {
                    self.fallback_candidate_count += 1;
                    evidence.action = AcousticChangeAction::ConservativeFallback;
                }
            }
        }
        if let Some(peak_evidence) = self.peak_selector.push(center_index, evidence) {
            self.emit_detected_boundary(peak_evidence);
        }
        self.last_evaluated_frame_index = Some(center_index);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct AcousticNormalizationSummary {
    normalized_voice_dimensions: usize,
    normalized_channel_dimensions: usize,
    missing_voice_dimensions: usize,
}

fn normalize_tracklet_features(tracklets: &mut [AcousticTracklet]) -> AcousticNormalizationSummary {
    let mut summary = AcousticNormalizationSummary::default();
    for dimension in 0..VOICE_VECTOR_DIMENSIONS {
        let mut values = tracklets
            .iter()
            .filter(|tracklet| tracklet.voice_valid[dimension])
            .map(|tracklet| tracklet.voice_mean[dimension])
            .collect::<Vec<_>>();
        if values.is_empty() {
            summary.missing_voice_dimensions += 1;
            continue;
        }
        values.sort_by(f32::total_cmp);
        let center = unweighted_quantile(&values, 0.5);
        let q25 = unweighted_quantile(&values, 0.25);
        let q75 = unweighted_quantile(&values, 0.75);
        let mut deviations = values
            .iter()
            .map(|value| (value - center).abs())
            .collect::<Vec<_>>();
        deviations.sort_by(f32::total_cmp);
        let scale = (1.4826 * unweighted_quantile(&deviations, 0.5))
            .max((q75 - q25).abs() / 1.349)
            .max(0.05);
        for tracklet in tracklets
            .iter_mut()
            .filter(|tracklet| tracklet.voice_valid[dimension])
        {
            tracklet.voice_mean[dimension] = (tracklet.voice_mean[dimension] - center) / scale;
            tracklet.voice_variance[dimension] /= scale * scale;
        }
        summary.normalized_voice_dimensions += 1;
    }
    for dimension in 0..CHANNEL_VECTOR_DIMENSIONS {
        let mut values = tracklets
            .iter()
            .filter(|tracklet| tracklet.channel_valid)
            .map(|tracklet| tracklet.channel_mean[dimension])
            .collect::<Vec<_>>();
        if values.is_empty() {
            continue;
        }
        values.sort_by(f32::total_cmp);
        let center = unweighted_quantile(&values, 0.5);
        let q25 = unweighted_quantile(&values, 0.25);
        let q75 = unweighted_quantile(&values, 0.75);
        let mut deviations = values
            .iter()
            .map(|value| (value - center).abs())
            .collect::<Vec<_>>();
        deviations.sort_by(f32::total_cmp);
        let scale = (1.4826 * unweighted_quantile(&deviations, 0.5))
            .max((q75 - q25).abs() / 1.349)
            .max(0.05);
        for tracklet in tracklets
            .iter_mut()
            .filter(|tracklet| tracklet.channel_valid)
        {
            tracklet.channel_mean[dimension] = (tracklet.channel_mean[dimension] - center) / scale;
            tracklet.channel_variance[dimension] /= scale * scale;
        }
        summary.normalized_channel_dimensions += 1;
    }
    summary
}

/// Segment streamed acoustic frames with bounded multiscale Haar-like
/// left/right contrasts.
///
/// Only [`CHANGE_RING_FRAMES`] frames are retained. Emitted output consists of
/// compact sufficient statistics, never raw PCM or a dense CWT.
pub fn segment_acoustic_frames<I, C>(
    frames: I,
    hints: &AcousticBoundaryHints,
    is_cancelled: C,
) -> FwResult<(Vec<AcousticTracklet>, AcousticSegmentationSummary)>
where
    I: IntoIterator<Item = AcousticFrameFeatures>,
    C: FnMut() -> bool,
{
    segment_acoustic_frames_with_schema(
        frames,
        hints,
        AcousticFeatureSchemaVersion::V2,
        is_cancelled,
    )
}

/// Segment frames with an explicitly selected representation.
///
/// Callers must opt into v1 deliberately; v2 remains the only default.
pub fn segment_acoustic_frames_with_schema<I, C>(
    frames: I,
    hints: &AcousticBoundaryHints,
    feature_schema_version: AcousticFeatureSchemaVersion,
    is_cancelled: C,
) -> FwResult<(Vec<AcousticTracklet>, AcousticSegmentationSummary)>
where
    I: IntoIterator<Item = AcousticFrameFeatures>,
    C: FnMut() -> bool,
{
    let feature_ablation = match feature_schema_version {
        AcousticFeatureSchemaVersion::V1 => AcousticFeatureAblation::V1,
        AcousticFeatureSchemaVersion::V2 => AcousticFeatureAblation::FullV2,
    };
    segment_acoustic_frames_with_ablation(frames, hints, feature_ablation, is_cancelled)
}

/// Segment frames with one explicit, frozen representation ablation.
///
/// The production entry point remains on the fixed-safe detector until a
/// hash-locked public development and certification artifact promotes a
/// posterior candidate. Detector comparisons use the explicit ablation API.
pub fn segment_acoustic_frames_with_ablation<I, C>(
    frames: I,
    hints: &AcousticBoundaryHints,
    feature_ablation: AcousticFeatureAblation,
    mut is_cancelled: C,
) -> FwResult<(Vec<AcousticTracklet>, AcousticSegmentationSummary)>
where
    I: IntoIterator<Item = AcousticFrameFeatures>,
    C: FnMut() -> bool,
{
    if is_cancelled() {
        return Err(FwError::Cancelled(
            "acoustic segmentation cancelled before frame zero".to_owned(),
        ));
    }
    let mut segmenter = AcousticSegmenter::new_with_supervised_boundaries(
        hints,
        &[],
        feature_ablation,
        AcousticChangeDetectorMode::FixedSafeV1,
    )?;
    for frame in frames {
        if segmenter.input_frame_count > 0
            && segmenter.input_frame_count % ACOUSTIC_CANCELLATION_INTERVAL_FRAMES == 0
            && is_cancelled()
        {
            return Err(FwError::Cancelled(format!(
                "acoustic segmentation cancelled at frame {}",
                frame.frame_index
            )));
        }
        segmenter.push(frame)?;
    }
    let (tracklets, summary, _) = segmenter.finish()?;
    Ok((tracklets, summary))
}

fn validate_acoustic_frame(frame: &AcousticFrameFeatures) -> FwResult<()> {
    let start_sample = frame
        .frame_index
        .checked_mul(ACOUSTIC_HOP_SAMPLES)
        .ok_or_else(|| {
            FwError::InvalidRequest("acoustic frame index exceeds the v2 cadence range".to_owned())
        })?;
    let end_sample = start_sample
        .checked_add(ACOUSTIC_FRAME_SAMPLES)
        .ok_or_else(|| {
            FwError::InvalidRequest("acoustic frame end exceeds the v2 cadence range".to_owned())
        })?;
    let expected_start_ms = checked_samples_to_ms(start_sample).ok_or_else(|| {
        FwError::InvalidRequest("acoustic frame timestamp exceeds the v2 cadence range".to_owned())
    })?;
    let expected_end_ms = checked_samples_to_ms(end_sample).ok_or_else(|| {
        FwError::InvalidRequest("acoustic frame timestamp exceeds the v2 cadence range".to_owned())
    })?;
    let f0_valid = frame
        .voice
        .f0_hz
        .is_none_or(|value| value.is_finite() && (54.0..=401.0).contains(&value));
    let unit_interval = |value: f32| value.is_finite() && (0.0..=1.0).contains(&value);
    let bounded_scalar = |value: f32| value.is_finite() && value.abs() <= MAX_ABS_ACOUSTIC_FEATURE;
    let bounded_voice = frame
        .voice
        .cepstral_envelope
        .iter()
        .chain(frame.voice.cepstral_delta.iter())
        .chain(frame.voice.cepstral_delta_delta.iter())
        .chain(frame.voice.formant_proxies_hz.iter())
        .copied()
        .all(bounded_scalar);
    let pitch_uncertainty_valid = frame
        .voice
        .pitch_uncertainty_octaves
        .is_none_or(|value| value.is_finite() && (0.0..=2.0).contains(&value));
    let bounded_channel = [
        frame.channel.rms_dbfs,
        frame.channel.dynamics_above_noise_db,
        frame.channel.spectral_centroid_hz,
        frame.channel.spectral_bandwidth_hz,
        frame.channel.spectral_rolloff_hz,
        frame.channel.spectral_flatness,
        frame.channel.spectral_tilt,
        frame.channel.low_band_fraction,
        frame.channel.mid_band_fraction,
        frame.channel.high_band_fraction,
        frame.channel.crest_factor,
        frame.channel.clipping_fraction,
        frame.channel.noise_floor_dbfs,
        frame.channel.spectral_flux,
        frame.channel.distortion_proxy,
        frame.channel.effective_band_limit_hz,
        frame.channel.high_frequency_attenuation,
        frame.channel.reverberation_proxy,
        frame.channel.muffling_proxy,
        frame.channel.stationary_coloration,
    ]
    .into_iter()
    .all(bounded_scalar);
    let band_fraction_sum = frame.channel.low_band_fraction
        + frame.channel.mid_band_fraction
        + frame.channel.high_band_fraction;
    let channel_domains_valid = (-241.0..=0.001).contains(&frame.channel.rms_dbfs)
        && (0.0..=241.0).contains(&frame.channel.dynamics_above_noise_db)
        && (0.0..=8_000.001).contains(&frame.channel.spectral_centroid_hz)
        && (0.0..=8_000.001).contains(&frame.channel.spectral_bandwidth_hz)
        && (0.0..=8_000.001).contains(&frame.channel.spectral_rolloff_hz)
        && (0.0..=1.001).contains(&frame.channel.spectral_flatness)
        && (0.0..=1.001).contains(&frame.channel.low_band_fraction)
        && (0.0..=1.001).contains(&frame.channel.mid_band_fraction)
        && (0.0..=1.001).contains(&frame.channel.high_band_fraction)
        && (0.0..=1.001).contains(&band_fraction_sum)
        && (0.0..=20.001).contains(&frame.channel.crest_factor)
        && (0.0..=1.0).contains(&frame.channel.clipping_fraction)
        && (-241.0..=0.001).contains(&frame.channel.noise_floor_dbfs)
        && (0.0..=1.001).contains(&frame.channel.spectral_flux)
        && (0.0..=1.0).contains(&frame.channel.distortion_proxy)
        && (0.0..=8_000.001).contains(&frame.channel.effective_band_limit_hz)
        && (0.0..=4.0).contains(&frame.channel.high_frequency_attenuation)
        && (0.0..=1.0).contains(&frame.channel.reverberation_proxy)
        && (0.0..=1.0).contains(&frame.channel.muffling_proxy)
        && (0.0..=1.0).contains(&frame.channel.stationary_coloration);
    let quality_consistent = frame.quality.voiced == frame.voice.f0_hz.is_some()
        && frame.voice.pitch_uncertainty_octaves.is_some() == frame.quality.voiced
        && frame.quality.reliable_pitch
            == (frame.quality.voiced
                && frame.voice.voicing_confidence >= 0.55
                && frame.channel.rms_dbfs >= -50.0)
        && frame.quality.low_energy == (frame.channel.rms_dbfs < -55.0)
        && frame.quality.clipped == (frame.channel.clipping_fraction > 0.005)
        && frame.quality.transient == (frame.channel.spectral_flux > 0.35);

    if frame.start_ms != expected_start_ms
        || frame.end_ms != expected_end_ms
        || !bounded_voice
        || !bounded_channel
        || !channel_domains_valid
        || !f0_valid
        || !pitch_uncertainty_valid
        || !unit_interval(frame.voice.voicing_confidence)
        || !unit_interval(frame.voice.harmonicity)
        || !frame.voice.harmonic_to_noise_db.is_finite()
        || !(-20.0..=40.0).contains(&frame.voice.harmonic_to_noise_db)
        || !frame.voice.temporal_modulation.is_finite()
        || frame.voice.temporal_modulation < 0.0
        || !unit_interval(frame.voice.voiced_fraction)
        || !unit_interval(frame.overlap_probability)
        || !quality_consistent
    {
        return Err(FwError::InvalidRequest(
            "acoustic frames must use the exact v2 cadence with finite, internally consistent feature values"
                .to_owned(),
        ));
    }
    Ok(())
}

fn checked_samples_to_ms(samples: usize) -> Option<u64> {
    u64::try_from(samples)
        .ok()?
        .checked_mul(1_000)
        .map(|milliseconds| milliseconds / crate::native_engine::mel::SAMPLE_RATE as u64)
}

fn validate_boundary_hints(hints: &AcousticBoundaryHints) -> FwResult<()> {
    let mut previous_end = 0u64;
    for &(start, end) in &hints.speech_regions_ms {
        if end <= start || start < previous_end {
            return Err(FwError::InvalidRequest(
                "VAD speech regions must be non-empty, ordered, and disjoint".to_owned(),
            ));
        }
        previous_end = end;
    }
    for (name, points) in [
        ("word", hints.word_boundaries_ms.as_slice()),
        ("tiny-diarize", hints.tiny_diarize_boundaries_ms.as_slice()),
    ] {
        if points.windows(2).any(|window| window[0] > window[1]) {
            return Err(FwError::InvalidRequest(format!(
                "{name} boundaries must be ordered"
            )));
        }
    }
    Ok(())
}

fn forced_boundary_map(
    hints: &AcousticBoundaryHints,
    supervised_boundaries_ms: &[u64],
    detector_mode: AcousticChangeDetectorMode,
) -> BTreeMap<usize, ChangePointEvidence> {
    let mut boundaries = BTreeMap::new();
    for &(start_ms, end_ms) in &hints.speech_regions_ms {
        for boundary_ms in [start_ms, end_ms] {
            let frame_index = ms_to_frame(boundary_ms);
            boundaries
                .entry(frame_index)
                .or_insert_with(|| ChangePointEvidence {
                    boundary_ms,
                    voice_distance: 0.0,
                    channel_distance: 0.0,
                    multiscale_scores: [0.0; 5],
                    raw_log_odds: 20.0,
                    change_probability: 1.0,
                    supporting_scale_mask: 0,
                    refinement_offset_frames: 0,
                    action: AcousticChangeAction::EmitBoundary,
                    fallback_reason: None,
                    detector_mode,
                    calibration_id: detector_mode.id(),
                    silence_gap: false,
                    snapped_to_word: false,
                    tiny_diarize_support: false,
                    vad_boundary: true,
                    supervised_boundary: false,
                });
        }
    }
    for &boundary_ms in supervised_boundaries_ms {
        let frame_index = ms_to_frame(boundary_ms);
        boundaries
            .entry(frame_index)
            .and_modify(|evidence| evidence.supervised_boundary = true)
            .or_insert_with(|| ChangePointEvidence {
                boundary_ms,
                voice_distance: 0.0,
                channel_distance: 0.0,
                multiscale_scores: [0.0; 5],
                raw_log_odds: 20.0,
                change_probability: 1.0,
                supporting_scale_mask: 0,
                refinement_offset_frames: 0,
                action: AcousticChangeAction::EmitBoundary,
                fallback_reason: None,
                detector_mode,
                calibration_id: detector_mode.id(),
                silence_gap: false,
                snapped_to_word: false,
                tiny_diarize_support: false,
                vad_boundary: false,
                supervised_boundary: true,
            });
    }
    boundaries
}

fn ms_to_frame(milliseconds: u64) -> usize {
    usize::try_from(milliseconds / 10).unwrap_or(usize::MAX)
}

/// Convert guarded known-speaker intervals into exact frame-split boundaries.
///
/// Acoustic frames overlap by 15 ms in both schemas. The end split therefore moves
/// inward by that overhang so the final enrollment frame remains fully inside
/// the guarded interval instead of reintroducing boundary bleed.
fn supervised_enrollment_boundaries_ms(request: &DiarizationRequest) -> Vec<u64> {
    let hop_ms = samples_to_ms(ACOUSTIC_HOP_SAMPLES);
    let frame_overhang_ms = samples_to_ms(ACOUSTIC_FRAME_SAMPLES).saturating_sub(hop_ms);
    let guard_ms = u64::from(request.enrollment_edge_guard_ms);
    let mut boundaries = Vec::with_capacity(request.known_intervals.len() * 2);
    for hint in &request.known_intervals {
        let guarded_start_ms = hint.start_ms.saturating_add(guard_ms);
        let guarded_end_ms = hint.end_ms.saturating_sub(guard_ms);
        if guarded_start_ms >= guarded_end_ms || guarded_end_ms <= frame_overhang_ms {
            continue;
        }
        let start_frame = guarded_start_ms.div_ceil(hop_ms);
        let end_frame = (guarded_end_ms - frame_overhang_ms) / hop_ms;
        if end_frame <= start_frame {
            continue;
        }
        boundaries.push(start_frame * hop_ms);
        boundaries.push(end_frame * hop_ms);
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
}

#[cfg(test)]
fn multiscale_change_evidence(
    ring: &VecDeque<AcousticFrameFeatures>,
    center: usize,
    hints: &AcousticBoundaryHints,
    feature_ablation: AcousticFeatureAblation,
) -> ChangePointEvidence {
    multiscale_change_evidence_with_detector(
        ring,
        center,
        hints,
        feature_ablation,
        AcousticChangeDetectorMode::CalibratedPosterior,
    )
}

fn multiscale_change_evidence_with_detector(
    ring: &VecDeque<AcousticFrameFeatures>,
    center: usize,
    hints: &AcousticBoundaryHints,
    feature_ablation: AcousticFeatureAblation,
    detector_mode: AcousticChangeDetectorMode,
) -> ChangePointEvidence {
    let calibration = acoustic_change_calibration();
    let mut scores = [0.0_f32; 5];
    let mut voice_evidence = [0.0_f32; 5];
    let mut page_hinkley_evidence = [0.0_f32; 5];
    let mut bayesian_evidence = [0.0_f32; 5];
    let mut channel_evidence = [0.0_f32; 5];
    let mut fallback_scores = [0.0_f32; 5];
    let mut voice_distance = 0.0_f32;
    let mut channel_distance = 0.0_f32;
    let mut supporting_scale_mask = 0_u8;
    let mut invalid_covariance = false;
    for (scale_index, &scale) in CHANGE_SCALES_FRAMES.iter().enumerate() {
        if center < scale || center.saturating_add(scale) > ring.len() {
            continue;
        }
        let scale_evidence =
            variance_aware_scale_evidence(ring, center, scale, feature_ablation, calibration);
        voice_evidence[scale_index] = scale_evidence.voice_evidence;
        page_hinkley_evidence[scale_index] = scale_evidence.page_hinkley_evidence;
        bayesian_evidence[scale_index] = scale_evidence.bayesian_evidence;
        channel_evidence[scale_index] = scale_evidence.channel_evidence;
        fallback_scores[scale_index] =
            0.78 * scale_evidence.voice_distance + 0.22 * scale_evidence.channel_distance.min(1.5);
        voice_distance = voice_distance.max(scale_evidence.voice_distance);
        channel_distance = channel_distance.max(scale_evidence.channel_distance);
        invalid_covariance |= scale_evidence.invalid_covariance;
        if scale_evidence.voice_dimensions >= 3 {
            supporting_scale_mask |= 1 << scale_index;
            scores[scale_index] = match detector_mode {
                AcousticChangeDetectorMode::CalibratedPosterior => {
                    let log_odds = calibration.logit_intercept
                        + calibration.voice_evidence_weight * scale_evidence.voice_evidence
                        + calibration.channel_evidence_weight * scale_evidence.channel_evidence;
                    logistic_probability(log_odds)
                }
                AcousticChangeDetectorMode::PageHinkleyV1 => {
                    let log_odds = calibration.page_hinkley_logit_intercept
                        + calibration.page_hinkley_voice_evidence_weight
                            * scale_evidence.page_hinkley_evidence
                        + calibration.channel_evidence_weight * scale_evidence.channel_evidence;
                    logistic_probability(log_odds)
                }
                AcousticChangeDetectorMode::BayesianTwoRegimeV1 => {
                    let log_odds = calibration.bayesian_logit_intercept
                        + calibration.bayesian_voice_evidence_weight
                            * scale_evidence.bayesian_evidence
                        + calibration.channel_evidence_weight * scale_evidence.channel_evidence;
                    logistic_probability(log_odds)
                }
                AcousticChangeDetectorMode::FixedSafeV1 => {
                    fixed_change_score_probability(fallback_scores[scale_index])
                }
            };
        }
    }
    let valid_scale_count = supporting_scale_mask.count_ones() as usize;
    let fallback_reason = match detector_mode {
        mode if mode.uses_variance_aware_model() && invalid_covariance => {
            Some(AcousticChangeFallbackReason::InvalidCovariance)
        }
        mode if mode.uses_variance_aware_model()
            && valid_scale_count < calibration.minimum_valid_scales =>
        {
            Some(AcousticChangeFallbackReason::InsufficientVoiceSupport)
        }
        AcousticChangeDetectorMode::CalibratedPosterior
        | AcousticChangeDetectorMode::PageHinkleyV1
        | AcousticChangeDetectorMode::BayesianTwoRegimeV1
        | AcousticChangeDetectorMode::FixedSafeV1 => None,
    };

    let selected_voice_evidence = match detector_mode {
        AcousticChangeDetectorMode::CalibratedPosterior => voice_evidence,
        AcousticChangeDetectorMode::PageHinkleyV1 => page_hinkley_evidence,
        AcousticChangeDetectorMode::BayesianTwoRegimeV1 => bayesian_evidence,
        AcousticChangeDetectorMode::FixedSafeV1 => voice_evidence,
    };
    let mut ranked_voice = selected_voice_evidence
        .into_iter()
        .enumerate()
        .filter(|(index, _)| supporting_scale_mask & (1 << index) != 0)
        .map(|(_, value)| value)
        .collect::<Vec<_>>();
    let mut ranked_channel = channel_evidence
        .into_iter()
        .enumerate()
        .filter(|(index, _)| supporting_scale_mask & (1 << index) != 0)
        .map(|(_, value)| value)
        .collect::<Vec<_>>();
    ranked_voice.sort_by(f32::total_cmp);
    ranked_channel.sort_by(f32::total_cmp);
    let fuse_ranked = |values: &[f32]| {
        let top = values.last().copied().unwrap_or(0.0);
        let second = values
            .get(values.len().saturating_sub(2))
            .copied()
            .unwrap_or(top);
        let third = values
            .get(values.len().saturating_sub(3))
            .copied()
            .unwrap_or(second);
        0.55 * top + 0.30 * second + 0.15 * third
    };
    let fused_voice = fuse_ranked(&ranked_voice);
    let fused_channel = fuse_ranked(&ranked_channel);
    let mut ranked_fallback = fallback_scores;
    ranked_fallback.sort_by(f32::total_cmp);
    let mut fixed_fused_score =
        0.55 * ranked_fallback[4] + 0.30 * ranked_fallback[3] + 0.15 * ranked_fallback[2];
    let (mut raw_log_odds, mut change_probability) = match detector_mode {
        mode if mode.uses_variance_aware_model() && fallback_reason.is_some() => {
            let raw_log_odds = fixed_change_score_log_odds(fixed_fused_score);
            (raw_log_odds, logistic_probability(raw_log_odds))
        }
        AcousticChangeDetectorMode::CalibratedPosterior => {
            let raw_log_odds = calibration.logit_intercept
                + calibration.voice_evidence_weight * fused_voice
                + calibration.channel_evidence_weight * fused_channel;
            (raw_log_odds, logistic_probability(raw_log_odds))
        }
        AcousticChangeDetectorMode::PageHinkleyV1 => {
            let raw_log_odds = calibration.page_hinkley_logit_intercept
                + calibration.page_hinkley_voice_evidence_weight * fused_voice
                + calibration.channel_evidence_weight * fused_channel;
            (raw_log_odds, logistic_probability(raw_log_odds))
        }
        AcousticChangeDetectorMode::BayesianTwoRegimeV1 => {
            let raw_log_odds = calibration.bayesian_logit_intercept
                + calibration.bayesian_voice_evidence_weight * fused_voice
                + calibration.channel_evidence_weight * fused_channel;
            (raw_log_odds, logistic_probability(raw_log_odds))
        }
        AcousticChangeDetectorMode::FixedSafeV1 => {
            let raw_log_odds = fixed_change_score_log_odds(fixed_fused_score);
            (raw_log_odds, logistic_probability(raw_log_odds))
        }
    };
    let silence_gap = (center.saturating_sub(10)..center)
        .filter(|&index| ring[index].quality.low_energy)
        .count()
        >= 7
        && (center..center + 10)
            .filter(|&index| !ring[index].quality.low_energy)
            .count()
            >= 7;
    if silence_gap {
        match detector_mode {
            AcousticChangeDetectorMode::CalibratedPosterior
            | AcousticChangeDetectorMode::PageHinkleyV1
            | AcousticChangeDetectorMode::BayesianTwoRegimeV1 => {
                if fallback_reason.is_none() {
                    raw_log_odds += calibration.silence_gap_logit_bonus;
                    change_probability = logistic_probability(raw_log_odds);
                }
            }
            AcousticChangeDetectorMode::FixedSafeV1 => {
                fixed_fused_score = fixed_fused_score.max(0.85);
                raw_log_odds = fixed_change_score_log_odds(fixed_fused_score);
                change_probability = logistic_probability(raw_log_odds);
            }
        }
    }

    let refined_index = match detector_mode {
        AcousticChangeDetectorMode::CalibratedPosterior
        | AcousticChangeDetectorMode::PageHinkleyV1
        | AcousticChangeDetectorMode::BayesianTwoRegimeV1 => {
            refine_boundary_index(ring, center, hints)
        }
        AcousticChangeDetectorMode::FixedSafeV1 => center,
    };
    let refinement_offset_frames =
        i16::try_from(refined_index as isize - center as isize).unwrap_or(0);
    let raw_boundary_ms = ring[refined_index].start_ms;
    let (boundary_ms, snapped_to_word) =
        snap_to_nearest(raw_boundary_ms, &hints.word_boundaries_ms, 80);
    let (tiny_boundary_ms, tiny_diarize_support) =
        snap_to_nearest(boundary_ms, &hints.tiny_diarize_boundaries_ms, 100);
    let boundary_ms = if snapped_to_word {
        boundary_ms
    } else {
        tiny_boundary_ms
    };
    if tiny_diarize_support {
        match detector_mode {
            AcousticChangeDetectorMode::CalibratedPosterior
            | AcousticChangeDetectorMode::PageHinkleyV1
            | AcousticChangeDetectorMode::BayesianTwoRegimeV1 => {
                raw_log_odds += calibration.tiny_diarize_logit_bonus;
                change_probability = logistic_probability(raw_log_odds);
            }
            AcousticChangeDetectorMode::FixedSafeV1 => {
                fixed_fused_score = (fixed_fused_score + 0.10).min(1.0);
                raw_log_odds = fixed_change_score_log_odds(fixed_fused_score);
                change_probability = logistic_probability(raw_log_odds);
            }
        }
    }
    ChangePointEvidence {
        boundary_ms,
        voice_distance,
        channel_distance,
        multiscale_scores: scores,
        raw_log_odds,
        change_probability,
        supporting_scale_mask,
        refinement_offset_frames,
        action: AcousticChangeAction::NoBoundary,
        fallback_reason,
        detector_mode,
        calibration_id: detector_mode.id(),
        silence_gap,
        snapped_to_word,
        tiny_diarize_support,
        vad_boundary: false,
        supervised_boundary: false,
    }
}

#[derive(Debug, Clone, Copy)]
struct DiagonalMoments<const N: usize> {
    count: [u32; N],
    mean: [f32; N],
    m2: [f32; N],
}

impl<const N: usize> Default for DiagonalMoments<N> {
    fn default() -> Self {
        Self {
            count: [0; N],
            mean: [0.0; N],
            m2: [0.0; N],
        }
    }
}

impl<const N: usize> DiagonalMoments<N> {
    fn push_masked(&mut self, values: &[f32; N], valid: &[bool; N], dimensions: usize) {
        for dimension in 0..dimensions.min(N) {
            if !valid[dimension] {
                continue;
            }
            self.count[dimension] = self.count[dimension].saturating_add(1);
            let count = self.count[dimension] as f32;
            let delta = values[dimension] - self.mean[dimension];
            self.mean[dimension] += delta / count;
            self.m2[dimension] += delta * (values[dimension] - self.mean[dimension]);
        }
    }

    fn push_prefix(&mut self, values: &[f32; N], dimensions: usize) {
        let valid = [true; N];
        self.push_masked(values, &valid, dimensions);
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct VarianceAwareScaleEvidence {
    voice_evidence: f32,
    page_hinkley_evidence: f32,
    bayesian_evidence: f32,
    channel_evidence: f32,
    voice_distance: f32,
    channel_distance: f32,
    voice_dimensions: usize,
    invalid_covariance: bool,
}

fn variance_aware_scale_evidence(
    ring: &VecDeque<AcousticFrameFeatures>,
    center: usize,
    scale: usize,
    feature_ablation: AcousticFeatureAblation,
    calibration: AcousticChangeCalibration,
) -> VarianceAwareScaleEvidence {
    let schema = acoustic_feature_schema(feature_ablation.schema_version());
    let mut left_voice = DiagonalMoments::<VOICE_VECTOR_DIMENSIONS>::default();
    let mut right_voice = DiagonalMoments::<VOICE_VECTOR_DIMENSIONS>::default();
    let mut left_channel = DiagonalMoments::<CHANNEL_VECTOR_DIMENSIONS>::default();
    let mut right_channel = DiagonalMoments::<CHANNEL_VECTOR_DIMENSIONS>::default();
    for frame in ring.iter().take(center).skip(center - scale) {
        let compact = compact_vectors_for_ablation(frame, feature_ablation);
        if compact.channel_valid {
            left_channel.push_prefix(&compact.channel, schema.channel_dimensions);
        }
        left_voice.push_masked(
            &compact.voice,
            &compact.voice_valid,
            schema.voice_dimensions,
        );
    }
    for frame in ring.iter().skip(center).take(scale) {
        let compact = compact_vectors_for_ablation(frame, feature_ablation);
        if compact.channel_valid {
            right_channel.push_prefix(&compact.channel, schema.channel_dimensions);
        }
        right_voice.push_masked(
            &compact.voice,
            &compact.voice_valid,
            schema.voice_dimensions,
        );
    }
    let voice = diagonal_glr_evidence(
        &left_voice,
        &right_voice,
        schema.voice_dimensions,
        calibration,
    );
    let channel = diagonal_glr_evidence(
        &left_channel,
        &right_channel,
        schema.channel_dimensions,
        calibration,
    );
    let page_hinkley = diagonal_page_hinkley_evidence(
        &left_voice,
        &right_voice,
        schema.voice_dimensions,
        calibration,
    );
    let bayesian = diagonal_bayesian_evidence(
        &left_voice,
        &right_voice,
        schema.voice_dimensions,
        calibration,
    );
    VarianceAwareScaleEvidence {
        voice_evidence: voice.evidence,
        page_hinkley_evidence: page_hinkley.evidence,
        bayesian_evidence: bayesian.evidence,
        channel_evidence: channel.evidence,
        voice_distance: voice.distance,
        channel_distance: channel.distance,
        voice_dimensions: voice.valid_dimensions,
        invalid_covariance: voice.invalid_covariance || channel.invalid_covariance,
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct DiagonalGlrEvidence {
    evidence: f32,
    distance: f32,
    valid_dimensions: usize,
    invalid_covariance: bool,
}

fn diagonal_glr_evidence<const N: usize>(
    left: &DiagonalMoments<N>,
    right: &DiagonalMoments<N>,
    dimensions: usize,
    calibration: AcousticChangeCalibration,
) -> DiagonalGlrEvidence {
    let dimensions = dimensions.min(N);
    let mut pooled_variances = Vec::with_capacity(dimensions);
    let mut usable = [false; N];
    for (dimension, usable_dimension) in usable.iter_mut().enumerate().take(dimensions) {
        let left_count = left.count[dimension];
        let right_count = right.count[dimension];
        if left_count < 3 || right_count < 3 {
            continue;
        }
        let degrees_of_freedom = left_count.saturating_add(right_count).saturating_sub(2);
        if degrees_of_freedom == 0 {
            continue;
        }
        let pooled = (left.m2[dimension] + right.m2[dimension]) / degrees_of_freedom as f32;
        if !pooled.is_finite() || pooled < 0.0 {
            return DiagonalGlrEvidence {
                invalid_covariance: true,
                ..DiagonalGlrEvidence::default()
            };
        }
        *usable_dimension = true;
        pooled_variances.push(pooled);
    }
    if pooled_variances.is_empty() {
        return DiagonalGlrEvidence::default();
    }
    pooled_variances.sort_by(f32::total_cmp);
    let shrinkage_target = unweighted_quantile(&pooled_variances, 0.5);
    let mut glr_dimensions = Vec::with_capacity(pooled_variances.len());
    let mut squared_distance = 0.0_f32;
    for (dimension, is_usable) in usable.iter().copied().enumerate().take(dimensions) {
        if !is_usable {
            continue;
        }
        let left_count = left.count[dimension] as f32;
        let right_count = right.count[dimension] as f32;
        let pooled = (left.m2[dimension] + right.m2[dimension]) / (left_count + right_count - 2.0);
        let shrunk_variance = ((1.0 - calibration.diagonal_shrinkage) * pooled
            + calibration.diagonal_shrinkage * shrinkage_target)
            .max(calibration.variance_floor);
        let difference = left.mean[dimension] - right.mean[dimension];
        squared_distance += difference * difference;
        let effective_count = (left_count * right_count / (left_count + right_count)).min(32.0);
        let glr = 0.5 * effective_count * (difference * difference / shrunk_variance).ln_1p();
        if !glr.is_finite() || glr < 0.0 {
            return DiagonalGlrEvidence {
                invalid_covariance: true,
                ..DiagonalGlrEvidence::default()
            };
        }
        glr_dimensions.push(glr);
    }
    glr_dimensions.sort_by(f32::total_cmp);
    let median = unweighted_quantile(&glr_dimensions, 0.5);
    let upper_quartile = unweighted_quantile(&glr_dimensions, 0.75);
    let aggregate = 0.4 * median + 0.6 * upper_quartile;
    DiagonalGlrEvidence {
        evidence: (aggregate.ln_1p() / 3.0).clamp(0.0, 1.0),
        distance: (squared_distance / glr_dimensions.len() as f32).sqrt(),
        valid_dimensions: glr_dimensions.len(),
        invalid_covariance: false,
    }
}

/// Terminal two-sided Page-Hinkley approximation over a bounded window.
///
/// The left and right sufficient statistics are the streaming state. For each
/// supported voice coordinate, the statistic is the terminal standardized
/// cumulative shift after the declared allowance. Robust median/upper-quartile
/// fusion prevents one unstable coordinate from deciding the boundary.
fn diagonal_page_hinkley_evidence<const N: usize>(
    left: &DiagonalMoments<N>,
    right: &DiagonalMoments<N>,
    dimensions: usize,
    calibration: AcousticChangeCalibration,
) -> DiagonalGlrEvidence {
    let dimensions = dimensions.min(N);
    let mut pooled_variances = Vec::with_capacity(dimensions);
    let mut usable = [false; N];
    for (dimension, usable_dimension) in usable.iter_mut().enumerate().take(dimensions) {
        let left_count = left.count[dimension];
        let right_count = right.count[dimension];
        if left_count < 3 || right_count < 3 {
            continue;
        }
        let degrees_of_freedom = left_count.saturating_add(right_count).saturating_sub(2);
        if degrees_of_freedom == 0 {
            continue;
        }
        let pooled = (left.m2[dimension] + right.m2[dimension]) / degrees_of_freedom as f32;
        if !pooled.is_finite() || pooled < 0.0 {
            return DiagonalGlrEvidence {
                invalid_covariance: true,
                ..DiagonalGlrEvidence::default()
            };
        }
        *usable_dimension = true;
        pooled_variances.push(pooled);
    }
    if pooled_variances.is_empty() {
        return DiagonalGlrEvidence::default();
    }
    pooled_variances.sort_by(f32::total_cmp);
    let shrinkage_target = unweighted_quantile(&pooled_variances, 0.5);
    let mut statistics = Vec::with_capacity(pooled_variances.len());
    let mut squared_distance = 0.0_f32;
    for (dimension, is_usable) in usable.iter().copied().enumerate().take(dimensions) {
        if !is_usable {
            continue;
        }
        let left_count = left.count[dimension] as f32;
        let right_count = right.count[dimension] as f32;
        let pooled = (left.m2[dimension] + right.m2[dimension]) / (left_count + right_count - 2.0);
        let shrunk_variance = ((1.0 - calibration.diagonal_shrinkage) * pooled
            + calibration.diagonal_shrinkage * shrinkage_target)
            .max(calibration.variance_floor);
        let difference = left.mean[dimension] - right.mean[dimension];
        squared_distance += difference * difference;
        let effective_count = (left_count * right_count / (left_count + right_count)).min(32.0);
        let standardized_shift = difference.abs() / shrunk_variance.sqrt();
        let statistic = effective_count.sqrt()
            * (standardized_shift - calibration.page_hinkley_allowance).max(0.0);
        if !statistic.is_finite() || statistic < 0.0 {
            return DiagonalGlrEvidence {
                invalid_covariance: true,
                ..DiagonalGlrEvidence::default()
            };
        }
        statistics.push(statistic);
    }
    statistics.sort_by(f32::total_cmp);
    let median = unweighted_quantile(&statistics, 0.5);
    let upper_quartile = unweighted_quantile(&statistics, 0.75);
    let aggregate = 0.4 * median + 0.6 * upper_quartile;
    DiagonalGlrEvidence {
        evidence: (aggregate.ln_1p() / 4.0).clamp(0.0, 1.0),
        distance: (squared_distance / statistics.len() as f32).sqrt(),
        valid_dimensions: statistics.len(),
        invalid_covariance: false,
    }
}

/// Bounded diagonal two-regime Bayes-factor approximation.
///
/// The known-variance likelihood gain is penalized by the additional mean
/// parameter's BIC/Occam term. Only left/right sufficient statistics are
/// retained, so memory remains fixed independently of call duration.
fn diagonal_bayesian_evidence<const N: usize>(
    left: &DiagonalMoments<N>,
    right: &DiagonalMoments<N>,
    dimensions: usize,
    calibration: AcousticChangeCalibration,
) -> DiagonalGlrEvidence {
    let dimensions = dimensions.min(N);
    let mut pooled_variances = Vec::with_capacity(dimensions);
    let mut usable = [false; N];
    for (dimension, usable_dimension) in usable.iter_mut().enumerate().take(dimensions) {
        let left_count = left.count[dimension];
        let right_count = right.count[dimension];
        if left_count < 3 || right_count < 3 {
            continue;
        }
        let degrees_of_freedom = left_count.saturating_add(right_count).saturating_sub(2);
        if degrees_of_freedom == 0 {
            continue;
        }
        let pooled = (left.m2[dimension] + right.m2[dimension]) / degrees_of_freedom as f32;
        if !pooled.is_finite() || pooled < 0.0 {
            return DiagonalGlrEvidence {
                invalid_covariance: true,
                ..DiagonalGlrEvidence::default()
            };
        }
        *usable_dimension = true;
        pooled_variances.push(pooled);
    }
    if pooled_variances.is_empty() {
        return DiagonalGlrEvidence::default();
    }
    pooled_variances.sort_by(f32::total_cmp);
    let shrinkage_target = unweighted_quantile(&pooled_variances, 0.5);
    let mut log_bayes_factors = Vec::with_capacity(pooled_variances.len());
    let mut squared_distance = 0.0_f32;
    for (dimension, is_usable) in usable.iter().copied().enumerate().take(dimensions) {
        if !is_usable {
            continue;
        }
        let left_count = left.count[dimension] as f32;
        let right_count = right.count[dimension] as f32;
        let pooled = (left.m2[dimension] + right.m2[dimension]) / (left_count + right_count - 2.0);
        let shrunk_variance = ((1.0 - calibration.diagonal_shrinkage) * pooled
            + calibration.diagonal_shrinkage * shrinkage_target)
            .max(calibration.variance_floor);
        let difference = left.mean[dimension] - right.mean[dimension];
        squared_distance += difference * difference;
        let effective_count = (left_count * right_count / (left_count + right_count)).min(32.0);
        let likelihood_gain = 0.5 * effective_count * difference * difference / shrunk_variance;
        let occam_penalty =
            0.5 * calibration.bayesian_occam_weight * (left_count + right_count).ln();
        let log_bayes_factor = (likelihood_gain - occam_penalty).max(0.0);
        if !log_bayes_factor.is_finite() {
            return DiagonalGlrEvidence {
                invalid_covariance: true,
                ..DiagonalGlrEvidence::default()
            };
        }
        log_bayes_factors.push(log_bayes_factor);
    }
    log_bayes_factors.sort_by(f32::total_cmp);
    let median = unweighted_quantile(&log_bayes_factors, 0.5);
    let upper_quartile = unweighted_quantile(&log_bayes_factors, 0.75);
    let aggregate = 0.4 * median + 0.6 * upper_quartile;
    DiagonalGlrEvidence {
        evidence: (aggregate.ln_1p() / 5.0).clamp(0.0, 1.0),
        distance: (squared_distance / log_bayes_factors.len() as f32).sqrt(),
        valid_dimensions: log_bayes_factors.len(),
        invalid_covariance: false,
    }
}

fn refine_boundary_index(
    ring: &VecDeque<AcousticFrameFeatures>,
    center: usize,
    hints: &AcousticBoundaryHints,
) -> usize {
    let radius = acoustic_change_calibration().refinement_radius_frames;
    let start = center.saturating_sub(radius).max(3);
    let end = center
        .saturating_add(radius)
        .min(ring.len().saturating_sub(4));
    (start..=end)
        .map(|candidate| {
            let mean_range =
                |range_start: usize, range_end: usize, value: fn(&AcousticFrameFeatures) -> f32| {
                    ring.iter()
                        .skip(range_start)
                        .take(range_end - range_start)
                        .map(value)
                        .sum::<f32>()
                        / (range_end - range_start) as f32
                };
            let voicing_jump = (mean_range(candidate - 3, candidate, |frame| {
                frame.voice.voicing_confidence
            }) - mean_range(candidate, candidate + 3, |frame| {
                frame.voice.voicing_confidence
            }))
            .abs();
            let pitch_jump = {
                let reliable_mean = |range_start: usize, range_end: usize| {
                    let mut count = 0_u32;
                    let total = ring
                        .iter()
                        .skip(range_start)
                        .take(range_end - range_start)
                        .filter_map(|frame| {
                            frame
                                .quality
                                .reliable_pitch
                                .then_some(frame.voice.f0_hz)
                                .flatten()
                                .map(|pitch| {
                                    count += 1;
                                    pitch.ln()
                                })
                        })
                        .sum::<f32>();
                    (count > 0).then_some(total / count as f32)
                };
                reliable_mean(candidate - 3, candidate)
                    .zip(reliable_mean(candidate, candidate + 3))
                    .map_or(0.0, |(left_pitch, right_pitch)| {
                        (left_pitch - right_pitch).abs().min(1.0)
                    })
            };
            let neighbor_rms =
                0.5 * (ring[candidate - 1].channel.rms_dbfs + ring[candidate + 1].channel.rms_dbfs);
            let energy_valley =
                ((neighbor_rms - ring[candidate].channel.rms_dbfs) / 12.0).clamp(0.0, 1.0);
            let boundary_ms = ring[candidate].start_ms;
            let word_support = hints
                .word_boundaries_ms
                .iter()
                .any(|timestamp| timestamp.abs_diff(boundary_ms) <= 30);
            let tiny_support = hints
                .tiny_diarize_boundaries_ms
                .iter()
                .any(|timestamp| timestamp.abs_diff(boundary_ms) <= 40);
            let score = 0.30 * ring[candidate].channel.spectral_flux
                + 0.25 * voicing_jump
                + 0.25 * pitch_jump
                + 0.10 * energy_valley
                + if word_support { 0.04 } else { 0.0 }
                + if tiny_support { 0.06 } else { 0.0 };
            (score, center.abs_diff(candidate), candidate)
        })
        .max_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| right.1.cmp(&left.1))
                .then_with(|| right.2.cmp(&left.2))
        })
        .map_or(center, |(_, _, candidate)| candidate)
}

fn logistic_probability(log_odds: f32) -> f32 {
    if log_odds >= 0.0 {
        1.0 / (1.0 + (-log_odds).exp())
    } else {
        let exponential = log_odds.exp();
        exponential / (1.0 + exponential)
    }
}

fn fixed_change_score_log_odds(score: f32) -> f32 {
    let threshold = acoustic_change_calibration().decision_probability;
    let threshold_log_odds = (threshold / (1.0 - threshold)).ln();
    threshold_log_odds + 10.0 * (score - CHANGE_FALLBACK_DISTANCE_THRESHOLD)
}

fn fixed_change_score_probability(score: f32) -> f32 {
    logistic_probability(fixed_change_score_log_odds(score))
}

/// Apply the runtime detector's deterministic peak suppression at an alternate
/// probability threshold.
///
/// This is evaluation-only support for development threshold sweeps. It
/// consumes the bounded public-corpus score stream and returns acoustic
/// candidates only; it never changes the production operating point.
pub(crate) fn select_acoustic_change_evidence_at_threshold(
    evaluated: &[ChangePointEvidence],
    threshold: f32,
) -> FwResult<Vec<ChangePointEvidence>> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(FwError::InvalidRequest(
            "speaker-change threshold must be finite and within [0, 1]".to_owned(),
        ));
    }
    let mut selected = BTreeMap::<usize, ChangePointEvidence>::new();
    let detector_mode = evaluated.first().map_or(
        AcousticChangeDetectorMode::CalibratedPosterior,
        |evidence| evidence.detector_mode,
    );
    if evaluated
        .iter()
        .any(|evidence| evidence.detector_mode != detector_mode)
    {
        return Err(FwError::InvalidRequest(
            "speaker-change evaluation stream mixes detector modes".to_owned(),
        ));
    }
    let mut selector = ChangePeakSelector::new(detector_mode, threshold);
    for (frame_index, evidence) in evaluated.iter().enumerate() {
        if !evidence.change_probability.is_finite() {
            return Err(FwError::InvalidRequest(
                "speaker-change evaluation probability must be finite".to_owned(),
            ));
        }
        if evidence.vad_boundary || evidence.supervised_boundary {
            continue;
        }
        if let Some(peak) = selector.push(frame_index, evidence.clone()) {
            insert_detected_boundary(&mut selected, peak);
        }
    }
    if let Some(peak) = selector.finish() {
        insert_detected_boundary(&mut selected, peak);
    }
    Ok(selected.into_values().collect())
}

#[derive(Debug, Clone, Copy)]
struct CompactFeatureVector {
    voice: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_valid: [bool; VOICE_VECTOR_DIMENSIONS],
    channel: [f32; CHANNEL_VECTOR_DIMENSIONS],
    channel_valid: bool,
    identity_quality: f32,
}

#[cfg(test)]
fn compact_vectors_for_schema(
    frame: &AcousticFrameFeatures,
    version: AcousticFeatureSchemaVersion,
) -> CompactFeatureVector {
    let ablation = match version {
        AcousticFeatureSchemaVersion::V1 => AcousticFeatureAblation::V1,
        AcousticFeatureSchemaVersion::V2 => AcousticFeatureAblation::FullV2,
    };
    compact_vectors_for_ablation(frame, ablation)
}

fn compact_vectors_for_ablation(
    frame: &AcousticFrameFeatures,
    ablation: AcousticFeatureAblation,
) -> CompactFeatureVector {
    let version = ablation.schema_version();
    let mut voice = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    let mut voice_valid = [false; VOICE_VECTOR_DIMENSIONS];
    voice[..CEPSTRAL_COEFFICIENTS].copy_from_slice(&frame.voice.cepstral_envelope);
    let high_information = frame.quality.voiced
        && !frame.quality.low_energy
        && !frame.quality.clipped
        && !frame.quality.transient;
    match version {
        AcousticFeatureSchemaVersion::V1 => {
            voice_valid[..6].fill(!frame.quality.low_energy);
            if let Some(pitch) = frame.voice.f0_hz {
                voice[6] = pitch.ln() / 6.0;
            }
            voice_valid[6] = frame.quality.reliable_pitch;
            voice[7] = frame.voice.harmonicity;
            voice_valid[7] = !frame.quality.low_energy;
        }
        AcousticFeatureSchemaVersion::V2 => {
            voice_valid[..12].fill(high_information);
            voice[12..16].copy_from_slice(&frame.voice.cepstral_delta[..4]);
            voice_valid[12..16].fill(high_information && frame.frame_index > 0);
            voice[16..20].copy_from_slice(&frame.voice.cepstral_delta_delta[..4]);
            voice_valid[16..20].fill(high_information && frame.frame_index > 1);
            if let Some(pitch) = frame.voice.f0_hz {
                voice[20] = pitch.ln();
            }
            voice_valid[20] = high_information && frame.quality.reliable_pitch;
            voice[21] = frame.voice.harmonicity;
            voice[22] = frame.voice.harmonic_to_noise_db / 40.0;
            voice_valid[21..23].fill(high_information);
            for (output, input) in voice[23..26].iter_mut().zip(frame.voice.formant_proxies_hz) {
                *output = input / 8_000.0;
            }
            voice_valid[23..26].fill(high_information);
            voice[26] = frame.voice.pitch_uncertainty_octaves.unwrap_or(0.0);
            voice_valid[26] = high_information && frame.quality.reliable_pitch;
            voice[27] = frame.voice.temporal_modulation;
            voice_valid[27] = high_information && frame.frame_index > 0;
        }
    }
    let mut channel = [0.0_f32; CHANNEL_VECTOR_DIMENSIONS];
    match version {
        AcousticFeatureSchemaVersion::V1 => {
            channel[..8].copy_from_slice(&[
                frame.channel.rms_dbfs / 40.0,
                frame.channel.spectral_centroid_hz / 8_000.0,
                frame.channel.spectral_bandwidth_hz / 8_000.0,
                frame.channel.spectral_flatness,
                frame.channel.spectral_tilt / 10.0,
                frame.channel.low_band_fraction,
                frame.channel.mid_band_fraction,
                frame.channel.high_band_fraction,
            ]);
        }
        AcousticFeatureSchemaVersion::V2 => {
            channel = [
                frame.channel.rms_dbfs / 40.0,
                frame.channel.noise_floor_dbfs / 40.0,
                frame.channel.dynamics_above_noise_db / 40.0,
                frame.channel.spectral_centroid_hz / 8_000.0,
                frame.channel.spectral_bandwidth_hz / 8_000.0,
                frame.channel.spectral_rolloff_hz / 8_000.0,
                frame.channel.effective_band_limit_hz / 8_000.0,
                frame.channel.high_frequency_attenuation,
                frame.channel.muffling_proxy,
                frame.channel.reverberation_proxy,
                frame.channel.spectral_tilt / 10.0,
                frame.channel.clipping_fraction,
                frame.channel.distortion_proxy,
                frame.channel.stationary_coloration,
            ];
        }
    }
    if ablation == AcousticFeatureAblation::NoPitch {
        voice_valid[20] = false;
        voice_valid[26] = false;
    } else if ablation == AcousticFeatureAblation::NoDeltas {
        voice_valid[12..20].fill(false);
    } else if ablation == AcousticFeatureAblation::NoModulation {
        voice_valid[27] = false;
    }
    let channel_valid = ablation != AcousticFeatureAblation::NoChannel
        && !frame.quality.low_energy
        && !frame.quality.clipped;
    CompactFeatureVector {
        voice,
        voice_valid,
        channel,
        channel_valid,
        identity_quality: if high_information {
            (0.65 * frame.voice.voicing_confidence
                + 0.20 * (frame.channel.dynamics_above_noise_db / 40.0).clamp(0.0, 1.0)
                + 0.15 * (1.0 - frame.channel.distortion_proxy))
                .clamp(0.0, 1.0)
        } else {
            0.0
        },
    }
}

fn scale_vector<const N: usize>(vector: &mut [f32; N], scale: f32) {
    for value in vector {
        *value *= scale;
    }
}

#[cfg(test)]
fn euclidean_distance<const N: usize>(left: &[f32; N], right: &[f32; N]) -> f32 {
    euclidean_distance_prefix(left, right, N)
}

fn euclidean_distance_prefix<const N: usize>(
    left: &[f32; N],
    right: &[f32; N],
    dimensions: usize,
) -> f32 {
    let dimensions = dimensions.clamp(1, N);
    (left
        .iter()
        .zip(right)
        .take(dimensions)
        .map(|(&left, &right)| {
            let difference = left - right;
            difference * difference
        })
        .sum::<f32>()
        / dimensions as f32)
        .sqrt()
}

fn masked_euclidean_distance<const N: usize>(
    left: &[f32; N],
    left_valid: &[bool; N],
    right: &[f32; N],
    right_valid: &[bool; N],
) -> Option<f32> {
    let mut squared_sum = 0.0_f32;
    let mut active = 0usize;
    for index in 0..N {
        if left_valid[index] && right_valid[index] {
            let difference = left[index] - right[index];
            squared_sum += difference * difference;
            active += 1;
        }
    }
    (active > 0).then(|| (squared_sum / active as f32).sqrt())
}

fn snap_to_nearest(value: u64, candidates: &[u64], tolerance_ms: u64) -> (u64, bool) {
    candidates
        .iter()
        .copied()
        .filter_map(|candidate| {
            let distance = candidate.abs_diff(value);
            (distance <= tolerance_ms).then_some((distance, candidate))
        })
        .min()
        .map_or((value, false), |(_, candidate)| (candidate, true))
}

fn insert_detected_boundary(
    boundaries: &mut BTreeMap<usize, ChangePointEvidence>,
    mut evidence: ChangePointEvidence,
) {
    if evidence.fallback_reason.is_none() {
        evidence.action = AcousticChangeAction::EmitBoundary;
    }
    let frame_index = ms_to_frame(evidence.boundary_ms);
    boundaries
        .entry(frame_index)
        .and_modify(|current| {
            if evidence.change_probability > current.change_probability {
                *current = evidence.clone();
            }
        })
        .or_insert(evidence);
}

#[derive(Debug, Default)]
struct TrackletAccumulator {
    start_ms: Option<u64>,
    end_ms: u64,
    frame_count: usize,
    voiced_frame_count: usize,
    identity_frame_count: usize,
    identity_candidates: Vec<CompactFeatureVector>,
    channel_mean: [f32; CHANNEL_VECTOR_DIMENSIONS],
    channel_m2: [f32; CHANNEL_VECTOR_DIMENSIONS],
    channel_frame_count: usize,
    channel_dimensions: usize,
    overlap_probability_sum: f32,
    overlap_evidence_frame_count: usize,
}

impl TrackletAccumulator {
    fn push(&mut self, frame: &AcousticFrameFeatures, feature_ablation: AcousticFeatureAblation) {
        self.start_ms.get_or_insert(frame.start_ms);
        self.end_ms = frame.end_ms;
        self.frame_count += 1;
        self.voiced_frame_count += usize::from(u8::from(frame.quality.voiced));
        if !frame.quality.low_energy && !frame.quality.clipped && !frame.quality.transient {
            self.overlap_probability_sum += frame.overlap_probability;
            self.overlap_evidence_frame_count += 1;
        }
        let compact = compact_vectors_for_ablation(frame, feature_ablation);
        if compact.channel_valid {
            self.channel_dimensions =
                acoustic_feature_schema(feature_ablation.schema_version()).channel_dimensions;
        }
        if compact.identity_quality > 0.0 {
            self.identity_frame_count += 1;
            if self.identity_candidates.len() < MAX_IDENTITY_SUBWINDOWS {
                self.identity_candidates.push(compact);
            } else if let Some((lowest_index, lowest)) = self
                .identity_candidates
                .iter()
                .enumerate()
                .min_by(|left, right| {
                    left.1
                        .identity_quality
                        .total_cmp(&right.1.identity_quality)
                        .then(left.0.cmp(&right.0))
                })
                && compact.identity_quality > lowest.identity_quality
            {
                self.identity_candidates[lowest_index] = compact;
            }
        }
        if compact.channel_valid {
            self.channel_frame_count += 1;
            welford_update(
                &mut self.channel_mean,
                &mut self.channel_m2,
                &compact.channel,
                self.channel_frame_count,
            );
        }
    }

    fn finish(
        &mut self,
        tracklet_index: usize,
        boundary_evidence: Option<ChangePointEvidence>,
    ) -> Option<AcousticTracklet> {
        let start_ms = self.start_ms?;
        let (voice_mean, voice_variance, voice_valid, voice_support) =
            robust_identity_statistics(&self.identity_candidates);
        let denominator = self.channel_frame_count.saturating_sub(1).max(1) as f32;
        let mut channel_variance = self.channel_m2;
        scale_vector(&mut channel_variance, 1.0 / denominator);
        let overlap_probability = if self.overlap_evidence_frame_count == 0 {
            0.0
        } else {
            self.overlap_probability_sum / self.overlap_evidence_frame_count as f32
        };
        let tracklet = AcousticTracklet {
            tracklet_index,
            start_ms,
            end_ms: self.end_ms,
            frame_count: self.frame_count,
            voiced_frame_count: self.voiced_frame_count,
            identity_frame_count: self.identity_frame_count,
            channel_frame_count: self.channel_frame_count,
            voice_mean,
            voice_variance,
            voice_valid,
            voice_support,
            channel_mean: self.channel_mean,
            channel_variance,
            channel_valid: self.channel_frame_count > 0,
            channel_dimensions: self.channel_dimensions,
            change_confidence: boundary_evidence
                .as_ref()
                .map_or(0.0, |evidence| evidence.change_probability),
            overlap_probability,
            overlap_suspected: self.overlap_evidence_frame_count >= 3
                && overlap_probability >= 0.55,
            boundary_evidence,
        };
        *self = Self::default();
        Some(tracklet)
    }
}

fn robust_identity_statistics(
    observations: &[CompactFeatureVector],
) -> (
    [f32; VOICE_VECTOR_DIMENSIONS],
    [f32; VOICE_VECTOR_DIMENSIONS],
    [bool; VOICE_VECTOR_DIMENSIONS],
    [u32; VOICE_VECTOR_DIMENSIONS],
) {
    let mut location = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    let mut variance = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    let mut valid = [false; VOICE_VECTOR_DIMENSIONS];
    let mut support = [0_u32; VOICE_VECTOR_DIMENSIONS];
    for dimension in 0..VOICE_VECTOR_DIMENSIONS {
        let mut values = observations
            .iter()
            .filter(|observation| observation.voice_valid[dimension])
            .map(|observation| observation.voice[dimension])
            .collect::<Vec<_>>();
        if values.is_empty() {
            continue;
        }
        values.sort_by(f32::total_cmp);
        let median = unweighted_quantile(&values, 0.5);
        let mut deviations = values
            .iter()
            .map(|value| (value - median).abs())
            .collect::<Vec<_>>();
        deviations.sort_by(f32::total_cmp);
        let mad = unweighted_quantile(&deviations, 0.5);
        location[dimension] = median;
        variance[dimension] = (1.4826 * mad).powi(2);
        valid[dimension] = true;
        support[dimension] = u32::try_from(values.len()).unwrap_or(u32::MAX);
    }
    (location, variance, valid, support)
}

fn unweighted_quantile(sorted: &[f32], quantile: f32) -> f32 {
    let last = sorted.len().saturating_sub(1);
    let index = (quantile.clamp(0.0, 1.0) * last as f32).round() as usize;
    sorted[index.min(last)]
}

fn welford_update<const N: usize>(
    mean: &mut [f32; N],
    m2: &mut [f32; N],
    sample: &[f32; N],
    count: usize,
) {
    let count = count as f32;
    for index in 0..N {
        let delta = sample[index] - mean[index];
        mean[index] += delta / count;
        let delta_after = sample[index] - mean[index];
        m2[index] += delta * delta_after;
    }
}

fn consume_segment_frame(
    frame: AcousticFrameFeatures,
    accumulator: &mut TrackletAccumulator,
    tracklets: &mut Vec<AcousticTracklet>,
    detected: &mut BTreeMap<usize, ChangePointEvidence>,
    forced: &mut BTreeMap<usize, ChangePointEvidence>,
    feature_ablation: AcousticFeatureAblation,
) {
    let evidence = forced
        .remove(&frame.frame_index)
        .or_else(|| detected.remove(&frame.frame_index));
    if let Some(evidence) = evidence {
        let must_preserve_boundary = evidence.vad_boundary || evidence.supervised_boundary;
        let long_enough = accumulator.frame_count >= MIN_TRACKLET_FRAMES;
        if (must_preserve_boundary || long_enough)
            && let Some(tracklet) = accumulator.finish(tracklets.len(), Some(evidence))
        {
            tracklets.push(tracklet);
        }
    }
    accumulator.push(&frame, feature_ablation);
}

fn merge_compatible_adjacent_tracklets(tracklets: &mut Vec<AcousticTracklet>) {
    if tracklets.len() < 2 {
        return;
    }
    let mut merged = Vec::with_capacity(tracklets.len());
    for tracklet in tracklets.drain(..) {
        let compatible = merged.last().is_some_and(|previous: &AcousticTracklet| {
            tracklet.start_ms <= previous.end_ms.saturating_add(50)
                && previous.change_confidence < acoustic_change_calibration().decision_probability
                && !previous
                    .boundary_evidence
                    .as_ref()
                    .is_some_and(|evidence| evidence.vad_boundary || evidence.supervised_boundary)
                && masked_euclidean_distance(
                    &previous.voice_mean,
                    &previous.voice_valid,
                    &tracklet.voice_mean,
                    &tracklet.voice_valid,
                )
                .is_some_and(|distance| distance < 0.08)
        });
        if compatible {
            if let Some(previous) = merged.last_mut() {
                merge_tracklet_statistics(previous, &tracklet);
            }
        } else {
            merged.push(tracklet);
        }
    }
    *tracklets = merged;
}

fn merge_tracklet_statistics(destination: &mut AcousticTracklet, source: &AcousticTracklet) {
    let total = destination.frame_count + source.frame_count;
    let combined_overlap_probability = if total == 0 {
        0.0
    } else {
        (destination.overlap_probability * destination.frame_count as f32
            + source.overlap_probability * source.frame_count as f32)
            / total as f32
    };
    for index in 0..VOICE_VECTOR_DIMENSIONS {
        let left_support = destination.voice_support[index];
        let right_support = source.voice_support[index];
        if left_support == 0 && right_support == 0 {
            continue;
        }
        if left_support == 0 {
            destination.voice_mean[index] = source.voice_mean[index];
            destination.voice_variance[index] = source.voice_variance[index];
            destination.voice_valid[index] = source.voice_valid[index];
            destination.voice_support[index] = right_support;
            continue;
        }
        if right_support == 0 {
            continue;
        }
        let left_mean = destination.voice_mean[index];
        let right_mean = source.voice_mean[index];
        let delta = right_mean - left_mean;
        let left_weight = left_support as f32;
        let right_weight = right_support as f32;
        let total_weight = left_weight + right_weight;
        destination.voice_mean[index] = left_mean + delta * right_weight / total_weight;
        let left_m2 = destination.voice_variance[index] * (left_weight - 1.0).max(0.0);
        let right_m2 = source.voice_variance[index] * (right_weight - 1.0).max(0.0);
        let between_m2 = delta * delta * (left_weight / total_weight) * right_weight;
        destination.voice_variance[index] =
            (left_m2 + right_m2 + between_m2) / (total_weight - 1.0).max(1.0);
        destination.voice_valid[index] = true;
        destination.voice_support[index] = left_support.saturating_add(right_support);
    }
    if !destination.channel_valid && source.channel_valid {
        destination.channel_mean = source.channel_mean;
        destination.channel_variance = source.channel_variance;
        destination.channel_valid = true;
        destination.channel_dimensions = source.channel_dimensions;
    } else if destination.channel_valid && source.channel_valid {
        let left_count = destination.channel_frame_count as f32;
        let right_count = source.channel_frame_count as f32;
        let total_count = left_count + right_count;
        destination.channel_dimensions = destination
            .channel_dimensions
            .min(source.channel_dimensions);
        for index in 0..destination.channel_dimensions {
            let left_mean = destination.channel_mean[index];
            let right_mean = source.channel_mean[index];
            let delta = right_mean - left_mean;
            destination.channel_mean[index] = left_mean + delta * right_count / total_count;
            let left_m2 = destination.channel_variance[index] * (left_count - 1.0).max(0.0);
            let right_m2 = source.channel_variance[index] * (right_count - 1.0).max(0.0);
            let between_m2 = delta * delta * (left_count / total_count) * right_count;
            destination.channel_variance[index] =
                (left_m2 + right_m2 + between_m2) / (total_count - 1.0).max(1.0);
        }
    }
    destination.end_ms = source.end_ms;
    destination.frame_count = total;
    destination.voiced_frame_count += source.voiced_frame_count;
    destination.identity_frame_count += source.identity_frame_count;
    destination.channel_frame_count += source.channel_frame_count;
    destination.overlap_probability = combined_overlap_probability;
    destination.overlap_suspected |= source.overlap_suspected;
}

/// Stable enrollment failure code for agent-facing diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileEnrollmentCode {
    InvalidRequest,
    EmptyHardEnrollment,
    ConflictingTrackletAttribution,
}

impl ProfileEnrollmentCode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "diarization.enrollment.invalid_request",
            Self::EmptyHardEnrollment => "diarization.enrollment.empty_hard_interval",
            Self::ConflictingTrackletAttribution => {
                "diarization.enrollment.conflicting_tracklet_attribution"
            }
        }
    }
}

/// Typed failure produced while constructing supervised profiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileEnrollmentError {
    pub code: ProfileEnrollmentCode,
    pub message: String,
    pub hint_index: Option<usize>,
}

impl std::fmt::Display for ProfileEnrollmentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for ProfileEnrollmentError {}

/// Per-hint enrollment audit without biometric feature values.
#[derive(Debug, Clone, PartialEq)]
pub struct HintEnrollmentEvidence {
    pub hint_index: usize,
    pub speaker_ref: String,
    pub policy: KnownSpeakerPolicy,
    pub usable_tracklet_count: usize,
    pub accepted_tracklet_count: usize,
    pub rejected_tracklet_count: usize,
    pub profile_accepted_tracklet_count: usize,
    pub profile_downweighted_tracklet_count: usize,
    pub profile_quarantined_tracklet_count: usize,
    pub applied_weight: f32,
    pub contradiction_score: Option<f32>,
}

/// Whether a trusted interval observation contributed to profile training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileTrainingDisposition {
    Accepted,
    Downweighted,
    Quarantined,
}

/// Feature-value-free reason for an enrollment training decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileTrainingReason {
    Consistent,
    LowVoicedCoverage,
    RobustDistanceOutlier,
    LeaveOneOutInconsistent,
}

/// Auditable profile-training decision. Hard attribution is intentionally
/// separate: a quarantined hard sample remains attributed to its speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileTrainingEvidence {
    pub tracklet_index: usize,
    pub hint_index: usize,
    pub speaker_ref: String,
    pub hard_attribution: bool,
    pub disposition: ProfileTrainingDisposition,
    pub reason: ProfileTrainingReason,
    pub applied_weight: f32,
}

/// Why supervised metric adaptation retained the deterministic unit metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileMetricAdaptationFallback {
    InsufficientSpeakers,
    InsufficientPerSpeakerSupport,
}

/// Privacy-safe audit for conservative, within-run metric adaptation.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileMetricAdaptationEvidence {
    pub policy_version: &'static str,
    pub enabled: bool,
    pub enrolled_speaker_count: usize,
    pub training_observation_count: usize,
    pub adapted_dimension_count: usize,
    pub maximum_absolute_weight_delta: f32,
    pub fallback: Option<ProfileMetricAdaptationFallback>,
}

#[derive(Debug, Clone)]
struct AcousticVoiceSubprofile {
    center: [f32; VOICE_VECTOR_DIMENSIONS],
    valid: [bool; VOICE_VECTOR_DIMENSIONS],
    scale: [f32; VOICE_VECTOR_DIMENSIONS],
    weight: f32,
}

#[derive(Debug, Clone)]
struct AcousticSpeakerProfile {
    speaker_ref: String,
    voice_median: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_valid: [bool; VOICE_VECTOR_DIMENSIONS],
    voice_mad: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_q25: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_q75: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_subprofiles: Vec<AcousticVoiceSubprofile>,
    channel_subprofiles: Vec<[f32; CHANNEL_VECTOR_DIMENSIONS]>,
    channel_dimensions: usize,
    frame_count: usize,
    voiced_duration_ms: u64,
    reliability: f32,
    anchored: bool,
    soft_hint_contradiction: Option<f32>,
    training_accepted_count: usize,
    training_downweighted_count: usize,
    training_quarantined_count: usize,
}

/// Validated within-run supervision and privacy-safe summaries.
///
/// Raw acoustic profiles remain private to this module. The public surface
/// intentionally exposes only hashes, counts, decisions, and summary quality.
#[derive(Debug, Clone)]
pub struct SpeakerEnrollment {
    pub hint_document_sha256: Option<String>,
    pub summaries: Vec<SpeakerProfileSummary>,
    pub evidence: Vec<HintEnrollmentEvidence>,
    pub training_evidence: Vec<ProfileTrainingEvidence>,
    pub metric_adaptation: ProfileMetricAdaptationEvidence,
    profiles: BTreeMap<String, AcousticSpeakerProfile>,
    voice_dimension_weights: [f32; VOICE_VECTOR_DIMENSIONS],
    hard_assignments: BTreeMap<usize, String>,
    soft_priors: BTreeMap<(usize, String), f32>,
    cannot_links: BTreeSet<(String, String)>,
    reserved_speaker_refs: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct EnrollmentObservation {
    tracklet_index: usize,
    hint_index: usize,
    voice: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_valid: [bool; VOICE_VECTOR_DIMENSIONS],
    channel: [f32; CHANNEL_VECTOR_DIMENSIONS],
    channel_valid: bool,
    channel_dimensions: usize,
    frame_count: usize,
    voiced_duration_ms: u64,
    weight: f32,
    hard: bool,
}

/// Validate and apply `speaker-hints-v1` intervals to tracklet statistics.
pub fn enroll_known_speaker_profiles(
    tracklets: &[AcousticTracklet],
    request: &DiarizationRequest,
    audio_duration_ms: u64,
) -> Result<SpeakerEnrollment, ProfileEnrollmentError> {
    request
        .validate(audio_duration_ms)
        .map_err(|error| ProfileEnrollmentError {
            code: ProfileEnrollmentCode::InvalidRequest,
            message: error.to_string(),
            hint_index: error.hint_index,
        })?;
    validate_tracklet_timeline(tracklets).map_err(|error| ProfileEnrollmentError {
        code: ProfileEnrollmentCode::InvalidRequest,
        message: error.to_string(),
        hint_index: None,
    })?;
    if tracklets
        .iter()
        .any(|tracklet| tracklet.end_ms > audio_duration_ms)
    {
        return Err(ProfileEnrollmentError {
            code: ProfileEnrollmentCode::InvalidRequest,
            message: "tracklet interval exceeds the canonical audio duration".to_owned(),
            hint_index: None,
        });
    }

    let hint_document_sha256 = (!request.known_intervals.is_empty())
        .then(|| canonical_hint_document_sha256(&request.known_intervals));
    let reserved_speaker_refs = request
        .known_intervals
        .iter()
        .map(|hint| hint.speaker_ref.clone())
        .collect();
    let mut by_speaker = BTreeMap::<String, Vec<EnrollmentObservation>>::new();
    let mut evidence = Vec::with_capacity(request.known_intervals.len());
    let mut training_evidence = Vec::new();
    let mut hard_assignments = BTreeMap::<usize, String>::new();
    let mut soft_priors = BTreeMap::<(usize, String), f32>::new();

    for (hint_index, hint) in request.known_intervals.iter().enumerate() {
        let guard = u64::from(request.enrollment_edge_guard_ms);
        let guarded_start = hint.start_ms.saturating_add(guard);
        let guarded_end = hint.end_ms.saturating_sub(guard);
        let candidates = if guarded_start < guarded_end {
            tracklets
                .iter()
                .filter(|tracklet| {
                    tracklet.start_ms >= guarded_start
                        && tracklet.end_ms <= guarded_end
                        && tracklet.frame_count >= 3
                        && tracklet.identity_frame_count.saturating_mul(4) >= tracklet.frame_count
                        && tracklet.voice_valid.iter().any(|valid| *valid)
                        && !tracklet.overlap_suspected
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if hint.policy == KnownSpeakerPolicy::HardMustLink && candidates.is_empty() {
            return Err(ProfileEnrollmentError {
                code: ProfileEnrollmentCode::EmptyHardEnrollment,
                message: format!(
                    "hard hint {hint_index} has no usable voiced evidence after edge guards"
                ),
                hint_index: Some(hint_index),
            });
        }

        let mut applied_weight = 0.0_f32;
        for tracklet in &candidates {
            let hard = hint.policy == KnownSpeakerPolicy::HardMustLink;
            let weight = if hard {
                tracklet.identity_frame_count.max(1) as f32
            } else {
                ((hint.confidence as f32) * tracklet.identity_frame_count.min(50) as f32).min(20.0)
            };
            if hard
                && let Some(existing) =
                    hard_assignments.insert(tracklet.tracklet_index, hint.speaker_ref.clone())
                && existing != hint.speaker_ref
            {
                return Err(ProfileEnrollmentError {
                    code: ProfileEnrollmentCode::ConflictingTrackletAttribution,
                    message: format!(
                        "tracklet {} is selected by hard hints for both {existing} and {}",
                        tracklet.tracklet_index, hint.speaker_ref
                    ),
                    hint_index: Some(hint_index),
                });
            }
            by_speaker
                .entry(hint.speaker_ref.clone())
                .or_default()
                .push(EnrollmentObservation {
                    tracklet_index: tracklet.tracklet_index,
                    hint_index,
                    voice: tracklet.voice_mean,
                    voice_valid: tracklet.voice_valid,
                    channel: tracklet.channel_mean,
                    channel_valid: tracklet.channel_valid,
                    channel_dimensions: tracklet.channel_dimensions,
                    frame_count: tracklet.frame_count,
                    voiced_duration_ms: tracklet.voiced_frame_count as u64 * 10,
                    weight,
                    hard,
                });
            applied_weight += weight;
        }
        evidence.push(HintEnrollmentEvidence {
            hint_index,
            speaker_ref: hint.speaker_ref.clone(),
            policy: hint.policy,
            usable_tracklet_count: candidates.len(),
            accepted_tracklet_count: if hint.policy == KnownSpeakerPolicy::HardMustLink {
                candidates.len()
            } else {
                0
            },
            rejected_tracklet_count: 0,
            profile_accepted_tracklet_count: 0,
            profile_downweighted_tracklet_count: 0,
            profile_quarantined_tracklet_count: 0,
            applied_weight,
            contradiction_score: None,
        });
    }

    let mut profiles = BTreeMap::new();
    let mut profile_training_observations = BTreeMap::<String, Vec<EnrollmentObservation>>::new();
    for (speaker_ref, observations) in by_speaker {
        let hard_observations = deduplicate_profile_observations(
            observations
                .iter()
                .filter(|observation| observation.hard)
                .cloned()
                .collect(),
        );
        let provisional_source = if hard_observations.is_empty() {
            deduplicate_profile_observations(observations.clone())
        } else {
            hard_observations
        };
        let (provisional, provisional_valid) =
            robust_location(&provisional_source, |observation| {
                (observation.voice, observation.voice_valid)
            });
        let mut accepted = Vec::new();
        let mut maximum_contradiction = 0.0_f32;
        for observation in observations {
            let contradiction = masked_euclidean_distance(
                &observation.voice,
                &observation.voice_valid,
                &provisional,
                &provisional_valid,
            )
            .unwrap_or(f32::INFINITY);
            let accept = observation.hard || contradiction <= 0.65;
            let hint_evidence = &mut evidence[observation.hint_index];
            if observation.hard {
                accepted.push(observation);
            } else if accept {
                hint_evidence.accepted_tracklet_count += 1;
                accepted.push(observation);
            } else {
                hint_evidence.rejected_tracklet_count += 1;
                hint_evidence.applied_weight =
                    (hint_evidence.applied_weight - observation.weight).max(0.0);
                maximum_contradiction = maximum_contradiction.max(contradiction);
                hint_evidence.contradiction_score = Some(
                    hint_evidence
                        .contradiction_score
                        .map_or(contradiction, |current| current.max(contradiction)),
                );
            }
        }
        if accepted.is_empty() {
            continue;
        }
        let accepted = deduplicate_profile_observations(accepted);
        let (accepted, speaker_training_evidence) =
            apply_profile_training_hygiene(&speaker_ref, accepted, &mut evidence);
        if accepted.is_empty() {
            continue;
        }
        for observation in &accepted {
            if !observation.hard {
                let prior = soft_priors
                    .entry((observation.tracklet_index, speaker_ref.clone()))
                    .or_default();
                *prior = prior.max(observation.weight.min(20.0));
            }
        }
        let training_accepted_count = speaker_training_evidence
            .iter()
            .filter(|item| item.disposition == ProfileTrainingDisposition::Accepted)
            .count();
        let training_downweighted_count = speaker_training_evidence
            .iter()
            .filter(|item| item.disposition == ProfileTrainingDisposition::Downweighted)
            .count();
        let training_quarantined_count = speaker_training_evidence
            .iter()
            .filter(|item| item.disposition == ProfileTrainingDisposition::Quarantined)
            .count();
        training_evidence.extend(speaker_training_evidence);
        let profile = build_speaker_profile(
            speaker_ref.clone(),
            &accepted,
            (maximum_contradiction > 0.0).then_some(maximum_contradiction),
            training_accepted_count,
            training_downweighted_count,
            training_quarantined_count,
        );
        profile_training_observations.insert(speaker_ref.clone(), accepted);
        profiles.insert(speaker_ref, profile);
    }

    let (voice_dimension_weights, metric_adaptation) =
        supervised_metric_adaptation(&profiles, &profile_training_observations);
    let cannot_links = hard_speaker_cannot_links(request);
    let summaries = profiles.values().map(profile_summary).collect::<Vec<_>>();
    Ok(SpeakerEnrollment {
        hint_document_sha256,
        summaries,
        evidence,
        training_evidence,
        metric_adaptation,
        profiles,
        voice_dimension_weights,
        hard_assignments,
        soft_priors,
        cannot_links,
        reserved_speaker_refs,
    })
}

fn deduplicate_profile_observations(
    observations: Vec<EnrollmentObservation>,
) -> Vec<EnrollmentObservation> {
    let mut by_tracklet = BTreeMap::<usize, EnrollmentObservation>::new();
    for observation in observations {
        by_tracklet
            .entry(observation.tracklet_index)
            .and_modify(|current| {
                if (!current.hard && observation.hard)
                    || (current.hard == observation.hard && observation.weight > current.weight)
                {
                    *current = observation.clone();
                }
            })
            .or_insert(observation);
    }
    by_tracklet.into_values().collect()
}

fn apply_profile_training_hygiene(
    speaker_ref: &str,
    observations: Vec<EnrollmentObservation>,
    hint_evidence: &mut [HintEnrollmentEvidence],
) -> (Vec<EnrollmentObservation>, Vec<ProfileTrainingEvidence>) {
    let (center, center_valid) = robust_location(&observations, |observation| {
        (observation.voice, observation.voice_valid)
    });
    let distances = observations
        .iter()
        .map(|observation| {
            masked_euclidean_distance(
                &observation.voice,
                &observation.voice_valid,
                &center,
                &center_valid,
            )
            .unwrap_or(f32::INFINITY)
        })
        .collect::<Vec<_>>();
    let weighted_distances = distances
        .iter()
        .zip(&observations)
        .filter_map(|(&distance, observation)| {
            distance
                .is_finite()
                .then_some((distance, observation.weight))
        })
        .collect::<Vec<_>>();
    let distance_median = weighted_quantile(&weighted_distances, 0.5);
    let distance_deviations = weighted_distances
        .iter()
        .map(|(distance, weight)| ((distance - distance_median).abs(), *weight))
        .collect::<Vec<_>>();
    let distance_mad = weighted_quantile(&distance_deviations, 0.5);
    let quarantine_threshold = distance_median + (3.0 * distance_mad).max(0.15);
    let downweight_threshold = distance_median + (1.5 * distance_mad).max(0.075);
    let leave_one_out_distances = (0..observations.len())
        .map(|excluded| {
            if observations.len() < 3 {
                return None;
            }
            let retained = observations
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != excluded)
                .map(|(_, observation)| observation.clone())
                .collect::<Vec<_>>();
            let (leave_one_out_center, leave_one_out_valid) =
                robust_location(&retained, |observation| {
                    (observation.voice, observation.voice_valid)
                });
            masked_euclidean_distance(
                &observations[excluded].voice,
                &observations[excluded].voice_valid,
                &leave_one_out_center,
                &leave_one_out_valid,
            )
        })
        .collect::<Vec<_>>();
    let nearest_peer_distances = observations
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            observations
                .iter()
                .enumerate()
                .filter(|(candidate, _)| *candidate != index)
                .filter_map(|(_, candidate)| {
                    masked_euclidean_distance(
                        &observation.voice,
                        &observation.voice_valid,
                        &candidate.voice,
                        &candidate.voice_valid,
                    )
                })
                .min_by(f32::total_cmp)
        })
        .collect::<Vec<_>>();

    let mut retained = Vec::with_capacity(observations.len());
    let mut decisions = Vec::with_capacity(observations.len());
    for (index, mut observation) in observations.into_iter().enumerate() {
        let global_distance = distances[index];
        let leave_one_out_distance = leave_one_out_distances[index];
        let voiced_denominator = observation.frame_count.saturating_mul(10).max(1) as f32;
        let voiced_coverage =
            (observation.voiced_duration_ms as f32 / voiced_denominator).clamp(0.0, 1.0);
        let low_voiced_coverage = voiced_coverage < 0.50;
        let disposition = if distances.len() >= 3
            && global_distance.is_finite()
            && global_distance > quarantine_threshold
            && leave_one_out_distance
                .is_some_and(|distance| distance > quarantine_threshold.max(0.50))
            && nearest_peer_distances[index]
                .is_none_or(|distance| distance > quarantine_threshold.max(0.50))
        {
            ProfileTrainingDisposition::Quarantined
        } else if low_voiced_coverage
            || (global_distance.is_finite() && global_distance > downweight_threshold)
        {
            ProfileTrainingDisposition::Downweighted
        } else {
            ProfileTrainingDisposition::Accepted
        };
        let reason = match disposition {
            ProfileTrainingDisposition::Accepted => ProfileTrainingReason::Consistent,
            ProfileTrainingDisposition::Downweighted if low_voiced_coverage => {
                ProfileTrainingReason::LowVoicedCoverage
            }
            ProfileTrainingDisposition::Downweighted => {
                ProfileTrainingReason::RobustDistanceOutlier
            }
            ProfileTrainingDisposition::Quarantined => {
                ProfileTrainingReason::LeaveOneOutInconsistent
            }
        };
        let original_weight = observation.weight;
        if disposition == ProfileTrainingDisposition::Downweighted {
            observation.weight *= 0.35;
        } else if disposition == ProfileTrainingDisposition::Quarantined {
            observation.weight = 0.0;
        }
        let evidence = &mut hint_evidence[observation.hint_index];
        match disposition {
            ProfileTrainingDisposition::Accepted => {
                evidence.profile_accepted_tracklet_count += 1;
            }
            ProfileTrainingDisposition::Downweighted => {
                evidence.profile_downweighted_tracklet_count += 1;
            }
            ProfileTrainingDisposition::Quarantined => {
                evidence.profile_quarantined_tracklet_count += 1;
            }
        }
        evidence.applied_weight =
            (evidence.applied_weight - original_weight + observation.weight).max(0.0);
        decisions.push(ProfileTrainingEvidence {
            tracklet_index: observation.tracklet_index,
            hint_index: observation.hint_index,
            speaker_ref: speaker_ref.to_owned(),
            hard_attribution: observation.hard,
            disposition,
            reason,
            applied_weight: observation.weight,
        });
        if disposition != ProfileTrainingDisposition::Quarantined {
            retained.push(observation);
        }
    }
    (retained, decisions)
}

fn canonical_hint_document_sha256(hints: &[KnownSpeakerInterval]) -> String {
    let mut canonical = hints.to_vec();
    canonical.sort_by(|left, right| {
        left.speaker_ref
            .cmp(&right.speaker_ref)
            .then(left.start_ms.cmp(&right.start_ms))
            .then(left.end_ms.cmp(&right.end_ms))
            .then(policy_rank(left.policy).cmp(&policy_rank(right.policy)))
            .then(left.confidence.total_cmp(&right.confidence))
            .then(left.provenance.cmp(&right.provenance))
    });
    let mut hasher = Sha256::new();
    hasher.update(b"speaker-hints-v1\0");
    for hint in canonical {
        hash_field(&mut hasher, hint.speaker_ref.as_bytes());
        hasher.update(hint.start_ms.to_le_bytes());
        hasher.update(hint.end_ms.to_le_bytes());
        hasher.update([policy_rank(hint.policy)]);
        hasher.update(hint.confidence.to_bits().to_le_bytes());
        hash_field(
            &mut hasher,
            hint.provenance.as_deref().unwrap_or_default().as_bytes(),
        );
    }
    format!("{:x}", hasher.finalize())
}

fn hash_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

const fn policy_rank(policy: KnownSpeakerPolicy) -> u8 {
    match policy {
        KnownSpeakerPolicy::HardMustLink => 0,
        KnownSpeakerPolicy::SoftEnrollment => 1,
    }
}

fn hard_speaker_cannot_links(request: &DiarizationRequest) -> BTreeSet<(String, String)> {
    let speakers = request
        .known_intervals
        .iter()
        .filter(|hint| hint.policy == KnownSpeakerPolicy::HardMustLink)
        .map(|hint| hint.speaker_ref.clone())
        .collect::<BTreeSet<_>>();
    let mut cannot_links = BTreeSet::new();
    for (left_index, left) in speakers.iter().enumerate() {
        for right in speakers.iter().skip(left_index + 1) {
            cannot_links.insert((left.clone(), right.clone()));
        }
    }
    cannot_links
}

fn robust_location<F>(
    observations: &[EnrollmentObservation],
    vector: F,
) -> (
    [f32; VOICE_VECTOR_DIMENSIONS],
    [bool; VOICE_VECTOR_DIMENSIONS],
)
where
    F: Fn(
        &EnrollmentObservation,
    ) -> (
        [f32; VOICE_VECTOR_DIMENSIONS],
        [bool; VOICE_VECTOR_DIMENSIONS],
    ),
{
    let mut output = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    let mut valid = [false; VOICE_VECTOR_DIMENSIONS];
    for (dimension, output_value) in output.iter_mut().enumerate() {
        let values = observations
            .iter()
            .filter_map(|observation| {
                let (values, mask) = vector(observation);
                mask[dimension].then_some((values[dimension], observation.weight))
            })
            .collect::<Vec<_>>();
        if !values.is_empty() {
            *output_value = weighted_quantile(&values, 0.5);
            valid[dimension] = true;
        }
    }
    (output, valid)
}

fn build_speaker_profile(
    speaker_ref: String,
    observations: &[EnrollmentObservation],
    soft_hint_contradiction: Option<f32>,
    training_accepted_count: usize,
    training_downweighted_count: usize,
    training_quarantined_count: usize,
) -> AcousticSpeakerProfile {
    let (voice_median, voice_valid) = robust_location(observations, |observation| {
        (observation.voice, observation.voice_valid)
    });
    let mut voice_mad = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    let mut voice_q25 = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    let mut voice_q75 = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    for dimension in 0..VOICE_VECTOR_DIMENSIONS {
        let values = observations
            .iter()
            .filter(|observation| observation.voice_valid[dimension])
            .map(|observation| (observation.voice[dimension], observation.weight))
            .collect::<Vec<_>>();
        let deviations = observations
            .iter()
            .filter(|observation| observation.voice_valid[dimension])
            .map(|observation| {
                (
                    (observation.voice[dimension] - voice_median[dimension]).abs(),
                    observation.weight,
                )
            })
            .collect::<Vec<_>>();
        if values.is_empty() {
            continue;
        }
        voice_mad[dimension] = weighted_quantile(&deviations, 0.5).max(0.025);
        voice_q25[dimension] = weighted_quantile(&values, 0.25);
        voice_q75[dimension] = weighted_quantile(&values, 0.75);
    }
    let channel_dimensions = observations
        .iter()
        .filter(|observation| observation.channel_valid)
        .map(|observation| observation.channel_dimensions)
        .min()
        .unwrap_or(0);
    let voice_subprofiles = build_voice_subprofiles(observations);
    let channel_subprofiles = build_channel_subprofiles(observations, channel_dimensions);
    let frame_count = observations
        .iter()
        .map(|observation| observation.frame_count)
        .sum();
    let voiced_duration_ms = observations
        .iter()
        .map(|observation| observation.voiced_duration_ms)
        .sum();
    let total_weight = observations
        .iter()
        .map(|observation| observation.weight)
        .sum::<f32>();
    let valid_dimensions = voice_valid.iter().filter(|&&valid| valid).count();
    let average_mad = voice_mad
        .iter()
        .zip(voice_valid)
        .filter_map(|(&mad, valid)| valid.then_some(mad))
        .sum::<f32>()
        / valid_dimensions.max(1) as f32;
    let reliability = ((total_weight / 100.0).min(1.0)
        * (valid_dimensions as f32 / VOICE_VECTOR_DIMENSIONS as f32)
        * (1.0 / (1.0 + 2.0 * average_mad)))
        .clamp(0.0, 1.0);
    AcousticSpeakerProfile {
        speaker_ref,
        voice_median,
        voice_valid,
        voice_mad,
        voice_q25,
        voice_q75,
        voice_subprofiles,
        channel_subprofiles,
        channel_dimensions,
        frame_count,
        voiced_duration_ms,
        reliability,
        anchored: observations.iter().any(|observation| observation.hard),
        soft_hint_contradiction,
        training_accepted_count,
        training_downweighted_count,
        training_quarantined_count,
    }
}

fn supervised_metric_adaptation(
    profiles: &BTreeMap<String, AcousticSpeakerProfile>,
    observations: &BTreeMap<String, Vec<EnrollmentObservation>>,
) -> (
    [f32; VOICE_VECTOR_DIMENSIONS],
    ProfileMetricAdaptationEvidence,
) {
    const POLICY_VERSION: &str = "speaker-profile-metric-shrinkage-v1";
    let unit_weights = [1.0_f32; VOICE_VECTOR_DIMENSIONS];
    let observation_count = observations.values().map(Vec::len).sum::<usize>();
    let fallback = if profiles.len() < 2 {
        Some(ProfileMetricAdaptationFallback::InsufficientSpeakers)
    } else if observations.values().any(|speaker| speaker.len() < 2)
        || observation_count < profiles.len().saturating_mul(2).max(6)
    {
        Some(ProfileMetricAdaptationFallback::InsufficientPerSpeakerSupport)
    } else {
        None
    };
    if let Some(fallback) = fallback {
        return (
            unit_weights,
            ProfileMetricAdaptationEvidence {
                policy_version: POLICY_VERSION,
                enabled: false,
                enrolled_speaker_count: profiles.len(),
                training_observation_count: observation_count,
                adapted_dimension_count: 0,
                maximum_absolute_weight_delta: 0.0,
                fallback: Some(fallback),
            },
        );
    }

    let mut weights = unit_weights;
    for (dimension, weight) in weights.iter_mut().enumerate() {
        let centers = profiles
            .values()
            .filter(|profile| profile.voice_valid[dimension])
            .map(|profile| (profile.voice_median[dimension], 1.0_f32))
            .collect::<Vec<_>>();
        let within_spreads = profiles
            .values()
            .filter(|profile| profile.voice_valid[dimension])
            .map(|profile| (profile.voice_mad[dimension], 1.0_f32))
            .collect::<Vec<_>>();
        if centers.len() < 2 {
            continue;
        }
        let minimum_center = centers
            .iter()
            .map(|(center, _)| *center)
            .min_by(f32::total_cmp)
            .unwrap_or(0.0);
        let maximum_center = centers
            .iter()
            .map(|(center, _)| *center)
            .max_by(f32::total_cmp)
            .unwrap_or(minimum_center);
        let between_spread = (maximum_center - minimum_center).abs() * 0.5;
        let within_spread = weighted_quantile(&within_spreads, 0.5);
        let discriminability = between_spread / (within_spread + 0.05);
        let bounded_signal = discriminability / (1.0 + discriminability);
        let target_weight = 0.75 + 0.50 * bounded_signal;
        *weight = (1.0 + 0.25 * (target_weight - 1.0)).clamp(0.9375, 1.0625);
    }
    let adapted_dimension_count = weights
        .iter()
        .filter(|&&weight| (weight - 1.0).abs() > 1e-3)
        .count();
    let maximum_absolute_weight_delta = weights
        .iter()
        .map(|weight| (weight - 1.0).abs())
        .max_by(f32::total_cmp)
        .unwrap_or(0.0);
    (
        weights,
        ProfileMetricAdaptationEvidence {
            policy_version: POLICY_VERSION,
            enabled: true,
            enrolled_speaker_count: profiles.len(),
            training_observation_count: observation_count,
            adapted_dimension_count,
            maximum_absolute_weight_delta,
            fallback: None,
        },
    )
}

fn weighted_quantile(values: &[(f32, f32)], quantile: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.0.total_cmp(&right.0));
    let total_weight = sorted
        .iter()
        .map(|(_, weight)| weight.max(0.0))
        .sum::<f32>();
    if total_weight <= f32::EPSILON {
        return sorted[sorted.len() / 2].0;
    }
    let target = quantile.clamp(0.0, 1.0) * total_weight;
    let mut cumulative = 0.0_f32;
    for (value, weight) in &sorted {
        cumulative += weight.max(0.0);
        if cumulative >= target {
            return *value;
        }
    }
    sorted.last().map_or(0.0, |(value, _)| *value)
}

fn build_voice_subprofiles(observations: &[EnrollmentObservation]) -> Vec<AcousticVoiceSubprofile> {
    const MAX_VOICE_SUBPROFILES: usize = 4;
    const VOICE_MODE_JOIN_DISTANCE: f32 = 0.40;

    let mut ordered = observations.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|observation| observation.tracklet_index);
    let mut groups = Vec::<Vec<EnrollmentObservation>>::new();
    for observation in ordered {
        let nearest = groups
            .iter()
            .enumerate()
            .filter_map(|(index, group)| {
                let (center, valid) = robust_location(group, |item| (item.voice, item.voice_valid));
                masked_euclidean_distance(
                    &observation.voice,
                    &observation.voice_valid,
                    &center,
                    &valid,
                )
                .map(|distance| (distance, index))
            })
            .min_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        let selected = match nearest {
            Some((distance, index)) if distance <= VOICE_MODE_JOIN_DISTANCE => index,
            _ if groups.len() < MAX_VOICE_SUBPROFILES => {
                groups.push(vec![observation.clone()]);
                continue;
            }
            Some((_, index)) => index,
            None => {
                groups.push(vec![observation.clone()]);
                continue;
            }
        };
        groups[selected].push(observation.clone());
    }

    groups
        .into_iter()
        .map(|group| {
            let (center, valid) = robust_location(&group, |observation| {
                (observation.voice, observation.voice_valid)
            });
            let mut scale = [0.025_f32; VOICE_VECTOR_DIMENSIONS];
            for dimension in 0..VOICE_VECTOR_DIMENSIONS {
                let deviations = group
                    .iter()
                    .filter(|observation| observation.voice_valid[dimension])
                    .map(|observation| {
                        (
                            (observation.voice[dimension] - center[dimension]).abs(),
                            observation.weight,
                        )
                    })
                    .collect::<Vec<_>>();
                if !deviations.is_empty() {
                    scale[dimension] = weighted_quantile(&deviations, 0.5).max(0.025);
                }
            }
            AcousticVoiceSubprofile {
                center,
                valid,
                scale,
                weight: group
                    .iter()
                    .map(|observation| observation.weight)
                    .sum::<f32>()
                    .max(f32::EPSILON),
            }
        })
        .collect()
}

fn build_channel_subprofiles(
    observations: &[EnrollmentObservation],
    channel_dimensions: usize,
) -> Vec<[f32; CHANNEL_VECTOR_DIMENSIONS]> {
    if channel_dimensions == 0 {
        return Vec::new();
    }
    let mut subprofiles = Vec::<([f32; CHANNEL_VECTOR_DIMENSIONS], f32)>::new();
    for observation in observations
        .iter()
        .filter(|observation| observation.channel_valid)
    {
        let nearest = subprofiles
            .iter()
            .enumerate()
            .map(|(index, (center, _))| {
                (
                    euclidean_distance_prefix(center, &observation.channel, channel_dimensions),
                    index,
                )
            })
            .min_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        let selected = match nearest {
            Some((distance, index)) if distance <= 0.18 => index,
            _ if subprofiles.len() < 4 => {
                subprofiles.push((observation.channel, observation.weight));
                continue;
            }
            Some((_, index)) => index,
            None => {
                subprofiles.push((observation.channel, observation.weight));
                continue;
            }
        };
        let (center, weight) = &mut subprofiles[selected];
        let new_total = *weight + observation.weight;
        for (dimension, center_value) in center.iter_mut().enumerate() {
            *center_value = (*weight * *center_value
                + observation.weight * observation.channel[dimension])
                / new_total.max(f32::EPSILON);
        }
        *weight = new_total;
    }
    subprofiles.into_iter().map(|(center, _)| center).collect()
}

fn profile_summary(profile: &AcousticSpeakerProfile) -> SpeakerProfileSummary {
    SpeakerProfileSummary {
        speaker_ref: profile.speaker_ref.clone(),
        frame_count: u64::try_from(profile.frame_count).unwrap_or(u64::MAX),
        voiced_duration_ms: profile.voiced_duration_ms,
        reliability: f64::from(profile.reliability),
        voice_profile_count: u32::try_from(profile.voice_subprofiles.len()).unwrap_or(u32::MAX),
        channel_profile_count: u32::try_from(profile.channel_subprofiles.len()).unwrap_or(u32::MAX),
        training_accepted_count: u32::try_from(profile.training_accepted_count).unwrap_or(u32::MAX),
        training_downweighted_count: u32::try_from(profile.training_downweighted_count)
            .unwrap_or(u32::MAX),
        training_quarantined_count: u32::try_from(profile.training_quarantined_count)
            .unwrap_or(u32::MAX),
        anchored: profile.anchored,
        soft_hint_contradiction: profile.soft_hint_contradiction.map(f64::from),
    }
}

/// One temporally smoothed tracklet assignment.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticSpeakerAssignment {
    pub tracklet_index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker_ref: Option<String>,
    pub speaker_confidence: f32,
    pub secondary_speaker_ref: Option<String>,
    pub secondary_speaker_confidence: Option<f32>,
    pub change_confidence: f32,
    pub overlap_suspected: bool,
    pub hard_attribution: bool,
}

/// Auditable, feature-value-free summary of one same/different comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticSpeakerPairEvidence {
    /// Variance-normalized vocal distance over the selected perturbation lane.
    pub voice_distance: f32,
    /// Separately retained channel distance; never treated as vocal identity.
    pub channel_distance: f32,
    /// Log odds for the different-speaker hypothesis.
    pub different_log_odds: f32,
    /// Posterior probability of the same-speaker hypothesis.
    pub same_speaker_probability: f32,
    pub active_voice_dimensions: usize,
    pub support: f32,
}

/// Feature-value-free evidence retained for one candidate speaker label.
///
/// This is deliberately distinct from cluster cardinality. A candidate may
/// exist in the constrained search while remaining unsupported and therefore
/// absent from authoritative assignments and profile summaries.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticSpeakerEvidenceSummary {
    pub speaker_ref: String,
    pub assigned_tracklet_count: usize,
    pub independent_tracklet_count: usize,
    pub recurrence_episode_count: usize,
    pub voiced_frame_count: usize,
    pub independent_voiced_frame_count: usize,
    pub voiced_duration_ms: u64,
    pub mean_assignment_confidence: f32,
    pub cluster_reliability: f32,
    pub hard_anchored: bool,
    pub separated_from_supported_speakers: bool,
    pub reasons: Vec<SpeakerEvidenceReason>,
    pub supported: bool,
}

/// Why the probabilistic clustering candidate reverted to the frozen path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcousticClusteringFallbackReason {
    InsufficientSharedVoiceDimensions,
    InvalidPosterior,
    UnstableSpeakerCount,
    SpeakerCountPriorUnresolved,
}

/// Non-biometric trace for one deterministic agglomeration step.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterMergeTrace {
    pub remaining_clusters: usize,
    pub distance: f32,
    pub same_speaker_probability: Option<f32>,
    pub voice_distance: Option<f32>,
    pub channel_distance: Option<f32>,
    pub left_anchor: Option<String>,
    pub right_anchor: Option<String>,
}

/// Diagnostics and assignments from constrained acoustic clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticClusteringResult {
    pub assignments: Vec<AcousticSpeakerAssignment>,
    pub profiles: Vec<SpeakerProfileSummary>,
    pub count_estimate: Option<SpeakerCountEstimate>,
    pub detected_speakers: usize,
    pub prototype_count: usize,
    pub prototype_cap: usize,
    pub cap_pressure: bool,
    pub constraints_satisfied: bool,
    pub speaker_evidence: Vec<AcousticSpeakerEvidenceSummary>,
    pub dominant_speaker_share: f32,
    pub unknown_voiced_share: f32,
    pub speaker_separation_satisfied: bool,
    pub bootstrap_stability: f32,
    pub requested_mode: AcousticClusteringMode,
    pub executed_mode: AcousticClusteringMode,
    pub fallback_reason: Option<AcousticClusteringFallbackReason>,
    pub speaker_pair_calibration_sha256: String,
    pub calibration_status: &'static str,
    pub merge_trace: Vec<ClusterMergeTrace>,
}

/// Lossless convenience projection of an independent acoustic turn timeline.
///
/// `segments` retain every input segment's text bytes, order, and ASR
/// confidence. The index lists carry evidence that does not fit in the legacy
/// [`TranscriptionSegment`] shape.
#[derive(Debug, Clone)]
pub struct DiarizationProjection {
    pub segments: Vec<TranscriptionSegment>,
    pub mixed_speaker_segment_indices: Vec<usize>,
    pub overlap_suspected_segment_indices: Vec<usize>,
}

/// Borrowed inputs for one complete native acoustic diarization pass.
#[derive(Debug, Clone, Copy)]
pub struct AcousticDiarizationInput<'a> {
    pub samples: &'a [f32],
    pub normalized_input_sha256: &'a str,
    pub segments: &'a [TranscriptionSegment],
    pub word_aligned: bool,
    pub request: &'a DiarizationRequest,
    pub boundary_hints: &'a AcousticBoundaryHints,
}

/// Run the complete Rust-native acoustic diarization pipeline over canonical
/// 16 kHz mono PCM.
pub fn diarize_acoustic_pcm<C>(
    input: AcousticDiarizationInput<'_>,
    is_cancelled: C,
) -> FwResult<(DiarizationReport, DiarizationProjection)>
where
    C: FnMut() -> bool,
{
    diarize_acoustic_pcm_with_ablation(input, AcousticFeatureAblation::FullV2, is_cancelled)
}

/// Run one frozen representation ablation through the otherwise identical
/// native acoustic pipeline.
///
/// This production-facing convenience path deliberately retains the
/// fixed-safe change detector while posterior candidates are fail-closed by
/// their public evidence gates.
pub fn diarize_acoustic_pcm_with_ablation<C>(
    input: AcousticDiarizationInput<'_>,
    feature_ablation: AcousticFeatureAblation,
    is_cancelled: C,
) -> FwResult<(DiarizationReport, DiarizationProjection)>
where
    C: FnMut() -> bool,
{
    diarize_acoustic_pcm_with_ablation_and_detector(
        input,
        feature_ablation,
        AcousticChangeDetectorMode::FixedSafeV1,
        is_cancelled,
    )
}

/// Run one frozen representation and change-detector ablation through the
/// otherwise identical native acoustic pipeline.
pub fn diarize_acoustic_pcm_with_ablation_and_detector<C>(
    input: AcousticDiarizationInput<'_>,
    feature_ablation: AcousticFeatureAblation,
    detector_mode: AcousticChangeDetectorMode,
    is_cancelled: C,
) -> FwResult<(DiarizationReport, DiarizationProjection)>
where
    C: FnMut() -> bool,
{
    let (report, projection, _, _) = diarize_acoustic_pcm_with_detector_evidence_internal(
        input,
        feature_ablation,
        detector_mode,
        AcousticClusteringMode::FixedSafeV1,
        false,
        is_cancelled,
    )?;
    Ok((report, projection))
}

/// Run one explicit detector/clustering combination for aggregate evaluation.
pub(crate) fn diarize_acoustic_pcm_with_modes_evidence<C>(
    input: AcousticDiarizationInput<'_>,
    feature_ablation: AcousticFeatureAblation,
    detector_mode: AcousticChangeDetectorMode,
    clustering_mode: AcousticClusteringMode,
    is_cancelled: C,
) -> FwResult<(
    DiarizationReport,
    DiarizationProjection,
    AcousticChangeEvaluationEvidence,
    AcousticClusteringEvaluationEvidence,
)>
where
    C: FnMut() -> bool,
{
    diarize_acoustic_pcm_with_detector_evidence_internal(
        input,
        feature_ablation,
        detector_mode,
        clustering_mode,
        true,
        is_cancelled,
    )
}

fn diarize_acoustic_pcm_with_detector_evidence_internal<C>(
    input: AcousticDiarizationInput<'_>,
    feature_ablation: AcousticFeatureAblation,
    detector_mode: AcousticChangeDetectorMode,
    clustering_mode: AcousticClusteringMode,
    capture_evaluation_evidence: bool,
    mut is_cancelled: C,
) -> FwResult<(
    DiarizationReport,
    DiarizationProjection,
    AcousticChangeEvaluationEvidence,
    AcousticClusteringEvaluationEvidence,
)>
where
    C: FnMut() -> bool,
{
    let AcousticDiarizationInput {
        samples,
        normalized_input_sha256,
        segments,
        word_aligned,
        request,
        boundary_hints,
    } = input;
    if !matches!(
        request.engine,
        DiarizationEngine::Auto | DiarizationEngine::Acoustic
    ) {
        return Err(FwError::InvalidRequest(format!(
            "native acoustic pipeline cannot execute {:?} diarization",
            request.engine
        )));
    }
    if normalized_input_sha256.len() != 64
        || !normalized_input_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(FwError::InvalidRequest(
            "normalized_input_sha256 must be exactly 64 lowercase hexadecimal characters"
                .to_owned(),
        ));
    }
    let audio_duration_ms = samples_to_ms(samples.len());
    request
        .validate(audio_duration_ms)
        .map_err(|error| FwError::InvalidRequest(error.to_string()))?;
    let supervised_boundaries_ms = supervised_enrollment_boundaries_ms(request);
    let mut segmenter = AcousticSegmenter::new_with_supervised_boundaries(
        boundary_hints,
        &supervised_boundaries_ms,
        feature_ablation,
        detector_mode,
    )?;
    segmenter.capture_evaluation_evidence = capture_evaluation_evidence;
    let extraction =
        extract_acoustic_features(samples, &mut is_cancelled, |frame| segmenter.push(frame))?;
    if is_cancelled() {
        return Err(FwError::Cancelled(
            "acoustic diarization cancelled after feature extraction".to_owned(),
        ));
    }
    let (tracklets, segmentation, change_evaluation_evidence) = segmenter.finish()?;
    let enrollment = enroll_known_speaker_profiles(&tracklets, request, audio_duration_ms)
        .map_err(|error| FwError::InvalidRequest(error.to_string()))?;
    let clustering = cluster_acoustic_tracklets_with_mode(
        &tracklets,
        &enrollment,
        &request.speaker_count,
        usize::from(request.max_prototypes),
        clustering_mode,
        is_cancelled,
    )?;
    let turns = diarization_turns_from_assignments(&clustering.assignments, audio_duration_ms)?;
    let speaker_queries =
        speaker_attribution_queries(&clustering.assignments, normalized_input_sha256);
    let projection = project_diarization_onto_segments(segments, &turns, word_aligned)?;
    let fallback_status = if matches!(
        request.speaker_count,
        SpeakerCountRequest::Infer
            | SpeakerCountRequest::Prior { .. }
            | SpeakerCountRequest::Range { .. }
    ) && clustering
        .count_estimate
        .as_ref()
        .and_then(|estimate| estimate.selected_count)
        .is_none()
    {
        DiarizationFallbackStatus::SpeakerCountUnresolved
    } else if !clustering.constraints_satisfied {
        DiarizationFallbackStatus::UnsatisfiedConstraints
    } else if clustering.detected_speakers == 0 {
        DiarizationFallbackStatus::InsufficientEvidence
    } else {
        DiarizationFallbackStatus::NotNeeded
    };
    if request.fallback == DiarizationFallbackPolicy::Error
        && fallback_status != DiarizationFallbackStatus::NotNeeded
    {
        return Err(FwError::InvalidRequest(format!(
            "native acoustic diarization fallback triggered: {fallback_status:?}"
        )));
    }
    let speaker_count = public_speaker_count_outcome(&request.speaker_count, &clustering)?;
    let hint_evidence = public_hint_evidence(&enrollment.evidence)?;
    let diagnostics = vec![
        format!(
            "feature_schema={} schema_sha256={} ablation={} extraction_schema={} features={} voiced={} reliable_pitch={} high_information={} missing_pitch={} retained_dsp_bytes<={}",
            feature_ablation.schema_version().id(),
            acoustic_feature_schema_sha256(feature_ablation.schema_version()),
            feature_ablation.id(),
            extraction.feature_schema,
            extraction.frame_count,
            extraction.voiced_frame_count,
            extraction.reliable_pitch_frame_count,
            extraction.high_information_frame_count,
            extraction.missing_pitch_frame_count,
            extraction.retained_state_bytes_upper_bound
        ),
        format!(
            "tracklets={} acoustic_changes={} posterior_candidate_frames={} page_hinkley_candidate_frames={} bayesian_candidate_frames={} fixed_candidate_frames={} fallback_candidate_frames={} change_detector={} change_calibration_sha256={} normalized_voice_dimensions={} normalized_channel_dimensions={} missing_voice_dimensions={} retained_segmentation_frames<={}",
            segmentation.tracklet_count,
            segmentation.acoustic_change_count,
            segmentation.posterior_candidate_count,
            segmentation.page_hinkley_candidate_count,
            segmentation.bayesian_candidate_count,
            segmentation.fixed_candidate_count,
            segmentation.fallback_candidate_count,
            detector_mode.id(),
            acoustic_change_calibration_sha256(),
            segmentation.normalized_voice_dimensions,
            segmentation.normalized_channel_dimensions,
            segmentation.missing_voice_dimensions,
            segmentation.maximum_retained_frames
        ),
        format!(
            "prototypes={}/{} cap_pressure={} count_stability={:.6} clustering_requested={} clustering_executed={} clustering_fallback={:?} speaker_pair_calibration_sha256={}",
            clustering.prototype_count,
            clustering.prototype_cap,
            clustering.cap_pressure,
            clustering.bootstrap_stability,
            clustering.requested_mode.id(),
            clustering.executed_mode.id(),
            clustering.fallback_reason,
            clustering.speaker_pair_calibration_sha256
        ),
        format!(
            "mixed_segments={} overlap_suspected_segments={} calibration={}",
            projection.mixed_speaker_segment_indices.len(),
            projection.overlap_suspected_segment_indices.len(),
            clustering.calibration_status
        ),
        clustering.count_estimate.as_ref().map_or_else(
            || "speaker_count_estimate=absent".to_owned(),
            |estimate| {
                format!(
                    "speaker_count_schema={} selected={:?} range={:?} unresolved_probability={:.6} entropy_bits={:.6} stability={:.6} calibration={} calibration_sha256={} evidence_sha256={} prototypes={} affinity_pair_evaluations={} sparse_edges={} estimated_peak_buffer_bytes={} stability_replicates={} solver_iterations={} solver_sparse_matvec_terms={} solver_residual={:?}",
                    estimate.schema_version,
                    estimate.selected_count,
                    estimate.supported_range,
                    estimate.unresolved_probability,
                    estimate.entropy_bits,
                    estimate.stability,
                    speaker_count_calibration_status_label(estimate.calibration_status),
                    estimate.calibration_sha256,
                    estimate.evidence_sha256,
                    estimate.resources.prototype_count,
                    estimate.resources.affinity_pair_evaluations,
                    estimate.resources.retained_sparse_edges,
                    estimate.resources.estimated_peak_buffer_bytes,
                    estimate.resources.stability_replicates,
                    estimate.resources.solver_iterations,
                    estimate.resources.solver_sparse_matvec_terms,
                    estimate.resources.solver_residual,
                )
            },
        ),
    ];
    let clustering_evaluation = AcousticClusteringEvaluationEvidence {
        requested_mode: clustering.requested_mode,
        executed_mode: clustering.executed_mode,
        fallback_reason: clustering.fallback_reason,
        speaker_count_stability: clustering.bootstrap_stability,
    };
    Ok((
        DiarizationReport {
            implementation: match feature_ablation.schema_version() {
                AcousticFeatureSchemaVersion::V1 => "native-acoustic-v1",
                AcousticFeatureSchemaVersion::V2 => "native-acoustic-v2",
            }
            .to_owned(),
            contract_version: ACOUSTIC_DIARIZATION_CONTRACT_VERSION.to_owned(),
            feature_schema: feature_ablation.schema_version().id().to_owned(),
            normalized_input_sha256: normalized_input_sha256.to_owned(),
            hint_document_sha256: enrollment.hint_document_sha256,
            turns,
            profiles: clustering.profiles,
            hint_evidence,
            speaker_queries,
            speaker_count,
            fallback_status,
            diagnostics,
        },
        projection,
        change_evaluation_evidence,
        clustering_evaluation,
    ))
}

fn public_speaker_count_outcome(
    request: &SpeakerCountRequest,
    clustering: &AcousticClusteringResult,
) -> FwResult<SpeakerCountOutcome> {
    let supported_speaker_count = u32::try_from(clustering.detected_speakers).map_err(|_| {
        FwError::InvalidRequest("supported speaker count exceeds report schema".to_owned())
    })?;
    let mut active_speaker_refs = clustering
        .speaker_evidence
        .iter()
        .filter(|evidence| evidence.supported)
        .map(|evidence| evidence.speaker_ref.clone())
        .collect::<Vec<_>>();
    active_speaker_refs.sort();
    let speaker_evidence = clustering
        .speaker_evidence
        .iter()
        .map(|evidence| -> FwResult<_> {
            Ok(SpeakerEvidenceSummary {
                speaker_ref: evidence.speaker_ref.clone(),
                assigned_tracklet_count: report_count(
                    evidence.assigned_tracklet_count,
                    "assigned tracklet count",
                )?,
                independent_tracklet_count: report_count(
                    evidence.independent_tracklet_count,
                    "independent tracklet count",
                )?,
                recurrence_episode_count: report_count(
                    evidence.recurrence_episode_count,
                    "recurrence episode count",
                )?,
                voiced_frame_count: report_count(
                    evidence.voiced_frame_count,
                    "voiced frame count",
                )?,
                independent_voiced_frame_count: report_count(
                    evidence.independent_voiced_frame_count,
                    "independent voiced frame count",
                )?,
                voiced_duration_ms: evidence.voiced_duration_ms,
                mean_assignment_confidence: f64::from(evidence.mean_assignment_confidence),
                profile_reliability: f64::from(evidence.cluster_reliability),
                hard_anchored: evidence.hard_anchored,
                separated_from_supported_speakers: evidence.separated_from_supported_speakers,
                reasons: evidence.reasons.clone(),
                supported: evidence.supported,
            })
        })
        .collect::<FwResult<Vec<_>>>()?;
    let status = match request {
        SpeakerCountRequest::Infer
        | SpeakerCountRequest::Prior { .. }
        | SpeakerCountRequest::Range { .. }
            if clustering
                .count_estimate
                .as_ref()
                .and_then(|estimate| estimate.selected_count)
                .is_some()
                && supported_speaker_count > 0 =>
        {
            SpeakerCountOutcomeStatus::Resolved
        }
        SpeakerCountRequest::Infer
        | SpeakerCountRequest::Prior { .. }
        | SpeakerCountRequest::Range { .. } => SpeakerCountOutcomeStatus::Unresolved,
        SpeakerCountRequest::HardConstraint { .. } if clustering.constraints_satisfied => {
            SpeakerCountOutcomeStatus::Satisfied
        }
        SpeakerCountRequest::HardConstraint { .. } => SpeakerCountOutcomeStatus::Unsatisfied,
    };
    let mut reasons = Vec::new();
    match status {
        SpeakerCountOutcomeStatus::Resolved => {
            reasons.push(SpeakerCountOutcomeReason::EvidenceSupportedCount);
        }
        SpeakerCountOutcomeStatus::Satisfied => {
            reasons.push(SpeakerCountOutcomeReason::RequestedCountMatched);
        }
        SpeakerCountOutcomeStatus::Unsatisfied => {
            reasons.push(SpeakerCountOutcomeReason::RequestedCountMismatch);
        }
        SpeakerCountOutcomeStatus::Unresolved => {
            if supported_speaker_count == 0 {
                reasons.push(SpeakerCountOutcomeReason::NoSupportedSpeakers);
            } else if matches!(request, SpeakerCountRequest::Prior { .. })
                && clustering.count_estimate.is_none()
            {
                reasons.push(SpeakerCountOutcomeReason::SpeakerCountPriorFusionUnavailable);
            } else {
                reasons.push(SpeakerCountOutcomeReason::SpeakerCountEvidenceUnresolved);
            }
        }
    }
    if clustering.detected_speakers > 1
        && clustering.dominant_speaker_share > MAX_MULTI_SPEAKER_DOMINANT_SHARE
    {
        reasons.push(SpeakerCountOutcomeReason::DominantSpeakerShareExceeded);
    }
    if !clustering.speaker_separation_satisfied {
        reasons.push(SpeakerCountOutcomeReason::AmbiguousSpeakerSeparation);
    }
    Ok(SpeakerCountOutcome {
        request: request.clone(),
        estimate: clustering.count_estimate.clone(),
        status,
        supported_speaker_count,
        active_speaker_refs,
        dominant_speaker_share: f64::from(clustering.dominant_speaker_share),
        unknown_voiced_share: f64::from(clustering.unknown_voiced_share),
        reasons,
        speaker_evidence,
    })
}

fn public_hint_evidence(
    evidence: &[HintEnrollmentEvidence],
) -> FwResult<Vec<SpeakerHintEvidenceSummary>> {
    evidence
        .iter()
        .map(|hint| {
            let disposition = if hint.usable_tracklet_count == 0 {
                SpeakerHintDisposition::NoUsableTracklets
            } else if hint.policy == KnownSpeakerPolicy::HardMustLink {
                SpeakerHintDisposition::HardAttributed
            } else if hint.accepted_tracklet_count == 0 {
                SpeakerHintDisposition::Rejected
            } else if hint.accepted_tracklet_count < hint.usable_tracklet_count
                || hint.rejected_tracklet_count > 0
                || hint.profile_downweighted_tracklet_count > 0
                || hint.profile_quarantined_tracklet_count > 0
            {
                SpeakerHintDisposition::PartiallyAccepted
            } else {
                SpeakerHintDisposition::Accepted
            };
            Ok(SpeakerHintEvidenceSummary {
                hint_index: report_count(hint.hint_index, "hint index")?,
                speaker_ref: hint.speaker_ref.clone(),
                policy: hint.policy,
                disposition,
                usable_tracklet_count: report_count(
                    hint.usable_tracklet_count,
                    "usable hint tracklet count",
                )?,
                accepted_tracklet_count: report_count(
                    hint.accepted_tracklet_count,
                    "accepted hint tracklet count",
                )?,
                rejected_tracklet_count: report_count(
                    hint.rejected_tracklet_count,
                    "rejected hint tracklet count",
                )?,
                profile_accepted_tracklet_count: report_count(
                    hint.profile_accepted_tracklet_count,
                    "profile-accepted hint tracklet count",
                )?,
                profile_downweighted_tracklet_count: report_count(
                    hint.profile_downweighted_tracklet_count,
                    "profile-downweighted hint tracklet count",
                )?,
                profile_quarantined_tracklet_count: report_count(
                    hint.profile_quarantined_tracklet_count,
                    "profile-quarantined hint tracklet count",
                )?,
                applied_weight: f64::from(hint.applied_weight),
                contradiction_score: hint.contradiction_score.map(f64::from),
            })
        })
        .collect()
}

fn report_count(value: usize, field: &str) -> FwResult<u64> {
    u64::try_from(value).map_err(|_| {
        FwError::InvalidRequest(format!("{field} exceeds the diarization report schema"))
    })
}

#[derive(Debug, Clone)]
struct AcousticPrototype {
    members: Vec<usize>,
    voice: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_valid: [bool; VOICE_VECTOR_DIMENSIONS],
    variance: [f32; VOICE_VECTOR_DIMENSIONS],
    channel: [f32; CHANNEL_VECTOR_DIMENSIONS],
    channel_valid: bool,
    channel_dimensions: usize,
    frame_count: usize,
    earliest_ms: u64,
    hard_anchor: Option<String>,
}

#[derive(Debug, Clone)]
struct AcousticCluster {
    prototype_members: Vec<usize>,
    voice: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_valid: [bool; VOICE_VECTOR_DIMENSIONS],
    scale: [f32; VOICE_VECTOR_DIMENSIONS],
    channel: [f32; CHANNEL_VECTOR_DIMENSIONS],
    channel_valid: bool,
    channel_dimensions: usize,
    weight: f32,
    sse: f32,
    earliest_ms: u64,
    hard_anchor: Option<String>,
    label_hint: Option<String>,
    reliability: f32,
}

#[derive(Debug, Clone, Copy)]
struct MergeCandidate {
    distance: f32,
    left: usize,
    right: usize,
    left_generation: u32,
    right_generation: u32,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.to_bits() == other.distance.to_bits()
            && self.left == other.left
            && self.right == other.right
            && self.left_generation == other.left_generation
            && self.right_generation == other.right_generation
    }
}

impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
            .then_with(|| other.left_generation.cmp(&self.left_generation))
            .then_with(|| other.right_generation.cmp(&self.right_generation))
    }
}

/// Cluster tracklets with hard supervision, bounded prototypes, deterministic
/// agglomeration, Viterbi smoothing, and honest UNKNOWN rejection.
pub fn cluster_acoustic_tracklets<C>(
    tracklets: &[AcousticTracklet],
    enrollment: &SpeakerEnrollment,
    speaker_count: &SpeakerCountRequest,
    requested_prototype_cap: usize,
    is_cancelled: C,
) -> FwResult<AcousticClusteringResult>
where
    C: FnMut() -> bool,
{
    cluster_acoustic_tracklets_with_mode(
        tracklets,
        enrollment,
        speaker_count,
        requested_prototype_cap,
        AcousticClusteringMode::FixedSafeV1,
        is_cancelled,
    )
}

/// Execute one explicit clustering ablation.
pub fn cluster_acoustic_tracklets_with_mode<C>(
    tracklets: &[AcousticTracklet],
    enrollment: &SpeakerEnrollment,
    speaker_count: &SpeakerCountRequest,
    requested_prototype_cap: usize,
    requested_mode: AcousticClusteringMode,
    mut is_cancelled: C,
) -> FwResult<AcousticClusteringResult>
where
    C: FnMut() -> bool,
{
    if requested_prototype_cap == 0 || requested_prototype_cap > 512 {
        return Err(FwError::InvalidRequest(
            "acoustic-v2 prototype cap must be within 1..=512".to_owned(),
        ));
    }
    if is_cancelled() {
        return Err(FwError::Cancelled(
            "acoustic clustering cancelled before prototype construction".to_owned(),
        ));
    }
    validate_tracklet_timeline(tracklets)?;
    crate::model::validate_speaker_count_request(speaker_count)
        .map_err(|error| FwError::InvalidRequest(error.to_string()))?;
    let constraint_lower_bound = enrollment
        .hard_assignments
        .values()
        .collect::<BTreeSet<_>>()
        .len()
        .max(1);
    if tracklets
        .iter()
        .all(|tracklet| !tracklet.voice_valid.iter().any(|valid| *valid))
    {
        return Ok(AcousticClusteringResult {
            assignments: tracklets.iter().map(unknown_assignment).collect(),
            profiles: enrollment.summaries.clone(),
            count_estimate: unavailable_speaker_count_estimate(
                speaker_count,
                constraint_lower_bound,
                1,
                0,
                SpeakerCountCalibrationStatus::Unavailable,
                SpeakerCountLaneUnavailableReason::InsufficientVoicedEvidence,
            ),
            detected_speakers: 0,
            prototype_count: 0,
            prototype_cap: requested_prototype_cap,
            cap_pressure: false,
            constraints_satisfied: count_request_allows_zero(speaker_count),
            speaker_evidence: Vec::new(),
            dominant_speaker_share: 0.0,
            unknown_voiced_share: 0.0,
            speaker_separation_satisfied: true,
            bootstrap_stability: 0.0,
            requested_mode,
            executed_mode: AcousticClusteringMode::FixedSafeV1,
            fallback_reason: (requested_mode == AcousticClusteringMode::ProbabilisticV1)
                .then_some(AcousticClusteringFallbackReason::InsufficientSharedVoiceDimensions),
            speaker_pair_calibration_sha256: acoustic_speaker_pair_calibration_sha256(),
            calibration_status: "insufficient_identity_evidence",
            merge_trace: Vec::new(),
        });
    }
    let (prototypes, cap_pressure) = build_capped_prototypes(
        tracklets,
        &enrollment.hard_assignments,
        requested_prototype_cap,
        &mut is_cancelled,
    )?;
    if prototypes.is_empty() {
        return Ok(AcousticClusteringResult {
            assignments: tracklets.iter().map(unknown_assignment).collect(),
            profiles: enrollment.summaries.clone(),
            count_estimate: unavailable_speaker_count_estimate(
                speaker_count,
                constraint_lower_bound,
                1,
                0,
                SpeakerCountCalibrationStatus::Unavailable,
                SpeakerCountLaneUnavailableReason::InsufficientPrototypes,
            ),
            detected_speakers: 0,
            prototype_count: 0,
            prototype_cap: requested_prototype_cap,
            cap_pressure,
            constraints_satisfied: count_request_allows_zero(speaker_count),
            speaker_evidence: Vec::new(),
            dominant_speaker_share: 0.0,
            unknown_voiced_share: 0.0,
            speaker_separation_satisfied: true,
            bootstrap_stability: 0.0,
            requested_mode,
            executed_mode: AcousticClusteringMode::FixedSafeV1,
            fallback_reason: (requested_mode == AcousticClusteringMode::ProbabilisticV1)
                .then_some(AcousticClusteringFallbackReason::InsufficientSharedVoiceDimensions),
            speaker_pair_calibration_sha256: acoustic_speaker_pair_calibration_sha256(),
            calibration_status: "insufficient_evidence",
            merge_trace: Vec::new(),
        });
    }

    let initial_clusters = initial_clusters(&prototypes, enrollment);
    let requested_minimum = match speaker_count {
        SpeakerCountRequest::HardConstraint { count } => *count as usize,
        SpeakerCountRequest::Infer
        | SpeakerCountRequest::Prior { .. }
        | SpeakerCountRequest::Range { .. } => 1,
    };
    let count_constraints_feasible = requested_minimum <= initial_clusters.len();
    let mut count_policy = if count_constraints_feasible {
        resolve_count_policy(speaker_count, initial_clusters.len())?
    } else {
        SpeakerCountPolicy {
            min: initial_clusters.len(),
            max: initial_clusters.len(),
            exact: Some(initial_clusters.len()),
        }
    };
    count_policy.min = count_policy.min.max(constraint_lower_bound);
    let (
        clusters,
        merge_trace,
        bootstrap_stability,
        mut count_estimate,
        executed_mode,
        fallback_reason,
    ) = if requested_mode == AcousticClusteringMode::ProbabilisticV1 {
        match probabilistic_agglomerate_clusters(
            &initial_clusters,
            &enrollment.cannot_links,
            speaker_count,
            count_policy,
            &mut is_cancelled,
        )? {
            ProbabilisticAgglomeration::Selected {
                clusters,
                merge_trace,
                stability,
                count_estimate,
            } => (
                clusters,
                merge_trace,
                stability,
                Some(count_estimate),
                AcousticClusteringMode::ProbabilisticV1,
                None,
            ),
            ProbabilisticAgglomeration::Fallback {
                reason,
                count_estimate,
            } => {
                let unavailable_reason = match reason {
                    AcousticClusteringFallbackReason::InsufficientSharedVoiceDimensions => {
                        SpeakerCountLaneUnavailableReason::InsufficientIndependentReplicates
                    }
                    AcousticClusteringFallbackReason::InvalidPosterior => {
                        SpeakerCountLaneUnavailableReason::CalibrationUnavailable
                    }
                    AcousticClusteringFallbackReason::UnstableSpeakerCount => {
                        SpeakerCountLaneUnavailableReason::SolverDidNotConverge
                    }
                    AcousticClusteringFallbackReason::SpeakerCountPriorUnresolved => {
                        SpeakerCountLaneUnavailableReason::CalibrationUnavailable
                    }
                };
                let count_estimate = count_estimate.or_else(|| {
                    unavailable_speaker_count_estimate(
                        speaker_count,
                        constraint_lower_bound,
                        initial_clusters.len(),
                        initial_clusters.len(),
                        SpeakerCountCalibrationStatus::Unavailable,
                        unavailable_reason,
                    )
                });
                let (clusters, merge_trace) = agglomerate_clusters(
                    initial_clusters,
                    &enrollment.cannot_links,
                    count_policy,
                    &mut is_cancelled,
                )?;
                (
                    clusters,
                    merge_trace,
                    0.0,
                    count_estimate,
                    AcousticClusteringMode::FixedSafeV1,
                    Some(reason),
                )
            }
        }
    } else {
        let initial_cluster_count = initial_clusters.len();
        let (clusters, merge_trace) = agglomerate_clusters(
            initial_clusters,
            &enrollment.cannot_links,
            count_policy,
            &mut is_cancelled,
        )?;
        let count_estimate = unavailable_speaker_count_estimate(
            speaker_count,
            constraint_lower_bound,
            initial_cluster_count,
            initial_cluster_count,
            SpeakerCountCalibrationStatus::FixedSafeUncalibrated,
            SpeakerCountLaneUnavailableReason::CalibrationUnavailable,
        );
        (
            clusters,
            merge_trace,
            0.0,
            count_estimate,
            AcousticClusteringMode::FixedSafeV1,
            None,
        )
    };
    let labels = canonical_cluster_labels(&clusters, enrollment);
    let mut assignments = viterbi_assignments(
        tracklets,
        &clusters,
        &labels,
        enrollment,
        executed_mode,
        &mut is_cancelled,
    )?;
    let speaker_evidence =
        evaluate_speaker_evidence(tracklets, &clusters, &labels, enrollment, &assignments)?;
    retain_supported_assignments(&speaker_evidence, &mut assignments);
    let detected_speakers = assignments
        .iter()
        .flat_map(|assignment| {
            [
                assignment.speaker_ref.as_ref(),
                assignment.secondary_speaker_ref.as_ref(),
            ]
            .into_iter()
            .flatten()
        })
        .collect::<BTreeSet<_>>()
        .len();
    let supported_speakers = speaker_evidence
        .iter()
        .filter(|evidence| evidence.supported)
        .count();
    count_estimate = count_estimate.and_then(|estimate| {
        finalize_speaker_count_estimate(estimate, &speaker_evidence, supported_speakers)
    });
    let hard_constraints_satisfied =
        hard_assignments_satisfied(tracklets, enrollment, &assignments);
    let (dominant_speaker_share, unknown_voiced_share) =
        assignment_voiced_shares(tracklets, &assignments)?;
    let dominant_share_satisfied =
        detected_speakers <= 1 || dominant_speaker_share <= MAX_MULTI_SPEAKER_DOMINANT_SHARE;
    let speaker_separation_satisfied = speaker_evidence.iter().all(|evidence| {
        !evidence
            .reasons
            .contains(&SpeakerEvidenceReason::MergeCompatibleWithSupportedSpeaker)
    });
    let constraints_satisfied = count_constraints_feasible
        && hard_constraints_satisfied
        && dominant_share_satisfied
        && detected_speakers == supported_speakers
        && speaker_count_satisfies(detected_speakers, speaker_count);
    let supported_labels = speaker_evidence
        .iter()
        .filter(|evidence| evidence.supported)
        .map(|evidence| evidence.speaker_ref.as_str())
        .collect::<BTreeSet<_>>();
    let profiles = clustering_profile_summaries(&clusters, &labels, enrollment, &supported_labels);
    Ok(AcousticClusteringResult {
        assignments,
        profiles,
        count_estimate,
        detected_speakers,
        prototype_count: prototypes.len(),
        prototype_cap: requested_prototype_cap,
        cap_pressure,
        constraints_satisfied,
        speaker_evidence,
        dominant_speaker_share,
        unknown_voiced_share,
        speaker_separation_satisfied,
        bootstrap_stability,
        requested_mode,
        executed_mode,
        fallback_reason,
        speaker_pair_calibration_sha256: acoustic_speaker_pair_calibration_sha256(),
        calibration_status: if executed_mode == AcousticClusteringMode::ProbabilisticV1 {
            "development_posterior_uncertified"
        } else {
            "heuristic_uncalibrated"
        },
        merge_trace,
    })
}

/// Convert smoothed acoustic assignments into a finite, monotonic turn
/// timeline independent of Whisper segment boundaries.
pub fn diarization_turns_from_assignments(
    assignments: &[AcousticSpeakerAssignment],
    audio_duration_ms: u64,
) -> FwResult<Vec<DiarizationTurn>> {
    let mut turns = Vec::<DiarizationTurn>::with_capacity(assignments.len());
    let mut secondary_turns = Vec::<DiarizationTurn>::new();
    let mut previous_start = 0u64;
    for (index, assignment) in assignments.iter().enumerate() {
        let secondary_is_valid = match (
            assignment.secondary_speaker_ref.as_ref(),
            assignment.secondary_speaker_confidence,
        ) {
            (None, None) => true,
            (Some(secondary), Some(confidence)) => {
                assignment.overlap_suspected
                    && assignment.speaker_ref.is_some()
                    && assignment.speaker_ref.as_ref() != Some(secondary)
                    && confidence.is_finite()
                    && (0.0..=1.0).contains(&confidence)
            }
            _ => false,
        };
        if assignment.end_ms <= assignment.start_ms
            || assignment.end_ms > audio_duration_ms
            || (index > 0 && assignment.start_ms < previous_start)
            || !assignment.speaker_confidence.is_finite()
            || !(0.0..=1.0).contains(&assignment.speaker_confidence)
            || !assignment.change_confidence.is_finite()
            || !(0.0..=1.0).contains(&assignment.change_confidence)
            || !secondary_is_valid
        {
            return Err(FwError::InvalidRequest(
                "acoustic assignments must be finite, bounded, and time-ordered".to_owned(),
            ));
        }
        let mut turn = DiarizationTurn {
            start_ms: assignment.start_ms,
            end_ms: assignment.end_ms,
            speaker_ref: assignment.speaker_ref.clone(),
            speaker_confidence: assignment.speaker_ref.as_ref().map(|_| {
                if assignment.hard_attribution {
                    1.0
                } else {
                    f64::from(assignment.speaker_confidence)
                }
            }),
            change_confidence: Some(f64::from(assignment.change_confidence)),
            overlap_suspected: assignment.overlap_suspected,
            hard_hint_attributed: assignment.hard_attribution,
        };
        previous_start = assignment.start_ms;

        let mut primary_merged = false;
        if let Some(previous) = turns.last_mut() {
            if turn.start_ms < previous.end_ms {
                if turn.start_ms < previous.end_ms.saturating_sub(25) {
                    return Err(FwError::InvalidRequest(
                        "acoustic assignments overlap beyond one analysis frame".to_owned(),
                    ));
                }
                let boundary = turn.start_ms + (previous.end_ms - turn.start_ms) / 2;
                if boundary <= previous.start_ms || boundary >= turn.end_ms {
                    return Err(FwError::InvalidRequest(
                        "acoustic assignment overlap leaves no positive turn duration".to_owned(),
                    ));
                }
                previous.end_ms = boundary;
                turn.start_ms = boundary;
            }
            if previous.speaker_ref == turn.speaker_ref
                && turn.start_ms <= previous.end_ms.saturating_add(25)
                && previous.hard_hint_attributed == turn.hard_hint_attributed
                && previous.overlap_suspected == turn.overlap_suspected
            {
                previous.end_ms = previous.end_ms.max(turn.end_ms);
                previous.speaker_confidence = minimum_optional_confidence(
                    previous.speaker_confidence,
                    turn.speaker_confidence,
                );
                previous.change_confidence =
                    maximum_optional_confidence(previous.change_confidence, turn.change_confidence);
                primary_merged = true;
            }
        }
        if !primary_merged {
            turns.push(turn);
        }

        if let (Some(speaker_ref), Some(confidence)) = (
            assignment.secondary_speaker_ref.as_ref(),
            assignment.secondary_speaker_confidence,
        ) {
            let secondary = DiarizationTurn {
                start_ms: assignment.start_ms,
                end_ms: assignment.end_ms,
                speaker_ref: Some(speaker_ref.clone()),
                speaker_confidence: Some(f64::from(confidence)),
                change_confidence: Some(f64::from(assignment.change_confidence)),
                overlap_suspected: true,
                hard_hint_attributed: false,
            };
            if let Some(previous) = secondary_turns.last_mut()
                && previous.speaker_ref == secondary.speaker_ref
                && secondary.start_ms <= previous.end_ms.saturating_add(25)
            {
                previous.end_ms = previous.end_ms.max(secondary.end_ms);
                previous.speaker_confidence = minimum_optional_confidence(
                    previous.speaker_confidence,
                    secondary.speaker_confidence,
                );
                previous.change_confidence = maximum_optional_confidence(
                    previous.change_confidence,
                    secondary.change_confidence,
                );
                continue;
            }
            secondary_turns.push(secondary);
        }
    }
    turns.extend(secondary_turns);
    turns.sort_by(|left, right| {
        left.start_ms
            .cmp(&right.start_ms)
            .then(left.end_ms.cmp(&right.end_ms))
            .then(left.speaker_ref.cmp(&right.speaker_ref))
    });
    Ok(turns)
}

fn speaker_attribution_queries(
    assignments: &[AcousticSpeakerAssignment],
    normalized_input_sha256: &str,
) -> Vec<SpeakerAttributionQuery> {
    const MAX_AGENT_QUERIES: usize = 32;
    let mut queries = Vec::<SpeakerAttributionQuery>::new();
    for assignment in assignments {
        if assignment.hard_attribution {
            continue;
        }
        let reason = if assignment.speaker_ref.is_none() {
            Some(SpeakerAttributionQueryReason::UnknownAttribution)
        } else if assignment.secondary_speaker_ref.is_some() {
            Some(SpeakerAttributionQueryReason::OverlapAmbiguity)
        } else if assignment.speaker_confidence < 0.60 {
            Some(SpeakerAttributionQueryReason::LowConfidence)
        } else {
            None
        };
        let Some(reason) = reason else {
            continue;
        };
        let mut candidate_speaker_refs = [
            assignment.speaker_ref.clone(),
            assignment.secondary_speaker_ref.clone(),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        candidate_speaker_refs.sort();
        candidate_speaker_refs.dedup();
        if let Some(previous) = queries.last_mut()
            && previous.reason == reason
            && previous.candidate_speaker_refs == candidate_speaker_refs
            && assignment.start_ms <= previous.end_ms.saturating_add(50)
        {
            previous.end_ms = previous.end_ms.max(assignment.end_ms);
            continue;
        }
        if queries.len() == MAX_AGENT_QUERIES {
            continue;
        }
        queries.push(SpeakerAttributionQuery {
            query_id_sha256: String::new(),
            start_ms: assignment.start_ms,
            end_ms: assignment.end_ms,
            reason,
            candidate_speaker_refs,
            suggested_policy: KnownSpeakerPolicy::SoftEnrollment,
        });
    }
    for query in &mut queries {
        query.query_id_sha256 = speaker_attribution_query_sha256(normalized_input_sha256, query);
    }
    queries
}

fn speaker_attribution_query_sha256(
    normalized_input_sha256: &str,
    query: &SpeakerAttributionQuery,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"speaker-attribution-query-v1\0");
    hash_field(&mut hasher, normalized_input_sha256.as_bytes());
    hasher.update(query.start_ms.to_le_bytes());
    hasher.update(query.end_ms.to_le_bytes());
    hasher.update([match query.reason {
        SpeakerAttributionQueryReason::UnknownAttribution => 0,
        SpeakerAttributionQueryReason::LowConfidence => 1,
        SpeakerAttributionQueryReason::OverlapAmbiguity => 2,
    }]);
    for speaker_ref in &query.candidate_speaker_refs {
        hash_field(&mut hasher, speaker_ref.as_bytes());
    }
    hasher.update([policy_rank(query.suggested_policy)]);
    format!("{:x}", hasher.finalize())
}

fn minimum_optional_confidence(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        _ => None,
    }
}

fn maximum_optional_confidence(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

/// Project an immutable acoustic turn timeline onto existing transcript
/// segments without modifying text or ASR confidence.
///
/// When `word_aligned` is true, each input segment is treated as a legal DTW
/// word unit. Otherwise a segment receives a label only when one speaker owns
/// at least 70% of its duration at calibrated-enough confidence.
pub fn project_diarization_onto_segments(
    segments: &[TranscriptionSegment],
    turns: &[DiarizationTurn],
    word_aligned: bool,
) -> FwResult<DiarizationProjection> {
    validate_diarization_turns(turns)?;
    validate_projection_segments(segments, word_aligned)?;

    let mut projected = Vec::with_capacity(segments.len());
    let mut mixed = Vec::new();
    let mut overlaps = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        let mut output = segment.clone();
        output.speaker = None;
        let Some((start_sec, end_sec)) = segment.start_sec.zip(segment.end_sec) else {
            if distinct_known_speakers(turns) > 1 {
                mixed.push(index);
            }
            projected.push(output);
            continue;
        };
        let overlapping = overlapping_turns(start_sec, end_sec, turns);
        if overlapping.iter().any(|(_, turn)| turn.overlap_suspected) {
            overlaps.push(index);
        }
        if word_aligned {
            output.speaker = choose_word_speaker(&overlapping);
        } else {
            let (speaker, is_mixed) =
                choose_dominant_segment_speaker(start_sec, end_sec, &overlapping);
            output.speaker = speaker;
            if is_mixed {
                mixed.push(index);
            }
        }
        projected.push(output);
    }
    Ok(DiarizationProjection {
        segments: projected,
        mixed_speaker_segment_indices: mixed,
        overlap_suspected_segment_indices: overlaps,
    })
}

fn validate_diarization_turns(turns: &[DiarizationTurn]) -> FwResult<()> {
    let mut previous_end = 0u64;
    for turn in turns {
        let valid_confidence = |value: Option<f64>| {
            value.is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        };
        if turn.end_ms <= turn.start_ms
            || turn.start_ms < previous_end
            || turn
                .speaker_ref
                .as_ref()
                .is_some_and(|speaker| speaker.trim().is_empty())
            || (turn.speaker_ref.is_none() && turn.speaker_confidence.is_some())
            || !valid_confidence(turn.speaker_confidence)
            || !valid_confidence(turn.change_confidence)
        {
            return Err(FwError::InvalidRequest(
                "diarization turns must be finite, labeled consistently, and monotonic".to_owned(),
            ));
        }
        previous_end = turn.end_ms;
    }
    Ok(())
}

fn validate_projection_segments(
    segments: &[TranscriptionSegment],
    word_aligned: bool,
) -> FwResult<()> {
    let mut previous_start = 0.0_f64;
    let mut previous_end = 0.0_f64;
    for (index, segment) in segments.iter().enumerate() {
        match (segment.start_sec, segment.end_sec) {
            (Some(start), Some(end))
                if start.is_finite()
                    && end.is_finite()
                    && start >= 0.0
                    && end > start
                    && (index == 0 || start >= previous_start)
                    && (!word_aligned
                        || index == 0
                        || start + CANONICAL_PROJECTION_EPSILON_SEC >= previous_end) =>
            {
                previous_start = start;
                previous_end = end;
            }
            (None, None) => {}
            _ => {
                return Err(FwError::InvalidRequest(
                    "projection segments must have paired finite timestamps in monotonic order"
                        .to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn overlapping_turns(
    start_sec: f64,
    end_sec: f64,
    turns: &[DiarizationTurn],
) -> Vec<(f64, &DiarizationTurn)> {
    turns
        .iter()
        .filter_map(|turn| {
            let turn_start = turn.start_ms as f64 / 1_000.0;
            let turn_end = turn.end_ms as f64 / 1_000.0;
            let overlap = end_sec.min(turn_end) - start_sec.max(turn_start);
            (overlap > 0.0).then_some((overlap, turn))
        })
        .collect()
}

fn choose_word_speaker(overlapping: &[(f64, &DiarizationTurn)]) -> Option<String> {
    const MIN_SPEAKER_CONFIDENCE: f64 = 0.30;
    let mut ranked = overlapping
        .iter()
        .filter(|(_, turn)| turn.speaker_ref.is_some())
        .collect::<Vec<_>>();
    ranked.sort_by(|(left_overlap, left), (right_overlap, right)| {
        right_overlap
            .total_cmp(left_overlap)
            .then(left.start_ms.cmp(&right.start_ms))
            .then(left.speaker_ref.cmp(&right.speaker_ref))
    });
    let (_, best) = ranked.first().copied()?;
    let confidence = best.speaker_confidence.unwrap_or(0.0);
    if confidence < MIN_SPEAKER_CONFIDENCE {
        return None;
    }
    best.speaker_ref.clone()
}

fn choose_dominant_segment_speaker(
    start_sec: f64,
    end_sec: f64,
    overlapping: &[(f64, &DiarizationTurn)],
) -> (Option<String>, bool) {
    const MIN_DOMINANCE: f64 = 0.70;
    const MIN_SPEAKER_CONFIDENCE: f64 = 0.30;
    let mut totals = BTreeMap::<String, (f64, f64)>::new();
    for (duration, turn) in overlapping {
        let Some(speaker) = &turn.speaker_ref else {
            continue;
        };
        let entry = totals.entry(speaker.clone()).or_default();
        entry.0 += duration;
        entry.1 += duration * turn.speaker_confidence.unwrap_or(0.0);
    }
    let mut ranked = totals.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|(left_label, left), (right_label, right)| {
        right.0.total_cmp(&left.0).then(left_label.cmp(right_label))
    });
    let Some((speaker, (duration, weighted_confidence))) = ranked.first() else {
        return (None, false);
    };
    let segment_duration = end_sec - start_sec;
    let confidence = weighted_confidence / duration.max(f64::EPSILON);
    let dominant = duration / segment_duration >= MIN_DOMINANCE
        && confidence >= MIN_SPEAKER_CONFIDENCE
        && ranked
            .get(1)
            .is_none_or(|(_, (second_duration, _))| duration > second_duration);
    if dominant {
        (Some(speaker.clone()), false)
    } else {
        (None, ranked.len() > 1)
    }
}

fn distinct_known_speakers(turns: &[DiarizationTurn]) -> usize {
    turns
        .iter()
        .filter_map(|turn| turn.speaker_ref.as_deref())
        .collect::<BTreeSet<_>>()
        .len()
}

fn validate_tracklet_timeline(tracklets: &[AcousticTracklet]) -> FwResult<()> {
    let mut previous_end = 0u64;
    let mut indexes = BTreeSet::new();
    let mut total_frame_count = 0usize;
    let maximum_total_frames = usize::try_from(u64::MAX / 10).unwrap_or(usize::MAX);
    for tracklet in tracklets {
        let Some(next_total_frame_count) = total_frame_count.checked_add(tracklet.frame_count)
        else {
            return Err(FwError::InvalidRequest(
                "tracklets must have finite non-negative statistics within acoustic-v2 bounds, valid counts, masks and confidence, unique indexes, and ordered positive intervals".to_owned(),
            ));
        };
        if next_total_frame_count > maximum_total_frames
            || tracklet.end_ms <= tracklet.start_ms
            || tracklet.start_ms < previous_end.saturating_sub(25)
            || tracklet.end_ms < previous_end
            || tracklet.frame_count == 0
            || tracklet.voiced_frame_count > tracklet.frame_count
            || tracklet.identity_frame_count > tracklet.voiced_frame_count
            || tracklet.channel_frame_count > tracklet.frame_count
            || tracklet.channel_valid != (tracklet.channel_frame_count > 0)
            || tracklet.channel_dimensions > CHANNEL_VECTOR_DIMENSIONS
            || tracklet.channel_valid != (tracklet.channel_dimensions > 0)
            || tracklet
                .voice_valid
                .iter()
                .zip(tracklet.voice_support)
                .any(|(&valid, support)| valid != (support > 0))
            || !indexes.insert(tracklet.tracklet_index)
            || !tracklet.change_confidence.is_finite()
            || !(0.0..=1.0).contains(&tracklet.change_confidence)
            || !tracklet.overlap_probability.is_finite()
            || !(0.0..=1.0).contains(&tracklet.overlap_probability)
            || tracklet
                .voice_mean
                .iter()
                .chain(tracklet.channel_mean.iter())
                .any(|value| !value.is_finite() || value.abs() > MAX_ABS_ACOUSTIC_FEATURE)
            || tracklet
                .voice_variance
                .iter()
                .chain(tracklet.channel_variance.iter())
                .any(|value| !value.is_finite() || *value < 0.0 || *value > MAX_ACOUSTIC_VARIANCE)
        {
            return Err(FwError::InvalidRequest(
                "tracklets must have finite non-negative statistics within acoustic-v2 bounds, valid counts, masks and confidence, unique indexes, and ordered positive intervals".to_owned(),
            ));
        }
        previous_end = tracklet.end_ms;
        total_frame_count = next_total_frame_count;
    }
    Ok(())
}

fn build_capped_prototypes<C>(
    tracklets: &[AcousticTracklet],
    hard_assignments: &BTreeMap<usize, String>,
    cap: usize,
    is_cancelled: &mut C,
) -> FwResult<(Vec<AcousticPrototype>, bool)>
where
    C: FnMut() -> bool,
{
    let mut ordered = tracklets.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.start_ms
            .cmp(&right.start_ms)
            .then(left.end_ms.cmp(&right.end_ms))
            .then_with(|| compare_float_vectors(&left.voice_mean, &right.voice_mean))
            .then(left.tracklet_index.cmp(&right.tracklet_index))
    });
    let mut prototypes = Vec::<AcousticPrototype>::with_capacity(cap);
    let mut cap_pressure = false;
    for (position, tracklet) in ordered.into_iter().enumerate() {
        if position % ACOUSTIC_CANCELLATION_INTERVAL_FRAMES == 0 && is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "acoustic clustering cancelled at tracklet {position}"
            )));
        }
        let prototype = AcousticPrototype {
            members: vec![tracklet.tracklet_index],
            voice: tracklet.voice_mean,
            voice_valid: tracklet.voice_valid,
            variance: tracklet.voice_variance.map(|value| value.max(0.025)),
            channel: tracklet.channel_mean,
            channel_valid: tracklet.channel_valid,
            channel_dimensions: tracklet.channel_dimensions,
            frame_count: tracklet.frame_count,
            earliest_ms: tracklet.start_ms,
            hard_anchor: hard_assignments.get(&tracklet.tracklet_index).cloned(),
        };
        if prototypes.len() < cap {
            prototypes.push(prototype);
            continue;
        }
        cap_pressure = true;
        let selected = prototypes
            .iter()
            .enumerate()
            .filter(|(_, existing)| {
                anchors_compatible(
                    existing.hard_anchor.as_ref(),
                    prototype.hard_anchor.as_ref(),
                )
            })
            .map(|(index, existing)| (prototype_distance(existing, &prototype), index))
            .min_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)))
            .map(|(_, index)| index)
            .ok_or_else(|| {
                FwError::InvalidRequest(
                    "prototype cap cannot preserve all incompatible hard anchors".to_owned(),
                )
            })?;
        merge_prototype(&mut prototypes[selected], &prototype);
    }
    Ok((prototypes, cap_pressure))
}

fn compare_float_vectors<const N: usize>(left: &[f32; N], right: &[f32; N]) -> std::cmp::Ordering {
    for index in 0..N {
        let ordering = left[index].total_cmp(&right[index]);
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }
    }
    std::cmp::Ordering::Equal
}

fn anchors_compatible(left: Option<&String>, right: Option<&String>) -> bool {
    left.is_none() || right.is_none() || left == right
}

fn prototype_distance(left: &AcousticPrototype, right: &AcousticPrototype) -> f32 {
    masked_variance_normalized_distance(
        &left.voice,
        &left.voice_valid,
        &left.variance,
        &right.voice,
        &right.voice_valid,
        &right.variance,
    )
    .unwrap_or(10.0)
        + channel_distance(
            &left.channel,
            left.channel_valid,
            &right.channel,
            right.channel_valid,
            left.channel_dimensions.min(right.channel_dimensions),
        )
}

fn merge_prototype(destination: &mut AcousticPrototype, source: &AcousticPrototype) {
    let total = destination.frame_count + source.frame_count;
    let left_weight = destination.frame_count as f32 / total as f32;
    let right_weight = source.frame_count as f32 / total as f32;
    for dimension in 0..VOICE_VECTOR_DIMENSIONS {
        if !destination.voice_valid[dimension] && !source.voice_valid[dimension] {
            continue;
        }
        if !destination.voice_valid[dimension] {
            destination.voice[dimension] = source.voice[dimension];
            destination.variance[dimension] = source.variance[dimension];
            destination.voice_valid[dimension] = true;
            continue;
        }
        if !source.voice_valid[dimension] {
            continue;
        }
        let delta = destination.voice[dimension] - source.voice[dimension];
        destination.voice[dimension] =
            left_weight * destination.voice[dimension] + right_weight * source.voice[dimension];
        destination.variance[dimension] = left_weight * destination.variance[dimension]
            + right_weight * source.variance[dimension]
            + left_weight * right_weight * delta * delta;
    }
    if !destination.channel_valid && source.channel_valid {
        destination.channel = source.channel;
        destination.channel_valid = true;
        destination.channel_dimensions = source.channel_dimensions;
    } else if destination.channel_valid && source.channel_valid {
        destination.channel_dimensions = destination
            .channel_dimensions
            .min(source.channel_dimensions);
        for dimension in 0..destination.channel_dimensions {
            destination.channel[dimension] = left_weight * destination.channel[dimension]
                + right_weight * source.channel[dimension];
        }
    }
    destination.members.extend_from_slice(&source.members);
    destination.members.sort_unstable();
    destination.frame_count = total;
    destination.earliest_ms = destination.earliest_ms.min(source.earliest_ms);
    if destination.hard_anchor.is_none() {
        destination.hard_anchor.clone_from(&source.hard_anchor);
    }
}

fn masked_variance_normalized_distance<const N: usize>(
    left: &[f32; N],
    left_valid: &[bool; N],
    left_scale: &[f32; N],
    right: &[f32; N],
    right_valid: &[bool; N],
    right_scale: &[f32; N],
) -> Option<f32> {
    let mut total = 0.0_f32;
    let mut active = 0usize;
    for dimension in 0..N {
        if left_valid[dimension] && right_valid[dimension] {
            let difference = left[dimension] - right[dimension];
            total +=
                difference * difference / (left_scale[dimension] + right_scale[dimension] + 0.05);
            active += 1;
        }
    }
    (active > 0).then(|| (total / active as f32).sqrt())
}

fn channel_distance<const N: usize>(
    left: &[f32; N],
    left_valid: bool,
    right: &[f32; N],
    right_valid: bool,
    dimensions: usize,
) -> f32 {
    CHANNEL_DISTANCE_WEIGHT * raw_channel_distance(left, left_valid, right, right_valid, dimensions)
}

fn raw_channel_distance<const N: usize>(
    left: &[f32; N],
    left_valid: bool,
    right: &[f32; N],
    right_valid: bool,
    dimensions: usize,
) -> f32 {
    if left_valid && right_valid && dimensions > 0 {
        euclidean_distance_prefix(left, right, dimensions).min(1.0)
    } else {
        0.0
    }
}

fn initial_clusters(
    prototypes: &[AcousticPrototype],
    enrollment: &SpeakerEnrollment,
) -> Vec<AcousticCluster> {
    let mut anchored = BTreeMap::<String, AcousticCluster>::new();
    let mut clusters = Vec::new();
    for (prototype_index, prototype) in prototypes.iter().enumerate() {
        let cluster = cluster_from_prototype(prototype_index, prototype);
        if let Some(anchor) = &prototype.hard_anchor {
            anchored
                .entry(anchor.clone())
                .and_modify(|existing| merge_cluster(existing, &cluster))
                .or_insert(cluster);
        } else {
            clusters.push(cluster);
        }
    }
    for (speaker_ref, mut cluster) in anchored {
        if let Some(profile) = enrollment.profiles.get(&speaker_ref) {
            apply_profile_prior(&mut cluster, profile);
        }
        cluster.hard_anchor = Some(speaker_ref.clone());
        cluster.label_hint = Some(speaker_ref);
        clusters.push(cluster);
    }
    for profile in enrollment
        .profiles
        .values()
        .filter(|profile| !profile.anchored)
    {
        clusters.push(cluster_from_profile(profile));
    }
    clusters.sort_by(|left, right| {
        left.hard_anchor
            .is_none()
            .cmp(&right.hard_anchor.is_none())
            .then(left.earliest_ms.cmp(&right.earliest_ms))
            .then_with(|| compare_float_vectors(&left.voice, &right.voice))
    });
    clusters
}

fn cluster_from_prototype(index: usize, prototype: &AcousticPrototype) -> AcousticCluster {
    AcousticCluster {
        prototype_members: vec![index],
        voice: prototype.voice,
        voice_valid: prototype.voice_valid,
        scale: prototype.variance.map(|value| value.max(0.025)),
        channel: prototype.channel,
        channel_valid: prototype.channel_valid,
        channel_dimensions: prototype.channel_dimensions,
        weight: prototype.frame_count as f32,
        sse: 0.0,
        earliest_ms: prototype.earliest_ms,
        hard_anchor: prototype.hard_anchor.clone(),
        label_hint: prototype.hard_anchor.clone(),
        reliability: (prototype.frame_count as f32 / 100.0).min(1.0),
    }
}

fn cluster_from_profile(profile: &AcousticSpeakerProfile) -> AcousticCluster {
    let mut scale = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    for (dimension, scale_value) in scale.iter_mut().enumerate() {
        let iqr = (profile.voice_q75[dimension] - profile.voice_q25[dimension]).abs() / 1.349;
        *scale_value = profile.voice_mad[dimension].max(iqr).max(0.025);
    }
    AcousticCluster {
        prototype_members: Vec::new(),
        voice: profile.voice_median,
        voice_valid: profile.voice_valid,
        scale,
        channel: profile
            .channel_subprofiles
            .first()
            .copied()
            .unwrap_or([0.0; CHANNEL_VECTOR_DIMENSIONS]),
        channel_valid: !profile.channel_subprofiles.is_empty(),
        channel_dimensions: profile.channel_dimensions,
        weight: (profile.frame_count as f32).clamp(1.0, 100.0),
        sse: 0.0,
        earliest_ms: u64::MAX,
        hard_anchor: profile.anchored.then(|| profile.speaker_ref.clone()),
        label_hint: Some(profile.speaker_ref.clone()),
        reliability: profile.reliability,
    }
}

fn apply_profile_prior(cluster: &mut AcousticCluster, profile: &AcousticSpeakerProfile) {
    let prior = cluster_from_profile(profile);
    merge_cluster(cluster, &prior);
    cluster.reliability = cluster.reliability.max(profile.reliability);
}

fn merge_cluster(destination: &mut AcousticCluster, source: &AcousticCluster) {
    let total = destination.weight + source.weight;
    let left_weight = destination.weight / total.max(f32::EPSILON);
    let right_weight = source.weight / total.max(f32::EPSILON);
    let merge_distance = cluster_distance(destination, source);
    for dimension in 0..VOICE_VECTOR_DIMENSIONS {
        if !destination.voice_valid[dimension] && !source.voice_valid[dimension] {
            continue;
        }
        if !destination.voice_valid[dimension] {
            destination.voice[dimension] = source.voice[dimension];
            destination.scale[dimension] = source.scale[dimension];
            destination.voice_valid[dimension] = true;
            continue;
        }
        if !source.voice_valid[dimension] {
            continue;
        }
        let delta = destination.voice[dimension] - source.voice[dimension];
        destination.voice[dimension] =
            left_weight * destination.voice[dimension] + right_weight * source.voice[dimension];
        destination.scale[dimension] = left_weight * destination.scale[dimension]
            + right_weight * source.scale[dimension]
            + left_weight * right_weight * delta * delta;
    }
    if !destination.channel_valid && source.channel_valid {
        destination.channel = source.channel;
        destination.channel_valid = true;
        destination.channel_dimensions = source.channel_dimensions;
    } else if destination.channel_valid && source.channel_valid {
        destination.channel_dimensions = destination
            .channel_dimensions
            .min(source.channel_dimensions);
        for dimension in 0..destination.channel_dimensions {
            destination.channel[dimension] = left_weight * destination.channel[dimension]
                + right_weight * source.channel[dimension];
        }
    }
    destination
        .prototype_members
        .extend_from_slice(&source.prototype_members);
    destination.prototype_members.sort_unstable();
    destination.prototype_members.dedup();
    destination.sse += source.sse
        + destination.weight * source.weight / total.max(f32::EPSILON)
            * merge_distance
            * merge_distance;
    destination.weight = total;
    destination.earliest_ms = destination.earliest_ms.min(source.earliest_ms);
    destination.reliability =
        left_weight * destination.reliability + right_weight * source.reliability;
    if destination.hard_anchor.is_none() {
        destination.hard_anchor.clone_from(&source.hard_anchor);
    }
    if destination.label_hint.is_none() {
        destination.label_hint.clone_from(&source.label_hint);
    }
}

fn cluster_distance(left: &AcousticCluster, right: &AcousticCluster) -> f32 {
    masked_variance_normalized_distance(
        &left.voice,
        &left.voice_valid,
        &left.scale,
        &right.voice,
        &right.voice_valid,
        &right.scale,
    )
    .unwrap_or(10.0)
        + channel_distance(
            &left.channel,
            left.channel_valid,
            &right.channel,
            right.channel_valid,
            left.channel_dimensions.min(right.channel_dimensions),
        )
}

#[derive(Debug, Clone, Copy)]
enum SpeakerPairPerturbation {
    Full,
    NoPitchCoordinates,
    NoDynamicCoordinates,
    NoFormantCoordinates,
    NoChannelEvidence,
}

impl SpeakerPairPerturbation {
    const ALL: [Self; SPEAKER_COUNT_PERTURBATION_LANES] = [
        Self::Full,
        Self::NoPitchCoordinates,
        Self::NoDynamicCoordinates,
        Self::NoFormantCoordinates,
        Self::NoChannelEvidence,
    ];

    const fn includes(self, dimension: usize) -> bool {
        match self {
            Self::Full | Self::NoChannelEvidence => true,
            Self::NoPitchCoordinates => dimension != 20 && dimension != 26,
            Self::NoDynamicCoordinates => dimension < 12 || (dimension >= 20 && dimension < 27),
            Self::NoFormantCoordinates => !(dimension >= 23 && dimension < 26),
        }
    }

    const fn includes_channel(self) -> bool {
        !matches!(self, Self::NoChannelEvidence)
    }
}

fn cluster_pair_evidence(
    left: &AcousticCluster,
    right: &AcousticCluster,
    perturbation: SpeakerPairPerturbation,
) -> Option<AcousticSpeakerPairEvidence> {
    speaker_pair_evidence_from_statistics(
        &left.voice,
        &left.voice_valid,
        &left.scale,
        left.weight,
        &left.channel,
        left.channel_valid,
        left.channel_dimensions,
        &right.voice,
        &right.voice_valid,
        &right.scale,
        right.weight,
        &right.channel,
        right.channel_valid,
        right.channel_dimensions,
        perturbation,
    )
}

#[allow(clippy::too_many_arguments)]
fn speaker_pair_evidence_from_statistics(
    left_voice: &[f32; VOICE_VECTOR_DIMENSIONS],
    left_voice_valid: &[bool; VOICE_VECTOR_DIMENSIONS],
    left_scale: &[f32; VOICE_VECTOR_DIMENSIONS],
    left_weight: f32,
    left_channel: &[f32; CHANNEL_VECTOR_DIMENSIONS],
    left_channel_valid: bool,
    left_channel_dimensions: usize,
    right_voice: &[f32; VOICE_VECTOR_DIMENSIONS],
    right_voice_valid: &[bool; VOICE_VECTOR_DIMENSIONS],
    right_scale: &[f32; VOICE_VECTOR_DIMENSIONS],
    right_weight: f32,
    right_channel: &[f32; CHANNEL_VECTOR_DIMENSIONS],
    right_channel_valid: bool,
    right_channel_dimensions: usize,
    perturbation: SpeakerPairPerturbation,
) -> Option<AcousticSpeakerPairEvidence> {
    let calibration = acoustic_speaker_pair_calibration();
    let mut squared_distance = 0.0_f32;
    let mut active_voice_dimensions = 0usize;
    for dimension in 0..VOICE_VECTOR_DIMENSIONS {
        if !perturbation.includes(dimension)
            || !left_voice_valid[dimension]
            || !right_voice_valid[dimension]
        {
            continue;
        }
        let variance = left_scale[dimension] + right_scale[dimension] + calibration.variance_floor;
        if !variance.is_finite() || variance <= 0.0 {
            return None;
        }
        let difference = left_voice[dimension] - right_voice[dimension];
        squared_distance += difference * difference / variance;
        active_voice_dimensions += 1;
    }
    if active_voice_dimensions < SPEAKER_PAIR_MINIMUM_ACTIVE_DIMENSIONS {
        return None;
    }
    let voice_distance = (squared_distance / active_voice_dimensions as f32).sqrt();
    let channel_distance = if perturbation.includes_channel() {
        raw_channel_distance(
            left_channel,
            left_channel_valid,
            right_channel,
            right_channel_valid,
            left_channel_dimensions.min(right_channel_dimensions),
        )
    } else {
        0.0
    };
    let support = (left_weight.min(right_weight) / calibration.full_support_frames).clamp(0.0, 1.0);
    let different_log_odds = support
        * (calibration.different_logit_intercept
            + calibration.voice_distance_weight * voice_distance)
        + calibration.channel_distance_weight * channel_distance;
    let same_speaker_probability = logistic_probability(-different_log_odds);
    if !voice_distance.is_finite()
        || !channel_distance.is_finite()
        || !different_log_odds.is_finite()
        || !same_speaker_probability.is_finite()
        || !(0.0..=1.0).contains(&same_speaker_probability)
    {
        return None;
    }
    Some(AcousticSpeakerPairEvidence {
        voice_distance,
        channel_distance,
        different_log_odds,
        same_speaker_probability,
        active_voice_dimensions,
        support,
    })
}

#[derive(Debug, Clone, Copy)]
struct SpeakerCountPolicy {
    min: usize,
    max: usize,
    exact: Option<usize>,
}

fn resolve_count_policy(
    request: &SpeakerCountRequest,
    available: usize,
) -> FwResult<SpeakerCountPolicy> {
    let (min, max, exact) = match request {
        SpeakerCountRequest::Infer => (1, available.clamp(1, 8), None),
        SpeakerCountRequest::Range { .. } => (1, available, None),
        SpeakerCountRequest::HardConstraint { count } => {
            let exact = *count as usize;
            (exact, exact, Some(exact))
        }
        SpeakerCountRequest::Prior { .. } => (1, available, None),
    };
    if min == 0 || min > max || min > available {
        return Err(FwError::InvalidRequest(format!(
            "speaker-count request requires {min}..={max} profiles but only {available} are available"
        )));
    }
    Ok(SpeakerCountPolicy {
        min,
        max: max.min(available),
        exact,
    })
}

fn clusters_compatible(
    left: &AcousticCluster,
    right: &AcousticCluster,
    cannot_links: &BTreeSet<(String, String)>,
) -> bool {
    match (&left.hard_anchor, &right.hard_anchor) {
        // Distinct hard references are a fail-closed cannot-link even if an
        // upstream graph serialization accidentally omitted the redundant
        // pair. Known-interval must-links have already compacted equal labels.
        (Some(left), Some(right)) if left != right => false,
        (Some(left), Some(right)) => {
            !cannot_links.contains(&(left.clone(), right.clone()))
                && !cannot_links.contains(&(right.clone(), left.clone()))
        }
        _ => true,
    }
}

fn agglomerate_clusters<C>(
    initial: Vec<AcousticCluster>,
    cannot_links: &BTreeSet<(String, String)>,
    policy: SpeakerCountPolicy,
    is_cancelled: &mut C,
) -> FwResult<(Vec<AcousticCluster>, Vec<ClusterMergeTrace>)>
where
    C: FnMut() -> bool,
{
    let mut clusters = initial.into_iter().map(Some).collect::<Vec<_>>();
    let mut generations = vec![0u32; clusters.len()];
    let mut heap = BinaryHeap::new();
    for left in 0..clusters.len() {
        for right in left + 1..clusters.len() {
            let Some(left_cluster) = clusters[left].as_ref() else {
                continue;
            };
            let Some(right_cluster) = clusters[right].as_ref() else {
                continue;
            };
            if clusters_compatible(left_cluster, right_cluster, cannot_links) {
                heap.push(MergeCandidate {
                    distance: cluster_distance(left_cluster, right_cluster),
                    left,
                    right,
                    left_generation: 0,
                    right_generation: 0,
                });
            }
        }
    }
    let mut active = clusters.len();
    let mut best: Option<(f32, Vec<AcousticCluster>, usize)> = None;
    let mut merge_trace = Vec::new();
    evaluate_cluster_count(&clusters, active, policy, merge_trace.len(), &mut best);
    let mut merge_count = 0usize;
    'merging: while active > policy.min {
        if merge_count.is_multiple_of(ACOUSTIC_CANCELLATION_INTERVAL_FRAMES) && is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "acoustic clustering cancelled after {merge_count} merges"
            )));
        }
        let candidate = loop {
            let Some(candidate) = heap.pop() else {
                break 'merging;
            };
            let valid = clusters[candidate.left].is_some()
                && clusters[candidate.right].is_some()
                && generations[candidate.left] == candidate.left_generation
                && generations[candidate.right] == candidate.right_generation;
            if valid {
                break candidate;
            }
        };
        let right = clusters[candidate.right].take().ok_or_else(|| {
            FwError::InvalidRequest("merge candidate right cluster disappeared".to_owned())
        })?;
        let left = clusters[candidate.left].as_mut().ok_or_else(|| {
            FwError::InvalidRequest("merge candidate left cluster disappeared".to_owned())
        })?;
        merge_trace.push(ClusterMergeTrace {
            remaining_clusters: active - 1,
            distance: candidate.distance,
            same_speaker_probability: None,
            voice_distance: None,
            channel_distance: None,
            left_anchor: left.hard_anchor.clone(),
            right_anchor: right.hard_anchor.clone(),
        });
        merge_cluster(left, &right);
        generations[candidate.left] = generations[candidate.left].wrapping_add(1);
        generations[candidate.right] = generations[candidate.right].wrapping_add(1);
        active -= 1;
        merge_count += 1;
        for other in 0..clusters.len() {
            if other == candidate.left || clusters[other].is_none() {
                continue;
            }
            let (first, second) = if candidate.left < other {
                (candidate.left, other)
            } else {
                (other, candidate.left)
            };
            let Some(first_cluster) = clusters[first].as_ref() else {
                continue;
            };
            let Some(second_cluster) = clusters[second].as_ref() else {
                continue;
            };
            if clusters_compatible(first_cluster, second_cluster, cannot_links) {
                heap.push(MergeCandidate {
                    distance: cluster_distance(first_cluster, second_cluster),
                    left: first,
                    right: second,
                    left_generation: generations[first],
                    right_generation: generations[second],
                });
            }
        }
        evaluate_cluster_count(&clusters, active, policy, merge_trace.len(), &mut best);
    }
    let (_, selected, selected_trace_len) = best.ok_or_else(|| {
        FwError::InvalidRequest("no feasible speaker-count solution was found".to_owned())
    })?;
    merge_trace.truncate(selected_trace_len);
    Ok((selected, merge_trace))
}

#[derive(Debug, Clone)]
struct ProbabilisticLaneMergeStep {
    left: usize,
    right: usize,
    remaining_clusters: usize,
    same_speaker_probability: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SpeakerCountRiskPoint {
    count: usize,
    expected_loss: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct SpeakerCountRiskCurve {
    selected_count: usize,
    points: Vec<SpeakerCountRiskPoint>,
}

#[derive(Debug, Clone)]
struct ProbabilisticLaneResult {
    selected_count: usize,
    groups: Vec<Vec<usize>>,
    risk_curve: SpeakerCountRiskCurve,
}

enum ProbabilisticAgglomeration {
    Selected {
        clusters: Vec<AcousticCluster>,
        merge_trace: Vec<ClusterMergeTrace>,
        stability: f32,
        count_estimate: SpeakerCountEstimate,
    },
    Fallback {
        reason: AcousticClusteringFallbackReason,
        count_estimate: Option<SpeakerCountEstimate>,
    },
}

fn probabilistic_agglomerate_clusters<C>(
    initial: &[AcousticCluster],
    cannot_links: &BTreeSet<(String, String)>,
    request: &SpeakerCountRequest,
    policy: SpeakerCountPolicy,
    is_cancelled: &mut C,
) -> FwResult<ProbabilisticAgglomeration>
where
    C: FnMut() -> bool,
{
    let mut lane_results = Vec::with_capacity(SPEAKER_COUNT_PERTURBATION_LANES);
    for perturbation in SpeakerPairPerturbation::ALL {
        let Some(result) = probabilistic_agglomeration_lane(
            initial,
            cannot_links,
            policy,
            perturbation,
            is_cancelled,
        )?
        else {
            return Ok(ProbabilisticAgglomeration::Fallback {
                reason: AcousticClusteringFallbackReason::InsufficientSharedVoiceDimensions,
                count_estimate: None,
            });
        };
        lane_results.push(result);
    }
    let affinity_build = build_sparse_speaker_affinity(initial, cannot_links, &mut *is_cancelled)?;
    let spectral_run = match affinity_build.graph.as_ref() {
        Some(graph) => {
            sparse_normalized_eigengap_run(graph, policy.min, policy.max, &mut *is_cancelled)?
        }
        None => SparseEigengapRun::unavailable(
            affinity_build
                .unavailable_reason
                .unwrap_or(SpeakerCountLaneUnavailableReason::InvalidAffinity),
            0,
            None,
            0,
        ),
    };
    let resources = speaker_count_resource_summary(&affinity_build, &spectral_run)?;
    let spectral_proposal = spectral_run.proposal.as_ref();
    let constraint_lower_bound = initial
        .iter()
        .filter_map(|cluster| cluster.hard_anchor.as_ref())
        .collect::<BTreeSet<_>>()
        .len()
        .max(1);
    let Some(count_estimate) = fused_speaker_count_estimate(
        &lane_results,
        spectral_proposal,
        spectral_run.unavailable_reason,
        request,
        policy,
        constraint_lower_bound,
        resources,
    ) else {
        return Ok(ProbabilisticAgglomeration::Fallback {
            reason: AcousticClusteringFallbackReason::InvalidPosterior,
            count_estimate: None,
        });
    };
    let Some(selected_count) = count_estimate
        .selected_count
        .and_then(|count| usize::try_from(count).ok())
    else {
        return Ok(ProbabilisticAgglomeration::Fallback {
            reason: AcousticClusteringFallbackReason::UnstableSpeakerCount,
            count_estimate: Some(count_estimate),
        });
    };
    let agreeing_lanes = lane_results
        .iter()
        .filter(|result| result.selected_count == selected_count)
        .count();
    let feature_stability = agreeing_lanes as f32 / SPEAKER_COUNT_PERTURBATION_LANES as f32;
    let stability = spectral_proposal.map_or(feature_stability, |proposal| {
        let spectral_support = if proposal.count == selected_count {
            proposal.confidence as f32
        } else {
            0.0
        };
        0.5 * (feature_stability + spectral_support)
    });
    if stability < acoustic_speaker_pair_calibration().minimum_stable_lane_fraction {
        return Ok(ProbabilisticAgglomeration::Fallback {
            reason: AcousticClusteringFallbackReason::UnstableSpeakerCount,
            count_estimate: Some(count_estimate),
        });
    }
    let Some((clusters, merge_trace)) = coassociation_consensus_clusters(
        initial,
        cannot_links,
        &lane_results,
        selected_count,
        is_cancelled,
    )?
    else {
        return Ok(ProbabilisticAgglomeration::Fallback {
            reason: AcousticClusteringFallbackReason::UnstableSpeakerCount,
            count_estimate: Some(count_estimate),
        });
    };
    Ok(ProbabilisticAgglomeration::Selected {
        clusters,
        merge_trace,
        stability,
        count_estimate,
    })
}

#[cfg(test)]
fn fused_merge_risk_count(
    lanes: &[ProbabilisticLaneResult],
    spectral: Option<&SparseEigengapProposal>,
    policy: SpeakerCountPolicy,
) -> Option<usize> {
    fused_count_loss_points(lanes, spectral, policy)?
        .into_iter()
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(count, _)| count)
}

fn fused_count_loss_points(
    lanes: &[ProbabilisticLaneResult],
    spectral: Option<&SparseEigengapProposal>,
    policy: SpeakerCountPolicy,
) -> Option<Vec<(usize, f64)>> {
    let merge_risk_points = merge_risk_loss_points(lanes, policy)?;
    let candidate_count = policy.max.checked_sub(policy.min)?.checked_add(1)?;
    let mut points = Vec::with_capacity(merge_risk_points.len());
    for (count, mean_normalized_loss) in merge_risk_points {
        let selected_lane_count = lanes
            .iter()
            .filter(|lane| lane.selected_count == count)
            .count();
        let smoothed_lane_probability = (selected_lane_count as f64 + 0.5)
            / (lanes.len() as f64 + 0.5 * candidate_count as f64);
        let spectral_probability = spectral.map_or(1.0, |proposal| {
            if proposal.eigenvalues.is_empty()
                || !proposal.residual.is_finite()
                || proposal.iterations == 0
            {
                return 1.0e-9;
            }
            if candidate_count == 1 {
                1.0
            } else if proposal.count == count {
                proposal.confidence.max(1.0e-9)
            } else {
                ((1.0 - proposal.confidence) / (candidate_count - 1) as f64).max(1.0e-9)
            }
        });
        let fused_loss = mean_normalized_loss
            - SPEAKER_COUNT_JACKKNIFE_LOG_WEIGHT * smoothed_lane_probability.ln()
            - SPEAKER_COUNT_SPECTRAL_LOG_WEIGHT * spectral_probability.ln();
        if !fused_loss.is_finite() {
            return None;
        }
        points.push((count, fused_loss));
    }
    (!points.is_empty()).then_some(points)
}

fn merge_risk_loss_points(
    lanes: &[ProbabilisticLaneResult],
    policy: SpeakerCountPolicy,
) -> Option<Vec<(usize, f64)>> {
    if lanes.is_empty() {
        return None;
    }
    let capacity = policy.max.checked_sub(policy.min)?.checked_add(1)?;
    let mut points = Vec::with_capacity(capacity);
    for count in policy.min..=policy.max {
        if !speaker_count_allowed(count, policy) {
            continue;
        }
        let mut normalized_loss_sum = 0.0;
        for lane in lanes {
            let minimum = lane
                .risk_curve
                .points
                .iter()
                .map(|point| point.expected_loss)
                .min_by(f64::total_cmp)?;
            let maximum = lane
                .risk_curve
                .points
                .iter()
                .map(|point| point.expected_loss)
                .max_by(f64::total_cmp)?;
            let point = lane
                .risk_curve
                .points
                .iter()
                .find(|point| point.count == count)?;
            let scale = (maximum - minimum).max(1.0e-9);
            normalized_loss_sum += (point.expected_loss - minimum) / scale;
        }
        let mean_normalized_loss = normalized_loss_sum / lanes.len() as f64;
        if !mean_normalized_loss.is_finite() {
            return None;
        }
        points.push((count, mean_normalized_loss));
    }
    (!points.is_empty()).then_some(points)
}

fn fused_speaker_count_estimate(
    lanes: &[ProbabilisticLaneResult],
    spectral: Option<&SparseEigengapProposal>,
    spectral_unavailable_reason: Option<SpeakerCountLaneUnavailableReason>,
    request: &SpeakerCountRequest,
    policy: SpeakerCountPolicy,
    constraint_lower_bound: usize,
    resources: SpeakerCountResourceSummary,
) -> Option<SpeakerCountEstimate> {
    let merge_risk_losses = merge_risk_loss_points(lanes, policy)?;
    let merge_risk_count = merge_risk_losses
        .iter()
        .min_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })?
        .0;
    let mut ordered_merge_risk_losses = merge_risk_losses
        .iter()
        .map(|(_, loss)| *loss)
        .collect::<Vec<_>>();
    ordered_merge_risk_losses.sort_by(f64::total_cmp);
    let risk_confidence = if ordered_merge_risk_losses.len() > 1 {
        (1.0 - (-(ordered_merge_risk_losses[1] - ordered_merge_risk_losses[0]).max(0.0)).exp())
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut losses = fused_count_loss_points(lanes, spectral, policy)?;
    losses.sort_by_key(|(count, _)| *count);
    let minimum_loss = losses
        .iter()
        .map(|(_, loss)| *loss)
        .min_by(f64::total_cmp)?;
    let mut concrete_weights = losses
        .iter()
        .map(|(count, loss)| (*count, (minimum_loss - *loss).exp()))
        .collect::<Vec<_>>();
    let concrete_weight_sum = concrete_weights
        .iter()
        .map(|(_, weight)| *weight)
        .sum::<f64>();
    if !concrete_weight_sum.is_finite() || concrete_weight_sum <= 0.0 {
        return None;
    }
    for (_, weight) in &mut concrete_weights {
        *weight /= concrete_weight_sum;
    }
    let acoustic_map_count = concrete_weights
        .iter()
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })?
        .0;
    let acoustic_selection_stability = lanes
        .iter()
        .filter(|lane| lane.selected_count == acoustic_map_count)
        .count() as f64
        / lanes.len() as f64;

    // A soft request is deliberately a bounded linear pool rather than a
    // log-pool. Log evidence gives a caller's zero or near-zero probability
    // unbounded leverage and can therefore behave like an undeclared hard
    // constraint. The linear pool guarantees that every acoustically
    // supported count retains most of its acoustic probability while still
    // allowing the caller to move posterior mass. Five-lane acoustic agreement
    // attenuates the caller's influence further, so a contradictory hint
    // cannot veto unanimous evidence indirectly through the unresolved-mass
    // threshold.
    let mut soft_prior_weights = vec![0.0; concrete_weights.len()];
    match request {
        SpeakerCountRequest::Prior { bins } => {
            for (index, (count, _)) in concrete_weights.iter().enumerate() {
                soft_prior_weights[index] = bins
                    .iter()
                    .find(|bin| bin.count as usize == *count)
                    .map_or(0.0, |bin| bin.probability);
            }
        }
        SpeakerCountRequest::Range { minimum, maximum } => {
            for (index, (count, _)) in concrete_weights.iter().enumerate() {
                let count = u32::try_from(*count).ok()?;
                if (*minimum..=*maximum).contains(&count) {
                    soft_prior_weights[index] = 1.0;
                }
            }
        }
        SpeakerCountRequest::Infer | SpeakerCountRequest::HardConstraint { .. } => {}
    }
    let soft_prior_weight_sum = soft_prior_weights.iter().sum::<f64>();
    if soft_prior_weight_sum.is_finite() && soft_prior_weight_sum > 0.0 {
        let effective_prior_mix_weight = SPEAKER_COUNT_SOFT_PRIOR_MIX_WEIGHT
            * (1.0 - SPEAKER_COUNT_SOFT_PRIOR_STABILITY_ATTENUATION * acoustic_selection_stability);
        for ((_, acoustic_weight), prior_weight) in
            concrete_weights.iter_mut().zip(soft_prior_weights)
        {
            let normalized_prior_weight = prior_weight / soft_prior_weight_sum;
            *acoustic_weight = (1.0 - effective_prior_mix_weight) * *acoustic_weight
                + effective_prior_mix_weight * normalized_prior_weight;
        }
    }
    let concrete_weight_sum = concrete_weights
        .iter()
        .map(|(_, weight)| *weight)
        .sum::<f64>();
    if !concrete_weight_sum.is_finite() || concrete_weight_sum <= 0.0 {
        return None;
    }

    let mut lane_counts = BTreeMap::<usize, usize>::new();
    for lane in lanes {
        *lane_counts.entry(lane.selected_count).or_default() += 1;
    }
    let (jackknife_count, agreeing_lanes) = lane_counts
        .into_iter()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))?;
    let jackknife_stability = agreeing_lanes as f64 / lanes.len() as f64;
    let raw_map_count = concrete_weights
        .iter()
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })?
        .0;
    let selection_stability = lanes
        .iter()
        .filter(|lane| lane.selected_count == raw_map_count)
        .count() as f64
        / lanes.len() as f64;
    let spectral_confidence = spectral.map_or(0.0, |proposal| proposal.confidence);
    let evidence_strength = SPEAKER_COUNT_STABILITY_EVIDENCE_WEIGHT * selection_stability
        + SPEAKER_COUNT_RISK_EVIDENCE_WEIGHT * risk_confidence
        + SPEAKER_COUNT_SPECTRAL_EVIDENCE_WEIGHT * spectral_confidence;
    let unresolved_probability = (SPEAKER_COUNT_UNRESOLVED_INTERCEPT
        - SPEAKER_COUNT_UNRESOLVED_EVIDENCE_SLOPE * evidence_strength)
        .clamp(
            SPEAKER_COUNT_MINIMUM_UNRESOLVED_MASS,
            SPEAKER_COUNT_MAXIMUM_UNRESOLVED_MASS,
        );
    let concrete_mass = 1.0 - unresolved_probability;
    for (_, weight) in &mut concrete_weights {
        *weight = concrete_mass * *weight / concrete_weight_sum;
    }
    let map = concrete_weights
        .iter()
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .copied()?;
    let selected_count = (selection_stability
        >= f64::from(acoustic_speaker_pair_calibration().minimum_stable_lane_fraction)
        && map.1 >= unresolved_probability)
        .then_some(u32::try_from(map.0).ok()?);

    let mut credible_bins = concrete_weights.clone();
    credible_bins.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let credible_target = SPEAKER_COUNT_CONCRETE_CREDIBLE_MASS * concrete_mass;
    let mut credible_mass = 0.0;
    let mut credible_minimum = usize::MAX;
    let mut credible_maximum = 0usize;
    for (count, probability) in credible_bins {
        credible_mass += probability;
        credible_minimum = credible_minimum.min(count);
        credible_maximum = credible_maximum.max(count);
        if credible_mass + f64::EPSILON >= credible_target {
            break;
        }
    }
    if credible_minimum == usize::MAX {
        return None;
    }
    let supported_range = SpeakerCountRange {
        minimum: u32::try_from(credible_minimum).ok()?,
        maximum: u32::try_from(credible_maximum).ok()?,
    };
    let posterior = concrete_weights
        .into_iter()
        .map(|(count, probability)| {
            Some(SpeakerCountPosteriorBin {
                count: u32::try_from(count).ok()?,
                probability,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let entropy_bits = posterior
        .iter()
        .map(|bin| count_entropy_term(bin.probability))
        .sum::<f64>()
        + count_entropy_term(unresolved_probability);
    let constraint_lower_bound = u32::try_from(constraint_lower_bound).ok()?;
    let candidate_upper_bound = u32::try_from(policy.max).ok()?;
    let calibration_sha256 = acoustic_speaker_pair_calibration_sha256();
    let evidence_sha256 = speaker_count_evidence_sha256(
        lanes,
        spectral,
        &resources,
        request,
        constraint_lower_bound,
        candidate_upper_bound,
    );
    let prior_lane =
        speaker_count_prior_lane(request, constraint_lower_bound, candidate_upper_bound)?;
    let estimate = SpeakerCountEstimate {
        schema_version: SPEAKER_COUNT_ESTIMATE_SCHEMA_VERSION.to_owned(),
        selected_count,
        supported_range: Some(supported_range),
        posterior,
        unresolved_probability,
        entropy_bits,
        stability: selection_stability,
        constraint_lower_bound,
        candidate_upper_bound,
        calibration_status: SpeakerCountCalibrationStatus::DevelopmentUncertified,
        calibration_sha256,
        evidence_sha256,
        lanes: vec![
            SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::MergeRisk,
                available: true,
                proposed_count: Some(u32::try_from(merge_risk_count).ok()?),
                confidence: risk_confidence,
                unavailable_reason: None,
            },
            SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::SparseNormalizedEigengap,
                available: spectral.is_some(),
                proposed_count: spectral.and_then(|proposal| u32::try_from(proposal.count).ok()),
                confidence: spectral_confidence,
                unavailable_reason: spectral.is_none().then_some(
                    spectral_unavailable_reason
                        .unwrap_or(SpeakerCountLaneUnavailableReason::InvalidAffinity),
                ),
            },
            SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::FeatureJackknife,
                available: true,
                proposed_count: Some(u32::try_from(jackknife_count).ok()?),
                confidence: jackknife_stability,
                unavailable_reason: None,
            },
            SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::EffectiveOccupancy,
                available: false,
                proposed_count: None,
                confidence: 0.0,
                unavailable_reason: Some(
                    SpeakerCountLaneUnavailableReason::InsufficientVoicedEvidence,
                ),
            },
            SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::ConstraintGraph,
                available: true,
                proposed_count: Some(constraint_lower_bound),
                confidence: 1.0,
                unavailable_reason: None,
            },
            prior_lane,
        ],
        resources,
    };
    estimate.validate().ok()?;
    Some(estimate)
}

fn speaker_count_evidence_sha256(
    lanes: &[ProbabilisticLaneResult],
    spectral: Option<&SparseEigengapProposal>,
    resources: &SpeakerCountResourceSummary,
    request: &SpeakerCountRequest,
    constraint_lower_bound: u32,
    candidate_upper_bound: u32,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(SPEAKER_COUNT_ESTIMATE_SCHEMA_VERSION.as_bytes());
    hasher.update(ACOUSTIC_CLUSTERING_PROBABILISTIC_VERSION.as_bytes());
    hasher.update(constraint_lower_bound.to_le_bytes());
    hasher.update(candidate_upper_bound.to_le_bytes());
    hash_speaker_count_resources(&mut hasher, resources);
    hash_speaker_count_request(&mut hasher, request);
    for lane in lanes {
        hasher.update((lane.selected_count as u64).to_le_bytes());
        for point in &lane.risk_curve.points {
            hasher.update((point.count as u64).to_le_bytes());
            hasher.update(point.expected_loss.to_bits().to_le_bytes());
        }
    }
    if let Some(spectral) = spectral {
        hasher.update((spectral.count as u64).to_le_bytes());
        hasher.update(spectral.confidence.to_bits().to_le_bytes());
        hasher.update(spectral.residual.to_bits().to_le_bytes());
        hasher.update((spectral.iterations as u64).to_le_bytes());
        for eigenvalue in &spectral.eigenvalues {
            hasher.update(eigenvalue.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
}

fn hash_speaker_count_resources(hasher: &mut Sha256, resources: &SpeakerCountResourceSummary) {
    hasher.update(resources.prototype_count.to_le_bytes());
    hasher.update(resources.affinity_pair_evaluations.to_le_bytes());
    hasher.update(resources.retained_sparse_edges.to_le_bytes());
    hasher.update(resources.estimated_peak_buffer_bytes.to_le_bytes());
    hasher.update(resources.stability_replicates.to_le_bytes());
    hasher.update(resources.solver_iterations.to_le_bytes());
    hasher.update(resources.solver_sparse_matvec_terms.to_le_bytes());
    match resources.solver_residual {
        Some(residual) => {
            hasher.update([1]);
            hasher.update(residual.to_bits().to_le_bytes());
        }
        None => hasher.update([0]),
    }
}

fn count_entropy_term(probability: f64) -> f64 {
    if probability > 0.0 {
        -probability * probability.log2()
    } else {
        0.0
    }
}

fn unavailable_speaker_count_resources(
    prototype_count: usize,
) -> Option<SpeakerCountResourceSummary> {
    Some(SpeakerCountResourceSummary {
        prototype_count: u32::try_from(prototype_count).ok()?,
        affinity_pair_evaluations: 0,
        retained_sparse_edges: 0,
        estimated_peak_buffer_bytes: 0,
        stability_replicates: 0,
        solver_iterations: 0,
        solver_sparse_matvec_terms: 0,
        solver_residual: None,
    })
}

fn unavailable_speaker_count_estimate(
    request: &SpeakerCountRequest,
    constraint_lower_bound: usize,
    available_candidates: usize,
    prototype_count: usize,
    calibration_status: SpeakerCountCalibrationStatus,
    unavailable_reason: SpeakerCountLaneUnavailableReason,
) -> Option<SpeakerCountEstimate> {
    let constraint_lower_bound = u32::try_from(constraint_lower_bound).ok()?;
    let available_candidates = u32::try_from(available_candidates)
        .ok()?
        .clamp(1, MAX_SPEAKER_COUNT);
    let requested_upper_bound = match request {
        SpeakerCountRequest::Infer => available_candidates,
        SpeakerCountRequest::Prior { bins } => bins.last()?.count.max(available_candidates),
        SpeakerCountRequest::Range { maximum, .. } => (*maximum).max(available_candidates),
        SpeakerCountRequest::HardConstraint { count } => *count,
    };
    let candidate_upper_bound = requested_upper_bound.max(constraint_lower_bound);
    let prior_lane =
        speaker_count_prior_lane(request, constraint_lower_bound, candidate_upper_bound)?;
    let mut hasher = Sha256::new();
    hasher.update(b"speaker-count-unavailable-v1");
    hash_speaker_count_request(&mut hasher, request);
    hasher.update(constraint_lower_bound.to_le_bytes());
    hasher.update(candidate_upper_bound.to_le_bytes());
    hasher.update(speaker_count_calibration_status_label(calibration_status).as_bytes());
    hasher.update(speaker_count_lane_unavailable_reason_label(unavailable_reason).as_bytes());
    let resources = unavailable_speaker_count_resources(prototype_count)?;
    hash_speaker_count_resources(&mut hasher, &resources);
    let evidence_sha256 = format!("{:x}", hasher.finalize());
    let mut lanes = [
        SpeakerCountEvidenceLane::MergeRisk,
        SpeakerCountEvidenceLane::SparseNormalizedEigengap,
        SpeakerCountEvidenceLane::FeatureJackknife,
        SpeakerCountEvidenceLane::EffectiveOccupancy,
    ]
    .into_iter()
    .map(|lane| SpeakerCountLaneEvidence {
        lane,
        available: false,
        proposed_count: None,
        confidence: 0.0,
        unavailable_reason: Some(unavailable_reason),
    })
    .collect::<Vec<_>>();
    lanes.push(SpeakerCountLaneEvidence {
        lane: SpeakerCountEvidenceLane::ConstraintGraph,
        available: true,
        proposed_count: Some(constraint_lower_bound),
        confidence: 1.0,
        unavailable_reason: None,
    });
    lanes.push(prior_lane);
    let estimate = SpeakerCountEstimate {
        schema_version: SPEAKER_COUNT_ESTIMATE_SCHEMA_VERSION.to_owned(),
        selected_count: None,
        supported_range: None,
        posterior: Vec::new(),
        unresolved_probability: 1.0,
        entropy_bits: 0.0,
        stability: 0.0,
        constraint_lower_bound,
        candidate_upper_bound,
        calibration_status,
        calibration_sha256: acoustic_speaker_pair_calibration_sha256(),
        evidence_sha256,
        lanes,
        resources,
    };
    estimate.validate().ok()?;
    Some(estimate)
}

fn unavailable_count_prior_lane(
    reason: SpeakerCountLaneUnavailableReason,
) -> SpeakerCountLaneEvidence {
    SpeakerCountLaneEvidence {
        lane: SpeakerCountEvidenceLane::CallerPrior,
        available: false,
        proposed_count: None,
        confidence: 0.0,
        unavailable_reason: Some(reason),
    }
}

fn speaker_count_prior_lane(
    request: &SpeakerCountRequest,
    constraint_lower_bound: u32,
    candidate_upper_bound: u32,
) -> Option<SpeakerCountLaneEvidence> {
    match request {
        SpeakerCountRequest::Prior { bins } => {
            let proposed = bins
                .iter()
                .filter(|bin| {
                    bin.count >= constraint_lower_bound && bin.count <= candidate_upper_bound
                })
                .max_by(|left, right| {
                    left.probability
                        .total_cmp(&right.probability)
                        .then_with(|| right.count.cmp(&left.count))
                });
            Some(proposed.map_or_else(
                || {
                    unavailable_count_prior_lane(
                        SpeakerCountLaneUnavailableReason::ContradictoryConstraints,
                    )
                },
                |proposed| SpeakerCountLaneEvidence {
                    lane: SpeakerCountEvidenceLane::CallerPrior,
                    available: true,
                    proposed_count: Some(proposed.count),
                    confidence: proposed.probability,
                    unavailable_reason: None,
                },
            ))
        }
        SpeakerCountRequest::Range { minimum, maximum } => {
            let width = maximum.checked_sub(*minimum)?.checked_add(1)?;
            if *maximum < constraint_lower_bound || *minimum > candidate_upper_bound {
                Some(unavailable_count_prior_lane(
                    SpeakerCountLaneUnavailableReason::ContradictoryConstraints,
                ))
            } else {
                Some(SpeakerCountLaneEvidence {
                    lane: SpeakerCountEvidenceLane::CallerPrior,
                    available: true,
                    proposed_count: Some((*minimum).max(constraint_lower_bound)),
                    confidence: 1.0 / f64::from(width),
                    unavailable_reason: None,
                })
            }
        }
        SpeakerCountRequest::Infer | SpeakerCountRequest::HardConstraint { .. } => Some(
            unavailable_count_prior_lane(SpeakerCountLaneUnavailableReason::NotRequested),
        ),
    }
}

fn hash_speaker_count_request(hasher: &mut Sha256, request: &SpeakerCountRequest) {
    match request {
        SpeakerCountRequest::Infer => hasher.update(b"infer"),
        SpeakerCountRequest::Range { minimum, maximum } => {
            hasher.update(b"range");
            hasher.update(minimum.to_le_bytes());
            hasher.update(maximum.to_le_bytes());
        }
        SpeakerCountRequest::HardConstraint { count } => {
            hasher.update(b"hard");
            hasher.update(count.to_le_bytes());
        }
        SpeakerCountRequest::Prior { bins } => {
            hasher.update(b"prior");
            for bin in bins {
                hasher.update(bin.count.to_le_bytes());
                hasher.update(bin.probability.to_bits().to_le_bytes());
            }
        }
    }
}

const fn speaker_count_calibration_status_label(
    status: SpeakerCountCalibrationStatus,
) -> &'static str {
    match status {
        SpeakerCountCalibrationStatus::Certified => "certified",
        SpeakerCountCalibrationStatus::DevelopmentUncertified => "development_uncertified",
        SpeakerCountCalibrationStatus::FixedSafeUncalibrated => "fixed_safe_uncalibrated",
        SpeakerCountCalibrationStatus::Unavailable => "unavailable",
    }
}

const fn speaker_count_lane_unavailable_reason_label(
    reason: SpeakerCountLaneUnavailableReason,
) -> &'static str {
    match reason {
        SpeakerCountLaneUnavailableReason::InsufficientPrototypes => "insufficient_prototypes",
        SpeakerCountLaneUnavailableReason::InvalidAffinity => "invalid_affinity",
        SpeakerCountLaneUnavailableReason::SolverDidNotConverge => "solver_did_not_converge",
        SpeakerCountLaneUnavailableReason::InsufficientIndependentReplicates => {
            "insufficient_independent_replicates"
        }
        SpeakerCountLaneUnavailableReason::InsufficientVoicedEvidence => {
            "insufficient_voiced_evidence"
        }
        SpeakerCountLaneUnavailableReason::NotRequested => "not_requested",
        SpeakerCountLaneUnavailableReason::CalibrationUnavailable => "calibration_unavailable",
        SpeakerCountLaneUnavailableReason::ResourceLimit => "resource_limit",
        SpeakerCountLaneUnavailableReason::ContradictoryConstraints => "contradictory_constraints",
    }
}

fn finalize_speaker_count_estimate(
    mut estimate: SpeakerCountEstimate,
    evidence: &[AcousticSpeakerEvidenceSummary],
    supported_speakers: usize,
) -> Option<SpeakerCountEstimate> {
    let supported_count = u32::try_from(supported_speakers).ok()?;
    let supported = evidence
        .iter()
        .filter(|speaker| speaker.supported)
        .collect::<Vec<_>>();
    let occupancy_confidence = if supported.is_empty() {
        0.0
    } else {
        supported
            .iter()
            .map(|speaker| {
                f64::from(
                    speaker
                        .mean_assignment_confidence
                        .min(speaker.cluster_reliability),
                )
            })
            .fold(1.0_f64, f64::min)
            .clamp(0.0, 1.0)
    };
    let occupancy_in_bounds = supported_count >= estimate.constraint_lower_bound
        && supported_count <= estimate.candidate_upper_bound
        && supported_count > 0;
    let occupancy_lane = estimate
        .lanes
        .iter_mut()
        .find(|lane| lane.lane == SpeakerCountEvidenceLane::EffectiveOccupancy)?;
    if occupancy_in_bounds {
        *occupancy_lane = SpeakerCountLaneEvidence {
            lane: SpeakerCountEvidenceLane::EffectiveOccupancy,
            available: true,
            proposed_count: Some(supported_count),
            confidence: occupancy_confidence,
            unavailable_reason: None,
        };
    } else {
        *occupancy_lane = SpeakerCountLaneEvidence {
            lane: SpeakerCountEvidenceLane::EffectiveOccupancy,
            available: false,
            proposed_count: None,
            confidence: 0.0,
            unavailable_reason: Some(if supported_count == 0 {
                SpeakerCountLaneUnavailableReason::InsufficientVoicedEvidence
            } else {
                SpeakerCountLaneUnavailableReason::ContradictoryConstraints
            }),
        };
    }

    let occupancy_agreement =
        if estimate.selected_count == Some(supported_count) && occupancy_in_bounds {
            occupancy_confidence
        } else {
            0.0
        };
    estimate.stability = ((1.0 - SPEAKER_COUNT_OCCUPANCY_STABILITY_WEIGHT) * estimate.stability
        + SPEAKER_COUNT_OCCUPANCY_STABILITY_WEIGHT * occupancy_agreement)
        .clamp(0.0, 1.0);
    if occupancy_agreement == 0.0 {
        estimate.selected_count = None;
        let unresolved_probability = estimate.unresolved_probability.max(0.51);
        let old_concrete_mass = 1.0 - estimate.unresolved_probability;
        let new_concrete_mass = 1.0 - unresolved_probability;
        if old_concrete_mass > f64::EPSILON {
            for bin in &mut estimate.posterior {
                bin.probability *= new_concrete_mass / old_concrete_mass;
            }
        } else {
            estimate.posterior.clear();
        }
        estimate.unresolved_probability = unresolved_probability;
    }
    estimate.entropy_bits = estimate
        .posterior
        .iter()
        .map(|bin| count_entropy_term(bin.probability))
        .sum::<f64>()
        + count_entropy_term(estimate.unresolved_probability);
    let mut hasher = Sha256::new();
    hasher.update(b"speaker-count-occupancy-finalization-v1");
    hasher.update(estimate.evidence_sha256.as_bytes());
    hasher.update(supported_count.to_le_bytes());
    for speaker in evidence {
        hasher.update((speaker.assigned_tracklet_count as u64).to_le_bytes());
        hasher.update((speaker.independent_tracklet_count as u64).to_le_bytes());
        hasher.update((speaker.recurrence_episode_count as u64).to_le_bytes());
        hasher.update((speaker.voiced_frame_count as u64).to_le_bytes());
        hasher.update((speaker.independent_voiced_frame_count as u64).to_le_bytes());
        hasher.update(speaker.voiced_duration_ms.to_le_bytes());
        hasher.update(speaker.mean_assignment_confidence.to_bits().to_le_bytes());
        hasher.update(speaker.cluster_reliability.to_bits().to_le_bytes());
        hasher.update([u8::from(speaker.hard_anchored), u8::from(speaker.supported)]);
    }
    estimate.evidence_sha256 = format!("{:x}", hasher.finalize());
    estimate.validate().ok()?;
    Some(estimate)
}

#[derive(Debug, Clone, PartialEq)]
struct SparseSpeakerAffinityGraph {
    rows: Vec<Vec<(usize, f64)>>,
    degrees: Vec<f64>,
    undirected_edge_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct SparseSpeakerAffinityBuild {
    graph: Option<SparseSpeakerAffinityGraph>,
    prototype_count: usize,
    affinity_pair_evaluations: u64,
    estimated_peak_buffer_bytes: u64,
    unavailable_reason: Option<SpeakerCountLaneUnavailableReason>,
}

impl SparseSpeakerAffinityGraph {
    fn apply_normalized(&self, input: &[f64]) -> Option<Vec<f64>> {
        if input.len() != self.rows.len() || self.degrees.len() != self.rows.len() {
            return None;
        }
        let mut output = vec![0.0; input.len()];
        for (row, neighbors) in self.rows.iter().enumerate() {
            let row_degree = *self.degrees.get(row)?;
            if !row_degree.is_finite() || row_degree <= 0.0 {
                return None;
            }
            output[row] += input[row] / row_degree;
            for &(column, weight) in neighbors {
                let column_degree = *self.degrees.get(column)?;
                let denominator = (row_degree * column_degree).sqrt();
                if !weight.is_finite()
                    || weight <= 0.0
                    || !denominator.is_finite()
                    || denominator <= 0.0
                {
                    return None;
                }
                output[row] += weight * input[column] / denominator;
            }
            if !output[row].is_finite() {
                return None;
            }
        }
        Some(output)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SparseEigengapProposal {
    count: usize,
    confidence: f64,
    eigenvalues: Vec<f64>,
    residual: f64,
    iterations: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct SparseEigengapRun {
    proposal: Option<SparseEigengapProposal>,
    iterations: usize,
    residual: Option<f64>,
    subspace_dimensions: usize,
    unavailable_reason: Option<SpeakerCountLaneUnavailableReason>,
}

impl SparseEigengapRun {
    fn unavailable(
        reason: SpeakerCountLaneUnavailableReason,
        iterations: usize,
        residual: Option<f64>,
        subspace_dimensions: usize,
    ) -> Self {
        Self {
            proposal: None,
            iterations,
            residual,
            subspace_dimensions,
            unavailable_reason: Some(reason),
        }
    }
}

fn build_sparse_speaker_affinity<C>(
    clusters: &[AcousticCluster],
    cannot_links: &BTreeSet<(String, String)>,
    mut is_cancelled: C,
) -> FwResult<SparseSpeakerAffinityBuild>
where
    C: FnMut() -> bool,
{
    let prototype_count = clusters.len();
    if clusters.len() < 2 {
        return Ok(SparseSpeakerAffinityBuild {
            graph: None,
            prototype_count,
            affinity_pair_evaluations: 0,
            estimated_peak_buffer_bytes: 0,
            unavailable_reason: Some(SpeakerCountLaneUnavailableReason::InsufficientPrototypes),
        });
    }
    let edge_capacity = clusters
        .len()
        .checked_mul(SPEAKER_COUNT_SPARSE_NEIGHBOR_DEGREE)
        .map(|directed_capacity| directed_capacity / 2)
        .ok_or_else(|| FwError::InvalidRequest("speaker affinity edge cap overflow".to_owned()))?;
    let mut affinity_pair_evaluations = 0_u64;
    let estimated_peak_buffer_bytes = estimated_sparse_affinity_peak_bytes(
        clusters.len(),
        edge_capacity,
        SPEAKER_COUNT_SPARSE_NEIGHBOR_DEGREE,
    )?;
    let mut candidate_edges = BTreeMap::<(usize, usize), f64>::new();
    for left in 0..clusters.len() {
        if left.is_multiple_of(ACOUSTIC_CANCELLATION_INTERVAL_FRAMES) && is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "speaker affinity construction cancelled after {left} prototype rows"
            )));
        }
        let mut neighbors = Vec::<(f64, usize)>::new();
        for right in 0..clusters.len() {
            if left == right
                || !clusters_compatible(&clusters[left], &clusters[right], cannot_links)
            {
                continue;
            }
            affinity_pair_evaluations =
                affinity_pair_evaluations.checked_add(1).ok_or_else(|| {
                    FwError::InvalidRequest("speaker affinity comparison count overflow".to_owned())
                })?;
            let Some(evidence) = cluster_pair_evidence(
                &clusters[left],
                &clusters[right],
                SpeakerPairPerturbation::Full,
            ) else {
                continue;
            };
            let weight = f64::from(
                ((2.0 * evidence.same_speaker_probability - 1.0).max(0.0)) * evidence.support,
            );
            if weight.is_finite() && weight > 0.0 {
                neighbors.push((weight, right));
            }
        }
        neighbors.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        neighbors.truncate(SPEAKER_COUNT_SPARSE_NEIGHBOR_DEGREE);
        for (weight, right) in neighbors {
            let edge = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            candidate_edges
                .entry(edge)
                .and_modify(|candidate| *candidate = candidate.max(weight))
                .or_insert(weight);
        }
    }
    if candidate_edges.is_empty() {
        return Ok(SparseSpeakerAffinityBuild {
            graph: None,
            prototype_count,
            affinity_pair_evaluations,
            estimated_peak_buffer_bytes,
            unavailable_reason: Some(SpeakerCountLaneUnavailableReason::InvalidAffinity),
        });
    }
    let mut candidate_edges = candidate_edges
        .into_iter()
        .map(|((left, right), weight)| (weight, left, right))
        .collect::<Vec<_>>();
    candidate_edges.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    let mut retained_degrees = vec![0usize; clusters.len()];
    let mut retained_edges = Vec::with_capacity(edge_capacity.min(candidate_edges.len()));
    for (weight, left, right) in candidate_edges {
        if retained_degrees[left] >= SPEAKER_COUNT_SPARSE_NEIGHBOR_DEGREE
            || retained_degrees[right] >= SPEAKER_COUNT_SPARSE_NEIGHBOR_DEGREE
        {
            continue;
        }
        retained_degrees[left] += 1;
        retained_degrees[right] += 1;
        retained_edges.push((left, right, weight));
    }
    if retained_edges.is_empty() {
        return Ok(SparseSpeakerAffinityBuild {
            graph: None,
            prototype_count,
            affinity_pair_evaluations,
            estimated_peak_buffer_bytes,
            unavailable_reason: Some(SpeakerCountLaneUnavailableReason::InvalidAffinity),
        });
    }
    let mut rows = vec![Vec::new(); clusters.len()];
    let mut degrees = vec![1.0; clusters.len()];
    for (left, right, weight) in retained_edges {
        rows[left].push((right, weight));
        rows[right].push((left, weight));
        degrees[left] += weight;
        degrees[right] += weight;
    }
    for row in &mut rows {
        row.sort_by_key(|(neighbor, _)| *neighbor);
    }
    if rows
        .iter()
        .any(|row| row.len() > SPEAKER_COUNT_SPARSE_NEIGHBOR_DEGREE)
    {
        return Err(FwError::InvalidRequest(
            "symmetrized speaker affinity exceeded its row-degree cap".to_owned(),
        ));
    }
    let undirected_edge_count = rows.iter().map(Vec::len).sum::<usize>() / 2;
    Ok(SparseSpeakerAffinityBuild {
        graph: Some(SparseSpeakerAffinityGraph {
            rows,
            degrees,
            undirected_edge_count,
        }),
        prototype_count,
        affinity_pair_evaluations,
        estimated_peak_buffer_bytes,
        unavailable_reason: None,
    })
}

fn estimated_sparse_affinity_peak_bytes(
    prototype_count: usize,
    retained_edge_capacity: usize,
    neighbor_degree: usize,
) -> FwResult<u64> {
    let node_bytes = prototype_count
        .checked_mul(
            std::mem::size_of::<Vec<(usize, f64)>>()
                + std::mem::size_of::<f64>()
                + std::mem::size_of::<usize>(),
        )
        .ok_or_else(|| {
            FwError::InvalidRequest("speaker affinity node-buffer byte count overflow".to_owned())
        })?;
    let retained_edge_bytes = retained_edge_capacity
        .checked_mul(2)
        .and_then(|directed_edges| directed_edges.checked_mul(std::mem::size_of::<(usize, f64)>()))
        .ok_or_else(|| {
            FwError::InvalidRequest("speaker affinity edge-buffer byte count overflow".to_owned())
        })?;
    let row_candidate_bytes = prototype_count
        .checked_mul(std::mem::size_of::<(f64, usize)>())
        .ok_or_else(|| {
            FwError::InvalidRequest("speaker affinity candidate-row byte count overflow".to_owned())
        })?;
    let retained_candidate_bytes = prototype_count
        .checked_mul(neighbor_degree)
        .and_then(|edges| edges.checked_mul(std::mem::size_of::<(f64, usize, usize)>()))
        .ok_or_else(|| {
            FwError::InvalidRequest(
                "speaker affinity candidate-edge byte count overflow".to_owned(),
            )
        })?;
    node_bytes
        .checked_add(retained_edge_bytes)
        .and_then(|bytes| bytes.checked_add(row_candidate_bytes))
        .and_then(|bytes| bytes.checked_add(retained_candidate_bytes))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            FwError::InvalidRequest("speaker affinity peak byte estimate overflow".to_owned())
        })
}

fn speaker_count_resource_summary(
    affinity: &SparseSpeakerAffinityBuild,
    solver: &SparseEigengapRun,
) -> FwResult<SpeakerCountResourceSummary> {
    let retained_sparse_edges = affinity
        .graph
        .as_ref()
        .map_or(0, |graph| graph.undirected_edge_count);
    let node_count = affinity.prototype_count;
    let sparse_product_terms = node_count
        .checked_add(retained_sparse_edges.checked_mul(2).ok_or_else(|| {
            FwError::InvalidRequest("speaker-count sparse term count overflow".to_owned())
        })?)
        .ok_or_else(|| {
            FwError::InvalidRequest("speaker-count sparse term count overflow".to_owned())
        })?;
    let solver_sparse_matvec_terms = sparse_product_terms
        .checked_mul(solver.subspace_dimensions)
        .and_then(|terms| terms.checked_mul(2))
        .and_then(|terms| terms.checked_mul(solver.iterations))
        .and_then(|terms| u64::try_from(terms).ok())
        .ok_or_else(|| {
            FwError::InvalidRequest("speaker-count solver operation count overflow".to_owned())
        })?;
    let solver_vector_bytes = node_count
        .checked_mul(solver.subspace_dimensions)
        .and_then(|values| values.checked_mul(3))
        .and_then(|values| values.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| {
            FwError::InvalidRequest("speaker-count solver vector byte count overflow".to_owned())
        })?;
    let solver_matrix_bytes = solver
        .subspace_dimensions
        .checked_mul(solver.subspace_dimensions)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| {
            FwError::InvalidRequest("speaker-count solver matrix byte count overflow".to_owned())
        })?;
    let estimated_peak_buffer_bytes = affinity
        .estimated_peak_buffer_bytes
        .checked_add(u64::try_from(solver_vector_bytes).map_err(|_| {
            FwError::InvalidRequest("speaker-count solver vector byte count overflow".to_owned())
        })?)
        .and_then(|bytes| bytes.checked_add(u64::try_from(solver_matrix_bytes).ok()?))
        .ok_or_else(|| {
            FwError::InvalidRequest("speaker-count peak byte estimate overflow".to_owned())
        })?;
    Ok(SpeakerCountResourceSummary {
        prototype_count: u32::try_from(affinity.prototype_count).map_err(|_| {
            FwError::InvalidRequest("speaker-count prototype count overflow".to_owned())
        })?,
        affinity_pair_evaluations: affinity.affinity_pair_evaluations,
        retained_sparse_edges: u32::try_from(retained_sparse_edges).map_err(|_| {
            FwError::InvalidRequest("speaker-count sparse edge count overflow".to_owned())
        })?,
        estimated_peak_buffer_bytes,
        stability_replicates: u32::try_from(SPEAKER_COUNT_PERTURBATION_LANES).map_err(|_| {
            FwError::InvalidRequest("speaker-count replicate count overflow".to_owned())
        })?,
        solver_iterations: u32::try_from(solver.iterations).map_err(|_| {
            FwError::InvalidRequest("speaker-count solver iteration count overflow".to_owned())
        })?,
        solver_sparse_matvec_terms,
        solver_residual: solver.residual,
    })
}

fn sparse_normalized_eigengap_run<C>(
    graph: &SparseSpeakerAffinityGraph,
    minimum_count: usize,
    maximum_count: usize,
    mut is_cancelled: C,
) -> FwResult<SparseEigengapRun>
where
    C: FnMut() -> bool,
{
    let node_count = graph.rows.len();
    if node_count == 0 || graph.degrees.len() != node_count {
        return Ok(SparseEigengapRun::unavailable(
            SpeakerCountLaneUnavailableReason::InvalidAffinity,
            0,
            None,
            0,
        ));
    }
    if node_count == 1 {
        let proposal = SparseEigengapProposal {
            count: 1,
            confidence: 1.0,
            eigenvalues: vec![1.0],
            residual: 0.0,
            iterations: 0,
        };
        return Ok(SparseEigengapRun {
            proposal: Some(proposal),
            iterations: 0,
            residual: Some(0.0),
            subspace_dimensions: 1,
            unavailable_reason: None,
        });
    }
    if graph.undirected_edge_count == 0 || minimum_count == 0 || minimum_count > maximum_count {
        return Ok(SparseEigengapRun::unavailable(
            SpeakerCountLaneUnavailableReason::InvalidAffinity,
            0,
            None,
            0,
        ));
    }
    let bounded_maximum = maximum_count.min(node_count);
    if minimum_count > bounded_maximum {
        return Ok(SparseEigengapRun::unavailable(
            SpeakerCountLaneUnavailableReason::ContradictoryConstraints,
            0,
            None,
            0,
        ));
    }
    let subspace_dimensions = bounded_maximum.saturating_add(1).min(node_count);
    let mut basis = deterministic_spectral_basis(graph, subspace_dimensions)?;
    let mut residual = f64::INFINITY;
    let mut iterations = 0usize;
    let mut rayleigh = vec![vec![0.0; subspace_dimensions]; subspace_dimensions];
    for iteration in 0..SPEAKER_COUNT_EIGENSOLVER_MAX_ITERATIONS {
        if is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "speaker-count eigensolver cancelled after {iteration} iterations"
            )));
        }
        let mut next = basis
            .iter()
            .map(|column| {
                let mut product = graph.apply_normalized(column)?;
                for (value, &input) in product.iter_mut().zip(column) {
                    *value += SPEAKER_COUNT_EIGENSOLVER_DIAGONAL_SHIFT * input;
                }
                Some(product)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                FwError::InvalidRequest(
                    "speaker-count eigensolver encountered an invalid sparse product".to_owned(),
                )
            })?;
        if !orthonormalize_columns(&mut next) {
            return Ok(SparseEigengapRun::unavailable(
                SpeakerCountLaneUnavailableReason::SolverDidNotConverge,
                iteration + 1,
                None,
                subspace_dimensions,
            ));
        }
        basis = next;
        let applied = basis
            .iter()
            .map(|column| graph.apply_normalized(column))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                FwError::InvalidRequest(
                    "speaker-count eigensolver encountered an invalid Rayleigh product".to_owned(),
                )
            })?;
        for row in 0..subspace_dimensions {
            for column in 0..subspace_dimensions {
                rayleigh[row][column] = dot_f64(&basis[row], &applied[column])?;
            }
        }
        residual = invariant_subspace_residual(&basis, &applied, &rayleigh)?;
        iterations = iteration + 1;
        if residual <= SPEAKER_COUNT_EIGENSOLVER_TOLERANCE {
            break;
        }
    }
    if residual > SPEAKER_COUNT_EIGENSOLVER_TOLERANCE {
        return Ok(SparseEigengapRun::unavailable(
            SpeakerCountLaneUnavailableReason::SolverDidNotConverge,
            iterations,
            residual.is_finite().then_some(residual),
            subspace_dimensions,
        ));
    }
    let mut eigenvalues = jacobi_symmetric_eigenvalues(rayleigh)?;
    eigenvalues.sort_by(|left, right| right.total_cmp(left));
    for eigenvalue in &mut eigenvalues {
        *eigenvalue = eigenvalue.clamp(-1.0, 1.0);
    }
    let mut scores = Vec::<(usize, f64)>::new();
    for count in minimum_count..=bounded_maximum {
        let Some(&left) = eigenvalues.get(count - 1) else {
            continue;
        };
        let Some(&right) = eigenvalues.get(count) else {
            // An eigengap at K requires both lambda_K and lambda_(K+1).
            // Treating a missing lambda_(K+1) as zero manufactures a large
            // terminal gap whenever K equals the number of prototypes.
            continue;
        };
        let gap = (left - right).max(0.0);
        let normalized_gap = gap / left.abs().max(1.0e-9);
        if normalized_gap.is_finite() {
            scores.push((count, normalized_gap));
        }
    }
    let Some(&(count, best_score)) = scores.iter().max_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| right.0.cmp(&left.0))
    }) else {
        return Ok(SparseEigengapRun::unavailable(
            SpeakerCountLaneUnavailableReason::InvalidAffinity,
            iterations,
            Some(residual),
            subspace_dimensions,
        ));
    };
    let total_score = scores.iter().map(|(_, score)| *score).sum::<f64>();
    if best_score <= 0.0 || !total_score.is_finite() || total_score <= 0.0 {
        return Ok(SparseEigengapRun::unavailable(
            SpeakerCountLaneUnavailableReason::InvalidAffinity,
            iterations,
            Some(residual),
            subspace_dimensions,
        ));
    }
    let proposal = SparseEigengapProposal {
        count,
        confidence: (best_score / total_score).clamp(0.0, 1.0),
        eigenvalues,
        residual,
        iterations,
    };
    Ok(SparseEigengapRun {
        proposal: Some(proposal),
        iterations,
        residual: Some(residual),
        subspace_dimensions,
        unavailable_reason: None,
    })
}

#[cfg(test)]
fn sparse_normalized_eigengap<C>(
    graph: &SparseSpeakerAffinityGraph,
    minimum_count: usize,
    maximum_count: usize,
    is_cancelled: C,
) -> FwResult<Option<SparseEigengapProposal>>
where
    C: FnMut() -> bool,
{
    Ok(sparse_normalized_eigengap_run(graph, minimum_count, maximum_count, is_cancelled)?.proposal)
}

fn deterministic_spectral_basis(
    graph: &SparseSpeakerAffinityGraph,
    dimensions: usize,
) -> FwResult<Vec<Vec<f64>>> {
    let node_count = graph.rows.len();
    if dimensions == 0 || dimensions > node_count {
        return Err(FwError::InvalidRequest(
            "speaker-count eigensolver basis dimensions are invalid".to_owned(),
        ));
    }
    let mut basis = Vec::with_capacity(dimensions);
    for column in 0..dimensions {
        let mut values = Vec::with_capacity(node_count);
        for (row, degree) in graph.degrees.iter().copied().enumerate() {
            let value = if column == 0 {
                degree.sqrt()
            } else {
                let angle =
                    std::f64::consts::PI * column as f64 * (row as f64 + 0.5) / node_count as f64;
                angle.cos()
            };
            values.push(value);
        }
        basis.push(values);
    }
    if !orthonormalize_columns(&mut basis) {
        return Err(FwError::InvalidRequest(
            "speaker-count eigensolver basis is rank deficient".to_owned(),
        ));
    }
    Ok(basis)
}

fn orthonormalize_columns(columns: &mut [Vec<f64>]) -> bool {
    for column in 0..columns.len() {
        let (previous_columns, current_and_remaining) = columns.split_at_mut(column);
        let current = &mut current_and_remaining[0];
        for _ in 0..2 {
            for previous in previous_columns.iter() {
                let Ok(projection) = dot_f64(current, previous) else {
                    return false;
                };
                for row in 0..current.len() {
                    current[row] -= projection * previous[row];
                }
            }
        }
        let Ok(norm_squared) = dot_f64(current, current) else {
            return false;
        };
        let norm = norm_squared.sqrt();
        if !norm.is_finite() || norm <= 1.0e-12 {
            return false;
        }
        for value in current {
            *value /= norm;
        }
    }
    true
}

fn invariant_subspace_residual(
    basis: &[Vec<f64>],
    applied: &[Vec<f64>],
    rayleigh: &[Vec<f64>],
) -> FwResult<f64> {
    if basis.len() != applied.len() || rayleigh.len() != basis.len() {
        return Err(FwError::InvalidRequest(
            "speaker-count eigensolver residual dimensions do not match".to_owned(),
        ));
    }
    let mut maximum = 0.0_f64;
    for column in 0..basis.len() {
        for row in 0..basis[column].len() {
            let mut projected = 0.0;
            for component in 0..basis.len() {
                projected += basis[component][row] * rayleigh[component][column];
            }
            maximum = maximum.max((applied[column][row] - projected).abs());
        }
    }
    if maximum.is_finite() {
        Ok(maximum)
    } else {
        Err(FwError::InvalidRequest(
            "speaker-count eigensolver residual is non-finite".to_owned(),
        ))
    }
}

fn jacobi_symmetric_eigenvalues(mut matrix: Vec<Vec<f64>>) -> FwResult<Vec<f64>> {
    let dimensions = matrix.len();
    if dimensions == 0 || matrix.iter().any(|row| row.len() != dimensions) {
        return Err(FwError::InvalidRequest(
            "speaker-count Rayleigh matrix is not square".to_owned(),
        ));
    }
    for matrix_row in &matrix {
        for value in matrix_row {
            if !value.is_finite() {
                return Err(FwError::InvalidRequest(
                    "speaker-count Rayleigh matrix is non-finite".to_owned(),
                ));
            }
        }
    }
    let mut row = 0;
    while row < dimensions {
        let mut column = row + 1;
        while column < dimensions {
            let symmetric = 0.5 * (matrix[row][column] + matrix[column][row]);
            matrix[row][column] = symmetric;
            matrix[column][row] = symmetric;
            column += 1;
        }
        row += 1;
    }
    let iteration_cap = dimensions
        .checked_mul(dimensions)
        .and_then(|value| value.checked_mul(64))
        .ok_or_else(|| {
            FwError::InvalidRequest("speaker-count Jacobi iteration cap overflow".to_owned())
        })?;
    for _ in 0..iteration_cap {
        let mut largest = 0.0_f64;
        let mut pivot = None;
        for (row, matrix_row) in matrix.iter().enumerate() {
            for (column, value) in matrix_row.iter().enumerate().skip(row + 1) {
                let magnitude = value.abs();
                if magnitude > largest {
                    largest = magnitude;
                    pivot = Some((row, column));
                }
            }
        }
        if largest <= 1.0e-12 {
            return Ok((0..dimensions).map(|index| matrix[index][index]).collect());
        }
        let Some((left, right)) = pivot else {
            break;
        };
        let angle =
            0.5 * (2.0 * matrix[left][right]).atan2(matrix[right][right] - matrix[left][left]);
        let cosine = angle.cos();
        let sine = angle.sin();
        let left_diagonal = matrix[left][left];
        let right_diagonal = matrix[right][right];
        let off_diagonal = matrix[left][right];
        matrix[left][left] = cosine * cosine * left_diagonal - 2.0 * sine * cosine * off_diagonal
            + sine * sine * right_diagonal;
        matrix[right][right] = sine * sine * left_diagonal
            + 2.0 * sine * cosine * off_diagonal
            + cosine * cosine * right_diagonal;
        matrix[left][right] = 0.0;
        matrix[right][left] = 0.0;
        let mut index = 0;
        while index < dimensions {
            if index == left || index == right {
                index += 1;
                continue;
            }
            let left_value = matrix[index][left];
            let right_value = matrix[index][right];
            matrix[index][left] = cosine * left_value - sine * right_value;
            matrix[left][index] = matrix[index][left];
            matrix[index][right] = sine * left_value + cosine * right_value;
            matrix[right][index] = matrix[index][right];
            index += 1;
        }
    }
    Err(FwError::InvalidRequest(
        "speaker-count Jacobi eigensolver did not converge".to_owned(),
    ))
}

fn dot_f64(left: &[f64], right: &[f64]) -> FwResult<f64> {
    if left.len() != right.len() {
        return Err(FwError::InvalidRequest(
            "speaker-count eigensolver vector dimensions do not match".to_owned(),
        ));
    }
    let value = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(FwError::InvalidRequest(
            "speaker-count eigensolver dot product is non-finite".to_owned(),
        ))
    }
}

fn probabilistic_agglomeration_lane<C>(
    initial: &[AcousticCluster],
    cannot_links: &BTreeSet<(String, String)>,
    policy: SpeakerCountPolicy,
    perturbation: SpeakerPairPerturbation,
    is_cancelled: &mut C,
) -> FwResult<Option<ProbabilisticLaneResult>>
where
    C: FnMut() -> bool,
{
    let mut clusters = initial.iter().cloned().map(Some).collect::<Vec<_>>();
    let mut generations = vec![0u32; clusters.len()];
    let mut heap = BinaryHeap::new();
    let mut compatible_pairs = 0usize;
    for left in 0..clusters.len() {
        for right in left + 1..clusters.len() {
            let Some(left_cluster) = clusters[left].as_ref() else {
                continue;
            };
            let Some(right_cluster) = clusters[right].as_ref() else {
                continue;
            };
            if !clusters_compatible(left_cluster, right_cluster, cannot_links) {
                continue;
            }
            compatible_pairs += 1;
            if let Some(evidence) = cluster_pair_evidence(left_cluster, right_cluster, perturbation)
            {
                heap.push(MergeCandidate {
                    distance: 1.0 - evidence.same_speaker_probability,
                    left,
                    right,
                    left_generation: 0,
                    right_generation: 0,
                });
            }
        }
    }
    if compatible_pairs > 0 && heap.is_empty() {
        return Ok(None);
    }

    let initial_count = clusters.len();
    let mut active = initial_count;
    let mut steps = Vec::with_capacity(initial_count.saturating_sub(policy.min));
    while active > policy.min {
        if steps
            .len()
            .is_multiple_of(ACOUSTIC_CANCELLATION_INTERVAL_FRAMES)
            && is_cancelled()
        {
            return Err(FwError::Cancelled(format!(
                "probabilistic acoustic clustering cancelled after {} merges",
                steps.len()
            )));
        }
        let candidate = loop {
            let Some(candidate) = heap.pop() else {
                if active > policy.max {
                    return Ok(None);
                }
                break None;
            };
            let valid = clusters[candidate.left].is_some()
                && clusters[candidate.right].is_some()
                && generations[candidate.left] == candidate.left_generation
                && generations[candidate.right] == candidate.right_generation;
            if valid {
                break Some(candidate);
            }
        };
        let Some(candidate) = candidate else {
            break;
        };
        let left_before = clusters[candidate.left].as_ref().ok_or_else(|| {
            FwError::InvalidRequest(
                "probabilistic merge candidate left cluster disappeared".to_owned(),
            )
        })?;
        let right_before = clusters[candidate.right].as_ref().ok_or_else(|| {
            FwError::InvalidRequest(
                "probabilistic merge candidate right cluster disappeared".to_owned(),
            )
        })?;
        let Some(evidence) = cluster_pair_evidence(left_before, right_before, perturbation) else {
            return Ok(None);
        };
        let right = clusters[candidate.right].take().ok_or_else(|| {
            FwError::InvalidRequest(
                "probabilistic merge candidate right cluster disappeared".to_owned(),
            )
        })?;
        let left = clusters[candidate.left].as_mut().ok_or_else(|| {
            FwError::InvalidRequest(
                "probabilistic merge candidate left cluster disappeared".to_owned(),
            )
        })?;
        merge_cluster(left, &right);
        generations[candidate.left] = generations[candidate.left].wrapping_add(1);
        generations[candidate.right] = generations[candidate.right].wrapping_add(1);
        active -= 1;
        steps.push(ProbabilisticLaneMergeStep {
            left: candidate.left,
            right: candidate.right,
            remaining_clusters: active,
            same_speaker_probability: evidence.same_speaker_probability,
        });
        for other in 0..clusters.len() {
            if other == candidate.left || clusters[other].is_none() {
                continue;
            }
            let (first, second) = if candidate.left < other {
                (candidate.left, other)
            } else {
                (other, candidate.left)
            };
            let Some(first_cluster) = clusters[first].as_ref() else {
                continue;
            };
            let Some(second_cluster) = clusters[second].as_ref() else {
                continue;
            };
            if !clusters_compatible(first_cluster, second_cluster, cannot_links) {
                continue;
            }
            if let Some(evidence) =
                cluster_pair_evidence(first_cluster, second_cluster, perturbation)
            {
                heap.push(MergeCandidate {
                    distance: 1.0 - evidence.same_speaker_probability,
                    left: first,
                    right: second,
                    left_generation: generations[first],
                    right_generation: generations[second],
                });
            }
        }
    }

    let Some(risk_curve) = speaker_count_expected_loss_curve(initial_count, &steps, policy) else {
        return Ok(None);
    };
    let selected_count = risk_curve.selected_count;

    let mut groups = (0..initial_count)
        .map(|index| Some(vec![index]))
        .collect::<Vec<_>>();
    let accepted_merges = initial_count.saturating_sub(selected_count);
    for step in steps.iter().take(accepted_merges) {
        let right = groups[step.right].take().ok_or_else(|| {
            FwError::InvalidRequest(
                "probabilistic lane replay found an absent right cluster".to_owned(),
            )
        })?;
        let left = groups[step.left].as_mut().ok_or_else(|| {
            FwError::InvalidRequest(
                "probabilistic lane replay found an absent left cluster".to_owned(),
            )
        })?;
        left.extend(right);
        left.sort_unstable();
    }
    Ok(Some(ProbabilisticLaneResult {
        selected_count,
        groups: groups.into_iter().flatten().collect(),
        risk_curve,
    }))
}

fn coassociation_consensus_clusters<C>(
    initial: &[AcousticCluster],
    cannot_links: &BTreeSet<(String, String)>,
    lanes: &[ProbabilisticLaneResult],
    selected_count: usize,
    is_cancelled: &mut C,
) -> FwResult<Option<(Vec<AcousticCluster>, Vec<ClusterMergeTrace>)>>
where
    C: FnMut() -> bool,
{
    let initial_count = initial.len();
    let mut coassociation = vec![vec![0u8; initial_count]; initial_count];
    for lane in lanes {
        for group in &lane.groups {
            for &left in group {
                for &right in group {
                    coassociation[left][right] = coassociation[left][right].saturating_add(1);
                }
            }
        }
    }
    let mut clusters = initial.iter().cloned().map(Some).collect::<Vec<_>>();
    let mut members = (0..initial_count)
        .map(|index| Some(vec![index]))
        .collect::<Vec<_>>();
    let mut active = initial_count;
    let mut merge_trace = Vec::with_capacity(initial_count.saturating_sub(selected_count));
    while active > selected_count {
        if merge_trace
            .len()
            .is_multiple_of(ACOUSTIC_CANCELLATION_INTERVAL_FRAMES)
            && is_cancelled()
        {
            return Err(FwError::Cancelled(format!(
                "co-association clustering cancelled after {} merges",
                merge_trace.len()
            )));
        }
        let mut best = None::<(f32, usize, usize)>;
        for left in 0..clusters.len() {
            let (Some(left_cluster), Some(left_members)) =
                (clusters[left].as_ref(), members[left].as_ref())
            else {
                continue;
            };
            for right in left + 1..clusters.len() {
                let (Some(right_cluster), Some(right_members)) =
                    (clusters[right].as_ref(), members[right].as_ref())
                else {
                    continue;
                };
                if !clusters_compatible(left_cluster, right_cluster, cannot_links) {
                    continue;
                }
                let mut support_sum = 0u64;
                for &left_member in left_members {
                    for &right_member in right_members {
                        support_sum = support_sum
                            .saturating_add(u64::from(coassociation[left_member][right_member]));
                    }
                }
                let denominator = left_members
                    .len()
                    .saturating_mul(right_members.len())
                    .saturating_mul(lanes.len());
                if denominator == 0 {
                    continue;
                }
                let support = support_sum as f32 / denominator as f32;
                if support + f32::EPSILON
                    < acoustic_speaker_pair_calibration().minimum_stable_lane_fraction
                {
                    continue;
                }
                if best
                    .as_ref()
                    .is_none_or(|&(best_support, best_left, best_right)| {
                        support > best_support
                            || (support.to_bits() == best_support.to_bits()
                                && (left, right) < (best_left, best_right))
                    })
                {
                    best = Some((support, left, right));
                }
            }
        }
        let Some((support, left_index, right_index)) = best else {
            return Ok(None);
        };
        let left_before = clusters[left_index].as_ref().ok_or_else(|| {
            FwError::InvalidRequest("consensus left cluster disappeared".to_owned())
        })?;
        let right_before = clusters[right_index].as_ref().ok_or_else(|| {
            FwError::InvalidRequest("consensus right cluster disappeared".to_owned())
        })?;
        let pair_evidence =
            cluster_pair_evidence(left_before, right_before, SpeakerPairPerturbation::Full);
        let left_anchor = left_before.hard_anchor.clone();
        let right_anchor = right_before.hard_anchor.clone();
        let right_cluster = clusters[right_index].take().ok_or_else(|| {
            FwError::InvalidRequest("consensus right cluster disappeared".to_owned())
        })?;
        let left_cluster = clusters[left_index].as_mut().ok_or_else(|| {
            FwError::InvalidRequest("consensus left cluster disappeared".to_owned())
        })?;
        merge_cluster(left_cluster, &right_cluster);
        let right_members = members[right_index].take().ok_or_else(|| {
            FwError::InvalidRequest("consensus right membership disappeared".to_owned())
        })?;
        let left_members = members[left_index].as_mut().ok_or_else(|| {
            FwError::InvalidRequest("consensus left membership disappeared".to_owned())
        })?;
        left_members.extend(right_members);
        left_members.sort_unstable();
        active -= 1;
        merge_trace.push(ClusterMergeTrace {
            remaining_clusters: active,
            distance: 1.0 - support,
            same_speaker_probability: Some(support),
            voice_distance: pair_evidence.map(|evidence| evidence.voice_distance),
            channel_distance: pair_evidence.map(|evidence| evidence.channel_distance),
            left_anchor,
            right_anchor,
        });
    }
    Ok(Some((
        clusters.into_iter().flatten().collect(),
        merge_trace,
    )))
}

fn speaker_count_expected_loss_curve(
    initial_count: usize,
    steps: &[ProbabilisticLaneMergeStep],
    policy: SpeakerCountPolicy,
) -> Option<SpeakerCountRiskCurve> {
    let calibration = acoustic_speaker_pair_calibration();
    let mut expected_loss = steps
        .iter()
        .map(|step| {
            f64::from(step.same_speaker_probability) * f64::from(calibration.false_split_loss)
        })
        .sum::<f64>();
    if !expected_loss.is_finite() {
        return None;
    }
    let mut points = Vec::with_capacity(steps.len().saturating_add(1));
    if speaker_count_allowed(initial_count, policy) {
        points.push(SpeakerCountRiskPoint {
            count: initial_count,
            expected_loss,
        });
    }
    for step in steps {
        let same_probability = f64::from(step.same_speaker_probability);
        expected_loss += (1.0 - same_probability) * f64::from(calibration.false_merge_loss)
            - same_probability * f64::from(calibration.false_split_loss);
        if !expected_loss.is_finite() {
            return None;
        }
        if speaker_count_allowed(step.remaining_clusters, policy) {
            points.push(SpeakerCountRiskPoint {
                count: step.remaining_clusters,
                expected_loss,
            });
        }
    }
    points.sort_by_key(|point| point.count);
    let selected_count = points
        .iter()
        .min_by(|left, right| {
            left.expected_loss
                .total_cmp(&right.expected_loss)
                .then_with(|| right.count.cmp(&left.count))
        })?
        .count;
    Some(SpeakerCountRiskCurve {
        selected_count,
        points,
    })
}

fn speaker_count_allowed(count: usize, policy: SpeakerCountPolicy) -> bool {
    policy
        .exact
        .map_or(count >= policy.min && count <= policy.max, |exact| {
            count == exact
        })
}

fn evaluate_cluster_count(
    clusters: &[Option<AcousticCluster>],
    active: usize,
    policy: SpeakerCountPolicy,
    merge_trace_len: usize,
    best: &mut Option<(f32, Vec<AcousticCluster>, usize)>,
) {
    let allowed = policy
        .exact
        .map_or(active >= policy.min && active <= policy.max, |exact| {
            active == exact
        });
    if !allowed {
        return;
    }
    let current = clusters.iter().flatten().cloned().collect::<Vec<_>>();
    let total_weight = current
        .iter()
        .map(|cluster| cluster.weight)
        .sum::<f32>()
        .max(2.0);
    let sse = current.iter().map(|cluster| cluster.sse).sum::<f32>();
    let voice_parameter_count = current
        .iter()
        .map(|cluster| cluster.voice_valid.iter().filter(|&&valid| valid).count())
        .sum::<usize>();
    let penalty = 0.035 * voice_parameter_count as f32 * total_weight.ln();
    let objective = sse + penalty;
    if best
        .as_ref()
        .is_none_or(|(best_objective, _, _)| objective < *best_objective)
    {
        *best = Some((objective, current, merge_trace_len));
    }
}

fn canonical_cluster_labels(
    clusters: &[AcousticCluster],
    enrollment: &SpeakerEnrollment,
) -> Vec<String> {
    let mut unanchored = clusters
        .iter()
        .enumerate()
        .filter(|(_, cluster)| cluster.hard_anchor.is_none())
        .collect::<Vec<_>>();
    unanchored.sort_by(|(left_index, left), (right_index, right)| {
        left.label_hint
            .is_none()
            .cmp(&right.label_hint.is_none())
            .then(left.earliest_ms.cmp(&right.earliest_ms))
            .then_with(|| compare_float_vectors(&left.voice, &right.voice))
            .then(left_index.cmp(right_index))
    });
    let reserved = clusters
        .iter()
        .filter_map(|cluster| cluster.hard_anchor.as_ref().or(cluster.label_hint.as_ref()))
        .cloned()
        .chain(enrollment.reserved_speaker_refs.iter().cloned())
        .collect::<BTreeSet<_>>();
    let mut generated = BTreeMap::new();
    let mut used = reserved.clone();
    let mut generated_index = 0usize;
    for (cluster_index, cluster) in unanchored {
        let label = if let Some(label_hint) = &cluster.label_hint {
            label_hint.clone()
        } else {
            loop {
                let candidate = format!("SPEAKER_{generated_index:02}");
                generated_index += 1;
                if !used.contains(&candidate) {
                    break candidate;
                }
            }
        };
        used.insert(label.clone());
        generated.insert(cluster_index, label);
    }
    clusters
        .iter()
        .enumerate()
        .map(|(index, cluster)| {
            cluster
                .hard_anchor
                .clone()
                .unwrap_or_else(|| generated[&index].clone())
        })
        .collect()
}

fn viterbi_assignments<C>(
    tracklets: &[AcousticTracklet],
    clusters: &[AcousticCluster],
    labels: &[String],
    enrollment: &SpeakerEnrollment,
    clustering_mode: AcousticClusteringMode,
    is_cancelled: &mut C,
) -> FwResult<Vec<AcousticSpeakerAssignment>>
where
    C: FnMut() -> bool,
{
    if tracklets.is_empty() {
        return Ok(Vec::new());
    }
    if clusters.len() != labels.len() {
        return Err(FwError::InvalidRequest(
            "acoustic cluster labels must have exactly one entry per cluster".to_owned(),
        ));
    }
    for tracklet in tracklets {
        if let Some(hard) = enrollment.hard_assignments.get(&tracklet.tracklet_index)
            && !labels.iter().any(|label| label == hard)
        {
            return Err(FwError::InvalidRequest(format!(
                "hard speaker reference {hard:?} has no acoustic assignment state"
            )));
        }
    }
    let state_count = clusters.len() + 1;
    let unknown_state = clusters.len();
    let mut emissions = Vec::with_capacity(tracklets.len());
    for (index, tracklet) in tracklets.iter().enumerate() {
        if index % ACOUSTIC_CANCELLATION_INTERVAL_FRAMES == 0 && is_cancelled() {
            return Err(FwError::Cancelled(format!(
                "acoustic smoothing cancelled at tracklet {index}"
            )));
        }
        let mut costs = clusters
            .iter()
            .enumerate()
            .map(|(cluster_index, cluster)| {
                if let Some(hard) = enrollment.hard_assignments.get(&tracklet.tracklet_index) {
                    return if labels[cluster_index] == *hard {
                        0.0
                    } else {
                        f32::INFINITY
                    };
                }
                let enrolled_profile = enrollment.profiles.get(&labels[cluster_index]);
                let mut cost = if clustering_mode == AcousticClusteringMode::ProbabilisticV1 {
                    let cluster_evidence = tracklet_cluster_pair_evidence(tracklet, cluster);
                    let profile_evidence = enrolled_profile.and_then(|profile| {
                        tracklet_profile_pair_evidence(
                            tracklet,
                            profile,
                            &enrollment.voice_dimension_weights,
                        )
                    });
                    cluster_evidence
                        .into_iter()
                        .chain(profile_evidence)
                        .max_by(|left, right| {
                            left.same_speaker_probability
                                .total_cmp(&right.same_speaker_probability)
                        })
                        .map_or(1_000_000.0, |value| {
                            -value.same_speaker_probability.max(1e-6).ln()
                        })
                } else {
                    enrolled_profile.map_or_else(
                        || tracklet_cluster_distance(tracklet, cluster),
                        |profile| {
                            tracklet_cluster_distance(tracklet, cluster).min(
                                tracklet_profile_distance(
                                    tracklet,
                                    profile,
                                    &enrollment.voice_dimension_weights,
                                ),
                            )
                        },
                    )
                };
                if let Some(prior) = enrollment
                    .soft_priors
                    .get(&(tracklet.tracklet_index, labels[cluster_index].clone()))
                {
                    cost = (cost - 0.015 * prior).max(0.0);
                }
                if tracklet.frame_count < MIN_TRACKLET_FRAMES {
                    cost += 0.35;
                }
                if tracklet.overlap_suspected {
                    cost += 0.20;
                }
                cost
            })
            .collect::<Vec<_>>();
        let unknown_cost = if enrollment
            .hard_assignments
            .contains_key(&tracklet.tracklet_index)
        {
            f32::INFINITY
        } else if clustering_mode == AcousticClusteringMode::ProbabilisticV1 {
            let best_same_probability = costs
                .iter()
                .copied()
                .filter(|cost| cost.is_finite())
                .map(|cost| (-cost).exp())
                .max_by(f32::total_cmp)
                .unwrap_or(0.0);
            let calibration = acoustic_speaker_pair_calibration();
            let unknown_probability = if tracklet.frame_count < MIN_TRACKLET_FRAMES
                || tracklet.voiced_frame_count.saturating_mul(4) < tracklet.frame_count
            {
                calibration.maximum_unknown_prior
            } else {
                (1.0 - best_same_probability).clamp(0.05, calibration.maximum_unknown_prior)
            };
            -unknown_probability.ln()
        } else if tracklet.frame_count < MIN_TRACKLET_FRAMES
            || tracklet.voiced_frame_count.saturating_mul(4) < tracklet.frame_count
        {
            0.20
        } else {
            0.90
        };
        costs.push(unknown_cost);
        emissions.push(costs);
    }

    let mut previous = emissions[0].clone();
    let initial_duration_ms = tracklets[0].end_ms.saturating_sub(tracklets[0].start_ms);
    let mut previous_run_duration_ms = vec![initial_duration_ms; state_count];
    let mut backpointers = vec![vec![0usize; state_count]; tracklets.len()];
    for time in 1..tracklets.len() {
        let mut current = vec![f32::INFINITY; state_count];
        let mut current_run_duration_ms = vec![0_u64; state_count];
        let current_duration_ms = tracklets[time]
            .end_ms
            .saturating_sub(tracklets[time].start_ms);
        let gap_ms = tracklets[time]
            .start_ms
            .saturating_sub(tracklets[time - 1].end_ms);
        for state in 0..state_count {
            for (previous_state, previous_cost) in previous.iter().enumerate().take(state_count) {
                let switch_penalty = if state == previous_state {
                    0.0
                } else if clustering_mode == AcousticClusteringMode::ProbabilisticV1 {
                    duration_aware_switch_penalty(
                        previous_run_duration_ms[previous_state],
                        current_duration_ms,
                        gap_ms,
                        tracklets[time].change_confidence,
                        state == unknown_state || previous_state == unknown_state,
                    )
                } else if state == unknown_state || previous_state == unknown_state {
                    0.08
                } else {
                    0.18
                };
                let candidate = *previous_cost + switch_penalty + emissions[time][state];
                if candidate < current[state] {
                    current[state] = candidate;
                    backpointers[time][state] = previous_state;
                    current_run_duration_ms[state] = if state == previous_state {
                        previous_run_duration_ms[previous_state].saturating_add(current_duration_ms)
                    } else {
                        current_duration_ms
                    };
                }
            }
        }
        previous = current;
        previous_run_duration_ms = current_run_duration_ms;
    }
    if previous.iter().all(|cost| !cost.is_finite()) {
        return Err(FwError::InvalidRequest(
            "hard speaker hints leave no finite acoustic assignment path".to_owned(),
        ));
    }
    let mut state = previous
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.total_cmp(right.1).then(left.0.cmp(&right.0)))
        .map_or(unknown_state, |(index, _)| index);
    let mut states = vec![unknown_state; tracklets.len()];
    for time in (0..tracklets.len()).rev() {
        states[time] = state;
        if time > 0 {
            state = backpointers[time][state];
        }
    }

    Ok(tracklets
        .iter()
        .enumerate()
        .map(|(index, tracklet)| {
            let hard = enrollment
                .hard_assignments
                .contains_key(&tracklet.tracklet_index);
            let state = states[index];
            if state == unknown_state {
                return unknown_assignment(tracklet);
            }
            let chosen_cost = emissions[index][state];
            let second_cost = emissions[index]
                .iter()
                .enumerate()
                .filter(|(candidate, _)| *candidate != state)
                .map(|(_, &cost)| cost)
                .min_by(f32::total_cmp)
                .unwrap_or(chosen_cost + 1.0);
            let margin = (second_cost - chosen_cost).max(0.0);
            let raw_confidence = if hard {
                1.0
            } else if clustering_mode == AcousticClusteringMode::ProbabilisticV1 {
                let chosen_likelihood = (-chosen_cost.min(80.0)).exp();
                let strongest_alternative = emissions[index]
                    .iter()
                    .enumerate()
                    .filter(|(candidate, _)| *candidate != state)
                    .map(|(_, cost)| (-cost.min(80.0)).exp())
                    .max_by(f32::total_cmp)
                    .unwrap_or(0.0);
                let pairwise_share =
                    chosen_likelihood / (chosen_likelihood + strongest_alternative).max(1e-6);
                let discrimination = 0.5 + 0.5 * pairwise_share;
                let reliability = 0.85 + 0.15 * clusters[state].reliability.clamp(0.0, 1.0);
                (chosen_likelihood * discrimination * reliability).clamp(0.0, 1.0)
            } else {
                // Assignment discrimination and accumulated profile reliability
                // are separate evidence gates. Multiplying by raw reliability
                // here counted cluster occupancy twice and systematically pushed
                // recurring minority speakers below the rejection threshold.
                // Keep only a bounded calibration adjustment in assignment
                // confidence; evaluate_speaker_evidence enforces reliability
                // independently before a label can become authoritative.
                let discrimination = margin / (margin + 0.5);
                let reliability = 0.85 + 0.15 * clusters[state].reliability.clamp(0.0, 1.0);
                (discrimination * reliability).clamp(0.0, 1.0)
            };
            let reject = if clustering_mode == AcousticClusteringMode::ProbabilisticV1 {
                raw_confidence < acoustic_speaker_pair_calibration().minimum_assignment_probability
            } else {
                chosen_cost > 1.35 || raw_confidence < 0.30
            };
            if !hard && reject {
                unknown_assignment(tracklet)
            } else {
                let reported_confidence = if hard {
                    1.0
                } else if clustering_mode == AcousticClusteringMode::ProbabilisticV1 {
                    calibrate_assignment_confidence(raw_confidence)
                } else {
                    raw_confidence
                };
                let secondary = if !hard
                    && clustering_mode == AcousticClusteringMode::ProbabilisticV1
                    && tracklet.overlap_suspected
                {
                    emissions[index][..clusters.len()]
                        .iter()
                        .enumerate()
                        .filter(|(candidate, cost)| *candidate != state && cost.is_finite())
                        .map(|(candidate, cost)| (candidate, (-cost.min(80.0)).exp()))
                        .filter(|(_, likelihood)| {
                            let chosen_likelihood = (-chosen_cost.min(80.0)).exp();
                            *likelihood >= 0.55
                                && chosen_likelihood >= 0.55
                                && *likelihood / chosen_likelihood.max(1e-6) >= 0.65
                        })
                        .max_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)))
                } else {
                    None
                };
                AcousticSpeakerAssignment {
                    tracklet_index: tracklet.tracklet_index,
                    start_ms: tracklet.start_ms,
                    end_ms: tracklet.end_ms,
                    speaker_ref: Some(labels[state].clone()),
                    speaker_confidence: reported_confidence,
                    secondary_speaker_ref: secondary
                        .map(|(secondary_state, _)| labels[secondary_state].clone()),
                    secondary_speaker_confidence: secondary.map(|(_, likelihood)| {
                        (likelihood * tracklet.overlap_probability).clamp(0.0, 1.0)
                    }),
                    change_confidence: tracklet.change_confidence,
                    overlap_suspected: tracklet.overlap_suspected,
                    hard_attribution: hard,
                }
            }
        })
        .collect())
}

fn calibrate_assignment_confidence(raw_confidence: f32) -> f32 {
    let bounded = raw_confidence.clamp(0.0, 1.0);
    (ACOUSTIC_ASSIGNMENT_CONFIDENCE_FLOOR + ACOUSTIC_ASSIGNMENT_CONFIDENCE_SCALE * bounded)
        .clamp(0.0, 1.0)
}

fn duration_aware_switch_penalty(
    previous_run_duration_ms: u64,
    current_tracklet_duration_ms: u64,
    gap_ms: u64,
    boundary_confidence: f32,
    touches_unknown: bool,
) -> f32 {
    let boundary_confidence = boundary_confidence.clamp(0.0, 1.0);
    let base = if touches_unknown {
        TEMPORAL_UNKNOWN_SWITCH_BASE
    } else {
        TEMPORAL_KNOWN_SWITCH_BASE
    };
    let boundary_credit = if touches_unknown {
        TEMPORAL_UNKNOWN_BOUNDARY_CREDIT * boundary_confidence
    } else {
        TEMPORAL_KNOWN_BOUNDARY_CREDIT * boundary_confidence
    };
    let gap_credit = TEMPORAL_MAX_GAP_CREDIT
        * (gap_ms.min(TEMPORAL_FULL_GAP_MS) as f32 / TEMPORAL_FULL_GAP_MS as f32);
    let premature_switch_penalty = if previous_run_duration_ms < TEMPORAL_SHORT_RUN_MS {
        TEMPORAL_PREMATURE_SWITCH_PENALTY
            * (1.0 - previous_run_duration_ms as f32 / TEMPORAL_SHORT_RUN_MS as f32)
            * (1.0 - boundary_confidence)
    } else {
        0.0
    };
    let fragment_penalty = if current_tracklet_duration_ms < TEMPORAL_FRAGMENT_MS {
        TEMPORAL_FRAGMENT_PENALTY
            * (1.0 - current_tracklet_duration_ms as f32 / TEMPORAL_FRAGMENT_MS as f32)
            * (1.0 - boundary_confidence)
    } else {
        0.0
    };
    (base - boundary_credit - gap_credit + premature_switch_penalty + fragment_penalty)
        .clamp(0.01, 0.45)
}

fn tracklet_cluster_distance(tracklet: &AcousticTracklet, cluster: &AcousticCluster) -> f32 {
    masked_variance_normalized_distance(
        &tracklet.voice_mean,
        &tracklet.voice_valid,
        &tracklet.voice_variance.map(|value| value.max(0.025)),
        &cluster.voice,
        &cluster.voice_valid,
        &cluster.scale,
    )
    .unwrap_or(10.0)
        + channel_distance(
            &tracklet.channel_mean,
            tracklet.channel_valid,
            &cluster.channel,
            cluster.channel_valid,
            tracklet.channel_dimensions.min(cluster.channel_dimensions),
        )
}

fn tracklet_cluster_pair_evidence(
    tracklet: &AcousticTracklet,
    cluster: &AcousticCluster,
) -> Option<AcousticSpeakerPairEvidence> {
    let tracklet_scale = tracklet.voice_variance.map(|variance| variance.max(0.025));
    speaker_pair_evidence_from_statistics(
        &tracklet.voice_mean,
        &tracklet.voice_valid,
        &tracklet_scale,
        tracklet.identity_frame_count as f32,
        &tracklet.channel_mean,
        tracklet.channel_valid,
        tracklet.channel_dimensions,
        &cluster.voice,
        &cluster.voice_valid,
        &cluster.scale,
        cluster.weight,
        &cluster.channel,
        cluster.channel_valid,
        cluster.channel_dimensions,
        SpeakerPairPerturbation::Full,
    )
}

fn tracklet_profile_distance(
    tracklet: &AcousticTracklet,
    profile: &AcousticSpeakerProfile,
    dimension_weights: &[f32; VOICE_VECTOR_DIMENSIONS],
) -> f32 {
    let tracklet_scale = tracklet.voice_variance.map(|variance| variance.max(0.025));
    let voice_distance = profile
        .voice_subprofiles
        .iter()
        .filter_map(|mode| {
            let adapted_scale = std::array::from_fn(|dimension| {
                mode.scale[dimension] / dimension_weights[dimension].max(0.25)
            });
            masked_variance_normalized_distance(
                &tracklet.voice_mean,
                &tracklet.voice_valid,
                &tracklet_scale,
                &mode.center,
                &mode.valid,
                &adapted_scale,
            )
        })
        .min_by(f32::total_cmp)
        .unwrap_or(10.0);
    let channel_distance = profile
        .channel_subprofiles
        .iter()
        .map(|channel| {
            channel_distance(
                &tracklet.channel_mean,
                tracklet.channel_valid,
                channel,
                true,
                tracklet.channel_dimensions.min(profile.channel_dimensions),
            )
        })
        .min_by(f32::total_cmp)
        .unwrap_or(0.0);
    voice_distance + channel_distance
}

fn tracklet_profile_pair_evidence(
    tracklet: &AcousticTracklet,
    profile: &AcousticSpeakerProfile,
    dimension_weights: &[f32; VOICE_VECTOR_DIMENSIONS],
) -> Option<AcousticSpeakerPairEvidence> {
    let tracklet_scale = tracklet.voice_variance.map(|variance| variance.max(0.025));
    let selected_channel = profile.channel_subprofiles.iter().min_by(|left, right| {
        raw_channel_distance(
            &tracklet.channel_mean,
            tracklet.channel_valid,
            left,
            true,
            tracklet.channel_dimensions.min(profile.channel_dimensions),
        )
        .total_cmp(&raw_channel_distance(
            &tracklet.channel_mean,
            tracklet.channel_valid,
            right,
            true,
            tracklet.channel_dimensions.min(profile.channel_dimensions),
        ))
    });
    let empty_channel = [0.0_f32; CHANNEL_VECTOR_DIMENSIONS];
    let profile_channel = selected_channel.copied().unwrap_or(empty_channel);
    let profile_channel_valid = selected_channel.is_some();
    profile
        .voice_subprofiles
        .iter()
        .filter_map(|mode| {
            let adapted_scale = std::array::from_fn(|dimension| {
                mode.scale[dimension] / dimension_weights[dimension].max(0.25)
            });
            speaker_pair_evidence_from_statistics(
                &tracklet.voice_mean,
                &tracklet.voice_valid,
                &tracklet_scale,
                tracklet.identity_frame_count as f32,
                &tracklet.channel_mean,
                tracklet.channel_valid,
                tracklet.channel_dimensions,
                &mode.center,
                &mode.valid,
                &adapted_scale,
                mode.weight,
                &profile_channel,
                profile_channel_valid,
                profile.channel_dimensions,
                SpeakerPairPerturbation::Full,
            )
        })
        .max_by(|left, right| {
            left.same_speaker_probability
                .total_cmp(&right.same_speaker_probability)
        })
}

fn unknown_assignment(tracklet: &AcousticTracklet) -> AcousticSpeakerAssignment {
    AcousticSpeakerAssignment {
        tracklet_index: tracklet.tracklet_index,
        start_ms: tracklet.start_ms,
        end_ms: tracklet.end_ms,
        speaker_ref: None,
        speaker_confidence: 0.0,
        secondary_speaker_ref: None,
        secondary_speaker_confidence: None,
        change_confidence: tracklet.change_confidence,
        overlap_suspected: tracklet.overlap_suspected,
        hard_attribution: false,
    }
}

fn evaluate_speaker_evidence(
    tracklets: &[AcousticTracklet],
    clusters: &[AcousticCluster],
    labels: &[String],
    enrollment: &SpeakerEnrollment,
    assignments: &[AcousticSpeakerAssignment],
) -> FwResult<Vec<AcousticSpeakerEvidenceSummary>> {
    #[derive(Debug, Clone, Copy, Default)]
    struct Accumulator {
        assigned_tracklet_count: usize,
        independent_tracklet_count: usize,
        recurrence_episode_count: usize,
        voiced_frame_count: usize,
        independent_voiced_frame_count: usize,
        voiced_duration_ms: u64,
        confidence_weighted_voiced_frames: f32,
        last_independent_position: Option<usize>,
        last_independent_end_ms: u64,
    }

    if tracklets.len() != assignments.len() {
        return Err(FwError::InvalidRequest(
            "speaker evidence requires exactly one assignment per acoustic tracklet".to_owned(),
        ));
    }
    if clusters.len() != labels.len() {
        return Err(FwError::InvalidRequest(
            "speaker evidence requires exactly one label per acoustic cluster".to_owned(),
        ));
    }
    let mut accumulated = labels
        .iter()
        .map(|label| (label.clone(), Accumulator::default()))
        .collect::<BTreeMap<_, _>>();
    for (position, (tracklet, assignment)) in tracklets.iter().zip(assignments).enumerate() {
        if assignment.tracklet_index != tracklet.tracklet_index {
            return Err(FwError::InvalidRequest(
                "speaker evidence assignments must preserve acoustic tracklet identity".to_owned(),
            ));
        }
        let Some(label) = assignment.speaker_ref.as_ref() else {
            continue;
        };
        let Some(evidence) = accumulated.get_mut(label) else {
            return Err(FwError::InvalidRequest(format!(
                "speaker assignment references unknown acoustic label {label:?}"
            )));
        };
        let voiced_duration_ms = report_count(
            tracklet.voiced_frame_count,
            "speaker-evidence voiced frame count",
        )?
        .checked_mul(10)
        .ok_or_else(|| {
            FwError::InvalidRequest(
                "speaker-evidence voiced duration exceeds the report schema".to_owned(),
            )
        })?;
        evidence.assigned_tracklet_count = evidence.assigned_tracklet_count.saturating_add(1);
        evidence.voiced_frame_count = evidence
            .voiced_frame_count
            .saturating_add(tracklet.voiced_frame_count);
        evidence.voiced_duration_ms = evidence
            .voiced_duration_ms
            .saturating_add(voiced_duration_ms);
        let is_own_soft_enrollment_observation = enrollment
            .soft_priors
            .contains_key(&(tracklet.tracklet_index, label.clone()));
        if !is_own_soft_enrollment_observation {
            let starts_new_episode = evidence.last_independent_position.is_none_or(|previous| {
                position != previous.saturating_add(1)
                    || tracklet.start_ms
                        > evidence
                            .last_independent_end_ms
                            .saturating_add(samples_to_ms(ACOUSTIC_HOP_SAMPLES))
            });
            evidence.independent_tracklet_count =
                evidence.independent_tracklet_count.saturating_add(1);
            evidence.independent_voiced_frame_count = evidence
                .independent_voiced_frame_count
                .saturating_add(tracklet.voiced_frame_count);
            evidence.confidence_weighted_voiced_frames +=
                assignment.speaker_confidence * tracklet.voiced_frame_count as f32;
            if starts_new_episode {
                evidence.recurrence_episode_count =
                    evidence.recurrence_episode_count.saturating_add(1);
            }
            evidence.last_independent_position = Some(position);
            evidence.last_independent_end_ms = tracklet.end_ms;
        }
    }

    let mut summaries = labels
        .iter()
        .zip(clusters)
        .map(|(label, cluster)| {
            let evidence = accumulated.get(label).copied().unwrap_or_default();
            let mean_assignment_confidence = if evidence.independent_voiced_frame_count == 0 {
                0.0
            } else {
                evidence.confidence_weighted_voiced_frames
                    / evidence.independent_voiced_frame_count as f32
            };
            let hard_anchored = cluster.hard_anchor.is_some();
            let has_independent_recurrence = evidence.recurrence_episode_count
                >= MIN_SPEAKER_EVIDENCE_RECURRENCE_EPISODES
                || (clusters.len() == 1 && evidence.independent_tracklet_count >= 2);
            let mut reasons = Vec::new();
            if evidence.assigned_tracklet_count == 0 {
                reasons.push(SpeakerEvidenceReason::NoAssignedSpeech);
            }
            if !hard_anchored && !has_independent_recurrence {
                reasons.push(SpeakerEvidenceReason::InsufficientIndependentRecurrence);
            }
            if !hard_anchored
                && evidence.independent_voiced_frame_count < MIN_SPEAKER_EVIDENCE_VOICED_FRAMES
            {
                reasons.push(SpeakerEvidenceReason::InsufficientVoicedFrames);
            }
            if !hard_anchored && mean_assignment_confidence < MIN_SPEAKER_EVIDENCE_CONFIDENCE {
                reasons.push(SpeakerEvidenceReason::InsufficientAssignmentConfidence);
            }
            if !hard_anchored && cluster.reliability < MIN_SPEAKER_EVIDENCE_RELIABILITY {
                reasons.push(SpeakerEvidenceReason::InsufficientProfileReliability);
            }
            let supported = hard_anchored || reasons.is_empty();
            AcousticSpeakerEvidenceSummary {
                speaker_ref: label.clone(),
                assigned_tracklet_count: evidence.assigned_tracklet_count,
                independent_tracklet_count: evidence.independent_tracklet_count,
                recurrence_episode_count: evidence.recurrence_episode_count,
                voiced_frame_count: evidence.voiced_frame_count,
                independent_voiced_frame_count: evidence.independent_voiced_frame_count,
                voiced_duration_ms: evidence.voiced_duration_ms,
                mean_assignment_confidence,
                cluster_reliability: cluster.reliability,
                hard_anchored,
                separated_from_supported_speakers: true,
                reasons,
                supported,
            }
        })
        .collect::<Vec<_>>();

    let mut candidates = summaries
        .iter()
        .enumerate()
        .filter_map(|(index, evidence)| evidence.supported.then_some(index))
        .collect::<Vec<_>>();
    candidates.sort_by(|&left, &right| {
        summaries[right]
            .hard_anchored
            .cmp(&summaries[left].hard_anchored)
            .then(
                summaries[right]
                    .independent_voiced_frame_count
                    .cmp(&summaries[left].independent_voiced_frame_count),
            )
            .then_with(|| {
                summaries[right]
                    .mean_assignment_confidence
                    .total_cmp(&summaries[left].mean_assignment_confidence)
            })
            .then_with(|| {
                summaries[left]
                    .speaker_ref
                    .cmp(&summaries[right].speaker_ref)
            })
    });
    let mut accepted = Vec::<usize>::new();
    for candidate in candidates {
        let merge_compatible = accepted.iter().copied().any(|existing| {
            if summaries[candidate].hard_anchored && summaries[existing].hard_anchored {
                return false;
            }
            !clusters_have_robust_different_speaker_evidence(
                &clusters[candidate],
                &clusters[existing],
            )
        });
        if merge_compatible {
            summaries[candidate].supported = false;
            summaries[candidate].separated_from_supported_speakers = false;
            summaries[candidate]
                .reasons
                .push(SpeakerEvidenceReason::MergeCompatibleWithSupportedSpeaker);
        } else {
            let support_reason = if summaries[candidate].hard_anchored {
                SpeakerEvidenceReason::SupportedByHardHint
            } else if summaries[candidate].recurrence_episode_count
                < MIN_SPEAKER_EVIDENCE_RECURRENCE_EPISODES
            {
                SpeakerEvidenceReason::SupportedByRepeatedTracklets
            } else {
                SpeakerEvidenceReason::SupportedByIndependentRecurrence
            };
            summaries[candidate].reasons.push(support_reason);
            accepted.push(candidate);
        }
    }
    Ok(summaries)
}

fn clusters_have_robust_different_speaker_evidence(
    left: &AcousticCluster,
    right: &AcousticCluster,
) -> bool {
    SpeakerPairPerturbation::ALL
        .into_iter()
        .filter_map(|perturbation| cluster_pair_evidence(left, right, perturbation))
        .filter(|evidence| evidence.support >= MIN_SPEAKER_SEPARATION_SUPPORT)
        .filter(|evidence| {
            evidence.same_speaker_probability <= MAX_SAME_SPEAKER_PROBABILITY_FOR_SEPARATION
        })
        .count()
        >= MIN_SPEAKER_SEPARATION_LANES
}

fn retain_supported_assignments(
    evidence: &[AcousticSpeakerEvidenceSummary],
    assignments: &mut [AcousticSpeakerAssignment],
) {
    let supported = evidence
        .iter()
        .filter(|evidence| evidence.supported)
        .map(|evidence| evidence.speaker_ref.as_str())
        .collect::<BTreeSet<_>>();
    for assignment in assignments {
        if assignment
            .speaker_ref
            .as_deref()
            .is_some_and(|speaker| !supported.contains(speaker))
            && !assignment.hard_attribution
        {
            assignment.speaker_ref = None;
            assignment.speaker_confidence = 0.0;
            assignment.secondary_speaker_ref = None;
            assignment.secondary_speaker_confidence = None;
        } else if assignment
            .secondary_speaker_ref
            .as_deref()
            .is_some_and(|speaker| !supported.contains(speaker))
        {
            assignment.secondary_speaker_ref = None;
            assignment.secondary_speaker_confidence = None;
        }
    }
}

fn hard_assignments_satisfied(
    tracklets: &[AcousticTracklet],
    enrollment: &SpeakerEnrollment,
    assignments: &[AcousticSpeakerAssignment],
) -> bool {
    if tracklets.len() != assignments.len() {
        return false;
    }
    tracklets
        .iter()
        .zip(assignments)
        .all(|(tracklet, assignment)| {
            assignment.tracklet_index == tracklet.tracklet_index
                && enrollment
                    .hard_assignments
                    .get(&tracklet.tracklet_index)
                    .is_none_or(|speaker| {
                        assignment.hard_attribution
                            && assignment.speaker_ref.as_ref() == Some(speaker)
                    })
        })
}

fn assignment_voiced_shares(
    tracklets: &[AcousticTracklet],
    assignments: &[AcousticSpeakerAssignment],
) -> FwResult<(f32, f32)> {
    if tracklets.len() != assignments.len() {
        return Err(FwError::InvalidRequest(
            "speaker occupancy requires exactly one assignment per acoustic tracklet".to_owned(),
        ));
    }
    let mut total_voiced_frames = 0usize;
    let mut unknown_voiced_frames = 0usize;
    let mut per_speaker = BTreeMap::<&str, usize>::new();
    for (tracklet, assignment) in tracklets.iter().zip(assignments) {
        if assignment.tracklet_index != tracklet.tracklet_index {
            return Err(FwError::InvalidRequest(
                "speaker occupancy assignments must preserve acoustic tracklet identity".to_owned(),
            ));
        }
        total_voiced_frames = total_voiced_frames.saturating_add(tracklet.voiced_frame_count);
        if let Some(speaker) = assignment.speaker_ref.as_deref() {
            let entry = per_speaker.entry(speaker).or_default();
            *entry = entry.saturating_add(tracklet.voiced_frame_count);
        } else {
            unknown_voiced_frames =
                unknown_voiced_frames.saturating_add(tracklet.voiced_frame_count);
        }
    }
    if total_voiced_frames == 0 {
        return Ok((0.0, 0.0));
    }
    let denominator = total_voiced_frames as f32;
    let dominant_share = per_speaker
        .values()
        .copied()
        .max()
        .map_or(0.0, |frames| frames as f32 / denominator);
    Ok((dominant_share, unknown_voiced_frames as f32 / denominator))
}

fn count_request_allows_zero(request: &SpeakerCountRequest) -> bool {
    matches!(
        request,
        SpeakerCountRequest::Infer
            | SpeakerCountRequest::Prior { .. }
            | SpeakerCountRequest::Range { .. }
    )
}

fn speaker_count_satisfies(count: usize, request: &SpeakerCountRequest) -> bool {
    match request {
        SpeakerCountRequest::Infer => true,
        SpeakerCountRequest::Range { .. } => true,
        SpeakerCountRequest::HardConstraint { count: exact } => count == *exact as usize,
        SpeakerCountRequest::Prior { .. } => true,
    }
}

fn clustering_profile_summaries(
    clusters: &[AcousticCluster],
    labels: &[String],
    enrollment: &SpeakerEnrollment,
    supported_labels: &BTreeSet<&str>,
) -> Vec<SpeakerProfileSummary> {
    clusters
        .iter()
        .enumerate()
        .filter(|(index, _)| supported_labels.contains(labels[*index].as_str()))
        .map(|(index, cluster)| {
            enrollment
                .summaries
                .iter()
                .find(|summary| summary.speaker_ref == labels[index])
                .cloned()
                .unwrap_or_else(|| SpeakerProfileSummary {
                    speaker_ref: labels[index].clone(),
                    frame_count: cluster.weight.max(0.0) as u64,
                    voiced_duration_ms: (cluster.weight.max(0.0) as u64) * 10,
                    reliability: f64::from(cluster.reliability.clamp(0.0, 1.0)),
                    voice_profile_count: 1,
                    channel_profile_count: u32::from(cluster.channel_valid),
                    training_accepted_count: 0,
                    training_downweighted_count: 0,
                    training_quarantined_count: 0,
                    anchored: cluster.hard_anchor.is_some(),
                    soft_hint_contradiction: None,
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, VecDeque};

    use super::{
        ACOUSTIC_CANCELLATION_INTERVAL_FRAMES, ACOUSTIC_FRAME_SAMPLES, ACOUSTIC_HOP_SAMPLES,
        ACOUSTIC_MODULATION_FREQUENCY_HZ, ACOUSTIC_MODULATION_HISTORY_FRAMES,
        ACOUSTIC_SCATTERING_SCALE_PAIRS, ACOUSTIC_SCATTERING_SCALE_SUPPORTS,
        ACOUSTIC_TRAJECTORY_HISTORY_FRAMES, AcousticBoundaryHints, AcousticDiarizationInput,
        AcousticFeatureStream, AcousticFrameFeatures, AcousticModulationSidecar,
        AcousticQualityMask, AcousticScatteringMode, AcousticSegmentationStream, AcousticSegmenter,
        AcousticSidecarFeatureOwner, AcousticSidecarStudy, AcousticSidecarStudyConfig,
        AcousticSidecarStudyMode, AcousticSpeakerAssignment, AcousticTracklet,
        AcousticTrajectoryFamily, AcousticTrajectoryWaveletMode, AcousticWaveletBasis,
        AcousticWaveletConfig, CEPSTRAL_COEFFICIENTS, CHANNEL_VECTOR_DIMENSIONS,
        CalibrationObservation, ChannelFeatureView, CorpusRecordingManifest,
        DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION, DIARIZATION_HYPOTHESIS_SCHEMA_VERSION,
        DIARIZATION_REFERENCE_SCHEMA_VERSION, DiarizationCorpusManifest,
        DiarizationHypothesisDocument, DiarizationReferenceDocument, DiarizationScorerConfig,
        EvaluationHintPolicy, EvaluationOverlapPolicy, EvaluationPerformanceObservation,
        EvaluationRegion, EvaluationSpeakerHint, EvaluationSplit, EvaluationTurn, EvaluationWord,
        LeakageKind, ProfileEnrollmentCode, ProfileMetricAdaptationFallback,
        ProfileTrainingDisposition, ProfileTrainingReason, ScoringTurn, VOICE_VECTOR_DIMENSIONS,
        VoiceFeatureView, acoustic_sidecar_study_config_sha256, analyze_acoustic_wavelet,
        audit_diarization_manifest, cluster_acoustic_tracklets, diarization_turns_from_assignments,
        diarize_acoustic_pcm, enroll_known_speaker_profiles, extract_acoustic_features,
        merge_tracklet_statistics, parse_diarization_corpus_manifest, parse_diarization_reference,
        project_diarization_onto_segments, score_calibration, score_change_points,
        score_diarization, score_diarization_documents, segment_acoustic_frames,
        verify_authoritative_score_hash, verify_leakage_audit_hash,
    };
    use crate::FwError;
    use crate::model::{
        DiarizationEngine, DiarizationFallbackStatus, DiarizationRequest, DiarizationTurn,
        KnownSpeakerInterval, KnownSpeakerPolicy, SpeakerAttributionQueryReason,
        SpeakerCountCalibrationStatus, SpeakerCountEstimate, SpeakerCountEvidenceLane,
        SpeakerCountLaneEvidence, SpeakerCountLaneUnavailableReason, SpeakerCountOutcomeReason,
        SpeakerCountOutcomeStatus, SpeakerCountPosteriorBin, SpeakerCountRange,
        SpeakerCountRequest, SpeakerCountResourceSummary, SpeakerEvidenceReason,
        SpeakerHintDisposition, TranscriptionSegment,
    };
    use sha2::{Digest, Sha256};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    fn count_estimate(
        selected_count: Option<u32>,
        posterior: &[(u32, f64)],
        unresolved_probability: f64,
    ) -> SpeakerCountEstimate {
        let posterior = posterior
            .iter()
            .map(|&(count, probability)| SpeakerCountPosteriorBin { count, probability })
            .collect::<Vec<_>>();
        let entropy_term = |probability: f64| {
            (probability > 0.0)
                .then(|| -probability * probability.log2())
                .unwrap_or(0.0)
        };
        let entropy_bits = posterior
            .iter()
            .map(|bin| entropy_term(bin.probability))
            .sum::<f64>()
            + entropy_term(unresolved_probability);
        let proposed_count = selected_count.or_else(|| posterior.first().map(|bin| bin.count));
        let available_lane = |lane| SpeakerCountLaneEvidence {
            lane,
            available: true,
            proposed_count,
            confidence: 0.75,
            unavailable_reason: None,
        };
        SpeakerCountEstimate {
            schema_version: "speaker-count-estimate-v2".to_owned(),
            selected_count,
            supported_range: selected_count.map(|count| SpeakerCountRange {
                minimum: count,
                maximum: count,
            }),
            posterior,
            unresolved_probability,
            entropy_bits,
            stability: 0.75,
            constraint_lower_bound: 1,
            candidate_upper_bound: 3,
            calibration_status: SpeakerCountCalibrationStatus::DevelopmentUncertified,
            calibration_sha256: "a".repeat(64),
            evidence_sha256: "b".repeat(64),
            lanes: vec![
                available_lane(SpeakerCountEvidenceLane::MergeRisk),
                available_lane(SpeakerCountEvidenceLane::SparseNormalizedEigengap),
                available_lane(SpeakerCountEvidenceLane::FeatureJackknife),
                available_lane(SpeakerCountEvidenceLane::EffectiveOccupancy),
                available_lane(SpeakerCountEvidenceLane::ConstraintGraph),
                SpeakerCountLaneEvidence {
                    lane: SpeakerCountEvidenceLane::CallerPrior,
                    available: false,
                    proposed_count: None,
                    confidence: 0.0,
                    unavailable_reason: Some(SpeakerCountLaneUnavailableReason::NotRequested),
                },
            ],
            resources: SpeakerCountResourceSummary {
                prototype_count: 4,
                affinity_pair_evaluations: 12,
                retained_sparse_edges: 6,
                estimated_peak_buffer_bytes: 1_024,
                stability_replicates: 3,
                solver_iterations: 4,
                solver_sparse_matvec_terms: 24,
                solver_residual: Some(0.01),
            },
        }
    }

    fn evaluation_reference() -> DiarizationReferenceDocument {
        DiarizationReferenceDocument {
            schema_version: DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
            recording_id: "synthetic-call-001".to_owned(),
            duration_ms: 6_000,
            turns: vec![
                EvaluationTurn::labeled(0, 2_000, "speaker-a"),
                EvaluationTurn::labeled(2_000, 4_000, "speaker-b"),
                EvaluationTurn::labeled(4_000, 5_000, "speaker-a"),
                EvaluationTurn::labeled(4_000, 5_000, "speaker-b"),
            ],
            ignored_regions: Vec::new(),
            speaker_hints: vec![
                EvaluationSpeakerHint {
                    start_ms: 0,
                    end_ms: 1_000,
                    speaker_ref: "speaker-a".to_owned(),
                    policy: EvaluationHintPolicy::Hard,
                },
                EvaluationSpeakerHint {
                    start_ms: 3_000,
                    end_ms: 4_000,
                    speaker_ref: "speaker-b".to_owned(),
                    policy: EvaluationHintPolicy::Soft,
                },
            ],
            words: Vec::new(),
        }
    }

    fn confident_turn(
        start_ms: u64,
        end_ms: u64,
        speaker: &str,
        confidence: f64,
        overlap_suspected: bool,
    ) -> EvaluationTurn {
        EvaluationTurn {
            start_ms,
            end_ms,
            speaker: Some(speaker.to_owned()),
            speaker_confidence: Some(confidence),
            overlap_suspected,
        }
    }

    fn evaluation_hypothesis() -> DiarizationHypothesisDocument {
        DiarizationHypothesisDocument {
            schema_version: DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: "synthetic-call-001".to_owned(),
            duration_ms: 6_000,
            turns: vec![
                confident_turn(0, 2_000, "cluster-x", 0.9, false),
                confident_turn(2_000, 3_000, "cluster-y", 0.8, false),
                EvaluationTurn::unknown(3_000, 4_000),
                confident_turn(4_000, 5_000, "cluster-x", 0.7, true),
                confident_turn(4_000, 5_000, "cluster-y", 0.6, true),
                confident_turn(5_000, 5_500, "cluster-x", 0.4, false),
            ],
            speaker_count_estimate: None,
            performance: Some(EvaluationPerformanceObservation {
                audio_duration_ms: 6_000,
                wall_time_ms: 120,
                peak_rss_bytes: 8_388_608,
            }),
        }
    }

    fn manifest_recording(
        recording_id: &str,
        split: EvaluationSplit,
        origin_recording_id: &str,
        speaker_refs: &[&str],
    ) -> CorpusRecordingManifest {
        CorpusRecordingManifest {
            recording_id: recording_id.to_owned(),
            split,
            origin_recording_id: origin_recording_id.to_owned(),
            speaker_refs: speaker_refs
                .iter()
                .map(|speaker| (*speaker).to_owned())
                .collect(),
            derived_from_recording_ids: Vec::new(),
            augmentation_group_id: None,
            enrollment_recording_ids: Vec::new(),
        }
    }

    #[test]
    fn authoritative_scorer_matches_hand_computed_multi_metric_example() {
        let result = score_diarization_documents(
            &evaluation_reference(),
            &evaluation_hypothesis(),
            &DiarizationScorerConfig::default(),
        )
        .expect("authoritative score");

        assert_close(result.diarization.reference_speaker_time_sec, 6.0);
        assert_close(result.diarization.missed_speech_sec, 0.0);
        assert_close(result.diarization.false_alarm_sec, 0.5);
        assert_close(result.diarization.speaker_confusion_sec, 1.0);
        assert_close(result.diarization.der.expect("DER"), 0.25);
        assert_close(result.diarization.jer.expect("JER"), 5.0 / 21.0);
        assert_eq!(result.diarization.speaker_mapping["cluster-x"], "speaker-a");
        assert_eq!(result.diarization.speaker_mapping["cluster-y"], "speaker-b");

        assert_close(result.speech_activity.reference_speech_sec, 5.0);
        assert_close(result.speech_activity.false_alarm_sec, 0.5);
        assert_close(
            result.speech_activity.error_rate.expect("speech error"),
            0.1,
        );
        assert_eq!(result.change_points.reference_count, 2);
        assert_eq!(result.change_points.hypothesis_count, 4);
        assert_eq!(result.change_points.matched_count, 2);
        assert_eq!(result.change_points.precision, Some(0.5));
        assert_eq!(result.change_points.recall, Some(1.0));
        assert_eq!(result.speaker_count.reference_speakers, 2);
        assert_eq!(result.speaker_count.hypothesis_speakers, 2);

        assert_close(result.overlap.reference_overlap_sec, 1.0);
        assert_eq!(result.overlap.f1, Some(1.0));
        assert_close(result.hints.hinted_sec, 2.0);
        assert_close(result.hints.adherent_sec, 1.0);
        assert_close(result.hints.unknown_sec, 1.0);
        assert_close(result.hints.hard_violation_sec, 0.0);
        assert_close(
            result.selective_attribution.coverage.expect("coverage"),
            5.0 / 6.0,
        );
        assert_eq!(result.selective_attribution.selective_risk, Some(0.0));
        assert_close(
            result.calibration.brier_score.expect("weighted Brier"),
            0.39 / 5.5,
        );
        assert_eq!(result.calibration.coverage, Some(1.0));
        assert_close(
            result
                .performance
                .as_ref()
                .and_then(|score| score.real_time_factor)
                .expect("RTF"),
            0.02,
        );
        verify_authoritative_score_hash(&result).expect("valid result hash");
    }

    #[test]
    fn speaker_count_posterior_uses_proper_scores_and_explicit_unresolved_mass() {
        let mut hypothesis = evaluation_hypothesis();
        hypothesis.speaker_count_estimate = Some(count_estimate(
            Some(2),
            &[(1, 0.1), (2, 0.7), (3, 0.1)],
            0.1,
        ));
        let result = score_diarization_documents(
            &evaluation_reference(),
            &hypothesis,
            &DiarizationScorerConfig::default(),
        )
        .expect("posterior score");
        let score = result.speaker_count_posterior;
        assert!(score.posterior_available);
        assert_eq!(score.selected_count, Some(2));
        assert!(!score.unresolved);
        assert_close(score.reference_probability.expect("reference mass"), 0.7);
        assert_close(
            score
                .negative_log_likelihood
                .expect("finite negative log likelihood"),
            -0.7_f64.ln(),
        );
        assert_close(score.brier_score.expect("multiclass Brier"), 0.12);
        assert_eq!(score.top_k_hit, Some(true));
        assert_eq!(score.credible_set_hit, Some(true));
        assert!(score.credible_set.contains(&2));
        assert_eq!(
            score.calibration_status,
            Some(SpeakerCountCalibrationStatus::DevelopmentUncertified)
        );

        let mut unresolved_hypothesis = evaluation_hypothesis();
        unresolved_hypothesis.speaker_count_estimate =
            Some(count_estimate(None, &[(1, 0.3), (2, 0.2)], 0.5));
        let unresolved = score_diarization_documents(
            &evaluation_reference(),
            &unresolved_hypothesis,
            &DiarizationScorerConfig::default(),
        )
        .expect("unresolved posterior")
        .speaker_count_posterior;
        assert!(unresolved.posterior_available);
        assert!(unresolved.unresolved);
        assert_eq!(unresolved.selected_count, None);
        assert_eq!(unresolved.unresolved_probability, Some(0.5));
        assert!(unresolved.credible_set_includes_unresolved);

        let unsupported_reference = super::score_speaker_count_posterior(
            crate::model::MAX_SPEAKER_COUNT as usize + 1,
            unresolved_hypothesis.speaker_count_estimate.as_ref(),
            &DiarizationScorerConfig::default(),
        );
        assert!(unsupported_reference.infinite_negative_log_likelihood);
        assert_close(
            unsupported_reference
                .brier_score
                .expect("out-of-support Brier"),
            1.38,
        );
    }

    #[test]
    fn occupancy_score_catches_exact_cardinality_with_dominant_speaker_collapse() {
        let reference = DiarizationReferenceDocument {
            schema_version: DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
            recording_id: "collapse-fixture".to_owned(),
            duration_ms: 10_000,
            turns: vec![
                EvaluationTurn::labeled(0, 5_000, "reference-a"),
                EvaluationTurn::labeled(5_000, 10_000, "reference-b"),
            ],
            ignored_regions: Vec::new(),
            speaker_hints: Vec::new(),
            words: Vec::new(),
        };
        let hypothesis = DiarizationHypothesisDocument {
            schema_version: DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: reference.recording_id.clone(),
            duration_ms: reference.duration_ms,
            turns: vec![
                EvaluationTurn::labeled(0, 9_970, "cluster-dominant"),
                EvaluationTurn::labeled(9_970, 10_000, "cluster-token"),
            ],
            speaker_count_estimate: None,
            performance: None,
        };
        let score = score_diarization_documents(
            &reference,
            &hypothesis,
            &DiarizationScorerConfig::default(),
        )
        .expect("collapse score");
        assert_eq!(score.speaker_count.absolute_error, 0);
        assert_close(
            score
                .speaker_occupancy
                .dominant_speaker_share
                .expect("dominant share"),
            0.997,
        );
        assert!(score.speaker_occupancy.dominant_collapse_detected);
        assert!(score.speaker_occupancy.any_reference_collapse_detected);
        assert_eq!(score.speaker_occupancy.effective_speaker_count, 1);
        assert_eq!(score.speaker_occupancy.collapsed_reference_speaker_count, 1);
        assert!(
            score
                .speaker_occupancy
                .minority_reference_recall
                .is_some_and(|recall| recall < 0.01)
        );
    }

    #[test]
    fn occupancy_ignores_labels_that_exist_only_in_excluded_regions() {
        let reference = DiarizationReferenceDocument {
            schema_version: DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
            recording_id: "ignored-occupancy-fixture".to_owned(),
            duration_ms: 2_000,
            turns: vec![
                EvaluationTurn::labeled(0, 1_000, "reference-a"),
                EvaluationTurn::labeled(1_000, 2_000, "excluded-reference"),
            ],
            ignored_regions: vec![EvaluationRegion {
                start_ms: 1_000,
                end_ms: 2_000,
                reason_code: "annotation_uncertain".to_owned(),
            }],
            speaker_hints: Vec::new(),
            words: Vec::new(),
        };
        let hypothesis = DiarizationHypothesisDocument {
            schema_version: DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: reference.recording_id.clone(),
            duration_ms: reference.duration_ms,
            turns: vec![
                EvaluationTurn::labeled(0, 1_000, "cluster-a"),
                EvaluationTurn::labeled(1_000, 2_000, "excluded-cluster"),
            ],
            speaker_count_estimate: None,
            performance: None,
        };
        let score = score_diarization_documents(
            &reference,
            &hypothesis,
            &DiarizationScorerConfig::default(),
        )
        .expect("ignored occupancy");
        let excluded = score
            .speaker_occupancy
            .speakers
            .iter()
            .find(|speaker| speaker.hypothesis_speaker == "excluded-cluster")
            .expect("excluded label evidence");
        assert_eq!(excluded.voiced_duration_sec, 0.0);
        assert_eq!(excluded.recurrence_episode_count, 0);
        assert!(!excluded.effective);
        assert_eq!(score.speaker_occupancy.phantom_speaker_count, 0);
        assert_eq!(score.speaker_occupancy.collapsed_reference_speaker_count, 0);
        assert_eq!(score.speaker_occupancy.minority_reference_recall, Some(1.0));
    }

    #[test]
    fn aligned_word_attribution_is_transcript_free_and_permutation_invariant() {
        let reference = DiarizationReferenceDocument {
            schema_version: DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
            recording_id: "word-fixture".to_owned(),
            duration_ms: 2_000,
            turns: vec![
                EvaluationTurn::labeled(0, 1_000, "reference-a"),
                EvaluationTurn::labeled(1_000, 2_000, "reference-b"),
            ],
            ignored_regions: Vec::new(),
            speaker_hints: Vec::new(),
            words: vec![
                EvaluationWord {
                    word_id: "word-001".to_owned(),
                    start_ms: 100,
                    end_ms: 200,
                    speaker_ref: "reference-a".to_owned(),
                },
                EvaluationWord {
                    word_id: "word-002".to_owned(),
                    start_ms: 1_100,
                    end_ms: 1_200,
                    speaker_ref: "reference-b".to_owned(),
                },
                EvaluationWord {
                    word_id: "word-003".to_owned(),
                    start_ms: 1_600,
                    end_ms: 1_700,
                    speaker_ref: "reference-b".to_owned(),
                },
            ],
        };
        let hypothesis = DiarizationHypothesisDocument {
            schema_version: DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: reference.recording_id.clone(),
            duration_ms: reference.duration_ms,
            turns: vec![
                EvaluationTurn::labeled(0, 1_000, "cluster-z"),
                EvaluationTurn::labeled(1_000, 1_500, "cluster-a"),
                EvaluationTurn::unknown(1_500, 2_000),
            ],
            speaker_count_estimate: None,
            performance: None,
        };
        let score = score_diarization_documents(
            &reference,
            &hypothesis,
            &DiarizationScorerConfig::default(),
        )
        .expect("word attribution")
        .word_attribution;
        assert_eq!(score.reference_word_count, 3);
        assert_eq!(score.scored_word_count, 3);
        assert_eq!(score.correct_word_count, 2);
        assert_eq!(score.unknown_word_count, 1);
        assert_eq!(score.incorrect_word_count, 0);
        assert_close(
            score.word_diarization_error_rate.expect("word error"),
            1.0 / 3.0,
        );
    }

    #[test]
    fn aligned_word_ids_reject_lexical_content() {
        let mut reference = evaluation_reference();
        reference.words = vec![EvaluationWord {
            word_id: "confidential-spoken-token".to_owned(),
            start_ms: 100,
            end_ms: 200,
            speaker_ref: "speaker-a".to_owned(),
        }];
        let error = parse_diarization_reference(
            &serde_json::to_vec(&reference).expect("reference serialization"),
        )
        .expect_err("lexical word identity must fail closed");
        assert!(error.to_string().contains("word_id_shape"));
    }

    #[test]
    fn invalid_hypothesis_count_posterior_fails_before_scoring() {
        let mut hypothesis = evaluation_hypothesis();
        let mut estimate = count_estimate(Some(2), &[(1, 0.1), (2, 0.7), (3, 0.1)], 0.1);
        estimate.posterior[1].probability = 0.8;
        hypothesis.speaker_count_estimate = Some(estimate);
        let error = score_diarization_documents(
            &evaluation_reference(),
            &hypothesis,
            &DiarizationScorerConfig::default(),
        )
        .expect_err("invalid posterior must fail");
        assert!(error.to_string().contains("speaker_count_estimate"));
    }

    #[test]
    fn authoritative_scorer_remains_permutation_invariant() {
        let reference = DiarizationReferenceDocument {
            speaker_hints: Vec::new(),
            turns: vec![
                EvaluationTurn::labeled(0, 1_000, "ref-a"),
                EvaluationTurn::labeled(1_000, 2_000, "ref-b"),
            ],
            duration_ms: 2_000,
            ..evaluation_reference()
        };
        let hypothesis = DiarizationHypothesisDocument {
            turns: vec![
                confident_turn(0, 1_000, "z-cluster", 1.0, false),
                confident_turn(1_000, 2_000, "a-cluster", 1.0, false),
            ],
            duration_ms: 2_000,
            performance: None,
            ..evaluation_hypothesis()
        };
        let result = score_diarization_documents(
            &reference,
            &hypothesis,
            &DiarizationScorerConfig::default(),
        )
        .expect("score");
        assert_eq!(result.diarization.der, Some(0.0));
        assert_eq!(result.diarization.jer, Some(0.0));
        assert_eq!(result.diarization.speaker_mapping["z-cluster"], "ref-a");
        assert_eq!(result.diarization.speaker_mapping["a-cluster"], "ref-b");
    }

    #[test]
    fn speaker_boundary_collar_changes_der_but_not_change_point_authority() {
        let reference = DiarizationReferenceDocument {
            speaker_hints: Vec::new(),
            turns: vec![
                EvaluationTurn::labeled(0, 2_000, "ref-a"),
                EvaluationTurn::labeled(2_000, 4_000, "ref-b"),
            ],
            duration_ms: 4_000,
            ..evaluation_reference()
        };
        let hypothesis = DiarizationHypothesisDocument {
            turns: vec![
                confident_turn(0, 2_200, "cluster-x", 0.9, false),
                confident_turn(2_200, 4_000, "cluster-y", 0.9, false),
            ],
            duration_ms: 4_000,
            performance: None,
            ..evaluation_hypothesis()
        };
        let strict = score_diarization_documents(
            &reference,
            &hypothesis,
            &DiarizationScorerConfig::default(),
        )
        .expect("strict score");
        assert_close(strict.diarization.der.expect("strict DER"), 0.05);

        let config = DiarizationScorerConfig {
            speaker_boundary_collar_ms: 250,
            ..DiarizationScorerConfig::default()
        };
        let collared =
            score_diarization_documents(&reference, &hypothesis, &config).expect("collared score");
        assert_eq!(collared.diarization.der, Some(0.0));
        assert_eq!(collared.change_points.reference_count, 1);
        assert_eq!(collared.change_points.matched_count, 1);
        assert_close(collared.ignored_duration_sec, 0.5);
    }

    #[test]
    fn overlap_exclusion_policy_is_explicit_and_reproducible() {
        let reference = DiarizationReferenceDocument {
            speaker_hints: Vec::new(),
            turns: vec![
                EvaluationTurn::labeled(0, 2_000, "ref-a"),
                EvaluationTurn::labeled(1_000, 2_000, "ref-b"),
            ],
            duration_ms: 2_000,
            ..evaluation_reference()
        };
        let hypothesis = DiarizationHypothesisDocument {
            turns: vec![confident_turn(0, 2_000, "cluster-x", 0.9, false)],
            duration_ms: 2_000,
            performance: None,
            ..evaluation_hypothesis()
        };
        let included = score_diarization_documents(
            &reference,
            &hypothesis,
            &DiarizationScorerConfig::default(),
        )
        .expect("included overlap");
        assert_close(included.diarization.der.expect("DER"), 1.0 / 3.0);

        let config = DiarizationScorerConfig {
            overlap_policy: EvaluationOverlapPolicy::Exclude,
            ..DiarizationScorerConfig::default()
        };
        let excluded = score_diarization_documents(&reference, &hypothesis, &config)
            .expect("excluded overlap");
        assert_eq!(excluded.diarization.der, Some(0.0));
        assert_close(excluded.ignored_duration_sec, 1.0);
        assert_close(excluded.overlap.reference_overlap_sec, 1.0);
        assert_eq!(excluded.overlap.recall, Some(0.0));
        assert_eq!(excluded.overlap.f1, Some(0.0));
    }

    #[test]
    fn ignored_regions_remove_accuracy_hint_and_calibration_opportunities() {
        let mut reference = evaluation_reference();
        reference.ignored_regions = vec![EvaluationRegion {
            start_ms: 2_000,
            end_ms: 4_000,
            reason_code: "annotation_uncertain".to_owned(),
        }];
        let result = score_diarization_documents(
            &reference,
            &evaluation_hypothesis(),
            &DiarizationScorerConfig::default(),
        )
        .expect("score");
        assert_close(result.ignored_duration_sec, 2.0);
        assert_close(result.diarization.reference_speaker_time_sec, 4.0);
        assert_close(result.diarization.speaker_confusion_sec, 0.0);
        assert_close(result.hints.hinted_sec, 1.0);
        assert_close(result.calibration.opportunity_duration_sec, 4.5);
    }

    #[test]
    fn empty_speech_and_missing_confidence_have_explicit_undefined_metrics() {
        let reference = DiarizationReferenceDocument {
            speaker_hints: Vec::new(),
            turns: Vec::new(),
            duration_ms: 1_000,
            ..evaluation_reference()
        };
        let empty_hypothesis = DiarizationHypothesisDocument {
            turns: Vec::new(),
            duration_ms: 1_000,
            performance: None,
            ..evaluation_hypothesis()
        };
        let empty = score_diarization_documents(
            &reference,
            &empty_hypothesis,
            &DiarizationScorerConfig::default(),
        )
        .expect("empty score");
        assert_eq!(empty.diarization.der, None);
        assert_eq!(empty.diarization.jer, None);
        assert_eq!(empty.speech_activity.error_rate, None);
        assert_eq!(empty.selective_attribution.coverage, None);
        assert_eq!(empty.calibration.coverage, None);
        assert_eq!(empty.calibration.brier_score, None);

        let labeled_reference = DiarizationReferenceDocument {
            turns: vec![EvaluationTurn::labeled(0, 1_000, "speaker-a")],
            ..reference
        };
        let no_confidence = DiarizationHypothesisDocument {
            turns: vec![EvaluationTurn::labeled(0, 1_000, "cluster-x")],
            ..empty_hypothesis
        };
        let score = score_diarization_documents(
            &labeled_reference,
            &no_confidence,
            &DiarizationScorerConfig::default(),
        )
        .expect("no-confidence score");
        assert_eq!(score.calibration.coverage, Some(0.0));
        assert_eq!(score.calibration.brier_score, None);
    }

    #[test]
    fn speaker_hint_must_name_a_reference_speaker() {
        let mut reference = evaluation_reference();
        reference.speaker_hints[0].speaker_ref = "speaker-not-present".to_owned();
        let error = score_diarization_documents(
            &reference,
            &evaluation_hypothesis(),
            &DiarizationScorerConfig::default(),
        )
        .expect_err("unbound hint");
        assert!(error.to_string().contains("hint_speaker_not_in_reference"));
    }

    #[test]
    fn scorer_serialization_and_hashes_are_deterministic() {
        let first = score_diarization_documents(
            &evaluation_reference(),
            &evaluation_hypothesis(),
            &DiarizationScorerConfig::default(),
        )
        .expect("first");
        let second = score_diarization_documents(
            &evaluation_reference(),
            &evaluation_hypothesis(),
            &DiarizationScorerConfig::default(),
        )
        .expect("second");
        assert_eq!(first, second);
        assert_eq!(
            serde_json::to_vec(&first).expect("serialize first"),
            serde_json::to_vec(&second).expect("serialize second")
        );

        let mut tampered = first;
        tampered.diarization.false_alarm_sec += 0.001;
        let error =
            verify_authoritative_score_hash(&tampered).expect_err("tampering must be detected");
        assert!(error.to_string().contains("result_hash_mismatch"));
    }

    #[test]
    fn parser_rejects_unknown_fields_wrong_versions_and_sensitive_markers() {
        let mut value = serde_json::to_value(evaluation_reference()).expect("reference JSON");
        value["unexpected"] = serde_json::json!(true);
        assert!(
            parse_diarization_reference(
                &serde_json::to_vec(&value).expect("serialize malformed reference")
            )
            .is_err()
        );

        let mut wrong_version = evaluation_reference();
        wrong_version.schema_version = "diarization-reference-v999".to_owned();
        let error = parse_diarization_reference(
            &serde_json::to_vec(&wrong_version).expect("serialize wrong version"),
        )
        .expect_err("wrong version");
        assert!(error.to_string().contains("reference_schema_version"));

        let mut sensitive = evaluation_reference();
        sensitive.recording_id = "Downloads/private-call.m4a".to_owned();
        let error = score_diarization_documents(
            &sensitive,
            &evaluation_hypothesis(),
            &DiarizationScorerConfig::default(),
        )
        .expect_err("sensitive path");
        assert!(
            error.to_string().contains("opaque_id_path")
                || error.to_string().contains("opaque_id_sensitive_marker")
        );
    }

    #[test]
    fn malformed_intervals_confidences_and_order_fail_with_stable_codes() {
        let mut invalid_interval = evaluation_reference();
        invalid_interval.turns[0].end_ms = 0;
        let error = score_diarization_documents(
            &invalid_interval,
            &evaluation_hypothesis(),
            &DiarizationScorerConfig::default(),
        )
        .expect_err("invalid interval");
        assert!(error.to_string().contains("interval_geometry"));

        let mut invalid_confidence = evaluation_hypothesis();
        invalid_confidence.turns[0].speaker_confidence = Some(f64::NAN);
        let error = score_diarization_documents(
            &evaluation_reference(),
            &invalid_confidence,
            &DiarizationScorerConfig::default(),
        )
        .expect_err("invalid confidence");
        assert!(error.to_string().contains("speaker_confidence_range"));

        let mut unsorted = evaluation_hypothesis();
        unsorted.turns.swap(0, 1);
        let error = score_diarization_documents(
            &evaluation_reference(),
            &unsorted,
            &DiarizationScorerConfig::default(),
        )
        .expect_err("unsorted");
        assert!(error.to_string().contains("turn_order"));
    }

    #[test]
    fn leakage_audit_detects_every_cross_split_identity_channel() {
        let mut train = manifest_recording(
            "recording-a",
            EvaluationSplit::Train,
            "origin-a",
            &["speaker-a"],
        );
        train.derived_from_recording_ids = vec!["source-a".to_owned()];
        train.augmentation_group_id = Some("augmentation-a".to_owned());
        train.enrollment_recording_ids = vec!["enrollment-a".to_owned()];
        let mut test = manifest_recording(
            "recording-a",
            EvaluationSplit::Test,
            "origin-a",
            &["speaker-a"],
        );
        test.derived_from_recording_ids = vec!["source-a".to_owned()];
        test.augmentation_group_id = Some("augmentation-a".to_owned());
        test.enrollment_recording_ids = vec!["enrollment-a".to_owned()];
        let manifest = DiarizationCorpusManifest {
            schema_version: DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION.to_owned(),
            corpus_id: "synthetic-corpus".to_owned(),
            license_id: "synthetic-only".to_owned(),
            recordings: vec![train, test],
        };
        let audit = audit_diarization_manifest(&manifest).expect("audit");
        assert!(!audit.passed);
        for kind in [
            LeakageKind::DuplicateRecording,
            LeakageKind::SharedOrigin,
            LeakageKind::SharedSpeaker,
            LeakageKind::SharedDerivedSource,
            LeakageKind::SharedAugmentation,
            LeakageKind::CrossSplitEnrollment,
        ] {
            assert!(
                audit.findings.iter().any(|finding| finding.kind == kind),
                "missing {kind:?}"
            );
        }
        verify_leakage_audit_hash(&audit).expect("audit hash");
    }

    #[test]
    fn clean_manifest_passes_and_remains_path_free() {
        let manifest = DiarizationCorpusManifest {
            schema_version: DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION.to_owned(),
            corpus_id: "public-corpus".to_owned(),
            license_id: "CC-BY-4.0".to_owned(),
            recordings: vec![
                manifest_recording(
                    "recording-a",
                    EvaluationSplit::Train,
                    "origin-a",
                    &["speaker-a"],
                ),
                manifest_recording(
                    "recording-b",
                    EvaluationSplit::Test,
                    "origin-b",
                    &["speaker-b"],
                ),
            ],
        };
        let audit = audit_diarization_manifest(&manifest).expect("audit");
        assert!(audit.passed);
        assert!(audit.findings.is_empty());
        let serialized = serde_json::to_string(&manifest).expect("serialize manifest");
        for forbidden in ["path", "transcript", "audio_file", "source_uri"] {
            assert!(!serialized.contains(forbidden));
        }
        let parsed = parse_diarization_corpus_manifest(serialized.as_bytes()).expect("parse");
        assert_eq!(parsed, manifest);
    }

    #[test]
    fn manifest_rejects_duplicate_entries_within_one_split() {
        let recording = manifest_recording(
            "recording-a",
            EvaluationSplit::Train,
            "origin-a",
            &["speaker-a"],
        );
        let manifest = DiarizationCorpusManifest {
            schema_version: DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION.to_owned(),
            corpus_id: "synthetic-corpus".to_owned(),
            license_id: "synthetic-only".to_owned(),
            recordings: vec![recording.clone(), recording],
        };
        let error = audit_diarization_manifest(&manifest).expect_err("duplicate manifest entry");
        assert!(error.to_string().contains("manifest_recording_order"));
    }

    #[test]
    fn diarization_score_is_invariant_to_speaker_label_permutation() {
        let reference = vec![
            ScoringTurn::labeled(0.0, 2.0, "alice"),
            ScoringTurn::labeled(2.0, 4.0, "bob"),
        ];
        let hypothesis = vec![
            ScoringTurn::labeled(0.0, 2.0, "SPEAKER_01"),
            ScoringTurn::labeled(2.0, 4.0, "SPEAKER_00"),
        ];
        let score = score_diarization(&reference, &hypothesis).expect("score");
        assert_eq!(score.der, Some(0.0));
        assert_eq!(score.jer, Some(0.0));
        assert_eq!(score.speaker_mapping["SPEAKER_01"], "alice");
        assert_eq!(score.speaker_mapping["SPEAKER_00"], "bob");
    }

    #[test]
    fn diarization_score_separates_miss_false_alarm_and_confusion() {
        let reference = vec![
            ScoringTurn::labeled(0.0, 2.0, "a"),
            ScoringTurn::labeled(2.0, 4.0, "b"),
        ];
        let hypothesis = vec![
            ScoringTurn::labeled(0.0, 1.0, "x"),
            ScoringTurn::labeled(2.0, 3.0, "x"),
            ScoringTurn::labeled(4.0, 5.0, "x"),
        ];
        let score = score_diarization(&reference, &hypothesis).expect("score");
        assert_close(score.reference_speaker_time_sec, 4.0);
        assert_close(score.missed_speech_sec, 2.0);
        assert_close(score.false_alarm_sec, 1.0);
        assert_close(score.speaker_confusion_sec, 1.0);
        assert_eq!(score.der, Some(1.0));
    }

    #[test]
    fn unknown_hypothesis_is_confusion_not_missed_speech() {
        let reference = vec![ScoringTurn::labeled(0.0, 1.0, "a")];
        let hypothesis = vec![ScoringTurn::unknown(0.0, 1.0)];
        let score = score_diarization(&reference, &hypothesis).expect("score");
        assert_eq!(score.missed_speech_sec, 0.0);
        assert_eq!(score.speaker_confusion_sec, 1.0);
        assert_eq!(score.der, Some(1.0));
    }

    #[test]
    fn labeled_hypothesis_against_empty_reference_is_false_alarm_without_panic() {
        let hypothesis = vec![ScoringTurn::labeled(0.0, 1.0, "cluster-a")];
        let score = score_diarization(&[], &hypothesis).expect("score");
        assert_eq!(score.reference_speaker_time_sec, 0.0);
        assert_eq!(score.missed_speech_sec, 0.0);
        assert_eq!(score.false_alarm_sec, 1.0);
        assert_eq!(score.speaker_confusion_sec, 0.0);
        assert_eq!(score.der, None);
        assert_eq!(score.jer, None);
        assert!(score.speaker_mapping.is_empty());
    }

    #[test]
    fn reference_overlap_contributes_speaker_time() {
        let reference = vec![
            ScoringTurn::labeled(0.0, 1.0, "a"),
            ScoringTurn::labeled(0.0, 1.0, "b"),
        ];
        let hypothesis = vec![ScoringTurn::labeled(0.0, 1.0, "x")];
        let score = score_diarization(&reference, &hypothesis).expect("score");
        assert_eq!(score.reference_speaker_time_sec, 2.0);
        assert_eq!(score.reference_overlap_sec, 1.0);
        assert_eq!(score.missed_speech_sec, 1.0);
    }

    #[test]
    fn invalid_reference_turn_fails_closed() {
        let reference = vec![ScoringTurn::unknown(0.0, 1.0)];
        let error = score_diarization(&reference, &[]).expect_err("reference must be labeled");
        assert!(error.to_string().contains("must have a speaker label"));
    }

    #[test]
    fn change_points_match_once_within_explicit_collar() {
        let score =
            score_change_points(&[1.0, 2.0, 3.0], &[0.95, 2.04, 2.08, 5.0], 0.1).expect("score");
        assert_eq!(score.matched_count, 2);
        assert_eq!(score.precision, Some(0.5));
        assert_close(score.recall.expect("recall"), 2.0 / 3.0);
        assert_close(score.mean_absolute_error_sec.expect("error"), 0.045);
    }

    #[test]
    fn change_point_matching_minimizes_error_after_cardinality() {
        let score = score_change_points(&[0.0, 10.0], &[0.0, 1.0, 10.0], 20.0).expect("score");
        assert_eq!(score.matched_count, 2);
        assert_eq!(score.mean_absolute_error_sec, Some(0.0));
    }

    #[test]
    fn overlapping_duplicate_hypothesis_label_is_one_active_speaker() {
        let reference = vec![ScoringTurn::labeled(0.0, 2.0, "a")];
        let hypothesis = vec![
            ScoringTurn::labeled(0.0, 2.0, "x"),
            ScoringTurn::labeled(0.5, 1.5, "x"),
        ];
        let score = score_diarization(&reference, &hypothesis).expect("score");
        assert_eq!(score.der, Some(0.0));
    }

    #[test]
    fn empty_change_points_have_undefined_ratios() {
        let score = score_change_points(&[], &[], 0.25).expect("score");
        assert_eq!(score.precision, None);
        assert_eq!(score.recall, None);
        assert_eq!(score.f1, None);
    }

    #[test]
    fn calibration_reports_accuracy_and_coverage() {
        let observations = [
            CalibrationObservation {
                confidence: 1.0,
                correct: true,
            },
            CalibrationObservation {
                confidence: 0.0,
                correct: false,
            },
            CalibrationObservation {
                confidence: 0.5,
                correct: true,
            },
        ];
        let score = score_calibration(&observations, 6, 2).expect("score");
        assert_eq!(score.coverage, 0.5);
        assert_close(score.brier_score.expect("brier"), 1.0 / 12.0);
        assert!(score.expected_calibration_error.expect("ece") >= 0.0);
    }

    #[test]
    fn calibration_rejects_confidence_outside_unit_interval() {
        let error = score_calibration(
            &[CalibrationObservation {
                confidence: 1.1,
                correct: true,
            }],
            1,
            10,
        )
        .expect_err("invalid confidence");
        assert!(error.to_string().contains("within [0, 1]"));
    }

    fn sine_wave(frequency_hz: f32, seconds: f32, amplitude: f32) -> Vec<f32> {
        let sample_count = (seconds * crate::native_engine::mel::SAMPLE_RATE as f32) as usize;
        (0..sample_count)
            .map(|sample| {
                let phase = 2.0 * std::f32::consts::PI * frequency_hz * sample as f32
                    / crate::native_engine::mel::SAMPLE_RATE as f32;
                amplitude * phase.sin()
            })
            .collect()
    }

    fn features(samples: &[f32]) -> Vec<AcousticFrameFeatures> {
        let mut output = Vec::new();
        extract_acoustic_features(
            samples,
            || false,
            |frame| {
                output.push(frame);
                Ok(())
            },
        )
        .expect("extract features");
        output
    }

    fn acoustic_feature_bytes(frames: &[AcousticFrameFeatures]) -> Vec<u8> {
        fn append_f32(output: &mut Vec<u8>, value: f32) {
            output.extend(value.to_bits().to_le_bytes());
        }

        fn append_optional_f32(output: &mut Vec<u8>, value: Option<f32>) {
            match value {
                Some(value) => {
                    output.push(1);
                    append_f32(output, value);
                }
                None => output.push(0),
            }
        }

        let mut output = Vec::new();
        output.extend((frames.len() as u64).to_le_bytes());
        for frame in frames {
            output.extend((frame.frame_index as u64).to_le_bytes());
            output.extend(frame.start_ms.to_le_bytes());
            output.extend(frame.end_ms.to_le_bytes());
            for value in frame
                .voice
                .cepstral_envelope
                .iter()
                .chain(frame.voice.cepstral_delta.iter())
                .chain(frame.voice.cepstral_delta_delta.iter())
            {
                append_f32(&mut output, *value);
            }
            append_optional_f32(&mut output, frame.voice.f0_hz);
            append_optional_f32(&mut output, frame.voice.pitch_uncertainty_octaves);
            for value in [
                frame.voice.voicing_confidence,
                frame.voice.harmonicity,
                frame.voice.harmonic_to_noise_db,
                frame.voice.formant_proxies_hz[0],
                frame.voice.formant_proxies_hz[1],
                frame.voice.formant_proxies_hz[2],
                frame.voice.temporal_modulation,
                frame.voice.voiced_fraction,
                frame.channel.rms_dbfs,
                frame.channel.dynamics_above_noise_db,
                frame.channel.spectral_centroid_hz,
                frame.channel.spectral_bandwidth_hz,
                frame.channel.spectral_rolloff_hz,
                frame.channel.spectral_flatness,
                frame.channel.spectral_tilt,
                frame.channel.low_band_fraction,
                frame.channel.mid_band_fraction,
                frame.channel.high_band_fraction,
                frame.channel.crest_factor,
                frame.channel.clipping_fraction,
                frame.channel.noise_floor_dbfs,
                frame.channel.spectral_flux,
                frame.channel.distortion_proxy,
                frame.channel.effective_band_limit_hz,
                frame.channel.high_frequency_attenuation,
                frame.channel.reverberation_proxy,
                frame.channel.muffling_proxy,
                frame.channel.stationary_coloration,
                frame.overlap_probability,
            ] {
                append_f32(&mut output, value);
            }
            output.extend([
                u8::from(frame.quality.voiced),
                u8::from(frame.quality.reliable_pitch),
                u8::from(frame.quality.low_energy),
                u8::from(frame.quality.clipped),
                u8::from(frame.quality.transient),
            ]);
        }
        output
    }

    fn median_pitch(frames: &[AcousticFrameFeatures]) -> f32 {
        let mut pitches = frames
            .iter()
            .filter_map(|frame| frame.voice.f0_hz)
            .collect::<Vec<_>>();
        pitches.sort_by(f32::total_cmp);
        pitches[pitches.len() / 2]
    }

    #[test]
    fn acoustic_features_distinguish_low_and_high_pitch_without_gender_labels() {
        let low = features(&sine_wave(120.0, 0.5, 0.4));
        let high = features(&sine_wave(260.0, 0.5, 0.4));
        let low_pitch = median_pitch(&low);
        let high_pitch = median_pitch(&high);
        assert!((low_pitch - 120.0).abs() < 8.0, "{low_pitch}");
        assert!((high_pitch - 260.0).abs() < 20.0, "{high_pitch}");
        assert!(high_pitch > low_pitch * 1.7);
    }

    #[test]
    fn complete_native_pipeline_uses_contextual_hints_without_changing_asr_bytes() {
        const SYNTHETIC_INPUT_SHA256: &str =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let samples = sine_wave(120.0, 5.0, 0.35);
        let segments = vec![
            TranscriptionSegment {
                start_sec: Some(0.5),
                end_sec: Some(2.0),
                text: "synthetic first".to_owned(),
                speaker: None,
                confidence: Some(0.91),
            },
            TranscriptionSegment {
                start_sec: Some(3.0),
                end_sec: Some(4.5),
                text: "synthetic second".to_owned(),
                speaker: None,
                confidence: Some(0.82),
            },
        ];
        let request = DiarizationRequest {
            engine: DiarizationEngine::Acoustic,
            speaker_count: SpeakerCountRequest::HardConstraint { count: 1 },
            known_intervals: vec![KnownSpeakerInterval {
                speaker_ref: "context_a".to_owned(),
                start_ms: 0,
                end_ms: 5_000,
                confidence: 1.0,
                policy: KnownSpeakerPolicy::HardMustLink,
                provenance: Some("synthetic-context".to_owned()),
            }],
            enrollment_edge_guard_ms: 0,
            max_prototypes: 16,
            ..DiarizationRequest::default()
        };
        let boundaries = AcousticBoundaryHints {
            speech_regions_ms: vec![(0, 5_000)],
            word_boundaries_ms: vec![500, 2_000, 3_000, 4_500],
            tiny_diarize_boundaries_ms: Vec::new(),
        };

        let (report, projection) = diarize_acoustic_pcm(
            AcousticDiarizationInput {
                samples: &samples,
                normalized_input_sha256: SYNTHETIC_INPUT_SHA256,
                segments: &segments,
                word_aligned: true,
                request: &request,
                boundary_hints: &boundaries,
            },
            || false,
        )
        .expect("complete native pipeline");
        let (repeated_report, repeated_projection) = diarize_acoustic_pcm(
            AcousticDiarizationInput {
                samples: &samples,
                normalized_input_sha256: SYNTHETIC_INPUT_SHA256,
                segments: &segments,
                word_aligned: true,
                request: &request,
                boundary_hints: &boundaries,
            },
            || false,
        )
        .expect("deterministic replay");
        let first_bytes = serde_json::to_vec(&(
            &report,
            &projection.segments,
            &projection.mixed_speaker_segment_indices,
            &projection.overlap_suspected_segment_indices,
        ))
        .expect("serialize first output");
        let repeated_bytes = serde_json::to_vec(&(
            &repeated_report,
            &repeated_projection.segments,
            &repeated_projection.mixed_speaker_segment_indices,
            &repeated_projection.overlap_suspected_segment_indices,
        ))
        .expect("serialize repeated output");
        assert_eq!(first_bytes, repeated_bytes);
        assert_eq!(
            Sha256::digest(&first_bytes),
            Sha256::digest(&repeated_bytes),
            "A/A output hashes must match"
        );

        assert_eq!(report.speaker_count.supported_speaker_count, 1);
        assert_eq!(
            report.speaker_count.status,
            SpeakerCountOutcomeStatus::Satisfied
        );
        assert_eq!(report.hint_evidence.len(), 1);
        assert_eq!(
            report.hint_evidence[0].disposition,
            SpeakerHintDisposition::HardAttributed
        );
        assert_eq!(
            report.speaker_count.speaker_evidence[0].reasons,
            vec![SpeakerEvidenceReason::SupportedByHardHint]
        );
        assert_eq!(report.fallback_status, DiarizationFallbackStatus::NotNeeded);
        assert_eq!(
            projection
                .segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>(),
            vec!["synthetic first", "synthetic second"]
        );
        assert_eq!(projection.segments[0].confidence, Some(0.91));
        assert_eq!(projection.segments[1].confidence, Some(0.82));
        assert_eq!(projection.segments[0].speaker.as_deref(), Some("context_a"));
        assert_eq!(projection.segments[1].speaker.as_deref(), Some("context_a"));
    }

    #[test]
    fn complete_native_pipeline_splits_stable_tracklet_for_bounded_hard_hint() {
        const SYNTHETIC_INPUT_SHA256: &str =
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let samples = sine_wave(140.0, 12.0, 0.35);
        let segments = [TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(12.0),
            text: "bounded synthetic supervision".to_owned(),
            speaker: None,
            confidence: Some(0.9),
        }];
        let request = DiarizationRequest {
            engine: DiarizationEngine::Acoustic,
            speaker_count: SpeakerCountRequest::HardConstraint { count: 1 },
            known_intervals: vec![KnownSpeakerInterval {
                speaker_ref: "bounded_context".to_owned(),
                start_ms: 4_000,
                end_ms: 8_000,
                confidence: 1.0,
                policy: KnownSpeakerPolicy::HardMustLink,
                provenance: Some("synthetic-bounded-context".to_owned()),
            }],
            enrollment_edge_guard_ms: 100,
            max_prototypes: 16,
            ..DiarizationRequest::default()
        };
        let (report, projection) = diarize_acoustic_pcm(
            AcousticDiarizationInput {
                samples: &samples,
                normalized_input_sha256: SYNTHETIC_INPUT_SHA256,
                segments: &segments,
                word_aligned: false,
                request: &request,
                boundary_hints: &AcousticBoundaryHints::default(),
            },
            || false,
        )
        .expect("bounded hard hint should create guarded enrollment tracklets");

        assert_eq!(report.speaker_count.supported_speaker_count, 1);
        assert_eq!(
            report.speaker_count.status,
            SpeakerCountOutcomeStatus::Satisfied
        );
        assert_eq!(report.fallback_status, DiarizationFallbackStatus::NotNeeded);
        assert!(
            report
                .profiles
                .iter()
                .any(|profile| profile.speaker_ref == "bounded_context" && profile.anchored)
        );
        assert!(report.turns.iter().any(|turn| turn.hard_hint_attributed));
        assert_eq!(
            projection.segments[0].speaker.as_deref(),
            Some("bounded_context")
        );
    }

    #[test]
    fn complete_native_pipeline_rejects_unbound_input_provenance() {
        let request = DiarizationRequest {
            engine: DiarizationEngine::Acoustic,
            ..DiarizationRequest::default()
        };
        let error = diarize_acoustic_pcm(
            AcousticDiarizationInput {
                samples: &[],
                normalized_input_sha256: "not-a-sha256",
                segments: &[],
                word_aligned: false,
                request: &request,
                boundary_hints: &AcousticBoundaryHints::default(),
            },
            || false,
        )
        .expect_err("unbound provenance must fail");
        assert!(
            error
                .to_string()
                .contains("64 lowercase hexadecimal characters")
        );
    }

    #[test]
    fn acoustic_features_retain_gain_and_muffling_evidence() {
        let mut clear = sine_wave(180.0, 0.5, 0.35);
        let high = sine_wave(2_200.0, 0.5, 0.15);
        for (sample, overtone) in clear.iter_mut().zip(high) {
            *sample += overtone;
        }
        let mut muffled = Vec::with_capacity(clear.len());
        let mut state = 0.0_f32;
        for sample in &clear {
            state = 0.12 * sample + 0.88 * state;
            muffled.push(0.45 * state);
        }
        let clear_features = features(&clear);
        let muffled_features = features(&muffled);
        let clear_channel = &clear_features[10].channel;
        let muffled_channel = &muffled_features[10].channel;
        assert!(clear_channel.rms_dbfs > muffled_channel.rms_dbfs + 6.0);
        assert!(clear_channel.high_band_fraction > muffled_channel.high_band_fraction * 2.0);
        assert!(clear_channel.spectral_centroid_hz > muffled_channel.spectral_centroid_hz);
        assert!(muffled_channel.muffling_proxy > clear_channel.muffling_proxy);
        assert!(
            muffled_channel.high_frequency_attenuation > clear_channel.high_frequency_attenuation
        );
        assert!(muffled_channel.effective_band_limit_hz < clear_channel.effective_band_limit_hz);
    }

    #[test]
    fn acoustic_v2_schema_is_complete_versioned_and_self_hashed() {
        let v1 = super::acoustic_feature_schema(super::AcousticFeatureSchemaVersion::V1);
        let v2 = super::acoustic_feature_schema(super::AcousticFeatureSchemaVersion::V2);
        assert_eq!(v1.version.id(), super::ACOUSTIC_FEATURE_SCHEMA_V1);
        assert_eq!(v1.voice_dimensions, 8);
        assert_eq!(v1.channel_dimensions, 8);
        assert_eq!(v2.version.id(), super::ACOUSTIC_FEATURE_SCHEMA_VERSION);
        assert_eq!(v2.voice_dimensions, VOICE_VECTOR_DIMENSIONS);
        assert_eq!(v2.channel_dimensions, CHANNEL_VECTOR_DIMENSIONS);
        assert_eq!(v2.frame_samples, ACOUSTIC_FRAME_SAMPLES);
        assert_eq!(v2.hop_samples, ACOUSTIC_HOP_SAMPLES);

        for owner in [
            super::AcousticFeatureOwner::Voice,
            super::AcousticFeatureOwner::Channel,
        ] {
            let dimensions = if owner == super::AcousticFeatureOwner::Voice {
                v2.voice_dimensions
            } else {
                v2.channel_dimensions
            };
            let mut covered = vec![false; dimensions];
            for family in v2.families.iter().filter(|family| family.owner == owner) {
                assert!(family.start_dimension < family.end_dimension_exclusive);
                assert!(family.end_dimension_exclusive <= dimensions);
                assert!(!family.unit.is_empty());
                assert!(!family.validity.is_empty());
                assert!(!family.normalization.is_empty());
                for coordinate in
                    &mut covered[family.start_dimension..family.end_dimension_exclusive]
                {
                    assert!(!*coordinate, "schema coordinate families overlap");
                    *coordinate = true;
                }
            }
            assert!(covered.into_iter().all(|coordinate| coordinate));
        }

        let v1_hash =
            super::acoustic_feature_schema_sha256(super::AcousticFeatureSchemaVersion::V1);
        let v2_hash =
            super::acoustic_feature_schema_sha256(super::AcousticFeatureSchemaVersion::V2);
        assert_eq!(v1_hash.len(), 64);
        assert_eq!(v2_hash.len(), 64);
        assert_ne!(v1_hash, v2_hash);
        assert_eq!(
            v2_hash,
            super::acoustic_feature_schema_sha256(super::AcousticFeatureSchemaVersion::V2)
        );
    }

    #[test]
    fn acoustic_ablation_ids_masks_and_schema_selection_are_exact() {
        let ids = super::AcousticFeatureAblation::ALL
            .into_iter()
            .map(super::AcousticFeatureAblation::id)
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), super::AcousticFeatureAblation::ALL.len());
        let frame = synthetic_feature(5, 0.25, false, false);
        let full =
            super::compact_vectors_for_ablation(&frame, super::AcousticFeatureAblation::FullV2);
        assert!(full.voice_valid[12..20].iter().all(|&valid| valid));
        assert!(full.voice_valid[20]);
        assert!(full.voice_valid[26]);
        assert!(full.voice_valid[27]);
        assert!(full.channel_valid);

        let no_pitch =
            super::compact_vectors_for_ablation(&frame, super::AcousticFeatureAblation::NoPitch);
        assert!(!no_pitch.voice_valid[20]);
        assert!(!no_pitch.voice_valid[26]);
        assert_eq!(no_pitch.voice_valid[21..26], full.voice_valid[21..26]);

        let no_channel =
            super::compact_vectors_for_ablation(&frame, super::AcousticFeatureAblation::NoChannel);
        assert!(!no_channel.channel_valid);
        assert_eq!(no_channel.voice_valid, full.voice_valid);

        let no_deltas =
            super::compact_vectors_for_ablation(&frame, super::AcousticFeatureAblation::NoDeltas);
        assert!(no_deltas.voice_valid[12..20].iter().all(|&valid| !valid));
        assert!(no_deltas.voice_valid[27]);

        let no_modulation = super::compact_vectors_for_ablation(
            &frame,
            super::AcousticFeatureAblation::NoModulation,
        );
        assert!(!no_modulation.voice_valid[27]);
        assert_eq!(
            super::AcousticFeatureAblation::V1.schema_version(),
            super::AcousticFeatureSchemaVersion::V1
        );
    }

    #[test]
    fn no_channel_ablation_preserves_voice_change_evidence() {
        let ring = (0..super::CHANGE_RING_FRAMES)
            .map(|index| {
                let value = if index < super::CHANGE_SCALES_FRAMES[4] {
                    -1.0
                } else {
                    1.0
                };
                synthetic_feature(index, value, false, false)
            })
            .collect::<VecDeque<_>>();
        let evidence = super::variance_aware_scale_evidence(
            &ring,
            super::CHANGE_SCALES_FRAMES[4],
            super::CHANGE_SCALES_FRAMES[3],
            super::AcousticFeatureAblation::NoChannel,
            super::acoustic_change_calibration(),
        );
        assert!(evidence.voice_evidence > 0.5);
        assert_eq!(evidence.channel_evidence, 0.0);
        assert_eq!(evidence.channel_distance, 0.0);
        assert!(evidence.voice_dimensions >= 3);
    }

    #[test]
    fn v1_channel_distance_uses_only_declared_coordinates() {
        let left = [0.0_f32; CHANNEL_VECTOR_DIMENSIONS];
        let mut right = [0.0_f32; CHANNEL_VECTOR_DIMENSIONS];
        right[..8].fill(1.0);
        assert_eq!(super::euclidean_distance_prefix(&left, &right, 8), 1.0);
        assert!(
            super::euclidean_distance(&left, &right)
                < super::euclidean_distance_prefix(&left, &right, 8)
        );
    }

    #[test]
    fn silence_is_unvoiced_instead_of_fabricating_pitch() {
        let frames = features(&vec![0.0; crate::native_engine::mel::SAMPLE_RATE / 2]);
        assert!(!frames.is_empty());
        assert!(frames.iter().all(|frame| frame.voice.f0_hz.is_none()));
        assert!(
            frames
                .iter()
                .all(|frame| frame.voice.pitch_uncertainty_octaves.is_none())
        );
        assert!(frames.iter().all(|frame| frame.quality.low_energy));
        for frame in &frames {
            let compact =
                super::compact_vectors_for_schema(frame, super::AcousticFeatureSchemaVersion::V2);
            assert!(!compact.voice_valid[20]);
            assert_eq!(compact.voice[20], 0.0);
            assert_eq!(compact.identity_quality, 0.0);
        }
    }

    #[test]
    fn gain_changes_channel_level_without_moving_pitch_or_envelope() {
        let quiet = features(&sine_wave(190.0, 0.5, 0.18));
        let loud = features(&sine_wave(190.0, 0.5, 0.72));
        let quiet_frame = &quiet[20];
        let loud_frame = &loud[20];
        assert!(
            loud_frame.channel.rms_dbfs > quiet_frame.channel.rms_dbfs + 11.0,
            "{quiet_frame:#?}\n{loud_frame:#?}"
        );
        assert_eq!(quiet_frame.voice.f0_hz, loud_frame.voice.f0_hz);
        for (quiet, loud) in quiet_frame
            .voice
            .cepstral_envelope
            .iter()
            .zip(loud_frame.voice.cepstral_envelope)
        {
            assert!((quiet - loud).abs() < 1e-4, "{quiet} vs {loud}");
        }
    }

    #[test]
    fn tones_harmonics_chirps_and_clipping_remain_finite_and_masked() {
        let mut harmonic = sine_wave(125.0, 0.5, 0.35);
        for (sample, overtone) in harmonic.iter_mut().zip(sine_wave(250.0, 0.5, 0.12)) {
            *sample += overtone;
        }
        let sample_rate = crate::native_engine::mel::SAMPLE_RATE as f32;
        let chirp = (0..crate::native_engine::mel::SAMPLE_RATE / 2)
            .map(|index| {
                let time = index as f32 / sample_rate;
                let phase = 2.0 * std::f32::consts::PI * (90.0 * time + 220.0 * time * time);
                0.4 * phase.sin()
            })
            .collect::<Vec<_>>();
        let clipped = sine_wave(160.0, 0.5, 1.0)
            .into_iter()
            .map(|sample| (2.0 * sample).clamp(-1.0, 1.0))
            .collect::<Vec<_>>();

        for signal in [&harmonic, &chirp, &clipped] {
            let extracted = features(signal);
            assert!(!extracted.is_empty());
            for frame in extracted {
                assert!(
                    frame
                        .voice
                        .cepstral_envelope
                        .iter()
                        .chain(frame.voice.cepstral_delta.iter())
                        .chain(frame.voice.cepstral_delta_delta.iter())
                        .chain(frame.voice.formant_proxies_hz.iter())
                        .all(|value| value.is_finite())
                );
                assert!(frame.voice.harmonic_to_noise_db.is_finite());
                assert!(frame.voice.temporal_modulation.is_finite());
                assert!(frame.channel.reverberation_proxy.is_finite());
            }
        }
        assert!(features(&clipped).iter().any(|frame| frame.quality.clipped));
    }

    #[test]
    fn arbitrary_finite_pcm_is_deterministic_and_bounded() {
        let mut state = 0x9e37_79b9_u32;
        let samples = (0..4_000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect::<Vec<_>>();
        let first = features(&samples);
        let repeated = features(&samples);
        assert_eq!(first, repeated);
        assert_eq!(
            first.len(),
            1 + (samples.len() - ACOUSTIC_FRAME_SAMPLES) / ACOUSTIC_HOP_SAMPLES
        );
    }

    #[test]
    fn short_input_emits_no_partial_or_padded_frame() {
        let mut emitted = 0usize;
        let summary = extract_acoustic_features(
            &vec![0.0; ACOUSTIC_FRAME_SAMPLES - 1],
            || false,
            |_| {
                emitted += 1;
                Ok(())
            },
        )
        .expect("short input");
        assert_eq!(emitted, 0);
        assert_eq!(summary.frame_count, 0);
        assert_eq!(
            summary.feature_schema,
            super::ACOUSTIC_FEATURE_SCHEMA_VERSION
        );
    }

    #[test]
    fn non_finite_pcm_fails_closed() {
        let mut samples = vec![0.0; ACOUSTIC_FRAME_SAMPLES];
        samples[17] = f32::NAN;
        let error =
            extract_acoustic_features(&samples, || false, |_| Ok(())).expect_err("NaN must fail");
        assert!(error.to_string().contains("non-finite PCM"));
    }

    #[test]
    fn entry_cancellation_precedes_whole_audio_validation() {
        let samples = [f32::NAN; ACOUSTIC_FRAME_SAMPLES];
        let error = extract_acoustic_features(&samples, || true, |_| Ok(()))
            .expect_err("pre-cancelled extraction must stop before inspecting PCM");
        assert!(matches!(error, FwError::Cancelled(_)));
    }

    #[test]
    fn out_of_range_pcm_fails_before_spectral_overflow() {
        let mut samples = vec![0.0; ACOUSTIC_FRAME_SAMPLES];
        samples[17] = 1.000_001;
        let error = extract_acoustic_features(&samples, || false, |_| Ok(()))
            .expect_err("non-normalized PCM must fail");
        assert!(error.to_string().contains("normalized [-1, 1] range"));
    }

    #[test]
    fn cancellation_is_checked_within_thirty_two_frames() {
        let samples = vec![0.0; ACOUSTIC_FRAME_SAMPLES + 100 * crate::native_engine::mel::HOP];
        let mut checks = 0usize;
        let mut emitted = 0usize;
        let error = extract_acoustic_features(
            &samples,
            || {
                checks += 1;
                checks >= 2
            },
            |_| {
                emitted += 1;
                Ok(())
            },
        )
        .expect_err("cancel");
        assert_eq!(emitted, ACOUSTIC_CANCELLATION_INTERVAL_FRAMES);
        assert!(error.to_string().contains("frame 32"));
    }

    #[test]
    fn extraction_state_is_bounded_independent_of_duration() {
        let short = extract_acoustic_features(
            &vec![0.0; crate::native_engine::mel::SAMPLE_RATE],
            || false,
            |_| Ok(()),
        )
        .expect("short");
        let long = extract_acoustic_features(
            &vec![0.0; crate::native_engine::mel::SAMPLE_RATE * 8],
            || false,
            |_| Ok(()),
        )
        .expect("long");
        assert!(long.frame_count > short.frame_count);
        assert_eq!(
            long.retained_state_bytes_upper_bound,
            short.retained_state_bytes_upper_bound
        );
    }

    #[test]
    fn factored_power_spectrum_matches_independent_dft_bins() {
        let frame = sine_wave(440.0, 0.025, 0.5);
        let mut actual = [0.0_f32; crate::native_engine::mel::N_FREQ_BINS];
        crate::native_engine::mel::fixed_frame_power_spectrum(&frame, &mut actual)
            .expect("spectrum");
        for bin in [0usize, 1, 10, 11, 25, 100, 200] {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for (sample, &value) in frame.iter().enumerate() {
                let window = 0.5
                    * (1.0
                        - (2.0 * std::f64::consts::PI * sample as f64
                            / ACOUSTIC_FRAME_SAMPLES as f64)
                            .cos());
                let phase = 2.0 * std::f64::consts::PI * bin as f64 * sample as f64
                    / ACOUSTIC_FRAME_SAMPLES as f64;
                re += f64::from(value) * window * phase.cos();
                im -= f64::from(value) * window * phase.sin();
            }
            let expected = (re * re + im * im) as f32;
            let tolerance = (expected.abs() * 2e-4).max(2e-5);
            assert!(
                (actual[bin] - expected).abs() <= tolerance,
                "bin {bin}: actual {}, expected {expected}, tolerance {tolerance}",
                actual[bin]
            );
        }
    }

    fn synthetic_feature(
        frame_index: usize,
        voice_value: f32,
        low_energy: bool,
        transient: bool,
    ) -> AcousticFrameFeatures {
        AcousticFrameFeatures {
            frame_index,
            start_ms: frame_index as u64 * 10,
            end_ms: frame_index as u64 * 10 + 25,
            voice: VoiceFeatureView {
                cepstral_envelope: [voice_value; CEPSTRAL_COEFFICIENTS],
                cepstral_delta: [0.0; CEPSTRAL_COEFFICIENTS],
                cepstral_delta_delta: [0.0; CEPSTRAL_COEFFICIENTS],
                f0_hz: (!low_energy).then_some(120.0 + 80.0 * voice_value),
                pitch_uncertainty_octaves: (!low_energy).then_some(0.2),
                voicing_confidence: if low_energy { 0.0 } else { 0.9 },
                harmonicity: if low_energy { 0.0 } else { 0.9 },
                harmonic_to_noise_db: if low_energy { -20.0 } else { 9.5 },
                formant_proxies_hz: [600.0, 1_500.0, 2_700.0],
                temporal_modulation: 0.0,
                voiced_fraction: if low_energy { 0.0 } else { 0.9 },
            },
            channel: ChannelFeatureView {
                rms_dbfs: if low_energy {
                    -80.0
                } else {
                    -20.0 + voice_value
                },
                dynamics_above_noise_db: if low_energy { 0.0 } else { 30.0 },
                spectral_centroid_hz: 1_000.0 + 500.0 * voice_value,
                spectral_bandwidth_hz: 1_200.0,
                spectral_rolloff_hz: 2_500.0,
                spectral_flatness: 0.1,
                spectral_tilt: -2.0,
                low_band_fraction: 0.4,
                mid_band_fraction: 0.5,
                high_band_fraction: 0.1,
                crest_factor: 1.5,
                clipping_fraction: 0.0,
                noise_floor_dbfs: -70.0,
                spectral_flux: if transient { 0.8 } else { 0.0 },
                distortion_proxy: 0.1,
                effective_band_limit_hz: 4_000.0,
                high_frequency_attenuation: 0.3,
                reverberation_proxy: 0.2,
                muffling_proxy: 0.4,
                stationary_coloration: if transient { 0.2 } else { 1.0 },
            },
            overlap_probability: 0.0,
            quality: AcousticQualityMask {
                voiced: !low_energy,
                reliable_pitch: !low_energy,
                low_energy,
                clipped: false,
                transient,
            },
        }
    }

    fn trajectory_fixture_values()
    -> [[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT] {
        std::array::from_fn(|family_index| {
            std::array::from_fn(|offset| {
                let phase = std::f32::consts::TAU * offset as f32
                    / ACOUSTIC_TRAJECTORY_HISTORY_FRAMES as f32;
                let family_phase = family_index as f32 * 0.37;
                (0.5 + 0.21 * (3.0 * phase + family_phase).sin()
                    + 0.07 * (7.0 * phase - family_phase).cos())
                .clamp(0.0, 1.0)
            })
        })
    }

    fn trajectory_window<'a>(
        values: &'a super::AcousticTrajectoryValues,
        valid: &'a super::AcousticTrajectoryValidity,
        oldest_index: usize,
        start_frame_index: usize,
        end_frame_index: usize,
    ) -> super::AcousticTrajectoryWindow<'a> {
        super::AcousticTrajectoryWindow::new(
            values,
            valid,
            oldest_index,
            start_frame_index,
            end_frame_index,
        )
    }

    fn assert_sidecar_study_state_matches(
        actual: &AcousticSidecarStudy,
        expected: &AcousticSidecarStudy,
    ) {
        assert_eq!(actual.config, expected.config);
        assert_eq!(
            actual.configuration_sha256_digest,
            expected.configuration_sha256_digest
        );
        assert_eq!(actual.modulation.voice, expected.modulation.voice);
        assert_eq!(
            actual.modulation.channel_level,
            expected.modulation.channel_level
        );
        assert_eq!(
            actual.modulation.channel_coloration,
            expected.modulation.channel_coloration
        );
        assert_eq!(
            actual.modulation.voice_valid,
            expected.modulation.voice_valid
        );
        assert_eq!(
            actual.modulation.channel_valid,
            expected.modulation.channel_valid
        );
        assert_eq!(actual.modulation.next_index, expected.modulation.next_index);
        assert_eq!(
            actual.modulation.buffered_frames,
            expected.modulation.buffered_frames
        );
        assert_eq!(
            actual.modulation.expected_next_frame_index,
            expected.modulation.expected_next_frame_index
        );
        assert_eq!(actual.trajectory.values, expected.trajectory.values);
        assert_eq!(actual.trajectory.valid, expected.trajectory.valid);
        assert_eq!(actual.trajectory.next_index, expected.trajectory.next_index);
        assert_eq!(
            actual.trajectory.buffered_frames,
            expected.trajectory.buffered_frames
        );
        assert_eq!(
            actual.trajectory.expected_next_frame_index,
            expected.trajectory.expected_next_frame_index
        );
    }

    fn reference_normalized_trajectory(
        input: &[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    ) -> [f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES] {
        let mean = input.iter().map(|value| f64::from(*value)).sum::<f64>()
            / ACOUSTIC_TRAJECTORY_HISTORY_FRAMES as f64;
        let energy = input
            .iter()
            .map(|value| {
                let centered = f64::from(*value) - mean;
                centered * centered
            })
            .sum::<f64>();
        let inverse_norm = energy.sqrt().recip();
        std::array::from_fn(|index| ((f64::from(input[index]) - mean) * inverse_norm) as f32)
    }

    fn scalar_reference_full_valid_trajectory_wavelet(
        input: &[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
        basis: AcousticWaveletBasis,
    ) -> [super::AcousticTrajectoryWaveletLevelSummary; super::ACOUSTIC_WAVELET_MAX_LEVELS] {
        const REFERENCE_HAAR_LOW: [f64; 2] = [
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ];
        const REFERENCE_HAAR_HIGH: [f64; 2] = [
            std::f64::consts::FRAC_1_SQRT_2,
            -std::f64::consts::FRAC_1_SQRT_2,
        ];
        const REFERENCE_D4_LOW: [f64; 4] = [
            0.482_962_913_144_534_16,
            0.836_516_303_737_807_9,
            0.224_143_868_042_013_4,
            -0.129_409_522_551_260_37,
        ];
        const REFERENCE_D4_HIGH: [f64; 4] = [
            0.129_409_522_551_260_37,
            0.224_143_868_042_013_4,
            -0.836_516_303_737_807_9,
            0.482_962_913_144_534_16,
        ];

        let (low_taps, high_taps): (&[f64], &[f64]) = match basis {
            AcousticWaveletBasis::Haar => (&REFERENCE_HAAR_LOW, &REFERENCE_HAAR_HIGH),
            AcousticWaveletBasis::DaubechiesFourTap => (&REFERENCE_D4_LOW, &REFERENCE_D4_HIGH),
        };
        let mut current = reference_normalized_trajectory(input);
        let mut current_len = ACOUSTIC_TRAJECTORY_HISTORY_FRAMES;
        std::array::from_fn(|level| {
            let dilation = 1usize << level;
            let output_len = current_len - (low_taps.len() - 1) * dilation;
            let mut approximation = [0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            let mut detail = [0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            for output_index in 0..output_len {
                let mut low = 0.0_f64;
                let mut high = 0.0_f64;
                for (tap, (&low_coefficient, &high_coefficient)) in
                    low_taps.iter().zip(high_taps).enumerate()
                {
                    let sample = f64::from(current[output_index + tap * dilation]);
                    low += low_coefficient * sample;
                    high += high_coefficient * sample;
                }
                approximation[output_index] = low as f32;
                detail[output_index] = high as f32;
            }
            let detail = &detail[..output_len];
            let detail_energy = detail
                .iter()
                .map(|value| {
                    let value = f64::from(*value);
                    value * value
                })
                .sum::<f64>();
            let mean_absolute_detail = detail
                .iter()
                .map(|value| f64::from(value.abs()))
                .sum::<f64>()
                / output_len as f64;
            let detail_rms = (detail_energy / output_len as f64).sqrt();
            let shape_available =
                detail_rms > f64::from(super::ACOUSTIC_TRAJECTORY_DETAIL_RMS_FLOOR);
            let normalized_entropy = if shape_available && output_len > 1 {
                detail
                    .iter()
                    .map(|value| {
                        let value = f64::from(*value);
                        let probability = value * value / detail_energy;
                        if probability > 0.0 {
                            -probability * probability.ln()
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / (output_len as f64).ln()
            } else {
                0.0
            };
            let normalized_detail_change = if shape_available && output_len > 1 {
                detail
                    .windows(2)
                    .map(|pair| (f64::from(pair[1]) - f64::from(pair[0])).abs())
                    .sum::<f64>()
                    / (output_len - 1) as f64
                    / detail_rms
            } else {
                0.0
            };
            current[..output_len].copy_from_slice(&approximation[..output_len]);
            current_len = output_len;
            super::AcousticTrajectoryWaveletLevelSummary {
                available: true,
                valid_coefficients: output_len,
                mean_absolute_detail: mean_absolute_detail as f32,
                detail_rms: detail_rms as f32,
                normalized_entropy_available: shape_available && output_len > 1,
                normalized_entropy: normalized_entropy.clamp(0.0, 1.0) as f32,
                adjacent_valid_pairs: output_len - 1,
                normalized_detail_change_available: shape_available && output_len > 1,
                normalized_detail_change: normalized_detail_change as f32,
            }
        })
    }

    fn scalar_reference_valid_haar_modulus(
        input: &[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
        input_len: usize,
        support: usize,
    ) -> ([f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES], usize) {
        let normalizer = 1.0 / (support as f64).sqrt();
        let output_len = input_len - support + 1;
        let mut output = [0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        for position in 0..output_len {
            let response = (0..support)
                .map(|tap| {
                    let sign = if tap < support / 2 { 1.0 } else { -1.0 };
                    sign * f64::from(input[position + tap])
                })
                .sum::<f64>();
            output[position] = (response * normalizer).abs() as f32;
        }
        (output, output_len)
    }

    fn scalar_reference_full_valid_scattering(
        input: &[f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES],
    ) -> ([f32; 3], [f32; 3]) {
        let normalized = reference_normalized_trajectory(input);
        let first_paths = [2, 4, 8].map(|support| {
            scalar_reference_valid_haar_modulus(
                &normalized,
                ACOUSTIC_TRAJECTORY_HISTORY_FRAMES,
                support,
            )
        });
        let first_order = first_paths.map(|(path, output_len)| {
            (path[..output_len]
                .iter()
                .map(|value| f64::from(*value))
                .sum::<f64>()
                / output_len as f64) as f32
        });
        let second_paths = [[0, 1], [0, 2], [1, 2]].map(|[first_scale, second_scale]| {
            scalar_reference_valid_haar_modulus(
                &first_paths[first_scale].0,
                first_paths[first_scale].1,
                [2, 4, 8][second_scale],
            )
        });
        let second_order = second_paths.map(|(path, output_len)| {
            (path[..output_len]
                .iter()
                .map(|value| f64::from(*value))
                .sum::<f64>()
                / output_len as f64) as f32
        });
        (first_order, second_order)
    }

    fn trajectory_sidecar_feature(frame_index: usize) -> AcousticFrameFeatures {
        let mut frame = synthetic_feature(frame_index, 0.0, false, false);
        let phase =
            std::f32::consts::TAU * frame_index as f32 / ACOUSTIC_TRAJECTORY_HISTORY_FRAMES as f32;
        for (coefficient, value) in frame.voice.cepstral_envelope.iter_mut().enumerate() {
            let coefficient_phase = coefficient as f32 * 0.23;
            *value = 0.35 * (4.0 * phase + coefficient_phase).sin()
                + 0.12 * (7.0 * phase - coefficient_phase).cos();
        }
        frame.voice.voiced_fraction = 0.55 + 0.25 * (3.0 * phase).sin();
        frame.voice.temporal_modulation = 0.5 + 0.3 * (4.0 * phase).sin();
        if frame_index % 5 == 0 {
            frame.voice.f0_hz = None;
            frame.voice.pitch_uncertainty_octaves = None;
            frame.voice.voicing_confidence = 0.0;
            frame.voice.harmonicity = 0.0;
            frame.voice.harmonic_to_noise_db = -20.0;
            frame.quality.voiced = false;
            frame.quality.reliable_pitch = false;
        }
        frame.channel.rms_dbfs = -22.0 + 3.0 * (2.0 * phase).sin();
        frame.channel.muffling_proxy = 0.45 + 0.2 * (5.0 * phase).cos();
        frame.channel.low_band_fraction = 0.3 + 0.08 * phase.sin();
        frame.channel.mid_band_fraction = 0.45 + 0.06 * (2.0 * phase).cos();
        frame.channel.high_band_fraction =
            1.0 - frame.channel.low_band_fraction - frame.channel.mid_band_fraction;
        frame
    }

    fn assert_trajectory_wavelet_summary_is_bounded(
        summary: &super::AcousticTrajectoryWaveletSummary,
    ) {
        assert!(summary.requested_levels <= super::ACOUSTIC_WAVELET_MAX_LEVELS);
        assert_eq!(
            summary.families.map(|family| family.family),
            [
                AcousticTrajectoryFamily::VoicedCepstralEnvelopeMagnitude,
                AcousticTrajectoryFamily::VoicedOccupancy,
                AcousticTrajectoryFamily::LowBandFraction,
                AcousticTrajectoryFamily::MidBandFraction,
                AcousticTrajectoryFamily::HighBandFraction,
            ]
        );
        assert_eq!(
            summary.families.map(|family| family.owner),
            [
                AcousticSidecarFeatureOwner::Voice,
                AcousticSidecarFeatureOwner::MixedAuxiliary,
                AcousticSidecarFeatureOwner::Channel,
                AcousticSidecarFeatureOwner::Channel,
                AcousticSidecarFeatureOwner::Channel,
            ]
        );
        for (family_index, family) in summary.families.iter().enumerate() {
            assert_eq!(family.family, AcousticTrajectoryFamily::ALL[family_index]);
            assert_eq!(family.owner, family.family.owner());
            assert!(family.input_valid_frames <= ACOUSTIC_TRAJECTORY_HISTORY_FRAMES);
            assert!(family.valid_level_count <= summary.requested_levels);
            for level in family.levels.iter().take(summary.requested_levels) {
                assert!(level.valid_coefficients <= ACOUSTIC_TRAJECTORY_HISTORY_FRAMES);
                assert!(level.mean_absolute_detail.is_finite());
                assert!(level.mean_absolute_detail >= 0.0);
                assert!(level.detail_rms.is_finite());
                assert!(level.detail_rms >= 0.0);
                assert!(level.normalized_entropy.is_finite());
                assert!((0.0..=1.0).contains(&level.normalized_entropy));
                if !level.normalized_entropy_available {
                    assert_eq!(level.normalized_entropy, 0.0);
                }
                assert!(level.adjacent_valid_pairs <= level.valid_coefficients.saturating_sub(1));
                assert!(level.normalized_detail_change.is_finite());
                assert!(level.normalized_detail_change >= 0.0);
                if !level.normalized_detail_change_available {
                    assert_eq!(level.normalized_detail_change, 0.0);
                }
                if !level.available {
                    assert_eq!(level.mean_absolute_detail, 0.0);
                    assert_eq!(level.detail_rms, 0.0);
                    assert!(!level.normalized_entropy_available);
                    assert_eq!(level.normalized_entropy, 0.0);
                    assert_eq!(level.adjacent_valid_pairs, 0);
                    assert!(!level.normalized_detail_change_available);
                    assert_eq!(level.normalized_detail_change, 0.0);
                }
            }
        }
    }

    fn assert_scattering_summary_is_bounded(summary: &super::AcousticScatteringSummary) {
        assert_eq!(
            summary.families.map(|family| family.family),
            [
                AcousticTrajectoryFamily::VoicedCepstralEnvelopeMagnitude,
                AcousticTrajectoryFamily::VoicedOccupancy,
                AcousticTrajectoryFamily::LowBandFraction,
                AcousticTrajectoryFamily::MidBandFraction,
                AcousticTrajectoryFamily::HighBandFraction,
            ]
        );
        assert_eq!(
            summary.families.map(|family| family.owner),
            [
                AcousticSidecarFeatureOwner::Voice,
                AcousticSidecarFeatureOwner::MixedAuxiliary,
                AcousticSidecarFeatureOwner::Channel,
                AcousticSidecarFeatureOwner::Channel,
                AcousticSidecarFeatureOwner::Channel,
            ]
        );
        for (family_index, family) in summary.families.iter().enumerate() {
            assert_eq!(family.family, AcousticTrajectoryFamily::ALL[family_index]);
            assert_eq!(family.owner, family.family.owner());
            assert!(family.input_valid_frames <= ACOUSTIC_TRAJECTORY_HISTORY_FRAMES);
            for scale_index in 0..ACOUSTIC_SCATTERING_SCALE_SUPPORTS.len() {
                assert!(
                    family.first_order_valid_positions[scale_index]
                        <= ACOUSTIC_TRAJECTORY_HISTORY_FRAMES
                );
                assert!(family.first_order_mean_modulus[scale_index].is_finite());
                assert!(family.first_order_mean_modulus[scale_index] >= 0.0);
                if !family.first_order_available[scale_index] {
                    assert_eq!(family.first_order_mean_modulus[scale_index], 0.0);
                }
            }
            for pair_index in 0..ACOUSTIC_SCATTERING_SCALE_PAIRS.len() {
                assert!(
                    family.second_order_valid_positions[pair_index]
                        <= ACOUSTIC_TRAJECTORY_HISTORY_FRAMES
                );
                assert!(family.second_order_mean_modulus[pair_index].is_finite());
                assert!(family.second_order_mean_modulus[pair_index] >= 0.0);
                if !family.second_order_available[pair_index] {
                    assert_eq!(family.second_order_mean_modulus[pair_index], 0.0);
                }
            }
        }
    }

    fn assert_wavelet_summary_is_bounded(summary: &super::AcousticWaveletSummary) {
        assert!(summary.valid_level_count <= super::ACOUSTIC_WAVELET_MAX_LEVELS);
        assert_eq!(summary.owner, AcousticSidecarFeatureOwner::MixedAuxiliary);
        assert!(summary.final_approximation_energy_fraction.is_finite());
        assert!((0.0..=1.0).contains(&summary.final_approximation_energy_fraction));
        assert!(
            summary
                .maximum_energy_conservation_relative_error
                .is_finite()
        );
        assert!(
            summary.maximum_energy_conservation_relative_error
                <= super::ACOUSTIC_WAVELET_ENERGY_TOLERANCE
        );
        for level in summary.levels.iter().take(summary.valid_level_count) {
            assert!(level.detail_energy_fraction.is_finite());
            assert!((0.0..=1.0).contains(&level.detail_energy_fraction));
            assert!(level.detail_log_energy.is_finite());
            assert!(level.normalized_entropy.is_finite());
            assert!((0.0..=1.0).contains(&level.normalized_entropy));
            assert!(level.coefficient_flatness.is_finite());
            assert!((0.0..=1.0).contains(&level.coefficient_flatness));
            assert!(level.crest_factor.is_finite());
            assert!(level.crest_factor >= 0.0);
            assert!(level.normalized_detail_change.is_finite());
            assert!(level.normalized_detail_change >= 0.0);
            assert!(level.energy_conservation_relative_error.is_finite());
            assert!(
                level.energy_conservation_relative_error
                    <= super::ACOUSTIC_WAVELET_ENERGY_TOLERANCE
            );
        }
    }

    #[test]
    fn sidecar_study_modes_are_separate_default_off_and_hash_complete() {
        assert_eq!(
            AcousticSidecarStudyConfig::default().mode,
            AcousticSidecarStudyMode::Off
        );
        assert_eq!(
            AcousticSidecarStudyConfig::default().frame_wavelet_levels,
            0
        );
        assert_eq!(
            AcousticSidecarStudyConfig::default().trajectory_wavelet_mode,
            AcousticTrajectoryWaveletMode::Off
        );
        assert_eq!(
            AcousticSidecarStudyConfig::default().trajectory_wavelet_levels,
            0
        );
        assert_eq!(
            AcousticSidecarStudyConfig::default().scattering_mode,
            AcousticScatteringMode::Off
        );
        assert_eq!(
            AcousticSidecarStudyMode::ALL.map(AcousticSidecarStudyMode::id),
            [
                "off",
                "haar",
                "daubechies_four_tap",
                "modulation",
                "haar_modulation",
                "daubechies_four_tap_modulation",
            ]
        );
        assert_eq!(
            AcousticSidecarStudyMode::ALL.map(|mode| {
                (
                    mode.wavelet_basis().map(AcousticWaveletBasis::id),
                    mode.uses_modulation(),
                )
            }),
            [
                (None, false),
                (Some("haar"), false),
                (Some("daubechies-four-tap"), false),
                (None, true),
                (Some("haar"), true),
                (Some("daubechies-four-tap"), true),
            ]
        );
        assert_eq!(
            AcousticTrajectoryWaveletMode::ALL.map(AcousticTrajectoryWaveletMode::id),
            ["off", "haar", "daubechies_four_tap"]
        );
        assert_eq!(
            AcousticScatteringMode::ALL.map(AcousticScatteringMode::id),
            [
                "off",
                "first_order",
                "second_order",
                "first_and_second_order",
            ]
        );
        assert_eq!(
            super::AcousticFeatureAblation::ALL.map(super::AcousticFeatureAblation::id),
            [
                "full_v2",
                "no_pitch",
                "no_channel",
                "no_deltas",
                "no_modulation",
                "v1",
            ]
        );
        assert_eq!(
            super::acoustic_feature_schema_sha256(super::AcousticFeatureSchemaVersion::V2),
            "093f04cec2743eeca83c1bb031cf15014955c3bd901cfec0baeb03cc0ec7744b"
        );

        let configs = [
            AcousticSidecarStudyConfig::default(),
            AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Haar,
                frame_wavelet_levels: 4,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::DaubechiesFourTap,
                frame_wavelet_levels: 4,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Modulation,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::HaarAndModulation,
                frame_wavelet_levels: 4,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::DaubechiesFourTapAndModulation,
                frame_wavelet_levels: 4,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Haar,
                trajectory_wavelet_levels: 4,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::DaubechiesFourTap,
                trajectory_wavelet_levels: 3,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                scattering_mode: AcousticScatteringMode::FirstOrder,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                scattering_mode: AcousticScatteringMode::SecondOrder,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                scattering_mode: AcousticScatteringMode::FirstAndSecondOrder,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::DaubechiesFourTapAndModulation,
                frame_wavelet_levels: 4,
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::DaubechiesFourTap,
                trajectory_wavelet_levels: 4,
                scattering_mode: AcousticScatteringMode::FirstAndSecondOrder,
            },
            AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Haar,
                frame_wavelet_levels: 1,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::DaubechiesFourTap,
                trajectory_wavelet_levels: 1,
                ..AcousticSidecarStudyConfig::default()
            },
        ];
        let hashes = configs
            .map(|config| acoustic_sidecar_study_config_sha256(config).expect("config hash"));
        assert!(hashes.iter().all(|hash| hash.len() == 64));
        assert_eq!(hashes.iter().collect::<BTreeSet<_>>().len(), hashes.len());
        assert_eq!(
            hashes,
            [
                "0fa6aecb32ccb25a6c10812195f62fedc4c0ed9d68d1bb351c054effa4d436e2",
                "77f8ef0ccbb32e433e8d5a808e9f54d0468a17ebfd0d02ce93b5052ee1513c06",
                "24bd8e1d74f20682a01e1510604e07a28e9baab1a86b3a4be819ea8c90d46515",
                "01e59ea5c0831a6a829f2df6cdf609d0a89acd08d63283b91501934e8c8e4945",
                "f5f77ac6027c1b6bb0aa939eefeaccd68003f2bafd4a3f0e35ef2c91cb040dbe",
                "ea36a9d513c9bc4bc1c0dc2eeb7c4ef46d7e0385792349442b5dd3f4a87997c5",
                "c8b9cc467933701b0b2083979f93627d9accf4c751931559a6346be693f73799",
                "ee1eac788d4b8be4928fe2585478ef12cfa34f4851d5c9d998e09eb3011c2ab4",
                "02375b0c6b68897d49e4e072e5eb840a3d156c26a6c3981d21a698eb7d0ed398",
                "a439d3242ef3db1291ffef5b6972f8eea5bd2608b332252a118605098037650a",
                "3a6db5ec18c5ebaa009593476564325acc6fff2a42cd4e96e3f116e331f66725",
                "32f6abdff8f6f79d65945e0112d81a31d366397ecc943dab4f2fb946ee40f1ce",
                "3ddfe65a5d6ace5c2433823629fa2e6dcb2e74491be16ceaeee888690241c77c",
                "d8829b9d2f59493ebc2d7798ae09dcf90b201248f2a5636e2f84a913a78d7684",
            ]
        );
        assert_ne!(hashes[1], hashes[12]);
        assert_ne!(hashes[7], hashes[13]);
        for invalid in [
            AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Off,
                frame_wavelet_levels: 1,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Haar,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::DaubechiesFourTap,
                frame_wavelet_levels: super::ACOUSTIC_WAVELET_MAX_LEVELS + 1,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                trajectory_wavelet_levels: 1,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Haar,
                ..AcousticSidecarStudyConfig::default()
            },
            AcousticSidecarStudyConfig {
                trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::DaubechiesFourTap,
                trajectory_wavelet_levels: super::ACOUSTIC_WAVELET_MAX_LEVELS + 1,
                ..AcousticSidecarStudyConfig::default()
            },
        ] {
            assert!(acoustic_sidecar_study_config_sha256(invalid).is_err());
        }
    }

    #[test]
    fn configured_sidecar_study_binds_mode_hash_and_fixed_frame_support() {
        let frame_samples = [0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        let frame = synthetic_feature(0, 0.0, false, false);
        let mut active = || false;

        let mut off = AcousticSidecarStudy::new(AcousticSidecarStudyConfig::default())
            .expect("default-off study");
        let off_observation = off
            .observe_normalized_16khz_frame(&frame_samples, &frame, &mut active)
            .expect("default-off observation");
        assert_eq!(
            off_observation.config(),
            AcousticSidecarStudyConfig::default()
        );
        assert_eq!(off_observation.frame_index(), 0);
        assert_eq!(
            off_observation.schema_version(),
            super::ACOUSTIC_SIDECAR_STUDY_SCHEMA_VERSION
        );
        assert!(off_observation.wavelet().is_none());
        assert!(off_observation.modulation().is_none());
        assert!(off_observation.trajectory_wavelet().is_none());
        assert!(off_observation.scattering().is_none());
        assert_eq!(
            off.configuration_sha256_hex(),
            acoustic_sidecar_study_config_sha256(AcousticSidecarStudyConfig::default())
                .expect("default hash")
        );
        assert_eq!(
            off_observation.configuration_sha256_digest(),
            super::acoustic_sidecar_study_config_digest(AcousticSidecarStudyConfig::default())
                .expect("default digest")
        );

        let config = AcousticSidecarStudyConfig {
            mode: AcousticSidecarStudyMode::DaubechiesFourTapAndModulation,
            frame_wavelet_levels: 4,
            ..AcousticSidecarStudyConfig::default()
        };
        let mut combined = AcousticSidecarStudy::new(config).expect("combined study");
        let combined_observation = combined
            .observe_normalized_16khz_frame(&frame_samples, &frame, &mut active)
            .expect("combined observation");
        assert_eq!(combined_observation.config(), config);
        assert_eq!(
            combined_observation.wavelet().map(|summary| summary.basis),
            Some(AcousticWaveletBasis::DaubechiesFourTap)
        );
        assert!(combined_observation.modulation().is_none());
        assert_eq!(combined.config(), config);
        assert_eq!(
            combined.configuration_sha256_hex(),
            acoustic_sidecar_study_config_sha256(config).expect("combined hash")
        );
        let debug = format!("{combined_observation:?}");
        assert_eq!(
            debug,
            format!(
                "AcousticSidecarStudyObservation {{ schema_version: {:?}, config: {:?}, frame_index: 0, wavelet_available: true, modulation_available: false, trajectory_wavelet_available: false, scattering_available: false, .. }}",
                super::ACOUSTIC_SIDECAR_STUDY_SCHEMA_VERSION,
                config
            )
        );

        assert!(
            AcousticSidecarStudy::new(AcousticSidecarStudyConfig {
                mode: AcousticSidecarStudyMode::Modulation,
                frame_wavelet_levels: 1,
                ..AcousticSidecarStudyConfig::default()
            })
            .is_err()
        );
    }

    #[test]
    fn configured_sidecar_study_rejects_invalid_pcm_before_every_mode_and_state_advance() {
        let valid_samples = [0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        let mut endpoint_samples = valid_samples;
        endpoint_samples[0] = -1.0;
        endpoint_samples[ACOUSTIC_FRAME_SAMPLES - 1] = 1.0;
        let mut configs = AcousticSidecarStudyMode::ALL
            .into_iter()
            .map(|mode| AcousticSidecarStudyConfig {
                mode,
                frame_wavelet_levels: usize::from(mode.wavelet_basis().is_some()),
                ..AcousticSidecarStudyConfig::default()
            })
            .collect::<Vec<_>>();
        configs.extend(
            AcousticTrajectoryWaveletMode::ALL
                .into_iter()
                .filter(|mode| mode.basis().is_some())
                .map(|trajectory_wavelet_mode| AcousticSidecarStudyConfig {
                    trajectory_wavelet_mode,
                    trajectory_wavelet_levels: 1,
                    ..AcousticSidecarStudyConfig::default()
                }),
        );
        configs.extend(
            AcousticScatteringMode::ALL
                .into_iter()
                .filter(|mode| mode.is_enabled())
                .map(|scattering_mode| AcousticSidecarStudyConfig {
                    scattering_mode,
                    ..AcousticSidecarStudyConfig::default()
                }),
        );
        for config in configs {
            for invalid_sample in [
                f32::NAN,
                f32::NEG_INFINITY,
                f32::INFINITY,
                -1.000_1,
                1.000_1,
            ] {
                let mut study = AcousticSidecarStudy::new(config).expect("configured study");
                let mut reference =
                    AcousticSidecarStudy::new(config).expect("reference configured study");
                for frame_index in 0..8 {
                    let frame =
                        synthetic_feature(frame_index, frame_index as f32 / 100.0, false, false);
                    let actual = study
                        .observe_normalized_16khz_frame(&valid_samples, &frame, &mut || false)
                        .expect("warm actual study");
                    let expected = reference
                        .observe_normalized_16khz_frame(&valid_samples, &frame, &mut || false)
                        .expect("warm reference study");
                    assert_eq!(actual, expected);
                }
                let frame = synthetic_feature(8, 0.08, false, false);
                let mut samples = valid_samples;
                samples[17] = invalid_sample;
                let error = study
                    .observe_normalized_16khz_frame(&samples, &frame, &mut || false)
                    .expect_err("invalid normalized PCM");
                assert!(matches!(&error, FwError::InvalidRequest(_)));
                assert!(error.to_string().contains(
                    "acoustic sidecar frame input must contain finite normalized PCM within [-1, 1]"
                ));
                assert_sidecar_study_state_matches(&study, &reference);

                let actual = study
                    .observe_normalized_16khz_frame(&endpoint_samples, &frame, &mut || false)
                    .expect("inclusive normalized endpoints succeed after rejected PCM");
                let expected = reference
                    .observe_normalized_16khz_frame(&endpoint_samples, &frame, &mut || false)
                    .expect("inclusive normalized endpoints succeed in reference study");
                assert_eq!(actual, expected);
                assert_sidecar_study_state_matches(&study, &reference);
            }
        }
    }

    #[test]
    fn configured_sidecar_study_validates_pcm_before_combined_stateful_modes() {
        let config = AcousticSidecarStudyConfig {
            mode: AcousticSidecarStudyMode::HaarAndModulation,
            frame_wavelet_levels: 1,
            trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Haar,
            trajectory_wavelet_levels: 1,
            scattering_mode: AcousticScatteringMode::FirstAndSecondOrder,
        };
        let mut study = AcousticSidecarStudy::new(config).expect("combined study");
        let mut reference = AcousticSidecarStudy::new(config).expect("combined reference study");
        let samples = [0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        for frame_index in 0..ACOUSTIC_TRAJECTORY_HISTORY_FRAMES - 1 {
            let frame =
                synthetic_feature(frame_index, (frame_index % 11) as f32 / 20.0, false, false);
            let actual = study
                .observe_normalized_16khz_frame(&samples, &frame, &mut || false)
                .expect("warm combined study");
            let expected = reference
                .observe_normalized_16khz_frame(&samples, &frame, &mut || false)
                .expect("warm combined reference study");
            assert_eq!(actual, expected);
        }
        let frame_index = ACOUSTIC_TRAJECTORY_HISTORY_FRAMES - 1;
        let frame = synthetic_feature(frame_index, 0.25, false, false);
        let mut invalid_samples = samples;
        invalid_samples[31] = f32::NAN;
        study
            .observe_normalized_16khz_frame(&invalid_samples, &frame, &mut || false)
            .expect_err("invalid PCM before first full-window result");
        assert_sidecar_study_state_matches(&study, &reference);

        let actual = study
            .observe_normalized_16khz_frame(&samples, &frame, &mut || false)
            .expect("combined retry result");
        let expected = reference
            .observe_normalized_16khz_frame(&samples, &frame, &mut || false)
            .expect("combined reference result");
        assert_eq!(actual, expected);
        assert!(actual.modulation().is_some());
        assert!(actual.trajectory_wavelet().is_some());
        assert!(actual.scattering().is_some());
        assert_sidecar_study_state_matches(&study, &reference);
    }

    #[test]
    fn configured_sidecar_study_emits_modulation_after_exact_fixed_window() {
        let config = AcousticSidecarStudyConfig {
            mode: AcousticSidecarStudyMode::HaarAndModulation,
            frame_wavelet_levels: 4,
            ..AcousticSidecarStudyConfig::default()
        };
        let mut study = AcousticSidecarStudy::new(config).expect("combined study");
        let frame_samples = [0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        let mut active = || false;
        let mut final_observation = None;
        for frame_index in 1..=ACOUSTIC_MODULATION_HISTORY_FRAMES {
            let mut frame = synthetic_feature(frame_index, 0.0, false, false);
            let phase = std::f32::consts::TAU * frame_index as f32
                / ACOUSTIC_MODULATION_HISTORY_FRAMES as f32;
            frame.voice.temporal_modulation = 0.5 + 0.3 * (4.0 * phase).sin();
            frame.channel.rms_dbfs = -20.0 + 3.0 * (2.0 * phase).sin();
            frame.channel.muffling_proxy = 0.5 + 0.3 * (8.0 * phase).sin();
            let observation = study
                .observe_normalized_16khz_frame(&frame_samples, &frame, &mut active)
                .expect("configured observation");
            assert_eq!(observation.frame_index(), frame_index);
            assert!(observation.wavelet().is_some());
            if frame_index < ACOUSTIC_MODULATION_HISTORY_FRAMES {
                assert!(observation.modulation().is_none());
            } else {
                final_observation = Some(observation);
            }
        }
        let observation = final_observation.expect("complete configured window");
        assert_eq!(observation.config(), config);
        assert_eq!(
            observation.configuration_sha256_digest(),
            super::acoustic_sidecar_study_config_digest(config).expect("configuration digest")
        );
        let modulation = observation.modulation().expect("modulation summary");
        assert_eq!(modulation.window_start_frame_index, 1);
        assert_eq!(
            modulation.window_end_frame_index,
            ACOUSTIC_MODULATION_HISTORY_FRAMES
        );
        assert!(modulation.voice_available);
        assert!(modulation.channel_level_available);
        assert!(modulation.channel_coloration_available);
    }

    #[test]
    fn daubechies_four_tap_periodic_goldens_freeze_phase_and_sign() {
        let mut input = [0.0_f32; super::ACOUSTIC_WAVELET_MAX_SAMPLES];
        input[0] = 1.0;
        let impulse_zero =
            super::wavelet_pair(AcousticWaveletBasis::DaubechiesFourTap, &input, 4, 0);
        let impulse_one =
            super::wavelet_pair(AcousticWaveletBasis::DaubechiesFourTap, &input, 4, 2);
        assert!((impulse_zero.0 - 0.482_962_9).abs() < 2e-6);
        assert!((impulse_zero.1 - 0.129_409_52).abs() < 2e-6);
        assert!((impulse_one.0 - 0.224_143_86).abs() < 2e-6);
        assert!((impulse_one.1 + 0.836_516_3).abs() < 2e-6);

        input[..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let ramp_zero = super::wavelet_pair(AcousticWaveletBasis::DaubechiesFourTap, &input, 4, 0);
        let ramp_one = super::wavelet_pair(AcousticWaveletBasis::DaubechiesFourTap, &input, 4, 2);
        assert!((ramp_zero.0 - 2.310_789).abs() < 2e-6);
        assert!(ramp_zero.1.abs() < 2e-6);
        assert!((ramp_one.0 - 4.760_278_7).abs() < 2e-6);
        assert!((ramp_one.1 - std::f32::consts::SQRT_2).abs() < 2e-6);

        input[..3].copy_from_slice(&[1.0, 2.0, 3.0]);
        let odd_zero = super::wavelet_pair(AcousticWaveletBasis::DaubechiesFourTap, &input, 3, 0);
        let odd_one = super::wavelet_pair(AcousticWaveletBasis::DaubechiesFourTap, &input, 3, 2);
        assert!((odd_zero.0 - 2.440_198_7).abs() < 2e-6);
        assert!((odd_zero.1 + 0.482_962_9).abs() < 2e-6);
        assert!((odd_one.0 - 3.923_762_6).abs() < 2e-6);
        assert!((odd_one.1 - 1.190_069_7).abs() < 2e-6);
    }

    #[test]
    fn haar_wavelet_places_period_two_and_period_four_energy_at_expected_scales() {
        let alternating = (0..16)
            .map(|index| if index % 2 == 0 { 0.5 } else { -0.5 })
            .collect::<Vec<_>>();
        let period_two = analyze_acoustic_wavelet(
            &alternating,
            AcousticWaveletConfig {
                basis: AcousticWaveletBasis::Haar,
                levels: 1,
            },
            || false,
        )
        .expect("period-two Haar summary");
        assert!(period_two.levels[0].detail_energy_fraction > 0.999_999);
        assert!(period_two.final_approximation_energy_fraction < 1e-6);

        let period_four = (0..16)
            .map(|index| if index % 4 < 2 { 0.5 } else { -0.5 })
            .collect::<Vec<_>>();
        let period_four = analyze_acoustic_wavelet(
            &period_four,
            AcousticWaveletConfig {
                basis: AcousticWaveletBasis::Haar,
                levels: 2,
            },
            || false,
        )
        .expect("period-four Haar summary");
        assert!(period_four.levels[0].detail_energy_fraction < 1e-6);
        assert!(period_four.levels[1].detail_energy_fraction > 0.999_999);
        assert!(period_four.final_approximation_energy_fraction < 1e-6);
    }

    #[test]
    fn orthogonal_wavelet_pairs_preserve_even_width_energy() {
        let mut original = [0.0_f32; super::ACOUSTIC_WAVELET_MAX_SAMPLES];
        for (index, sample) in original[..64].iter_mut().enumerate() {
            let phase = std::f32::consts::TAU * index as f32 / 64.0;
            *sample = 0.37 * phase.sin() + 0.19 * (3.0 * phase).cos();
        }
        for basis in [
            AcousticWaveletBasis::Haar,
            AcousticWaveletBasis::DaubechiesFourTap,
        ] {
            let mut current = original;
            let mut approximation = [0.0_f32; super::ACOUSTIC_WAVELET_MAX_SAMPLES];
            let mut current_len = 64usize;
            for level in 0..4 {
                let input_energy = current[..current_len]
                    .iter()
                    .map(|value| f64::from(*value) * f64::from(*value))
                    .sum::<f64>();
                let mut output_energy = 0.0_f64;
                for output_index in 0..current_len / 2 {
                    let (low, high, _) =
                        super::wavelet_pair(basis, &current, current_len, output_index * 2);
                    approximation[output_index] = low;
                    output_energy +=
                        f64::from(low) * f64::from(low) + f64::from(high) * f64::from(high);
                }
                let relative_error = (output_energy - input_energy).abs() / input_energy;
                assert!(
                    relative_error < 2e-6,
                    "{basis:?} level {level} relative energy error {relative_error}"
                );
                current_len /= 2;
                current[..current_len].copy_from_slice(&approximation[..current_len]);
            }
        }
    }

    #[test]
    fn wavelet_sidecar_is_gain_and_dc_invariant_above_floor_with_fixed_accounting() {
        let quiet_samples = sine_wave(440.0, 0.025, 0.2);
        let loud_samples = sine_wave(440.0, 0.025, 0.8);
        let dc_shifted_samples = quiet_samples
            .iter()
            .map(|sample| sample + 0.3)
            .collect::<Vec<_>>();
        let config = AcousticWaveletConfig {
            basis: AcousticWaveletBasis::DaubechiesFourTap,
            levels: 4,
        };
        let quiet =
            analyze_acoustic_wavelet(&quiet_samples, config, || false).expect("quiet summary");
        let loud = analyze_acoustic_wavelet(&loud_samples, config, || false).expect("loud summary");
        let dc_shifted = analyze_acoustic_wavelet(&dc_shifted_samples, config, || false)
            .expect("DC-shifted summary");
        assert_eq!(quiet.filter_tap_terms, 3_000);
        assert_eq!(quiet.filter_tap_terms, loud.filter_tap_terms);
        assert_eq!(quiet.filter_tap_terms, dc_shifted.filter_tap_terms);
        assert_eq!(quiet.scratch_buffer_payload_bytes, 3 * 400 * 4);
        assert_eq!(
            format!("{quiet:?}"),
            format!(
                "AcousticWaveletSummary {{ basis: {:?}, owner: {:?}, input_samples: {}, valid_level_count: {}, input_was_silent_or_near_constant: {}, filter_tap_terms: {}, maximum_energy_conservation_relative_error: {:?}, scratch_buffer_payload_bytes: {}, .. }}",
                quiet.basis,
                quiet.owner,
                quiet.input_samples,
                quiet.valid_level_count,
                quiet.input_was_silent_or_near_constant,
                quiet.filter_tap_terms,
                quiet.maximum_energy_conservation_relative_error,
                quiet.scratch_buffer_payload_bytes
            )
        );
        for (level_index, ((quiet_level, loud_level), dc_shifted_level)) in quiet
            .levels
            .iter()
            .zip(&loud.levels)
            .zip(&dc_shifted.levels)
            .enumerate()
        {
            for (variant, candidate, flatness_tolerance) in [
                ("gain", loud_level, 2e-6),
                // Adding a non-dyadic DC value re-quantizes the source samples
                // before mean removal. Bound that representational loss by a
                // small multiple of the f32 machine epsilon while retaining
                // the stricter tolerance for exactly representable gain.
                ("DC", dc_shifted_level, 64.0 * f32::EPSILON),
            ] {
                assert!(
                    (quiet_level.detail_energy_fraction - candidate.detail_energy_fraction).abs()
                        < 2e-6
                );
                assert!((quiet_level.detail_log_energy - candidate.detail_log_energy).abs() < 2e-6);
                assert!(
                    (quiet_level.normalized_entropy - candidate.normalized_entropy).abs() < 2e-6
                );
                assert!(
                    (quiet_level.coefficient_flatness - candidate.coefficient_flatness).abs()
                        < flatness_tolerance,
                    "level {level_index} {variant} flatness differs: reference={} candidate={} difference={}",
                    quiet_level.coefficient_flatness,
                    candidate.coefficient_flatness,
                    (quiet_level.coefficient_flatness - candidate.coefficient_flatness).abs()
                );
                assert!((quiet_level.crest_factor - candidate.crest_factor).abs() < 2e-6);
                assert!(
                    (quiet_level.normalized_detail_change - candidate.normalized_detail_change)
                        .abs()
                        < 2e-6
                );
            }
        }
    }

    #[test]
    fn wavelet_sidecar_suppresses_f32_jitter_but_retains_above_floor_variation() {
        let config = AcousticWaveletConfig {
            basis: AcousticWaveletBasis::Haar,
            levels: 4,
        };
        let mut quantization_only = [0.5_f32; 64];
        quantization_only[63] = f32::from_bits(0.5_f32.to_bits() + 1);
        let suppressed = analyze_acoustic_wavelet(&quantization_only, config, || false)
            .expect("quantization-only frame wavelet");
        assert!(suppressed.input_was_silent_or_near_constant);
        assert_eq!(suppressed.valid_level_count, 0);
        assert_eq!(suppressed.filter_tap_terms, 0);
        assert_eq!(suppressed.final_approximation_energy_fraction, 0.0);
        assert_eq!(suppressed.maximum_energy_conservation_relative_error, 0.0);
        assert!(
            suppressed
                .levels
                .iter()
                .all(|level| *level == super::AcousticWaveletLevelSummary::default())
        );

        let at_floor: [f32; 64] = std::array::from_fn(|index| {
            if index % 2 == 0 {
                super::ACOUSTIC_WAVELET_CENTERED_RMS_RELATIVE_FLOOR
            } else {
                -super::ACOUSTIC_WAVELET_CENTERED_RMS_RELATIVE_FLOOR
            }
        });
        let boundary =
            analyze_acoustic_wavelet(&at_floor, config, || false).expect("at-floor frame wavelet");
        assert!(boundary.input_was_silent_or_near_constant);
        assert_eq!(boundary.valid_level_count, 0);

        let delta = 2.0 * super::ACOUSTIC_WAVELET_CENTERED_RMS_RELATIVE_FLOOR;
        let above_floor: [f32; 64] =
            std::array::from_fn(|index| 0.5 + if index % 2 == 0 { delta } else { -delta });
        let retained = analyze_acoustic_wavelet(&above_floor, config, || false)
            .expect("above-floor frame wavelet");
        assert!(!retained.input_was_silent_or_near_constant);
        assert_eq!(retained.valid_level_count, 4);
        assert!(retained.filter_tap_terms > 0);
        assert!(retained.levels[0].detail_energy_fraction > 0.99);
    }

    #[test]
    fn wavelet_tiny_detail_retains_its_local_energy_fraction() {
        let detail = [0.1 * super::PCM_EPSILON, -0.1 * super::PCM_EPSILON];
        let detail_energy = detail
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        let (summary, observed_energy) =
            super::summarize_wavelet_detail(&detail, 2.0 * detail_energy);
        assert_eq!(observed_energy, detail_energy);
        assert!((summary.detail_energy_fraction - 0.5).abs() < f32::EPSILON);
        assert_eq!(summary.detail_log_energy, super::POWER_EPSILON.ln());
        assert_eq!(summary.normalized_entropy, 0.0);
        assert_eq!(summary.coefficient_flatness, 0.0);
        assert_eq!(summary.crest_factor, 0.0);
        assert_eq!(summary.normalized_detail_change, 0.0);
    }

    #[test]
    fn wavelet_sidecar_is_bounded_for_declared_adversarial_signals() {
        let mut impulse = vec![0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        impulse[ACOUSTIC_FRAME_SAMPLES / 2] = 1.0;
        let mut chirp = vec![0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        for (index, sample) in chirp.iter_mut().enumerate() {
            let time = index as f32 / crate::native_engine::mel::SAMPLE_RATE as f32;
            let phase = std::f32::consts::TAU * (80.0 * time + 60_000.0 * time * time);
            *sample = 0.7 * phase.sin();
        }
        let mut noise = vec![0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        let mut state = 0x4d59_5df4_u32;
        for sample in &mut noise {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = 0.8 * ((state >> 8) as f32 / 16_777_215.0 * 2.0 - 1.0);
        }
        let clipped = (0..ACOUSTIC_FRAME_SAMPLES)
            .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let silence = vec![0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        for fixture in [&impulse, &chirp, &noise, &clipped, &silence] {
            for basis in [
                AcousticWaveletBasis::Haar,
                AcousticWaveletBasis::DaubechiesFourTap,
            ] {
                let summary = analyze_acoustic_wavelet(
                    fixture,
                    AcousticWaveletConfig { basis, levels: 4 },
                    || false,
                )
                .expect("bounded fixture summary");
                assert_wavelet_summary_is_bounded(&summary);
            }
        }

        let odd = &chirp[..ACOUSTIC_FRAME_SAMPLES - 1];
        let odd_summary = analyze_acoustic_wavelet(
            odd,
            AcousticWaveletConfig {
                basis: AcousticWaveletBasis::DaubechiesFourTap,
                levels: 4,
            },
            || false,
        )
        .expect("odd-width summary");
        assert_eq!(odd_summary.input_samples, 399);
        assert_wavelet_summary_is_bounded(&odd_summary);
    }

    #[test]
    fn wavelet_sidecar_rejects_invalid_geometry_values_and_levels() {
        let valid = vec![0.25_f32; 16];
        for levels in [0, super::ACOUSTIC_WAVELET_MAX_LEVELS + 1] {
            let error = analyze_acoustic_wavelet(
                &valid,
                AcousticWaveletConfig {
                    basis: AcousticWaveletBasis::Haar,
                    levels,
                },
                || false,
            )
            .expect_err("invalid level count");
            assert!(error.to_string().contains("wavelet levels"));
        }
        let error = analyze_acoustic_wavelet(
            &[0.0; 7],
            AcousticWaveletConfig {
                basis: AcousticWaveletBasis::Haar,
                levels: 3,
            },
            || false,
        )
        .expect_err("short input");
        assert!(error.to_string().contains("8..=400"));
        let error = analyze_acoustic_wavelet(
            &[0.0; 16],
            AcousticWaveletConfig {
                basis: AcousticWaveletBasis::DaubechiesFourTap,
                levels: 4,
            },
            || false,
        )
        .expect_err("D4 support collision");
        assert!(error.to_string().contains("32..=400"));

        let too_long = vec![0.0_f32; ACOUSTIC_FRAME_SAMPLES + 1];
        assert!(
            analyze_acoustic_wavelet(&too_long, AcousticWaveletConfig::default(), || false)
                .is_err()
        );
        for invalid in [f32::NAN, f32::INFINITY, 1.01, -1.01] {
            let mut input = valid.clone();
            input[3] = invalid;
            let error = analyze_acoustic_wavelet(
                &input,
                AcousticWaveletConfig {
                    basis: AcousticWaveletBasis::Haar,
                    levels: 1,
                },
                || false,
            )
            .expect_err("invalid sample");
            assert!(error.to_string().contains("finite normalized PCM"));
        }
    }

    #[test]
    fn wavelet_sidecar_cancellation_stops_between_complete_levels() {
        let input = sine_wave(440.0, 0.025, 0.5);
        let mut checks = 0usize;
        let error = analyze_acoustic_wavelet(&input, AcousticWaveletConfig::default(), || {
            checks += 1;
            checks == 3
        })
        .expect_err("cancel before level one");
        assert!(matches!(error, FwError::Cancelled(_)));
        assert!(error.to_string().contains("before level 1"));
        assert_eq!(checks, 3);
    }

    #[test]
    fn modulation_sidecar_resolves_declared_voice_and_channel_frequencies() {
        assert_eq!(
            ACOUSTIC_MODULATION_FREQUENCY_HZ,
            [1.5625, 3.125, 6.25, 12.5]
        );
        let mut sidecar = AcousticModulationSidecar::new();
        let mut is_cancelled = || false;
        let mut first_window = None;
        let mut final_summary = None;
        for frame_index in 0..=ACOUSTIC_MODULATION_HISTORY_FRAMES {
            let mut frame = synthetic_feature(frame_index, 0.0, false, false);
            let voice_phase = std::f32::consts::TAU * 4.0 * frame_index as f32
                / ACOUSTIC_MODULATION_HISTORY_FRAMES as f32;
            let level_phase = std::f32::consts::TAU * 2.0 * frame_index as f32
                / ACOUSTIC_MODULATION_HISTORY_FRAMES as f32;
            let color_phase = std::f32::consts::TAU * 8.0 * frame_index as f32
                / ACOUSTIC_MODULATION_HISTORY_FRAMES as f32;
            frame.voice.temporal_modulation = 0.5 + 0.4 * voice_phase.sin();
            frame.channel.rms_dbfs = -20.0 + 4.0 * level_phase.sin();
            frame.channel.muffling_proxy = 0.5 + 0.4 * color_phase.sin();
            let summary = sidecar
                .push(&frame, &mut is_cancelled)
                .expect("modulation push");
            if frame_index + 1 < ACOUSTIC_MODULATION_HISTORY_FRAMES {
                assert!(summary.is_none());
            } else if frame_index + 1 == ACOUSTIC_MODULATION_HISTORY_FRAMES {
                first_window = summary;
            } else {
                final_summary = summary;
            }
        }
        let historyless = first_window.expect("first complete channel window");
        assert_eq!(historyless.window_start_frame_index, 0);
        assert_eq!(historyless.window_end_frame_index, 63);
        assert!(historyless.voice_available);
        assert_eq!(historyless.voice_valid_frames, 63);
        assert!(historyless.channel_level_available);
        assert!(historyless.channel_coloration_available);
        assert!(historyless.voice_normalized_power[2] > 0.999);
        for (frequency, power) in historyless.voice_normalized_power.iter().enumerate() {
            if frequency != 2 {
                assert!(
                    *power < 0.01,
                    "unexpected voice leakage at {frequency}: {power}"
                );
            }
        }
        assert_eq!(
            historyless.projection_sample_frequency_visits,
            (63 + 64 + 64) * 4 * 2
        );

        let summary = final_summary.expect("first complete voice window");
        assert_eq!(summary.window_start_frame_index, 1);
        assert_eq!(summary.window_end_frame_index, 64);
        assert!(summary.voice_available);
        assert!(summary.channel_level_available);
        assert!(summary.channel_coloration_available);
        assert_eq!(summary.voice_valid_frames, 64);
        assert_eq!(summary.channel_valid_frames, 64);
        assert!(summary.voice_normalized_power[2] > 0.999);
        assert!(summary.channel_level_normalized_power[1] > 0.999);
        assert!(summary.channel_coloration_normalized_power[3] > 0.999);
        for (frequency, power) in summary.voice_normalized_power.iter().enumerate() {
            if frequency != 2 {
                assert!(
                    *power < 0.01,
                    "unexpected voice leakage at {frequency}: {power}"
                );
            }
        }
        for (frequency, power) in summary.channel_level_normalized_power.iter().enumerate() {
            if frequency != 1 {
                assert!(
                    *power < 0.01,
                    "unexpected level leakage at {frequency}: {power}"
                );
            }
        }
        for (frequency, power) in summary
            .channel_coloration_normalized_power
            .iter()
            .enumerate()
        {
            if frequency != 3 {
                assert!(
                    *power < 0.01,
                    "unexpected color leakage at {frequency}: {power}"
                );
            }
        }
        assert_eq!(summary.voice_owner, AcousticSidecarFeatureOwner::Voice);
        assert_eq!(
            summary.channel_level_owner,
            AcousticSidecarFeatureOwner::Channel
        );
        assert_eq!(
            summary.channel_coloration_owner,
            AcousticSidecarFeatureOwner::Channel
        );
        assert_eq!(summary.projection_sample_frequency_visits, 3 * 4 * 64 * 2);
        assert_eq!(
            summary.retained_state_bytes_on_target,
            sidecar.retained_state_bytes_on_target()
        );
        assert_eq!(summary.cached_twiddle_payload_bytes, 4 * 64 * 2 * 8);
        assert_eq!(
            format!("{summary:?}"),
            format!(
                "AcousticModulationSummary {{ window_start_frame_index: {}, window_end_frame_index: {}, voice_available: {}, channel_level_available: {}, channel_coloration_available: {}, voice_valid_frames: {}, channel_valid_frames: {}, projection_sample_frequency_visits: {}, retained_state_bytes_on_target: {}, cached_twiddle_payload_bytes: {}, .. }}",
                summary.window_start_frame_index,
                summary.window_end_frame_index,
                summary.voice_available,
                summary.channel_level_available,
                summary.channel_coloration_available,
                summary.voice_valid_frames,
                summary.channel_valid_frames,
                summary.projection_sample_frequency_visits,
                summary.retained_state_bytes_on_target,
                summary.cached_twiddle_payload_bytes
            )
        );
    }

    #[test]
    fn modulation_sidecar_fails_closed_without_mutating_on_cancel_or_gap() {
        let mut sidecar = AcousticModulationSidecar::new();
        let frame_zero = synthetic_feature(0, 0.0, false, false);
        let mut cancelled = || true;
        let error = sidecar
            .push(&frame_zero, &mut cancelled)
            .expect_err("cancelled first frame");
        assert!(matches!(error, FwError::Cancelled(_)));

        let mut active = || false;
        assert!(
            sidecar
                .push(&frame_zero, &mut active)
                .expect("first frame after cancellation")
                .is_none()
        );
        let frame_two = synthetic_feature(2, 0.0, false, false);
        let error = sidecar
            .push(&frame_two, &mut active)
            .expect_err("non-contiguous frame");
        assert!(error.to_string().contains("expected frame 1, got 2"));
        let frame_one = synthetic_feature(1, 0.0, false, false);
        assert!(
            sidecar
                .push(&frame_one, &mut active)
                .expect("state preserved after gap")
                .is_none()
        );
    }

    #[test]
    fn modulation_sidecar_cancellation_between_frequencies_is_atomic() {
        let mut sidecar = AcousticModulationSidecar::new();
        let mut active = || false;
        for frame_index in 1..ACOUSTIC_MODULATION_HISTORY_FRAMES {
            let mut frame = synthetic_feature(frame_index, 0.0, false, false);
            frame.voice.temporal_modulation = 0.5 + 0.2 * (frame_index as f32).sin();
            assert!(sidecar.push(&frame, &mut active).expect("warmup").is_none());
        }
        let mut final_frame = synthetic_feature(64, 0.0, false, false);
        final_frame.voice.temporal_modulation = 0.5 + 0.2 * 64.0_f32.sin();
        final_frame.channel.rms_dbfs = -20.0 + 64.0_f32.sin();
        final_frame.channel.muffling_proxy = 0.5 + 0.2 * 64.0_f32.cos();
        let mut checks = 0usize;
        let error = sidecar
            .push(&final_frame, &mut || {
                checks += 1;
                checks == 4
            })
            .expect_err("cancel between voice frequencies");
        assert!(matches!(error, FwError::Cancelled(_)));
        assert!(error.to_string().contains("voice frequency 1"));
        assert_eq!(checks, 4);

        let summary = sidecar
            .push(&final_frame, &mut active)
            .expect("retry complete frame")
            .expect("complete window after atomic retry");
        assert_eq!(summary.window_start_frame_index, 1);
        assert_eq!(summary.window_end_frame_index, 64);
    }

    #[test]
    fn modulation_sidecar_cancellation_between_families_is_atomic() {
        let mut sidecar = AcousticModulationSidecar::new();
        let mut active = || false;
        for frame_index in 1..ACOUSTIC_MODULATION_HISTORY_FRAMES {
            let mut frame = synthetic_feature(frame_index, 0.0, false, false);
            frame.voice.temporal_modulation = 0.5 + 0.2 * (frame_index as f32).sin();
            assert!(sidecar.push(&frame, &mut active).expect("warmup").is_none());
        }
        let mut final_frame = synthetic_feature(64, 0.0, false, false);
        final_frame.voice.temporal_modulation = 0.5 + 0.2 * 64.0_f32.sin();
        final_frame.channel.rms_dbfs = -20.0 + 64.0_f32.sin();
        final_frame.channel.muffling_proxy = 0.5 + 0.2 * 64.0_f32.cos();
        let mut checks = 0usize;
        let error = sidecar
            .push(&final_frame, &mut || {
                checks += 1;
                checks == 7
            })
            .expect_err("cancel between voice and channel-level families");
        assert!(matches!(error, FwError::Cancelled(_)));
        assert!(error.to_string().contains("channel-level projection"));
        assert_eq!(checks, 7);

        let summary = sidecar
            .push(&final_frame, &mut active)
            .expect("retry complete frame")
            .expect("complete window after atomic retry");
        assert_eq!(summary.window_start_frame_index, 1);
        assert_eq!(summary.window_end_frame_index, 64);
    }

    #[test]
    fn modulation_sidecar_marks_insufficient_evidence_unavailable() {
        let mut sidecar = AcousticModulationSidecar::new();
        let mut active = || false;
        let mut summary = None;
        for frame_index in 0..ACOUSTIC_MODULATION_HISTORY_FRAMES {
            let low_energy = frame_index < 40;
            let mut frame = synthetic_feature(frame_index, 0.0, low_energy, false);
            frame.voice.temporal_modulation = if low_energy {
                0.0
            } else {
                0.5 + 0.2 * (frame_index as f32).sin()
            };
            summary = sidecar.push(&frame, &mut active).expect("push");
        }
        let summary = summary.expect("complete unavailable window");
        assert_eq!(summary.voice_valid_frames, 24);
        assert_eq!(summary.channel_valid_frames, 24);
        assert!(!summary.voice_available);
        assert!(!summary.channel_level_available);
        assert!(!summary.channel_coloration_available);
        assert_eq!(summary.voice_normalized_power, [0.0; 4]);
        assert_eq!(summary.channel_level_normalized_power, [0.0; 4]);
        assert_eq!(summary.channel_coloration_normalized_power, [0.0; 4]);
        assert_eq!(summary.projection_sample_frequency_visits, 0);
    }

    #[test]
    fn modulation_spectrum_enforces_exact_minimum_valid_boundary() {
        let mut values = [0.0_f32; ACOUSTIC_MODULATION_HISTORY_FRAMES];
        for (offset, value) in values.iter_mut().enumerate() {
            let phase =
                std::f32::consts::TAU * offset as f32 / ACOUSTIC_MODULATION_HISTORY_FRAMES as f32;
            *value = (4.0 * phase).sin();
        }
        let mut valid = [false; ACOUSTIC_MODULATION_HISTORY_FRAMES];
        valid[..super::ACOUSTIC_MODULATION_MIN_VALID_FRAMES - 1].fill(true);
        let mut active = || false;
        let below = super::modulation_spectrum(&values, &valid, 0, "boundary", &mut active)
            .expect("below-boundary spectrum");
        assert!(!below.0);
        assert_eq!(below.1, super::ACOUSTIC_MODULATION_MIN_VALID_FRAMES - 1);
        assert_eq!(below.2, [0.0; 4]);
        assert_eq!(below.3, 0);

        valid[super::ACOUSTIC_MODULATION_MIN_VALID_FRAMES - 1] = true;
        let boundary = super::modulation_spectrum(&values, &valid, 0, "boundary", &mut active)
            .expect("at-boundary spectrum");
        assert!(boundary.0);
        assert_eq!(boundary.1, super::ACOUSTIC_MODULATION_MIN_VALID_FRAMES);
        assert!(boundary.2[2] > 0.999);
        assert_eq!(
            boundary.3,
            4 * super::ACOUSTIC_MODULATION_MIN_VALID_FRAMES * 2
        );
    }

    #[test]
    fn modulation_sidecar_regresses_over_valid_samples_without_zero_fill() {
        let mut sidecar = AcousticModulationSidecar::new();
        let mut active = || false;
        let mut summary = None;
        for frame_index in 1..=ACOUSTIC_MODULATION_HISTORY_FRAMES {
            let low_energy = frame_index % 4 == 0;
            let mut frame = synthetic_feature(frame_index, 0.0, low_energy, false);
            let voice_phase = std::f32::consts::TAU * 4.0 * frame_index as f32
                / ACOUSTIC_MODULATION_HISTORY_FRAMES as f32;
            let level_phase = std::f32::consts::TAU * 2.0 * frame_index as f32
                / ACOUSTIC_MODULATION_HISTORY_FRAMES as f32;
            let color_phase = std::f32::consts::TAU * 8.0 * frame_index as f32
                / ACOUSTIC_MODULATION_HISTORY_FRAMES as f32;
            frame.voice.temporal_modulation = 0.5 + 0.4 * voice_phase.sin();
            if !low_energy {
                frame.channel.rms_dbfs = -20.0 + 4.0 * level_phase.sin();
            }
            frame.channel.muffling_proxy = 0.5 + 0.4 * color_phase.sin();
            summary = sidecar.push(&frame, &mut active).expect("masked push");
        }
        let summary = summary.expect("masked complete window");
        assert_eq!(summary.voice_valid_frames, 48);
        assert_eq!(summary.channel_valid_frames, 48);
        assert!(summary.voice_available);
        assert!(summary.channel_level_available);
        assert!(summary.channel_coloration_available);
        assert!(summary.voice_normalized_power[2] > 0.999);
        assert!(summary.channel_level_normalized_power[1] > 0.999);
        assert!(summary.channel_coloration_normalized_power[3] > 0.999);
        assert_eq!(summary.projection_sample_frequency_visits, 3 * 4 * 48 * 2);
    }

    #[test]
    fn modulation_sidecar_marks_all_valid_constant_trajectories_unavailable() {
        let mut sidecar = AcousticModulationSidecar::new();
        let mut active = || false;
        let mut summary = None;
        for frame_index in 1..=ACOUSTIC_MODULATION_HISTORY_FRAMES {
            let mut frame = synthetic_feature(frame_index, 0.0, false, false);
            frame.voice.temporal_modulation = 0.25;
            frame.channel.rms_dbfs = -20.0;
            frame.channel.muffling_proxy = 0.4;
            summary = sidecar.push(&frame, &mut active).expect("constant push");
        }
        let summary = summary.expect("constant complete window");
        assert_eq!(summary.voice_valid_frames, 64);
        assert_eq!(summary.channel_valid_frames, 64);
        assert!(!summary.voice_available);
        assert!(!summary.channel_level_available);
        assert!(!summary.channel_coloration_available);
        assert_eq!(summary.projection_sample_frequency_visits, 0);
    }

    #[test]
    fn modulation_suppresses_f32_jitter_but_retains_above_floor_variation() {
        let valid = [true; ACOUSTIC_MODULATION_HISTORY_FRAMES];
        let mut active = || false;
        for mean in [0.5_f32, 200.0] {
            let next = f32::from_bits(mean.to_bits() + 1);
            let quantization_only: [f32; ACOUSTIC_MODULATION_HISTORY_FRAMES] =
                std::array::from_fn(|index| if index % 16 < 8 { next } else { mean });
            let suppressed = super::modulation_spectrum(
                &quantization_only,
                &valid,
                0,
                "quantization-only",
                &mut active,
            )
            .expect("quantization-only modulation");
            assert!(!suppressed.0);
            assert_eq!(suppressed.1, ACOUSTIC_MODULATION_HISTORY_FRAMES);
            assert_eq!(suppressed.2, [0.0; 4]);
            assert_eq!(suppressed.3, 0);

            let at_floor: [f32; ACOUSTIC_MODULATION_HISTORY_FRAMES] =
                std::array::from_fn(|index| {
                    if index % 2 == 0 {
                        super::ACOUSTIC_MODULATION_CENTERED_RMS_RELATIVE_FLOOR
                    } else {
                        -super::ACOUSTIC_MODULATION_CENTERED_RMS_RELATIVE_FLOOR
                    }
                });
            let boundary =
                super::modulation_spectrum(&at_floor, &valid, 0, "at-floor", &mut active)
                    .expect("at-floor modulation");
            assert!(!boundary.0);
            assert_eq!(boundary.1, ACOUSTIC_MODULATION_HISTORY_FRAMES);
            assert_eq!(boundary.2, [0.0; 4]);
            assert_eq!(boundary.3, 0);

            let delta =
                2.0 * super::ACOUSTIC_MODULATION_CENTERED_RMS_RELATIVE_FLOOR * mean.abs().max(1.0);
            let above_floor: [f32; ACOUSTIC_MODULATION_HISTORY_FRAMES] =
                std::array::from_fn(|index| mean + if index % 16 < 8 { delta } else { -delta });
            let retained =
                super::modulation_spectrum(&above_floor, &valid, 0, "above-floor", &mut active)
                    .expect("above-floor modulation");
            assert!(retained.0);
            assert_eq!(retained.1, ACOUSTIC_MODULATION_HISTORY_FRAMES);
            assert!(retained.2.iter().all(|power| power.is_finite()));
            assert!(retained.2[2] > 0.75);
            assert_eq!(retained.3, 4 * ACOUSTIC_MODULATION_HISTORY_FRAMES * 2);
        }
    }

    #[test]
    fn modulation_frequency_power_is_circular_shift_tolerant() {
        let mut values = [0.0_f32; ACOUSTIC_MODULATION_HISTORY_FRAMES];
        for (offset, value) in values.iter_mut().enumerate() {
            let phase =
                std::f32::consts::TAU * offset as f32 / ACOUSTIC_MODULATION_HISTORY_FRAMES as f32;
            *value = 0.6 * (3.0 * phase).sin() + 0.3 * (7.0 * phase).cos();
        }
        values[11] += 0.2;
        let mut shifted = [0.0_f32; ACOUSTIC_MODULATION_HISTORY_FRAMES];
        for (offset, value) in values.into_iter().enumerate() {
            shifted[(offset + 7) % ACOUSTIC_MODULATION_HISTORY_FRAMES] = value;
        }
        let valid = [true; ACOUSTIC_MODULATION_HISTORY_FRAMES];
        let mut active = || false;
        let original = super::modulation_spectrum(&values, &valid, 0, "test", &mut active)
            .expect("original spectrum");
        let shifted = super::modulation_spectrum(&shifted, &valid, 0, "test", &mut active)
            .expect("shifted spectrum");
        assert!(original.0);
        assert!(shifted.0);
        assert_eq!(original.1, 64);
        assert_eq!(original.3, 4 * 64 * 2);
        for (left, right) in original.2.into_iter().zip(shifted.2) {
            assert!((left - right).abs() < 2e-6, "{left} versus {right}");
        }
    }

    #[test]
    fn trajectory_wavelets_freeze_masked_levels_accounting_and_affine_invariance() {
        let values = trajectory_fixture_values();
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let affine = values.map(|family| family.map(|value| 0.25 + 0.5 * value));
        let mut active = || false;
        for (basis, expected_filter_terms, expected_validity_visits, expected_support) in [
            (AcousticWaveletBasis::Haar, 4_600, 2_300, [63, 61, 57, 49]),
            (
                AcousticWaveletBasis::DaubechiesFourTap,
                7_120,
                3_560,
                [61, 55, 43, 19],
            ),
        ] {
            let summary = super::analyze_acoustic_trajectory_wavelet(
                trajectory_window(&values, &valid, 0, 11, 74),
                basis,
                4,
                &mut active,
            )
            .expect("trajectory wavelet summary");
            let affine_summary = super::analyze_acoustic_trajectory_wavelet(
                trajectory_window(&affine, &valid, 0, 11, 74),
                basis,
                4,
                &mut active,
            )
            .expect("affine trajectory wavelet summary");
            assert_eq!(summary.basis, basis);
            assert_eq!(summary.requested_levels, 4);
            assert_eq!(summary.window_start_frame_index, 11);
            assert_eq!(summary.window_end_frame_index, 74);
            assert_eq!(summary.filter_tap_terms, expected_filter_terms);
            assert_eq!(summary.validity_sample_visits, expected_validity_visits);
            assert_eq!(summary.scratch_buffer_payload_bytes, 3 * (64 * 4 + 64));
            assert_trajectory_wavelet_summary_is_bounded(&summary);
            for (family, affine_family) in summary.families.iter().zip(affine_summary.families) {
                assert_eq!(family.input_valid_frames, 64);
                assert!(!family.input_was_constant_or_near_constant);
                assert_eq!(family.valid_level_count, 4);
                assert_eq!(
                    family.levels.map(|level| level.valid_coefficients),
                    expected_support
                );
                for (level, affine_level) in family.levels.iter().zip(affine_family.levels) {
                    assert_eq!(level.available, affine_level.available);
                    assert_eq!(level.valid_coefficients, affine_level.valid_coefficients);
                    assert_eq!(
                        level.normalized_entropy_available,
                        affine_level.normalized_entropy_available
                    );
                    assert_eq!(
                        level.adjacent_valid_pairs,
                        affine_level.adjacent_valid_pairs
                    );
                    assert_eq!(
                        level.normalized_detail_change_available,
                        affine_level.normalized_detail_change_available
                    );
                    assert!(
                        (level.mean_absolute_detail - affine_level.mean_absolute_detail).abs()
                            < 2e-5
                    );
                    assert!((level.detail_rms - affine_level.detail_rms).abs() < 2e-5);
                    assert!(
                        (level.normalized_entropy - affine_level.normalized_entropy).abs() < 2e-5
                    );
                    assert!(
                        (level.normalized_detail_change - affine_level.normalized_detail_change)
                            .abs()
                            < 2e-5
                    );
                }
            }
            assert_eq!(
                format!("{summary:?}"),
                format!(
                    "AcousticTrajectoryWaveletSummary {{ basis: {:?}, requested_levels: 4, window_start_frame_index: 11, window_end_frame_index: 74, available_family_count: 5, filter_tap_terms: {expected_filter_terms}, validity_sample_visits: {expected_validity_visits}, scratch_buffer_payload_bytes: {}, .. }}",
                    basis, summary.scratch_buffer_payload_bytes
                )
            );
        }
    }

    #[test]
    fn trajectory_wavelets_match_scalar_full_valid_haar_and_d4_references() {
        let values = trajectory_fixture_values();
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut active = || false;
        for basis in [
            AcousticWaveletBasis::Haar,
            AcousticWaveletBasis::DaubechiesFourTap,
        ] {
            let actual = super::analyze_acoustic_trajectory_wavelet(
                trajectory_window(&values, &valid, 0, 0, 63),
                basis,
                4,
                &mut active,
            )
            .expect("full-valid differential stationary wavelet");
            let expected = scalar_reference_full_valid_trajectory_wavelet(&values[0], basis);
            for (actual, expected) in actual.families[0].levels.iter().zip(expected) {
                assert_eq!(actual.available, expected.available);
                assert_eq!(actual.valid_coefficients, expected.valid_coefficients);
                assert_eq!(actual.adjacent_valid_pairs, expected.adjacent_valid_pairs);
                assert_eq!(
                    actual.normalized_entropy_available,
                    expected.normalized_entropy_available
                );
                assert_eq!(
                    actual.normalized_detail_change_available,
                    expected.normalized_detail_change_available
                );
                assert!((actual.mean_absolute_detail - expected.mean_absolute_detail).abs() < 2e-6);
                assert!((actual.detail_rms - expected.detail_rms).abs() < 2e-6);
                assert!((actual.normalized_entropy - expected.normalized_entropy).abs() < 2e-6);
                assert!(
                    (actual.normalized_detail_change - expected.normalized_detail_change).abs()
                        < 2e-6
                );
            }
        }
    }

    #[test]
    fn stationary_d4_boundary_impulse_matches_closed_form_all_level_goldens() {
        let mut values = [[0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        for family in &mut values {
            family[0] = 1.0;
        }
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut active = || false;
        let summary = super::analyze_acoustic_trajectory_wavelet(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticWaveletBasis::DaubechiesFourTap,
            4,
            &mut active,
        )
        .expect("closed-form stationary D4 impulse");
        let expected_valid_coefficients = [61, 55, 43, 19];
        let expected_mean_absolute = [
            0.002_138_238_3,
            0.001_145_346_9,
            0.000_707_530_3,
            0.000_773_345_7,
        ];
        let expected_rms = [0.016_700_175, 0.008_494_12, 0.004_639_586_4, 0.003_370_936];
        let expected_normalized_change = [0.130_170_82, 0.137_337_01, 0.156_129_5, 0.242_161_05];
        for (level_index, level) in summary.families[0].levels.iter().enumerate() {
            assert!(level.available);
            assert_eq!(
                level.valid_coefficients,
                expected_valid_coefficients[level_index]
            );
            assert_eq!(
                level.adjacent_valid_pairs,
                expected_valid_coefficients[level_index] - 1
            );
            assert!(
                (level.mean_absolute_detail - expected_mean_absolute[level_index]).abs() < 1e-7
            );
            assert!((level.detail_rms - expected_rms[level_index]).abs() < 1e-7);
            assert!(level.normalized_entropy_available);
            assert!(level.normalized_entropy < 1e-6);
            assert!(level.normalized_detail_change_available);
            assert!(
                (level.normalized_detail_change - expected_normalized_change[level_index]).abs()
                    < 1e-7
            );
        }
    }

    #[test]
    fn trajectory_detail_change_distinguishes_missing_adjacency_from_measured_zero() {
        let mut detail = [0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        let mut valid = [false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        detail[0] = 1.0;
        detail[2] = 1.0;
        valid[0] = true;
        valid[2] = true;
        let separated = super::summarize_masked_trajectory_detail(&detail, &valid, 4);
        assert!(separated.available);
        assert_eq!(separated.valid_coefficients, 2);
        assert_eq!(separated.adjacent_valid_pairs, 0);
        assert!(!separated.normalized_detail_change_available);
        assert_eq!(separated.normalized_detail_change, 0.0);

        detail[1] = 1.0;
        valid[1] = true;
        let adjacent = super::summarize_masked_trajectory_detail(&detail, &valid, 4);
        assert!(adjacent.available);
        assert_eq!(adjacent.valid_coefficients, 3);
        assert_eq!(adjacent.adjacent_valid_pairs, 2);
        assert!(adjacent.normalized_detail_change_available);
        assert_eq!(adjacent.normalized_detail_change, 0.0);
    }

    #[test]
    fn trajectory_detail_shape_floor_separates_roundoff_from_measured_structure() {
        let floor = super::ACOUSTIC_TRAJECTORY_DETAIL_RMS_FLOOR;
        let mut detail = [0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        let mut valid = [false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
        valid[..4].fill(true);
        detail[..4].copy_from_slice(&[0.5 * floor, -0.5 * floor, 0.5 * floor, -0.5 * floor]);
        let below = super::summarize_masked_trajectory_detail(&detail, &valid, 4);
        assert!(below.available);
        assert!(below.detail_rms > 0.0);
        assert!(!below.normalized_entropy_available);
        assert_eq!(below.normalized_entropy, 0.0);
        assert!(!below.normalized_detail_change_available);
        assert_eq!(below.normalized_detail_change, 0.0);

        detail[..4].copy_from_slice(&[floor, -floor, floor, -floor]);
        let at_floor = super::summarize_masked_trajectory_detail(&detail, &valid, 4);
        assert!(!at_floor.normalized_entropy_available);
        assert_eq!(at_floor.normalized_entropy, 0.0);
        assert!(!at_floor.normalized_detail_change_available);
        assert_eq!(at_floor.normalized_detail_change, 0.0);

        detail[..4].copy_from_slice(&[2.0 * floor, -2.0 * floor, 2.0 * floor, -2.0 * floor]);
        let above = super::summarize_masked_trajectory_detail(&detail, &valid, 4);
        assert!(above.normalized_entropy_available);
        assert!((above.normalized_entropy - 1.0).abs() < f32::EPSILON);
        assert!(above.normalized_detail_change_available);
        assert!(above.normalized_detail_change > 0.0);
    }

    #[test]
    fn trajectory_wavelets_enforce_minimum_support_without_zero_imputation() {
        let values = trajectory_fixture_values();
        let mut valid =
            [[false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        for family_valid in &mut valid {
            family_valid[..super::ACOUSTIC_TRAJECTORY_MIN_VALID_FRAMES - 1].fill(true);
        }
        let mut active = || false;
        let below = super::analyze_acoustic_trajectory_wavelet(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticWaveletBasis::Haar,
            4,
            &mut active,
        )
        .expect("below-boundary masked stationary wavelet");
        assert!(
            below
                .families
                .iter()
                .all(|family| family.input_valid_frames == 31 && family.valid_level_count == 0)
        );
        assert_eq!(below.filter_tap_terms, 0);
        assert_eq!(below.validity_sample_visits, 0);

        for family_valid in &mut valid {
            family_valid[super::ACOUSTIC_TRAJECTORY_MIN_VALID_FRAMES - 1] = true;
        }
        let boundary = super::analyze_acoustic_trajectory_wavelet(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticWaveletBasis::Haar,
            4,
            &mut active,
        )
        .expect("at-boundary masked stationary wavelet");
        assert!(boundary.families.iter().all(|family| {
            family.input_valid_frames == 32
                && family.valid_level_count == 4
                && family.levels.map(|level| level.valid_coefficients) == [31, 29, 25, 17]
        }));

        let mut sparse_valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        for family_valid in &mut sparse_valid {
            family_valid[48..].fill(false);
        }
        let mut changed_invalid_values = values;
        for family_index in 0..super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT {
            for offset in 0..ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
                if !sparse_valid[family_index][offset] {
                    changed_invalid_values[family_index][offset] = if offset % 2 == 0 {
                        f32::NAN
                    } else {
                        f32::INFINITY
                    };
                }
            }
        }
        let masked = super::analyze_acoustic_trajectory_wavelet(
            trajectory_window(&values, &sparse_valid, 0, 0, 63),
            AcousticWaveletBasis::DaubechiesFourTap,
            4,
            &mut active,
        )
        .expect("masked trajectory wavelet");
        let changed = super::analyze_acoustic_trajectory_wavelet(
            trajectory_window(&changed_invalid_values, &sparse_valid, 0, 0, 63),
            AcousticWaveletBasis::DaubechiesFourTap,
            4,
            &mut active,
        )
        .expect("masked trajectory wavelet with changed invalid values");
        assert!(masked.filter_tap_terms > 0);
        assert!(
            masked
                .families
                .iter()
                .all(|family| family.levels[0].valid_coefficients == 45)
        );
        assert_eq!(masked, changed);
    }

    #[test]
    fn stationary_trajectory_wavelet_is_stable_under_one_frame_translation() {
        let first = std::array::from_fn(|_| {
            std::array::from_fn(|offset| if offset % 2 == 0 { 0.1 } else { 0.9 })
        });
        let shifted = std::array::from_fn(|_| {
            std::array::from_fn(|offset| if offset % 2 == 0 { 0.9 } else { 0.1 })
        });
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut active = || false;
        for basis in [
            AcousticWaveletBasis::Haar,
            AcousticWaveletBasis::DaubechiesFourTap,
        ] {
            let first = super::analyze_acoustic_trajectory_wavelet(
                trajectory_window(&first, &valid, 0, 0, 63),
                basis,
                4,
                &mut active,
            )
            .expect("first stationary trajectory window");
            let shifted = super::analyze_acoustic_trajectory_wavelet(
                trajectory_window(&shifted, &valid, 0, 1, 64),
                basis,
                4,
                &mut active,
            )
            .expect("translated stationary trajectory window");
            if basis == AcousticWaveletBasis::Haar {
                let level = first.families[0].levels[0];
                assert_eq!(level.valid_coefficients, 63);
                assert!((level.mean_absolute_detail - 0.176_776_69).abs() < 2e-6);
                assert!((level.detail_rms - 0.176_776_69).abs() < 2e-6);
                assert!(level.normalized_entropy_available);
                assert!((level.normalized_entropy - 1.0).abs() < 2e-6);
                assert!(level.normalized_detail_change_available);
                assert!((level.normalized_detail_change - 2.0).abs() < 2e-6);
            }
            for (left, right) in first.families[0]
                .levels
                .iter()
                .zip(shifted.families[0].levels)
            {
                assert_eq!(left.valid_coefficients, right.valid_coefficients);
                assert!((left.mean_absolute_detail - right.mean_absolute_detail).abs() < 2e-6);
                assert!((left.detail_rms - right.detail_rms).abs() < 2e-6);
                assert!((left.normalized_entropy - right.normalized_entropy).abs() < 2e-6);
                assert!(
                    (left.normalized_detail_change - right.normalized_detail_change).abs() < 2e-6
                );
            }
        }
    }

    #[test]
    fn stationary_trajectory_wavelet_translates_nonzero_interior_detail() {
        let mut first = [[0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut shifted = first;
        for family_index in 0..super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT {
            let amplitude = 0.5 + family_index as f32 * 0.05;
            first[family_index][16] = amplitude;
            shifted[family_index][17] = amplitude;
        }
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        for basis in [
            AcousticWaveletBasis::Haar,
            AcousticWaveletBasis::DaubechiesFourTap,
        ] {
            let first = super::analyze_acoustic_trajectory_wavelet(
                trajectory_window(&first, &valid, 0, 0, 63),
                basis,
                2,
                &mut || false,
            )
            .expect("first impulse trajectory window");
            let shifted = super::analyze_acoustic_trajectory_wavelet(
                trajectory_window(&shifted, &valid, 0, 1, 64),
                basis,
                2,
                &mut || false,
            )
            .expect("translated impulse trajectory window");
            assert_eq!(first.filter_tap_terms, shifted.filter_tap_terms);
            assert_eq!(first.validity_sample_visits, shifted.validity_sample_visits);
            for (left_family, right_family) in first.families.iter().zip(shifted.families) {
                for (left, right) in left_family.levels.iter().zip(right_family.levels).take(2) {
                    assert!(left.detail_rms > super::ACOUSTIC_TRAJECTORY_DETAIL_RMS_FLOOR);
                    assert_eq!(left.valid_coefficients, right.valid_coefficients);
                    assert!((left.mean_absolute_detail - right.mean_absolute_detail).abs() < 2e-6);
                    assert!((left.detail_rms - right.detail_rms).abs() < 2e-6);
                    assert!((left.normalized_entropy - right.normalized_entropy).abs() < 2e-6);
                    assert!(
                        (left.normalized_detail_change - right.normalized_detail_change).abs()
                            < 2e-6
                    );
                }
            }
        }
    }

    #[test]
    fn stationary_d4_does_not_invent_detail_shape_at_a_linear_window_seam() {
        let values = std::array::from_fn(|_| {
            std::array::from_fn(|offset| {
                offset as f32 / (ACOUSTIC_TRAJECTORY_HISTORY_FRAMES - 1) as f32
            })
        });
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut active = || false;
        let summary = super::analyze_acoustic_trajectory_wavelet(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticWaveletBasis::DaubechiesFourTap,
            4,
            &mut active,
        )
        .expect("linear stationary D4 trajectory");
        for level in summary.families[0].levels {
            assert!(level.available);
            assert!(level.detail_rms <= super::ACOUSTIC_TRAJECTORY_DETAIL_RMS_FLOOR);
            assert!(!level.normalized_entropy_available);
            assert_eq!(level.normalized_entropy, 0.0);
            assert!(!level.normalized_detail_change_available);
            assert_eq!(level.normalized_detail_change, 0.0);
        }
    }

    #[test]
    fn trajectory_families_freeze_sources_ownership_domains_and_default_invariants() {
        assert_eq!(
            AcousticTrajectoryFamily::ALL.map(AcousticTrajectoryFamily::id),
            [
                "voiced_cepstral_envelope_magnitude",
                "voiced_occupancy",
                "low_band_fraction",
                "mid_band_fraction",
                "high_band_fraction",
            ]
        );
        let wavelet_default = super::AcousticTrajectoryWaveletFamilySummary::default();
        let scattering_default = super::AcousticScatteringFamilySummary::default();
        assert_eq!(wavelet_default.owner, wavelet_default.family.owner());
        assert_eq!(scattering_default.owner, scattering_default.family.owner());

        let mut frame = synthetic_feature(0, 0.0, false, false);
        frame.voice.cepstral_envelope = [2.0; CEPSTRAL_COEFFICIENTS];
        frame.voice.temporal_modulation = 17.0;
        frame.voice.voiced_fraction = 0.9;
        frame.quality.voiced = false;
        let (magnitude, magnitude_valid) =
            AcousticTrajectoryFamily::VoicedCepstralEnvelopeMagnitude.value_and_valid(&frame);
        let (occupancy, occupancy_valid) =
            AcousticTrajectoryFamily::VoicedOccupancy.value_and_valid(&frame);
        assert_eq!(magnitude, 2.0);
        assert!(!magnitude_valid);
        assert_eq!(occupancy, 0.0);
        assert!(occupancy_valid);

        frame.voice.voiced_fraction = 0.0;
        frame.quality.voiced = true;
        let (magnitude, magnitude_valid) =
            AcousticTrajectoryFamily::VoicedCepstralEnvelopeMagnitude.value_and_valid(&frame);
        let (occupancy, occupancy_valid) =
            AcousticTrajectoryFamily::VoicedOccupancy.value_and_valid(&frame);
        assert_eq!(magnitude, 2.0);
        assert!(magnitude_valid);
        assert_eq!(occupancy, 1.0);
        assert!(occupancy_valid);

        frame.quality.transient = true;
        assert!(
            !AcousticTrajectoryFamily::VoicedCepstralEnvelopeMagnitude
                .value_and_valid(&frame)
                .1
        );
        assert!(AcousticTrajectoryFamily::VoicedCepstralEnvelopeMagnitude.value_is_in_domain(2.0));
        for family in &AcousticTrajectoryFamily::ALL[1..] {
            assert!(!family.value_is_in_domain(2.0));
        }
        for family in AcousticTrajectoryFamily::ALL {
            assert!(!family.value_is_in_domain(f32::NAN));
        }
    }

    #[test]
    fn trajectory_kernels_reject_invalid_configuration_geometry_and_values() {
        let mut values = trajectory_fixture_values();
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut active = || false;
        for levels in [0, super::ACOUSTIC_WAVELET_MAX_LEVELS + 1] {
            assert!(
                super::analyze_acoustic_trajectory_wavelet(
                    trajectory_window(&values, &valid, 0, 0, 63),
                    AcousticWaveletBasis::Haar,
                    levels,
                    &mut active,
                )
                .is_err()
            );
        }
        assert!(
            super::analyze_acoustic_trajectory_wavelet(
                trajectory_window(&values, &valid, ACOUSTIC_TRAJECTORY_HISTORY_FRAMES, 0, 63,),
                AcousticWaveletBasis::Haar,
                4,
                &mut active,
            )
            .is_err()
        );
        assert!(
            super::analyze_acoustic_trajectory_wavelet(
                trajectory_window(&values, &valid, 0, 1, 63),
                AcousticWaveletBasis::Haar,
                4,
                &mut active,
            )
            .is_err()
        );
        assert!(
            super::analyze_acoustic_scattering(
                trajectory_window(&values, &valid, 0, 0, 63),
                AcousticScatteringMode::Off,
                &mut active,
            )
            .is_err()
        );
        values[0][3] = f32::NAN;
        assert!(
            super::analyze_acoustic_scattering(
                trajectory_window(&values, &valid, 0, 0, 63),
                AcousticScatteringMode::FirstOrder,
                &mut active,
            )
            .is_err()
        );
    }

    #[test]
    fn trajectory_kernels_mark_supported_constant_families_unavailable() {
        let values = [[0.5_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut active = || false;
        let wavelet = super::analyze_acoustic_trajectory_wavelet(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticWaveletBasis::Haar,
            4,
            &mut active,
        )
        .expect("constant trajectory wavelet");
        assert!(wavelet.families.iter().all(|family| {
            family.input_valid_frames == 64
                && family.input_was_constant_or_near_constant
                && family.valid_level_count == 0
        }));
        assert_eq!(wavelet.filter_tap_terms, 0);
        assert_eq!(wavelet.validity_sample_visits, 0);

        let scattering = super::analyze_acoustic_scattering(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticScatteringMode::FirstAndSecondOrder,
            &mut active,
        )
        .expect("constant scattering");
        assert!(scattering.families.iter().all(|family| {
            family.input_valid_frames == 64
                && family.input_was_constant_or_near_constant
                && family.first_order_available == [false; 3]
                && family.second_order_available == [false; 3]
        }));
        assert_eq!(scattering.filter_sample_terms, 0);
        assert_eq!(scattering.validity_sample_visits, 0);
    }

    #[test]
    fn trajectory_kernels_suppress_f32_jitter_but_retain_above_floor_variation() {
        let mut quantization_only = [[0.5_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        for family in &mut quantization_only {
            family[ACOUSTIC_TRAJECTORY_HISTORY_FRAMES - 1] = f32::from_bits(0.5_f32.to_bits() + 1);
        }
        quantization_only[0].fill(500_000.0);
        quantization_only[0][ACOUSTIC_TRAJECTORY_HISTORY_FRAMES - 1] =
            f32::from_bits(500_000.0_f32.to_bits() + 1);
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut active = || false;
        let wavelet = super::analyze_acoustic_trajectory_wavelet(
            trajectory_window(&quantization_only, &valid, 0, 0, 63),
            AcousticWaveletBasis::Haar,
            4,
            &mut active,
        )
        .expect("quantization-only trajectory wavelet");
        assert!(wavelet.families.iter().all(|family| {
            family.input_was_constant_or_near_constant && family.valid_level_count == 0
        }));
        assert_eq!(wavelet.filter_tap_terms, 0);
        assert_eq!(wavelet.validity_sample_visits, 0);

        let scattering = super::analyze_acoustic_scattering(
            trajectory_window(&quantization_only, &valid, 0, 0, 63),
            AcousticScatteringMode::FirstAndSecondOrder,
            &mut active,
        )
        .expect("quantization-only scattering");
        assert!(
            scattering
                .families
                .iter()
                .all(|family| family.input_was_constant_or_near_constant)
        );
        assert_eq!(scattering.filter_sample_terms, 0);
        assert_eq!(scattering.validity_sample_visits, 0);

        let relative_floor = super::ACOUSTIC_TRAJECTORY_CENTERED_RMS_RELATIVE_FLOOR;
        let mut above_floor = [[0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        for (family_index, family) in above_floor.iter_mut().enumerate() {
            let mean: f32 = if family_index == 0 { 500_000.0 } else { 0.5 };
            let delta = 2.0 * relative_floor * mean.max(1.0);
            for (offset, value) in family.iter_mut().enumerate() {
                *value = mean + if offset % 2 == 0 { delta } else { -delta };
            }
        }
        let wavelet = super::analyze_acoustic_trajectory_wavelet(
            trajectory_window(&above_floor, &valid, 0, 0, 63),
            AcousticWaveletBasis::Haar,
            4,
            &mut active,
        )
        .expect("above-floor trajectory wavelet");
        assert!(wavelet.families.iter().all(|family| {
            !family.input_was_constant_or_near_constant && family.valid_level_count == 4
        }));
        let scattering = super::analyze_acoustic_scattering(
            trajectory_window(&above_floor, &valid, 0, 0, 63),
            AcousticScatteringMode::FirstAndSecondOrder,
            &mut active,
        )
        .expect("above-floor scattering");
        assert!(
            scattering
                .families
                .iter()
                .all(|family| !family.input_was_constant_or_near_constant)
        );
    }

    #[test]
    fn configured_trajectory_study_rejects_gap_without_advancing() {
        let config = AcousticSidecarStudyConfig {
            trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Haar,
            trajectory_wavelet_levels: 4,
            scattering_mode: AcousticScatteringMode::FirstOrder,
            ..AcousticSidecarStudyConfig::default()
        };
        let mut study = AcousticSidecarStudy::new(config).expect("trajectory cadence study");
        let frame_samples = [0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        let mut active = || false;
        study
            .observe_normalized_16khz_frame(
                &frame_samples,
                &trajectory_sidecar_feature(0),
                &mut active,
            )
            .expect("trajectory frame zero");
        let error = study
            .observe_normalized_16khz_frame(
                &frame_samples,
                &trajectory_sidecar_feature(2),
                &mut active,
            )
            .expect_err("trajectory gap");
        assert!(error.to_string().contains("expected frame 1, got 2"));
        study
            .observe_normalized_16khz_frame(
                &frame_samples,
                &trajectory_sidecar_feature(1),
                &mut active,
            )
            .expect("trajectory frame one after rejected gap");
    }

    #[test]
    fn trajectory_kernels_honor_family_level_scale_and_pair_cancellation() {
        let values = trajectory_fixture_values();
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut wavelet_checks = 0usize;
        let wavelet_error = super::analyze_acoustic_trajectory_wavelet(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticWaveletBasis::Haar,
            4,
            &mut || {
                wavelet_checks += 1;
                wavelet_checks == 3
            },
        )
        .expect_err("cancel before trajectory level one");
        assert!(matches!(wavelet_error, FwError::Cancelled(_)));
        assert!(
            wavelet_error
                .to_string()
                .contains("family voiced_cepstral_envelope_magnitude level 1")
        );
        assert_eq!(wavelet_checks, 3);

        let mut scattering_checks = 0usize;
        let scattering_error = super::analyze_acoustic_scattering(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticScatteringMode::SecondOrder,
            &mut || {
                scattering_checks += 1;
                scattering_checks == 4
            },
        )
        .expect_err("cancel before first second-order pair");
        assert!(matches!(scattering_error, FwError::Cancelled(_)));
        assert!(
            scattering_error
                .to_string()
                .contains("family voiced_cepstral_envelope_magnitude second-order pair 0")
        );
        assert_eq!(scattering_checks, 4);
    }

    #[test]
    fn scattering_orders_freeze_goldens_selection_and_operation_accounting() {
        let values = std::array::from_fn(|_| {
            std::array::from_fn(|offset| if offset % 2 == 0 { 0.0 } else { 1.0 })
        });
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut active = || false;
        let first = super::analyze_acoustic_scattering(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticScatteringMode::FirstOrder,
            &mut active,
        )
        .expect("first-order scattering");
        let second = super::analyze_acoustic_scattering(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticScatteringMode::SecondOrder,
            &mut active,
        )
        .expect("second-order scattering");
        let combined = super::analyze_acoustic_scattering(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticScatteringMode::FirstAndSecondOrder,
            &mut active,
        )
        .expect("combined scattering");
        assert_eq!(first.filter_sample_terms, 4_130);
        assert_eq!(first.validity_sample_visits, 4_130);
        assert_eq!(second.filter_sample_terms, 7_450);
        assert_eq!(second.validity_sample_visits, 7_450);
        assert_eq!(combined.filter_sample_terms, 9_730);
        assert_eq!(combined.validity_sample_visits, 9_730);
        assert_eq!(first.scratch_buffer_payload_bytes, 1_280);
        assert_eq!(second.scratch_buffer_payload_bytes, 1_600);
        assert_eq!(combined.scratch_buffer_payload_bytes, 1_600);
        for family_index in 0..super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT {
            let first_family = first.families[family_index];
            let second_family = second.families[family_index];
            let combined_family = combined.families[family_index];
            assert_eq!(first_family.first_order_available, [true; 3]);
            assert_eq!(first_family.first_order_valid_positions, [63, 61, 57]);
            assert!((first_family.first_order_mean_modulus[0] - 0.176_776_69).abs() < 2e-6);
            assert!(first_family.first_order_mean_modulus[1].abs() < 2e-6);
            assert!(first_family.first_order_mean_modulus[2].abs() < 2e-6);
            assert_eq!(first_family.second_order_available, [false; 3]);
            assert_eq!(second_family.first_order_available, [false; 3]);
            assert_eq!(second_family.first_order_valid_positions, [0; 3]);
            assert_eq!(second_family.first_order_mean_modulus, [0.0; 3]);
            assert_eq!(second_family.second_order_available, [true; 3]);
            assert_eq!(second_family.second_order_valid_positions, [60, 56, 54]);
            assert_eq!(second_family.second_order_mean_modulus, [0.0; 3]);
            assert_eq!(combined_family.first_order_available, [true; 3]);
            assert_eq!(
                combined_family.first_order_mean_modulus,
                first_family.first_order_mean_modulus
            );
            assert_eq!(
                combined_family.second_order_mean_modulus,
                second_family.second_order_mean_modulus
            );
        }
        assert_scattering_summary_is_bounded(&combined);
        assert_eq!(
            format!("{combined:?}"),
            "AcousticScatteringSummary { mode: FirstAndSecondOrder, window_start_frame_index: 0, window_end_frame_index: 63, available_first_order_count: 15, available_second_order_count: 15, filter_sample_terms: 9730, validity_sample_visits: 9730, scratch_buffer_payload_bytes: 1600, .. }"
        );
    }

    #[test]
    fn scattering_matches_scalar_nonzero_first_and_second_order_reference() {
        let values = trajectory_fixture_values();
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut active = || false;
        let actual = super::analyze_acoustic_scattering(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticScatteringMode::FirstAndSecondOrder,
            &mut active,
        )
        .expect("full-valid differential scattering");
        let (expected_first, expected_second) = scalar_reference_full_valid_scattering(&values[0]);
        assert!(expected_first.iter().all(|value| *value > 1e-4));
        assert!(expected_second.iter().all(|value| *value > 1e-4));
        for (actual, expected) in actual.families[0]
            .first_order_mean_modulus
            .iter()
            .zip(expected_first)
        {
            assert!(
                (*actual - expected).abs() < 2e-6,
                "{actual} versus {expected}"
            );
        }
        for (actual, expected) in actual.families[0]
            .second_order_mean_modulus
            .iter()
            .zip(expected_second)
        {
            assert!(
                (*actual - expected).abs() < 2e-6,
                "{actual} versus {expected}"
            );
        }
    }

    #[test]
    fn scattering_boundary_impulse_matches_closed_form_nonzero_order_goldens() {
        let mut values = [[0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        for family in &mut values {
            family[0] = 1.0;
        }
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut active = || false;
        let summary = super::analyze_acoustic_scattering(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticScatteringMode::FirstAndSecondOrder,
            &mut active,
        )
        .expect("closed-form scattering impulse");
        let family = summary.families[0];
        assert!(!family.input_was_constant_or_near_constant);
        assert_eq!(family.first_order_available, [true; 3]);
        assert_eq!(family.first_order_valid_positions, [63, 61, 57]);
        assert_eq!(family.second_order_available, [true; 3]);
        assert_eq!(family.second_order_valid_positions, [60, 56, 54]);
        for (actual, expected) in family.first_order_mean_modulus.into_iter().zip([
            0.011_312_645,
            0.008_261_519,
            0.006_251_725,
        ]) {
            assert!(
                (actual - expected).abs() < 1e-7,
                "{actual} versus {expected}"
            );
        }
        for (actual, expected) in family.second_order_mean_modulus.into_iter().zip([
            0.005_939_139,
            0.004_499_577,
            0.003_299_521_5,
        ]) {
            assert!(
                (actual - expected).abs() < 1e-7,
                "{actual} versus {expected}"
            );
        }
    }

    #[test]
    fn configured_scattering_only_runner_freezes_each_selector_and_hidden_outputs() {
        let frame_samples = [0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        for mode in [
            AcousticScatteringMode::FirstOrder,
            AcousticScatteringMode::SecondOrder,
            AcousticScatteringMode::FirstAndSecondOrder,
        ] {
            let config = AcousticSidecarStudyConfig {
                scattering_mode: mode,
                ..AcousticSidecarStudyConfig::default()
            };
            let mut study = AcousticSidecarStudy::new(config).expect("scattering-only study");
            let mut final_observation = None;
            let mut active = || false;
            for frame_index in 0..ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
                let mut frame = trajectory_sidecar_feature(frame_index);
                let voiced = frame_index < ACOUSTIC_TRAJECTORY_HISTORY_FRAMES / 2;
                frame.voice.f0_hz = voiced.then_some(140.0);
                frame.voice.pitch_uncertainty_octaves = voiced.then_some(0.2);
                frame.voice.voicing_confidence = if voiced { 0.9 } else { 0.0 };
                frame.voice.harmonicity = if voiced { 0.9 } else { 0.0 };
                frame.voice.harmonic_to_noise_db = if voiced { 9.5 } else { -20.0 };
                frame.quality.voiced = voiced;
                frame.quality.reliable_pitch = voiced;
                assert!(super::validate_acoustic_frame(&frame).is_ok());
                final_observation = Some(
                    study
                        .observe_normalized_16khz_frame(&frame_samples, &frame, &mut active)
                        .expect("scattering-only observation"),
                );
            }
            let observation = final_observation.expect("complete scattering-only window");
            assert!(observation.wavelet().is_none());
            assert!(observation.modulation().is_none());
            assert!(observation.trajectory_wavelet().is_none());
            let summary = observation
                .scattering()
                .expect("selected scattering result");
            assert_eq!(observation.config(), config);
            assert!(
                format!("{observation:?}").contains(&format!("config: {config:?}")),
                "observation Debug must expose every configured study axis"
            );
            assert_eq!(summary.mode, mode);
            let occupancy = summary.families[1];
            if mode.emits_first_order() {
                assert_eq!(occupancy.first_order_available, [true; 3]);
                assert!(
                    occupancy
                        .first_order_mean_modulus
                        .iter()
                        .all(|value| *value > 0.0)
                );
            } else {
                assert_eq!(occupancy.first_order_available, [false; 3]);
                assert_eq!(occupancy.first_order_valid_positions, [0; 3]);
                assert_eq!(occupancy.first_order_mean_modulus, [0.0; 3]);
            }
            if mode.emits_second_order() {
                assert_eq!(occupancy.second_order_available, [true; 3]);
                assert!(
                    occupancy
                        .second_order_mean_modulus
                        .iter()
                        .all(|value| *value > 0.0)
                );
                assert_eq!(summary.scratch_buffer_payload_bytes, 1_600);
            } else {
                assert_eq!(occupancy.second_order_available, [false; 3]);
                assert_eq!(occupancy.second_order_valid_positions, [0; 3]);
                assert_eq!(occupancy.second_order_mean_modulus, [0.0; 3]);
                assert_eq!(summary.scratch_buffer_payload_bytes, 1_280);
            }
        }
    }

    #[test]
    fn scattering_nonwrapping_ramp_goldens_and_masked_values() {
        let ramp = std::array::from_fn(|family_index| {
            std::array::from_fn(|offset| {
                if family_index == 0 {
                    offset as f32
                } else {
                    offset as f32 / (ACOUSTIC_TRAJECTORY_HISTORY_FRAMES - 1) as f32
                }
            })
        });
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut active = || false;
        let ramp_summary = super::analyze_acoustic_scattering(
            trajectory_window(&ramp, &valid, 0, 0, 63),
            AcousticScatteringMode::FirstOrder,
            &mut active,
        )
        .expect("non-wrapping ramp scattering");
        let normalized_step = (21_840.0_f64).sqrt().recip();
        let expected = [
            normalized_step * std::f64::consts::FRAC_1_SQRT_2,
            2.0 * normalized_step,
            16.0 * normalized_step / 8.0_f64.sqrt(),
        ];
        let ramp_family = ramp_summary.families[0];
        assert_eq!(ramp_family.first_order_valid_positions, [63, 61, 57]);
        for (actual, expected) in ramp_family
            .first_order_mean_modulus
            .into_iter()
            .zip(expected)
        {
            assert!((f64::from(actual) - expected).abs() < 2e-6);
        }

        let values = trajectory_fixture_values();
        let mut sparse_valid = valid;
        let mut changed_invalid_values = values;
        for family_index in 0..super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT {
            for offset in (family_index..ACOUSTIC_TRAJECTORY_HISTORY_FRAMES).step_by(5) {
                sparse_valid[family_index][offset] = false;
                changed_invalid_values[family_index][offset] = if offset % 2 == 0 {
                    f32::NAN
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
        let masked = super::analyze_acoustic_scattering(
            trajectory_window(&values, &sparse_valid, 0, 0, 63),
            AcousticScatteringMode::FirstAndSecondOrder,
            &mut active,
        )
        .expect("masked scattering");
        let changed = super::analyze_acoustic_scattering(
            trajectory_window(&changed_invalid_values, &sparse_valid, 0, 0, 63),
            AcousticScatteringMode::FirstAndSecondOrder,
            &mut active,
        )
        .expect("masked scattering with changed invalid values");
        assert_eq!(masked, changed);
    }

    #[test]
    fn scattering_is_stable_under_one_frame_translation() {
        let mut first = [[0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
            super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        let mut shifted = first;
        for family_index in 0..super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT {
            let amplitude = 0.5 + family_index as f32 * 0.05;
            first[family_index][16] = amplitude;
            shifted[family_index][17] = amplitude;
        }
        let valid =
            [[true; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        for mode in [
            AcousticScatteringMode::FirstOrder,
            AcousticScatteringMode::SecondOrder,
            AcousticScatteringMode::FirstAndSecondOrder,
        ] {
            let first = super::analyze_acoustic_scattering(
                trajectory_window(&first, &valid, 0, 0, 63),
                mode,
                &mut || false,
            )
            .expect("first scattering window");
            let shifted = super::analyze_acoustic_scattering(
                trajectory_window(&shifted, &valid, 0, 1, 64),
                mode,
                &mut || false,
            )
            .expect("translated scattering window");
            assert_eq!(first.filter_sample_terms, shifted.filter_sample_terms);
            assert_eq!(first.validity_sample_visits, shifted.validity_sample_visits);
            if mode.emits_first_order() {
                assert!(first.families.iter().any(|family| {
                    family
                        .first_order_mean_modulus
                        .iter()
                        .any(|value| *value > 0.0)
                }));
            }
            if mode.emits_second_order() {
                assert!(first.families.iter().any(|family| {
                    family
                        .second_order_mean_modulus
                        .iter()
                        .any(|value| *value > 0.0)
                }));
            }
            for (left, right) in first.families.iter().zip(shifted.families) {
                assert_eq!(left.input_valid_frames, right.input_valid_frames);
                assert_eq!(left.first_order_available, right.first_order_available);
                assert_eq!(
                    left.first_order_valid_positions,
                    right.first_order_valid_positions
                );
                assert_eq!(left.second_order_available, right.second_order_available);
                assert_eq!(
                    left.second_order_valid_positions,
                    right.second_order_valid_positions
                );
                for (left, right) in left
                    .first_order_mean_modulus
                    .into_iter()
                    .zip(right.first_order_mean_modulus)
                {
                    assert!((left - right).abs() < 2e-6, "{left} versus {right}");
                }
                for (left, right) in left
                    .second_order_mean_modulus
                    .into_iter()
                    .zip(right.second_order_mean_modulus)
                {
                    assert!((left - right).abs() < 2e-6, "{left} versus {right}");
                }
            }
        }
    }

    #[test]
    fn scattering_enforces_exact_first_order_minimum_valid_output_boundary() {
        let values = trajectory_fixture_values();
        let mut valid =
            [[false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        for family_valid in &mut valid {
            family_valid[..14].fill(true);
            for offset in (16..=50).step_by(2) {
                family_valid[offset] = true;
            }
        }
        let mut active = || false;
        let below = super::analyze_acoustic_scattering(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticScatteringMode::FirstOrder,
            &mut active,
        )
        .expect("below-boundary scattering");
        assert!(below.families.iter().all(|family| {
            family.input_valid_frames == 32
                && family.first_order_valid_positions[2] == 7
                && !family.first_order_available[2]
                && family.first_order_mean_modulus[2] == 0.0
        }));

        for family_valid in &mut valid {
            family_valid[14] = true;
        }
        let boundary = super::analyze_acoustic_scattering(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticScatteringMode::FirstOrder,
            &mut active,
        )
        .expect("at-boundary scattering");
        assert!(boundary.families.iter().all(|family| {
            family.input_valid_frames == 33
                && family.first_order_valid_positions[2] == 8
                && family.first_order_available[2]
                && family.first_order_mean_modulus[2].is_finite()
        }));
    }

    #[test]
    fn scattering_enforces_exact_second_order_minimum_valid_output_boundary() {
        let values = trajectory_fixture_values();
        let mut valid =
            [[false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES]; super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
        for family_valid in &mut valid {
            family_valid[..17].fill(true);
            for offset in (20..=48).step_by(2) {
                family_valid[offset] = true;
            }
        }
        let below = super::analyze_acoustic_scattering(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticScatteringMode::SecondOrder,
            &mut || false,
        )
        .expect("below-boundary second-order scattering");
        assert!(below.families.iter().all(|family| {
            family.input_valid_frames == 32
                && family.second_order_valid_positions[2] == 7
                && !family.second_order_available[2]
                && family.second_order_mean_modulus[2] == 0.0
        }));

        for family_valid in &mut valid {
            family_valid[17] = true;
        }
        let boundary = super::analyze_acoustic_scattering(
            trajectory_window(&values, &valid, 0, 0, 63),
            AcousticScatteringMode::SecondOrder,
            &mut || false,
        )
        .expect("at-boundary second-order scattering");
        assert!(boundary.families.iter().all(|family| {
            family.input_valid_frames == 33
                && family.second_order_valid_positions[2] == 8
                && family.second_order_available[2]
                && family.second_order_mean_modulus[2].is_finite()
        }));
    }

    #[test]
    fn configured_trajectory_masks_voice_activity_separately_from_channel_bands() {
        let config = AcousticSidecarStudyConfig {
            trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Haar,
            trajectory_wavelet_levels: 4,
            ..AcousticSidecarStudyConfig::default()
        };
        let mut study = AcousticSidecarStudy::new(config).expect("masked-family study");
        let frame_samples = [0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        let mut active = || false;
        let mut final_observation = None;
        for frame_index in 0..ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
            let low_energy = frame_index < 40;
            let mut frame = synthetic_feature(frame_index, 0.0, low_energy, false);
            if !low_energy {
                let phase = std::f32::consts::TAU * frame_index as f32
                    / ACOUSTIC_TRAJECTORY_HISTORY_FRAMES as f32;
                frame.voice.voiced_fraction = 0.5 + 0.25 * phase.sin();
                frame.channel.low_band_fraction = 0.3 + 0.05 * phase.sin();
                frame.channel.mid_band_fraction = 0.45 + 0.05 * phase.cos();
                frame.channel.high_band_fraction =
                    1.0 - frame.channel.low_band_fraction - frame.channel.mid_band_fraction;
            }
            final_observation = Some(
                study
                    .observe_normalized_16khz_frame(&frame_samples, &frame, &mut active)
                    .expect("masked-family observation"),
            );
        }
        let summary = final_observation
            .expect("complete masked-family observation")
            .trajectory_wavelet()
            .expect("masked-family trajectory wavelet");
        assert_eq!(summary.families[0].input_valid_frames, 24);
        assert_eq!(summary.families[0].valid_level_count, 0);
        assert_eq!(summary.families[1].input_valid_frames, 64);
        assert!(summary.families[1].valid_level_count > 0);
        for family in &summary.families[2..] {
            assert_eq!(family.input_valid_frames, 24);
            assert_eq!(family.valid_level_count, 0);
        }
    }

    #[test]
    fn trajectory_candidates_remain_finite_under_deterministic_adversarial_masks() {
        let mut state = 0x7f4a_7c15_u32;
        let mut active = || false;
        for case_index in 0..32 {
            let mut values = [[0.0_f32; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
                super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
            let mut valid = [[false; ACOUSTIC_TRAJECTORY_HISTORY_FRAMES];
                super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT];
            for family_index in 0..super::ACOUSTIC_TRAJECTORY_FAMILY_COUNT {
                for offset in 0..ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    values[family_index][offset] = (state >> 8) as f32 / 16_777_215.0;
                    valid[family_index][offset] = ((state >> (case_index % 7)) & 3) != 0;
                }
            }
            for basis in [
                AcousticWaveletBasis::Haar,
                AcousticWaveletBasis::DaubechiesFourTap,
            ] {
                let summary = super::analyze_acoustic_trajectory_wavelet(
                    trajectory_window(
                        &values,
                        &valid,
                        case_index % ACOUSTIC_TRAJECTORY_HISTORY_FRAMES,
                        case_index,
                        case_index + ACOUSTIC_TRAJECTORY_HISTORY_FRAMES - 1,
                    ),
                    basis,
                    4,
                    &mut active,
                )
                .expect("adversarial masked trajectory wavelet");
                assert_trajectory_wavelet_summary_is_bounded(&summary);
            }
            let scattering = super::analyze_acoustic_scattering(
                trajectory_window(
                    &values,
                    &valid,
                    case_index % ACOUSTIC_TRAJECTORY_HISTORY_FRAMES,
                    case_index,
                    case_index + ACOUSTIC_TRAJECTORY_HISTORY_FRAMES - 1,
                ),
                AcousticScatteringMode::FirstAndSecondOrder,
                &mut active,
            )
            .expect("adversarial masked scattering");
            assert_scattering_summary_is_bounded(&scattering);
        }
    }

    #[test]
    fn configured_sidecar_study_emits_all_candidates_after_one_shared_window() {
        let config = AcousticSidecarStudyConfig {
            mode: AcousticSidecarStudyMode::HaarAndModulation,
            frame_wavelet_levels: 4,
            trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::DaubechiesFourTap,
            trajectory_wavelet_levels: 4,
            scattering_mode: AcousticScatteringMode::FirstAndSecondOrder,
        };
        let mut study = AcousticSidecarStudy::new(config).expect("all-candidate study");
        let retained_state_bytes = study.retained_state_bytes_on_target();
        let frame_samples: [f32; ACOUSTIC_FRAME_SAMPLES] = std::array::from_fn(|sample| {
            let phase = std::f32::consts::TAU * 440.0 * sample as f32
                / crate::native_engine::mel::SAMPLE_RATE as f32;
            0.25 * phase.sin()
        });
        let mut active = || false;
        let mut final_observation = None;
        for frame_index in 1..=ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
            let frame = trajectory_sidecar_feature(frame_index);
            let observation = study
                .observe_normalized_16khz_frame(&frame_samples, &frame, &mut active)
                .expect("all-candidate observation");
            assert!(observation.wavelet().is_some());
            if frame_index < ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
                assert!(observation.modulation().is_none());
                assert!(observation.trajectory_wavelet().is_none());
                assert!(observation.scattering().is_none());
            } else {
                final_observation = Some(observation);
            }
        }
        assert_eq!(study.retained_state_bytes_on_target(), retained_state_bytes);
        let observation = final_observation.expect("complete shared trajectory window");
        assert!(observation.modulation().is_some());
        let trajectory = observation
            .trajectory_wavelet()
            .expect("configured trajectory wavelet");
        let scattering = observation.scattering().expect("configured scattering");
        assert_eq!(trajectory.window_start_frame_index, 1);
        assert_eq!(trajectory.window_end_frame_index, 64);
        assert_eq!(trajectory.filter_tap_terms, 5_800);
        assert_eq!(trajectory.validity_sample_visits, 3_312);
        assert_eq!(trajectory.scratch_buffer_payload_bytes, 960);
        assert_eq!(scattering.window_start_frame_index, 1);
        assert_eq!(scattering.window_end_frame_index, 64);
        assert_eq!(scattering.filter_sample_terms, 7_914);
        assert_eq!(scattering.validity_sample_visits, 9_730);
        assert_eq!(scattering.scratch_buffer_payload_bytes, 1_600);
        assert_trajectory_wavelet_summary_is_bounded(&trajectory);
        assert_scattering_summary_is_bounded(&scattering);
        assert_eq!(
            format!("{observation:?}"),
            format!(
                "AcousticSidecarStudyObservation {{ schema_version: {:?}, config: {:?}, frame_index: 64, wavelet_available: true, modulation_available: true, trajectory_wavelet_available: true, scattering_available: true, .. }}",
                super::ACOUSTIC_SIDECAR_STUDY_SCHEMA_VERSION,
                config
            )
        );
        let next_observation = study
            .observe_normalized_16khz_frame(
                &frame_samples,
                &trajectory_sidecar_feature(ACOUSTIC_TRAJECTORY_HISTORY_FRAMES + 1),
                &mut active,
            )
            .expect("one-frame-sliding observation");
        assert_eq!(
            next_observation
                .trajectory_wavelet()
                .expect("sliding trajectory wavelet")
                .window_start_frame_index,
            2
        );
        assert_eq!(
            next_observation
                .scattering()
                .expect("sliding scattering")
                .window_end_frame_index,
            65
        );
        let mut sliding_reference =
            AcousticSidecarStudy::new(config).expect("sliding reference study");
        let mut expected_sliding_observation = None;
        for frame_index in 2..=ACOUSTIC_TRAJECTORY_HISTORY_FRAMES + 1 {
            expected_sliding_observation = Some(
                sliding_reference
                    .observe_normalized_16khz_frame(
                        &frame_samples,
                        &trajectory_sidecar_feature(frame_index),
                        &mut active,
                    )
                    .expect("sliding reference observation"),
            );
        }
        assert_eq!(
            next_observation,
            expected_sliding_observation.expect("complete sliding reference window")
        );
        assert_eq!(study.retained_state_bytes_on_target(), retained_state_bytes);
    }

    #[test]
    fn configured_sidecar_cancellation_after_modulation_rolls_back_every_ring() {
        let config = AcousticSidecarStudyConfig {
            mode: AcousticSidecarStudyMode::Modulation,
            trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Haar,
            trajectory_wavelet_levels: 4,
            scattering_mode: AcousticScatteringMode::FirstOrder,
            ..AcousticSidecarStudyConfig::default()
        };
        let frame_samples = [0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        let mut study = AcousticSidecarStudy::new(config).expect("atomic study");
        let mut active = || false;
        for frame_index in 1..ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
            let frame = trajectory_sidecar_feature(frame_index);
            let observation = study
                .observe_normalized_16khz_frame(&frame_samples, &frame, &mut active)
                .expect("atomic warmup");
            assert!(observation.modulation().is_none());
            assert!(observation.trajectory_wavelet().is_none());
        }
        let final_frame = trajectory_sidecar_feature(ACOUSTIC_TRAJECTORY_HISTORY_FRAMES);
        let mut checks = 0usize;
        let error = study
            .observe_normalized_16khz_frame(&frame_samples, &final_frame, &mut || {
                checks += 1;
                checks == 18
            })
            .expect_err("cancel after completed staged modulation");
        assert!(matches!(error, FwError::Cancelled(_)));
        assert!(
            error
                .to_string()
                .contains("acoustic trajectory sidecar cancelled before frame 64")
        );
        assert_eq!(checks, 18);

        let retry = study
            .observe_normalized_16khz_frame(&frame_samples, &final_frame, &mut active)
            .expect("atomic retry");
        let mut reference = AcousticSidecarStudy::new(config).expect("reference study");
        let mut expected = None;
        for frame_index in 1..=ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
            expected = Some(
                reference
                    .observe_normalized_16khz_frame(
                        &frame_samples,
                        &trajectory_sidecar_feature(frame_index),
                        &mut active,
                    )
                    .expect("reference observation"),
            );
        }
        assert_eq!(retry, expected.expect("reference final observation"));
        assert_eq!(
            retry
                .modulation()
                .expect("retried modulation")
                .window_start_frame_index,
            1
        );
        assert_eq!(
            retry
                .trajectory_wavelet()
                .expect("retried trajectory")
                .window_start_frame_index,
            1
        );
    }

    #[test]
    fn configured_sidecar_cancellation_after_trajectory_wavelet_rolls_back_ring() {
        let config = AcousticSidecarStudyConfig {
            trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::Haar,
            trajectory_wavelet_levels: 4,
            scattering_mode: AcousticScatteringMode::FirstOrder,
            ..AcousticSidecarStudyConfig::default()
        };
        let frame_samples = [0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        let mut study = AcousticSidecarStudy::new(config).expect("deep-atomic study");
        let mut active = || false;
        for frame_index in 1..ACOUSTIC_TRAJECTORY_HISTORY_FRAMES {
            study
                .observe_normalized_16khz_frame(
                    &frame_samples,
                    &trajectory_sidecar_feature(frame_index),
                    &mut active,
                )
                .expect("deep-atomic warmup");
        }
        let final_frame = trajectory_sidecar_feature(ACOUSTIC_TRAJECTORY_HISTORY_FRAMES);
        let mut checks = 0usize;
        let error = study
            .observe_normalized_16khz_frame(&frame_samples, &final_frame, &mut || {
                checks += 1;
                checks == 28
            })
            .expect_err("cancel after completed trajectory wavelet");
        assert!(matches!(error, FwError::Cancelled(_)));
        assert!(
            error
                .to_string()
                .contains("family voiced_cepstral_envelope_magnitude first-order scale 0")
        );
        assert_eq!(checks, 28);

        let retry = study
            .observe_normalized_16khz_frame(&frame_samples, &final_frame, &mut active)
            .expect("deep-atomic retry");
        assert_eq!(
            retry
                .trajectory_wavelet()
                .expect("deep-retry trajectory")
                .window_start_frame_index,
            1
        );
        assert_eq!(
            retry
                .scattering()
                .expect("deep-retry scattering")
                .window_start_frame_index,
            1
        );
    }

    #[test]
    fn experimental_sidecars_leave_default_acoustic_v2_output_byte_exact() {
        let samples = sine_wave(180.0, 0.8, 0.35);
        let baseline = features(&samples);
        let baseline_bytes = acoustic_feature_bytes(&baseline);
        let _wavelet = analyze_acoustic_wavelet(
            &samples[..ACOUSTIC_FRAME_SAMPLES],
            AcousticWaveletConfig {
                basis: AcousticWaveletBasis::DaubechiesFourTap,
                levels: 4,
            },
            || false,
        )
        .expect("opt-in wavelet");
        let mut modulation = AcousticModulationSidecar::new();
        let mut active = || false;
        for frame in &baseline {
            let _ = modulation
                .push(frame, &mut active)
                .expect("opt-in modulation");
        }
        let mut trajectory_study = AcousticSidecarStudy::new(AcousticSidecarStudyConfig {
            trajectory_wavelet_mode: AcousticTrajectoryWaveletMode::DaubechiesFourTap,
            trajectory_wavelet_levels: 4,
            scattering_mode: AcousticScatteringMode::FirstAndSecondOrder,
            ..AcousticSidecarStudyConfig::default()
        })
        .expect("opt-in trajectory study");
        let frame_samples = [0.0_f32; ACOUSTIC_FRAME_SAMPLES];
        for frame in &baseline {
            let _ = trajectory_study
                .observe_normalized_16khz_frame(&frame_samples, frame, &mut active)
                .expect("opt-in trajectory observation");
        }
        let repeated = features(&samples);
        let repeated_bytes = acoustic_feature_bytes(&repeated);
        assert_eq!(baseline, repeated);
        assert_eq!(baseline_bytes, repeated_bytes);
    }

    #[test]
    fn robust_tracklet_identity_ignores_silence_clipping_and_transients() {
        let mut accumulator = super::TrackletAccumulator::default();
        for index in 0..8 {
            accumulator.push(
                &synthetic_feature(index, 0.25, false, false),
                super::AcousticFeatureAblation::FullV2,
            );
        }
        for index in 8..108 {
            let mut contaminated = synthetic_feature(index, 0.25, false, true);
            contaminated.voice.cepstral_envelope.fill(20.0);
            contaminated.channel.clipping_fraction = 0.01;
            contaminated.quality.clipped = true;
            accumulator.push(&contaminated, super::AcousticFeatureAblation::FullV2);
        }
        let tracklet = accumulator.finish(0, None).expect("tracklet");
        assert_eq!(tracklet.frame_count, 108);
        assert_eq!(tracklet.identity_frame_count, 8);
        assert!(tracklet.voice_valid.iter().all(|valid| *valid));
        assert!(
            tracklet.voice_mean[..12]
                .iter()
                .all(|value| (*value - 0.25).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn invalid_final_frame_does_not_erase_tracklet_channel_dimensions() {
        let mut accumulator = super::TrackletAccumulator::default();
        accumulator.push(
            &synthetic_feature(0, 0.25, false, false),
            super::AcousticFeatureAblation::FullV2,
        );
        accumulator.push(
            &synthetic_feature(1, 0.0, true, false),
            super::AcousticFeatureAblation::FullV2,
        );

        let tracklet = accumulator.finish(0, None).expect("tracklet");
        assert!(tracklet.channel_valid);
        assert_eq!(tracklet.channel_frame_count, 1);
        assert_eq!(
            tracklet.channel_dimensions,
            super::CHANNEL_VECTOR_DIMENSIONS
        );
    }

    #[test]
    fn identity_reservoir_is_bounded_and_keeps_high_information_frames() {
        let mut accumulator = super::TrackletAccumulator::default();
        for index in 0..200 {
            let mut frame = synthetic_feature(index, index as f32 / 1_000.0, false, false);
            frame.voice.voicing_confidence = 0.56 + 0.4 * index as f32 / 199.0;
            frame.voice.harmonicity = frame.voice.voicing_confidence;
            frame.voice.harmonic_to_noise_db =
                super::harmonic_to_noise_db(frame.voice.voicing_confidence);
            frame.voice.pitch_uncertainty_octaves =
                Some((1.0 - frame.voice.voicing_confidence) * 2.0);
            accumulator.push(&frame, super::AcousticFeatureAblation::FullV2);
        }
        let tracklet = accumulator.finish(0, None).expect("tracklet");
        assert_eq!(tracklet.identity_frame_count, 200);
        assert!(
            tracklet
                .voice_support
                .iter()
                .all(|support| *support <= super::MAX_IDENTITY_SUBWINDOWS as u32)
        );
        assert_eq!(
            tracklet.voice_support[0],
            super::MAX_IDENTITY_SUBWINDOWS as u32
        );
        assert!(
            tracklet.voice_mean[0] > 0.1,
            "low-quality early frames should not dominate the retained subwindows"
        );
    }

    #[test]
    fn equal_tracklet_normalization_preserves_minority_separation() {
        let mut tracklets = (0..10)
            .map(|index| {
                profile_tracklet(
                    index,
                    index as u64 * 100,
                    index as u64 * 100 + 100,
                    0.0,
                    0.0,
                    10,
                )
            })
            .collect::<Vec<_>>();
        tracklets.push(profile_tracklet(10, 1_000, 1_100, 1.0, 1.0, 10));
        let summary = super::normalize_tracklet_features(&mut tracklets);
        assert_eq!(summary.normalized_voice_dimensions, VOICE_VECTOR_DIMENSIONS);
        assert_eq!(
            summary.normalized_channel_dimensions,
            CHANNEL_VECTOR_DIMENSIONS
        );
        assert!(tracklets[..10].iter().all(|tracklet| {
            tracklet
                .voice_mean
                .iter()
                .chain(tracklet.channel_mean.iter())
                .all(|value| value.abs() < f32::EPSILON)
        }));
        assert!(
            tracklets[10].voice_mean.iter().all(|value| *value >= 20.0),
            "a majority speaker must not erase a minority tracklet"
        );
    }

    #[test]
    fn v1_representation_requires_explicit_selection_and_stays_lower_dimensional() {
        let frames = (0..80)
            .map(|index| synthetic_feature(index, 0.1, false, false))
            .collect::<Vec<_>>();
        let compact =
            super::compact_vectors_for_schema(&frames[5], super::AcousticFeatureSchemaVersion::V1);
        assert_eq!(
            &compact.channel[..8],
            &[
                -19.9 / 40.0,
                1_050.0 / 8_000.0,
                1_200.0 / 8_000.0,
                0.1,
                -2.0 / 10.0,
                0.4,
                0.5,
                0.1,
            ]
        );
        assert!(compact.channel[8..].iter().all(|value| *value == 0.0));
        let (tracklets, summary) = super::segment_acoustic_frames_with_schema(
            frames,
            &AcousticBoundaryHints::default(),
            super::AcousticFeatureSchemaVersion::V1,
            || false,
        )
        .expect("explicit v1 fallback");
        assert_eq!(tracklets.len(), 1);
        assert!(tracklets[0].voice_valid[..8].iter().all(|valid| *valid));
        assert!(tracklets[0].voice_valid[8..].iter().all(|valid| !*valid));
        assert_eq!(summary.normalized_voice_dimensions, 0);
        assert_eq!(summary.normalized_channel_dimensions, 0);
        assert_eq!(
            summary.missing_voice_dimensions,
            VOICE_VECTOR_DIMENSIONS - 8
        );
    }

    #[test]
    fn acoustic_v2_compact_vector_matches_golden_synthetic_coordinates() {
        let frame = synthetic_feature(5, 0.25, false, false);
        let compact =
            super::compact_vectors_for_schema(&frame, super::AcousticFeatureSchemaVersion::V2);
        assert_eq!(
            super::AcousticFeatureSchemaVersion::V2.id(),
            super::ACOUSTIC_FEATURE_SCHEMA_VERSION
        );
        assert_eq!(compact.voice.len(), VOICE_VECTOR_DIMENSIONS);
        assert_eq!(compact.channel.len(), CHANNEL_VECTOR_DIMENSIONS);
        assert!(compact.voice_valid.iter().all(|valid| *valid));
        assert!(compact.channel_valid);
        assert!(compact.voice[..12].iter().all(|value| *value == 0.25));
        assert!(compact.voice[12..20].iter().all(|value| *value == 0.0));
        assert!((compact.voice[20] - 140.0_f32.ln()).abs() < 1e-6);
        assert!((compact.voice[21] - 0.9).abs() < f32::EPSILON);
        assert!((compact.voice[22] - 9.5 / 40.0).abs() < f32::EPSILON);
        assert_eq!(
            &compact.voice[23..26],
            &[600.0 / 8_000.0, 1_500.0 / 8_000.0, 2_700.0 / 8_000.0]
        );
        assert!((compact.voice[26] - 0.2).abs() < f32::EPSILON);
        assert_eq!(compact.voice[27], 0.0);
        assert_eq!(
            compact.channel,
            [
                -19.75 / 40.0,
                -70.0 / 40.0,
                30.0 / 40.0,
                1_125.0 / 8_000.0,
                1_200.0 / 8_000.0,
                2_500.0 / 8_000.0,
                4_000.0 / 8_000.0,
                0.3,
                0.4,
                0.2,
                -2.0 / 10.0,
                0.0,
                0.1,
                1.0,
            ]
        );
    }

    #[test]
    fn multiscale_segmentation_detects_a_sustained_acoustic_step() {
        let frames = (0..650)
            .map(|index| {
                synthetic_feature(index, if index < 325 { 0.0 } else { 1.0 }, false, false)
            })
            .collect::<Vec<_>>();
        let hints = AcousticBoundaryHints {
            word_boundaries_ms: vec![3_270],
            ..AcousticBoundaryHints::default()
        };
        let (tracklets, summary) =
            segment_acoustic_frames(frames, &hints, || false).expect("segment");
        assert!(tracklets.len() >= 2, "{tracklets:#?}");
        let evidence = tracklets
            .iter()
            .filter_map(|tracklet| tracklet.boundary_evidence.as_ref())
            .max_by(|left, right| left.change_probability.total_cmp(&right.change_probability))
            .expect("change evidence");
        assert!(evidence.boundary_ms.abs_diff(3_250) <= 100);
        assert!(evidence.snapped_to_word);
        assert!(summary.maximum_retained_frames <= 401);
    }

    #[test]
    fn change_calibration_contract_is_loss_consistent_and_content_addressed() {
        let calibration = super::acoustic_change_calibration();
        let bayes_threshold = calibration.false_split_loss
            / (calibration.false_split_loss + calibration.missed_change_loss);
        assert!((calibration.decision_probability - bayes_threshold).abs() < f32::EPSILON);
        assert!(calibration.hint_contradiction_loss > calibration.false_split_loss);
        assert!(calibration.timing_error_loss_per_second > 0.0);
        assert!(calibration.latency_loss_per_second > 0.0);
        assert!(calibration.fallback_loss > 0.0);
        let hash = super::acoustic_change_calibration_sha256();
        assert_eq!(hash.len(), 64);
        assert!(hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(hash, super::acoustic_change_calibration_sha256());
    }

    #[test]
    fn calibrated_posterior_separates_stationary_noise_from_an_abrupt_step() {
        let stationary = (0..401)
            .map(|index| {
                let dither = ((index * 17 % 11) as f32 - 5.0) * 0.001;
                synthetic_feature(index, 0.1 + dither, false, false)
            })
            .collect::<VecDeque<_>>();
        let abrupt = (0..401)
            .map(|index| {
                synthetic_feature(index, if index < 200 { 0.0 } else { 1.0 }, false, false)
            })
            .collect::<VecDeque<_>>();
        let stationary_evidence = super::multiscale_change_evidence(
            &stationary,
            200,
            &AcousticBoundaryHints::default(),
            super::AcousticFeatureAblation::FullV2,
        );
        let abrupt_evidence = super::multiscale_change_evidence(
            &abrupt,
            200,
            &AcousticBoundaryHints::default(),
            super::AcousticFeatureAblation::FullV2,
        );
        assert!(
            stationary_evidence.change_probability
                < super::acoustic_change_calibration().decision_probability,
            "{stationary_evidence:#?}"
        );
        assert!(
            abrupt_evidence.change_probability
                >= super::acoustic_change_calibration().decision_probability,
            "{abrupt_evidence:#?}"
        );
        assert!(abrupt_evidence.change_probability > stationary_evidence.change_probability + 0.25);
        assert_eq!(abrupt_evidence.supporting_scale_mask, 0b1_1111);
    }

    #[test]
    fn bounded_change_candidates_are_finite_deterministic_and_step_sensitive() {
        let stationary = (0..401)
            .map(|index| {
                let dither = ((index * 17 % 11) as f32 - 5.0) * 0.001;
                synthetic_feature(index, 0.1 + dither, false, false)
            })
            .collect::<VecDeque<_>>();
        let abrupt = (0..401)
            .map(|index| {
                synthetic_feature(index, if index < 200 { 0.0 } else { 1.0 }, false, false)
            })
            .collect::<VecDeque<_>>();
        for (mode, expected_id) in [
            (
                super::AcousticChangeDetectorMode::PageHinkleyV1,
                super::ACOUSTIC_CHANGE_PAGE_HINKLEY_VERSION,
            ),
            (
                super::AcousticChangeDetectorMode::BayesianTwoRegimeV1,
                super::ACOUSTIC_CHANGE_BAYESIAN_VERSION,
            ),
        ] {
            let stationary_evidence = super::multiscale_change_evidence_with_detector(
                &stationary,
                200,
                &AcousticBoundaryHints::default(),
                super::AcousticFeatureAblation::FullV2,
                mode,
            );
            let abrupt_evidence = super::multiscale_change_evidence_with_detector(
                &abrupt,
                200,
                &AcousticBoundaryHints::default(),
                super::AcousticFeatureAblation::FullV2,
                mode,
            );
            let repeated = super::multiscale_change_evidence_with_detector(
                &abrupt,
                200,
                &AcousticBoundaryHints::default(),
                super::AcousticFeatureAblation::FullV2,
                mode,
            );
            assert_eq!(abrupt_evidence, repeated);
            assert_eq!(abrupt_evidence.detector_mode, mode);
            assert_eq!(abrupt_evidence.calibration_id, expected_id);
            assert!(stationary_evidence.change_probability.is_finite());
            assert!((0.0..=1.0).contains(&stationary_evidence.change_probability));
            assert!(abrupt_evidence.change_probability.is_finite());
            assert!((0.0..=1.0).contains(&abrupt_evidence.change_probability));
            assert!(
                stationary_evidence.change_probability
                    < super::acoustic_change_calibration().decision_probability,
                "{mode:?}: {stationary_evidence:#?}"
            );
            assert!(
                abrupt_evidence.change_probability
                    >= super::acoustic_change_calibration().decision_probability,
                "{mode:?}: {abrupt_evidence:#?}"
            );
            assert!(
                abrupt_evidence.change_probability > stationary_evidence.change_probability + 0.25,
                "{mode:?}: stationary={stationary_evidence:#?} abrupt={abrupt_evidence:#?}"
            );
        }
    }

    #[test]
    fn fixed_safe_detector_remains_an_explicit_reproducible_ablation() {
        let ring = (0..401)
            .map(|index| {
                synthetic_feature(index, if index < 200 { 0.0 } else { 1.0 }, false, false)
            })
            .collect::<VecDeque<_>>();
        let evidence = super::multiscale_change_evidence_with_detector(
            &ring,
            200,
            &AcousticBoundaryHints::default(),
            super::AcousticFeatureAblation::FullV2,
            super::AcousticChangeDetectorMode::FixedSafeV1,
        );
        assert_eq!(
            evidence.detector_mode,
            super::AcousticChangeDetectorMode::FixedSafeV1
        );
        assert_eq!(
            evidence.calibration_id,
            super::ACOUSTIC_CHANGE_FIXED_SAFE_VERSION
        );
        assert_eq!(evidence.refinement_offset_frames, 0);
        assert_eq!(evidence.fallback_reason, None);
        assert!(
            evidence.change_probability
                >= super::acoustic_change_calibration().decision_probability
        );
        assert_eq!(super::CHANGE_FIXED_SAFE_SUPPRESSION_FRAMES, 20);
    }

    #[test]
    fn gain_and_channel_only_step_does_not_masquerade_as_speaker_identity() {
        let ring = (0..401)
            .map(|index| {
                let mut frame = synthetic_feature(index, 0.2, false, false);
                if index >= 200 {
                    frame.channel.rms_dbfs = -8.0;
                    frame.channel.dynamics_above_noise_db = 50.0;
                    frame.channel.spectral_centroid_hz = 3_500.0;
                    frame.channel.spectral_rolloff_hz = 6_500.0;
                    frame.channel.high_frequency_attenuation = 0.0;
                    frame.channel.muffling_proxy = 0.0;
                }
                frame
            })
            .collect::<VecDeque<_>>();
        let evidence = super::multiscale_change_evidence(
            &ring,
            200,
            &AcousticBoundaryHints::default(),
            super::AcousticFeatureAblation::FullV2,
        );
        assert!(evidence.channel_distance > 0.1, "{evidence:#?}");
        assert!(evidence.voice_distance < 1e-5, "{evidence:#?}");
        assert!(
            evidence.change_probability < super::acoustic_change_calibration().decision_probability,
            "{evidence:#?}"
        );
        assert_eq!(evidence.fallback_reason, None);
    }

    #[test]
    fn insufficient_voiced_support_uses_the_explicit_safe_fallback() {
        let ring = (0..401)
            .map(|index| synthetic_feature(index, 0.0, true, false))
            .collect::<VecDeque<_>>();
        let evidence = super::multiscale_change_evidence(
            &ring,
            200,
            &AcousticBoundaryHints::default(),
            super::AcousticFeatureAblation::FullV2,
        );
        assert_eq!(
            evidence.fallback_reason,
            Some(super::AcousticChangeFallbackReason::InsufficientVoiceSupport)
        );
        assert_eq!(evidence.supporting_scale_mask, 0);
        assert!(
            evidence.change_probability < super::acoustic_change_calibration().decision_probability,
            "{evidence:#?}"
        );
    }

    #[test]
    fn invalid_covariance_fails_closed_instead_of_emitting_nan() {
        let mut left = super::DiagonalMoments::<4>::default();
        let mut right = super::DiagonalMoments::<4>::default();
        left.count = [5; 4];
        right.count = [5; 4];
        left.mean = [0.0; 4];
        right.mean = [1.0; 4];
        left.m2 = [-1.0, 0.0, 0.0, 0.0];
        right.m2 = [0.0; 4];
        let evidence =
            super::diagonal_glr_evidence(&left, &right, 4, super::acoustic_change_calibration());
        assert!(evidence.invalid_covariance);
        assert!(evidence.evidence.is_finite());
        assert!(evidence.distance.is_finite());
        assert_eq!(evidence.valid_dimensions, 0);
    }

    #[test]
    fn every_multiscale_probability_is_finite_and_bounded() {
        let ring = (0..401)
            .map(|index| {
                let value = if index < 173 {
                    (index % 7) as f32 * 0.01
                } else {
                    0.8 + (index % 5) as f32 * 0.01
                };
                synthetic_feature(index, value, false, index == 173)
            })
            .collect::<VecDeque<_>>();
        for center in 10..=391 {
            let evidence = super::multiscale_change_evidence(
                &ring,
                center,
                &AcousticBoundaryHints::default(),
                super::AcousticFeatureAblation::FullV2,
            );
            assert!(evidence.raw_log_odds.is_finite(), "center {center}");
            assert!(
                evidence.change_probability.is_finite()
                    && (0.0..=1.0).contains(&evidence.change_probability),
                "center {center}: {evidence:#?}"
            );
            assert!(
                evidence
                    .multiscale_scores
                    .iter()
                    .all(|probability| probability.is_finite()
                        && (0.0..=1.0).contains(probability)),
                "center {center}: {evidence:#?}"
            );
        }
    }

    #[test]
    fn bounded_refinement_recovers_a_known_local_offset() {
        let true_boundary = 205;
        let ring = (0..401)
            .map(|index| {
                synthetic_feature(
                    index,
                    if index < true_boundary { 0.0 } else { 1.0 },
                    false,
                    index == true_boundary,
                )
            })
            .collect::<VecDeque<_>>();
        let evidence = super::multiscale_change_evidence(
            &ring,
            200,
            &AcousticBoundaryHints::default(),
            super::AcousticFeatureAblation::FullV2,
        );
        assert_eq!(evidence.refinement_offset_frames, 5);
        assert_eq!(evidence.boundary_ms, true_boundary as u64 * 10);
        assert!(
            evidence.refinement_offset_frames.unsigned_abs()
                <= super::CHANGE_REFINEMENT_RADIUS_FRAMES as u16
        );
    }

    #[test]
    fn word_geometry_has_timing_authority_over_nearby_tiny_diarize_support() {
        let ring = (0..401)
            .map(|index| {
                synthetic_feature(index, if index < 200 { 0.0 } else { 1.0 }, false, false)
            })
            .collect::<VecDeque<_>>();
        let hints = AcousticBoundaryHints {
            word_boundaries_ms: vec![1_970],
            tiny_diarize_boundaries_ms: vec![2_050],
            ..AcousticBoundaryHints::default()
        };
        let evidence = super::multiscale_change_evidence(
            &ring,
            200,
            &hints,
            super::AcousticFeatureAblation::FullV2,
        );
        assert_eq!(evidence.boundary_ms, 1_970);
        assert!(evidence.snapped_to_word);
        assert!(evidence.tiny_diarize_support);
    }

    #[test]
    fn pending_change_near_end_of_stream_is_not_lost() {
        let frames = (0..650)
            .map(|index| {
                synthetic_feature(index, if index < 590 { 0.0 } else { 1.0 }, false, false)
            })
            .collect::<Vec<_>>();
        let (tracklets, summary) =
            segment_acoustic_frames(frames, &AcousticBoundaryHints::default(), || false)
                .expect("segment");
        assert!(tracklets.len() >= 2, "{tracklets:#?}");
        assert!(tracklets.iter().any(|tracklet| {
            tracklet
                .boundary_evidence
                .as_ref()
                .is_some_and(|evidence| evidence.boundary_ms.abs_diff(5_900) <= 100)
        }));
        assert!(summary.fixed_candidate_count > 0);
    }

    #[test]
    fn hysteretic_peak_selector_emits_once_per_weak_excursion_and_rearms_after_a_low_interval() {
        let ring = (0..401)
            .map(|index| synthetic_feature(index, 0.2, false, false))
            .collect::<VecDeque<_>>();
        let mut template = super::multiscale_change_evidence(
            &ring,
            200,
            &AcousticBoundaryHints::default(),
            super::AcousticFeatureAblation::FullV2,
        );
        let mut selector = super::ChangePeakSelector::new(
            super::AcousticChangeDetectorMode::CalibratedPosterior,
            0.10,
        );
        let mut emitted = Vec::new();
        for frame_index in 0..300 {
            template.boundary_ms = frame_index as u64 * 10;
            template.change_probability = 0.2;
            if let Some(evidence) = selector.push(frame_index, template.clone()) {
                emitted.push(evidence);
            }
        }
        assert_eq!(emitted.len(), 1, "{emitted:#?}");
        for frame_index in 300..320 {
            template.boundary_ms = frame_index as u64 * 10;
            template.change_probability = 0.01;
            assert!(selector.push(frame_index, template.clone()).is_none());
        }
        for frame_index in 320..330 {
            template.boundary_ms = frame_index as u64 * 10;
            template.change_probability = 0.2;
            if let Some(evidence) = selector.push(frame_index, template.clone()) {
                emitted.push(evidence);
            }
        }
        if let Some(evidence) = selector.finish() {
            emitted.push(evidence);
        }
        assert_eq!(emitted.len(), 2, "{emitted:#?}");
    }

    #[test]
    fn fixed_safe_peak_selector_retains_the_frozen_suppression_policy() {
        let ring = (0..401)
            .map(|index| synthetic_feature(index, 0.2, false, false))
            .collect::<VecDeque<_>>();
        let mut template = super::multiscale_change_evidence_with_detector(
            &ring,
            200,
            &AcousticBoundaryHints::default(),
            super::AcousticFeatureAblation::FullV2,
            super::AcousticChangeDetectorMode::FixedSafeV1,
        );
        template.change_probability = 0.9;
        let mut selector =
            super::ChangePeakSelector::new(super::AcousticChangeDetectorMode::FixedSafeV1, 0.6);
        let mut emitted = 0;
        for frame_index in 0..50 {
            template.boundary_ms = frame_index as u64 * 10;
            emitted += usize::from(selector.push(frame_index, template.clone()).is_some());
        }
        emitted += usize::from(selector.finish().is_some());
        assert_eq!(emitted, 3);
    }

    #[test]
    fn short_turn_and_rapid_alternation_remain_distinct_after_peak_suppression() {
        let frames = (0..900)
            .map(|index| {
                let voice = match index {
                    0..300 | 360..620 | 680.. => 0.0,
                    _ => 1.0,
                };
                synthetic_feature(index, voice, false, matches!(index, 300 | 360 | 620 | 680))
            })
            .collect::<Vec<_>>();
        let (tracklets, summary) =
            segment_acoustic_frames(frames, &AcousticBoundaryHints::default(), || false)
                .expect("segment");
        for expected_ms in [3_000_u64, 3_600, 6_200, 6_800] {
            assert!(
                tracklets.iter().any(|tracklet| {
                    tracklet
                        .boundary_evidence
                        .as_ref()
                        .is_some_and(|evidence| evidence.boundary_ms.abs_diff(expected_ms) <= 120)
                }),
                "missing {expected_ms} ms boundary: {tracklets:#?}"
            );
        }
        assert!(summary.acoustic_change_count >= 4);
    }

    #[test]
    fn streaming_segmenter_memory_is_bounded_for_long_calls() {
        let hints = AcousticBoundaryHints::default();
        let mut segmenter = AcousticSegmenter::new(&hints).expect("segmenter");
        for index in 0..20_000 {
            segmenter
                .push(synthetic_feature(index, 0.0, false, false))
                .expect("streaming frame");
        }
        let (tracklets, summary, _) = segmenter.finish().expect("finish");
        assert_eq!(summary.input_frame_count, 20_000);
        assert_eq!(tracklets.len(), 1);
        assert!(
            summary.maximum_retained_frames <= 401,
            "segmenter retained {} frames",
            summary.maximum_retained_frames
        );
    }

    #[test]
    fn no_change_and_one_frame_noise_burst_do_not_create_turns() {
        let frames = (0..650)
            .map(|index| {
                synthetic_feature(
                    index,
                    if index == 325 { 1.0 } else { 0.0 },
                    false,
                    index == 325,
                )
            })
            .collect::<Vec<_>>();
        let (tracklets, _) =
            segment_acoustic_frames(frames, &AcousticBoundaryHints::default(), || false)
                .expect("segment");
        assert_eq!(tracklets.len(), 1);
    }

    #[test]
    fn short_vad_regions_remain_structural_boundaries() {
        let frames = (0..40)
            .map(|index| synthetic_feature(index, 0.0, false, false))
            .collect::<Vec<_>>();
        let hints = AcousticBoundaryHints {
            speech_regions_ms: vec![(0, 100), (100, 400)],
            ..AcousticBoundaryHints::default()
        };
        let (tracklets, summary) =
            segment_acoustic_frames(frames, &hints, || false).expect("segment");
        assert_eq!(tracklets.len(), 2, "{tracklets:#?}");
        assert_eq!(tracklets[0].frame_count, 10);
        assert!(
            tracklets[0]
                .boundary_evidence
                .as_ref()
                .is_some_and(|evidence| evidence.vad_boundary)
        );
        assert_eq!(summary.forced_boundary_count, 3);
    }

    #[test]
    fn vad_mask_excludes_non_speech_tracklets_from_clustering() {
        let frames = (0..40)
            .map(|index| {
                let speech = (10..30).contains(&index);
                synthetic_feature(index, 0.0, !speech, false)
            })
            .collect::<Vec<_>>();
        let hints = AcousticBoundaryHints {
            speech_regions_ms: vec![(100, 300)],
            ..AcousticBoundaryHints::default()
        };
        let (tracklets, _) = segment_acoustic_frames(frames, &hints, || false).expect("segment");
        assert_eq!(tracklets.len(), 1, "{tracklets:#?}");
        assert!(tracklets[0].start_ms >= 100);
        assert!(tracklets[0].end_ms <= 325);
        assert_eq!(tracklets[0].voiced_frame_count, tracklets[0].frame_count);
    }

    #[test]
    fn adjacent_tracklet_merge_uses_pooled_sample_variance() {
        let mut destination = profile_tracklet(0, 0, 20, 0.0, 0.0, 2);
        let mut source = profile_tracklet(1, 20, 40, 2.0, 4.0, 2);
        destination.voice_variance = [0.0; VOICE_VECTOR_DIMENSIONS];
        destination.channel_variance = [0.0; CHANNEL_VECTOR_DIMENSIONS];
        source.voice_variance = [0.0; VOICE_VECTOR_DIMENSIONS];
        source.channel_variance = [0.0; CHANNEL_VECTOR_DIMENSIONS];

        merge_tracklet_statistics(&mut destination, &source);

        assert_eq!(destination.frame_count, 4);
        assert!(destination.voice_mean.iter().all(|value| *value == 1.0));
        assert!(destination.channel_mean.iter().all(|value| *value == 2.0));
        assert!(
            destination
                .voice_variance
                .iter()
                .all(|value| (*value - 4.0 / 3.0).abs() < 1e-6)
        );
        assert!(
            destination
                .channel_variance
                .iter()
                .all(|value| (*value - 16.0 / 3.0).abs() < 1e-6)
        );

        let large_count = usize::MAX / 4;
        let mut large_destination = profile_tracklet(0, 0, 20, -1_000_000.0, -1_000_000.0, 0);
        let mut large_source = profile_tracklet(1, 20, 40, 1_000_000.0, 1_000_000.0, 0);
        large_destination.frame_count = large_count;
        large_source.frame_count = large_count;
        merge_tracklet_statistics(&mut large_destination, &large_source);
        assert!(
            large_destination
                .voice_variance
                .iter()
                .chain(large_destination.channel_variance.iter())
                .all(|value| value.is_finite()),
            "bounded statistics must not overflow pooled-variance intermediates"
        );
    }

    #[test]
    fn silence_gap_is_change_evidence_not_a_speaker_identity() {
        let frames = (0..700)
            .map(|index| {
                let silence = (300..340).contains(&index);
                synthetic_feature(index, if index >= 340 { 1.0 } else { 0.0 }, silence, false)
            })
            .collect::<Vec<_>>();
        let (tracklets, _) =
            segment_acoustic_frames(frames, &AcousticBoundaryHints::default(), || false)
                .expect("segment");
        assert!(tracklets.iter().any(|tracklet| {
            tracklet
                .boundary_evidence
                .as_ref()
                .is_some_and(|evidence| evidence.silence_gap)
        }));
    }

    #[test]
    fn same_voice_pause_does_not_become_a_speaker_change() {
        let hints = AcousticBoundaryHints::default();
        let mut segmenter = super::AcousticSegmenter::new(&hints).expect("segmenter");
        for index in 0..700 {
            let silence = (300..340).contains(&index);
            segmenter
                .push(synthetic_feature(index, 0.2, silence, false))
                .expect("frame");
        }
        let (_, _, emitted) = segmenter.finish().expect("finish");
        assert!(
            emitted.emitted.iter().all(|evidence| !evidence.silence_gap),
            "{emitted:#?}"
        );
    }

    #[test]
    fn gradual_drift_is_conservative_and_deterministic() {
        let make_frames = || {
            (0..800)
                .map(|index| synthetic_feature(index, index as f32 / 4_000.0, false, false))
                .collect::<Vec<_>>()
        };
        let first =
            segment_acoustic_frames(make_frames(), &AcousticBoundaryHints::default(), || false)
                .expect("first");
        let second =
            segment_acoustic_frames(make_frames(), &AcousticBoundaryHints::default(), || false)
                .expect("second");
        assert_eq!(first, second);
        assert!(first.0.len() <= 2);
    }

    #[test]
    fn segmentation_cancellation_is_bounded() {
        let frames = (0..800)
            .map(|index| synthetic_feature(index, 0.0, false, false))
            .collect::<Vec<_>>();
        let mut checks = 0usize;
        let error = segment_acoustic_frames(frames, &AcousticBoundaryHints::default(), || {
            checks += 1;
            checks >= 2
        })
        .expect_err("cancel");
        assert!(error.to_string().contains("segmentation cancelled"));
    }

    #[test]
    fn malformed_boundary_hints_fail_closed() {
        let hints = AcousticBoundaryHints {
            speech_regions_ms: vec![(100, 200), (150, 300)],
            ..AcousticBoundaryHints::default()
        };
        let error =
            segment_acoustic_frames(Vec::new(), &hints, || false).expect_err("overlapping VAD");
        assert!(error.to_string().contains("ordered, and disjoint"));
    }

    #[test]
    fn malformed_acoustic_frames_fail_before_segmentation() {
        let mut wrong_cadence = synthetic_feature(0, 0.0, false, false);
        wrong_cadence.start_ms = 1;
        let error =
            segment_acoustic_frames([wrong_cadence], &AcousticBoundaryHints::default(), || false)
                .expect_err("wrong frame cadence must fail");
        assert!(error.to_string().contains("exact v2 cadence"));

        let mut invalid_pitch = synthetic_feature(0, 0.0, false, false);
        invalid_pitch.voice.f0_hz = Some(-120.0);
        let error =
            segment_acoustic_frames([invalid_pitch], &AcousticBoundaryHints::default(), || false)
                .expect_err("negative pitch must fail");
        assert!(error.to_string().contains("internally consistent"));

        let mut overflowing_distance = synthetic_feature(0, 0.0, false, false);
        overflowing_distance.voice.cepstral_envelope[0] = f32::MAX;
        let error = segment_acoustic_frames(
            [overflowing_distance],
            &AcousticBoundaryHints::default(),
            || false,
        )
        .expect_err("finite values that overflow distance arithmetic must fail");
        assert!(error.to_string().contains("internally consistent"));

        let mut contradictory_quality = synthetic_feature(0, 0.0, false, false);
        contradictory_quality.channel.clipping_fraction = 1.0;
        let error = segment_acoustic_frames(
            [contradictory_quality],
            &AcousticBoundaryHints::default(),
            || false,
        )
        .expect_err("feature values and quality flags must agree");
        assert!(error.to_string().contains("internally consistent"));

        #[cfg(target_pointer_width = "64")]
        {
            let overflowing_index =
                usize::try_from((u64::MAX / 1_000 + 1) / ACOUSTIC_HOP_SAMPLES as u64)
                    .expect("64-bit frame index");
            let mut overflowing = synthetic_feature(0, 0.0, false, false);
            overflowing.frame_index = overflowing_index;
            let error =
                segment_acoustic_frames([overflowing], &AcousticBoundaryHints::default(), || false)
                    .expect_err("timestamp conversion overflow must fail");
            assert!(error.to_string().contains("cadence range"));
        }
    }

    fn profile_tracklet(
        index: usize,
        start_ms: u64,
        end_ms: u64,
        voice: f32,
        channel: f32,
        voiced_frames: usize,
    ) -> AcousticTracklet {
        let frame_count = usize::try_from((end_ms - start_ms) / 10).expect("fixture frame count");
        let valid_voice_frames = voiced_frames.min(frame_count);
        let voice_support = u32::try_from(valid_voice_frames).unwrap_or(u32::MAX);
        AcousticTracklet {
            tracklet_index: index,
            start_ms,
            end_ms,
            frame_count,
            voiced_frame_count: valid_voice_frames,
            identity_frame_count: valid_voice_frames,
            channel_frame_count: frame_count,
            voice_mean: [voice; VOICE_VECTOR_DIMENSIONS],
            voice_variance: [0.01; VOICE_VECTOR_DIMENSIONS],
            voice_valid: [valid_voice_frames > 0; VOICE_VECTOR_DIMENSIONS],
            voice_support: [voice_support; VOICE_VECTOR_DIMENSIONS],
            channel_mean: [channel; CHANNEL_VECTOR_DIMENSIONS],
            channel_variance: [0.01; CHANNEL_VECTOR_DIMENSIONS],
            channel_valid: true,
            channel_dimensions: CHANNEL_VECTOR_DIMENSIONS,
            change_confidence: 0.8,
            overlap_probability: 0.0,
            overlap_suspected: false,
            boundary_evidence: None,
        }
    }

    fn sequential_profile_tracklets(
        voices: &[f32],
        duration_ms: u64,
        voiced_frames: usize,
    ) -> Vec<AcousticTracklet> {
        voices
            .iter()
            .copied()
            .enumerate()
            .map(|(index, voice)| {
                let start_ms = u64::try_from(index)
                    .expect("fixture index")
                    .saturating_mul(duration_ms);
                profile_tracklet(
                    index,
                    start_ms,
                    start_ms.saturating_add(duration_ms),
                    voice,
                    0.0,
                    voiced_frames,
                )
            })
            .collect()
    }

    fn cluster_with_hard_count(
        tracklets: &[AcousticTracklet],
        count: u32,
    ) -> super::AcousticClusteringResult {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::HardConstraint { count },
            ..DiarizationRequest::default()
        };
        let enrollment = enroll_known_speaker_profiles(
            tracklets,
            &request,
            tracklets.last().map_or(0, |tracklet| tracklet.end_ms),
        )
        .expect("fixture enrollment");
        cluster_acoustic_tracklets(tracklets, &enrollment, &request.speaker_count, 512, || {
            false
        })
        .expect("hard-count clustering")
    }

    fn primary_voiced_occupancy(
        tracklets: &[AcousticTracklet],
        assignments: &[AcousticSpeakerAssignment],
    ) -> BTreeMap<String, (usize, usize, u64)> {
        let mut occupancy = BTreeMap::<String, (usize, usize, u64)>::new();
        for (tracklet, assignment) in tracklets.iter().zip(assignments) {
            let Some(speaker) = assignment.speaker_ref.as_ref() else {
                continue;
            };
            let entry = occupancy.entry(speaker.clone()).or_default();
            entry.0 += 1;
            entry.1 = entry.1.saturating_add(tracklet.voiced_frame_count);
            entry.2 = entry.2.saturating_add(
                u64::try_from(tracklet.voiced_frame_count)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(10),
            );
        }
        occupancy
    }

    fn known_interval(
        speaker_ref: &str,
        start_ms: u64,
        end_ms: u64,
        policy: KnownSpeakerPolicy,
    ) -> KnownSpeakerInterval {
        KnownSpeakerInterval {
            speaker_ref: speaker_ref.to_owned(),
            start_ms,
            end_ms,
            confidence: 0.9,
            policy,
            provenance: Some("agent-context".to_owned()),
        }
    }

    #[test]
    fn no_channel_tracklets_cannot_create_zero_valued_channel_profiles() {
        let mut tracklet = profile_tracklet(0, 0, 500, 0.0, 7.0, 50);
        tracklet.channel_valid = false;
        tracklet.channel_frame_count = 0;
        tracklet.channel_dimensions = 0;
        let request = DiarizationRequest {
            known_intervals: vec![known_interval(
                "speaker-a",
                0,
                500,
                KnownSpeakerPolicy::HardMustLink,
            )],
            enrollment_edge_guard_ms: 0,
            ..DiarizationRequest::default()
        };
        let enrollment = enroll_known_speaker_profiles(&[tracklet], &request, 500)
            .expect("voice-only enrollment");
        assert_eq!(enrollment.summaries.len(), 1);
        assert_eq!(enrollment.summaries[0].channel_profile_count, 0);
        assert_eq!(
            enrollment.profiles["speaker-a"].channel_dimensions, 0,
            "no-channel enrollment must remain channel-free"
        );
    }

    #[test]
    fn hard_hint_requires_usable_voiced_evidence_after_edge_guards() {
        let request = DiarizationRequest {
            known_intervals: vec![known_interval(
                "alice",
                0,
                1_000,
                KnownSpeakerPolicy::HardMustLink,
            )],
            ..DiarizationRequest::default()
        };
        let tracklets = vec![profile_tracklet(0, 200, 800, 0.0, 0.0, 0)];
        let error = enroll_known_speaker_profiles(&tracklets, &request, 1_000)
            .expect_err("empty hard enrollment");
        assert_eq!(error.code, ProfileEnrollmentCode::EmptyHardEnrollment);
    }

    #[test]
    fn edge_guards_exclude_boundary_bleed_from_hard_profile() {
        let request = DiarizationRequest {
            known_intervals: vec![known_interval(
                "alice",
                0,
                1_000,
                KnownSpeakerPolicy::HardMustLink,
            )],
            enrollment_edge_guard_ms: 100,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 0, 80, 5.0, 0.0, 8),
            profile_tracklet(1, 300, 700, 0.0, 0.0, 40),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 1_000).expect("enrollment");
        assert_eq!(enrollment.summaries[0].frame_count, 40);
        assert_eq!(
            enrollment.hard_assignments.get(&1),
            Some(&"alice".to_owned())
        );
        assert!(!enrollment.hard_assignments.contains_key(&0));
    }

    #[test]
    fn edge_guards_require_the_whole_tracklet_to_be_trusted() {
        let request = DiarizationRequest {
            known_intervals: vec![known_interval(
                "alice",
                0,
                1_000,
                KnownSpeakerPolicy::HardMustLink,
            )],
            enrollment_edge_guard_ms: 100,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            // Its midpoint is inside the guarded interval, but its leading edge is not.
            profile_tracklet(0, 50, 150, 5.0, 0.0, 10),
            profile_tracklet(1, 200, 700, 0.0, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 1_000).expect("enrollment");
        assert_eq!(enrollment.summaries[0].frame_count, 50);
        assert!(!enrollment.hard_assignments.contains_key(&0));
        assert_eq!(
            enrollment.hard_assignments.get(&1),
            Some(&"alice".to_owned())
        );
    }

    #[test]
    fn contradictory_soft_enrollment_is_rejected_with_evidence() {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::HardConstraint { count: 2 },
            known_intervals: vec![
                known_interval("alice", 0, 900, KnownSpeakerPolicy::HardMustLink),
                known_interval("alice", 1_000, 1_900, KnownSpeakerPolicy::SoftEnrollment),
            ],
            enrollment_edge_guard_ms: 50,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 200, 700, 0.0, 0.0, 50),
            profile_tracklet(1, 1_200, 1_700, 3.0, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 2_000).expect("enrollment");
        assert_eq!(enrollment.evidence[1].accepted_tracklet_count, 0);
        assert_eq!(enrollment.evidence[1].rejected_tracklet_count, 1);
        assert!(enrollment.evidence[1].contradiction_score.is_some());
        assert!(enrollment.soft_priors.is_empty());
    }

    #[test]
    fn one_voice_can_retain_multiple_channel_subprofiles() {
        let request = DiarizationRequest {
            known_intervals: vec![
                known_interval("alice", 0, 900, KnownSpeakerPolicy::HardMustLink),
                known_interval("alice", 1_000, 1_900, KnownSpeakerPolicy::HardMustLink),
            ],
            enrollment_edge_guard_ms: 50,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 200, 700, 0.1, 0.0, 50),
            profile_tracklet(1, 1_200, 1_700, 0.1, 1.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 2_000).expect("enrollment");
        assert_eq!(enrollment.summaries[0].channel_profile_count, 2);
    }

    #[test]
    fn enrollment_hygiene_quarantines_isolated_hard_outlier_without_changing_attribution() {
        let request = DiarizationRequest {
            known_intervals: vec![
                known_interval("alice", 0, 500, KnownSpeakerPolicy::HardMustLink),
                known_interval("alice", 500, 1_000, KnownSpeakerPolicy::HardMustLink),
                known_interval("alice", 1_000, 1_500, KnownSpeakerPolicy::HardMustLink),
                known_interval("alice", 1_500, 2_000, KnownSpeakerPolicy::HardMustLink),
            ],
            enrollment_edge_guard_ms: 0,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 0, 500, 0.00, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 0.03, 0.0, 50),
            profile_tracklet(2, 1_000, 1_500, 0.06, 0.0, 50),
            profile_tracklet(3, 1_500, 2_000, 3.00, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 2_000).expect("enrollment");
        assert_eq!(enrollment.hard_assignments.len(), 4);
        assert_eq!(
            enrollment.hard_assignments.get(&3).map(String::as_str),
            Some("alice"),
            "training quarantine must not weaken a hard timestamp attribution"
        );
        assert_eq!(enrollment.summaries[0].frame_count, 150);
        assert_eq!(enrollment.summaries[0].training_quarantined_count, 1);
        let quarantined = enrollment
            .training_evidence
            .iter()
            .find(|item| item.tracklet_index == 3)
            .expect("outlier decision");
        assert_eq!(
            quarantined.disposition,
            ProfileTrainingDisposition::Quarantined
        );
        assert_eq!(
            quarantined.reason,
            ProfileTrainingReason::LeaveOneOutInconsistent
        );
        assert!(quarantined.hard_attribution);
        assert_eq!(quarantined.applied_weight, 0.0);
    }

    #[test]
    fn enrollment_hygiene_retains_repeated_secondary_voice_as_bounded_mixture_mode() {
        let request = DiarizationRequest {
            known_intervals: vec![
                known_interval("alice", 0, 500, KnownSpeakerPolicy::HardMustLink),
                known_interval("alice", 500, 1_000, KnownSpeakerPolicy::HardMustLink),
                known_interval("alice", 1_000, 1_500, KnownSpeakerPolicy::HardMustLink),
                known_interval("alice", 1_500, 2_000, KnownSpeakerPolicy::HardMustLink),
            ],
            enrollment_edge_guard_ms: 0,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 0.1, 0.0, 50),
            profile_tracklet(2, 1_000, 1_500, 3.0, 0.0, 50),
            profile_tracklet(3, 1_500, 2_000, 3.1, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 2_000).expect("enrollment");
        assert_eq!(enrollment.summaries[0].voice_profile_count, 2);
        assert_eq!(enrollment.summaries[0].training_quarantined_count, 0);
        assert_eq!(enrollment.profiles["alice"].voice_subprofiles.len(), 2);
        assert!(
            enrollment
                .training_evidence
                .iter()
                .all(|item| item.disposition != ProfileTrainingDisposition::Quarantined)
        );
    }

    #[test]
    fn enrollment_hygiene_downweights_and_audits_marginal_voiced_coverage() {
        let request = DiarizationRequest {
            known_intervals: vec![known_interval(
                "alice",
                0,
                500,
                KnownSpeakerPolicy::HardMustLink,
            )],
            enrollment_edge_guard_ms: 0,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![profile_tracklet(0, 0, 500, 0.0, 0.0, 15)];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 500).expect("enrollment");
        assert_eq!(enrollment.summaries[0].training_downweighted_count, 1);
        assert_eq!(
            enrollment.training_evidence[0].disposition,
            ProfileTrainingDisposition::Downweighted
        );
        assert_eq!(
            enrollment.training_evidence[0].reason,
            ProfileTrainingReason::LowVoicedCoverage
        );
        assert_eq!(
            enrollment.hard_assignments.get(&0).map(String::as_str),
            Some("alice")
        );
        assert!(!enrollment.metric_adaptation.enabled);
        assert_eq!(
            enrollment.metric_adaptation.fallback,
            Some(ProfileMetricAdaptationFallback::InsufficientSpeakers)
        );
        assert!(
            enrollment
                .voice_dimension_weights
                .iter()
                .all(|weight| *weight == 1.0)
        );
    }

    #[test]
    fn enrollment_hygiene_metric_adaptation_is_bounded_and_reversible() {
        let request = DiarizationRequest {
            known_intervals: vec![
                known_interval("alice", 0, 500, KnownSpeakerPolicy::HardMustLink),
                known_interval("alice", 500, 1_000, KnownSpeakerPolicy::HardMustLink),
                known_interval("alice", 1_000, 1_500, KnownSpeakerPolicy::HardMustLink),
                known_interval("bob", 1_500, 2_000, KnownSpeakerPolicy::HardMustLink),
                known_interval("bob", 2_000, 2_500, KnownSpeakerPolicy::HardMustLink),
                known_interval("bob", 2_500, 3_000, KnownSpeakerPolicy::HardMustLink),
            ],
            enrollment_edge_guard_ms: 0,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 0, 500, 0.00, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 0.03, 0.0, 50),
            profile_tracklet(2, 1_000, 1_500, 0.06, 0.0, 50),
            profile_tracklet(3, 1_500, 2_000, 2.00, 0.0, 50),
            profile_tracklet(4, 2_000, 2_500, 2.03, 0.0, 50),
            profile_tracklet(5, 2_500, 3_000, 2.06, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 3_000).expect("enrollment");
        assert!(enrollment.metric_adaptation.enabled);
        assert_eq!(enrollment.metric_adaptation.fallback, None);
        assert_eq!(enrollment.metric_adaptation.enrolled_speaker_count, 2);
        assert_eq!(enrollment.metric_adaptation.training_observation_count, 6);
        assert!(
            enrollment
                .voice_dimension_weights
                .iter()
                .all(|weight| (0.9375..=1.0625).contains(weight))
        );
        assert!(
            enrollment.metric_adaptation.maximum_absolute_weight_delta <= 0.0625 + f32::EPSILON
        );
        assert!(enrollment.metric_adaptation.adapted_dimension_count > 0);
    }

    #[test]
    fn duration_aware_temporal_prior_preserves_evidence_backed_short_turns() {
        let low_evidence_short = super::duration_aware_switch_penalty(80, 80, 0, 0.10, false);
        let high_evidence_short = super::duration_aware_switch_penalty(80, 80, 0, 0.95, false);
        let gap_backed_short = super::duration_aware_switch_penalty(80, 80, 500, 0.10, false);
        assert!(
            high_evidence_short < low_evidence_short,
            "strong acoustic boundary evidence must override the short-run prior"
        );
        assert!(
            gap_backed_short < low_evidence_short,
            "a real silence gap must make a speaker transition less costly"
        );
        assert!(low_evidence_short <= 0.45);
        assert!(high_evidence_short >= 0.01);
    }

    #[test]
    fn duration_aware_temporal_prior_handles_unknown_transitions_conservatively() {
        let known_switch = super::duration_aware_switch_penalty(1_000, 500, 0, 0.0, false);
        let unknown_switch = super::duration_aware_switch_penalty(1_000, 500, 0, 0.0, true);
        let evidence_backed_unknown =
            super::duration_aware_switch_penalty(1_000, 500, 0, 1.0, true);
        assert!(unknown_switch < known_switch);
        assert!(evidence_backed_unknown < unknown_switch);
    }

    #[test]
    fn overlap_dual_periodicity_distinguishes_a_nonharmonic_voice_mixture() {
        let pure = sine_wave(120.0, 0.025, 0.35);
        let mut mixture = pure.clone();
        for (sample, second_voice) in mixture.iter_mut().zip(sine_wave(205.0, 0.025, 0.30)) {
            *sample += second_voice;
        }
        let (_, _, pure_overlap) = super::estimate_f0(&pure, -12.0);
        let (_, _, mixture_overlap) = super::estimate_f0(&mixture, -9.0);
        assert!(
            mixture_overlap > pure_overlap,
            "non-harmonic dual periodicity should exceed the single-source control"
        );
    }

    #[test]
    fn clipping_and_transients_do_not_masquerade_as_overlap() {
        let frames = (0..20)
            .map(|index| {
                let mut frame = synthetic_feature(index, 0.0, false, true);
                frame.overlap_probability = 1.0;
                frame.quality.clipped = index % 2 == 0;
                frame.channel.clipping_fraction = if frame.quality.clipped { 0.01 } else { 0.0 };
                frame
            })
            .collect::<Vec<_>>();
        let (tracklets, _) =
            segment_acoustic_frames(frames, &AcousticBoundaryHints::default(), || false)
                .expect("segmentation");
        assert!(tracklets.iter().all(|tracklet| !tracklet.overlap_suspected));
    }

    #[test]
    fn streaming_feature_chunks_are_exactly_batch_equivalent() {
        let mut samples = sine_wave(137.0, 1.0, 0.30);
        for (sample, overtone) in samples.iter_mut().zip(sine_wave(274.0, 1.0, 0.08)) {
            *sample += overtone;
        }
        let mut batch_frames = Vec::new();
        let batch_summary = extract_acoustic_features(
            &samples,
            || false,
            |frame| {
                batch_frames.push(frame);
                Ok(())
            },
        )
        .expect("batch features");

        let chunk_sizes = [1_usize, 159, 160, 399, 7, 401, 997];
        let mut stream = AcousticFeatureStream::new();
        let mut stream_frames = Vec::new();
        let mut offset = 0usize;
        let mut chunk_index = 0usize;
        while offset < samples.len() {
            let end = offset
                .saturating_add(chunk_sizes[chunk_index % chunk_sizes.len()])
                .min(samples.len());
            stream
                .push_chunk(&samples[offset..end], &mut || false, &mut |frame| {
                    stream_frames.push(frame);
                    Ok(())
                })
                .expect("stream chunk");
            offset = end;
            chunk_index += 1;
        }
        let stream_summary = stream.finish();
        assert_eq!(stream_frames, batch_frames);
        assert_eq!(stream_summary.frame_count, batch_summary.frame_count);
        assert_eq!(
            stream_summary.voiced_frame_count,
            batch_summary.voiced_frame_count
        );
        assert_eq!(
            stream_summary.reliable_pitch_frame_count,
            batch_summary.reliable_pitch_frame_count
        );
        assert_eq!(
            stream_summary.high_information_frame_count,
            batch_summary.high_information_frame_count
        );
        assert_eq!(
            stream_summary.missing_pitch_frame_count,
            batch_summary.missing_pitch_frame_count
        );
        assert_eq!(
            stream_summary.low_energy_frame_count,
            batch_summary.low_energy_frame_count
        );
    }

    #[test]
    fn streaming_feature_state_is_bounded_and_cancel_correct() {
        let samples = sine_wave(120.0, 1.0, 0.25);
        let mut stream = AcousticFeatureStream::new();
        let mut cancellation_checks = 0usize;
        let mut emitted = 0usize;
        let error = stream
            .push_chunk(
                &samples,
                &mut || {
                    cancellation_checks += 1;
                    cancellation_checks >= 2
                },
                &mut |_| {
                    emitted += 1;
                    Ok(())
                },
            )
            .expect_err("second scheduled cancellation check");
        assert!(matches!(error, FwError::Cancelled(_)));
        assert_eq!(emitted, ACOUSTIC_CANCELLATION_INTERVAL_FRAMES);

        let mut short = AcousticFeatureStream::new();
        short
            .push_chunk(&samples[..800], &mut || false, &mut |_| Ok(()))
            .expect("short stream");
        let short_bound = short.finish().retained_state_bytes_upper_bound;
        let mut long = AcousticFeatureStream::new();
        long.push_chunk(&samples, &mut || false, &mut |_| Ok(()))
            .expect("long stream");
        let long_bound = long.finish().retained_state_bytes_upper_bound;
        assert_eq!(short_bound, long_bound);
        assert!(long_bound < 16 * 1024);
    }

    #[test]
    fn streaming_segmentation_matches_batch_across_chunk_boundaries() {
        let mut samples = sine_wave(125.0, 1.5, 0.25);
        let second = sine_wave(220.0, 0.75, 0.20);
        for (sample, replacement) in samples.iter_mut().skip(12_000).zip(second) {
            *sample = replacement;
        }
        let hints = AcousticBoundaryHints {
            speech_regions_ms: vec![(0, 1_500)],
            word_boundaries_ms: vec![750],
            tiny_diarize_boundaries_ms: Vec::new(),
        };
        let mut batch_frames = Vec::new();
        let batch_features = extract_acoustic_features(
            &samples,
            || false,
            |frame| {
                batch_frames.push(frame);
                Ok(())
            },
        )
        .expect("batch features");
        let (batch_tracklets, batch_segmentation) =
            segment_acoustic_frames(batch_frames, &hints, || false).expect("batch segmentation");

        let mut stream = AcousticSegmentationStream::new(
            &hints,
            &[],
            super::AcousticFeatureAblation::FullV2,
            super::AcousticChangeDetectorMode::FixedSafeV1,
        )
        .expect("stream");
        for chunk in samples.chunks(733) {
            stream
                .push_chunk(chunk, &mut || false)
                .expect("segmentation chunk");
        }
        let (stream_features, stream_tracklets, stream_segmentation) =
            stream.finish().expect("stream finish");
        assert_eq!(stream_features.frame_count, batch_features.frame_count);
        assert_eq!(stream_tracklets, batch_tracklets);
        assert_eq!(stream_segmentation, batch_segmentation);
    }

    #[test]
    fn overlapping_same_speaker_hints_do_not_double_count_a_tracklet() {
        let request = DiarizationRequest {
            known_intervals: vec![
                known_interval("alice", 0, 900, KnownSpeakerPolicy::HardMustLink),
                known_interval("alice", 100, 800, KnownSpeakerPolicy::HardMustLink),
            ],
            enrollment_edge_guard_ms: 50,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![profile_tracklet(0, 200, 700, 0.1, 0.0, 50)];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 1_000).expect("enrollment");
        assert_eq!(enrollment.summaries[0].frame_count, 50);
        assert_eq!(enrollment.summaries[0].voiced_duration_ms, 500);
        assert_eq!(enrollment.hard_assignments.len(), 1);
        assert_eq!(
            enrollment
                .evidence
                .iter()
                .map(|evidence| evidence.accepted_tracklet_count)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
    }

    #[test]
    fn canonical_hint_hash_is_order_independent_and_provenance_is_not_weight() {
        let first = known_interval("alice", 0, 900, KnownSpeakerPolicy::SoftEnrollment);
        let mut second = known_interval("bob", 1_000, 1_900, KnownSpeakerPolicy::SoftEnrollment);
        second.provenance = Some("untrusted free form".to_owned());
        let tracklets = vec![
            profile_tracklet(0, 200, 700, 0.0, 0.0, 50),
            profile_tracklet(1, 1_200, 1_700, 1.0, 1.0, 50),
        ];
        let request_a = DiarizationRequest {
            known_intervals: vec![first.clone(), second.clone()],
            enrollment_edge_guard_ms: 50,
            ..DiarizationRequest::default()
        };
        let request_b = DiarizationRequest {
            known_intervals: vec![second.clone(), first.clone()],
            enrollment_edge_guard_ms: 50,
            ..DiarizationRequest::default()
        };
        let mut changed_provenance = second;
        changed_provenance.provenance = Some("query:content-bound-id".to_owned());
        let request_c = DiarizationRequest {
            known_intervals: vec![first, changed_provenance],
            enrollment_edge_guard_ms: 50,
            ..DiarizationRequest::default()
        };
        let enrollment_a =
            enroll_known_speaker_profiles(&tracklets, &request_a, 2_000).expect("enrollment a");
        let enrollment_b =
            enroll_known_speaker_profiles(&tracklets, &request_b, 2_000).expect("enrollment b");
        let enrollment_c =
            enroll_known_speaker_profiles(&tracklets, &request_c, 2_000).expect("enrollment c");
        assert_eq!(
            enrollment_a.hint_document_sha256,
            enrollment_b.hint_document_sha256
        );
        assert_eq!(enrollment_a.summaries, enrollment_b.summaries);
        assert_ne!(
            enrollment_a.hint_document_sha256, enrollment_c.hint_document_sha256,
            "provenance is immutable hash input even though it never changes acoustic weight"
        );
        assert_eq!(enrollment_a.summaries, enrollment_c.summaries);
    }

    #[test]
    fn distinct_hard_references_create_cannot_links() {
        let request = DiarizationRequest {
            known_intervals: vec![
                known_interval("alice", 0, 900, KnownSpeakerPolicy::HardMustLink),
                known_interval("bob", 1_000, 1_900, KnownSpeakerPolicy::HardMustLink),
            ],
            enrollment_edge_guard_ms: 50,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 200, 700, 0.0, 0.0, 50),
            profile_tracklet(1, 1_200, 1_700, 1.0, 1.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 2_000).expect("enrollment");
        assert!(
            enrollment
                .cannot_links
                .contains(&("alice".to_owned(), "bob".to_owned()))
        );
    }

    #[test]
    fn hard_anchors_survive_clustering_and_smoothing_exactly() {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::HardConstraint { count: 2 },
            known_intervals: vec![
                known_interval("alice", 0, 900, KnownSpeakerPolicy::HardMustLink),
                known_interval("bob", 1_000, 1_900, KnownSpeakerPolicy::HardMustLink),
            ],
            enrollment_edge_guard_ms: 50,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 200, 700, 0.0, 0.0, 50),
            profile_tracklet(1, 1_200, 1_700, 2.0, 1.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 2_000).expect("enrollment");
        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("cluster");
        assert_eq!(result.assignments[0].speaker_ref.as_deref(), Some("alice"));
        assert_eq!(result.assignments[1].speaker_ref.as_deref(), Some("bob"));
        assert!(
            result
                .assignments
                .iter()
                .all(|value| value.hard_attribution)
        );
        assert!(result.constraints_satisfied);
    }

    #[test]
    fn generated_labels_never_collide_with_hard_speaker_references() {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::HardConstraint { count: 2 },
            known_intervals: vec![
                known_interval("SPEAKER_00", 0, 500, KnownSpeakerPolicy::HardMustLink),
                known_interval("SPEAKER_00", 1_000, 1_500, KnownSpeakerPolicy::HardMustLink),
            ],
            enrollment_edge_guard_ms: 0,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 3.0, 0.0, 50),
            profile_tracklet(2, 1_000, 1_500, 0.0, 0.0, 50),
            profile_tracklet(3, 1_500, 2_000, 3.0, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 2_000).expect("enrollment");
        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("cluster");
        assert_eq!(
            result.assignments[0].speaker_ref.as_deref(),
            Some("SPEAKER_00")
        );
        assert_eq!(
            result.assignments[1].speaker_ref.as_deref(),
            Some("SPEAKER_01")
        );
        assert_eq!(
            result.assignments[2].speaker_ref.as_deref(),
            Some("SPEAKER_00")
        );
        assert_eq!(
            result.assignments[3].speaker_ref.as_deref(),
            Some("SPEAKER_01")
        );
        assert_eq!(result.detected_speakers, 2);
        assert!(result.constraints_satisfied);
    }

    #[test]
    fn generated_labels_reserve_rejected_soft_speaker_references() {
        let request = DiarizationRequest {
            known_intervals: vec![known_interval(
                "SPEAKER_00",
                0,
                100,
                KnownSpeakerPolicy::SoftEnrollment,
            )],
            enrollment_edge_guard_ms: 50,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 200, 700, 0.0, 0.0, 50),
            profile_tracklet(1, 800, 1_300, 0.0, 0.0, 50),
        ];
        let mut enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 1_500).expect("enrollment");
        assert_eq!(enrollment.evidence[0].usable_tracklet_count, 0);
        enrollment.evidence.clear();

        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("cluster");
        assert_eq!(result.profiles[0].speaker_ref, "SPEAKER_01");
    }

    #[test]
    fn compatible_bounded_solution_survives_exhausted_cannot_link_heap() {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::Range {
                minimum: 1,
                maximum: 3,
            },
            known_intervals: vec![
                known_interval("alice", 0, 500, KnownSpeakerPolicy::HardMustLink),
                known_interval("bob", 500, 1_000, KnownSpeakerPolicy::HardMustLink),
            ],
            enrollment_edge_guard_ms: 0,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 1.0, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 1_000).expect("enrollment");
        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("the two anchored speakers already satisfy the bounded request");
        assert_eq!(result.detected_speakers, 2);
        assert!(result.constraints_satisfied);
        assert!(result.merge_trace.is_empty());
    }

    #[test]
    fn soft_enrollment_influences_but_does_not_force_unrelated_speech() {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::HardConstraint { count: 2 },
            known_intervals: vec![known_interval(
                "alice",
                0,
                900,
                KnownSpeakerPolicy::SoftEnrollment,
            )],
            enrollment_edge_guard_ms: 50,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 200, 700, 0.0, 0.0, 50),
            profile_tracklet(1, 1_200, 1_700, 3.0, 0.0, 50),
            profile_tracklet(2, 1_800, 2_300, 0.02, 0.0, 50),
            profile_tracklet(3, 2_400, 2_900, 3.02, 0.0, 50),
            profile_tracklet(4, 3_000, 3_500, -0.02, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 3_500).expect("enrollment");
        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("cluster");
        assert_eq!(result.assignments[0].speaker_ref.as_deref(), Some("alice"));
        assert_ne!(result.assignments[1].speaker_ref.as_deref(), Some("alice"));
        assert!(
            result
                .assignments
                .iter()
                .all(|assignment| !assignment.hard_attribution)
        );
    }

    #[test]
    fn hard_count_is_enforced_while_soft_range_permits_disagreement() {
        let request = DiarizationRequest::default();
        let tracklets =
            sequential_profile_tracklets(&[0.0, 2.0, 4.0, 0.0, 2.0, 4.0, 0.0, 2.0, 4.0], 500, 50);
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 4_500).expect("enrollment");
        let exact = SpeakerCountRequest::HardConstraint { count: 3 };
        let exact_result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, &exact, 512, || false)
                .expect("exact clustering");
        assert_eq!(exact_result.detected_speakers, 3);
        assert!(exact_result.constraints_satisfied);

        let soft_range = SpeakerCountRequest::Range {
            minimum: 1,
            maximum: 1,
        };
        let ranged_result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, &soft_range, 512, || false)
                .expect("soft-range clustering");
        assert_eq!(
            ranged_result.detected_speakers, 3,
            "soft range must not force three recurrent separated regimes into one speaker"
        );
        assert!(ranged_result.constraints_satisfied);
    }

    #[test]
    fn long_imbalanced_three_speaker_fixture_retains_supported_minority_speakers() {
        let tracklets = (0..90)
            .map(|index| {
                let (voice, channel) = match index % 15 {
                    5 => (4.0, -1.5),
                    10 => (-4.0, 0.0),
                    _ => (0.0, 1.5),
                };
                let start_ms = index * 500;
                profile_tracklet(
                    usize::try_from(index).expect("fixture index"),
                    start_ms,
                    start_ms + 500,
                    voice,
                    channel,
                    50,
                )
            })
            .collect::<Vec<_>>();
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::HardConstraint { count: 3 },
            known_intervals: vec![
                known_interval("near", 0, 500, KnownSpeakerPolicy::SoftEnrollment),
                known_interval("far", 2_500, 3_000, KnownSpeakerPolicy::SoftEnrollment),
                known_interval("minority", 5_000, 5_500, KnownSpeakerPolicy::SoftEnrollment),
            ],
            enrollment_edge_guard_ms: 0,
            ..DiarizationRequest::default()
        };
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 45_000).expect("enrollment");
        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("cluster long imbalanced synthetic call");
        let occupancy = primary_voiced_occupancy(&tracklets, &result.assignments);

        assert!(
            result.constraints_satisfied,
            "condition=long-near-far-imbalanced requested=3 detected={} evidence={:#?}",
            result.detected_speakers, result.speaker_evidence
        );
        assert_eq!(result.detected_speakers, 3);
        assert_eq!(occupancy.len(), 3);
        assert!(
            result.dominant_speaker_share < 0.90,
            "dominant speaker collapsed the call: share={} occupancy={occupancy:#?}",
            result.dominant_speaker_share
        );
        let total_duration = occupancy
            .values()
            .map(|(_, _, duration_ms)| *duration_ms)
            .sum::<u64>();
        let minority_duration = occupancy
            .values()
            .map(|(_, _, duration_ms)| *duration_ms)
            .min()
            .expect("three supported speakers");
        assert!(
            minority_duration * 20 >= total_duration,
            "minority speaker lost despite 6.7% recurring support: {occupancy:#?}"
        );
        assert!(
            result
                .speaker_evidence
                .iter()
                .filter(|evidence| evidence.supported)
                .all(|evidence| evidence.independent_tracklet_count >= 2)
        );
    }

    #[test]
    fn hard_count_with_only_n_minus_one_supported_speakers_fails_closed() {
        let tracklets = sequential_profile_tracklets(&[0.0, 4.0, 0.0, 4.0, 0.0, 4.0], 500, 50);
        let result = cluster_with_hard_count(&tracklets, 3);
        let occupancy = primary_voiced_occupancy(&tracklets, &result.assignments);

        assert!(
            !result.constraints_satisfied,
            "condition=n-minus-one-supported requested=3 detected={} evidence={:#?}",
            result.detected_speakers, result.speaker_evidence
        );
        assert!(
            result.detected_speakers <= 2,
            "a third label must not be fabricated from two acoustic regimes: {occupancy:#?}"
        );
        assert_eq!(
            occupancy.len(),
            result.detected_speakers,
            "detected count must describe authoritative primary occupancy"
        );
    }

    #[test]
    fn hard_count_does_not_promote_short_outliers() {
        let mut tracklets = sequential_profile_tracklets(&[0.0_f32; 20], 500, 50);
        let next_start = tracklets.last().expect("dominant fixture").end_ms;
        tracklets.push(profile_tracklet(
            20,
            next_start,
            next_start + 100,
            -8.0,
            0.0,
            10,
        ));
        tracklets.push(profile_tracklet(
            21,
            next_start + 100,
            next_start + 200,
            8.0,
            0.0,
            10,
        ));

        let result = cluster_with_hard_count(&tracklets, 3);
        assert!(
            !result.constraints_satisfied,
            "short outliers cannot satisfy a hard count: {:#?}",
            result.speaker_evidence
        );
        assert!(
            result.detected_speakers <= 1,
            "short outliers became phantom speakers: {:#?}",
            result.assignments
        );
        assert!(
            result.assignments[20..]
                .iter()
                .all(|assignment| assignment.speaker_ref.is_none()),
            "short outliers must remain UNKNOWN"
        );
    }

    #[test]
    fn hard_count_does_not_accept_a_single_self_validating_interval() {
        let tracklets = vec![profile_tracklet(0, 0, 3_000, 0.0, 0.0, 300)];
        let result = cluster_with_hard_count(&tracklets, 1);

        assert!(!result.constraints_satisfied);
        assert_eq!(result.detected_speakers, 0);
        assert!(result.assignments[0].speaker_ref.is_none());
        assert_eq!(result.speaker_evidence.len(), 1);
        assert!(
            result.speaker_evidence[0]
                .reasons
                .contains(&SpeakerEvidenceReason::InsufficientIndependentRecurrence)
        );
    }

    #[test]
    fn hard_count_with_identical_features_is_unsatisfied_and_deterministic() {
        let tracklets = sequential_profile_tracklets(&[0.0; 6], 500, 50);
        let first = cluster_with_hard_count(&tracklets, 3);
        let second = cluster_with_hard_count(&tracklets, 3);

        assert_eq!(
            first, second,
            "identical evidence must replay deterministically"
        );
        assert!(!first.constraints_satisfied);
        assert!(
            first.detected_speakers <= 1,
            "identical observations cannot support three identities: {:#?}",
            first.speaker_evidence
        );
    }

    #[test]
    fn increasing_hard_count_cannot_restore_constraint_satisfaction() {
        let tracklets = sequential_profile_tracklets(&[0.0; 6], 500, 50);
        let one = cluster_with_hard_count(&tracklets, 1);
        assert!(
            one.constraints_satisfied,
            "one homogeneous regime should satisfy K=1: {:#?}",
            one.speaker_evidence
        );

        let mut became_unsatisfied = false;
        for count in 2..=5 {
            let result = cluster_with_hard_count(&tracklets, count);
            if !result.constraints_satisfied {
                became_unsatisfied = true;
            }
            assert!(
                !became_unsatisfied || !result.constraints_satisfied,
                "adding unsupported K restored satisfaction at K={count}: {:#?}",
                result.speaker_evidence
            );
            assert!(
                !result.constraints_satisfied,
                "identical evidence must explicitly reject unsupported K={count}"
            );
        }
    }

    #[test]
    fn satisfied_count_requires_assignment_occupancy_evidence() {
        let tracklets = sequential_profile_tracklets(
            &[0.0, 3.0, 6.0, 0.0, 3.0, 6.0, 0.0, 3.0, 6.0, 0.0, 3.0, 6.0],
            500,
            50,
        );
        let result = cluster_with_hard_count(&tracklets, 3);
        let occupancy = primary_voiced_occupancy(&tracklets, &result.assignments);

        assert!(
            result.constraints_satisfied,
            "three recurrent separated regimes should satisfy K=3: {:#?}",
            result.speaker_evidence
        );
        assert_eq!(result.detected_speakers, 3);
        for evidence in &result.speaker_evidence {
            if !evidence.supported {
                continue;
            }
            let actual = occupancy
                .get(&evidence.speaker_ref)
                .expect("every supported speaker has authoritative occupancy");
            assert_eq!(evidence.assigned_tracklet_count, actual.0);
            assert_eq!(evidence.voiced_frame_count, actual.1);
            assert_eq!(evidence.voiced_duration_ms, actual.2);
            assert!(evidence.mean_assignment_confidence >= super::MIN_SPEAKER_EVIDENCE_CONFIDENCE);
        }
        assert_eq!(
            result
                .speaker_evidence
                .iter()
                .filter(|evidence| evidence.supported)
                .count(),
            result.detected_speakers
        );
    }

    #[test]
    fn infeasible_speaker_count_reports_unsatisfied_instead_of_aborting() {
        let request = DiarizationRequest::default();
        let tracklets = vec![profile_tracklet(0, 0, 500, 0.0, 0.0, 50)];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 500).expect("enrollment");
        let impossible = SpeakerCountRequest::HardConstraint { count: 2 };
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, &impossible, 512, || false)
                .expect("insufficient evidence is a fallback state, not a malformed request");
        assert!(!result.constraints_satisfied);
        assert!(result.detected_speakers <= 1);
    }

    #[test]
    fn malformed_speaker_count_remains_an_error_when_evidence_is_insufficient() {
        let tracklets = vec![profile_tracklet(0, 0, 500, 0.0, 0.0, 50)];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &DiarizationRequest::default(), 500)
                .expect("enrollment");
        let malformed = SpeakerCountRequest::Range {
            minimum: 3,
            maximum: 2,
        };
        let error = cluster_acoustic_tracklets(&tracklets, &enrollment, &malformed, 512, || false)
            .expect_err("malformed constraints must not become an evidence fallback");
        assert!(error.to_string().contains("minimum <= maximum"));
    }

    #[test]
    fn impossible_exact_coverage_does_not_leave_partial_forced_assignments() {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::HardConstraint { count: 2 },
            known_intervals: vec![known_interval(
                "known",
                0,
                500,
                KnownSpeakerPolicy::SoftEnrollment,
            )],
            enrollment_edge_guard_ms: 0,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![profile_tracklet(0, 0, 500, 0.0, 0.0, 50)];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 500).expect("enrollment");
        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("coverage failure must remain a typed unsatisfied result");
        assert!(!result.constraints_satisfied);
        assert_eq!(result.assignments.len(), 1);
        assert!(result.detected_speakers <= 1);
    }

    #[test]
    fn merge_trace_stops_at_the_selected_cluster_solution() {
        let request = DiarizationRequest::default();
        let tracklets = vec![
            profile_tracklet(0, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 0.01, 0.0, 50),
            profile_tracklet(2, 1_000, 1_500, 8.0, 0.0, 50),
            profile_tracklet(3, 1_500, 2_000, 0.01, 0.0, 50),
            profile_tracklet(4, 2_000, 2_500, 8.0, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 2_500).expect("enrollment");
        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("cluster");
        assert_eq!(result.profiles.len(), 2, "{result:#?}");
        assert_eq!(
            result
                .merge_trace
                .last()
                .map(|step| step.remaining_clusters),
            Some(result.profiles.len()),
            "the audit trace must omit exploratory merges past the selected solution"
        );
    }

    #[test]
    fn voice_identity_outweighs_a_large_channel_change() {
        let request = DiarizationRequest::default();
        let tracklets = vec![
            profile_tracklet(0, 0, 500, 0.25, -1.0, 50),
            profile_tracklet(1, 500, 1_000, 0.25, 1.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 1_000).expect("enrollment");
        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("cluster");
        assert_eq!(result.detected_speakers, 1);
        assert_eq!(
            result.assignments[0].speaker_ref,
            result.assignments[1].speaker_ref
        );
    }

    #[test]
    fn probabilistic_scoring_merges_channel_variants_but_separates_voices() {
        let request = DiarizationRequest::default();
        let same_voice = vec![
            profile_tracklet(0, 0, 500, 0.25, -1.0, 50),
            profile_tracklet(1, 500, 1_000, 0.25, 1.0, 50),
        ];
        let same_enrollment =
            enroll_known_speaker_profiles(&same_voice, &request, 1_000).expect("enrollment");
        let same_result = super::cluster_acoustic_tracklets_with_mode(
            &same_voice,
            &same_enrollment,
            &request.speaker_count,
            512,
            super::AcousticClusteringMode::ProbabilisticV1,
            || false,
        )
        .expect("probabilistic same-speaker clustering");
        assert_eq!(
            same_result.executed_mode,
            super::AcousticClusteringMode::ProbabilisticV1
        );
        assert_eq!(same_result.detected_speakers, 1, "{same_result:#?}");
        assert_eq!(same_result.bootstrap_stability, 1.0);
        assert!(
            same_result
                .merge_trace
                .iter()
                .all(|step| step.channel_distance.is_some()
                    && step.same_speaker_probability.is_some())
        );

        let different_voices = vec![
            profile_tracklet(0, 0, 500, -1.5, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 1.5, 0.0, 50),
            profile_tracklet(2, 1_000, 1_500, -1.5, 0.0, 50),
            profile_tracklet(3, 1_500, 2_000, 1.5, 0.0, 50),
        ];
        let different_enrollment =
            enroll_known_speaker_profiles(&different_voices, &request, 2_000).expect("enrollment");
        let different_result = super::cluster_acoustic_tracklets_with_mode(
            &different_voices,
            &different_enrollment,
            &request.speaker_count,
            512,
            super::AcousticClusteringMode::ProbabilisticV1,
            || false,
        )
        .expect("probabilistic different-speaker clustering");
        assert_eq!(
            different_result.executed_mode,
            super::AcousticClusteringMode::ProbabilisticV1
        );
        assert_eq!(
            different_result.detected_speakers, 2,
            "{different_result:#?}"
        );
    }

    #[test]
    fn probabilistic_count_selection_reverses_many_false_boundaries() {
        let request = DiarizationRequest::default();
        let tracklets = (0..8)
            .map(|index| {
                profile_tracklet(
                    index,
                    index as u64 * 400,
                    index as u64 * 400 + 400,
                    if index.is_multiple_of(2) { 0.20 } else { 0.21 },
                    if index.is_multiple_of(2) { -0.8 } else { 0.8 },
                    40,
                )
            })
            .collect::<Vec<_>>();
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 3_200).expect("enrollment");
        let result = super::cluster_acoustic_tracklets_with_mode(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            super::AcousticClusteringMode::ProbabilisticV1,
            || false,
        )
        .expect("probabilistic clustering");
        assert_eq!(
            result.executed_mode,
            super::AcousticClusteringMode::ProbabilisticV1,
            "{result:#?}"
        );
        assert_eq!(result.detected_speakers, 1, "{result:#?}");
        assert!(
            result
                .assignments
                .iter()
                .all(|assignment| assignment.speaker_ref.is_some())
        );
        let resources = &result
            .count_estimate
            .as_ref()
            .expect("count estimate")
            .resources;
        assert_eq!(resources.prototype_count, 8);
        assert_eq!(resources.affinity_pair_evaluations, 56);
        assert!(resources.retained_sparse_edges <= 32, "{resources:#?}");
        assert_eq!(resources.stability_replicates, 5);
        assert!(resources.solver_iterations > 0, "{resources:#?}");
        assert!(
            resources
                .solver_residual
                .is_some_and(|residual| { residual <= super::SPEAKER_COUNT_EIGENSOLVER_TOLERANCE }),
            "{resources:#?}"
        );
    }

    #[test]
    fn fused_merge_risk_count_is_bounded_and_permutation_invariant() {
        let lane = |selected_count, losses: [f64; 3]| super::ProbabilisticLaneResult {
            selected_count,
            groups: vec![vec![0]],
            risk_curve: super::SpeakerCountRiskCurve {
                selected_count,
                points: losses
                    .into_iter()
                    .enumerate()
                    .map(|(index, expected_loss)| super::SpeakerCountRiskPoint {
                        count: index + 1,
                        expected_loss,
                    })
                    .collect(),
            },
        };
        let lanes = vec![
            lane(2, [5.0, 0.0, 3.0]),
            lane(2, [4.0, 0.0, 2.0]),
            lane(2, [6.0, 0.0, 4.0]),
            lane(3, [6.0, 1.0, 0.0]),
            lane(3, [7.0, 1.5, 0.0]),
        ];
        let policy = super::SpeakerCountPolicy {
            min: 1,
            max: 3,
            exact: None,
        };
        assert_eq!(super::fused_merge_risk_count(&lanes, None, policy), Some(2));

        let mut reversed = lanes.clone();
        reversed.reverse();
        assert_eq!(
            super::fused_merge_risk_count(&reversed, None, policy),
            Some(2)
        );

        let exact = super::SpeakerCountPolicy {
            min: 3,
            max: 3,
            exact: Some(3),
        };
        assert_eq!(
            super::fused_merge_risk_count(&lanes, None, exact),
            Some(3),
            "a hard count restricts candidate search without changing lane losses"
        );
    }

    #[test]
    fn fused_speaker_count_estimate_normalizes_with_explicit_unresolved_mass() {
        let lane = |losses: [f64; 3]| super::ProbabilisticLaneResult {
            selected_count: 2,
            groups: vec![vec![0], vec![1]],
            risk_curve: super::SpeakerCountRiskCurve {
                selected_count: 2,
                points: losses
                    .into_iter()
                    .enumerate()
                    .map(|(index, expected_loss)| super::SpeakerCountRiskPoint {
                        count: index + 1,
                        expected_loss,
                    })
                    .collect(),
            },
        };
        let lanes = vec![
            lane([4.0, 0.0, 3.0]),
            lane([5.0, 0.0, 2.0]),
            lane([6.0, 0.0, 4.0]),
            lane([4.5, 0.0, 2.5]),
            lane([5.5, 0.0, 3.5]),
        ];
        let estimate = super::fused_speaker_count_estimate(
            &lanes,
            None,
            Some(SpeakerCountLaneUnavailableReason::InvalidAffinity),
            &SpeakerCountRequest::Infer,
            super::SpeakerCountPolicy {
                min: 1,
                max: 3,
                exact: None,
            },
            1,
            super::unavailable_speaker_count_resources(3).expect("resource summary"),
        )
        .expect("fused estimate");
        estimate.validate().expect("valid estimate");
        assert_eq!(estimate.selected_count, Some(2), "{estimate:#?}");
        assert_eq!(estimate.posterior.len(), 3);
        assert_eq!(estimate.lanes.len(), 6);
        let total = estimate
            .posterior
            .iter()
            .map(|bin| bin.probability)
            .sum::<f64>()
            + estimate.unresolved_probability;
        assert!((total - 1.0).abs() <= 1.0e-9, "{estimate:#?}");
        assert!(
            estimate.unresolved_probability >= 0.15,
            "development evidence must retain unresolved mass: {estimate:#?}"
        );
    }

    #[test]
    fn fixed_safe_infer_reports_explicit_uncalibrated_count_evidence() {
        let request = DiarizationRequest::default();
        let tracklets = sequential_profile_tracklets(&[0.0; 6], 500, 50);
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 3_000).expect("enrollment");
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, &request.speaker_count, 64, || {
                false
            })
            .expect("fixed-safe clustering");
        let estimate = result
            .count_estimate
            .as_ref()
            .expect("native count object must never silently disappear");
        estimate.validate().expect("valid count estimate");
        assert_eq!(
            estimate.calibration_status,
            SpeakerCountCalibrationStatus::FixedSafeUncalibrated
        );
        assert_eq!(estimate.selected_count, None);
        assert!(estimate.posterior.is_empty());
        assert_eq!(estimate.unresolved_probability, 1.0);
        assert_eq!(estimate.lanes.len(), 6);

        let outcome =
            super::public_speaker_count_outcome(&request.speaker_count, &result).expect("outcome");
        assert_eq!(outcome.status, SpeakerCountOutcomeStatus::Unresolved);
        assert!(
            outcome
                .reasons
                .contains(&SpeakerCountOutcomeReason::SpeakerCountEvidenceUnresolved),
            "{outcome:#?}"
        );
    }

    #[test]
    fn no_voice_infer_returns_unavailable_count_without_fabricated_bins() {
        let request = DiarizationRequest::default();
        let mut tracklet = profile_tracklet(0, 0, 500, 0.0, 0.0, 50);
        tracklet.voice_valid = [false; super::VOICE_VECTOR_DIMENSIONS];
        tracklet.voice_mean = [0.0; super::VOICE_VECTOR_DIMENSIONS];
        tracklet.voice_variance = [0.0; super::VOICE_VECTOR_DIMENSIONS];
        tracklet.voice_support = [0; super::VOICE_VECTOR_DIMENSIONS];
        tracklet.voiced_frame_count = 0;
        tracklet.identity_frame_count = 0;
        let enrollment =
            enroll_known_speaker_profiles(std::slice::from_ref(&tracklet), &request, 500)
                .expect("enrollment");
        let result =
            cluster_acoustic_tracklets(&[tracklet], &enrollment, &request.speaker_count, 8, || {
                false
            })
            .expect("no-voice count fallback");
        let estimate = result
            .count_estimate
            .as_ref()
            .expect("unavailable estimate");
        estimate.validate().expect("valid unavailable estimate");
        assert_eq!(
            estimate.calibration_status,
            SpeakerCountCalibrationStatus::Unavailable
        );
        assert_eq!(estimate.selected_count, None);
        assert!(estimate.posterior.is_empty());
        assert_eq!(estimate.unresolved_probability, 1.0);
        assert_eq!(result.detected_speakers, 0);
    }

    #[test]
    fn soft_count_prior_disagreement_cannot_override_acoustic_selection() {
        let lane = || super::ProbabilisticLaneResult {
            selected_count: 2,
            groups: vec![vec![0], vec![1]],
            risk_curve: super::SpeakerCountRiskCurve {
                selected_count: 2,
                points: vec![
                    super::SpeakerCountRiskPoint {
                        count: 2,
                        expected_loss: 0.0,
                    },
                    super::SpeakerCountRiskPoint {
                        count: 3,
                        expected_loss: 0.1,
                    },
                ],
            },
        };
        let lanes = (0..super::SPEAKER_COUNT_PERTURBATION_LANES)
            .map(|_| lane())
            .collect::<Vec<_>>();
        let estimate = super::fused_speaker_count_estimate(
            &lanes,
            None,
            Some(SpeakerCountLaneUnavailableReason::InvalidAffinity),
            &SpeakerCountRequest::Prior {
                bins: vec![
                    crate::model::SpeakerCountPriorMass {
                        count: 2,
                        probability: 0.001,
                    },
                    crate::model::SpeakerCountPriorMass {
                        count: 3,
                        probability: 0.999,
                    },
                ],
            },
            super::SpeakerCountPolicy {
                min: 2,
                max: 3,
                exact: None,
            },
            1,
            super::unavailable_speaker_count_resources(3).expect("resource summary"),
        )
        .expect("prior-disagreement estimate");
        estimate.validate().expect("valid estimate");
        assert_eq!(
            estimate.selected_count,
            Some(2),
            "a soft prior may move mass but cannot override unanimous acoustic evidence"
        );
        assert_eq!(
            estimate
                .lanes
                .iter()
                .find(|lane| lane.lane == SpeakerCountEvidenceLane::CallerPrior)
                .and_then(|lane| lane.proposed_count),
            Some(3)
        );
        assert_eq!(
            estimate
                .lanes
                .iter()
                .find(|lane| lane.lane == SpeakerCountEvidenceLane::MergeRisk)
                .and_then(|lane| lane.proposed_count),
            Some(2),
            "merge-risk evidence must remain independent of the conflicting caller prior"
        );
    }

    #[test]
    fn soft_count_prior_does_not_exclude_acoustically_supported_counts() {
        let lane = || super::ProbabilisticLaneResult {
            selected_count: 1,
            groups: vec![vec![0, 1, 2]],
            risk_curve: super::SpeakerCountRiskCurve {
                selected_count: 1,
                points: vec![
                    super::SpeakerCountRiskPoint {
                        count: 1,
                        expected_loss: 0.0,
                    },
                    super::SpeakerCountRiskPoint {
                        count: 2,
                        expected_loss: 8.0,
                    },
                    super::SpeakerCountRiskPoint {
                        count: 3,
                        expected_loss: 12.0,
                    },
                ],
            },
        };
        let lanes = (0..super::SPEAKER_COUNT_PERTURBATION_LANES)
            .map(|_| lane())
            .collect::<Vec<_>>();
        let request = SpeakerCountRequest::Prior {
            bins: vec![crate::model::SpeakerCountPriorMass {
                count: 3,
                probability: 1.0,
            }],
        };
        let policy = super::resolve_count_policy(&request, 3).expect("soft-prior search policy");
        assert_eq!(policy.min, 1);
        assert_eq!(policy.max, 3);
        assert_eq!(policy.exact, None);
        let estimate = super::fused_speaker_count_estimate(
            &lanes,
            None,
            Some(SpeakerCountLaneUnavailableReason::InvalidAffinity),
            &request,
            policy,
            1,
            super::unavailable_speaker_count_resources(3).expect("resource summary"),
        )
        .expect("soft-prior estimate");
        estimate.validate().expect("valid estimate");
        assert_eq!(
            estimate.selected_count,
            Some(1),
            "strong acoustic agreement must be allowed to select outside soft prior support"
        );
        assert_eq!(
            estimate
                .lanes
                .iter()
                .find(|lane| lane.lane == SpeakerCountEvidenceLane::CallerPrior)
                .and_then(|lane| lane.proposed_count),
            Some(3)
        );
        let acoustic_only = super::fused_speaker_count_estimate(
            &lanes,
            None,
            Some(SpeakerCountLaneUnavailableReason::InvalidAffinity),
            &SpeakerCountRequest::Infer,
            policy,
            1,
            super::unavailable_speaker_count_resources(3).expect("resource summary"),
        )
        .expect("acoustic-only estimate");
        let count_three_probability = |estimate: &crate::model::SpeakerCountEstimate| {
            estimate
                .posterior
                .iter()
                .find(|bin| bin.count == 3)
                .map(|bin| bin.probability)
                .expect("count-three posterior bin")
        };
        assert!(
            count_three_probability(&estimate) > count_three_probability(&acoustic_only),
            "the bounded soft prior must still move posterior mass toward its support"
        );
    }

    #[test]
    fn partially_feasible_soft_count_input_retains_a_non_authoritative_lane() {
        let range = super::speaker_count_prior_lane(
            &SpeakerCountRequest::Range {
                minimum: 1,
                maximum: 4,
            },
            2,
            3,
        )
        .expect("range lane");
        assert!(range.available);
        assert_eq!(range.proposed_count, Some(2));
        assert_eq!(range.unavailable_reason, None);

        let prior = super::speaker_count_prior_lane(
            &SpeakerCountRequest::Prior {
                bins: vec![
                    crate::model::SpeakerCountPriorMass {
                        count: 1,
                        probability: 0.9,
                    },
                    crate::model::SpeakerCountPriorMass {
                        count: 2,
                        probability: 0.1,
                    },
                ],
            },
            2,
            3,
        )
        .expect("prior lane");
        assert!(prior.available);
        assert_eq!(prior.proposed_count, Some(2));
        assert_eq!(prior.confidence, 0.1);

        let contradictory = super::speaker_count_prior_lane(
            &SpeakerCountRequest::Range {
                minimum: 4,
                maximum: 6,
            },
            1,
            3,
        )
        .expect("contradictory range lane");
        assert!(!contradictory.available);
        assert_eq!(
            contradictory.unavailable_reason,
            Some(SpeakerCountLaneUnavailableReason::ContradictoryConstraints)
        );
    }

    #[test]
    fn sparse_normalized_eigengap_distinguishes_one_and_two_components() {
        let graph = |node_count: usize, edges: &[(usize, usize, f64)]| {
            let mut rows = vec![Vec::new(); node_count];
            let mut degrees = vec![1.0; node_count];
            for &(left, right, weight) in edges {
                rows[left].push((right, weight));
                rows[right].push((left, weight));
                degrees[left] += weight;
                degrees[right] += weight;
            }
            super::SparseSpeakerAffinityGraph {
                rows,
                degrees,
                undirected_edge_count: edges.len(),
            }
        };
        let two_components = graph(4, &[(0, 1, 1.0), (2, 3, 1.0)]);
        let proposal = super::sparse_normalized_eigengap(&two_components, 1, 3, || false)
            .expect("spectral solver")
            .expect("two-component proposal");
        assert_eq!(proposal.count, 2, "{proposal:#?}");
        assert!(proposal.confidence > 0.99, "{proposal:#?}");
        assert!(proposal.residual <= super::SPEAKER_COUNT_EIGENSOLVER_TOLERANCE);

        let connected = graph(
            4,
            &[
                (0, 1, 1.0),
                (0, 2, 1.0),
                (0, 3, 1.0),
                (1, 2, 1.0),
                (1, 3, 1.0),
                (2, 3, 1.0),
            ],
        );
        let proposal = super::sparse_normalized_eigengap(&connected, 1, 3, || false)
            .expect("spectral solver")
            .expect("connected proposal");
        assert_eq!(proposal.count, 1, "{proposal:#?}");

        let proposal = super::sparse_normalized_eigengap(&connected, 1, 4, || false)
            .expect("spectral solver")
            .expect("connected proposal at the prototype ceiling");
        assert_eq!(
            proposal.count, 1,
            "a missing fifth eigenvalue must not fabricate a terminal K=4 gap: {proposal:#?}"
        );
    }

    #[test]
    fn sparse_normalized_eigengap_is_cancellable() {
        let graph = super::SparseSpeakerAffinityGraph {
            rows: vec![vec![(1, 1.0)], vec![(0, 1.0)]],
            degrees: vec![2.0, 2.0],
            undirected_edge_count: 1,
        };
        let error = super::sparse_normalized_eigengap(&graph, 1, 2, || true)
            .expect_err("cancellation must stop spectral iteration");
        assert!(
            matches!(error, crate::error::FwError::Cancelled(_)),
            "{error:?}"
        );
    }

    #[test]
    fn probabilistic_clustering_falls_back_when_voice_overlap_is_too_sparse() {
        let request = DiarizationRequest::default();
        let mut tracklets = vec![
            profile_tracklet(0, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 0.1, 0.0, 50),
        ];
        for tracklet in &mut tracklets {
            for dimension in 2..VOICE_VECTOR_DIMENSIONS {
                tracklet.voice_valid[dimension] = false;
                tracklet.voice_support[dimension] = 0;
            }
        }
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 1_000).expect("enrollment");
        let result = super::cluster_acoustic_tracklets_with_mode(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            super::AcousticClusteringMode::ProbabilisticV1,
            || false,
        )
        .expect("deterministic fallback");
        assert_eq!(
            result.executed_mode,
            super::AcousticClusteringMode::FixedSafeV1
        );
        assert_eq!(
            result.fallback_reason,
            Some(super::AcousticClusteringFallbackReason::InsufficientSharedVoiceDimensions)
        );
        assert_eq!(result.bootstrap_stability, 0.0);
    }

    #[test]
    fn probabilistic_hard_constraint_graph_never_merges_distinct_references() {
        let request = DiarizationRequest {
            known_intervals: vec![
                known_interval("alice", 0, 500, KnownSpeakerPolicy::HardMustLink),
                known_interval("bob", 500, 1_000, KnownSpeakerPolicy::HardMustLink),
            ],
            enrollment_edge_guard_ms: 0,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 0.0, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 1_000).expect("enrollment");
        let result = super::cluster_acoustic_tracklets_with_mode(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            super::AcousticClusteringMode::ProbabilisticV1,
            || false,
        )
        .expect("constrained clustering");
        assert_eq!(result.detected_speakers, 2, "{result:#?}");
        assert_eq!(result.assignments[0].speaker_ref.as_deref(), Some("alice"));
        assert_eq!(result.assignments[1].speaker_ref.as_deref(), Some("bob"));
        assert!(result.constraints_satisfied);
    }

    #[test]
    fn speaker_pair_loss_threshold_and_hash_are_stable() {
        let calibration = super::acoustic_speaker_pair_calibration();
        let merge_threshold = calibration.false_merge_loss
            / (calibration.false_merge_loss + calibration.false_split_loss);
        assert!((merge_threshold - (12.0 / 13.0)).abs() < f32::EPSILON);
        assert_eq!(super::SpeakerPairPerturbation::ALL.len(), 5);
        assert!(
            !super::SpeakerPairPerturbation::NoPitchCoordinates.includes(20)
                && !super::SpeakerPairPerturbation::NoPitchCoordinates.includes(26)
        );
        assert!(!super::SpeakerPairPerturbation::NoFormantCoordinates.includes(23));
        assert!(!super::SpeakerPairPerturbation::NoChannelEvidence.includes_channel());
        let hash = super::acoustic_speaker_pair_calibration_sha256();
        assert_eq!(hash.len(), 64);
        assert!(hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn development_assignment_calibration_is_bounded_monotone_and_does_not_change_rejection() {
        let raw = [0.0_f32, 0.25, 0.55, 0.75, 1.0];
        let calibrated = raw.map(super::calibrate_assignment_confidence);
        assert!((calibrated[0] - 0.0).abs() < f32::EPSILON);
        assert!((calibrated[1] - 0.25).abs() < f32::EPSILON);
        assert!((calibrated[4] - 1.0).abs() < f32::EPSILON);
        assert!(
            calibrated
                .windows(2)
                .all(|pair| pair[0] <= pair[1] && (0.0..=1.0).contains(&pair[0]))
        );
        assert!((0.0..=1.0).contains(calibrated.last().expect("last confidence")));
        let rejection_threshold =
            super::acoustic_speaker_pair_calibration().minimum_assignment_probability;
        assert!(raw[1] < rejection_threshold);
        assert_eq!(super::calibrate_assignment_confidence(raw[1]), raw[1]);
    }

    #[test]
    fn separated_voices_do_not_false_merge_when_two_speakers_are_required() {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::HardConstraint { count: 2 },
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 0, 500, -1.5, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 1.5, 0.0, 50),
            profile_tracklet(2, 1_000, 1_500, -1.5, 0.0, 50),
            profile_tracklet(3, 1_500, 2_000, 1.5, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 2_000).expect("enrollment");
        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("cluster");
        assert_ne!(
            result.assignments[0].speaker_ref,
            result.assignments[1].speaker_ref
        );
    }

    #[test]
    fn speaker_return_preserves_the_first_speaker_label() {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::HardConstraint { count: 2 },
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 2.0, 0.0, 50),
            profile_tracklet(2, 1_000, 1_500, 0.0, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 1_500).expect("enrollment");
        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("cluster");
        assert_eq!(
            result.assignments[0].speaker_ref,
            result.assignments[2].speaker_ref
        );
        assert_ne!(
            result.assignments[0].speaker_ref,
            result.assignments[1].speaker_ref
        );
    }

    #[test]
    fn short_ambiguous_tracklet_is_rejected_as_unknown() {
        let request = DiarizationRequest::default();
        let tracklets = vec![profile_tracklet(0, 0, 30, 0.0, 0.0, 3)];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 30).expect("enrollment");
        let result = cluster_acoustic_tracklets(
            &tracklets,
            &enrollment,
            &request.speaker_count,
            512,
            || false,
        )
        .expect("cluster");
        assert_eq!(result.assignments[0].speaker_ref, None);
        assert_eq!(result.detected_speakers, 0);
    }

    #[test]
    fn clustering_is_deterministic_and_labels_ignore_tracklet_ids() {
        let request = DiarizationRequest::default();
        let first_tracklets = vec![
            profile_tracklet(1, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(8, 500, 1_000, 2.0, 0.0, 50),
            profile_tracklet(3, 1_000, 1_500, 0.0, 0.0, 50),
        ];
        let second_tracklets = vec![
            profile_tracklet(90, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(4, 500, 1_000, 2.0, 0.0, 50),
            profile_tracklet(77, 1_000, 1_500, 0.0, 0.0, 50),
        ];
        let constraints = SpeakerCountRequest::HardConstraint { count: 2 };
        let first_enrollment = enroll_known_speaker_profiles(&first_tracklets, &request, 1_500)
            .expect("first enrollment");
        let second_enrollment = enroll_known_speaker_profiles(&second_tracklets, &request, 1_500)
            .expect("second enrollment");
        let first = cluster_acoustic_tracklets(
            &first_tracklets,
            &first_enrollment,
            &constraints,
            512,
            || false,
        )
        .expect("first cluster");
        let repeated = cluster_acoustic_tracklets(
            &first_tracklets,
            &first_enrollment,
            &constraints,
            512,
            || false,
        )
        .expect("repeated cluster");
        let second = cluster_acoustic_tracklets(
            &second_tracklets,
            &second_enrollment,
            &constraints,
            512,
            || false,
        )
        .expect("second cluster");
        assert_eq!(first, repeated);
        assert_eq!(
            first
                .assignments
                .iter()
                .map(|assignment| assignment.speaker_ref.as_deref())
                .collect::<Vec<_>>(),
            second
                .assignments
                .iter()
                .map(|assignment| assignment.speaker_ref.as_deref())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn prototype_pressure_is_reported_and_never_exceeds_the_cap() {
        let request = DiarizationRequest::default();
        let tracklets = (0..600)
            .map(|index| {
                profile_tracklet(
                    index,
                    index as u64 * 100,
                    index as u64 * 100 + 100,
                    (index % 3) as f32,
                    (index % 5) as f32 / 5.0,
                    10,
                )
            })
            .collect::<Vec<_>>();
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 60_000).expect("enrollment");
        let constraints = SpeakerCountRequest::HardConstraint { count: 1 };
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, &constraints, 32, || false)
                .expect("cluster");
        assert!(result.cap_pressure);
        assert_eq!(result.prototype_count, 32);
        assert_eq!(result.prototype_cap, 32);
        assert_eq!(
            result.detected_speakers, 0,
            "prototype pressure must not promote 100 ms outliers into a supported speaker"
        );
        assert!(!result.constraints_satisfied);
        assert!(
            result
                .assignments
                .iter()
                .all(|assignment| assignment.speaker_ref.is_none())
        );
    }

    #[test]
    fn clustering_cancellation_is_checked_before_unbounded_work() {
        let request = DiarizationRequest::default();
        let tracklets = (0..100)
            .map(|index| {
                profile_tracklet(
                    index,
                    index as u64 * 100,
                    index as u64 * 100 + 100,
                    index as f32 / 100.0,
                    0.0,
                    10,
                )
            })
            .collect::<Vec<_>>();
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, 10_000).expect("enrollment");
        let mut checks = 0usize;
        let error =
            cluster_acoustic_tracklets(&tracklets, &enrollment, &request.speaker_count, 64, || {
                checks += 1;
                checks >= 2
            })
            .expect_err("cancel");
        assert!(error.to_string().contains("clustering cancelled"));
    }

    #[test]
    fn clustering_rejects_invalid_tracklet_statistics_and_counts() {
        let request = DiarizationRequest::default();
        let enrollment = enroll_known_speaker_profiles(&[], &request, 100).expect("enrollment");
        let mut invalid = profile_tracklet(0, 0, 100, 0.0, 0.0, 10);
        invalid.channel_variance[0] = f32::NAN;
        assert!(
            cluster_acoustic_tracklets(
                &[invalid.clone()],
                &enrollment,
                &request.speaker_count,
                8,
                || false,
            )
            .expect_err("non-finite channel variance")
            .to_string()
            .contains("finite non-negative statistics")
        );

        invalid.channel_variance[0] = 0.01;
        invalid.voice_variance[0] = -0.01;
        assert!(
            cluster_acoustic_tracklets(
                &[invalid.clone()],
                &enrollment,
                &request.speaker_count,
                8,
                || false,
            )
            .expect_err("negative sample variance")
            .to_string()
            .contains("finite non-negative statistics")
        );

        invalid.voice_variance[0] = 0.01;
        invalid.voice_mean[0] = f32::MAX;
        assert!(
            cluster_acoustic_tracklets(
                &[invalid.clone()],
                &enrollment,
                &request.speaker_count,
                8,
                || false,
            )
            .expect_err("finite tracklet means must not overflow distance arithmetic")
            .to_string()
            .contains("within acoustic-v2 bounds")
        );

        invalid.voice_mean[0] = 0.0;
        invalid.channel_variance[0] = f32::MAX;
        assert!(
            cluster_acoustic_tracklets(
                &[invalid.clone()],
                &enrollment,
                &request.speaker_count,
                8,
                || false,
            )
            .expect_err("finite tracklet variances must not overflow distance arithmetic")
            .to_string()
            .contains("within acoustic-v2 bounds")
        );

        invalid.channel_variance[0] = 0.01;
        invalid.channel_dimensions = 0;
        assert!(
            cluster_acoustic_tracklets(
                &[invalid.clone()],
                &enrollment,
                &request.speaker_count,
                8,
                || false,
            )
            .expect_err("valid channel evidence requires an explicit coordinate prefix")
            .to_string()
            .contains("valid counts")
        );

        invalid.channel_dimensions = CHANNEL_VECTOR_DIMENSIONS;
        invalid.overlap_probability = f32::NAN;
        assert!(
            cluster_acoustic_tracklets(
                &[invalid.clone()],
                &enrollment,
                &request.speaker_count,
                8,
                || false,
            )
            .expect_err("overlap evidence must be a finite probability")
            .to_string()
            .contains("within acoustic-v2 bounds")
        );

        invalid.overlap_probability = 0.0;
        invalid.voiced_frame_count = invalid.frame_count + 1;
        assert!(
            cluster_acoustic_tracklets(&[invalid], &enrollment, &request.speaker_count, 8, || {
                false
            },)
            .expect_err("voiced count exceeds total")
            .to_string()
            .contains("valid counts")
        );

        let mut huge = profile_tracklet(0, 0, 100, 0.0, 0.0, 0);
        huge.frame_count = usize::MAX;
        let following = profile_tracklet(1, 100, 200, 0.0, 0.0, 1);
        assert!(
            cluster_acoustic_tracklets(
                &[huge, following],
                &enrollment,
                &request.speaker_count,
                8,
                || false,
            )
            .expect_err("aggregate frame counts must not overflow")
            .to_string()
            .contains("valid counts")
        );

        let nested = vec![
            profile_tracklet(0, 0, 100, 0.0, 0.0, 10),
            profile_tracklet(1, 80, 90, 0.0, 0.0, 1),
        ];
        assert!(
            cluster_acoustic_tracklets(&nested, &enrollment, &request.speaker_count, 8, || false,)
                .expect_err("nested tracklet timeline")
                .to_string()
                .contains("ordered positive intervals")
        );
    }

    fn assignment(
        tracklet_index: usize,
        start_ms: u64,
        end_ms: u64,
        speaker_ref: Option<&str>,
        confidence: f32,
    ) -> AcousticSpeakerAssignment {
        AcousticSpeakerAssignment {
            tracklet_index,
            start_ms,
            end_ms,
            speaker_ref: speaker_ref.map(str::to_owned),
            speaker_confidence: confidence,
            secondary_speaker_ref: None,
            secondary_speaker_confidence: None,
            change_confidence: 0.8,
            overlap_suspected: false,
            hard_attribution: false,
        }
    }

    fn turn(
        start_ms: u64,
        end_ms: u64,
        speaker_ref: Option<&str>,
        confidence: Option<f64>,
    ) -> DiarizationTurn {
        DiarizationTurn {
            start_ms,
            end_ms,
            speaker_ref: speaker_ref.map(str::to_owned),
            speaker_confidence: confidence,
            change_confidence: Some(0.8),
            overlap_suspected: false,
            hard_hint_attributed: false,
        }
    }

    fn transcript_segment(
        start_sec: Option<f64>,
        end_sec: Option<f64>,
        text: &str,
        confidence: Option<f64>,
    ) -> TranscriptionSegment {
        TranscriptionSegment {
            start_sec,
            end_sec,
            text: text.to_owned(),
            speaker: Some("stale-backend-label".to_owned()),
            confidence,
        }
    }

    #[test]
    fn assignments_become_independent_non_overlapping_turns() {
        let assignments = vec![
            assignment(0, 0, 525, Some("alice"), 0.8),
            assignment(1, 500, 1_025, Some("bob"), 0.7),
            assignment(2, 1_000, 1_500, Some("bob"), 0.6),
        ];
        let turns = diarization_turns_from_assignments(&assignments, 1_500).expect("turn timeline");
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker_ref.as_deref(), Some("alice"));
        assert_eq!(turns[1].speaker_ref.as_deref(), Some("bob"));
        assert_eq!(turns[0].end_ms, turns[1].start_ms);
        assert_eq!(turns[1].end_ms, 1_500);
        assert!((turns[1].speaker_confidence.expect("confidence") - 0.6).abs() < 1e-6);
    }

    #[test]
    fn turn_merging_does_not_launder_hard_or_overlap_provenance() {
        let ordinary = assignment(0, 0, 500, Some("alice"), 0.8);
        let mut hard = assignment(1, 500, 1_000, Some("alice"), 1.0);
        hard.hard_attribution = true;
        let mut overlap = assignment(2, 1_000, 1_500, Some("alice"), 0.7);
        overlap.overlap_suspected = true;

        let turns = diarization_turns_from_assignments(&[ordinary, hard, overlap], 1_500)
            .expect("provenance-preserving turn timeline");
        assert_eq!(turns.len(), 3);
        assert!(!turns[0].hard_hint_attributed);
        assert!(!turns[0].overlap_suspected);
        assert!(turns[1].hard_hint_attributed);
        assert!(!turns[1].overlap_suspected);
        assert!(!turns[2].hard_hint_attributed);
        assert!(turns[2].overlap_suspected);
    }

    #[test]
    fn explicit_secondary_assignment_projects_as_two_overlapping_turns() {
        let mut overlapped = assignment(0, 0, 500, Some("alice"), 0.8);
        overlapped.overlap_suspected = true;
        overlapped.secondary_speaker_ref = Some("bob".to_owned());
        overlapped.secondary_speaker_confidence = Some(0.65);
        let turns =
            diarization_turns_from_assignments(&[overlapped], 500).expect("overlap timeline");
        assert_eq!(turns.len(), 2);
        assert_eq!(
            turns
                .iter()
                .filter_map(|turn| turn.speaker_ref.as_deref())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["alice", "bob"])
        );
        assert!(
            turns
                .iter()
                .all(|turn| turn.start_ms == 0 && turn.end_ms == 500 && turn.overlap_suspected)
        );
    }

    #[test]
    fn secondary_assignment_without_overlap_evidence_fails_closed() {
        let mut invalid = assignment(0, 0, 500, Some("alice"), 0.8);
        invalid.secondary_speaker_ref = Some("bob".to_owned());
        invalid.secondary_speaker_confidence = Some(0.65);
        assert!(
            diarization_turns_from_assignments(&[invalid], 500)
                .expect_err("unbacked secondary speaker")
                .to_string()
                .contains("finite, bounded, and time-ordered")
        );
    }

    #[test]
    fn active_agent_queries_are_bounded_merged_and_content_bound() {
        let unknown_a = assignment(0, 0, 200, None, 0.0);
        let unknown_b = assignment(1, 200, 400, None, 0.0);
        let low_confidence = assignment(2, 500, 800, Some("alice"), 0.40);
        let queries = super::speaker_attribution_queries(
            &[unknown_a, unknown_b, low_confidence],
            &"a".repeat(64),
        );
        assert_eq!(queries.len(), 2);
        assert_eq!(
            queries[0].reason,
            SpeakerAttributionQueryReason::UnknownAttribution
        );
        assert_eq!((queries[0].start_ms, queries[0].end_ms), (0, 400));
        assert_eq!(
            queries[1].reason,
            SpeakerAttributionQueryReason::LowConfidence
        );
        assert_eq!(queries[1].candidate_speaker_refs, vec!["alice"]);
        assert!(
            queries
                .iter()
                .all(|query| query.query_id_sha256.len() == 64)
        );

        let different_input = super::speaker_attribution_queries(
            &[assignment(0, 0, 400, None, 0.0)],
            &"b".repeat(64),
        );
        assert_ne!(
            queries[0].query_id_sha256,
            different_input[0].query_id_sha256
        );

        let separated_unknowns = (0..40)
            .map(|index| {
                let start = index * 200;
                assignment(index as usize, start, start + 100, None, 0.0)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            super::speaker_attribution_queries(&separated_unknowns, &"c".repeat(64)).len(),
            32
        );
    }

    #[test]
    fn external_style_turn_without_calibration_remains_structurally_valid() {
        let turns = vec![turn(0, 1_000, Some("external_a"), None)];
        let projection =
            project_diarization_onto_segments(&[], &turns, false).expect("valid turn timeline");
        assert!(projection.segments.is_empty());
    }

    #[test]
    fn dtw_word_projection_preserves_text_order_bytes_and_asr_confidence() {
        let turns = vec![
            turn(0, 1_000, Some("alice"), Some(0.9)),
            turn(1_000, 2_000, Some("bob"), Some(0.8)),
        ];
        let segments = vec![
            transcript_segment(Some(0.1), Some(0.8), " hello", Some(0.91)),
            transcript_segment(Some(0.8), Some(1.2), "bridge,", Some(0.72)),
            transcript_segment(Some(1.2), Some(1.8), "world!", None),
        ];
        let projection =
            project_diarization_onto_segments(&segments, &turns, true).expect("projection");
        assert_eq!(
            projection
                .segments
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>(),
            vec![" hello", "bridge,", "world!"]
        );
        assert_eq!(
            projection
                .segments
                .iter()
                .map(|segment| segment.confidence)
                .collect::<Vec<_>>(),
            vec![Some(0.91), Some(0.72), None]
        );
        assert_eq!(
            projection
                .segments
                .iter()
                .map(|segment| segment.speaker.as_deref())
                .collect::<Vec<_>>(),
            vec![Some("alice"), Some("alice"), Some("bob")]
        );
    }

    #[test]
    fn tied_word_uses_earlier_speaker_only_above_confidence_policy() {
        let low_confidence_turns = vec![
            turn(0, 1_000, Some("alice"), Some(0.2)),
            turn(1_000, 2_000, Some("bob"), Some(0.9)),
        ];
        let word = vec![transcript_segment(
            Some(0.8),
            Some(1.2),
            "bridge",
            Some(0.9),
        )];
        let low =
            project_diarization_onto_segments(&word, &low_confidence_turns, true).expect("low");
        assert_eq!(low.segments[0].speaker, None);

        let high_confidence_turns = vec![
            turn(0, 1_000, Some("alice"), Some(0.8)),
            turn(1_000, 2_000, Some("bob"), Some(0.9)),
        ];
        let high =
            project_diarization_onto_segments(&word, &high_confidence_turns, true).expect("high");
        assert_eq!(high.segments[0].speaker.as_deref(), Some("alice"));
    }

    #[test]
    fn no_word_timing_requires_duration_dominance() {
        let mixed_turns = vec![
            turn(0, 1_000, Some("alice"), Some(0.9)),
            turn(1_000, 2_000, Some("bob"), Some(0.9)),
        ];
        let segment = vec![transcript_segment(
            Some(0.0),
            Some(2.0),
            "untouched transcript bytes",
            Some(0.83),
        )];
        let mixed =
            project_diarization_onto_segments(&segment, &mixed_turns, false).expect("mixed");
        assert_eq!(mixed.segments[0].speaker, None);
        assert_eq!(mixed.mixed_speaker_segment_indices, vec![0]);
        assert_eq!(mixed.segments[0].confidence, Some(0.83));

        let dominant_turns = vec![
            turn(0, 1_600, Some("alice"), Some(0.9)),
            turn(1_600, 2_000, Some("bob"), Some(0.9)),
        ];
        let dominant =
            project_diarization_onto_segments(&segment, &dominant_turns, false).expect("dominant");
        assert_eq!(dominant.segments[0].speaker.as_deref(), Some("alice"));
        assert!(dominant.mixed_speaker_segment_indices.is_empty());
        assert_eq!(dominant.segments[0].text, segment[0].text);
    }

    #[test]
    fn overlap_evidence_survives_projection_outside_legacy_segment_shape() {
        let mut suspected = turn(0, 1_000, Some("alice"), Some(0.9));
        suspected.overlap_suspected = true;
        let segments = vec![
            transcript_segment(Some(0.2), Some(0.4), "one", None),
            transcript_segment(Some(1.2), Some(1.4), "two", None),
        ];
        let projection = project_diarization_onto_segments(
            &segments,
            &[suspected, turn(1_000, 2_000, Some("bob"), Some(0.9))],
            true,
        )
        .expect("projection");
        assert_eq!(projection.overlap_suspected_segment_indices, vec![0]);
    }

    #[test]
    fn untimed_segment_with_multiple_speakers_is_honestly_mixed_and_unknown() {
        let turns = vec![
            turn(0, 1_000, Some("alice"), Some(0.9)),
            turn(1_000, 2_000, Some("bob"), Some(0.9)),
        ];
        let segments = vec![transcript_segment(
            None,
            None,
            "time unavailable",
            Some(0.4),
        )];
        let projection =
            project_diarization_onto_segments(&segments, &turns, false).expect("projection");
        assert_eq!(projection.segments[0].speaker, None);
        assert_eq!(projection.mixed_speaker_segment_indices, vec![0]);
    }

    #[test]
    fn malformed_turn_and_word_timelines_fail_closed() {
        let malformed_turns = vec![turn(100, 100, Some("alice"), Some(0.9))];
        let segments = vec![transcript_segment(Some(0.0), Some(1.0), "word", None)];
        assert!(
            project_diarization_onto_segments(&segments, &malformed_turns, true)
                .expect_err("zero turn")
                .to_string()
                .contains("diarization turns")
        );

        let turns = vec![turn(0, 2_000, Some("alice"), Some(0.9))];
        let overlapping_words = vec![
            transcript_segment(Some(0.0), Some(1.1), "one", None),
            transcript_segment(Some(1.0), Some(2.0), "two", None),
        ];
        assert!(
            project_diarization_onto_segments(&overlapping_words, &turns, true)
                .expect_err("overlapping words")
                .to_string()
                .contains("projection segments")
        );
    }

    #[test]
    fn projection_accepts_only_sub_epsilon_adjacency_noise() {
        let turns = vec![turn(0, 2_000, Some("alice"), Some(0.9))];
        let harmless = vec![
            transcript_segment(Some(0.0), Some(1.0), "one", None),
            transcript_segment(
                Some(1.0 - super::CANONICAL_PROJECTION_EPSILON_SEC / 2.0),
                Some(2.0),
                "two",
                None,
            ),
        ];
        project_diarization_onto_segments(&harmless, &turns, true)
            .expect("shared projection epsilon must absorb floating-point adjacency noise");

        let material = vec![
            transcript_segment(Some(0.0), Some(1.0), "one", None),
            transcript_segment(
                Some(1.0 - super::CANONICAL_PROJECTION_EPSILON_SEC * 2.0),
                Some(2.0),
                "two",
                None,
            ),
        ];
        assert!(
            project_diarization_onto_segments(&material, &turns, true)
                .expect_err("material overlap must remain invalid")
                .to_string()
                .contains("projection segments")
        );
    }
}
