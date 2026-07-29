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

use crate::error::{FwError, FwResult};
use crate::model::{
    DiarizationEngine, DiarizationFallbackPolicy, DiarizationFallbackStatus, DiarizationReport,
    DiarizationRequest, DiarizationTurn, KnownSpeakerInterval, KnownSpeakerPolicy,
    SpeakerConstraints, SpeakerProfileSummary, TranscriptionSegment,
};

/// Stable identifier for the native acoustic diarization contract.
pub const ACOUSTIC_DIARIZATION_CONTRACT_VERSION: &str = "acoustic-diarization-v1";
/// Frozen implementation identity for retained diarization evaluation results.
pub const DIARIZATION_SCORER_VERSION: &str = "diarization-scorer-v1";
/// Schema identity for reference annotations accepted by the frozen scorer.
pub const DIARIZATION_REFERENCE_SCHEMA_VERSION: &str = "diarization-reference-v1";
/// Schema identity for system hypotheses accepted by the frozen scorer.
pub const DIARIZATION_HYPOTHESIS_SCHEMA_VERSION: &str = "diarization-hypothesis-v1";
/// Schema identity for scorer configuration.
pub const DIARIZATION_SCORER_CONFIG_SCHEMA_VERSION: &str = "diarization-scorer-config-v1";
/// Schema identity for retained scorer results.
pub const DIARIZATION_SCORE_RESULT_SCHEMA_VERSION: &str = "diarization-score-result-v1";
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
}

impl Default for DiarizationScorerConfig {
    fn default() -> Self {
        Self {
            schema_version: DIARIZATION_SCORER_CONFIG_SCHEMA_VERSION.to_owned(),
            speaker_boundary_collar_ms: 0,
            change_boundary_collar_ms: 250,
            overlap_policy: EvaluationOverlapPolicy::Include,
            calibration_bins: 10,
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

    let hypothesis_to_reference = maximum_overlap_mapping(&overlap_weights);
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

fn speaker_change_points_ms(
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

fn score_overlap_detection(atoms: &[EvaluationAtomicInterval]) -> OverlapDetectionScore {
    let mut reference_overlap_sec = 0.0;
    let mut hypothesis_overlap_sec = 0.0;
    let mut true_positive_sec = 0.0;
    let mut false_positive_sec = 0.0;
    let mut false_negative_sec = 0.0;
    for atom in atoms.iter().filter(|atom| !atom.excluded) {
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
    OverlapDetectionScore {
        reference_overlap_sec,
        hypothesis_overlap_sec,
        true_positive_sec,
        false_positive_sec,
        false_negative_sec,
        precision,
        recall,
        f1: f1_or_none(precision, recall),
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

fn f1_or_none(precision: Option<f64>, recall: Option<f64>) -> Option<f64> {
    match (precision, recall) {
        (Some(precision), Some(recall)) if precision + recall > SCORE_EPSILON_SEC => {
            Some(2.0 * precision * recall / (precision + recall))
        }
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    }
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

fn maximum_overlap_mapping(weights: &[Vec<f64>]) -> Vec<Option<usize>> {
    let reference_count = weights.len();
    let hypothesis_count = weights.first().map_or(0, Vec::len);
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

/// Stable identity for the first native acoustic feature layout.
pub const ACOUSTIC_FEATURE_SCHEMA_VERSION: &str = "acoustic-feature-v1";
/// Fixed analysis cadence shared with the native Whisper frontend.
pub const ACOUSTIC_FRAME_SAMPLES: usize = crate::native_engine::mel::N_FFT;
/// Fixed frame advance shared with the native Whisper frontend.
pub const ACOUSTIC_HOP_SAMPLES: usize = crate::native_engine::mel::HOP;
/// Maximum number of frames between cancellation checks.
pub const ACOUSTIC_CANCELLATION_INTERVAL_FRAMES: usize = 32;

const ENVELOPE_BANDS: usize = 12;
const CEPSTRAL_COEFFICIENTS: usize = 6;
const POWER_EPSILON: f32 = 1e-20;
const PCM_EPSILON: f32 = 1e-12;
const MAX_ABS_ACOUSTIC_FEATURE: f32 = 1_000_000.0;
const MAX_ACOUSTIC_VARIANCE: f32 = MAX_ABS_ACOUSTIC_FEATURE * MAX_ABS_ACOUSTIC_FEATURE;

/// Vocal-source and vocal-tract evidence.
///
/// Pitch is deliberately nullable and is never converted into a demographic
/// label. The cepstral envelope excludes absolute energy so it can remain
/// useful when one person moves relative to the microphone.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceFeatureView {
    pub cepstral_envelope: [f32; CEPSTRAL_COEFFICIENTS],
    pub cepstral_delta: [f32; CEPSTRAL_COEFFICIENTS],
    pub f0_hz: Option<f32>,
    pub voicing_confidence: f32,
    pub harmonicity: f32,
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

/// One bounded acoustic observation produced at the v1 10 ms cadence.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticFrameFeatures {
    pub frame_index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub voice: VoiceFeatureView,
    pub channel: ChannelFeatureView,
    pub quality: AcousticQualityMask,
}

/// Resource and quality summary returned after streaming extraction.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureExtractionSummary {
    pub feature_schema: &'static str,
    pub frame_count: usize,
    pub voiced_frame_count: usize,
    pub low_energy_frame_count: usize,
    /// Fixed upper bound on retained DSP state, independent of call duration.
    pub retained_state_bytes_upper_bound: usize,
}

/// Stream acoustic-v1 features from normalized 16 kHz mono PCM.
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
    let mut previous_normalized_power = [0.0_f32; crate::native_engine::mel::N_FREQ_BINS];
    let mut has_previous = false;
    let mut noise_floor_dbfs = -90.0_f32;
    let mut voiced_fraction = 0.0_f32;
    let mut voiced_frame_count = 0usize;
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
        let frame = &samples[start..start + ACOUSTIC_FRAME_SAMPLES];
        let mut power = [0.0_f32; crate::native_engine::mel::N_FREQ_BINS];
        crate::native_engine::mel::fixed_frame_power_spectrum(frame, &mut power)?;

        let (rms_dbfs, crest_factor, clipping_fraction) = waveform_descriptors(frame);
        if rms_dbfs < -55.0 {
            low_energy_frame_count += 1;
        }
        noise_floor_dbfs = update_noise_floor(noise_floor_dbfs, rms_dbfs);

        let (f0_hz, voicing_confidence) = estimate_f0(frame, rms_dbfs);
        let voiced = f0_hz.is_some();
        if voiced {
            voiced_frame_count += 1;
        }
        voiced_fraction = 0.95 * voiced_fraction + 0.05 * if voiced { 1.0 } else { 0.0 };

        let cepstral_envelope = cepstral_envelope(&power);
        let mut cepstral_delta = [0.0_f32; CEPSTRAL_COEFFICIENTS];
        if has_previous {
            for coefficient in 0..CEPSTRAL_COEFFICIENTS {
                cepstral_delta[coefficient] =
                    cepstral_envelope[coefficient] - previous_cepstrum[coefficient];
            }
        }
        previous_cepstrum = cepstral_envelope;

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

        let features = AcousticFrameFeatures {
            frame_index,
            start_ms: samples_to_ms(start),
            end_ms: samples_to_ms(start + ACOUSTIC_FRAME_SAMPLES),
            voice: VoiceFeatureView {
                cepstral_envelope,
                cepstral_delta,
                f0_hz,
                voicing_confidence,
                harmonicity: voicing_confidence,
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
            },
            quality: AcousticQualityMask {
                voiced,
                reliable_pitch,
                low_energy: rms_dbfs < -55.0,
                clipped: clipping_fraction > 0.005,
                transient,
            },
        };
        sink(features)?;
    }

    Ok(FeatureExtractionSummary {
        feature_schema: ACOUSTIC_FEATURE_SCHEMA_VERSION,
        frame_count,
        voiced_frame_count,
        low_energy_frame_count,
        retained_state_bytes_upper_bound: std::mem::size_of::<
            [f32; crate::native_engine::mel::N_FREQ_BINS],
        >() + std::mem::size_of::<[f32; CEPSTRAL_COEFFICIENTS]>()
            + 2 * std::mem::size_of::<[f32; ACOUSTIC_FRAME_SAMPLES]>(),
    })
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

fn estimate_f0(frame: &[f32], rms_dbfs: f32) -> (Option<f32>, f32) {
    if rms_dbfs < -60.0 {
        return (None, 0.0);
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
        (None, best_correlation.clamp(0.0, 1.0))
    } else {
        (
            Some(crate::native_engine::mel::SAMPLE_RATE as f32 / best_lag as f32),
            best_correlation.clamp(0.0, 1.0),
        )
    }
}

fn cepstral_envelope(power: &[f32; crate::native_engine::mel::N_FREQ_BINS]) -> [f32; 6] {
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

const VOICE_VECTOR_DIMENSIONS: usize = 8;
const CHANNEL_VECTOR_DIMENSIONS: usize = 8;
const CHANGE_SCALES_FRAMES: [usize; 5] = [10, 25, 50, 100, 200];
const CHANGE_RING_FRAMES: usize = 2 * CHANGE_SCALES_FRAMES[4] + 1;
const CHANGE_THRESHOLD: f32 = 0.34;
const CHANGE_HYSTERESIS_FRAMES: usize = 20;
const MIN_TRACKLET_FRAMES: usize = 20;
// Channel conditions are useful within a recording, but must remain secondary:
// the same vocal source may legitimately appear through both a nearby
// microphone and a distant loudspeaker.
const CHANNEL_DISTANCE_WEIGHT: f32 = 0.08;

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
    pub fused_score: f32,
    pub silence_gap: bool,
    pub snapped_to_word: bool,
    pub tiny_diarize_support: bool,
    pub vad_boundary: bool,
    /// Boundary forced at the guarded edge of a known-speaker interval.
    pub supervised_boundary: bool,
}

/// Compact speaker-homogeneous observation retained after frame segmentation.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticTracklet {
    pub tracklet_index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub frame_count: usize,
    pub voiced_frame_count: usize,
    pub voice_mean: [f32; VOICE_VECTOR_DIMENSIONS],
    pub voice_variance: [f32; VOICE_VECTOR_DIMENSIONS],
    pub channel_mean: [f32; CHANNEL_VECTOR_DIMENSIONS],
    pub channel_variance: [f32; CHANNEL_VECTOR_DIMENSIONS],
    pub change_confidence: f32,
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
}

struct AcousticSegmenter<'a> {
    hints: &'a AcousticBoundaryHints,
    forced_boundaries: BTreeMap<usize, ChangePointEvidence>,
    forced_boundary_count: usize,
    detected_boundaries: BTreeMap<usize, ChangePointEvidence>,
    pending_peak: Option<(usize, ChangePointEvidence)>,
    ring: VecDeque<AcousticFrameFeatures>,
    accumulator: TrackletAccumulator,
    tracklets: Vec<AcousticTracklet>,
    input_frame_count: usize,
    last_frame: Option<(usize, u64)>,
    maximum_retained_frames: usize,
}

impl<'a> AcousticSegmenter<'a> {
    fn new(hints: &'a AcousticBoundaryHints) -> FwResult<Self> {
        Self::new_with_supervised_boundaries(hints, &[])
    }

    fn new_with_supervised_boundaries(
        hints: &'a AcousticBoundaryHints,
        supervised_boundaries_ms: &[u64],
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
        let forced_boundaries = forced_boundary_map(hints, supervised_boundaries_ms);
        let forced_boundary_count = forced_boundaries.len();
        Ok(Self {
            hints,
            forced_boundaries,
            forced_boundary_count,
            detected_boundaries: BTreeMap::new(),
            pending_peak: None,
            ring: VecDeque::with_capacity(CHANGE_RING_FRAMES),
            accumulator: TrackletAccumulator::default(),
            tracklets: Vec::new(),
            input_frame_count: 0,
            last_frame: None,
            maximum_retained_frames: 0,
        })
    }

    fn push(&mut self, frame: AcousticFrameFeatures) -> FwResult<()> {
        validate_acoustic_frame(&frame)?;
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

        if self.ring.len() == CHANGE_RING_FRAMES {
            let center = CHANGE_SCALES_FRAMES[4];
            let center_index = self.ring[center].frame_index;
            let evidence = multiscale_change_evidence(&self.ring, center, self.hints);
            if evidence.fused_score >= CHANGE_THRESHOLD {
                match &mut self.pending_peak {
                    Some((peak_index, peak_evidence))
                        if center_index <= *peak_index + CHANGE_HYSTERESIS_FRAMES =>
                    {
                        if evidence.fused_score > peak_evidence.fused_score {
                            *peak_index = center_index;
                            *peak_evidence = evidence;
                        }
                    }
                    Some(_) => {
                        if let Some((peak_index, peak_evidence)) = self.pending_peak.take() {
                            insert_detected_boundary(
                                &mut self.detected_boundaries,
                                peak_index,
                                peak_evidence,
                            );
                        }
                        self.pending_peak = Some((center_index, evidence));
                    }
                    None => self.pending_peak = Some((center_index, evidence)),
                }
            } else if self.pending_peak.as_ref().is_some_and(|(peak_index, _)| {
                center_index > *peak_index + CHANGE_HYSTERESIS_FRAMES
            }) && let Some((peak_index, peak_evidence)) = self.pending_peak.take()
            {
                insert_detected_boundary(&mut self.detected_boundaries, peak_index, peak_evidence);
            }

            let oldest = self.ring.pop_front().ok_or_else(|| {
                FwError::InvalidRequest("segmentation ring unexpectedly empty".to_owned())
            })?;
            consume_segment_frame(
                oldest,
                &mut self.accumulator,
                &mut self.tracklets,
                &mut self.detected_boundaries,
                &mut self.forced_boundaries,
            );
        }
        Ok(())
    }

    fn finish(mut self) -> FwResult<(Vec<AcousticTracklet>, AcousticSegmentationSummary)> {
        if let Some((peak_index, peak_evidence)) = self.pending_peak {
            insert_detected_boundary(&mut self.detected_boundaries, peak_index, peak_evidence);
        }
        while let Some(frame) = self.ring.pop_front() {
            consume_segment_frame(
                frame,
                &mut self.accumulator,
                &mut self.tracklets,
                &mut self.detected_boundaries,
                &mut self.forced_boundaries,
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
        };
        Ok((self.tracklets, summary))
    }
}

/// Segment streamed acoustic frames with bounded multiscale Haar-like
/// left/right contrasts.
///
/// Only [`CHANGE_RING_FRAMES`] frames are retained. Emitted output consists of
/// compact sufficient statistics, never raw PCM or a dense CWT.
pub fn segment_acoustic_frames<I, C>(
    frames: I,
    hints: &AcousticBoundaryHints,
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
    let mut segmenter = AcousticSegmenter::new(hints)?;
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
    segmenter.finish()
}

fn validate_acoustic_frame(frame: &AcousticFrameFeatures) -> FwResult<()> {
    let start_sample = frame
        .frame_index
        .checked_mul(ACOUSTIC_HOP_SAMPLES)
        .ok_or_else(|| {
            FwError::InvalidRequest("acoustic frame index exceeds the v1 cadence range".to_owned())
        })?;
    let end_sample = start_sample
        .checked_add(ACOUSTIC_FRAME_SAMPLES)
        .ok_or_else(|| {
            FwError::InvalidRequest("acoustic frame end exceeds the v1 cadence range".to_owned())
        })?;
    let expected_start_ms = checked_samples_to_ms(start_sample).ok_or_else(|| {
        FwError::InvalidRequest("acoustic frame timestamp exceeds the v1 cadence range".to_owned())
    })?;
    let expected_end_ms = checked_samples_to_ms(end_sample).ok_or_else(|| {
        FwError::InvalidRequest("acoustic frame timestamp exceeds the v1 cadence range".to_owned())
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
        .copied()
        .all(bounded_scalar);
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
        && (0.0..=1.0).contains(&frame.channel.distortion_proxy);
    let quality_consistent = frame.quality.voiced == frame.voice.f0_hz.is_some()
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
        || !unit_interval(frame.voice.voicing_confidence)
        || !unit_interval(frame.voice.harmonicity)
        || !unit_interval(frame.voice.voiced_fraction)
        || !quality_consistent
    {
        return Err(FwError::InvalidRequest(
            "acoustic frames must use the exact v1 cadence with finite, internally consistent feature values"
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
                    fused_score: 1.0,
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
                fused_score: 1.0,
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
/// Acoustic frames overlap by 15 ms in v1. The end split therefore moves
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

fn multiscale_change_evidence(
    ring: &VecDeque<AcousticFrameFeatures>,
    center: usize,
    hints: &AcousticBoundaryHints,
) -> ChangePointEvidence {
    let mut scores = [0.0_f32; 5];
    let mut voice_distance = 0.0_f32;
    let mut channel_distance = 0.0_f32;
    for (scale_index, &scale) in CHANGE_SCALES_FRAMES.iter().enumerate() {
        let (voice, channel) = distribution_distance(ring, center, scale);
        scores[scale_index] = 0.78 * voice + 0.22 * channel.min(1.5);
        voice_distance = voice_distance.max(voice);
        channel_distance = channel_distance.max(channel);
    }
    let mut ranked_scores = scores;
    ranked_scores.sort_by(f32::total_cmp);
    let mut fused_score =
        0.55 * ranked_scores[4] + 0.30 * ranked_scores[3] + 0.15 * ranked_scores[2];
    let silence_gap = (center.saturating_sub(10)..center)
        .filter(|&index| ring[index].quality.low_energy)
        .count()
        >= 7
        && (center..center + 10)
            .filter(|&index| !ring[index].quality.low_energy)
            .count()
            >= 7;
    if silence_gap {
        fused_score = fused_score.max(0.85);
    }

    let raw_boundary_ms = ring[center].start_ms;
    let (boundary_ms, snapped_to_word) =
        snap_to_nearest(raw_boundary_ms, &hints.word_boundaries_ms, 80);
    let (boundary_ms, tiny_diarize_support) =
        snap_to_nearest(boundary_ms, &hints.tiny_diarize_boundaries_ms, 100);
    if tiny_diarize_support {
        fused_score = (fused_score + 0.10).min(1.0);
    }
    ChangePointEvidence {
        boundary_ms,
        voice_distance,
        channel_distance,
        multiscale_scores: scores,
        fused_score: fused_score.clamp(0.0, 1.0),
        silence_gap,
        snapped_to_word,
        tiny_diarize_support,
        vad_boundary: false,
        supervised_boundary: false,
    }
}

fn distribution_distance(
    ring: &VecDeque<AcousticFrameFeatures>,
    center: usize,
    scale: usize,
) -> (f32, f32) {
    let mut left_voice = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    let mut right_voice = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    let mut left_channel = [0.0_f32; CHANNEL_VECTOR_DIMENSIONS];
    let mut right_channel = [0.0_f32; CHANNEL_VECTOR_DIMENSIONS];
    let mut left_count = 0.0_f32;
    let mut right_count = 0.0_f32;
    for frame in ring.iter().take(center).skip(center - scale) {
        let (voice, channel) = compact_vectors(frame);
        if !frame.quality.low_energy {
            add_vector(&mut left_voice, &voice);
            add_vector(&mut left_channel, &channel);
            left_count += 1.0;
        }
    }
    for frame in ring.iter().skip(center).take(scale) {
        let (voice, channel) = compact_vectors(frame);
        if !frame.quality.low_energy {
            add_vector(&mut right_voice, &voice);
            add_vector(&mut right_channel, &channel);
            right_count += 1.0;
        }
    }
    if left_count < 3.0 || right_count < 3.0 {
        return (0.0, 0.0);
    }
    scale_vector(&mut left_voice, 1.0 / left_count);
    scale_vector(&mut right_voice, 1.0 / right_count);
    scale_vector(&mut left_channel, 1.0 / left_count);
    scale_vector(&mut right_channel, 1.0 / right_count);
    (
        euclidean_distance(&left_voice, &right_voice),
        euclidean_distance(&left_channel, &right_channel),
    )
}

fn compact_vectors(
    frame: &AcousticFrameFeatures,
) -> (
    [f32; VOICE_VECTOR_DIMENSIONS],
    [f32; CHANNEL_VECTOR_DIMENSIONS],
) {
    let mut voice = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    voice[..CEPSTRAL_COEFFICIENTS].copy_from_slice(&frame.voice.cepstral_envelope);
    voice[6] = frame.voice.f0_hz.map_or(0.0, |pitch| pitch.ln() / 6.0);
    voice[7] = frame.voice.harmonicity;
    let channel = [
        frame.channel.rms_dbfs / 40.0,
        frame.channel.spectral_centroid_hz / 8_000.0,
        frame.channel.spectral_bandwidth_hz / 8_000.0,
        frame.channel.spectral_flatness,
        frame.channel.spectral_tilt / 10.0,
        frame.channel.low_band_fraction,
        frame.channel.mid_band_fraction,
        frame.channel.high_band_fraction,
    ];
    (voice, channel)
}

fn add_vector<const N: usize>(destination: &mut [f32; N], source: &[f32; N]) {
    for (output, &input) in destination.iter_mut().zip(source) {
        *output += input;
    }
}

fn scale_vector<const N: usize>(vector: &mut [f32; N], scale: f32) {
    for value in vector {
        *value *= scale;
    }
}

fn euclidean_distance<const N: usize>(left: &[f32; N], right: &[f32; N]) -> f32 {
    (left
        .iter()
        .zip(right)
        .map(|(&left, &right)| {
            let difference = left - right;
            difference * difference
        })
        .sum::<f32>()
        / N as f32)
        .sqrt()
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
    original_frame_index: usize,
    evidence: ChangePointEvidence,
) {
    let frame_index = ms_to_frame(evidence.boundary_ms);
    let candidate = if frame_index.abs_diff(original_frame_index) <= 10 {
        frame_index
    } else {
        original_frame_index
    };
    boundaries
        .entry(candidate)
        .and_modify(|current| {
            if evidence.fused_score > current.fused_score {
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
    voice_mean: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_m2: [f32; VOICE_VECTOR_DIMENSIONS],
    channel_mean: [f32; CHANNEL_VECTOR_DIMENSIONS],
    channel_m2: [f32; CHANNEL_VECTOR_DIMENSIONS],
    anomaly_count: usize,
}

impl TrackletAccumulator {
    fn push(&mut self, frame: &AcousticFrameFeatures) {
        self.start_ms.get_or_insert(frame.start_ms);
        self.end_ms = frame.end_ms;
        self.frame_count += 1;
        self.voiced_frame_count += usize::from(u8::from(frame.quality.voiced));
        self.anomaly_count +=
            usize::from(u8::from(frame.quality.clipped || frame.quality.transient));
        let (voice, channel) = compact_vectors(frame);
        welford_update(
            &mut self.voice_mean,
            &mut self.voice_m2,
            &voice,
            self.frame_count,
        );
        welford_update(
            &mut self.channel_mean,
            &mut self.channel_m2,
            &channel,
            self.frame_count,
        );
    }

    fn finish(
        &mut self,
        tracklet_index: usize,
        boundary_evidence: Option<ChangePointEvidence>,
    ) -> Option<AcousticTracklet> {
        let start_ms = self.start_ms?;
        let denominator = self.frame_count.saturating_sub(1).max(1) as f32;
        let mut voice_variance = self.voice_m2;
        let mut channel_variance = self.channel_m2;
        scale_vector(&mut voice_variance, 1.0 / denominator);
        scale_vector(&mut channel_variance, 1.0 / denominator);
        let tracklet = AcousticTracklet {
            tracklet_index,
            start_ms,
            end_ms: self.end_ms,
            frame_count: self.frame_count,
            voiced_frame_count: self.voiced_frame_count,
            voice_mean: self.voice_mean,
            voice_variance,
            channel_mean: self.channel_mean,
            channel_variance,
            change_confidence: boundary_evidence
                .as_ref()
                .map_or(0.0, |evidence| evidence.fused_score),
            overlap_suspected: self.anomaly_count * 5 > self.frame_count,
            boundary_evidence,
        };
        *self = Self::default();
        Some(tracklet)
    }
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
    accumulator.push(&frame);
}

fn merge_compatible_adjacent_tracklets(tracklets: &mut Vec<AcousticTracklet>) {
    if tracklets.len() < 2 {
        return;
    }
    let mut merged = Vec::with_capacity(tracklets.len());
    for tracklet in tracklets.drain(..) {
        let compatible = merged.last().is_some_and(|previous: &AcousticTracklet| {
            tracklet.start_ms <= previous.end_ms + 50
                && previous.change_confidence < CHANGE_THRESHOLD
                && !previous
                    .boundary_evidence
                    .as_ref()
                    .is_some_and(|evidence| evidence.vad_boundary || evidence.supervised_boundary)
                && euclidean_distance(&previous.voice_mean, &tracklet.voice_mean) < 0.08
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
    let left_count = destination.frame_count as f32;
    let right_count = source.frame_count as f32;
    let total_count = total as f32;
    for index in 0..VOICE_VECTOR_DIMENSIONS {
        let left_mean = destination.voice_mean[index];
        let right_mean = source.voice_mean[index];
        let delta = right_mean - left_mean;
        destination.voice_mean[index] = left_mean + delta * right_count / total_count;
        let left_m2 = destination.voice_variance[index] * (left_count - 1.0).max(0.0);
        let right_m2 = source.voice_variance[index] * (right_count - 1.0).max(0.0);
        let between_m2 = delta * delta * (left_count / total_count) * right_count;
        destination.voice_variance[index] =
            (left_m2 + right_m2 + between_m2) / (total_count - 1.0).max(1.0);
    }
    for index in 0..CHANNEL_VECTOR_DIMENSIONS {
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
    destination.end_ms = source.end_ms;
    destination.frame_count = total;
    destination.voiced_frame_count += source.voiced_frame_count;
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
    pub applied_weight: f32,
    pub contradiction_score: Option<f32>,
}

#[derive(Debug, Clone)]
struct AcousticSpeakerProfile {
    speaker_ref: String,
    voice_median: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_mad: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_q25: [f32; VOICE_VECTOR_DIMENSIONS],
    voice_q75: [f32; VOICE_VECTOR_DIMENSIONS],
    channel_subprofiles: Vec<[f32; CHANNEL_VECTOR_DIMENSIONS]>,
    frame_count: usize,
    voiced_duration_ms: u64,
    reliability: f32,
    anchored: bool,
    soft_hint_contradiction: Option<f32>,
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
    profiles: BTreeMap<String, AcousticSpeakerProfile>,
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
    channel: [f32; CHANNEL_VECTOR_DIMENSIONS],
    frame_count: usize,
    voiced_duration_ms: u64,
    weight: f32,
    hard: bool,
}

/// Validate and apply `speaker-hints-v1` intervals to tracklet statistics.
pub fn enroll_known_speaker_profiles(
    tracklets: &[AcousticTracklet],
    request: &DiarizationRequest,
    constraints: Option<&SpeakerConstraints>,
    audio_duration_ms: u64,
) -> Result<SpeakerEnrollment, ProfileEnrollmentError> {
    request
        .validate(audio_duration_ms, constraints)
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
                        && tracklet.voiced_frame_count * 4 >= tracklet.frame_count
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
                tracklet.voiced_frame_count.max(1) as f32
            } else {
                ((hint.confidence as f32) * tracklet.voiced_frame_count.min(50) as f32).min(20.0)
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
                    channel: tracklet.channel_mean,
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
            applied_weight,
            contradiction_score: None,
        });
    }

    let mut profiles = BTreeMap::new();
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
        let provisional = robust_location(&provisional_source, |observation| observation.voice);
        let mut accepted = Vec::new();
        let mut maximum_contradiction = 0.0_f32;
        for observation in observations {
            let contradiction = euclidean_distance(&observation.voice, &provisional);
            let accept = observation.hard || contradiction <= 0.65;
            let hint_evidence = &mut evidence[observation.hint_index];
            if observation.hard {
                accepted.push(observation);
            } else if accept {
                hint_evidence.accepted_tracklet_count += 1;
                let prior = soft_priors
                    .entry((observation.tracklet_index, speaker_ref.clone()))
                    .or_default();
                *prior = prior.max(observation.weight.min(20.0));
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
        let profile = build_speaker_profile(
            speaker_ref.clone(),
            &accepted,
            (maximum_contradiction > 0.0).then_some(maximum_contradiction),
        );
        profiles.insert(speaker_ref, profile);
    }

    let cannot_links = hard_speaker_cannot_links(request);
    let summaries = profiles.values().map(profile_summary).collect::<Vec<_>>();
    Ok(SpeakerEnrollment {
        hint_document_sha256,
        summaries,
        evidence,
        profiles,
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
) -> [f32; VOICE_VECTOR_DIMENSIONS]
where
    F: Fn(&EnrollmentObservation) -> [f32; VOICE_VECTOR_DIMENSIONS],
{
    let mut output = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    for (dimension, output_value) in output.iter_mut().enumerate() {
        let values = observations
            .iter()
            .map(|observation| (vector(observation)[dimension], observation.weight))
            .collect::<Vec<_>>();
        *output_value = weighted_quantile(&values, 0.5);
    }
    output
}

fn build_speaker_profile(
    speaker_ref: String,
    observations: &[EnrollmentObservation],
    soft_hint_contradiction: Option<f32>,
) -> AcousticSpeakerProfile {
    let voice_median = robust_location(observations, |observation| observation.voice);
    let mut voice_mad = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    let mut voice_q25 = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    let mut voice_q75 = [0.0_f32; VOICE_VECTOR_DIMENSIONS];
    for dimension in 0..VOICE_VECTOR_DIMENSIONS {
        let values = observations
            .iter()
            .map(|observation| (observation.voice[dimension], observation.weight))
            .collect::<Vec<_>>();
        let deviations = observations
            .iter()
            .map(|observation| {
                (
                    (observation.voice[dimension] - voice_median[dimension]).abs(),
                    observation.weight,
                )
            })
            .collect::<Vec<_>>();
        voice_mad[dimension] = weighted_quantile(&deviations, 0.5).max(0.025);
        voice_q25[dimension] = weighted_quantile(&values, 0.25);
        voice_q75[dimension] = weighted_quantile(&values, 0.75);
    }
    let channel_subprofiles = build_channel_subprofiles(observations);
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
    let average_mad = voice_mad.iter().copied().sum::<f32>() / VOICE_VECTOR_DIMENSIONS as f32;
    let reliability =
        ((total_weight / 100.0).min(1.0) * (1.0 / (1.0 + 2.0 * average_mad))).clamp(0.0, 1.0);
    AcousticSpeakerProfile {
        speaker_ref,
        voice_median,
        voice_mad,
        voice_q25,
        voice_q75,
        channel_subprofiles,
        frame_count,
        voiced_duration_ms,
        reliability,
        anchored: observations.iter().any(|observation| observation.hard),
        soft_hint_contradiction,
    }
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

fn build_channel_subprofiles(
    observations: &[EnrollmentObservation],
) -> Vec<[f32; CHANNEL_VECTOR_DIMENSIONS]> {
    let mut subprofiles = Vec::<([f32; CHANNEL_VECTOR_DIMENSIONS], f32)>::new();
    for observation in observations {
        let nearest = subprofiles
            .iter()
            .enumerate()
            .map(|(index, (center, _))| (euclidean_distance(center, &observation.channel), index))
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
        channel_profile_count: u32::try_from(profile.channel_subprofiles.len()).unwrap_or(u32::MAX),
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
    pub change_confidence: f32,
    pub overlap_suspected: bool,
    pub hard_attribution: bool,
}

/// Non-biometric trace for one deterministic agglomeration step.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterMergeTrace {
    pub remaining_clusters: usize,
    pub distance: f32,
    pub left_anchor: Option<String>,
    pub right_anchor: Option<String>,
}

/// Diagnostics and assignments from constrained acoustic clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct AcousticClusteringResult {
    pub assignments: Vec<AcousticSpeakerAssignment>,
    pub profiles: Vec<SpeakerProfileSummary>,
    pub detected_speakers: usize,
    pub prototype_count: usize,
    pub prototype_cap: usize,
    pub cap_pressure: bool,
    pub constraints_satisfied: bool,
    pub bootstrap_stability: f32,
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
    pub constraints: Option<&'a SpeakerConstraints>,
    pub boundary_hints: &'a AcousticBoundaryHints,
}

/// Run the complete Rust-native acoustic diarization pipeline over canonical
/// 16 kHz mono PCM.
pub fn diarize_acoustic_pcm<C>(
    input: AcousticDiarizationInput<'_>,
    mut is_cancelled: C,
) -> FwResult<(DiarizationReport, DiarizationProjection)>
where
    C: FnMut() -> bool,
{
    let AcousticDiarizationInput {
        samples,
        normalized_input_sha256,
        segments,
        word_aligned,
        request,
        constraints,
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
        .validate(audio_duration_ms, constraints)
        .map_err(|error| FwError::InvalidRequest(error.to_string()))?;
    let supervised_boundaries_ms = supervised_enrollment_boundaries_ms(request);
    let mut segmenter = AcousticSegmenter::new_with_supervised_boundaries(
        boundary_hints,
        &supervised_boundaries_ms,
    )?;
    let extraction =
        extract_acoustic_features(samples, &mut is_cancelled, |frame| segmenter.push(frame))?;
    if is_cancelled() {
        return Err(FwError::Cancelled(
            "acoustic diarization cancelled after feature extraction".to_owned(),
        ));
    }
    let (tracklets, segmentation) = segmenter.finish()?;
    let enrollment =
        enroll_known_speaker_profiles(&tracklets, request, constraints, audio_duration_ms)
            .map_err(|error| FwError::InvalidRequest(error.to_string()))?;
    let clustering = cluster_acoustic_tracklets(
        &tracklets,
        &enrollment,
        constraints,
        usize::from(request.max_prototypes),
        is_cancelled,
    )?;
    let turns = diarization_turns_from_assignments(&clustering.assignments, audio_duration_ms)?;
    let projection = project_diarization_onto_segments(segments, &turns, word_aligned)?;
    let fallback_status = if !clustering.constraints_satisfied {
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
    let detected_speakers = u32::try_from(clustering.detected_speakers).map_err(|_| {
        FwError::InvalidRequest("detected speaker count exceeds report schema".to_owned())
    })?;
    let diagnostics = vec![
        format!(
            "features={} voiced={} retained_dsp_bytes<={}",
            extraction.frame_count,
            extraction.voiced_frame_count,
            extraction.retained_state_bytes_upper_bound
        ),
        format!(
            "tracklets={} acoustic_changes={} retained_segmentation_frames<={}",
            segmentation.tracklet_count,
            segmentation.acoustic_change_count,
            segmentation.maximum_retained_frames
        ),
        format!(
            "prototypes={}/{} cap_pressure={} bootstrap_stability={:.6}",
            clustering.prototype_count,
            clustering.prototype_cap,
            clustering.cap_pressure,
            clustering.bootstrap_stability
        ),
        format!(
            "mixed_segments={} overlap_suspected_segments={} calibration={}",
            projection.mixed_speaker_segment_indices.len(),
            projection.overlap_suspected_segment_indices.len(),
            clustering.calibration_status
        ),
    ];
    Ok((
        DiarizationReport {
            implementation: "native-acoustic-v1".to_owned(),
            contract_version: ACOUSTIC_DIARIZATION_CONTRACT_VERSION.to_owned(),
            feature_schema: ACOUSTIC_FEATURE_SCHEMA_VERSION.to_owned(),
            normalized_input_sha256: normalized_input_sha256.to_owned(),
            hint_document_sha256: enrollment.hint_document_sha256,
            turns,
            profiles: clustering.profiles,
            detected_speakers,
            constraints_satisfied: clustering.constraints_satisfied,
            fallback_status,
            diagnostics,
        },
        projection,
    ))
}

#[derive(Debug, Clone)]
struct AcousticPrototype {
    members: Vec<usize>,
    voice: [f32; VOICE_VECTOR_DIMENSIONS],
    variance: [f32; VOICE_VECTOR_DIMENSIONS],
    channel: [f32; CHANNEL_VECTOR_DIMENSIONS],
    frame_count: usize,
    earliest_ms: u64,
    hard_anchor: Option<String>,
}

#[derive(Debug, Clone)]
struct AcousticCluster {
    prototype_members: Vec<usize>,
    voice: [f32; VOICE_VECTOR_DIMENSIONS],
    scale: [f32; VOICE_VECTOR_DIMENSIONS],
    channel: [f32; CHANNEL_VECTOR_DIMENSIONS],
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
    constraints: Option<&SpeakerConstraints>,
    requested_prototype_cap: usize,
    mut is_cancelled: C,
) -> FwResult<AcousticClusteringResult>
where
    C: FnMut() -> bool,
{
    if requested_prototype_cap == 0 || requested_prototype_cap > 512 {
        return Err(FwError::InvalidRequest(
            "acoustic-v1 prototype cap must be within 1..=512".to_owned(),
        ));
    }
    if is_cancelled() {
        return Err(FwError::Cancelled(
            "acoustic clustering cancelled before prototype construction".to_owned(),
        ));
    }
    validate_tracklet_timeline(tracklets)?;
    crate::model::validate_speaker_constraints(constraints)
        .map_err(|error| FwError::InvalidRequest(error.to_string()))?;
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
            detected_speakers: 0,
            prototype_count: 0,
            prototype_cap: requested_prototype_cap,
            cap_pressure,
            constraints_satisfied: constraint_allows_zero(constraints),
            bootstrap_stability: 0.0,
            calibration_status: "insufficient_evidence",
            merge_trace: Vec::new(),
        });
    }

    let initial_clusters = initial_clusters(&prototypes, enrollment);
    let requested_minimum = constraints
        .and_then(|constraints| constraints.num_speakers.or(constraints.min_speakers))
        .map_or(1, |value| value as usize);
    let count_constraints_feasible = requested_minimum <= initial_clusters.len();
    let count_policy = if count_constraints_feasible {
        resolve_count_policy(constraints, initial_clusters.len())?
    } else {
        SpeakerCountPolicy {
            min: initial_clusters.len(),
            max: initial_clusters.len(),
            exact: Some(initial_clusters.len()),
        }
    };
    let (clusters, merge_trace) = agglomerate_clusters(
        initial_clusters,
        &enrollment.cannot_links,
        count_policy,
        &mut is_cancelled,
    )?;
    let labels = canonical_cluster_labels(&clusters, enrollment);
    let require_exact_coverage = count_constraints_feasible
        && constraints.is_some_and(|constraints| constraints.num_speakers.is_some());
    let mut assignments = viterbi_assignments(
        tracklets,
        &clusters,
        &labels,
        enrollment,
        require_exact_coverage,
        &mut is_cancelled,
    )?;
    let mut exact_coverage_satisfied = true;
    if require_exact_coverage {
        let mut covered_assignments = assignments.clone();
        exact_coverage_satisfied = ensure_cluster_coverage(
            tracklets,
            &clusters,
            &labels,
            enrollment,
            &mut covered_assignments,
        );
        if exact_coverage_satisfied {
            assignments = covered_assignments;
        }
    }
    let detected_speakers = assignments
        .iter()
        .filter_map(|assignment| assignment.speaker_ref.as_ref())
        .collect::<BTreeSet<_>>()
        .len();
    let constraints_satisfied = count_constraints_feasible
        && exact_coverage_satisfied
        && speaker_count_satisfies(detected_speakers, constraints);
    let profiles = clustering_profile_summaries(&clusters, &labels, enrollment);
    let bootstrap_stability = clusters
        .iter()
        .map(|cluster| cluster.reliability)
        .sum::<f32>()
        / clusters.len() as f32;
    Ok(AcousticClusteringResult {
        assignments,
        profiles,
        detected_speakers,
        prototype_count: prototypes.len(),
        prototype_cap: requested_prototype_cap,
        cap_pressure,
        constraints_satisfied,
        bootstrap_stability,
        calibration_status: "heuristic_uncalibrated",
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
    let mut previous_start = 0u64;
    for (index, assignment) in assignments.iter().enumerate() {
        if assignment.end_ms <= assignment.start_ms
            || assignment.end_ms > audio_duration_ms
            || (index > 0 && assignment.start_ms < previous_start)
            || !assignment.speaker_confidence.is_finite()
            || !(0.0..=1.0).contains(&assignment.speaker_confidence)
            || !assignment.change_confidence.is_finite()
            || !(0.0..=1.0).contains(&assignment.change_confidence)
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
            {
                previous.end_ms = previous.end_ms.max(turn.end_ms);
                previous.speaker_confidence = minimum_optional_confidence(
                    previous.speaker_confidence,
                    turn.speaker_confidence,
                );
                previous.change_confidence =
                    maximum_optional_confidence(previous.change_confidence, turn.change_confidence);
                previous.overlap_suspected |= turn.overlap_suspected;
                previous.hard_hint_attributed |= turn.hard_hint_attributed;
                continue;
            }
        }
        turns.push(turn);
    }
    Ok(turns)
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
                    && (!word_aligned || index == 0 || start + 1e-9 >= previous_end) =>
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
                "tracklets must have finite non-negative statistics within acoustic-v1 bounds, valid counts and confidence, unique indexes, and ordered positive intervals".to_owned(),
            ));
        };
        if next_total_frame_count > maximum_total_frames
            || tracklet.end_ms <= tracklet.start_ms
            || tracklet.start_ms < previous_end.saturating_sub(25)
            || tracklet.end_ms < previous_end
            || tracklet.frame_count == 0
            || tracklet.voiced_frame_count > tracklet.frame_count
            || !indexes.insert(tracklet.tracklet_index)
            || !tracklet.change_confidence.is_finite()
            || !(0.0..=1.0).contains(&tracklet.change_confidence)
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
                "tracklets must have finite non-negative statistics within acoustic-v1 bounds, valid counts and confidence, unique indexes, and ordered positive intervals".to_owned(),
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
            variance: tracklet.voice_variance.map(|value| value.max(0.025)),
            channel: tracklet.channel_mean,
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
    variance_normalized_distance(&left.voice, &left.variance, &right.voice, &right.variance)
        + CHANNEL_DISTANCE_WEIGHT * euclidean_distance(&left.channel, &right.channel).min(1.0)
}

fn merge_prototype(destination: &mut AcousticPrototype, source: &AcousticPrototype) {
    let total = destination.frame_count + source.frame_count;
    let left_weight = destination.frame_count as f32 / total as f32;
    let right_weight = source.frame_count as f32 / total as f32;
    for dimension in 0..VOICE_VECTOR_DIMENSIONS {
        let delta = destination.voice[dimension] - source.voice[dimension];
        destination.voice[dimension] =
            left_weight * destination.voice[dimension] + right_weight * source.voice[dimension];
        destination.variance[dimension] = left_weight * destination.variance[dimension]
            + right_weight * source.variance[dimension]
            + left_weight * right_weight * delta * delta;
    }
    for dimension in 0..CHANNEL_VECTOR_DIMENSIONS {
        destination.channel[dimension] =
            left_weight * destination.channel[dimension] + right_weight * source.channel[dimension];
    }
    destination.members.extend_from_slice(&source.members);
    destination.members.sort_unstable();
    destination.frame_count = total;
    destination.earliest_ms = destination.earliest_ms.min(source.earliest_ms);
    if destination.hard_anchor.is_none() {
        destination.hard_anchor.clone_from(&source.hard_anchor);
    }
}

fn variance_normalized_distance<const N: usize>(
    left: &[f32; N],
    left_scale: &[f32; N],
    right: &[f32; N],
    right_scale: &[f32; N],
) -> f32 {
    (left
        .iter()
        .zip(left_scale)
        .zip(right.iter().zip(right_scale))
        .map(|((&left, &left_scale), (&right, &right_scale))| {
            let difference = left - right;
            difference * difference / (left_scale + right_scale + 0.05)
        })
        .sum::<f32>()
        / N as f32)
        .sqrt()
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
        scale: prototype.variance.map(|value| value.max(0.025)),
        channel: prototype.channel,
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
        scale,
        channel: profile
            .channel_subprofiles
            .first()
            .copied()
            .unwrap_or([0.0; CHANNEL_VECTOR_DIMENSIONS]),
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
        let delta = destination.voice[dimension] - source.voice[dimension];
        destination.voice[dimension] =
            left_weight * destination.voice[dimension] + right_weight * source.voice[dimension];
        destination.scale[dimension] = left_weight * destination.scale[dimension]
            + right_weight * source.scale[dimension]
            + left_weight * right_weight * delta * delta;
    }
    for dimension in 0..CHANNEL_VECTOR_DIMENSIONS {
        destination.channel[dimension] =
            left_weight * destination.channel[dimension] + right_weight * source.channel[dimension];
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
    variance_normalized_distance(&left.voice, &left.scale, &right.voice, &right.scale)
        + CHANNEL_DISTANCE_WEIGHT * euclidean_distance(&left.channel, &right.channel).min(1.0)
}

#[derive(Debug, Clone, Copy)]
struct SpeakerCountPolicy {
    min: usize,
    max: usize,
    exact: Option<usize>,
}

fn resolve_count_policy(
    constraints: Option<&SpeakerConstraints>,
    available: usize,
) -> FwResult<SpeakerCountPolicy> {
    let exact = constraints
        .and_then(|value| value.num_speakers)
        .map(|value| value as usize);
    let min = exact.unwrap_or_else(|| {
        constraints
            .and_then(|value| value.min_speakers)
            .map_or(1, |value| value as usize)
    });
    let max = exact.unwrap_or_else(|| {
        constraints
            .and_then(|value| value.max_speakers)
            .map_or(available.min(8).max(min), |value| value as usize)
    });
    if min == 0 || min > max || min > available {
        return Err(FwError::InvalidRequest(format!(
            "speaker-count constraints require {min}..={max} profiles but only {available} are available"
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
        (Some(left), Some(right)) if left != right => {
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
    let penalty = 0.035 * active as f32 * VOICE_VECTOR_DIMENSIONS as f32 * total_weight.ln();
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
    require_known: bool,
    is_cancelled: &mut C,
) -> FwResult<Vec<AcousticSpeakerAssignment>>
where
    C: FnMut() -> bool,
{
    if tracklets.is_empty() {
        return Ok(Vec::new());
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
                if let Some(hard) = enrollment.hard_assignments.get(&tracklet.tracklet_index)
                    && labels[cluster_index] != *hard
                {
                    return 1_000_000.0;
                }
                let mut cost = tracklet_cluster_distance(tracklet, cluster);
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
        let unknown_cost = if require_known
            || enrollment
                .hard_assignments
                .contains_key(&tracklet.tracklet_index)
        {
            1_000_000.0
        } else if tracklet.frame_count < MIN_TRACKLET_FRAMES
            || tracklet.voiced_frame_count * 4 < tracklet.frame_count
        {
            0.20
        } else {
            0.90
        };
        costs.push(unknown_cost);
        emissions.push(costs);
    }

    let mut previous = emissions[0].clone();
    let mut backpointers = vec![vec![0usize; state_count]; tracklets.len()];
    for time in 1..tracklets.len() {
        let mut current = vec![f32::INFINITY; state_count];
        for state in 0..state_count {
            for (previous_state, previous_cost) in previous.iter().enumerate().take(state_count) {
                let switch_penalty = if state == previous_state {
                    0.0
                } else if state == unknown_state || previous_state == unknown_state {
                    0.08
                } else {
                    0.18
                };
                let candidate = *previous_cost + switch_penalty + emissions[time][state];
                if candidate < current[state] {
                    current[state] = candidate;
                    backpointers[time][state] = previous_state;
                }
            }
        }
        previous = current;
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
            let confidence = if hard {
                1.0
            } else {
                (margin / (margin + 0.5) * clusters[state].reliability).clamp(0.0, 1.0)
            };
            if !hard && !require_known && (chosen_cost > 1.35 || confidence < 0.30) {
                unknown_assignment(tracklet)
            } else {
                AcousticSpeakerAssignment {
                    tracklet_index: tracklet.tracklet_index,
                    start_ms: tracklet.start_ms,
                    end_ms: tracklet.end_ms,
                    speaker_ref: Some(labels[state].clone()),
                    speaker_confidence: confidence,
                    change_confidence: tracklet.change_confidence,
                    overlap_suspected: tracklet.overlap_suspected,
                    hard_attribution: hard,
                }
            }
        })
        .collect())
}

fn tracklet_cluster_distance(tracklet: &AcousticTracklet, cluster: &AcousticCluster) -> f32 {
    variance_normalized_distance(
        &tracklet.voice_mean,
        &tracklet.voice_variance.map(|value| value.max(0.025)),
        &cluster.voice,
        &cluster.scale,
    ) + CHANNEL_DISTANCE_WEIGHT
        * euclidean_distance(&tracklet.channel_mean, &cluster.channel).min(1.0)
}

fn unknown_assignment(tracklet: &AcousticTracklet) -> AcousticSpeakerAssignment {
    AcousticSpeakerAssignment {
        tracklet_index: tracklet.tracklet_index,
        start_ms: tracklet.start_ms,
        end_ms: tracklet.end_ms,
        speaker_ref: None,
        speaker_confidence: 0.0,
        change_confidence: tracklet.change_confidence,
        overlap_suspected: tracklet.overlap_suspected,
        hard_attribution: false,
    }
}

fn ensure_cluster_coverage(
    tracklets: &[AcousticTracklet],
    clusters: &[AcousticCluster],
    labels: &[String],
    enrollment: &SpeakerEnrollment,
    assignments: &mut [AcousticSpeakerAssignment],
) -> bool {
    let mut counts = BTreeMap::<String, usize>::new();
    for assignment in assignments.iter() {
        if let Some(label) = &assignment.speaker_ref {
            *counts.entry(label.clone()).or_default() += 1;
        }
    }
    for (cluster_index, label) in labels.iter().enumerate() {
        if counts.get(label).copied().unwrap_or(0) > 0 {
            continue;
        }
        let candidate = tracklets
            .iter()
            .enumerate()
            .filter(|(index, tracklet)| {
                assignments[*index]
                    .speaker_ref
                    .as_ref()
                    .is_none_or(|current| counts.get(current).copied().unwrap_or(0) > 1)
                    && enrollment
                        .hard_assignments
                        .get(&tracklet.tracklet_index)
                        .is_none_or(|hard| hard == label)
            })
            .map(|(index, tracklet)| {
                (
                    tracklet_cluster_distance(tracklet, &clusters[cluster_index]),
                    index,
                )
            })
            .min_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)))
            .map(|(_, index)| index);
        let Some(candidate) = candidate else {
            return false;
        };
        if let Some(previous) = assignments[candidate].speaker_ref.take()
            && let Some(count) = counts.get_mut(&previous)
        {
            *count = count.saturating_sub(1);
        }
        assignments[candidate].speaker_ref = Some(label.clone());
        counts.insert(label.clone(), 1);
        assignments[candidate].speaker_confidence =
            clusters[cluster_index].reliability.clamp(0.30, 1.0);
        assignments[candidate].hard_attribution = enrollment
            .hard_assignments
            .contains_key(&tracklets[candidate].tracklet_index);
    }
    true
}

fn constraint_allows_zero(constraints: Option<&SpeakerConstraints>) -> bool {
    constraints.is_none_or(|constraints| {
        constraints.num_speakers.is_none()
            && constraints.min_speakers.is_none_or(|minimum| minimum == 0)
    })
}

fn speaker_count_satisfies(count: usize, constraints: Option<&SpeakerConstraints>) -> bool {
    constraints.is_none_or(|constraints| {
        constraints
            .num_speakers
            .is_none_or(|exact| count == exact as usize)
            && constraints
                .min_speakers
                .is_none_or(|minimum| count >= minimum as usize)
            && constraints
                .max_speakers
                .is_none_or(|maximum| count <= maximum as usize)
    })
}

fn clustering_profile_summaries(
    clusters: &[AcousticCluster],
    labels: &[String],
    enrollment: &SpeakerEnrollment,
) -> Vec<SpeakerProfileSummary> {
    clusters
        .iter()
        .enumerate()
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
                    channel_profile_count: 1,
                    anchored: cluster.hard_anchor.is_some(),
                    soft_hint_contradiction: None,
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        ACOUSTIC_CANCELLATION_INTERVAL_FRAMES, ACOUSTIC_FRAME_SAMPLES, ACOUSTIC_HOP_SAMPLES,
        AcousticBoundaryHints, AcousticDiarizationInput, AcousticFrameFeatures,
        AcousticQualityMask, AcousticSegmenter, AcousticSpeakerAssignment, AcousticTracklet,
        CalibrationObservation, ChannelFeatureView, CorpusRecordingManifest,
        DIARIZATION_CORPUS_MANIFEST_SCHEMA_VERSION, DIARIZATION_HYPOTHESIS_SCHEMA_VERSION,
        DIARIZATION_REFERENCE_SCHEMA_VERSION, DiarizationCorpusManifest,
        DiarizationHypothesisDocument, DiarizationReferenceDocument, DiarizationScorerConfig,
        EvaluationHintPolicy, EvaluationOverlapPolicy, EvaluationPerformanceObservation,
        EvaluationRegion, EvaluationSpeakerHint, EvaluationSplit, EvaluationTurn, LeakageKind,
        ProfileEnrollmentCode, ScoringTurn, VoiceFeatureView, audit_diarization_manifest,
        cluster_acoustic_tracklets, diarization_turns_from_assignments, diarize_acoustic_pcm,
        enroll_known_speaker_profiles, extract_acoustic_features, merge_tracklet_statistics,
        parse_diarization_corpus_manifest, parse_diarization_reference,
        project_diarization_onto_segments, score_calibration, score_change_points,
        score_diarization, score_diarization_documents, segment_acoustic_frames,
        verify_authoritative_score_hash, verify_leakage_audit_hash,
    };
    use crate::FwError;
    use crate::model::{
        DiarizationEngine, DiarizationFallbackStatus, DiarizationRequest, DiarizationTurn,
        KnownSpeakerInterval, KnownSpeakerPolicy, SpeakerConstraints, TranscriptionSegment,
    };
    use sha2::{Digest, Sha256};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
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
        let constraints = SpeakerConstraints {
            num_speakers: Some(1),
            min_speakers: None,
            max_speakers: None,
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
                constraints: Some(&constraints),
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
                constraints: Some(&constraints),
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

        assert_eq!(report.detected_speakers, 1);
        assert!(report.constraints_satisfied);
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
        let constraints = SpeakerConstraints {
            num_speakers: Some(1),
            ..SpeakerConstraints::default()
        };

        let (report, projection) = diarize_acoustic_pcm(
            AcousticDiarizationInput {
                samples: &samples,
                normalized_input_sha256: SYNTHETIC_INPUT_SHA256,
                segments: &segments,
                word_aligned: false,
                request: &request,
                constraints: Some(&constraints),
                boundary_hints: &AcousticBoundaryHints::default(),
            },
            || false,
        )
        .expect("bounded hard hint should create guarded enrollment tracklets");

        assert_eq!(report.detected_speakers, 1);
        assert!(report.constraints_satisfied);
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
                constraints: None,
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
    }

    #[test]
    fn silence_is_unvoiced_instead_of_fabricating_pitch() {
        let frames = features(&vec![0.0; crate::native_engine::mel::SAMPLE_RATE / 2]);
        assert!(!frames.is_empty());
        assert!(frames.iter().all(|frame| frame.voice.f0_hz.is_none()));
        assert!(frames.iter().all(|frame| frame.quality.low_energy));
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
                cepstral_envelope: [voice_value; 6],
                cepstral_delta: [0.0; 6],
                f0_hz: (!low_energy).then_some(120.0 + 80.0 * voice_value),
                voicing_confidence: if low_energy { 0.0 } else { 0.9 },
                harmonicity: if low_energy { 0.0 } else { 0.9 },
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
            },
            quality: AcousticQualityMask {
                voiced: !low_energy,
                reliable_pitch: !low_energy,
                low_energy,
                clipped: false,
                transient,
            },
        }
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
            .max_by(|left, right| left.fused_score.total_cmp(&right.fused_score))
            .expect("change evidence");
        assert!(evidence.boundary_ms.abs_diff(3_250) <= 100);
        assert!(evidence.snapped_to_word);
        assert!(summary.maximum_retained_frames <= 401);
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
        let (tracklets, summary) = segmenter.finish().expect("finish");
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
        destination.voice_variance = [0.0; 8];
        destination.channel_variance = [0.0; 8];
        source.voice_variance = [0.0; 8];
        source.channel_variance = [0.0; 8];

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
        assert!(error.to_string().contains("exact v1 cadence"));

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
        AcousticTracklet {
            tracklet_index: index,
            start_ms,
            end_ms,
            frame_count,
            voiced_frame_count: voiced_frames.min(frame_count),
            voice_mean: [voice; 8],
            voice_variance: [0.01; 8],
            channel_mean: [channel; 8],
            channel_variance: [0.01; 8],
            change_confidence: 0.8,
            overlap_suspected: false,
            boundary_evidence: None,
        }
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
        let error = enroll_known_speaker_profiles(&tracklets, &request, None, 1_000)
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
            enroll_known_speaker_profiles(&tracklets, &request, None, 1_000).expect("enrollment");
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
            enroll_known_speaker_profiles(&tracklets, &request, None, 1_000).expect("enrollment");
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
            enroll_known_speaker_profiles(&tracklets, &request, None, 2_000).expect("enrollment");
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
            enroll_known_speaker_profiles(&tracklets, &request, None, 2_000).expect("enrollment");
        assert_eq!(enrollment.summaries[0].channel_profile_count, 2);
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
            enroll_known_speaker_profiles(&tracklets, &request, None, 1_000).expect("enrollment");
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
            known_intervals: vec![second, first],
            enrollment_edge_guard_ms: 50,
            ..DiarizationRequest::default()
        };
        let enrollment_a = enroll_known_speaker_profiles(&tracklets, &request_a, None, 2_000)
            .expect("enrollment a");
        let enrollment_b = enroll_known_speaker_profiles(&tracklets, &request_b, None, 2_000)
            .expect("enrollment b");
        assert_eq!(
            enrollment_a.hint_document_sha256,
            enrollment_b.hint_document_sha256
        );
        assert_eq!(enrollment_a.summaries, enrollment_b.summaries);
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
            enroll_known_speaker_profiles(&tracklets, &request, None, 2_000).expect("enrollment");
        assert!(
            enrollment
                .cannot_links
                .contains(&("alice".to_owned(), "bob".to_owned()))
        );
    }

    #[test]
    fn hard_anchors_survive_clustering_and_smoothing_exactly() {
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
            profile_tracklet(1, 1_200, 1_700, 2.0, 1.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, None, 2_000).expect("enrollment");
        let constraints = SpeakerConstraints {
            num_speakers: Some(2),
            ..SpeakerConstraints::default()
        };
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&constraints), 512, || false)
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
            known_intervals: vec![known_interval(
                "SPEAKER_00",
                0,
                500,
                KnownSpeakerPolicy::HardMustLink,
            )],
            enrollment_edge_guard_ms: 0,
            ..DiarizationRequest::default()
        };
        let tracklets = vec![
            profile_tracklet(0, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 3.0, 0.0, 50),
        ];
        let constraints = SpeakerConstraints {
            num_speakers: Some(2),
            ..SpeakerConstraints::default()
        };
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, Some(&constraints), 1_000)
                .expect("enrollment");
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&constraints), 512, || false)
                .expect("cluster");
        assert_eq!(
            result.assignments[0].speaker_ref.as_deref(),
            Some("SPEAKER_00")
        );
        assert_eq!(
            result.assignments[1].speaker_ref.as_deref(),
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
        let tracklets = vec![profile_tracklet(0, 200, 700, 0.0, 0.0, 50)];
        let mut enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, None, 1_000).expect("enrollment");
        assert_eq!(enrollment.evidence[0].usable_tracklet_count, 0);
        enrollment.evidence.clear();

        let result = cluster_acoustic_tracklets(&tracklets, &enrollment, None, 512, || false)
            .expect("cluster");
        assert_eq!(result.profiles[0].speaker_ref, "SPEAKER_01");
    }

    #[test]
    fn compatible_bounded_solution_survives_exhausted_cannot_link_heap() {
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
            profile_tracklet(1, 500, 1_000, 1.0, 0.0, 50),
        ];
        let constraints = SpeakerConstraints {
            min_speakers: Some(1),
            max_speakers: Some(3),
            ..SpeakerConstraints::default()
        };
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, Some(&constraints), 1_000)
                .expect("enrollment");
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&constraints), 512, || false)
                .expect("the two anchored speakers already satisfy the bounded request");
        assert_eq!(result.detected_speakers, 2);
        assert!(result.constraints_satisfied);
        assert!(result.merge_trace.is_empty());
    }

    #[test]
    fn soft_enrollment_influences_but_does_not_force_unrelated_speech() {
        let request = DiarizationRequest {
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
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, None, 2_000).expect("enrollment");
        let constraints = SpeakerConstraints {
            num_speakers: Some(2),
            ..SpeakerConstraints::default()
        };
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&constraints), 512, || false)
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
    fn exact_and_bounded_speaker_counts_are_enforced() {
        let request = DiarizationRequest::default();
        let tracklets = vec![
            profile_tracklet(4, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(9, 500, 1_000, 2.0, 0.0, 50),
            profile_tracklet(2, 1_000, 1_500, 4.0, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, None, 1_500).expect("enrollment");
        let exact = SpeakerConstraints {
            num_speakers: Some(3),
            ..SpeakerConstraints::default()
        };
        let exact_result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&exact), 512, || false)
                .expect("exact clustering");
        assert_eq!(exact_result.detected_speakers, 3);
        assert!(exact_result.constraints_satisfied);

        let bounded = SpeakerConstraints {
            min_speakers: Some(1),
            max_speakers: Some(2),
            ..SpeakerConstraints::default()
        };
        let bounded_result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&bounded), 512, || false)
                .expect("bounded clustering");
        assert!((1..=2).contains(&bounded_result.detected_speakers));
        assert!(bounded_result.constraints_satisfied);
    }

    #[test]
    fn infeasible_speaker_count_reports_unsatisfied_instead_of_aborting() {
        let request = DiarizationRequest::default();
        let tracklets = vec![profile_tracklet(0, 0, 500, 0.0, 0.0, 50)];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, None, 500).expect("enrollment");
        let impossible = SpeakerConstraints {
            num_speakers: Some(2),
            ..SpeakerConstraints::default()
        };
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&impossible), 512, || false)
                .expect("insufficient evidence is a fallback state, not a malformed request");
        assert!(!result.constraints_satisfied);
        assert!(result.detected_speakers <= 1);
    }

    #[test]
    fn malformed_speaker_count_remains_an_error_when_evidence_is_insufficient() {
        let tracklets = vec![profile_tracklet(0, 0, 500, 0.0, 0.0, 50)];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &DiarizationRequest::default(), None, 500)
                .expect("enrollment");
        let malformed = SpeakerConstraints {
            min_speakers: Some(3),
            max_speakers: Some(2),
            ..SpeakerConstraints::default()
        };
        let error =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&malformed), 512, || false)
                .expect_err("malformed constraints must not become an evidence fallback");
        assert!(error.to_string().contains("min <= exact <= max"));
    }

    #[test]
    fn impossible_exact_coverage_does_not_leave_partial_forced_assignments() {
        let request = DiarizationRequest {
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
        let constraints = SpeakerConstraints {
            num_speakers: Some(2),
            ..SpeakerConstraints::default()
        };
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, Some(&constraints), 500)
                .expect("enrollment");
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&constraints), 512, || false)
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
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, None, 1_500).expect("enrollment");
        let result = cluster_acoustic_tracklets(&tracklets, &enrollment, None, 512, || false)
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
            enroll_known_speaker_profiles(&tracklets, &request, None, 1_000).expect("enrollment");
        let result = cluster_acoustic_tracklets(&tracklets, &enrollment, None, 512, || false)
            .expect("cluster");
        assert_eq!(result.detected_speakers, 1);
        assert_eq!(
            result.assignments[0].speaker_ref,
            result.assignments[1].speaker_ref
        );
    }

    #[test]
    fn separated_voices_do_not_false_merge_when_two_speakers_are_required() {
        let request = DiarizationRequest::default();
        let tracklets = vec![
            profile_tracklet(0, 0, 500, -1.5, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 1.5, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, None, 1_000).expect("enrollment");
        let constraints = SpeakerConstraints {
            num_speakers: Some(2),
            ..SpeakerConstraints::default()
        };
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&constraints), 512, || false)
                .expect("cluster");
        assert_ne!(
            result.assignments[0].speaker_ref,
            result.assignments[1].speaker_ref
        );
    }

    #[test]
    fn speaker_return_preserves_the_first_speaker_label() {
        let request = DiarizationRequest::default();
        let tracklets = vec![
            profile_tracklet(0, 0, 500, 0.0, 0.0, 50),
            profile_tracklet(1, 500, 1_000, 2.0, 0.0, 50),
            profile_tracklet(2, 1_000, 1_500, 0.0, 0.0, 50),
        ];
        let enrollment =
            enroll_known_speaker_profiles(&tracklets, &request, None, 1_500).expect("enrollment");
        let constraints = SpeakerConstraints {
            num_speakers: Some(2),
            ..SpeakerConstraints::default()
        };
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&constraints), 512, || false)
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
            enroll_known_speaker_profiles(&tracklets, &request, None, 30).expect("enrollment");
        let result = cluster_acoustic_tracklets(&tracklets, &enrollment, None, 512, || false)
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
        let constraints = SpeakerConstraints {
            num_speakers: Some(2),
            ..SpeakerConstraints::default()
        };
        let first_enrollment =
            enroll_known_speaker_profiles(&first_tracklets, &request, None, 1_500)
                .expect("first enrollment");
        let second_enrollment =
            enroll_known_speaker_profiles(&second_tracklets, &request, None, 1_500)
                .expect("second enrollment");
        let first = cluster_acoustic_tracklets(
            &first_tracklets,
            &first_enrollment,
            Some(&constraints),
            512,
            || false,
        )
        .expect("first cluster");
        let repeated = cluster_acoustic_tracklets(
            &first_tracklets,
            &first_enrollment,
            Some(&constraints),
            512,
            || false,
        )
        .expect("repeated cluster");
        let second = cluster_acoustic_tracklets(
            &second_tracklets,
            &second_enrollment,
            Some(&constraints),
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
            enroll_known_speaker_profiles(&tracklets, &request, None, 60_000).expect("enrollment");
        let constraints = SpeakerConstraints {
            num_speakers: Some(1),
            ..SpeakerConstraints::default()
        };
        let result =
            cluster_acoustic_tracklets(&tracklets, &enrollment, Some(&constraints), 32, || false)
                .expect("cluster");
        assert!(result.cap_pressure);
        assert_eq!(result.prototype_count, 32);
        assert_eq!(result.prototype_cap, 32);
        assert_eq!(result.detected_speakers, 1);
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
            enroll_known_speaker_profiles(&tracklets, &request, None, 10_000).expect("enrollment");
        let mut checks = 0usize;
        let error = cluster_acoustic_tracklets(&tracklets, &enrollment, None, 64, || {
            checks += 1;
            checks >= 2
        })
        .expect_err("cancel");
        assert!(error.to_string().contains("clustering cancelled"));
    }

    #[test]
    fn clustering_rejects_invalid_tracklet_statistics_and_counts() {
        let request = DiarizationRequest::default();
        let enrollment =
            enroll_known_speaker_profiles(&[], &request, None, 100).expect("enrollment");
        let mut invalid = profile_tracklet(0, 0, 100, 0.0, 0.0, 10);
        invalid.channel_variance[0] = f32::NAN;
        assert!(
            cluster_acoustic_tracklets(&[invalid.clone()], &enrollment, None, 8, || false)
                .expect_err("non-finite channel variance")
                .to_string()
                .contains("finite non-negative statistics")
        );

        invalid.channel_variance[0] = 0.01;
        invalid.voice_variance[0] = -0.01;
        assert!(
            cluster_acoustic_tracklets(&[invalid.clone()], &enrollment, None, 8, || false)
                .expect_err("negative sample variance")
                .to_string()
                .contains("finite non-negative statistics")
        );

        invalid.voice_variance[0] = 0.01;
        invalid.voice_mean[0] = f32::MAX;
        assert!(
            cluster_acoustic_tracklets(&[invalid.clone()], &enrollment, None, 8, || false)
                .expect_err("finite tracklet means must not overflow distance arithmetic")
                .to_string()
                .contains("within acoustic-v1 bounds")
        );

        invalid.voice_mean[0] = 0.0;
        invalid.channel_variance[0] = f32::MAX;
        assert!(
            cluster_acoustic_tracklets(&[invalid.clone()], &enrollment, None, 8, || false)
                .expect_err("finite tracklet variances must not overflow distance arithmetic")
                .to_string()
                .contains("within acoustic-v1 bounds")
        );

        invalid.channel_variance[0] = 0.01;
        invalid.voiced_frame_count = invalid.frame_count + 1;
        assert!(
            cluster_acoustic_tracklets(&[invalid], &enrollment, None, 8, || false)
                .expect_err("voiced count exceeds total")
                .to_string()
                .contains("valid counts")
        );

        let mut huge = profile_tracklet(0, 0, 100, 0.0, 0.0, 0);
        huge.frame_count = usize::MAX;
        let following = profile_tracklet(1, 100, 200, 0.0, 0.0, 1);
        assert!(
            cluster_acoustic_tracklets(&[huge, following], &enrollment, None, 8, || false)
                .expect_err("aggregate frame counts must not overflow")
                .to_string()
                .contains("valid counts")
        );

        let nested = vec![
            profile_tracklet(0, 0, 100, 0.0, 0.0, 10),
            profile_tracklet(1, 80, 90, 0.0, 0.0, 1),
        ];
        assert!(
            cluster_acoustic_tracklets(&nested, &enrollment, None, 8, || false)
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
}
