//! Local-only, aggregate-only diarization evaluation for confidential inputs.
//!
//! The path-bearing manifest types in this module intentionally implement only
//! `Deserialize`: they must never be logged, returned, or serialized into a
//! repository artifact. Inputs are canonicalized beneath one explicit external
//! root, outputs are refused beneath the project tree, and public results contain
//! only aggregate measurements plus opaque content fingerprints.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::diarization::{
    AuthoritativeDiarizationScore, DIARIZATION_SCORER_VERSION, DiarizationScorerConfig,
    parse_diarization_hypothesis, parse_diarization_reference, score_diarization_documents,
};
use crate::error::{FwError, FwResult};

/// Schema for an external, path-bearing local evaluation manifest.
pub const CONFIDENTIAL_EVALUATION_MANIFEST_SCHEMA_VERSION: &str =
    "confidential-diarization-evaluation-manifest-v2";
/// Schema for the path-free aggregate emitted by this module.
pub const CONFIDENTIAL_EVALUATION_AGGREGATE_SCHEMA_VERSION: &str =
    "confidential-diarization-evaluation-aggregate-v2";

const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const MAX_DOCUMENT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_RECORDINGS: usize = 10_000;
const HASH_BUFFER_BYTES: usize = 64 * 1024;
const HASH_DOMAIN_SEPARATOR: &[u8] = b"franken-whisper-confidential-evaluation-v2\0";

/// Path-free micro/macro diarization aggregate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidentialDiarizationAggregate {
    pub reference_speaker_time_sec: f64,
    pub missed_speech_sec: f64,
    pub false_alarm_sec: f64,
    pub speaker_confusion_sec: f64,
    pub micro_der: Option<f64>,
    pub macro_der: Option<f64>,
    pub macro_jer: Option<f64>,
}

/// Path-free aggregate of change-boundary matching.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidentialChangeAggregate {
    pub reference_count: u64,
    pub hypothesis_count: u64,
    pub matched_count: u64,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub mean_absolute_error_sec: Option<f64>,
}

/// Aggregate speaker-count error without per-recording counts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidentialSpeakerCountAggregate {
    pub exact_recordings: u64,
    pub exact_rate: Option<f64>,
    pub mean_absolute_error: Option<f64>,
}

/// Aggregate proper scores and coverage for automatic speaker-count posteriors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidentialSpeakerCountPosteriorAggregate {
    pub observed_recordings: u64,
    pub unavailable_recordings: u64,
    pub unresolved_recordings: u64,
    pub zero_reference_probability_recordings: u64,
    pub finite_negative_log_likelihood_recordings: u64,
    pub mean_finite_negative_log_likelihood: Option<f64>,
    pub brier_observation_count: u64,
    pub mean_brier_score: Option<f64>,
    pub top_k_observation_count: u64,
    pub top_k_hit_count: u64,
    pub top_k_coverage: Option<f64>,
    pub credible_set_observation_count: u64,
    pub credible_set_hit_count: u64,
    pub credible_set_coverage: Option<f64>,
    pub entropy_observation_count: u64,
    pub mean_entropy_bits: Option<f64>,
}

/// Aggregate occupancy diagnostics without retaining speaker or recording IDs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidentialSpeakerOccupancyAggregate {
    pub dominant_collapse_recordings: u64,
    pub reference_collapse_recordings: u64,
    pub phantom_speaker_count: u64,
    pub collapsed_reference_speaker_count: u64,
    pub mean_effective_speaker_count: Option<f64>,
    pub dominant_share_observation_count: u64,
    pub mean_dominant_speaker_share: Option<f64>,
    pub maximum_dominant_speaker_share: Option<f64>,
    pub unknown_share_observation_count: u64,
    pub mean_unknown_speaker_share: Option<f64>,
    pub maximum_unknown_speaker_share: Option<f64>,
    pub minority_recall_observation_count: u64,
    pub mean_minority_reference_recall: Option<f64>,
}

/// Aggregate transcript-free aligned-word speaker attribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidentialWordAttributionAggregate {
    pub reference_word_count: u64,
    pub scored_word_count: u64,
    pub correct_word_count: u64,
    pub incorrect_word_count: u64,
    pub unknown_word_count: u64,
    pub excluded_word_count: u64,
    pub macro_observation_count: u64,
    pub micro_word_diarization_error_rate: Option<f64>,
    pub macro_word_diarization_error_rate: Option<f64>,
}

/// Duration-weighted overlap-detection aggregate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidentialOverlapAggregate {
    pub reference_overlap_sec: f64,
    pub hypothesis_overlap_sec: f64,
    pub true_positive_sec: f64,
    pub false_positive_sec: f64,
    pub false_negative_sec: f64,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
}

/// Aggregate confidence coverage and duration-weighted calibration means.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidentialCalibrationAggregate {
    pub observed_duration_sec: f64,
    pub opportunity_duration_sec: f64,
    pub coverage: Option<f64>,
    pub duration_weighted_mean_brier_score: Option<f64>,
    pub duration_weighted_mean_ece: Option<f64>,
}

/// Aggregate performance available only when every recording supplied it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidentialPerformanceAggregate {
    pub audio_duration_sec: f64,
    pub wall_time_sec: f64,
    pub real_time_factor: Option<f64>,
    pub peak_rss_bytes: u64,
}

/// Aggregate-only output safe to retain outside the project tree.
///
/// It contains no filenames, paths, transcript text, timestamps, speaker IDs,
/// recording IDs, per-recording metrics, or acoustic representations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidentialEvaluationAggregate {
    pub schema_version: String,
    pub scorer_version: String,
    pub dataset_fingerprint_sha256: String,
    pub config_sha256: String,
    pub recording_count: u64,
    pub total_annotation_duration_sec: f64,
    pub diarization: ConfidentialDiarizationAggregate,
    pub change_points: ConfidentialChangeAggregate,
    pub speaker_count: ConfidentialSpeakerCountAggregate,
    pub speaker_count_posterior: ConfidentialSpeakerCountPosteriorAggregate,
    pub speaker_occupancy: ConfidentialSpeakerOccupancyAggregate,
    pub word_attribution: ConfidentialWordAttributionAggregate,
    pub overlap: ConfidentialOverlapAggregate,
    pub calibration: ConfidentialCalibrationAggregate,
    pub performance_observed_recordings: u64,
    pub performance: Option<ConfidentialPerformanceAggregate>,
    /// Hash of this aggregate with this field set to the empty string.
    pub result_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfidentialEvaluationManifest {
    schema_version: String,
    dataset_id: String,
    config: DiarizationScorerConfig,
    recordings: Vec<ConfidentialEvaluationRecording>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfidentialEvaluationRecording {
    recording_id: String,
    audio_path: PathBuf,
    reference_path: PathBuf,
    hypothesis_path: PathBuf,
}

struct EvaluationBoundary {
    project_root: PathBuf,
    input_root: PathBuf,
    output_path: PathBuf,
}

#[derive(Default)]
struct AggregateAccumulator {
    recording_count: u64,
    annotation_duration_sec: f64,
    reference_speaker_time_sec: f64,
    missed_speech_sec: f64,
    false_alarm_sec: f64,
    speaker_confusion_sec: f64,
    macro_der_sum: f64,
    macro_der_count: u64,
    macro_jer_sum: f64,
    macro_jer_count: u64,
    change_reference_count: u64,
    change_hypothesis_count: u64,
    change_matched_count: u64,
    change_error_sum_sec: f64,
    exact_speaker_count_recordings: u64,
    absolute_speaker_count_error_sum: u64,
    count_posterior_recordings: u64,
    count_posterior_unavailable_recordings: u64,
    count_unresolved_recordings: u64,
    count_zero_reference_probability_recordings: u64,
    count_negative_log_likelihood_sum: f64,
    count_negative_log_likelihood_count: u64,
    count_brier_sum: f64,
    count_brier_count: u64,
    count_top_k_observation_count: u64,
    count_top_k_hit_count: u64,
    count_credible_set_observation_count: u64,
    count_credible_set_hit_count: u64,
    count_entropy_sum: f64,
    count_entropy_count: u64,
    dominant_collapse_recordings: u64,
    reference_collapse_recordings: u64,
    phantom_speaker_count: u64,
    collapsed_reference_speaker_count: u64,
    effective_speaker_count_sum: u64,
    dominant_share_sum: f64,
    dominant_share_count: u64,
    maximum_dominant_share: Option<f64>,
    unknown_share_sum: f64,
    unknown_share_count: u64,
    maximum_unknown_share: Option<f64>,
    minority_recall_sum: f64,
    minority_recall_count: u64,
    reference_word_count: u64,
    scored_word_count: u64,
    correct_word_count: u64,
    incorrect_word_count: u64,
    unknown_word_count: u64,
    excluded_word_count: u64,
    macro_word_error_sum: f64,
    macro_word_error_count: u64,
    reference_overlap_sec: f64,
    hypothesis_overlap_sec: f64,
    overlap_true_positive_sec: f64,
    overlap_false_positive_sec: f64,
    overlap_false_negative_sec: f64,
    calibration_observed_sec: f64,
    calibration_opportunity_sec: f64,
    brier_weighted_sum: f64,
    brier_weight_sec: f64,
    ece_weighted_sum: f64,
    ece_weight_sec: f64,
    performance_count: u64,
    audio_duration_sec: f64,
    wall_time_sec: f64,
    peak_rss_bytes: u64,
}

impl AggregateAccumulator {
    fn push(&mut self, score: &AuthoritativeDiarizationScore) -> FwResult<()> {
        self.recording_count = self.recording_count.checked_add(1).ok_or_else(|| {
            confidential_error(
                "aggregate_overflow",
                "recording count exceeds the supported range",
            )
        })?;
        self.annotation_duration_sec += score.scored_duration_sec + score.ignored_duration_sec;
        self.reference_speaker_time_sec += score.diarization.reference_speaker_time_sec;
        self.missed_speech_sec += score.diarization.missed_speech_sec;
        self.false_alarm_sec += score.diarization.false_alarm_sec;
        self.speaker_confusion_sec += score.diarization.speaker_confusion_sec;
        if let Some(value) = score.diarization.der {
            self.macro_der_sum += value;
            self.macro_der_count += 1;
        }
        if let Some(value) = score.diarization.jer {
            self.macro_jer_sum += value;
            self.macro_jer_count += 1;
        }

        let reference_count = u64::try_from(score.change_points.reference_count).map_err(|_| {
            confidential_error(
                "aggregate_overflow",
                "change reference count exceeds the supported range",
            )
        })?;
        let hypothesis_count =
            u64::try_from(score.change_points.hypothesis_count).map_err(|_| {
                confidential_error(
                    "aggregate_overflow",
                    "change hypothesis count exceeds the supported range",
                )
            })?;
        let matched_count = u64::try_from(score.change_points.matched_count).map_err(|_| {
            confidential_error(
                "aggregate_overflow",
                "change match count exceeds the supported range",
            )
        })?;
        self.change_reference_count = self
            .change_reference_count
            .checked_add(reference_count)
            .ok_or_else(|| {
                confidential_error(
                    "aggregate_overflow",
                    "change reference count exceeds the supported range",
                )
            })?;
        self.change_hypothesis_count = self
            .change_hypothesis_count
            .checked_add(hypothesis_count)
            .ok_or_else(|| {
            confidential_error(
                "aggregate_overflow",
                "change hypothesis count exceeds the supported range",
            )
        })?;
        self.change_matched_count = self
            .change_matched_count
            .checked_add(matched_count)
            .ok_or_else(|| {
                confidential_error(
                    "aggregate_overflow",
                    "change match count exceeds the supported range",
                )
            })?;
        if let Some(error) = score.change_points.mean_absolute_error_sec {
            self.change_error_sum_sec += error * score.change_points.matched_count as f64;
        }

        if score.speaker_count.absolute_error == 0 {
            self.exact_speaker_count_recordings += 1;
        }
        self.absolute_speaker_count_error_sum = self
            .absolute_speaker_count_error_sum
            .checked_add(score.speaker_count.absolute_error)
            .ok_or_else(|| {
                confidential_error(
                    "aggregate_overflow",
                    "speaker-count error exceeds the supported range",
                )
            })?;

        if score.speaker_count_posterior.posterior_available {
            self.count_posterior_recordings = checked_aggregate_add(
                self.count_posterior_recordings,
                1,
                "count posterior observation",
            )?;
        } else {
            self.count_posterior_unavailable_recordings = checked_aggregate_add(
                self.count_posterior_unavailable_recordings,
                1,
                "count posterior unavailability",
            )?;
        }
        if score.speaker_count_posterior.unresolved {
            self.count_unresolved_recordings = checked_aggregate_add(
                self.count_unresolved_recordings,
                1,
                "unresolved count posterior",
            )?;
        }
        if score
            .speaker_count_posterior
            .infinite_negative_log_likelihood
        {
            self.count_zero_reference_probability_recordings = checked_aggregate_add(
                self.count_zero_reference_probability_recordings,
                1,
                "zero-probability count posterior",
            )?;
        }
        if let Some(value) = score.speaker_count_posterior.negative_log_likelihood {
            self.count_negative_log_likelihood_sum += value;
            self.count_negative_log_likelihood_count = checked_aggregate_add(
                self.count_negative_log_likelihood_count,
                1,
                "count negative-log-likelihood observation",
            )?;
        }
        if let Some(value) = score.speaker_count_posterior.brier_score {
            self.count_brier_sum += value;
            self.count_brier_count =
                checked_aggregate_add(self.count_brier_count, 1, "count Brier observation")?;
        }
        if let Some(hit) = score.speaker_count_posterior.top_k_hit {
            self.count_top_k_observation_count = checked_aggregate_add(
                self.count_top_k_observation_count,
                1,
                "count top-k observation",
            )?;
            self.count_top_k_hit_count = checked_aggregate_add(
                self.count_top_k_hit_count,
                u64::from(hit),
                "count top-k hit",
            )?;
        }
        if let Some(hit) = score.speaker_count_posterior.credible_set_hit {
            self.count_credible_set_observation_count = checked_aggregate_add(
                self.count_credible_set_observation_count,
                1,
                "count credible-set observation",
            )?;
            self.count_credible_set_hit_count = checked_aggregate_add(
                self.count_credible_set_hit_count,
                u64::from(hit),
                "count credible-set hit",
            )?;
        }
        if score.speaker_count_posterior.posterior_available
            && let Some(value) = score.speaker_count_posterior.entropy_bits
        {
            self.count_entropy_sum += value;
            self.count_entropy_count =
                checked_aggregate_add(self.count_entropy_count, 1, "count entropy observation")?;
        }

        self.dominant_collapse_recordings = checked_aggregate_add(
            self.dominant_collapse_recordings,
            u64::from(score.speaker_occupancy.dominant_collapse_detected),
            "dominant-collapse recording",
        )?;
        self.reference_collapse_recordings = checked_aggregate_add(
            self.reference_collapse_recordings,
            u64::from(score.speaker_occupancy.any_reference_collapse_detected),
            "reference-collapse recording",
        )?;
        self.phantom_speaker_count = checked_aggregate_add(
            self.phantom_speaker_count,
            u64::try_from(score.speaker_occupancy.phantom_speaker_count).map_err(|_| {
                confidential_error(
                    "aggregate_overflow",
                    "phantom-speaker count exceeds the supported range",
                )
            })?,
            "phantom-speaker count",
        )?;
        self.collapsed_reference_speaker_count = checked_aggregate_add(
            self.collapsed_reference_speaker_count,
            u64::try_from(score.speaker_occupancy.collapsed_reference_speaker_count).map_err(
                |_| {
                    confidential_error(
                        "aggregate_overflow",
                        "collapsed reference-speaker count exceeds the supported range",
                    )
                },
            )?,
            "collapsed reference-speaker count",
        )?;
        self.effective_speaker_count_sum = checked_aggregate_add(
            self.effective_speaker_count_sum,
            u64::try_from(score.speaker_occupancy.effective_speaker_count).map_err(|_| {
                confidential_error(
                    "aggregate_overflow",
                    "effective-speaker count exceeds the supported range",
                )
            })?,
            "effective-speaker count",
        )?;
        if let Some(value) = score.speaker_occupancy.dominant_speaker_share {
            self.dominant_share_sum += value;
            self.dominant_share_count =
                checked_aggregate_add(self.dominant_share_count, 1, "dominant-share observation")?;
            self.maximum_dominant_share = Some(
                self.maximum_dominant_share
                    .map_or(value, |maximum| maximum.max(value)),
            );
        }
        if let Some(value) = score.speaker_occupancy.unknown_speaker_share {
            self.unknown_share_sum += value;
            self.unknown_share_count =
                checked_aggregate_add(self.unknown_share_count, 1, "unknown-share observation")?;
            self.maximum_unknown_share = Some(
                self.maximum_unknown_share
                    .map_or(value, |maximum| maximum.max(value)),
            );
        }
        if let Some(value) = score.speaker_occupancy.minority_reference_recall {
            self.minority_recall_sum += value;
            self.minority_recall_count = checked_aggregate_add(
                self.minority_recall_count,
                1,
                "minority-recall observation",
            )?;
        }

        self.reference_word_count = checked_aggregate_add(
            self.reference_word_count,
            score.word_attribution.reference_word_count,
            "reference-word count",
        )?;
        self.scored_word_count = checked_aggregate_add(
            self.scored_word_count,
            score.word_attribution.scored_word_count,
            "scored-word count",
        )?;
        self.correct_word_count = checked_aggregate_add(
            self.correct_word_count,
            score.word_attribution.correct_word_count,
            "correct-word count",
        )?;
        self.incorrect_word_count = checked_aggregate_add(
            self.incorrect_word_count,
            score.word_attribution.incorrect_word_count,
            "incorrect-word count",
        )?;
        self.unknown_word_count = checked_aggregate_add(
            self.unknown_word_count,
            score.word_attribution.unknown_word_count,
            "unknown-word count",
        )?;
        self.excluded_word_count = checked_aggregate_add(
            self.excluded_word_count,
            score.word_attribution.excluded_word_count,
            "excluded-word count",
        )?;
        if let Some(value) = score.word_attribution.word_diarization_error_rate {
            self.macro_word_error_sum += value;
            self.macro_word_error_count = checked_aggregate_add(
                self.macro_word_error_count,
                1,
                "word-attribution recording",
            )?;
        }

        self.reference_overlap_sec += score.overlap.reference_overlap_sec;
        self.hypothesis_overlap_sec += score.overlap.hypothesis_overlap_sec;
        self.overlap_true_positive_sec += score.overlap.true_positive_sec;
        self.overlap_false_positive_sec += score.overlap.false_positive_sec;
        self.overlap_false_negative_sec += score.overlap.false_negative_sec;

        let calibration_weight = score.calibration.observed_duration_sec;
        self.calibration_observed_sec += calibration_weight;
        self.calibration_opportunity_sec += score.calibration.opportunity_duration_sec;
        if let Some(value) = score.calibration.brier_score {
            self.brier_weighted_sum += value * calibration_weight;
            self.brier_weight_sec += calibration_weight;
        }
        if let Some(value) = score.calibration.expected_calibration_error {
            self.ece_weighted_sum += value * calibration_weight;
            self.ece_weight_sec += calibration_weight;
        }

        if let Some(performance) = &score.performance {
            self.performance_count += 1;
            self.audio_duration_sec += performance.audio_duration_sec;
            self.wall_time_sec += performance.wall_time_sec;
            self.peak_rss_bytes = self.peak_rss_bytes.max(performance.peak_rss_bytes);
        }
        Ok(())
    }

    fn finish(
        self,
        dataset_fingerprint_sha256: String,
        config_sha256: String,
    ) -> FwResult<ConfidentialEvaluationAggregate> {
        let diarization_error_sec =
            self.missed_speech_sec + self.false_alarm_sec + self.speaker_confusion_sec;
        let micro_der = ratio(diarization_error_sec, self.reference_speaker_time_sec);
        let change_precision = ratio(
            self.change_matched_count as f64,
            self.change_hypothesis_count as f64,
        );
        let change_recall = ratio(
            self.change_matched_count as f64,
            self.change_reference_count as f64,
        );
        let overlap_precision = ratio(
            self.overlap_true_positive_sec,
            self.overlap_true_positive_sec + self.overlap_false_positive_sec,
        );
        let overlap_recall = ratio(
            self.overlap_true_positive_sec,
            self.overlap_true_positive_sec + self.overlap_false_negative_sec,
        );
        let performance =
            if self.performance_count == self.recording_count && self.recording_count > 0 {
                Some(ConfidentialPerformanceAggregate {
                    audio_duration_sec: self.audio_duration_sec,
                    wall_time_sec: self.wall_time_sec,
                    real_time_factor: ratio(self.wall_time_sec, self.audio_duration_sec),
                    peak_rss_bytes: self.peak_rss_bytes,
                })
            } else {
                None
            };

        let mut aggregate = ConfidentialEvaluationAggregate {
            schema_version: CONFIDENTIAL_EVALUATION_AGGREGATE_SCHEMA_VERSION.to_owned(),
            scorer_version: DIARIZATION_SCORER_VERSION.to_owned(),
            dataset_fingerprint_sha256,
            config_sha256,
            recording_count: self.recording_count,
            total_annotation_duration_sec: self.annotation_duration_sec,
            diarization: ConfidentialDiarizationAggregate {
                reference_speaker_time_sec: self.reference_speaker_time_sec,
                missed_speech_sec: self.missed_speech_sec,
                false_alarm_sec: self.false_alarm_sec,
                speaker_confusion_sec: self.speaker_confusion_sec,
                micro_der,
                macro_der: ratio(self.macro_der_sum, self.macro_der_count as f64),
                macro_jer: ratio(self.macro_jer_sum, self.macro_jer_count as f64),
            },
            change_points: ConfidentialChangeAggregate {
                reference_count: self.change_reference_count,
                hypothesis_count: self.change_hypothesis_count,
                matched_count: self.change_matched_count,
                precision: change_precision,
                recall: change_recall,
                f1: f1(change_precision, change_recall),
                mean_absolute_error_sec: ratio(
                    self.change_error_sum_sec,
                    self.change_matched_count as f64,
                ),
            },
            speaker_count: ConfidentialSpeakerCountAggregate {
                exact_recordings: self.exact_speaker_count_recordings,
                exact_rate: ratio(
                    self.exact_speaker_count_recordings as f64,
                    self.recording_count as f64,
                ),
                mean_absolute_error: ratio(
                    self.absolute_speaker_count_error_sum as f64,
                    self.recording_count as f64,
                ),
            },
            speaker_count_posterior: ConfidentialSpeakerCountPosteriorAggregate {
                observed_recordings: self.count_posterior_recordings,
                unavailable_recordings: self.count_posterior_unavailable_recordings,
                unresolved_recordings: self.count_unresolved_recordings,
                zero_reference_probability_recordings: self
                    .count_zero_reference_probability_recordings,
                finite_negative_log_likelihood_recordings: self.count_negative_log_likelihood_count,
                mean_finite_negative_log_likelihood: ratio(
                    self.count_negative_log_likelihood_sum,
                    self.count_negative_log_likelihood_count as f64,
                ),
                brier_observation_count: self.count_brier_count,
                mean_brier_score: ratio(self.count_brier_sum, self.count_brier_count as f64),
                top_k_observation_count: self.count_top_k_observation_count,
                top_k_hit_count: self.count_top_k_hit_count,
                top_k_coverage: ratio(
                    self.count_top_k_hit_count as f64,
                    self.count_top_k_observation_count as f64,
                ),
                credible_set_observation_count: self.count_credible_set_observation_count,
                credible_set_hit_count: self.count_credible_set_hit_count,
                credible_set_coverage: ratio(
                    self.count_credible_set_hit_count as f64,
                    self.count_credible_set_observation_count as f64,
                ),
                entropy_observation_count: self.count_entropy_count,
                mean_entropy_bits: ratio(self.count_entropy_sum, self.count_entropy_count as f64),
            },
            speaker_occupancy: ConfidentialSpeakerOccupancyAggregate {
                dominant_collapse_recordings: self.dominant_collapse_recordings,
                reference_collapse_recordings: self.reference_collapse_recordings,
                phantom_speaker_count: self.phantom_speaker_count,
                collapsed_reference_speaker_count: self.collapsed_reference_speaker_count,
                mean_effective_speaker_count: ratio(
                    self.effective_speaker_count_sum as f64,
                    self.recording_count as f64,
                ),
                dominant_share_observation_count: self.dominant_share_count,
                mean_dominant_speaker_share: ratio(
                    self.dominant_share_sum,
                    self.dominant_share_count as f64,
                ),
                maximum_dominant_speaker_share: self.maximum_dominant_share,
                unknown_share_observation_count: self.unknown_share_count,
                mean_unknown_speaker_share: ratio(
                    self.unknown_share_sum,
                    self.unknown_share_count as f64,
                ),
                maximum_unknown_speaker_share: self.maximum_unknown_share,
                minority_recall_observation_count: self.minority_recall_count,
                mean_minority_reference_recall: ratio(
                    self.minority_recall_sum,
                    self.minority_recall_count as f64,
                ),
            },
            word_attribution: ConfidentialWordAttributionAggregate {
                reference_word_count: self.reference_word_count,
                scored_word_count: self.scored_word_count,
                correct_word_count: self.correct_word_count,
                incorrect_word_count: self.incorrect_word_count,
                unknown_word_count: self.unknown_word_count,
                excluded_word_count: self.excluded_word_count,
                macro_observation_count: self.macro_word_error_count,
                micro_word_diarization_error_rate: ratio(
                    self.incorrect_word_count
                        .checked_add(self.unknown_word_count)
                        .ok_or_else(|| {
                            confidential_error(
                                "aggregate_overflow",
                                "word-attribution error count exceeds the supported range",
                            )
                        })? as f64,
                    self.scored_word_count as f64,
                ),
                macro_word_diarization_error_rate: ratio(
                    self.macro_word_error_sum,
                    self.macro_word_error_count as f64,
                ),
            },
            overlap: ConfidentialOverlapAggregate {
                reference_overlap_sec: self.reference_overlap_sec,
                hypothesis_overlap_sec: self.hypothesis_overlap_sec,
                true_positive_sec: self.overlap_true_positive_sec,
                false_positive_sec: self.overlap_false_positive_sec,
                false_negative_sec: self.overlap_false_negative_sec,
                precision: overlap_precision,
                recall: overlap_recall,
                f1: f1(overlap_precision, overlap_recall),
            },
            calibration: ConfidentialCalibrationAggregate {
                observed_duration_sec: self.calibration_observed_sec,
                opportunity_duration_sec: self.calibration_opportunity_sec,
                coverage: ratio(
                    self.calibration_observed_sec,
                    self.calibration_opportunity_sec,
                ),
                duration_weighted_mean_brier_score: ratio(
                    self.brier_weighted_sum,
                    self.brier_weight_sec,
                ),
                duration_weighted_mean_ece: ratio(self.ece_weighted_sum, self.ece_weight_sec),
            },
            performance_observed_recordings: self.performance_count,
            performance,
            result_sha256: String::new(),
        };
        validate_finite_aggregate(&aggregate)?;
        aggregate.result_sha256 = canonical_sha256(&aggregate)?;
        Ok(aggregate)
    }
}

/// Locate the nearest repository root without exposing its path in errors.
///
/// Confidential evaluation is intentionally a developer-only workflow and
/// refuses to run when invoked outside a checkout.
pub fn discover_project_root(start: &Path) -> FwResult<PathBuf> {
    let mut current = start
        .canonicalize()
        .map_err(|_| confidential_error("project_root", "project root could not be resolved"))?;
    if current.is_file() {
        current = current
            .parent()
            .ok_or_else(|| {
                confidential_error("project_root", "project root could not be resolved")
            })?
            .to_owned();
    }
    loop {
        if current.join(".git").exists() && current.join("Cargo.toml").is_file() {
            return Ok(current);
        }
        if !current.pop() {
            return Err(confidential_error(
                "project_root",
                "confidential evaluation must run from a project checkout",
            ));
        }
    }
}

/// Score external reference/hypothesis pairs and write one aggregate-only JSON.
///
/// The manifest, audio, reference, and hypothesis files are read in place
/// beneath `input_root`. `output_path` must be a new absolute JSON path whose
/// canonical parent is outside `project_root`.
pub fn run_confidential_evaluation(
    project_root: &Path,
    input_root: &Path,
    manifest_path: &Path,
    output_path: &Path,
) -> FwResult<ConfidentialEvaluationAggregate> {
    run_confidential_evaluation_with_cancel(
        project_root,
        input_root,
        manifest_path,
        output_path,
        || false,
    )
}

/// Cancellable form of [`run_confidential_evaluation`].
pub fn run_confidential_evaluation_with_cancel<C>(
    project_root: &Path,
    input_root: &Path,
    manifest_path: &Path,
    output_path: &Path,
    mut is_cancelled: C,
) -> FwResult<ConfidentialEvaluationAggregate>
where
    C: FnMut() -> bool,
{
    let boundary = EvaluationBoundary::new(project_root, input_root, manifest_path, output_path)?;
    check_cancelled(&mut is_cancelled)?;
    let manifest_bytes = read_capped(
        &boundary.canonical_input_file(manifest_path, InputKind::Manifest)?,
        MAX_MANIFEST_BYTES,
        "manifest_read",
    )?;
    let manifest: ConfidentialEvaluationManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|_| {
            confidential_error(
                "manifest_schema",
                "confidential evaluation manifest is invalid",
            )
        })?;
    validate_manifest_shape(&manifest)?;

    let config_sha256 = canonical_sha256(&manifest.config)?;
    let mut dataset_hasher = Sha256::new();
    dataset_hasher.update(HASH_DOMAIN_SEPARATOR);
    hash_fingerprint_field(
        &mut dataset_hasher,
        b"dataset_id",
        manifest.dataset_id.as_bytes(),
    )?;
    hash_fingerprint_field(
        &mut dataset_hasher,
        b"config_sha256",
        config_sha256.as_bytes(),
    )?;
    hash_fingerprint_field(
        &mut dataset_hasher,
        b"recording_count",
        &u64::try_from(manifest.recordings.len())
            .map_err(|_| {
                confidential_error(
                    "aggregate_overflow",
                    "recording count exceeds the supported range",
                )
            })?
            .to_be_bytes(),
    )?;
    let mut aggregate = AggregateAccumulator::default();
    let mut seen_audio = BTreeSet::new();
    let mut seen_references = BTreeSet::new();
    let mut seen_hypotheses = BTreeSet::new();

    for recording in &manifest.recordings {
        check_cancelled(&mut is_cancelled)?;
        let audio_path = boundary.canonical_input_file(&recording.audio_path, InputKind::Audio)?;
        let reference_path =
            boundary.canonical_input_file(&recording.reference_path, InputKind::Reference)?;
        let hypothesis_path =
            boundary.canonical_input_file(&recording.hypothesis_path, InputKind::Hypothesis)?;
        if !seen_audio.insert(audio_path.clone())
            || !seen_references.insert(reference_path.clone())
            || !seen_hypotheses.insert(hypothesis_path.clone())
        {
            return Err(confidential_error(
                "duplicate_source",
                "confidential recordings must not reuse source files",
            ));
        }

        let audio_sha256 = hash_file(&audio_path, &mut is_cancelled)?;
        let reference_bytes = read_capped(&reference_path, MAX_DOCUMENT_BYTES, "reference_read")?;
        let hypothesis_bytes =
            read_capped(&hypothesis_path, MAX_DOCUMENT_BYTES, "hypothesis_read")?;
        let reference = parse_diarization_reference(&reference_bytes).map_err(|_| {
            confidential_error(
                "reference_schema",
                "one confidential reference document is invalid",
            )
        })?;
        let hypothesis = parse_diarization_hypothesis(&hypothesis_bytes).map_err(|_| {
            confidential_error(
                "hypothesis_schema",
                "one confidential hypothesis document is invalid",
            )
        })?;
        if reference.recording_id != recording.recording_id
            || hypothesis.recording_id != recording.recording_id
        {
            return Err(confidential_error(
                "recording_identity",
                "one confidential recording identity is inconsistent",
            ));
        }
        let score = score_diarization_documents(&reference, &hypothesis, &manifest.config)
            .map_err(|_| {
                confidential_error("scoring", "one confidential recording could not be scored")
            })?;
        hash_fingerprint_field(
            &mut dataset_hasher,
            b"recording_id",
            recording.recording_id.as_bytes(),
        )?;
        hash_fingerprint_field(
            &mut dataset_hasher,
            b"audio_sha256",
            audio_sha256.as_bytes(),
        )?;
        hash_fingerprint_field(
            &mut dataset_hasher,
            b"reference_sha256",
            score.reference_sha256.as_bytes(),
        )?;
        hash_fingerprint_field(
            &mut dataset_hasher,
            b"hypothesis_sha256",
            score.hypothesis_sha256.as_bytes(),
        )?;
        aggregate.push(&score)?;
    }

    let result = aggregate.finish(format!("{:x}", dataset_hasher.finalize()), config_sha256)?;
    check_cancelled(&mut is_cancelled)?;
    write_aggregate(&boundary.output_path, &result)?;
    Ok(result)
}

/// Verify schema identity, finite aggregate values, and the self-hash.
pub fn verify_confidential_evaluation_aggregate(
    aggregate: &ConfidentialEvaluationAggregate,
) -> FwResult<()> {
    if aggregate.schema_version != CONFIDENTIAL_EVALUATION_AGGREGATE_SCHEMA_VERSION
        || aggregate.scorer_version != DIARIZATION_SCORER_VERSION
    {
        return Err(confidential_error(
            "aggregate_version",
            "unsupported confidential aggregate version",
        ));
    }
    if !is_sha256_hex(&aggregate.dataset_fingerprint_sha256)
        || !is_sha256_hex(&aggregate.config_sha256)
        || !is_sha256_hex(&aggregate.result_sha256)
    {
        return Err(confidential_error(
            "aggregate_hash",
            "confidential aggregate contains an invalid hash",
        ));
    }
    validate_finite_aggregate(aggregate)?;
    let mut unhashed = aggregate.clone();
    let expected = unhashed.result_sha256.clone();
    unhashed.result_sha256.clear();
    if canonical_sha256(&unhashed)? == expected {
        Ok(())
    } else {
        Err(confidential_error(
            "aggregate_hash",
            "confidential aggregate self-hash does not match",
        ))
    }
}

#[derive(Clone, Copy)]
enum InputKind {
    Manifest,
    Audio,
    Reference,
    Hypothesis,
}

impl EvaluationBoundary {
    fn new(
        project_root: &Path,
        input_root: &Path,
        manifest_path: &Path,
        output_path: &Path,
    ) -> FwResult<Self> {
        let project_root = canonical_absolute_directory(project_root, "project_root")?;
        let input_root = canonical_absolute_directory(input_root, "input_root")?;
        if input_root.starts_with(&project_root) || project_root.starts_with(&input_root) {
            return Err(confidential_error(
                "input_root",
                "confidential input and project roots must be disjoint",
            ));
        }
        if !manifest_path.is_absolute() {
            return Err(confidential_error(
                "manifest_path",
                "confidential manifest path must be absolute",
            ));
        }
        if !output_path.is_absolute() {
            return Err(confidential_error(
                "output_path",
                "confidential output path must be absolute",
            ));
        }
        if output_path.exists() {
            return Err(confidential_error(
                "output_exists",
                "confidential aggregate output must not already exist",
            ));
        }
        if output_path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_none_or(|extension| !extension.eq_ignore_ascii_case("json"))
        {
            return Err(confidential_error(
                "output_extension",
                "confidential aggregate output must use a JSON extension",
            ));
        }
        let output_parent = output_path.parent().ok_or_else(|| {
            confidential_error(
                "output_path",
                "confidential aggregate output requires a parent directory",
            )
        })?;
        let canonical_output_parent = canonical_absolute_directory(output_parent, "output_path")?;
        if canonical_output_parent.starts_with(&project_root) {
            return Err(confidential_error(
                "output_inside_project",
                "confidential evaluation refuses output beneath the project tree",
            ));
        }
        let file_name = output_path.file_name().ok_or_else(|| {
            confidential_error(
                "output_path",
                "confidential aggregate output requires a filename",
            )
        })?;
        let resolved_output = canonical_output_parent.join(file_name);
        let boundary = Self {
            project_root,
            input_root,
            output_path: resolved_output,
        };
        boundary.canonical_input_file(manifest_path, InputKind::Manifest)?;
        Ok(boundary)
    }

    fn canonical_input_file(&self, path: &Path, kind: InputKind) -> FwResult<PathBuf> {
        if !path.is_absolute() {
            return Err(confidential_error(
                "source_path",
                "every confidential source path must be absolute",
            ));
        }
        let canonical = path.canonicalize().map_err(|_| {
            confidential_error(
                "source_path",
                "one confidential source file could not be resolved",
            )
        })?;
        if !canonical.is_file()
            || !canonical.starts_with(&self.input_root)
            || canonical.starts_with(&self.project_root)
        {
            return Err(confidential_error(
                "source_boundary",
                "one confidential source file is outside the allowed external root",
            ));
        }
        let extension = canonical
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase);
        let valid_extension = match kind {
            InputKind::Manifest | InputKind::Reference | InputKind::Hypothesis => {
                extension.as_deref() == Some("json")
            }
            InputKind::Audio => extension.as_deref().is_some_and(|extension| {
                matches!(
                    extension,
                    "wav"
                        | "ulaw"
                        | "mp3"
                        | "flac"
                        | "ogg"
                        | "m4a"
                        | "aac"
                        | "aif"
                        | "aiff"
                        | "amr"
                        | "caf"
                        | "opus"
                        | "wma"
                        | "3gp"
                        | "mp4"
                        | "mov"
                        | "webm"
                )
            }),
        };
        if !valid_extension {
            return Err(confidential_error(
                "source_extension",
                "one confidential source file has an unsupported extension",
            ));
        }
        Ok(canonical)
    }
}

fn canonical_absolute_directory(path: &Path, code: &str) -> FwResult<PathBuf> {
    if !path.is_absolute() {
        return Err(confidential_error(
            code,
            "confidential evaluation roots must be absolute directories",
        ));
    }
    let canonical = path.canonicalize().map_err(|_| {
        confidential_error(
            code,
            "one confidential evaluation root could not be resolved",
        )
    })?;
    if canonical.is_dir() {
        Ok(canonical)
    } else {
        Err(confidential_error(
            code,
            "one confidential evaluation root is not a directory",
        ))
    }
}

fn validate_manifest_shape(manifest: &ConfidentialEvaluationManifest) -> FwResult<()> {
    if manifest.schema_version != CONFIDENTIAL_EVALUATION_MANIFEST_SCHEMA_VERSION {
        return Err(confidential_error(
            "manifest_version",
            "unsupported confidential evaluation manifest version",
        ));
    }
    validate_local_opaque_id(&manifest.dataset_id)?;
    if manifest.recordings.is_empty() || manifest.recordings.len() > MAX_RECORDINGS {
        return Err(confidential_error(
            "manifest_recording_count",
            "confidential evaluation manifest recording count is outside the supported range",
        ));
    }
    for recording in &manifest.recordings {
        validate_local_opaque_id(&recording.recording_id)?;
    }
    if !manifest
        .recordings
        .windows(2)
        .all(|window| window[0].recording_id < window[1].recording_id)
    {
        return Err(confidential_error(
            "manifest_order",
            "confidential recordings must be strictly sorted and unique",
        ));
    }
    Ok(())
}

fn validate_local_opaque_id(value: &str) -> FwResult<()> {
    let valid = !value.is_empty()
        && value.len() <= 160
        && value.trim() == value
        && !value.chars().any(char::is_control)
        && !value.contains('/')
        && !value.contains('\\')
        && !value.contains("..");
    if valid {
        Ok(())
    } else {
        Err(confidential_error(
            "opaque_id",
            "confidential manifest identifiers must be opaque and path-free",
        ))
    }
}

fn read_capped(path: &Path, max_bytes: u64, code: &str) -> FwResult<Vec<u8>> {
    let file = File::open(path)
        .map_err(|_| confidential_error(code, "one confidential source could not be read"))?;
    if file
        .metadata()
        .map_err(|_| confidential_error(code, "one confidential source could not be read"))?
        .len()
        > max_bytes
    {
        return Err(confidential_error(
            code,
            "one confidential source exceeds its bounded size limit",
        ));
    }
    let mut bytes = Vec::new();
    file.take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| confidential_error(code, "one confidential source could not be read"))?;
    let byte_len = u64::try_from(bytes.len()).map_err(|_| {
        confidential_error(
            code,
            "one confidential source exceeds its bounded size limit",
        )
    })?;
    if byte_len > max_bytes {
        return Err(confidential_error(
            code,
            "one confidential source exceeds its bounded size limit",
        ));
    }
    Ok(bytes)
}

fn hash_file<C>(path: &Path, is_cancelled: &mut C) -> FwResult<String>
where
    C: FnMut() -> bool,
{
    let mut file = File::open(path).map_err(|_| {
        confidential_error(
            "audio_read",
            "one confidential audio source could not be read",
        )
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; HASH_BUFFER_BYTES];
    let mut total_bytes = 0_u64;
    loop {
        check_cancelled(is_cancelled)?;
        let read = file.read(&mut buffer).map_err(|_| {
            confidential_error(
                "audio_read",
                "one confidential audio source could not be read",
            )
        })?;
        if read == 0 {
            break;
        }
        total_bytes = total_bytes
            .checked_add(u64::try_from(read).map_err(|_| {
                confidential_error(
                    "audio_read",
                    "one confidential audio source exceeds the supported range",
                )
            })?)
            .ok_or_else(|| {
                confidential_error(
                    "audio_read",
                    "one confidential audio source exceeds the supported range",
                )
            })?;
        hasher.update(&buffer[..read]);
    }
    if total_bytes == 0 {
        Err(confidential_error(
            "audio_empty",
            "one confidential audio source is empty",
        ))
    } else {
        Ok(format!("{:x}", hasher.finalize()))
    }
}

fn check_cancelled<C>(is_cancelled: &mut C) -> FwResult<()>
where
    C: FnMut() -> bool,
{
    if is_cancelled() {
        Err(FwError::Cancelled(
            "confidential evaluation cancelled without writing output".to_owned(),
        ))
    } else {
        Ok(())
    }
}

fn write_aggregate(path: &Path, aggregate: &ConfidentialEvaluationAggregate) -> FwResult<()> {
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|_| {
            confidential_error(
                "output_write",
                "confidential aggregate output could not be created",
            )
        })?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, aggregate).map_err(|_| {
        confidential_error(
            "output_write",
            "confidential aggregate output could not be serialized",
        )
    })?;
    writer.write_all(b"\n").map_err(|_| {
        confidential_error(
            "output_write",
            "confidential aggregate output could not be written",
        )
    })?;
    writer.flush().map_err(|_| {
        confidential_error(
            "output_write",
            "confidential aggregate output could not be written",
        )
    })?;
    writer.get_ref().sync_all().map_err(|_| {
        confidential_error(
            "output_write",
            "confidential aggregate output could not be synchronized",
        )
    })
}

fn canonical_sha256<T: Serialize>(value: &T) -> FwResult<String> {
    let bytes = serde_json::to_vec(value).map_err(|_| {
        confidential_error(
            "aggregate_serialization",
            "confidential aggregate could not be serialized",
        )
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn hash_fingerprint_field(hasher: &mut Sha256, tag: &[u8], value: &[u8]) -> FwResult<()> {
    let tag_len = u64::try_from(tag.len()).map_err(|_| {
        confidential_error(
            "aggregate_overflow",
            "fingerprint field tag exceeds the supported range",
        )
    })?;
    let value_len = u64::try_from(value.len()).map_err(|_| {
        confidential_error(
            "aggregate_overflow",
            "fingerprint field value exceeds the supported range",
        )
    })?;
    hasher.update(tag_len.to_be_bytes());
    hasher.update(tag);
    hasher.update(value_len.to_be_bytes());
    hasher.update(value);
    Ok(())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn checked_aggregate_add(current: u64, increment: u64, name: &str) -> FwResult<u64> {
    current.checked_add(increment).ok_or_else(|| {
        confidential_error(
            "aggregate_overflow",
            &format!("{name} exceeds the supported range"),
        )
    })
}

fn validate_finite_aggregate(aggregate: &ConfidentialEvaluationAggregate) -> FwResult<()> {
    let required = [
        aggregate.total_annotation_duration_sec,
        aggregate.diarization.reference_speaker_time_sec,
        aggregate.diarization.missed_speech_sec,
        aggregate.diarization.false_alarm_sec,
        aggregate.diarization.speaker_confusion_sec,
        aggregate.overlap.reference_overlap_sec,
        aggregate.overlap.hypothesis_overlap_sec,
        aggregate.overlap.true_positive_sec,
        aggregate.overlap.false_positive_sec,
        aggregate.overlap.false_negative_sec,
        aggregate.calibration.observed_duration_sec,
        aggregate.calibration.opportunity_duration_sec,
    ];
    let optional = [
        aggregate.diarization.micro_der,
        aggregate.diarization.macro_der,
        aggregate.diarization.macro_jer,
        aggregate.change_points.precision,
        aggregate.change_points.recall,
        aggregate.change_points.f1,
        aggregate.change_points.mean_absolute_error_sec,
        aggregate.speaker_count.exact_rate,
        aggregate.speaker_count.mean_absolute_error,
        aggregate
            .speaker_count_posterior
            .mean_finite_negative_log_likelihood,
        aggregate.speaker_count_posterior.mean_brier_score,
        aggregate.speaker_count_posterior.top_k_coverage,
        aggregate.speaker_count_posterior.credible_set_coverage,
        aggregate.speaker_count_posterior.mean_entropy_bits,
        aggregate.speaker_occupancy.mean_effective_speaker_count,
        aggregate.speaker_occupancy.mean_dominant_speaker_share,
        aggregate.speaker_occupancy.maximum_dominant_speaker_share,
        aggregate.speaker_occupancy.mean_unknown_speaker_share,
        aggregate.speaker_occupancy.maximum_unknown_speaker_share,
        aggregate.speaker_occupancy.mean_minority_reference_recall,
        aggregate.word_attribution.micro_word_diarization_error_rate,
        aggregate.word_attribution.macro_word_diarization_error_rate,
        aggregate.overlap.precision,
        aggregate.overlap.recall,
        aggregate.overlap.f1,
        aggregate.calibration.coverage,
        aggregate.calibration.duration_weighted_mean_brier_score,
        aggregate.calibration.duration_weighted_mean_ece,
    ];
    let performance_finite = aggregate.performance.as_ref().is_none_or(|performance| {
        performance.audio_duration_sec.is_finite()
            && performance.wall_time_sec.is_finite()
            && performance.real_time_factor.is_none_or(f64::is_finite)
    });
    let bounded_rates = [
        aggregate.change_points.precision,
        aggregate.change_points.recall,
        aggregate.change_points.f1,
        aggregate.speaker_count.exact_rate,
        aggregate.speaker_count_posterior.top_k_coverage,
        aggregate.speaker_count_posterior.credible_set_coverage,
        aggregate.speaker_occupancy.mean_dominant_speaker_share,
        aggregate.speaker_occupancy.maximum_dominant_speaker_share,
        aggregate.speaker_occupancy.mean_unknown_speaker_share,
        aggregate.speaker_occupancy.maximum_unknown_speaker_share,
        aggregate.speaker_occupancy.mean_minority_reference_recall,
        aggregate.word_attribution.micro_word_diarization_error_rate,
        aggregate.word_attribution.macro_word_diarization_error_rate,
        aggregate.overlap.precision,
        aggregate.overlap.recall,
        aggregate.overlap.f1,
        aggregate.diarization.macro_jer,
        aggregate.calibration.coverage,
        aggregate.calibration.duration_weighted_mean_brier_score,
        aggregate.calibration.duration_weighted_mean_ece,
    ];
    let nonnegative_optional = [
        aggregate.diarization.micro_der,
        aggregate.diarization.macro_der,
        aggregate.change_points.mean_absolute_error_sec,
        aggregate.speaker_count.mean_absolute_error,
        aggregate
            .speaker_count_posterior
            .mean_finite_negative_log_likelihood,
        aggregate.speaker_count_posterior.mean_entropy_bits,
        aggregate.speaker_occupancy.mean_effective_speaker_count,
    ];
    let performance_presence_is_consistent = match &aggregate.performance {
        Some(_) => aggregate.performance_observed_recordings == aggregate.recording_count,
        None => aggregate.performance_observed_recordings < aggregate.recording_count,
    };
    let counts_are_consistent = aggregate.recording_count > 0
        && aggregate.performance_observed_recordings <= aggregate.recording_count
        && aggregate.change_points.matched_count <= aggregate.change_points.reference_count
        && aggregate.change_points.matched_count <= aggregate.change_points.hypothesis_count
        && aggregate.speaker_count.exact_recordings <= aggregate.recording_count
        && performance_presence_is_consistent;
    let posterior = &aggregate.speaker_count_posterior;
    let posterior_counts_are_consistent = posterior
        .observed_recordings
        .checked_add(posterior.unavailable_recordings)
        == Some(aggregate.recording_count)
        && posterior.unresolved_recordings <= aggregate.recording_count
        && posterior.zero_reference_probability_recordings <= posterior.observed_recordings
        && posterior
            .finite_negative_log_likelihood_recordings
            .checked_add(posterior.zero_reference_probability_recordings)
            == Some(posterior.observed_recordings)
        && posterior.brier_observation_count == posterior.observed_recordings
        && posterior.top_k_observation_count == posterior.observed_recordings
        && posterior.credible_set_observation_count == posterior.observed_recordings
        && posterior.entropy_observation_count == posterior.observed_recordings
        && posterior.top_k_hit_count <= posterior.top_k_observation_count
        && posterior.credible_set_hit_count <= posterior.credible_set_observation_count
        && option_matches_count(
            posterior.mean_finite_negative_log_likelihood,
            posterior.finite_negative_log_likelihood_recordings,
        )
        && option_matches_count(
            posterior.mean_brier_score,
            posterior.brier_observation_count,
        )
        && option_matches_count(posterior.top_k_coverage, posterior.top_k_observation_count)
        && option_matches_count(
            posterior.credible_set_coverage,
            posterior.credible_set_observation_count,
        )
        && option_matches_count(
            posterior.mean_entropy_bits,
            posterior.entropy_observation_count,
        )
        && posterior
            .mean_brier_score
            .is_none_or(|value| (0.0..=2.0).contains(&value));
    let occupancy = &aggregate.speaker_occupancy;
    let occupancy_is_consistent = occupancy.dominant_collapse_recordings
        <= aggregate.recording_count
        && occupancy.reference_collapse_recordings <= aggregate.recording_count
        && occupancy.dominant_share_observation_count <= aggregate.recording_count
        && occupancy.unknown_share_observation_count <= aggregate.recording_count
        && occupancy.minority_recall_observation_count <= aggregate.recording_count
        && option_matches_count(
            occupancy.mean_dominant_speaker_share,
            occupancy.dominant_share_observation_count,
        )
        && option_matches_count(
            occupancy.maximum_dominant_speaker_share,
            occupancy.dominant_share_observation_count,
        )
        && option_matches_count(
            occupancy.mean_unknown_speaker_share,
            occupancy.unknown_share_observation_count,
        )
        && option_matches_count(
            occupancy.maximum_unknown_speaker_share,
            occupancy.unknown_share_observation_count,
        )
        && option_matches_count(
            occupancy.mean_minority_reference_recall,
            occupancy.minority_recall_observation_count,
        )
        && occupancy.mean_effective_speaker_count.is_some()
        && occupancy
            .mean_effective_speaker_count
            .is_none_or(|value| value <= f64::from(crate::model::MAX_SPEAKER_COUNT))
        && occupancy.mean_dominant_speaker_share <= occupancy.maximum_dominant_speaker_share
        && occupancy.mean_unknown_speaker_share <= occupancy.maximum_unknown_speaker_share;
    let words = &aggregate.word_attribution;
    let word_errors = words
        .incorrect_word_count
        .checked_add(words.unknown_word_count);
    let words_are_consistent = words
        .scored_word_count
        .checked_add(words.excluded_word_count)
        == Some(words.reference_word_count)
        && words
            .correct_word_count
            .checked_add(words.incorrect_word_count)
            .and_then(|value| value.checked_add(words.unknown_word_count))
            == Some(words.scored_word_count)
        && words.macro_observation_count <= aggregate.recording_count
        && option_matches_count(
            words.micro_word_diarization_error_rate,
            words.scored_word_count,
        )
        && option_matches_count(
            words.macro_word_diarization_error_rate,
            words.macro_observation_count,
        )
        && words.micro_word_diarization_error_rate
            == word_errors.and_then(|errors| ratio(errors as f64, words.scored_word_count as f64));
    let expected_change_precision = ratio(
        aggregate.change_points.matched_count as f64,
        aggregate.change_points.hypothesis_count as f64,
    );
    let expected_change_recall = ratio(
        aggregate.change_points.matched_count as f64,
        aggregate.change_points.reference_count as f64,
    );
    let expected_overlap_precision = ratio(
        aggregate.overlap.true_positive_sec,
        aggregate.overlap.true_positive_sec + aggregate.overlap.false_positive_sec,
    );
    let expected_overlap_recall = ratio(
        aggregate.overlap.true_positive_sec,
        aggregate.overlap.true_positive_sec + aggregate.overlap.false_negative_sec,
    );
    let derived_metrics_are_consistent = aggregate.diarization.micro_der
        == ratio(
            aggregate.diarization.missed_speech_sec
                + aggregate.diarization.false_alarm_sec
                + aggregate.diarization.speaker_confusion_sec,
            aggregate.diarization.reference_speaker_time_sec,
        )
        && aggregate.change_points.precision == expected_change_precision
        && aggregate.change_points.recall == expected_change_recall
        && aggregate.change_points.f1 == f1(expected_change_precision, expected_change_recall)
        && aggregate.speaker_count.exact_rate
            == ratio(
                aggregate.speaker_count.exact_recordings as f64,
                aggregate.recording_count as f64,
            )
        && aggregate.speaker_count_posterior.top_k_coverage
            == ratio(
                aggregate.speaker_count_posterior.top_k_hit_count as f64,
                aggregate.speaker_count_posterior.top_k_observation_count as f64,
            )
        && aggregate.speaker_count_posterior.credible_set_coverage
            == ratio(
                aggregate.speaker_count_posterior.credible_set_hit_count as f64,
                aggregate
                    .speaker_count_posterior
                    .credible_set_observation_count as f64,
            )
        && aggregate.overlap.precision == expected_overlap_precision
        && aggregate.overlap.recall == expected_overlap_recall
        && aggregate.overlap.f1 == f1(expected_overlap_precision, expected_overlap_recall)
        && aggregate.calibration.coverage
            == ratio(
                aggregate.calibration.observed_duration_sec,
                aggregate.calibration.opportunity_duration_sec,
            )
        && aggregate.performance.as_ref().is_none_or(|performance| {
            performance.real_time_factor
                == ratio(performance.wall_time_sec, performance.audio_duration_sec)
        });
    let durations_are_consistent = required.iter().all(|value| *value >= 0.0)
        && aggregate.calibration.observed_duration_sec
            <= aggregate.calibration.opportunity_duration_sec;
    if required.iter().all(|value| value.is_finite())
        && optional.iter().flatten().all(|value| value.is_finite())
        && bounded_rates
            .iter()
            .flatten()
            .all(|value| (0.0..=1.0).contains(value))
        && nonnegative_optional
            .iter()
            .flatten()
            .all(|value| *value >= 0.0)
        && counts_are_consistent
        && posterior_counts_are_consistent
        && occupancy_is_consistent
        && words_are_consistent
        && derived_metrics_are_consistent
        && durations_are_consistent
        && performance_finite
        && aggregate.performance.as_ref().is_none_or(|performance| {
            performance.audio_duration_sec >= 0.0 && performance.wall_time_sec >= 0.0
        })
    {
        Ok(())
    } else {
        Err(confidential_error(
            "aggregate_semantics",
            "confidential aggregate contains invalid semantic values",
        ))
    }
}

fn option_matches_count(value: Option<f64>, count: u64) -> bool {
    value.is_some() == (count > 0)
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then_some(numerator / denominator)
}

fn f1(precision: Option<f64>, recall: Option<f64>) -> Option<f64> {
    match (precision, recall) {
        (Some(precision), Some(recall)) if precision + recall > 0.0 => {
            Some(2.0 * precision * recall / (precision + recall))
        }
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    }
}

fn confidential_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("confidential_evaluation.{code}: {message}"))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use serde_json::json;
    use tempfile::tempdir;

    use crate::FwError;
    use crate::diarization::{
        DIARIZATION_HYPOTHESIS_SCHEMA_VERSION, DIARIZATION_REFERENCE_SCHEMA_VERSION,
        DiarizationHypothesisDocument, DiarizationReferenceDocument,
        EvaluationPerformanceObservation, EvaluationTurn, EvaluationWord,
    };
    use crate::model::{
        SpeakerCountCalibrationStatus, SpeakerCountEstimate, SpeakerCountEvidenceLane,
        SpeakerCountLaneEvidence, SpeakerCountLaneUnavailableReason, SpeakerCountPosteriorBin,
        SpeakerCountRange, SpeakerCountResourceSummary,
    };

    use super::{
        CONFIDENTIAL_EVALUATION_AGGREGATE_SCHEMA_VERSION,
        CONFIDENTIAL_EVALUATION_MANIFEST_SCHEMA_VERSION, ConfidentialEvaluationAggregate,
        canonical_sha256, run_confidential_evaluation, run_confidential_evaluation_with_cancel,
        verify_confidential_evaluation_aggregate,
    };

    fn rehash_for_test(
        mut aggregate: ConfidentialEvaluationAggregate,
    ) -> ConfidentialEvaluationAggregate {
        aggregate.result_sha256.clear();
        aggregate.result_sha256 = canonical_sha256(&aggregate).expect("recompute aggregate hash");
        aggregate
    }

    fn speaker_count_estimate() -> SpeakerCountEstimate {
        let posterior = vec![
            SpeakerCountPosteriorBin {
                count: 1,
                probability: 0.2,
            },
            SpeakerCountPosteriorBin {
                count: 2,
                probability: 0.7,
            },
        ];
        let unresolved_probability = 0.1_f64;
        let entropy_bits = posterior
            .iter()
            .map(|bin| -bin.probability * bin.probability.log2())
            .sum::<f64>()
            - unresolved_probability * unresolved_probability.log2();
        let available_lane = |lane| SpeakerCountLaneEvidence {
            lane,
            available: true,
            proposed_count: Some(2),
            confidence: 0.75,
            unavailable_reason: None,
        };
        SpeakerCountEstimate {
            schema_version: "speaker-count-estimate-v2".to_owned(),
            selected_count: Some(2),
            supported_range: Some(SpeakerCountRange {
                minimum: 2,
                maximum: 2,
            }),
            posterior,
            unresolved_probability,
            entropy_bits,
            stability: 0.75,
            constraint_lower_bound: 1,
            candidate_upper_bound: 2,
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
                prototype_count: 2,
                affinity_pair_evaluations: 2,
                retained_sparse_edges: 1,
                estimated_peak_buffer_bytes: 1_024,
                stability_replicates: 3,
                solver_iterations: 4,
                solver_sparse_matvec_terms: 8,
                solver_residual: Some(0.01),
            },
        }
    }

    fn write_fixture(input_root: &Path, manifest_path: &Path, source_override: Option<&Path>) {
        let audio_path = source_override
            .map(Path::to_owned)
            .unwrap_or_else(|| input_root.join("PRIVATE_AUDIO_SENTINEL.m4a"));
        fs::write(&audio_path, b"synthetic external audio bytes").expect("audio");
        let reference_path = input_root.join("PRIVATE_REFERENCE_SENTINEL.json");
        let hypothesis_path = input_root.join("PRIVATE_HYPOTHESIS_SENTINEL.json");
        let reference = DiarizationReferenceDocument {
            schema_version: DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
            recording_id: "private-recording-sentinel".to_owned(),
            duration_ms: 2_000,
            turns: vec![
                EvaluationTurn::labeled(0, 1_000, "speaker-private-a"),
                EvaluationTurn::labeled(1_000, 2_000, "speaker-private-b"),
            ],
            ignored_regions: vec![],
            speaker_hints: vec![],
            words: vec![
                EvaluationWord {
                    word_id: "word-000001".to_owned(),
                    start_ms: 200,
                    end_ms: 400,
                    speaker_ref: "speaker-private-a".to_owned(),
                },
                EvaluationWord {
                    word_id: "word-000002".to_owned(),
                    start_ms: 1_200,
                    end_ms: 1_400,
                    speaker_ref: "speaker-private-b".to_owned(),
                },
            ],
        };
        let hypothesis = DiarizationHypothesisDocument {
            schema_version: DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: "private-recording-sentinel".to_owned(),
            duration_ms: 2_000,
            turns: vec![
                EvaluationTurn {
                    speaker_confidence: Some(0.9),
                    ..EvaluationTurn::labeled(0, 1_000, "cluster-x")
                },
                EvaluationTurn {
                    speaker_confidence: Some(0.8),
                    ..EvaluationTurn::labeled(1_000, 2_000, "cluster-y")
                },
            ],
            speaker_count_estimate: Some(speaker_count_estimate()),
            performance: Some(EvaluationPerformanceObservation {
                audio_duration_ms: 2_000,
                wall_time_ms: 1_000,
                peak_rss_bytes: 4096,
            }),
        };
        fs::write(
            &reference_path,
            serde_json::to_vec(&reference).expect("reference JSON"),
        )
        .expect("reference");
        fs::write(
            &hypothesis_path,
            serde_json::to_vec(&hypothesis).expect("hypothesis JSON"),
        )
        .expect("hypothesis");
        fs::write(
            manifest_path,
            serde_json::to_vec(&json!({
                "schema_version": CONFIDENTIAL_EVALUATION_MANIFEST_SCHEMA_VERSION,
                "dataset_id": "private-dataset-sentinel",
                "config": crate::diarization::DiarizationScorerConfig::default(),
                "recordings": [{
                    "recording_id": "private-recording-sentinel",
                    "audio_path": audio_path,
                    "reference_path": reference_path,
                    "hypothesis_path": hypothesis_path
                }]
            }))
            .expect("manifest JSON"),
        )
        .expect("manifest");
    }

    #[test]
    fn external_evaluation_is_deterministic_and_path_free() {
        let project = tempdir().expect("project");
        let input = tempdir().expect("input");
        let output = tempdir().expect("output");
        let manifest = input.path().join("PRIVATE_MANIFEST_SENTINEL.json");
        write_fixture(input.path(), &manifest, None);

        let first_path = output.path().join("aggregate-one.json");
        let second_path = output.path().join("aggregate-two.json");
        let first =
            run_confidential_evaluation(project.path(), input.path(), &manifest, &first_path)
                .expect("first aggregate");
        let second =
            run_confidential_evaluation(project.path(), input.path(), &manifest, &second_path)
                .expect("second aggregate");
        assert_eq!(first, second);
        verify_confidential_evaluation_aggregate(&first).expect("aggregate hash");
        assert_eq!(
            first.schema_version,
            CONFIDENTIAL_EVALUATION_AGGREGATE_SCHEMA_VERSION
        );
        assert_eq!(first.recording_count, 1);
        assert_eq!(first.diarization.micro_der, Some(0.0));
        assert_eq!(first.speaker_count_posterior.observed_recordings, 1);
        assert_eq!(
            first
                .speaker_count_posterior
                .zero_reference_probability_recordings,
            0
        );
        assert_eq!(first.speaker_occupancy.dominant_collapse_recordings, 0);
        assert_eq!(first.word_attribution.reference_word_count, 2);
        assert_eq!(
            first.word_attribution.micro_word_diarization_error_rate,
            Some(0.0)
        );
        assert_eq!(
            first
                .performance
                .as_ref()
                .and_then(|value| value.real_time_factor),
            Some(0.5)
        );
        let serialized = fs::read_to_string(first_path).expect("aggregate output");
        for forbidden in [
            "PRIVATE_",
            "private-recording-sentinel",
            "private-dataset-sentinel",
            "speaker-private",
            "cluster-",
            "audio_path",
            "reference_path",
            "hypothesis_path",
            "transcript",
        ] {
            assert!(
                !serialized.contains(forbidden),
                "aggregate leaked forbidden marker {forbidden}"
            );
        }

        let mut tampered = first.clone();
        tampered.diarization.speaker_confusion_sec = 1.0;
        assert!(
            verify_confidential_evaluation_aggregate(&tampered)
                .expect_err("tampering must fail")
                .to_string()
                .contains("aggregate_hash")
        );

        let mut forged_posterior = first.clone();
        forged_posterior.speaker_count_posterior.top_k_hit_count = 2;
        forged_posterior.result_sha256.clear();
        forged_posterior.result_sha256 =
            canonical_sha256(&forged_posterior).expect("forged posterior hash");
        assert!(
            verify_confidential_evaluation_aggregate(&forged_posterior)
                .expect_err("posterior semantic forgery must fail")
                .to_string()
                .contains("aggregate_semantics")
        );

        let mut semantically_invalid = first;
        semantically_invalid.diarization.speaker_confusion_sec = -1.0;
        semantically_invalid.result_sha256.clear();
        semantically_invalid.result_sha256 =
            canonical_sha256(&semantically_invalid).expect("invalid aggregate hash");
        assert!(
            verify_confidential_evaluation_aggregate(&semantically_invalid)
                .expect_err("semantic forgery must fail")
                .to_string()
                .contains("aggregate_semantics")
        );
    }

    #[test]
    fn rehashed_derived_metric_forgeries_fail_semantic_verification() {
        let project = tempdir().expect("project");
        let input = tempdir().expect("input");
        let output = tempdir().expect("output");
        let manifest = input.path().join("PRIVATE_MANIFEST_SENTINEL.json");
        write_fixture(input.path(), &manifest, None);
        let aggregate = run_confidential_evaluation(
            project.path(),
            input.path(),
            &manifest,
            &output.path().join("aggregate.json"),
        )
        .expect("aggregate");
        verify_confidential_evaluation_aggregate(&aggregate).expect("writer aggregate verifies");

        let forgeries = [
            ("micro DER", {
                let mut forged = aggregate.clone();
                forged.diarization.micro_der = Some(0.5);
                rehash_for_test(forged)
            }),
            ("change precision", {
                let mut forged = aggregate.clone();
                forged.change_points.precision = Some(0.5);
                rehash_for_test(forged)
            }),
            ("change recall", {
                let mut forged = aggregate.clone();
                forged.change_points.recall = Some(0.5);
                rehash_for_test(forged)
            }),
            ("change F1", {
                let mut forged = aggregate.clone();
                forged.change_points.f1 = Some(0.5);
                rehash_for_test(forged)
            }),
            ("exact count rate", {
                let mut forged = aggregate.clone();
                forged.speaker_count.exact_rate = Some(0.5);
                rehash_for_test(forged)
            }),
            ("top-k coverage", {
                let mut forged = aggregate.clone();
                forged.speaker_count_posterior.top_k_coverage = Some(0.5);
                rehash_for_test(forged)
            }),
            ("credible-set coverage", {
                let mut forged = aggregate.clone();
                forged.speaker_count_posterior.credible_set_coverage = Some(0.5);
                rehash_for_test(forged)
            }),
            ("overlap precision", {
                let mut forged = aggregate.clone();
                forged.overlap.precision = Some(0.5);
                rehash_for_test(forged)
            }),
            ("overlap recall", {
                let mut forged = aggregate.clone();
                forged.overlap.recall = Some(0.5);
                rehash_for_test(forged)
            }),
            ("overlap F1", {
                let mut forged = aggregate.clone();
                forged.overlap.f1 = Some(0.5);
                rehash_for_test(forged)
            }),
            ("calibration coverage", {
                let mut forged = aggregate.clone();
                forged.calibration.coverage = Some(0.5);
                rehash_for_test(forged)
            }),
            ("performance RTF", {
                let mut forged = aggregate.clone();
                forged
                    .performance
                    .as_mut()
                    .expect("fixture performance")
                    .real_time_factor = Some(0.25);
                rehash_for_test(forged)
            }),
        ];

        for (name, forged) in forgeries {
            let error = match verify_confidential_evaluation_aggregate(&forged) {
                Ok(()) => panic!("rehashed {name} forgery must fail"),
                Err(error) => error,
            };
            assert!(
                error.to_string().contains("aggregate_semantics"),
                "rehashed {name} forgery returned {error}"
            );
        }
    }

    #[test]
    fn output_beneath_project_is_rejected_before_creation() {
        let project = tempdir().expect("project");
        let input = tempdir().expect("input");
        let manifest = input.path().join("PRIVATE_MANIFEST_SENTINEL.json");
        write_fixture(input.path(), &manifest, None);
        let output = project.path().join("must-not-exist.json");
        let error = run_confidential_evaluation(project.path(), input.path(), &manifest, &output)
            .expect_err("project output must fail");
        assert!(error.to_string().contains("output_inside_project"));
        assert!(!output.exists());
        assert!(!error.to_string().contains("PRIVATE_MANIFEST_SENTINEL"));
    }

    #[test]
    fn source_escape_is_rejected_without_echoing_private_names() {
        let project = tempdir().expect("project");
        let input = tempdir().expect("input");
        let output = tempdir().expect("output");
        let manifest = input.path().join("PRIVATE_MANIFEST_SENTINEL.json");
        let escaped_audio = project.path().join("PRIVATE_ESCAPE_SENTINEL.m4a");
        write_fixture(input.path(), &manifest, Some(&escaped_audio));
        let error = run_confidential_evaluation(
            project.path(),
            input.path(),
            &manifest,
            &output.path().join("aggregate.json"),
        )
        .expect_err("source escape must fail");
        assert!(error.to_string().contains("source_boundary"));
        assert!(!error.to_string().contains("PRIVATE_ESCAPE_SENTINEL"));
        assert!(!error.to_string().contains("PRIVATE_MANIFEST_SENTINEL"));
    }

    #[test]
    fn cancellation_never_creates_an_aggregate() {
        let project = tempdir().expect("project");
        let input = tempdir().expect("input");
        let output = tempdir().expect("output");
        let manifest = input.path().join("PRIVATE_MANIFEST_SENTINEL.json");
        let aggregate = output.path().join("must-not-exist.json");
        write_fixture(input.path(), &manifest, None);
        let error = run_confidential_evaluation_with_cancel(
            project.path(),
            input.path(),
            &manifest,
            &aggregate,
            || true,
        )
        .expect_err("cancellation");
        assert!(matches!(error, FwError::Cancelled(_)));
        assert!(!aggregate.exists());
        assert!(!error.to_string().contains("PRIVATE_MANIFEST_SENTINEL"));
    }

    #[test]
    fn duplicate_sources_fail_without_per_recording_output() {
        let project = tempdir().expect("project");
        let input = tempdir().expect("input");
        let output = tempdir().expect("output");
        let manifest = input.path().join("PRIVATE_MANIFEST_SENTINEL.json");
        let aggregate = output.path().join("must-not-exist.json");
        write_fixture(input.path(), &manifest, None);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest).expect("read manifest"))
                .expect("parse manifest");
        let duplicate = value["recordings"][0].clone();
        value["recordings"]
            .as_array_mut()
            .expect("recordings")
            .push({
                let mut duplicate = duplicate;
                duplicate["recording_id"] = json!("private-recording-z");
                duplicate
            });
        fs::write(
            &manifest,
            serde_json::to_vec(&value).expect("duplicate manifest JSON"),
        )
        .expect("duplicate manifest");
        let error =
            run_confidential_evaluation(project.path(), input.path(), &manifest, &aggregate)
                .expect_err("duplicate sources");
        assert!(error.to_string().contains("duplicate_source"));
        assert!(!aggregate.exists());
        assert!(!error.to_string().contains("PRIVATE_"));
    }

    #[test]
    fn empty_audio_fails_without_echoing_its_name() {
        let project = tempdir().expect("project");
        let input = tempdir().expect("input");
        let output = tempdir().expect("output");
        let manifest = input.path().join("PRIVATE_MANIFEST_SENTINEL.json");
        let aggregate = output.path().join("must-not-exist.json");
        write_fixture(input.path(), &manifest, None);
        fs::write(input.path().join("PRIVATE_AUDIO_SENTINEL.m4a"), []).expect("empty audio");
        let error =
            run_confidential_evaluation(project.path(), input.path(), &manifest, &aggregate)
                .expect_err("empty audio");
        assert!(error.to_string().contains("audio_empty"));
        assert!(!aggregate.exists());
        assert!(!error.to_string().contains("PRIVATE_AUDIO_SENTINEL"));
    }
}
