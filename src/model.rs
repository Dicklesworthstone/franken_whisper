use std::collections::BTreeSet;
use std::path::PathBuf;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Phase 3: backend-specific parameter types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    Txt,
    Vtt,
    Srt,
    Csv,
    Json,
    JsonFull,
    Lrc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum TimestampLevel {
    Chunk,
    Word,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DecodingParams {
    pub best_of: Option<u32>,
    pub beam_size: Option<u32>,
    pub max_context: Option<i32>,
    pub max_segment_length: Option<u32>,
    pub temperature: Option<f32>,
    pub temperature_increment: Option<f32>,
    pub entropy_threshold: Option<f32>,
    pub logprob_threshold: Option<f32>,
    pub no_speech_threshold: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VadParams {
    pub model_path: Option<PathBuf>,
    pub threshold: Option<f32>,
    pub min_speech_duration_ms: Option<u32>,
    pub min_silence_duration_ms: Option<u32>,
    pub max_speech_duration_s: Option<f32>,
    pub speech_pad_ms: Option<u32>,
    pub samples_overlap: Option<f32>,
}

/// One point in an explicitly supplied speaker-count prior.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeakerCountPriorMass {
    pub count: u32,
    pub probability: f64,
}

/// Typed speaker-count semantics shared by native and external diarizers.
///
/// A hard count restricts candidate model search; it never forces assignments
/// or removes UNKNOWN. `Prior` is representable and validated now, but remains
/// unresolved until calibrated count-posterior fusion is implemented.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SpeakerCountRequest {
    #[default]
    Infer,
    Prior {
        bins: Vec<SpeakerCountPriorMass>,
    },
    Range {
        minimum: u32,
        maximum: u32,
    },
    HardConstraint {
        count: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiarizationConfig {
    pub no_stem: bool,
    pub whisper_model: Option<String>,
    pub suppress_numerals: bool,
    pub device: Option<String>,
    pub batch_size: Option<u32>,
}

/// Speaker-diarization implementation selected by a library or CLI request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DiarizationEngine {
    /// Select the best admitted implementation, preferring the native acoustic
    /// engine when its input and evidence requirements are satisfied.
    Auto,
    /// Rust-native, waveform-only acoustic diarization.
    Acoustic,
    /// User-installed subprocess backend.
    External,
    /// Optional in-process neural speaker-embedding engine.
    Neural,
}

/// Evidence-gated rollout stage for `auto` acoustic diarization.
///
/// Explicit `DiarizationEngine::Acoustic` requests are not changed by this
/// stage; the gate controls only whether `auto` may select the acoustic engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AcousticDiarizationRolloutStage {
    /// Acoustic output is not user-visible and cannot satisfy an `auto`
    /// request. Focused development evidence may still be collected directly.
    #[default]
    Shadow,
    /// The implementation contract is validated, but `auto` remains off.
    Validated,
    /// `auto` uses verified external output when present, then acoustic.
    Fallback,
    /// `auto` prefers acoustic even when external output is present.
    Primary,
    /// `auto` admits only acoustic; external output is not selected.
    Sole,
}

/// Conservative action when the selected diarizer cannot make a supported
/// assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DiarizationFallbackPolicy {
    /// Preserve attributable hard-hint regions and emit unknown elsewhere.
    Unknown,
    /// Permit an admitted external backend to attempt the request.
    External,
    /// Fail the request with a structured error.
    Error,
}

/// Strength assigned to a caller-provided known-speaker interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnownSpeakerPolicy {
    /// Frames from this interval are immutable must-link evidence.
    HardMustLink,
    /// Frames may enroll a profile, but contradictory evidence may downweight
    /// or reject them.
    SoftEnrollment,
}

/// One `speaker-hints-v1` interval.
///
/// `speaker_ref` is an opaque identifier scoped to this run. It is not a
/// biometric or legal identity claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnownSpeakerInterval {
    pub speaker_ref: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub confidence: f64,
    pub policy: KnownSpeakerPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<String>,
}

/// Typed native diarization request (`speaker-hints-v1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizationRequest {
    pub engine: DiarizationEngine,
    pub fallback: DiarizationFallbackPolicy,
    pub speaker_count: SpeakerCountRequest,
    #[serde(default)]
    pub known_intervals: Vec<KnownSpeakerInterval>,
    /// Samples removed from each enrollment edge to avoid boundary bleed.
    pub enrollment_edge_guard_ms: u32,
    /// Maximum number of global clustering prototypes.
    pub max_prototypes: u16,
    /// Record explicit consent for reusable-profile persistence. Default-off.
    /// Schema v5 deliberately persists only privacy-safe summaries; raw
    /// acoustic vectors remain excluded until a separately reviewed schema.
    #[serde(default)]
    pub persist_profiles: bool,
}

impl Default for DiarizationRequest {
    fn default() -> Self {
        Self {
            engine: DiarizationEngine::Auto,
            fallback: DiarizationFallbackPolicy::Unknown,
            speaker_count: SpeakerCountRequest::Infer,
            known_intervals: Vec::new(),
            enrollment_edge_guard_ms: 100,
            max_prototypes: 512,
            persist_profiles: false,
        }
    }
}

/// Hard request-size limits for the bounded `speaker-hints-v1` surface.
pub const MAX_KNOWN_SPEAKER_INTERVALS: usize = 1_024;
pub const MAX_SPEAKER_REF_BYTES: usize = 256;
pub const MAX_HINT_PROVENANCE_BYTES: usize = 4_096;
pub const MAX_SPEAKER_COUNT: u32 = 64;

/// Stable validation code for malformed diarization requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiarizationValidationCode {
    TooManyKnownIntervals,
    EmptySpeakerRef,
    SpeakerRefTooLong,
    ProvenanceTooLong,
    InvalidHintConfidence,
    ReversedHintInterval,
    HintOutsideAudio,
    ContradictoryHardHints,
    InvalidSpeakerCount,
    InvalidPrototypeCap,
}

impl DiarizationValidationCode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TooManyKnownIntervals => "diarization.too_many_known_intervals",
            Self::EmptySpeakerRef => "diarization.empty_speaker_ref",
            Self::SpeakerRefTooLong => "diarization.speaker_ref_too_long",
            Self::ProvenanceTooLong => "diarization.provenance_too_long",
            Self::InvalidHintConfidence => "diarization.invalid_hint_confidence",
            Self::ReversedHintInterval => "diarization.reversed_hint_interval",
            Self::HintOutsideAudio => "diarization.hint_outside_audio",
            Self::ContradictoryHardHints => "diarization.contradictory_hard_hints",
            Self::InvalidSpeakerCount => "diarization.invalid_speaker_count",
            Self::InvalidPrototypeCap => "diarization.invalid_prototype_cap",
        }
    }
}

/// Structured validation failure suitable for robot-mode serialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiarizationValidationError {
    pub code: DiarizationValidationCode,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint_index: Option<usize>,
}

impl std::fmt::Display for DiarizationValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for DiarizationValidationError {}

impl DiarizationRequest {
    /// Validate speaker hints and count constraints against the canonical
    /// normalized audio duration.
    pub fn validate(&self, audio_duration_ms: u64) -> Result<(), DiarizationValidationError> {
        if self.max_prototypes == 0 || self.max_prototypes > 512 {
            return Err(DiarizationValidationError {
                code: DiarizationValidationCode::InvalidPrototypeCap,
                message: "max_prototypes must be within 1..=512 for acoustic-v2".to_owned(),
                hint_index: None,
            });
        }
        if self.known_intervals.len() > MAX_KNOWN_SPEAKER_INTERVALS {
            return Err(DiarizationValidationError {
                code: DiarizationValidationCode::TooManyKnownIntervals,
                message: format!(
                    "known_intervals exceeds the acoustic-v2 limit of {MAX_KNOWN_SPEAKER_INTERVALS}"
                ),
                hint_index: None,
            });
        }
        validate_speaker_count_request(&self.speaker_count)?;

        for (index, hint) in self.known_intervals.iter().enumerate() {
            if hint.speaker_ref.trim().is_empty() {
                return Err(DiarizationValidationError {
                    code: DiarizationValidationCode::EmptySpeakerRef,
                    message: "speaker_ref must not be empty".to_owned(),
                    hint_index: Some(index),
                });
            }
            if hint.speaker_ref.len() > MAX_SPEAKER_REF_BYTES {
                return Err(DiarizationValidationError {
                    code: DiarizationValidationCode::SpeakerRefTooLong,
                    message: format!(
                        "speaker_ref exceeds the {MAX_SPEAKER_REF_BYTES}-byte acoustic-v2 limit"
                    ),
                    hint_index: Some(index),
                });
            }
            if hint
                .provenance
                .as_ref()
                .is_some_and(|value| value.len() > MAX_HINT_PROVENANCE_BYTES)
            {
                return Err(DiarizationValidationError {
                    code: DiarizationValidationCode::ProvenanceTooLong,
                    message: format!(
                        "hint provenance exceeds the {MAX_HINT_PROVENANCE_BYTES}-byte acoustic-v2 limit"
                    ),
                    hint_index: Some(index),
                });
            }
            if !hint.confidence.is_finite() || !(0.0..=1.0).contains(&hint.confidence) {
                return Err(DiarizationValidationError {
                    code: DiarizationValidationCode::InvalidHintConfidence,
                    message: "hint confidence must be finite and within [0, 1]".to_owned(),
                    hint_index: Some(index),
                });
            }
            if hint.end_ms <= hint.start_ms {
                return Err(DiarizationValidationError {
                    code: DiarizationValidationCode::ReversedHintInterval,
                    message: "hint interval must satisfy start_ms < end_ms".to_owned(),
                    hint_index: Some(index),
                });
            }
            if hint.end_ms > audio_duration_ms {
                return Err(DiarizationValidationError {
                    code: DiarizationValidationCode::HintOutsideAudio,
                    message: format!(
                        "hint end_ms {} exceeds audio duration {audio_duration_ms}",
                        hint.end_ms
                    ),
                    hint_index: Some(index),
                });
            }
        }

        for (left_index, left) in self.known_intervals.iter().enumerate() {
            if left.policy != KnownSpeakerPolicy::HardMustLink {
                continue;
            }
            for (right_index, right) in self.known_intervals.iter().enumerate().skip(left_index + 1)
            {
                if right.policy == KnownSpeakerPolicy::HardMustLink
                    && left.speaker_ref != right.speaker_ref
                    && left.start_ms < right.end_ms
                    && right.start_ms < left.end_ms
                {
                    return Err(DiarizationValidationError {
                        code: DiarizationValidationCode::ContradictoryHardHints,
                        message: format!(
                            "hard hints {left_index} and {right_index} overlap with different speaker_ref values"
                        ),
                        hint_index: Some(right_index),
                    });
                }
            }
        }

        let hard_speaker_count = self
            .known_intervals
            .iter()
            .filter(|hint| hint.policy == KnownSpeakerPolicy::HardMustLink)
            .map(|hint| hint.speaker_ref.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let constrained_maximum = match self.speaker_count {
            SpeakerCountRequest::Range { maximum, .. } => Some(maximum as usize),
            SpeakerCountRequest::HardConstraint { count } => Some(count as usize),
            SpeakerCountRequest::Infer | SpeakerCountRequest::Prior { .. } => None,
        };
        if let Some(maximum) = constrained_maximum
            && maximum < hard_speaker_count
        {
            return Err(DiarizationValidationError {
                code: DiarizationValidationCode::InvalidSpeakerCount,
                message: format!(
                    "speaker-count request permits at most {maximum} speakers but hard hints name {hard_speaker_count} distinct speakers"
                ),
                hint_index: None,
            });
        }
        Ok(())
    }
}

pub(crate) fn validate_speaker_count_request(
    request: &SpeakerCountRequest,
) -> Result<(), DiarizationValidationError> {
    let invalid = |message: String| DiarizationValidationError {
        code: DiarizationValidationCode::InvalidSpeakerCount,
        message,
        hint_index: None,
    };
    match request {
        SpeakerCountRequest::Infer => Ok(()),
        SpeakerCountRequest::HardConstraint { count } => {
            if !(1..=MAX_SPEAKER_COUNT).contains(count) {
                return Err(invalid(format!(
                    "hard speaker count must be within 1..={MAX_SPEAKER_COUNT}"
                )));
            }
            Ok(())
        }
        SpeakerCountRequest::Range { minimum, maximum } => {
            if *minimum == 0 || minimum > maximum || *maximum > MAX_SPEAKER_COUNT {
                return Err(invalid(format!(
                    "speaker-count range must satisfy 1 <= minimum <= maximum <= {MAX_SPEAKER_COUNT}"
                )));
            }
            Ok(())
        }
        SpeakerCountRequest::Prior { bins } => {
            if bins.is_empty() {
                return Err(invalid(
                    "speaker-count prior must contain at least one bin".to_owned(),
                ));
            }
            let mut previous_count = None;
            let mut total = 0.0_f64;
            for bin in bins {
                if !(1..=MAX_SPEAKER_COUNT).contains(&bin.count) {
                    return Err(invalid(format!(
                        "speaker-count prior support must be within 1..={MAX_SPEAKER_COUNT}"
                    )));
                }
                if previous_count.is_some_and(|previous| bin.count <= previous) {
                    return Err(invalid(
                        "speaker-count prior bins must be unique and strictly increasing"
                            .to_owned(),
                    ));
                }
                if !bin.probability.is_finite() || bin.probability < 0.0 {
                    return Err(invalid(
                        "speaker-count prior probabilities must be finite and non-negative"
                            .to_owned(),
                    ));
                }
                total += bin.probability;
                previous_count = Some(bin.count);
            }
            if !total.is_finite() || (total - 1.0).abs() > 1e-9 {
                return Err(invalid(
                    "speaker-count prior probabilities must sum to exactly 1 within 1e-9"
                        .to_owned(),
                ));
            }
            Ok(())
        }
    }
}

/// Why a diarization result conservatively fell back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiarizationFallbackStatus {
    NotNeeded,
    InsufficientEvidence,
    CalibrationInvalid,
    ResourceLimit,
    UnsatisfiedConstraints,
    SpeakerCountUnresolved,
    ExternalBackend,
}

/// One acoustic speaker turn, independent of ASR segment boundaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizationTurn {
    pub start_ms: u64,
    pub end_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub change_confidence: Option<f64>,
    pub overlap_suspected: bool,
    pub hard_hint_attributed: bool,
}

/// Why an agent may want to provide another contextual timestamp interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerAttributionQueryReason {
    UnknownAttribution,
    LowConfidence,
    OverlapAmbiguity,
}

/// Content-bound, feature-value-free request for optional agent supervision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerAttributionQuery {
    pub query_id_sha256: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub reason: SpeakerAttributionQueryReason,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidate_speaker_refs: Vec<String>,
    pub suggested_policy: KnownSpeakerPolicy,
}

/// Privacy-safe quality summary for one within-run speaker profile.
///
/// Raw acoustic vectors and audio are intentionally absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerProfileSummary {
    pub speaker_ref: String,
    pub frame_count: u64,
    pub voiced_duration_ms: u64,
    pub reliability: f64,
    #[serde(default)]
    pub voice_profile_count: u32,
    pub channel_profile_count: u32,
    #[serde(default)]
    pub training_accepted_count: u32,
    #[serde(default)]
    pub training_downweighted_count: u32,
    #[serde(default)]
    pub training_quarantined_count: u32,
    pub anchored: bool,
    pub soft_hint_contradiction: Option<f64>,
}

/// Final, feature-value-free disposition of one known-speaker interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerHintDisposition {
    HardAttributed,
    Accepted,
    PartiallyAccepted,
    Rejected,
    NoUsableTracklets,
}

/// Privacy-safe enrollment audit for one known-speaker interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerHintEvidenceSummary {
    pub hint_index: u64,
    pub speaker_ref: String,
    pub policy: KnownSpeakerPolicy,
    pub disposition: SpeakerHintDisposition,
    pub usable_tracklet_count: u64,
    pub accepted_tracklet_count: u64,
    pub rejected_tracklet_count: u64,
    pub profile_accepted_tracklet_count: u64,
    pub profile_downweighted_tracklet_count: u64,
    pub profile_quarantined_tracklet_count: u64,
    pub applied_weight: f64,
    pub contradiction_score: Option<f64>,
}

/// Why one candidate speaker was retained or rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerEvidenceReason {
    SupportedByHardHint,
    SupportedByIndependentRecurrence,
    SupportedByRepeatedTracklets,
    SupportedByExternalAttribution,
    NoAssignedSpeech,
    InsufficientIndependentRecurrence,
    InsufficientVoicedFrames,
    InsufficientAssignmentConfidence,
    InsufficientProfileReliability,
    MergeCompatibleWithSupportedSpeaker,
}

/// Feature-value-free occupancy and quality evidence for one speaker label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerEvidenceSummary {
    pub speaker_ref: String,
    pub assigned_tracklet_count: u64,
    pub independent_tracklet_count: u64,
    pub recurrence_episode_count: u64,
    pub voiced_frame_count: u64,
    pub independent_voiced_frame_count: u64,
    pub voiced_duration_ms: u64,
    pub mean_assignment_confidence: f64,
    pub profile_reliability: f64,
    pub hard_anchored: bool,
    pub separated_from_supported_speakers: bool,
    pub reasons: Vec<SpeakerEvidenceReason>,
    pub supported: bool,
}

/// Resolution state for a typed speaker-count request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerCountOutcomeStatus {
    Resolved,
    Satisfied,
    Unsatisfied,
    Unresolved,
}

/// Run-level reason attached to speaker-count resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerCountOutcomeReason {
    EvidenceSupportedCount,
    RequestedCountMatched,
    RequestedCountMismatch,
    SpeakerCountPriorFusionUnavailable,
    NoSupportedSpeakers,
    DominantSpeakerShareExceeded,
    AmbiguousSpeakerSeparation,
    ExternalAttribution,
}

/// Auditable result of applying one speaker-count request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerCountOutcome {
    pub request: SpeakerCountRequest,
    pub status: SpeakerCountOutcomeStatus,
    pub supported_speaker_count: u32,
    pub active_speaker_refs: Vec<String>,
    pub dominant_speaker_share: f64,
    pub unknown_voiced_share: f64,
    pub reasons: Vec<SpeakerCountOutcomeReason>,
    pub speaker_evidence: Vec<SpeakerEvidenceSummary>,
}

/// Complete typed diarization result attached to [`TranscriptionResult`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiarizationReport {
    pub implementation: String,
    pub contract_version: String,
    pub feature_schema: String,
    pub normalized_input_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hint_document_sha256: Option<String>,
    pub turns: Vec<DiarizationTurn>,
    pub profiles: Vec<SpeakerProfileSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hint_evidence: Vec<SpeakerHintEvidenceSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub speaker_queries: Vec<SpeakerAttributionQuery>,
    pub speaker_count: SpeakerCountOutcome,
    pub fallback_status: DiarizationFallbackStatus,
    #[serde(default)]
    pub diagnostics: Vec<String>,
}

// ---------------------------------------------------------------------------
// bd-1rj.2: Word-level timestamp parameters (whisper.cpp)
// ---------------------------------------------------------------------------

/// Configuration for word-level timestamp extraction in whisper.cpp.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WordTimestampParams {
    /// Enable word-level timestamps (whisper.cpp: -ml 1).
    #[serde(default)]
    pub enabled: bool,
    /// Maximum number of characters per word segment.
    pub max_len: Option<u32>,
    /// Word timestamp probability threshold (whisper.cpp: -wt).
    pub token_threshold: Option<f32>,
    /// Token sum probability threshold for splitting words (whisper.cpp: -wtps).
    pub token_sum_threshold: Option<f32>,
}

// ---------------------------------------------------------------------------
// bd-1rj.3: Insanely-fast-whisper tuning parameters
// ---------------------------------------------------------------------------

/// Device map strategy for insanely-fast-whisper model placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceMapStrategy {
    /// Automatic device placement across available GPUs.
    Auto,
    /// Sequential placement on a single device.
    Sequential,
}

/// Extended tuning parameters for insanely-fast-whisper.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InsanelyFastTuningParams {
    /// Device map strategy for multi-GPU setups.
    pub device_map: Option<DeviceMapStrategy>,
    /// Torch dtype for inference (e.g. "float16", "bfloat16", "float32").
    pub torch_dtype: Option<String>,
    /// Disable BetterTransformer optimization.
    #[serde(default)]
    pub disable_better_transformer: bool,
}

// ---------------------------------------------------------------------------
// bd-1rj.4: Diarization pipeline extension parameters
// ---------------------------------------------------------------------------

/// Forced alignment configuration for whisper-diarization.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlignmentConfig {
    /// Alignment model identifier (e.g. "WAV2VEC2_ASR_LARGE_LV60K_960H").
    pub alignment_model: Option<String>,
    /// Interpolation resolution character for alignment (e.g. words, characters).
    pub interpolate_method: Option<String>,
    /// Return character-level alignments in addition to word-level.
    #[serde(default)]
    pub return_char_alignments: bool,
}

/// Punctuation restoration parameters for whisper-diarization.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PunctuationConfig {
    /// Punctuation restoration model identifier.
    pub model: Option<String>,
    /// Enable punctuation restoration post-processing.
    #[serde(default)]
    pub enabled: bool,
}

/// Source separation (Demucs) parameters for whisper-diarization.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SourceSeparationConfig {
    /// Enable Demucs source separation before diarization.
    #[serde(default)]
    pub enabled: bool,
    /// Demucs model name (e.g. "htdemucs", "htdemucs_ft").
    pub model: Option<String>,
    /// Number of audio shifts for test-time augmentation.
    pub shifts: Option<u32>,
    /// Overlap between audio chunks (0.0 to 1.0).
    pub overlap: Option<f32>,
}

/// Aggregated backend-specific parameters for Phase 3 parity.
///
/// Each backend picks the fields it supports and ignores the rest.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackendParams {
    /// Additional output formats to request from whisper.cpp (beyond JSON).
    pub output_formats: Vec<OutputFormat>,
    /// Timestamp granularity for insanely-fast-whisper.
    pub timestamp_level: Option<TimestampLevel>,
    /// Decoding parameters (whisper.cpp).
    pub decoding: Option<DecodingParams>,
    /// Voice Activity Detection parameters (whisper.cpp).
    pub vad: Option<VadParams>,
    /// Diarization-specific pipeline options.
    pub diarization_config: Option<DiarizationConfig>,
    /// Backend-independent native acoustic diarization request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acoustic_diarization: Option<DiarizationRequest>,
    /// GPU device identifier (insanely-fast, diarization).
    pub gpu_device: Option<String>,
    /// Enable Flash Attention 2 (insanely-fast).
    pub flash_attention: Option<bool>,
    /// Explicit HuggingFace token override for insanely-fast diarization.
    /// Never serialized to avoid leaking secrets in JSON output or persistence.
    #[serde(skip_serializing)]
    pub insanely_fast_hf_token: Option<String>,
    /// Explicit transcript artifact path override for insanely-fast output.
    pub insanely_fast_transcript_path: Option<PathBuf>,
    /// Suppress timestamps in output (whisper.cpp).
    pub no_timestamps: bool,
    /// Exit after detecting language (whisper.cpp).
    pub detect_language_only: bool,
    /// Batch size for inference (insanely-fast, diarization).
    pub batch_size: Option<u32>,
    /// Split on word boundaries (whisper.cpp).
    pub split_on_word: bool,
    /// Number of threads for computation (whisper.cpp: -t).
    pub threads: Option<u32>,
    /// Number of processors for parallel processing (whisper.cpp: -p).
    pub processors: Option<u32>,
    /// Disable GPU acceleration (whisper.cpp: -ng).
    #[serde(default)]
    pub no_gpu: bool,
    /// Initial text prompt for biasing transcription (whisper.cpp: --prompt).
    pub prompt: Option<String>,
    /// Always prepend initial prompt to every segment (whisper.cpp: --carry-initial-prompt).
    #[serde(default)]
    pub carry_initial_prompt: bool,
    /// Disable temperature fallback during decoding (whisper.cpp: -nf).
    #[serde(default)]
    pub no_fallback: bool,
    /// Suppress non-speech tokens (whisper.cpp: -sns).
    #[serde(default)]
    pub suppress_nst: bool,
    /// Time offset in milliseconds (whisper.cpp: -ot).
    pub offset_ms: Option<u64>,
    /// Duration of audio to process in milliseconds (whisper.cpp: -d).
    pub duration_ms: Option<u64>,
    /// Audio context size, 0 = all (whisper.cpp: -ac).
    pub audio_ctx: Option<i32>,
    /// Word timestamp probability threshold (whisper.cpp: -wt).
    pub word_threshold: Option<f32>,
    /// Regex pattern to suppress matching tokens (whisper.cpp: --suppress-regex).
    pub suppress_regex: Option<String>,
    /// Enable TinyDiarize speaker-turn token injection (whisper.cpp: --tdrz).
    #[serde(default)]
    pub tiny_diarize: bool,
    // -----------------------------------------------------------------
    // bd-1rj.2: whisper.cpp word-level timestamps
    // -----------------------------------------------------------------
    /// Word-level timestamp extraction configuration (whisper.cpp).
    pub word_timestamps: Option<WordTimestampParams>,
    // -----------------------------------------------------------------
    // bd-1rj.3: insanely-fast-whisper tuning
    // -----------------------------------------------------------------
    /// Extended tuning parameters for insanely-fast-whisper.
    pub insanely_fast_tuning: Option<InsanelyFastTuningParams>,
    // -----------------------------------------------------------------
    // bd-1rj.4: diarization pipeline extensions
    // -----------------------------------------------------------------
    /// Forced alignment configuration (whisper-diarization).
    pub alignment: Option<AlignmentConfig>,
    /// Punctuation restoration parameters (whisper-diarization).
    pub punctuation: Option<PunctuationConfig>,
    /// Source separation (Demucs) parameters (whisper-diarization).
    pub source_separation: Option<SourceSeparationConfig>,
    // -----------------------------------------------------------------
    // Speculative cancel-correct streaming request.
    //
    // When set, the orchestrator routes the run through the
    // `streaming::SpeculativeStreamingPipeline` instead of the
    // single-backend `Backend` stage. The request shape is a
    // serde-friendly mirror of the user-visible CLI knobs; the
    // dispatch path converts it into a `streaming::SpeculativeConfig`
    // at runtime.
    // -----------------------------------------------------------------
    pub speculative: Option<SpeculativeRequest>,
}

/// Serde-friendly mirror of the speculative-streaming request configuration.
///
/// `BackendParams.speculative` carries this struct to thread CLI flags through
/// to the orchestrator, which converts it into the execution-shape
/// [`crate::streaming::SpeculativeConfig`] at dispatch time. Kept separate from
/// `SpeculativeConfig` so that `model.rs` does not need to depend on
/// `streaming.rs` (which would create a circular module dependency).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeRequest {
    /// Initial speculation window size in milliseconds.
    pub window_size_ms: u64,
    /// Window overlap in milliseconds.
    pub overlap_ms: u64,
    /// Model name passed to the fast lane (low latency).
    pub fast_model_name: String,
    /// Model name passed to the quality lane (correction / verification).
    pub quality_model_name: String,
    /// Maximum word-error-rate tolerance before correction triggers.
    /// `None` falls back to the [`CorrectionTolerance::default`] value (0.1).
    pub max_wer_tolerance: Option<f64>,
    /// Whether to enable adaptive window sizing via
    /// [`crate::speculation::SpeculationWindowController`].
    pub adaptive: bool,
    /// Force a correction on every window (evaluation mode).
    pub always_correct: bool,
}

impl Default for SpeculativeRequest {
    fn default() -> Self {
        Self {
            window_size_ms: 3000,
            overlap_ms: 500,
            fast_model_name: "auto-fast".to_owned(),
            quality_model_name: "auto-quality".to_owned(),
            max_wer_tolerance: None,
            adaptive: true,
            always_correct: false,
        }
    }
}

/// Describes the capabilities of a specific engine implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCapabilities {
    pub supports_diarization: bool,
    pub supports_translation: bool,
    pub supports_word_timestamps: bool,
    pub supports_gpu: bool,
    pub supports_streaming: bool,
}

/// A single backend entry in the `backends.discovery` report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendDiscoveryEntry {
    /// Machine-readable backend identifier (matches [`BackendKind`] serialization).
    pub name: String,
    /// Which [`BackendKind`] this entry corresponds to.
    pub kind: BackendKind,
    /// Whether the backend's external binary/script is currently available.
    pub available: bool,
    /// Declared capabilities of this engine.
    pub capabilities: EngineCapabilities,
}

/// Top-level report for the `backends.discovery` NDJSON event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendsReport {
    pub backends: Vec<BackendDiscoveryEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Auto,
    WhisperCpp,
    InsanelyFast,
    WhisperDiarization,
}

impl BackendKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::WhisperCpp => "whisper_cpp",
            Self::InsanelyFast => "insanely_fast",
            Self::WhisperDiarization => "whisper_diarization",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InputSource {
    File {
        path: PathBuf,
    },
    Stdin {
        hint_extension: Option<String>,
    },
    Microphone {
        seconds: u32,
        device: Option<String>,
        ffmpeg_format: Option<String>,
        ffmpeg_source: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeRequest {
    pub input: InputSource,
    pub backend: BackendKind,
    pub model: Option<String>,
    pub language: Option<String>,
    pub translate: bool,
    pub diarize: bool,
    pub persist: bool,
    pub db_path: PathBuf,
    pub timeout_ms: Option<u64>,
    /// Phase 3 backend-specific parameters (backward-compatible default).
    #[serde(default)]
    pub backend_params: BackendParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub start_sec: Option<f64>,
    pub end_sec: Option<f64>,
    pub text: String,
    pub speaker: Option<String>,
    /// ASR token/text confidence. Speaker assignment confidence lives on
    /// [`DiarizationTurn::speaker_confidence`].
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerationBackend {
    None,
    Frankentorch,
    Frankenjax,
}

impl AccelerationBackend {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Frankentorch => "frankentorch",
            Self::Frankenjax => "frankenjax",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerationReport {
    pub backend: AccelerationBackend,
    pub input_values: usize,
    pub normalized_confidences: bool,
    pub pre_mass: Option<f64>,
    pub post_mass: Option<f64>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub backend: BackendKind,
    pub transcript: String,
    pub language: Option<String>,
    pub segments: Vec<TranscriptionSegment>,
    pub acceleration: Option<AccelerationReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diarization: Option<DiarizationReport>,
    pub raw_output: Value,
    pub artifact_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEvent {
    pub seq: u64,
    pub ts_rfc3339: String,
    pub stage: String,
    pub code: String,
    pub message: String,
    pub payload: Value,
}

/// Deterministic replay envelope for regression drift detection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplayEnvelope {
    /// SHA-256 of the normalized WAV input bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_content_hash: Option<String>,
    /// Identity string for the backend command that produced output (e.g. "whisper-cli 1.7.2").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_identity: Option<String>,
    /// Version string reported by the backend command/runtime when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_version: Option<String>,
    /// SHA-256 of the raw backend output JSON payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_payload_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunReport {
    pub run_id: String,
    pub trace_id: String,
    pub started_at_rfc3339: String,
    pub finished_at_rfc3339: String,
    pub input_path: String,
    pub normalized_wav_path: String,
    pub request: TranscribeRequest,
    pub result: TranscriptionResult,
    pub events: Vec<RunEvent>,
    pub warnings: Vec<String>,
    pub evidence: Vec<Value>,
    /// Deterministic replay envelope for regression drift detection.
    #[serde(default)]
    pub replay: ReplayEnvelope,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub started_at_rfc3339: String,
    pub finished_at_rfc3339: String,
    pub backend: BackendKind,
    pub transcript_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamedRunEvent {
    pub run_id: String,
    pub event: RunEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredRunDetails {
    pub run_id: String,
    pub started_at_rfc3339: String,
    pub finished_at_rfc3339: String,
    pub backend: BackendKind,
    pub transcript: String,
    pub segments: Vec<TranscriptionSegment>,
    /// Complete typed diarization report recovered from the canonical
    /// `result_json` payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diarization: Option<DiarizationReport>,
    /// Privacy-safe projection provenance recovered from the canonical
    /// `result_json` payload.
    ///
    /// The rest of backend `raw_output` is deliberately not exposed from run
    /// history because it may contain internal model paths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_timeline: Option<Value>,
    pub events: Vec<RunEvent>,
    pub warnings: Vec<String>,
    pub acceleration: Option<AccelerationReport>,
    #[serde(default)]
    pub replay: ReplayEnvelope,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn backend_kind_serialization_round_trip() {
        for kind in [
            BackendKind::Auto,
            BackendKind::WhisperCpp,
            BackendKind::InsanelyFast,
            BackendKind::WhisperDiarization,
        ] {
            let serialized = serde_json::to_string(&kind).unwrap();
            let deserialized: BackendKind = serde_json::from_str(&serialized).unwrap();
            assert_eq!(kind, deserialized);
        }
    }

    #[test]
    fn backend_kind_as_str_matches_serde() {
        assert_eq!(BackendKind::Auto.as_str(), "auto");
        assert_eq!(BackendKind::WhisperCpp.as_str(), "whisper_cpp");
        assert_eq!(BackendKind::InsanelyFast.as_str(), "insanely_fast");
        assert_eq!(
            BackendKind::WhisperDiarization.as_str(),
            "whisper_diarization"
        );
    }

    #[test]
    fn output_format_serialization_round_trip() {
        for fmt in [
            OutputFormat::Txt,
            OutputFormat::Vtt,
            OutputFormat::Srt,
            OutputFormat::Csv,
            OutputFormat::Json,
            OutputFormat::JsonFull,
            OutputFormat::Lrc,
        ] {
            let serialized = serde_json::to_string(&fmt).unwrap();
            let deserialized: OutputFormat = serde_json::from_str(&serialized).unwrap();
            assert_eq!(fmt, deserialized);
        }
    }

    #[test]
    fn timestamp_level_serialization_round_trip() {
        for level in [TimestampLevel::Chunk, TimestampLevel::Word] {
            let serialized = serde_json::to_string(&level).unwrap();
            let deserialized: TimestampLevel = serde_json::from_str(&serialized).unwrap();
            assert_eq!(level, deserialized);
        }
    }

    #[test]
    fn acceleration_backend_as_str() {
        assert_eq!(AccelerationBackend::None.as_str(), "none");
        assert_eq!(AccelerationBackend::Frankentorch.as_str(), "frankentorch");
        assert_eq!(AccelerationBackend::Frankenjax.as_str(), "frankenjax");
    }

    #[test]
    fn backend_params_default_is_empty() {
        let bp = BackendParams::default();
        assert!(bp.output_formats.is_empty());
        assert!(bp.timestamp_level.is_none());
        assert!(bp.decoding.is_none());
        assert!(bp.vad.is_none());
        assert!(bp.acoustic_diarization.is_none());
        assert!(bp.diarization_config.is_none());
        assert!(bp.gpu_device.is_none());
        assert!(bp.flash_attention.is_none());
        assert!(!bp.no_timestamps);
        assert!(!bp.detect_language_only);
        assert!(bp.batch_size.is_none());
        assert!(!bp.split_on_word);
    }

    #[test]
    fn transcription_result_serialization_round_trip() {
        let result = TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: "hello world".to_owned(),
            language: Some("en".to_owned()),
            segments: vec![TranscriptionSegment {
                start_sec: Some(0.0),
                end_sec: Some(1.5),
                text: "hello world".to_owned(),
                speaker: Some("SPEAKER_00".to_owned()),
                confidence: Some(0.95),
            }],
            acceleration: None,
            diarization: None,
            raw_output: json!({"test": true}),
            artifact_paths: vec!["output.json".to_owned()],
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: TranscriptionResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.backend, BackendKind::WhisperCpp);
        assert_eq!(deserialized.transcript, "hello world");
        assert_eq!(deserialized.language.as_deref(), Some("en"));
        assert_eq!(deserialized.segments.len(), 1);
        assert_eq!(
            deserialized.segments[0].speaker.as_deref(),
            Some("SPEAKER_00")
        );
    }

    #[test]
    fn replay_envelope_default_has_all_none() {
        let envelope = ReplayEnvelope::default();
        assert!(envelope.input_content_hash.is_none());
        assert!(envelope.backend_identity.is_none());
        assert!(envelope.backend_version.is_none());
        assert!(envelope.output_payload_hash.is_none());
    }

    #[test]
    fn replay_envelope_skip_serializing_if_none() {
        let envelope = ReplayEnvelope::default();
        let json = serde_json::to_value(&envelope).unwrap();
        let obj = json.as_object().unwrap();
        assert!(
            obj.is_empty(),
            "default envelope should serialize empty: {obj:?}"
        );
    }

    #[test]
    fn transcribe_request_with_default_backend_params() {
        let request_json = json!({
            "input": {"kind": "file", "path": "test.wav"},
            "backend": "auto",
            "model": null,
            "language": "en",
            "translate": false,
            "diarize": false,
            "persist": true,
            "db_path": "db.sqlite3",
            "timeout_ms": null
        });

        let request: TranscribeRequest = serde_json::from_value(request_json).unwrap();
        assert_eq!(request.backend, BackendKind::Auto);
        assert!(request.backend_params.output_formats.is_empty());
    }

    #[test]
    fn input_source_file_variant_round_trip() {
        let source = InputSource::File {
            path: PathBuf::from("test.wav"),
        };
        let json = serde_json::to_value(&source).unwrap();
        assert_eq!(json["kind"], "file");
        let deserialized: InputSource = serde_json::from_value(json).unwrap();
        assert!(
            matches!(deserialized, InputSource::File { ref path } if path.as_os_str() == "test.wav")
        );
    }

    #[test]
    fn input_source_stdin_variant_round_trip() {
        let source = InputSource::Stdin {
            hint_extension: Some("wav".to_owned()),
        };
        let json = serde_json::to_value(&source).unwrap();
        assert_eq!(json["kind"], "stdin");
        let deserialized: InputSource = serde_json::from_value(json).unwrap();
        assert!(
            matches!(deserialized, InputSource::Stdin { hint_extension: Some(ext) } if ext == "wav")
        );
    }

    #[test]
    fn decoding_params_default_all_none() {
        let dp = DecodingParams::default();
        assert!(dp.best_of.is_none());
        assert!(dp.beam_size.is_none());
        assert!(dp.max_context.is_none());
        assert!(dp.max_segment_length.is_none());
        assert!(dp.temperature.is_none());
        assert!(dp.temperature_increment.is_none());
        assert!(dp.entropy_threshold.is_none());
        assert!(dp.logprob_threshold.is_none());
        assert!(dp.no_speech_threshold.is_none());
    }

    #[test]
    fn vad_params_default_all_none() {
        let vp = VadParams::default();
        assert!(vp.model_path.is_none());
        assert!(vp.threshold.is_none());
        assert!(vp.min_speech_duration_ms.is_none());
        assert!(vp.min_silence_duration_ms.is_none());
        assert!(vp.max_speech_duration_s.is_none());
        assert!(vp.speech_pad_ms.is_none());
        assert!(vp.samples_overlap.is_none());
    }

    #[test]
    fn input_source_microphone_variant_round_trip() {
        let source = InputSource::Microphone {
            seconds: 30,
            device: Some("hw:1,0".to_owned()),
            ffmpeg_format: Some("alsa".to_owned()),
            ffmpeg_source: Some("hw:1,0".to_owned()),
        };
        let json = serde_json::to_value(&source).unwrap();
        assert_eq!(json["kind"], "microphone");
        assert_eq!(json["seconds"], 30);
        assert_eq!(json["device"], "hw:1,0");
        let deserialized: InputSource = serde_json::from_value(json).unwrap();
        assert!(matches!(
            deserialized,
            InputSource::Microphone { seconds: 30, .. }
        ));
    }

    #[test]
    fn speaker_count_request_serialization_round_trip() {
        for request in [
            SpeakerCountRequest::Infer,
            SpeakerCountRequest::Range {
                minimum: 1,
                maximum: 5,
            },
            SpeakerCountRequest::HardConstraint { count: 3 },
            SpeakerCountRequest::Prior {
                bins: vec![
                    SpeakerCountPriorMass {
                        count: 2,
                        probability: 0.25,
                    },
                    SpeakerCountPriorMass {
                        count: 3,
                        probability: 0.75,
                    },
                ],
            },
        ] {
            let json = serde_json::to_string(&request).unwrap();
            let deserialized: SpeakerCountRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, request);
        }
    }

    #[test]
    fn diarization_config_serialization_round_trip() {
        let dc = DiarizationConfig {
            no_stem: true,
            whisper_model: Some("large-v2".to_owned()),
            suppress_numerals: true,
            device: Some("cuda:0".to_owned()),
            batch_size: Some(16),
        };
        let json = serde_json::to_string(&dc).unwrap();
        let deserialized: DiarizationConfig = serde_json::from_str(&json).unwrap();
        assert!(deserialized.no_stem);
        assert_eq!(deserialized.whisper_model.as_deref(), Some("large-v2"));
        assert!(deserialized.suppress_numerals);
        assert_eq!(deserialized.device.as_deref(), Some("cuda:0"));
        assert_eq!(deserialized.batch_size, Some(16));
    }

    #[test]
    fn run_event_serialization_round_trip() {
        let event = RunEvent {
            seq: 42,
            ts_rfc3339: "2025-01-15T10:30:00Z".to_owned(),
            stage: "backend".to_owned(),
            code: "ok".to_owned(),
            message: "completed".to_owned(),
            payload: json!({"elapsed_ms": 1234}),
        };
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: RunEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.seq, 42);
        assert_eq!(deserialized.stage, "backend");
    }

    #[test]
    fn acceleration_report_serialization_round_trip() {
        let report = AccelerationReport {
            backend: AccelerationBackend::Frankentorch,
            input_values: 100,
            normalized_confidences: true,
            pre_mass: Some(0.8),
            post_mass: Some(1.0),
            notes: vec!["normalized".to_owned()],
        };
        let json = serde_json::to_string(&report).unwrap();
        let deserialized: AccelerationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.backend, AccelerationBackend::Frankentorch);
        assert_eq!(deserialized.input_values, 100);
        assert!(deserialized.normalized_confidences);
    }

    #[test]
    fn backend_params_phase4_fields_default_to_none_false() {
        let bp = BackendParams::default();
        assert!(bp.threads.is_none());
        assert!(bp.processors.is_none());
        assert!(!bp.no_gpu);
        assert!(bp.prompt.is_none());
        assert!(!bp.carry_initial_prompt);
        assert!(!bp.no_fallback);
        assert!(!bp.suppress_nst);
        assert!(bp.offset_ms.is_none());
        assert!(bp.duration_ms.is_none());
        assert!(bp.audio_ctx.is_none());
        assert!(bp.word_threshold.is_none());
        assert!(bp.suppress_regex.is_none());
        assert!(bp.insanely_fast_hf_token.is_none());
        assert!(bp.insanely_fast_transcript_path.is_none());
    }

    #[test]
    fn backend_params_phase4_serde_round_trip() {
        let bp = BackendParams {
            threads: Some(4),
            processors: Some(2),
            no_gpu: true,
            prompt: Some("test prompt".to_owned()),
            carry_initial_prompt: true,
            no_fallback: true,
            suppress_nst: true,
            offset_ms: Some(5000),
            duration_ms: Some(30000),
            audio_ctx: Some(0),
            word_threshold: Some(0.25),
            suppress_regex: Some(r"\[.*\]".to_owned()),
            ..BackendParams::default()
        };
        let json = serde_json::to_string(&bp).unwrap();
        let parsed: BackendParams = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.threads, Some(4));
        assert_eq!(parsed.processors, Some(2));
        assert!(parsed.no_gpu);
        assert_eq!(parsed.prompt.as_deref(), Some("test prompt"));
        assert!(parsed.carry_initial_prompt);
        assert!(parsed.no_fallback);
        assert!(parsed.suppress_nst);
        assert_eq!(parsed.offset_ms, Some(5000));
        assert_eq!(parsed.duration_ms, Some(30000));
        assert_eq!(parsed.audio_ctx, Some(0));
        assert_eq!(parsed.word_threshold, Some(0.25));
        assert_eq!(parsed.suppress_regex.as_deref(), Some(r"\[.*\]"));
    }

    // --- ReplayEnvelope serde edge cases ---

    #[test]
    fn replay_envelope_populated_round_trip() {
        let envelope = ReplayEnvelope {
            input_content_hash: Some("abc123".to_owned()),
            backend_identity: Some("whisper-cli".to_owned()),
            backend_version: Some("1.7.2".to_owned()),
            output_payload_hash: Some("def456".to_owned()),
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let parsed: ReplayEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.input_content_hash.as_deref(), Some("abc123"));
        assert_eq!(parsed.backend_identity.as_deref(), Some("whisper-cli"));
        assert_eq!(parsed.backend_version.as_deref(), Some("1.7.2"));
        assert_eq!(parsed.output_payload_hash.as_deref(), Some("def456"));
    }

    #[test]
    fn replay_envelope_deserializes_from_empty_object() {
        let parsed: ReplayEnvelope = serde_json::from_str("{}").unwrap();
        assert!(parsed.input_content_hash.is_none());
        assert!(parsed.backend_identity.is_none());
        assert!(parsed.backend_version.is_none());
        assert!(parsed.output_payload_hash.is_none());
    }

    // --- TranscriptionResult edge cases ---

    #[test]
    fn transcription_result_empty_transcript_and_segments() {
        let result = TranscriptionResult {
            backend: BackendKind::Auto,
            transcript: String::new(),
            language: None,
            segments: vec![],
            acceleration: None,
            diarization: None,
            raw_output: json!(null),
            artifact_paths: vec![],
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: TranscriptionResult = serde_json::from_str(&json).unwrap();
        assert!(parsed.transcript.is_empty());
        assert!(parsed.segments.is_empty());
        assert!(parsed.language.is_none());
        assert!(parsed.acceleration.is_none());
    }

    // --- AccelerationReport edge cases ---

    #[test]
    fn acceleration_report_none_backend_round_trip() {
        let report = AccelerationReport {
            backend: AccelerationBackend::None,
            input_values: 0,
            normalized_confidences: false,
            pre_mass: None,
            post_mass: None,
            notes: vec![],
        };
        let json = serde_json::to_string(&report).unwrap();
        let parsed: AccelerationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.backend, AccelerationBackend::None);
        assert_eq!(parsed.input_values, 0);
        assert!(!parsed.normalized_confidences);
        assert!(parsed.notes.is_empty());
    }

    // --- EngineCapabilities ---

    #[test]
    fn engine_capabilities_serde_round_trip() {
        let caps = EngineCapabilities {
            supports_diarization: true,
            supports_translation: false,
            supports_word_timestamps: true,
            supports_gpu: true,
            supports_streaming: false,
        };
        let json = serde_json::to_string(&caps).unwrap();
        let parsed: EngineCapabilities = serde_json::from_str(&json).unwrap();
        assert!(parsed.supports_diarization);
        assert!(!parsed.supports_translation);
        assert!(parsed.supports_word_timestamps);
        assert!(parsed.supports_gpu);
        assert!(!parsed.supports_streaming);
    }

    // --- RunSummary ---

    #[test]
    fn run_summary_serde_round_trip() {
        let summary = RunSummary {
            run_id: "run-42".to_owned(),
            started_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-01-01T00:01:00Z".to_owned(),
            backend: BackendKind::InsanelyFast,
            transcript_preview: "hello world...".to_owned(),
        };
        let json = serde_json::to_string(&summary).unwrap();
        let parsed: RunSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.run_id, "run-42");
        assert_eq!(parsed.backend, BackendKind::InsanelyFast);
        assert_eq!(parsed.transcript_preview, "hello world...");
    }

    // --- StreamedRunEvent ---

    #[test]
    fn streamed_run_event_serde_round_trip() {
        let sre = StreamedRunEvent {
            run_id: "run-99".to_owned(),
            event: RunEvent {
                seq: 1,
                ts_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
                stage: "ingest".to_owned(),
                code: "ingest.ok".to_owned(),
                message: "done".to_owned(),
                payload: json!({}),
            },
        };
        let json = serde_json::to_string(&sre).unwrap();
        let parsed: StreamedRunEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.run_id, "run-99");
        assert_eq!(parsed.event.code, "ingest.ok");
    }

    // --- VadParams and DecodingParams serde ---

    #[test]
    fn vad_params_populated_round_trip() {
        let vp = VadParams {
            model_path: Some(PathBuf::from("/models/vad.bin")),
            threshold: Some(0.5),
            min_speech_duration_ms: Some(250),
            min_silence_duration_ms: Some(100),
            max_speech_duration_s: Some(30.0),
            speech_pad_ms: Some(200),
            samples_overlap: Some(0.1),
        };
        let json = serde_json::to_string(&vp).unwrap();
        let parsed: VadParams = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.threshold, Some(0.5));
        assert_eq!(parsed.min_speech_duration_ms, Some(250));
        assert_eq!(
            parsed.model_path.as_deref(),
            Some(std::path::Path::new("/models/vad.bin"))
        );
    }

    #[test]
    fn decoding_params_populated_round_trip() {
        let dp = DecodingParams {
            best_of: Some(5),
            beam_size: Some(3),
            max_context: Some(-1),
            max_segment_length: Some(50),
            temperature: Some(0.0),
            temperature_increment: Some(0.2),
            entropy_threshold: Some(2.4),
            logprob_threshold: Some(-1.0),
            no_speech_threshold: Some(0.6),
        };
        let json = serde_json::to_string(&dp).unwrap();
        let parsed: DecodingParams = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.best_of, Some(5));
        assert_eq!(parsed.max_context, Some(-1));
        assert_eq!(parsed.temperature, Some(0.0));
    }

    // --- DiarizationConfig defaults ---

    #[test]
    fn diarization_config_default_all_false_none() {
        let dc = DiarizationConfig::default();
        assert!(!dc.no_stem);
        assert!(dc.whisper_model.is_none());
        assert!(!dc.suppress_numerals);
        assert!(dc.device.is_none());
        assert!(dc.batch_size.is_none());
    }

    #[test]
    fn backend_params_phase4_backward_compatible_deserialization() {
        // JSON without Phase 4 fields should still parse successfully.
        let json = r#"{"output_formats":[],"no_timestamps":false,"detect_language_only":false,"split_on_word":false}"#;
        let parsed: BackendParams = serde_json::from_str(json).unwrap();
        assert!(parsed.threads.is_none());
        assert!(!parsed.no_gpu);
        assert!(parsed.prompt.is_none());
        assert!(!parsed.carry_initial_prompt);
    }

    // --- StoredRunDetails ---

    #[test]
    fn stored_run_details_serde_round_trip() {
        let details = StoredRunDetails {
            run_id: "run-777".to_owned(),
            started_at_rfc3339: "2026-01-15T10:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-01-15T10:01:30Z".to_owned(),
            backend: BackendKind::WhisperDiarization,
            transcript: "hello world from diarization".to_owned(),
            segments: vec![
                TranscriptionSegment {
                    start_sec: Some(0.0),
                    end_sec: Some(1.0),
                    text: "hello".to_owned(),
                    speaker: Some("SPEAKER_00".to_owned()),
                    confidence: Some(0.99),
                },
                TranscriptionSegment {
                    start_sec: Some(1.0),
                    end_sec: Some(2.5),
                    text: "world from diarization".to_owned(),
                    speaker: Some("SPEAKER_01".to_owned()),
                    confidence: Some(0.85),
                },
            ],
            diarization: None,
            projection_timeline: Some(json!({
                "schema_version": "dtw-projection-v2",
                "word_aligned_safe": true
            })),
            events: vec![RunEvent {
                seq: 0,
                ts_rfc3339: "2026-01-15T10:00:00Z".to_owned(),
                stage: "ingest".to_owned(),
                code: "ok".to_owned(),
                message: "ingested".to_owned(),
                payload: json!({}),
            }],
            warnings: vec!["low confidence segment".to_owned()],
            acceleration: Some(AccelerationReport {
                backend: AccelerationBackend::Frankenjax,
                input_values: 50,
                normalized_confidences: true,
                pre_mass: Some(0.7),
                post_mass: Some(1.0),
                notes: vec!["jax accelerated".to_owned()],
            }),
            replay: ReplayEnvelope {
                input_content_hash: Some("sha256-abc".to_owned()),
                backend_identity: Some("whisper-diarization".to_owned()),
                backend_version: Some("0.0.15".to_owned()),
                output_payload_hash: Some("sha256-def".to_owned()),
            },
        };
        let json = serde_json::to_string(&details).unwrap();
        let parsed: StoredRunDetails = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.run_id, "run-777");
        assert_eq!(parsed.backend, BackendKind::WhisperDiarization);
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.segments[1].speaker.as_deref(), Some("SPEAKER_01"));
        assert_eq!(
            parsed
                .projection_timeline
                .as_ref()
                .expect("projection timeline")["schema_version"],
            "dtw-projection-v2"
        );
        assert_eq!(parsed.events.len(), 1);
        assert_eq!(parsed.warnings.len(), 1);
        assert!(parsed.acceleration.is_some());
        assert_eq!(
            parsed.replay.backend_identity.as_deref(),
            Some("whisper-diarization")
        );
    }

    #[test]
    fn stored_run_details_minimal_round_trip() {
        let details = StoredRunDetails {
            run_id: "run-min".to_owned(),
            started_at_rfc3339: "2026-01-01T00:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-01-01T00:00:01Z".to_owned(),
            backend: BackendKind::Auto,
            transcript: String::new(),
            segments: vec![],
            diarization: None,
            projection_timeline: None,
            events: vec![],
            warnings: vec![],
            acceleration: None,
            replay: ReplayEnvelope::default(),
        };
        let json = serde_json::to_string(&details).unwrap();
        let parsed: StoredRunDetails = serde_json::from_str(&json).unwrap();
        assert!(parsed.transcript.is_empty());
        assert!(parsed.segments.is_empty());
        assert!(parsed.diarization.is_none());
        assert!(parsed.projection_timeline.is_none());
        assert!(parsed.events.is_empty());
        assert!(parsed.warnings.is_empty());
        assert!(parsed.acceleration.is_none());
    }

    // --- RunReport full round-trip ---

    fn make_test_run_report() -> RunReport {
        RunReport {
            run_id: "run-full".to_owned(),
            trace_id: "trace-abc123".to_owned(),
            started_at_rfc3339: "2026-02-01T12:00:00Z".to_owned(),
            finished_at_rfc3339: "2026-02-01T12:01:00Z".to_owned(),
            input_path: "/tmp/input.wav".to_owned(),
            normalized_wav_path: "/tmp/normalized.wav".to_owned(),
            request: TranscribeRequest {
                input: InputSource::File {
                    path: PathBuf::from("/tmp/input.wav"),
                },
                backend: BackendKind::WhisperCpp,
                model: Some("large-v3".to_owned()),
                language: Some("en".to_owned()),
                translate: false,
                diarize: false,
                persist: true,
                db_path: PathBuf::from("db.sqlite3"),
                timeout_ms: Some(120_000),
                backend_params: BackendParams::default(),
            },
            result: TranscriptionResult {
                backend: BackendKind::WhisperCpp,
                transcript: "hello world".to_owned(),
                language: Some("en".to_owned()),
                segments: vec![TranscriptionSegment {
                    start_sec: Some(0.0),
                    end_sec: Some(1.5),
                    text: "hello world".to_owned(),
                    speaker: None,
                    confidence: Some(0.97),
                }],
                acceleration: None,
                diarization: None,
                raw_output: json!({"text": "hello world"}),
                artifact_paths: vec![],
            },
            events: vec![
                RunEvent {
                    seq: 0,
                    ts_rfc3339: "2026-02-01T12:00:00Z".to_owned(),
                    stage: "ingest".to_owned(),
                    code: "ok".to_owned(),
                    message: "ingested".to_owned(),
                    payload: json!({}),
                },
                RunEvent {
                    seq: 1,
                    ts_rfc3339: "2026-02-01T12:00:30Z".to_owned(),
                    stage: "backend".to_owned(),
                    code: "ok".to_owned(),
                    message: "transcribed".to_owned(),
                    payload: json!({"elapsed_ms": 30000}),
                },
            ],
            warnings: vec![],
            evidence: vec![json!({"decision": "whisper_cpp", "score": 0.9})],
            replay: ReplayEnvelope {
                input_content_hash: Some("sha256-input".to_owned()),
                backend_identity: Some("whisper-cli".to_owned()),
                backend_version: Some("1.7.2".to_owned()),
                output_payload_hash: Some("sha256-output".to_owned()),
            },
        }
    }

    #[test]
    fn run_report_full_serde_round_trip() {
        let report = make_test_run_report();
        let json = serde_json::to_string(&report).unwrap();
        let parsed: RunReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.run_id, "run-full");
        assert_eq!(parsed.trace_id, "trace-abc123");
        assert_eq!(parsed.events.len(), 2);
        assert_eq!(parsed.evidence.len(), 1);
        assert_eq!(parsed.result.transcript, "hello world");
        assert_eq!(
            parsed.replay.input_content_hash.as_deref(),
            Some("sha256-input")
        );
    }

    #[test]
    fn run_report_preserves_evidence_through_serde() {
        let mut report = make_test_run_report();
        report.evidence = vec![
            json!({"contract": "backend_selection", "action": "whisper_cpp"}),
            json!({"contract": "retry_decision", "action": "no_retry"}),
        ];
        let json = serde_json::to_string(&report).unwrap();
        let parsed: RunReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.evidence.len(), 2);
        assert_eq!(parsed.evidence[0]["contract"], "backend_selection");
        assert_eq!(parsed.evidence[1]["action"], "no_retry");
    }

    #[test]
    fn run_report_empty_evidence_round_trips() {
        let mut report = make_test_run_report();
        report.evidence = vec![];
        let json = serde_json::to_string(&report).unwrap();
        let parsed: RunReport = serde_json::from_str(&json).unwrap();
        assert!(parsed.evidence.is_empty());
    }

    // --- BackendKind as_str consistency with serde ---

    #[test]
    fn backend_kind_as_str_matches_serde_serialized_value() {
        for kind in [
            BackendKind::Auto,
            BackendKind::WhisperCpp,
            BackendKind::InsanelyFast,
            BackendKind::WhisperDiarization,
        ] {
            let serialized = serde_json::to_string(&kind).unwrap();
            // serde serializes with quotes: "auto", "whisper_cpp", etc.
            let expected = format!("\"{}\"", kind.as_str());
            assert_eq!(
                serialized, expected,
                "as_str() and serde disagree for {kind:?}"
            );
        }
    }

    // --- AccelerationBackend as_str consistency with serde ---

    #[test]
    fn acceleration_backend_as_str_matches_serde() {
        for ab in [
            AccelerationBackend::None,
            AccelerationBackend::Frankentorch,
            AccelerationBackend::Frankenjax,
        ] {
            let serialized = serde_json::to_string(&ab).unwrap();
            let expected = format!("\"{}\"", ab.as_str());
            assert_eq!(
                serialized, expected,
                "as_str() and serde disagree for {ab:?}"
            );
        }
    }

    // --- InputSource::Microphone edge cases ---

    #[test]
    fn input_source_microphone_all_optional_fields_none() {
        let source = InputSource::Microphone {
            seconds: 10,
            device: None,
            ffmpeg_format: None,
            ffmpeg_source: None,
        };
        let json = serde_json::to_value(&source).unwrap();
        assert_eq!(json["kind"], "microphone");
        assert_eq!(json["seconds"], 10);
        assert!(json["device"].is_null());
        assert!(json["ffmpeg_format"].is_null());
        assert!(json["ffmpeg_source"].is_null());
        let deserialized: InputSource = serde_json::from_value(json).unwrap();
        let input = deserialized;
        assert!(
            matches!(
                &input,
                InputSource::Microphone {
                    seconds: _,
                    device: _,
                    ffmpeg_format: _,
                    ffmpeg_source: _
                }
            ),
            "expected Microphone, got {input:?}"
        );
        if let InputSource::Microphone {
            seconds,
            device,
            ffmpeg_format,
            ffmpeg_source,
        } = input
        {
            assert_eq!(seconds, 10);
            assert!(device.is_none());
            assert!(ffmpeg_format.is_none());
            assert!(ffmpeg_source.is_none());
        }
    }

    #[test]
    fn input_source_stdin_no_hint_round_trip() {
        let source = InputSource::Stdin {
            hint_extension: None,
        };
        let json = serde_json::to_value(&source).unwrap();
        assert_eq!(json["kind"], "stdin");
        assert!(json["hint_extension"].is_null());
        let deserialized: InputSource = serde_json::from_value(json).unwrap();
        assert!(matches!(
            deserialized,
            InputSource::Stdin {
                hint_extension: None
            }
        ));
    }

    // --- TranscriptionSegment edge cases ---

    #[test]
    fn transcription_segment_all_optional_none() {
        let seg = TranscriptionSegment {
            start_sec: None,
            end_sec: None,
            text: "no timestamps".to_owned(),
            speaker: None,
            confidence: None,
        };
        let json = serde_json::to_string(&seg).unwrap();
        let parsed: TranscriptionSegment = serde_json::from_str(&json).unwrap();
        assert!(parsed.start_sec.is_none());
        assert!(parsed.end_sec.is_none());
        assert!(parsed.speaker.is_none());
        assert!(parsed.confidence.is_none());
        assert_eq!(parsed.text, "no timestamps");
    }

    // --- TranscribeRequest full round-trip ---

    #[test]
    fn transcribe_request_full_round_trip() {
        let request = TranscribeRequest {
            input: InputSource::Microphone {
                seconds: 60,
                device: Some("mic1".to_owned()),
                ffmpeg_format: None,
                ffmpeg_source: None,
            },
            backend: BackendKind::InsanelyFast,
            model: Some("large-v3".to_owned()),
            language: Some("ja".to_owned()),
            translate: true,
            diarize: true,
            persist: false,
            db_path: PathBuf::from("/custom/db.sqlite3"),
            timeout_ms: Some(300_000),
            backend_params: BackendParams {
                gpu_device: Some("cuda:0".to_owned()),
                batch_size: Some(24),
                flash_attention: Some(true),
                timestamp_level: Some(TimestampLevel::Word),
                ..BackendParams::default()
            },
        };
        let json = serde_json::to_string(&request).unwrap();
        let parsed: TranscribeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.backend, BackendKind::InsanelyFast);
        assert!(parsed.translate);
        assert!(parsed.diarize);
        assert!(!parsed.persist);
        assert_eq!(parsed.timeout_ms, Some(300_000));
        assert_eq!(parsed.backend_params.gpu_device.as_deref(), Some("cuda:0"));
        assert_eq!(parsed.backend_params.batch_size, Some(24));
        assert_eq!(parsed.backend_params.flash_attention, Some(true));
        assert_eq!(
            parsed.backend_params.timestamp_level,
            Some(TimestampLevel::Word)
        );
        assert_eq!(parsed.language.as_deref(), Some("ja"));
    }

    // --- AccelerationBackend serde round-trip ---

    #[test]
    fn acceleration_backend_serde_round_trip() {
        for ab in [
            AccelerationBackend::None,
            AccelerationBackend::Frankentorch,
            AccelerationBackend::Frankenjax,
        ] {
            let serialized = serde_json::to_string(&ab).unwrap();
            let deserialized: AccelerationBackend = serde_json::from_str(&serialized).unwrap();
            assert_eq!(ab, deserialized);
        }
    }

    // --- SpeakerCountRequest defaults ---

    #[test]
    fn speaker_count_request_defaults_to_infer() {
        assert_eq!(SpeakerCountRequest::default(), SpeakerCountRequest::Infer);
    }

    #[test]
    fn transcription_segment_full_round_trip() {
        let seg = TranscriptionSegment {
            start_sec: Some(1.5),
            end_sec: Some(3.75),
            text: "hello world".to_owned(),
            speaker: Some("SPEAKER_00".to_owned()),
            confidence: Some(0.95),
        };
        let json = serde_json::to_string(&seg).unwrap();
        let parsed: TranscriptionSegment = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.start_sec, Some(1.5));
        assert_eq!(parsed.end_sec, Some(3.75));
        assert_eq!(parsed.text, "hello world");
        assert_eq!(parsed.speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(parsed.confidence, Some(0.95));
    }

    #[test]
    fn backend_params_with_output_formats_round_trip() {
        let params = BackendParams {
            output_formats: vec![OutputFormat::Srt, OutputFormat::Vtt, OutputFormat::Lrc],
            ..BackendParams::default()
        };
        let json = serde_json::to_string(&params).unwrap();
        assert!(
            !json.contains("hf_example_token"),
            "hf token should be redacted from serialized params"
        );
        assert!(
            !json.contains("insanely_fast_hf_token"),
            "hf token field should be omitted from serialized params"
        );
        let parsed: BackendParams = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.output_formats.len(), 3);
        assert_eq!(parsed.output_formats[0], OutputFormat::Srt);
        assert_eq!(parsed.output_formats[1], OutputFormat::Vtt);
        assert_eq!(parsed.output_formats[2], OutputFormat::Lrc);
    }

    #[test]
    fn backend_params_with_all_populated_round_trip() {
        let params = BackendParams {
            output_formats: vec![OutputFormat::Txt],
            timestamp_level: Some(TimestampLevel::Word),
            decoding: Some(DecodingParams {
                best_of: Some(5),
                beam_size: Some(3),
                ..DecodingParams::default()
            }),
            vad: Some(VadParams {
                threshold: Some(0.5),
                ..VadParams::default()
            }),
            acoustic_diarization: Some(DiarizationRequest {
                speaker_count: SpeakerCountRequest::HardConstraint { count: 4 },
                ..DiarizationRequest::default()
            }),
            diarization_config: Some(DiarizationConfig {
                no_stem: true,
                whisper_model: Some("large".to_owned()),
                suppress_numerals: true,
                device: Some("cuda:0".to_owned()),
                batch_size: Some(16),
            }),
            gpu_device: Some("cuda:0".to_owned()),
            flash_attention: Some(true),
            insanely_fast_hf_token: Some("hf_example_token".to_owned()),
            insanely_fast_transcript_path: Some(PathBuf::from("artifacts/insanely-fast.json")),
            no_timestamps: true,
            detect_language_only: true,
            batch_size: Some(16),
            split_on_word: true,
            threads: Some(8),
            processors: Some(2),
            no_gpu: true,
            prompt: Some("medical".to_owned()),
            carry_initial_prompt: true,
            no_fallback: true,
            suppress_nst: true,
            offset_ms: Some(5000),
            duration_ms: Some(60000),
            audio_ctx: Some(0),
            word_threshold: Some(0.01),
            suppress_regex: Some(r"\[.*\]".to_owned()),
            ..BackendParams::default()
        };
        let json = serde_json::to_string(&params).unwrap();
        let parsed: BackendParams = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.output_formats.len(), 1);
        assert_eq!(parsed.timestamp_level, Some(TimestampLevel::Word));
        assert!(parsed.decoding.is_some());
        assert!(parsed.vad.is_some());
        assert!(matches!(
            parsed
                .acoustic_diarization
                .as_ref()
                .map(|request| &request.speaker_count),
            Some(SpeakerCountRequest::HardConstraint { count: 4 })
        ));
        assert!(parsed.diarization_config.is_some());
        assert_eq!(parsed.gpu_device.as_deref(), Some("cuda:0"));
        assert_eq!(parsed.flash_attention, Some(true));
        assert!(
            parsed.insanely_fast_hf_token.is_none(),
            "hf token should not round-trip through serialized params"
        );
        assert_eq!(
            parsed.insanely_fast_transcript_path.as_deref(),
            Some(PathBuf::from("artifacts/insanely-fast.json").as_path())
        );
        assert!(parsed.no_timestamps);
        assert!(parsed.detect_language_only);
        assert!(parsed.split_on_word);
        assert_eq!(parsed.threads, Some(8));
        assert_eq!(parsed.processors, Some(2));
        assert!(parsed.no_gpu);
        assert_eq!(parsed.prompt.as_deref(), Some("medical"));
        assert!(parsed.carry_initial_prompt);
        assert!(parsed.no_fallback);
        assert!(parsed.suppress_nst);
        assert_eq!(parsed.offset_ms, Some(5000));
        assert_eq!(parsed.duration_ms, Some(60000));
        assert_eq!(parsed.audio_ctx, Some(0));
        assert_eq!(parsed.word_threshold, Some(0.01));
        assert_eq!(parsed.suppress_regex.as_deref(), Some(r"\[.*\]"));
    }

    #[test]
    fn run_event_with_complex_payload_round_trip() {
        let event = RunEvent {
            seq: 42,
            ts_rfc3339: "2026-02-22T12:34:56Z".to_owned(),
            stage: "backend".to_owned(),
            code: "backend.selected".to_owned(),
            message: "chose whisper_cpp via Bayesian selection".to_owned(),
            payload: json!({
                "posterior": [0.7, 0.2, 0.1],
                "action": "try_whisper_cpp",
                "nested": {"key": "value", "arr": [1, 2, 3]}
            }),
        };
        let json = serde_json::to_string(&event).unwrap();
        let parsed: RunEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.seq, 42);
        assert_eq!(parsed.stage, "backend");
        assert_eq!(parsed.payload["action"], "try_whisper_cpp");
        assert_eq!(parsed.payload["nested"]["arr"][2], 3);
    }

    #[test]
    fn replay_envelope_partial_fields_round_trip() {
        let envelope = ReplayEnvelope {
            input_content_hash: Some("abc".to_owned()),
            backend_identity: None,
            backend_version: Some("1.0".to_owned()),
            output_payload_hash: None,
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let parsed: ReplayEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.input_content_hash.as_deref(), Some("abc"));
        assert!(parsed.backend_identity.is_none());
        assert_eq!(parsed.backend_version.as_deref(), Some("1.0"));
        assert!(parsed.output_payload_hash.is_none());
    }

    #[test]
    fn output_format_all_variants_as_str() {
        let variants = [
            (OutputFormat::Txt, "txt"),
            (OutputFormat::Vtt, "vtt"),
            (OutputFormat::Srt, "srt"),
            (OutputFormat::Csv, "csv"),
            (OutputFormat::Json, "json"),
            (OutputFormat::JsonFull, "json_full"),
            (OutputFormat::Lrc, "lrc"),
        ];
        for (fmt, expected) in variants {
            let serialized = serde_json::to_string(&fmt).unwrap();
            assert_eq!(
                serialized,
                format!("\"{expected}\""),
                "OutputFormat serde for {fmt:?}"
            );
        }
    }

    #[test]
    fn timestamp_level_chunk_round_trip() {
        let level = TimestampLevel::Chunk;
        let json = serde_json::to_string(&level).unwrap();
        let parsed: TimestampLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, TimestampLevel::Chunk);
    }

    #[test]
    fn unicode_in_all_string_fields_round_trips() {
        let seg = TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(1.0),
            text: "日本語のテスト 🎤".to_owned(),
            speaker: Some("話者_01".to_owned()),
            confidence: Some(0.88),
        };
        let json = serde_json::to_string(&seg).unwrap();
        let parsed: TranscriptionSegment = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.text, "日本語のテスト 🎤");
        assert_eq!(parsed.speaker.as_deref(), Some("話者_01"));
    }

    #[test]
    fn run_report_backward_compat_without_replay_field() {
        // JSON without the `replay` field should deserialize with default.
        let json = json!({
            "run_id": "run-old",
            "trace_id": "trace-old",
            "started_at_rfc3339": "2026-01-01T00:00:00Z",
            "finished_at_rfc3339": "2026-01-01T00:00:01Z",
            "input_path": "in.wav",
            "normalized_wav_path": "norm.wav",
            "request": {
                "input": {"kind": "file", "path": "in.wav"},
                "backend": "auto",
                "model": null,
                "language": null,
                "translate": false,
                "diarize": false,
                "persist": false,
                "db_path": "db.sqlite3",
                "timeout_ms": null
            },
            "result": {
                "backend": "whisper_cpp",
                "transcript": "test",
                "language": null,
                "segments": [],
                "acceleration": null,
                "raw_output": {},
                "artifact_paths": []
            },
            "events": [],
            "warnings": [],
            "evidence": []
        });
        let parsed: RunReport = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.run_id, "run-old");
        assert!(parsed.replay.input_content_hash.is_none());
        assert!(parsed.replay.backend_identity.is_none());
    }

    #[test]
    fn stored_run_details_backward_compat_without_replay_field() {
        let json = json!({
            "run_id": "run-legacy",
            "started_at_rfc3339": "2026-01-01T00:00:00Z",
            "finished_at_rfc3339": "2026-01-01T00:00:01Z",
            "backend": "insanely_fast",
            "transcript": "hello",
            "segments": [],
            "events": [],
            "warnings": [],
            "acceleration": null
        });
        let parsed: StoredRunDetails = serde_json::from_value(json).unwrap();
        assert_eq!(parsed.run_id, "run-legacy");
        assert_eq!(parsed.backend, BackendKind::InsanelyFast);
        assert!(parsed.replay.input_content_hash.is_none());
    }

    #[test]
    fn transcription_result_with_acceleration_round_trip() {
        let result = TranscriptionResult {
            backend: BackendKind::WhisperCpp,
            transcript: "accelerated".to_owned(),
            language: Some("en".to_owned()),
            segments: vec![],
            acceleration: Some(AccelerationReport {
                backend: AccelerationBackend::Frankenjax,
                input_values: 50,
                normalized_confidences: false,
                pre_mass: None,
                post_mass: Some(0.99),
                notes: vec!["jax".to_owned(), "fast".to_owned()],
            }),
            diarization: None,
            raw_output: json!({}),
            artifact_paths: vec!["a.json".to_owned(), "b.srt".to_owned()],
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: TranscriptionResult = serde_json::from_str(&json).unwrap();
        let accel = parsed.acceleration.expect("should have acceleration");
        assert_eq!(accel.backend, AccelerationBackend::Frankenjax);
        assert_eq!(accel.input_values, 50);
        assert!(!accel.normalized_confidences);
        assert!(accel.pre_mass.is_none());
        assert_eq!(accel.post_mass, Some(0.99));
        assert_eq!(accel.notes.len(), 2);
        assert_eq!(parsed.artifact_paths.len(), 2);
    }

    #[test]
    fn decoding_params_extreme_f32_values_round_trip() {
        let dp = DecodingParams {
            best_of: Some(u32::MAX),
            beam_size: Some(0),
            max_context: Some(i32::MIN),
            max_segment_length: Some(u32::MAX),
            temperature: Some(f32::MIN_POSITIVE),
            temperature_increment: Some(f32::MAX),
            entropy_threshold: Some(f32::NEG_INFINITY),
            logprob_threshold: Some(f32::INFINITY),
            no_speech_threshold: Some(0.0),
        };
        let json = serde_json::to_string(&dp).unwrap();
        let parsed: DecodingParams = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.best_of, Some(u32::MAX));
        assert_eq!(parsed.max_context, Some(i32::MIN));
        assert_eq!(parsed.temperature, Some(f32::MIN_POSITIVE));
    }

    #[test]
    fn run_report_with_many_warnings_round_trips() {
        let mut report = make_test_run_report();
        report.warnings = (0..100).map(|i| format!("warning {i}")).collect();
        let json = serde_json::to_string(&report).unwrap();
        let parsed: RunReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.warnings.len(), 100);
        assert_eq!(parsed.warnings[0], "warning 0");
        assert_eq!(parsed.warnings[99], "warning 99");
    }

    #[test]
    fn vad_params_extreme_values_round_trip() {
        let vp = VadParams {
            model_path: Some(PathBuf::from("/a/very/long/path/to/model.bin")),
            threshold: Some(0.0),
            min_speech_duration_ms: Some(0),
            min_silence_duration_ms: Some(u32::MAX),
            max_speech_duration_s: Some(f32::MAX),
            speech_pad_ms: Some(0),
            samples_overlap: Some(1.0),
        };
        let json = serde_json::to_string(&vp).unwrap();
        let parsed: VadParams = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.threshold, Some(0.0));
        assert_eq!(parsed.min_silence_duration_ms, Some(u32::MAX));
        assert_eq!(parsed.samples_overlap, Some(1.0));
    }

    #[test]
    fn input_source_file_with_special_path_characters_round_trips() {
        let paths = [
            "/tmp/données/résultat.wav",
            "/path/with spaces/file name.wav",
            "/dir/ファイル/音声.wav",
            "/backslash\\in\\path.wav",
            "/dots/../../../etc/file.wav",
        ];
        for p in paths {
            let source = InputSource::File {
                path: PathBuf::from(p),
            };
            let json = serde_json::to_string(&source).unwrap();
            let parsed: InputSource = serde_json::from_str(&json).unwrap();
            let input = parsed;
            assert!(
                matches!(&input, InputSource::File { .. }),
                "expected File variant, got {input:?}"
            );
            if let InputSource::File { path } = input {
                assert_eq!(path, PathBuf::from(p), "path: {p}");
            } else {
                return;
            }
        }
    }

    #[test]
    fn raw_output_with_diverse_json_structures_round_trips() {
        let payloads = [
            json!(null),
            json!([1, "two", null, [3, 4]]),
            json!({"nested": {"deep": {"value": 42, "arr": [true, false]}}}),
            json!("a plain string"),
            json!(1.2345),
            json!(true),
        ];
        for payload in payloads {
            let result = TranscriptionResult {
                backend: BackendKind::Auto,
                transcript: String::new(),
                language: None,
                segments: vec![],
                acceleration: None,
                diarization: None,
                raw_output: payload.clone(),
                artifact_paths: vec![],
            };
            let json_str = serde_json::to_string(&result).unwrap();
            let parsed: TranscriptionResult = serde_json::from_str(&json_str).unwrap();
            assert_eq!(parsed.raw_output, payload, "payload: {payload}");
        }
    }

    #[test]
    fn malformed_json_fails_to_deserialize_backend_kind() {
        let bad_inputs = [
            r#""unknown_backend""#,
            r#""WHISPER_CPP""#,
            r#"42"#,
            r#"null"#,
        ];
        for input in bad_inputs {
            let result = serde_json::from_str::<BackendKind>(input);
            assert!(result.is_err(), "should reject: {input}");
        }
    }

    #[test]
    fn malformed_json_fails_to_deserialize_input_source() {
        // Missing required `kind` tag
        let no_kind = r#"{"path": "test.wav"}"#;
        assert!(serde_json::from_str::<InputSource>(no_kind).is_err());

        // Invalid `kind` value
        let bad_kind = r#"{"kind": "url", "path": "http://example.com"}"#;
        assert!(serde_json::from_str::<InputSource>(bad_kind).is_err());

        // File variant missing required `path` field
        let no_path = r#"{"kind": "file"}"#;
        assert!(serde_json::from_str::<InputSource>(no_path).is_err());
    }

    #[test]
    fn vad_params_model_path_with_unicode_round_trips() {
        let vp = VadParams {
            model_path: Some(PathBuf::from("/modèles/données/vad_模型.onnx")),
            ..VadParams::default()
        };
        let json = serde_json::to_string(&vp).unwrap();
        let parsed: VadParams = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.model_path.as_deref(),
            Some(std::path::Path::new("/modèles/données/vad_模型.onnx"))
        );
    }

    #[test]
    fn transcribe_request_db_path_with_spaces_round_trips() {
        let req = TranscribeRequest {
            input: InputSource::File {
                path: PathBuf::from("in.wav"),
            },
            backend: BackendKind::Auto,
            model: None,
            language: None,
            translate: false,
            diarize: false,
            persist: true,
            db_path: PathBuf::from("/Users/Name/My Projects/data base.sqlite3"),
            timeout_ms: None,
            backend_params: BackendParams::default(),
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: TranscribeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.db_path,
            PathBuf::from("/Users/Name/My Projects/data base.sqlite3")
        );
    }

    #[test]
    fn malformed_json_fails_to_deserialize_output_format_and_timestamp_level() {
        // OutputFormat rejects unknown values.
        let bad_formats = [r#""mp3""#, r#""JSON""#, r#"42"#, r#"null"#];
        for input in bad_formats {
            assert!(
                serde_json::from_str::<OutputFormat>(input).is_err(),
                "OutputFormat should reject: {input}"
            );
        }
        // TimestampLevel rejects unknown values.
        let bad_levels = [r#""segment""#, r#""WORD""#, r#"true"#, r#"null"#];
        for input in bad_levels {
            assert!(
                serde_json::from_str::<TimestampLevel>(input).is_err(),
                "TimestampLevel should reject: {input}"
            );
        }
        // AccelerationBackend rejects unknown values.
        let bad_accel = [r#""gpu""#, r#""NONE""#, r#"0"#];
        for input in bad_accel {
            assert!(
                serde_json::from_str::<AccelerationBackend>(input).is_err(),
                "AccelerationBackend should reject: {input}"
            );
        }
    }

    #[test]
    fn backend_params_serde_default_fields_can_be_omitted() {
        // Only no_gpu, carry_initial_prompt, no_fallback, suppress_nst have
        // #[serde(default)]. The other required bools must be present.
        // Provide the required fields, omit the serde(default) ones.
        let json = r#"{
            "output_formats": [],
            "no_timestamps": false,
            "detect_language_only": false,
            "split_on_word": false
        }"#;
        let parsed: BackendParams = serde_json::from_str(json).unwrap();
        // Verify the serde(default) fields defaulted to false.
        assert!(!parsed.no_gpu);
        assert!(!parsed.carry_initial_prompt);
        assert!(!parsed.no_fallback);
        assert!(!parsed.suppress_nst);
        // All Option fields default to None.
        assert!(parsed.decoding.is_none());
        assert!(parsed.vad.is_none());
        assert!(parsed.flash_attention.is_none());
        assert!(parsed.gpu_device.is_none());
        assert!(parsed.insanely_fast_hf_token.is_none());
        assert!(parsed.insanely_fast_transcript_path.is_none());
        assert!(parsed.batch_size.is_none());
        assert!(parsed.threads.is_none());
        assert!(parsed.processors.is_none());
        assert!(parsed.prompt.is_none());
    }

    #[test]
    fn flash_attention_explicit_false_round_trips_distinct_from_none() {
        let params = BackendParams {
            flash_attention: Some(false),
            ..BackendParams::default()
        };
        let json = serde_json::to_string(&params).unwrap();
        let parsed: BackendParams = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.flash_attention,
            Some(false),
            "Some(false) must round-trip distinctly from None"
        );

        // Verify None stays None (default).
        let params_none = BackendParams::default();
        let json_none = serde_json::to_string(&params_none).unwrap();
        let parsed_none: BackendParams = serde_json::from_str(&json_none).unwrap();
        assert_eq!(parsed_none.flash_attention, None);
    }

    #[test]
    fn decoding_params_nan_temperature_round_trips_as_null() {
        // serde_json serializes NaN as null, then deserializes null as None.
        let dp = DecodingParams {
            temperature: Some(f32::NAN),
            ..DecodingParams::default()
        };
        let json = serde_json::to_string(&dp).unwrap();
        let parsed: DecodingParams = serde_json::from_str(&json).unwrap();
        // NaN → null → None: the value is lost during serialization.
        assert!(
            parsed.temperature.is_none(),
            "NaN should round-trip as None (via null)"
        );
    }

    #[test]
    fn input_source_microphone_zero_and_max_seconds_round_trip() {
        for seconds in [0_u32, u32::MAX] {
            let source = InputSource::Microphone {
                seconds,
                device: None,
                ffmpeg_format: None,
                ffmpeg_source: None,
            };
            let json = serde_json::to_string(&source).unwrap();
            let parsed: InputSource = serde_json::from_str(&json).unwrap();
            assert!(
                matches!(&parsed, InputSource::Microphone { .. }),
                "expected Microphone, got {parsed:?}"
            );
            if let InputSource::Microphone { seconds: s, .. } = parsed {
                assert_eq!(s, seconds);
            } else {
                return;
            }
        }
    }

    #[test]
    fn input_source_file_missing_path_fails_deserialization() {
        // InputSource::File requires a `path` field. Omitting it must fail.
        let json = r#"{"kind":"file"}"#;
        let result = serde_json::from_str::<InputSource>(json);
        assert!(result.is_err(), "missing path should fail: {result:?}");
    }

    #[test]
    fn input_source_invalid_kind_tag_fails_deserialization() {
        // An unrecognized tag value for the serde(tag = "kind") discriminator
        // should produce a deserialization error.
        let json = r#"{"kind":"bluetooth","path":"test.wav"}"#;
        let result = serde_json::from_str::<InputSource>(json);
        assert!(result.is_err(), "unknown kind should fail: {result:?}");
    }

    #[test]
    fn replay_envelope_default_has_all_none_fields() {
        let envelope = ReplayEnvelope::default();
        assert!(envelope.input_content_hash.is_none());
        assert!(envelope.backend_identity.is_none());
        assert!(envelope.backend_version.is_none());
        assert!(envelope.output_payload_hash.is_none());
        // Default serializes to just "{}" since all fields use skip_serializing_if.
        let json = serde_json::to_string(&envelope).unwrap();
        assert_eq!(
            json, "{}",
            "default envelope should serialize to empty object"
        );
    }

    #[test]
    fn acceleration_backend_as_str_all_variants() {
        assert_eq!(AccelerationBackend::None.as_str(), "none");
        assert_eq!(AccelerationBackend::Frankentorch.as_str(), "frankentorch");
        assert_eq!(AccelerationBackend::Frankenjax.as_str(), "frankenjax");
    }

    #[test]
    fn transcription_segment_confidence_infinity_round_trips_as_null() {
        // Like NaN, serde_json serializes INFINITY as null.
        let seg = TranscriptionSegment {
            start_sec: Some(f64::INFINITY),
            end_sec: Some(f64::NEG_INFINITY),
            text: "test".to_owned(),
            speaker: None,
            confidence: Some(f64::INFINITY),
        };
        let json = serde_json::to_string(&seg).unwrap();
        let parsed: TranscriptionSegment = serde_json::from_str(&json).unwrap();
        // INFINITY → null → None: the values are lost.
        assert!(
            parsed.start_sec.is_none(),
            "INFINITY should serialize as null → None"
        );
        assert!(
            parsed.end_sec.is_none(),
            "NEG_INFINITY should serialize as null → None"
        );
        assert!(
            parsed.confidence.is_none(),
            "INFINITY confidence should serialize as null → None"
        );
    }

    #[test]
    fn transcription_segment_start_after_end_serializes_without_error() {
        // No runtime validator prevents start_sec > end_sec — this is the caller's
        // responsibility. Verify the struct is fully serializable in this state.
        let seg = TranscriptionSegment {
            start_sec: Some(10.0),
            end_sec: Some(2.0),
            text: "backwards".to_owned(),
            speaker: None,
            confidence: Some(0.5),
        };
        let json = serde_json::to_string(&seg).unwrap();
        let parsed: TranscriptionSegment = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.start_sec, Some(10.0));
        assert_eq!(parsed.end_sec, Some(2.0));
        assert_eq!(parsed.text, "backwards");
    }

    #[test]
    fn backend_params_carry_initial_prompt_true_with_no_prompt() {
        // Edge case: carry_initial_prompt=true but prompt=None.
        // This is semantically questionable but should serialize fine.
        let params = BackendParams {
            carry_initial_prompt: true,
            prompt: None,
            ..Default::default()
        };
        let json = serde_json::to_string(&params).unwrap();
        let parsed: BackendParams = serde_json::from_str(&json).unwrap();
        assert!(parsed.carry_initial_prompt);
        assert!(parsed.prompt.is_none());
    }

    #[test]
    fn input_source_file_empty_path_round_trips() {
        let source = InputSource::File {
            path: PathBuf::from(""),
        };
        let json = serde_json::to_string(&source).unwrap();
        let parsed: InputSource = serde_json::from_str(&json).unwrap();
        assert!(
            matches!(&parsed, InputSource::File { .. }),
            "expected File variant, got {parsed:?}"
        );
        if let InputSource::File { path } = parsed {
            assert_eq!(path, PathBuf::from(""));
        }
    }

    #[test]
    fn diarization_config_all_boolean_combinations_round_trip() {
        for (no_stem, suppress) in [(true, true), (true, false), (false, true), (false, false)] {
            let config = DiarizationConfig {
                no_stem,
                suppress_numerals: suppress,
                ..Default::default()
            };
            let json = serde_json::to_string(&config).unwrap();
            let parsed: DiarizationConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.no_stem, no_stem, "no_stem={no_stem} mismatch");
            assert_eq!(
                parsed.suppress_numerals, suppress,
                "suppress_numerals={suppress} mismatch"
            );
        }
    }

    #[test]
    fn run_report_without_replay_field_deserializes_with_default() {
        // RunReport's replay field has #[serde(default)], so JSON without "replay"
        // should deserialize with ReplayEnvelope::default().
        let json = json!({
            "run_id": "test",
            "trace_id": "00000000000000000000000000000000",
            "started_at_rfc3339": "2026-01-01T00:00:00Z",
            "finished_at_rfc3339": "2026-01-01T00:00:01Z",
            "input_path": "test.wav",
            "normalized_wav_path": "norm.wav",
            "request": {
                "input": {"kind": "file", "path": "test.wav"},
                "backend": "auto",
                "model": null,
                "language": null,
                "translate": false,
                "diarize": false,
                "persist": true,
                "db_path": "/tmp/test.sqlite3",
                "timeout_ms": null,
            },
            "result": {
                "backend": "whisper_cpp",
                "transcript": "hello",
                "language": null,
                "segments": [],
                "acceleration": null,
                "raw_output": {},
                "artifact_paths": [],
            },
            "events": [],
            "warnings": [],
            "evidence": [],
        });
        let report: RunReport =
            serde_json::from_value(json).expect("should deserialize without replay field");
        assert!(report.replay.input_content_hash.is_none());
        assert!(report.replay.backend_identity.is_none());
    }

    // --- BackendDiscoveryEntry ---

    #[test]
    fn backend_discovery_entry_serde_round_trip() {
        let entry = BackendDiscoveryEntry {
            name: "whisper.cpp".to_owned(),
            kind: BackendKind::WhisperCpp,
            available: true,
            capabilities: EngineCapabilities {
                supports_diarization: false,
                supports_translation: true,
                supports_word_timestamps: true,
                supports_gpu: true,
                supports_streaming: false,
            },
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: BackendDiscoveryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "whisper.cpp");
        assert_eq!(parsed.kind, BackendKind::WhisperCpp);
        assert!(parsed.available);
        assert!(!parsed.capabilities.supports_diarization);
        assert!(parsed.capabilities.supports_translation);
    }

    #[test]
    fn backend_discovery_entry_unavailable_round_trip() {
        let entry = BackendDiscoveryEntry {
            name: "insanely-fast-whisper".to_owned(),
            kind: BackendKind::InsanelyFast,
            available: false,
            capabilities: EngineCapabilities {
                supports_diarization: true,
                supports_translation: true,
                supports_word_timestamps: true,
                supports_gpu: true,
                supports_streaming: false,
            },
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: BackendDiscoveryEntry = serde_json::from_str(&json).unwrap();
        assert!(!parsed.available);
        assert!(parsed.capabilities.supports_diarization);
    }

    // --- BackendsReport ---

    #[test]
    fn backends_report_serde_round_trip() {
        let report = BackendsReport {
            backends: vec![
                BackendDiscoveryEntry {
                    name: "whisper.cpp".to_owned(),
                    kind: BackendKind::WhisperCpp,
                    available: true,
                    capabilities: EngineCapabilities {
                        supports_diarization: false,
                        supports_translation: true,
                        supports_word_timestamps: true,
                        supports_gpu: true,
                        supports_streaming: false,
                    },
                },
                BackendDiscoveryEntry {
                    name: "whisper-diarization".to_owned(),
                    kind: BackendKind::WhisperDiarization,
                    available: false,
                    capabilities: EngineCapabilities {
                        supports_diarization: true,
                        supports_translation: false,
                        supports_word_timestamps: false,
                        supports_gpu: true,
                        supports_streaming: false,
                    },
                },
            ],
        };
        let json = serde_json::to_string(&report).unwrap();
        let parsed: BackendsReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.backends.len(), 2);
        assert_eq!(parsed.backends[0].name, "whisper.cpp");
        assert_eq!(parsed.backends[1].kind, BackendKind::WhisperDiarization);
    }

    #[test]
    fn backends_report_empty_round_trip() {
        let report = BackendsReport { backends: vec![] };
        let json = serde_json::to_string(&report).unwrap();
        let parsed: BackendsReport = serde_json::from_str(&json).unwrap();
        assert!(parsed.backends.is_empty());
    }

    #[test]
    fn device_map_strategy_serde_round_trip_and_rejects_unknown() {
        for (variant, expected_wire) in [
            (DeviceMapStrategy::Auto, "\"auto\""),
            (DeviceMapStrategy::Sequential, "\"sequential\""),
        ] {
            let serialized = serde_json::to_string(&variant).unwrap();
            assert_eq!(serialized, expected_wire);
            let deserialized: DeviceMapStrategy = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized, variant);
        }
        for bad in [r#""Auto""#, r#""parallel""#, r#"null"#, r#"0"#] {
            assert!(
                serde_json::from_str::<DeviceMapStrategy>(bad).is_err(),
                "should reject: {bad}"
            );
        }
    }

    #[test]
    fn word_timestamp_params_default_and_serde() {
        let wt = WordTimestampParams::default();
        assert!(!wt.enabled);
        assert!(wt.max_len.is_none());
        assert!(wt.token_threshold.is_none());
        assert!(wt.token_sum_threshold.is_none());

        let populated = WordTimestampParams {
            enabled: true,
            max_len: Some(50),
            token_threshold: Some(0.01),
            token_sum_threshold: Some(0.05),
        };
        let json = serde_json::to_string(&populated).unwrap();
        let parsed: WordTimestampParams = serde_json::from_str(&json).unwrap();
        assert!(parsed.enabled);
        assert_eq!(parsed.max_len, Some(50));
        assert_eq!(parsed.token_threshold, Some(0.01));

        // serde(default) on enabled: omitting it yields false
        let no_enabled = r#"{"max_len": 30}"#;
        let parsed2: WordTimestampParams = serde_json::from_str(no_enabled).unwrap();
        assert!(!parsed2.enabled);
        assert_eq!(parsed2.max_len, Some(30));
    }

    #[test]
    fn insanely_fast_tuning_params_default_and_serde() {
        let p = InsanelyFastTuningParams::default();
        assert!(p.device_map.is_none());
        assert!(p.torch_dtype.is_none());
        assert!(!p.disable_better_transformer);

        let populated = InsanelyFastTuningParams {
            device_map: Some(DeviceMapStrategy::Sequential),
            torch_dtype: Some("bfloat16".to_owned()),
            disable_better_transformer: true,
        };
        let json = serde_json::to_string(&populated).unwrap();
        let parsed: InsanelyFastTuningParams = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.device_map, Some(DeviceMapStrategy::Sequential));
        assert_eq!(parsed.torch_dtype.as_deref(), Some("bfloat16"));
        assert!(parsed.disable_better_transformer);
    }

    #[test]
    fn diarization_pipeline_extension_structs_default_and_serde() {
        let ac = AlignmentConfig::default();
        assert!(ac.alignment_model.is_none());
        assert!(ac.interpolate_method.is_none());
        assert!(!ac.return_char_alignments);

        let ac_full = AlignmentConfig {
            alignment_model: Some("WAV2VEC2_ASR_LARGE_LV60K_960H".to_owned()),
            interpolate_method: Some("nearest".to_owned()),
            return_char_alignments: true,
        };
        let ac_json = serde_json::to_string(&ac_full).unwrap();
        let ac_parsed: AlignmentConfig = serde_json::from_str(&ac_json).unwrap();
        assert!(ac_parsed.return_char_alignments);
        assert_eq!(
            ac_parsed.alignment_model.as_deref(),
            Some("WAV2VEC2_ASR_LARGE_LV60K_960H")
        );

        let pc = PunctuationConfig::default();
        assert!(!pc.enabled);
        assert!(pc.model.is_none());

        let sc = SourceSeparationConfig::default();
        assert!(!sc.enabled);
        assert!(sc.shifts.is_none());
        assert!(sc.overlap.is_none());

        let sc_full = SourceSeparationConfig {
            enabled: true,
            model: Some("htdemucs".to_owned()),
            shifts: Some(4),
            overlap: Some(0.25),
        };
        let sc_json = serde_json::to_string(&sc_full).unwrap();
        let sc_parsed: SourceSeparationConfig = serde_json::from_str(&sc_json).unwrap();
        assert!(sc_parsed.enabled);
        assert_eq!(sc_parsed.shifts, Some(4));
        assert_eq!(sc_parsed.overlap, Some(0.25));
    }

    #[test]
    fn backend_params_extension_fields_round_trip() {
        let params = BackendParams {
            word_timestamps: Some(WordTimestampParams {
                enabled: true,
                max_len: Some(100),
                token_threshold: Some(0.02),
                token_sum_threshold: None,
            }),
            insanely_fast_tuning: Some(InsanelyFastTuningParams {
                device_map: Some(DeviceMapStrategy::Auto),
                torch_dtype: Some("float16".to_owned()),
                disable_better_transformer: true,
            }),
            alignment: Some(AlignmentConfig {
                alignment_model: Some("WAV2VEC2_ASR_BASE_960H".to_owned()),
                interpolate_method: None,
                return_char_alignments: false,
            }),
            punctuation: Some(PunctuationConfig {
                model: Some("punct-base".to_owned()),
                enabled: true,
            }),
            source_separation: Some(SourceSeparationConfig {
                enabled: true,
                model: Some("htdemucs".to_owned()),
                shifts: Some(2),
                overlap: Some(0.5),
            }),
            ..BackendParams::default()
        };
        let json = serde_json::to_string(&params).unwrap();
        let parsed: BackendParams = serde_json::from_str(&json).unwrap();

        let wt = parsed.word_timestamps.expect("word_timestamps");
        assert!(wt.enabled);
        assert_eq!(wt.max_len, Some(100));

        let tuning = parsed.insanely_fast_tuning.expect("insanely_fast_tuning");
        assert_eq!(tuning.device_map, Some(DeviceMapStrategy::Auto));
        assert!(tuning.disable_better_transformer);

        let al = parsed.alignment.expect("alignment");
        assert_eq!(
            al.alignment_model.as_deref(),
            Some("WAV2VEC2_ASR_BASE_960H")
        );

        let punct = parsed.punctuation.expect("punctuation");
        assert!(punct.enabled);

        let sep = parsed.source_separation.expect("source_separation");
        assert!(sep.enabled);
        assert_eq!(sep.shifts, Some(2));
    }

    #[test]
    fn zero_speaker_count_is_rejected() {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::HardConstraint { count: 0 },
            ..DiarizationRequest::default()
        };
        assert_eq!(
            request.validate(1_000).expect_err("zero count").code,
            DiarizationValidationCode::InvalidSpeakerCount
        );
    }

    #[test]
    fn transcribe_request_stdin_full_round_trip() {
        let request = TranscribeRequest {
            input: InputSource::Stdin {
                hint_extension: Some("mp3".to_owned()),
            },
            backend: BackendKind::WhisperDiarization,
            model: None,
            language: None,
            translate: false,
            diarize: true,
            persist: false,
            db_path: PathBuf::from("/tmp/test.sqlite3"),
            timeout_ms: Some(60_000),
            backend_params: BackendParams {
                batch_size: Some(8),
                ..BackendParams::default()
            },
        };
        let json = serde_json::to_string(&request).unwrap();
        let parsed: TranscribeRequest = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed.input,
            InputSource::Stdin {
                hint_extension: Some(ref ext)
            } if ext == "mp3"
        ));
        assert_eq!(parsed.backend, BackendKind::WhisperDiarization);
        assert!(parsed.diarize);
        assert_eq!(parsed.timeout_ms, Some(60_000));
        assert_eq!(parsed.backend_params.batch_size, Some(8));
    }

    #[test]
    fn punctuation_config_enabled_without_model_round_trip() {
        let pc = PunctuationConfig {
            enabled: true,
            model: None,
        };
        let json = serde_json::to_string(&pc).unwrap();
        let parsed: PunctuationConfig = serde_json::from_str(&json).unwrap();
        assert!(parsed.enabled);
        assert!(parsed.model.is_none());
    }

    #[test]
    fn source_separation_overlap_boundary_values_round_trip() {
        for overlap in [0.0_f32, 1.0_f32] {
            let sc = SourceSeparationConfig {
                enabled: true,
                model: None,
                shifts: None,
                overlap: Some(overlap),
            };
            let json = serde_json::to_string(&sc).unwrap();
            let parsed: SourceSeparationConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.overlap, Some(overlap), "overlap={overlap}");
        }
    }

    #[test]
    fn run_report_diverse_evidence_entries_round_trip() {
        let mut report = make_test_run_report();
        report.evidence = vec![
            json!(null),
            json!("plain string evidence"),
            json!(42),
            json!({"nested": {"key": [1, 2, 3]}}),
            json!([true, false, null]),
        ];
        let json = serde_json::to_string(&report).unwrap();
        let parsed: RunReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.evidence.len(), 5);
        assert!(parsed.evidence[0].is_null());
        assert_eq!(parsed.evidence[1], "plain string evidence");
        assert_eq!(parsed.evidence[2], 42);
        assert_eq!(parsed.evidence[3]["nested"]["key"][1], 2);
        assert_eq!(parsed.evidence[4][0], true);
    }

    fn speaker_hint(
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
            provenance: Some("contextual transcript cue".to_owned()),
        }
    }

    #[test]
    fn acoustic_diarization_request_round_trips_snake_case() {
        let request = DiarizationRequest {
            engine: DiarizationEngine::Acoustic,
            fallback: DiarizationFallbackPolicy::Unknown,
            known_intervals: vec![speaker_hint(
                "caller",
                100,
                900,
                KnownSpeakerPolicy::HardMustLink,
            )],
            ..DiarizationRequest::default()
        };
        let json = serde_json::to_string(&request).expect("serialize request");
        assert!(json.contains("\"engine\":\"acoustic\""));
        assert!(json.contains("\"policy\":\"hard_must_link\""));
        let parsed: DiarizationRequest = serde_json::from_str(&json).expect("deserialize request");
        assert_eq!(parsed, request);
        assert!(request.validate(1_000).is_ok());
    }

    #[test]
    fn contradictory_hard_hints_fail_with_stable_code() {
        let request = DiarizationRequest {
            known_intervals: vec![
                speaker_hint("near", 0, 800, KnownSpeakerPolicy::HardMustLink),
                speaker_hint("remote", 700, 1_000, KnownSpeakerPolicy::HardMustLink),
            ],
            ..DiarizationRequest::default()
        };
        let error = request
            .validate(1_000)
            .expect_err("overlapping hard identities must fail");
        assert_eq!(
            error.code,
            DiarizationValidationCode::ContradictoryHardHints
        );
        assert_eq!(error.code.as_str(), "diarization.contradictory_hard_hints");
    }

    #[test]
    fn speaker_count_request_cannot_merge_distinct_hard_references() {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::Range {
                minimum: 1,
                maximum: 1,
            },
            known_intervals: vec![
                speaker_hint("near", 0, 400, KnownSpeakerPolicy::HardMustLink),
                speaker_hint("remote", 600, 1_000, KnownSpeakerPolicy::HardMustLink),
            ],
            ..DiarizationRequest::default()
        };
        let error = request
            .validate(1_000)
            .expect_err("two immutable hard references cannot fit one speaker");
        assert_eq!(error.code, DiarizationValidationCode::InvalidSpeakerCount);
        assert!(error.message.contains("2 distinct speakers"));
    }

    #[test]
    fn overlapping_soft_hints_are_advisory() {
        let request = DiarizationRequest {
            known_intervals: vec![
                speaker_hint("near", 0, 800, KnownSpeakerPolicy::SoftEnrollment),
                speaker_hint("remote", 700, 1_000, KnownSpeakerPolicy::SoftEnrollment),
            ],
            ..DiarizationRequest::default()
        };
        assert!(request.validate(1_000).is_ok());
    }

    #[test]
    fn malformed_hint_and_constraint_codes_are_stable() {
        let mut request = DiarizationRequest {
            known_intervals: vec![speaker_hint(
                "",
                100,
                200,
                KnownSpeakerPolicy::SoftEnrollment,
            )],
            ..DiarizationRequest::default()
        };
        assert_eq!(
            request.validate(1_000).expect_err("empty").code,
            DiarizationValidationCode::EmptySpeakerRef
        );

        request.known_intervals[0].speaker_ref = "speaker".to_owned();
        request.known_intervals[0].confidence = f64::NAN;
        assert_eq!(
            request.validate(1_000).expect_err("NaN").code,
            DiarizationValidationCode::InvalidHintConfidence
        );

        request.known_intervals[0].confidence = 0.8;
        request.known_intervals[0].end_ms = 1_001;
        assert_eq!(
            request.validate(1_000).expect_err("bounds").code,
            DiarizationValidationCode::HintOutsideAudio
        );

        request.known_intervals.clear();
        request.speaker_count = SpeakerCountRequest::Range {
            minimum: 4,
            maximum: 3,
        };
        assert_eq!(
            request.validate(1_000).expect_err("constraints").code,
            DiarizationValidationCode::InvalidSpeakerCount
        );
    }

    #[test]
    fn acoustic_hint_request_limits_fail_before_quadratic_validation() {
        let hint = speaker_hint("speaker", 0, 1, KnownSpeakerPolicy::SoftEnrollment);
        let request = DiarizationRequest {
            known_intervals: vec![hint.clone(); MAX_KNOWN_SPEAKER_INTERVALS + 1],
            ..DiarizationRequest::default()
        };
        assert_eq!(
            request
                .validate(1)
                .expect_err("interval count must be bounded")
                .code,
            DiarizationValidationCode::TooManyKnownIntervals
        );

        let mut request = DiarizationRequest {
            known_intervals: vec![hint],
            ..DiarizationRequest::default()
        };
        request.known_intervals[0].speaker_ref = "s".repeat(MAX_SPEAKER_REF_BYTES + 1);
        assert_eq!(
            request
                .validate(1)
                .expect_err("speaker reference must be bounded")
                .code,
            DiarizationValidationCode::SpeakerRefTooLong
        );

        request.known_intervals[0].speaker_ref = "speaker".to_owned();
        request.known_intervals[0].provenance = Some("p".repeat(MAX_HINT_PROVENANCE_BYTES + 1));
        assert_eq!(
            request
                .validate(1)
                .expect_err("provenance must be bounded")
                .code,
            DiarizationValidationCode::ProvenanceTooLong
        );
    }

    #[test]
    fn diarization_report_is_typed_and_privacy_safe() {
        let report = DiarizationReport {
            implementation: "native_acoustic".to_owned(),
            contract_version: "acoustic-diarization-v2".to_owned(),
            feature_schema: "acoustic-feature-v1".to_owned(),
            normalized_input_sha256:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            hint_document_sha256: Some(
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
            ),
            turns: vec![DiarizationTurn {
                start_ms: 0,
                end_ms: 1_000,
                speaker_ref: Some("near".to_owned()),
                speaker_confidence: Some(0.91),
                change_confidence: Some(0.84),
                overlap_suspected: false,
                hard_hint_attributed: true,
            }],
            profiles: vec![SpeakerProfileSummary {
                speaker_ref: "near".to_owned(),
                frame_count: 72,
                voiced_duration_ms: 720,
                reliability: 0.9,
                voice_profile_count: 1,
                channel_profile_count: 1,
                training_accepted_count: 1,
                training_downweighted_count: 0,
                training_quarantined_count: 0,
                anchored: true,
                soft_hint_contradiction: None,
            }],
            hint_evidence: vec![SpeakerHintEvidenceSummary {
                hint_index: 0,
                speaker_ref: "near".to_owned(),
                policy: KnownSpeakerPolicy::HardMustLink,
                disposition: SpeakerHintDisposition::HardAttributed,
                usable_tracklet_count: 1,
                accepted_tracklet_count: 1,
                rejected_tracklet_count: 0,
                profile_accepted_tracklet_count: 1,
                profile_downweighted_tracklet_count: 0,
                profile_quarantined_tracklet_count: 0,
                applied_weight: 1.0,
                contradiction_score: None,
            }],
            speaker_queries: vec![SpeakerAttributionQuery {
                query_id_sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                    .to_owned(),
                start_ms: 1_000,
                end_ms: 1_500,
                reason: SpeakerAttributionQueryReason::LowConfidence,
                candidate_speaker_refs: vec!["near".to_owned()],
                suggested_policy: KnownSpeakerPolicy::SoftEnrollment,
            }],
            speaker_count: SpeakerCountOutcome {
                request: SpeakerCountRequest::HardConstraint { count: 1 },
                status: SpeakerCountOutcomeStatus::Satisfied,
                supported_speaker_count: 1,
                active_speaker_refs: vec!["near".to_owned()],
                dominant_speaker_share: 1.0,
                unknown_voiced_share: 0.0,
                reasons: vec![SpeakerCountOutcomeReason::RequestedCountMatched],
                speaker_evidence: vec![SpeakerEvidenceSummary {
                    speaker_ref: "near".to_owned(),
                    assigned_tracklet_count: 1,
                    independent_tracklet_count: 1,
                    recurrence_episode_count: 1,
                    voiced_frame_count: 72,
                    independent_voiced_frame_count: 72,
                    voiced_duration_ms: 720,
                    mean_assignment_confidence: 0.91,
                    profile_reliability: 0.9,
                    hard_anchored: true,
                    separated_from_supported_speakers: true,
                    reasons: vec![SpeakerEvidenceReason::SupportedByHardHint],
                    supported: true,
                }],
            },
            fallback_status: DiarizationFallbackStatus::NotNeeded,
            diagnostics: Vec::new(),
        };
        let json = serde_json::to_string(&report).expect("serialize report");
        assert!(!json.contains("feature_vector"));
        assert!(!json.contains("raw_audio"));
        assert_eq!(
            serde_json::from_str::<DiarizationReport>(&json).expect("deserialize report"),
            report
        );
    }
}
