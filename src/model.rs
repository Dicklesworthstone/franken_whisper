use std::collections::BTreeSet;
use std::path::PathBuf;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

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
#[serde(deny_unknown_fields)]
pub struct SpeakerCountPriorMass {
    pub count: u32,
    pub probability: f64,
}

/// Typed speaker-count semantics at the common request boundary.
///
/// A hard count restricts candidate model search; it never forces assignments
/// or removes UNKNOWN. `Prior` contributes bounded linear evidence over its
/// declared bins; acoustic evidence may select outside that support or remain
/// unresolved. An engine that cannot preserve these semantics must reject the
/// request instead of silently approximating it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpeakerCountRequest {
    #[default]
    Infer,
    Prior {
        bins: Vec<SpeakerCountPriorMass>,
    },
    /// Soft preference over an inclusive interval; evidence may select
    /// outside the interval or remain unresolved.
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
    /// In-process ECAPA speaker identity without acoustic channel fusion.
    Ecapa,
    /// In-process ECAPA identity with separately bounded acoustic channel
    /// evidence when a compatible channel-valid pair can actually be scored.
    #[value(alias = "ecapa_fused")]
    EcapaFused,
}

/// Speaker-evidence representation that actually produced a diarization
/// report. This is distinct from the requested engine because a neural request
/// may conservatively execute the native acoustic fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiarizationSpeakerEvidenceMode {
    AcousticV2,
    EcapaOnly,
    EcapaWithAcousticChannel,
    External,
    None,
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
    /// If a neural representation is unavailable or insufficient, rerun the
    /// exact common stack with the native acoustic-v2 identity representation.
    Acoustic,
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct DiarizationRequest {
    pub engine: DiarizationEngine,
    pub fallback: DiarizationFallbackPolicy,
    pub speaker_count: SpeakerCountRequest,
    #[serde(default)]
    pub known_intervals: Vec<KnownSpeakerInterval>,
    /// Samples removed from each enrollment edge to avoid boundary bleed.
    pub enrollment_edge_guard_ms: u32,
    /// Maximum number of global clustering prototypes, including any
    /// enrollment-profile nodes admitted into the affinity graph.
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
        if hard_speaker_count > MAX_SPEAKER_COUNT as usize {
            return Err(DiarizationValidationError {
                code: DiarizationValidationCode::InvalidSpeakerCount,
                message: format!(
                    "hard speaker hints name {hard_speaker_count} distinct speakers but the \
                     bounded speaker-count domain permits at most {MAX_SPEAKER_COUNT}"
                ),
                hint_index: None,
            });
        }
        if hard_speaker_count > usize::from(self.max_prototypes) {
            return Err(DiarizationValidationError {
                code: DiarizationValidationCode::InvalidPrototypeCap,
                message: format!(
                    "max_prototypes={} cannot preserve {hard_speaker_count} distinct hard speaker references",
                    self.max_prototypes
                ),
                hint_index: None,
            });
        }
        let constrained_maximum = match self.speaker_count {
            SpeakerCountRequest::HardConstraint { count } => Some(count as usize),
            SpeakerCountRequest::Infer
            | SpeakerCountRequest::Prior { .. }
            | SpeakerCountRequest::Range { .. } => None,
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

/// Stable execution state for the optional neural speaker representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuralSpeakerRepresentationStatus {
    Ready,
    Degraded,
    Unavailable,
}

/// Feature-value-free reason that the neural representation was degraded or
/// unavailable. These codes are safe to emit in robot mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuralSpeakerRepresentationReason {
    ShortTracklet,
    InsufficientIdentityEvidence,
    InsufficientTracklets,
    ContradictoryEnrollment,
    ModelUnavailable,
    ModelInvalid,
    InferenceFailed,
}

/// Typed provenance for a loaded ECAPA model package. `PackageVerified`
/// records only that the admitted package digest was verified; it makes no
/// claim about whether a process-local cache was hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuralModelLoadSource {
    PackageVerified,
}

/// Privacy-safe provenance and coverage for an optional ECAPA execution.
///
/// Raw embeddings and model paths are deliberately absent. The exact package
/// digest is public model provenance rather than user-derived information.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeuralSpeakerRepresentationSummary {
    pub schema_version: String,
    pub provider_version: String,
    pub expected_model_package_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loaded_model_package_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_load_source: Option<NeuralModelLoadSource>,
    pub status: NeuralSpeakerRepresentationStatus,
    pub embedded_tracklet_count: u64,
    pub zero_padded_tracklet_count: u64,
    pub skipped_tracklet_count: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<NeuralSpeakerRepresentationReason>,
}

impl NeuralSpeakerRepresentationSummary {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != "neural-speaker-representation-summary-v1" {
            return Err("neural representation summary schema version is unsupported".to_owned());
        }
        if self.provider_version.trim().is_empty() || self.provider_version.len() > 128 {
            return Err("neural representation provider version is invalid".to_owned());
        }
        if !lowercase_sha256(&self.expected_model_package_sha256)
            || self
                .loaded_model_package_sha256
                .as_ref()
                .is_some_and(|digest| !lowercase_sha256(digest))
        {
            return Err("neural representation model digest is not lowercase SHA-256".to_owned());
        }
        if self
            .loaded_model_package_sha256
            .as_deref()
            .is_some_and(|digest| digest != self.expected_model_package_sha256.as_str())
        {
            return Err(
                "loaded neural model digest does not match the expected package".to_owned(),
            );
        }
        match self.status {
            NeuralSpeakerRepresentationStatus::Ready
                if self.loaded_model_package_sha256.as_deref()
                    != Some(self.expected_model_package_sha256.as_str())
                    || self.embedded_tracklet_count < 2
                    || self.zero_padded_tracklet_count != 0
                    || self.skipped_tracklet_count != 0
                    || !self.reasons.is_empty() =>
            {
                return Err(
                    "ready neural representation summary is internally inconsistent".to_owned(),
                );
            }
            NeuralSpeakerRepresentationStatus::Degraded
                if self.loaded_model_package_sha256.as_deref()
                    != Some(self.expected_model_package_sha256.as_str())
                    || self.embedded_tracklet_count == 0
                    || self.reasons.is_empty() =>
            {
                return Err(
                    "degraded neural representation summary is internally inconsistent".to_owned(),
                );
            }
            NeuralSpeakerRepresentationStatus::Unavailable
                if self.embedded_tracklet_count != 0
                    || self.zero_padded_tracklet_count != 0
                    || self.reasons.is_empty() =>
            {
                return Err(
                    "unavailable neural representation summary claims observed model evidence"
                        .to_owned(),
                );
            }
            _ => {}
        }
        let model_was_unavailable = self.reasons.iter().any(|reason| {
            matches!(
                reason,
                NeuralSpeakerRepresentationReason::ModelUnavailable
                    | NeuralSpeakerRepresentationReason::ModelInvalid
            )
        });
        if model_was_unavailable && self.loaded_model_package_sha256.is_some() {
            return Err(
                "unavailable or invalid neural model claims a loaded package digest".to_owned(),
            );
        }
        if self.loaded_model_package_sha256.is_some() != self.model_load_source.is_some() {
            return Err(
                "neural model digest and load source must either both be present or both be absent"
                    .to_owned(),
            );
        }
        if self
            .reasons
            .contains(&NeuralSpeakerRepresentationReason::InferenceFailed)
            && self.loaded_model_package_sha256.is_none()
        {
            return Err(
                "neural inference failure must identify the successfully loaded model source"
                    .to_owned(),
            );
        }
        if self.zero_padded_tracklet_count > self.embedded_tracklet_count {
            return Err(
                "zero-padded neural tracklet count exceeds embedded tracklet count".to_owned(),
            );
        }
        if slice_has_duplicates(&self.reasons) {
            return Err("neural representation reasons are duplicated".to_owned());
        }
        if self.zero_padded_tracklet_count > 0
            && !self
                .reasons
                .contains(&NeuralSpeakerRepresentationReason::ShortTracklet)
        {
            return Err("zero-padded neural tracklets lack short-tracklet provenance".to_owned());
        }
        if self.skipped_tracklet_count > 0
            && !self.reasons.iter().any(|reason| {
                matches!(
                    reason,
                    NeuralSpeakerRepresentationReason::ShortTracklet
                        | NeuralSpeakerRepresentationReason::InsufficientIdentityEvidence
                )
            })
        {
            return Err(
                "skipped neural tracklets lack short-or-insufficient-identity provenance"
                    .to_owned(),
            );
        }
        Ok(())
    }
}

/// Algorithm that produced the concrete partition consumed by temporal
/// decoding. This is deliberately separate from the calibrated speaker-count
/// estimate: an unresolved posterior may coexist with a conservative,
/// explicitly non-authoritative operational partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiarizationOperationalPartitionMethod {
    FixedSafeAgglomerative,
    ProbabilisticConsensus,
    /// ECAPA-only spherical clustering over neural speaker embeddings.
    EcapaSpherical,
    /// Five-lane ECAPA consensus clustering whose pair evidence, speaker-count
    /// estimate, separation checks, and downstream emissions retain the
    /// authorized acoustic-channel side evidence. Emission of this method
    /// proves that at least one selected consensus merge joined a compatible
    /// channel-valid pair.
    EcapaFusedConsensus,
}

/// Typed, privacy-safe provenance for the concrete partition actually used by
/// the common diarization stack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiarizationOperationalPartitionSummary {
    pub schema_version: String,
    pub method: DiarizationOperationalPartitionMethod,
    pub selected_count: u32,
    pub confidence: f64,
    pub calibration_sha256: String,
    pub authority: SpeakerCountCalibrationStatus,
}

impl DiarizationOperationalPartitionSummary {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != "diarization-operational-partition-v2" {
            return Err("operational partition schema version is unsupported".to_owned());
        }
        if self.authority == SpeakerCountCalibrationStatus::Certified {
            return Err(
                "certified operational partitions are not admitted without a registered calibration digest"
                    .to_owned(),
            );
        }
        if !(1..=MAX_SPEAKER_COUNT).contains(&self.selected_count) {
            return Err(format!(
                "operational partition count must stay within 1..={MAX_SPEAKER_COUNT}"
            ));
        }
        if !unit_interval(self.confidence) {
            return Err("operational partition confidence is not finite in 0..=1".to_owned());
        }
        if !lowercase_sha256(&self.calibration_sha256) {
            return Err(
                "operational partition calibration digest is not lowercase SHA-256".to_owned(),
            );
        }
        match self.method {
            DiarizationOperationalPartitionMethod::EcapaSpherical
            | DiarizationOperationalPartitionMethod::EcapaFusedConsensus
            | DiarizationOperationalPartitionMethod::ProbabilisticConsensus
                if self.authority != SpeakerCountCalibrationStatus::DevelopmentUncertified =>
            {
                return Err(
                    "probabilistic operational partition has incompatible authority".to_owned(),
                );
            }
            DiarizationOperationalPartitionMethod::FixedSafeAgglomerative
                if self.authority != SpeakerCountCalibrationStatus::FixedSafeUncalibrated =>
            {
                return Err(
                    "fixed-safe operational partition must be explicitly uncalibrated".to_owned(),
                );
            }
            _ => {}
        }
        if self.method == DiarizationOperationalPartitionMethod::FixedSafeAgglomerative
            && self.confidence.to_bits() != 0.0_f64.to_bits()
        {
            return Err("fixed-safe operational partition must have zero confidence".to_owned());
        }
        Ok(())
    }
}

/// One acoustic speaker turn, independent of ASR segment boundaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct SpeakerProfileSummary {
    pub speaker_ref: String,
    pub frame_count: u64,
    pub voiced_duration_ms: u64,
    pub reliability: f64,
    pub voice_profile_count: u32,
    pub channel_profile_count: u32,
    pub training_accepted_count: u32,
    pub training_downweighted_count: u32,
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
#[serde(deny_unknown_fields)]
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
    /// One long observation was split into disjoint discovery and validation
    /// windows; the held-out window independently selected and separated the
    /// discovery cluster.
    SupportedByHeldoutObservation,
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
#[serde(deny_unknown_fields)]
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
    SpeakerCountEvidenceUnresolved,
    NoSupportedSpeakers,
    DominantSpeakerShareExceeded,
    AmbiguousSpeakerSeparation,
    ExternalAttribution,
}

/// One normalized probability mass in a bounded speaker-count estimate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakerCountPosteriorBin {
    pub count: u32,
    pub probability: f64,
}

/// Inclusive count interval supported by the retained acoustic evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakerCountRange {
    pub minimum: u32,
    pub maximum: u32,
}

/// Independent evidence view contributing to speaker-count selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerCountEvidenceLane {
    MergeRisk,
    SparseNormalizedEigengap,
    FeatureJackknife,
    EffectiveOccupancy,
    ConstraintGraph,
    CallerPrior,
}

/// Why one count-evidence lane could not contribute authoritative evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerCountLaneUnavailableReason {
    InsufficientPrototypes,
    InvalidAffinity,
    SolverDidNotConverge,
    InsufficientIndependentReplicates,
    InsufficientVoicedEvidence,
    NotRequested,
    CalibrationUnavailable,
    ResourceLimit,
    ContradictoryConstraints,
}

/// Privacy-safe summary of one count-evidence lane.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakerCountLaneEvidence {
    pub lane: SpeakerCountEvidenceLane,
    pub available: bool,
    pub proposed_count: Option<u32>,
    pub confidence: f64,
    pub unavailable_reason: Option<SpeakerCountLaneUnavailableReason>,
}

/// Bounded, content-free resource accounting for speaker-count inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakerCountResourceSummary {
    pub prototype_count: u32,
    pub affinity_pair_evaluations: u64,
    pub retained_sparse_edges: u32,
    pub estimated_peak_buffer_bytes: u64,
    pub stability_replicates: u32,
    pub solver_iterations: u32,
    pub solver_sparse_matvec_terms: u64,
    pub solver_residual: Option<f64>,
}

/// Authority attached to the fused speaker-count distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerCountCalibrationStatus {
    Certified,
    DevelopmentUncertified,
    FixedSafeUncalibrated,
    Unavailable,
}

/// Versioned, bounded and privacy-safe speaker-count estimate.
///
/// `posterior` contains only concrete count bins. `unresolved_probability` is
/// separate so weak, contradictory, out-of-domain, or resource-limited
/// evidence cannot be normalized into fabricated certainty over a count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakerCountEstimate {
    pub schema_version: String,
    pub selected_count: Option<u32>,
    pub supported_range: Option<SpeakerCountRange>,
    pub posterior: Vec<SpeakerCountPosteriorBin>,
    pub unresolved_probability: f64,
    pub entropy_bits: f64,
    pub stability: f64,
    pub constraint_lower_bound: u32,
    pub candidate_upper_bound: u32,
    pub calibration_status: SpeakerCountCalibrationStatus,
    pub calibration_sha256: String,
    pub evidence_sha256: String,
    pub lanes: Vec<SpeakerCountLaneEvidence>,
    pub resources: SpeakerCountResourceSummary,
}

impl SpeakerCountEstimate {
    /// Validate normalization, bounds, deterministic ordering and lane
    /// availability semantics before this estimate crosses a public boundary.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != "speaker-count-estimate-v2" {
            return Err("speaker-count estimate schema version is unsupported".to_owned());
        }
        if self.calibration_status == SpeakerCountCalibrationStatus::Certified {
            return Err(
                "certified speaker-count estimates are not admitted without a registered calibration digest"
                    .to_owned(),
            );
        }
        if self.constraint_lower_bound == 0
            || self.candidate_upper_bound == 0
            || self.candidate_upper_bound > MAX_SPEAKER_COUNT
        {
            return Err(format!(
                "speaker-count estimate bounds must stay within 1..={MAX_SPEAKER_COUNT}"
            ));
        }
        if self.constraint_lower_bound > self.candidate_upper_bound {
            return Err(
                "speaker-count estimate lower bound exceeds its candidate upper bound".to_owned(),
            );
        }
        if !unit_interval(self.unresolved_probability) {
            return Err("speaker-count unresolved probability is not finite in 0..=1".to_owned());
        }
        if !unit_interval(self.stability) {
            return Err("speaker-count stability is not finite in 0..=1".to_owned());
        }
        if !self.entropy_bits.is_finite() || self.entropy_bits < 0.0 {
            return Err("speaker-count entropy is not finite and non-negative".to_owned());
        }
        if !lowercase_sha256(&self.calibration_sha256) {
            return Err(
                "speaker-count calibration fingerprint is not lowercase SHA-256".to_owned(),
            );
        }
        if !lowercase_sha256(&self.evidence_sha256) {
            return Err("speaker-count evidence fingerprint is not lowercase SHA-256".to_owned());
        }
        if self
            .resources
            .solver_residual
            .is_some_and(|residual| !residual.is_finite() || residual < 0.0)
        {
            return Err("speaker-count solver residual is not finite and non-negative".to_owned());
        }
        let maximum_edges = u64::from(self.resources.prototype_count)
            .saturating_mul(u64::from(self.resources.prototype_count.saturating_sub(1)))
            / 2;
        if u64::from(self.resources.retained_sparse_edges) > maximum_edges {
            return Err(
                "speaker-count retained sparse edges exceed the simple graph bound".to_owned(),
            );
        }
        let maximum_directed_pairs = u64::from(self.resources.prototype_count)
            .saturating_mul(u64::from(self.resources.prototype_count.saturating_sub(1)));
        if self.resources.affinity_pair_evaluations > maximum_directed_pairs {
            return Err(
                "speaker-count affinity evaluations exceed the directed pair bound".to_owned(),
            );
        }
        if self.resources.prototype_count == 0
            && (self.resources.affinity_pair_evaluations != 0
                || self.resources.retained_sparse_edges != 0
                || self.resources.estimated_peak_buffer_bytes != 0
                || self.resources.stability_replicates != 0
                || self.resources.solver_iterations != 0
                || self.resources.solver_sparse_matvec_terms != 0
                || self.resources.solver_residual.is_some())
        {
            return Err(
                "speaker-count resource accounting claims work without prototypes".to_owned(),
            );
        }
        if self.resources.prototype_count == 0
            && self.calibration_status != SpeakerCountCalibrationStatus::Unavailable
        {
            return Err(
                "zero-prototype speaker-count estimate must be explicitly unavailable".to_owned(),
            );
        }

        let mut previous_count = None;
        let mut concrete_mass = 0.0;
        let mut entropy_bits = entropy_term(self.unresolved_probability);
        let mut map = None::<(f64, u32)>;
        for bin in &self.posterior {
            if bin.count < self.constraint_lower_bound || bin.count > self.candidate_upper_bound {
                return Err("speaker-count posterior bin lies outside candidate bounds".to_owned());
            }
            if previous_count.is_some_and(|previous| bin.count <= previous) {
                return Err(
                    "speaker-count posterior bins are not strictly count-ordered".to_owned(),
                );
            }
            if !bin.probability.is_finite() || bin.probability <= 0.0 || bin.probability > 1.0 {
                return Err("speaker-count posterior probability is not finite in 0..=1".to_owned());
            }
            previous_count = Some(bin.count);
            concrete_mass += bin.probability;
            entropy_bits += entropy_term(bin.probability);
            if map.as_ref().is_none_or(|&(probability, count)| {
                bin.probability > probability
                    || (bin.probability.to_bits() == probability.to_bits() && bin.count < count)
            }) {
                map = Some((bin.probability, bin.count));
            }
        }
        if (concrete_mass + self.unresolved_probability - 1.0).abs() > 1.0e-9 {
            return Err(
                "speaker-count posterior and unresolved mass do not normalize to one".to_owned(),
            );
        }
        if (entropy_bits - self.entropy_bits).abs() > 1.0e-9 {
            return Err(
                "speaker-count entropy does not match retained probability mass".to_owned(),
            );
        }

        if let Some(range) = self.supported_range
            && (range.minimum > range.maximum
                || range.minimum < self.constraint_lower_bound
                || range.maximum > self.candidate_upper_bound)
        {
            return Err("speaker-count supported range lies outside candidate bounds".to_owned());
        }
        if self.posterior.is_empty() {
            if self.supported_range.is_some() || self.stability.to_bits() != 0.0_f64.to_bits() {
                return Err(
                    "speaker-count estimate without posterior mass claims a range or stability"
                        .to_owned(),
                );
            }
        } else {
            let range = self.supported_range.ok_or_else(|| {
                "speaker-count estimate with posterior mass omits its supported range".to_owned()
            })?;
            let has_minimum = self.posterior.iter().any(|bin| bin.count == range.minimum);
            let has_maximum = self.posterior.iter().any(|bin| bin.count == range.maximum);
            if !has_minimum || !has_maximum {
                return Err(
                    "speaker-count supported range endpoints lack posterior bins".to_owned(),
                );
            }
        }
        if let Some(selected) = self.selected_count {
            let Some((map_probability, map_count)) = map else {
                return Err(
                    "speaker-count selection exists without concrete posterior mass".to_owned(),
                );
            };
            if selected != map_count || map_probability < self.unresolved_probability {
                return Err(
                    "speaker-count selection is not the authoritative posterior action".to_owned(),
                );
            }
            if self
                .supported_range
                .is_none_or(|range| selected < range.minimum || selected > range.maximum)
            {
                return Err("speaker-count selection lies outside the supported range".to_owned());
            }
        }

        let mut seen_lanes = [false; 6];
        for (expected_lane_index, lane) in self.lanes.iter().enumerate() {
            if !unit_interval(lane.confidence) {
                return Err("speaker-count lane confidence is not finite in 0..=1".to_owned());
            }
            let lane_index = match lane.lane {
                SpeakerCountEvidenceLane::MergeRisk => 0,
                SpeakerCountEvidenceLane::SparseNormalizedEigengap => 1,
                SpeakerCountEvidenceLane::FeatureJackknife => 2,
                SpeakerCountEvidenceLane::EffectiveOccupancy => 3,
                SpeakerCountEvidenceLane::ConstraintGraph => 4,
                SpeakerCountEvidenceLane::CallerPrior => 5,
            };
            if lane_index != expected_lane_index {
                return Err("speaker-count evidence lanes are not canonically ordered".to_owned());
            }
            if std::mem::replace(&mut seen_lanes[lane_index], true) {
                return Err("speaker-count evidence lane is duplicated".to_owned());
            }
            if lane.available {
                if lane.unavailable_reason.is_some() {
                    return Err(
                        "available speaker-count lane carries an unavailable reason".to_owned()
                    );
                }
                if lane.proposed_count.is_none() {
                    return Err("available speaker-count lane has no proposed count".to_owned());
                }
            } else if lane.proposed_count.is_some()
                || lane.confidence != 0.0
                || lane.unavailable_reason.is_none()
            {
                return Err(
                    "unavailable speaker-count lane carries authoritative evidence".to_owned(),
                );
            }
            if lane.proposed_count.is_some_and(|count| {
                count < self.constraint_lower_bound || count > self.candidate_upper_bound
            }) {
                return Err("speaker-count lane proposal lies outside candidate bounds".to_owned());
            }
        }
        if seen_lanes.iter().any(|seen| !seen) {
            return Err("speaker-count estimate is missing a required evidence lane".to_owned());
        }
        if matches!(
            self.calibration_status,
            SpeakerCountCalibrationStatus::FixedSafeUncalibrated
                | SpeakerCountCalibrationStatus::Unavailable
        ) && (self.selected_count.is_some()
            || !self.posterior.is_empty()
            || self.supported_range.is_some()
            || self.unresolved_probability.to_bits() != 1.0_f64.to_bits()
            || self.entropy_bits.to_bits() != 0.0_f64.to_bits()
            || self.stability.to_bits() != 0.0_f64.to_bits())
        {
            return Err(
                "uncalibrated or unavailable speaker-count evidence claims authority".to_owned(),
            );
        }
        Ok(())
    }
}

fn unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Content digest for the ordered speaker-enrollment request.
///
/// Request order is intentionally part of the digest because public hint
/// evidence uses positional `hint_index` identities. The edge guard is also
/// bound because it changes which tracklets each interval may enroll.
pub(crate) fn speaker_hint_document_sha256(request: &DiarizationRequest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"speaker-enrollment-request-v2\0");
    hasher.update(request.enrollment_edge_guard_ms.to_le_bytes());
    for hint in &request.known_intervals {
        hash_diarization_field(&mut hasher, hint.speaker_ref.as_bytes());
        hasher.update(hint.start_ms.to_le_bytes());
        hasher.update(hint.end_ms.to_le_bytes());
        hasher.update([known_speaker_policy_rank(hint.policy)]);
        hasher.update(hint.confidence.to_bits().to_le_bytes());
        match hint.provenance.as_deref() {
            None => hasher.update([0]),
            Some(provenance) => {
                hasher.update([1]);
                hash_diarization_field(&mut hasher, provenance.as_bytes());
            }
        }
    }
    format!("{:x}", hasher.finalize())
}

/// Content digest binding an attribution query to its normalized input and
/// complete privacy-safe query fields.
pub(crate) fn speaker_attribution_query_sha256(
    normalized_input_sha256: &str,
    query: &SpeakerAttributionQuery,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"speaker-attribution-query-v1\0");
    hash_diarization_field(&mut hasher, normalized_input_sha256.as_bytes());
    hasher.update(query.start_ms.to_le_bytes());
    hasher.update(query.end_ms.to_le_bytes());
    hasher.update([match query.reason {
        SpeakerAttributionQueryReason::UnknownAttribution => 0,
        SpeakerAttributionQueryReason::LowConfidence => 1,
        SpeakerAttributionQueryReason::OverlapAmbiguity => 2,
    }]);
    for speaker_ref in &query.candidate_speaker_refs {
        hash_diarization_field(&mut hasher, speaker_ref.as_bytes());
    }
    hasher.update([known_speaker_policy_rank(query.suggested_policy)]);
    format!("{:x}", hasher.finalize())
}

fn hash_diarization_field(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(bytes);
}

const fn known_speaker_policy_rank(policy: KnownSpeakerPolicy) -> u8 {
    match policy {
        KnownSpeakerPolicy::HardMustLink => 0,
        KnownSpeakerPolicy::SoftEnrollment => 1,
    }
}

fn entropy_term(probability: f64) -> f64 {
    if probability > 0.0 {
        -probability * probability.log2()
    } else {
        0.0
    }
}

/// Auditable result of applying one speaker-count request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakerCountOutcome {
    pub request: SpeakerCountRequest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimate: Option<SpeakerCountEstimate>,
    pub status: SpeakerCountOutcomeStatus,
    pub supported_speaker_count: u32,
    pub active_speaker_refs: Vec<String>,
    pub dominant_speaker_share: f64,
    pub unknown_voiced_share: f64,
    pub reasons: Vec<SpeakerCountOutcomeReason>,
    pub speaker_evidence: Vec<SpeakerEvidenceSummary>,
}

/// Stable, payload-free report code for a neural-to-acoustic fallback.
pub const DIARIZATION_DIAGNOSTIC_NATIVE_ACOUSTIC_FALLBACK: &str = "native_acoustic_fallback";
/// Stable, payload-free report code for a fallback into an external backend.
pub const DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_FALLBACK: &str = "external_backend_fallback";
/// Stable, payload-free report code for an unavailable external backend.
pub const DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_UNAVAILABLE: &str =
    "external_backend_unavailable";
/// Stable, payload-free report code for unavailable neural identity evidence.
pub const DIARIZATION_DIAGNOSTIC_NEURAL_IDENTITY_UNAVAILABLE: &str = "neural_identity_unavailable";
/// Stable, payload-free report code for accepted external speaker attribution.
pub const DIARIZATION_DIAGNOSTIC_EXTERNAL_ATTRIBUTION_ACCEPTED: &str =
    "external_backend_attribution_accepted";

const DIARIZATION_DIAGNOSTIC_CODES: [&str; 5] = [
    DIARIZATION_DIAGNOSTIC_NATIVE_ACOUSTIC_FALLBACK,
    DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_FALLBACK,
    DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_UNAVAILABLE,
    DIARIZATION_DIAGNOSTIC_NEURAL_IDENTITY_UNAVAILABLE,
    DIARIZATION_DIAGNOSTIC_EXTERNAL_ATTRIBUTION_ACCEPTED,
];

/// Complete typed diarization result attached to [`TranscriptionResult`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiarizationReport {
    pub implementation: String,
    pub contract_version: String,
    pub feature_schema: String,
    pub speaker_evidence_mode: DiarizationSpeakerEvidenceMode,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operational_partition: Option<DiarizationOperationalPartitionSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neural_representation: Option<NeuralSpeakerRepresentationSummary>,
    #[serde(default)]
    pub diagnostics: Vec<String>,
}

impl DiarizationReport {
    /// Validate the complete typed contract before a report crosses a durable
    /// public boundary.
    pub fn validate(&self) -> Result<(), String> {
        if !lowercase_sha256(&self.normalized_input_sha256) {
            return Err("normalized input digest is not lowercase SHA-256".to_owned());
        }
        if self
            .hint_document_sha256
            .as_ref()
            .is_some_and(|digest| !lowercase_sha256(digest))
        {
            return Err("hint document digest is not lowercase SHA-256".to_owned());
        }

        let report_kind = classify_diarization_report(self)?;
        if let Some(estimate) = self.speaker_count.estimate.as_ref() {
            estimate
                .validate()
                .map_err(|error| format!("invalid speaker-count estimate: {error}"))?;
        }
        if let Some(partition) = self.operational_partition.as_ref() {
            partition
                .validate()
                .map_err(|error| format!("invalid operational partition: {error}"))?;
        }
        if let Some(summary) = self.neural_representation.as_ref() {
            summary
                .validate()
                .map_err(|error| format!("invalid neural representation: {error}"))?;
            validate_neural_provider_binding(summary)?;
        }

        validate_diarization_turns(&self.turns)?;
        validate_speaker_profiles(&self.profiles)?;
        validate_diarization_diagnostics(&self.diagnostics)?;
        validate_hint_evidence(&self.hint_evidence)?;
        validate_speaker_queries(&self.normalized_input_sha256, &self.speaker_queries)?;
        validate_speaker_count_outcome(&self.speaker_count, report_kind)?;
        validate_speaker_count_estimate_request_binding(&self.speaker_count)?;
        validate_fallback_status_binding(self.fallback_status, &self.speaker_count)?;
        validate_diarization_references(self, report_kind)?;

        if self.hint_document_sha256.is_some() != !self.hint_evidence.is_empty() {
            return Err(
                "hint document digest and hint evidence must either both be present or both be absent"
                    .to_owned(),
            );
        }
        if let Some(partition) = self.operational_partition.as_ref() {
            let estimate = self.speaker_count.estimate.as_ref().ok_or_else(|| {
                "operational partition exists without a speaker-count estimate".to_owned()
            })?;
            if partition.calibration_sha256 != estimate.calibration_sha256 {
                return Err(
                    "operational partition and speaker-count estimate calibration digests differ"
                        .to_owned(),
                );
            }
            if partition.authority != estimate.calibration_status {
                return Err(
                    "operational partition authority differs from the speaker-count estimate"
                        .to_owned(),
                );
            }
            if partition.selected_count < estimate.constraint_lower_bound
                || partition.selected_count > estimate.candidate_upper_bound
            {
                return Err(
                    "operational partition count lies outside speaker-count candidate bounds"
                        .to_owned(),
                );
            }
            if partition.selected_count > estimate.resources.prototype_count {
                return Err(
                    "operational partition count exceeds the reported affinity-node count"
                        .to_owned(),
                );
            }
            if partition.method == DiarizationOperationalPartitionMethod::FixedSafeAgglomerative
                && (estimate.selected_count.is_some()
                    || estimate.supported_range.is_some()
                    || !estimate.posterior.is_empty()
                    || estimate.unresolved_probability.to_bits() != 1.0_f64.to_bits()
                    || estimate.entropy_bits.to_bits() != 0.0_f64.to_bits()
                    || estimate.stability.to_bits() != 0.0_f64.to_bits())
            {
                return Err(
                    "fixed-safe operational partition is paired with authoritative posterior fields"
                        .to_owned(),
                );
            }
        }

        validate_report_calibration_binding(self)?;
        validate_report_kind_invariants(self, report_kind)?;
        Ok(())
    }

    /// Validate a report against the exact request that authorized its
    /// execution. When the normalized duration is unavailable, the greatest
    /// request/report interval end is used only as a conservative lower bound
    /// so non-duration request invariants can still be checked.
    pub fn validate_against_request(
        &self,
        request: &DiarizationRequest,
        audio_duration_ms: Option<u64>,
    ) -> Result<(), String> {
        self.validate()?;

        let observed_duration_lower_bound = request
            .known_intervals
            .iter()
            .map(|hint| hint.end_ms)
            .chain(self.turns.iter().map(|turn| turn.end_ms))
            .chain(self.speaker_queries.iter().map(|query| query.end_ms))
            .max()
            .unwrap_or(0);
        if let Some(duration_ms) = audio_duration_ms
            && duration_ms < observed_duration_lower_bound
        {
            return Err(format!(
                "diarization request/report interval end {observed_duration_lower_bound} exceeds audio duration {duration_ms}"
            ));
        }
        request
            .validate(audio_duration_ms.unwrap_or(observed_duration_lower_bound))
            .map_err(|error| format!("invalid diarization request: {error}"))?;

        if let Some(duration_ms) = audio_duration_ms
            && (self
                .profiles
                .iter()
                .any(|profile| profile.voiced_duration_ms > duration_ms)
                || self
                    .speaker_count
                    .speaker_evidence
                    .iter()
                    .any(|evidence| evidence.voiced_duration_ms > duration_ms))
        {
            return Err(
                "diarization speaker evidence duration exceeds the normalized audio duration"
                    .to_owned(),
            );
        }

        if self.speaker_count.request != request.speaker_count {
            return Err(
                "diarization report speaker-count request differs from the execution request"
                    .to_owned(),
            );
        }

        let report_kind = classify_diarization_report(self)?;
        validate_report_request_kind_binding(self, request, report_kind)?;
        validate_report_request_diagnostic_binding(self, request, report_kind)?;
        validate_report_hint_binding(self, request, report_kind)?;
        validate_report_request_resource_binding(self, request)?;
        validate_error_fallback_policy_binding(self, request, report_kind)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiarizationReportKind {
    NativeAcoustic,
    NativeEcapaOnly,
    NativeEcapaFused,
    NativeEcapaUnavailable,
    External,
    FallbackUnknown,
}

fn classify_diarization_report(
    report: &DiarizationReport,
) -> Result<DiarizationReportKind, String> {
    const ACOUSTIC_CONTRACT: &str = "acoustic-diarization-v3";
    const NEURAL_CONTRACT: &str = "neural-diarization-common-v2";
    const ACOUSTIC_FEATURE_V1: &str = "acoustic-feature-v1";
    const ACOUSTIC_FEATURE_V2: &str = "acoustic-feature-v2";

    let (kind, contract, feature_schema, evidence_mode) = match report.implementation.as_str() {
        "native-acoustic-v1" => (
            DiarizationReportKind::NativeAcoustic,
            ACOUSTIC_CONTRACT,
            ACOUSTIC_FEATURE_V1,
            DiarizationSpeakerEvidenceMode::AcousticV2,
        ),
        "native-acoustic-v2" => (
            DiarizationReportKind::NativeAcoustic,
            ACOUSTIC_CONTRACT,
            ACOUSTIC_FEATURE_V2,
            DiarizationSpeakerEvidenceMode::AcousticV2,
        ),
        "native-ecapa-only-v1" => (
            DiarizationReportKind::NativeEcapaOnly,
            NEURAL_CONTRACT,
            ACOUSTIC_FEATURE_V2,
            DiarizationSpeakerEvidenceMode::EcapaOnly,
        ),
        "native-ecapa-fused-v1" => (
            DiarizationReportKind::NativeEcapaFused,
            NEURAL_CONTRACT,
            ACOUSTIC_FEATURE_V2,
            DiarizationSpeakerEvidenceMode::EcapaWithAcousticChannel,
        ),
        "native-ecapa-unavailable-v1" => (
            DiarizationReportKind::NativeEcapaUnavailable,
            NEURAL_CONTRACT,
            ACOUSTIC_FEATURE_V2,
            DiarizationSpeakerEvidenceMode::None,
        ),
        "external-backend" => (
            DiarizationReportKind::External,
            ACOUSTIC_CONTRACT,
            "external-unreported",
            DiarizationSpeakerEvidenceMode::External,
        ),
        "fallback-unknown" => (
            DiarizationReportKind::FallbackUnknown,
            ACOUSTIC_CONTRACT,
            ACOUSTIC_FEATURE_V2,
            DiarizationSpeakerEvidenceMode::None,
        ),
        implementation => {
            return Err(format!(
                "diarization implementation {implementation:?} is not admitted by the current report contract"
            ));
        }
    };
    if report.contract_version != contract
        || report.feature_schema != feature_schema
        || report.speaker_evidence_mode != evidence_mode
    {
        return Err(format!(
            "diarization implementation {} requires contract {contract}, feature schema {feature_schema}, and evidence mode {evidence_mode:?}",
            report.implementation
        ));
    }
    Ok(kind)
}

fn validate_neural_provider_binding(
    summary: &NeuralSpeakerRepresentationSummary,
) -> Result<(), String> {
    validate_current_ecapa_provider(summary)
}

fn validate_current_ecapa_provider(
    summary: &NeuralSpeakerRepresentationSummary,
) -> Result<(), String> {
    if summary.provider_version != crate::diarization::ECAPA_SPEAKER_REPRESENTATION_VERSION {
        return Err("ECAPA execution declares an unrecognized provider version".to_owned());
    }
    let admitted_package_sha256 = crate::ecapa_conformance::ECAPA_PACKAGE_SHA256;
    if summary.expected_model_package_sha256 != admitted_package_sha256 {
        return Err(
            "known ECAPA provider declares an unrecognized expected model package".to_owned(),
        );
    }
    if summary
        .loaded_model_package_sha256
        .as_deref()
        .is_some_and(|digest| digest != admitted_package_sha256)
    {
        return Err("known ECAPA provider claims an unrecognized loaded model package".to_owned());
    }
    Ok(())
}

fn validate_report_calibration_binding(report: &DiarizationReport) -> Result<(), String> {
    let expected_calibration_sha256 = match report.speaker_evidence_mode {
        DiarizationSpeakerEvidenceMode::AcousticV2 => {
            Some(crate::diarization::acoustic_speaker_pair_calibration_sha256())
        }
        DiarizationSpeakerEvidenceMode::EcapaOnly
        | DiarizationSpeakerEvidenceMode::EcapaWithAcousticChannel => Some(
            crate::diarization::ecapa_speaker_pair_calibration_sha256(report.speaker_evidence_mode),
        ),
        DiarizationSpeakerEvidenceMode::External | DiarizationSpeakerEvidenceMode::None => None,
    };
    let Some(expected_calibration_sha256) = expected_calibration_sha256 else {
        return Ok(());
    };

    if report
        .speaker_count
        .estimate
        .as_ref()
        .is_some_and(|estimate| estimate.calibration_sha256 != expected_calibration_sha256)
    {
        return Err(
            "speaker-count estimate calibration digest is incompatible with the evidence mode"
                .to_owned(),
        );
    }
    if report
        .operational_partition
        .as_ref()
        .is_some_and(|partition| partition.calibration_sha256 != expected_calibration_sha256)
    {
        return Err(
            "operational partition calibration digest is incompatible with the evidence mode"
                .to_owned(),
        );
    }
    Ok(())
}

fn validate_report_request_kind_binding(
    report: &DiarizationReport,
    request: &DiarizationRequest,
    kind: DiarizationReportKind,
) -> Result<(), String> {
    match kind {
        DiarizationReportKind::NativeAcoustic => match request.engine {
            DiarizationEngine::Auto | DiarizationEngine::Acoustic => {
                if report.neural_representation.is_some() {
                    return Err(
                        "direct native acoustic report claims an unrequested neural attempt"
                            .to_owned(),
                    );
                }
            }
            DiarizationEngine::Ecapa | DiarizationEngine::EcapaFused
                if request.fallback == DiarizationFallbackPolicy::Acoustic =>
            {
                let summary = report.neural_representation.as_ref().ok_or_else(|| {
                    "neural-to-acoustic fallback report omits neural attempt provenance".to_owned()
                })?;
                validate_current_ecapa_provider(summary)?;
            }
            _ => {
                return Err(
                    "native acoustic implementation is incompatible with the requested engine and fallback"
                        .to_owned(),
                );
            }
        },
        DiarizationReportKind::NativeEcapaOnly => {
            if request.engine != DiarizationEngine::Ecapa {
                return Err(
                    "ECAPA-only implementation is incompatible with the requested engine"
                        .to_owned(),
                );
            }
        }
        DiarizationReportKind::NativeEcapaFused => {
            if request.engine != DiarizationEngine::EcapaFused {
                return Err(
                    "fused ECAPA implementation is incompatible with the requested engine"
                        .to_owned(),
                );
            }
        }
        DiarizationReportKind::NativeEcapaUnavailable => {
            if !matches!(
                request.engine,
                DiarizationEngine::Ecapa | DiarizationEngine::EcapaFused
            ) || request.fallback != DiarizationFallbackPolicy::Unknown
            {
                return Err(
                    "unavailable ECAPA implementation is incompatible with the requested engine and fallback"
                        .to_owned(),
                );
            }
        }
        DiarizationReportKind::External => {
            validate_external_request_semantics(request)?;
            match request.engine {
                DiarizationEngine::Auto | DiarizationEngine::External => {
                    if report.neural_representation.is_some() {
                        return Err(
                            "direct external report claims an unrequested neural attempt"
                                .to_owned(),
                        );
                    }
                }
                DiarizationEngine::Acoustic
                    if request.fallback == DiarizationFallbackPolicy::External =>
                {
                    if report.neural_representation.is_some() {
                        return Err(
                            "acoustic-to-external fallback report claims a neural attempt"
                                .to_owned(),
                        );
                    }
                }
                DiarizationEngine::Ecapa | DiarizationEngine::EcapaFused
                    if request.fallback == DiarizationFallbackPolicy::External =>
                {
                    let summary = report.neural_representation.as_ref().ok_or_else(|| {
                        "neural-to-external fallback report omits neural attempt provenance"
                            .to_owned()
                    })?;
                    validate_current_ecapa_provider(summary)?;
                }
                _ => {
                    return Err(
                        "external implementation is incompatible with the requested engine and fallback"
                            .to_owned(),
                    );
                }
            }
        }
        DiarizationReportKind::FallbackUnknown => {
            validate_external_request_semantics(request)?;
            if !matches!(
                request.engine,
                DiarizationEngine::Auto | DiarizationEngine::External
            ) || request.fallback != DiarizationFallbackPolicy::Unknown
            {
                return Err(
                    "unknown fallback implementation is incompatible with the requested engine and fallback"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

fn validate_report_request_diagnostic_binding(
    report: &DiarizationReport,
    request: &DiarizationRequest,
    kind: DiarizationReportKind,
) -> Result<(), String> {
    let expected: &[&str] = match kind {
        DiarizationReportKind::NativeAcoustic
            if matches!(
                request.engine,
                DiarizationEngine::Ecapa | DiarizationEngine::EcapaFused
            ) =>
        {
            &[DIARIZATION_DIAGNOSTIC_NATIVE_ACOUSTIC_FALLBACK]
        }
        DiarizationReportKind::NativeAcoustic
        | DiarizationReportKind::NativeEcapaOnly
        | DiarizationReportKind::NativeEcapaFused => &[],
        DiarizationReportKind::NativeEcapaUnavailable => {
            &[DIARIZATION_DIAGNOSTIC_NEURAL_IDENTITY_UNAVAILABLE]
        }
        DiarizationReportKind::External
            if matches!(
                request.engine,
                DiarizationEngine::Acoustic
                    | DiarizationEngine::Ecapa
                    | DiarizationEngine::EcapaFused
            ) =>
        {
            &[
                DIARIZATION_DIAGNOSTIC_EXTERNAL_ATTRIBUTION_ACCEPTED,
                DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_FALLBACK,
            ]
        }
        DiarizationReportKind::External => &[DIARIZATION_DIAGNOSTIC_EXTERNAL_ATTRIBUTION_ACCEPTED],
        DiarizationReportKind::FallbackUnknown => {
            &[DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_UNAVAILABLE]
        }
    };
    if !diagnostic_codes_equal(&report.diagnostics, expected) {
        return Err(
            "diarization diagnostic codes disagree with the authorized execution path".to_owned(),
        );
    }
    Ok(())
}

fn validate_external_request_semantics(request: &DiarizationRequest) -> Result<(), String> {
    if !request.known_intervals.is_empty() {
        return Err(
            "external diarization cannot preserve known-speaker interval semantics".to_owned(),
        );
    }
    if matches!(
        request.speaker_count,
        SpeakerCountRequest::Prior { .. } | SpeakerCountRequest::Range { .. }
    ) {
        return Err(
            "external diarization cannot preserve soft speaker-count request semantics".to_owned(),
        );
    }
    Ok(())
}

fn validate_report_request_resource_binding(
    report: &DiarizationReport,
    request: &DiarizationRequest,
) -> Result<(), String> {
    if let Some(estimate) = report.speaker_count.estimate.as_ref() {
        if estimate.resources.prototype_count > u32::from(request.max_prototypes) {
            return Err(
                "speaker-count estimate exceeds the request's global prototype resource cap"
                    .to_owned(),
            );
        }
        let hard_speaker_count = request
            .known_intervals
            .iter()
            .filter(|hint| hint.policy == KnownSpeakerPolicy::HardMustLink)
            .map(|hint| hint.speaker_ref.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let expected_constraint_lower_bound = match &request.speaker_count {
            SpeakerCountRequest::HardConstraint { count } => *count,
            SpeakerCountRequest::Infer
            | SpeakerCountRequest::Prior { .. }
            | SpeakerCountRequest::Range { .. } => u32::try_from(hard_speaker_count.max(1))
                .map_err(|_| "hard speaker count exceeds the report schema".to_owned())?,
        };
        if estimate.constraint_lower_bound != expected_constraint_lower_bound {
            return Err(
                "speaker-count estimate constraint lower bound disagrees with the execution request"
                    .to_owned(),
            );
        }
        let expected_candidate_upper_bound = canonical_speaker_count_candidate_upper_bound(
            &request.speaker_count,
            expected_constraint_lower_bound,
            estimate,
        )?;
        if estimate.candidate_upper_bound != expected_candidate_upper_bound {
            return Err(
                "speaker-count estimate candidate upper bound disagrees with the execution request and observed resources"
                    .to_owned(),
            );
        }
    }
    Ok(())
}

fn canonical_speaker_count_candidate_upper_bound(
    request: &SpeakerCountRequest,
    constraint_lower_bound: u32,
    estimate: &SpeakerCountEstimate,
) -> Result<u32, String> {
    if let SpeakerCountRequest::HardConstraint { count } = request {
        return Ok(*count);
    }

    let prototype_count = estimate.resources.prototype_count;
    let candidate_upper_bound = match estimate.calibration_status {
        SpeakerCountCalibrationStatus::DevelopmentUncertified => match request {
            SpeakerCountRequest::Infer => prototype_count.min(8),
            SpeakerCountRequest::Prior { .. } | SpeakerCountRequest::Range { .. } => {
                prototype_count.min(MAX_SPEAKER_COUNT)
            }
            SpeakerCountRequest::HardConstraint { .. } => {
                return Err("hard speaker-count policy was not finalized exactly".to_owned());
            }
        },
        SpeakerCountCalibrationStatus::FixedSafeUncalibrated
        | SpeakerCountCalibrationStatus::Unavailable => {
            let available_candidates = prototype_count.clamp(1, MAX_SPEAKER_COUNT);
            match request {
                SpeakerCountRequest::Infer => available_candidates,
                SpeakerCountRequest::Prior { bins } => bins
                    .last()
                    .ok_or_else(|| "speaker-count prior is empty".to_owned())?
                    .count
                    .max(available_candidates),
                SpeakerCountRequest::Range { maximum, .. } => (*maximum).max(available_candidates),
                SpeakerCountRequest::HardConstraint { .. } => {
                    return Err("hard speaker-count policy was not finalized exactly".to_owned());
                }
            }
        }
        SpeakerCountCalibrationStatus::Certified => {
            return Err(
                "certified speaker-count estimates lack an admitted candidate policy".to_owned(),
            );
        }
    };
    Ok(candidate_upper_bound.max(constraint_lower_bound))
}

fn validate_error_fallback_policy_binding(
    report: &DiarizationReport,
    request: &DiarizationRequest,
    kind: DiarizationReportKind,
) -> Result<(), String> {
    if request.fallback != DiarizationFallbackPolicy::Error {
        return Ok(());
    }
    if report.speaker_count.status == SpeakerCountOutcomeStatus::Unsatisfied {
        return Err(
            "error fallback policy cannot return unsatisfied speaker-count constraints".to_owned(),
        );
    }
    if report.speaker_count.status == SpeakerCountOutcomeStatus::Unresolved {
        return Err(
            "error fallback policy cannot return unresolved speaker-count evidence".to_owned(),
        );
    }
    if matches!(
        kind,
        DiarizationReportKind::NativeAcoustic
            | DiarizationReportKind::NativeEcapaOnly
            | DiarizationReportKind::NativeEcapaFused
            | DiarizationReportKind::NativeEcapaUnavailable
    ) && report.fallback_status != DiarizationFallbackStatus::NotNeeded
    {
        return Err("error fallback policy cannot return a native fallback report".to_owned());
    }
    Ok(())
}

fn validate_report_hint_binding(
    report: &DiarizationReport,
    request: &DiarizationRequest,
    kind: DiarizationReportKind,
) -> Result<(), String> {
    if matches!(
        kind,
        DiarizationReportKind::External | DiarizationReportKind::FallbackUnknown
    ) {
        return Ok(());
    }

    let expected_digest =
        (!request.known_intervals.is_empty()).then(|| speaker_hint_document_sha256(request));
    if report.hint_document_sha256 != expected_digest {
        return Err("diarization report hint digest differs from the execution request".to_owned());
    }
    if report.hint_evidence.len() != request.known_intervals.len() {
        return Err(
            "diarization report hint evidence does not cover every requested interval".to_owned(),
        );
    }
    for (index, (evidence, hint)) in report
        .hint_evidence
        .iter()
        .zip(&request.known_intervals)
        .enumerate()
    {
        let expected_index = u64::try_from(index)
            .map_err(|_| "diarization hint index exceeds the report schema".to_owned())?;
        if evidence.hint_index != expected_index
            || evidence.speaker_ref != hint.speaker_ref
            || evidence.policy != hint.policy
        {
            return Err(format!(
                "diarization hint evidence {index} does not identify its requested interval"
            ));
        }
        if hint.policy == KnownSpeakerPolicy::HardMustLink {
            match evidence.disposition {
                SpeakerHintDisposition::HardAttributed => {
                    validate_hard_hint_attribution(report, hint)?;
                }
                SpeakerHintDisposition::NoUsableTracklets => {}
                _ => {
                    return Err(format!(
                        "hard diarization hint evidence {index} has an incompatible disposition"
                    ));
                }
            }
        }
    }
    for evidence in report
        .speaker_count
        .speaker_evidence
        .iter()
        .filter(|evidence| evidence.hard_anchored)
    {
        if !request
            .known_intervals
            .iter()
            .zip(&report.hint_evidence)
            .any(|(hint, hint_evidence)| {
                hint.speaker_ref == evidence.speaker_ref
                    && hint.policy == KnownSpeakerPolicy::HardMustLink
                    && hint_evidence.disposition == SpeakerHintDisposition::HardAttributed
            })
        {
            return Err(
                "hard-anchored speaker evidence lacks hard-attributed request evidence".to_owned(),
            );
        }
    }
    for turn in &report.turns {
        if turn.hard_hint_attributed
            && !request
                .known_intervals
                .iter()
                .zip(&report.hint_evidence)
                .any(|(hint, evidence)| {
                    hint.policy == KnownSpeakerPolicy::HardMustLink
                        && evidence.disposition == SpeakerHintDisposition::HardAttributed
                        && turn.speaker_ref.as_deref() == Some(hint.speaker_ref.as_str())
                        && turn.start_ms < hint.end_ms
                        && hint.start_ms < turn.end_ms
                })
        {
            return Err(
                "hard-attributed diarization turn is not bound to an overlapping hard hint"
                    .to_owned(),
            );
        }
    }
    Ok(())
}

fn validate_hard_hint_attribution(
    report: &DiarizationReport,
    hint: &KnownSpeakerInterval,
) -> Result<(), String> {
    let has_supported_anchor = report
        .speaker_count
        .speaker_evidence
        .iter()
        .any(|evidence| {
            evidence.speaker_ref == hint.speaker_ref
                && evidence.supported
                && evidence.hard_anchored
                && evidence
                    .reasons
                    .contains(&SpeakerEvidenceReason::SupportedByHardHint)
        });
    if !has_supported_anchor {
        return Err("hard-attributed hint lacks active, supported hard-anchor evidence".to_owned());
    }
    if !report.turns.iter().any(|turn| {
        turn.hard_hint_attributed
            && turn.speaker_ref.as_deref() == Some(hint.speaker_ref.as_str())
            && turn.start_ms < hint.end_ms
            && hint.start_ms < turn.end_ms
    }) {
        return Err(
            "hard-attributed hint lacks an overlapping hard-attributed speaker turn".to_owned(),
        );
    }
    Ok(())
}

fn validate_report_kind_invariants(
    report: &DiarizationReport,
    kind: DiarizationReportKind,
) -> Result<(), String> {
    let diagnostic_shape_is_admitted = match kind {
        DiarizationReportKind::NativeAcoustic => {
            (report.neural_representation.is_none() && report.diagnostics.is_empty())
                || (report.neural_representation.is_some()
                    && diagnostic_codes_equal(
                        &report.diagnostics,
                        &[DIARIZATION_DIAGNOSTIC_NATIVE_ACOUSTIC_FALLBACK],
                    ))
        }
        DiarizationReportKind::NativeEcapaOnly | DiarizationReportKind::NativeEcapaFused => {
            report.diagnostics.is_empty()
        }
        DiarizationReportKind::NativeEcapaUnavailable => diagnostic_codes_equal(
            &report.diagnostics,
            &[DIARIZATION_DIAGNOSTIC_NEURAL_IDENTITY_UNAVAILABLE],
        ),
        DiarizationReportKind::External => {
            (report.neural_representation.is_none()
                && diagnostic_codes_equal(
                    &report.diagnostics,
                    &[DIARIZATION_DIAGNOSTIC_EXTERNAL_ATTRIBUTION_ACCEPTED],
                ))
                || diagnostic_codes_equal(
                    &report.diagnostics,
                    &[
                        DIARIZATION_DIAGNOSTIC_EXTERNAL_ATTRIBUTION_ACCEPTED,
                        DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_FALLBACK,
                    ],
                )
        }
        DiarizationReportKind::FallbackUnknown => diagnostic_codes_equal(
            &report.diagnostics,
            &[DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_UNAVAILABLE],
        ),
    };
    if !diagnostic_shape_is_admitted {
        return Err(
            "diarization diagnostic codes are not canonical for the report implementation"
                .to_owned(),
        );
    }

    if kind != DiarizationReportKind::External
        && (report
            .speaker_count
            .reasons
            .contains(&SpeakerCountOutcomeReason::ExternalAttribution)
            || report
                .speaker_count
                .speaker_evidence
                .iter()
                .any(|evidence| {
                    evidence
                        .reasons
                        .contains(&SpeakerEvidenceReason::SupportedByExternalAttribution)
                }))
    {
        return Err(
            "non-external diarization report claims external attribution evidence".to_owned(),
        );
    }

    let native_success = matches!(
        kind,
        DiarizationReportKind::NativeAcoustic
            | DiarizationReportKind::NativeEcapaOnly
            | DiarizationReportKind::NativeEcapaFused
    );
    if native_success && report.speaker_count.estimate.is_none() {
        return Err("native diarization reports require a speaker-count estimate".to_owned());
    }
    if native_success && report.fallback_status == DiarizationFallbackStatus::ExternalBackend {
        return Err("native diarization cannot claim external-backend fallback status".to_owned());
    }
    if native_success
        && matches!(
            report.fallback_status,
            DiarizationFallbackStatus::InsufficientEvidence
                | DiarizationFallbackStatus::CalibrationInvalid
                | DiarizationFallbackStatus::ResourceLimit
        )
    {
        return Err(
            "native diarization claims a fallback cause that no current producer emits".to_owned(),
        );
    }
    if native_success {
        let estimate = report.speaker_count.estimate.as_ref().ok_or_else(|| {
            "native diarization reports require a speaker-count estimate".to_owned()
        })?;
        if (estimate.resources.prototype_count > 0) != report.operational_partition.is_some() {
            return Err(
                "native operational partition presence disagrees with observed prototypes"
                    .to_owned(),
            );
        }
        if estimate.resources.prototype_count == 0
            && report.speaker_count.supported_speaker_count != 0
        {
            return Err("zero-prototype native report claims supported speakers".to_owned());
        }
        if report
            .operational_partition
            .as_ref()
            .is_some_and(|partition| {
                partition.selected_count < report.speaker_count.supported_speaker_count
            })
        {
            return Err(
                "native operational partition has fewer clusters than supported speakers"
                    .to_owned(),
            );
        }
        if matches!(
            &report.speaker_count.request,
            SpeakerCountRequest::Infer
                | SpeakerCountRequest::Prior { .. }
                | SpeakerCountRequest::Range { .. }
        ) {
            let expected_status = if estimate.selected_count.is_some() {
                SpeakerCountOutcomeStatus::Resolved
            } else {
                SpeakerCountOutcomeStatus::Unresolved
            };
            if report.speaker_count.status != expected_status {
                return Err(
                    "native soft speaker-count status disagrees with the finalized estimate"
                        .to_owned(),
                );
            }
        }
        let dominance_breached = report.speaker_count.supported_speaker_count > 1
            && report.speaker_count.dominant_speaker_share > f64::from(0.98_f32);
        if report
            .speaker_count
            .reasons
            .contains(&SpeakerCountOutcomeReason::DominantSpeakerShareExceeded)
            != dominance_breached
        {
            return Err(
                "native dominance reason disagrees with supported-speaker occupancy".to_owned(),
            );
        }
        if dominance_breached && report.fallback_status == DiarizationFallbackStatus::NotNeeded {
            return Err(
                "native dominant-speaker breach cannot claim that no fallback was needed"
                    .to_owned(),
            );
        }
    }

    match kind {
        DiarizationReportKind::NativeAcoustic => {
            if report
                .operational_partition
                .as_ref()
                .is_some_and(|partition| {
                    matches!(
                        partition.method,
                        DiarizationOperationalPartitionMethod::EcapaSpherical
                            | DiarizationOperationalPartitionMethod::EcapaFusedConsensus
                    )
                })
            {
                return Err("native acoustic evidence cannot claim an ECAPA partition".to_owned());
            }
        }
        DiarizationReportKind::NativeEcapaOnly => {
            let summary = report.neural_representation.as_ref().ok_or_else(|| {
                "successful ECAPA-only diarization requires neural representation provenance"
                    .to_owned()
            })?;
            if !matches!(
                summary.status,
                NeuralSpeakerRepresentationStatus::Ready
                    | NeuralSpeakerRepresentationStatus::Degraded
            ) {
                return Err(
                    "successful ECAPA-only diarization requires ready or degraded neural evidence"
                        .to_owned(),
                );
            }
            if report
                .operational_partition
                .as_ref()
                .is_some_and(|partition| {
                    partition.method == DiarizationOperationalPartitionMethod::EcapaFusedConsensus
                })
            {
                return Err(
                    "ECAPA-only evidence cannot claim a fused consensus partition".to_owned(),
                );
            }
        }
        DiarizationReportKind::NativeEcapaFused => {
            let summary = report.neural_representation.as_ref().ok_or_else(|| {
                "successful fused ECAPA diarization requires neural representation provenance"
                    .to_owned()
            })?;
            if !matches!(
                summary.status,
                NeuralSpeakerRepresentationStatus::Ready
                    | NeuralSpeakerRepresentationStatus::Degraded
            ) {
                return Err(
                    "successful fused ECAPA diarization requires ready or degraded neural evidence"
                        .to_owned(),
                );
            }
            if report
                .operational_partition
                .as_ref()
                .is_some_and(|partition| {
                    partition.method == DiarizationOperationalPartitionMethod::EcapaSpherical
                })
            {
                return Err(
                    "fused ECAPA evidence has incompatible operational partition provenance"
                        .to_owned(),
                );
            }
        }
        DiarizationReportKind::NativeEcapaUnavailable => {
            let summary = report.neural_representation.as_ref().ok_or_else(|| {
                "unavailable ECAPA report requires neural attempt provenance".to_owned()
            })?;
            if summary.status != NeuralSpeakerRepresentationStatus::Unavailable {
                return Err(
                    "unavailable ECAPA report must carry unavailable neural provenance".to_owned(),
                );
            }
            if report.speaker_count.estimate.is_some()
                || report.operational_partition.is_some()
                || !report.profiles.is_empty()
                || !report.speaker_queries.is_empty()
                || matches!(
                    report.fallback_status,
                    DiarizationFallbackStatus::NotNeeded
                        | DiarizationFallbackStatus::ExternalBackend
                )
            {
                return Err(
                    "unavailable ECAPA report claims evidence that its fallback path cannot produce"
                        .to_owned(),
                );
            }
            if report.turns.iter().any(|turn| {
                match (turn.speaker_ref.as_ref(), turn.speaker_confidence) {
                    (None, None) => turn.hard_hint_attributed,
                    (Some(_), Some(confidence)) => {
                        !turn.hard_hint_attributed || confidence.to_bits() != 1.0_f64.to_bits()
                    }
                    _ => true,
                }
            }) || report
                .speaker_count
                .speaker_evidence
                .iter()
                .any(|evidence| {
                    !evidence.supported
                        || !evidence.hard_anchored
                        || evidence.reasons != vec![SpeakerEvidenceReason::SupportedByHardHint]
                        || !report.hint_evidence.iter().any(|hint| {
                            hint.speaker_ref == evidence.speaker_ref
                                && hint.policy == KnownSpeakerPolicy::HardMustLink
                                && hint.disposition == SpeakerHintDisposition::HardAttributed
                        })
                })
            {
                return Err(
                    "unavailable ECAPA report claims non-hard speaker identity evidence".to_owned(),
                );
            }
            let expected_fallback = match &report.speaker_count.request {
                SpeakerCountRequest::Infer
                | SpeakerCountRequest::Prior { .. }
                | SpeakerCountRequest::Range { .. } => {
                    DiarizationFallbackStatus::SpeakerCountUnresolved
                }
                SpeakerCountRequest::HardConstraint { .. }
                    if report.speaker_count.status == SpeakerCountOutcomeStatus::Satisfied =>
                {
                    DiarizationFallbackStatus::InsufficientEvidence
                }
                SpeakerCountRequest::HardConstraint { .. } => {
                    DiarizationFallbackStatus::UnsatisfiedConstraints
                }
            };
            if report.fallback_status != expected_fallback {
                return Err(
                    "unavailable ECAPA fallback status disagrees with its count request".to_owned(),
                );
            }
        }
        DiarizationReportKind::External => {
            if matches!(
                &report.speaker_count.request,
                SpeakerCountRequest::Prior { .. } | SpeakerCountRequest::Range { .. }
            ) {
                return Err(
                    "external diarization cannot preserve soft speaker-count request semantics"
                        .to_owned(),
                );
            }
            if matches!(&report.speaker_count.request, SpeakerCountRequest::Infer) {
                let expected_status =
                    if report.speaker_count.unknown_voiced_share.to_bits() == 0.0_f64.to_bits() {
                        SpeakerCountOutcomeStatus::Resolved
                    } else {
                        SpeakerCountOutcomeStatus::Unresolved
                    };
                if report.speaker_count.status != expected_status {
                    return Err(
                        "external inferred speaker-count status disagrees with attribution coverage"
                            .to_owned(),
                    );
                }
            }
            if report.fallback_status != DiarizationFallbackStatus::ExternalBackend
                || report.speaker_count.estimate.is_some()
                || report.operational_partition.is_some()
                || report.hint_document_sha256.is_some()
                || !report.hint_evidence.is_empty()
                || !report.speaker_queries.is_empty()
            {
                return Err(
                    "external diarization report claims unsupported native or hint evidence"
                        .to_owned(),
                );
            }
            if !report
                .speaker_count
                .reasons
                .contains(&SpeakerCountOutcomeReason::ExternalAttribution)
                || report
                    .speaker_count
                    .speaker_evidence
                    .iter()
                    .any(|evidence| {
                        evidence.reasons
                            != vec![SpeakerEvidenceReason::SupportedByExternalAttribution]
                    })
            {
                return Err(
                    "external diarization report has inconsistent attribution evidence".to_owned(),
                );
            }
            if report.turns.is_empty()
                || report.turns.iter().any(|turn| {
                    turn.speaker_ref.is_none()
                        || turn.speaker_confidence.is_some()
                        || turn.change_confidence.is_some()
                        || turn.overlap_suspected
                        || turn.hard_hint_attributed
                })
                || report.profiles.iter().any(|profile| {
                    profile.frame_count != 0
                        || profile.reliability.to_bits() != 0.0_f64.to_bits()
                        || profile.voice_profile_count != 0
                        || profile.channel_profile_count != 0
                        || profile.training_accepted_count != 0
                        || profile.training_downweighted_count != 0
                        || profile.training_quarantined_count != 0
                        || profile.anchored
                        || profile.soft_hint_contradiction.is_some()
                })
                || report
                    .speaker_count
                    .speaker_evidence
                    .iter()
                    .any(|evidence| {
                        evidence.voiced_frame_count != 0
                            || evidence.independent_voiced_frame_count != 0
                            || evidence.mean_assignment_confidence.to_bits() != 0.0_f64.to_bits()
                            || evidence.profile_reliability.to_bits() != 0.0_f64.to_bits()
                            || evidence.hard_anchored
                            || !evidence.separated_from_supported_speakers
                            || evidence.independent_tracklet_count
                                != evidence.assigned_tracklet_count
                            || evidence.recurrence_episode_count != evidence.assigned_tracklet_count
                    })
            {
                return Err(
                    "external diarization report claims unavailable native evidence".to_owned(),
                );
            }
            for profile in &report.profiles {
                let evidence = report
                    .speaker_count
                    .speaker_evidence
                    .iter()
                    .find(|evidence| evidence.speaker_ref == profile.speaker_ref)
                    .ok_or_else(|| {
                        "external profile has no matching speaker evidence".to_owned()
                    })?;
                let matching_turns = report
                    .turns
                    .iter()
                    .filter(|turn| turn.speaker_ref.as_ref() == Some(&profile.speaker_ref))
                    .collect::<Vec<_>>();
                let duration = matching_turns.iter().try_fold(0u64, |total, turn| {
                    total.checked_add(turn.end_ms - turn.start_ms)
                });
                if u64::try_from(matching_turns.len()).ok()
                    != Some(evidence.assigned_tracklet_count)
                    || duration != Some(profile.voiced_duration_ms)
                    || evidence.voiced_duration_ms != profile.voiced_duration_ms
                {
                    return Err(
                        "external turn totals disagree with profile and evidence summaries"
                            .to_owned(),
                    );
                }
            }
            if report.speaker_count.unknown_voiced_share.to_bits() != 0.0_f64.to_bits() {
                let expected_status = match &report.speaker_count.request {
                    SpeakerCountRequest::HardConstraint { .. } => {
                        SpeakerCountOutcomeStatus::Unsatisfied
                    }
                    SpeakerCountRequest::Infer
                    | SpeakerCountRequest::Prior { .. }
                    | SpeakerCountRequest::Range { .. } => SpeakerCountOutcomeStatus::Unresolved,
                };
                if report.speaker_count.status != expected_status {
                    return Err(
                        "partial external attribution claims a resolved speaker count".to_owned(),
                    );
                }
            }
        }
        DiarizationReportKind::FallbackUnknown => {
            let exact_fallback = match &report.speaker_count.request {
                SpeakerCountRequest::Infer
                    if report.speaker_count.status == SpeakerCountOutcomeStatus::Unresolved =>
                {
                    DiarizationFallbackStatus::InsufficientEvidence
                }
                SpeakerCountRequest::HardConstraint { .. }
                    if report.speaker_count.status == SpeakerCountOutcomeStatus::Unsatisfied =>
                {
                    DiarizationFallbackStatus::UnsatisfiedConstraints
                }
                SpeakerCountRequest::Prior { .. } | SpeakerCountRequest::Range { .. } => {
                    return Err(
                        "unknown fallback cannot preserve soft speaker-count request semantics"
                            .to_owned(),
                    );
                }
                SpeakerCountRequest::Infer | SpeakerCountRequest::HardConstraint { .. } => {
                    return Err(
                        "unknown fallback speaker-count status disagrees with its request"
                            .to_owned(),
                    );
                }
            };
            if report.fallback_status != exact_fallback {
                return Err(
                    "unknown fallback cause disagrees with its speaker-count request".to_owned(),
                );
            }
            if report.neural_representation.is_some()
                || report.speaker_count.estimate.is_some()
                || report.operational_partition.is_some()
                || report.hint_document_sha256.is_some()
                || !report.turns.is_empty()
                || !report.profiles.is_empty()
                || !report.hint_evidence.is_empty()
                || !report.speaker_queries.is_empty()
                || report.speaker_count.supported_speaker_count != 0
                || !report.speaker_count.active_speaker_refs.is_empty()
                || !report.speaker_count.speaker_evidence.is_empty()
                || report.speaker_count.unknown_voiced_share.to_bits() != 1.0_f64.to_bits()
                || matches!(
                    report.fallback_status,
                    DiarizationFallbackStatus::NotNeeded
                        | DiarizationFallbackStatus::ExternalBackend
                )
            {
                return Err(
                    "unknown fallback report claims speaker/model evidence or known voiced occupancy that it cannot produce"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

fn validate_diarization_turns(turns: &[DiarizationTurn]) -> Result<(), String> {
    for (index, turn) in turns.iter().enumerate() {
        if turn.end_ms <= turn.start_ms {
            return Err(format!(
                "diarization turn {index} has a non-positive interval"
            ));
        }
        if let Some(speaker_ref) = turn.speaker_ref.as_deref() {
            validate_report_speaker_ref(speaker_ref, "diarization turn speaker_ref")?;
        } else if turn.speaker_confidence.is_some() {
            return Err(format!(
                "diarization turn {index} has confidence without a speaker reference"
            ));
        }
        for (field, confidence) in [
            ("speaker_confidence", turn.speaker_confidence),
            ("change_confidence", turn.change_confidence),
        ] {
            if confidence.is_some_and(|value| !unit_interval(value)) {
                return Err(format!(
                    "diarization turn {index} {field} is not finite in 0..=1"
                ));
            }
        }
        if turn.hard_hint_attributed
            && (turn.speaker_ref.is_none()
                || turn.speaker_confidence.map(f64::to_bits) != Some(1.0_f64.to_bits()))
        {
            return Err(format!(
                "diarization turn {index} has inconsistent hard-hint attribution"
            ));
        }
        if let Some(previous) = index.checked_sub(1).and_then(|prior| turns.get(prior)) {
            let previous_key = (
                previous.start_ms,
                previous.end_ms,
                previous.speaker_ref.as_deref(),
            );
            let current_key = (turn.start_ms, turn.end_ms, turn.speaker_ref.as_deref());
            if previous_key > current_key {
                return Err("diarization turns are not deterministically time-ordered".to_owned());
            }
            if previous_key == current_key {
                return Err("diarization turns contain a duplicate interval and speaker".to_owned());
            }
        }
    }
    for (left_index, left) in turns.iter().enumerate() {
        for right in turns.iter().skip(left_index + 1) {
            if right.start_ms >= left.end_ms {
                break;
            }
            if left.start_ms < right.end_ms && right.start_ms < left.end_ms {
                if left.speaker_ref == right.speaker_ref {
                    return Err(
                        "overlapping diarization turns duplicate the same speaker".to_owned()
                    );
                }
                if left.speaker_ref.is_none()
                    || right.speaker_ref.is_none()
                    || !left.overlap_suspected
                    || !right.overlap_suspected
                {
                    return Err(
                        "overlapping diarization turns lack two labeled, overlap-flagged speakers"
                            .to_owned(),
                    );
                }
            }
        }
    }
    Ok(())
}

fn validate_diarization_diagnostics(diagnostics: &[String]) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for diagnostic in diagnostics {
        if !DIARIZATION_DIAGNOSTIC_CODES.contains(&diagnostic.as_str()) {
            return Err(format!(
                "diarization diagnostic {diagnostic:?} is not an admitted stable code"
            ));
        }
        if !seen.insert(diagnostic.as_str()) {
            return Err("diarization diagnostics contain a duplicate stable code".to_owned());
        }
    }
    Ok(())
}

fn diagnostic_codes_equal(actual: &[String], expected: &[&str]) -> bool {
    actual
        .iter()
        .map(String::as_str)
        .eq(expected.iter().copied())
}

fn validate_speaker_profiles(profiles: &[SpeakerProfileSummary]) -> Result<(), String> {
    let mut speaker_refs = BTreeSet::new();
    for profile in profiles {
        validate_report_speaker_ref(&profile.speaker_ref, "speaker profile speaker_ref")?;
        if !speaker_refs.insert(profile.speaker_ref.as_str()) {
            return Err("speaker profiles contain a duplicate speaker_ref".to_owned());
        }
        if !unit_interval(profile.reliability) {
            return Err("speaker profile reliability is not finite in 0..=1".to_owned());
        }
        if profile
            .soft_hint_contradiction
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err("speaker profile contradiction is not finite and non-negative".to_owned());
        }
    }
    Ok(())
}

fn validate_hint_evidence(hints: &[SpeakerHintEvidenceSummary]) -> Result<(), String> {
    for (index, hint) in hints.iter().enumerate() {
        validate_report_speaker_ref(&hint.speaker_ref, "speaker hint evidence speaker_ref")?;
        let expected_index = u64::try_from(index)
            .map_err(|_| "speaker hint evidence index exceeds the report schema".to_owned())?;
        if hint.hint_index != expected_index {
            return Err("speaker hint evidence indices are not contiguous from zero".to_owned());
        }
        if !hint.applied_weight.is_finite() || hint.applied_weight < 0.0 {
            return Err("speaker hint applied weight is not finite and non-negative".to_owned());
        }
        if hint
            .contradiction_score
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err("speaker hint contradiction is not finite and non-negative".to_owned());
        }
        let disposition_count = hint
            .accepted_tracklet_count
            .checked_add(hint.rejected_tracklet_count)
            .ok_or_else(|| "speaker hint tracklet counts overflow".to_owned())?;
        if disposition_count != hint.usable_tracklet_count {
            return Err(
                "speaker hint accepted and rejected counts do not cover usable tracklets"
                    .to_owned(),
            );
        }
        let profile_count = hint
            .profile_accepted_tracklet_count
            .checked_add(hint.profile_downweighted_tracklet_count)
            .and_then(|count| count.checked_add(hint.profile_quarantined_tracklet_count))
            .ok_or_else(|| "speaker hint profile counts overflow".to_owned())?;
        if profile_count > hint.accepted_tracklet_count {
            return Err("speaker hint profile counts exceed accepted tracklets".to_owned());
        }
        if hint.usable_tracklet_count == 0 && hint.applied_weight.to_bits() != 0.0_f64.to_bits() {
            return Err("speaker hint without usable tracklets carries applied weight".to_owned());
        }
        match hint.disposition {
            SpeakerHintDisposition::HardAttributed
                if hint.policy != KnownSpeakerPolicy::HardMustLink
                    || hint.accepted_tracklet_count != hint.usable_tracklet_count
                    || hint.rejected_tracklet_count != 0 =>
            {
                return Err("hard-attributed hint has inconsistent hard evidence".to_owned());
            }
            SpeakerHintDisposition::Accepted
                if hint.policy != KnownSpeakerPolicy::SoftEnrollment
                    || hint.accepted_tracklet_count == 0
                    || hint.rejected_tracklet_count != 0
                    || hint.profile_downweighted_tracklet_count != 0
                    || hint.profile_quarantined_tracklet_count != 0 =>
            {
                return Err("accepted speaker hint has inconsistent evidence counts".to_owned());
            }
            SpeakerHintDisposition::PartiallyAccepted
                if hint.policy != KnownSpeakerPolicy::SoftEnrollment
                    || hint.accepted_tracklet_count == 0
                    || (hint.rejected_tracklet_count == 0
                        && hint.profile_downweighted_tracklet_count == 0
                        && hint.profile_quarantined_tracklet_count == 0) =>
            {
                return Err(
                    "partially accepted speaker hint has inconsistent evidence counts".to_owned(),
                );
            }
            SpeakerHintDisposition::Rejected
                if hint.policy != KnownSpeakerPolicy::SoftEnrollment
                    || hint.usable_tracklet_count == 0
                    || hint.accepted_tracklet_count != 0 =>
            {
                return Err("rejected speaker hint has inconsistent evidence counts".to_owned());
            }
            SpeakerHintDisposition::NoUsableTracklets if hint.usable_tracklet_count != 0 => {
                return Err("no-usable-tracklets hint claims usable evidence".to_owned());
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_speaker_queries(
    normalized_input_sha256: &str,
    queries: &[SpeakerAttributionQuery],
) -> Result<(), String> {
    const MAX_SPEAKER_QUERIES: usize = 32;
    if queries.len() > MAX_SPEAKER_QUERIES {
        return Err(format!(
            "speaker attribution queries exceed the limit of {MAX_SPEAKER_QUERIES}"
        ));
    }
    let mut query_ids = BTreeSet::new();
    let mut previous_interval = None;
    for query in queries {
        if !lowercase_sha256(&query.query_id_sha256) {
            return Err("speaker attribution query ID is not lowercase SHA-256".to_owned());
        }
        if !query_ids.insert(query.query_id_sha256.as_str()) {
            return Err("speaker attribution queries contain a duplicate query ID".to_owned());
        }
        if query.end_ms <= query.start_ms {
            return Err("speaker attribution query has a non-positive interval".to_owned());
        }
        if previous_interval.is_some_and(|previous| previous > (query.start_ms, query.end_ms)) {
            return Err("speaker attribution queries are not time-ordered".to_owned());
        }
        previous_interval = Some((query.start_ms, query.end_ms));
        let mut previous_speaker = None;
        for speaker_ref in &query.candidate_speaker_refs {
            validate_report_speaker_ref(speaker_ref, "speaker query candidate speaker_ref")?;
            if previous_speaker.is_some_and(|previous: &str| previous >= speaker_ref.as_str()) {
                return Err(
                    "speaker query candidate references are not strictly ordered".to_owned(),
                );
            }
            previous_speaker = Some(speaker_ref.as_str());
        }
        let expected_candidate_count = match query.reason {
            SpeakerAttributionQueryReason::UnknownAttribution => 0,
            SpeakerAttributionQueryReason::LowConfidence => 1,
            SpeakerAttributionQueryReason::OverlapAmbiguity => 2,
        };
        if query.candidate_speaker_refs.len() != expected_candidate_count {
            return Err(
                "speaker attribution query candidate count is inconsistent with its reason"
                    .to_owned(),
            );
        }
        if query.suggested_policy != KnownSpeakerPolicy::SoftEnrollment {
            return Err(
                "machine-generated speaker attribution query suggests a hard policy".to_owned(),
            );
        }
        if query.query_id_sha256 != speaker_attribution_query_sha256(normalized_input_sha256, query)
        {
            return Err(
                "speaker attribution query ID does not bind its normalized input and fields"
                    .to_owned(),
            );
        }
    }
    Ok(())
}

fn validate_speaker_count_outcome(
    outcome: &SpeakerCountOutcome,
    kind: DiarizationReportKind,
) -> Result<(), String> {
    const F32_OCCUPANCY_SUM_TOLERANCE: f64 = 2.0 * f32::EPSILON as f64;
    let minimum_recurrence_episodes =
        crate::diarization::MIN_SPEAKER_EVIDENCE_RECURRENCE_EPISODES as u64;

    validate_speaker_count_request(&outcome.request)
        .map_err(|error| format!("invalid speaker-count request in report: {error}"))?;
    if outcome.supported_speaker_count > MAX_SPEAKER_COUNT {
        return Err(format!(
            "supported speaker count exceeds the limit of {MAX_SPEAKER_COUNT}"
        ));
    }
    if !unit_interval(outcome.dominant_speaker_share)
        || !unit_interval(outcome.unknown_voiced_share)
    {
        return Err("speaker occupancy shares are not finite in 0..=1".to_owned());
    }
    if outcome.dominant_speaker_share + outcome.unknown_voiced_share
        > 1.0 + F32_OCCUPANCY_SUM_TOLERANCE
    {
        return Err(
            "speaker occupancy shares exceed the common voiced-time denominator".to_owned(),
        );
    }
    if outcome.supported_speaker_count == 0
        && outcome.dominant_speaker_share.to_bits() != 0.0_f64.to_bits()
    {
        return Err("zero supported speakers carry a non-zero dominant share".to_owned());
    }
    if outcome.reasons.is_empty() || slice_has_duplicates(&outcome.reasons) {
        return Err("speaker-count outcome reasons are empty or duplicated".to_owned());
    }

    let mut active_refs = BTreeSet::new();
    let mut previous_active_ref = None;
    for speaker_ref in &outcome.active_speaker_refs {
        validate_report_speaker_ref(speaker_ref, "active speaker_ref")?;
        if previous_active_ref.is_some_and(|previous: &str| previous >= speaker_ref.as_str()) {
            return Err("active speaker references are not strictly ordered".to_owned());
        }
        previous_active_ref = Some(speaker_ref.as_str());
        active_refs.insert(speaker_ref.as_str());
    }
    if outcome.active_speaker_refs.len() != outcome.supported_speaker_count as usize {
        return Err("supported speaker count does not match active references".to_owned());
    }

    let mut evidence_refs = BTreeSet::new();
    let mut supported_evidence_refs = BTreeSet::new();
    for evidence in &outcome.speaker_evidence {
        validate_report_speaker_ref(&evidence.speaker_ref, "speaker evidence speaker_ref")?;
        if !evidence_refs.insert(evidence.speaker_ref.as_str()) {
            return Err("speaker evidence contains a duplicate speaker_ref".to_owned());
        }
        if evidence.independent_tracklet_count > evidence.assigned_tracklet_count
            || evidence.recurrence_episode_count > evidence.independent_tracklet_count
            || evidence.independent_voiced_frame_count > evidence.voiced_frame_count
        {
            return Err("speaker evidence independent counts exceed total counts".to_owned());
        }
        if !unit_interval(evidence.mean_assignment_confidence)
            || !unit_interval(evidence.profile_reliability)
        {
            return Err("speaker evidence confidence is not finite in 0..=1".to_owned());
        }
        if evidence.reasons.is_empty() || slice_has_duplicates(&evidence.reasons) {
            return Err("speaker evidence reasons are empty or duplicated".to_owned());
        }
        let has_no_assigned_speech = evidence
            .reasons
            .contains(&SpeakerEvidenceReason::NoAssignedSpeech);
        if has_no_assigned_speech != (evidence.assigned_tracklet_count == 0)
            && !(evidence.assigned_tracklet_count == 0
                && evidence.hard_anchored
                && evidence.supported)
        {
            return Err(
                "speaker evidence no-assigned-speech reason disagrees with its counts".to_owned(),
            );
        }
        if evidence
            .reasons
            .contains(&SpeakerEvidenceReason::SupportedByIndependentRecurrence)
            && evidence.recurrence_episode_count < minimum_recurrence_episodes
        {
            return Err(
                "independent-recurrence support lacks repeated recurrence episodes".to_owned(),
            );
        }
        if evidence
            .reasons
            .contains(&SpeakerEvidenceReason::SupportedByRepeatedTracklets)
            && evidence.independent_tracklet_count < 2
        {
            return Err("repeated-tracklet support lacks independent tracklets".to_owned());
        }
        if evidence
            .reasons
            .contains(&SpeakerEvidenceReason::SupportedByHeldoutObservation)
            && (evidence.assigned_tracklet_count == 0
                || evidence.independent_tracklet_count != 1
                || evidence.independent_voiced_frame_count == 0)
        {
            return Err("held-out speaker support lacks its observed evidence".to_owned());
        }
        let has_support_reason = evidence.reasons.iter().any(|reason| {
            matches!(
                reason,
                SpeakerEvidenceReason::SupportedByHardHint
                    | SpeakerEvidenceReason::SupportedByIndependentRecurrence
                    | SpeakerEvidenceReason::SupportedByRepeatedTracklets
                    | SpeakerEvidenceReason::SupportedByHeldoutObservation
                    | SpeakerEvidenceReason::SupportedByExternalAttribution
            )
        });
        if evidence.supported != has_support_reason {
            return Err("speaker evidence support flag disagrees with its reasons".to_owned());
        }
        let has_hard_hint_reason = evidence
            .reasons
            .contains(&SpeakerEvidenceReason::SupportedByHardHint);
        if evidence.hard_anchored != has_hard_hint_reason {
            return Err(
                "speaker evidence hard-anchor flag disagrees with its support reasons".to_owned(),
            );
        }
        if evidence.supported && !evidence.separated_from_supported_speakers {
            return Err("supported speaker evidence is not robustly separated".to_owned());
        }
        if evidence.supported
            && !evidence.hard_anchored
            && kind != DiarizationReportKind::External
            && (evidence.assigned_tracklet_count == 0
                || evidence.independent_tracklet_count == 0
                || evidence.independent_voiced_frame_count == 0
                || evidence.mean_assignment_confidence.to_bits() == 0.0_f64.to_bits()
                || evidence.profile_reliability.to_bits() == 0.0_f64.to_bits())
        {
            return Err(
                "native non-hard speaker support lacks observable identity evidence".to_owned(),
            );
        }
        if matches!(
            kind,
            DiarizationReportKind::NativeAcoustic
                | DiarizationReportKind::NativeEcapaOnly
                | DiarizationReportKind::NativeEcapaFused
        ) {
            validate_native_speaker_evidence(evidence, kind)?;
        }
        if evidence.supported {
            supported_evidence_refs.insert(evidence.speaker_ref.as_str());
        }
    }
    if active_refs != supported_evidence_refs {
        return Err("active speaker references do not match supported speaker evidence".to_owned());
    }

    let partial_external = kind == DiarizationReportKind::External
        && outcome.unknown_voiced_share.to_bits() != 0.0_f64.to_bits();
    match &outcome.request {
        SpeakerCountRequest::HardConstraint { count } => {
            let expected_status = if outcome.supported_speaker_count == *count && !partial_external
            {
                SpeakerCountOutcomeStatus::Satisfied
            } else {
                SpeakerCountOutcomeStatus::Unsatisfied
            };
            if outcome.status != expected_status {
                return Err("hard speaker-count request has inconsistent outcome status".to_owned());
            }
        }
        SpeakerCountRequest::Infer
        | SpeakerCountRequest::Prior { .. }
        | SpeakerCountRequest::Range { .. }
            if !matches!(
                outcome.status,
                SpeakerCountOutcomeStatus::Resolved | SpeakerCountOutcomeStatus::Unresolved
            ) =>
        {
            return Err(
                "soft speaker-count request has a hard-constraint outcome status".to_owned(),
            );
        }
        SpeakerCountRequest::Infer
        | SpeakerCountRequest::Prior { .. }
        | SpeakerCountRequest::Range { .. } => {}
    }
    let status_reason_count = outcome
        .reasons
        .iter()
        .filter(|reason| {
            matches!(
                reason,
                SpeakerCountOutcomeReason::EvidenceSupportedCount
                    | SpeakerCountOutcomeReason::RequestedCountMatched
                    | SpeakerCountOutcomeReason::RequestedCountMismatch
                    | SpeakerCountOutcomeReason::SpeakerCountPriorFusionUnavailable
                    | SpeakerCountOutcomeReason::SpeakerCountEvidenceUnresolved
                    | SpeakerCountOutcomeReason::NoSupportedSpeakers
            )
        })
        .count();
    if status_reason_count != 1 {
        return Err("speaker-count outcome must carry exactly one status reason".to_owned());
    }

    let status_reason_matches = match outcome.status {
        SpeakerCountOutcomeStatus::Resolved if outcome.supported_speaker_count == 0 => {
            return Err("resolved speaker-count outcome has no supported speakers".to_owned());
        }
        SpeakerCountOutcomeStatus::Resolved => outcome
            .reasons
            .contains(&SpeakerCountOutcomeReason::EvidenceSupportedCount),
        SpeakerCountOutcomeStatus::Satisfied => outcome
            .reasons
            .contains(&SpeakerCountOutcomeReason::RequestedCountMatched),
        SpeakerCountOutcomeStatus::Unsatisfied => outcome
            .reasons
            .contains(&SpeakerCountOutcomeReason::RequestedCountMismatch),
        SpeakerCountOutcomeStatus::Unresolved if outcome.supported_speaker_count == 0 => outcome
            .reasons
            .contains(&SpeakerCountOutcomeReason::NoSupportedSpeakers),
        SpeakerCountOutcomeStatus::Unresolved => outcome.reasons.iter().any(|reason| {
            matches!(
                reason,
                SpeakerCountOutcomeReason::SpeakerCountPriorFusionUnavailable
                    | SpeakerCountOutcomeReason::SpeakerCountEvidenceUnresolved
            )
        }),
    };
    if !status_reason_matches {
        return Err("speaker-count outcome status disagrees with its reasons".to_owned());
    }
    if outcome
        .reasons
        .contains(&SpeakerCountOutcomeReason::SpeakerCountPriorFusionUnavailable)
        && !matches!(outcome.request, SpeakerCountRequest::Prior { .. })
    {
        return Err(
            "speaker-count prior-fusion-unavailable reason lacks a prior request".to_owned(),
        );
    }
    let has_ambiguous_reason = outcome
        .reasons
        .contains(&SpeakerCountOutcomeReason::AmbiguousSpeakerSeparation);
    let has_merge_compatible_evidence = outcome.speaker_evidence.iter().any(|evidence| {
        evidence
            .reasons
            .contains(&SpeakerEvidenceReason::MergeCompatibleWithSupportedSpeaker)
    });
    if has_ambiguous_reason != has_merge_compatible_evidence {
        return Err(
            "speaker-count separation reason disagrees with merge-compatible evidence".to_owned(),
        );
    }
    let has_dominance_reason = outcome
        .reasons
        .contains(&SpeakerCountOutcomeReason::DominantSpeakerShareExceeded);
    if has_dominance_reason
        && (outcome.supported_speaker_count <= 1
            || outcome.dominant_speaker_share <= f64::from(0.98_f32))
    {
        return Err("dominant-speaker reason lacks a multi-speaker dominance breach".to_owned());
    }
    Ok(())
}

fn validate_native_speaker_evidence(
    evidence: &SpeakerEvidenceSummary,
    kind: DiarizationReportKind,
) -> Result<(), String> {
    let minimum_voiced_frames = crate::diarization::MIN_SPEAKER_EVIDENCE_VOICED_FRAMES as u64;
    let minimum_recurrence_episodes =
        crate::diarization::MIN_SPEAKER_EVIDENCE_RECURRENCE_EPISODES as u64;
    let minimum_assignment_confidence =
        f64::from(crate::diarization::MIN_SPEAKER_EVIDENCE_CONFIDENCE);
    let minimum_profile_reliability =
        f64::from(crate::diarization::MIN_SPEAKER_EVIDENCE_RELIABILITY);

    if evidence.hard_anchored {
        let mut expected_reasons = Vec::with_capacity(2);
        if evidence.assigned_tracklet_count == 0 {
            expected_reasons.push(SpeakerEvidenceReason::NoAssignedSpeech);
        }
        expected_reasons.push(SpeakerEvidenceReason::SupportedByHardHint);
        if evidence.reasons != expected_reasons {
            return Err(
                "native hard-anchor evidence does not carry its exact canonical reasons".to_owned(),
            );
        }
        return Ok(());
    }

    let insufficient_voiced_frames =
        evidence.independent_voiced_frame_count < minimum_voiced_frames;
    let insufficient_assignment_confidence =
        evidence.mean_assignment_confidence < minimum_assignment_confidence;
    let insufficient_profile_reliability =
        evidence.profile_reliability < minimum_profile_reliability;
    if evidence
        .reasons
        .contains(&SpeakerEvidenceReason::InsufficientVoicedFrames)
        != insufficient_voiced_frames
        || evidence
            .reasons
            .contains(&SpeakerEvidenceReason::InsufficientAssignmentConfidence)
            != insufficient_assignment_confidence
        || evidence
            .reasons
            .contains(&SpeakerEvidenceReason::InsufficientProfileReliability)
            != insufficient_profile_reliability
    {
        return Err(
            "native speaker evidence rejection reasons disagree with the exact support thresholds"
                .to_owned(),
        );
    }

    let has_merge_compatible_reason = evidence
        .reasons
        .contains(&SpeakerEvidenceReason::MergeCompatibleWithSupportedSpeaker);
    if has_merge_compatible_reason {
        if kind == DiarizationReportKind::NativeAcoustic
            && evidence.recurrence_episode_count < minimum_recurrence_episodes
            && evidence.independent_tracklet_count < 2
        {
            return Err(
                "native acoustic evidence cannot claim a singleton held-out candidate".to_owned(),
            );
        }
        if evidence.separated_from_supported_speakers
            || evidence.reasons != vec![SpeakerEvidenceReason::MergeCompatibleWithSupportedSpeaker]
        {
            return Err(
                "merge-compatible native speaker evidence is not a canonical rejected candidate"
                    .to_owned(),
            );
        }
        return Ok(());
    }
    if !evidence.separated_from_supported_speakers {
        return Err(
            "native speaker evidence lacks robust separation without a merge-compatible reason"
                .to_owned(),
        );
    }

    if evidence.supported {
        if insufficient_voiced_frames
            || insufficient_assignment_confidence
            || insufficient_profile_reliability
            || evidence.assigned_tracklet_count == 0
            || evidence.independent_tracklet_count == 0
        {
            return Err(
                "native non-hard speaker support does not meet the exact identity evidence gates"
                    .to_owned(),
            );
        }
        let expected_support_reason = if evidence.independent_tracklet_count == 1 {
            if kind == DiarizationReportKind::NativeAcoustic {
                return Err(
                    "native acoustic evidence cannot claim held-out neural speaker support"
                        .to_owned(),
                );
            }
            SpeakerEvidenceReason::SupportedByHeldoutObservation
        } else if evidence.recurrence_episode_count < minimum_recurrence_episodes {
            SpeakerEvidenceReason::SupportedByRepeatedTracklets
        } else {
            SpeakerEvidenceReason::SupportedByIndependentRecurrence
        };
        if evidence.reasons != vec![expected_support_reason] {
            return Err(
                "native non-hard speaker support does not carry its exact canonical reason"
                    .to_owned(),
            );
        }
        return Ok(());
    }

    let has_insufficient_recurrence = evidence
        .reasons
        .contains(&SpeakerEvidenceReason::InsufficientIndependentRecurrence);
    if has_insufficient_recurrence
        && (evidence.recurrence_episode_count >= minimum_recurrence_episodes
            || evidence.independent_tracklet_count >= 2)
    {
        return Err(
            "native recurrence rejection contradicts independently repeated evidence".to_owned(),
        );
    }
    if (evidence.independent_tracklet_count == 0
        || (kind == DiarizationReportKind::NativeAcoustic
            && evidence.recurrence_episode_count < minimum_recurrence_episodes
            && evidence.independent_tracklet_count < 2))
        && !has_insufficient_recurrence
    {
        return Err(
            "native speaker evidence omits its required independent-recurrence rejection"
                .to_owned(),
        );
    }
    if !evidence.reasons.iter().any(|reason| {
        matches!(
            reason,
            SpeakerEvidenceReason::NoAssignedSpeech
                | SpeakerEvidenceReason::InsufficientIndependentRecurrence
                | SpeakerEvidenceReason::InsufficientVoicedFrames
                | SpeakerEvidenceReason::InsufficientAssignmentConfidence
                | SpeakerEvidenceReason::InsufficientProfileReliability
        )
    }) {
        return Err("unsupported native speaker evidence has no rejection reason".to_owned());
    }
    Ok(())
}

fn validate_speaker_count_estimate_request_binding(
    outcome: &SpeakerCountOutcome,
) -> Result<(), String> {
    let Some(estimate) = outcome.estimate.as_ref() else {
        return Ok(());
    };
    if let Some(selected_count) = estimate.selected_count
        && selected_count != outcome.supported_speaker_count
    {
        return Err(
            "selected speaker-count estimate disagrees with supported speaker evidence".to_owned(),
        );
    }
    if let SpeakerCountRequest::HardConstraint { count } = &outcome.request
        && estimate.candidate_upper_bound != *count
    {
        return Err(
            "hard speaker-count request does not match the estimate candidate upper bound"
                .to_owned(),
        );
    }

    let caller_prior = estimate
        .lanes
        .iter()
        .find(|lane| lane.lane == SpeakerCountEvidenceLane::CallerPrior)
        .ok_or_else(|| "speaker-count estimate is missing its caller-prior lane".to_owned())?;
    let expected_caller_prior = canonical_caller_prior_lane(
        &outcome.request,
        estimate.constraint_lower_bound,
        estimate.candidate_upper_bound,
    )?;
    if caller_prior != &expected_caller_prior {
        return Err(
            "caller-prior lane disagrees with the exact authorized request and count bounds"
                .to_owned(),
        );
    }
    Ok(())
}

fn canonical_caller_prior_lane(
    request: &SpeakerCountRequest,
    constraint_lower_bound: u32,
    candidate_upper_bound: u32,
) -> Result<SpeakerCountLaneEvidence, String> {
    let unavailable = |reason| SpeakerCountLaneEvidence {
        lane: SpeakerCountEvidenceLane::CallerPrior,
        available: false,
        proposed_count: None,
        confidence: 0.0,
        unavailable_reason: Some(reason),
    };
    match request {
        SpeakerCountRequest::Infer | SpeakerCountRequest::HardConstraint { .. } => {
            Ok(unavailable(SpeakerCountLaneUnavailableReason::NotRequested))
        }
        SpeakerCountRequest::Prior { bins } => {
            let feasible = bins.iter().filter(|bin| {
                bin.count >= constraint_lower_bound && bin.count <= candidate_upper_bound
            });
            let retained_feasible_mass = feasible.clone().map(|bin| bin.probability).sum::<f64>();
            let proposed = feasible.max_by(|left, right| {
                left.probability
                    .total_cmp(&right.probability)
                    .then_with(|| right.count.cmp(&left.count))
            });
            if !retained_feasible_mass.is_finite() || retained_feasible_mass <= 0.0 {
                return Ok(unavailable(
                    SpeakerCountLaneUnavailableReason::ContradictoryConstraints,
                ));
            }
            let proposed = proposed.ok_or_else(|| {
                "positive caller-prior mass has no feasible proposed count".to_owned()
            })?;
            Ok(SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::CallerPrior,
                available: true,
                proposed_count: Some(proposed.count),
                confidence: retained_feasible_mass,
                unavailable_reason: None,
            })
        }
        SpeakerCountRequest::Range { minimum, maximum } => {
            let declared_width = maximum
                .checked_sub(*minimum)
                .and_then(|width| width.checked_add(1))
                .ok_or_else(|| "speaker-count range width overflows".to_owned())?;
            let feasible_minimum = (*minimum).max(constraint_lower_bound);
            let feasible_maximum = (*maximum).min(candidate_upper_bound);
            if feasible_minimum > feasible_maximum {
                return Ok(unavailable(
                    SpeakerCountLaneUnavailableReason::ContradictoryConstraints,
                ));
            }
            let feasible_width = feasible_maximum
                .checked_sub(feasible_minimum)
                .and_then(|width| width.checked_add(1))
                .ok_or_else(|| "feasible speaker-count range width overflows".to_owned())?;
            Ok(SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::CallerPrior,
                available: true,
                proposed_count: Some(feasible_minimum),
                confidence: f64::from(feasible_width) / f64::from(declared_width),
                unavailable_reason: None,
            })
        }
    }
}

fn validate_fallback_status_binding(
    fallback_status: DiarizationFallbackStatus,
    outcome: &SpeakerCountOutcome,
) -> Result<(), String> {
    let compatible = match fallback_status {
        DiarizationFallbackStatus::NotNeeded => matches!(
            outcome.status,
            SpeakerCountOutcomeStatus::Resolved | SpeakerCountOutcomeStatus::Satisfied
        ),
        DiarizationFallbackStatus::SpeakerCountUnresolved => {
            outcome.status == SpeakerCountOutcomeStatus::Unresolved
        }
        DiarizationFallbackStatus::UnsatisfiedConstraints => {
            outcome.status != SpeakerCountOutcomeStatus::Unresolved
        }
        DiarizationFallbackStatus::InsufficientEvidence
        | DiarizationFallbackStatus::CalibrationInvalid
        | DiarizationFallbackStatus::ResourceLimit
        | DiarizationFallbackStatus::ExternalBackend => true,
    };
    if !compatible {
        return Err("diarization fallback status disagrees with speaker-count outcome".to_owned());
    }
    Ok(())
}

fn validate_diarization_references(
    report: &DiarizationReport,
    kind: DiarizationReportKind,
) -> Result<(), String> {
    let active_refs = report
        .speaker_count
        .active_speaker_refs
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for turn in &report.turns {
        if turn
            .speaker_ref
            .as_deref()
            .is_some_and(|speaker_ref| !active_refs.contains(speaker_ref))
        {
            return Err("diarization turn references a non-active speaker".to_owned());
        }
    }
    if active_refs.iter().any(|speaker_ref| {
        !report
            .turns
            .iter()
            .any(|turn| turn.speaker_ref.as_deref() == Some(*speaker_ref))
    }) {
        return Err("active speaker reference has no emitted diarization turn".to_owned());
    }
    for query in &report.speaker_queries {
        if query
            .candidate_speaker_refs
            .iter()
            .any(|speaker_ref| !active_refs.contains(speaker_ref.as_str()))
        {
            return Err("speaker query references a non-active candidate".to_owned());
        }
        let overlaps =
            |turn: &&DiarizationTurn| turn.start_ms < query.end_ms && query.start_ms < turn.end_ms;
        let query_is_grounded = match query.reason {
            SpeakerAttributionQueryReason::UnknownAttribution => report
                .turns
                .iter()
                .filter(overlaps)
                .any(|turn| turn.speaker_ref.is_none()),
            SpeakerAttributionQueryReason::LowConfidence => query
                .candidate_speaker_refs
                .first()
                .is_some_and(|candidate| {
                    report.turns.iter().filter(overlaps).any(|turn| {
                        !turn.hard_hint_attributed
                            && turn.speaker_ref.as_ref() == Some(candidate)
                            && turn
                                .speaker_confidence
                                .is_some_and(|confidence| confidence < 0.60)
                    })
                }),
            SpeakerAttributionQueryReason::OverlapAmbiguity => {
                match query.candidate_speaker_refs.as_slice() {
                    [left_candidate, right_candidate] => report.turns.iter().any(|left| {
                        left.overlap_suspected
                            && left.speaker_ref.as_ref() == Some(left_candidate)
                            && left.start_ms < query.end_ms
                            && query.start_ms < left.end_ms
                            && report.turns.iter().any(|right| {
                                right.overlap_suspected
                                    && right.speaker_ref.as_ref() == Some(right_candidate)
                                    && right.start_ms < query.end_ms
                                    && query.start_ms < right.end_ms
                                    && left.start_ms.max(right.start_ms).max(query.start_ms)
                                        < left.end_ms.min(right.end_ms).min(query.end_ms)
                            })
                    }),
                    _ => false,
                }
            }
        };
        if !query_is_grounded {
            return Err(
                "speaker attribution query is not grounded in overlapping diarization turns"
                    .to_owned(),
            );
        }
    }
    let native_kind = matches!(
        kind,
        DiarizationReportKind::NativeAcoustic
            | DiarizationReportKind::NativeEcapaOnly
            | DiarizationReportKind::NativeEcapaFused
            | DiarizationReportKind::NativeEcapaUnavailable
    );
    if native_kind
        && report
            .turns
            .iter()
            .any(|turn| turn.speaker_ref.is_some() && turn.speaker_confidence.is_none())
    {
        return Err("native labeled diarization turn omits speaker confidence".to_owned());
    }
    if matches!(
        kind,
        DiarizationReportKind::NativeAcoustic
            | DiarizationReportKind::NativeEcapaOnly
            | DiarizationReportKind::NativeEcapaFused
            | DiarizationReportKind::External
    ) {
        let profile_refs = report
            .profiles
            .iter()
            .map(|profile| profile.speaker_ref.as_str())
            .collect::<BTreeSet<_>>();
        if profile_refs != active_refs {
            return Err(
                "speaker profile references do not match active supported speakers".to_owned(),
            );
        }
        for profile in &report.profiles {
            let evidence = report
                .speaker_count
                .speaker_evidence
                .iter()
                .find(|evidence| evidence.speaker_ref == profile.speaker_ref)
                .ok_or_else(|| {
                    "speaker profile has no matching supported speaker evidence".to_owned()
                })?;
            if profile.anchored != evidence.hard_anchored {
                return Err(
                    "speaker profile anchor flag disagrees with speaker evidence".to_owned(),
                );
            }
        }
    }
    Ok(())
}

fn validate_report_speaker_ref(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} is empty"));
    }
    if value.len() > MAX_SPEAKER_REF_BYTES {
        return Err(format!(
            "{field} exceeds the {MAX_SPEAKER_REF_BYTES}-byte limit"
        ));
    }
    Ok(())
}

fn slice_has_duplicates<T: PartialEq>(values: &[T]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, value)| values[..index].contains(value))
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
    #[value(alias = "whisper_cpp")]
    WhisperCpp,
    #[value(alias = "insanely_fast")]
    InsanelyFast,
    #[value(alias = "whisper_diarization")]
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

    fn complete_speaker_count_lanes() -> Vec<SpeakerCountLaneEvidence> {
        vec![
            SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::MergeRisk,
                available: true,
                proposed_count: Some(2),
                confidence: 0.75,
                unavailable_reason: None,
            },
            SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::SparseNormalizedEigengap,
                available: false,
                proposed_count: None,
                confidence: 0.0,
                unavailable_reason: Some(SpeakerCountLaneUnavailableReason::SolverDidNotConverge),
            },
            SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::FeatureJackknife,
                available: true,
                proposed_count: Some(2),
                confidence: 0.8,
                unavailable_reason: None,
            },
            SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::EffectiveOccupancy,
                available: true,
                proposed_count: Some(2),
                confidence: 0.7,
                unavailable_reason: None,
            },
            SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::ConstraintGraph,
                available: true,
                proposed_count: Some(1),
                confidence: 1.0,
                unavailable_reason: None,
            },
            SpeakerCountLaneEvidence {
                lane: SpeakerCountEvidenceLane::CallerPrior,
                available: false,
                proposed_count: None,
                confidence: 0.0,
                unavailable_reason: Some(SpeakerCountLaneUnavailableReason::NotRequested),
            },
        ]
    }

    fn hard_one_speaker_count_lanes() -> Vec<SpeakerCountLaneEvidence> {
        let mut lanes = complete_speaker_count_lanes();
        for lane in &mut lanes {
            if lane.proposed_count.is_some() {
                lane.proposed_count = Some(1);
            }
        }
        lanes
    }

    fn speaker_count_resources() -> SpeakerCountResourceSummary {
        SpeakerCountResourceSummary {
            prototype_count: 5,
            affinity_pair_evaluations: 20,
            retained_sparse_edges: 6,
            estimated_peak_buffer_bytes: 4_096,
            stability_replicates: 5,
            solver_iterations: 12,
            solver_sparse_matvec_terms: 2_040,
            solver_residual: Some(1.0e-8),
        }
    }

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
    fn speaker_count_estimate_serialization_round_trip() {
        let entropy_bits = [0.62, 0.23, 0.15].into_iter().map(entropy_term).sum();
        let estimate = SpeakerCountEstimate {
            schema_version: "speaker-count-estimate-v2".to_owned(),
            selected_count: Some(2),
            supported_range: Some(SpeakerCountRange {
                minimum: 2,
                maximum: 3,
            }),
            posterior: vec![
                SpeakerCountPosteriorBin {
                    count: 2,
                    probability: 0.62,
                },
                SpeakerCountPosteriorBin {
                    count: 3,
                    probability: 0.23,
                },
            ],
            unresolved_probability: 0.15,
            entropy_bits,
            stability: 0.8,
            constraint_lower_bound: 1,
            candidate_upper_bound: 5,
            calibration_status: SpeakerCountCalibrationStatus::DevelopmentUncertified,
            calibration_sha256: "a".repeat(64),
            evidence_sha256: "b".repeat(64),
            lanes: complete_speaker_count_lanes(),
            resources: speaker_count_resources(),
        };

        let json = serde_json::to_string(&estimate).unwrap();
        let deserialized: SpeakerCountEstimate = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, estimate);
        estimate.validate().unwrap();
        assert_eq!(
            serde_json::to_value(&estimate).unwrap()["calibration_status"],
            "development_uncertified"
        );
        assert_eq!(
            serde_json::to_value(&estimate).unwrap()["lanes"][1]["unavailable_reason"],
            "solver_did_not_converge"
        );
        assert_eq!(
            serde_json::to_value(&estimate).unwrap()["resources"]["retained_sparse_edges"],
            6
        );
    }

    #[test]
    fn speaker_count_estimate_validation_fails_closed() {
        let mut estimate = SpeakerCountEstimate {
            schema_version: "speaker-count-estimate-v2".to_owned(),
            selected_count: None,
            supported_range: Some(SpeakerCountRange {
                minimum: 1,
                maximum: 2,
            }),
            posterior: vec![
                SpeakerCountPosteriorBin {
                    count: 1,
                    probability: 0.2,
                },
                SpeakerCountPosteriorBin {
                    count: 2,
                    probability: 0.3,
                },
            ],
            unresolved_probability: 0.5,
            entropy_bits: [0.2, 0.3, 0.5].into_iter().map(entropy_term).sum(),
            stability: 0.0,
            constraint_lower_bound: 1,
            candidate_upper_bound: 2,
            calibration_status: SpeakerCountCalibrationStatus::DevelopmentUncertified,
            calibration_sha256: "a".repeat(64),
            evidence_sha256: "b".repeat(64),
            lanes: complete_speaker_count_lanes(),
            resources: speaker_count_resources(),
        };
        estimate.validate().unwrap();

        estimate.constraint_lower_bound = 0;
        assert_eq!(
            estimate.validate().unwrap_err(),
            "speaker-count estimate bounds must stay within 1..=64"
        );
        estimate.constraint_lower_bound = 1;
        estimate.posterior.swap(0, 1);
        assert_eq!(
            estimate.validate().unwrap_err(),
            "speaker-count posterior bins are not strictly count-ordered"
        );
        estimate.posterior.swap(0, 1);
        estimate.selected_count = Some(1);
        assert_eq!(
            estimate.validate().unwrap_err(),
            "speaker-count selection is not the authoritative posterior action"
        );
        estimate.selected_count = None;
        estimate.lanes[1].available = true;
        assert_eq!(
            estimate.validate().unwrap_err(),
            "available speaker-count lane carries an unavailable reason"
        );
        estimate.lanes[1].available = false;
        let caller_prior = estimate.lanes.pop().expect("caller-prior lane");
        assert_eq!(
            estimate.validate().unwrap_err(),
            "speaker-count estimate is missing a required evidence lane"
        );
        estimate.lanes.push(caller_prior);
        estimate.resources.solver_residual = Some(f64::NAN);
        assert_eq!(
            estimate.validate().unwrap_err(),
            "speaker-count solver residual is not finite and non-negative"
        );
        estimate.resources.solver_residual = Some(1.0e-8);
        estimate.resources.retained_sparse_edges = 11;
        assert_eq!(
            estimate.validate().unwrap_err(),
            "speaker-count retained sparse edges exceed the simple graph bound"
        );
        estimate.resources.retained_sparse_edges = 6;
        estimate.resources.affinity_pair_evaluations = 21;
        assert_eq!(
            estimate.validate().unwrap_err(),
            "speaker-count affinity evaluations exceed the directed pair bound"
        );
        estimate.resources.affinity_pair_evaluations = 20;
        let mut unresolved = estimate.clone();
        unresolved.unresolved_probability = 0.2;
        unresolved.posterior[0].probability = 0.4;
        unresolved.posterior[1].probability = 0.4;
        unresolved.entropy_bits = [0.4, 0.4, 0.2].into_iter().map(entropy_term).sum();
        unresolved
            .validate()
            .expect("consensus gates may withhold selection despite a dominant concrete bin");
        estimate.calibration_status = SpeakerCountCalibrationStatus::Unavailable;
        assert_eq!(
            estimate.validate().unwrap_err(),
            "uncalibrated or unavailable speaker-count evidence claims authority"
        );

        let mut missing_range = unresolved.clone();
        missing_range.supported_range = None;
        assert!(
            missing_range
                .validate()
                .expect_err("posterior authority requires an explicit supported range")
                .contains("omits its supported range")
        );

        let mut reordered_lanes = unresolved.clone();
        reordered_lanes.lanes.swap(0, 1);
        assert!(
            reordered_lanes
                .validate()
                .expect_err("evidence lanes have one deterministic durable ordering")
                .contains("not canonically ordered")
        );

        let mut zero_prototypes = unresolved;
        zero_prototypes.resources = SpeakerCountResourceSummary {
            prototype_count: 0,
            affinity_pair_evaluations: 0,
            retained_sparse_edges: 0,
            estimated_peak_buffer_bytes: 0,
            stability_replicates: 0,
            solver_iterations: 0,
            solver_sparse_matvec_terms: 0,
            solver_residual: None,
        };
        assert!(
            zero_prototypes
                .validate()
                .expect_err("zero prototypes cannot carry development authority")
                .contains("explicitly unavailable")
        );
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

        let acoustic_fallback = DiarizationRequest {
            engine: DiarizationEngine::EcapaFused,
            fallback: DiarizationFallbackPolicy::Acoustic,
            ..DiarizationRequest::default()
        };
        let json = serde_json::to_string(&acoustic_fallback).expect("serialize acoustic fallback");
        assert!(json.contains("\"fallback\":\"acoustic\""));
        assert_eq!(
            serde_json::from_str::<DiarizationRequest>(&json)
                .expect("deserialize acoustic fallback"),
            acoustic_fallback
        );
        for (engine, encoded) in [
            (DiarizationEngine::Ecapa, "ecapa"),
            (DiarizationEngine::EcapaFused, "ecapa_fused"),
        ] {
            let request = DiarizationRequest {
                engine,
                ..DiarizationRequest::default()
            };
            let json = serde_json::to_string(&request).expect("serialize ECAPA mode");
            assert!(json.contains(&format!("\"engine\":\"{encoded}\"")));
            assert_eq!(
                serde_json::from_str::<DiarizationRequest>(&json).expect("parse ECAPA mode"),
                request
            );
        }
        assert!(
            serde_json::from_str::<DiarizationRequest>(
                r#"{"engine":"neural","fallback":"unknown","speaker_count":"infer","known_intervals":[],"enrollment_edge_guard_ms":100,"max_prototypes":512,"persist_profiles":false}"#,
            )
            .is_err(),
            "the ambiguous legacy neural mode must not silently choose a fusion policy"
        );
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
            speaker_count: SpeakerCountRequest::HardConstraint { count: 1 },
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
    fn prototype_cap_cannot_drop_distinct_hard_references() {
        let request = DiarizationRequest {
            max_prototypes: 1,
            known_intervals: vec![
                speaker_hint("near", 0, 400, KnownSpeakerPolicy::HardMustLink),
                speaker_hint("remote", 600, 1_000, KnownSpeakerPolicy::HardMustLink),
            ],
            ..DiarizationRequest::default()
        };
        let error = request
            .validate(1_000)
            .expect_err("prototype cap cannot discard immutable speaker identities");
        assert_eq!(error.code, DiarizationValidationCode::InvalidPrototypeCap);
        assert!(error.message.contains("2 distinct hard speaker references"));
    }

    #[test]
    fn soft_count_range_cannot_override_hard_anchor_lower_bound() {
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
        assert!(
            request.validate(1_000).is_ok(),
            "a soft range may disagree but cannot invalidate immutable acoustic anchors"
        );
    }

    #[test]
    fn hard_anchor_lower_bound_cannot_exceed_the_bounded_count_domain() {
        let known_intervals = (0..=MAX_SPEAKER_COUNT)
            .map(|index| {
                let start_ms = u64::from(index) * 10;
                speaker_hint(
                    &format!("speaker_{index:02}"),
                    start_ms,
                    start_ms + 5,
                    KnownSpeakerPolicy::HardMustLink,
                )
            })
            .collect();
        let request = DiarizationRequest {
            known_intervals,
            ..DiarizationRequest::default()
        };
        let error = request
            .validate(u64::from(MAX_SPEAKER_COUNT + 1) * 10)
            .expect_err("hard-anchor lower bound above K_max must fail closed");
        assert_eq!(error.code, DiarizationValidationCode::InvalidSpeakerCount);
        assert!(error.message.contains("permits at most 64"));
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

    fn typed_diarization_report_fixture() -> DiarizationReport {
        let unresolved_probability = 0.2_f64;
        let concrete_probability = 0.8_f64;
        let calibration_sha256 = crate::diarization::acoustic_speaker_pair_calibration_sha256();
        let mut report = DiarizationReport {
            implementation: "native-acoustic-v2".to_owned(),
            contract_version: "acoustic-diarization-v3".to_owned(),
            feature_schema: "acoustic-feature-v2".to_owned(),
            speaker_evidence_mode: DiarizationSpeakerEvidenceMode::AcousticV2,
            normalized_input_sha256:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned(),
            hint_document_sha256: Some(
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned(),
            ),
            turns: vec![
                DiarizationTurn {
                    start_ms: 0,
                    end_ms: 1_000,
                    speaker_ref: Some("near".to_owned()),
                    speaker_confidence: Some(1.0),
                    change_confidence: Some(0.84),
                    overlap_suspected: false,
                    hard_hint_attributed: true,
                },
                DiarizationTurn {
                    start_ms: 1_000,
                    end_ms: 1_500,
                    speaker_ref: Some("near".to_owned()),
                    speaker_confidence: Some(0.5),
                    change_confidence: Some(0.2),
                    overlap_suspected: false,
                    hard_hint_attributed: false,
                },
            ],
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
                query_id_sha256: String::new(),
                start_ms: 1_000,
                end_ms: 1_500,
                reason: SpeakerAttributionQueryReason::LowConfidence,
                candidate_speaker_refs: vec!["near".to_owned()],
                suggested_policy: KnownSpeakerPolicy::SoftEnrollment,
            }],
            speaker_count: SpeakerCountOutcome {
                request: SpeakerCountRequest::HardConstraint { count: 1 },
                estimate: Some(SpeakerCountEstimate {
                    schema_version: "speaker-count-estimate-v2".to_owned(),
                    selected_count: Some(1),
                    supported_range: Some(SpeakerCountRange {
                        minimum: 1,
                        maximum: 1,
                    }),
                    posterior: vec![SpeakerCountPosteriorBin {
                        count: 1,
                        probability: concrete_probability,
                    }],
                    unresolved_probability,
                    entropy_bits: entropy_term(concrete_probability)
                        + entropy_term(unresolved_probability),
                    stability: 0.8,
                    constraint_lower_bound: 1,
                    candidate_upper_bound: 1,
                    calibration_status: SpeakerCountCalibrationStatus::DevelopmentUncertified,
                    calibration_sha256: calibration_sha256.clone(),
                    evidence_sha256:
                        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                            .to_owned(),
                    lanes: hard_one_speaker_count_lanes(),
                    resources: speaker_count_resources(),
                }),
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
                    mean_assignment_confidence: 1.0,
                    profile_reliability: 0.9,
                    hard_anchored: true,
                    separated_from_supported_speakers: true,
                    reasons: vec![SpeakerEvidenceReason::SupportedByHardHint],
                    supported: true,
                }],
            },
            fallback_status: DiarizationFallbackStatus::NotNeeded,
            operational_partition: Some(DiarizationOperationalPartitionSummary {
                schema_version: "diarization-operational-partition-v2".to_owned(),
                method: DiarizationOperationalPartitionMethod::ProbabilisticConsensus,
                selected_count: 1,
                confidence: 0.42,
                calibration_sha256,
                authority: SpeakerCountCalibrationStatus::DevelopmentUncertified,
            }),
            neural_representation: None,
            diagnostics: Vec::new(),
        };
        let query_id_sha256 = speaker_attribution_query_sha256(
            &report.normalized_input_sha256,
            report.speaker_queries.first().expect("query fixture"),
        );
        report
            .speaker_queries
            .first_mut()
            .expect("query fixture")
            .query_id_sha256 = query_id_sha256;
        report
    }

    fn ready_neural_summary() -> NeuralSpeakerRepresentationSummary {
        NeuralSpeakerRepresentationSummary {
            schema_version: "neural-speaker-representation-summary-v1".to_owned(),
            provider_version: crate::diarization::ECAPA_SPEAKER_REPRESENTATION_VERSION.to_owned(),
            expected_model_package_sha256: crate::ecapa_conformance::ECAPA_PACKAGE_SHA256
                .to_owned(),
            loaded_model_package_sha256: Some(
                crate::ecapa_conformance::ECAPA_PACKAGE_SHA256.to_owned(),
            ),
            model_load_source: Some(NeuralModelLoadSource::PackageVerified),
            status: NeuralSpeakerRepresentationStatus::Ready,
            embedded_tracklet_count: 2,
            zero_padded_tracklet_count: 0,
            skipped_tracklet_count: 0,
            reasons: Vec::new(),
        }
    }

    fn unavailable_neural_summary() -> NeuralSpeakerRepresentationSummary {
        NeuralSpeakerRepresentationSummary {
            schema_version: "neural-speaker-representation-summary-v1".to_owned(),
            provider_version: crate::diarization::ECAPA_SPEAKER_REPRESENTATION_VERSION.to_owned(),
            expected_model_package_sha256: crate::ecapa_conformance::ECAPA_PACKAGE_SHA256
                .to_owned(),
            loaded_model_package_sha256: None,
            model_load_source: None,
            status: NeuralSpeakerRepresentationStatus::Unavailable,
            embedded_tracklet_count: 0,
            zero_padded_tracklet_count: 0,
            skipped_tracklet_count: 0,
            reasons: vec![NeuralSpeakerRepresentationReason::ModelUnavailable],
        }
    }

    fn set_report_calibration_for_mode(
        report: &mut DiarizationReport,
        mode: DiarizationSpeakerEvidenceMode,
    ) -> Result<(), &'static str> {
        let calibration_sha256 = match mode {
            DiarizationSpeakerEvidenceMode::AcousticV2 => {
                crate::diarization::acoustic_speaker_pair_calibration_sha256()
            }
            DiarizationSpeakerEvidenceMode::EcapaOnly
            | DiarizationSpeakerEvidenceMode::EcapaWithAcousticChannel => {
                crate::diarization::ecapa_speaker_pair_calibration_sha256(mode)
            }
            DiarizationSpeakerEvidenceMode::External | DiarizationSpeakerEvidenceMode::None => {
                return Err("test helper requires a native evidence mode");
            }
        };
        report
            .speaker_count
            .estimate
            .as_mut()
            .expect("native test report estimate")
            .calibration_sha256
            .clone_from(&calibration_sha256);
        report
            .operational_partition
            .as_mut()
            .expect("native test report partition")
            .calibration_sha256 = calibration_sha256;
        Ok(())
    }

    fn external_diarization_report_fixture() -> DiarizationReport {
        let mut report = typed_diarization_report_fixture();
        report.implementation = "external-backend".to_owned();
        report.feature_schema = "external-unreported".to_owned();
        report.speaker_evidence_mode = DiarizationSpeakerEvidenceMode::External;
        report.hint_document_sha256 = None;
        report.hint_evidence.clear();
        report.speaker_queries.clear();
        report.turns.truncate(1);
        report.speaker_count.estimate = None;
        report.speaker_count.reasons = vec![
            SpeakerCountOutcomeReason::ExternalAttribution,
            SpeakerCountOutcomeReason::RequestedCountMatched,
        ];
        report.speaker_count.speaker_evidence = vec![SpeakerEvidenceSummary {
            speaker_ref: "near".to_owned(),
            assigned_tracklet_count: 1,
            independent_tracklet_count: 1,
            recurrence_episode_count: 1,
            voiced_frame_count: 0,
            independent_voiced_frame_count: 0,
            voiced_duration_ms: 1_000,
            mean_assignment_confidence: 0.0,
            profile_reliability: 0.0,
            hard_anchored: false,
            separated_from_supported_speakers: true,
            reasons: vec![SpeakerEvidenceReason::SupportedByExternalAttribution],
            supported: true,
        }];
        report.turns[0].speaker_confidence = None;
        report.turns[0].change_confidence = None;
        report.turns[0].hard_hint_attributed = false;
        report.profiles[0] = SpeakerProfileSummary {
            speaker_ref: "near".to_owned(),
            frame_count: 0,
            voiced_duration_ms: 1_000,
            reliability: 0.0,
            voice_profile_count: 0,
            channel_profile_count: 0,
            training_accepted_count: 0,
            training_downweighted_count: 0,
            training_quarantined_count: 0,
            anchored: false,
            soft_hint_contradiction: None,
        };
        report.fallback_status = DiarizationFallbackStatus::ExternalBackend;
        report.operational_partition = None;
        report.diagnostics = vec![DIARIZATION_DIAGNOSTIC_EXTERNAL_ATTRIBUTION_ACCEPTED.to_owned()];
        report
    }

    fn unknown_diarization_report_fixture() -> DiarizationReport {
        let mut report = typed_diarization_report_fixture();
        report.implementation = "fallback-unknown".to_owned();
        report.feature_schema = "acoustic-feature-v2".to_owned();
        report.speaker_evidence_mode = DiarizationSpeakerEvidenceMode::None;
        report.hint_document_sha256 = None;
        report.turns.clear();
        report.profiles.clear();
        report.hint_evidence.clear();
        report.speaker_queries.clear();
        report.speaker_count = SpeakerCountOutcome {
            request: SpeakerCountRequest::Infer,
            estimate: None,
            status: SpeakerCountOutcomeStatus::Unresolved,
            supported_speaker_count: 0,
            active_speaker_refs: Vec::new(),
            dominant_speaker_share: 0.0,
            unknown_voiced_share: 1.0,
            reasons: vec![SpeakerCountOutcomeReason::NoSupportedSpeakers],
            speaker_evidence: Vec::new(),
        };
        report.fallback_status = DiarizationFallbackStatus::InsufficientEvidence;
        report.operational_partition = None;
        report.neural_representation = None;
        report.diagnostics = vec![DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_UNAVAILABLE.to_owned()];
        report
    }

    fn fixed_safe_diarization_report_fixture() -> DiarizationReport {
        let mut report = typed_diarization_report_fixture();
        let estimate = report
            .speaker_count
            .estimate
            .as_mut()
            .expect("native test report estimate");
        estimate.selected_count = None;
        estimate.supported_range = None;
        estimate.posterior.clear();
        estimate.unresolved_probability = 1.0;
        estimate.entropy_bits = 0.0;
        estimate.stability = 0.0;
        estimate.calibration_status = SpeakerCountCalibrationStatus::FixedSafeUncalibrated;
        let partition = report
            .operational_partition
            .as_mut()
            .expect("native test report partition");
        partition.method = DiarizationOperationalPartitionMethod::FixedSafeAgglomerative;
        partition.confidence = 0.0;
        partition.authority = SpeakerCountCalibrationStatus::FixedSafeUncalibrated;
        report
    }

    fn use_infer_count_request(report: &mut DiarizationReport) {
        report.speaker_count.request = SpeakerCountRequest::Infer;
        report.speaker_count.status = SpeakerCountOutcomeStatus::Resolved;
        report.speaker_count.reasons = vec![SpeakerCountOutcomeReason::EvidenceSupportedCount];
        let estimate = report
            .speaker_count
            .estimate
            .as_mut()
            .expect("native test report estimate");
        estimate.candidate_upper_bound = 5;
        estimate.lanes = complete_speaker_count_lanes();
    }

    fn matching_acoustic_request(report: &mut DiarizationReport) -> DiarizationRequest {
        let request = DiarizationRequest {
            engine: DiarizationEngine::Acoustic,
            fallback: DiarizationFallbackPolicy::Unknown,
            speaker_count: SpeakerCountRequest::HardConstraint { count: 1 },
            known_intervals: vec![speaker_hint(
                "near",
                0,
                1_000,
                KnownSpeakerPolicy::HardMustLink,
            )],
            ..DiarizationRequest::default()
        };
        report.hint_document_sha256 = Some(speaker_hint_document_sha256(&request));
        request
    }

    #[test]
    fn diarization_report_is_typed_and_privacy_safe() {
        let report = typed_diarization_report_fixture();
        let json = serde_json::to_string(&report).expect("serialize report");
        assert!(!json.contains("feature_vector"));
        assert!(!json.contains("raw_audio"));
        report
            .operational_partition
            .as_ref()
            .expect("typed operational partition")
            .validate()
            .expect("valid operational partition");
        assert_eq!(
            serde_json::from_str::<DiarizationReport>(&json).expect("deserialize report"),
            report
        );
        report.validate().expect("valid report contract");
    }

    #[test]
    fn admitted_diarization_report_tuples_validate() {
        let mut acoustic_v1 = typed_diarization_report_fixture();
        acoustic_v1.implementation = "native-acoustic-v1".to_owned();
        acoustic_v1.feature_schema = "acoustic-feature-v1".to_owned();
        acoustic_v1.validate().expect("native acoustic v1 tuple");

        let mut acoustic_fallback = typed_diarization_report_fixture();
        acoustic_fallback.neural_representation = Some(ready_neural_summary());
        acoustic_fallback.diagnostics =
            vec![DIARIZATION_DIAGNOSTIC_NATIVE_ACOUSTIC_FALLBACK.to_owned()];
        acoustic_fallback
            .validate()
            .expect("acoustic fallback may retain valid neural attempt provenance");

        let mut missing_fallback_code = acoustic_fallback.clone();
        missing_fallback_code.diagnostics.clear();
        assert!(
            missing_fallback_code
                .validate()
                .expect_err("neural provenance requires the acoustic-fallback code")
                .contains("not canonical for the report implementation")
        );

        let mut forged_fallback_code = typed_diarization_report_fixture();
        forged_fallback_code.diagnostics =
            vec![DIARIZATION_DIAGNOSTIC_NATIVE_ACOUSTIC_FALLBACK.to_owned()];
        assert!(
            forged_fallback_code
                .validate()
                .expect_err("an acoustic-fallback code requires neural attempt provenance")
                .contains("not canonical for the report implementation")
        );

        let mut ecapa_only = typed_diarization_report_fixture();
        ecapa_only.implementation = "native-ecapa-only-v1".to_owned();
        ecapa_only.contract_version = "neural-diarization-common-v2".to_owned();
        ecapa_only.speaker_evidence_mode = DiarizationSpeakerEvidenceMode::EcapaOnly;
        ecapa_only.neural_representation = Some(ready_neural_summary());
        set_report_calibration_for_mode(&mut ecapa_only, DiarizationSpeakerEvidenceMode::EcapaOnly)
            .expect("ECAPA-only test calibration");
        ecapa_only
            .operational_partition
            .as_mut()
            .expect("partition")
            .method = DiarizationOperationalPartitionMethod::EcapaSpherical;
        ecapa_only.validate().expect("native ECAPA-only tuple");

        let mut ecapa_fused = ecapa_only.clone();
        ecapa_fused.implementation = "native-ecapa-fused-v1".to_owned();
        ecapa_fused.speaker_evidence_mode =
            DiarizationSpeakerEvidenceMode::EcapaWithAcousticChannel;
        set_report_calibration_for_mode(
            &mut ecapa_fused,
            DiarizationSpeakerEvidenceMode::EcapaWithAcousticChannel,
        )
        .expect("fused ECAPA test calibration");
        ecapa_fused
            .operational_partition
            .as_mut()
            .expect("partition")
            .method = DiarizationOperationalPartitionMethod::EcapaFusedConsensus;
        ecapa_fused.validate().expect("native fused ECAPA tuple");
        assert!(
            serde_json::to_string(&ecapa_fused)
                .expect("serialize fused ECAPA tuple")
                .contains("ecapa_fused_consensus")
        );

        let mut ecapa_only_with_fused_partition = ecapa_only.clone();
        ecapa_only_with_fused_partition
            .operational_partition
            .as_mut()
            .expect("partition")
            .method = DiarizationOperationalPartitionMethod::EcapaFusedConsensus;
        assert!(
            ecapa_only_with_fused_partition
                .validate()
                .expect_err("ECAPA-only report cannot claim a fused consensus partition")
                .contains("cannot claim a fused consensus")
        );

        let mut external = external_diarization_report_fixture();
        external.neural_representation = Some(ready_neural_summary());
        external
            .diagnostics
            .push(DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_FALLBACK.to_owned());
        external
            .validate()
            .expect("external fallback may retain valid neural attempt provenance");

        let mut missing_external_fallback_code = external.clone();
        missing_external_fallback_code.diagnostics.pop();
        assert!(
            missing_external_fallback_code
                .validate()
                .expect_err("external neural provenance requires fallback routing")
                .contains("not canonical for the report implementation")
        );

        let mut partial_external = external_diarization_report_fixture();
        partial_external.speaker_count.dominant_speaker_share = 0.6;
        partial_external.speaker_count.unknown_voiced_share = 0.4;
        partial_external.speaker_count.status = SpeakerCountOutcomeStatus::Unsatisfied;
        partial_external.speaker_count.reasons = vec![
            SpeakerCountOutcomeReason::ExternalAttribution,
            SpeakerCountOutcomeReason::RequestedCountMismatch,
        ];
        partial_external
            .validate()
            .expect("external attribution may honestly leave timed speech unknown");

        unknown_diarization_report_fixture()
            .validate()
            .expect("unknown fallback tuple");

        let mut forged_unknown_cause = unknown_diarization_report_fixture();
        forged_unknown_cause.fallback_status = DiarizationFallbackStatus::ResourceLimit;
        assert!(
            forged_unknown_cause
                .validate()
                .expect_err("unknown fallback cannot invent an unproduced resource cause")
                .contains("cause disagrees")
        );

        let mut hard_unknown = unknown_diarization_report_fixture();
        hard_unknown.speaker_count.request = SpeakerCountRequest::HardConstraint { count: 2 };
        hard_unknown.speaker_count.status = SpeakerCountOutcomeStatus::Unsatisfied;
        hard_unknown.speaker_count.reasons =
            vec![SpeakerCountOutcomeReason::RequestedCountMismatch];
        hard_unknown.fallback_status = DiarizationFallbackStatus::UnsatisfiedConstraints;
        hard_unknown
            .validate()
            .expect("unknown hard-count fallback carries the exact unsatisfied cause");
        hard_unknown.fallback_status = DiarizationFallbackStatus::InsufficientEvidence;
        assert!(
            hard_unknown
                .validate()
                .expect_err("unknown hard-count mismatch cannot claim mere insufficiency")
                .contains("cause disagrees")
        );

        let mut prior_unknown = unknown_diarization_report_fixture();
        prior_unknown.speaker_count.request = SpeakerCountRequest::Prior {
            bins: vec![SpeakerCountPriorMass {
                count: 1,
                probability: 1.0,
            }],
        };
        prior_unknown.fallback_status = DiarizationFallbackStatus::SpeakerCountUnresolved;
        let error = prior_unknown
            .validate()
            .expect_err("unknown fallback cannot embed a soft prior");
        assert!(
            error.contains("cannot preserve soft speaker-count"),
            "{error}"
        );

        let mut unavailable = unknown_diarization_report_fixture();
        unavailable.implementation = "native-ecapa-unavailable-v1".to_owned();
        unavailable.contract_version = "neural-diarization-common-v2".to_owned();
        unavailable.neural_representation = Some(unavailable_neural_summary());
        unavailable.fallback_status = DiarizationFallbackStatus::SpeakerCountUnresolved;
        unavailable.diagnostics =
            vec![DIARIZATION_DIAGNOSTIC_NEURAL_IDENTITY_UNAVAILABLE.to_owned()];
        unavailable
            .validate()
            .expect("unavailable ECAPA tuple with honest provenance");
    }

    #[test]
    fn external_and_unavailable_reports_cannot_claim_native_authority() {
        let mut external_native_profile = external_diarization_report_fixture();
        external_native_profile.profiles[0].reliability = 0.1;
        assert!(
            external_native_profile
                .validate()
                .expect_err("external profile cannot claim native reliability")
                .contains("unavailable native evidence")
        );

        let mut partial_resolved = external_diarization_report_fixture();
        partial_resolved.speaker_count.request = SpeakerCountRequest::Infer;
        partial_resolved.speaker_count.status = SpeakerCountOutcomeStatus::Resolved;
        partial_resolved.speaker_count.dominant_speaker_share = 0.6;
        partial_resolved.speaker_count.unknown_voiced_share = 0.4;
        partial_resolved.speaker_count.reasons = vec![
            SpeakerCountOutcomeReason::ExternalAttribution,
            SpeakerCountOutcomeReason::EvidenceSupportedCount,
        ];
        assert!(
            partial_resolved
                .validate()
                .expect_err("partial external labels cannot resolve the total speaker count")
                .contains("attribution coverage")
        );

        let mut fully_attributed_unresolved = external_diarization_report_fixture();
        fully_attributed_unresolved.speaker_count.request = SpeakerCountRequest::Infer;
        fully_attributed_unresolved.speaker_count.status = SpeakerCountOutcomeStatus::Unresolved;
        fully_attributed_unresolved.speaker_count.reasons = vec![
            SpeakerCountOutcomeReason::ExternalAttribution,
            SpeakerCountOutcomeReason::SpeakerCountEvidenceUnresolved,
        ];
        assert!(
            fully_attributed_unresolved
                .validate()
                .expect_err("complete external attribution resolves an inferred count")
                .contains("attribution coverage")
        );

        let mut fully_attributed_infer = external_diarization_report_fixture();
        fully_attributed_infer.speaker_count.request = SpeakerCountRequest::Infer;
        fully_attributed_infer.speaker_count.status = SpeakerCountOutcomeStatus::Resolved;
        fully_attributed_infer.speaker_count.reasons = vec![
            SpeakerCountOutcomeReason::ExternalAttribution,
            SpeakerCountOutcomeReason::EvidenceSupportedCount,
        ];
        fully_attributed_infer
            .validate()
            .expect("complete external attribution resolves an inferred count");

        let mut external_prior = external_diarization_report_fixture();
        external_prior.speaker_count.request = SpeakerCountRequest::Prior {
            bins: vec![SpeakerCountPriorMass {
                count: 1,
                probability: 1.0,
            }],
        };
        external_prior.speaker_count.status = SpeakerCountOutcomeStatus::Unresolved;
        external_prior.speaker_count.reasons = vec![
            SpeakerCountOutcomeReason::ExternalAttribution,
            SpeakerCountOutcomeReason::SpeakerCountPriorFusionUnavailable,
        ];
        assert!(
            external_prior
                .validate()
                .expect_err("external standalone reports cannot embed a soft prior")
                .contains("cannot preserve soft speaker-count")
        );

        let mut unavailable = typed_diarization_report_fixture();
        unavailable.implementation = "native-ecapa-unavailable-v1".to_owned();
        unavailable.contract_version = "neural-diarization-common-v2".to_owned();
        unavailable.speaker_evidence_mode = DiarizationSpeakerEvidenceMode::None;
        unavailable.neural_representation = Some(unavailable_neural_summary());
        unavailable.speaker_count.estimate = None;
        unavailable.operational_partition = None;
        unavailable.profiles.clear();
        unavailable.speaker_queries.clear();
        unavailable.diagnostics =
            vec![DIARIZATION_DIAGNOSTIC_NEURAL_IDENTITY_UNAVAILABLE.to_owned()];
        unavailable.turns.truncate(1);
        unavailable.fallback_status = DiarizationFallbackStatus::InsufficientEvidence;
        unavailable.speaker_count.speaker_evidence[0].assigned_tracklet_count = 3;
        unavailable.speaker_count.speaker_evidence[0].independent_tracklet_count = 3;
        unavailable.speaker_count.speaker_evidence[0].recurrence_episode_count = 1;
        unavailable
            .validate()
            .expect("multiple hard-attributed tracklets may form one recurrence episode");

        unavailable.turns.push(DiarizationTurn {
            start_ms: 1_000,
            end_ms: 1_500,
            speaker_ref: None,
            speaker_confidence: None,
            change_confidence: Some(0.2),
            overlap_suspected: false,
            hard_hint_attributed: false,
        });
        unavailable.speaker_count.dominant_speaker_share = f64::from(2.0_f32 / 3.0);
        unavailable.speaker_count.unknown_voiced_share = f64::from(1.0_f32 / 3.0);
        unavailable
            .validate()
            .expect("unavailable ECAPA may retain explicitly unknown timed speech");

        unavailable.turns[0].hard_hint_attributed = false;
        assert!(
            unavailable
                .validate()
                .expect_err("unavailable ECAPA cannot invent non-hard identity evidence")
                .contains("non-hard speaker identity evidence")
        );
    }

    #[test]
    fn diarization_request_rejects_unknown_sensitive_fields_at_every_nested_level() {
        let request = DiarizationRequest {
            speaker_count: SpeakerCountRequest::Prior {
                bins: vec![SpeakerCountPriorMass {
                    count: 1,
                    probability: 1.0,
                }],
            },
            known_intervals: vec![speaker_hint(
                "near",
                0,
                100,
                KnownSpeakerPolicy::SoftEnrollment,
            )],
            ..DiarizationRequest::default()
        };
        let value = serde_json::to_value(request).expect("request JSON");

        let mut top_level = value.clone();
        top_level["raw_audio"] = json!([0, 1]);
        assert!(serde_json::from_value::<DiarizationRequest>(top_level).is_err());

        let mut interval = value.clone();
        interval["known_intervals"][0]["embedding"] = json!([0.1, 0.2]);
        assert!(serde_json::from_value::<DiarizationRequest>(interval).is_err());

        let mut count_request = value.clone();
        count_request["speaker_count"]["raw_audio"] = json!([0, 1]);
        assert!(serde_json::from_value::<DiarizationRequest>(count_request).is_err());

        let mut prior_bin = value;
        prior_bin["speaker_count"]["bins"][0]["embedding"] = json!([0.1]);
        assert!(serde_json::from_value::<DiarizationRequest>(prior_bin).is_err());
    }

    #[test]
    fn diarization_report_rejects_unknown_sensitive_fields_at_every_nested_level() {
        let value = serde_json::to_value(typed_diarization_report_fixture()).expect("report JSON");

        let mut top_level = value.clone();
        top_level["embedding"] = json!([0.1]);
        assert!(serde_json::from_value::<DiarizationReport>(top_level).is_err());

        let mut turn = value.clone();
        turn["turns"][0]["raw_audio"] = json!([0, 1]);
        assert!(serde_json::from_value::<DiarizationReport>(turn).is_err());

        let mut profile = value.clone();
        profile["profiles"][0]["embedding"] = json!([0.1]);
        assert!(serde_json::from_value::<DiarizationReport>(profile).is_err());
        for required_field in [
            "voice_profile_count",
            "training_accepted_count",
            "training_downweighted_count",
            "training_quarantined_count",
        ] {
            let mut missing_profile_field = value.clone();
            missing_profile_field["profiles"][0]
                .as_object_mut()
                .expect("profile JSON object")
                .remove(required_field);
            assert!(
                serde_json::from_value::<DiarizationReport>(missing_profile_field).is_err(),
                "missing authority field {required_field} must fail closed"
            );
        }

        let mut hint = value.clone();
        hint["hint_evidence"][0]["raw_audio"] = json!([0, 1]);
        assert!(serde_json::from_value::<DiarizationReport>(hint).is_err());

        let mut query = value.clone();
        query["speaker_queries"][0]["embedding"] = json!([0.1]);
        assert!(serde_json::from_value::<DiarizationReport>(query).is_err());

        let mut outcome = value.clone();
        outcome["speaker_count"]["raw_audio"] = json!([0, 1]);
        assert!(serde_json::from_value::<DiarizationReport>(outcome).is_err());

        let mut evidence = value.clone();
        evidence["speaker_count"]["speaker_evidence"][0]["embedding"] = json!([0.1]);
        assert!(serde_json::from_value::<DiarizationReport>(evidence).is_err());

        let mut estimate = value.clone();
        estimate["speaker_count"]["estimate"]["raw_audio"] = json!([0, 1]);
        assert!(serde_json::from_value::<DiarizationReport>(estimate).is_err());

        let mut range = value.clone();
        range["speaker_count"]["estimate"]["supported_range"]["embedding"] = json!([0.1]);
        assert!(serde_json::from_value::<DiarizationReport>(range).is_err());

        let mut posterior = value.clone();
        posterior["speaker_count"]["estimate"]["posterior"][0]["embedding"] = json!([0.1]);
        assert!(serde_json::from_value::<DiarizationReport>(posterior).is_err());

        let mut lane = value.clone();
        lane["speaker_count"]["estimate"]["lanes"][0]["embedding"] = json!([0.1]);
        assert!(serde_json::from_value::<DiarizationReport>(lane).is_err());

        let mut resources = value.clone();
        resources["speaker_count"]["estimate"]["resources"]["raw_audio"] = json!([0, 1]);
        assert!(serde_json::from_value::<DiarizationReport>(resources).is_err());

        let mut partition = value;
        partition["operational_partition"]["embedding"] = json!([0.1]);
        assert!(serde_json::from_value::<DiarizationReport>(partition).is_err());

        let mut neural_report = typed_diarization_report_fixture();
        neural_report.neural_representation = Some(ready_neural_summary());
        let mut neural = serde_json::to_value(neural_report).expect("neural report JSON");
        neural["neural_representation"]["embedding"] = json!([0.1]);
        assert!(serde_json::from_value::<DiarizationReport>(neural).is_err());
    }

    #[test]
    fn forged_certification_is_rejected_without_an_admitted_digest_registry() {
        let mut report = typed_diarization_report_fixture();
        report
            .speaker_count
            .estimate
            .as_mut()
            .expect("estimate")
            .calibration_status = SpeakerCountCalibrationStatus::Certified;
        assert!(
            report
                .validate()
                .expect_err("forged estimate certification must fail")
                .contains("not admitted")
        );

        let mut report = typed_diarization_report_fixture();
        report
            .operational_partition
            .as_mut()
            .expect("partition")
            .authority = SpeakerCountCalibrationStatus::Certified;
        assert!(
            report
                .validate()
                .expect_err("forged partition certification must fail")
                .contains("not admitted")
        );
    }

    #[test]
    fn operational_partition_authority_matrix_preserves_distinct_actions() {
        fixed_safe_diarization_report_fixture()
            .validate()
            .expect("fixed-safe partition with unavailable posterior fields");

        let mut report = typed_diarization_report_fixture();
        use_infer_count_request(&mut report);
        report
            .operational_partition
            .as_mut()
            .expect("partition")
            .selected_count = 2;
        report
            .validate()
            .expect("operational hierarchy count may differ from the posterior MAP");

        let mut report = typed_diarization_report_fixture();
        use_infer_count_request(&mut report);
        report
            .speaker_count
            .estimate
            .as_mut()
            .expect("estimate")
            .selected_count = None;
        report.speaker_count.status = SpeakerCountOutcomeStatus::Unresolved;
        report.speaker_count.reasons =
            vec![SpeakerCountOutcomeReason::SpeakerCountEvidenceUnresolved];
        report.fallback_status = DiarizationFallbackStatus::SpeakerCountUnresolved;
        report
            .validate()
            .expect("finalization may clear a posterior selection without erasing the partition");

        let mut report = fixed_safe_diarization_report_fixture();
        report
            .operational_partition
            .as_mut()
            .expect("partition")
            .method = DiarizationOperationalPartitionMethod::ProbabilisticConsensus;
        report
            .operational_partition
            .as_mut()
            .expect("partition")
            .authority = SpeakerCountCalibrationStatus::DevelopmentUncertified;
        assert!(
            report
                .validate()
                .expect_err("partition and estimate authorities must match")
                .contains("authority differs")
        );

        let mut estimate = fixed_safe_diarization_report_fixture()
            .speaker_count
            .estimate
            .expect("estimate");
        estimate.stability = 0.5;
        assert!(
            estimate
                .validate()
                .expect_err("fixed-safe estimates cannot claim posterior stability")
                .contains("range or stability")
        );

        let mut false_fixed_confidence = fixed_safe_diarization_report_fixture();
        false_fixed_confidence
            .operational_partition
            .as_mut()
            .expect("partition")
            .confidence = 0.9;
        assert!(
            false_fixed_confidence
                .validate()
                .expect_err("fixed-safe action cannot claim calibrated confidence")
                .contains("must have zero confidence")
        );
    }

    #[test]
    fn diarization_report_binds_known_provider_and_evidence_calibration() {
        let mut report = typed_diarization_report_fixture();
        report
            .speaker_count
            .estimate
            .as_mut()
            .expect("estimate")
            .calibration_sha256 = "f".repeat(64);
        report
            .operational_partition
            .as_mut()
            .expect("partition")
            .calibration_sha256 = "f".repeat(64);
        assert!(
            report
                .validate()
                .expect_err("acoustic evidence must use the admitted calibration")
                .contains("incompatible with the evidence mode")
        );

        let mut report = typed_diarization_report_fixture();
        report.implementation = "native-ecapa-only-v1".to_owned();
        report.contract_version = "neural-diarization-common-v2".to_owned();
        report.speaker_evidence_mode = DiarizationSpeakerEvidenceMode::EcapaOnly;
        let mut summary = ready_neural_summary();
        summary.provider_version = "unrecognized-ecapa-provider".to_owned();
        report.neural_representation = Some(summary);
        report
            .operational_partition
            .as_mut()
            .expect("partition")
            .method = DiarizationOperationalPartitionMethod::EcapaSpherical;
        set_report_calibration_for_mode(&mut report, DiarizationSpeakerEvidenceMode::EcapaOnly)
            .expect("ECAPA-only test calibration");
        assert!(
            report
                .validate()
                .expect_err("native ECAPA report must use the admitted provider version")
                .contains("unrecognized provider version")
        );

        let mut report = external_diarization_report_fixture();
        let mut summary = ready_neural_summary();
        summary.provider_version =
            crate::diarization::ECAPA_SPEAKER_REPRESENTATION_VERSION.to_owned();
        summary.expected_model_package_sha256 = "f".repeat(64);
        summary.loaded_model_package_sha256 = Some("f".repeat(64));
        report.neural_representation = Some(summary);
        assert!(
            report
                .validate()
                .expect_err("known provider must bind its package digest")
                .contains("unrecognized expected model package")
        );

        let mut report = external_diarization_report_fixture();
        let mut summary = ready_neural_summary();
        summary.provider_version =
            crate::diarization::ECAPA_SPEAKER_REPRESENTATION_VERSION.to_owned();
        summary.expected_model_package_sha256 =
            crate::ecapa_conformance::ECAPA_PACKAGE_SHA256.to_owned();
        summary.loaded_model_package_sha256 =
            Some(crate::ecapa_conformance::ECAPA_PACKAGE_SHA256.to_owned());
        report.neural_representation = Some(summary);
        report
            .diagnostics
            .push(DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_FALLBACK.to_owned());
        report
            .validate()
            .expect("known provider with the admitted package digest");
    }

    #[test]
    fn diarization_report_validates_against_exact_request() {
        let mut report = typed_diarization_report_fixture();
        let request = matching_acoustic_request(&mut report);
        report
            .validate_against_request(&request, None)
            .expect("unknown audio duration uses observed intervals as a lower bound");
        report
            .validate_against_request(&request, Some(1_500))
            .expect("known audio duration covers all intervals");
        assert!(
            report
                .validate_against_request(&request, Some(1_499))
                .expect_err("reported intervals cannot exceed the known audio duration")
                .contains("exceeds audio duration")
        );

        let mut mismatched_count = request.clone();
        mismatched_count.speaker_count = SpeakerCountRequest::HardConstraint { count: 2 };
        assert!(
            report
                .validate_against_request(&mismatched_count, None)
                .expect_err("report must bind the exact count request")
                .contains("speaker-count request differs")
        );

        let mut mismatched_hint = request.clone();
        mismatched_hint.known_intervals[0].confidence = 0.5;
        assert!(
            report
                .validate_against_request(&mismatched_hint, None)
                .expect_err("report must bind the canonical hint document")
                .contains("hint digest differs")
        );

        let mut mismatched_identity = request.clone();
        mismatched_identity.known_intervals[0].speaker_ref = "other".to_owned();
        let mut identity_report = report.clone();
        identity_report.hint_document_sha256 =
            Some(speaker_hint_document_sha256(&mismatched_identity));
        assert!(
            identity_report
                .validate_against_request(&mismatched_identity, None)
                .expect_err("hint evidence identities must match the request")
                .contains("does not identify")
        );

        let mut mismatched_engine = request.clone();
        mismatched_engine.engine = DiarizationEngine::External;
        assert!(
            report
                .validate_against_request(&mismatched_engine, None)
                .expect_err("native report cannot satisfy an external request")
                .contains("incompatible with the requested engine")
        );
    }

    #[test]
    fn ordered_hint_digest_binds_positional_evidence_to_the_exact_interval_sequence() {
        let mut report = typed_diarization_report_fixture();
        let mut request = matching_acoustic_request(&mut report);
        request.known_intervals.push(speaker_hint(
            "near",
            1_200,
            1_400,
            KnownSpeakerPolicy::HardMustLink,
        ));
        report.hint_evidence.push(SpeakerHintEvidenceSummary {
            hint_index: 1,
            speaker_ref: "near".to_owned(),
            policy: KnownSpeakerPolicy::HardMustLink,
            disposition: SpeakerHintDisposition::NoUsableTracklets,
            usable_tracklet_count: 0,
            accepted_tracklet_count: 0,
            rejected_tracklet_count: 0,
            profile_accepted_tracklet_count: 0,
            profile_downweighted_tracklet_count: 0,
            profile_quarantined_tracklet_count: 0,
            applied_weight: 0.0,
            contradiction_score: None,
        });
        report.hint_document_sha256 = Some(speaker_hint_document_sha256(&request));
        report
            .validate_against_request(&request, None)
            .expect("ordered hard hints with honest no-usable evidence");

        let mut reordered = request;
        reordered.known_intervals.reverse();
        let reordered_digest = speaker_hint_document_sha256(&reordered);
        assert_ne!(
            report.hint_document_sha256.as_deref(),
            Some(reordered_digest.as_str())
        );
        assert!(
            report
                .validate_against_request(&reordered, None)
                .expect_err("same-ref same-policy intervals cannot be silently reordered")
                .contains("hint digest differs")
        );
    }

    #[test]
    fn hint_digest_binds_edge_guard_and_provenance_presence() {
        let mut request = DiarizationRequest {
            known_intervals: vec![speaker_hint(
                "near",
                0,
                1_000,
                KnownSpeakerPolicy::HardMustLink,
            )],
            ..DiarizationRequest::default()
        };
        request.known_intervals[0].provenance = None;
        let without_provenance = speaker_hint_document_sha256(&request);

        request.known_intervals[0].provenance = Some(String::new());
        assert_ne!(without_provenance, speaker_hint_document_sha256(&request));

        let before_edge_guard = speaker_hint_document_sha256(&request);
        request.enrollment_edge_guard_ms = request.enrollment_edge_guard_ms.saturating_add(1);
        assert_ne!(before_edge_guard, speaker_hint_document_sha256(&request));
    }

    #[test]
    fn hard_hint_attribution_requires_supported_anchor_and_overlapping_hard_turn() {
        let mut report = typed_diarization_report_fixture();
        let request = matching_acoustic_request(&mut report);

        let mut missing_anchor = report.clone();
        missing_anchor.profiles[0].anchored = false;
        let speaker_evidence = &mut missing_anchor.speaker_count.speaker_evidence[0];
        speaker_evidence.hard_anchored = false;
        speaker_evidence.assigned_tracklet_count = 2;
        speaker_evidence.independent_tracklet_count = 2;
        speaker_evidence.recurrence_episode_count = 1;
        speaker_evidence.reasons = vec![SpeakerEvidenceReason::SupportedByRepeatedTracklets];
        missing_anchor
            .validate()
            .expect("report-local evidence remains internally consistent");
        assert!(
            missing_anchor
                .validate_against_request(&request, None)
                .expect_err("hard attribution cannot omit hard-anchor evidence")
                .contains("lacks active, supported hard-anchor evidence")
        );

        let mut missing_turn = report.clone();
        missing_turn.turns[0].hard_hint_attributed = false;
        assert!(
            missing_turn
                .validate_against_request(&request, None)
                .expect_err("hard attribution must have an overlapping hard turn")
                .contains("lacks an overlapping hard-attributed")
        );

        let mut trusted_timestamp_only = report.clone();
        let hint_evidence = &mut trusted_timestamp_only.hint_evidence[0];
        hint_evidence.usable_tracklet_count = 0;
        hint_evidence.accepted_tracklet_count = 0;
        hint_evidence.profile_accepted_tracklet_count = 0;
        hint_evidence.applied_weight = 0.0;
        trusted_timestamp_only
            .validate_against_request(&request, None)
            .expect("immutable timestamp label does not fabricate enrolled tracklets");

        let mut no_usable_but_attributed = trusted_timestamp_only.clone();
        no_usable_but_attributed.hint_evidence[0].disposition =
            SpeakerHintDisposition::NoUsableTracklets;
        no_usable_but_attributed.profiles[0].anchored = false;
        let speaker_evidence = &mut no_usable_but_attributed.speaker_count.speaker_evidence[0];
        speaker_evidence.hard_anchored = false;
        speaker_evidence.assigned_tracklet_count = 2;
        speaker_evidence.independent_tracklet_count = 2;
        speaker_evidence.recurrence_episode_count = 1;
        speaker_evidence.reasons = vec![SpeakerEvidenceReason::SupportedByRepeatedTracklets];
        let error = no_usable_but_attributed
            .validate_against_request(&request, None)
            .expect_err("no-usable hard hint cannot authorize a hard-attributed turn");
        assert!(
            error.contains("not bound to an overlapping hard hint"),
            "{error}"
        );

        let mut no_usable_but_anchored = trusted_timestamp_only;
        no_usable_but_anchored.hint_evidence[0].disposition =
            SpeakerHintDisposition::NoUsableTracklets;
        no_usable_but_anchored.turns[0].hard_hint_attributed = false;
        assert!(
            no_usable_but_anchored
                .validate_against_request(&request, None)
                .expect_err("hard anchor cannot survive without hard-attributed hint evidence")
                .contains("hard-anchored speaker evidence lacks")
        );

        let mut forged_turn = report;
        forged_turn.turns[0].start_ms = 1_000;
        forged_turn.turns[0].end_ms = 1_100;
        forged_turn.turns[1].start_ms = 1_100;
        let error = forged_turn
            .validate_against_request(&request, None)
            .expect_err("hard turn outside its hint cannot claim hard attribution");
        assert!(
            error.contains("lacks an overlapping hard-attributed"),
            "{error}"
        );
    }

    #[test]
    fn no_usable_hard_hint_does_not_fabricate_an_active_speaker() {
        let mut report = typed_diarization_report_fixture();
        let request = matching_acoustic_request(&mut report);
        report.turns.clear();
        report.profiles.clear();
        report.hint_evidence[0] = SpeakerHintEvidenceSummary {
            hint_index: 0,
            speaker_ref: "near".to_owned(),
            policy: KnownSpeakerPolicy::HardMustLink,
            disposition: SpeakerHintDisposition::NoUsableTracklets,
            usable_tracklet_count: 0,
            accepted_tracklet_count: 0,
            rejected_tracklet_count: 0,
            profile_accepted_tracklet_count: 0,
            profile_downweighted_tracklet_count: 0,
            profile_quarantined_tracklet_count: 0,
            applied_weight: 0.0,
            contradiction_score: None,
        };
        report.speaker_queries.clear();
        report.speaker_count.status = SpeakerCountOutcomeStatus::Unsatisfied;
        report.speaker_count.supported_speaker_count = 0;
        report.speaker_count.active_speaker_refs.clear();
        report.speaker_count.dominant_speaker_share = 0.0;
        report.speaker_count.unknown_voiced_share = 1.0;
        report.speaker_count.reasons = vec![SpeakerCountOutcomeReason::RequestedCountMismatch];
        report.speaker_count.speaker_evidence.clear();
        report
            .speaker_count
            .estimate
            .as_mut()
            .expect("estimate")
            .selected_count = None;
        report.fallback_status = DiarizationFallbackStatus::UnsatisfiedConstraints;
        report
            .validate_against_request(&request, None)
            .expect("no-usable hard hint remains explicit without fabricating identity");
    }

    #[test]
    fn report_request_binding_admits_only_honest_fallback_paths() {
        let mut acoustic_fallback = typed_diarization_report_fixture();
        let mut acoustic_fallback_request = matching_acoustic_request(&mut acoustic_fallback);
        acoustic_fallback_request.engine = DiarizationEngine::EcapaFused;
        acoustic_fallback_request.fallback = DiarizationFallbackPolicy::Acoustic;
        acoustic_fallback.neural_representation = Some(unavailable_neural_summary());
        acoustic_fallback.diagnostics =
            vec![DIARIZATION_DIAGNOSTIC_NATIVE_ACOUSTIC_FALLBACK.to_owned()];
        acoustic_fallback
            .validate_against_request(&acoustic_fallback_request, None)
            .expect("neural request may fall back to acoustic with attempt provenance");

        let mut forged_direct_acoustic = typed_diarization_report_fixture();
        let direct_acoustic_request = matching_acoustic_request(&mut forged_direct_acoustic);
        forged_direct_acoustic.diagnostics =
            vec![DIARIZATION_DIAGNOSTIC_NATIVE_ACOUSTIC_FALLBACK.to_owned()];
        assert!(
            forged_direct_acoustic
                .validate_against_request(&direct_acoustic_request, None)
                .expect_err("direct acoustic execution cannot claim a neural fallback")
                .contains("not canonical for the report implementation")
        );

        let mut forged_acoustic_fallback = acoustic_fallback.clone();
        forged_acoustic_fallback
            .neural_representation
            .as_mut()
            .expect("neural attempt")
            .provider_version = "unrecognized-ecapa-provider".to_owned();
        assert!(
            forged_acoustic_fallback
                .validate_against_request(&acoustic_fallback_request, None)
                .expect_err("ECAPA fallback provenance must bind the admitted provider")
                .contains("unrecognized provider version")
        );

        let external = external_diarization_report_fixture();
        let external_request = DiarizationRequest {
            engine: DiarizationEngine::External,
            fallback: DiarizationFallbackPolicy::Error,
            speaker_count: SpeakerCountRequest::HardConstraint { count: 1 },
            ..DiarizationRequest::default()
        };
        external
            .validate_against_request(&external_request, None)
            .expect("direct external report with preservable request semantics");

        let mut external_fallback = external.clone();
        external_fallback.neural_representation = Some(unavailable_neural_summary());
        external_fallback
            .diagnostics
            .push(DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_FALLBACK.to_owned());
        let neural_external_request = DiarizationRequest {
            engine: DiarizationEngine::Ecapa,
            fallback: DiarizationFallbackPolicy::External,
            speaker_count: SpeakerCountRequest::HardConstraint { count: 1 },
            ..DiarizationRequest::default()
        };
        external_fallback
            .validate_against_request(&neural_external_request, None)
            .expect("neural request may fall back externally with attempt provenance");

        let mut forged_direct_external = external.clone();
        forged_direct_external
            .diagnostics
            .push(DIARIZATION_DIAGNOSTIC_EXTERNAL_BACKEND_FALLBACK.to_owned());
        assert!(
            forged_direct_external
                .validate_against_request(&external_request, None)
                .expect_err("direct external execution cannot claim fallback routing")
                .contains("authorized execution path")
        );

        let mut lossy_external_request = external_request.clone();
        lossy_external_request.known_intervals = vec![speaker_hint(
            "near",
            0,
            500,
            KnownSpeakerPolicy::HardMustLink,
        )];
        assert!(
            external
                .validate_against_request(&lossy_external_request, None)
                .expect_err("external report cannot silently discard hints")
                .contains("cannot preserve known-speaker")
        );

        let unknown = unknown_diarization_report_fixture();
        let unknown_request = DiarizationRequest {
            engine: DiarizationEngine::External,
            fallback: DiarizationFallbackPolicy::Unknown,
            ..DiarizationRequest::default()
        };
        unknown
            .validate_against_request(&unknown_request, None)
            .expect("external unavailability may conservatively emit unknown");

        let mut forged_unknown_occupancy = unknown;
        forged_unknown_occupancy.speaker_count.unknown_voiced_share = 0.0;
        assert!(
            forged_unknown_occupancy
                .validate()
                .expect_err("unknown fallback must attribute all voiced occupancy to unknown")
                .contains("known voiced occupancy")
        );
    }

    #[test]
    fn request_resource_caps_and_error_fallback_policy_are_enforced() {
        let mut report = typed_diarization_report_fixture();
        let mut capped_request = matching_acoustic_request(&mut report);
        capped_request.max_prototypes = 4;
        assert!(
            report
                .validate_against_request(&capped_request, None)
                .expect_err("report resource use cannot exceed the authorized prototype cap")
                .contains("global prototype resource cap")
        );

        let mut forged_constraint_floor = typed_diarization_report_fixture();
        use_infer_count_request(&mut forged_constraint_floor);
        let hard_identity_request = DiarizationRequest {
            engine: DiarizationEngine::Acoustic,
            fallback: DiarizationFallbackPolicy::Unknown,
            speaker_count: SpeakerCountRequest::Infer,
            known_intervals: vec![
                speaker_hint("speaker-1", 0, 100, KnownSpeakerPolicy::HardMustLink),
                speaker_hint("speaker-2", 200, 300, KnownSpeakerPolicy::HardMustLink),
                speaker_hint("speaker-3", 400, 500, KnownSpeakerPolicy::HardMustLink),
            ],
            ..DiarizationRequest::default()
        };
        assert!(
            validate_report_request_resource_binding(
                &forged_constraint_floor,
                &hard_identity_request,
            )
            .expect_err("three hard identities require a lower bound of three")
            .contains("constraint lower bound")
        );

        let mut forged_infer_ceiling = typed_diarization_report_fixture();
        use_infer_count_request(&mut forged_infer_ceiling);
        let mut infer_request = matching_acoustic_request(&mut forged_infer_ceiling);
        infer_request.speaker_count = SpeakerCountRequest::Infer;
        let estimate = forged_infer_ceiling
            .speaker_count
            .estimate
            .as_mut()
            .expect("estimate");
        estimate.resources.prototype_count = 10;
        estimate.candidate_upper_bound = 10;
        assert!(
            forged_infer_ceiling
                .validate_against_request(&infer_request, None)
                .expect_err("automatic development inference is capped at eight candidates")
                .contains("candidate upper bound")
        );

        let mut policy_estimate = forged_infer_ceiling
            .speaker_count
            .estimate
            .clone()
            .expect("estimate");
        policy_estimate.calibration_status = SpeakerCountCalibrationStatus::FixedSafeUncalibrated;
        assert_eq!(
            canonical_speaker_count_candidate_upper_bound(
                &SpeakerCountRequest::Prior {
                    bins: vec![SpeakerCountPriorMass {
                        count: 12,
                        probability: 1.0,
                    }],
                },
                1,
                &policy_estimate,
            )
            .expect("fixed-safe prior ceiling"),
            12,
        );
        policy_estimate.calibration_status = SpeakerCountCalibrationStatus::Unavailable;
        policy_estimate.resources.prototype_count = 0;
        assert_eq!(
            canonical_speaker_count_candidate_upper_bound(
                &SpeakerCountRequest::Range {
                    minimum: 2,
                    maximum: 4,
                },
                1,
                &policy_estimate,
            )
            .expect("unavailable range ceiling"),
            4,
        );

        let mut native_fallback = typed_diarization_report_fixture();
        let mut error_request = matching_acoustic_request(&mut native_fallback);
        error_request.fallback = DiarizationFallbackPolicy::Error;
        native_fallback.fallback_status = DiarizationFallbackStatus::UnsatisfiedConstraints;
        assert!(
            native_fallback
                .validate_against_request(&error_request, None)
                .expect_err("error policy cannot return a native fallback")
                .contains("cannot return a native fallback")
        );

        let mut external = external_diarization_report_fixture();
        external.speaker_count.request = SpeakerCountRequest::HardConstraint { count: 2 };
        external.speaker_count.status = SpeakerCountOutcomeStatus::Unsatisfied;
        external.speaker_count.reasons = vec![
            SpeakerCountOutcomeReason::ExternalAttribution,
            SpeakerCountOutcomeReason::RequestedCountMismatch,
        ];
        let request = DiarizationRequest {
            engine: DiarizationEngine::External,
            fallback: DiarizationFallbackPolicy::Error,
            speaker_count: SpeakerCountRequest::HardConstraint { count: 2 },
            ..DiarizationRequest::default()
        };
        assert!(
            external
                .validate_against_request(&request, None)
                .expect_err("error policy cannot return unsatisfied external constraints")
                .contains("cannot return unsatisfied")
        );

        let mut partial_external = external_diarization_report_fixture();
        partial_external.speaker_count.request = SpeakerCountRequest::Infer;
        partial_external.speaker_count.status = SpeakerCountOutcomeStatus::Unresolved;
        partial_external.speaker_count.dominant_speaker_share = 0.6;
        partial_external.speaker_count.unknown_voiced_share = 0.4;
        partial_external.speaker_count.reasons = vec![
            SpeakerCountOutcomeReason::ExternalAttribution,
            SpeakerCountOutcomeReason::SpeakerCountEvidenceUnresolved,
        ];
        let partial_external_request = DiarizationRequest {
            engine: DiarizationEngine::External,
            fallback: DiarizationFallbackPolicy::Error,
            speaker_count: SpeakerCountRequest::Infer,
            ..DiarizationRequest::default()
        };
        assert!(
            partial_external
                .validate_against_request(&partial_external_request, None)
                .expect_err("error policy cannot return unresolved partial attribution")
                .contains("cannot return unresolved")
        );

        for unproduced_cause in [
            DiarizationFallbackStatus::InsufficientEvidence,
            DiarizationFallbackStatus::CalibrationInvalid,
            DiarizationFallbackStatus::ResourceLimit,
        ] {
            let mut forged_native_cause = typed_diarization_report_fixture();
            forged_native_cause.fallback_status = unproduced_cause;
            assert!(
                forged_native_cause
                    .validate()
                    .expect_err("native reports cannot claim an unproduced fallback cause")
                    .contains("no current producer emits")
            );
        }
    }

    #[test]
    fn diarization_report_validation_rejects_cross_object_inconsistency() {
        let mut report = typed_diarization_report_fixture();
        report.normalized_input_sha256 = "A".repeat(64);
        assert!(report.validate().unwrap_err().contains("normalized input"));

        let mut report = typed_diarization_report_fixture();
        report.turns[0].speaker_confidence = Some(1.1);
        assert!(
            report
                .validate()
                .unwrap_err()
                .contains("speaker_confidence")
        );

        let mut report = typed_diarization_report_fixture();
        report.profiles[0].reliability = f64::NAN;
        assert!(
            report
                .validate()
                .unwrap_err()
                .contains("profile reliability")
        );

        let mut report = typed_diarization_report_fixture();
        report.hint_evidence[0].accepted_tracklet_count = 0;
        assert!(report.validate().unwrap_err().contains("do not cover"));

        let mut report = typed_diarization_report_fixture();
        report.hint_evidence[0].hint_index = 1;
        assert!(
            report
                .validate()
                .unwrap_err()
                .contains("not contiguous from zero")
        );

        let mut report = typed_diarization_report_fixture();
        report.speaker_queries[0].query_id_sha256 = "invalid".to_owned();
        assert!(report.validate().unwrap_err().contains("query ID"));

        let mut report = typed_diarization_report_fixture();
        report.speaker_count.active_speaker_refs.clear();
        assert!(report.validate().unwrap_err().contains("active references"));

        let mut report = typed_diarization_report_fixture();
        report.speaker_count.unknown_voiced_share = 0.1;
        assert!(
            report
                .validate()
                .unwrap_err()
                .contains("common voiced-time denominator")
        );

        let mut report = typed_diarization_report_fixture();
        report
            .operational_partition
            .as_mut()
            .expect("partition")
            .calibration_sha256 = "f".repeat(64);
        assert!(report.validate().unwrap_err().contains("digests differ"));

        let mut fused = typed_diarization_report_fixture();
        fused.implementation = "native-ecapa-fused-v1".to_owned();
        fused.contract_version = "neural-diarization-common-v2".to_owned();
        fused.speaker_evidence_mode = DiarizationSpeakerEvidenceMode::EcapaWithAcousticChannel;
        fused.neural_representation = Some(ready_neural_summary());
        set_report_calibration_for_mode(
            &mut fused,
            DiarizationSpeakerEvidenceMode::EcapaWithAcousticChannel,
        )
        .expect("fused ECAPA test calibration");
        fused
            .operational_partition
            .as_mut()
            .expect("partition")
            .method = DiarizationOperationalPartitionMethod::EcapaSpherical;
        assert!(
            fused
                .validate()
                .unwrap_err()
                .contains("incompatible operational partition provenance")
        );

        fused
            .operational_partition
            .as_mut()
            .expect("partition")
            .method = DiarizationOperationalPartitionMethod::ProbabilisticConsensus;
        fused
            .validate()
            .expect("degraded fused ECAPA may underclaim generic consensus provenance");

        let mut private_diagnostic = typed_diarization_report_fixture();
        private_diagnostic.diagnostics = vec![
            "input_path=/Users/example/Downloads/private-call.m4a transcript=secret".to_owned(),
        ];
        assert!(
            private_diagnostic
                .validate()
                .expect_err("diagnostics cannot carry free-form or private payloads")
                .contains("not an admitted stable code")
        );
        let mut duplicate_diagnostic = typed_diarization_report_fixture();
        duplicate_diagnostic.diagnostics = vec![
            DIARIZATION_DIAGNOSTIC_NATIVE_ACOUSTIC_FALLBACK.to_owned(),
            DIARIZATION_DIAGNOSTIC_NATIVE_ACOUSTIC_FALLBACK.to_owned(),
        ];
        assert!(
            duplicate_diagnostic
                .validate()
                .expect_err("stable diagnostic codes are set-like")
                .contains("duplicate stable code")
        );

        let mut excessive_duration = typed_diarization_report_fixture();
        let request = matching_acoustic_request(&mut excessive_duration);
        excessive_duration.profiles[0].voiced_duration_ms = 1_501;
        assert!(
            excessive_duration
                .validate_against_request(&request, Some(1_500))
                .expect_err("per-speaker evidence cannot exceed the whole recording")
                .contains("evidence duration exceeds")
        );
    }

    #[test]
    fn attribution_queries_are_content_bound_and_reason_cardinality_is_exact() {
        let mut report = typed_diarization_report_fixture();
        report.speaker_queries[0].end_ms = 1_400;
        assert!(
            report
                .validate()
                .expect_err("query interval mutation must invalidate its ID")
                .contains("does not bind")
        );

        let mut report = typed_diarization_report_fixture();
        report.speaker_queries[0]
            .candidate_speaker_refs
            .push("other".to_owned());
        assert!(
            report
                .validate()
                .expect_err("low-confidence query must name exactly one candidate")
                .contains("candidate count")
        );

        let mut report = typed_diarization_report_fixture();
        report.speaker_queries[0].suggested_policy = KnownSpeakerPolicy::HardMustLink;
        assert!(
            report
                .validate()
                .expect_err("machine query cannot suggest immutable attribution")
                .contains("suggests a hard policy")
        );

        let mut report = typed_diarization_report_fixture();
        report.speaker_queries[0].reason = SpeakerAttributionQueryReason::OverlapAmbiguity;
        assert!(
            report
                .validate()
                .expect_err("overlap query must name exactly two candidates")
                .contains("candidate count")
        );

        let mut ungrounded_low_confidence = typed_diarization_report_fixture();
        ungrounded_low_confidence.speaker_queries[0].start_ms = 0;
        ungrounded_low_confidence.speaker_queries[0].end_ms = 500;
        ungrounded_low_confidence.speaker_queries[0].query_id_sha256 =
            speaker_attribution_query_sha256(
                &ungrounded_low_confidence.normalized_input_sha256,
                &ungrounded_low_confidence.speaker_queries[0],
            );
        assert!(
            ungrounded_low_confidence
                .validate()
                .expect_err("self-hashed query must still be grounded in low-confidence speech")
                .contains("not grounded")
        );

        let mut forged_unknown = typed_diarization_report_fixture();
        forged_unknown.speaker_queries[0].reason =
            SpeakerAttributionQueryReason::UnknownAttribution;
        forged_unknown.speaker_queries[0]
            .candidate_speaker_refs
            .clear();
        forged_unknown.speaker_queries[0].query_id_sha256 = speaker_attribution_query_sha256(
            &forged_unknown.normalized_input_sha256,
            &forged_unknown.speaker_queries[0],
        );
        assert!(
            forged_unknown
                .validate()
                .expect_err("self-hashed unknown query requires an overlapping unknown turn")
                .contains("not grounded")
        );

        let mut disjoint_overlap_query = typed_diarization_report_fixture();
        disjoint_overlap_query.turns = vec![
            DiarizationTurn {
                start_ms: 0,
                end_ms: 500,
                speaker_ref: Some("near".to_owned()),
                speaker_confidence: Some(0.5),
                change_confidence: None,
                overlap_suspected: true,
                hard_hint_attributed: false,
            },
            DiarizationTurn {
                start_ms: 1_000,
                end_ms: 1_500,
                speaker_ref: Some("other".to_owned()),
                speaker_confidence: Some(0.5),
                change_confidence: None,
                overlap_suspected: true,
                hard_hint_attributed: false,
            },
        ];
        let mut other_profile = disjoint_overlap_query.profiles[0].clone();
        other_profile.speaker_ref = "other".to_owned();
        disjoint_overlap_query.profiles.push(other_profile);
        let mut other_evidence = disjoint_overlap_query.speaker_count.speaker_evidence[0].clone();
        other_evidence.speaker_ref = "other".to_owned();
        disjoint_overlap_query
            .speaker_count
            .speaker_evidence
            .push(other_evidence);
        disjoint_overlap_query.speaker_count.supported_speaker_count = 2;
        disjoint_overlap_query.speaker_count.active_speaker_refs =
            vec!["near".to_owned(), "other".to_owned()];
        disjoint_overlap_query.speaker_queries = vec![SpeakerAttributionQuery {
            query_id_sha256: "f".repeat(64),
            start_ms: 0,
            end_ms: 1_500,
            reason: SpeakerAttributionQueryReason::OverlapAmbiguity,
            candidate_speaker_refs: vec!["near".to_owned(), "other".to_owned()],
            suggested_policy: KnownSpeakerPolicy::SoftEnrollment,
        }];
        assert!(
            validate_diarization_references(
                &disjoint_overlap_query,
                DiarizationReportKind::NativeAcoustic,
            )
            .expect_err("disjoint candidates cannot ground an overlap query")
            .contains("not grounded")
        );
    }

    #[test]
    fn turn_overlap_and_native_confidence_semantics_fail_closed() {
        let mut hard_overlap = typed_diarization_report_fixture();
        hard_overlap.turns[0].overlap_suspected = true;
        hard_overlap
            .validate()
            .expect("hard identity attribution remains valid during suspected overlap");

        let mut same_speaker_overlap = typed_diarization_report_fixture();
        same_speaker_overlap.turns[1].start_ms = 900;
        assert!(
            same_speaker_overlap
                .validate()
                .expect_err("same-speaker duplicate overlap is not a second voice")
                .contains("duplicate the same speaker")
        );

        let mut missing_confidence = typed_diarization_report_fixture();
        missing_confidence.turns[1].speaker_confidence = None;
        missing_confidence.speaker_queries.clear();
        assert!(
            missing_confidence
                .validate()
                .expect_err("native labeled turn must carry assignment confidence")
                .contains("omits speaker confidence")
        );
    }

    #[test]
    fn report_reference_sets_and_anchor_flags_are_exact() {
        let mut missing_profile = typed_diarization_report_fixture();
        missing_profile.profiles.clear();
        assert!(
            missing_profile
                .validate()
                .expect_err("every active speaker requires exactly one profile")
                .contains("profile references do not match")
        );

        let mut extra_profile = typed_diarization_report_fixture();
        let mut profile = extra_profile.profiles[0].clone();
        profile.speaker_ref = "other".to_owned();
        profile.anchored = false;
        extra_profile.profiles.push(profile);
        assert!(
            extra_profile
                .validate()
                .expect_err("inactive speaker cannot retain a public profile")
                .contains("profile references do not match")
        );

        let mut wrong_anchor = typed_diarization_report_fixture();
        wrong_anchor.profiles[0].anchored = false;
        assert!(
            wrong_anchor
                .validate()
                .expect_err("profile anchor must match speaker evidence")
                .contains("anchor flag disagrees")
        );

        let mut wrong_evidence_anchor = typed_diarization_report_fixture();
        wrong_evidence_anchor.speaker_count.speaker_evidence[0].hard_anchored = false;
        assert!(
            wrong_evidence_anchor
                .validate()
                .expect_err("hard-anchor evidence flag must match its reason")
                .contains("hard-anchor flag disagrees")
        );
    }

    #[test]
    fn report_rejects_external_evidence_on_native_implementations() {
        let mut report = typed_diarization_report_fixture();
        report
            .speaker_count
            .reasons
            .push(SpeakerCountOutcomeReason::ExternalAttribution);
        let error = report
            .validate()
            .expect_err("native outcome cannot claim external attribution");
        assert!(error.contains("claims external attribution"), "{error}");

        let mut report = typed_diarization_report_fixture();
        report.speaker_count.speaker_evidence[0]
            .reasons
            .push(SpeakerEvidenceReason::SupportedByExternalAttribution);
        let error = report
            .validate()
            .expect_err("native speaker evidence cannot claim external support");
        assert!(error.contains("exact canonical reasons"), "{error}");
    }

    #[test]
    fn speaker_count_status_reason_bounds_and_fallback_status_are_consistent() {
        let mut rounded_occupancy = typed_diarization_report_fixture();
        rounded_occupancy.speaker_count.dominant_speaker_share = f64::from(1.0_f32 / 3.0);
        rounded_occupancy.speaker_count.unknown_voiced_share = f64::from(2.0_f32 / 3.0);
        rounded_occupancy
            .validate()
            .expect("f32 producer rounding cannot invalidate a unit occupancy sum");

        let mut forged_dominance_fallback = typed_diarization_report_fixture();
        forged_dominance_fallback
            .speaker_count
            .supported_speaker_count = 2;
        forged_dominance_fallback
            .speaker_count
            .dominant_speaker_share = 0.99;
        forged_dominance_fallback
            .speaker_count
            .reasons
            .push(SpeakerCountOutcomeReason::DominantSpeakerShareExceeded);
        forged_dominance_fallback
            .operational_partition
            .as_mut()
            .expect("partition")
            .selected_count = 2;
        assert!(
            validate_report_kind_invariants(
                &forged_dominance_fallback,
                DiarizationReportKind::NativeAcoustic,
            )
            .expect_err("dominant-speaker breach requires a native fallback")
            .contains("no fallback was needed")
        );

        let mut report = typed_diarization_report_fixture();
        report.speaker_count.reasons = vec![SpeakerCountOutcomeReason::EvidenceSupportedCount];
        assert!(
            report
                .validate()
                .expect_err("satisfied hard count requires requested-count-matched")
                .contains("status disagrees")
        );

        let mut report = typed_diarization_report_fixture();
        report
            .speaker_count
            .reasons
            .push(SpeakerCountOutcomeReason::EvidenceSupportedCount);
        assert!(
            report
                .validate()
                .expect_err("outcome cannot carry two primary status reasons")
                .contains("exactly one status reason")
        );

        let mut report = typed_diarization_report_fixture();
        report.fallback_status = DiarizationFallbackStatus::SpeakerCountUnresolved;
        assert!(
            report
                .validate()
                .expect_err("resolved hard count cannot claim unresolved fallback")
                .contains("fallback status disagrees")
        );

        let mut report = typed_diarization_report_fixture();
        let estimate = report.speaker_count.estimate.as_mut().expect("estimate");
        estimate.candidate_upper_bound = 2;
        estimate.lanes = complete_speaker_count_lanes();
        assert!(
            report
                .validate()
                .expect_err("satisfied hard count must constrain the estimate domain exactly")
                .contains("estimate candidate upper bound")
        );

        let mut hard_with_caller_prior = typed_diarization_report_fixture();
        let caller_prior = hard_with_caller_prior
            .speaker_count
            .estimate
            .as_mut()
            .expect("estimate")
            .lanes
            .last_mut()
            .expect("caller-prior lane");
        caller_prior.available = true;
        caller_prior.proposed_count = Some(1);
        caller_prior.confidence = 1.0;
        caller_prior.unavailable_reason = None;
        assert!(
            hard_with_caller_prior
                .validate()
                .expect_err("hard request did not authorize caller-prior evidence")
                .contains("exact authorized request")
        );

        let mut prior_report = typed_diarization_report_fixture();
        use_infer_count_request(&mut prior_report);
        prior_report.speaker_count.request = SpeakerCountRequest::Prior {
            bins: vec![
                SpeakerCountPriorMass {
                    count: 1,
                    probability: 0.4,
                },
                SpeakerCountPriorMass {
                    count: 2,
                    probability: 0.6,
                },
            ],
        };
        let caller_prior = prior_report
            .speaker_count
            .estimate
            .as_mut()
            .expect("estimate")
            .lanes
            .last_mut()
            .expect("caller-prior lane");
        caller_prior.available = true;
        caller_prior.proposed_count = Some(2);
        caller_prior.confidence = 1.0;
        caller_prior.unavailable_reason = None;
        prior_report
            .validate()
            .expect("caller-prior lane exactly reflects the authorized prior");
        prior_report
            .speaker_count
            .estimate
            .as_mut()
            .expect("estimate")
            .lanes
            .last_mut()
            .expect("caller-prior lane")
            .confidence = 0.9;
        assert!(
            prior_report
                .validate()
                .expect_err("caller-prior confidence cannot be self-asserted")
                .contains("exact authorized request")
        );

        let mut infer_with_prior_failure = typed_diarization_report_fixture();
        use_infer_count_request(&mut infer_with_prior_failure);
        infer_with_prior_failure
            .speaker_count
            .estimate
            .as_mut()
            .expect("estimate")
            .selected_count = None;
        infer_with_prior_failure.speaker_count.status = SpeakerCountOutcomeStatus::Unresolved;
        infer_with_prior_failure.speaker_count.reasons =
            vec![SpeakerCountOutcomeReason::SpeakerCountPriorFusionUnavailable];
        infer_with_prior_failure.fallback_status =
            DiarizationFallbackStatus::SpeakerCountUnresolved;
        assert!(
            infer_with_prior_failure
                .validate()
                .expect_err("Infer cannot claim that unrequested prior fusion failed")
                .contains("lacks a prior request")
        );
    }

    #[test]
    fn native_speaker_support_uses_exact_producer_thresholds_and_provenance() {
        let minimum_voiced_frames = crate::diarization::MIN_SPEAKER_EVIDENCE_VOICED_FRAMES as u64;
        let minimum_assignment_confidence =
            f64::from(crate::diarization::MIN_SPEAKER_EVIDENCE_CONFIDENCE);
        let minimum_profile_reliability =
            f64::from(crate::diarization::MIN_SPEAKER_EVIDENCE_RELIABILITY);
        let evidence = SpeakerEvidenceSummary {
            speaker_ref: "speaker-1".to_owned(),
            assigned_tracklet_count: 2,
            independent_tracklet_count: 2,
            recurrence_episode_count: 1,
            voiced_frame_count: minimum_voiced_frames,
            independent_voiced_frame_count: minimum_voiced_frames,
            voiced_duration_ms: minimum_voiced_frames * 10,
            mean_assignment_confidence: minimum_assignment_confidence,
            profile_reliability: minimum_profile_reliability,
            hard_anchored: false,
            separated_from_supported_speakers: true,
            reasons: vec![SpeakerEvidenceReason::SupportedByRepeatedTracklets],
            supported: true,
        };
        let outcome = SpeakerCountOutcome {
            request: SpeakerCountRequest::Infer,
            estimate: None,
            status: SpeakerCountOutcomeStatus::Resolved,
            supported_speaker_count: 1,
            active_speaker_refs: vec!["speaker-1".to_owned()],
            dominant_speaker_share: 1.0,
            unknown_voiced_share: 0.0,
            reasons: vec![SpeakerCountOutcomeReason::EvidenceSupportedCount],
            speaker_evidence: vec![evidence],
        };
        validate_speaker_count_outcome(&outcome, DiarizationReportKind::NativeAcoustic)
            .expect("evidence exactly at every native producer threshold is supported");

        let mut forged_acoustic_heldout = outcome.clone();
        let heldout_evidence = &mut forged_acoustic_heldout.speaker_evidence[0];
        heldout_evidence.assigned_tracklet_count = 1;
        heldout_evidence.independent_tracklet_count = 1;
        heldout_evidence.recurrence_episode_count = 1;
        heldout_evidence.reasons = vec![SpeakerEvidenceReason::SupportedByHeldoutObservation];
        assert!(
            validate_speaker_count_outcome(
                &forged_acoustic_heldout,
                DiarizationReportKind::NativeAcoustic,
            )
            .expect_err("acoustic evidence cannot invent neural held-out support")
            .contains("held-out neural")
        );

        let mut below_frames = outcome.clone();
        below_frames.speaker_evidence[0].independent_voiced_frame_count -= 1;
        assert!(
            validate_speaker_count_outcome(&below_frames, DiarizationReportKind::NativeAcoustic,)
                .expect_err("sub-threshold voiced evidence cannot remain supported")
                .contains("exact support thresholds")
        );

        let mut below_confidence = outcome.clone();
        below_confidence.speaker_evidence[0].mean_assignment_confidence = f64::from(
            f32::from_bits(crate::diarization::MIN_SPEAKER_EVIDENCE_CONFIDENCE.to_bits() - 1),
        );
        assert!(
            validate_speaker_count_outcome(
                &below_confidence,
                DiarizationReportKind::NativeAcoustic,
            )
            .expect_err("sub-threshold assignment confidence cannot remain supported")
            .contains("exact support thresholds")
        );

        let mut below_reliability = outcome.clone();
        below_reliability.speaker_evidence[0].profile_reliability = f64::from(f32::from_bits(
            crate::diarization::MIN_SPEAKER_EVIDENCE_RELIABILITY.to_bits() - 1,
        ));
        assert!(
            validate_speaker_count_outcome(
                &below_reliability,
                DiarizationReportKind::NativeAcoustic,
            )
            .expect_err("sub-threshold profile reliability cannot remain supported")
            .contains("exact support thresholds")
        );

        let mut wrong_provenance = outcome.clone();
        wrong_provenance.speaker_evidence[0].recurrence_episode_count =
            crate::diarization::MIN_SPEAKER_EVIDENCE_RECURRENCE_EPISODES as u64;
        assert!(
            validate_speaker_count_outcome(
                &wrong_provenance,
                DiarizationReportKind::NativeAcoustic,
            )
            .expect_err("recurrence evidence cannot claim repeated-tracklet provenance")
            .contains("exact canonical reason")
        );

        let mut rejected = outcome;
        let rejected_evidence = &mut rejected.speaker_evidence[0];
        rejected_evidence.mean_assignment_confidence = f64::from(f32::from_bits(
            crate::diarization::MIN_SPEAKER_EVIDENCE_CONFIDENCE.to_bits() - 1,
        ));
        rejected_evidence.reasons = vec![SpeakerEvidenceReason::InsufficientAssignmentConfidence];
        rejected_evidence.supported = false;
        rejected.status = SpeakerCountOutcomeStatus::Unresolved;
        rejected.supported_speaker_count = 0;
        rejected.active_speaker_refs.clear();
        rejected.dominant_speaker_share = 0.0;
        rejected.reasons = vec![SpeakerCountOutcomeReason::NoSupportedSpeakers];
        validate_speaker_count_outcome(&rejected, DiarizationReportKind::NativeAcoustic)
            .expect("a rejected candidate carries the exact below-threshold reason");
    }

    #[test]
    fn diarization_report_validation_rejects_nested_and_mode_mismatches() {
        let mut report = typed_diarization_report_fixture();
        report
            .operational_partition
            .as_mut()
            .expect("partition")
            .selected_count = 0;
        assert!(
            report
                .validate()
                .expect_err("zero operational count must fail")
                .contains("operational partition")
        );

        let mut report = typed_diarization_report_fixture();
        report.implementation = "native-ecapa-only-v1".to_owned();
        report.contract_version = "neural-diarization-common-v2".to_owned();
        report.speaker_evidence_mode = DiarizationSpeakerEvidenceMode::EcapaOnly;
        set_report_calibration_for_mode(&mut report, DiarizationSpeakerEvidenceMode::EcapaOnly)
            .expect("ECAPA-only test calibration");
        assert!(
            report
                .validate()
                .expect_err("ECAPA mode without provenance must fail")
                .contains("requires neural representation provenance")
        );

        let mut report = typed_diarization_report_fixture();
        report
            .operational_partition
            .as_mut()
            .expect("partition")
            .method = DiarizationOperationalPartitionMethod::EcapaSpherical;
        assert!(
            report
                .validate()
                .expect_err("acoustic evidence cannot claim ECAPA spherical partition")
                .contains("native acoustic evidence")
        );

        let mut report = typed_diarization_report_fixture();
        report
            .operational_partition
            .as_mut()
            .expect("partition")
            .method = DiarizationOperationalPartitionMethod::EcapaFusedConsensus;
        assert!(
            report
                .validate()
                .expect_err("acoustic evidence cannot claim fused ECAPA consensus")
                .contains("native acoustic evidence")
        );

        for mode in [
            DiarizationSpeakerEvidenceMode::External,
            DiarizationSpeakerEvidenceMode::None,
        ] {
            let mut report = typed_diarization_report_fixture();
            report.speaker_evidence_mode = mode;
            assert!(
                report
                    .validate()
                    .expect_err("non-native evidence cannot claim a native partition")
                    .contains("requires contract")
            );
        }

        let mut report = typed_diarization_report_fixture();
        report.neural_representation = Some(NeuralSpeakerRepresentationSummary {
            schema_version: "malformed".to_owned(),
            provider_version: "ecapa-test-v1".to_owned(),
            expected_model_package_sha256: "d".repeat(64),
            loaded_model_package_sha256: None,
            model_load_source: None,
            status: NeuralSpeakerRepresentationStatus::Unavailable,
            embedded_tracklet_count: 0,
            zero_padded_tracklet_count: 0,
            skipped_tracklet_count: 0,
            reasons: vec![NeuralSpeakerRepresentationReason::ModelUnavailable],
        });
        assert!(
            report
                .validate()
                .expect_err("malformed neural summary must fail")
                .contains("neural representation")
        );
    }

    #[test]
    fn neural_summary_rejects_unknown_biometric_fields() {
        let value = serde_json::json!({
            "schema_version": "neural-speaker-representation-summary-v1",
            "provider_version": "ecapa-test-v1",
            "expected_model_package_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "status": "unavailable",
            "embedded_tracklet_count": 0,
            "zero_padded_tracklet_count": 0,
            "skipped_tracklet_count": 0,
            "reasons": ["model_unavailable"],
            "embedding": [0.1, 0.2]
        });
        assert!(
            serde_json::from_value::<NeuralSpeakerRepresentationSummary>(value).is_err(),
            "unknown biometric fields must fail closed"
        );
    }

    #[test]
    fn neural_model_source_rejects_obsolete_cache_claims() {
        let value = serde_json::to_value(ready_neural_summary()).expect("neural summary JSON");
        for obsolete in ["cache_hit", "cache_miss", "direct"] {
            let mut value = value.clone();
            value["model_load_source"] = json!(obsolete);
            assert!(
                serde_json::from_value::<NeuralSpeakerRepresentationSummary>(value).is_err(),
                "obsolete model source {obsolete} must fail closed"
            );
        }
    }

    #[test]
    fn neural_representation_summary_distinguishes_expected_from_loaded_model() {
        let expected = "d".repeat(64);
        let unavailable = NeuralSpeakerRepresentationSummary {
            schema_version: "neural-speaker-representation-summary-v1".to_owned(),
            provider_version: "ecapa-test-v1".to_owned(),
            expected_model_package_sha256: expected.clone(),
            loaded_model_package_sha256: None,
            model_load_source: None,
            status: NeuralSpeakerRepresentationStatus::Unavailable,
            embedded_tracklet_count: 0,
            zero_padded_tracklet_count: 0,
            skipped_tracklet_count: 0,
            reasons: vec![NeuralSpeakerRepresentationReason::ModelUnavailable],
        };
        unavailable.validate().expect("honest unavailable summary");

        let mut false_observation = unavailable;
        false_observation.loaded_model_package_sha256 = Some(expected);
        assert!(
            false_observation
                .validate()
                .expect_err("missing model cannot claim a loaded digest")
                .contains("claims a loaded package")
        );

        let mut padded_without_reason = ready_neural_summary();
        padded_without_reason.status = NeuralSpeakerRepresentationStatus::Degraded;
        padded_without_reason.zero_padded_tracklet_count = 1;
        padded_without_reason.reasons =
            vec![NeuralSpeakerRepresentationReason::InsufficientTracklets];
        assert!(
            padded_without_reason
                .validate()
                .expect_err("zero-padding requires typed short-tracklet provenance")
                .contains("short-tracklet provenance")
        );

        let mut duplicate_reasons = padded_without_reason;
        duplicate_reasons.reasons = vec![
            NeuralSpeakerRepresentationReason::ShortTracklet,
            NeuralSpeakerRepresentationReason::ShortTracklet,
        ];
        assert!(
            duplicate_reasons
                .validate()
                .expect_err("neural reasons must have one canonical set")
                .contains("duplicated")
        );
    }
}
