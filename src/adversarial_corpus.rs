//! Public-safe adversarial and metamorphic acoustic call generation.
//!
//! This module intentionally stores recipes rather than media. A retained
//! recipe contains synthetic speaker parameters, bounded perturbations, and
//! SHA-256 fingerprints; it cannot contain a source path, transcript, audio
//! samples, embedding, or real-world speaker identifier.

use std::collections::BTreeMap;
use std::f32::consts::TAU;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::audio::resample_mono_linear;
use crate::error::{FwError, FwResult};

/// Schema identity for retained synthetic call recipes.
pub const SYNTHETIC_CALL_SCHEMA_VERSION: &str = "adversarial-synthetic-call-v1";
/// Schema identity for retained transformation plans.
pub const TRANSFORM_PLAN_SCHEMA_VERSION: &str = "adversarial-transform-plan-v1";
/// Schema identity for path-free transformation evidence.
pub const TRANSFORM_EVIDENCE_SCHEMA_VERSION: &str = "adversarial-transform-evidence-v1";
/// Schema identity for minimized regression seeds.
pub const MINIMIZED_REPRO_SCHEMA_VERSION: &str = "adversarial-minimized-repro-v1";
/// Frozen implementation identity for recipe generation and execution.
pub const ADVERSARIAL_CORPUS_ENGINE_VERSION: &str = "adversarial-corpus-engine-v1";

const HASH_HEX_LEN: usize = 64;
const MAX_CHANNELS: usize = 2;
const MIN_SAMPLE_RATE_HZ: u32 = 8_000;
const MAX_SAMPLE_RATE_HZ: u32 = 48_000;
const MAX_INTERLEAVED_SAMPLES: usize = 64 * 1024 * 1024;
const MAX_SYNTHETIC_SPEAKERS: usize = 16;
const MAX_SYNTHETIC_TURNS: usize = 1_024;
const MAX_TRANSFORM_STEPS: usize = 64;
const MAX_REVERB_REPEATS: u8 = 8;
const MAX_MINIMIZER_EVALUATIONS: usize = 256;
const MILLION: u32 = 1_000_000;

/// An owned, finite, interleaved PCM buffer.
///
/// The audio itself is deliberately not serializable. Only its fingerprint and
/// shape may enter retained evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct AdversarialAudio {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
    pub channels: usize,
}

impl AdversarialAudio {
    /// Validate the bounded in-memory representation.
    pub fn validate(&self) -> FwResult<()> {
        if !(MIN_SAMPLE_RATE_HZ..=MAX_SAMPLE_RATE_HZ).contains(&self.sample_rate_hz) {
            return Err(adversarial_error(
                "sample_rate_out_of_range",
                "sample_rate_hz must be in 8000..=48000",
            ));
        }
        if !(1..=MAX_CHANNELS).contains(&self.channels) {
            return Err(adversarial_error(
                "channels_out_of_range",
                "channels must be one or two",
            ));
        }
        if self.samples.len() > MAX_INTERLEAVED_SAMPLES {
            return Err(adversarial_error(
                "audio_too_large",
                "interleaved sample count exceeds the bounded evaluation limit",
            ));
        }
        if !self.samples.len().is_multiple_of(self.channels) {
            return Err(adversarial_error(
                "partial_audio_frame",
                "interleaved sample count must be divisible by channels",
            ));
        }
        if self.samples.iter().any(|sample| !sample.is_finite()) {
            return Err(adversarial_error(
                "non_finite_audio",
                "audio samples must all be finite",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels.max(1)
    }

    /// Content-bind format and exact IEEE-754 sample bits.
    #[must_use]
    pub fn sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"franken-whisper-adversarial-audio-v1\0");
        hasher.update(self.sample_rate_hz.to_le_bytes());
        hasher.update((self.channels as u64).to_le_bytes());
        hasher.update((self.samples.len() as u64).to_le_bytes());
        for sample in &self.samples {
            hasher.update(sample.to_bits().to_le_bytes());
        }
        format!("{hasher:x}")
    }
}

/// A synthetic voice-like source profile. It encodes no identity attributes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticSpeakerProfile {
    /// Fundamental frequency in millihertz.
    pub fundamental_millihz: u32,
    /// Number of harmonics in the deterministic oscillator bank.
    pub harmonic_count: u8,
    /// Geometric attenuation applied between adjacent harmonics.
    pub harmonic_decay_millionths: u32,
    /// Peak profile amplitude before per-turn gain and the safety limiter.
    pub amplitude_millionths: u32,
    /// One-pole smoothing retention used as a stationary channel proxy.
    pub muffle_retention_millionths: u32,
    /// Stereo position in `-1_000_000..=1_000_000`; ignored for mono.
    pub stereo_position_millionths: i32,
}

/// One transcript-free synthetic speaking interval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticTurn {
    pub speaker_index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub gain_millionths: u32,
    /// Per-turn pitch movement in millihertz, used for voice-state challenges.
    pub pitch_shift_millihz: i32,
    /// Apply deterministic loudspeaker-like attenuation and coloration.
    pub playback: bool,
}

/// A complete public-safe recipe for a synthetic multi-speaker call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticCallPlan {
    pub schema_version: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub seed: u64,
    pub speakers: Vec<SyntheticSpeakerProfile>,
    pub turns: Vec<SyntheticTurn>,
}

impl SyntheticCallPlan {
    /// Construct a versioned plan. Validation occurs during materialization.
    #[must_use]
    pub fn new(
        sample_rate_hz: u32,
        channels: usize,
        seed: u64,
        speakers: Vec<SyntheticSpeakerProfile>,
        turns: Vec<SyntheticTurn>,
    ) -> Self {
        Self {
            schema_version: SYNTHETIC_CALL_SCHEMA_VERSION.to_owned(),
            sample_rate_hz,
            channels,
            seed,
            speakers,
            turns,
        }
    }

    pub fn sha256(&self) -> FwResult<String> {
        canonical_sha256(self)
    }
}

/// Transcript-free reference interval produced from a synthetic recipe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticReferenceTurn {
    pub speaker_index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// Path-free evidence for one synthetic source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticCallEvidence {
    pub schema_version: String,
    pub engine_version: String,
    pub plan_sha256: String,
    pub audio_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub frame_count: usize,
    pub speaker_count: usize,
    pub turn_count: usize,
}

/// Materialized in-memory audio plus safe reference and evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedSyntheticCall {
    pub audio: AdversarialAudio,
    pub reference_turns: Vec<SyntheticReferenceTurn>,
    pub evidence: SyntheticCallEvidence,
}

/// Generate a deterministic harmonic call from a path-free recipe.
pub fn generate_synthetic_call(
    plan: &SyntheticCallPlan,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<GeneratedSyntheticCall> {
    validate_synthetic_call_plan(plan)?;
    checkpoint_cancelled(is_cancelled)?;

    let duration_ms = plan
        .turns
        .iter()
        .map(|turn| turn.end_ms)
        .max()
        .unwrap_or_default();
    let frames = milliseconds_to_frames_ceil(duration_ms, plan.sample_rate_hz)?;
    let sample_count = frames.checked_mul(plan.channels).ok_or_else(|| {
        adversarial_error(
            "audio_size_overflow",
            "synthetic interleaved sample count overflowed",
        )
    })?;
    if sample_count > MAX_INTERLEAVED_SAMPLES {
        return Err(adversarial_error(
            "audio_too_large",
            "synthetic recipe exceeds the bounded evaluation limit",
        ));
    }

    let mut samples = vec![0.0f32; sample_count];
    for (turn_index, turn) in plan.turns.iter().enumerate() {
        checkpoint_cancelled(is_cancelled)?;
        render_synthetic_turn(&mut samples, plan, turn, turn_index)?;
    }
    for sample in &mut samples {
        *sample = sample.clamp(-1.0, 1.0);
    }

    let audio = AdversarialAudio {
        samples,
        sample_rate_hz: plan.sample_rate_hz,
        channels: plan.channels,
    };
    audio.validate()?;
    let plan_sha256 = plan.sha256()?;
    let audio_sha256 = audio.sha256();
    let reference_turns = plan
        .turns
        .iter()
        .map(|turn| SyntheticReferenceTurn {
            speaker_index: turn.speaker_index,
            start_ms: turn.start_ms,
            end_ms: turn.end_ms,
        })
        .collect();
    let evidence = SyntheticCallEvidence {
        schema_version: SYNTHETIC_CALL_SCHEMA_VERSION.to_owned(),
        engine_version: ADVERSARIAL_CORPUS_ENGINE_VERSION.to_owned(),
        plan_sha256,
        audio_sha256,
        sample_rate_hz: plan.sample_rate_hz,
        channels: plan.channels,
        frame_count: audio.frame_count(),
        speaker_count: plan.speakers.len(),
        turn_count: plan.turns.len(),
    };
    Ok(GeneratedSyntheticCall {
        audio,
        reference_turns,
        evidence,
    })
}

/// Generate a synthetic source without cancellation.
pub fn generate_synthetic_call_uncancellable(
    plan: &SyntheticCallPlan,
) -> FwResult<GeneratedSyntheticCall> {
    generate_synthetic_call(plan, &mut || false)
}

/// How a transform should affect a diarization result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum TransformExpectation {
    /// Speaker identity and time geometry should be preserved.
    SpeakerLabelsInvariant,
    /// Speaker identity should remain stable while signal quality degrades.
    SpeakerLabelsInvariantWithQualityLoss,
    /// All reference boundaries move by the declared positive offset.
    TimelineShift { delta_ms: u64 },
    /// Swapping stereo channels must not change speaker identity.
    ChannelPermutationInvariant,
    /// Simultaneous-speech evidence should increase in the declared interval.
    IncreasedOverlap,
}

/// One bounded deterministic perturbation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum AcousticPerturbation {
    Gain {
        gain_millionths: u32,
    },
    StationaryMuffle {
        retention_millionths: u32,
    },
    BandLimit {
        lower_hz: u32,
        upper_hz: u32,
    },
    ResampleRoundTrip {
        intermediate_rate_hz: u32,
    },
    Quantize {
        bits: u8,
    },
    Clip {
        threshold_millionths: u32,
    },
    AddNoise {
        amplitude_millionths: u32,
        seed_xor: u64,
    },
    Reverberate {
        delay_ms: u64,
        decay_millionths: u32,
        repeats: u8,
    },
    Interrupt {
        start_ms: u64,
        duration_ms: u64,
    },
    PadSilence {
        before_ms: u64,
        after_ms: u64,
    },
    SegmentChannelShift {
        start_ms: u64,
        end_ms: u64,
        gain_millionths: u32,
        muffle_retention_millionths: u32,
    },
    SpeakerPlayback {
        gain_millionths: u32,
        muffle_retention_millionths: u32,
        delay_ms: u64,
        decay_millionths: u32,
    },
    StereoChannelSwap,
    SelfOverlap {
        source_start_ms: u64,
        destination_start_ms: u64,
        duration_ms: u64,
        gain_millionths: u32,
    },
}

impl AcousticPerturbation {
    #[must_use]
    pub const fn expectation(&self) -> TransformExpectation {
        match self {
            Self::Gain { .. } => TransformExpectation::SpeakerLabelsInvariant,
            Self::StationaryMuffle { .. }
            | Self::BandLimit { .. }
            | Self::ResampleRoundTrip { .. }
            | Self::Quantize { .. }
            | Self::Clip { .. }
            | Self::AddNoise { .. }
            | Self::Reverberate { .. }
            | Self::Interrupt { .. }
            | Self::SegmentChannelShift { .. }
            | Self::SpeakerPlayback { .. } => {
                TransformExpectation::SpeakerLabelsInvariantWithQualityLoss
            }
            Self::PadSilence { before_ms, .. } => TransformExpectation::TimelineShift {
                delta_ms: *before_ms,
            },
            Self::StereoChannelSwap => TransformExpectation::ChannelPermutationInvariant,
            Self::SelfOverlap { .. } => TransformExpectation::IncreasedOverlap,
        }
    }
}

/// Proof boundary for sources admitted to the adversarial harness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum AdversarialSourceAuthority {
    /// The source is generated entirely from `SyntheticCallPlan`.
    Synthetic,
    /// The operator has separately verified a public or user-held license.
    PublicLicensed {
        /// Hash of an external acknowledgement or license record, never its path.
        acknowledgement_sha256: String,
    },
}

/// A content-bound sequence of perturbations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformPlan {
    pub schema_version: String,
    pub source_audio_sha256: String,
    pub source_authority: AdversarialSourceAuthority,
    pub seed: u64,
    pub steps: Vec<AcousticPerturbation>,
}

impl TransformPlan {
    /// Construct a plan whose source was produced by the synthetic generator.
    #[must_use]
    pub fn new_synthetic(
        source_audio_sha256: String,
        seed: u64,
        steps: Vec<AcousticPerturbation>,
    ) -> Self {
        Self {
            schema_version: TRANSFORM_PLAN_SCHEMA_VERSION.to_owned(),
            source_audio_sha256,
            source_authority: AdversarialSourceAuthority::Synthetic,
            seed,
            steps,
        }
    }

    /// Construct a plan for an externally held, license-cleared public source.
    #[must_use]
    pub fn new_public_licensed(
        source_audio_sha256: String,
        acknowledgement_sha256: String,
        seed: u64,
        steps: Vec<AcousticPerturbation>,
    ) -> Self {
        Self {
            schema_version: TRANSFORM_PLAN_SCHEMA_VERSION.to_owned(),
            source_audio_sha256,
            source_authority: AdversarialSourceAuthority::PublicLicensed {
                acknowledgement_sha256,
            },
            seed,
            steps,
        }
    }

    pub fn sha256(&self) -> FwResult<String> {
        canonical_sha256(self)
    }

    fn with_steps(&self, steps: Vec<AcousticPerturbation>) -> Self {
        Self {
            schema_version: self.schema_version.clone(),
            source_audio_sha256: self.source_audio_sha256.clone(),
            source_authority: self.source_authority.clone(),
            seed: self.seed,
            steps,
        }
    }
}

/// One node in the retained content-addressed transformation graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformGraphNode {
    pub step_index: usize,
    pub perturbation_sha256: String,
    pub input_audio_sha256: String,
    pub output_audio_sha256: String,
    pub expectation: TransformExpectation,
}

/// Path-free evidence produced by executing a transformation plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransformEvidence {
    pub schema_version: String,
    pub engine_version: String,
    pub plan_sha256: String,
    pub source_audio_sha256: String,
    pub output_audio_sha256: String,
    pub graph: Vec<TransformGraphNode>,
}

/// In-memory output plus safe transformation evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformedAudio {
    pub audio: AdversarialAudio,
    pub evidence: TransformEvidence,
}

/// Execute a bounded transformation plan with cancellation checkpoints.
pub fn apply_transform_plan(
    source: &AdversarialAudio,
    plan: &TransformPlan,
    is_cancelled: &mut impl FnMut() -> bool,
) -> FwResult<TransformedAudio> {
    source.validate()?;
    validate_transform_plan(plan)?;
    let source_sha256 = source.sha256();
    if source_sha256 != plan.source_audio_sha256 {
        return Err(adversarial_error(
            "source_hash_mismatch",
            "transform plan does not bind the supplied audio",
        ));
    }

    let mut audio = source.clone();
    let mut graph = Vec::with_capacity(plan.steps.len());
    for (step_index, perturbation) in plan.steps.iter().enumerate() {
        checkpoint_cancelled(is_cancelled)?;
        let input_audio_sha256 = audio.sha256();
        apply_perturbation(&mut audio, perturbation, plan.seed, step_index)?;
        audio.validate()?;
        let output_audio_sha256 = audio.sha256();
        graph.push(TransformGraphNode {
            step_index,
            perturbation_sha256: canonical_sha256(perturbation)?,
            input_audio_sha256,
            output_audio_sha256,
            expectation: perturbation.expectation(),
        });
    }

    let evidence = TransformEvidence {
        schema_version: TRANSFORM_EVIDENCE_SCHEMA_VERSION.to_owned(),
        engine_version: ADVERSARIAL_CORPUS_ENGINE_VERSION.to_owned(),
        plan_sha256: plan.sha256()?,
        source_audio_sha256,
        output_audio_sha256: audio.sha256(),
        graph,
    };
    Ok(TransformedAudio { audio, evidence })
}

/// Execute a transformation plan without cancellation.
pub fn apply_transform_plan_uncancellable(
    source: &AdversarialAudio,
    plan: &TransformPlan,
) -> FwResult<TransformedAudio> {
    apply_transform_plan(source, plan, &mut || false)
}

/// Shift transcript-free reference boundaries through time-moving transforms.
pub fn transform_reference_turns(
    reference: &[SyntheticReferenceTurn],
    plan: &TransformPlan,
) -> FwResult<Vec<SyntheticReferenceTurn>> {
    validate_transform_plan(plan)?;
    for (index, turn) in reference.iter().enumerate() {
        if turn.start_ms >= turn.end_ms {
            return Err(adversarial_error(
                "invalid_reference_turn",
                &format!("reference turn {index} must have positive duration"),
            ));
        }
    }
    let mut shift_ms = 0u64;
    for step in &plan.steps {
        if let AcousticPerturbation::PadSilence { before_ms, .. } = step {
            shift_ms = shift_ms.checked_add(*before_ms).ok_or_else(|| {
                adversarial_error(
                    "timeline_overflow",
                    "cumulative reference time shift overflowed",
                )
            })?;
        }
    }
    reference
        .iter()
        .map(|turn| {
            let start_ms = turn.start_ms.checked_add(shift_ms).ok_or_else(|| {
                adversarial_error("timeline_overflow", "reference start time overflowed")
            })?;
            let end_ms = turn.end_ms.checked_add(shift_ms).ok_or_else(|| {
                adversarial_error("timeline_overflow", "reference end time overflowed")
            })?;
            Ok(SyntheticReferenceTurn {
                speaker_index: turn.speaker_index,
                start_ms,
                end_ms,
            })
        })
        .collect()
}

/// Ordered pipeline stages used for first-divergence attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdversarialPipelineStage {
    Input,
    Normalization,
    SpeechMask,
    FeatureExtraction,
    ChangeDetection,
    Clustering,
    Projection,
    Scoring,
}

/// Hash of a path-free aggregate at one pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageFingerprint {
    pub stage: AdversarialPipelineStage,
    pub sha256: String,
}

/// Earliest stage whose aggregate fingerprint differs or is missing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageDivergence {
    pub stage: AdversarialPipelineStage,
    pub baseline_sha256: Option<String>,
    pub candidate_sha256: Option<String>,
}

/// Find the earliest divergent stage independently of caller vector order.
pub fn first_stage_divergence(
    baseline: &[StageFingerprint],
    candidate: &[StageFingerprint],
) -> FwResult<Option<StageDivergence>> {
    let baseline = validate_stage_fingerprints(baseline, "baseline")?;
    let candidate = validate_stage_fingerprints(candidate, "candidate")?;
    const STAGES: [AdversarialPipelineStage; 8] = [
        AdversarialPipelineStage::Input,
        AdversarialPipelineStage::Normalization,
        AdversarialPipelineStage::SpeechMask,
        AdversarialPipelineStage::FeatureExtraction,
        AdversarialPipelineStage::ChangeDetection,
        AdversarialPipelineStage::Clustering,
        AdversarialPipelineStage::Projection,
        AdversarialPipelineStage::Scoring,
    ];
    for stage in STAGES {
        let baseline_sha256 = baseline.get(&stage).cloned();
        let candidate_sha256 = candidate.get(&stage).cloned();
        if baseline_sha256 != candidate_sha256 {
            return Ok(Some(StageDivergence {
                stage,
                baseline_sha256,
                candidate_sha256,
            }));
        }
    }
    Ok(None)
}

/// Stable failure identity used by the deterministic minimizer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegressionClassification {
    pub code: String,
    pub first_divergent_stage: AdversarialPipelineStage,
}

/// Retained, content-free result of deterministic delta minimization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MinimizedReproSeed {
    pub schema_version: String,
    pub engine_version: String,
    pub classification: RegressionClassification,
    pub original_plan_sha256: String,
    pub minimized_plan_sha256: String,
    pub minimized_plan: TransformPlan,
    pub retained_original_step_indices: Vec<usize>,
    pub evaluation_count: usize,
}

/// Deterministically remove perturbations while preserving exact failure class.
pub fn minimize_failing_plan(
    plan: &TransformPlan,
    expected: &RegressionClassification,
    mut classify: impl FnMut(&TransformPlan) -> FwResult<Option<RegressionClassification>>,
) -> FwResult<MinimizedReproSeed> {
    validate_transform_plan(plan)?;
    validate_regression_classification(expected)?;
    let original_plan_sha256 = plan.sha256()?;
    let initial = classify(plan)?;
    let mut evaluation_count = 1usize;
    if initial.as_ref() != Some(expected) {
        return Err(adversarial_error(
            "failure_classification_mismatch",
            "the original plan does not reproduce the expected failure",
        ));
    }

    let mut retained: Vec<(usize, AcousticPerturbation)> =
        plan.steps.iter().cloned().enumerate().collect();
    let mut granularity = 2usize;
    while !retained.is_empty() && evaluation_count < MAX_MINIMIZER_EVALUATIONS {
        let chunk_size = retained.len().div_ceil(granularity);
        let mut reduced = false;
        let mut start = 0usize;
        while start < retained.len() && evaluation_count < MAX_MINIMIZER_EVALUATIONS {
            let end = (start + chunk_size).min(retained.len());
            let candidate_items: Vec<_> = retained
                .iter()
                .enumerate()
                .filter(|(index, _)| *index < start || *index >= end)
                .map(|(_, item)| item.clone())
                .collect();
            let candidate = plan.with_steps(
                candidate_items
                    .iter()
                    .map(|(_, step)| step.clone())
                    .collect(),
            );
            evaluation_count += 1;
            if classify(&candidate)?.as_ref() == Some(expected) {
                retained = candidate_items;
                granularity = granularity.saturating_sub(1).max(2);
                reduced = true;
                break;
            }
            start = end;
        }
        if reduced {
            continue;
        }
        if granularity >= retained.len() {
            break;
        }
        granularity = (granularity * 2).min(retained.len());
    }

    let minimized_plan = plan.with_steps(retained.iter().map(|(_, step)| step.clone()).collect());
    let minimized_plan_sha256 = minimized_plan.sha256()?;
    Ok(MinimizedReproSeed {
        schema_version: MINIMIZED_REPRO_SCHEMA_VERSION.to_owned(),
        engine_version: ADVERSARIAL_CORPUS_ENGINE_VERSION.to_owned(),
        classification: expected.clone(),
        original_plan_sha256,
        minimized_plan_sha256,
        retained_original_step_indices: retained.iter().map(|(index, _)| *index).collect(),
        minimized_plan,
        evaluation_count,
    })
}

/// Integer-valued comparison suitable for path-free consistency evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioConsistencyMeasurement {
    pub expected_frame_count: usize,
    pub observed_frame_count: usize,
    pub frame_count_delta: i64,
    pub max_abs_error_millionths: u64,
    pub rms_error_millionths: u64,
    pub exact_sample_bits: bool,
}

/// Measure chunking or resampling consistency without retaining samples.
pub fn measure_audio_consistency(
    expected: &AdversarialAudio,
    observed: &AdversarialAudio,
) -> FwResult<AudioConsistencyMeasurement> {
    expected.validate()?;
    observed.validate()?;
    if expected.sample_rate_hz != observed.sample_rate_hz || expected.channels != observed.channels
    {
        return Err(adversarial_error(
            "incompatible_audio_shapes",
            "consistency inputs must have the same sample rate and channel count",
        ));
    }
    let compared = expected.samples.len().min(observed.samples.len());
    let mut max_abs_error = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut exact_sample_bits = expected.samples.len() == observed.samples.len();
    for index in 0..compared {
        let left = expected.samples[index];
        let right = observed.samples[index];
        exact_sample_bits &= left.to_bits() == right.to_bits();
        let error = f64::from((left - right).abs());
        max_abs_error = max_abs_error.max(error);
        squared_error += error * error;
    }
    let missing = expected.samples.len().abs_diff(observed.samples.len());
    if missing > 0 {
        exact_sample_bits = false;
        max_abs_error = max_abs_error.max(1.0);
        squared_error += missing as f64;
    }
    let denominator = expected.samples.len().max(observed.samples.len()).max(1) as f64;
    let rms_error = (squared_error / denominator).sqrt();
    Ok(AudioConsistencyMeasurement {
        expected_frame_count: expected.frame_count(),
        observed_frame_count: observed.frame_count(),
        frame_count_delta: signed_difference(observed.frame_count(), expected.frame_count()),
        max_abs_error_millionths: scaled_error(max_abs_error),
        rms_error_millionths: scaled_error(rms_error),
        exact_sample_bits,
    })
}

/// Public-safe acoustic challenge families required by the v1 matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcousticChallengeFamily {
    GainDistanceImbalance,
    StationaryEqMuffling,
    BandLimitation,
    ResamplingCodecDistortion,
    Clipping,
    Noise,
    Reverberation,
    Interruptions,
    Silence,
    RapidTurns,
    LongTurns,
    SimilarPitch,
    VoiceStateShift,
    WithinSpeakerChannelChange,
    SpeakerPlayback,
    StereoChannelSwap,
    ControlledOverlap,
}

/// A complete content-free seed for one acoustic challenge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcousticChallengeSeed {
    pub family: AcousticChallengeFamily,
    pub synthetic_call: SyntheticCallPlan,
    pub transform: TransformPlan,
}

/// Materialized challenge audio, reference, and path-free evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedAcousticChallenge {
    pub family: AcousticChallengeFamily,
    pub audio: AdversarialAudio,
    pub reference_turns: Vec<SyntheticReferenceTurn>,
    pub source_evidence: SyntheticCallEvidence,
    pub transform_evidence: TransformEvidence,
}

impl AcousticChallengeSeed {
    pub fn sha256(&self) -> FwResult<String> {
        canonical_sha256(self)
    }

    pub fn materialize(&self) -> FwResult<MaterializedAcousticChallenge> {
        let source = generate_synthetic_call_uncancellable(&self.synthetic_call)?;
        let transformed = apply_transform_plan_uncancellable(&source.audio, &self.transform)?;
        let reference_turns = transform_reference_turns(&source.reference_turns, &self.transform)?;
        Ok(MaterializedAcousticChallenge {
            family: self.family,
            audio: transformed.audio,
            reference_turns,
            source_evidence: source.evidence,
            transform_evidence: transformed.evidence,
        })
    }
}

/// Return one deterministic, public-safe seed for every required challenge.
pub fn known_acoustic_challenge_seeds() -> FwResult<Vec<AcousticChallengeSeed>> {
    use AcousticChallengeFamily as Family;
    use AcousticPerturbation as Perturbation;

    let families = [
        Family::GainDistanceImbalance,
        Family::StationaryEqMuffling,
        Family::BandLimitation,
        Family::ResamplingCodecDistortion,
        Family::Clipping,
        Family::Noise,
        Family::Reverberation,
        Family::Interruptions,
        Family::Silence,
        Family::RapidTurns,
        Family::LongTurns,
        Family::SimilarPitch,
        Family::VoiceStateShift,
        Family::WithinSpeakerChannelChange,
        Family::SpeakerPlayback,
        Family::StereoChannelSwap,
        Family::ControlledOverlap,
    ];
    let mut seeds = Vec::with_capacity(families.len());
    for (index, family) in families.into_iter().enumerate() {
        let mut source_plan = base_synthetic_plan(2, index as u64 + 1);
        let steps = match family {
            Family::GainDistanceImbalance => {
                source_plan.speakers[1].amplitude_millionths = 220_000;
                vec![Perturbation::Gain {
                    gain_millionths: 850_000,
                }]
            }
            Family::StationaryEqMuffling => vec![Perturbation::StationaryMuffle {
                retention_millionths: 940_000,
            }],
            Family::BandLimitation => vec![Perturbation::BandLimit {
                lower_hz: 250,
                upper_hz: 3_200,
            }],
            Family::ResamplingCodecDistortion => vec![
                Perturbation::ResampleRoundTrip {
                    intermediate_rate_hz: 8_000,
                },
                Perturbation::Quantize { bits: 8 },
            ],
            Family::Clipping => vec![Perturbation::Clip {
                threshold_millionths: 180_000,
            }],
            Family::Noise => vec![Perturbation::AddNoise {
                amplitude_millionths: 120_000,
                seed_xor: 0x4e4f_4953_455f_5631,
            }],
            Family::Reverberation => vec![Perturbation::Reverberate {
                delay_ms: 75,
                decay_millionths: 420_000,
                repeats: 4,
            }],
            Family::Interruptions => vec![Perturbation::Interrupt {
                start_ms: 1_050,
                duration_ms: 300,
            }],
            Family::Silence => vec![Perturbation::PadSilence {
                before_ms: 700,
                after_ms: 900,
            }],
            Family::RapidTurns => {
                source_plan.turns = rapid_turns();
                Vec::new()
            }
            Family::LongTurns => {
                source_plan.turns =
                    vec![synthetic_turn(0, 0, 4_000), synthetic_turn(1, 4_200, 8_200)];
                Vec::new()
            }
            Family::SimilarPitch => {
                source_plan.speakers[0].fundamental_millihz = 148_000;
                source_plan.speakers[1].fundamental_millihz = 153_000;
                Vec::new()
            }
            Family::VoiceStateShift => {
                source_plan.turns[2].pitch_shift_millihz = 38_000;
                Vec::new()
            }
            Family::WithinSpeakerChannelChange => {
                vec![Perturbation::SegmentChannelShift {
                    start_ms: 1_900,
                    end_ms: 3_100,
                    gain_millionths: 480_000,
                    muffle_retention_millionths: 930_000,
                }]
            }
            Family::SpeakerPlayback => vec![Perturbation::SpeakerPlayback {
                gain_millionths: 620_000,
                muffle_retention_millionths: 950_000,
                delay_ms: 55,
                decay_millionths: 300_000,
            }],
            Family::StereoChannelSwap => vec![Perturbation::StereoChannelSwap],
            Family::ControlledOverlap => {
                source_plan.turns = vec![
                    synthetic_turn(0, 0, 2_400),
                    synthetic_turn(1, 1_400, 3_600),
                    synthetic_turn(0, 3_800, 5_000),
                ];
                vec![Perturbation::SelfOverlap {
                    source_start_ms: 250,
                    destination_start_ms: 4_050,
                    duration_ms: 550,
                    gain_millionths: 420_000,
                }]
            }
        };
        let source = generate_synthetic_call_uncancellable(&source_plan)?;
        let transform =
            TransformPlan::new_synthetic(source.audio.sha256(), 10_000 + index as u64, steps);
        seeds.push(AcousticChallengeSeed {
            family,
            synthetic_call: source_plan,
            transform,
        });
    }
    Ok(seeds)
}

fn validate_synthetic_call_plan(plan: &SyntheticCallPlan) -> FwResult<()> {
    if plan.schema_version != SYNTHETIC_CALL_SCHEMA_VERSION {
        return Err(adversarial_error(
            "synthetic_schema_mismatch",
            "unsupported synthetic call schema_version",
        ));
    }
    if !(MIN_SAMPLE_RATE_HZ..=MAX_SAMPLE_RATE_HZ).contains(&plan.sample_rate_hz) {
        return Err(adversarial_error(
            "sample_rate_out_of_range",
            "sample_rate_hz must be in 8000..=48000",
        ));
    }
    if !(1..=MAX_CHANNELS).contains(&plan.channels) {
        return Err(adversarial_error(
            "channels_out_of_range",
            "channels must be one or two",
        ));
    }
    if plan.speakers.is_empty() || plan.speakers.len() > MAX_SYNTHETIC_SPEAKERS {
        return Err(adversarial_error(
            "speaker_count_out_of_range",
            "synthetic speaker count must be in 1..=16",
        ));
    }
    if plan.turns.is_empty() || plan.turns.len() > MAX_SYNTHETIC_TURNS {
        return Err(adversarial_error(
            "turn_count_out_of_range",
            "synthetic turn count must be in 1..=1024",
        ));
    }
    for (index, speaker) in plan.speakers.iter().enumerate() {
        let fundamental_hz = speaker.fundamental_millihz / 1_000;
        if !(60..=500).contains(&fundamental_hz)
            || speaker.harmonic_count == 0
            || speaker.harmonic_count > 8
            || speaker.harmonic_decay_millionths > MILLION
            || speaker.amplitude_millionths == 0
            || speaker.amplitude_millionths > MILLION
            || speaker.muffle_retention_millionths >= MILLION
            || !(-1_000_000..=1_000_000).contains(&speaker.stereo_position_millionths)
        {
            return Err(adversarial_error(
                "invalid_speaker_profile",
                &format!("synthetic speaker profile {index} is outside its bounded domain"),
            ));
        }
        let highest_harmonic_hz =
            u64::from(speaker.fundamental_millihz) * u64::from(speaker.harmonic_count) / 1_000;
        if highest_harmonic_hz >= u64::from(plan.sample_rate_hz / 2) {
            return Err(adversarial_error(
                "speaker_profile_aliasing",
                &format!("synthetic speaker profile {index} exceeds Nyquist"),
            ));
        }
    }
    for (index, turn) in plan.turns.iter().enumerate() {
        if turn.speaker_index >= plan.speakers.len()
            || turn.start_ms >= turn.end_ms
            || turn.gain_millionths == 0
            || turn.gain_millionths > MILLION
        {
            return Err(adversarial_error(
                "invalid_synthetic_turn",
                &format!("synthetic turn {index} is outside its bounded domain"),
            ));
        }
        let speaker = &plan.speakers[turn.speaker_index];
        let shifted = i64::from(speaker.fundamental_millihz) + i64::from(turn.pitch_shift_millihz);
        if !(60_000..=500_000).contains(&shifted) {
            return Err(adversarial_error(
                "turn_pitch_out_of_range",
                &format!("synthetic turn {index} pitch is outside 60..=500 Hz"),
            ));
        }
    }
    Ok(())
}

fn render_synthetic_turn(
    destination: &mut [f32],
    plan: &SyntheticCallPlan,
    turn: &SyntheticTurn,
    turn_index: usize,
) -> FwResult<()> {
    let start = milliseconds_to_frames_floor(turn.start_ms, plan.sample_rate_hz)?;
    let end = milliseconds_to_frames_ceil(turn.end_ms, plan.sample_rate_hz)?;
    let profile = &plan.speakers[turn.speaker_index];
    let shifted_millihz =
        i64::from(profile.fundamental_millihz) + i64::from(turn.pitch_shift_millihz);
    let fundamental_hz = shifted_millihz as f32 / 1_000.0;
    let profile_gain = profile.amplitude_millionths as f32 / MILLION as f32
        * turn.gain_millionths as f32
        / MILLION as f32;
    let playback_gain = if turn.playback { 0.62 } else { 1.0 };
    let retention = if turn.playback {
        profile.muffle_retention_millionths.max(940_000)
    } else {
        profile.muffle_retention_millionths
    } as f32
        / MILLION as f32;
    let mut channel_state = [0.0f32; MAX_CHANNELS];
    let attack_frames = (plan.sample_rate_hz as usize / 100).max(1);
    let release_start = end.saturating_sub(attack_frames);
    let mut harmonic_weight_sum = 0.0f32;
    let mut weight = 1.0f32;
    for _ in 0..profile.harmonic_count {
        harmonic_weight_sum += weight;
        weight *= profile.harmonic_decay_millionths as f32 / MILLION as f32;
    }
    for frame in start..end {
        let time = frame as f32 / plan.sample_rate_hz as f32;
        let local_frame = frame - start;
        let attack = (local_frame as f32 / attack_frames as f32).min(1.0);
        let release = if frame >= release_start {
            ((end - frame) as f32 / attack_frames as f32).min(1.0)
        } else {
            1.0
        };
        let envelope = attack.min(release);
        let modulation = 0.94 + 0.06 * (TAU * 2.3 * time).sin();
        let mut raw = 0.0f32;
        let mut harmonic_weight = 1.0f32;
        for harmonic in 1..=profile.harmonic_count {
            let phase_seed = plan.seed
                ^ ((turn.speaker_index as u64 + 1) << 32)
                ^ ((turn_index as u64 + 1) << 8)
                ^ u64::from(harmonic);
            let phase = unit_phase(phase_seed);
            raw += harmonic_weight * (TAU * fundamental_hz * harmonic as f32 * time + phase).sin();
            harmonic_weight *= profile.harmonic_decay_millionths as f32 / MILLION as f32;
        }
        raw = raw / harmonic_weight_sum.max(f32::EPSILON)
            * profile_gain
            * playback_gain
            * envelope
            * modulation;
        for channel in 0..plan.channels {
            let channel_gain =
                synthetic_channel_gain(channel, plan.channels, profile.stereo_position_millionths);
            let filtered = raw.mul_add(1.0 - retention, retention * channel_state[channel]);
            channel_state[channel] = filtered;
            let index = frame
                .checked_mul(plan.channels)
                .and_then(|value| value.checked_add(channel))
                .ok_or_else(|| {
                    adversarial_error("audio_size_overflow", "synthetic frame index overflowed")
                })?;
            destination[index] += filtered * channel_gain;
        }
    }
    Ok(())
}

fn synthetic_channel_gain(channel: usize, channels: usize, position_millionths: i32) -> f32 {
    if channels == 1 {
        return 1.0;
    }
    let position = position_millionths as f32 / 1_000_000.0;
    if channel == 0 {
        1.0 - position.max(0.0) * 0.65
    } else {
        1.0 + position.min(0.0) * 0.65
    }
}

fn unit_phase(seed: u64) -> f32 {
    let mixed = mix_seed(seed);
    (mixed >> 40) as f32 / (1u64 << 24) as f32 * TAU
}

fn validate_transform_plan(plan: &TransformPlan) -> FwResult<()> {
    if plan.schema_version != TRANSFORM_PLAN_SCHEMA_VERSION {
        return Err(adversarial_error(
            "transform_schema_mismatch",
            "unsupported transform plan schema_version",
        ));
    }
    validate_sha256(&plan.source_audio_sha256, "source_audio_sha256")?;
    match &plan.source_authority {
        AdversarialSourceAuthority::Synthetic => {}
        AdversarialSourceAuthority::PublicLicensed {
            acknowledgement_sha256,
        } => validate_sha256(
            acknowledgement_sha256,
            "source_authority.acknowledgement_sha256",
        )?,
    }
    if plan.steps.len() > MAX_TRANSFORM_STEPS {
        return Err(adversarial_error(
            "too_many_transform_steps",
            "transform plans may contain at most 64 steps",
        ));
    }
    for (index, step) in plan.steps.iter().enumerate() {
        validate_perturbation(step).map_err(|error| match error {
            FwError::InvalidRequest(message) => adversarial_error(
                "invalid_transform_step",
                &format!("step {index}: {message}"),
            ),
            other => other,
        })?;
    }
    Ok(())
}

fn validate_perturbation(perturbation: &AcousticPerturbation) -> FwResult<()> {
    use AcousticPerturbation as Perturbation;
    let valid_fraction = |value: u32| value > 0 && value <= MILLION;
    match perturbation {
        Perturbation::Gain { gain_millionths } if !valid_fraction(*gain_millionths) => {
            Err(adversarial_error(
                "gain_out_of_range",
                "gain_millionths must be in 1..=1000000",
            ))
        }
        Perturbation::StationaryMuffle {
            retention_millionths,
        } if *retention_millionths >= MILLION => Err(adversarial_error(
            "muffle_out_of_range",
            "retention_millionths must be below 1000000",
        )),
        Perturbation::BandLimit { lower_hz, upper_hz }
            if *lower_hz >= *upper_hz || *upper_hz > MAX_SAMPLE_RATE_HZ / 2 =>
        {
            Err(adversarial_error(
                "band_limit_out_of_range",
                "band limits must be ordered and upper_hz must be at most 24000",
            ))
        }
        Perturbation::ResampleRoundTrip {
            intermediate_rate_hz,
        } if !(MIN_SAMPLE_RATE_HZ..=MAX_SAMPLE_RATE_HZ).contains(intermediate_rate_hz) => {
            Err(adversarial_error(
                "resample_rate_out_of_range",
                "intermediate_rate_hz must be in 8000..=48000",
            ))
        }
        Perturbation::Quantize { bits } if !(4..=24).contains(bits) => Err(adversarial_error(
            "quantization_bits_out_of_range",
            "quantization bits must be in 4..=24",
        )),
        Perturbation::Clip {
            threshold_millionths,
        } if !(10_000..=MILLION).contains(threshold_millionths) => Err(adversarial_error(
            "clip_threshold_out_of_range",
            "clip threshold must be in 10000..=1000000",
        )),
        Perturbation::AddNoise {
            amplitude_millionths,
            ..
        } if *amplitude_millionths > MILLION => Err(adversarial_error(
            "noise_amplitude_out_of_range",
            "noise amplitude must be at most 1000000",
        )),
        Perturbation::Reverberate {
            delay_ms,
            decay_millionths,
            repeats,
        } if *delay_ms == 0
            || !valid_fraction(*decay_millionths)
            || *repeats == 0
            || *repeats > MAX_REVERB_REPEATS =>
        {
            Err(adversarial_error(
                "reverb_out_of_range",
                "reverb requires positive delay, decay in 1..=1000000, and repeats in 1..=8",
            ))
        }
        Perturbation::Interrupt { duration_ms, .. } if *duration_ms == 0 => Err(adversarial_error(
            "interrupt_out_of_range",
            "interrupt duration must be positive",
        )),
        Perturbation::PadSilence {
            before_ms,
            after_ms,
        } if *before_ms == 0 && *after_ms == 0 => Err(adversarial_error(
            "silence_padding_empty",
            "at least one silence padding side must be positive",
        )),
        Perturbation::SegmentChannelShift {
            start_ms,
            end_ms,
            gain_millionths,
            muffle_retention_millionths,
        } if start_ms >= end_ms
            || !valid_fraction(*gain_millionths)
            || *muffle_retention_millionths >= MILLION =>
        {
            Err(adversarial_error(
                "segment_channel_shift_out_of_range",
                "segment channel shift has invalid time, gain, or muffle retention",
            ))
        }
        Perturbation::SpeakerPlayback {
            gain_millionths,
            muffle_retention_millionths,
            delay_ms,
            decay_millionths,
        } if !valid_fraction(*gain_millionths)
            || *muffle_retention_millionths >= MILLION
            || *delay_ms == 0
            || !valid_fraction(*decay_millionths) =>
        {
            Err(adversarial_error(
                "speaker_playback_out_of_range",
                "speaker playback has invalid gain, muffle, delay, or decay",
            ))
        }
        Perturbation::SelfOverlap {
            duration_ms,
            gain_millionths,
            ..
        } if *duration_ms == 0 || !valid_fraction(*gain_millionths) => Err(adversarial_error(
            "overlap_out_of_range",
            "self-overlap duration and gain must be positive and bounded",
        )),
        _ => Ok(()),
    }
}

fn apply_perturbation(
    audio: &mut AdversarialAudio,
    perturbation: &AcousticPerturbation,
    plan_seed: u64,
    step_index: usize,
) -> FwResult<()> {
    use AcousticPerturbation as Perturbation;
    match perturbation {
        Perturbation::Gain { gain_millionths } => {
            apply_gain(audio, *gain_millionths);
        }
        Perturbation::StationaryMuffle {
            retention_millionths,
        } => apply_muffle(audio, *retention_millionths),
        Perturbation::BandLimit { lower_hz, upper_hz } => {
            apply_band_limit(audio, *lower_hz, *upper_hz)?;
        }
        Perturbation::ResampleRoundTrip {
            intermediate_rate_hz,
        } => apply_resample_round_trip(audio, *intermediate_rate_hz)?,
        Perturbation::Quantize { bits } => apply_quantization(audio, *bits),
        Perturbation::Clip {
            threshold_millionths,
        } => apply_clipping(audio, *threshold_millionths),
        Perturbation::AddNoise {
            amplitude_millionths,
            seed_xor,
        } => apply_noise(
            audio,
            *amplitude_millionths,
            mix_seed(plan_seed ^ *seed_xor ^ step_index as u64),
        ),
        Perturbation::Reverberate {
            delay_ms,
            decay_millionths,
            repeats,
        } => apply_reverb(audio, *delay_ms, *decay_millionths, *repeats)?,
        Perturbation::Interrupt {
            start_ms,
            duration_ms,
        } => apply_interruption(audio, *start_ms, *duration_ms)?,
        Perturbation::PadSilence {
            before_ms,
            after_ms,
        } => apply_silence_padding(audio, *before_ms, *after_ms)?,
        Perturbation::SegmentChannelShift {
            start_ms,
            end_ms,
            gain_millionths,
            muffle_retention_millionths,
        } => apply_segment_channel_shift(
            audio,
            *start_ms,
            *end_ms,
            *gain_millionths,
            *muffle_retention_millionths,
        )?,
        Perturbation::SpeakerPlayback {
            gain_millionths,
            muffle_retention_millionths,
            delay_ms,
            decay_millionths,
        } => {
            apply_gain(audio, *gain_millionths);
            apply_muffle(audio, *muffle_retention_millionths);
            apply_reverb(audio, *delay_ms, *decay_millionths, 2)?;
        }
        Perturbation::StereoChannelSwap => apply_stereo_swap(audio)?,
        Perturbation::SelfOverlap {
            source_start_ms,
            destination_start_ms,
            duration_ms,
            gain_millionths,
        } => apply_self_overlap(
            audio,
            *source_start_ms,
            *destination_start_ms,
            *duration_ms,
            *gain_millionths,
        )?,
    }
    Ok(())
}

fn apply_gain(audio: &mut AdversarialAudio, gain_millionths: u32) {
    let gain = gain_millionths as f32 / MILLION as f32;
    for sample in &mut audio.samples {
        *sample = (*sample * gain).clamp(-1.0, 1.0);
    }
}

fn apply_muffle(audio: &mut AdversarialAudio, retention_millionths: u32) {
    let retention = retention_millionths as f32 / MILLION as f32;
    let fresh = 1.0 - retention;
    let mut state = [0.0f32; MAX_CHANNELS];
    for frame in audio.samples.chunks_exact_mut(audio.channels) {
        for channel in 0..audio.channels {
            let filtered = frame[channel].mul_add(fresh, state[channel] * retention);
            state[channel] = filtered;
            frame[channel] = filtered.clamp(-1.0, 1.0);
        }
    }
}

fn apply_band_limit(audio: &mut AdversarialAudio, lower_hz: u32, upper_hz: u32) -> FwResult<()> {
    if upper_hz >= audio.sample_rate_hz / 2 {
        return Err(adversarial_error(
            "band_limit_above_nyquist",
            "upper_hz must be below the input Nyquist frequency",
        ));
    }
    let original = audio.samples.clone();
    let low_passed = one_pole_low_pass(&original, audio.channels, audio.sample_rate_hz, upper_hz);
    if lower_hz == 0 {
        audio.samples = low_passed;
        return Ok(());
    }
    let lower = one_pole_low_pass(&original, audio.channels, audio.sample_rate_hz, lower_hz);
    for ((output, low), removed_low) in audio.samples.iter_mut().zip(low_passed).zip(lower) {
        *output = (low - removed_low).clamp(-1.0, 1.0);
    }
    Ok(())
}

fn one_pole_low_pass(
    input: &[f32],
    channels: usize,
    sample_rate_hz: u32,
    cutoff_hz: u32,
) -> Vec<f32> {
    if cutoff_hz == 0 {
        return vec![0.0; input.len()];
    }
    let retention = (-TAU * cutoff_hz as f32 / sample_rate_hz as f32).exp();
    let fresh = 1.0 - retention;
    let mut state = [0.0f32; MAX_CHANNELS];
    let mut output = Vec::with_capacity(input.len());
    for frame in input.chunks_exact(channels) {
        for channel in 0..channels {
            let filtered = frame[channel].mul_add(fresh, state[channel] * retention);
            state[channel] = filtered;
            output.push(filtered);
        }
    }
    output
}

fn apply_resample_round_trip(
    audio: &mut AdversarialAudio,
    intermediate_rate_hz: u32,
) -> FwResult<()> {
    let frames = audio.frame_count();
    let mut channels = Vec::with_capacity(audio.channels);
    for channel in 0..audio.channels {
        let mono: Vec<_> = audio
            .samples
            .chunks_exact(audio.channels)
            .map(|frame| frame[channel])
            .collect();
        let intermediate = resample_mono_linear(&mono, audio.sample_rate_hz, intermediate_rate_hz);
        let mut restored =
            resample_mono_linear(&intermediate, intermediate_rate_hz, audio.sample_rate_hz);
        restored.resize(frames, 0.0);
        restored.truncate(frames);
        channels.push(restored);
    }
    let sample_count = frames.checked_mul(audio.channels).ok_or_else(|| {
        adversarial_error(
            "audio_size_overflow",
            "resampled interleaved sample count overflowed",
        )
    })?;
    let mut samples = vec![0.0; sample_count];
    for frame in 0..frames {
        for channel in 0..audio.channels {
            samples[frame * audio.channels + channel] = channels[channel][frame].clamp(-1.0, 1.0);
        }
    }
    audio.samples = samples;
    Ok(())
}

fn apply_quantization(audio: &mut AdversarialAudio, bits: u8) {
    let levels = ((1u32 << (bits - 1)) - 1) as f32;
    for sample in &mut audio.samples {
        *sample = (*sample * levels).round() / levels;
    }
}

fn apply_clipping(audio: &mut AdversarialAudio, threshold_millionths: u32) {
    let threshold = threshold_millionths as f32 / MILLION as f32;
    for sample in &mut audio.samples {
        *sample = sample.clamp(-threshold, threshold);
    }
}

fn apply_noise(audio: &mut AdversarialAudio, amplitude_millionths: u32, seed: u64) {
    let amplitude = amplitude_millionths as f32 / MILLION as f32;
    let mut rng = DeterministicRng::new(seed);
    for sample in &mut audio.samples {
        *sample = (*sample + amplitude * rng.next_signed()).clamp(-1.0, 1.0);
    }
}

fn apply_reverb(
    audio: &mut AdversarialAudio,
    delay_ms: u64,
    decay_millionths: u32,
    repeats: u8,
) -> FwResult<()> {
    let delay_frames = milliseconds_to_frames_floor(delay_ms, audio.sample_rate_hz)?;
    if delay_frames == 0 {
        return Err(adversarial_error(
            "reverb_delay_too_short",
            "reverb delay resolves to zero frames",
        ));
    }
    let original = audio.samples.clone();
    let decay = decay_millionths as f32 / MILLION as f32;
    let frames = audio.frame_count();
    let mut weight = decay;
    for repeat in 1..=usize::from(repeats) {
        let offset = delay_frames.checked_mul(repeat).ok_or_else(|| {
            adversarial_error("timeline_overflow", "reverb delay offset overflowed")
        })?;
        if offset >= frames {
            break;
        }
        for frame in offset..frames {
            for channel in 0..audio.channels {
                let output_index = frame * audio.channels + channel;
                let input_index = (frame - offset) * audio.channels + channel;
                audio.samples[output_index] += original[input_index] * weight;
            }
        }
        weight *= decay;
    }
    for sample in &mut audio.samples {
        *sample = sample.clamp(-1.0, 1.0);
    }
    Ok(())
}

fn apply_interruption(
    audio: &mut AdversarialAudio,
    start_ms: u64,
    duration_ms: u64,
) -> FwResult<()> {
    let start = milliseconds_to_frames_floor(start_ms, audio.sample_rate_hz)?;
    let duration = milliseconds_to_frames_ceil(duration_ms, audio.sample_rate_hz)?;
    let end = start.checked_add(duration).ok_or_else(|| {
        adversarial_error("timeline_overflow", "interruption end time overflowed")
    })?;
    require_frame_range(audio, start, end, "interruption")?;
    audio.samples[start * audio.channels..end * audio.channels].fill(0.0);
    Ok(())
}

fn apply_silence_padding(
    audio: &mut AdversarialAudio,
    before_ms: u64,
    after_ms: u64,
) -> FwResult<()> {
    let before_frames = milliseconds_to_frames_ceil(before_ms, audio.sample_rate_hz)?;
    let after_frames = milliseconds_to_frames_ceil(after_ms, audio.sample_rate_hz)?;
    let total_frames = before_frames
        .checked_add(audio.frame_count())
        .and_then(|value| value.checked_add(after_frames))
        .ok_or_else(|| adversarial_error("audio_size_overflow", "padded frame count overflowed"))?;
    let sample_count = total_frames.checked_mul(audio.channels).ok_or_else(|| {
        adversarial_error("audio_size_overflow", "padded sample count overflowed")
    })?;
    if sample_count > MAX_INTERLEAVED_SAMPLES {
        return Err(adversarial_error(
            "audio_too_large",
            "silence padding exceeds the bounded evaluation limit",
        ));
    }
    let before_samples = before_frames * audio.channels;
    let mut padded = vec![0.0; sample_count];
    padded[before_samples..before_samples + audio.samples.len()].copy_from_slice(&audio.samples);
    audio.samples = padded;
    Ok(())
}

fn apply_segment_channel_shift(
    audio: &mut AdversarialAudio,
    start_ms: u64,
    end_ms: u64,
    gain_millionths: u32,
    retention_millionths: u32,
) -> FwResult<()> {
    let start = milliseconds_to_frames_floor(start_ms, audio.sample_rate_hz)?;
    let end = milliseconds_to_frames_ceil(end_ms, audio.sample_rate_hz)?;
    require_frame_range(audio, start, end, "segment channel shift")?;
    let gain = gain_millionths as f32 / MILLION as f32;
    let retention = retention_millionths as f32 / MILLION as f32;
    let fresh = 1.0 - retention;
    let mut state = [0.0f32; MAX_CHANNELS];
    for frame_index in start..end {
        let frame =
            &mut audio.samples[frame_index * audio.channels..(frame_index + 1) * audio.channels];
        for channel in 0..audio.channels {
            let filtered = frame[channel].mul_add(fresh, state[channel] * retention);
            state[channel] = filtered;
            frame[channel] = (filtered * gain).clamp(-1.0, 1.0);
        }
    }
    Ok(())
}

fn apply_stereo_swap(audio: &mut AdversarialAudio) -> FwResult<()> {
    if audio.channels != 2 {
        return Err(adversarial_error(
            "stereo_required",
            "stereo channel swap requires exactly two channels",
        ));
    }
    for frame in audio.samples.chunks_exact_mut(2) {
        frame.swap(0, 1);
    }
    Ok(())
}

fn apply_self_overlap(
    audio: &mut AdversarialAudio,
    source_start_ms: u64,
    destination_start_ms: u64,
    duration_ms: u64,
    gain_millionths: u32,
) -> FwResult<()> {
    let source_start = milliseconds_to_frames_floor(source_start_ms, audio.sample_rate_hz)?;
    let destination_start =
        milliseconds_to_frames_floor(destination_start_ms, audio.sample_rate_hz)?;
    let duration = milliseconds_to_frames_ceil(duration_ms, audio.sample_rate_hz)?;
    let source_end = source_start
        .checked_add(duration)
        .ok_or_else(|| adversarial_error("timeline_overflow", "overlap source end overflowed"))?;
    let destination_end = destination_start.checked_add(duration).ok_or_else(|| {
        adversarial_error("timeline_overflow", "overlap destination end overflowed")
    })?;
    require_frame_range(audio, source_start, source_end, "overlap source")?;
    require_frame_range(
        audio,
        destination_start,
        destination_end,
        "overlap destination",
    )?;
    let source = audio.samples[source_start * audio.channels..source_end * audio.channels].to_vec();
    let gain = gain_millionths as f32 / MILLION as f32;
    let destination =
        &mut audio.samples[destination_start * audio.channels..destination_end * audio.channels];
    for (output, input) in destination.iter_mut().zip(source) {
        *output = (*output + input * gain).clamp(-1.0, 1.0);
    }
    Ok(())
}

fn require_frame_range(
    audio: &AdversarialAudio,
    start: usize,
    end: usize,
    field: &str,
) -> FwResult<()> {
    if start >= end || end > audio.frame_count() {
        Err(adversarial_error(
            "timeline_out_of_range",
            &format!("{field} range falls outside the audio"),
        ))
    } else {
        Ok(())
    }
}

fn validate_stage_fingerprints(
    fingerprints: &[StageFingerprint],
    field: &str,
) -> FwResult<BTreeMap<AdversarialPipelineStage, String>> {
    let mut output = BTreeMap::new();
    for fingerprint in fingerprints {
        validate_sha256(&fingerprint.sha256, field)?;
        if output
            .insert(fingerprint.stage, fingerprint.sha256.clone())
            .is_some()
        {
            return Err(adversarial_error(
                "duplicate_stage_fingerprint",
                &format!("{field} contains a duplicate stage"),
            ));
        }
    }
    Ok(output)
}

fn validate_regression_classification(classification: &RegressionClassification) -> FwResult<()> {
    if classification.code.is_empty()
        || classification.code.len() > 64
        || !classification
            .code
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(adversarial_error(
            "invalid_failure_code",
            "failure code must contain 1..=64 uppercase ASCII letters, digits, or hyphens",
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> FwResult<()> {
    if value.len() != HASH_HEX_LEN
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(adversarial_error(
            "invalid_sha256",
            &format!("{field} must be 64 lowercase hexadecimal characters"),
        ));
    }
    Ok(())
}

fn milliseconds_to_frames_floor(milliseconds: u64, sample_rate_hz: u32) -> FwResult<usize> {
    let frames = milliseconds
        .checked_mul(u64::from(sample_rate_hz))
        .ok_or_else(|| {
            adversarial_error("timeline_overflow", "time-to-frame conversion overflowed")
        })?
        / 1_000;
    usize::try_from(frames)
        .map_err(|_| adversarial_error("timeline_overflow", "frame index does not fit usize"))
}

fn milliseconds_to_frames_ceil(milliseconds: u64, sample_rate_hz: u32) -> FwResult<usize> {
    let numerator = milliseconds
        .checked_mul(u64::from(sample_rate_hz))
        .and_then(|value| value.checked_add(999))
        .ok_or_else(|| {
            adversarial_error("timeline_overflow", "time-to-frame conversion overflowed")
        })?;
    usize::try_from(numerator / 1_000)
        .map_err(|_| adversarial_error("timeline_overflow", "frame index does not fit usize"))
}

fn canonical_sha256<T: Serialize>(value: &T) -> FwResult<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

fn checkpoint_cancelled(is_cancelled: &mut impl FnMut() -> bool) -> FwResult<()> {
    if is_cancelled() {
        Err(FwError::Cancelled(
            "adversarial_corpus.cancelled: adversarial corpus operation cancelled".to_owned(),
        ))
    } else {
        Ok(())
    }
}

fn adversarial_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("adversarial_corpus.{code}: {message}"))
}

fn mix_seed(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: mix_seed(seed).max(1),
        }
    }

    fn next_signed(&mut self) -> f32 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        let output = value.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let unit = (output >> 40) as f32 / (1u64 << 24) as f32;
        unit.mul_add(2.0, -1.0)
    }
}

fn scaled_error(value: f64) -> u64 {
    let scaled = (value * f64::from(MILLION)).round();
    if scaled >= u64::MAX as f64 {
        u64::MAX
    } else {
        scaled as u64
    }
}

fn signed_difference(left: usize, right: usize) -> i64 {
    if left >= right {
        i64::try_from(left - right).unwrap_or(i64::MAX)
    } else {
        -i64::try_from(right - left).unwrap_or(i64::MAX)
    }
}

fn synthetic_speaker(
    fundamental_millihz: u32,
    position_millionths: i32,
) -> SyntheticSpeakerProfile {
    SyntheticSpeakerProfile {
        fundamental_millihz,
        harmonic_count: 5,
        harmonic_decay_millionths: 540_000,
        amplitude_millionths: 620_000,
        muffle_retention_millionths: 250_000,
        stereo_position_millionths: position_millionths,
    }
}

fn synthetic_turn(speaker_index: usize, start_ms: u64, end_ms: u64) -> SyntheticTurn {
    SyntheticTurn {
        speaker_index,
        start_ms,
        end_ms,
        gain_millionths: MILLION,
        pitch_shift_millihz: 0,
        playback: false,
    }
}

fn base_synthetic_plan(channels: usize, seed: u64) -> SyntheticCallPlan {
    SyntheticCallPlan::new(
        16_000,
        channels,
        seed,
        vec![
            synthetic_speaker(118_000, -520_000),
            synthetic_speaker(212_000, 580_000),
        ],
        vec![
            synthetic_turn(0, 0, 1_600),
            synthetic_turn(1, 1_800, 3_300),
            synthetic_turn(0, 3_500, 5_100),
            synthetic_turn(1, 5_300, 6_700),
        ],
    )
}

fn rapid_turns() -> Vec<SyntheticTurn> {
    (0..12)
        .map(|index| {
            let start_ms = index * 310;
            synthetic_turn(index as usize % 2, start_ms, start_ms + 240)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn materialized_base(channels: usize) -> GeneratedSyntheticCall {
        generate_synthetic_call_uncancellable(&base_synthetic_plan(channels, 7))
            .expect("base synthetic call")
    }

    #[test]
    fn synthetic_generation_is_deterministic_and_content_bound() {
        let plan = base_synthetic_plan(2, 41);
        let left = generate_synthetic_call_uncancellable(&plan).expect("left");
        let right = generate_synthetic_call_uncancellable(&plan).expect("right");
        assert_eq!(left.audio, right.audio);
        assert_eq!(left.evidence, right.evidence);
        assert_eq!(left.evidence.audio_sha256, left.audio.sha256());
        assert_eq!(left.evidence.plan_sha256, plan.sha256().expect("plan hash"));
        assert!(left.audio.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn synthetic_generation_supports_overlap_and_voice_state_shift() {
        let mut plan = base_synthetic_plan(1, 42);
        plan.turns = vec![
            synthetic_turn(0, 0, 1_500),
            synthetic_turn(1, 800, 2_200),
            SyntheticTurn {
                pitch_shift_millihz: 35_000,
                ..synthetic_turn(0, 2_400, 3_500)
            },
        ];
        let generated = generate_synthetic_call_uncancellable(&plan).expect("generated");
        assert_eq!(generated.reference_turns.len(), 3);
        assert_eq!(generated.reference_turns[1].start_ms, 800);
        assert!(generated.audio.samples.iter().any(|sample| *sample != 0.0));
    }

    #[test]
    fn malformed_synthetic_parameters_fail_safely() {
        let mut plan = base_synthetic_plan(1, 1);
        plan.sample_rate_hz = 0;
        let error = generate_synthetic_call_uncancellable(&plan).expect_err("invalid rate");
        assert!(error.to_string().contains("sample_rate_out_of_range"));

        let mut plan = base_synthetic_plan(1, 1);
        plan.turns[0].end_ms = plan.turns[0].start_ms;
        let error = generate_synthetic_call_uncancellable(&plan).expect_err("invalid turn");
        assert!(error.to_string().contains("invalid_synthetic_turn"));
    }

    #[test]
    fn every_transform_is_deterministic_and_finite() {
        let source = materialized_base(2).audio;
        let steps = vec![
            AcousticPerturbation::Gain {
                gain_millionths: 800_000,
            },
            AcousticPerturbation::StationaryMuffle {
                retention_millionths: 800_000,
            },
            AcousticPerturbation::BandLimit {
                lower_hz: 200,
                upper_hz: 3_400,
            },
            AcousticPerturbation::ResampleRoundTrip {
                intermediate_rate_hz: 8_000,
            },
            AcousticPerturbation::Quantize { bits: 10 },
            AcousticPerturbation::Clip {
                threshold_millionths: 300_000,
            },
            AcousticPerturbation::AddNoise {
                amplitude_millionths: 40_000,
                seed_xor: 99,
            },
            AcousticPerturbation::Reverberate {
                delay_ms: 50,
                decay_millionths: 300_000,
                repeats: 3,
            },
            AcousticPerturbation::Interrupt {
                start_ms: 500,
                duration_ms: 100,
            },
            AcousticPerturbation::SegmentChannelShift {
                start_ms: 900,
                end_ms: 1_200,
                gain_millionths: 600_000,
                muffle_retention_millionths: 750_000,
            },
            AcousticPerturbation::SpeakerPlayback {
                gain_millionths: 700_000,
                muffle_retention_millionths: 850_000,
                delay_ms: 45,
                decay_millionths: 250_000,
            },
            AcousticPerturbation::StereoChannelSwap,
            AcousticPerturbation::SelfOverlap {
                source_start_ms: 100,
                destination_start_ms: 2_000,
                duration_ms: 200,
                gain_millionths: 400_000,
            },
            AcousticPerturbation::PadSilence {
                before_ms: 120,
                after_ms: 80,
            },
        ];
        let plan = TransformPlan::new_synthetic(source.sha256(), 123, steps);
        let left = apply_transform_plan_uncancellable(&source, &plan).expect("left");
        let right = apply_transform_plan_uncancellable(&source, &plan).expect("right");
        assert_eq!(left.audio, right.audio);
        assert_eq!(left.evidence, right.evidence);
        assert_eq!(left.evidence.graph.len(), plan.steps.len());
        assert!(left.audio.samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn transform_hash_mismatch_and_invalid_ranges_fail_closed() {
        let source = materialized_base(1).audio;
        let plan = TransformPlan::new_synthetic(
            "0".repeat(64),
            1,
            vec![AcousticPerturbation::Gain {
                gain_millionths: 500_000,
            }],
        );
        let error = apply_transform_plan_uncancellable(&source, &plan).expect_err("hash mismatch");
        assert!(error.to_string().contains("source_hash_mismatch"));

        let plan = TransformPlan::new_synthetic(
            source.sha256(),
            1,
            vec![AcousticPerturbation::BandLimit {
                lower_hz: 4_000,
                upper_hz: 2_000,
            }],
        );
        let error = apply_transform_plan_uncancellable(&source, &plan).expect_err("bad band");
        assert!(error.to_string().contains("invalid_transform_step"));

        let plan = TransformPlan::new_public_licensed(
            source.sha256(),
            "not-a-hash".to_owned(),
            1,
            Vec::new(),
        );
        let error =
            apply_transform_plan_uncancellable(&source, &plan).expect_err("bad license proof");
        assert!(error.to_string().contains("invalid_sha256"));
    }

    #[test]
    fn transform_directional_effects_match_their_declared_contracts() {
        let source = materialized_base(2).audio;

        let gain_plan = TransformPlan::new_synthetic(
            source.sha256(),
            1,
            vec![AcousticPerturbation::Gain {
                gain_millionths: 500_000,
            }],
        );
        let gained = apply_transform_plan_uncancellable(&source, &gain_plan)
            .expect("gain")
            .audio;
        for (before, after) in source.samples.iter().zip(&gained.samples) {
            assert!((*after - *before * 0.5).abs() <= f32::EPSILON);
        }

        let clip_plan = TransformPlan::new_synthetic(
            source.sha256(),
            2,
            vec![AcousticPerturbation::Clip {
                threshold_millionths: 100_000,
            }],
        );
        let clipped = apply_transform_plan_uncancellable(&source, &clip_plan)
            .expect("clip")
            .audio;
        assert!(
            clipped
                .samples
                .iter()
                .all(|sample| sample.abs() <= 0.100_001)
        );

        let interrupt_plan = TransformPlan::new_synthetic(
            source.sha256(),
            3,
            vec![AcousticPerturbation::Interrupt {
                start_ms: 400,
                duration_ms: 100,
            }],
        );
        let interrupted = apply_transform_plan_uncancellable(&source, &interrupt_plan)
            .expect("interrupt")
            .audio;
        let start = 400 * source.sample_rate_hz as usize / 1_000;
        let end = 500 * source.sample_rate_hz as usize / 1_000;
        assert!(
            interrupted.samples[start * 2..end * 2]
                .iter()
                .all(|sample| *sample == 0.0)
        );

        let resample_plan = TransformPlan::new_synthetic(
            source.sha256(),
            4,
            vec![AcousticPerturbation::ResampleRoundTrip {
                intermediate_rate_hz: 8_000,
            }],
        );
        let resampled = apply_transform_plan_uncancellable(&source, &resample_plan)
            .expect("resample")
            .audio;
        assert_eq!(resampled.frame_count(), source.frame_count());
        assert_ne!(resampled.sha256(), source.sha256());

        let swap_plan = TransformPlan::new_synthetic(
            source.sha256(),
            5,
            vec![
                AcousticPerturbation::StereoChannelSwap,
                AcousticPerturbation::StereoChannelSwap,
            ],
        );
        let swapped_twice = apply_transform_plan_uncancellable(&source, &swap_plan)
            .expect("swap twice")
            .audio;
        assert_eq!(swapped_twice, source);
    }

    #[test]
    fn non_finite_audio_fails_before_transforming() {
        let source = AdversarialAudio {
            samples: vec![0.0, f32::NAN],
            sample_rate_hz: 16_000,
            channels: 1,
        };
        let plan = TransformPlan::new_synthetic(
            source.sha256(),
            1,
            vec![AcousticPerturbation::Gain {
                gain_millionths: MILLION,
            }],
        );
        let error = apply_transform_plan_uncancellable(&source, &plan).expect_err("non-finite");
        assert!(error.to_string().contains("non_finite_audio"));
    }

    #[test]
    fn silence_padding_has_exact_reference_shift() {
        let source = materialized_base(1);
        let plan = TransformPlan::new_synthetic(
            source.audio.sha256(),
            1,
            vec![
                AcousticPerturbation::PadSilence {
                    before_ms: 300,
                    after_ms: 100,
                },
                AcousticPerturbation::PadSilence {
                    before_ms: 200,
                    after_ms: 0,
                },
            ],
        );
        let shifted = transform_reference_turns(&source.reference_turns, &plan).expect("shift");
        assert_eq!(
            shifted[0].start_ms,
            source.reference_turns[0].start_ms + 500
        );
        assert_eq!(shifted[0].end_ms, source.reference_turns[0].end_ms + 500);

        let malformed = [SyntheticReferenceTurn {
            speaker_index: 0,
            start_ms: 10,
            end_ms: 10,
        }];
        let error = transform_reference_turns(&malformed, &plan).expect_err("malformed");
        assert!(error.to_string().contains("invalid_reference_turn"));
    }

    #[test]
    fn stage_divergence_finds_first_changed_or_missing_stage() {
        let hash_a = "a".repeat(64);
        let hash_b = "b".repeat(64);
        let baseline = vec![
            StageFingerprint {
                stage: AdversarialPipelineStage::Input,
                sha256: hash_a.clone(),
            },
            StageFingerprint {
                stage: AdversarialPipelineStage::FeatureExtraction,
                sha256: hash_a.clone(),
            },
        ];
        let candidate = vec![
            StageFingerprint {
                stage: AdversarialPipelineStage::Input,
                sha256: hash_a,
            },
            StageFingerprint {
                stage: AdversarialPipelineStage::FeatureExtraction,
                sha256: hash_b,
            },
        ];
        let divergence = first_stage_divergence(&baseline, &candidate)
            .expect("comparison")
            .expect("divergence");
        assert_eq!(
            divergence.stage,
            AdversarialPipelineStage::FeatureExtraction
        );
    }

    #[test]
    fn duplicate_stage_fingerprints_fail_safely() {
        let fingerprints = vec![
            StageFingerprint {
                stage: AdversarialPipelineStage::Input,
                sha256: "a".repeat(64),
            },
            StageFingerprint {
                stage: AdversarialPipelineStage::Input,
                sha256: "b".repeat(64),
            },
        ];
        let error = first_stage_divergence(&fingerprints, &[]).expect_err("duplicate");
        assert!(error.to_string().contains("duplicate_stage_fingerprint"));
    }

    #[test]
    fn minimizer_preserves_exact_failure_classification() {
        let source_hash = "a".repeat(64);
        let essential = AcousticPerturbation::Clip {
            threshold_millionths: 200_000,
        };
        let plan = TransformPlan::new_synthetic(
            source_hash,
            5,
            vec![
                AcousticPerturbation::Gain {
                    gain_millionths: 900_000,
                },
                essential.clone(),
                AcousticPerturbation::AddNoise {
                    amplitude_millionths: 10_000,
                    seed_xor: 7,
                },
            ],
        );
        let expected = RegressionClassification {
            code: "FW-ADVERSARIAL-CLUSTER-REGRESSION".to_owned(),
            first_divergent_stage: AdversarialPipelineStage::Clustering,
        };
        let minimized = minimize_failing_plan(&plan, &expected, |candidate| {
            Ok(candidate
                .steps
                .contains(&essential)
                .then(|| expected.clone()))
        })
        .expect("minimized");
        assert_eq!(minimized.minimized_plan.steps, vec![essential]);
        assert_eq!(minimized.retained_original_step_indices, vec![1]);
        assert!(minimized.evaluation_count <= MAX_MINIMIZER_EVALUATIONS);
    }

    #[test]
    fn minimizer_rejects_non_reproducing_original_plan() {
        let plan = TransformPlan::new_synthetic("a".repeat(64), 1, Vec::new());
        let expected = RegressionClassification {
            code: "FW-ADVERSARIAL-SCORING-REGRESSION".to_owned(),
            first_divergent_stage: AdversarialPipelineStage::Scoring,
        };
        let error =
            minimize_failing_plan(&plan, &expected, |_| Ok(None)).expect_err("not reproducing");
        assert!(
            error
                .to_string()
                .contains("failure_classification_mismatch")
        );
    }

    #[test]
    fn consistency_measurement_detects_exact_and_approximate_results() {
        let expected = materialized_base(1).audio;
        let exact = measure_audio_consistency(&expected, &expected).expect("exact");
        assert!(exact.exact_sample_bits);
        assert_eq!(exact.max_abs_error_millionths, 0);

        let mut observed = expected.clone();
        observed.samples[0] += 0.01;
        let approximate = measure_audio_consistency(&expected, &observed).expect("approximate");
        assert!(!approximate.exact_sample_bits);
        assert!(approximate.max_abs_error_millionths >= 9_999);
    }

    #[test]
    fn all_known_challenge_families_have_unique_reproducible_safe_seeds() {
        let seeds = known_acoustic_challenge_seeds().expect("seeds");
        assert_eq!(seeds.len(), 17);
        let mut families = std::collections::BTreeSet::new();
        let mut hashes = std::collections::BTreeSet::new();
        for seed in &seeds {
            assert!(families.insert(seed.family));
            assert!(hashes.insert(seed.sha256().expect("seed hash")));
            let left = seed.materialize().expect("left");
            let right = seed.materialize().expect("right");
            assert_eq!(left.audio, right.audio);
            assert_eq!(left.transform_evidence, right.transform_evidence);
            assert!(!left.reference_turns.is_empty());
        }
    }

    #[test]
    fn retained_evidence_and_seeds_are_media_and_identity_free() {
        let seed = known_acoustic_challenge_seeds()
            .expect("seeds")
            .into_iter()
            .next()
            .expect("first");
        let materialized = seed.materialize().expect("materialized");
        let serialized = serde_json::to_string(&(
            seed,
            materialized.source_evidence,
            materialized.transform_evidence,
        ))
        .expect("serialize");
        for forbidden in [
            "path",
            "filename",
            "transcript",
            "embedding",
            ".wav",
            ".m4a",
            "private_person_name",
        ] {
            assert!(!serialized.contains(forbidden), "{forbidden}");
        }
        assert!(!serialized.contains("samples"));
    }

    #[test]
    fn cancellation_is_observed_before_materialization_and_each_step() {
        let plan = base_synthetic_plan(1, 1);
        let error = generate_synthetic_call(&plan, &mut || true).expect_err("generation cancelled");
        assert!(matches!(error, FwError::Cancelled(_)));

        let source = materialized_base(1).audio;
        let plan = TransformPlan::new_synthetic(
            source.sha256(),
            1,
            vec![AcousticPerturbation::Gain {
                gain_millionths: 500_000,
            }],
        );
        let error =
            apply_transform_plan(&source, &plan, &mut || true).expect_err("transform cancelled");
        assert!(matches!(error, FwError::Cancelled(_)));
    }
}
