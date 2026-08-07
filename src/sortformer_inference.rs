//! Evaluation-only native inference building blocks for the pinned four-speaker
//! Streaming Sortformer.
//!
//! This module is intentionally not a diarization route. It begins the f32
//! equivalence ladder with the exact stored frontend buffers and preprocessing
//! graph. Promotion, automatic routing, streaming state, and speaker decisions
//! remain separate gates.
//!
//! The initial frontend emulates offline whole-file evaluation and materializes
//! the complete `[128, T]` result. Incremental preemphasis/STFT continuity is a
//! separate seam; live streaming is not certified by this module.

use std::fmt;

use sha2::{Digest, Sha256};

use crate::error::{FwError, FwResult};
use crate::native_engine::Mat;
use crate::native_engine::mel::{REAL_FFT_512_BINS, REAL_FFT_512_LEN, real_fft_512};
use crate::native_engine::nn;
use crate::native_engine::weights::SafetensorsFile;
use crate::sortformer_conformance::{
    VerifiedSortformerActivationPack, VerifiedSortformerPackage,
    VerifiedSortformerPublicActivationPack,
};

pub const SORTFORMER_SAMPLE_RATE_HZ: usize = 16_000;
pub const SORTFORMER_CHANNELS: usize = 1;
pub const SORTFORMER_MAX_AUDIO_SECONDS: usize = 2 * 60 * 60;
pub const SORTFORMER_MAX_AUDIO_SAMPLES: usize =
    SORTFORMER_SAMPLE_RATE_HZ * SORTFORMER_MAX_AUDIO_SECONDS;
pub const SORTFORMER_WINDOW_SAMPLES: usize = 400;
pub const SORTFORMER_HOP_SAMPLES: usize = 160;
pub const SORTFORMER_MEL_BINS: usize = 128;
pub const SORTFORMER_SUBSAMPLING_CHANNELS: usize = 256;
pub const SORTFORMER_ENCODER_WIDTH: usize = 512;
pub const SORTFORMER_FASTCONFORMER_LAYERS: usize = 17;
pub const SORTFORMER_TRANSFORMER_WIDTH: usize = 192;
pub const SORTFORMER_TRANSFORMER_LAYERS: usize = 18;
pub const SORTFORMER_TRANSFORMER_HEADS: usize = 8;
pub const SORTFORMER_TRANSFORMER_HEAD_WIDTH: usize = 24;
pub const SORTFORMER_TRANSFORMER_INNER_WIDTH: usize = 768;
pub const SORTFORMER_SPEAKER_LANES: usize = 4;
pub const SORTFORMER_SUBSAMPLING_FACTOR: usize = 8;
pub const SORTFORMER_ENCODER_XSCALE: f32 = 22.627_417; // f32(sqrt(512))

pub const SORTFORMER_WINDOW_TENSOR_NAME: &str = "preprocessor.featurizer.window";
pub const SORTFORMER_WINDOW_TENSOR_SHA256: &str =
    "7d6b2ab4944b0b65650e1bba1132821fd1d2ed000df84dbd893316788d0ef062";
pub const SORTFORMER_MEL_TENSOR_NAME: &str = "preprocessor.featurizer.fb";
pub const SORTFORMER_MEL_TENSOR_SHA256: &str =
    "82663f1145f6965d8b27a85f32a44fa4f3bffef9bd0d6c2d1902b334a012367b";

const PREEMPHASIS: f32 = 0.97;
const LOG_GUARD: f32 = f32::from_bits(0x3380_0000); // exactly 2^-24
const WINDOW_FFT_LEFT_PAD: usize = (REAL_FFT_512_LEN - SORTFORMER_WINDOW_SAMPLES) / 2;
const WINDOW_CENTER: usize = SORTFORMER_WINDOW_SAMPLES / 2;
const ZERO_FILL_CHUNK_VALUES: usize = 1024 * 1024;
const WINDOW_SHAPE: &[usize] = &[SORTFORMER_WINDOW_SAMPLES];
const MEL_SHAPE: &[usize] = &[1, SORTFORMER_MEL_BINS, REAL_FFT_512_BINS];

/// Predeclared implementation envelope for the diagnostic synthetic L1 probe.
/// These are cross-kernel budgets, not the reference oracle's measured floor;
/// the latter is independently required to remain byte-exact at zero.
pub const SORTFORMER_FRONTEND_MAX_ABS_DIFF: f64 = 0.000_244_140_625; // 2^-12
pub const SORTFORMER_FRONTEND_MAX_MEAN_ABS_DIFF: f64 = 0.000_007_629_394_531_25; // 2^-17
pub const SORTFORMER_FRONTEND_MAX_RELATIVE_L2: f64 = 0.000_001_907_348_632_812_5; // 2^-19

/// Predeclared Rust/PyTorch cross-kernel envelope for each public L2 seam.
/// The independently measured source-replay floor remains separate and may be
/// tighter. These constants must not be changed in response to a failed Rust
/// comparison; a failure is evidence against the implementation.
pub const SORTFORMER_L2_MAX_ABS_DIFF: f64 = 0.000_976_562_5; // 2^-10
pub const SORTFORMER_L2_MAX_RELATIVE_L2: f64 = 0.000_015_258_789_062_5; // 2^-16
pub const SORTFORMER_L3_INPUT_MAX_ABS_DIFF: f64 = 0.031_25; // 2^-5
pub const SORTFORMER_L3_INPUT_MAX_RELATIVE_L2: f64 = 0.000_015_258_789_062_5; // 2^-16
pub const SORTFORMER_L3_FFN_MAX_ABS_DIFF: f64 = 0.003_906_25; // 2^-8
pub const SORTFORMER_L3_FFN_MAX_RELATIVE_L2: f64 = 0.000_061_035_156_25; // 2^-14
pub const SORTFORMER_L3_QKV_MAX_ABS_DIFF: f64 = 0.007_812_5; // 2^-7
pub const SORTFORMER_L3_QKV_MAX_RELATIVE_L2: f64 = 0.000_122_070_312_5; // 2^-13
pub const SORTFORMER_L3_ATTENTION_MAX_ABS_DIFF: f64 = 0.015_625; // 2^-6
pub const SORTFORMER_L3_ATTENTION_MAX_RELATIVE_L2: f64 = 0.000_488_281_25; // 2^-11
pub const SORTFORMER_L3_CONV_MAX_ABS_DIFF: f64 = 0.031_25; // 2^-5
pub const SORTFORMER_L3_CONV_MAX_RELATIVE_L2: f64 = 0.000_244_140_625; // 2^-12
pub const SORTFORMER_L3_BLOCK_MAX_ABS_DIFF: f64 = 0.062_5; // 2^-4
pub const SORTFORMER_L3_BLOCK_MAX_RELATIVE_L2: f64 = 0.000_976_562_5; // 2^-10
pub const SORTFORMER_L4_MAX_ABS_DIFF: f64 = 0.062_5; // 2^-4
pub const SORTFORMER_L4_MAX_RELATIVE_L2: f64 = 0.000_976_562_5; // 2^-10
pub const SORTFORMER_L5_MAX_ABS_DIFF: f64 = 0.062_5; // 2^-4
pub const SORTFORMER_L5_MAX_RELATIVE_L2: f64 = 0.000_976_562_5; // 2^-10
pub const SORTFORMER_L6_MAX_ABS_DIFF: f64 = 0.062_5; // 2^-4
pub const SORTFORMER_L6_MAX_RELATIVE_L2: f64 = 0.000_976_562_5; // 2^-10

const SORTFORMER_SPEAKER_CACHE_FRAMES: usize = 188;
const SORTFORMER_FIFO_FRAMES: usize = 0;
const SORTFORMER_CACHE_UPDATE_FRAMES: usize = 188;
const SORTFORMER_SILENCE_FRAMES_PER_SPEAKER: usize = 3;
const SORTFORMER_SILENCE_THRESHOLD: f32 = 0.2;
const SORTFORMER_PREDICTION_SCORE_THRESHOLD: f32 = 0.25;
const SORTFORMER_STRONG_BOOST_RATE: f32 = 0.75;
const SORTFORMER_WEAK_BOOST_RATE: f32 = 1.5;
const SORTFORMER_MIN_POSITIVE_SCORE_RATE: f32 = 0.5;
const SORTFORMER_LATEST_SCORE_BOOST: f32 = 0.05;

#[derive(Clone, Copy)]
struct TensorContract<'a> {
    name: &'a str,
    shape: &'a [usize],
    sha256: &'a str,
}

const PRODUCTION_WINDOW_CONTRACT: TensorContract<'static> = TensorContract {
    name: SORTFORMER_WINDOW_TENSOR_NAME,
    shape: WINDOW_SHAPE,
    sha256: SORTFORMER_WINDOW_TENSOR_SHA256,
};
const PRODUCTION_MEL_CONTRACT: TensorContract<'static> = TensorContract {
    name: SORTFORMER_MEL_TENSOR_NAME,
    shape: MEL_SHAPE,
    sha256: SORTFORMER_MEL_TENSOR_SHA256,
};

/// Exact pinned analysis buffers needed by the first Sortformer oracle seam.
///
/// Construction authenticates the raw little-endian f32 payloads by name,
/// shape, dtype, and SHA-256. It does not replace whole-package conversion
/// receipt verification.
pub struct SortformerFrontend {
    window: [f32; SORTFORMER_WINDOW_SAMPLES],
    mel_filterbank: Vec<f32>,
    mel_nonzero_ranges: [(usize, usize); SORTFORMER_MEL_BINS],
}

impl fmt::Debug for SortformerFrontend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SortformerFrontend")
            .field("window", &"<authenticated>")
            .field("mel_filterbank", &"<authenticated>")
            .field("mel_nonzero_ranges", &"<derived>")
            .finish()
    }
}

/// Canonical mel-major Sortformer frontend output.
///
/// `data[mel * valid_frames + frame]` addresses one value. Debug formatting is
/// redacted because these features are derived from caller audio.
#[derive(PartialEq)]
pub struct SortformerFrontendOutput {
    pub mel_bins: usize,
    pub valid_frames: usize,
    pub data: Vec<f32>,
}

/// Numeric result for one authenticated non-human fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct SortformerFrontendParityMetrics {
    pub fixture: String,
    pub compared_values: usize,
    pub mismatch_count: usize,
    pub byte_exact: bool,
    pub max_abs_diff: f64,
    pub mean_abs_diff: f64,
    pub relative_l2: f64,
}

/// Diagnostic partial-L1 result. This report is not routing or promotion
/// authority and contains only aggregate drift over frozen synthetic inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct SortformerFrontendParityReport {
    pub oracle_floor_byte_exact: bool,
    pub fixtures: Vec<SortformerFrontendParityMetrics>,
}

/// Audio boundary for the evaluation frontend.
///
/// Carrying rate and channel count with the samples makes the two preprocessing
/// preconditions machine-enforceable instead of relying on caller convention.
#[derive(Clone, Copy)]
pub struct SortformerPcm<'a> {
    pub samples: &'a [f32],
    pub sample_rate_hz: usize,
    pub channels: usize,
}

impl<'a> SortformerPcm<'a> {
    #[must_use]
    pub const fn new(samples: &'a [f32], sample_rate_hz: usize, channels: usize) -> Self {
        Self {
            samples,
            sample_rate_hz,
            channels,
        }
    }

    #[must_use]
    pub const fn mono_16khz(samples: &'a [f32]) -> Self {
        Self::new(samples, SORTFORMER_SAMPLE_RATE_HZ, SORTFORMER_CHANNELS)
    }
}

impl fmt::Debug for SortformerPcm<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SortformerPcm")
            .field("samples", &"<redacted>")
            .field("sample_rate_hz", &self.sample_rate_hz)
            .field("channels", &self.channels)
            .finish()
    }
}

impl fmt::Debug for SortformerFrontendOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SortformerFrontendOutput")
            .field("mel_bins", &self.mel_bins)
            .field("valid_frames", &self.valid_frames)
            .field("data", &"<redacted>")
            .finish()
    }
}

impl SortformerFrontend {
    /// Load from a package whose full receipt and payload were authenticated,
    /// then independently re-authenticate the two frontend buffers at this
    /// semantic boundary.
    pub fn from_verified_package(package: &VerifiedSortformerPackage) -> FwResult<Self> {
        Self::from_verified_package_with_checkpoint(package, &|| Ok(()))
    }

    /// Load from a fully authenticated package with cooperative cancellation.
    pub fn from_verified_package_with_checkpoint(
        package: &VerifiedSortformerPackage,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        Self::from_package_with_checkpoint(package.safetensors(), checkpoint)
    }

    /// Load and authenticate the two pinned frontend buffers.
    ///
    /// This accepts an already parsed package so a higher-level conversion
    /// verifier can authenticate the whole package once and pass the same owned
    /// bytes through without reopening a path.
    #[cfg(test)]
    fn from_package(package: &SafetensorsFile) -> FwResult<Self> {
        Self::from_package_with_checkpoint(package, &|| Ok(()))
    }

    /// Load the pinned frontend buffers with cooperative cancellation.
    pub(crate) fn from_package_with_checkpoint(
        package: &SafetensorsFile,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        Self::from_package_contracts(
            package,
            PRODUCTION_WINDOW_CONTRACT,
            PRODUCTION_MEL_CONTRACT,
            checkpoint,
        )
    }

    fn from_package_contracts(
        package: &SafetensorsFile,
        window_contract: TensorContract<'_>,
        mel_contract: TensorContract<'_>,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let window_values = load_exact_f32_tensor(package, window_contract, checkpoint)?;
        let mel_filterbank = load_exact_f32_tensor(package, mel_contract, checkpoint)?;
        let window = window_values.try_into().map_err(|_| {
            frontend_error(
                "window_count",
                "authenticated window did not contain exactly 400 values",
            )
        })?;
        let mel_nonzero_ranges = mel_nonzero_ranges(&mel_filterbank)?;
        frontend_checkpoint(checkpoint)?;
        Ok(Self {
            window,
            mel_filterbank,
            mel_nonzero_ranges,
        })
    }

    /// Compute the pinned evaluation frontend for finite mono 16 kHz PCM.
    ///
    /// This initial whole-file seam accepts at most two hours and materializes
    /// the complete mel tensor. It is not a live-streaming API.
    pub fn compute(&self, pcm: SortformerPcm<'_>) -> FwResult<SortformerFrontendOutput> {
        self.compute_with_checkpoint(pcm, &|| Ok(()))
    }

    /// Compute the pinned frontend with cooperative cancellation.
    ///
    /// The physical centered STFT produces `floor(samples / 160) + 1` frames.
    /// NeMo's declared length marks the last frame invalid, so the returned
    /// canonical tensor is cropped to `floor(samples / 160)` frames.
    pub fn compute_with_checkpoint(
        &self,
        pcm: SortformerPcm<'_>,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<SortformerFrontendOutput> {
        validate_pcm(pcm, checkpoint)?;
        let samples = pcm.samples;
        let (valid_frames, physical_frames) = frame_geometry(samples.len())?;
        let value_count = valid_frames
            .checked_mul(SORTFORMER_MEL_BINS)
            .ok_or_else(|| frontend_error("output_size", "frontend output size overflows"))?;

        let mut data = zeroed_output_buffer(value_count, checkpoint)?;
        let mut fft_input = [0.0_f32; REAL_FFT_512_LEN];
        let mut spectrum = [(0.0_f32, 0.0_f32); REAL_FFT_512_BINS];
        let mut power = [0.0_f32; REAL_FFT_512_BINS];

        for frame in 0..physical_frames {
            frontend_checkpoint(checkpoint)?;
            fft_input.fill(0.0);
            let frame_hop = frame
                .checked_mul(SORTFORMER_HOP_SAMPLES)
                .ok_or_else(|| frontend_error("frame_offset", "STFT frame offset overflows"))?;
            for (window_index, (&sample_window, fft_slot)) in self
                .window
                .iter()
                .zip(
                    fft_input[WINDOW_FFT_LEFT_PAD..WINDOW_FFT_LEFT_PAD + SORTFORMER_WINDOW_SAMPLES]
                        .iter_mut(),
                )
                .enumerate()
            {
                let centered = frame_hop.checked_add(window_index).ok_or_else(|| {
                    frontend_error("frame_offset", "STFT sample offset overflows")
                })?;
                let sample = centered
                    .checked_sub(WINDOW_CENTER)
                    .and_then(|source| preemphasized_sample(samples, source))
                    .unwrap_or(0.0);
                *fft_slot = sample * sample_window;
            }

            real_fft_512(&fft_input, &mut spectrum)?;
            for (destination, &(real, imaginary)) in power.iter_mut().zip(spectrum.iter()) {
                // Preserve the pinned NeMo graph literally: abs(complex).pow(2),
                // not a substituted direct re^2 + im^2 expression.
                let magnitude = (real.powi(2) + imaginary.powi(2)).sqrt();
                *destination = magnitude.powi(2);
            }
            if power.iter().any(|value| !value.is_finite()) {
                return Err(frontend_error(
                    "output_nonfinite",
                    "frontend produced a non-finite power spectrum",
                ));
            }

            for mel in 0..SORTFORMER_MEL_BINS {
                let filter_start = mel * REAL_FFT_512_BINS;
                let filter = &self.mel_filterbank[filter_start..filter_start + REAL_FFT_512_BINS];
                let energy =
                    sparse_mel_energy(power.as_slice(), filter, self.mel_nonzero_ranges[mel]);
                let log_mel = (energy + LOG_GUARD).ln();
                if !log_mel.is_finite() {
                    return Err(frontend_error(
                        "output_nonfinite",
                        "frontend produced a non-finite log-mel value",
                    ));
                }
                if frame < valid_frames {
                    data[mel * valid_frames + frame] = log_mel;
                }
            }
        }
        frontend_checkpoint(checkpoint)?;

        Ok(SortformerFrontendOutput {
            mel_bins: SORTFORMER_MEL_BINS,
            valid_frames,
            data,
        })
    }
}

/// Output of the authenticated f32 depthwise-striding subsampler.
///
/// Values are time-major `[frames, 512]`. Debug output is redacted because the
/// embeddings are derived from caller audio.
#[derive(PartialEq)]
pub struct SortformerSubsamplingOutput {
    pub frames: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

/// Numeric result for one authenticated public L2 seam probe.
#[derive(Debug, Clone, PartialEq)]
pub struct SortformerSeamParityMetrics {
    pub stage: String,
    pub compared_values: usize,
    pub mismatch_count: usize,
    pub byte_exact: bool,
    pub max_abs_diff: f64,
    pub relative_l2: f64,
    pub accepted_abs_tolerance: f64,
    pub accepted_relative_l2: f64,
}

/// Exact integer result for one authenticated public seam probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortformerI64ParityMetrics {
    pub stage: String,
    pub compared_values: usize,
    pub mismatch_count: usize,
    pub byte_exact: bool,
}

/// Evaluation-only L6 state-boundary parity report.
#[derive(Debug, Clone, PartialEq)]
pub struct SortformerL6ParityReport {
    pub fixture: String,
    pub step: usize,
    pub boundary: String,
    pub f32_seams: Vec<SortformerSeamParityMetrics>,
    pub i64_seams: Vec<SortformerI64ParityMetrics>,
}

/// Evaluation-only L2 parity report for one public fixture streaming step.
#[derive(Debug, Clone, PartialEq)]
pub struct SortformerL2ParityReport {
    pub fixture: String,
    pub step: usize,
    pub seams: Vec<SortformerSeamParityMetrics>,
}

/// Exact synchronous Streaming Sortformer state for one audio stream.
///
/// The optional prediction buffers deliberately preserve NeMo's `None`
/// semantics: an empty tensor is not interchangeable with an absent tensor at
/// the first cache-compression transition.
#[derive(Clone, PartialEq)]
pub struct SortformerStreamingState {
    speaker_cache: Mat,
    speaker_cache_predictions: Option<Mat>,
    fifo: Mat,
    fifo_predictions: Option<Mat>,
    speaker_permutation: Option<[usize; SORTFORMER_SPEAKER_LANES]>,
    mean_silence_embedding: Vec<f32>,
    silence_frames: i64,
}

/// Exact discrete activity products emitted from the four-lane Sortformer head.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SortformerActivityOutput {
    pub frames: usize,
    pub activity: Vec<i64>,
    pub speech: Vec<i64>,
    pub overlap: Vec<i64>,
    pub change_indices: Vec<[i64; 2]>,
}

/// One post-processed Sortformer speaker interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SortformerSpeakerTurn {
    pub start_seconds: f32,
    pub end_seconds: f32,
    pub speaker: usize,
}

/// Complete native Sortformer result for one finite mono recording.
#[derive(Clone, Debug, PartialEq)]
pub struct SortformerDiarizationOutput {
    pub frames: usize,
    pub probabilities: Vec<f32>,
    pub activity: SortformerActivityOutput,
    pub turns: Vec<SortformerSpeakerTurn>,
}

/// Authenticated native whole-recording Sortformer session.
pub struct SortformerSession {
    frontend: SortformerFrontend,
    graph: SortformerF32Facade,
}

impl fmt::Debug for SortformerSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SortformerSession")
            .field("frontend", &self.frontend)
            .field("graph", &self.graph)
            .finish()
    }
}

/// Apply the frozen inclusive 0.5 L7 activity threshold.
pub fn sortformer_activity_output(
    probabilities: &[f32],
    frames: usize,
) -> FwResult<SortformerActivityOutput> {
    let expected = frames
        .checked_mul(SORTFORMER_SPEAKER_LANES)
        .ok_or_else(|| reference_error("l7_shape", "L7 probability size overflows"))?;
    if frames == 0
        || probabilities.len() != expected
        || probabilities.iter().any(|value| !value.is_finite())
    {
        return Err(reference_error(
            "l7_input",
            "L7 probabilities have invalid geometry or a non-finite value",
        ));
    }
    let mut activity = Vec::new();
    activity
        .try_reserve_exact(expected)
        .map_err(|_| reference_error("allocation", "L7 activity allocation failed"))?;
    let mut speech = Vec::new();
    speech
        .try_reserve_exact(frames)
        .map_err(|_| reference_error("allocation", "L7 speech allocation failed"))?;
    let mut overlap = Vec::new();
    overlap
        .try_reserve_exact(frames)
        .map_err(|_| reference_error("allocation", "L7 overlap allocation failed"))?;
    for row in probabilities.chunks(SORTFORMER_SPEAKER_LANES) {
        let active = row.iter().filter(|&&value| value >= 0.5).count();
        activity.extend(row.iter().map(|&value| i64::from(value >= 0.5)));
        speech.push(i64::from(active > 0));
        overlap.push(i64::from(active > 1));
    }
    let mut change_indices = Vec::new();
    for frame in 1..frames {
        let previous =
            &activity[(frame - 1) * SORTFORMER_SPEAKER_LANES..frame * SORTFORMER_SPEAKER_LANES];
        let current =
            &activity[frame * SORTFORMER_SPEAKER_LANES..(frame + 1) * SORTFORMER_SPEAKER_LANES];
        if previous != current {
            change_indices.push([
                0,
                i64::try_from(frame - 1)
                    .map_err(|_| reference_error("l7_shape", "L7 frame index exceeds i64"))?,
            ]);
        }
    }
    Ok(SortformerActivityOutput {
        frames,
        activity,
        speech,
        overlap,
        change_indices,
    })
}

/// Reproduce NeMo's default L8 TS-VAD post-processing at an 80 ms model stride.
pub fn sortformer_speaker_turns(
    probabilities: &[f32],
    frames: usize,
) -> FwResult<Vec<SortformerSpeakerTurn>> {
    let _ = sortformer_activity_output(probabilities, frames)?;
    const REPEATS: usize = 8;
    const TEN_MS_SECONDS: f64 = 0.01;
    let repeated_frames = frames
        .checked_mul(REPEATS)
        .ok_or_else(|| reference_error("l8_shape", "L8 repeated frame count overflows"))?;
    let mut turns = Vec::new();
    for speaker in 0..SORTFORMER_SPEAKER_LANES {
        let mut active_start = None;
        for frame in 0..frames {
            let value = probabilities[frame * SORTFORMER_SPEAKER_LANES + speaker];
            if active_start.is_some() {
                if value < 0.5 {
                    let start = active_start.take().ok_or_else(|| {
                        reference_error("l8_state", "L8 active start disappeared")
                    })?;
                    turns.push(SortformerSpeakerTurn {
                        start_seconds: (start as f64 * TEN_MS_SECONDS) as f32,
                        end_seconds: ((frame * REPEATS) as f64 * TEN_MS_SECONDS) as f32,
                        speaker,
                    });
                }
            } else if value > 0.5 {
                active_start = Some(frame * REPEATS);
            }
        }
        if let Some(start) = active_start {
            turns.push(SortformerSpeakerTurn {
                start_seconds: (start as f64 * TEN_MS_SECONDS) as f32,
                end_seconds: ((repeated_frames - 1) as f64 * TEN_MS_SECONDS) as f32,
                speaker,
            });
        }
    }
    turns.sort_by(|left, right| {
        left.start_seconds
            .total_cmp(&right.start_seconds)
            .then_with(|| left.end_seconds.total_cmp(&right.end_seconds))
            .then_with(|| left.speaker.cmp(&right.speaker))
    });
    Ok(turns)
}

fn clamp_sortformer_turns_to_duration(
    turns: &mut Vec<SortformerSpeakerTurn>,
    duration_seconds: f32,
) {
    turns.retain_mut(|turn| {
        turn.start_seconds = turn.start_seconds.min(duration_seconds);
        turn.end_seconds = turn.end_seconds.min(duration_seconds);
        turn.start_seconds < turn.end_seconds
    });
}

impl fmt::Debug for SortformerStreamingState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SortformerStreamingState")
            .field("speaker_cache_frames", &self.speaker_cache.rows)
            .field(
                "speaker_cache_predictions_present",
                &self.speaker_cache_predictions.is_some(),
            )
            .field("fifo_frames", &self.fifo.rows)
            .field("fifo_predictions_present", &self.fifo_predictions.is_some())
            .field(
                "speaker_permutation_present",
                &self.speaker_permutation.is_some(),
            )
            .field("silence_frames", &self.silence_frames)
            .field("audio_derived_values", &"<redacted>")
            .finish()
    }
}

impl SortformerStreamingState {
    /// Construct the pinned synchronous evaluation state. No caller audio or
    /// predictions are retained outside this owned value.
    #[must_use]
    pub fn new() -> Self {
        Self {
            speaker_cache: Mat::from_vec(0, SORTFORMER_ENCODER_WIDTH, Vec::new()),
            speaker_cache_predictions: None,
            fifo: Mat::from_vec(0, SORTFORMER_ENCODER_WIDTH, Vec::new()),
            fifo_predictions: None,
            speaker_permutation: None,
            mean_silence_embedding: vec![0.0; SORTFORMER_ENCODER_WIDTH],
            silence_frames: 0,
        }
    }

    #[must_use]
    pub fn speaker_cache_frames(&self) -> usize {
        self.speaker_cache.rows
    }
}

impl Default for SortformerStreamingState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SortformerSubsamplingOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SortformerSubsamplingOutput")
            .field("frames", &self.frames)
            .field("width", &self.width)
            .field("data", &"<redacted>")
            .finish()
    }
}

/// Narrow f32 compute facade for the pinned Streaming Sortformer graph.
///
/// This reference implementation intentionally exposes only admitted graph
/// operations. Dense projection delegates to FrankenTorch's checked f32 GEMM;
/// model-specific grouped Conv2d remains a safe scalar reference until parity
/// and profiling justify a fused kernel.
pub struct SortformerF32Facade {
    subsampler: SortformerSubsampler,
    fastconformer_blocks: Vec<FastConformerBlock>,
    encoder_projection: Affine,
    transformer_blocks: Vec<TransformerBlock>,
    speaker_hidden: Affine,
    speaker_logits: Affine,
}

impl fmt::Debug for SortformerF32Facade {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SortformerF32Facade")
            .field("subsampler", &"<authenticated weights redacted>")
            .field(
                "fastconformer_blocks",
                &"<17 authenticated blocks redacted>",
            )
            .field("encoder_projection", &"<authenticated weights redacted>")
            .field("transformer_blocks", &"<18 authenticated blocks redacted>")
            .field("speaker_head", &"<authenticated weights redacted>")
            .finish()
    }
}

impl SortformerF32Facade {
    /// Load the L2 reference graph from a fully authenticated model package.
    pub fn from_verified_package(package: &VerifiedSortformerPackage) -> FwResult<Self> {
        Self::from_verified_package_with_checkpoint(package, &|| Ok(()))
    }

    /// Load the L2 reference graph with cooperative cancellation.
    pub fn from_verified_package_with_checkpoint(
        package: &VerifiedSortformerPackage,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let tensors = package.safetensors();
        let mut fastconformer_blocks = Vec::new();
        fastconformer_blocks
            .try_reserve_exact(SORTFORMER_FASTCONFORMER_LAYERS)
            .map_err(|_| reference_error("allocation", "FastConformer block allocation failed"))?;
        for layer in 0..SORTFORMER_FASTCONFORMER_LAYERS {
            reference_checkpoint(checkpoint)?;
            fastconformer_blocks.push(FastConformerBlock::load(tensors, layer, checkpoint)?);
        }
        let encoder_projection = Affine::load(
            tensors,
            "sortformer_modules.encoder_proj",
            SORTFORMER_ENCODER_WIDTH,
            SORTFORMER_TRANSFORMER_WIDTH,
            true,
            checkpoint,
        )?;
        let mut transformer_blocks = Vec::new();
        transformer_blocks
            .try_reserve_exact(SORTFORMER_TRANSFORMER_LAYERS)
            .map_err(|_| reference_error("allocation", "Transformer block allocation failed"))?;
        for layer in 0..SORTFORMER_TRANSFORMER_LAYERS {
            reference_checkpoint(checkpoint)?;
            transformer_blocks.push(TransformerBlock::load(tensors, layer, checkpoint)?);
        }
        let speaker_hidden = Affine::load(
            tensors,
            "sortformer_modules.first_hidden_to_hidden",
            SORTFORMER_TRANSFORMER_WIDTH,
            SORTFORMER_TRANSFORMER_WIDTH,
            true,
            checkpoint,
        )?;
        let speaker_logits = Affine::load(
            tensors,
            "sortformer_modules.single_hidden_to_spks",
            SORTFORMER_TRANSFORMER_WIDTH,
            SORTFORMER_SPEAKER_LANES,
            true,
            checkpoint,
        )?;
        Ok(Self {
            subsampler: SortformerSubsampler::load(tensors, checkpoint)?,
            fastconformer_blocks,
            encoder_projection,
            transformer_blocks,
            speaker_hidden,
            speaker_logits,
        })
    }

    /// Run one fixed-profile streaming chunk and update its recurrent state.
    pub fn forward_streaming_chunk_with_checkpoint(
        &self,
        state: &mut SortformerStreamingState,
        time_major_features: &[f32],
        feature_frames: usize,
        left_feature_frames: usize,
        right_feature_frames: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Vec<f32>> {
        frontend_checkpoint(checkpoint)?;
        if !matches!(left_feature_frames, 0 | SORTFORMER_SUBSAMPLING_FACTOR)
            || !matches!(right_feature_frames, 0 | SORTFORMER_SUBSAMPLING_FACTOR)
            || left_feature_frames
                .checked_add(right_feature_frames)
                .is_none_or(|context| context >= feature_frames)
        {
            return Err(reference_error(
                "stream_context",
                "Sortformer stream context is outside the fixed one-frame profile",
            ));
        }
        let chunk_embeddings = self.subsample_feature_chunk_with_checkpoint(
            time_major_features,
            feature_frames,
            checkpoint,
        )?;
        let prefix_frames = state
            .speaker_cache
            .rows
            .checked_add(state.fifo.rows)
            .ok_or_else(|| reference_error("stream_shape", "stream prefix length overflows"))?;
        let total_frames = prefix_frames
            .checked_add(chunk_embeddings.frames)
            .ok_or_else(|| reference_error("stream_shape", "stream input length overflows"))?;
        let total_values = total_frames
            .checked_mul(SORTFORMER_ENCODER_WIDTH)
            .ok_or_else(|| reference_error("stream_shape", "stream input size overflows"))?;
        let mut encoder_values = Vec::new();
        encoder_values
            .try_reserve_exact(total_values)
            .map_err(|_| reference_error("allocation", "stream encoder input allocation failed"))?;
        encoder_values.extend_from_slice(&state.speaker_cache.data);
        encoder_values.extend_from_slice(&state.fifo.data);
        encoder_values.extend_from_slice(&chunk_embeddings.data);
        for value in &mut encoder_values {
            *value *= SORTFORMER_ENCODER_XSCALE;
        }
        let mut current = Mat::from_vec(total_frames, SORTFORMER_ENCODER_WIDTH, encoder_values);
        for block in &self.fastconformer_blocks {
            frontend_checkpoint(checkpoint)?;
            current = block.forward(&current)?;
        }
        current = self.encoder_projection.forward(&current)?;
        for block in &self.transformer_blocks {
            frontend_checkpoint(checkpoint)?;
            current = block.forward(&current)?;
        }
        relu_in_place(&mut current.data);
        let mut hidden = self.speaker_hidden.forward(&current)?;
        relu_in_place(&mut hidden.data);
        let logits = self.speaker_logits.forward(&hidden)?;
        let mut probabilities = logits;
        for value in &mut probabilities.data {
            *value = sigmoid_f32(*value)?;
        }
        let left_embedding_frames = left_feature_frames / SORTFORMER_SUBSAMPLING_FACTOR;
        let right_embedding_frames = right_feature_frames.div_ceil(SORTFORMER_SUBSAMPLING_FACTOR);
        let chunk_predictions = self.update_streaming_state(
            state,
            &chunk_embeddings,
            &probabilities,
            left_embedding_frames,
            right_embedding_frames,
        )?;
        frontend_checkpoint(checkpoint)?;
        Ok(chunk_predictions.data)
    }

    fn fastconformer_block(&self, layer: usize) -> FwResult<&FastConformerBlock> {
        self.fastconformer_blocks.get(layer).ok_or_else(|| {
            reference_error("fastconformer_layer", "FastConformer layer is out of range")
        })
    }

    /// Subsample one time-major `[feature_frames, 128]` streaming feature
    /// chunk. The caller must perform chunk/context selection before this seam.
    pub fn subsample_feature_chunk(
        &self,
        time_major_features: &[f32],
        feature_frames: usize,
    ) -> FwResult<SortformerSubsamplingOutput> {
        self.subsample_feature_chunk_with_checkpoint(
            time_major_features,
            feature_frames,
            &|| Ok(()),
        )
    }

    /// Cancellation-aware form of [`Self::subsample_feature_chunk`].
    pub fn subsample_feature_chunk_with_checkpoint(
        &self,
        time_major_features: &[f32],
        feature_frames: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<SortformerSubsamplingOutput> {
        let projection =
            self.subsampler
                .forward_output(time_major_features, feature_frames, checkpoint)?;
        Ok(SortformerSubsamplingOutput {
            frames: projection.rows,
            width: projection.cols,
            data: projection.data,
        })
    }

    /// Compare every captured L2 boundary for one already-selected public
    /// feature chunk against the authenticated probe pack.
    ///
    /// `time_major_features` is the exact chunk emitted by the frozen feature
    /// loader, including its left/right context. This API never opens corpus
    /// paths and therefore cannot accidentally make private audio part of the
    /// model package.
    pub fn verify_public_l2_chunk_parity(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        time_major_features: &[f32],
        feature_frames: usize,
    ) -> FwResult<SortformerL2ParityReport> {
        self.verify_public_l2_chunk_parity_with_checkpoint(
            activation_pack,
            fixture,
            step,
            time_major_features,
            feature_frames,
            &|| Ok(()),
        )
    }

    /// Cancellation-aware form of [`Self::verify_public_l2_chunk_parity`].
    pub fn verify_public_l2_chunk_parity_with_checkpoint(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        time_major_features: &[f32],
        feature_frames: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<SortformerL2ParityReport> {
        if !activation_pack
            .receipt()
            .fixtures
            .iter()
            .any(|candidate| candidate.name == fixture)
        {
            return Err(reference_error(
                "parity_fixture",
                "requested fixture is absent from the authenticated public receipt",
            ));
        }
        let transition = activation_pack
            .receipt()
            .streaming_transitions
            .get(fixture)
            .and_then(|transitions| transitions.get(step))
            .ok_or_else(|| {
                reference_error(
                    "parity_step",
                    "requested streaming step is absent from the authenticated public receipt",
                )
            })?;
        if usize::try_from(transition.step).ok() != Some(step)
            || usize::try_from(transition.input_feature_frames).ok() != Some(feature_frames)
            || usize::try_from(transition.valid_feature_frames).ok() != Some(feature_frames)
        {
            return Err(reference_error(
                "parity_chunk",
                "feature chunk geometry does not match the authenticated transition",
            ));
        }

        let trace = self
            .subsampler
            .forward(time_major_features, feature_frames, checkpoint)?;
        let prefix = format!("fixture.{fixture}.step.{step:03}.l2.subsampling");
        let candidates = [
            (format!("{prefix}.0"), trace.conv0.shape(), trace.conv0.data),
            (format!("{prefix}.2"), trace.conv2.shape(), trace.conv2.data),
            (format!("{prefix}.3"), trace.conv3.shape(), trace.conv3.data),
            (format!("{prefix}.5"), trace.conv5.shape(), trace.conv5.data),
            (format!("{prefix}.6"), trace.conv6.shape(), trace.conv6.data),
            (
                format!("{prefix}.projection"),
                vec![1, trace.projection.rows, trace.projection.cols],
                trace.projection.data,
            ),
        ];
        let mut seams = Vec::new();
        seams
            .try_reserve_exact(candidates.len())
            .map_err(|_| reference_error("allocation", "L2 parity report allocation failed"))?;
        for (stage, shape, values) in candidates {
            reference_checkpoint(checkpoint)?;
            seams.push(compare_public_f32_probe(
                activation_pack,
                &stage,
                &shape,
                &values,
                SORTFORMER_L2_MAX_ABS_DIFF,
                SORTFORMER_L2_MAX_RELATIVE_L2,
            )?);
        }
        Ok(SortformerL2ParityReport {
            fixture: fixture.to_owned(),
            step,
            seams,
        })
    }

    /// Verify the exact scaled input boundary of FastConformer block zero.
    /// This is the first L3 seam and intentionally stops before any block
    /// operator so a later block failure cannot obscure L2-to-L3 handoff.
    pub fn verify_public_l3_encoder_input(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        prior_stream_embeddings: &[f32],
        prior_stream_frames: usize,
        time_major_features: &[f32],
        feature_frames: usize,
    ) -> FwResult<SortformerSeamParityMetrics> {
        let input = self.assemble_l3_encoder_input(
            activation_pack,
            fixture,
            step,
            prior_stream_embeddings,
            prior_stream_frames,
            time_major_features,
            feature_frames,
        )?;
        let stage = format!("fixture.{fixture}.step.{step:03}.l3.fastconformer.block.00.input");
        compare_public_f32_probe(
            activation_pack,
            &stage,
            &[1, input.rows, input.cols],
            &input.data,
            SORTFORMER_L3_INPUT_MAX_ABS_DIFF,
            SORTFORMER_L3_INPUT_MAX_RELATIVE_L2,
        )
    }

    /// Verify the first FastConformer feed-forward module before its half-step
    /// residual is applied.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_public_l3_block0_feed_forward1(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        prior_stream_embeddings: &[f32],
        prior_stream_frames: usize,
        time_major_features: &[f32],
        feature_frames: usize,
    ) -> FwResult<SortformerSeamParityMetrics> {
        let mut input = self.assemble_l3_encoder_input(
            activation_pack,
            fixture,
            step,
            prior_stream_embeddings,
            prior_stream_frames,
            time_major_features,
            feature_frames,
        )?;
        let block = self.fastconformer_block(0)?;
        block.norm_feed_forward1.forward_in_place(&mut input)?;
        let output = block.feed_forward1.forward(&input)?;
        let stage =
            format!("fixture.{fixture}.step.{step:03}.l3.fastconformer.block.00.feed_forward1");
        compare_public_f32_probe(
            activation_pack,
            &stage,
            &[1, output.rows, output.cols],
            &output.data,
            SORTFORMER_L3_FFN_MAX_ABS_DIFF,
            SORTFORMER_L3_FFN_MAX_RELATIVE_L2,
        )
    }

    /// Verify the three raw block-00 Q/K/V affine seams after FFN1's
    /// half-step residual and self-attention LayerNorm.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_public_l3_block0_qkv(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        prior_stream_embeddings: &[f32],
        prior_stream_frames: usize,
        time_major_features: &[f32],
        feature_frames: usize,
    ) -> FwResult<Vec<SortformerSeamParityMetrics>> {
        let input = self.assemble_l3_encoder_input(
            activation_pack,
            fixture,
            step,
            prior_stream_embeddings,
            prior_stream_frames,
            time_major_features,
            feature_frames,
        )?;
        let mut normalized = input.clone();
        let block = self.fastconformer_block(0)?;
        block.norm_feed_forward1.forward_in_place(&mut normalized)?;
        let feed_forward = block.feed_forward1.forward(&normalized)?;
        let mut residual = input;
        add_scaled_residual_in_place(&mut residual, &feed_forward, 0.5)?;
        block.norm_self_att.forward_in_place(&mut residual)?;
        let (query, key, value) = block.self_attention.project_qkv(&residual)?;
        let prefix = format!("fixture.{fixture}.step.{step:03}.l3.fastconformer.block.00");
        let candidates = [
            (format!("{prefix}.attention_query"), query),
            (format!("{prefix}.attention_key"), key),
            (format!("{prefix}.attention_value"), value),
        ];
        let mut metrics = Vec::new();
        metrics
            .try_reserve_exact(candidates.len())
            .map_err(|_| reference_error("allocation", "Q/K/V parity report allocation failed"))?;
        for (stage, output) in candidates {
            metrics.push(compare_public_f32_probe(
                activation_pack,
                &stage,
                &[1, output.rows, output.cols],
                &output.data,
                SORTFORMER_L3_QKV_MAX_ABS_DIFF,
                SORTFORMER_L3_QKV_MAX_RELATIVE_L2,
            )?);
        }
        Ok(metrics)
    }

    /// Verify the encoder projection and all 18 post-LayerNorm Transformer
    /// blocks from an already-authenticated native L3 output.
    pub fn verify_public_l4_transformer(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        l3_output: &Mat,
    ) -> FwResult<(Vec<SortformerSeamParityMetrics>, Mat)> {
        if l3_output.rows == 0 || l3_output.cols != SORTFORMER_ENCODER_WIDTH {
            return Err(reference_error(
                "l4_input",
                "L4 requires a non-empty [frames, 512] L3 output",
            ));
        }
        let mut current = self.encoder_projection.forward(l3_output)?;
        let seam_count = SORTFORMER_TRANSFORMER_LAYERS
            .checked_mul(8)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| reference_error("allocation", "L4 seam count overflows"))?;
        let mut metrics = Vec::new();
        metrics
            .try_reserve_exact(seam_count)
            .map_err(|_| reference_error("allocation", "L4 parity report allocation failed"))?;
        metrics.push(compare_public_f32_probe(
            activation_pack,
            &format!("fixture.{fixture}.step.{step:03}.l4.encoder_projection"),
            &[1, current.rows, current.cols],
            &current.data,
            SORTFORMER_L4_MAX_ABS_DIFF,
            SORTFORMER_L4_MAX_RELATIVE_L2,
        )?);
        for (layer, block) in self.transformer_blocks.iter().enumerate() {
            let prefix =
                format!("fixture.{fixture}.step.{step:03}.l4.transformer.block.{layer:02}");
            metrics.push(compare_public_f32_probe(
                activation_pack,
                &format!("{prefix}.input"),
                &[1, current.rows, current.cols],
                &current.data,
                SORTFORMER_L4_MAX_ABS_DIFF,
                SORTFORMER_L4_MAX_RELATIVE_L2,
            )?);
            let trace = block.forward_trace(&current)?;
            let next = trace.output.clone();
            let candidates = [
                (format!("{prefix}.attention_query"), trace.attention_query),
                (format!("{prefix}.attention_key"), trace.attention_key),
                (format!("{prefix}.attention_value"), trace.attention_value),
                (format!("{prefix}.attention_output"), trace.attention_output),
                (
                    format!("{prefix}.feed_forward_inner"),
                    trace.feed_forward_inner,
                ),
                (
                    format!("{prefix}.feed_forward_output"),
                    trace.feed_forward_output,
                ),
                (format!("{prefix}.output"), trace.output),
            ];
            for (stage, output) in candidates {
                metrics.push(compare_public_f32_probe(
                    activation_pack,
                    &stage,
                    &[1, output.rows, output.cols],
                    &output.data,
                    SORTFORMER_L4_MAX_ABS_DIFF,
                    SORTFORMER_L4_MAX_RELATIVE_L2,
                )?);
            }
            current = next;
        }
        Ok((metrics, current))
    }

    /// Verify the exact evaluation speaker head: ReLU, 192-to-192 affine,
    /// ReLU, 192-to-4 affine, then elementwise sigmoid. Dropout is identity.
    pub fn verify_public_l5_speaker_head(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        l4_output: &Mat,
    ) -> FwResult<(Vec<SortformerSeamParityMetrics>, Mat)> {
        if l4_output.rows == 0 || l4_output.cols != SORTFORMER_TRANSFORMER_WIDTH {
            return Err(reference_error(
                "l5_input",
                "L5 requires a non-empty [frames, 192] Transformer output",
            ));
        }
        let mut activated_input = l4_output.clone();
        relu_in_place(&mut activated_input.data);
        let hidden = self.speaker_hidden.forward(&activated_input)?;
        let mut activated_hidden = hidden.clone();
        relu_in_place(&mut activated_hidden.data);
        let logits = self.speaker_logits.forward(&activated_hidden)?;
        let mut probabilities = logits.clone();
        for value in &mut probabilities.data {
            *value = sigmoid_f32(*value)?;
        }
        let prefix = format!("fixture.{fixture}.step.{step:03}.l5");
        let candidates = [
            (format!("{prefix}.hidden"), hidden),
            (format!("{prefix}.logits"), logits),
            (format!("{prefix}.probabilities"), probabilities.clone()),
        ];
        let mut metrics = Vec::new();
        metrics
            .try_reserve_exact(candidates.len() + 1)
            .map_err(|_| reference_error("allocation", "L5 parity report allocation failed"))?;
        for (stage, output) in candidates {
            metrics.push(compare_public_f32_probe(
                activation_pack,
                &stage,
                &[1, output.rows, output.cols],
                &output.data,
                SORTFORMER_L5_MAX_ABS_DIFF,
                SORTFORMER_L5_MAX_RELATIVE_L2,
            )?);
        }
        let transition = activation_pack
            .receipt()
            .streaming_transitions
            .get(fixture)
            .and_then(|transitions| transitions.get(step))
            .ok_or_else(|| reference_error("parity_step", "public streaming step is absent"))?;
        let cache_frames = usize::try_from(transition.before_cache_frames)
            .map_err(|_| reference_error("l5_stream", "cache frame count exceeds usize"))?;
        let left_frames = usize::try_from(transition.left_offset)
            .map_err(|_| reference_error("l5_stream", "left context exceeds usize"))?
            / SORTFORMER_SUBSAMPLING_FACTOR;
        let output_frames = usize::try_from(transition.output_frames)
            .map_err(|_| reference_error("l5_stream", "output frame count exceeds usize"))?;
        let stream_start = cache_frames
            .checked_add(left_frames)
            .ok_or_else(|| reference_error("l5_stream", "stream output offset overflows"))?;
        let stream_end = stream_start
            .checked_add(output_frames)
            .ok_or_else(|| reference_error("l5_stream", "stream output extent overflows"))?;
        let start_value = stream_start
            .checked_mul(SORTFORMER_SPEAKER_LANES)
            .ok_or_else(|| reference_error("l5_stream", "stream value offset overflows"))?;
        let end_value = stream_end
            .checked_mul(SORTFORMER_SPEAKER_LANES)
            .ok_or_else(|| reference_error("l5_stream", "stream value extent overflows"))?;
        let stream_values = probabilities
            .data
            .get(start_value..end_value)
            .ok_or_else(|| {
                reference_error(
                    "l5_stream",
                    "stream output interval exceeds speaker probabilities",
                )
            })?;
        metrics.push(compare_public_f32_probe(
            activation_pack,
            &format!("{prefix}.stream_output"),
            &[1, output_frames, SORTFORMER_SPEAKER_LANES],
            stream_values,
            SORTFORMER_L5_MAX_ABS_DIFF,
            SORTFORMER_L5_MAX_RELATIVE_L2,
        )?);
        Ok((metrics, probabilities))
    }

    /// Apply the pinned synchronous L6 cache/FIFO update for one native
    /// speaker-head invocation and return its context-trimmed stream output.
    ///
    /// `chunk_embeddings` are the L2 pre-encode embeddings including left and
    /// right context. `probabilities` cover the concatenated prior cache, FIFO,
    /// and complete contextual chunk in that order.
    pub fn update_streaming_state(
        &self,
        state: &mut SortformerStreamingState,
        chunk_embeddings: &SortformerSubsamplingOutput,
        probabilities: &Mat,
        left_context_frames: usize,
        right_context_frames: usize,
    ) -> FwResult<Mat> {
        validate_streaming_state(state)?;
        if chunk_embeddings.width != SORTFORMER_ENCODER_WIDTH
            || chunk_embeddings.frames.checked_mul(chunk_embeddings.width)
                != Some(chunk_embeddings.data.len())
            || chunk_embeddings.data.iter().any(|value| !value.is_finite())
        {
            return Err(reference_error(
                "l6_chunk",
                "L6 chunk embeddings have invalid geometry or values",
            ));
        }
        let chunk_frames = chunk_embeddings
            .frames
            .checked_sub(left_context_frames)
            .and_then(|frames| frames.checked_sub(right_context_frames))
            .ok_or_else(|| reference_error("l6_chunk", "L6 context exceeds chunk frames"))?;
        if chunk_frames == 0 {
            return Err(reference_error(
                "l6_chunk",
                "L6 requires at least one non-context chunk frame",
            ));
        }
        let expected_probability_rows = state
            .speaker_cache
            .rows
            .checked_add(state.fifo.rows)
            .and_then(|rows| rows.checked_add(chunk_embeddings.frames))
            .ok_or_else(|| reference_error("l6_shape", "L6 probability rows overflow"))?;
        if probabilities.rows != expected_probability_rows
            || probabilities.cols != SORTFORMER_SPEAKER_LANES
            || probabilities.rows.checked_mul(probabilities.cols) != Some(probabilities.data.len())
            || probabilities.data.iter().any(|value| !value.is_finite())
        {
            return Err(reference_error(
                "l6_probabilities",
                "L6 probabilities have invalid concatenated geometry or values",
            ));
        }

        let ordered_probabilities = if let Some(permutation) = state.speaker_permutation {
            inverse_permute_speaker_probabilities(probabilities, permutation)?
        } else {
            probabilities.clone()
        };
        let cache_frames = state.speaker_cache.rows;
        let fifo_frames = state.fifo.rows;
        state.fifo_predictions = Some(slice_mat_rows(
            &ordered_probabilities,
            cache_frames,
            cache_frames
                .checked_add(fifo_frames)
                .ok_or_else(|| reference_error("l6_shape", "L6 FIFO extent overflows"))?,
        )?);
        let central_start = cache_frames
            .checked_add(fifo_frames)
            .and_then(|frames| frames.checked_add(left_context_frames))
            .ok_or_else(|| reference_error("l6_shape", "L6 chunk offset overflows"))?;
        let central_end = central_start
            .checked_add(chunk_frames)
            .ok_or_else(|| reference_error("l6_shape", "L6 chunk extent overflows"))?;
        let chunk_predictions = slice_mat_rows(&ordered_probabilities, central_start, central_end)?;
        let chunk = Mat::from_vec(
            chunk_frames,
            SORTFORMER_ENCODER_WIDTH,
            slice_rows(
                &chunk_embeddings.data,
                SORTFORMER_ENCODER_WIDTH,
                left_context_frames,
                left_context_frames
                    .checked_add(chunk_frames)
                    .ok_or_else(|| reference_error("l6_shape", "L6 chunk extent overflows"))?,
            )?
            .to_vec(),
        );
        append_mat_rows(&mut state.fifo, &chunk)?;
        append_mat_rows(
            state
                .fifo_predictions
                .as_mut()
                .ok_or_else(|| reference_error("l6_state", "L6 FIFO predictions disappeared"))?,
            &chunk_predictions,
        )?;

        if fifo_frames
            .checked_add(chunk_frames)
            .ok_or_else(|| reference_error("l6_shape", "L6 FIFO length overflows"))?
            > SORTFORMER_FIFO_FRAMES
        {
            let total_fifo_frames = fifo_frames + chunk_frames;
            let required_pop = chunk_frames
                .checked_sub(SORTFORMER_FIFO_FRAMES)
                .and_then(|frames| frames.checked_add(fifo_frames))
                .ok_or_else(|| reference_error("l6_shape", "L6 pop length underflows"))?;
            let pop_frames = SORTFORMER_CACHE_UPDATE_FRAMES
                .max(required_pop)
                .min(total_fifo_frames);
            let pop_embeddings = slice_mat_rows(&state.fifo, 0, pop_frames)?;
            let pop_predictions = slice_mat_rows(
                state
                    .fifo_predictions
                    .as_ref()
                    .ok_or_else(|| reference_error("l6_state", "L6 FIFO predictions absent"))?,
                0,
                pop_frames,
            )?;
            update_silence_profile(state, &pop_embeddings, &pop_predictions)?;
            state.fifo = slice_mat_rows(&state.fifo, pop_frames, total_fifo_frames)?;
            state.fifo_predictions = Some(slice_mat_rows(
                state
                    .fifo_predictions
                    .as_ref()
                    .ok_or_else(|| reference_error("l6_state", "L6 FIFO predictions absent"))?,
                pop_frames,
                total_fifo_frames,
            )?);

            append_mat_rows(&mut state.speaker_cache, &pop_embeddings)?;
            if let Some(cache_predictions) = state.speaker_cache_predictions.as_mut() {
                append_mat_rows(cache_predictions, &pop_predictions)?;
            }
            if state.speaker_cache.rows > SORTFORMER_SPEAKER_CACHE_FRAMES {
                if state.speaker_cache_predictions.is_none() {
                    let mut synthesized = slice_mat_rows(&ordered_probabilities, 0, cache_frames)?;
                    append_mat_rows(&mut synthesized, &pop_predictions)?;
                    state.speaker_cache_predictions = Some(synthesized);
                }
                compress_speaker_cache(state)?;
            }
        }
        validate_streaming_state(state)?;
        Ok(chunk_predictions)
    }

    /// Compare one complete native L6 state boundary, including the exact
    /// presence/absence contract that controls first-cache compression.
    pub fn verify_public_l6_state(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        boundary: &str,
        state: &SortformerStreamingState,
    ) -> FwResult<SortformerL6ParityReport> {
        validate_streaming_state(state)?;
        let transition = activation_pack
            .receipt()
            .streaming_transitions
            .get(fixture)
            .and_then(|transitions| transitions.get(step))
            .ok_or_else(|| reference_error("parity_step", "public streaming step is absent"))?;
        let expected_options = match boundary {
            "before" => &transition.before_options,
            "after" => &transition.after_options,
            _ => {
                return Err(reference_error(
                    "l6_boundary",
                    "L6 boundary must be `before` or `after`",
                ));
            }
        };
        let option_value = |name: &str| {
            expected_options.get(name).copied().ok_or_else(|| {
                reference_error("l6_options", "public L6 option contract is incomplete")
            })
        };
        let observed_options = [
            ("spkcache", true),
            ("spkcache_lengths", false),
            ("spkcache_preds", state.speaker_cache_predictions.is_some()),
            ("fifo", true),
            ("fifo_lengths", false),
            ("fifo_preds", state.fifo_predictions.is_some()),
            ("spk_perm", state.speaker_permutation.is_some()),
            ("mean_sil_emb", true),
            ("n_sil_frames", true),
        ];
        for (name, observed) in observed_options {
            if option_value(name)? != observed {
                return Err(reference_error(
                    "l6_options",
                    &format!("native L6 `{name}` presence differs at {boundary}"),
                ));
            }
        }

        let prefix = format!("fixture.{fixture}.step.{step:03}.l6.{boundary}");
        let mut candidates = Vec::new();
        candidates
            .try_reserve_exact(5)
            .map_err(|_| reference_error("allocation", "L6 parity allocation failed"))?;
        candidates.push((
            format!("{prefix}.spkcache"),
            vec![1, state.speaker_cache.rows, SORTFORMER_ENCODER_WIDTH],
            state.speaker_cache.data.as_slice(),
        ));
        candidates.push((
            format!("{prefix}.fifo"),
            vec![1, state.fifo.rows, SORTFORMER_ENCODER_WIDTH],
            state.fifo.data.as_slice(),
        ));
        candidates.push((
            format!("{prefix}.mean_sil_emb"),
            vec![1, SORTFORMER_ENCODER_WIDTH],
            state.mean_silence_embedding.as_slice(),
        ));
        if let Some(predictions) = &state.speaker_cache_predictions {
            candidates.push((
                format!("{prefix}.spkcache_preds"),
                vec![1, predictions.rows, SORTFORMER_SPEAKER_LANES],
                predictions.data.as_slice(),
            ));
        }
        if let Some(predictions) = &state.fifo_predictions {
            candidates.push((
                format!("{prefix}.fifo_preds"),
                vec![1, predictions.rows, SORTFORMER_SPEAKER_LANES],
                predictions.data.as_slice(),
            ));
        }
        let mut f32_seams = Vec::new();
        f32_seams
            .try_reserve_exact(candidates.len())
            .map_err(|_| reference_error("allocation", "L6 parity allocation failed"))?;
        for (stage, shape, values) in candidates {
            f32_seams.push(compare_public_f32_probe(
                activation_pack,
                &stage,
                &shape,
                values,
                SORTFORMER_L6_MAX_ABS_DIFF,
                SORTFORMER_L6_MAX_RELATIVE_L2,
            )?);
        }
        let silence_stage = format!("{prefix}.n_sil_frames");
        let silence = [state.silence_frames];
        let i64_seams = vec![compare_public_i64_probe(
            activation_pack,
            &silence_stage,
            &[1],
            &silence,
        )?];
        Ok(SortformerL6ParityReport {
            fixture: fixture.to_owned(),
            step,
            boundary: boundary.to_owned(),
            f32_seams,
            i64_seams,
        })
    }

    /// Verify the complete block-00 relative-attention output projection.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_public_l3_block0_attention_output(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        prior_stream_embeddings: &[f32],
        prior_stream_frames: usize,
        time_major_features: &[f32],
        feature_frames: usize,
    ) -> FwResult<SortformerSeamParityMetrics> {
        let input = self.assemble_l3_encoder_input(
            activation_pack,
            fixture,
            step,
            prior_stream_embeddings,
            prior_stream_frames,
            time_major_features,
            feature_frames,
        )?;
        let mut normalized = input.clone();
        let block = self.fastconformer_block(0)?;
        block.norm_feed_forward1.forward_in_place(&mut normalized)?;
        let feed_forward = block.feed_forward1.forward(&normalized)?;
        let mut residual = input;
        add_scaled_residual_in_place(&mut residual, &feed_forward, 0.5)?;
        block.norm_self_att.forward_in_place(&mut residual)?;
        let output = block.self_attention.forward(&residual)?;
        let stage =
            format!("fixture.{fixture}.step.{step:03}.l3.fastconformer.block.00.attention_output");
        compare_public_f32_probe(
            activation_pack,
            &stage,
            &[1, output.rows, output.cols],
            &output.data,
            SORTFORMER_L3_ATTENTION_MAX_ABS_DIFF,
            SORTFORMER_L3_ATTENTION_MAX_RELATIVE_L2,
        )
    }

    /// Verify the captured convolution-depthwise, FFN2, and final output seams
    /// for FastConformer block zero.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_public_l3_block0_tail(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        prior_stream_embeddings: &[f32],
        prior_stream_frames: usize,
        time_major_features: &[f32],
        feature_frames: usize,
    ) -> FwResult<Vec<SortformerSeamParityMetrics>> {
        let input = self.assemble_l3_encoder_input(
            activation_pack,
            fixture,
            step,
            prior_stream_embeddings,
            prior_stream_frames,
            time_major_features,
            feature_frames,
        )?;
        let trace = self.fastconformer_block(0)?.forward_trace(&input)?;

        let prefix = format!("fixture.{fixture}.step.{step:03}.l3.fastconformer.block.00");
        let candidates = [
            (
                format!("{prefix}.attention_output"),
                vec![1, trace.attention.rows, trace.attention.cols],
                trace.attention.data,
                SORTFORMER_L3_ATTENTION_MAX_ABS_DIFF,
                SORTFORMER_L3_ATTENTION_MAX_RELATIVE_L2,
            ),
            (
                format!("{prefix}.convolution_depthwise"),
                vec![1, SORTFORMER_ENCODER_WIDTH, trace.output.rows],
                trace.convolution_depthwise_channel_major,
                SORTFORMER_L3_CONV_MAX_ABS_DIFF,
                SORTFORMER_L3_CONV_MAX_RELATIVE_L2,
            ),
            (
                format!("{prefix}.feed_forward2"),
                vec![1, trace.feed_forward2.rows, trace.feed_forward2.cols],
                trace.feed_forward2.data,
                SORTFORMER_L3_BLOCK_MAX_ABS_DIFF,
                SORTFORMER_L3_BLOCK_MAX_RELATIVE_L2,
            ),
            (
                format!("{prefix}.output"),
                vec![1, trace.output.rows, trace.output.cols],
                trace.output.data,
                SORTFORMER_L3_BLOCK_MAX_ABS_DIFF,
                SORTFORMER_L3_BLOCK_MAX_RELATIVE_L2,
            ),
        ];
        let mut metrics = Vec::new();
        metrics
            .try_reserve_exact(candidates.len())
            .map_err(|_| reference_error("allocation", "block tail report allocation failed"))?;
        for (stage, shape, values, abs_tolerance, relative_tolerance) in candidates {
            metrics.push(compare_public_f32_probe(
                activation_pack,
                &stage,
                &shape,
                &values,
                abs_tolerance,
                relative_tolerance,
            )?);
        }
        Ok(metrics)
    }

    /// Verify every captured seam through the complete 17-layer
    /// FastConformer chain for one independently reconstructable public state.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_public_l3_fastconformer(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        prior_stream_embeddings: &[f32],
        prior_stream_frames: usize,
        time_major_features: &[f32],
        feature_frames: usize,
    ) -> FwResult<(Vec<SortformerSeamParityMetrics>, Mat)> {
        let mut current = self.assemble_l3_encoder_input(
            activation_pack,
            fixture,
            step,
            prior_stream_embeddings,
            prior_stream_frames,
            time_major_features,
            feature_frames,
        )?;
        let seam_count = SORTFORMER_FASTCONFORMER_LAYERS
            .checked_mul(9)
            .ok_or_else(|| reference_error("allocation", "FastConformer seam count overflows"))?;
        let mut metrics = Vec::new();
        metrics.try_reserve_exact(seam_count).map_err(|_| {
            reference_error(
                "allocation",
                "FastConformer parity report allocation failed",
            )
        })?;

        for layer in 0..SORTFORMER_FASTCONFORMER_LAYERS {
            let prefix =
                format!("fixture.{fixture}.step.{step:03}.l3.fastconformer.block.{layer:02}");
            let (abs_tolerance, relative_tolerance) = if layer == 0 {
                (
                    SORTFORMER_L3_INPUT_MAX_ABS_DIFF,
                    SORTFORMER_L3_INPUT_MAX_RELATIVE_L2,
                )
            } else {
                (
                    SORTFORMER_L3_BLOCK_MAX_ABS_DIFF,
                    SORTFORMER_L3_BLOCK_MAX_RELATIVE_L2,
                )
            };
            metrics.push(compare_public_f32_probe(
                activation_pack,
                &format!("{prefix}.input"),
                &[1, current.rows, current.cols],
                &current.data,
                abs_tolerance,
                relative_tolerance,
            )?);
            let trace = self.fastconformer_block(layer)?.forward_trace(&current)?;
            let next = trace.output.clone();
            let candidates = [
                (format!("{prefix}.feed_forward1"), trace.feed_forward1),
                (format!("{prefix}.attention_query"), trace.attention_query),
                (format!("{prefix}.attention_key"), trace.attention_key),
                (format!("{prefix}.attention_value"), trace.attention_value),
                (format!("{prefix}.attention_output"), trace.attention),
                (format!("{prefix}.feed_forward2"), trace.feed_forward2),
                (format!("{prefix}.output"), trace.output),
            ];
            for (stage, output) in candidates {
                metrics.push(compare_public_f32_probe(
                    activation_pack,
                    &stage,
                    &[1, output.rows, output.cols],
                    &output.data,
                    SORTFORMER_L3_BLOCK_MAX_ABS_DIFF,
                    SORTFORMER_L3_BLOCK_MAX_RELATIVE_L2,
                )?);
            }
            metrics.push(compare_public_f32_probe(
                activation_pack,
                &format!("{prefix}.convolution_depthwise"),
                &[1, SORTFORMER_ENCODER_WIDTH, next.rows],
                &trace.convolution_depthwise_channel_major,
                SORTFORMER_L3_CONV_MAX_ABS_DIFF,
                SORTFORMER_L3_CONV_MAX_RELATIVE_L2,
            )?);
            current = next;
        }
        Ok((metrics, current))
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble_l3_encoder_input(
        &self,
        activation_pack: &VerifiedSortformerPublicActivationPack,
        fixture: &str,
        step: usize,
        prior_stream_embeddings: &[f32],
        prior_stream_frames: usize,
        time_major_features: &[f32],
        feature_frames: usize,
    ) -> FwResult<Mat> {
        let transition = activation_pack
            .receipt()
            .streaming_transitions
            .get(fixture)
            .and_then(|transitions| transitions.get(step))
            .ok_or_else(|| reference_error("parity_step", "public streaming step is absent"))?;
        if usize::try_from(transition.input_feature_frames).ok() != Some(feature_frames) {
            return Err(reference_error(
                "parity_chunk",
                "L3 feature chunk geometry differs from the public transition",
            ));
        }
        let trace = self
            .subsampler
            .forward(time_major_features, feature_frames, &|| Ok(()))?;
        let prior_elements = prior_stream_frames
            .checked_mul(SORTFORMER_ENCODER_WIDTH)
            .ok_or_else(|| reference_error("shape", "L3 stream prefix geometry overflows"))?;
        if prior_stream_embeddings.len() != prior_elements {
            return Err(reference_error(
                "shape",
                "L3 stream prefix payload differs from its declared geometry",
            ));
        }
        let total_frames = prior_stream_frames
            .checked_add(trace.projection.rows)
            .ok_or_else(|| reference_error("shape", "L3 concatenated frame count overflows"))?;
        let total_elements = total_frames
            .checked_mul(SORTFORMER_ENCODER_WIDTH)
            .ok_or_else(|| reference_error("shape", "L3 concatenated payload size overflows"))?;
        let mut scaled = Vec::new();
        scaled
            .try_reserve_exact(total_elements)
            .map_err(|_| reference_error("allocation", "L3 input allocation failed"))?;
        scaled.extend_from_slice(prior_stream_embeddings);
        scaled.extend_from_slice(&trace.projection.data);
        for value in &mut scaled {
            *value *= SORTFORMER_ENCODER_XSCALE;
        }
        Ok(Mat::from_vec(total_frames, trace.projection.cols, scaled))
    }
}

impl SortformerSession {
    /// Construct a reusable native session from an authenticated converted package.
    pub fn from_verified_package(package: &VerifiedSortformerPackage) -> FwResult<Self> {
        Self::from_verified_package_with_checkpoint(package, &|| Ok(()))
    }

    /// Cancellation-aware authenticated session construction.
    pub fn from_verified_package_with_checkpoint(
        package: &VerifiedSortformerPackage,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        Ok(Self {
            frontend: SortformerFrontend::from_verified_package_with_checkpoint(
                package, checkpoint,
            )?,
            graph: SortformerF32Facade::from_verified_package_with_checkpoint(package, checkpoint)?,
        })
    }

    /// Diarize one finite mono 16 kHz recording with the frozen streaming profile.
    pub fn diarize(&self, pcm: SortformerPcm<'_>) -> FwResult<SortformerDiarizationOutput> {
        self.diarize_with_checkpoint(pcm, &|| Ok(()))
    }

    /// Cancellation-aware form of [`Self::diarize`].
    pub fn diarize_with_checkpoint(
        &self,
        pcm: SortformerPcm<'_>,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<SortformerDiarizationOutput> {
        let duration_seconds = (pcm.samples.len() as f64 / pcm.sample_rate_hz as f64) as f32;
        let features = self.frontend.compute_with_checkpoint(pcm, checkpoint)?;
        const CENTRAL_FEATURE_FRAMES: usize =
            SORTFORMER_CACHE_UPDATE_FRAMES * SORTFORMER_SUBSAMPLING_FACTOR;
        let expected_output_frames = features
            .valid_frames
            .div_ceil(SORTFORMER_SUBSAMPLING_FACTOR);
        let expected_values = expected_output_frames
            .checked_mul(SORTFORMER_SPEAKER_LANES)
            .ok_or_else(|| reference_error("stream_shape", "output probability size overflows"))?;
        let mut probabilities = Vec::new();
        probabilities
            .try_reserve_exact(expected_values)
            .map_err(|_| reference_error("allocation", "output probability allocation failed"))?;
        let mut state = SortformerStreamingState::new();
        let mut central_start = 0usize;
        while central_start < features.valid_frames {
            frontend_checkpoint(checkpoint)?;
            let left = if central_start == 0 {
                0
            } else {
                SORTFORMER_SUBSAMPLING_FACTOR
            };
            let central_end = central_start
                .checked_add(CENTRAL_FEATURE_FRAMES)
                .map_or(features.valid_frames, |end| end.min(features.valid_frames));
            let right = (features.valid_frames - central_end).min(SORTFORMER_SUBSAMPLING_FACTOR);
            let chunk_start = central_start - left;
            let chunk_end = central_end + right;
            let feature_frames = chunk_end - chunk_start;
            let feature_values = feature_frames
                .checked_mul(SORTFORMER_MEL_BINS)
                .ok_or_else(|| reference_error("stream_shape", "feature chunk size overflows"))?;
            let mut time_major = Vec::new();
            time_major
                .try_reserve_exact(feature_values)
                .map_err(|_| reference_error("allocation", "feature chunk allocation failed"))?;
            for frame in chunk_start..chunk_end {
                for mel in 0..SORTFORMER_MEL_BINS {
                    time_major.push(features.data[mel * features.valid_frames + frame]);
                }
            }
            probabilities.extend(self.graph.forward_streaming_chunk_with_checkpoint(
                &mut state,
                &time_major,
                feature_frames,
                left,
                right,
                checkpoint,
            )?);
            central_start = central_end;
        }
        let frames = probabilities.len() / SORTFORMER_SPEAKER_LANES;
        if frames != expected_output_frames || probabilities.len() != expected_values {
            return Err(reference_error(
                "stream_shape",
                "native stream produced an unexpected probability length",
            ));
        }
        let activity = sortformer_activity_output(&probabilities, frames)?;
        let mut turns = sortformer_speaker_turns(&probabilities, frames)?;
        clamp_sortformer_turns_to_duration(&mut turns, duration_seconds);
        Ok(SortformerDiarizationOutput {
            frames,
            probabilities,
            activity,
            turns,
        })
    }
}

struct Affine {
    input: usize,
    output: usize,
    weight_t: Mat,
    bias: Option<Vec<f32>>,
}

impl Affine {
    fn load(
        package: &SafetensorsFile,
        prefix: &str,
        input: usize,
        output: usize,
        use_bias: bool,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let weight_name = format!("{prefix}.weight");
        let natural = load_model_f32(package, &weight_name, &[output, input], checkpoint)?;
        let weight_t = transpose_affine_weight(&weight_name, output, input, natural)?;
        let bias = if use_bias {
            let bias_name = format!("{prefix}.bias");
            Some(load_model_f32(package, &bias_name, &[output], checkpoint)?)
        } else {
            None
        };
        Ok(Self {
            input,
            output,
            weight_t,
            bias,
        })
    }

    fn forward(&self, input: &Mat) -> FwResult<Mat> {
        let expected_values = input.rows.checked_mul(input.cols);
        if input.rows == 0
            || input.cols != self.input
            || expected_values != Some(input.data.len())
            || input.data.iter().any(|value| !value.is_finite())
        {
            return Err(reference_error(
                "affine_input",
                "affine input has invalid geometry or a non-finite value",
            ));
        }
        let output = nn::matmul_bias(input, &self.weight_t, self.bias.as_deref())
            .map_err(|_| reference_error("affine_kernel", "FrankenTorch rejected affine input"))?;
        if output.cols != self.output || output.data.iter().any(|value| !value.is_finite()) {
            return Err(reference_error(
                "affine_output",
                "affine output has invalid geometry or a non-finite value",
            ));
        }
        Ok(output)
    }

    fn load_pointwise_conv1d(
        package: &SafetensorsFile,
        prefix: &str,
        input: usize,
        output: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        let natural = load_model_f32(package, &weight_name, &[output, input, 1], checkpoint)?;
        Ok(Self {
            input,
            output,
            weight_t: transpose_affine_weight(&weight_name, output, input, natural)?,
            bias: Some(load_model_f32(package, &bias_name, &[output], checkpoint)?),
        })
    }
}

struct LayerNorm {
    width: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    const EPSILON: f32 = 1.0e-5;

    fn load(
        package: &SafetensorsFile,
        prefix: &str,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        Self::load_width(package, prefix, SORTFORMER_ENCODER_WIDTH, checkpoint)
    }

    fn load_width(
        package: &SafetensorsFile,
        prefix: &str,
        width: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        if width == 0 {
            return Err(reference_error(
                "layer_norm_width",
                "LayerNorm width must be non-zero",
            ));
        }
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        let weight = load_model_f32(package, &weight_name, &[width], checkpoint)?;
        let bias = load_model_f32(package, &bias_name, &[width], checkpoint)?;
        Ok(Self {
            width,
            weight,
            bias,
        })
    }

    fn forward_in_place(&self, input: &mut Mat) -> FwResult<()> {
        let expected_values = input.rows.checked_mul(input.cols);
        if input.rows == 0
            || input.cols != self.width
            || expected_values != Some(input.data.len())
            || input.data.iter().any(|value| !value.is_finite())
        {
            return Err(reference_error(
                "layer_norm_input",
                "LayerNorm input has invalid geometry or a non-finite value",
            ));
        }
        nn::layer_norm(input, &self.weight, &self.bias, Self::EPSILON);
        if input.data.iter().any(|value| !value.is_finite()) {
            return Err(reference_error(
                "layer_norm_output",
                "LayerNorm produced a non-finite value",
            ));
        }
        Ok(())
    }
}

struct ConformerFeedForward {
    linear1: Affine,
    linear2: Affine,
}

impl ConformerFeedForward {
    const INNER_WIDTH: usize = 2048;

    fn load(
        package: &SafetensorsFile,
        prefix: &str,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        Ok(Self {
            linear1: Affine::load(
                package,
                &format!("{prefix}.linear1"),
                SORTFORMER_ENCODER_WIDTH,
                Self::INNER_WIDTH,
                true,
                checkpoint,
            )?,
            linear2: Affine::load(
                package,
                &format!("{prefix}.linear2"),
                Self::INNER_WIDTH,
                SORTFORMER_ENCODER_WIDTH,
                true,
                checkpoint,
            )?,
        })
    }

    fn forward(&self, input: &Mat) -> FwResult<Mat> {
        let mut hidden = self.linear1.forward(input)?;
        swish_in_place(&mut hidden.data)?;
        self.linear2.forward(&hidden)
    }
}

struct BatchNorm1d {
    width: usize,
    scale: Vec<f32>,
    bias: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
}

impl BatchNorm1d {
    const EPSILON: f32 = 1.0e-5;

    fn load(
        package: &SafetensorsFile,
        prefix: &str,
        width: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let load = |suffix: &str| {
            load_model_f32(package, &format!("{prefix}.{suffix}"), &[width], checkpoint)
        };
        Ok(Self {
            width,
            scale: load("weight")?,
            bias: load("bias")?,
            running_mean: load("running_mean")?,
            running_var: load("running_var")?,
        })
    }

    fn forward_in_place(&self, input: &mut Mat) -> FwResult<()> {
        if input.rows == 0
            || input.cols != self.width
            || input.rows.checked_mul(input.cols) != Some(input.data.len())
            || input.data.iter().any(|value| !value.is_finite())
        {
            return Err(reference_error(
                "batch_norm_input",
                "BatchNorm input has invalid geometry or a non-finite value",
            ));
        }
        for row in input.data.chunks_mut(self.width) {
            for channel in 0..self.width {
                let variance = self.running_var[channel];
                if variance < 0.0 {
                    return Err(reference_error(
                        "batch_norm_state",
                        "BatchNorm running variance is negative",
                    ));
                }
                row[channel] = (row[channel] - self.running_mean[channel])
                    / (variance + Self::EPSILON).sqrt()
                    * self.scale[channel]
                    + self.bias[channel];
                if !row[channel].is_finite() {
                    return Err(reference_error(
                        "batch_norm_output",
                        "BatchNorm produced a non-finite value",
                    ));
                }
            }
        }
        Ok(())
    }
}

struct ConformerConvolution {
    pointwise1: Affine,
    depthwise_weight: Vec<f32>,
    depthwise_bias: Vec<f32>,
    batch_norm: BatchNorm1d,
    pointwise2: Affine,
}

struct ConformerConvolutionTrace {
    depthwise_channel_major: Vec<f32>,
    output: Mat,
}

impl ConformerConvolution {
    const KERNEL: usize = 9;
    const PADDING: usize = 4;

    fn load(
        package: &SafetensorsFile,
        prefix: &str,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        Ok(Self {
            pointwise1: Affine::load_pointwise_conv1d(
                package,
                &format!("{prefix}.pointwise_conv1"),
                SORTFORMER_ENCODER_WIDTH,
                SORTFORMER_ENCODER_WIDTH * 2,
                checkpoint,
            )?,
            depthwise_weight: load_model_f32(
                package,
                &format!("{prefix}.depthwise_conv.weight"),
                &[SORTFORMER_ENCODER_WIDTH, 1, Self::KERNEL],
                checkpoint,
            )?,
            depthwise_bias: load_model_f32(
                package,
                &format!("{prefix}.depthwise_conv.bias"),
                &[SORTFORMER_ENCODER_WIDTH],
                checkpoint,
            )?,
            batch_norm: BatchNorm1d::load(
                package,
                &format!("{prefix}.batch_norm"),
                SORTFORMER_ENCODER_WIDTH,
                checkpoint,
            )?,
            pointwise2: Affine::load_pointwise_conv1d(
                package,
                &format!("{prefix}.pointwise_conv2"),
                SORTFORMER_ENCODER_WIDTH,
                SORTFORMER_ENCODER_WIDTH,
                checkpoint,
            )?,
        })
    }

    fn forward(&self, input: &Mat) -> FwResult<ConformerConvolutionTrace> {
        let frames = input.rows;
        let gated = self.pointwise1.forward(input)?;
        let glu_values = frames
            .checked_mul(SORTFORMER_ENCODER_WIDTH)
            .ok_or_else(|| reference_error("convolution_shape", "GLU size overflows"))?;
        let mut glu = Vec::new();
        glu.try_reserve_exact(glu_values)
            .map_err(|_| reference_error("allocation", "GLU allocation failed"))?;
        for row in gated.data.chunks(SORTFORMER_ENCODER_WIDTH * 2) {
            for channel in 0..SORTFORMER_ENCODER_WIDTH {
                glu.push(row[channel] * sigmoid_f32(row[SORTFORMER_ENCODER_WIDTH + channel])?);
            }
        }
        let glu = Mat::from_vec(frames, SORTFORMER_ENCODER_WIDTH, glu);
        let mut depthwise = Mat::zeros(frames, SORTFORMER_ENCODER_WIDTH);
        for time in 0..frames {
            for channel in 0..SORTFORMER_ENCODER_WIDTH {
                let mut sum = self.depthwise_bias[channel];
                for kernel in 0..Self::KERNEL {
                    if let Some(source_time) = time
                        .checked_add(kernel)
                        .and_then(|value| value.checked_sub(Self::PADDING))
                        .filter(|&value| value < frames)
                    {
                        sum += glu.data[source_time * SORTFORMER_ENCODER_WIDTH + channel]
                            * self.depthwise_weight[channel * Self::KERNEL + kernel];
                    }
                }
                depthwise.data[time * SORTFORMER_ENCODER_WIDTH + channel] = sum;
            }
        }
        if depthwise.data.iter().any(|value| !value.is_finite()) {
            return Err(reference_error(
                "convolution_nonfinite",
                "depthwise convolution produced a non-finite value",
            ));
        }
        let mut depthwise_channel_major = Vec::new();
        depthwise_channel_major
            .try_reserve_exact(depthwise.data.len())
            .map_err(|_| reference_error("allocation", "depthwise trace allocation failed"))?;
        for channel in 0..SORTFORMER_ENCODER_WIDTH {
            for time in 0..frames {
                depthwise_channel_major
                    .push(depthwise.data[time * SORTFORMER_ENCODER_WIDTH + channel]);
            }
        }
        self.batch_norm.forward_in_place(&mut depthwise)?;
        swish_in_place(&mut depthwise.data)?;
        let output = self.pointwise2.forward(&depthwise)?;
        Ok(ConformerConvolutionTrace {
            depthwise_channel_major,
            output,
        })
    }
}

struct RelPositionAttention {
    query: Affine,
    key: Affine,
    value: Affine,
    output: Affine,
    position: Affine,
    pos_bias_u: Vec<f32>,
    pos_bias_v: Vec<f32>,
}

impl RelPositionAttention {
    const HEADS: usize = 8;
    const HEAD_WIDTH: usize = 64;

    fn load(
        package: &SafetensorsFile,
        prefix: &str,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let affine = |suffix: &str, use_bias: bool| {
            Affine::load(
                package,
                &format!("{prefix}.{suffix}"),
                SORTFORMER_ENCODER_WIDTH,
                SORTFORMER_ENCODER_WIDTH,
                use_bias,
                checkpoint,
            )
        };
        let pos_bias_u = load_model_f32(
            package,
            &format!("{prefix}.pos_bias_u"),
            &[Self::HEADS, Self::HEAD_WIDTH],
            checkpoint,
        )?;
        let pos_bias_v = load_model_f32(
            package,
            &format!("{prefix}.pos_bias_v"),
            &[Self::HEADS, Self::HEAD_WIDTH],
            checkpoint,
        )?;
        Ok(Self {
            query: affine("linear_q", true)?,
            key: affine("linear_k", true)?,
            value: affine("linear_v", true)?,
            output: affine("linear_out", true)?,
            position: affine("linear_pos", false)?,
            pos_bias_u,
            pos_bias_v,
        })
    }

    fn project_qkv(&self, input: &Mat) -> FwResult<(Mat, Mat, Mat)> {
        Ok((
            self.query.forward(input)?,
            self.key.forward(input)?,
            self.value.forward(input)?,
        ))
    }

    fn forward(&self, input: &Mat) -> FwResult<Mat> {
        let frames = input.rows;
        if frames == 0 || input.cols != SORTFORMER_ENCODER_WIDTH {
            return Err(reference_error(
                "attention_shape",
                "relative attention requires a non-empty [frames, 512] input",
            ));
        }
        let (query, key, value) = self.project_qkv(input)?;
        let position_rows = frames
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| reference_error("attention_shape", "position length overflows"))?;
        let position = relative_position_encoding(frames, SORTFORMER_ENCODER_WIDTH)?;
        let position = self.position.forward(&Mat::from_vec(
            position_rows,
            SORTFORMER_ENCODER_WIDTH,
            position,
        ))?;
        let score_values = Self::HEADS
            .checked_mul(frames)
            .and_then(|value| value.checked_mul(frames))
            .ok_or_else(|| reference_error("attention_shape", "attention score size overflows"))?;
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(score_values)
            .map_err(|_| reference_error("allocation", "attention score allocation failed"))?;
        scores.resize(score_values, 0.0);
        let head_values = frames
            .checked_mul(Self::HEAD_WIDTH)
            .ok_or_else(|| reference_error("attention_shape", "attention head size overflows"))?;
        let position_head_values = position_rows
            .checked_mul(Self::HEAD_WIDTH)
            .ok_or_else(|| reference_error("attention_shape", "position head size overflows"))?;
        let scale = (Self::HEAD_WIDTH as f32).sqrt();
        let mut context = Mat::zeros(frames, SORTFORMER_ENCODER_WIDTH);

        for head in 0..Self::HEADS {
            let mut query_u = Vec::new();
            query_u
                .try_reserve_exact(head_values)
                .map_err(|_| reference_error("allocation", "attention query allocation failed"))?;
            let mut query_v = Vec::new();
            query_v
                .try_reserve_exact(head_values)
                .map_err(|_| reference_error("allocation", "attention query allocation failed"))?;
            let mut key_head = Vec::new();
            key_head
                .try_reserve_exact(head_values)
                .map_err(|_| reference_error("allocation", "attention key allocation failed"))?;
            let mut value_head = Vec::new();
            value_head
                .try_reserve_exact(head_values)
                .map_err(|_| reference_error("allocation", "attention value allocation failed"))?;
            let mut position_head = Vec::new();
            position_head
                .try_reserve_exact(position_head_values)
                .map_err(|_| {
                    reference_error("allocation", "attention position allocation failed")
                })?;
            let head_offset = head * Self::HEAD_WIDTH;
            for frame in 0..frames {
                let base = frame * SORTFORMER_ENCODER_WIDTH + head_offset;
                for inner in 0..Self::HEAD_WIDTH {
                    let bias_index = head_offset + inner;
                    query_u.push(query.data[base + inner] + self.pos_bias_u[bias_index]);
                    query_v.push(query.data[base + inner] + self.pos_bias_v[bias_index]);
                    key_head.push(key.data[base + inner]);
                    value_head.push(value.data[base + inner]);
                }
            }
            for position_frame in 0..position_rows {
                let base = position_frame * SORTFORMER_ENCODER_WIDTH + head_offset;
                position_head.extend_from_slice(&position.data[base..base + Self::HEAD_WIDTH]);
            }

            let content = ft_kernel_cpu::matmul_rhs_transposed_contiguous_f32(
                frames,
                Self::HEAD_WIDTH,
                frames,
                &query_u,
                &key_head,
            )
            .map_err(|_| reference_error("attention_kernel", "content GEMM failed"))?;
            let relative = ft_kernel_cpu::matmul_rhs_transposed_contiguous_f32(
                frames,
                Self::HEAD_WIDTH,
                position_rows,
                &query_v,
                &position_head,
            )
            .map_err(|_| reference_error("attention_kernel", "position GEMM failed"))?;
            let head_score_offset = head * frames * frames;
            for query_frame in 0..frames {
                let output_row = &mut scores[head_score_offset + query_frame * frames
                    ..head_score_offset + (query_frame + 1) * frames];
                let content_row = &content[query_frame * frames..(query_frame + 1) * frames];
                let relative_row =
                    &relative[query_frame * position_rows..(query_frame + 1) * position_rows];
                for key_frame in 0..frames {
                    let relative_index = key_frame + frames - 1 - query_frame;
                    output_row[key_frame] =
                        (content_row[key_frame] + relative_row[relative_index]) / scale;
                }
                softmax_f32_in_place(output_row)?;
            }
            let probabilities = Mat::from_vec(
                frames,
                frames,
                scores[head_score_offset..head_score_offset + frames * frames].to_vec(),
            );
            let head_context = nn::matmul(
                &probabilities,
                &Mat::from_vec(frames, Self::HEAD_WIDTH, value_head),
            )
            .map_err(|_| reference_error("attention_kernel", "value GEMM failed"))?;
            for frame in 0..frames {
                let source =
                    &head_context.data[frame * Self::HEAD_WIDTH..(frame + 1) * Self::HEAD_WIDTH];
                let destination = &mut context.data[frame * SORTFORMER_ENCODER_WIDTH + head_offset
                    ..frame * SORTFORMER_ENCODER_WIDTH + head_offset + Self::HEAD_WIDTH];
                destination.copy_from_slice(source);
            }
        }
        if context.data.iter().any(|value| !value.is_finite()) {
            return Err(reference_error(
                "attention_output",
                "relative attention produced a non-finite context",
            ));
        }
        self.output.forward(&context)
    }

    #[cfg(test)]
    fn forward_scalar(&self, input: &Mat) -> FwResult<Mat> {
        let frames = input.rows;
        if frames == 0 || input.cols != SORTFORMER_ENCODER_WIDTH {
            return Err(reference_error(
                "attention_shape",
                "relative attention requires a non-empty [frames, 512] input",
            ));
        }
        let (query, key, value) = self.project_qkv(input)?;
        let position_rows = frames
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| reference_error("attention_shape", "position length overflows"))?;
        let position = relative_position_encoding(frames, SORTFORMER_ENCODER_WIDTH)?;
        let position = self.position.forward(&Mat::from_vec(
            position_rows,
            SORTFORMER_ENCODER_WIDTH,
            position,
        ))?;
        let score_values = Self::HEADS
            .checked_mul(frames)
            .and_then(|value| value.checked_mul(frames))
            .ok_or_else(|| reference_error("attention_shape", "attention score size overflows"))?;
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(score_values)
            .map_err(|_| reference_error("allocation", "attention score allocation failed"))?;
        scores.resize(score_values, 0.0);
        let scale = (Self::HEAD_WIDTH as f32).sqrt();
        for head in 0..Self::HEADS {
            for query_frame in 0..frames {
                let query_base = query_frame * SORTFORMER_ENCODER_WIDTH + head * Self::HEAD_WIDTH;
                for key_frame in 0..frames {
                    let key_base = key_frame * SORTFORMER_ENCODER_WIDTH + head * Self::HEAD_WIDTH;
                    let relative_index = key_frame + frames - 1 - query_frame;
                    let position_base =
                        relative_index * SORTFORMER_ENCODER_WIDTH + head * Self::HEAD_WIDTH;
                    let mut content_score = 0.0_f32;
                    let mut position_score = 0.0_f32;
                    for inner in 0..Self::HEAD_WIDTH {
                        let bias_index = head * Self::HEAD_WIDTH + inner;
                        content_score += (query.data[query_base + inner]
                            + self.pos_bias_u[bias_index])
                            * key.data[key_base + inner];
                        position_score += (query.data[query_base + inner]
                            + self.pos_bias_v[bias_index])
                            * position.data[position_base + inner];
                    }
                    scores[(head * frames + query_frame) * frames + key_frame] =
                        (content_score + position_score) / scale;
                }
                softmax_f32_in_place(
                    &mut scores[(head * frames + query_frame) * frames
                        ..(head * frames + query_frame + 1) * frames],
                )?;
            }
        }
        let context_values = frames
            .checked_mul(SORTFORMER_ENCODER_WIDTH)
            .ok_or_else(|| reference_error("attention_shape", "attention output size overflows"))?;
        let mut context = Vec::new();
        context
            .try_reserve_exact(context_values)
            .map_err(|_| reference_error("allocation", "attention output allocation failed"))?;
        context.resize(context_values, 0.0);
        for query_frame in 0..frames {
            for head in 0..Self::HEADS {
                for inner in 0..Self::HEAD_WIDTH {
                    let mut sum = 0.0_f32;
                    for key_frame in 0..frames {
                        let probability =
                            scores[(head * frames + query_frame) * frames + key_frame];
                        let value_index =
                            key_frame * SORTFORMER_ENCODER_WIDTH + head * Self::HEAD_WIDTH + inner;
                        sum += probability * value.data[value_index];
                    }
                    context[query_frame * SORTFORMER_ENCODER_WIDTH
                        + head * Self::HEAD_WIDTH
                        + inner] = sum;
                }
            }
        }
        self.output
            .forward(&Mat::from_vec(frames, SORTFORMER_ENCODER_WIDTH, context))
    }
}

struct FastConformerBlock {
    feed_forward1: ConformerFeedForward,
    norm_feed_forward1: LayerNorm,
    norm_self_att: LayerNorm,
    self_attention: RelPositionAttention,
    norm_conv: LayerNorm,
    convolution: ConformerConvolution,
    norm_feed_forward2: LayerNorm,
    feed_forward2: ConformerFeedForward,
    norm_out: LayerNorm,
}

struct FastConformerBlockTrace {
    feed_forward1: Mat,
    attention_query: Mat,
    attention_key: Mat,
    attention_value: Mat,
    attention: Mat,
    convolution_depthwise_channel_major: Vec<f32>,
    feed_forward2: Mat,
    output: Mat,
}

impl FastConformerBlock {
    fn load(
        package: &SafetensorsFile,
        layer: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        if layer >= SORTFORMER_FASTCONFORMER_LAYERS {
            return Err(reference_error(
                "fastconformer_layer",
                "FastConformer layer is out of range",
            ));
        }
        let prefix = format!("encoder.layers.{layer}");
        Ok(Self {
            feed_forward1: ConformerFeedForward::load(
                package,
                &format!("{prefix}.feed_forward1"),
                checkpoint,
            )?,
            norm_feed_forward1: LayerNorm::load(
                package,
                &format!("{prefix}.norm_feed_forward1"),
                checkpoint,
            )?,
            norm_self_att: LayerNorm::load(
                package,
                &format!("{prefix}.norm_self_att"),
                checkpoint,
            )?,
            self_attention: RelPositionAttention::load(
                package,
                &format!("{prefix}.self_attn"),
                checkpoint,
            )?,
            norm_conv: LayerNorm::load(package, &format!("{prefix}.norm_conv"), checkpoint)?,
            convolution: ConformerConvolution::load(
                package,
                &format!("{prefix}.conv"),
                checkpoint,
            )?,
            norm_feed_forward2: LayerNorm::load(
                package,
                &format!("{prefix}.norm_feed_forward2"),
                checkpoint,
            )?,
            feed_forward2: ConformerFeedForward::load(
                package,
                &format!("{prefix}.feed_forward2"),
                checkpoint,
            )?,
            norm_out: LayerNorm::load(package, &format!("{prefix}.norm_out"), checkpoint)?,
        })
    }

    fn forward_trace(&self, input: &Mat) -> FwResult<FastConformerBlockTrace> {
        let mut normalized = input.clone();
        self.norm_feed_forward1.forward_in_place(&mut normalized)?;
        let feed_forward1 = self.feed_forward1.forward(&normalized)?;
        let mut residual = input.clone();
        add_scaled_residual_in_place(&mut residual, &feed_forward1, 0.5)?;

        normalized = residual.clone();
        self.norm_self_att.forward_in_place(&mut normalized)?;
        let (attention_query, attention_key, attention_value) =
            self.self_attention.project_qkv(&normalized)?;
        let attention = self.self_attention.forward(&normalized)?;
        add_scaled_residual_in_place(&mut residual, &attention, 1.0)?;

        normalized = residual.clone();
        self.norm_conv.forward_in_place(&mut normalized)?;
        let convolution = self.convolution.forward(&normalized)?;
        add_scaled_residual_in_place(&mut residual, &convolution.output, 1.0)?;

        normalized = residual.clone();
        self.norm_feed_forward2.forward_in_place(&mut normalized)?;
        let feed_forward2 = self.feed_forward2.forward(&normalized)?;
        add_scaled_residual_in_place(&mut residual, &feed_forward2, 0.5)?;
        self.norm_out.forward_in_place(&mut residual)?;

        Ok(FastConformerBlockTrace {
            feed_forward1,
            attention_query,
            attention_key,
            attention_value,
            attention,
            convolution_depthwise_channel_major: convolution.depthwise_channel_major,
            feed_forward2,
            output: residual,
        })
    }

    fn forward(&self, input: &Mat) -> FwResult<Mat> {
        let mut normalized = input.clone();
        self.norm_feed_forward1.forward_in_place(&mut normalized)?;
        let feed_forward = self.feed_forward1.forward(&normalized)?;
        let mut residual = input.clone();
        add_scaled_residual_in_place(&mut residual, &feed_forward, 0.5)?;

        normalized.clone_from(&residual);
        self.norm_self_att.forward_in_place(&mut normalized)?;
        let attention = self.self_attention.forward(&normalized)?;
        add_scaled_residual_in_place(&mut residual, &attention, 1.0)?;

        normalized.clone_from(&residual);
        self.norm_conv.forward_in_place(&mut normalized)?;
        let convolution = self.convolution.forward(&normalized)?;
        add_scaled_residual_in_place(&mut residual, &convolution.output, 1.0)?;

        normalized.clone_from(&residual);
        self.norm_feed_forward2.forward_in_place(&mut normalized)?;
        let feed_forward = self.feed_forward2.forward(&normalized)?;
        add_scaled_residual_in_place(&mut residual, &feed_forward, 0.5)?;
        self.norm_out.forward_in_place(&mut residual)?;
        Ok(residual)
    }
}

struct TransformerAttention {
    query: Affine,
    key: Affine,
    value: Affine,
    output: Affine,
}

struct TransformerAttentionTrace {
    query: Mat,
    key: Mat,
    value: Mat,
    output: Mat,
}

impl TransformerAttention {
    fn load(
        package: &SafetensorsFile,
        prefix: &str,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let affine = |suffix: &str| {
            Affine::load(
                package,
                &format!("{prefix}.{suffix}"),
                SORTFORMER_TRANSFORMER_WIDTH,
                SORTFORMER_TRANSFORMER_WIDTH,
                true,
                checkpoint,
            )
        };
        Ok(Self {
            query: affine("query_net")?,
            key: affine("key_net")?,
            value: affine("value_net")?,
            output: affine("out_projection")?,
        })
    }

    fn forward_trace(&self, input: &Mat) -> FwResult<TransformerAttentionTrace> {
        if input.rows == 0 || input.cols != SORTFORMER_TRANSFORMER_WIDTH {
            return Err(reference_error(
                "transformer_attention_shape",
                "Transformer attention requires a non-empty [frames, 192] input",
            ));
        }
        let query = self.query.forward(input)?;
        let key = self.key.forward(input)?;
        let value = self.value.forward(input)?;
        let frames = input.rows;
        let head_values = frames
            .checked_mul(SORTFORMER_TRANSFORMER_HEAD_WIDTH)
            .ok_or_else(|| {
                reference_error(
                    "transformer_attention_shape",
                    "attention head size overflows",
                )
            })?;
        let scale = (SORTFORMER_TRANSFORMER_HEAD_WIDTH as f32).sqrt().sqrt();
        let mut context = Mat::zeros(frames, SORTFORMER_TRANSFORMER_WIDTH);
        for head in 0..SORTFORMER_TRANSFORMER_HEADS {
            let mut query_head = Vec::new();
            query_head.try_reserve_exact(head_values).map_err(|_| {
                reference_error("allocation", "Transformer query allocation failed")
            })?;
            let mut key_head = Vec::new();
            key_head
                .try_reserve_exact(head_values)
                .map_err(|_| reference_error("allocation", "Transformer key allocation failed"))?;
            let mut value_head = Vec::new();
            value_head.try_reserve_exact(head_values).map_err(|_| {
                reference_error("allocation", "Transformer value allocation failed")
            })?;
            let head_offset = head * SORTFORMER_TRANSFORMER_HEAD_WIDTH;
            for frame in 0..frames {
                let base = frame * SORTFORMER_TRANSFORMER_WIDTH + head_offset;
                for inner in 0..SORTFORMER_TRANSFORMER_HEAD_WIDTH {
                    query_head.push(query.data[base + inner] / scale);
                    key_head.push(key.data[base + inner] / scale);
                    value_head.push(value.data[base + inner]);
                }
            }
            let mut scores = ft_kernel_cpu::matmul_rhs_transposed_contiguous_f32(
                frames,
                SORTFORMER_TRANSFORMER_HEAD_WIDTH,
                frames,
                &query_head,
                &key_head,
            )
            .map_err(|_| {
                reference_error(
                    "transformer_attention_kernel",
                    "attention score GEMM failed",
                )
            })?;
            for row in scores.chunks_mut(frames) {
                softmax_f32_in_place(row)?;
            }
            let head_context = nn::matmul(
                &Mat::from_vec(frames, frames, scores),
                &Mat::from_vec(frames, SORTFORMER_TRANSFORMER_HEAD_WIDTH, value_head),
            )
            .map_err(|_| {
                reference_error(
                    "transformer_attention_kernel",
                    "attention value GEMM failed",
                )
            })?;
            for frame in 0..frames {
                let source = &head_context.data[frame * SORTFORMER_TRANSFORMER_HEAD_WIDTH
                    ..(frame + 1) * SORTFORMER_TRANSFORMER_HEAD_WIDTH];
                let destination = &mut context.data[frame * SORTFORMER_TRANSFORMER_WIDTH
                    + head_offset
                    ..frame * SORTFORMER_TRANSFORMER_WIDTH
                        + head_offset
                        + SORTFORMER_TRANSFORMER_HEAD_WIDTH];
                destination.copy_from_slice(source);
            }
        }
        let output = self.output.forward(&context)?;
        Ok(TransformerAttentionTrace {
            query,
            key,
            value,
            output,
        })
    }
}

struct TransformerBlock {
    attention: TransformerAttention,
    layer_norm_1: LayerNorm,
    dense_in: Affine,
    dense_out: Affine,
    layer_norm_2: LayerNorm,
}

struct TransformerBlockTrace {
    attention_query: Mat,
    attention_key: Mat,
    attention_value: Mat,
    attention_output: Mat,
    feed_forward_inner: Mat,
    feed_forward_output: Mat,
    output: Mat,
}

impl TransformerBlock {
    fn load(
        package: &SafetensorsFile,
        layer: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        if layer >= SORTFORMER_TRANSFORMER_LAYERS {
            return Err(reference_error(
                "transformer_layer",
                "Transformer layer is out of range",
            ));
        }
        let prefix = format!("transformer_encoder.layers.{layer}");
        Ok(Self {
            attention: TransformerAttention::load(
                package,
                &format!("{prefix}.first_sub_layer"),
                checkpoint,
            )?,
            layer_norm_1: LayerNorm::load_width(
                package,
                &format!("{prefix}.layer_norm_1"),
                SORTFORMER_TRANSFORMER_WIDTH,
                checkpoint,
            )?,
            dense_in: Affine::load(
                package,
                &format!("{prefix}.second_sub_layer.dense_in"),
                SORTFORMER_TRANSFORMER_WIDTH,
                SORTFORMER_TRANSFORMER_INNER_WIDTH,
                true,
                checkpoint,
            )?,
            dense_out: Affine::load(
                package,
                &format!("{prefix}.second_sub_layer.dense_out"),
                SORTFORMER_TRANSFORMER_INNER_WIDTH,
                SORTFORMER_TRANSFORMER_WIDTH,
                true,
                checkpoint,
            )?,
            layer_norm_2: LayerNorm::load_width(
                package,
                &format!("{prefix}.layer_norm_2"),
                SORTFORMER_TRANSFORMER_WIDTH,
                checkpoint,
            )?,
        })
    }

    fn forward_trace(&self, input: &Mat) -> FwResult<TransformerBlockTrace> {
        let attention = self.attention.forward_trace(input)?;
        let mut attention_residual = input.clone();
        add_scaled_residual_in_place(&mut attention_residual, &attention.output, 1.0)?;
        // The pinned oracle's CPU forward hook aliases out_projection's output.
        // NeMo immediately applies `self_attn_output += encoder_query`, so the
        // authenticated `attention_output` seam is this post-residual,
        // pre-LayerNorm state rather than the raw affine output.
        let captured_attention_output = attention_residual.clone();
        self.layer_norm_1
            .forward_in_place(&mut attention_residual)?;

        let feed_forward_inner = self.dense_in.forward(&attention_residual)?;
        let mut feed_forward_activated = feed_forward_inner.clone();
        relu_in_place(&mut feed_forward_activated.data);
        let feed_forward_output = self.dense_out.forward(&feed_forward_activated)?;
        let mut output = attention_residual;
        add_scaled_residual_in_place(&mut output, &feed_forward_output, 1.0)?;
        // The dense-out hook has the same aliasing behavior: NeMo mutates its
        // result with the residual before LayerNorm2.
        let captured_feed_forward_output = output.clone();
        self.layer_norm_2.forward_in_place(&mut output)?;

        Ok(TransformerBlockTrace {
            attention_query: attention.query,
            attention_key: attention.key,
            attention_value: attention.value,
            attention_output: captured_attention_output,
            feed_forward_inner,
            feed_forward_output: captured_feed_forward_output,
            output,
        })
    }

    fn forward(&self, input: &Mat) -> FwResult<Mat> {
        let attention = self.attention.forward_trace(input)?.output;
        let mut output = input.clone();
        add_scaled_residual_in_place(&mut output, &attention, 1.0)?;
        self.layer_norm_1.forward_in_place(&mut output)?;

        let mut feed_forward = self.dense_in.forward(&output)?;
        relu_in_place(&mut feed_forward.data);
        let feed_forward = self.dense_out.forward(&feed_forward)?;
        add_scaled_residual_in_place(&mut output, &feed_forward, 1.0)?;
        self.layer_norm_2.forward_in_place(&mut output)?;
        Ok(output)
    }
}

struct SortformerSubsampler {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    projection_weight_t: Mat,
    projection_bias: Vec<f32>,
}

struct SortformerSubsamplingTrace {
    conv0: Tensor4,
    conv2: Tensor4,
    conv3: Tensor4,
    conv5: Tensor4,
    conv6: Tensor4,
    projection: Mat,
}

impl SortformerSubsampler {
    fn load(
        package: &SafetensorsFile,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let conv0 = Conv2d::load(
            package,
            "encoder.pre_encode.conv.0",
            [SORTFORMER_SUBSAMPLING_CHANNELS, 1, 3, 3],
            1,
            checkpoint,
        )?;
        let conv2 = Conv2d::load(
            package,
            "encoder.pre_encode.conv.2",
            [SORTFORMER_SUBSAMPLING_CHANNELS, 1, 3, 3],
            SORTFORMER_SUBSAMPLING_CHANNELS,
            checkpoint,
        )?;
        let conv3 = Conv2d::load(
            package,
            "encoder.pre_encode.conv.3",
            [
                SORTFORMER_SUBSAMPLING_CHANNELS,
                SORTFORMER_SUBSAMPLING_CHANNELS,
                1,
                1,
            ],
            1,
            checkpoint,
        )?;
        let conv5 = Conv2d::load(
            package,
            "encoder.pre_encode.conv.5",
            [SORTFORMER_SUBSAMPLING_CHANNELS, 1, 3, 3],
            SORTFORMER_SUBSAMPLING_CHANNELS,
            checkpoint,
        )?;
        let conv6 = Conv2d::load(
            package,
            "encoder.pre_encode.conv.6",
            [
                SORTFORMER_SUBSAMPLING_CHANNELS,
                SORTFORMER_SUBSAMPLING_CHANNELS,
                1,
                1,
            ],
            1,
            checkpoint,
        )?;
        let projection_weight = load_model_f32(
            package,
            "encoder.pre_encode.out.weight",
            &[SORTFORMER_ENCODER_WIDTH, 4096],
            checkpoint,
        )?;
        let projection_weight_t = transpose_affine_weight(
            "encoder.pre_encode.out.weight",
            SORTFORMER_ENCODER_WIDTH,
            4096,
            projection_weight,
        )?;
        let projection_bias = load_model_f32(
            package,
            "encoder.pre_encode.out.bias",
            &[SORTFORMER_ENCODER_WIDTH],
            checkpoint,
        )?;
        reference_checkpoint(checkpoint)?;
        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            projection_weight_t,
            projection_bias,
        })
    }

    fn forward(
        &self,
        time_major_features: &[f32],
        feature_frames: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<SortformerSubsamplingTrace> {
        let expected_values = feature_frames
            .checked_mul(SORTFORMER_MEL_BINS)
            .ok_or_else(|| reference_error("feature_shape", "feature value count overflows"))?;
        if feature_frames == 0 || time_major_features.len() != expected_values {
            return Err(reference_error(
                "feature_shape",
                "subsampler input must be a non-empty time-major [frames, 128] tensor",
            ));
        }
        if time_major_features.iter().any(|value| !value.is_finite()) {
            return Err(reference_error(
                "feature_nonfinite",
                "subsampler input contains a non-finite value",
            ));
        }
        let input = Tensor4::from_vec(
            1,
            1,
            feature_frames,
            SORTFORMER_MEL_BINS,
            time_major_features.to_vec(),
        )?;
        let conv0 = self.conv0.forward(&input, 2, 1, checkpoint)?;
        let mut stage = conv0.clone();
        relu_in_place(&mut stage.data);
        let conv2 = self.conv2.forward(&stage, 2, 1, checkpoint)?;
        let conv3 = self.conv3.forward(&conv2, 1, 0, checkpoint)?;
        stage = conv3.clone();
        relu_in_place(&mut stage.data);
        let conv5 = self.conv5.forward(&stage, 2, 1, checkpoint)?;
        let conv6 = self.conv6.forward(&conv5, 1, 0, checkpoint)?;
        stage = conv6.clone();
        relu_in_place(&mut stage.data);

        let projection = self.project_stage(&stage, checkpoint)?;
        Ok(SortformerSubsamplingTrace {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            projection,
        })
    }

    fn forward_output(
        &self,
        time_major_features: &[f32],
        feature_frames: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Mat> {
        let expected_values = feature_frames
            .checked_mul(SORTFORMER_MEL_BINS)
            .ok_or_else(|| reference_error("feature_shape", "feature value count overflows"))?;
        if feature_frames == 0 || time_major_features.len() != expected_values {
            return Err(reference_error(
                "feature_shape",
                "subsampler input must be a non-empty time-major [frames, 128] tensor",
            ));
        }
        if time_major_features.iter().any(|value| !value.is_finite()) {
            return Err(reference_error(
                "feature_nonfinite",
                "subsampler input contains a non-finite value",
            ));
        }
        let mut stage = Tensor4::from_vec(
            1,
            1,
            feature_frames,
            SORTFORMER_MEL_BINS,
            time_major_features.to_vec(),
        )?;
        stage = self.conv0.forward(&stage, 2, 1, checkpoint)?;
        relu_in_place(&mut stage.data);
        stage = self.conv2.forward(&stage, 2, 1, checkpoint)?;
        stage = self.conv3.forward(&stage, 1, 0, checkpoint)?;
        relu_in_place(&mut stage.data);
        stage = self.conv5.forward(&stage, 2, 1, checkpoint)?;
        stage = self.conv6.forward(&stage, 1, 0, checkpoint)?;
        relu_in_place(&mut stage.data);
        self.project_stage(&stage, checkpoint)
    }

    fn project_stage(
        &self,
        stage: &Tensor4,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Mat> {
        if stage.n != 1 || stage.c != SORTFORMER_SUBSAMPLING_CHANNELS || stage.w != 16 {
            return Err(reference_error(
                "subsampling_geometry",
                "three stride-2 stages did not produce [1, 256, frames, 16]",
            ));
        }
        let flattened_cols = stage
            .c
            .checked_mul(stage.w)
            .ok_or_else(|| reference_error("subsampling_geometry", "flatten width overflows"))?;
        let flattened_values = stage
            .h
            .checked_mul(flattened_cols)
            .ok_or_else(|| reference_error("subsampling_geometry", "flatten size overflows"))?;
        let mut flattened = Vec::new();
        flattened.try_reserve_exact(flattened_values).map_err(|_| {
            reference_error("allocation", "subsampling flatten buffer allocation failed")
        })?;
        for time in 0..stage.h {
            reference_checkpoint(checkpoint)?;
            for channel in 0..stage.c {
                let start = stage.offset(0, channel, time, 0)?;
                flattened.extend_from_slice(&stage.data[start..start + stage.w]);
            }
        }
        let flattened = Mat::from_vec(stage.h, flattened_cols, flattened);
        let projection = nn::matmul_bias(
            &flattened,
            &self.projection_weight_t,
            Some(&self.projection_bias),
        )
        .map_err(|_| {
            reference_error("projection", "FrankenTorch projection rejected L2 geometry")
        })?;
        if projection.data.iter().any(|value| !value.is_finite()) {
            return Err(reference_error(
                "output_nonfinite",
                "L2 projection produced a non-finite value",
            ));
        }
        reference_checkpoint(checkpoint)?;
        Ok(projection)
    }
}

#[derive(Clone)]
struct Tensor4 {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    data: Vec<f32>,
}

impl Tensor4 {
    fn from_vec(n: usize, c: usize, h: usize, w: usize, data: Vec<f32>) -> FwResult<Self> {
        let expected = checked_product(&[n, c, h, w], "tensor4 shape")?;
        if data.len() != expected {
            return Err(reference_error(
                "tensor_shape",
                "four-dimensional tensor payload length does not match its shape",
            ));
        }
        Ok(Self { n, c, h, w, data })
    }

    fn zeros(n: usize, c: usize, h: usize, w: usize) -> FwResult<Self> {
        let values = checked_product(&[n, c, h, w], "tensor4 output shape")?;
        let mut data = Vec::new();
        data.try_reserve_exact(values)
            .map_err(|_| reference_error("allocation", "tensor4 allocation failed"))?;
        data.resize(values, 0.0);
        Ok(Self { n, c, h, w, data })
    }

    fn offset(&self, n: usize, c: usize, h: usize, w: usize) -> FwResult<usize> {
        if n >= self.n || c >= self.c || h >= self.h || w >= self.w {
            return Err(reference_error(
                "tensor_index",
                "tensor4 index is out of bounds",
            ));
        }
        Ok(((n * self.c + c) * self.h + h) * self.w + w)
    }

    fn shape(&self) -> Vec<usize> {
        vec![self.n, self.c, self.h, self.w]
    }
}

struct Conv2d {
    out_channels: usize,
    in_per_group: usize,
    kernel_h: usize,
    kernel_w: usize,
    groups: usize,
    weight: Vec<f32>,
    dense_weight_t: Option<Mat>,
    bias: Vec<f32>,
}

impl Conv2d {
    fn load(
        package: &SafetensorsFile,
        prefix: &str,
        shape: [usize; 4],
        groups: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Self> {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        let weight = load_model_f32(package, &weight_name, &shape, checkpoint)?;
        let bias = load_model_f32(package, &bias_name, &[shape[0]], checkpoint)?;
        Self::new(shape, groups, weight, bias)
    }

    fn new(shape: [usize; 4], groups: usize, weight: Vec<f32>, bias: Vec<f32>) -> FwResult<Self> {
        if groups == 0 || shape.contains(&0) || shape[0] % groups != 0 {
            return Err(reference_error(
                "conv_contract",
                "invalid Conv2d group geometry",
            ));
        }
        if weight.len() != checked_product(&shape, "conv weight shape")? || bias.len() != shape[0] {
            return Err(reference_error(
                "conv_contract",
                "Conv2d payload shape mismatch",
            ));
        }
        let dense_weight_t = if groups == 1 {
            let input_width = shape[1]
                .checked_mul(shape[2])
                .and_then(|value| value.checked_mul(shape[3]))
                .ok_or_else(|| {
                    reference_error("conv_contract", "dense Conv2d input width overflows")
                })?;
            Some(transpose_affine_weight(
                "Conv2d.weight",
                shape[0],
                input_width,
                weight.clone(),
            )?)
        } else {
            None
        };
        Ok(Self {
            out_channels: shape[0],
            in_per_group: shape[1],
            kernel_h: shape[2],
            kernel_w: shape[3],
            groups,
            weight,
            dense_weight_t,
            bias,
        })
    }

    fn forward(
        &self,
        input: &Tensor4,
        stride: usize,
        padding: usize,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Tensor4> {
        if stride == 0
            || input.c
                != self.in_per_group.checked_mul(self.groups).ok_or_else(|| {
                    reference_error("conv_contract", "Conv2d input channel count overflows")
                })?
        {
            return Err(reference_error(
                "conv_contract",
                "Conv2d input geometry mismatch",
            ));
        }
        let padded_h = input
            .h
            .checked_add(padding.saturating_mul(2))
            .ok_or_else(|| reference_error("conv_geometry", "Conv2d padded height overflows"))?;
        let padded_w = input
            .w
            .checked_add(padding.saturating_mul(2))
            .ok_or_else(|| reference_error("conv_geometry", "Conv2d padded width overflows"))?;
        if padded_h < self.kernel_h || padded_w < self.kernel_w {
            return Err(reference_error(
                "conv_geometry",
                "Conv2d kernel exceeds padded input",
            ));
        }
        let output_h = (padded_h - self.kernel_h) / stride + 1;
        let output_w = (padded_w - self.kernel_w) / stride + 1;
        if let Some(weight_t) = &self.dense_weight_t {
            return self.forward_dense(
                input, stride, padding, output_h, output_w, weight_t, checkpoint,
            );
        }
        let mut output = Tensor4::zeros(input.n, self.out_channels, output_h, output_w)?;
        let out_per_group = self.out_channels / self.groups;

        for batch in 0..input.n {
            for out_channel in 0..self.out_channels {
                reference_checkpoint(checkpoint)?;
                let group = out_channel / out_per_group;
                let input_channel_start = group * self.in_per_group;
                for out_h in 0..output_h {
                    for out_w in 0..output_w {
                        let mut sum = self.bias[out_channel];
                        for local_input_channel in 0..self.in_per_group {
                            let input_channel = input_channel_start + local_input_channel;
                            for kernel_h in 0..self.kernel_h {
                                let padded_input_h = out_h * stride + kernel_h;
                                let Some(input_h) = padded_input_h.checked_sub(padding) else {
                                    continue;
                                };
                                if input_h >= input.h {
                                    continue;
                                }
                                for kernel_w in 0..self.kernel_w {
                                    let padded_input_w = out_w * stride + kernel_w;
                                    let Some(input_w) = padded_input_w.checked_sub(padding) else {
                                        continue;
                                    };
                                    if input_w >= input.w {
                                        continue;
                                    }
                                    let input_index =
                                        input.offset(batch, input_channel, input_h, input_w)?;
                                    let weight_index = (((out_channel * self.in_per_group
                                        + local_input_channel)
                                        * self.kernel_h
                                        + kernel_h)
                                        * self.kernel_w)
                                        + kernel_w;
                                    sum += input.data[input_index] * self.weight[weight_index];
                                }
                            }
                        }
                        let output_index = output.offset(batch, out_channel, out_h, out_w)?;
                        output.data[output_index] = sum;
                    }
                }
            }
        }
        if output.data.iter().any(|value| !value.is_finite()) {
            return Err(reference_error(
                "output_nonfinite",
                "Conv2d produced a non-finite value",
            ));
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_dense(
        &self,
        input: &Tensor4,
        stride: usize,
        padding: usize,
        output_h: usize,
        output_w: usize,
        weight_t: &Mat,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<Tensor4> {
        let rows = checked_product(&[input.n, output_h, output_w], "Conv2d im2col rows")?;
        let columns = checked_product(
            &[input.c, self.kernel_h, self.kernel_w],
            "Conv2d im2col columns",
        )?;
        let values = rows
            .checked_mul(columns)
            .ok_or_else(|| reference_error("conv_geometry", "Conv2d im2col size overflows"))?;
        let mut lowered = Vec::new();
        lowered
            .try_reserve_exact(values)
            .map_err(|_| reference_error("allocation", "Conv2d im2col allocation failed"))?;
        for batch in 0..input.n {
            for out_h in 0..output_h {
                reference_checkpoint(checkpoint)?;
                for out_w in 0..output_w {
                    for input_channel in 0..input.c {
                        for kernel_h in 0..self.kernel_h {
                            let padded_input_h = out_h * stride + kernel_h;
                            for kernel_w in 0..self.kernel_w {
                                let padded_input_w = out_w * stride + kernel_w;
                                let value = match (
                                    padded_input_h.checked_sub(padding),
                                    padded_input_w.checked_sub(padding),
                                ) {
                                    (Some(input_h), Some(input_w))
                                        if input_h < input.h && input_w < input.w =>
                                    {
                                        let index =
                                            input.offset(batch, input_channel, input_h, input_w)?;
                                        input.data[index]
                                    }
                                    _ => 0.0,
                                };
                                lowered.push(value);
                            }
                        }
                    }
                }
            }
        }
        if lowered.len() != values {
            return Err(reference_error(
                "conv_geometry",
                "Conv2d im2col produced an invalid value count",
            ));
        }
        let lowered = Mat::from_vec(rows, columns, lowered);
        let dense = nn::matmul_bias(&lowered, weight_t, Some(&self.bias)).map_err(|_| {
            reference_error("conv_kernel", "FrankenTorch rejected dense Conv2d lowering")
        })?;
        if dense.rows != rows || dense.cols != self.out_channels {
            return Err(reference_error(
                "conv_kernel",
                "FrankenTorch returned invalid dense Conv2d geometry",
            ));
        }
        let mut output = Tensor4::zeros(input.n, self.out_channels, output_h, output_w)?;
        for batch in 0..input.n {
            for out_h in 0..output_h {
                reference_checkpoint(checkpoint)?;
                for out_w in 0..output_w {
                    let dense_row = (batch * output_h + out_h) * output_w + out_w;
                    for out_channel in 0..self.out_channels {
                        let output_index = output.offset(batch, out_channel, out_h, out_w)?;
                        output.data[output_index] =
                            dense.data[dense_row * self.out_channels + out_channel];
                    }
                }
            }
        }
        if output.data.iter().any(|value| !value.is_finite()) {
            return Err(reference_error(
                "output_nonfinite",
                "dense Conv2d produced a non-finite value",
            ));
        }
        Ok(output)
    }
}

fn load_model_f32(
    package: &SafetensorsFile,
    name: &str,
    expected_shape: &[usize],
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<Vec<f32>> {
    reference_checkpoint(checkpoint)?;
    if package
        .dtype_name(name)
        .map_err(|_| reference_error("tensor_missing", "required model tensor is absent"))?
        != "F32"
    {
        return Err(reference_error(
            "tensor_dtype",
            "Sortformer reference weights must use exact F32 storage",
        ));
    }
    if package
        .shape(name)
        .map_err(|_| reference_error("tensor_missing", "required model tensor is absent"))?
        != expected_shape
    {
        return Err(reference_error(
            "tensor_shape",
            "Sortformer reference tensor shape changed",
        ));
    }
    let (_, values) = package
        .tensor_f32(name)
        .map_err(|_| reference_error("tensor_payload", "model tensor could not be decoded"))?;
    for chunk in values.chunks(4096) {
        reference_checkpoint(checkpoint)?;
        if chunk.iter().any(|value| !value.is_finite()) {
            return Err(reference_error(
                "tensor_nonfinite",
                "Sortformer reference tensor contains a non-finite value",
            ));
        }
    }
    Ok(values)
}

fn transpose_affine_weight(
    name: &str,
    output: usize,
    input: usize,
    natural: Vec<f32>,
) -> FwResult<Mat> {
    if natural.len()
        != output.checked_mul(input).ok_or_else(|| {
            reference_error("tensor_shape", "affine weight element count overflows")
        })?
    {
        return Err(reference_error(
            "tensor_shape",
            &format!("affine weight {name} has an invalid payload length"),
        ));
    }
    let mut transposed = Vec::new();
    transposed
        .try_reserve_exact(natural.len())
        .map_err(|_| reference_error("allocation", "affine transpose allocation failed"))?;
    transposed.resize(natural.len(), 0.0);
    for out in 0..output {
        for inner in 0..input {
            transposed[inner * output + out] = natural[out * input + inner];
        }
    }
    Ok(Mat::from_vec(input, output, transposed))
}

fn checked_product(dimensions: &[usize], label: &str) -> FwResult<usize> {
    dimensions.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(dimension).ok_or_else(|| {
            reference_error("tensor_shape", &format!("{label} element count overflows"))
        })
    })
}

fn validate_streaming_state(state: &SortformerStreamingState) -> FwResult<()> {
    let cache_values = state
        .speaker_cache
        .rows
        .checked_mul(SORTFORMER_ENCODER_WIDTH);
    let fifo_values = state.fifo.rows.checked_mul(SORTFORMER_ENCODER_WIDTH);
    if state.speaker_cache.cols != SORTFORMER_ENCODER_WIDTH
        || cache_values != Some(state.speaker_cache.data.len())
        || state.fifo.cols != SORTFORMER_ENCODER_WIDTH
        || fifo_values != Some(state.fifo.data.len())
        || state.fifo.rows > SORTFORMER_FIFO_FRAMES
        || state.mean_silence_embedding.len() != SORTFORMER_ENCODER_WIDTH
        || state.silence_frames < 0
        || state
            .speaker_cache
            .data
            .iter()
            .chain(&state.fifo.data)
            .chain(&state.mean_silence_embedding)
            .any(|value| !value.is_finite())
    {
        return Err(reference_error(
            "l6_state",
            "L6 streaming state has invalid embedding geometry or values",
        ));
    }
    if let Some(predictions) = &state.speaker_cache_predictions
        && (predictions.rows != state.speaker_cache.rows
            || predictions.cols != SORTFORMER_SPEAKER_LANES
            || predictions.rows.checked_mul(predictions.cols) != Some(predictions.data.len())
            || predictions.data.iter().any(|value| !value.is_finite()))
    {
        return Err(reference_error(
            "l6_state",
            "L6 speaker-cache predictions differ from cache geometry",
        ));
    }
    if let Some(predictions) = &state.fifo_predictions
        && (predictions.rows != state.fifo.rows
            || predictions.cols != SORTFORMER_SPEAKER_LANES
            || predictions.rows.checked_mul(predictions.cols) != Some(predictions.data.len())
            || predictions.data.iter().any(|value| !value.is_finite()))
    {
        return Err(reference_error(
            "l6_state",
            "L6 FIFO predictions differ from FIFO geometry",
        ));
    }
    if let Some(permutation) = state.speaker_permutation {
        let mut seen = [false; SORTFORMER_SPEAKER_LANES];
        for lane in permutation {
            let slot = seen.get_mut(lane).ok_or_else(|| {
                reference_error("l6_state", "L6 speaker permutation is out of range")
            })?;
            if *slot {
                return Err(reference_error(
                    "l6_state",
                    "L6 speaker permutation contains a duplicate lane",
                ));
            }
            *slot = true;
        }
    }
    Ok(())
}

fn slice_rows(values: &[f32], width: usize, start: usize, end: usize) -> FwResult<&[f32]> {
    if width == 0 || start > end {
        return Err(reference_error("l6_shape", "L6 row slice is invalid"));
    }
    let start_value = start
        .checked_mul(width)
        .ok_or_else(|| reference_error("l6_shape", "L6 row offset overflows"))?;
    let end_value = end
        .checked_mul(width)
        .ok_or_else(|| reference_error("l6_shape", "L6 row extent overflows"))?;
    values
        .get(start_value..end_value)
        .ok_or_else(|| reference_error("l6_shape", "L6 row slice exceeds its tensor"))
}

fn slice_mat_rows(input: &Mat, start: usize, end: usize) -> FwResult<Mat> {
    Ok(Mat::from_vec(
        end.checked_sub(start)
            .ok_or_else(|| reference_error("l6_shape", "L6 matrix slice is reversed"))?,
        input.cols,
        slice_rows(&input.data, input.cols, start, end)?.to_vec(),
    ))
}

fn append_mat_rows(destination: &mut Mat, source: &Mat) -> FwResult<()> {
    if destination.cols != source.cols
        || destination.rows.checked_mul(destination.cols) != Some(destination.data.len())
        || source.rows.checked_mul(source.cols) != Some(source.data.len())
    {
        return Err(reference_error(
            "l6_shape",
            "L6 matrix append has incompatible geometry",
        ));
    }
    let rows = destination
        .rows
        .checked_add(source.rows)
        .ok_or_else(|| reference_error("l6_shape", "L6 appended row count overflows"))?;
    destination
        .data
        .try_reserve_exact(source.data.len())
        .map_err(|_| reference_error("allocation", "L6 matrix append allocation failed"))?;
    destination.data.extend_from_slice(&source.data);
    destination.rows = rows;
    Ok(())
}

fn inverse_permute_speaker_probabilities(
    probabilities: &Mat,
    permutation: [usize; SORTFORMER_SPEAKER_LANES],
) -> FwResult<Mat> {
    let mut inverse = [0usize; SORTFORMER_SPEAKER_LANES];
    for (output_lane, &input_lane) in permutation.iter().enumerate() {
        let slot = inverse
            .get_mut(input_lane)
            .ok_or_else(|| reference_error("l6_state", "L6 speaker permutation is out of range"))?;
        *slot = output_lane;
    }
    let mut data = Vec::new();
    data.try_reserve_exact(probabilities.data.len())
        .map_err(|_| reference_error("allocation", "L6 permutation allocation failed"))?;
    for row in probabilities.data.chunks(SORTFORMER_SPEAKER_LANES) {
        for lane in inverse {
            data.push(row[lane]);
        }
    }
    Ok(Mat::from_vec(
        probabilities.rows,
        SORTFORMER_SPEAKER_LANES,
        data,
    ))
}

fn update_silence_profile(
    state: &mut SortformerStreamingState,
    embeddings: &Mat,
    predictions: &Mat,
) -> FwResult<()> {
    if embeddings.rows != predictions.rows
        || embeddings.cols != SORTFORMER_ENCODER_WIDTH
        || predictions.cols != SORTFORMER_SPEAKER_LANES
    {
        return Err(reference_error(
            "l6_silence",
            "L6 silence inputs have incompatible geometry",
        ));
    }
    let mut new_count = 0_i64;
    let mut new_sum = vec![0.0_f32; SORTFORMER_ENCODER_WIDTH];
    for frame in 0..embeddings.rows {
        let prediction_row = &predictions.data
            [frame * SORTFORMER_SPEAKER_LANES..(frame + 1) * SORTFORMER_SPEAKER_LANES];
        if prediction_row.iter().copied().sum::<f32>() < SORTFORMER_SILENCE_THRESHOLD {
            new_count = new_count
                .checked_add(1)
                .ok_or_else(|| reference_error("l6_silence", "silence count overflows"))?;
            let embedding_row = &embeddings.data
                [frame * SORTFORMER_ENCODER_WIDTH..(frame + 1) * SORTFORMER_ENCODER_WIDTH];
            for (sum, value) in new_sum.iter_mut().zip(embedding_row) {
                *sum += *value;
            }
        }
    }
    if new_count == 0 {
        return Ok(());
    }
    let total_count = state
        .silence_frames
        .checked_add(new_count)
        .ok_or_else(|| reference_error("l6_silence", "silence count overflows"))?;
    let old_count = state.silence_frames as f32;
    let denominator = total_count as f32;
    for (mean, added) in state.mean_silence_embedding.iter_mut().zip(new_sum) {
        *mean = (*mean * old_count + added) / denominator;
        if !mean.is_finite() {
            return Err(reference_error(
                "l6_silence",
                "L6 silence mean became non-finite",
            ));
        }
    }
    state.silence_frames = total_count;
    Ok(())
}

fn ranked_speaker_indices(
    scores: &[f32],
    frames: usize,
    speaker: usize,
    count: usize,
) -> FwResult<Vec<usize>> {
    let mut indices = (0..frames).collect::<Vec<_>>();
    indices.sort_unstable_by(|left, right| {
        let left_score = scores[left * SORTFORMER_SPEAKER_LANES + speaker];
        let right_score = scores[right * SORTFORMER_SPEAKER_LANES + speaker];
        right_score
            .total_cmp(&left_score)
            .then_with(|| left.cmp(right))
    });
    if indices.len() != frames {
        return Err(reference_error("l6_compression", "L6 ranking lost a frame"));
    }
    let keep = count.min(frames);
    indices.truncate(keep);
    Ok(indices)
}

fn boost_top_scores(scores: &mut [f32], frames: usize, count: usize, scale: f32) -> FwResult<()> {
    let adjustment = scale * 0.5_f32.ln();
    for speaker in 0..SORTFORMER_SPEAKER_LANES {
        let indices = ranked_speaker_indices(scores, frames, speaker, count)?;
        for frame in indices {
            scores[frame * SORTFORMER_SPEAKER_LANES + speaker] -= adjustment;
        }
    }
    Ok(())
}

fn compress_speaker_cache(state: &mut SortformerStreamingState) -> FwResult<()> {
    let frames = state.speaker_cache.rows;
    let predictions = state
        .speaker_cache_predictions
        .as_ref()
        .ok_or_else(|| reference_error("l6_compression", "L6 cache predictions are absent"))?;
    if frames <= SORTFORMER_SPEAKER_CACHE_FRAMES || predictions.rows != frames {
        return Err(reference_error(
            "l6_compression",
            "L6 compression requires an overfull aligned cache",
        ));
    }
    let values = frames
        .checked_mul(SORTFORMER_SPEAKER_LANES)
        .ok_or_else(|| reference_error("l6_compression", "L6 score size overflows"))?;
    let mut scores = Vec::new();
    scores
        .try_reserve_exact(values)
        .map_err(|_| reference_error("allocation", "L6 score allocation failed"))?;
    for row in predictions.data.chunks(SORTFORMER_SPEAKER_LANES) {
        let mut log_inactive_sum = 0.0_f32;
        let mut log_inactive = [0.0_f32; SORTFORMER_SPEAKER_LANES];
        for speaker in 0..SORTFORMER_SPEAKER_LANES {
            log_inactive[speaker] = (1.0 - row[speaker])
                .max(SORTFORMER_PREDICTION_SCORE_THRESHOLD)
                .ln();
            log_inactive_sum += log_inactive[speaker];
        }
        for speaker in 0..SORTFORMER_SPEAKER_LANES {
            scores.push(
                row[speaker].max(SORTFORMER_PREDICTION_SCORE_THRESHOLD).ln()
                    - log_inactive[speaker]
                    + log_inactive_sum
                    - 0.5_f32.ln(),
            );
        }
    }
    let cache_frames_per_speaker = SORTFORMER_SPEAKER_CACHE_FRAMES / SORTFORMER_SPEAKER_LANES
        - SORTFORMER_SILENCE_FRAMES_PER_SPEAKER;
    let minimum_positive =
        (cache_frames_per_speaker as f32 * SORTFORMER_MIN_POSITIVE_SCORE_RATE).floor() as usize;
    for speaker in 0..SORTFORMER_SPEAKER_LANES {
        let positive_count = (0..frames)
            .filter(|&frame| {
                predictions.data[frame * SORTFORMER_SPEAKER_LANES + speaker] > 0.5
                    && scores[frame * SORTFORMER_SPEAKER_LANES + speaker] > 0.0
            })
            .count();
        for frame in 0..frames {
            let index = frame * SORTFORMER_SPEAKER_LANES + speaker;
            let is_speech = predictions.data[index] > 0.5;
            if !is_speech || (positive_count >= minimum_positive && scores[index] <= 0.0) {
                scores[index] = f32::NEG_INFINITY;
            } else if frame >= SORTFORMER_SPEAKER_CACHE_FRAMES {
                scores[index] += SORTFORMER_LATEST_SCORE_BOOST;
            }
        }
    }
    let strong = (cache_frames_per_speaker as f32 * SORTFORMER_STRONG_BOOST_RATE).floor() as usize;
    let weak = (cache_frames_per_speaker as f32 * SORTFORMER_WEAK_BOOST_RATE).floor() as usize;
    boost_top_scores(&mut scores, frames, strong, 2.0)?;
    boost_top_scores(&mut scores, frames, weak, 1.0)?;

    let score_frames = frames
        .checked_add(SORTFORMER_SILENCE_FRAMES_PER_SPEAKER)
        .ok_or_else(|| reference_error("l6_compression", "L6 silence padding overflows"))?;
    let mut flattened = Vec::new();
    flattened
        .try_reserve_exact(
            score_frames
                .checked_mul(SORTFORMER_SPEAKER_LANES)
                .ok_or_else(|| reference_error("l6_compression", "L6 top-k size overflows"))?,
        )
        .map_err(|_| reference_error("allocation", "L6 top-k allocation failed"))?;
    for speaker in 0..SORTFORMER_SPEAKER_LANES {
        for frame in 0..score_frames {
            let score = if frame < frames {
                scores[frame * SORTFORMER_SPEAKER_LANES + speaker]
            } else {
                f32::INFINITY
            };
            flattened.push((score, speaker * score_frames + frame));
        }
    }
    flattened.sort_unstable_by(|(left_score, left_index), (right_score, right_index)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_index.cmp(right_index))
    });
    flattened.truncate(SORTFORMER_SPEAKER_CACHE_FRAMES);
    let mut selected = flattened
        .into_iter()
        .map(|(score, index)| {
            if score == f32::NEG_INFINITY {
                usize::MAX
            } else {
                index
            }
        })
        .collect::<Vec<_>>();
    selected.sort_unstable();

    let cache_values = SORTFORMER_SPEAKER_CACHE_FRAMES
        .checked_mul(SORTFORMER_ENCODER_WIDTH)
        .ok_or_else(|| reference_error("l6_compression", "L6 cache size overflows"))?;
    let prediction_values = SORTFORMER_SPEAKER_CACHE_FRAMES
        .checked_mul(SORTFORMER_SPEAKER_LANES)
        .ok_or_else(|| reference_error("l6_compression", "L6 prediction size overflows"))?;
    let mut compressed_cache = Vec::new();
    compressed_cache
        .try_reserve_exact(cache_values)
        .map_err(|_| reference_error("allocation", "L6 cache allocation failed"))?;
    let mut compressed_predictions = Vec::new();
    compressed_predictions
        .try_reserve_exact(prediction_values)
        .map_err(|_| reference_error("allocation", "L6 prediction allocation failed"))?;
    for flattened_index in selected {
        let frame = flattened_index % score_frames;
        let disabled = flattened_index == usize::MAX || frame >= frames;
        if disabled {
            compressed_cache.extend_from_slice(&state.mean_silence_embedding);
            compressed_predictions.extend([0.0; SORTFORMER_SPEAKER_LANES]);
        } else {
            compressed_cache.extend_from_slice(slice_rows(
                &state.speaker_cache.data,
                SORTFORMER_ENCODER_WIDTH,
                frame,
                frame + 1,
            )?);
            compressed_predictions.extend_from_slice(slice_rows(
                &predictions.data,
                SORTFORMER_SPEAKER_LANES,
                frame,
                frame + 1,
            )?);
        }
    }
    state.speaker_cache = Mat::from_vec(
        SORTFORMER_SPEAKER_CACHE_FRAMES,
        SORTFORMER_ENCODER_WIDTH,
        compressed_cache,
    );
    state.speaker_cache_predictions = Some(Mat::from_vec(
        SORTFORMER_SPEAKER_CACHE_FRAMES,
        SORTFORMER_SPEAKER_LANES,
        compressed_predictions,
    ));
    state.speaker_permutation = None;
    Ok(())
}

fn relu_in_place(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn swish_in_place(values: &mut [f32]) -> FwResult<()> {
    for value in values {
        let sigmoid = 1.0 / (1.0 + (-*value).exp());
        *value *= sigmoid;
        if !value.is_finite() {
            return Err(reference_error(
                "activation_nonfinite",
                "Swish produced a non-finite value",
            ));
        }
    }
    Ok(())
}

fn sigmoid_f32(value: f32) -> FwResult<f32> {
    if !value.is_finite() {
        return Err(reference_error(
            "activation_nonfinite",
            "sigmoid input is non-finite",
        ));
    }
    let output = 1.0 / (1.0 + (-value).exp());
    if !output.is_finite() {
        return Err(reference_error(
            "activation_nonfinite",
            "sigmoid produced a non-finite value",
        ));
    }
    Ok(output)
}

fn add_scaled_residual_in_place(destination: &mut Mat, update: &Mat, scale: f32) -> FwResult<()> {
    if destination.rows != update.rows
        || destination.cols != update.cols
        || destination.data.len() != update.data.len()
        || !scale.is_finite()
    {
        return Err(reference_error(
            "residual_shape",
            "residual update geometry or scale is invalid",
        ));
    }
    for (destination, &update) in destination.data.iter_mut().zip(&update.data) {
        *destination += scale * update;
        if !destination.is_finite() {
            return Err(reference_error(
                "residual_nonfinite",
                "residual update produced a non-finite value",
            ));
        }
    }
    Ok(())
}

fn relative_position_encoding(frames: usize, width: usize) -> FwResult<Vec<f32>> {
    if frames == 0 || width == 0 || !width.is_multiple_of(2) {
        return Err(reference_error(
            "position_shape",
            "relative positional encoding requires positive frames and an even width",
        ));
    }
    let rows = frames
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| reference_error("position_shape", "relative position length overflows"))?;
    let values = rows
        .checked_mul(width)
        .ok_or_else(|| reference_error("position_shape", "relative position size overflows"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(values)
        .map_err(|_| reference_error("allocation", "relative position allocation failed"))?;
    output.resize(values, 0.0);
    let exponent_scale = -(10_000.0_f32.ln() / width as f32);
    for row in 0..rows {
        let position = (frames - 1) as f32 - row as f32;
        for even in (0..width).step_by(2) {
            let angle = position * (even as f32 * exponent_scale).exp();
            output[row * width + even] = angle.sin();
            output[row * width + even + 1] = angle.cos();
        }
    }
    Ok(output)
}

fn softmax_f32_in_place(values: &mut [f32]) -> FwResult<()> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(reference_error(
            "softmax_input",
            "softmax input must be non-empty and finite",
        ));
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(reference_error(
            "softmax_output",
            "softmax normalization sum is invalid",
        ));
    }
    for value in values {
        *value /= sum;
    }
    Ok(())
}

fn compare_public_f32_probe(
    pack: &VerifiedSortformerPublicActivationPack,
    stage: &str,
    observed_shape: &[usize],
    observed_values: &[f32],
    cross_kernel_abs_tolerance: f64,
    cross_kernel_relative_l2: f64,
) -> FwResult<SortformerSeamParityMetrics> {
    let contract = pack
        .receipt()
        .seam_contracts
        .iter()
        .find(|contract| contract.stage == stage)
        .ok_or_else(|| reference_error("parity_seam", "L2 seam contract is absent"))?;
    let contract_shape = contract
        .full_shape
        .iter()
        .map(|&dimension| usize::try_from(dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| reference_error("parity_shape", "L2 seam shape does not fit usize"))?;
    if contract_shape != observed_shape
        || usize::try_from(contract.full_elements).ok() != Some(observed_values.len())
    {
        return Err(reference_error(
            "parity_shape",
            "native L2 seam geometry differs from the authenticated full tensor contract",
        ));
    }
    let expected = load_public_probe_f32(pack, stage, &contract.probe_shape)?;
    let mut observed_probe = Vec::new();
    observed_probe
        .try_reserve_exact(expected.len())
        .map_err(|_| reference_error("allocation", "L2 probe allocation failed"))?;
    match contract.probe_selection.as_str() {
        "complete_tensor" => observed_probe.extend_from_slice(observed_values),
        "linear_index_endpoints_inclusive_v1" => {
            if expected.len() < 2 || observed_values.len() < expected.len() {
                return Err(reference_error(
                    "parity_probe",
                    "linear endpoint probe geometry is invalid",
                ));
            }
            let last = observed_values.len() - 1;
            let denominator = expected.len() - 1;
            for probe_index in 0..expected.len() {
                let source = probe_index
                    .checked_mul(last)
                    .ok_or_else(|| reference_error("parity_probe", "probe index overflows"))?
                    / denominator;
                observed_probe.push(observed_values[source]);
            }
        }
        _ => {
            return Err(reference_error(
                "parity_probe",
                "authenticated L2 seam uses an unsupported probe selection rule",
            ));
        }
    }
    if observed_probe.len() != expected.len() {
        return Err(reference_error(
            "parity_probe",
            "native and authenticated L2 probes have different lengths",
        ));
    }

    let observation = pack
        .receipt()
        .oracle_floor
        .observations
        .iter()
        .find(|observation| observation.stage == stage)
        .ok_or_else(|| reference_error("parity_floor", "L2 oracle-floor row is absent"))?;
    let source_abs_tolerance = f64::from(parse_f32_hex_bits(
        &observation.accepted_abs_tolerance_f32_bits,
    )?);
    let source_relative_l2 = parse_f64_hex_bits(&observation.accepted_relative_l2_f64_bits)?;
    let accepted_abs_tolerance = source_abs_tolerance.max(cross_kernel_abs_tolerance);
    let accepted_relative_l2 = source_relative_l2.max(cross_kernel_relative_l2);
    let mut mismatch_count = 0usize;
    let mut max_abs_diff = 0.0_f64;
    let mut squared_difference_sum = 0.0_f64;
    let mut squared_scale_sum = 0.0_f64;
    for (&left, &right) in observed_probe.iter().zip(&expected) {
        if !left.is_finite() || !right.is_finite() {
            return Err(reference_error(
                "parity_nonfinite",
                "native or authenticated L2 probe contains a non-finite value",
            ));
        }
        if left.to_bits() != right.to_bits() {
            mismatch_count = mismatch_count.saturating_add(1);
        }
        let difference = (f64::from(left) - f64::from(right)).abs();
        max_abs_diff = max_abs_diff.max(difference);
        squared_difference_sum += difference * difference;
        let scale = f64::from(left).abs().max(f64::from(right).abs());
        squared_scale_sum += scale * scale;
    }
    let relative_l2 = if squared_scale_sum == 0.0 {
        0.0
    } else {
        (squared_difference_sum / squared_scale_sum).sqrt()
    };
    if max_abs_diff > accepted_abs_tolerance || relative_l2 > accepted_relative_l2 {
        return Err(reference_error(
            "parity_budget",
            &format!(
                "L2 seam {stage} exceeds its authenticated envelope: max_abs={max_abs_diff:.9e}, \
                 relative_l2={relative_l2:.9e}"
            ),
        ));
    }
    Ok(SortformerSeamParityMetrics {
        stage: stage.to_owned(),
        compared_values: expected.len(),
        mismatch_count,
        byte_exact: mismatch_count == 0,
        max_abs_diff,
        relative_l2,
        accepted_abs_tolerance,
        accepted_relative_l2,
    })
}

fn compare_public_i64_probe(
    pack: &VerifiedSortformerPublicActivationPack,
    stage: &str,
    observed_shape: &[usize],
    observed_values: &[i64],
) -> FwResult<SortformerI64ParityMetrics> {
    let contract = pack
        .receipt()
        .seam_contracts
        .iter()
        .find(|contract| contract.stage == stage)
        .ok_or_else(|| reference_error("parity_seam", "integer seam contract is absent"))?;
    let full_shape = contract
        .full_shape
        .iter()
        .map(|&dimension| usize::try_from(dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| reference_error("parity_shape", "integer seam shape does not fit usize"))?;
    if full_shape != observed_shape
        || usize::try_from(contract.full_elements).ok() != Some(observed_values.len())
    {
        return Err(reference_error(
            "parity_shape",
            "native integer seam geometry differs from its authenticated contract",
        ));
    }
    let expected_shape = contract
        .probe_shape
        .iter()
        .map(|&dimension| usize::try_from(dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| reference_error("parity_shape", "integer probe shape does not fit usize"))?;
    if pack
        .safetensors()
        .dtype_name(stage)
        .map_err(|_| reference_error("parity_tensor", "integer probe is absent"))?
        != "I64"
        || pack
            .safetensors()
            .shape(stage)
            .map_err(|_| reference_error("parity_tensor", "integer probe shape is absent"))?
            != expected_shape
    {
        return Err(reference_error(
            "parity_tensor",
            "authenticated integer probe has an invalid dtype or shape",
        ));
    }
    let raw = pack
        .safetensors()
        .tensor_raw_bytes(stage)
        .map_err(|_| reference_error("parity_tensor", "integer probe is unreadable"))?;
    if raw.len() % std::mem::size_of::<i64>() != 0 {
        return Err(reference_error(
            "parity_tensor",
            "integer probe byte length is not I64-aligned",
        ));
    }
    let mut expected = Vec::new();
    expected
        .try_reserve_exact(raw.len() / std::mem::size_of::<i64>())
        .map_err(|_| reference_error("allocation", "integer probe allocation failed"))?;
    for bytes in raw.chunks_exact(std::mem::size_of::<i64>()) {
        expected.push(i64::from_le_bytes(bytes.try_into().map_err(|_| {
            reference_error("parity_tensor", "integer probe chunk has invalid width")
        })?));
    }
    let mut observed_probe = Vec::new();
    observed_probe
        .try_reserve_exact(expected.len())
        .map_err(|_| reference_error("allocation", "integer parity allocation failed"))?;
    match contract.probe_selection.as_str() {
        "complete_tensor" => observed_probe.extend_from_slice(observed_values),
        "linear_index_endpoints_inclusive_v1" => {
            if expected.len() < 2 || observed_values.len() < expected.len() {
                return Err(reference_error(
                    "parity_probe",
                    "integer endpoint probe geometry is invalid",
                ));
            }
            let last = observed_values.len() - 1;
            let denominator = expected.len() - 1;
            for probe_index in 0..expected.len() {
                let source = probe_index
                    .checked_mul(last)
                    .ok_or_else(|| reference_error("parity_probe", "probe index overflows"))?
                    / denominator;
                observed_probe.push(observed_values[source]);
            }
        }
        _ => {
            return Err(reference_error(
                "parity_probe",
                "authenticated integer seam uses an unsupported probe rule",
            ));
        }
    }
    let mismatch_count = observed_probe
        .iter()
        .zip(&expected)
        .filter(|(observed, expected)| observed != expected)
        .count();
    if mismatch_count != 0 {
        return Err(reference_error(
            "parity_integer",
            &format!("integer seam {stage} has {mismatch_count} mismatches"),
        ));
    }
    Ok(SortformerI64ParityMetrics {
        stage: stage.to_owned(),
        compared_values: expected.len(),
        mismatch_count,
        byte_exact: true,
    })
}

fn load_public_probe_f32(
    pack: &VerifiedSortformerPublicActivationPack,
    stage: &str,
    expected_shape_u64: &[u64],
) -> FwResult<Vec<f32>> {
    let expected_shape = expected_shape_u64
        .iter()
        .map(|&dimension| usize::try_from(dimension))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| reference_error("parity_shape", "probe shape does not fit usize"))?;
    if pack
        .safetensors()
        .dtype_name(stage)
        .map_err(|_| reference_error("parity_tensor", "authenticated L2 probe is absent"))?
        != "F32"
    {
        return Err(reference_error(
            "parity_tensor",
            "authenticated L2 probe is not exact F32",
        ));
    }
    let (shape, values) = pack
        .safetensors()
        .tensor_f32(stage)
        .map_err(|_| reference_error("parity_tensor", "authenticated L2 probe is unreadable"))?;
    if shape != expected_shape {
        return Err(reference_error(
            "parity_shape",
            "authenticated L2 probe shape differs from its receipt",
        ));
    }
    Ok(values)
}

fn parse_f32_hex_bits(encoded: &str) -> FwResult<f32> {
    let digits = encoded
        .strip_prefix("0x")
        .filter(|digits| digits.len() == 8)
        .ok_or_else(|| reference_error("parity_floor", "f32 tolerance encoding is invalid"))?;
    let bits = u32::from_str_radix(digits, 16)
        .map_err(|_| reference_error("parity_floor", "f32 tolerance bits are invalid"))?;
    let value = f32::from_bits(bits);
    if !value.is_finite() || value < 0.0 {
        return Err(reference_error(
            "parity_floor",
            "f32 tolerance must be finite and non-negative",
        ));
    }
    Ok(value)
}

fn parse_f64_hex_bits(encoded: &str) -> FwResult<f64> {
    let digits = encoded
        .strip_prefix("0x")
        .filter(|digits| digits.len() == 16)
        .ok_or_else(|| reference_error("parity_floor", "f64 tolerance encoding is invalid"))?;
    let bits = u64::from_str_radix(digits, 16)
        .map_err(|_| reference_error("parity_floor", "f64 tolerance bits are invalid"))?;
    let value = f64::from_bits(bits);
    if !value.is_finite() || value < 0.0 {
        return Err(reference_error(
            "parity_floor",
            "f64 tolerance must be finite and non-negative",
        ));
    }
    Ok(value)
}

fn reference_checkpoint(checkpoint: &(dyn Fn() -> FwResult<()> + Sync)) -> FwResult<()> {
    match checkpoint() {
        Ok(()) => Ok(()),
        Err(FwError::Cancelled(_)) => Err(FwError::Cancelled(
            "sortformer.reference_cancelled: cooperative checkpoint requested cancellation"
                .to_owned(),
        )),
        Err(_) => Err(FwError::ContractViolation(
            "sortformer.reference_checkpoint_failure: caller checkpoint returned a non-cancellation failure"
                .to_owned(),
        )),
    }
}

fn reference_error(code: &str, detail: &str) -> FwError {
    FwError::InvalidRequest(format!("sortformer.reference_{code}: {detail}"))
}

/// Run the native frontend against the independently authenticated synthetic
/// oracle pack and enforce the predeclared cross-kernel envelope.
///
/// The frontend buffers come from `model_package`; the expected activations
/// come from `activation_pack`. Keeping those trust chains separate prevents a
/// tautological replay of the oracle's own analysis buffers.
pub fn verify_sortformer_frontend_synthetic_parity(
    model_package: &VerifiedSortformerPackage,
    activation_pack: &VerifiedSortformerActivationPack,
) -> FwResult<SortformerFrontendParityReport> {
    verify_sortformer_frontend_synthetic_parity_with_checkpoint(
        model_package,
        activation_pack,
        &|| Ok(()),
    )
}

/// Cancellation-aware form of [`verify_sortformer_frontend_synthetic_parity`].
pub fn verify_sortformer_frontend_synthetic_parity_with_checkpoint(
    model_package: &VerifiedSortformerPackage,
    activation_pack: &VerifiedSortformerActivationPack,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<SortformerFrontendParityReport> {
    let frontend =
        SortformerFrontend::from_verified_package_with_checkpoint(model_package, checkpoint)?;
    let oracle_floor_byte_exact = activation_pack.receipt().oracle_floor.all_byte_exact;
    if !oracle_floor_byte_exact {
        return Err(frontend_error(
            "parity_oracle_floor",
            "authenticated activation pack does not carry a byte-exact source floor",
        ));
    }

    let mut fixtures = Vec::new();
    fixtures
        .try_reserve_exact(activation_pack.receipt().fixtures.len())
        .map_err(|_| {
            frontend_error(
                "parity_allocation",
                "synthetic parity report could not be allocated",
            )
        })?;
    for fixture in &activation_pack.receipt().fixtures {
        frontend_checkpoint(checkpoint)?;
        let decoded_name = format!("fixture.{}.decoded_pcm_f32", fixture.name);
        let decoded_shape = [
            1,
            usize::try_from(fixture.sample_count).map_err(|_| {
                frontend_error(
                    "parity_shape",
                    "synthetic fixture sample count does not fit this platform",
                )
            })?,
        ];
        let decoded = load_activation_f32(activation_pack, &decoded_name, &decoded_shape)?;
        let observed =
            frontend.compute_with_checkpoint(SortformerPcm::mono_16khz(&decoded), checkpoint)?;

        let valid_frames = usize::try_from(fixture.valid_frames).map_err(|_| {
            frontend_error(
                "parity_shape",
                "synthetic fixture frame count does not fit this platform",
            )
        })?;
        if observed.mel_bins != SORTFORMER_MEL_BINS || observed.valid_frames != valid_frames {
            return Err(frontend_error(
                "parity_shape",
                "native frontend output geometry changed from the authenticated fixture",
            ));
        }
        let expected_name = format!("fixture.{}.log_mel_f32", fixture.name);
        let expected = load_activation_f32(
            activation_pack,
            &expected_name,
            &[1, SORTFORMER_MEL_BINS, valid_frames],
        )?;
        let metrics = frontend_parity_metrics(&fixture.name, &observed.data, &expected)?;
        let silence_requires_exact = fixture.generator == "all_zero_i16_v1";
        if (silence_requires_exact && !metrics.byte_exact)
            || metrics.max_abs_diff > SORTFORMER_FRONTEND_MAX_ABS_DIFF
            || metrics.mean_abs_diff > SORTFORMER_FRONTEND_MAX_MEAN_ABS_DIFF
            || metrics.relative_l2 > SORTFORMER_FRONTEND_MAX_RELATIVE_L2
        {
            return Err(frontend_error(
                "parity_budget",
                &format!(
                    "synthetic fixture {} exceeds the frozen frontend envelope: \
                     mismatches={}, max_abs={:.9e}, mean_abs={:.9e}, relative_l2={:.9e}",
                    metrics.fixture,
                    metrics.mismatch_count,
                    metrics.max_abs_diff,
                    metrics.mean_abs_diff,
                    metrics.relative_l2,
                ),
            ));
        }
        fixtures.push(metrics);
    }
    frontend_checkpoint(checkpoint)?;
    Ok(SortformerFrontendParityReport {
        oracle_floor_byte_exact,
        fixtures,
    })
}

fn load_activation_f32(
    pack: &VerifiedSortformerActivationPack,
    name: &str,
    expected_shape: &[usize],
) -> FwResult<Vec<f32>> {
    if pack
        .safetensors()
        .dtype_name(name)
        .map_err(|_| frontend_error("parity_tensor", "authenticated activation tensor is absent"))?
        != "F32"
    {
        return Err(frontend_error(
            "parity_tensor",
            "authenticated activation tensor is not exact F32",
        ));
    }
    let (shape, values) = pack.safetensors().tensor_f32(name).map_err(|_| {
        frontend_error(
            "parity_tensor",
            "authenticated activation tensor could not be materialized",
        )
    })?;
    if shape != expected_shape {
        return Err(frontend_error(
            "parity_shape",
            "authenticated activation tensor shape changed",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(frontend_error(
            "parity_nonfinite",
            "authenticated activation tensor contains a non-finite value",
        ));
    }
    Ok(values)
}

fn frontend_parity_metrics(
    fixture: &str,
    observed: &[f32],
    expected: &[f32],
) -> FwResult<SortformerFrontendParityMetrics> {
    if observed.len() != expected.len() {
        return Err(frontend_error(
            "parity_shape",
            "native and oracle frontend tensors have different value counts",
        ));
    }
    let mut mismatch_count = 0usize;
    let mut max_abs_diff = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    let mut squared_difference_sum = 0.0_f64;
    let mut squared_scale_sum = 0.0_f64;
    for (&left, &right) in observed.iter().zip(expected) {
        if !left.is_finite() || !right.is_finite() {
            return Err(frontend_error(
                "parity_nonfinite",
                "native or oracle frontend comparison value is non-finite",
            ));
        }
        if left.to_bits() != right.to_bits() {
            mismatch_count = mismatch_count.saturating_add(1);
        }
        let difference = (f64::from(left) - f64::from(right)).abs();
        max_abs_diff = max_abs_diff.max(difference);
        absolute_sum += difference;
        squared_difference_sum += difference * difference;
        let scale = f64::from(left).abs().max(f64::from(right).abs());
        squared_scale_sum += scale * scale;
    }
    let mean_abs_diff = if observed.is_empty() {
        0.0
    } else {
        absolute_sum / observed.len() as f64
    };
    let relative_l2 = if squared_scale_sum == 0.0 {
        0.0
    } else {
        (squared_difference_sum / squared_scale_sum).sqrt()
    };
    Ok(SortformerFrontendParityMetrics {
        fixture: fixture.to_owned(),
        compared_values: observed.len(),
        mismatch_count,
        byte_exact: mismatch_count == 0,
        max_abs_diff,
        mean_abs_diff,
        relative_l2,
    })
}

fn mel_nonzero_ranges(filterbank: &[f32]) -> FwResult<[(usize, usize); SORTFORMER_MEL_BINS]> {
    if filterbank.len() != SORTFORMER_MEL_BINS * REAL_FFT_512_BINS {
        return Err(frontend_error(
            "mel_count",
            "authenticated mel filterbank has the wrong value count",
        ));
    }

    Ok(std::array::from_fn(|mel| {
        let start = mel * REAL_FFT_512_BINS;
        let filter = &filterbank[start..start + REAL_FFT_512_BINS];
        let first = filter.iter().position(|&weight| weight != 0.0);
        let last = filter.iter().rposition(|&weight| weight != 0.0);
        match (first, last) {
            (Some(first), Some(last)) => (first, last + 1),
            _ => (0, 0),
        }
    }))
}

fn sparse_mel_energy(power: &[f32], filter: &[f32], range: (usize, usize)) -> f32 {
    let mut energy = 0.0_f32;
    for frequency in range.0..range.1 {
        let weight = filter[frequency];
        if weight != 0.0 {
            // The full authenticated filterbank remains authoritative. Exact
            // zeros may be skipped because `power` was proven finite above;
            // all retained additions stay in ascending frequency order.
            energy += power[frequency] * weight;
        }
    }
    energy
}

fn preemphasized_sample(samples: &[f32], source: usize) -> Option<f32> {
    let &sample = samples.get(source)?;
    if source == 0 {
        Some(sample)
    } else {
        Some(sample - PREEMPHASIS * samples[source - 1])
    }
}

fn frame_geometry(sample_count: usize) -> FwResult<(usize, usize)> {
    let valid = sample_count / SORTFORMER_HOP_SAMPLES;
    let physical = valid
        .checked_add(1)
        .ok_or_else(|| frontend_error("frame_count", "STFT frame count overflows"))?;
    Ok((valid, physical))
}

fn zeroed_output_buffer(
    value_count: usize,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<Vec<f32>> {
    frontend_checkpoint(checkpoint)?;
    let mut data = Vec::new();
    data.try_reserve_exact(value_count).map_err(|_| {
        frontend_error(
            "output_allocation",
            "frontend output buffer could not be allocated within the admitted resource envelope",
        )
    })?;
    while data.len() < value_count {
        frontend_checkpoint(checkpoint)?;
        let next_len = data
            .len()
            .saturating_add(ZERO_FILL_CHUNK_VALUES)
            .min(value_count);
        data.resize(next_len, 0.0);
    }
    Ok(data)
}

fn validate_pcm(
    pcm: SortformerPcm<'_>,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    validate_pcm_contract(pcm.samples.len(), pcm.sample_rate_hz, pcm.channels)?;
    for chunk in pcm.samples.chunks(4096) {
        frontend_checkpoint(checkpoint)?;
        if chunk
            .iter()
            .any(|sample| !sample.is_finite() || !(-1.0..=1.0).contains(sample))
        {
            return Err(frontend_error(
                "pcm_value",
                "frontend PCM must be finite and within [-1, 1]",
            ));
        }
    }
    Ok(())
}

fn validate_pcm_contract(
    sample_count: usize,
    sample_rate_hz: usize,
    channels: usize,
) -> FwResult<()> {
    if sample_rate_hz != SORTFORMER_SAMPLE_RATE_HZ {
        return Err(frontend_error(
            "sample_rate",
            "frontend PCM must be sampled at exactly 16000 Hz",
        ));
    }
    if channels != SORTFORMER_CHANNELS {
        return Err(frontend_error(
            "channels",
            "frontend PCM must contain exactly one channel",
        ));
    }
    if sample_count == 0 {
        return Err(frontend_error(
            "pcm_empty",
            "frontend PCM must contain at least one sample",
        ));
    }
    if sample_count > SORTFORMER_MAX_AUDIO_SAMPLES {
        return Err(frontend_error(
            "pcm_duration",
            "whole-file frontend PCM exceeds the two-hour resource ceiling",
        ));
    }
    Ok(())
}

fn load_exact_f32_tensor(
    package: &SafetensorsFile,
    contract: TensorContract<'_>,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<Vec<f32>> {
    frontend_checkpoint(checkpoint)?;
    let actual_shape = package.shape(contract.name).map_err(|_| {
        frontend_error(
            "tensor_missing",
            "required pinned frontend tensor is absent",
        )
    })?;
    if actual_shape != contract.shape {
        return Err(frontend_error(
            "tensor_shape",
            "pinned frontend tensor has the wrong shape",
        ));
    }
    if package.dtype_name(contract.name).map_err(|_| {
        frontend_error(
            "tensor_dtype",
            "pinned frontend tensor dtype is unavailable",
        )
    })? != "F32"
    {
        return Err(frontend_error(
            "tensor_dtype",
            "pinned frontend tensor must use exact F32 storage",
        ));
    }

    let raw = package.tensor_raw_bytes(contract.name).map_err(|_| {
        frontend_error(
            "tensor_payload",
            "pinned frontend tensor payload is unavailable",
        )
    })?;
    let digest = sha256_hex(raw);
    if digest != contract.sha256 {
        return Err(frontend_error(
            "tensor_hash",
            "pinned frontend tensor payload hash does not match",
        ));
    }
    frontend_checkpoint(checkpoint)?;

    let (_, values) = package.tensor_f32(contract.name).map_err(|_| {
        frontend_error(
            "tensor_materialize",
            "pinned frontend tensor could not be materialized",
        )
    })?;
    for chunk in values.chunks(4096) {
        frontend_checkpoint(checkpoint)?;
        if chunk.iter().any(|value| !value.is_finite()) {
            return Err(frontend_error(
                "tensor_nonfinite",
                "pinned frontend tensor contains a non-finite value",
            ));
        }
    }
    Ok(values)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn frontend_checkpoint(checkpoint: &(dyn Fn() -> FwResult<()> + Sync)) -> FwResult<()> {
    match checkpoint() {
        Ok(()) => Ok(()),
        Err(FwError::Cancelled(_)) => Err(FwError::Cancelled(
            "sortformer.frontend_cancelled: cooperative checkpoint requested cancellation"
                .to_owned(),
        )),
        Err(_) => Err(FwError::ContractViolation(
            "sortformer.frontend_checkpoint_failure: caller checkpoint returned a non-cancellation failure"
                .to_owned(),
        )),
    }
}

fn frontend_error(code: &str, detail: &str) -> FwError {
    FwError::InvalidRequest(format!("sortformer.frontend_{code}: {detail}"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn complete_public_mat(
        pack: &VerifiedSortformerPublicActivationPack,
        stage: &str,
        width: usize,
    ) -> Mat {
        let contract = pack
            .receipt()
            .seam_contracts
            .iter()
            .find(|contract| contract.stage == stage)
            .expect("diagnostic seam contract exists");
        assert_eq!(contract.probe_selection, "complete_tensor");
        let full_shape = contract
            .full_shape
            .iter()
            .map(|&dimension| usize::try_from(dimension).expect("diagnostic shape fits usize"))
            .collect::<Vec<_>>();
        assert_eq!(full_shape.len(), 3);
        assert_eq!(full_shape[0], 1);
        assert_eq!(full_shape[2], width);
        let values = load_public_probe_f32(pack, stage, &contract.probe_shape)
            .expect("authenticated complete diagnostic tensor");
        Mat::from_vec(full_shape[1], width, values)
    }

    fn clipped_rttm_turns(
        annotation_path: &std::path::Path,
        recording_id: &str,
        clip_start_ms: u64,
        clip_duration_ms: u64,
    ) -> Vec<crate::diarization::EvaluationTurn> {
        let annotation = std::fs::read_to_string(annotation_path).expect("read public RTTM");
        let clip_end_ms = clip_start_ms
            .checked_add(clip_duration_ms)
            .expect("bounded public clip");
        let mut turns = Vec::new();
        for line in annotation.lines() {
            let fields = line.split_ascii_whitespace().collect::<Vec<_>>();
            if fields.len() != 10 || fields[0] != "SPEAKER" || fields[1] != recording_id {
                continue;
            }
            let start_ms = (fields[3].parse::<f64>().expect("RTTM start") * 1_000.0).round() as u64;
            let duration_ms =
                (fields[4].parse::<f64>().expect("RTTM duration") * 1_000.0).round() as u64;
            let end_ms = start_ms
                .checked_add(duration_ms)
                .expect("bounded RTTM turn");
            let clipped_start = start_ms.max(clip_start_ms);
            let clipped_end = end_ms.min(clip_end_ms);
            if clipped_start < clipped_end {
                turns.push(crate::diarization::EvaluationTurn::labeled(
                    clipped_start - clip_start_ms,
                    clipped_end - clip_start_ms,
                    fields[7],
                ));
            }
        }
        for index in 0..turns.len() {
            turns[index].overlap_suspected =
                turns.iter().enumerate().any(|(other_index, other)| {
                    other_index != index
                        && other.speaker != turns[index].speaker
                        && other.start_ms < turns[index].end_ms
                        && turns[index].start_ms < other.end_ms
                });
        }
        turns.sort_by(|left, right| {
            (left.start_ms, left.end_ms, left.speaker.as_deref()).cmp(&(
                right.start_ms,
                right.end_ms,
                right.speaker.as_deref(),
            ))
        });
        assert!(
            !turns.is_empty(),
            "public clip must contain reference turns"
        );
        turns
    }

    #[test]
    fn grouped_conv2d_matches_hand_computed_depthwise_padding() {
        let input = Tensor4::from_vec(1, 2, 2, 2, vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
            .expect("input tensor");
        let convolution = Conv2d::new(
            [2, 1, 3, 3],
            2,
            vec![1.0; 9].into_iter().chain(vec![2.0; 9]).collect(),
            vec![0.5, -1.0],
        )
        .expect("depthwise convolution");
        let output = convolution
            .forward(&input, 2, 1, &|| Ok(()))
            .expect("grouped convolution output");
        assert_eq!(output.shape(), vec![1, 2, 1, 1]);
        assert_eq!(output.data, vec![10.5, 199.0]);
    }

    #[test]
    fn conv2d_rejects_bad_groups_payloads_and_input_geometry() {
        assert!(Conv2d::new([2, 1, 1, 1], 0, vec![1.0; 2], vec![0.0; 2]).is_err());
        assert!(Conv2d::new([3, 1, 1, 1], 2, vec![1.0; 3], vec![0.0; 3]).is_err());
        assert!(Conv2d::new([2, 1, 1, 1], 1, vec![1.0], vec![0.0; 2]).is_err());

        let convolution = Conv2d::new([1, 1, 1, 1], 1, vec![1.0], vec![0.0]).expect("valid conv");
        let wrong_channels =
            Tensor4::from_vec(1, 2, 1, 1, vec![1.0, 2.0]).expect("two-channel input");
        let error = convolution
            .forward(&wrong_channels, 1, 0, &|| Ok(()))
            .err()
            .expect("channel mismatch");
        assert!(error.to_string().contains("conv_contract"));
    }

    #[test]
    fn affine_transpose_preserves_natural_linear_layer_semantics() {
        let weight = transpose_affine_weight("test", 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("transposed affine weight");
        assert_eq!(weight.rows, 3);
        assert_eq!(weight.cols, 2);
        assert_eq!(weight.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let input = Mat::from_vec(1, 3, vec![1.0, 10.0, 100.0]);
        let output =
            nn::matmul_bias(&input, &weight, Some(&[0.5, -0.5])).expect("FrankenTorch affine");
        assert_eq!(output.data, vec![321.5, 653.5]);
    }

    #[test]
    fn affine_and_layer_norm_fail_closed_on_malformed_or_nonfinite_inputs() {
        let affine = Affine {
            input: 2,
            output: 1,
            weight_t: Mat::from_vec(2, 1, vec![1.0, 1.0]),
            bias: Some(vec![0.0]),
        };
        let malformed = Mat {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0],
        };
        assert!(affine.forward(&malformed).is_err());
        assert!(
            affine
                .forward(&Mat::from_vec(1, 2, vec![f32::NAN, 0.0]))
                .is_err()
        );

        let norm = LayerNorm {
            width: 2,
            weight: vec![1.0, 1.0],
            bias: vec![0.0, 0.0],
        };
        let mut empty = Mat::from_vec(0, 2, Vec::new());
        assert!(norm.forward_in_place(&mut empty).is_err());
        let mut infinite = Mat::from_vec(1, 2, vec![f32::INFINITY, 0.0]);
        assert!(norm.forward_in_place(&mut infinite).is_err());
    }

    #[test]
    fn layer_norm_and_swish_match_their_scalar_definitions() {
        let norm = LayerNorm {
            width: 2,
            weight: vec![2.0, 0.5],
            bias: vec![0.25, -0.75],
        };
        let mut input = Mat::from_vec(1, 2, vec![1.0, 3.0]);
        norm.forward_in_place(&mut input).expect("finite LayerNorm");
        let inverse_std = 1.0_f64 / (1.0 + f64::from(LayerNorm::EPSILON)).sqrt();
        assert_eq!(input.data[0], (-inverse_std * 2.0 + 0.25) as f32);
        assert_eq!(input.data[1], (inverse_std * 0.5 - 0.75) as f32);

        let mut values = vec![-2.0, 0.0, 2.0];
        swish_in_place(&mut values).expect("finite Swish");
        assert_eq!(values[0], -2.0 / (1.0 + 2.0_f32.exp()));
        assert_eq!(values[1], 0.0);
        assert_eq!(values[2], 2.0 / (1.0 + (-2.0_f32).exp()));
    }

    #[test]
    fn scaled_residual_checks_geometry_and_nonfinite_results() {
        let mut destination = Mat::from_vec(1, 2, vec![1.0, -2.0]);
        let update = Mat::from_vec(1, 2, vec![4.0, 6.0]);
        add_scaled_residual_in_place(&mut destination, &update, 0.5)
            .expect("matching finite residual");
        assert_eq!(destination.data, vec![3.0, 1.0]);
        assert!(
            add_scaled_residual_in_place(
                &mut destination,
                &Mat::from_vec(2, 1, vec![1.0, 2.0]),
                1.0,
            )
            .is_err()
        );
        assert!(add_scaled_residual_in_place(&mut destination, &update, f32::NAN).is_err());
    }

    #[test]
    fn relative_positions_and_softmax_have_exact_geometry_and_symmetry() {
        let positions = relative_position_encoding(2, 4).expect("relative positions");
        assert_eq!(positions.len(), 12);
        assert_eq!(&positions[4..8], &[0.0, 1.0, 0.0, 1.0]);
        assert_eq!(positions[0], -positions[8]);
        assert_eq!(positions[1], positions[9]);
        assert!(relative_position_encoding(0, 4).is_err());
        assert!(relative_position_encoding(2, 3).is_err());

        let mut probabilities = vec![1.0, 1.0, 1.0, 1.0];
        softmax_f32_in_place(&mut probabilities).expect("finite softmax");
        assert_eq!(probabilities, vec![0.25; 4]);
        assert!(softmax_f32_in_place(&mut []).is_err());
        assert!(softmax_f32_in_place(&mut [f32::NAN]).is_err());
    }

    #[test]
    fn sigmoid_and_eval_batch_norm_match_scalar_definitions() {
        assert_eq!(sigmoid_f32(0.0).unwrap(), 0.5);
        assert!(sigmoid_f32(f32::NAN).is_err());
        let norm = BatchNorm1d {
            width: 2,
            scale: vec![2.0, 0.5],
            bias: vec![1.0, -1.0],
            running_mean: vec![3.0, 4.0],
            running_var: vec![4.0 - BatchNorm1d::EPSILON, 1.0 - BatchNorm1d::EPSILON],
        };
        let mut input = Mat::from_vec(1, 2, vec![5.0, 6.0]);
        norm.forward_in_place(&mut input).expect("eval BatchNorm");
        assert_eq!(input.data, vec![3.0, 0.0]);

        let invalid = BatchNorm1d {
            width: 1,
            scale: vec![1.0],
            bias: vec![0.0],
            running_mean: vec![0.0],
            running_var: vec![-1.0],
        };
        assert!(
            invalid
                .forward_in_place(&mut Mat::from_vec(1, 1, vec![0.0]))
                .is_err()
        );
    }

    #[test]
    fn tolerance_bit_parsers_are_strict_and_finite() {
        assert_eq!(parse_f32_hex_bits("0x3f000000").unwrap(), 0.5);
        assert_eq!(parse_f64_hex_bits("0x3fe0000000000000").unwrap(), 0.5);
        for invalid in ["3f000000", "0x7fc00000", "0xbf000000", "0x0000"] {
            assert!(parse_f32_hex_bits(invalid).is_err(), "accepted {invalid}");
        }
        for invalid in [
            "3fe0000000000000",
            "0x7ff8000000000000",
            "0xbfe0000000000000",
            "0x0000",
        ] {
            assert!(parse_f64_hex_bits(invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn reference_checkpoint_redacts_cancellation_and_rejects_other_failures() {
        let cancelled =
            reference_checkpoint(&|| Err(FwError::Cancelled("private cancellation".to_owned())))
                .expect_err("cancelled checkpoint");
        assert!(matches!(cancelled, FwError::Cancelled(_)));
        assert!(!cancelled.to_string().contains("private cancellation"));

        let rejected =
            reference_checkpoint(&|| Err(FwError::InvalidRequest("private failure".to_owned())))
                .expect_err("non-cancellation checkpoint failure");
        assert!(matches!(rejected, FwError::ContractViolation(_)));
        assert!(!rejected.to_string().contains("private failure"));
    }

    fn synthetic_package(
        window: &[f32],
        window_shape: &[usize],
        mel: &[f32],
        mel_shape: &[usize],
    ) -> SafetensorsFile {
        let window_bytes = f32_bytes(window);
        let mel_bytes = f32_bytes(mel);
        synthetic_package_raw("F32", &window_bytes, window_shape, &mel_bytes, mel_shape)
    }

    fn synthetic_package_raw(
        window_dtype: &str,
        window_bytes: &[u8],
        window_shape: &[usize],
        mel_bytes: &[u8],
        mel_shape: &[usize],
    ) -> SafetensorsFile {
        let mut header = BTreeMap::new();
        header.insert(
            SORTFORMER_WINDOW_TENSOR_NAME,
            json!({
                "dtype": window_dtype,
                "shape": window_shape,
                "data_offsets": [0, window_bytes.len()],
            }),
        );
        header.insert(
            SORTFORMER_MEL_TENSOR_NAME,
            json!({
                "dtype": "F32",
                "shape": mel_shape,
                "data_offsets": [window_bytes.len(), window_bytes.len() + mel_bytes.len()],
            }),
        );
        let header_bytes = serde_json::to_vec(&header).expect("synthetic header");
        let mut package =
            Vec::with_capacity(8 + header_bytes.len() + window_bytes.len() + mel_bytes.len());
        package.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        package.extend_from_slice(&header_bytes);
        package.extend_from_slice(window_bytes);
        package.extend_from_slice(mel_bytes);
        SafetensorsFile::from_owned_bytes(package).expect("synthetic package")
    }

    fn synthetic_frontend() -> SortformerFrontend {
        let window: Vec<f32> = (0..SORTFORMER_WINDOW_SAMPLES)
            .map(|index| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * index as f32 / 399.0).cos())
            .collect();
        let mut mel = vec![0.0_f32; SORTFORMER_MEL_BINS * REAL_FFT_512_BINS];
        for band in 0..SORTFORMER_MEL_BINS {
            mel[band * REAL_FFT_512_BINS + (band * 2).min(REAL_FFT_512_BINS - 1)] = 1.0;
        }
        let package = synthetic_package(&window, WINDOW_SHAPE, &mel, MEL_SHAPE);
        let window_hash = sha256_hex(&f32_bytes(&window));
        let window_contract = TensorContract {
            name: SORTFORMER_WINDOW_TENSOR_NAME,
            shape: WINDOW_SHAPE,
            sha256: &window_hash,
        };
        let mel_hash = sha256_hex(&f32_bytes(&mel));
        let mel_contract = TensorContract {
            name: SORTFORMER_MEL_TENSOR_NAME,
            shape: MEL_SHAPE,
            sha256: &mel_hash,
        };
        SortformerFrontend::from_package_contracts(&package, window_contract, mel_contract, &|| {
            Ok(())
        })
        .expect("synthetic frontend")
    }

    fn sparse_frontend(window_index: usize, mel_weight: f32) -> SortformerFrontend {
        let mut window = [0.0_f32; SORTFORMER_WINDOW_SAMPLES];
        window[window_index] = 1.0;
        let mut mel_filterbank = vec![0.0_f32; SORTFORMER_MEL_BINS * REAL_FFT_512_BINS];
        mel_filterbank[0] = mel_weight;
        let mel_nonzero_ranges =
            mel_nonzero_ranges(&mel_filterbank).expect("synthetic filterbank shape");
        SortformerFrontend {
            window,
            mel_filterbank,
            mel_nonzero_ranges,
        }
    }

    fn dense_mel_energy(power: &[f32], filter: &[f32]) -> f32 {
        let mut energy = 0.0_f32;
        for frequency in 0..REAL_FFT_512_BINS {
            energy += power[frequency] * filter[frequency];
        }
        energy
    }

    #[test]
    fn synthetic_buffers_load_and_production_hashes_reject_them() {
        let frontend = synthetic_frontend();
        assert_eq!(frontend.window.len(), SORTFORMER_WINDOW_SAMPLES);
        assert_eq!(
            frontend.mel_filterbank.len(),
            SORTFORMER_MEL_BINS * REAL_FFT_512_BINS
        );

        let window = vec![1.0_f32; SORTFORMER_WINDOW_SAMPLES];
        let mel = vec![0.0_f32; SORTFORMER_MEL_BINS * REAL_FFT_512_BINS];
        let package = synthetic_package(&window, WINDOW_SHAPE, &mel, MEL_SHAPE);
        let error = SortformerFrontend::from_package(&package)
            .expect_err("untrusted synthetic tensors must not pass production pins");
        assert!(error.to_string().contains("tensor_hash"));
    }

    #[test]
    fn frontend_silence_is_exact_guard_log_and_is_mel_major() {
        let frontend = synthetic_frontend();
        let silence = vec![0.0_f32; 2 * SORTFORMER_HOP_SAMPLES];
        let output = frontend
            .compute(SortformerPcm::mono_16khz(&silence))
            .expect("silence frontend");
        assert_eq!(output.mel_bins, SORTFORMER_MEL_BINS);
        assert_eq!(output.valid_frames, 2);
        assert_eq!(output.data.len(), SORTFORMER_MEL_BINS * 2);
        let expected = LOG_GUARD.ln();
        assert!(output.data.iter().all(|&value| value == expected));
    }

    #[test]
    fn sparse_projection_is_bit_exact_to_dense_reference() {
        let power: Vec<f32> = (0..REAL_FFT_512_BINS)
            .map(|frequency| {
                let value = (frequency as f32 + 0.25) / REAL_FFT_512_BINS as f32;
                value * value
            })
            .collect();
        let mut filterbank = vec![0.0_f32; SORTFORMER_MEL_BINS * REAL_FFT_512_BINS];

        // Band zero is intentionally empty. Other bands exercise leading,
        // trailing, and interior exact zeros without changing the ascending
        // order of retained multiply-adds.
        filterbank[REAL_FFT_512_BINS + 3] = 0.25;
        filterbank[REAL_FFT_512_BINS + 9] = 0.5;
        filterbank[REAL_FFT_512_BINS + 17] = 0.25;
        filterbank[2 * REAL_FFT_512_BINS] = 1.0;
        filterbank[2 * REAL_FFT_512_BINS + REAL_FFT_512_BINS - 1] = 0.5;
        filterbank[3 * REAL_FFT_512_BINS + 4] = -0.0;
        for mel in 4..SORTFORMER_MEL_BINS {
            let center = (mel * 2).min(REAL_FFT_512_BINS - 1);
            filterbank[mel * REAL_FFT_512_BINS + center] = 1.0 / (mel + 1) as f32;
        }

        let ranges = mel_nonzero_ranges(&filterbank).expect("synthetic filterbank ranges");
        assert_eq!(ranges[0], (0, 0));
        assert_eq!(ranges[1], (3, 18));
        assert_eq!(ranges[3], (0, 0));
        for (mel, &range) in ranges.iter().enumerate() {
            let start = mel * REAL_FFT_512_BINS;
            let filter = &filterbank[start..start + REAL_FFT_512_BINS];
            let dense = (dense_mel_energy(&power, filter) + LOG_GUARD).ln();
            let sparse = (sparse_mel_energy(&power, filter, range) + LOG_GUARD).ln();
            assert_eq!(
                sparse.to_bits(),
                dense.to_bits(),
                "mel band {mel} diverged from the dense reference"
            );
        }
    }

    #[test]
    fn frontend_geometry_has_one_physical_tail_then_crops() {
        assert_eq!(frame_geometry(1).expect("geometry"), (0, 1));
        assert_eq!(frame_geometry(159).expect("geometry"), (0, 1));
        assert_eq!(frame_geometry(160).expect("geometry"), (1, 2));
        assert_eq!(frame_geometry(321).expect("geometry"), (2, 3));

        let frontend = synthetic_frontend();
        let samples = [0.25_f32; 159];
        let output = frontend
            .compute(SortformerPcm::mono_16khz(&samples))
            .expect("short clip");
        assert_eq!(output.valid_frames, 0);
        assert!(output.data.is_empty());
    }

    #[test]
    fn centered_window_and_preemphasis_have_explicit_sample_mapping() {
        // In physical frame zero, stored-window coordinate 201 addresses PCM
        // sample 1 because the 400-sample window is centered at coordinate 200.
        // The selected DC-only mel lane therefore sees preemphasis
        // `x[1] - 0.97*x[0]`, while every other lane remains at the log guard.
        let frontend = sparse_frontend(WINDOW_CENTER + 1, 1.0);
        let mut samples = [0.0_f32; SORTFORMER_HOP_SAMPLES];
        samples[0] = 1.0;
        samples[1] = 1.0;
        let output = frontend
            .compute(SortformerPcm::mono_16khz(&samples))
            .expect("centered frontend");
        let emphasized = samples[1] - PREEMPHASIS * samples[0];
        let magnitude = (emphasized.powi(2) + 0.0_f32.powi(2)).sqrt();
        let expected = (magnitude.powi(2) + LOG_GUARD).ln();
        assert_eq!(output.data[0].to_bits(), expected.to_bits());
        for mel in 1..SORTFORMER_MEL_BINS {
            assert_eq!(
                output.data[mel * output.valid_frames].to_bits(),
                LOG_GUARD.ln().to_bits()
            );
        }

        let first_sample_frontend = sparse_frontend(WINDOW_CENTER, 1.0);
        let mut first_sample_only = [0.0_f32; SORTFORMER_HOP_SAMPLES];
        first_sample_only[0] = 0.5;
        let output = first_sample_frontend
            .compute(SortformerPcm::mono_16khz(&first_sample_only))
            .expect("first sample preservation");
        let first_magnitude = (0.5_f32.powi(2) + 0.0_f32.powi(2)).sqrt();
        let first_expected = (first_magnitude.powi(2) + LOG_GUARD).ln();
        assert_eq!(output.data[0].to_bits(), first_expected.to_bits());
    }

    #[test]
    fn physical_tail_is_computed_but_cropped_from_canonical_output() {
        // At frame zero, window coordinate 50 points left of the clip. At the
        // physical F+1 tail it points at PCM sample 10. A positive filter proves
        // the tail cannot leak into the sole returned frame.
        let mut samples = [0.0_f32; SORTFORMER_HOP_SAMPLES];
        samples[10] = 1.0;
        let positive = sparse_frontend(50, 1.0);
        let output = positive
            .compute(SortformerPcm::mono_16khz(&samples))
            .expect("cropped physical tail");
        assert_eq!(output.valid_frames, 1);
        assert_eq!(output.data[0].to_bits(), LOG_GUARD.ln().to_bits());
    }

    #[test]
    fn output_allocation_is_fallible_and_checkpointed() {
        let error = zeroed_output_buffer(usize::MAX, &|| Ok(()))
            .expect_err("an impossible capacity must fail without aborting");
        assert!(error.to_string().contains("output_allocation"));

        let checkpoints = AtomicUsize::new(0);
        let error = zeroed_output_buffer(ZERO_FILL_CHUNK_VALUES * 2 + 1, &|| {
            let call = checkpoints.fetch_add(1, Ordering::SeqCst);
            if call >= 2 {
                Err(FwError::Cancelled("test cancellation".to_owned()))
            } else {
                Ok(())
            }
        })
        .expect_err("zero-fill must remain cancellable");
        assert!(matches!(error, FwError::Cancelled(_)));
    }

    #[test]
    fn frontend_replay_is_bit_deterministic() {
        let frontend = synthetic_frontend();
        let samples: Vec<f32> = (0..777)
            .map(|index| ((index as f32 * 0.03125).sin() * 0.75).clamp(-1.0, 1.0))
            .collect();
        let first = frontend
            .compute(SortformerPcm::mono_16khz(&samples))
            .expect("first replay");
        let second = frontend
            .compute(SortformerPcm::mono_16khz(&samples))
            .expect("second replay");
        assert_eq!(first, second);
    }

    #[test]
    fn frontend_rejects_bad_pcm_and_redacts_derived_features() {
        let frontend = synthetic_frontend();
        for samples in [Vec::new(), vec![f32::NAN], vec![1.000_001]] {
            assert!(
                frontend
                    .compute(SortformerPcm::mono_16khz(&samples))
                    .is_err()
            );
        }
        let samples = [0.0_f32; 160];
        let rate_error = frontend
            .compute(SortformerPcm::new(&samples, 8_000, 1))
            .expect_err("wrong sample rate");
        assert!(rate_error.to_string().contains("sample_rate"));
        let channel_error = frontend
            .compute(SortformerPcm::new(&samples, 16_000, 2))
            .expect_err("interleaved stereo is outside the mono contract");
        assert!(channel_error.to_string().contains("channels"));
        let pcm_debug = format!("{:?}", SortformerPcm::mono_16khz(&samples));
        assert!(pcm_debug.contains("<redacted>"));
        assert!(!pcm_debug.contains("[0.0"));

        let output = frontend
            .compute(SortformerPcm::mono_16khz(&samples))
            .expect("frontend");
        let rendered = format!("{output:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains(&LOG_GUARD.ln().to_string()));
    }

    #[test]
    fn whole_file_resource_ceiling_accepts_two_hours_and_rejects_one_more_sample() {
        // Exercise the checked contract without allocating a 439 MiB PCM test
        // vector. Value scanning is covered independently by the PCM tests.
        validate_pcm_contract(
            SORTFORMER_MAX_AUDIO_SAMPLES,
            SORTFORMER_SAMPLE_RATE_HZ,
            SORTFORMER_CHANNELS,
        )
        .expect("exact two-hour ceiling");
        let error = validate_pcm_contract(
            SORTFORMER_MAX_AUDIO_SAMPLES + 1,
            SORTFORMER_SAMPLE_RATE_HZ,
            SORTFORMER_CHANNELS,
        )
        .expect_err("one sample beyond resource ceiling");
        assert!(error.to_string().contains("pcm_duration"));
    }

    #[test]
    fn frontend_honors_mid_computation_cancellation() {
        let frontend = synthetic_frontend();
        let calls = AtomicUsize::new(0);
        let checkpoint = || {
            // Eight checkpoints validate the 32,000 samples. Cancellation on
            // call ten therefore lands after FFT work has begun.
            if calls.fetch_add(1, Ordering::SeqCst) == 10 {
                Err(FwError::Cancelled("private detail".to_owned()))
            } else {
                Ok(())
            }
        };
        let samples = [0.125_f32; 32_000];
        let error = frontend
            .compute_with_checkpoint(SortformerPcm::mono_16khz(&samples), &checkpoint)
            .expect_err("cancelled frontend");
        assert!(matches!(error, FwError::Cancelled(_)));
        assert!(!error.to_string().contains("private detail"));
    }

    #[test]
    fn frontend_honors_cancellation_during_buffer_load() {
        let window = vec![1.0_f32; SORTFORMER_WINDOW_SAMPLES];
        let mel = vec![0.0_f32; SORTFORMER_MEL_BINS * REAL_FFT_512_BINS];
        let package = synthetic_package(&window, WINDOW_SHAPE, &mel, MEL_SHAPE);
        let window_hash = sha256_hex(&f32_bytes(&window));
        let mel_hash = sha256_hex(&f32_bytes(&mel));
        let calls = AtomicUsize::new(0);
        let checkpoint = || {
            if calls.fetch_add(1, Ordering::SeqCst) == 1 {
                Err(FwError::Cancelled("private load detail".to_owned()))
            } else {
                Ok(())
            }
        };
        let error = SortformerFrontend::from_package_contracts(
            &package,
            TensorContract {
                name: SORTFORMER_WINDOW_TENSOR_NAME,
                shape: WINDOW_SHAPE,
                sha256: &window_hash,
            },
            TensorContract {
                name: SORTFORMER_MEL_TENSOR_NAME,
                shape: MEL_SHAPE,
                sha256: &mel_hash,
            },
            &checkpoint,
        )
        .expect_err("cancelled frontend load");
        assert!(matches!(error, FwError::Cancelled(_)));
        assert!(!error.to_string().contains("private load detail"));
    }

    #[test]
    fn tensor_shape_and_nonfinite_payloads_fail_closed() {
        let window = vec![1.0_f32; SORTFORMER_WINDOW_SAMPLES];
        let mel = vec![0.0_f32; SORTFORMER_MEL_BINS * REAL_FFT_512_BINS];
        let bad_shape = synthetic_package(&window, &[200, 2], &mel, MEL_SHAPE);
        let contract_hash = sha256_hex(&f32_bytes(&window));
        let window_contract = TensorContract {
            name: SORTFORMER_WINDOW_TENSOR_NAME,
            shape: WINDOW_SHAPE,
            sha256: &contract_hash,
        };
        let mel_hash = sha256_hex(&f32_bytes(&mel));
        let mel_contract = TensorContract {
            name: SORTFORMER_MEL_TENSOR_NAME,
            shape: MEL_SHAPE,
            sha256: &mel_hash,
        };
        let error = SortformerFrontend::from_package_contracts(
            &bad_shape,
            window_contract,
            mel_contract,
            &|| Ok(()),
        )
        .expect_err("wrong window shape");
        assert!(error.to_string().contains("tensor_shape"));

        let mut mutated_window = window.clone();
        mutated_window[0] = 0.5;
        let mutated_package = synthetic_package(&mutated_window, WINDOW_SHAPE, &mel, MEL_SHAPE);
        let error = SortformerFrontend::from_package_contracts(
            &mutated_package,
            window_contract,
            mel_contract,
            &|| Ok(()),
        )
        .expect_err("payload mutation must fail a previously pinned hash");
        assert!(error.to_string().contains("tensor_hash"));

        let half_window = vec![0_u8; SORTFORMER_WINDOW_SAMPLES * 2];
        let mel_bytes = f32_bytes(&mel);
        let half_package =
            synthetic_package_raw("F16", &half_window, WINDOW_SHAPE, &mel_bytes, MEL_SHAPE);
        let error = SortformerFrontend::from_package_contracts(
            &half_package,
            TensorContract {
                name: SORTFORMER_WINDOW_TENSOR_NAME,
                shape: WINDOW_SHAPE,
                sha256: &sha256_hex(&half_window),
            },
            mel_contract,
            &|| Ok(()),
        )
        .expect_err("half-precision frontend buffer must fail closed");
        assert!(error.to_string().contains("tensor_dtype"));

        let mut nonfinite_window = window;
        nonfinite_window[17] = f32::INFINITY;
        let package = synthetic_package(&nonfinite_window, WINDOW_SHAPE, &mel, MEL_SHAPE);
        let nonfinite_hash = sha256_hex(&f32_bytes(&nonfinite_window));
        let nonfinite_contract = TensorContract {
            name: SORTFORMER_WINDOW_TENSOR_NAME,
            shape: WINDOW_SHAPE,
            sha256: &nonfinite_hash,
        };
        let error = SortformerFrontend::from_package_contracts(
            &package,
            nonfinite_contract,
            mel_contract,
            &|| Ok(()),
        )
        .expect_err("non-finite window");
        assert!(error.to_string().contains("tensor_nonfinite"));
    }

    #[test]
    fn parity_metrics_use_f64_aggregates_and_bit_mismatches() {
        let metrics = frontend_parity_metrics("synthetic", &[1.0, 2.0], &[1.0, 1.0])
            .expect("finite equal-shape metrics");
        assert_eq!(metrics.compared_values, 2);
        assert_eq!(metrics.mismatch_count, 1);
        assert!(!metrics.byte_exact);
        assert_eq!(metrics.max_abs_diff, 1.0);
        assert_eq!(metrics.mean_abs_diff, 0.5);
        assert_eq!(metrics.relative_l2, (1.0_f64 / 5.0).sqrt());

        let error = frontend_parity_metrics("nonfinite", &[f32::NAN], &[0.0])
            .expect_err("non-finite comparison must fail closed");
        assert!(error.to_string().contains("parity_nonfinite"));
    }

    #[test]
    fn synthetic_frontend_envelope_is_frozen_to_binary_powers() {
        assert_eq!(SORTFORMER_FRONTEND_MAX_ABS_DIFF, 2.0_f64.powi(-12));
        assert_eq!(SORTFORMER_FRONTEND_MAX_MEAN_ABS_DIFF, 2.0_f64.powi(-17));
        assert_eq!(SORTFORMER_FRONTEND_MAX_RELATIVE_L2, 2.0_f64.powi(-19));
    }

    #[test]
    fn l2_cross_kernel_envelope_is_frozen_to_binary_powers() {
        assert_eq!(SORTFORMER_L2_MAX_ABS_DIFF, 2.0_f64.powi(-10));
        assert_eq!(SORTFORMER_L2_MAX_RELATIVE_L2, 2.0_f64.powi(-16));
        assert_eq!(SORTFORMER_L3_INPUT_MAX_ABS_DIFF, 2.0_f64.powi(-5));
        assert_eq!(SORTFORMER_L3_INPUT_MAX_RELATIVE_L2, 2.0_f64.powi(-16));
        assert_eq!(SORTFORMER_L3_FFN_MAX_ABS_DIFF, 2.0_f64.powi(-8));
        assert_eq!(SORTFORMER_L3_FFN_MAX_RELATIVE_L2, 2.0_f64.powi(-14));
        assert_eq!(SORTFORMER_L3_QKV_MAX_ABS_DIFF, 2.0_f64.powi(-7));
        assert_eq!(SORTFORMER_L3_QKV_MAX_RELATIVE_L2, 2.0_f64.powi(-13));
        assert_eq!(SORTFORMER_L3_ATTENTION_MAX_ABS_DIFF, 2.0_f64.powi(-6));
        assert_eq!(SORTFORMER_L3_ATTENTION_MAX_RELATIVE_L2, 2.0_f64.powi(-11));
        assert_eq!(SORTFORMER_L3_CONV_MAX_ABS_DIFF, 2.0_f64.powi(-5));
        assert_eq!(SORTFORMER_L3_CONV_MAX_RELATIVE_L2, 2.0_f64.powi(-12));
        assert_eq!(SORTFORMER_L3_BLOCK_MAX_ABS_DIFF, 2.0_f64.powi(-4));
        assert_eq!(SORTFORMER_L3_BLOCK_MAX_RELATIVE_L2, 2.0_f64.powi(-10));
        assert_eq!(SORTFORMER_L4_MAX_ABS_DIFF, 2.0_f64.powi(-4));
        assert_eq!(SORTFORMER_L4_MAX_RELATIVE_L2, 2.0_f64.powi(-10));
        assert_eq!(SORTFORMER_L5_MAX_ABS_DIFF, 2.0_f64.powi(-4));
        assert_eq!(SORTFORMER_L5_MAX_RELATIVE_L2, 2.0_f64.powi(-10));
        assert_eq!(SORTFORMER_L6_MAX_ABS_DIFF, 2.0_f64.powi(-4));
        assert_eq!(SORTFORMER_L6_MAX_RELATIVE_L2, 2.0_f64.powi(-10));
    }

    #[test]
    fn l6_all_disabled_cache_uses_silence_profile_deterministically() {
        let frames = SORTFORMER_SPEAKER_CACHE_FRAMES + 1;
        let mut state = SortformerStreamingState::new();
        state.speaker_cache = Mat::from_vec(
            frames,
            SORTFORMER_ENCODER_WIDTH,
            (0..frames * SORTFORMER_ENCODER_WIDTH)
                .map(|index| (index % 97) as f32 * 0.001)
                .collect(),
        );
        state.speaker_cache_predictions = Some(Mat::from_vec(
            frames,
            SORTFORMER_SPEAKER_LANES,
            vec![0.0; frames * SORTFORMER_SPEAKER_LANES],
        ));
        state.mean_silence_embedding.fill(0.25);
        let mut repeated = state.clone();

        compress_speaker_cache(&mut state).expect("all-disabled cache compresses");
        compress_speaker_cache(&mut repeated).expect("repeated compression succeeds");
        assert_eq!(state, repeated);
        assert_eq!(state.speaker_cache.rows, SORTFORMER_SPEAKER_CACHE_FRAMES);
        assert!(state.speaker_cache.data.iter().all(|&value| value == 0.25));
        assert!(
            state
                .speaker_cache_predictions
                .as_ref()
                .expect("compression creates predictions")
                .data
                .iter()
                .all(|&value| value == 0.0)
        );
        validate_streaming_state(&state).expect("compressed state remains valid");
    }

    #[test]
    fn l6_silence_profile_counts_only_low_total_activity() {
        let mut state = SortformerStreamingState::new();
        let embeddings = Mat::from_vec(
            2,
            SORTFORMER_ENCODER_WIDTH,
            [
                vec![1.0; SORTFORMER_ENCODER_WIDTH],
                vec![3.0; SORTFORMER_ENCODER_WIDTH],
            ]
            .concat(),
        );
        let predictions = Mat::from_vec(
            2,
            SORTFORMER_SPEAKER_LANES,
            vec![0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1],
        );
        update_silence_profile(&mut state, &embeddings, &predictions)
            .expect("finite aligned silence inputs succeed");
        assert_eq!(state.silence_frames, 1);
        assert!(
            state
                .mean_silence_embedding
                .iter()
                .all(|&value| value == 1.0)
        );
    }

    #[test]
    fn l7_threshold_change_and_overlap_semantics_are_exact() {
        let probabilities = [0.5, 0.49, 0.0, 0.0, 0.51, 0.5, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4];
        let output = sortformer_activity_output(&probabilities, 3).expect("valid L7 input");
        assert_eq!(output.activity, vec![1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(output.speech, vec![1, 1, 0]);
        assert_eq!(output.overlap, vec![0, 1, 0]);
        assert_eq!(output.change_indices, vec![[0, 0], [0, 1]]);
    }

    #[test]
    fn l7_and_l8_reject_empty_malformed_overflowing_and_nonfinite_inputs() {
        for (probabilities, frames) in [
            (Vec::new(), 0),
            (vec![0.0; SORTFORMER_SPEAKER_LANES - 1], 1),
            (vec![f32::NAN; SORTFORMER_SPEAKER_LANES], 1),
        ] {
            assert!(sortformer_activity_output(&probabilities, frames).is_err());
            assert!(sortformer_speaker_turns(&probabilities, frames).is_err());
        }
        assert!(sortformer_activity_output(&[], usize::MAX).is_err());
        assert!(sortformer_speaker_turns(&[], usize::MAX).is_err());
    }

    #[test]
    fn l8_uses_strict_edges_and_last_repeated_frame_endpoint() {
        let probabilities = [0.5, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0];
        let turns = sortformer_speaker_turns(&probabilities, 3).expect("valid L8 input");
        assert_eq!(
            turns,
            vec![SortformerSpeakerTurn {
                start_seconds: 0.08,
                end_seconds: (23.0_f64 * 0.01) as f32,
                speaker: 0,
            }]
        );
    }

    #[test]
    fn production_turns_are_clamped_to_physical_audio_duration() {
        let mut turns = vec![
            SortformerSpeakerTurn {
                start_seconds: 0.5,
                end_seconds: 1.25,
                speaker: 0,
            },
            SortformerSpeakerTurn {
                start_seconds: 1.1,
                end_seconds: 1.3,
                speaker: 1,
            },
        ];
        clamp_sortformer_turns_to_duration(&mut turns, 1.0);
        assert_eq!(
            turns,
            vec![SortformerSpeakerTurn {
                start_seconds: 0.5,
                end_seconds: 1.0,
                speaker: 0,
            }]
        );
    }

    #[test]
    #[ignore = "requires operator-local public truth pack"]
    fn operator_local_public_l7_l8_exact_postprocessing_parity() {
        let activation_receipt =
            std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_RECEIPT")
                .expect("set FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_RECEIPT");
        let activation_package =
            std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_PACKAGE")
                .expect("set FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_PACKAGE");
        let activations =
            crate::sortformer_conformance::load_verified_sortformer_public_activation_pack(
                std::path::Path::new(&activation_receipt),
                std::path::Path::new(&activation_package),
            )
            .expect("operator-local public activation admission");
        for fixture in [
            "hiyis_exact_two_chunks",
            "mevkw_overlap_two_speakers",
            "syiwe_complete_three_speakers",
        ] {
            let probabilities = complete_public_mat(
                &activations,
                &format!("fixture.{fixture}.l5.final_probabilities_f32"),
                SORTFORMER_SPEAKER_LANES,
            );
            let output = sortformer_activity_output(&probabilities.data, probabilities.rows)
                .expect("public L7 post-processing");
            for (suffix, shape, values) in [
                (
                    "activity_i64",
                    vec![1, output.frames, SORTFORMER_SPEAKER_LANES],
                    output.activity.as_slice(),
                ),
                (
                    "speech_i64",
                    vec![1, output.frames],
                    output.speech.as_slice(),
                ),
                (
                    "overlap_i64",
                    vec![1, output.frames],
                    output.overlap.as_slice(),
                ),
            ] {
                let metrics = compare_public_i64_probe(
                    &activations,
                    &format!("fixture.{fixture}.l7.{suffix}"),
                    &shape,
                    values,
                )
                .expect("public L7 seam is exact");
                assert!(metrics.byte_exact);
            }
            let changes = output.change_indices.concat();
            let metrics = compare_public_i64_probe(
                &activations,
                &format!("fixture.{fixture}.l7.change_indices_i64"),
                &[output.change_indices.len(), 2],
                &changes,
            )
            .expect("public L7 changes are exact");
            assert!(metrics.byte_exact);

            let turns = sortformer_speaker_turns(&probabilities.data, probabilities.rows)
                .expect("public L8 post-processing");
            let flattened = turns
                .iter()
                .flat_map(|turn| [turn.start_seconds, turn.end_seconds, turn.speaker as f32])
                .collect::<Vec<_>>();
            let metrics = compare_public_f32_probe(
                &activations,
                &format!("fixture.{fixture}.l8.turns_f32"),
                &[turns.len(), 3],
                &flattened,
                0.0,
                0.0,
            )
            .expect("public L8 turns are exact");
            assert!(metrics.byte_exact);
        }
    }

    #[test]
    #[ignore = "requires operator-local weights, public truth pack, and external VoxConverse root"]
    fn operator_local_public_session_mevkw_end_to_end() {
        let model_receipt = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_RECEIPT")
            .expect("set FRANKEN_WHISPER_SORTFORMER_RECEIPT");
        let model_package = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PACKAGE")
            .expect("set FRANKEN_WHISPER_SORTFORMER_PACKAGE");
        let activation_receipt =
            std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_RECEIPT")
                .expect("set FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_RECEIPT");
        let activation_package =
            std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_PACKAGE")
                .expect("set FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_PACKAGE");
        let public_root = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PUBLIC_CORPUS_ROOT")
            .expect("set FRANKEN_WHISPER_SORTFORMER_PUBLIC_CORPUS_ROOT");
        let model = crate::sortformer_conformance::load_verified_sortformer_package(
            std::path::Path::new(&model_receipt),
            std::path::Path::new(&model_package),
        )
        .expect("operator-local converted model admission");
        let activations =
            crate::sortformer_conformance::load_verified_sortformer_public_activation_pack(
                std::path::Path::new(&activation_receipt),
                std::path::Path::new(&activation_package),
            )
            .expect("operator-local public activation admission");
        let fixture = activations
            .receipt()
            .fixtures
            .iter()
            .find(|fixture| fixture.name == "mevkw_overlap_two_speakers")
            .expect("mevkw fixture");
        let wav_path = std::path::Path::new(&public_root)
            .join("audio")
            .join(format!("{}.wav", fixture.recording_id));
        let wav_bytes = std::fs::read(&wav_path).expect("read external public WAV");
        assert_eq!(sha256_hex(&wav_bytes), fixture.audio_sha256);
        let mut reader = hound::WavReader::new(std::io::Cursor::new(wav_bytes))
            .expect("parse external public WAV");
        let all_samples = reader
            .samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .expect("decode public PCM16");
        let start = usize::try_from(fixture.start_sample).expect("fixture start fits usize");
        let count = usize::try_from(fixture.sample_count).expect("fixture count fits usize");
        let pcm = all_samples[start..start + count]
            .iter()
            .map(|&sample| f32::from(sample) / 32_768.0)
            .collect::<Vec<_>>();
        let load_started = std::time::Instant::now();
        let session = SortformerSession::from_verified_package(&model).expect("native session");
        let load_seconds = load_started.elapsed().as_secs_f64();
        let inference_started = std::time::Instant::now();
        let output = session
            .diarize(SortformerPcm::mono_16khz(&pcm))
            .expect("native whole-recording diarization");
        let inference_seconds = inference_started.elapsed().as_secs_f64();
        let audio_seconds = pcm.len() as f64 / SORTFORMER_SAMPLE_RATE_HZ as f64;
        let probability_metrics = compare_public_f32_probe(
            &activations,
            "fixture.mevkw_overlap_two_speakers.l5.final_probabilities_f32",
            &[1, output.frames, SORTFORMER_SPEAKER_LANES],
            &output.probabilities,
            SORTFORMER_L5_MAX_ABS_DIFF,
            SORTFORMER_L5_MAX_RELATIVE_L2,
        )
        .expect("production session probabilities stay inside the L5 gate");
        assert!(
            compare_public_i64_probe(
                &activations,
                "fixture.mevkw_overlap_two_speakers.l7.activity_i64",
                &[1, output.frames, SORTFORMER_SPEAKER_LANES],
                &output.activity.activity,
            )
            .expect("production L7 activity")
            .byte_exact
        );
        let raw_turns = sortformer_speaker_turns(&output.probabilities, output.frames)
            .expect("raw production L8 postprocessing");
        let turns = raw_turns
            .iter()
            .flat_map(|turn| [turn.start_seconds, turn.end_seconds, turn.speaker as f32])
            .collect::<Vec<_>>();
        assert!(
            compare_public_f32_probe(
                &activations,
                "fixture.mevkw_overlap_two_speakers.l8.turns_f32",
                &[raw_turns.len(), 3],
                &turns,
                0.0,
                0.0,
            )
            .expect("production L8 turns")
            .byte_exact
        );
        assert!(
            output
                .turns
                .iter()
                .all(|turn| f64::from(turn.end_seconds) <= audio_seconds)
        );
        let clip_duration_ms = fixture
            .sample_count
            .checked_mul(1_000)
            .expect("bounded clip duration")
            / fixture.sample_rate_hz;
        let clip_start_ms = fixture
            .start_sample
            .checked_mul(1_000)
            .expect("bounded clip offset")
            / fixture.sample_rate_hz;
        let reference_turns = clipped_rttm_turns(
            &std::path::Path::new(&public_root)
                .join("annotations")
                .join(format!("{}.rttm", fixture.recording_id)),
            &fixture.recording_id,
            clip_start_ms,
            clip_duration_ms,
        );
        let hypothesis_turns = output
            .turns
            .iter()
            .filter_map(|turn| {
                let start_ms = ((f64::from(turn.start_seconds) * 1_000.0).round() as u64)
                    .min(clip_duration_ms);
                let end_ms =
                    ((f64::from(turn.end_seconds) * 1_000.0).round() as u64).min(clip_duration_ms);
                (start_ms < end_ms).then(|| crate::diarization::EvaluationTurn {
                    start_ms,
                    end_ms,
                    speaker: Some(format!("sortformer-lane-{}", turn.speaker)),
                    speaker_confidence: None,
                    overlap_suspected: output.turns.iter().any(|other| {
                        other.speaker != turn.speaker
                            && other.start_seconds < turn.end_seconds
                            && turn.start_seconds < other.end_seconds
                    }),
                })
            })
            .collect::<Vec<_>>();
        let reference = crate::diarization::DiarizationReferenceDocument {
            schema_version: crate::diarization::DIARIZATION_REFERENCE_SCHEMA_VERSION.to_owned(),
            recording_id: format!("{}-public-clip", fixture.recording_id),
            duration_ms: clip_duration_ms,
            turns: reference_turns,
            ignored_regions: Vec::new(),
            speaker_hints: Vec::new(),
            words: Vec::new(),
        };
        let hypothesis = crate::diarization::DiarizationHypothesisDocument {
            schema_version: crate::diarization::DIARIZATION_HYPOTHESIS_SCHEMA_VERSION.to_owned(),
            recording_id: reference.recording_id.clone(),
            duration_ms: clip_duration_ms,
            turns: hypothesis_turns,
            speaker_count_estimate: None,
            performance: Some(crate::diarization::EvaluationPerformanceObservation {
                audio_duration_ms: clip_duration_ms,
                wall_time_ms: u64::try_from(inference_started.elapsed().as_millis())
                    .unwrap_or(u64::MAX),
                peak_rss_bytes: 0,
            }),
        };
        let accuracy = crate::diarization::score_diarization_documents(
            &reference,
            &hypothesis,
            &crate::diarization::DiarizationScorerConfig::default(),
        )
        .expect("authoritative transcript-free public DER/JER score");
        let cancellation = session
            .diarize_with_checkpoint(SortformerPcm::mono_16khz(&pcm), &|| {
                Err(FwError::Cancelled("private cancellation detail".to_owned()))
            })
            .expect_err("production session must honor immediate cancellation");
        assert!(matches!(cancellation, FwError::Cancelled(_)));
        assert!(
            !cancellation
                .to_string()
                .contains("private cancellation detail")
        );
        eprintln!(
            "sortformer_session_e2e load_seconds={load_seconds:.6} inference_seconds={inference_seconds:.6} audio_seconds={audio_seconds:.6} rtf={:.6} probability_max_abs={:.9e} probability_relative_l2={:.9e} der={:.9} jer={:.9} miss_seconds={:.6} false_alarm_seconds={:.6} confusion_seconds={:.6}",
            inference_seconds / audio_seconds,
            probability_metrics.max_abs_diff,
            probability_metrics.relative_l2,
            accuracy.diarization.der.expect("defined DER"),
            accuracy.diarization.jer.expect("defined JER"),
            accuracy.diarization.missed_speech_sec,
            accuracy.diarization.false_alarm_sec,
            accuracy.diarization.speaker_confusion_sec,
        );
    }

    #[test]
    fn l6_duplicate_speaker_permutation_fails_closed() {
        let mut state = SortformerStreamingState::new();
        state.speaker_permutation = Some([0, 0, 2, 3]);
        let error = validate_streaming_state(&state)
            .expect_err("duplicate speaker permutation must be rejected");
        assert!(error.to_string().contains("duplicate lane"));
    }

    #[test]
    #[ignore = "requires operator-local converted weights and synthetic activation truth pack"]
    fn operator_local_synthetic_frontend_parity_is_within_frozen_envelope() {
        let model_receipt = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_RECEIPT")
            .expect("set FRANKEN_WHISPER_SORTFORMER_RECEIPT");
        let model_package = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PACKAGE")
            .expect("set FRANKEN_WHISPER_SORTFORMER_PACKAGE");
        let activation_receipt = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_ACTIVATION_RECEIPT")
            .expect("set FRANKEN_WHISPER_SORTFORMER_ACTIVATION_RECEIPT");
        let activation_package = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_ACTIVATION_PACKAGE")
            .expect("set FRANKEN_WHISPER_SORTFORMER_ACTIVATION_PACKAGE");
        let model = crate::sortformer_conformance::load_verified_sortformer_package(
            std::path::Path::new(&model_receipt),
            std::path::Path::new(&model_package),
        )
        .expect("operator-local converted model admission");
        let activations = crate::sortformer_conformance::load_verified_sortformer_activation_pack(
            std::path::Path::new(&activation_receipt),
            std::path::Path::new(&activation_package),
        )
        .expect("operator-local activation admission");
        let report = verify_sortformer_frontend_synthetic_parity(&model, &activations)
            .expect("native frontend must stay within the predeclared synthetic envelope");
        assert!(report.oracle_floor_byte_exact);
        assert_eq!(report.fixtures.len(), 4);
        for metrics in report.fixtures {
            eprintln!(
                "fixture={} compared={} mismatches={} byte_exact={} max_abs={:.9e} \
                 mean_abs={:.9e} relative_l2={:.9e}",
                metrics.fixture,
                metrics.compared_values,
                metrics.mismatch_count,
                metrics.byte_exact,
                metrics.max_abs_diff,
                metrics.mean_abs_diff,
                metrics.relative_l2,
            );
        }
    }

    #[test]
    #[ignore = "requires operator-local weights and public activation truth pack"]
    fn operator_local_l6_exact_score_compression_matches_public_oracle() {
        let model_receipt = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_RECEIPT")
            .expect("set FRANKEN_WHISPER_SORTFORMER_RECEIPT");
        let model_package = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PACKAGE")
            .expect("set FRANKEN_WHISPER_SORTFORMER_PACKAGE");
        let activation_receipt =
            std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_RECEIPT")
                .expect("set FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_RECEIPT");
        let activation_package =
            std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_PACKAGE")
                .expect("set FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_PACKAGE");
        let model = crate::sortformer_conformance::load_verified_sortformer_package(
            std::path::Path::new(&model_receipt),
            std::path::Path::new(&model_package),
        )
        .expect("operator-local converted model admission");
        let activations =
            crate::sortformer_conformance::load_verified_sortformer_public_activation_pack(
                std::path::Path::new(&activation_receipt),
                std::path::Path::new(&activation_package),
            )
            .expect("operator-local public activation admission");
        let reference =
            SortformerF32Facade::from_verified_package(&model).expect("authenticated graph");
        let fixture = "iqtde_complete_four_speakers";
        let step = 2usize;
        let exact_cache_predictions = complete_public_mat(
            &activations,
            &format!("fixture.{fixture}.step.{step:03}.l6.before.spkcache_preds"),
            SORTFORMER_SPEAKER_LANES,
        );
        let exact_probabilities = complete_public_mat(
            &activations,
            &format!("fixture.{fixture}.step.{step:03}.l5.probabilities"),
            SORTFORMER_SPEAKER_LANES,
        );
        let transition = &activations.receipt().streaming_transitions[fixture][step];
        let left = usize::try_from(transition.left_offset).expect("left offset fits")
            / SORTFORMER_SUBSAMPLING_FACTOR;
        let right = usize::try_from(transition.right_offset)
            .expect("right offset fits")
            .div_ceil(SORTFORMER_SUBSAMPLING_FACTOR);
        let chunk_frames = exact_probabilities
            .rows
            .checked_sub(SORTFORMER_SPEAKER_CACHE_FRAMES)
            .expect("exact probabilities include the cache prefix");
        let mut state = SortformerStreamingState::new();
        let mut cache_embeddings =
            vec![0.0; SORTFORMER_SPEAKER_CACHE_FRAMES * SORTFORMER_ENCODER_WIDTH];
        for frame in 0..SORTFORMER_SPEAKER_CACHE_FRAMES {
            cache_embeddings[frame * SORTFORMER_ENCODER_WIDTH] = frame as f32;
        }
        state.speaker_cache = Mat::from_vec(
            SORTFORMER_SPEAKER_CACHE_FRAMES,
            SORTFORMER_ENCODER_WIDTH,
            cache_embeddings,
        );
        state.speaker_cache_predictions = Some(exact_cache_predictions);
        state.fifo_predictions = Some(Mat::from_vec(0, SORTFORMER_SPEAKER_LANES, Vec::new()));
        let mut chunk_embeddings = vec![0.0; chunk_frames * SORTFORMER_ENCODER_WIDTH];
        for frame in 0..chunk_frames {
            chunk_embeddings[frame * SORTFORMER_ENCODER_WIDTH] =
                (SORTFORMER_SPEAKER_CACHE_FRAMES - left + frame) as f32;
        }
        let chunk = SortformerSubsamplingOutput {
            frames: chunk_frames,
            width: SORTFORMER_ENCODER_WIDTH,
            data: chunk_embeddings,
        };
        reference
            .update_streaming_state(&mut state, &chunk, &exact_probabilities, left, right)
            .expect("exact-score compression succeeds");
        let observed = state
            .speaker_cache_predictions
            .expect("compression produces cache predictions");
        let metrics = compare_public_f32_probe(
            &activations,
            &format!("fixture.{fixture}.step.{step:03}.l6.after.spkcache_preds"),
            &[1, SORTFORMER_SPEAKER_CACHE_FRAMES, SORTFORMER_SPEAKER_LANES],
            &observed.data,
            SORTFORMER_L6_MAX_ABS_DIFF,
            SORTFORMER_L6_MAX_RELATIVE_L2,
        )
        .expect("exact scores reproduce authenticated cache-prediction selection");
        eprintln!(
            "l6_exact_score_selection compared={} mismatches={} max_abs={:.9e} relative_l2={:.9e}",
            metrics.compared_values,
            metrics.mismatch_count,
            metrics.max_abs_diff,
            metrics.relative_l2,
        );
    }

    #[test]
    #[ignore = "requires operator-local weights, public truth pack, and external VoxConverse root"]
    fn operator_local_public_l1_l6_parity_is_within_frozen_envelopes() {
        let model_receipt = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_RECEIPT")
            .expect("set FRANKEN_WHISPER_SORTFORMER_RECEIPT");
        let model_package = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PACKAGE")
            .expect("set FRANKEN_WHISPER_SORTFORMER_PACKAGE");
        let activation_receipt =
            std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_RECEIPT")
                .expect("set FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_RECEIPT");
        let activation_package =
            std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_PACKAGE")
                .expect("set FRANKEN_WHISPER_SORTFORMER_PUBLIC_ACTIVATION_PACKAGE");
        let public_root = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PUBLIC_CORPUS_ROOT")
            .expect("set FRANKEN_WHISPER_SORTFORMER_PUBLIC_CORPUS_ROOT");
        let model = crate::sortformer_conformance::load_verified_sortformer_package(
            std::path::Path::new(&model_receipt),
            std::path::Path::new(&model_package),
        )
        .expect("operator-local converted model admission");
        let activations =
            crate::sortformer_conformance::load_verified_sortformer_public_activation_pack(
                std::path::Path::new(&activation_receipt),
                std::path::Path::new(&activation_package),
            )
            .expect("operator-local public activation admission");
        let frontend = SortformerFrontend::from_verified_package(&model)
            .expect("authenticated native frontend");
        let reference =
            SortformerF32Facade::from_verified_package(&model).expect("authenticated L2 graph");

        for fixture in &activations.receipt().fixtures {
            let wav_path = std::path::Path::new(&public_root)
                .join("audio")
                .join(format!("{}.wav", fixture.recording_id));
            let wav_bytes = std::fs::read(&wav_path).expect("read external public WAV");
            assert_eq!(sha256_hex(&wav_bytes), fixture.audio_sha256);
            let mut reader = hound::WavReader::new(std::io::Cursor::new(wav_bytes))
                .expect("parse external public WAV");
            let spec = reader.spec();
            assert_eq!(spec.channels, 1);
            assert_eq!(spec.sample_rate, SORTFORMER_SAMPLE_RATE_HZ as u32);
            assert_eq!(spec.bits_per_sample, 16);
            assert_eq!(spec.sample_format, hound::SampleFormat::Int);
            let all_samples = reader
                .samples::<i16>()
                .collect::<Result<Vec<_>, _>>()
                .expect("decode public PCM16");
            let start = usize::try_from(fixture.start_sample).expect("fixture start fits usize");
            let count = usize::try_from(fixture.sample_count).expect("fixture count fits usize");
            let end = start
                .checked_add(count)
                .expect("fixture end does not overflow");
            let pcm = all_samples
                .get(start..end)
                .expect("fixture lies within public recording")
                .iter()
                .map(|&sample| f32::from(sample) / 32_768.0)
                .collect::<Vec<_>>();
            let features = frontend
                .compute(SortformerPcm::mono_16khz(&pcm))
                .expect("native public frontend");
            assert_eq!(features.valid_frames as u64, fixture.valid_frames);

            let transitions = activations
                .receipt()
                .streaming_transitions
                .get(&fixture.name)
                .expect("fixture transitions");
            let mut central_start = 0usize;
            let mut streaming_state = SortformerStreamingState::new();
            let mut compared_neural_steps = 0usize;
            for transition in transitions {
                let step = usize::try_from(transition.step).expect("step fits usize");
                let before_state = reference
                    .verify_public_l6_state(
                        &activations,
                        &fixture.name,
                        step,
                        "before",
                        &streaming_state,
                    )
                    .expect("native L6 before-state matches the authenticated boundary");
                for seam in before_state.f32_seams {
                    eprintln!(
                        "stage={} compared={} mismatches={} byte_exact={} max_abs={:.9e} \
                         relative_l2={:.9e} abs_limit={:.9e} relative_limit={:.9e}",
                        seam.stage,
                        seam.compared_values,
                        seam.mismatch_count,
                        seam.byte_exact,
                        seam.max_abs_diff,
                        seam.relative_l2,
                        seam.accepted_abs_tolerance,
                        seam.accepted_relative_l2,
                    );
                }
                for seam in before_state.i64_seams {
                    eprintln!(
                        "stage={} compared={} mismatches={} byte_exact={}",
                        seam.stage, seam.compared_values, seam.mismatch_count, seam.byte_exact,
                    );
                }
                let left = usize::try_from(transition.left_offset).expect("left offset");
                let right = usize::try_from(transition.right_offset).expect("right offset");
                let input_frames =
                    usize::try_from(transition.input_feature_frames).expect("input frames");
                let central_frames = input_frames
                    .checked_sub(left)
                    .and_then(|value| value.checked_sub(right))
                    .expect("context fits input chunk");
                let chunk_start = central_start
                    .checked_sub(left)
                    .expect("left context exists");
                let chunk_end = central_start
                    .checked_add(central_frames)
                    .and_then(|value| value.checked_add(right))
                    .expect("chunk end does not overflow");
                assert!(chunk_end <= features.valid_frames);
                let mut time_major = Vec::with_capacity(input_frames * SORTFORMER_MEL_BINS);
                for frame in chunk_start..chunk_end {
                    for mel in 0..SORTFORMER_MEL_BINS {
                        time_major.push(features.data[mel * features.valid_frames + frame]);
                    }
                }
                let report = reference
                    .verify_public_l2_chunk_parity(
                        &activations,
                        &fixture.name,
                        usize::try_from(transition.step).expect("step fits usize"),
                        &time_major,
                        input_frames,
                    )
                    .expect("native L2 seams stay within predeclared envelope");
                for seam in report.seams {
                    eprintln!(
                        "stage={} compared={} mismatches={} byte_exact={} max_abs={:.9e} \
                         relative_l2={:.9e} abs_limit={:.9e} relative_limit={:.9e}",
                        seam.stage,
                        seam.compared_values,
                        seam.mismatch_count,
                        seam.byte_exact,
                        seam.max_abs_diff,
                        seam.relative_l2,
                        seam.accepted_abs_tolerance,
                        seam.accepted_relative_l2,
                    );
                }
                {
                    let prefix_frames = streaming_state
                        .speaker_cache
                        .rows
                        .checked_add(streaming_state.fifo.rows)
                        .expect("native stream prefix does not overflow");
                    let mut stream_prefix = streaming_state.speaker_cache.data.clone();
                    stream_prefix.extend_from_slice(&streaming_state.fifo.data);
                    let l3_input = reference
                        .verify_public_l3_encoder_input(
                            &activations,
                            &fixture.name,
                            usize::try_from(transition.step).expect("step fits usize"),
                            &stream_prefix,
                            prefix_frames,
                            &time_major,
                            input_frames,
                        )
                        .expect(
                            "scaled FastConformer input stays within the frozen handoff envelope",
                        );
                    eprintln!(
                        "stage={} compared={} mismatches={} byte_exact={} max_abs={:.9e} \
                         relative_l2={:.9e} abs_limit={:.9e} relative_limit={:.9e}",
                        l3_input.stage,
                        l3_input.compared_values,
                        l3_input.mismatch_count,
                        l3_input.byte_exact,
                        l3_input.max_abs_diff,
                        l3_input.relative_l2,
                        l3_input.accepted_abs_tolerance,
                        l3_input.accepted_relative_l2,
                    );
                    let feed_forward1 = reference
                        .verify_public_l3_block0_feed_forward1(
                            &activations,
                            &fixture.name,
                            usize::try_from(transition.step).expect("step fits usize"),
                            &stream_prefix,
                            prefix_frames,
                            &time_major,
                            input_frames,
                        )
                        .expect("block-00 FFN1 stays within the frozen operator envelope");
                    eprintln!(
                        "stage={} compared={} mismatches={} byte_exact={} max_abs={:.9e} \
                         relative_l2={:.9e} abs_limit={:.9e} relative_limit={:.9e}",
                        feed_forward1.stage,
                        feed_forward1.compared_values,
                        feed_forward1.mismatch_count,
                        feed_forward1.byte_exact,
                        feed_forward1.max_abs_diff,
                        feed_forward1.relative_l2,
                        feed_forward1.accepted_abs_tolerance,
                        feed_forward1.accepted_relative_l2,
                    );
                    let qkv = reference
                        .verify_public_l3_block0_qkv(
                            &activations,
                            &fixture.name,
                            usize::try_from(transition.step).expect("step fits usize"),
                            &stream_prefix,
                            prefix_frames,
                            &time_major,
                            input_frames,
                        )
                        .expect("block-00 Q/K/V stay within the frozen operator envelope");
                    for projection in qkv {
                        eprintln!(
                            "stage={} compared={} mismatches={} byte_exact={} max_abs={:.9e} \
                             relative_l2={:.9e} abs_limit={:.9e} relative_limit={:.9e}",
                            projection.stage,
                            projection.compared_values,
                            projection.mismatch_count,
                            projection.byte_exact,
                            projection.max_abs_diff,
                            projection.relative_l2,
                            projection.accepted_abs_tolerance,
                            projection.accepted_relative_l2,
                        );
                    }
                    let l5_probabilities = {
                        if fixture.name == "hiyis_exact_two_chunks" && transition.step == 0 {
                            let attention_input = reference
                                .assemble_l3_encoder_input(
                                    &activations,
                                    &fixture.name,
                                    usize::try_from(transition.step).expect("step fits usize"),
                                    &stream_prefix,
                                    prefix_frames,
                                    &time_major,
                                    input_frames,
                                )
                                .expect("assemble benchmark attention input");
                            let mut attention_normalized = attention_input.clone();
                            let block0 = reference
                                .fastconformer_block(0)
                                .expect("authenticated block zero");
                            block0
                                .norm_feed_forward1
                                .forward_in_place(&mut attention_normalized)
                                .expect("normalize benchmark FFN1 input");
                            let attention_feed_forward = block0
                                .feed_forward1
                                .forward(&attention_normalized)
                                .expect("benchmark FFN1");
                            let mut attention_residual = attention_input;
                            add_scaled_residual_in_place(
                                &mut attention_residual,
                                &attention_feed_forward,
                                0.5,
                            )
                            .expect("benchmark FFN1 residual");
                            block0
                                .norm_self_att
                                .forward_in_place(&mut attention_residual)
                                .expect("normalize benchmark attention input");
                            let scalar_start = std::time::Instant::now();
                            let scalar_attention = block0
                                .self_attention
                                .forward_scalar(&attention_residual)
                                .expect("scalar attention incumbent");
                            let scalar_elapsed = scalar_start.elapsed();
                            let kernel_start = std::time::Instant::now();
                            let kernel_attention = block0
                                .self_attention
                                .forward(&attention_residual)
                                .expect("FrankenTorch attention candidate");
                            let kernel_elapsed = kernel_start.elapsed();
                            let path_metrics = frontend_parity_metrics(
                                "block0_attention_scalar_vs_frankentorch",
                                &kernel_attention.data,
                                &scalar_attention.data,
                            )
                            .expect("attention paths have matching finite geometry");
                            eprintln!(
                                "attention_path_parity compared={} max_abs={:.9e} relative_l2={:.9e} \
                             scalar_seconds={:.6} frankentorch_seconds={:.6}",
                                path_metrics.compared_values,
                                path_metrics.max_abs_diff,
                                path_metrics.relative_l2,
                                scalar_elapsed.as_secs_f64(),
                                kernel_elapsed.as_secs_f64(),
                            );
                            assert!(
                                path_metrics.max_abs_diff <= SORTFORMER_L3_ATTENTION_MAX_ABS_DIFF
                            );
                            assert!(
                                path_metrics.relative_l2 <= SORTFORMER_L3_ATTENTION_MAX_RELATIVE_L2
                            );
                        }
                        let (fastconformer, l3_output) = reference
                            .verify_public_l3_fastconformer(
                                &activations,
                                &fixture.name,
                                usize::try_from(transition.step).expect("step fits usize"),
                                &stream_prefix,
                                prefix_frames,
                                &time_major,
                                input_frames,
                            )
                            .expect("all FastConformer seams stay within frozen envelopes");
                        for seam in fastconformer {
                            eprintln!(
                                "stage={} compared={} mismatches={} byte_exact={} max_abs={:.9e} \
                                 relative_l2={:.9e} abs_limit={:.9e} relative_limit={:.9e}",
                                seam.stage,
                                seam.compared_values,
                                seam.mismatch_count,
                                seam.byte_exact,
                                seam.max_abs_diff,
                                seam.relative_l2,
                                seam.accepted_abs_tolerance,
                                seam.accepted_relative_l2,
                            );
                        }
                        let (transformer, l4_output) = reference
                            .verify_public_l4_transformer(
                                &activations,
                                &fixture.name,
                                usize::try_from(transition.step).expect("step fits usize"),
                                &l3_output,
                            )
                            .expect("all L4 Transformer seams stay within frozen envelopes");
                        for seam in transformer {
                            eprintln!(
                                "stage={} compared={} mismatches={} byte_exact={} max_abs={:.9e} \
                                 relative_l2={:.9e} abs_limit={:.9e} relative_limit={:.9e}",
                                seam.stage,
                                seam.compared_values,
                                seam.mismatch_count,
                                seam.byte_exact,
                                seam.max_abs_diff,
                                seam.relative_l2,
                                seam.accepted_abs_tolerance,
                                seam.accepted_relative_l2,
                            );
                        }
                        let (speaker_head, l5_probabilities) = reference
                            .verify_public_l5_speaker_head(
                                &activations,
                                &fixture.name,
                                usize::try_from(transition.step).expect("step fits usize"),
                                &l4_output,
                            )
                            .expect("all L5 speaker-head seams stay within frozen envelopes");
                        for seam in speaker_head {
                            eprintln!(
                                "stage={} compared={} mismatches={} byte_exact={} max_abs={:.9e} \
                                 relative_l2={:.9e} abs_limit={:.9e} relative_limit={:.9e}",
                                seam.stage,
                                seam.compared_values,
                                seam.mismatch_count,
                                seam.byte_exact,
                                seam.max_abs_diff,
                                seam.relative_l2,
                                seam.accepted_abs_tolerance,
                                seam.accepted_relative_l2,
                            );
                        }
                        l5_probabilities
                    };
                    compared_neural_steps += 1;

                    let current_embeddings = reference
                        .subsample_feature_chunk(&time_major, input_frames)
                        .expect("reconstruct native pre-encode embeddings");
                    let left_embedding_frames = left / SORTFORMER_SUBSAMPLING_FACTOR;
                    let right_embedding_frames = right
                        .checked_add(SORTFORMER_SUBSAMPLING_FACTOR - 1)
                        .expect("right embedding context does not overflow")
                        / SORTFORMER_SUBSAMPLING_FACTOR;
                    let diagnostic_state_before = streaming_state.clone();
                    let chunk_predictions = reference
                        .update_streaming_state(
                            &mut streaming_state,
                            &current_embeddings,
                            &l5_probabilities,
                            left_embedding_frames,
                            right_embedding_frames,
                        )
                        .expect("native L6 state update succeeds");
                    assert_eq!(
                        chunk_predictions.rows,
                        usize::try_from(transition.output_frames).expect("output frames fit usize")
                    );
                    if fixture.name == "iqtde_complete_four_speakers" && step == 2 {
                        let mut exact_score_state = diagnostic_state_before;
                        exact_score_state.speaker_cache_predictions = Some(complete_public_mat(
                            &activations,
                            &format!(
                                "fixture.{}.step.{step:03}.l6.before.spkcache_preds",
                                fixture.name
                            ),
                            SORTFORMER_SPEAKER_LANES,
                        ));
                        let exact_probabilities = complete_public_mat(
                            &activations,
                            &format!("fixture.{}.step.{step:03}.l5.probabilities", fixture.name),
                            SORTFORMER_SPEAKER_LANES,
                        );
                        reference
                            .update_streaming_state(
                                &mut exact_score_state,
                                &current_embeddings,
                                &exact_probabilities,
                                left_embedding_frames,
                                right_embedding_frames,
                            )
                            .expect("exact-score L6 diagnostic update succeeds");
                        let exact_score_result = reference.verify_public_l6_state(
                            &activations,
                            &fixture.name,
                            step,
                            "after",
                            &exact_score_state,
                        );
                        eprintln!(
                            "l6_exact_score_diagnostic={}",
                            if exact_score_result.is_ok() {
                                "pass"
                            } else {
                                "fail"
                            }
                        );
                        if let Err(error) = exact_score_result {
                            eprintln!("l6_exact_score_error={error}");
                        }
                    }
                    let after_state = reference
                        .verify_public_l6_state(
                            &activations,
                            &fixture.name,
                            step,
                            "after",
                            &streaming_state,
                        )
                        .expect("native L6 after-state matches the authenticated boundary");
                    for seam in after_state.f32_seams {
                        eprintln!(
                            "stage={} compared={} mismatches={} byte_exact={} max_abs={:.9e} \
                             relative_l2={:.9e} abs_limit={:.9e} relative_limit={:.9e}",
                            seam.stage,
                            seam.compared_values,
                            seam.mismatch_count,
                            seam.byte_exact,
                            seam.max_abs_diff,
                            seam.relative_l2,
                            seam.accepted_abs_tolerance,
                            seam.accepted_relative_l2,
                        );
                    }
                    for seam in after_state.i64_seams {
                        eprintln!(
                            "stage={} compared={} mismatches={} byte_exact={}",
                            seam.stage, seam.compared_values, seam.mismatch_count, seam.byte_exact,
                        );
                    }
                }
                central_start = central_start
                    .checked_add(central_frames)
                    .expect("central feature offset does not overflow");
            }
            assert_eq!(central_start, features.valid_frames);
            assert_eq!(compared_neural_steps, transitions.len());
        }
    }

    #[test]
    fn transformer_score_kernel_matches_scalar_at_streaming_geometry() {
        let frames = 189_usize;
        let width = SORTFORMER_TRANSFORMER_HEAD_WIDTH;
        let lhs: Vec<f32> = (0..frames * width)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.031_25)
            .collect();
        let rhs: Vec<f32> = (0..frames * width)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.015_625)
            .collect();
        let observed =
            ft_kernel_cpu::matmul_rhs_transposed_contiguous_f32(frames, width, frames, &lhs, &rhs)
                .expect("production-shape score kernel accepts finite inputs");
        for row in 0..frames {
            for column in 0..frames {
                let mut expected = 0.0_f32;
                for inner in 0..width {
                    expected += lhs[row * width + inner] * rhs[column * width + inner];
                }
                let actual = observed[row * frames + column];
                assert!(
                    (actual - expected).abs() <= 2.0e-6,
                    "score mismatch at ({row}, {column}): actual={actual} expected={expected}"
                );
            }
        }

        let mut probabilities = observed;
        for row in probabilities.chunks_mut(frames) {
            softmax_f32_in_place(row).expect("finite score rows admit softmax");
        }
        let values: Vec<f32> = (0..frames * width)
            .map(|index| ((index % 37) as f32 - 18.0) * 0.007_812_5)
            .collect();
        let context = nn::matmul(
            &Mat::from_vec(frames, frames, probabilities.clone()),
            &Mat::from_vec(frames, width, values.clone()),
        )
        .expect("production-shape probability-value kernel accepts finite inputs");
        for row in 0..frames {
            for column in 0..width {
                let mut expected = 0.0_f32;
                for inner in 0..frames {
                    expected +=
                        probabilities[row * frames + inner] * values[inner * width + column];
                }
                let actual = context.data[row * width + column];
                assert!(
                    (actual - expected).abs() <= 2.0e-6,
                    "context mismatch at ({row}, {column}): actual={actual} expected={expected}"
                );
            }
        }
    }
}
