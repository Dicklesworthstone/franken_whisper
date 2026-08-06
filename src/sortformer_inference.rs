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
use crate::native_engine::mel::{REAL_FFT_512_BINS, REAL_FFT_512_LEN, real_fft_512};
use crate::native_engine::weights::SafetensorsFile;
use crate::sortformer_conformance::VerifiedSortformerPackage;

pub const SORTFORMER_SAMPLE_RATE_HZ: usize = 16_000;
pub const SORTFORMER_CHANNELS: usize = 1;
pub const SORTFORMER_MAX_AUDIO_SECONDS: usize = 2 * 60 * 60;
pub const SORTFORMER_MAX_AUDIO_SAMPLES: usize =
    SORTFORMER_SAMPLE_RATE_HZ * SORTFORMER_MAX_AUDIO_SECONDS;
pub const SORTFORMER_WINDOW_SAMPLES: usize = 400;
pub const SORTFORMER_HOP_SAMPLES: usize = 160;
pub const SORTFORMER_MEL_BINS: usize = 128;

pub const SORTFORMER_WINDOW_TENSOR_NAME: &str = "preprocessor.featurizer.window";
pub const SORTFORMER_WINDOW_TENSOR_SHA256: &str =
    "c427e2029118cf789649e5a4d439b6115d0dd0cbf95dcd22f65e3c848add8c5b";
pub const SORTFORMER_MEL_TENSOR_NAME: &str = "preprocessor.featurizer.fb";
pub const SORTFORMER_MEL_TENSOR_SHA256: &str =
    "bce5ec5f194a5913f6508cee5a85512e7bad2352db8fc28f5c6ff75af8b09137";

const PREEMPHASIS: f32 = 0.97;
const LOG_GUARD: f32 = f32::from_bits(0x3380_0000); // exactly 2^-24
const WINDOW_FFT_LEFT_PAD: usize = (REAL_FFT_512_LEN - SORTFORMER_WINDOW_SAMPLES) / 2;
const WINDOW_CENTER: usize = SORTFORMER_WINDOW_SAMPLES / 2;
const ZERO_FILL_CHUNK_VALUES: usize = 1024 * 1024;
const WINDOW_SHAPE: &[usize] = &[SORTFORMER_WINDOW_SAMPLES];
const MEL_SHAPE: &[usize] = &[1, SORTFORMER_MEL_BINS, REAL_FFT_512_BINS];

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
}
