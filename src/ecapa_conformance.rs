//! Frozen ECAPA-TDNN model, export, frontend, and conformance contract.
//!
//! The optional neural diarizer is not admitted by this module. This module
//! freezes the independently reproducible source model and provides a
//! fail-closed bridge from an unsafe framework checkpoint to the native
//! safetensors loader consumed by [`crate::ecapa_inference`]. No source
//! weights, exported weights, or framework runtime are vendored.

#![forbid(unsafe_code)]

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{FwError, FwResult};
use crate::native_engine::weights::{SafetensorsFile, WeightsManifest, validate};
use crate::orchestrator::CancellationToken;

pub const ECAPA_CONTRACT_SCHEMA: &str = "franken-whisper-ecapa-contract-v2";
pub const ECAPA_CONTRACT_SHA256: &str =
    "9eb3e323aaa5550c87057996978d38ce57f9b280b829be6217440c8e63cef7a4";
pub const ECAPA_EXPORTER_VERSION: &str = "franken-whisper-ecapa-export-v1";
pub const ECAPA_EXPORT_PROFILE: &str = "ecapa-tdnn-voxceleb-v1";
pub const ECAPA_PACKAGE_FILENAME: &str = "ecapa_tdnn_voxceleb.safetensors";
pub const ECAPA_PACKAGE_SHA256: &str =
    "9276a840c52cdd2e9afb73cd87a38e15749e12bf494d3ca47b5bc162f237cbcc";
pub const ECAPA_PACKAGE_BYTES: u64 = 83_246_544;
pub const ECAPA_EXPORT_NUMPY_VERSION: &str = "2.2.6";
pub const ECAPA_EXPORT_TORCH_VERSION: &str = "2.7.1";
pub const ECAPA_EXPORT_SAFETENSORS_VERSION: &str = "0.5.3";
pub const ECAPA_GOLDEN_SCHEMA: &str = "franken-whisper-ecapa-golden-v1";
pub const ECAPA_GOLDEN_EVIDENCE_SHA256: &str =
    "073a910a2a8d171dca45e28940387ebfc0642e63224d62ebd62abe2b8efd9ac2";
pub const ECAPA_FULL_ORACLE_SCHEMA: &str = "franken-whisper-ecapa-full-oracle-v1";
pub const ECAPA_FULL_ORACLE_EXPORTER_VERSION: &str = "franken-whisper-ecapa-full-oracle-export-v1";
pub const ECAPA_FULL_ORACLE_FILENAME: &str = "ecapa_tdnn_voxceleb_full_oracle.safetensors";
pub const ECAPA_FULL_ORACLE_SHA256: &str =
    "2c80806fbf68262ab1e0a1b52af18139f08272b7802fc3b0fd96011192dcf485";
pub const ECAPA_FULL_ORACLE_BYTES: u64 = 2_160_320;
pub const ECAPA_FULL_ORACLE_TENSOR_COUNT: usize = 7;
pub const ECAPA_FULL_ORACLE_F32_ELEMENTS: u64 = 539_616;
pub const ECAPA_FULL_ORACLE_FRONTEND_F32_ELEMENTS: u64 = 16_160;
pub const ECAPA_FULL_ORACLE_NEURAL_F32_ELEMENTS: u64 = 523_456;
pub const ECAPA_FULL_ORACLE_PAYLOAD_BYTES: u64 = ECAPA_FULL_ORACLE_F32_ELEMENTS * 4;
pub const ECAPA_ORACLE_SPEECHBRAIN_VERSION: &str = "0.5.16";
pub const ECAPA_ORACLE_TORCHAUDIO_VERSION: &str = "2.7.1";
pub const ECAPA_ORACLE_PYTHON_VERSION: &str = "3.12.12";
pub const ECAPA_FIXTURE_ID: &str = "analytic-harmonic-chirp-impulse-v1";
pub const ECAPA_FIXTURE_PCM_SHA256: &str =
    "acc240c07370020bbd1b3aaf9b8b81be43ef053b8da950969e86f62b6f1dba2f";
pub const ECAPA_MODEL_ID: &str = "speechbrain/spkrec-ecapa-voxceleb";
pub const ECAPA_MODEL_REVISION: &str = "eac27266f68caa806381260bd44ace38b136c76a";
pub const ECAPA_TRAINING_CODE_REVISION: &str = "aa0185408025e80f6c748d2c7af7fa96958c2231";
pub const ECAPA_SOURCE_CHECKPOINT_SHA256: &str =
    "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2";
pub const ECAPA_SOURCE_HYPERPARAMS_SHA256: &str =
    "ecd11c44202b32edb72709dd1013a16f2f060ebee3438ae8a9f9fecb0666ecd2";
pub const ECAPA_SOURCE_CHECKPOINT_BYTES: u64 = 83_316_686;
pub const ECAPA_SOURCE_TENSOR_COUNT: usize = 231;
pub const ECAPA_EXPORTED_TENSOR_COUNT: usize = 200;
pub const ECAPA_DROPPED_BATCH_COUNTER_COUNT: usize = 31;
pub const ECAPA_EXPORTED_F32_ELEMENTS: u64 = 20_805_952;
pub const ECAPA_EXPORTED_PAYLOAD_BYTES: u64 = ECAPA_EXPORTED_F32_ELEMENTS * 4;
pub const ECAPA_SAMPLE_RATE_HZ: usize = 16_000;
pub const ECAPA_WINDOW_SAMPLES: usize = 400;
pub const ECAPA_HOP_SAMPLES: usize = 160;
pub const ECAPA_FFT_BINS: usize = 201;
pub const ECAPA_MEL_BANDS: usize = 80;
pub const ECAPA_EMBEDDING_DIMENSIONS: usize = 192;
pub const ECAPA_MINIMUM_RUNTIME_SAMPLES: usize = ECAPA_SAMPLE_RATE_HZ / 2;
pub const ECAPA_MAXIMUM_RUNTIME_SAMPLES: usize = ECAPA_SAMPLE_RATE_HZ * 3;
pub const ECAPA_MAXIMUM_CONFORMANCE_SAMPLES: usize = ECAPA_SAMPLE_RATE_HZ;
pub const ECAPA_MINIMUM_INFERENCE_FRAMES: usize =
    ECAPA_MINIMUM_RUNTIME_SAMPLES / ECAPA_HOP_SAMPLES + 1;
pub const ECAPA_MAXIMUM_INFERENCE_FRAMES: usize = 301;
pub const ECAPA_MAXIMUM_ABSOLUTE_INPUT_FEATURE: f32 = 160.0;
pub const ECAPA_BATCH_NORM_EPSILON: f32 = 1.0e-5;
pub const ECAPA_POOLING_VARIANCE_FLOOR: f32 = 1.0e-12;

const HASH_HEX_LEN: usize = 64;
const READ_CHUNK_BYTES: usize = 64 * 1024;
const FRONTEND_AMIN: f32 = 1.0e-10;
const FRONTEND_TOP_DB: f32 = 80.0;
const EMBEDDING_NORM_EPSILON: f32 = 1.0e-6;

/// Immutable source and numerical semantics for the optional ECAPA model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EcapaContract {
    pub schema_version: String,
    pub model_id: String,
    pub model_revision: String,
    pub license_spdx: String,
    pub training_code_revision: String,
    pub source_checkpoint_sha256: String,
    pub source_hyperparams_sha256: String,
    pub source_checkpoint_bytes: u64,
    pub source_tensor_count: usize,
    pub exported_tensor_count: usize,
    pub dropped_batch_counter_count: usize,
    pub exported_payload_bytes: u64,
    pub frontend: EcapaFrontendContract,
    pub export: EcapaExportContract,
    pub architecture: EcapaArchitectureContract,
    pub training_domains: Vec<String>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EcapaFrontendContract {
    pub sample_rate_hz: usize,
    pub channels: usize,
    pub window_samples: usize,
    pub hop_samples: usize,
    pub fft_bins: usize,
    pub window: String,
    pub centered: bool,
    pub pad_mode: String,
    pub spectrum: String,
    pub mel_bands: usize,
    pub minimum_hz: usize,
    pub maximum_hz: usize,
    pub mel_scale: String,
    pub mel_filter_shape: String,
    pub amplitude_minimum: f32,
    pub top_db: f32,
    pub sentence_mean_normalization: bool,
    pub sentence_std_normalization: bool,
    pub minimum_runtime_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EcapaExportContract {
    pub exporter_version: String,
    pub export_profile: String,
    pub package_format: String,
    pub package_filename: String,
    pub package_sha256: String,
    pub package_bytes: u64,
    pub dtype: String,
    pub byte_order: String,
    pub tensor_order: String,
    pub logical_layout: String,
    pub numpy_version: String,
    pub torch_version: String,
    pub safetensors_version: String,
    pub batch_norm_folding: bool,
    pub batch_norm_epsilon: f32,
    pub batch_norm_momentum: f32,
    pub dropped_source_tensors: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EcapaArchitectureContract {
    pub input_features: usize,
    pub channels: Vec<usize>,
    pub kernel_sizes: Vec<usize>,
    pub dilations: Vec<usize>,
    pub groups: Vec<usize>,
    pub res2net_scale: usize,
    pub squeeze_excitation_channels: usize,
    pub attention_channels: usize,
    pub global_attention_context: bool,
    pub convolution_padding: String,
    pub tdnn_operation_order: String,
    pub res2net_accumulation: String,
    pub attention_softmax_axis: String,
    pub pooling_variance_floor: f32,
    pub minimum_inference_frames: usize,
    pub maximum_inference_frames: usize,
    pub maximum_absolute_input_feature: f32,
    pub embedding_dimensions: usize,
    pub golden_embedding_stage: String,
    pub embedding_normalization: String,
    pub embedding_norm_epsilon: f32,
    pub dropout: f32,
}

/// Frozen model contract. The model remains optional and default-off.
#[must_use]
pub fn frozen_ecapa_contract() -> EcapaContract {
    EcapaContract {
        schema_version: ECAPA_CONTRACT_SCHEMA.to_owned(),
        model_id: ECAPA_MODEL_ID.to_owned(),
        model_revision: ECAPA_MODEL_REVISION.to_owned(),
        license_spdx: "Apache-2.0".to_owned(),
        training_code_revision: ECAPA_TRAINING_CODE_REVISION.to_owned(),
        source_checkpoint_sha256: ECAPA_SOURCE_CHECKPOINT_SHA256.to_owned(),
        source_hyperparams_sha256: ECAPA_SOURCE_HYPERPARAMS_SHA256.to_owned(),
        source_checkpoint_bytes: ECAPA_SOURCE_CHECKPOINT_BYTES,
        source_tensor_count: ECAPA_SOURCE_TENSOR_COUNT,
        exported_tensor_count: ECAPA_EXPORTED_TENSOR_COUNT,
        dropped_batch_counter_count: ECAPA_DROPPED_BATCH_COUNTER_COUNT,
        exported_payload_bytes: ECAPA_EXPORTED_PAYLOAD_BYTES,
        frontend: EcapaFrontendContract {
            sample_rate_hz: ECAPA_SAMPLE_RATE_HZ,
            channels: 1,
            window_samples: ECAPA_WINDOW_SAMPLES,
            hop_samples: ECAPA_HOP_SAMPLES,
            fft_bins: ECAPA_FFT_BINS,
            window: "periodic_hamming".to_owned(),
            centered: true,
            pad_mode: "constant_zero".to_owned(),
            spectrum: "squared_magnitude".to_owned(),
            mel_bands: ECAPA_MEL_BANDS,
            minimum_hz: 0,
            maximum_hz: ECAPA_SAMPLE_RATE_HZ / 2,
            mel_scale: "htk_2595_log10".to_owned(),
            mel_filter_shape: "speechbrain_symmetric_triangular".to_owned(),
            amplitude_minimum: FRONTEND_AMIN,
            top_db: FRONTEND_TOP_DB,
            sentence_mean_normalization: true,
            sentence_std_normalization: false,
            minimum_runtime_samples: ECAPA_MINIMUM_RUNTIME_SAMPLES,
        },
        export: EcapaExportContract {
            exporter_version: ECAPA_EXPORTER_VERSION.to_owned(),
            export_profile: ECAPA_EXPORT_PROFILE.to_owned(),
            package_format: "safetensors".to_owned(),
            package_filename: ECAPA_PACKAGE_FILENAME.to_owned(),
            package_sha256: ECAPA_PACKAGE_SHA256.to_owned(),
            package_bytes: ECAPA_PACKAGE_BYTES,
            dtype: "ieee754_f32".to_owned(),
            byte_order: "little_endian".to_owned(),
            tensor_order: "lexicographic_name_order".to_owned(),
            logical_layout: "pytorch_contiguous_row_major".to_owned(),
            numpy_version: ECAPA_EXPORT_NUMPY_VERSION.to_owned(),
            torch_version: ECAPA_EXPORT_TORCH_VERSION.to_owned(),
            safetensors_version: ECAPA_EXPORT_SAFETENSORS_VERSION.to_owned(),
            batch_norm_folding: false,
            batch_norm_epsilon: ECAPA_BATCH_NORM_EPSILON,
            batch_norm_momentum: 0.1,
            dropped_source_tensors: "batch_norm_num_batches_tracked_i64_only".to_owned(),
        },
        architecture: EcapaArchitectureContract {
            input_features: ECAPA_MEL_BANDS,
            channels: vec![1_024, 1_024, 1_024, 1_024, 3_072],
            kernel_sizes: vec![5, 3, 3, 3, 1],
            dilations: vec![1, 2, 3, 4, 1],
            groups: vec![1, 1, 1, 1, 1],
            res2net_scale: 8,
            squeeze_excitation_channels: 128,
            attention_channels: 128,
            global_attention_context: true,
            convolution_padding: "same_reflect_effective_kernel".to_owned(),
            tdnn_operation_order: "conv_relu_batch_norm_eval".to_owned(),
            res2net_accumulation: "chunk_0_identity_then_recursive_previous_output".to_owned(),
            attention_softmax_axis: "time_per_channel".to_owned(),
            pooling_variance_floor: ECAPA_POOLING_VARIANCE_FLOOR,
            minimum_inference_frames: ECAPA_MINIMUM_INFERENCE_FRAMES,
            maximum_inference_frames: ECAPA_MAXIMUM_INFERENCE_FRAMES,
            maximum_absolute_input_feature: ECAPA_MAXIMUM_ABSOLUTE_INPUT_FEATURE,
            embedding_dimensions: ECAPA_EMBEDDING_DIMENSIONS,
            golden_embedding_stage: "raw_model_output_before_l2_normalization".to_owned(),
            embedding_normalization: "l2_unit_norm_fail_below_epsilon".to_owned(),
            embedding_norm_epsilon: EMBEDDING_NORM_EPSILON,
            dropout: 0.0,
        },
        training_domains: vec![
            "voxceleb1".to_owned(),
            "voxceleb2".to_owned(),
            "english_web_video".to_owned(),
        ],
        limitations: vec![
            "speaker_verification_training_not_diarization_calibration".to_owned(),
            "cross_domain_performance_not_warranted".to_owned(),
            "telephone_far_field_and_playback_shift_require_held_out_validation".to_owned(),
            "embedding_is_not_a_person_identity_claim".to_owned(),
        ],
    }
}

pub fn ecapa_contract_sha256() -> FwResult<String> {
    let observed = canonical_sha256(&frozen_ecapa_contract())?;
    if observed != ECAPA_CONTRACT_SHA256 {
        return Err(ecapa_error(
            "contract_internal_drift",
            &format!(
                "compiled ECAPA contract hash {observed} does not match frozen \
                 {ECAPA_CONTRACT_SHA256}"
            ),
        ));
    }
    Ok(observed)
}

/// One exact inference tensor expected after the pinned checkpoint is exported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EcapaTensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

/// Generate the exact 200-tensor inference inventory in checkpoint order.
#[must_use]
pub fn expected_ecapa_tensors() -> Vec<EcapaTensorSpec> {
    let mut tensors = Vec::with_capacity(ECAPA_EXPORTED_TENSOR_COUNT);
    push_conv_bn(
        &mut tensors,
        "blocks.0",
        vec![1_024, ECAPA_MEL_BANDS, 5],
        1_024,
    );
    for block in 1..=3 {
        push_conv_bn(
            &mut tensors,
            &format!("blocks.{block}.tdnn1"),
            vec![1_024, 1_024, 1],
            1_024,
        );
        for inner in 0..7 {
            push_conv_bn(
                &mut tensors,
                &format!("blocks.{block}.res2net_block.blocks.{inner}"),
                vec![128, 128, 3],
                128,
            );
        }
        push_conv_bn(
            &mut tensors,
            &format!("blocks.{block}.tdnn2"),
            vec![1_024, 1_024, 1],
            1_024,
        );
        push_bare_conv(
            &mut tensors,
            &format!("blocks.{block}.se_block.conv1"),
            vec![128, 1_024, 1],
            128,
        );
        push_bare_conv(
            &mut tensors,
            &format!("blocks.{block}.se_block.conv2"),
            vec![1_024, 128, 1],
            1_024,
        );
    }
    push_conv_bn(&mut tensors, "mfa", vec![3_072, 3_072, 1], 3_072);
    push_conv_bn(&mut tensors, "asp.tdnn", vec![128, 9_216, 1], 128);
    push_bare_conv(&mut tensors, "asp.conv", vec![3_072, 128, 1], 3_072);
    push_batch_norm(&mut tensors, "asp_bn", 6_144, false);
    tensors.push(EcapaTensorSpec {
        name: "fc.conv.weight".to_owned(),
        shape: vec![ECAPA_EMBEDDING_DIMENSIONS, 6_144, 1],
    });
    tensors.push(EcapaTensorSpec {
        name: "fc.conv.bias".to_owned(),
        shape: vec![ECAPA_EMBEDDING_DIMENSIONS],
    });
    tensors
}

fn push_conv_bn(
    tensors: &mut Vec<EcapaTensorSpec>,
    prefix: &str,
    weight_shape: Vec<usize>,
    channels: usize,
) {
    push_conv(tensors, prefix, weight_shape, channels);
    push_batch_norm(tensors, prefix, channels, true);
}

fn push_conv(
    tensors: &mut Vec<EcapaTensorSpec>,
    prefix: &str,
    weight_shape: Vec<usize>,
    channels: usize,
) {
    tensors.push(EcapaTensorSpec {
        name: format!("{prefix}.conv.conv.weight"),
        shape: weight_shape,
    });
    tensors.push(EcapaTensorSpec {
        name: format!("{prefix}.conv.conv.bias"),
        shape: vec![channels],
    });
}

fn push_bare_conv(
    tensors: &mut Vec<EcapaTensorSpec>,
    prefix: &str,
    weight_shape: Vec<usize>,
    channels: usize,
) {
    tensors.push(EcapaTensorSpec {
        name: format!("{prefix}.conv.weight"),
        shape: weight_shape,
    });
    tensors.push(EcapaTensorSpec {
        name: format!("{prefix}.conv.bias"),
        shape: vec![channels],
    });
}

fn push_batch_norm(
    tensors: &mut Vec<EcapaTensorSpec>,
    prefix: &str,
    channels: usize,
    nested_norm: bool,
) {
    let norm = if nested_norm {
        format!("{prefix}.norm.norm")
    } else {
        format!("{prefix}.norm")
    };
    for suffix in ["weight", "bias", "running_mean", "running_var"] {
        tensors.push(EcapaTensorSpec {
            name: format!("{norm}.{suffix}"),
            shape: vec![channels],
        });
    }
}

/// Validate the exact frozen safetensors package without exposing its path.
pub fn verify_ecapa_weight_package(package_path: &Path) -> FwResult<()> {
    let token = CancellationToken::unbounded(); // ubs:ignore — cancellation token is not a secret
    verify_ecapa_weight_package_with_token(package_path, &token)
}

pub(crate) fn verify_ecapa_weight_package_with_token(
    package_path: &Path,
    token: &CancellationToken,
) -> FwResult<()> {
    load_verified_ecapa_weight_package_with_token(package_path, token).map(drop)
}

/// Load and authenticate the exact frozen package without reopening its path.
///
/// Callers receive the [`SafetensorsFile`] backed by the same owned bytes that
/// passed the size and SHA-256 checks. This prevents a path replacement between
/// verification and model construction while keeping all path and tensor data
/// out of diagnostics.
pub fn load_verified_ecapa_weight_package(package_path: &Path) -> FwResult<SafetensorsFile> {
    let token = CancellationToken::unbounded(); // ubs:ignore — cancellation token is not a secret
    load_verified_ecapa_weight_package_with_token(package_path, &token)
}

pub(crate) fn load_verified_ecapa_weight_package_with_token(
    package_path: &Path,
    token: &CancellationToken,
) -> FwResult<SafetensorsFile> {
    load_verified_ecapa_weight_package_with_checkpoint(package_path, &|| token.checkpoint())
}

/// Load and authenticate the exact package with a caller-owned checkpoint.
///
/// The checkpoint runs before opening the package, between every bounded read,
/// and around safetensors parsing and census validation.
pub fn load_verified_ecapa_weight_package_with_checkpoint(
    package_path: &Path,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<SafetensorsFile> {
    let package_bytes = read_verified_ecapa_artifact(
        package_path,
        ECAPA_PACKAGE_BYTES,
        ECAPA_PACKAGE_SHA256,
        checkpoint,
    )?;
    ecapa_load_checkpoint(checkpoint)?;
    let package = SafetensorsFile::from_owned_bytes(package_bytes).map_err(|_| {
        ecapa_error(
            "safetensors_structure",
            "weight package is not structurally valid safetensors",
        )
    })?;
    verify_loaded_ecapa_package(&package)?;
    ecapa_load_checkpoint(checkpoint)?;
    Ok(package)
}

/// Load and authenticate the complete public seven-stage ECAPA oracle.
///
/// The returned tensors are backed by the same owned bytes whose exact length
/// and SHA-256 were checked. The tensor census, shapes, F32 payload hashes, and
/// provenance metadata are then checked independently before the artifact is
/// returned. No audio or transcript supplied by a user is read by this path.
pub fn load_verified_ecapa_full_oracle(oracle_path: &Path) -> FwResult<SafetensorsFile> {
    let token = CancellationToken::unbounded(); // ubs:ignore — cancellation token is not a secret
    load_verified_ecapa_full_oracle_with_checkpoint(oracle_path, &|| token.checkpoint())
}

/// Authenticate the public full oracle with a cooperative checkpoint.
pub fn load_verified_ecapa_full_oracle_with_checkpoint(
    oracle_path: &Path,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<SafetensorsFile> {
    let oracle_bytes = read_verified_ecapa_artifact(
        oracle_path,
        ECAPA_FULL_ORACLE_BYTES,
        ECAPA_FULL_ORACLE_SHA256,
        checkpoint,
    )?;
    ecapa_load_checkpoint(checkpoint)?;
    let oracle = SafetensorsFile::from_owned_bytes(oracle_bytes).map_err(|_| {
        ecapa_error(
            "oracle_structure",
            "full oracle is not structurally valid safetensors",
        )
    })?;
    verify_loaded_ecapa_full_oracle(&oracle, checkpoint)?;
    ecapa_load_checkpoint(checkpoint)?;
    Ok(oracle)
}

pub(crate) fn ecapa_load_checkpoint(
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    match checkpoint() {
        Ok(()) => Ok(()),
        Err(FwError::Cancelled(_)) => Err(FwError::Cancelled(
            "ecapa.load_cancelled: cooperative checkpoint requested cancellation".to_owned(),
        )),
        Err(_) => Err(ecapa_error(
            "checkpoint_failure",
            "caller checkpoint returned a non-cancellation failure",
        )),
    }
}

fn read_verified_ecapa_artifact(
    artifact_path: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<Vec<u8>> {
    ecapa_load_checkpoint(checkpoint)?;
    let file = File::open(artifact_path)
        .map_err(|_| ecapa_error("artifact_open", "ECAPA artifact could not be opened"))?;
    let actual_bytes = file
        .metadata()
        .map_err(|_| {
            ecapa_error(
                "artifact_metadata",
                "ECAPA artifact metadata is unavailable",
            )
        })?
        .len();
    if actual_bytes != expected_bytes {
        return Err(ecapa_error(
            "artifact_identity",
            "ECAPA artifact length does not match the frozen identity",
        ));
    }
    let mut reader = BufReader::new(file);
    let mut package_hasher = Sha256::new();
    let capacity = usize::try_from(expected_bytes).map_err(|_| {
        ecapa_error(
            "artifact_identity",
            "ECAPA artifact length does not fit this platform",
        )
    })?;
    let mut package_bytes = Vec::with_capacity(capacity);
    let mut buffer = [0u8; READ_CHUNK_BYTES];
    loop {
        ecapa_load_checkpoint(checkpoint)?;
        let read = reader
            .read(&mut buffer)
            .map_err(|_| ecapa_error("artifact_read", "ECAPA artifact could not be read"))?;
        if read == 0 {
            break;
        }
        if read > capacity.saturating_sub(package_bytes.len()) {
            return Err(ecapa_error(
                "artifact_identity",
                "ECAPA artifact length changed while it was read",
            ));
        }
        let chunk = buffer
            .get(..read)
            .ok_or_else(|| ecapa_error("artifact_read", "ECAPA artifact read is invalid"))?;
        package_hasher.update(chunk);
        package_bytes.extend_from_slice(chunk);
    }
    if package_bytes.len() != capacity {
        return Err(ecapa_error(
            "artifact_identity",
            "ECAPA artifact length changed while it was read",
        ));
    }
    if hex_digest(package_hasher.finalize()) != expected_sha256 {
        return Err(ecapa_error(
            "artifact_identity",
            "ECAPA artifact checksum does not match the frozen identity",
        ));
    }
    Ok(package_bytes)
}

fn verify_loaded_ecapa_package(package: &SafetensorsFile) -> FwResult<()> {
    let expected = expected_ecapa_tensors();
    if expected.len() != ECAPA_EXPORTED_TENSOR_COUNT {
        return Err(ecapa_error(
            "contract_internal_drift",
            "compiled ECAPA tensor census is inconsistent",
        ));
    }
    let payload_bytes = expected.iter().try_fold(0u64, |total, tensor| {
        let tensor_bytes = checked_element_count(&tensor.shape)?
            .checked_mul(4)
            .ok_or_else(|| ecapa_error("tensor_shape", "tensor byte count overflows"))?;
        total
            .checked_add(tensor_bytes)
            .ok_or_else(|| ecapa_error("tensor_shape", "package byte count overflows"))
    })?;
    if payload_bytes != ECAPA_EXPORTED_PAYLOAD_BYTES {
        return Err(ecapa_error(
            "contract_internal_drift",
            "compiled ECAPA tensor byte census is inconsistent",
        ));
    }
    verify_loaded_ecapa_package_against(package, &expected, &expected_ecapa_package_metadata())
}

fn verify_loaded_ecapa_full_oracle(
    oracle: &SafetensorsFile,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    frozen_ecapa_golden_evidence().map(drop)?;
    let expected = expected_ecapa_full_oracle_tensors();
    if expected.len() != ECAPA_FULL_ORACLE_TENSOR_COUNT {
        return Err(ecapa_error(
            "contract_internal_drift",
            "compiled ECAPA oracle tensor census is inconsistent",
        ));
    }
    let value_count = expected.iter().try_fold(0u64, |total, tensor| {
        total
            .checked_add(checked_element_count(&tensor.shape)?)
            .ok_or_else(|| ecapa_error("oracle_shape", "oracle value count overflows"))
    })?;
    let frontend_value_count = expected.iter().take(2).try_fold(0u64, |total, tensor| {
        total
            .checked_add(checked_element_count(&tensor.shape)?)
            .ok_or_else(|| ecapa_error("oracle_shape", "oracle frontend value count overflows"))
    })?;
    let neural_value_count = expected.iter().skip(2).try_fold(0u64, |total, tensor| {
        total
            .checked_add(checked_element_count(&tensor.shape)?)
            .ok_or_else(|| ecapa_error("oracle_shape", "oracle neural value count overflows"))
    })?;
    if value_count != ECAPA_FULL_ORACLE_F32_ELEMENTS
        || frontend_value_count != ECAPA_FULL_ORACLE_FRONTEND_F32_ELEMENTS
        || neural_value_count != ECAPA_FULL_ORACLE_NEURAL_F32_ELEMENTS
        || value_count
            .checked_mul(4)
            .ok_or_else(|| ecapa_error("oracle_shape", "oracle payload size overflows"))?
            != ECAPA_FULL_ORACLE_PAYLOAD_BYTES
    {
        return Err(ecapa_error(
            "contract_internal_drift",
            "compiled ECAPA oracle payload census is inconsistent",
        ));
    }
    let manifest = WeightsManifest::new(
        expected
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.shape.clone())),
    );
    validate(oracle, &manifest).map_err(|_| {
        ecapa_error(
            "oracle_mapping",
            "full oracle names or shapes do not match the frozen census",
        )
    })?;
    if oracle.metadata() != Some(&expected_ecapa_full_oracle_metadata()) {
        return Err(ecapa_error(
            "oracle_metadata",
            "full oracle metadata does not match the frozen provenance",
        ));
    }
    for tensor in expected {
        ecapa_load_checkpoint(checkpoint)?;
        if oracle.dtype_name(&tensor.name).map_err(|_| {
            ecapa_error(
                "oracle_mapping",
                "full oracle tensor is absent from the frozen census",
            )
        })? != "F32"
        {
            return Err(ecapa_error(
                "oracle_dtype",
                "every full oracle tensor must be F32",
            ));
        }
        let (shape, values) = oracle.tensor_f32(&tensor.name).map_err(|_| {
            ecapa_error(
                "oracle_payload",
                "full oracle tensor payload could not be decoded",
            )
        })?;
        if shape != tensor.shape
            || ecapa_reference_f32_sha256(&values, &shape)? != tensor.reference_sha256
        {
            return Err(ecapa_error(
                "oracle_payload",
                "full oracle tensor payload does not match the frozen evidence",
            ));
        }
        ecapa_load_checkpoint(checkpoint)?;
    }
    Ok(())
}

fn verify_loaded_ecapa_package_against(
    package: &SafetensorsFile,
    expected: &[EcapaTensorSpec],
    expected_metadata: &serde_json::Value,
) -> FwResult<()> {
    let manifest = WeightsManifest::new(
        expected
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.shape.clone())),
    );
    validate(package, &manifest).map_err(|error| {
        ecapa_error(
            "tensor_mapping",
            &format!("safetensors names or shapes do not match the frozen census: {error}"),
        )
    })?;
    for tensor in expected {
        if package.dtype_name(&tensor.name).map_err(|_| {
            ecapa_error(
                "tensor_mapping",
                "safetensors tensor is absent from the frozen census",
            )
        })? != "F32"
        {
            return Err(ecapa_error(
                "tensor_dtype",
                "every exported ECAPA tensor must be F32",
            ));
        }
    }
    if package.metadata() != Some(expected_metadata) {
        return Err(ecapa_error(
            "package_metadata",
            "safetensors metadata does not match the frozen export profile",
        ));
    }
    Ok(())
}

fn expected_ecapa_package_metadata() -> serde_json::Value {
    serde_json::json!({
        "converter": "franken_whisper/scripts/convert_to_safetensors.py",
        "dropped_batch_counter_count": ECAPA_DROPPED_BATCH_COUNTER_COUNT.to_string(),
        "exported_dtype": "F32",
        "exported_tensor_count": ECAPA_EXPORTED_TENSOR_COUNT.to_string(),
        "exporter_version": ECAPA_EXPORTER_VERSION,
        "numpy_version": ECAPA_EXPORT_NUMPY_VERSION,
        "profile": ECAPA_EXPORT_PROFILE,
        "safetensors_version": ECAPA_EXPORT_SAFETENSORS_VERSION,
        "source_checkpoint_bytes": ECAPA_SOURCE_CHECKPOINT_BYTES.to_string(),
        "source_checkpoint_sha256": ECAPA_SOURCE_CHECKPOINT_SHA256,
        "source_model_id": ECAPA_MODEL_ID,
        "source_model_revision": ECAPA_MODEL_REVISION,
        "source_tensor_count": ECAPA_SOURCE_TENSOR_COUNT.to_string(),
        "torch_version": ECAPA_EXPORT_TORCH_VERSION,
    })
}

fn expected_ecapa_full_oracle_metadata() -> serde_json::Value {
    serde_json::json!({
        "canonical_layout": "speechbrain_cpu_contiguous_c_order",
        "contract_sha256": ECAPA_CONTRACT_SHA256,
        "device": "cpu",
        "evaluation_mode": "true",
        "exported_dtype": "F32",
        "exported_tensor_count": ECAPA_FULL_ORACLE_TENSOR_COUNT.to_string(),
        "exporter_version": ECAPA_FULL_ORACLE_EXPORTER_VERSION,
        "fixture_id": ECAPA_FIXTURE_ID,
        "fixture_pcm_sha256": ECAPA_FIXTURE_PCM_SHA256,
        "fixture_sample_count": ECAPA_SAMPLE_RATE_HZ.to_string(),
        "generator": "franken_whisper/scripts/convert_to_safetensors.py",
        "golden_evidence_sha256": ECAPA_GOLDEN_EVIDENCE_SHA256,
        "numpy_version": ECAPA_EXPORT_NUMPY_VERSION,
        "python_version": ECAPA_ORACLE_PYTHON_VERSION,
        "safetensors_version": ECAPA_EXPORT_SAFETENSORS_VERSION,
        "schema_version": ECAPA_FULL_ORACLE_SCHEMA,
        "source_checkpoint_sha256": ECAPA_SOURCE_CHECKPOINT_SHA256,
        "source_model_id": ECAPA_MODEL_ID,
        "source_model_revision": ECAPA_MODEL_REVISION,
        "source_weight_package_sha256": ECAPA_PACKAGE_SHA256,
        "speechbrain_version": ECAPA_ORACLE_SPEECHBRAIN_VERSION,
        "torch_version": ECAPA_EXPORT_TORCH_VERSION,
        "torchaudio_version": ECAPA_ORACLE_TORCHAUDIO_VERSION,
        "training_code_revision": ECAPA_TRAINING_CODE_REVISION,
    })
}

fn checked_element_count(shape: &[usize]) -> FwResult<u64> {
    if shape.is_empty() || shape.len() > 4 || shape.contains(&0) {
        return Err(ecapa_error("tensor_shape", "tensor shape is invalid"));
    }
    shape.iter().try_fold(1u64, |product, dimension| {
        let dimension = u64::try_from(*dimension)
            .map_err(|_| ecapa_error("tensor_shape", "tensor dimension is invalid"))?;
        product
            .checked_mul(dimension)
            .ok_or_else(|| ecapa_error("tensor_shape", "tensor element count overflows"))
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EcapaConformanceStage {
    FbankPreNormalization,
    FbankSentenceMeanNormalized,
    InitialTdnn,
    FirstSeRes2,
    MultiFeatureAggregation,
    AttentivePooling,
    Embedding,
}

/// One complete tensor in the independently generated public ECAPA oracle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EcapaFullOracleTensorSpec {
    pub stage: EcapaConformanceStage,
    pub name: String,
    pub shape: Vec<usize>,
    pub reference_sha256: String,
}

/// Exact tensor census, canonical layouts, and payload hashes for the oracle.
#[must_use]
pub fn expected_ecapa_full_oracle_tensors() -> Vec<EcapaFullOracleTensorSpec> {
    vec![
        full_oracle_tensor(
            EcapaConformanceStage::FbankPreNormalization,
            "fbank_pre_normalization",
            &[1, 101, ECAPA_MEL_BANDS],
            "8fd529b6f2d3ec34d7b45bf39196ec8ebfb0c2b407d8b2e308717fe5bf8fcde8",
        ),
        full_oracle_tensor(
            EcapaConformanceStage::FbankSentenceMeanNormalized,
            "fbank_sentence_mean_normalized",
            &[1, 101, ECAPA_MEL_BANDS],
            "32afe9ace7c803c7e777e1d19ffe0630549f59da69c6593fe4aa4bff30cb5370",
        ),
        full_oracle_tensor(
            EcapaConformanceStage::InitialTdnn,
            "initial_tdnn",
            &[1, 1_024, 101],
            "18274d7866b0181b17f9d3d58d0b585d9eb99ba7c9b8fabda6d3d7d23478d112",
        ),
        full_oracle_tensor(
            EcapaConformanceStage::FirstSeRes2,
            "first_se_res2",
            &[1, 1_024, 101],
            "b37629ffd2cca7c00533cd8f2baf23a22ce6b5b7348343c10b855cc37ef7bc24",
        ),
        full_oracle_tensor(
            EcapaConformanceStage::MultiFeatureAggregation,
            "multi_feature_aggregation",
            &[1, 3_072, 101],
            "f8787f6f3fd0038d11feeb49b4e821993a9f4e890f518e03d890384e3ddbafb0",
        ),
        full_oracle_tensor(
            EcapaConformanceStage::AttentivePooling,
            "attentive_pooling",
            &[1, 6_144, 1],
            "31261217b61f9519c6756330a8e9d6797626c49ece4fe2d4ee39f18c408e62b2",
        ),
        full_oracle_tensor(
            EcapaConformanceStage::Embedding,
            "embedding",
            &[1, 1, ECAPA_EMBEDDING_DIMENSIONS],
            "ff4b056c34a75e59ff51662faa22293cc7ef18785441d584b2b61dfd0b8cb5ae",
        ),
    ]
}

fn full_oracle_tensor(
    stage: EcapaConformanceStage,
    name: &str,
    shape: &[usize],
    reference_sha256: &str,
) -> EcapaFullOracleTensorSpec {
    EcapaFullOracleTensorSpec {
        stage,
        name: name.to_owned(),
        shape: shape.to_vec(),
        reference_sha256: reference_sha256.to_owned(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EcapaTolerance {
    pub absolute: f32,
    pub relative: f32,
}

impl EcapaConformanceStage {
    #[must_use]
    pub const fn tolerance(self) -> EcapaTolerance {
        match self {
            Self::FbankPreNormalization => EcapaTolerance {
                absolute: 0.05,
                relative: 0.005,
            },
            Self::FbankSentenceMeanNormalized => EcapaTolerance {
                absolute: 0.08,
                relative: 0.005,
            },
            Self::InitialTdnn | Self::FirstSeRes2 | Self::MultiFeatureAggregation => {
                EcapaTolerance {
                    absolute: 0.002,
                    relative: 0.002,
                }
            }
            Self::AttentivePooling => EcapaTolerance {
                absolute: 0.001,
                relative: 0.002,
            },
            Self::Embedding => EcapaTolerance {
                absolute: 0.02,
                relative: 0.002,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EcapaConformanceComparison {
    pub stage: EcapaConformanceStage,
    pub compared_values: usize,
    pub maximum_absolute_error: f32,
    pub maximum_relative_error: f32,
    pub passes: bool,
}

pub fn compare_ecapa_values(
    stage: EcapaConformanceStage,
    expected: &[f32],
    observed: &[f32],
) -> FwResult<EcapaConformanceComparison> {
    if expected.is_empty() || expected.len() != observed.len() {
        return Err(ecapa_error(
            "comparison_shape",
            "conformance vectors must be non-empty and equally sized",
        ));
    }
    let tolerance = stage.tolerance();
    let mut maximum_absolute_error = 0.0f32;
    let mut maximum_relative_error = 0.0f32;
    let mut passes = true;
    for (&expected, &observed) in expected.iter().zip(observed) {
        if !expected.is_finite() || !observed.is_finite() {
            return Err(ecapa_error(
                "comparison_value",
                "conformance vectors must contain finite values",
            ));
        }
        let absolute = (expected - observed).abs();
        let relative = absolute / expected.abs().max(1.0e-6);
        if !absolute.is_finite() || !relative.is_finite() {
            return Err(ecapa_error(
                "comparison_value",
                "conformance error arithmetic must remain finite",
            ));
        }
        maximum_absolute_error = maximum_absolute_error.max(absolute);
        maximum_relative_error = maximum_relative_error.max(relative);
        passes &= absolute <= tolerance.absolute + tolerance.relative * expected.abs();
    }
    Ok(EcapaConformanceComparison {
        stage,
        compared_values: expected.len(),
        maximum_absolute_error,
        maximum_relative_error,
        passes,
    })
}

/// Hash one reference tensor in the frozen oracle byte domain.
///
/// `values` must already be ordered in the declared stage shape using
/// C-contiguous row-major order. Each finite value is serialized as one
/// little-endian IEEE-754 `f32` before SHA-256 is applied. Native time-major
/// matrices therefore need a logical transpose before hashing a declared
/// `[1, channels, time]` network stage.
pub fn ecapa_reference_f32_sha256(values: &[f32], shape: &[usize]) -> FwResult<String> {
    let expected = shape.iter().try_fold(1usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| ecapa_error("reference_shape", "reference tensor shape overflows"))
    })?;
    if shape.is_empty() || shape.contains(&0) || values.len() != expected {
        return Err(ecapa_error(
            "reference_shape",
            "reference tensor values do not match the declared shape",
        ));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ecapa_error(
            "reference_value",
            "reference tensor contains a non-finite value",
        ));
    }
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    Ok(hex_digest(hasher.finalize()))
}

/// Bounded reference frontend used only for numerical conformance.
#[derive(Clone, PartialEq)]
pub struct EcapaFrontendOutput {
    pub frame_count: usize,
    pub mel_band_count: usize,
    pub log_fbank_db: Vec<f32>,
    pub sentence_mean_normalized: Vec<f32>,
}

impl fmt::Debug for EcapaFrontendOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EcapaFrontendOutput")
            .field("frame_count", &self.frame_count)
            .field("mel_band_count", &self.mel_band_count)
            .field("log_fbank_db", &"<redacted>")
            .field("sentence_mean_normalized", &"<redacted>")
            .finish()
    }
}

/// Compute the pinned SpeechBrain 80-band frontend for a short public fixture.
///
/// This deliberately bounded scalar reference is not the later production
/// kernel. It exists so optimized safe-Rust kernels have an executable oracle.
pub fn ecapa_frontend_conformance(samples: &[f32]) -> FwResult<EcapaFrontendOutput> {
    if samples.is_empty() || samples.len() > ECAPA_MAXIMUM_CONFORMANCE_SAMPLES {
        return Err(ecapa_error(
            "frontend_length",
            "conformance PCM length is outside the bounded range",
        ));
    }
    if samples
        .iter()
        .any(|sample| !sample.is_finite() || sample.abs() > 1.0)
    {
        return Err(ecapa_error(
            "frontend_pcm",
            "conformance PCM must be finite and within [-1, 1]",
        ));
    }
    let frame_count = samples.len() / ECAPA_HOP_SAMPLES + 1;
    let value_count = frame_count
        .checked_mul(ECAPA_MEL_BANDS)
        .ok_or_else(|| ecapa_error("frontend_size", "frontend output size overflows"))?;
    let mut log_fbank_db = Vec::with_capacity(value_count);
    let twiddles = frontend_twiddles();
    let mel_filters = frontend_mel_filters();
    let mut global_max = f32::NEG_INFINITY;
    let mut power = [0.0f32; ECAPA_FFT_BINS];
    for frame_index in 0..frame_count {
        let frame_origin =
            frame_index as isize * ECAPA_HOP_SAMPLES as isize - (ECAPA_WINDOW_SAMPLES / 2) as isize;
        for (frequency, output) in power.iter_mut().enumerate() {
            let mut real = 0.0f64;
            let mut imaginary = 0.0f64;
            let twiddle_base = frequency * ECAPA_WINDOW_SAMPLES;
            for sample_index in 0..ECAPA_WINDOW_SAMPLES {
                let source_index = frame_origin + sample_index as isize;
                let sample = usize::try_from(source_index)
                    .ok()
                    .and_then(|index| samples.get(index))
                    .copied()
                    .unwrap_or(0.0);
                let windowed = f64::from(sample) * periodic_hamming(sample_index);
                let (cosine, sine) = twiddles[twiddle_base + sample_index];
                real += windowed * cosine;
                imaginary += windowed * sine;
            }
            *output = (real * real + imaginary * imaginary) as f32;
        }
        for mel in 0..ECAPA_MEL_BANDS {
            let filter_base = mel * ECAPA_FFT_BINS;
            let energy = power
                .iter()
                .enumerate()
                .map(|(frequency, value)| value * mel_filters[filter_base + frequency])
                .sum::<f32>();
            let db = 10.0 * energy.max(FRONTEND_AMIN).log10();
            global_max = global_max.max(db);
            log_fbank_db.push(db);
        }
    }
    finalize_ecapa_frontend(frame_count, log_fbank_db, global_max)
}

/// Compute the pinned SpeechBrain frontend for one admitted production window.
///
/// This path uses the same safe-Rust 400-point FFT as the native Whisper
/// frontend, with ECAPA's distinct periodic-Hamming window applied explicitly.
/// The scalar DFT in [`ecapa_frontend_conformance`] remains the independent
/// numerical oracle.
pub fn ecapa_frontend_runtime(
    samples: &[f32],
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<EcapaFrontendOutput> {
    validate_ecapa_runtime_input(samples)?;
    checkpoint()?;

    let frame_count = samples.len() / ECAPA_HOP_SAMPLES + 1;
    let value_count = frame_count
        .checked_mul(ECAPA_MEL_BANDS)
        .ok_or_else(|| ecapa_error("frontend_size", "frontend output size overflows"))?;
    let mel_filters = frontend_mel_filters();
    let mut log_fbank_db = Vec::with_capacity(value_count);
    let mut global_max = f32::NEG_INFINITY;
    let mut windowed = [0.0_f32; ECAPA_WINDOW_SAMPLES];
    let mut power = [0.0_f32; ECAPA_FFT_BINS];

    for frame_index in 0..frame_count {
        if frame_index.is_multiple_of(16) {
            checkpoint()?;
        }
        let frame_origin =
            frame_index as isize * ECAPA_HOP_SAMPLES as isize - (ECAPA_WINDOW_SAMPLES / 2) as isize;
        for (sample_index, output) in windowed.iter_mut().enumerate() {
            let source_index = frame_origin + sample_index as isize;
            let sample = usize::try_from(source_index)
                .ok()
                .and_then(|index| samples.get(index))
                .copied()
                .unwrap_or(0.0);
            *output = sample * periodic_hamming(sample_index) as f32;
        }
        crate::native_engine::mel::fixed_windowed_frame_power_spectrum(&windowed, &mut power)?;
        for mel in 0..ECAPA_MEL_BANDS {
            let filter_base = mel * ECAPA_FFT_BINS;
            let energy = power
                .iter()
                .enumerate()
                .map(|(frequency, value)| value * mel_filters[filter_base + frequency])
                .sum::<f32>();
            let db = 10.0 * energy.max(FRONTEND_AMIN).log10();
            global_max = global_max.max(db);
            log_fbank_db.push(db);
        }
    }
    checkpoint()?;
    finalize_ecapa_frontend(frame_count, log_fbank_db, global_max)
}

fn finalize_ecapa_frontend(
    frame_count: usize,
    mut log_fbank_db: Vec<f32>,
    global_max: f32,
) -> FwResult<EcapaFrontendOutput> {
    if frame_count == 0 || !global_max.is_finite() {
        return Err(ecapa_error(
            "frontend_value",
            "frontend produced no finite frames",
        ));
    }
    let floor = global_max - FRONTEND_TOP_DB;
    for value in &mut log_fbank_db {
        *value = value.max(floor);
    }
    let mut means = [0.0f64; ECAPA_MEL_BANDS];
    let (frames, remainder) = log_fbank_db.as_chunks::<ECAPA_MEL_BANDS>();
    if !remainder.is_empty() {
        return Err(ecapa_error(
            "frontend_size",
            "frontend output is not frame aligned",
        ));
    }
    for frame in frames {
        for (mean, value) in means.iter_mut().zip(frame) {
            *mean += f64::from(*value);
        }
    }
    for mean in &mut means {
        *mean /= frame_count as f64;
    }
    let sentence_mean_normalized = log_fbank_db
        .iter()
        .enumerate()
        .map(|(index, value)| *value - means[index % ECAPA_MEL_BANDS] as f32)
        .collect();
    Ok(EcapaFrontendOutput {
        frame_count,
        mel_band_count: ECAPA_MEL_BANDS,
        log_fbank_db,
        sentence_mean_normalized,
    })
}

/// Enforce the product-side duration and PCM boundary separately from fixtures.
pub fn validate_ecapa_runtime_input(samples: &[f32]) -> FwResult<()> {
    if !(ECAPA_MINIMUM_RUNTIME_SAMPLES..=ECAPA_MAXIMUM_RUNTIME_SAMPLES).contains(&samples.len()) {
        return Err(ecapa_error(
            "runtime_duration",
            "neural speaker window is outside the admitted duration range",
        ));
    }
    if samples
        .iter()
        .any(|sample| !sample.is_finite() || sample.abs() > 1.0)
    {
        return Err(ecapa_error(
            "runtime_pcm",
            "neural speaker PCM must be finite and within [-1, 1]",
        ));
    }
    Ok(())
}

/// Reject audio which has not passed the exact frontend normalization boundary.
///
/// Resampling and downmixing belong to the existing audio-normalization stage,
/// not to the model. This check prevents accidental inference over mislabeled
/// 8 kHz samples or interleaved multichannel PCM.
pub fn validate_ecapa_input_format(sample_rate_hz: usize, channels: usize) -> FwResult<()> {
    if sample_rate_hz != ECAPA_SAMPLE_RATE_HZ {
        return Err(ecapa_error(
            "input_sample_rate",
            "ECAPA input must be normalized to exactly 16000 Hz",
        ));
    }
    if channels != 1 {
        return Err(ecapa_error(
            "input_channels",
            "ECAPA input must be normalized to exactly one mono channel",
        ));
    }
    Ok(())
}

/// Apply the post-network normalization frozen by the contract.
pub fn normalize_ecapa_embedding(embedding: &mut [f32]) -> FwResult<()> {
    if embedding.len() != ECAPA_EMBEDDING_DIMENSIONS
        || embedding.iter().any(|value| !value.is_finite())
    {
        return Err(ecapa_error(
            "embedding_shape",
            "ECAPA embedding must contain exactly 192 finite values",
        ));
    }
    let squared_norm = embedding
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>();
    let norm = squared_norm.sqrt();
    if !norm.is_finite() || norm < f64::from(EMBEDDING_NORM_EPSILON) {
        return Err(ecapa_error(
            "embedding_norm",
            "ECAPA embedding norm is below the admitted minimum",
        ));
    }
    for value in embedding {
        *value = (f64::from(*value) / norm) as f32;
    }
    Ok(())
}

fn frontend_twiddles() -> &'static [(f64, f64)] {
    static TWIDDLES: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    TWIDDLES.get_or_init(|| {
        let mut values = Vec::with_capacity(ECAPA_FFT_BINS * ECAPA_WINDOW_SAMPLES);
        for frequency in 0..ECAPA_FFT_BINS {
            for sample in 0..ECAPA_WINDOW_SAMPLES {
                let angle = -2.0 * std::f64::consts::PI * frequency as f64 * sample as f64
                    / ECAPA_WINDOW_SAMPLES as f64;
                values.push((angle.cos(), angle.sin()));
            }
        }
        values
    })
}

fn frontend_mel_filters() -> &'static [f32] {
    static FILTERS: OnceLock<Vec<f32>> = OnceLock::new();
    FILTERS.get_or_init(|| {
        let maximum_mel = 2_595.0f32 * (1.0f32 + 8_000.0f32 / 700.0f32).log10();
        let mut hz = Vec::with_capacity(ECAPA_MEL_BANDS + 2);
        for index in 0..ECAPA_MEL_BANDS + 2 {
            let mel = maximum_mel * index as f32 / (ECAPA_MEL_BANDS + 1) as f32;
            hz.push(700.0 * (10.0f32.powf(mel / 2_595.0) - 1.0));
        }
        let mut filters = vec![0.0f32; ECAPA_MEL_BANDS * ECAPA_FFT_BINS];
        for mel in 0..ECAPA_MEL_BANDS {
            let center = hz[mel + 1];
            let band = hz[mel + 1] - hz[mel];
            for frequency in 0..ECAPA_FFT_BINS {
                let hz = 8_000.0 * frequency as f32 / (ECAPA_FFT_BINS - 1) as f32;
                let slope = (hz - center) / band;
                filters[mel * ECAPA_FFT_BINS + frequency] =
                    0.0f32.max((slope + 1.0).min(1.0 - slope));
            }
        }
        filters
    })
}

fn periodic_hamming(index: usize) -> f64 {
    0.54 - 0.46 * (2.0 * std::f64::consts::PI * index as f64 / ECAPA_WINDOW_SAMPLES as f64).cos()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EcapaGoldenEvidence {
    pub schema_version: String,
    pub model_revision: String,
    pub oracle_speechbrain_version: String,
    pub oracle_torch_version: String,
    pub fixture_id: String,
    pub fixture_pcm_sha256: String,
    pub fixture_sample_count: usize,
    pub stages: Vec<EcapaGoldenStage>,
    pub evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EcapaGoldenStage {
    pub stage: EcapaConformanceStage,
    pub shape: Vec<usize>,
    pub reference_sha256: String,
    pub points: Vec<EcapaGoldenPoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EcapaGoldenPoint {
    pub index: Vec<usize>,
    pub value: f32,
}

/// Public analytic fixture evidence generated through the pinned model.
pub fn frozen_ecapa_golden_evidence() -> FwResult<EcapaGoldenEvidence> {
    let mut evidence = EcapaGoldenEvidence {
        schema_version: ECAPA_GOLDEN_SCHEMA.to_owned(),
        model_revision: ECAPA_MODEL_REVISION.to_owned(),
        oracle_speechbrain_version: ECAPA_ORACLE_SPEECHBRAIN_VERSION.to_owned(),
        oracle_torch_version: ECAPA_EXPORT_TORCH_VERSION.to_owned(),
        fixture_id: ECAPA_FIXTURE_ID.to_owned(),
        fixture_pcm_sha256: ECAPA_FIXTURE_PCM_SHA256.to_owned(),
        fixture_sample_count: ECAPA_SAMPLE_RATE_HZ,
        stages: vec![
            golden_stage(
                EcapaConformanceStage::FbankPreNormalization,
                &[1, 101, 80],
                "8fd529b6f2d3ec34d7b45bf39196ec8ebfb0c2b407d8b2e308717fe5bf8fcde8",
                &[
                    (&[0, 0, 0], 8.821_338),
                    (&[0, 1, 10], 0.172_932_5),
                    (&[0, 25, 40], -50.939_823),
                    (&[0, 50, 79], -22.926_104),
                    (&[0, 100, 1], 12.569_429),
                ],
            ),
            golden_stage(
                EcapaConformanceStage::FbankSentenceMeanNormalized,
                &[1, 101, 80],
                "32afe9ace7c803c7e777e1d19ffe0630549f59da69c6593fe4aa4bff30cb5370",
                &[
                    (&[0, 0, 0], 32.338_14),
                    (&[0, 1, 10], 0.086_695_48),
                    (&[0, 25, 40], -21.164_53),
                    (&[0, 50, 79], 11.397_993),
                    (&[0, 100, 1], 29.971_508),
                ],
            ),
            golden_stage(
                EcapaConformanceStage::InitialTdnn,
                &[1, 1_024, 101],
                "18274d7866b0181b17f9d3d58d0b585d9eb99ba7c9b8fabda6d3d7d23478d112",
                &[
                    (&[0, 0, 0], 0.028_214_231),
                    (&[0, 0, 50], 0.069_608_38),
                    (&[0, 17, 23], -0.181_440_04),
                    (&[0, 511, 50], 0.039_706_632),
                    (&[0, 1_023, 100], -0.086_247_01),
                ],
            ),
            golden_stage(
                EcapaConformanceStage::FirstSeRes2,
                &[1, 1_024, 101],
                "b37629ffd2cca7c00533cd8f2baf23a22ce6b5b7348343c10b855cc37ef7bc24",
                &[
                    (&[0, 0, 0], 0.101_756_126),
                    (&[0, 0, 50], 0.495_432_76),
                    (&[0, 17, 23], 0.261_554_36),
                    (&[0, 511, 50], -0.185_389_71),
                    (&[0, 1_023, 100], -0.181_889_56),
                ],
            ),
            golden_stage(
                EcapaConformanceStage::MultiFeatureAggregation,
                &[1, 3_072, 101],
                "f8787f6f3fd0038d11feeb49b4e821993a9f4e890f518e03d890384e3ddbafb0",
                &[
                    (&[0, 0, 0], -0.138_677_43),
                    (&[0, 0, 50], -0.138_677_43),
                    (&[0, 17, 23], 0.020_858_98),
                    (&[0, 1_535, 50], -0.038_260_62),
                    (&[0, 3_071, 100], -0.036_232_326),
                ],
            ),
            golden_stage(
                EcapaConformanceStage::AttentivePooling,
                &[1, 6_144, 1],
                "31261217b61f9519c6756330a8e9d6797626c49ece4fe2d4ee39f18c408e62b2",
                &[
                    (&[0, 0, 0], -0.130_729_08),
                    (&[0, 17, 0], 0.009_684_245),
                    (&[0, 3_071, 0], -0.005_759_360_3),
                    (&[0, 3_072, 0], 0.036_187_105),
                    (&[0, 6_143, 0], 0.019_499_47),
                ],
            ),
            golden_stage(
                EcapaConformanceStage::Embedding,
                &[1, 1, ECAPA_EMBEDDING_DIMENSIONS],
                "ff4b056c34a75e59ff51662faa22293cc7ef18785441d584b2b61dfd0b8cb5ae",
                &[
                    (&[0, 0, 0], 13.187_779),
                    (&[0, 0, 1], -10.624_47),
                    (&[0, 0, 17], -9.378_475),
                    (&[0, 0, 96], 17.990_408),
                    (&[0, 0, 191], 25.197_382),
                ],
            ),
        ],
        evidence_sha256: String::new(),
    };
    evidence.evidence_sha256 = canonical_sha256(&evidence)?;
    verify_ecapa_golden_evidence(&evidence)?;
    Ok(evidence)
}

fn golden_stage(
    stage: EcapaConformanceStage,
    shape: &[usize],
    reference_sha256: &str,
    points: &[(&[usize], f32)],
) -> EcapaGoldenStage {
    EcapaGoldenStage {
        stage,
        shape: shape.to_vec(),
        reference_sha256: reference_sha256.to_owned(),
        points: points
            .iter()
            .map(|(index, value)| EcapaGoldenPoint {
                index: index.to_vec(),
                value: *value,
            })
            .collect(),
    }
}

pub fn verify_ecapa_golden_evidence(evidence: &EcapaGoldenEvidence) -> FwResult<()> {
    if evidence.schema_version != ECAPA_GOLDEN_SCHEMA
        || evidence.model_revision != ECAPA_MODEL_REVISION
        || evidence.oracle_speechbrain_version != ECAPA_ORACLE_SPEECHBRAIN_VERSION
        || evidence.oracle_torch_version != ECAPA_EXPORT_TORCH_VERSION
        || evidence.fixture_id != ECAPA_FIXTURE_ID
        || evidence.fixture_pcm_sha256 != ECAPA_FIXTURE_PCM_SHA256
        || evidence.fixture_sample_count != ECAPA_SAMPLE_RATE_HZ
        || evidence.stages.len() != ECAPA_FULL_ORACLE_TENSOR_COUNT
    {
        return Err(ecapa_error(
            "golden_identity",
            "ECAPA golden evidence identity is invalid",
        ));
    }
    let expected_stages = expected_ecapa_full_oracle_tensors();
    for (stage, expected_stage) in evidence.stages.iter().zip(expected_stages) {
        if stage.stage != expected_stage.stage
            || stage.shape != expected_stage.shape
            || stage.reference_sha256 != expected_stage.reference_sha256
            || stage.points.is_empty()
            || stage.points.iter().any(|point| {
                point.index.len() != stage.shape.len()
                    || point
                        .index
                        .iter()
                        .zip(&stage.shape)
                        .any(|(index, dimension)| index >= dimension)
                    || !point.value.is_finite()
            })
        {
            return Err(ecapa_error("golden_stage", "ECAPA golden stage is invalid"));
        }
    }
    if !is_sha256_hex(&evidence.evidence_sha256) {
        return Err(ecapa_error(
            "golden_hash",
            "ECAPA golden evidence hash is invalid",
        ));
    }
    let mut unhashed = evidence.clone();
    unhashed.evidence_sha256.clear();
    if canonical_sha256(&unhashed)? != evidence.evidence_sha256 {
        return Err(ecapa_error(
            "golden_hash",
            "ECAPA golden evidence hash does not match",
        ));
    }
    if evidence.evidence_sha256 != ECAPA_GOLDEN_EVIDENCE_SHA256 {
        return Err(ecapa_error(
            "golden_version",
            "ECAPA golden evidence does not match the frozen version",
        ));
    }
    Ok(())
}

fn canonical_sha256<T: Serialize>(value: &T) -> FwResult<String> {
    Ok(bytes_sha256(&serde_json::to_vec(value)?))
}

fn bytes_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_digest(hasher.finalize())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == HASH_HEX_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
pub(crate) fn ecapa_analytic_fixture() -> Vec<f32> {
    (0..ECAPA_SAMPLE_RATE_HZ)
        .map(|index| {
            let time = index as f64 / ECAPA_SAMPLE_RATE_HZ as f64;
            let chirp_phase = 2.0 * std::f64::consts::PI * (120.0 * time + 180.0 * time * time);
            let mut value = 0.22 * (2.0 * std::f64::consts::PI * 173.0 * time).sin();
            value += 0.11 * (2.0 * std::f64::consts::PI * 347.0 * time).sin();
            value += 0.07 * chirp_phase.sin();
            if index == 1_234 {
                value += 0.5;
            }
            value as f32
        })
        .collect()
}

fn ecapa_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("ecapa.{code}: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_safetensors(
        name: &str,
        dtype: &str,
        shape: &[usize],
        payload: &[u8],
        metadata: &serde_json::Value,
    ) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_owned(), metadata.clone());
        header.insert(
            name.to_owned(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [0, payload.len()],
            }),
        );
        let mut header_bytes =
            serde_json::to_vec(&serde_json::Value::Object(header)).expect("header");
        header_bytes.resize(header_bytes.len().next_multiple_of(8), b' ');
        let mut package = Vec::with_capacity(8 + header_bytes.len() + payload.len());
        package.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        package.extend_from_slice(&header_bytes);
        package.extend_from_slice(payload);
        package
    }

    #[test]
    fn frozen_contract_and_inventory_match_the_pinned_checkpoint() {
        let contract = frozen_ecapa_contract();
        assert_eq!(contract.license_spdx, "Apache-2.0");
        assert_eq!(contract.model_revision, ECAPA_MODEL_REVISION);
        assert_eq!(contract.frontend.mel_bands, 80);
        assert!(!contract.export.batch_norm_folding);
        assert_eq!(contract.export.package_format, "safetensors");
        assert_eq!(contract.export.package_sha256, ECAPA_PACKAGE_SHA256);
        assert_eq!(contract.export.package_bytes, ECAPA_PACKAGE_BYTES);
        assert_eq!(contract.architecture.embedding_dimensions, 192);
        assert_eq!(
            contract.architecture.golden_embedding_stage,
            "raw_model_output_before_l2_normalization"
        );
        assert_eq!(
            ecapa_contract_sha256().expect("contract hash"),
            ECAPA_CONTRACT_SHA256
        );

        let tensors = expected_ecapa_tensors();
        assert_eq!(tensors.len(), ECAPA_EXPORTED_TENSOR_COUNT);
        assert_eq!(tensors[0].name, "blocks.0.conv.conv.weight");
        assert_eq!(tensors[0].shape, vec![1_024, 80, 5]);
        assert_eq!(tensors[60].name, "blocks.1.se_block.conv1.conv.weight");
        assert!(
            tensors
                .iter()
                .any(|tensor| tensor.name == "asp.conv.conv.weight")
        );
        assert!(
            tensors
                .iter()
                .all(|tensor| tensor.name != "asp.conv.conv.conv.weight")
        );
        assert_eq!(
            tensors.last().map(|tensor| tensor.name.as_str()),
            Some("fc.conv.bias")
        );
        let elements = tensors
            .iter()
            .map(|tensor| checked_element_count(&tensor.shape).expect("shape"))
            .sum::<u64>();
        assert_eq!(elements, ECAPA_EXPORTED_F32_ELEMENTS);

        let oracle_tensors = expected_ecapa_full_oracle_tensors();
        assert_eq!(oracle_tensors.len(), ECAPA_FULL_ORACLE_TENSOR_COUNT);
        let oracle_elements = oracle_tensors
            .iter()
            .map(|tensor| checked_element_count(&tensor.shape).expect("oracle shape"))
            .sum::<u64>();
        assert_eq!(oracle_elements, ECAPA_FULL_ORACLE_F32_ELEMENTS);
    }

    #[test]
    fn package_identity_detects_corruption_truncation_and_cancellation() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("weights.safetensors");
        let payload = b"abcdefgh";
        std::fs::write(&path, payload).expect("write payload");
        let no_cancel = || Ok(());
        let verified = read_verified_ecapa_artifact(
            &path,
            payload.len() as u64,
            &bytes_sha256(payload),
            &no_cancel,
        )
        .expect("valid identity");
        assert_eq!(verified, payload);

        assert!(
            read_verified_ecapa_artifact(&path, payload.len() as u64, &"f".repeat(64), &no_cancel,)
                .is_err()
        );
        assert!(
            read_verified_ecapa_artifact(
                &path,
                payload.len() as u64 + 1,
                &bytes_sha256(payload),
                &no_cancel,
            )
            .is_err()
        );
        let expired = CancellationToken::already_expired();
        let cancel = || expired.checkpoint();
        assert!(matches!(
            read_verified_ecapa_artifact(
                &path,
                payload.len() as u64,
                &bytes_sha256(payload),
                &cancel,
            ),
            Err(FwError::Cancelled(_))
        ));

        let secret_failure = || {
            Err(FwError::InvalidRequest(
                "confidential caller checkpoint detail".to_owned(),
            ))
        };
        let error = read_verified_ecapa_artifact(
            &path,
            payload.len() as u64,
            &bytes_sha256(payload),
            &secret_failure,
        )
        .expect_err("non-cancellation checkpoint failure must fail closed");
        let rendered = error.to_string();
        assert!(rendered.contains("ecapa.checkpoint_failure"));
        assert!(!rendered.contains("confidential"));

        std::fs::write(&path, payload).expect("restore payload");
        let checkpoint_calls = std::sync::atomic::AtomicUsize::new(0);
        let append_after_metadata = || {
            if checkpoint_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 1 {
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(&path)
                    .expect("open package for concurrent append");
                std::io::Write::write_all(&mut file, b"extra bytes").expect("append package bytes");
            }
            Ok(())
        };
        let error = read_verified_ecapa_artifact(
            &path,
            payload.len() as u64,
            &bytes_sha256(payload),
            &append_after_metadata,
        )
        .expect_err("growth after metadata validation must fail closed");
        assert!(
            error
                .to_string()
                .contains("ECAPA artifact length changed while it was read")
        );
    }

    #[test]
    fn safetensors_wrapper_rejects_mapping_dtype_and_metadata_drift() {
        let expected_metadata = serde_json::json!({"profile": "test"});
        let expected = vec![EcapaTensorSpec {
            name: "weight".to_owned(),
            shape: vec![1],
        }];

        let valid_bytes = tiny_safetensors(
            "weight",
            "F32",
            &[1],
            &0.25f32.to_le_bytes(),
            &expected_metadata,
        );
        let valid = SafetensorsFile::from_bytes(&valid_bytes).expect("valid safetensors");
        verify_loaded_ecapa_package_against(&valid, &expected, &expected_metadata)
            .expect("valid package contract");

        let wrong_shape_bytes =
            tiny_safetensors("weight", "F32", &[2], &[0; 8], &expected_metadata);
        let wrong_shape =
            SafetensorsFile::from_bytes(&wrong_shape_bytes).expect("shape package parses");
        let shape_error =
            verify_loaded_ecapa_package_against(&wrong_shape, &expected, &expected_metadata)
                .expect_err("shape drift fails");
        assert!(shape_error.to_string().contains("ecapa.tensor_mapping"));

        let wrong_dtype_bytes =
            tiny_safetensors("weight", "F16", &[1], &[0; 2], &expected_metadata);
        let wrong_dtype =
            SafetensorsFile::from_bytes(&wrong_dtype_bytes).expect("dtype package parses");
        let dtype_error =
            verify_loaded_ecapa_package_against(&wrong_dtype, &expected, &expected_metadata)
                .expect_err("dtype drift fails");
        assert!(dtype_error.to_string().contains("ecapa.tensor_dtype"));

        let wrong_metadata = serde_json::json!({"profile": "wrong"});
        let metadata_bytes = tiny_safetensors("weight", "F32", &[1], &[0; 4], &wrong_metadata);
        let metadata_package =
            SafetensorsFile::from_bytes(&metadata_bytes).expect("metadata package parses");
        let metadata_error =
            verify_loaded_ecapa_package_against(&metadata_package, &expected, &expected_metadata)
                .expect_err("metadata drift fails");
        assert!(
            metadata_error
                .to_string()
                .contains("ecapa.package_metadata")
        );
    }

    #[test]
    #[ignore = "requires an externally converted public ECAPA package"]
    fn external_frozen_safetensors_package_verifies_end_to_end() {
        let path = std::env::var_os("FRANKEN_WHISPER_ECAPA_TEST_WEIGHTS")
            .map(std::path::PathBuf::from)
            .expect("set FRANKEN_WHISPER_ECAPA_TEST_WEIGHTS");
        verify_ecapa_weight_package(&path).expect("frozen package");
    }

    #[test]
    #[ignore = "requires the externally generated public ECAPA full oracle"]
    fn external_full_oracle_verifies_end_to_end() {
        let path = std::env::var_os("FRANKEN_WHISPER_ECAPA_TEST_ORACLE")
            .map(std::path::PathBuf::from)
            .expect("set FRANKEN_WHISPER_ECAPA_TEST_ORACLE");
        let oracle = load_verified_ecapa_full_oracle(&path).expect("frozen full oracle");
        assert_eq!(oracle.len(), ECAPA_FULL_ORACLE_TENSOR_COUNT);
    }

    #[test]
    fn frontend_matches_independent_speechbrain_golden_points() {
        let samples = ecapa_analytic_fixture();
        assert_eq!(
            bytes_sha256(
                &samples
                    .iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect::<Vec<_>>()
            ),
            "acc240c07370020bbd1b3aaf9b8b81be43ef053b8da950969e86f62b6f1dba2f"
        );
        let output = ecapa_frontend_conformance(&samples).expect("frontend");
        assert_eq!(output.frame_count, 101);
        let evidence = frozen_ecapa_golden_evidence().expect("golden evidence");
        for stage in evidence.stages.iter().take(2) {
            let values = if stage.stage == EcapaConformanceStage::FbankPreNormalization {
                &output.log_fbank_db
            } else {
                &output.sentence_mean_normalized
            };
            let observed = stage
                .points
                .iter()
                .map(|point| values[point.index[1] * ECAPA_MEL_BANDS + point.index[2]])
                .collect::<Vec<_>>();
            let expected = stage
                .points
                .iter()
                .map(|point| point.value)
                .collect::<Vec<_>>();
            let comparison =
                compare_ecapa_values(stage.stage, &expected, &observed).expect("comparison");
            assert!(
                comparison.passes,
                "{:?} max_abs={} max_rel={}",
                stage.stage, comparison.maximum_absolute_error, comparison.maximum_relative_error
            );
            assert!(
                comparison.maximum_absolute_error < 1.0e-3,
                "{:?} scalar oracle drifted by {}",
                stage.stage,
                comparison.maximum_absolute_error
            );
        }
    }

    #[test]
    fn runtime_frontend_matches_scalar_oracle() {
        let samples = ecapa_analytic_fixture();
        let expected = ecapa_frontend_conformance(&samples).expect("scalar frontend");
        let actual = ecapa_frontend_runtime(&samples, &|| Ok(())).expect("runtime frontend");
        assert_eq!(actual.frame_count, expected.frame_count);
        assert_eq!(actual.mel_band_count, expected.mel_band_count);
        for (stage, expected_values, actual_values) in [
            (
                "log_fbank_db",
                expected.log_fbank_db.as_slice(),
                actual.log_fbank_db.as_slice(),
            ),
            (
                "sentence_mean_normalized",
                expected.sentence_mean_normalized.as_slice(),
                actual.sentence_mean_normalized.as_slice(),
            ),
        ] {
            let maximum_error = expected_values
                .iter()
                .zip(actual_values)
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                maximum_error <= 2.0e-3,
                "{stage} runtime FFT drifted from the scalar oracle by {maximum_error}"
            );
        }
    }

    #[test]
    fn runtime_frontend_honors_validation_and_cancellation() {
        let admitted = vec![0.0; ECAPA_MINIMUM_RUNTIME_SAMPLES];
        assert!(ecapa_frontend_runtime(&admitted, &|| Ok(())).is_ok());
        assert!(
            ecapa_frontend_runtime(&admitted[..ECAPA_MINIMUM_RUNTIME_SAMPLES - 1], &|| Ok(()))
                .is_err()
        );
        let error = ecapa_frontend_runtime(&admitted, &|| {
            Err(FwError::Cancelled("test cancellation".to_owned()))
        })
        .expect_err("cancelled frontend fails");
        assert!(matches!(error, FwError::Cancelled(_)));
    }

    #[test]
    fn frontend_edge_cases_are_bounded_and_finite() {
        let silence = vec![0.0; 401];
        let output = ecapa_frontend_conformance(&silence).expect("silence");
        assert_eq!(output.frame_count, 3);
        assert!(
            output
                .sentence_mean_normalized
                .iter()
                .all(|value| value.abs() < 1.0e-6)
        );
        let mut clipped = vec![0.0; 799];
        clipped[0] = -1.0;
        clipped[1] = 1.0;
        let output = ecapa_frontend_conformance(&clipped).expect("odd clipped fixture");
        assert!(
            output
                .log_fbank_db
                .iter()
                .chain(&output.sentence_mean_normalized)
                .all(|value| value.is_finite())
        );
        assert!(ecapa_frontend_conformance(&[]).is_err());
        assert!(ecapa_frontend_conformance(&[f32::NAN]).is_err());
        assert!(ecapa_frontend_conformance(&[1.01]).is_err());
        assert!(
            ecapa_frontend_conformance(&vec![0.0; ECAPA_MAXIMUM_CONFORMANCE_SAMPLES + 1]).is_err()
        );
    }

    #[test]
    fn runtime_input_has_an_explicit_duration_range() {
        assert!(validate_ecapa_input_format(ECAPA_SAMPLE_RATE_HZ, 1).is_ok());
        assert!(validate_ecapa_input_format(8_000, 1).is_err());
        assert!(validate_ecapa_input_format(ECAPA_SAMPLE_RATE_HZ, 2).is_err());
        assert!(validate_ecapa_runtime_input(&vec![0.0; ECAPA_MINIMUM_RUNTIME_SAMPLES]).is_ok());
        assert!(validate_ecapa_runtime_input(&vec![0.0; ECAPA_MAXIMUM_RUNTIME_SAMPLES]).is_ok());
        assert!(
            validate_ecapa_runtime_input(&vec![0.0; ECAPA_MINIMUM_RUNTIME_SAMPLES - 1]).is_err()
        );
        assert!(
            validate_ecapa_runtime_input(&vec![0.0; ECAPA_MAXIMUM_RUNTIME_SAMPLES + 1]).is_err()
        );
    }

    #[test]
    fn frontend_debug_redacts_audio_derived_features() {
        let output = EcapaFrontendOutput {
            frame_count: 1,
            mel_band_count: ECAPA_MEL_BANDS,
            log_fbank_db: vec![12_345.625],
            sentence_mean_normalized: vec![-54_321.125],
        };
        let rendered = format!("{output:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("12345.625"));
        assert!(!rendered.contains("54321.125"));
    }

    #[test]
    fn embedding_normalization_is_explicit_and_fail_closed() {
        let mut embedding = vec![0.0; ECAPA_EMBEDDING_DIMENSIONS];
        embedding[0] = 3.0;
        embedding[1] = 4.0;
        normalize_ecapa_embedding(&mut embedding).expect("normalize");
        assert!((embedding[0] - 0.6).abs() < 1.0e-6);
        assert!((embedding[1] - 0.8).abs() < 1.0e-6);
        let norm = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1.0e-6);

        assert!(normalize_ecapa_embedding(&mut vec![0.0; 191]).is_err());
        assert!(normalize_ecapa_embedding(&mut vec![0.0; 192]).is_err());
        let mut nonfinite = vec![0.0; 192];
        nonfinite[0] = f32::INFINITY;
        assert!(normalize_ecapa_embedding(&mut nonfinite).is_err());
    }

    #[test]
    fn comparison_rejects_nonfinite_and_applies_stage_tolerances() {
        assert!(compare_ecapa_values(EcapaConformanceStage::Embedding, &[], &[]).is_err());
        assert!(
            compare_ecapa_values(EcapaConformanceStage::Embedding, &[0.0], &[f32::NAN]).is_err()
        );
        assert!(
            compare_ecapa_values(EcapaConformanceStage::Embedding, &[f32::MAX], &[-f32::MAX],)
                .is_err()
        );
        assert!(
            compare_ecapa_values(EcapaConformanceStage::Embedding, &[10.0], &[10.01])
                .expect("within tolerance")
                .passes
        );
        assert!(
            !compare_ecapa_values(EcapaConformanceStage::Embedding, &[10.0], &[11.0])
                .expect("outside tolerance")
                .passes
        );
    }

    #[test]
    fn golden_evidence_is_ordered_self_hashed_and_content_free() {
        let evidence = frozen_ecapa_golden_evidence().expect("golden");
        verify_ecapa_golden_evidence(&evidence).expect("verify");
        assert_eq!(evidence.evidence_sha256, ECAPA_GOLDEN_EVIDENCE_SHA256);
        let json = serde_json::to_string(&evidence).expect("json");
        for forbidden in [
            "transcript",
            "audio_path",
            "speaker_name",
            "embedding_values",
        ] {
            assert!(!json.contains(forbidden));
        }
        let mut tampered = evidence;
        tampered.stages[0].points[0].value += 1.0;
        assert!(verify_ecapa_golden_evidence(&tampered).is_err());
    }
}
