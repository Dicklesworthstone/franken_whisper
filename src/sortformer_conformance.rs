//! Fail-closed L0 artifact verifier for the native Streaming Sortformer port.
//!
//! This module does not run the model and does not admit a production route.
//! It authenticates an operator-local, non-executable safetensors conversion
//! against an independently reviewed canonical receipt. Production admission
//! uses compiled converter, topology-projection, receipt, and package trust
//! roots; arbitrary caller-supplied digests are not an admissible authority.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{FwError, FwResult};
use crate::native_engine::weights::SafetensorsFile;

pub use crate::differential_oracle::SORTFORMER_ORACLE_ADAPTER_SHA256;

pub const SORTFORMER_RECEIPT_SCHEMA: &str = "franken-whisper-sortformer-conversion-receipt-v1";
pub const SORTFORMER_TENSOR_MANIFEST_SCHEMA: &str = "franken-whisper-sortformer-tensor-manifest-v1";
/// Receipt schema label only. The compiled converter-source and canonical
/// receipt digests below are the executable and instance trust roots.
pub const SORTFORMER_CONVERTER_ID: &str = "franken-whisper-native-sortformer-converter";
pub const SORTFORMER_CONVERTER_VERSION: &str = "1";
pub const SORTFORMER_CONVERTER_SOURCE_SHA256: &str =
    "6a946cc6647bf52244d0eaad89db834bdc52cc61fd08d9563632dd1f9d239c1e";
pub const SORTFORMER_CONVERSION_RECEIPT_SHA256: &str =
    "a1c6dce95ef4fd715965951bdaaa136e55e2219f93cf78122f8b462fbd07cbbe";
pub const SORTFORMER_MODEL_ID: &str = "nvidia/diar_streaming_sortformer_4spk-v2.1";
pub const SORTFORMER_MODEL_REVISION: &str = "fafaab5faa1617a0ca52d38dd3dc4bd636800d3d";
pub const SORTFORMER_NEMO_BYTES: u64 = 471_367_680;
pub const SORTFORMER_NEMO_SHA256: &str =
    "8abd32832159c6ac1148c926b7276f35ba34582c444e559dce1f1253fea42ef8";
pub const SORTFORMER_CONFIG_SHA256: &str =
    "2865d469c4d2aac54aa5b8a956b2423c053806dd20d5bf5d08675942a1acface";
pub const SORTFORMER_CHECKPOINT_BYTES: u64 = 471_352_898;
pub const SORTFORMER_CHECKPOINT_SHA256: &str =
    "eca9773c2dab91dd41fbaa4473cebb9d00811d67788ce2de609dadc6e499cdf4";
pub const SORTFORMER_STATE_INVENTORY_SHA256: &str =
    "f4f219cf4ac6f755247b56d19e425db3d6a7c23c4509176549b363b63abdf532";
pub const SORTFORMER_TENSOR_MANIFEST_SHA256: &str =
    "2c32b0b9e48bb296e66615b038827d0fdde4b4fda2ce044a6c30cd317456c8d7";
pub const SORTFORMER_NEMO_SOURCE_REVISION: &str = "40ace43c7cf151af78dc22027c02feeca7e06b6a";
pub const SORTFORMER_EXTERNAL_CONTRACT_SHA256: &str =
    "7ac048e3372fe4c622840beddfbeef42944d961408360324cb7276a69c8542c5";
pub const SORTFORMER_RUNTIME_FINGERPRINT_SHA256: &str =
    "3713fd3f024c1cef7d860706baf0dbaaf18058c03c26331da6254687693d564c";
pub const SORTFORMER_PARAMETER_TENSORS: u64 = 937;
pub const SORTFORMER_TRAINABLE_PARAMETERS: u64 = 117_693_960;
pub const SORTFORMER_STATE_TENSORS: u64 = 990;
pub const SORTFORMER_STATE_ELEMENTS: u64 = 117_744_681;
pub const SORTFORMER_STATE_F32_TENSORS: u64 = 973;
pub const SORTFORMER_STATE_F32_ELEMENTS: u64 = 117_744_664;
pub const SORTFORMER_STATE_F32_BYTES: u64 = 470_978_656;
pub const SORTFORMER_STATE_I64_TENSORS: u64 = 17;
pub const SORTFORMER_STATE_PAYLOAD_BYTES: u64 = 470_978_792;
pub const SORTFORMER_SOURCE_RECORDS: u64 = 992;
pub const SORTFORMER_EXPORTED_TENSORS: u64 = 974;
pub const SORTFORMER_DROPPED_TENSORS: u64 = 18;
pub const SORTFORMER_PACKAGE_F32_ELEMENTS: u64 = 122_864_152;
pub const SORTFORMER_PACKAGE_PAYLOAD_BYTES: u64 = 491_456_608;
pub const SORTFORMER_PACKAGE_BYTES: u64 = 491_570_584;
pub const SORTFORMER_PACKAGE_SHA256: &str =
    "487fa30cb0aa9799c77bd9985e6787962c3991fab8d4d576a4f1221d45298f6a";

pub const SORTFORMER_ACTIVATION_RECEIPT_SCHEMA: &str =
    "franken-whisper-sortformer-activation-receipt-v1";
pub const SORTFORMER_ACTIVATION_FLOOR_SCHEMA: &str = "franken-whisper-sortformer-oracle-floor-v1";
pub const SORTFORMER_ACTIVATION_RECEIPT_SHA256: &str =
    "058ecfebe91cea90dd669c28ee3e8976ef3ea3e3494f9dc0c5b93f0a26fa17f2";
pub const SORTFORMER_ACTIVATION_PACKAGE_SHA256: &str =
    "294edcc0a9d80fa9470c2cd45f2c1556a47a56b7c98ba444984f764a1f398a8b";
pub const SORTFORMER_ACTIVATION_EXPORTER_SHA256: &str =
    "a5e4a8a29e1ce8e4227ff8973a6279fe5aa560a57e5f06947be2a14cf68adf33";
pub const SORTFORMER_ACTIVATION_PACKAGE_BYTES: u64 = 282_716;
pub const SORTFORMER_ACTIVATION_PAYLOAD_BYTES: u64 = 278_076;
pub const SORTFORMER_ACTIVATION_F32_ELEMENTS: u64 = 69_503;
pub const SORTFORMER_ACTIVATION_I64_ELEMENTS: u64 = 8;
pub const SORTFORMER_ACTIVATION_TENSORS: u64 = 46;

pub const SORTFORMER_POSITION_TENSOR: &str = "encoder.pos_enc.pe";
pub const SORTFORMER_DTYPE_SENTINEL: &str = "preprocessor.dtype_sentinel_tensor";
pub const SORTFORMER_LICENSE_URL: &str =
    "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/";
pub const SORTFORMER_MODEL_LICENSE_SNAPSHOT_SHA256: &str =
    "13c9c998e24abd5211cff4b5c912902f566bd710294da98580be7b3376626f04";
pub const SORTFORMER_NEMO_LICENSE_SHA256: &str =
    "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1";

const HASH_HEX_LEN: usize = 64;
const READ_CHUNK_BYTES: usize = 64 * 1024;
const MAX_RECEIPT_BYTES: u64 = 8 * 1024 * 1024;
const MAX_PACKAGE_BYTES: u64 = 640 * 1024 * 1024;
const MAX_PACKAGE_HEADER_BYTES: u64 = 8 * 1024 * 1024;
const MAX_ACTIVATION_RECEIPT_BYTES: u64 = 1024 * 1024;
const MAX_ACTIVATION_PACKAGE_BYTES: u64 = 4 * 1024 * 1024;
const F32_BYTES: u64 = 4;
const I64_BYTES: u64 = 8;
const SOURCE_LAYOUT: &str = "pytorch_contiguous_row_major";
const PACKAGE_FORMAT: &str = "safetensors";
const PACKAGE_DTYPE: &str = "f32";
const PACKAGE_BYTE_ORDER: &str = "little_endian";
const PACKAGE_TENSOR_ORDER: &str = "lexicographic_name_order";
const PACKAGE_METADATA_POLICY: &str = "absent";
const LICENSE_ID: &str = "NVIDIA Open Model License";
const LICENSE_POLICY: &str = "operator_local_no_git_no_release";
const EMPTY_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Complete canonical conversion receipt. The receipt intentionally contains
/// no self-hash; its expected digest must arrive through an independent trust
/// channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerConversionReceipt {
    pub schema_version: String,
    pub model: SortformerModelIdentity,
    pub execution: SortformerExecutionConfig,
    pub source_files: Vec<SortformerSourceFileIdentity>,
    pub converter: SortformerConverterIdentity,
    pub runtime: SortformerRuntimeIdentity,
    pub license: SortformerLicenseIdentity,
    pub package: SortformerPackageIdentity,
    pub records: Vec<SortformerTensorRecord>,
}

/// Execution-defining configuration that is unsafe to infer from tensor names
/// or hide behind the archive-config digest alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerExecutionConfig {
    pub streaming_mode: bool,
    pub async_streaming: bool,
    pub encoder_attention_context: [i64; 2],
    pub encoder_attention_style: String,
    pub transformer_mask_future: bool,
    pub transformer_pre_ln: bool,
    pub drop_extra_pre_encoded: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerModelIdentity {
    pub model_id: String,
    pub model_revision: String,
    pub nemo_bytes: u64,
    pub nemo_sha256: String,
    pub config_sha256: String,
    pub checkpoint_bytes: u64,
    pub checkpoint_sha256: String,
    pub state_inventory_sha256: String,
    pub tensor_manifest_sha256: String,
    pub nemo_source_revision: String,
    pub external_contract_sha256: String,
    pub runtime_fingerprint_sha256: String,
    pub oracle_adapter_sha256: String,
    pub trainable_parameters: u64,
    pub parameter_tensors: u64,
    pub state_tensors: u64,
    pub state_elements: u64,
    pub state_f32_tensors: u64,
    pub state_f32_elements: u64,
    pub state_f32_bytes: u64,
    pub state_i64_tensors: u64,
    pub state_payload_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerSourceFileIdentity {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerConverterIdentity {
    pub converter_id: String,
    pub converter_version: String,
    pub source_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerRuntimeIdentity {
    pub python: String,
    pub nemo: String,
    pub torch: String,
    pub torchaudio: String,
    pub numpy: String,
    pub safetensors: String,
    pub librosa: String,
    pub lhotse: String,
    pub soundfile: String,
    pub scipy: String,
    pub omegaconf: String,
    pub hydra_core: String,
    pub lightning: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerLicenseIdentity {
    pub model_license_id: String,
    pub model_license_url: String,
    pub model_license_snapshot_retrieved_date: String,
    pub model_license_last_modified: String,
    pub model_license_etag: String,
    pub model_license_payload_sha256: String,
    pub model_weight_distribution_policy: String,
    pub nemo_source_license_spdx: String,
    pub nemo_source_license_sha256: String,
    pub embedded_notice_source_path: String,
    pub embedded_notice_source_sha256: String,
    pub embedded_notice_license_spdx: String,
    pub embedded_notice_attribution: String,
    pub embedded_notice_attribution_required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerPackageIdentity {
    pub format: String,
    pub sha256: String,
    pub bytes: u64,
    pub payload_bytes: u64,
    pub f32_elements: u64,
    pub tensor_count: u64,
    pub dtype: String,
    pub byte_order: String,
    pub tensor_order: String,
    pub logical_layout: String,
    pub metadata_policy: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortformerTensorOrigin {
    StateDict,
    NonpersistentBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortformerTensorDtype {
    F32,
    I64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortformerTensorTransform {
    IdentityContiguousF32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerTensorRecord {
    pub source_name: String,
    pub source_origin: SortformerTensorOrigin,
    pub source_dtype: SortformerTensorDtype,
    pub source_shape: Vec<u64>,
    pub source_logical_layout: String,
    pub source_value_sha256: String,
    pub source_elements: u64,
    pub source_bytes: u64,
    pub disposition: SortformerTensorDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SortformerTensorDisposition {
    Exported {
        transform: SortformerTensorTransform,
        destination: SortformerDestinationTensor,
    },
    DroppedTrainOnly,
    DroppedRuntimeSentinel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerDestinationTensor {
    pub name: String,
    pub dtype: SortformerTensorDtype,
    pub shape: Vec<u64>,
    pub logical_layout: String,
    pub value_sha256: String,
    pub elements: u64,
    pub bytes: u64,
}

/// Canonical receipt for the deterministic, non-human L1 frontend truth pack.
/// The receipt and tensor payload remain operator-local; only their independent
/// trust roots are compiled into the library.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerActivationReceipt {
    pub schema_version: String,
    pub canonical_json_version: String,
    pub authority: String,
    pub equivalence_level: String,
    pub fixture_set: String,
    pub model: SortformerActivationModelIdentity,
    pub exporter: SortformerActivationExporterIdentity,
    pub runtime: SortformerRuntimeIdentity,
    pub source_files: Vec<SortformerSourceFileIdentity>,
    pub execution: SortformerActivationExecutionIdentity,
    pub fixtures: Vec<SortformerActivationFixture>,
    pub oracle_floor: SortformerActivationOracleFloor,
    pub package: SortformerActivationPackageIdentity,
    pub records: Vec<SortformerActivationTensorRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerActivationModelIdentity {
    pub model_id: String,
    pub model_revision: String,
    pub nemo_sha256: String,
    pub nemo_bytes: u64,
    pub config_sha256: String,
    pub checkpoint_sha256: String,
    pub state_inventory_sha256: String,
    pub nemo_source_revision: String,
    pub external_contract_sha256: String,
    pub conversion_receipt_sha256: String,
    pub converted_package_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerActivationExporterIdentity {
    pub exporter_id: String,
    pub exporter_version: String,
    pub source_sha256: String,
    pub conversion_helper_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerActivationExecutionIdentity {
    pub operating_system: String,
    pub machine_architecture: String,
    pub device: String,
    pub compute_dtype: String,
    pub autocast: bool,
    pub quantization: String,
    pub deterministic_algorithms: bool,
    pub torch_intraop_thread_counts: Vec<u64>,
    pub torch_interop_threads: u64,
    pub data_loader_workers: u64,
    pub torch_blas_backend: String,
    pub torch_configuration_sha256: String,
    pub numpy_configuration_sha256: String,
    pub python_executable_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerActivationFixture {
    pub name: String,
    pub generator: String,
    pub generator_parameters_sha256: String,
    pub sample_rate_hz: u64,
    pub channels: u64,
    pub sample_count: u64,
    pub valid_frames: u64,
    pub physical_frames: u64,
    pub pcm16_sha256: String,
    pub decoded_f32_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerActivationOracleFloor {
    pub schema_version: String,
    pub baseline_threads: u64,
    pub baseline_repetition: u64,
    pub thread_counts: Vec<u64>,
    pub repetitions_per_thread: u64,
    pub all_byte_exact: bool,
    pub mismatch_count: u64,
    pub comparison_rule: String,
    pub absolute_tolerance_f32_bits: String,
    pub relative_tolerance_f32_bits: String,
    pub margin_basis: String,
    pub observations: Vec<SortformerActivationFloorObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerActivationFloorObservation {
    pub fixture: String,
    pub stage: String,
    pub run_count: u64,
    pub pair_count: u64,
    pub compared_values: u64,
    pub mismatch_count: u64,
    pub byte_exact: bool,
    pub max_abs_diff_f32_bits: String,
    pub mean_abs_diff_f64_bits: String,
    pub relative_l2_f64_bits: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerActivationPackageIdentity {
    pub format: String,
    pub dtype_set: Vec<SortformerTensorDtype>,
    pub byte_order: String,
    pub tensor_order: String,
    pub logical_layout: String,
    pub metadata_policy: String,
    pub tensor_count: u64,
    pub f32_elements: u64,
    pub i64_elements: u64,
    pub payload_bytes: u64,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerActivationTensorRecord {
    pub name: String,
    pub dtype: SortformerTensorDtype,
    pub shape: Vec<u64>,
    pub logical_layout: String,
    pub elements: u64,
    pub bytes: u64,
    pub value_sha256: String,
}

/// Authenticated receipt and the exact owned package bytes it authorized.
pub struct VerifiedSortformerPackage {
    receipt: SortformerConversionReceipt,
    package: SafetensorsFile,
}

impl fmt::Debug for VerifiedSortformerPackage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedSortformerPackage")
            .field("receipt_schema", &self.receipt.schema_version)
            .field("model_id", &self.receipt.model.model_id)
            .field("package_sha256", &self.receipt.package.sha256)
            .field("tensor_count", &self.receipt.package.tensor_count)
            .field("package", &"<authenticated model bytes redacted>")
            .finish()
    }
}

impl VerifiedSortformerPackage {
    /// Borrow the immutable authenticated receipt.
    pub const fn receipt(&self) -> &SortformerConversionReceipt {
        &self.receipt
    }

    /// Borrow the authenticated package without permitting callers outside the
    /// crate to unwrap it into an unauthenticated generic weight container.
    #[allow(dead_code)] // consumed by the f32 inference slice after L0 admission
    pub(crate) const fn safetensors(&self) -> &SafetensorsFile {
        &self.package
    }
}

/// Authenticated synthetic frontend truth pack. Debug output never exposes
/// activation values, even though the frozen fixtures contain no human audio.
pub struct VerifiedSortformerActivationPack {
    receipt: SortformerActivationReceipt,
    package: SafetensorsFile,
}

impl fmt::Debug for VerifiedSortformerActivationPack {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedSortformerActivationPack")
            .field("receipt_schema", &self.receipt.schema_version)
            .field("fixture_set", &self.receipt.fixture_set)
            .field("package_sha256", &self.receipt.package.sha256)
            .field("tensor_count", &self.receipt.package.tensor_count)
            .field("package", &"<authenticated synthetic activations redacted>")
            .finish()
    }
}

impl VerifiedSortformerActivationPack {
    /// Borrow the immutable authenticated diagnostic receipt.
    pub const fn receipt(&self) -> &SortformerActivationReceipt {
        &self.receipt
    }

    /// Borrow the exact authenticated tensors for an in-crate parity probe.
    pub(crate) const fn safetensors(&self) -> &SafetensorsFile {
        &self.package
    }
}

/// Authenticate the frozen, synthetic-only L1 frontend truth pack.
///
/// This does not promote the Sortformer route and does not accept a caller-
/// selected checksum. Both files must match independently compiled trust roots.
pub fn load_verified_sortformer_activation_pack(
    receipt_path: &Path,
    package_path: &Path,
) -> FwResult<VerifiedSortformerActivationPack> {
    load_verified_sortformer_activation_pack_with_checkpoint(receipt_path, package_path, &|| Ok(()))
}

/// Authenticate the frozen truth pack with cooperative cancellation between
/// bounded reads and tensor validations.
pub fn load_verified_sortformer_activation_pack_with_checkpoint(
    receipt_path: &Path,
    package_path: &Path,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<VerifiedSortformerActivationPack> {
    sortformer_checkpoint(checkpoint, SortformerArtifactDomain::Activation)?;
    let receipt_bytes = read_bounded_file(
        receipt_path,
        "activation_receipt",
        MAX_ACTIVATION_RECEIPT_BYTES,
        None,
        SortformerArtifactDomain::Activation,
        checkpoint,
    )?;
    if sha256_bytes(&receipt_bytes) != SORTFORMER_ACTIVATION_RECEIPT_SHA256 {
        return Err(sortformer_activation_error(
            "receipt_identity",
            "activation receipt checksum does not match the independent trust root",
        ));
    }
    let receipt: SortformerActivationReceipt =
        serde_json::from_slice(&receipt_bytes).map_err(|_| {
            sortformer_activation_error(
                "receipt_schema",
                "activation receipt is not valid strict receipt JSON",
            )
        })?;
    let canonical = canonical_json_bytes(&receipt).map_err(|_| {
        sortformer_activation_error(
            "receipt_schema",
            "activation receipt could not be serialized canonically",
        )
    })?;
    if canonical != receipt_bytes {
        return Err(sortformer_activation_error(
            "receipt_canonical",
            "activation receipt bytes are not the canonical JSON encoding",
        ));
    }
    verify_activation_receipt(&receipt, checkpoint)?;

    let package_bytes = read_bounded_file(
        package_path,
        "activation_package",
        MAX_ACTIVATION_PACKAGE_BYTES,
        Some((receipt.package.bytes, receipt.package.sha256.as_str())),
        SortformerArtifactDomain::Activation,
        checkpoint,
    )?;
    verify_compact_safetensors_layout(
        &package_bytes,
        receipt.package.payload_bytes,
        SortformerArtifactDomain::Activation,
        checkpoint,
    )?;
    let package = SafetensorsFile::from_owned_bytes(package_bytes).map_err(|_| {
        sortformer_activation_error(
            "package_structure",
            "activation package is not structurally valid safetensors",
        )
    })?;
    verify_activation_package(&package, &receipt, checkpoint)?;
    sortformer_checkpoint(checkpoint, SortformerArtifactDomain::Activation)?;
    Ok(VerifiedSortformerActivationPack { receipt, package })
}

/// Authenticate the frozen pinned-model census using the compiled, independently
/// reviewed canonical-receipt digest.
pub fn load_verified_sortformer_package(
    receipt_path: &Path,
    package_path: &Path,
) -> FwResult<VerifiedSortformerPackage> {
    load_verified_sortformer_package_with_checkpoint(receipt_path, package_path, &|| Ok(()))
}

/// Authenticate the frozen pinned-model census with cooperative cancellation.
/// The checkpoint runs before opening either file, between every bounded read,
/// and between tensor validations.
pub fn load_verified_sortformer_package_with_checkpoint(
    receipt_path: &Path,
    package_path: &Path,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<VerifiedSortformerPackage> {
    load_verified_sortformer_package_against(
        receipt_path,
        package_path,
        SORTFORMER_CONVERSION_RECEIPT_SHA256,
        checkpoint,
        &pinned_model_expectations(),
    )
}

#[derive(Debug, Clone)]
struct ReceiptExpectations {
    model: SortformerModelIdentity,
    execution: SortformerExecutionConfig,
    converter_source_sha256: String,
    package_sha256: String,
    package_bytes: u64,
    source_files: Vec<SortformerSourceFileIdentity>,
    runtime: SortformerRuntimeIdentity,
    license: SortformerLicenseIdentity,
    counter_names: BTreeSet<String>,
    position_shape: Vec<u64>,
    state_tensors: u64,
    state_elements: u64,
    state_f32_tensors: u64,
    state_f32_elements: u64,
    state_f32_bytes: u64,
    state_i64_tensors: u64,
    state_payload_bytes: u64,
    source_records: u64,
    exported_tensors: u64,
    dropped_tensors: u64,
    package_f32_elements: u64,
    package_payload_bytes: u64,
}

fn pinned_model_expectations() -> ReceiptExpectations {
    let counter_names = (0..17)
        .map(|layer| format!("encoder.layers.{layer}.conv.batch_norm.num_batches_tracked"))
        .collect();
    ReceiptExpectations {
        model: frozen_model_identity(),
        execution: frozen_execution_config(),
        converter_source_sha256: SORTFORMER_CONVERTER_SOURCE_SHA256.to_owned(),
        package_sha256: SORTFORMER_PACKAGE_SHA256.to_owned(),
        package_bytes: SORTFORMER_PACKAGE_BYTES,
        source_files: frozen_source_files(),
        runtime: frozen_runtime_identity(),
        license: frozen_license_identity(),
        counter_names,
        position_shape: vec![1, 9_999, 512],
        state_tensors: SORTFORMER_STATE_TENSORS,
        state_elements: SORTFORMER_STATE_ELEMENTS,
        state_f32_tensors: SORTFORMER_STATE_F32_TENSORS,
        state_f32_elements: SORTFORMER_STATE_F32_ELEMENTS,
        state_f32_bytes: SORTFORMER_STATE_F32_BYTES,
        state_i64_tensors: SORTFORMER_STATE_I64_TENSORS,
        state_payload_bytes: SORTFORMER_STATE_PAYLOAD_BYTES,
        source_records: SORTFORMER_SOURCE_RECORDS,
        exported_tensors: SORTFORMER_EXPORTED_TENSORS,
        dropped_tensors: SORTFORMER_DROPPED_TENSORS,
        package_f32_elements: SORTFORMER_PACKAGE_F32_ELEMENTS,
        package_payload_bytes: SORTFORMER_PACKAGE_PAYLOAD_BYTES,
    }
}

fn frozen_execution_config() -> SortformerExecutionConfig {
    SortformerExecutionConfig {
        streaming_mode: true,
        async_streaming: false,
        encoder_attention_context: [-1, -1],
        encoder_attention_style: "regular".to_owned(),
        transformer_mask_future: false,
        transformer_pre_ln: false,
        drop_extra_pre_encoded: 0,
    }
}

fn frozen_model_identity() -> SortformerModelIdentity {
    SortformerModelIdentity {
        model_id: SORTFORMER_MODEL_ID.to_owned(),
        model_revision: SORTFORMER_MODEL_REVISION.to_owned(),
        nemo_bytes: SORTFORMER_NEMO_BYTES,
        nemo_sha256: SORTFORMER_NEMO_SHA256.to_owned(),
        config_sha256: SORTFORMER_CONFIG_SHA256.to_owned(),
        checkpoint_bytes: SORTFORMER_CHECKPOINT_BYTES,
        checkpoint_sha256: SORTFORMER_CHECKPOINT_SHA256.to_owned(),
        state_inventory_sha256: SORTFORMER_STATE_INVENTORY_SHA256.to_owned(),
        tensor_manifest_sha256: SORTFORMER_TENSOR_MANIFEST_SHA256.to_owned(),
        nemo_source_revision: SORTFORMER_NEMO_SOURCE_REVISION.to_owned(),
        external_contract_sha256: SORTFORMER_EXTERNAL_CONTRACT_SHA256.to_owned(),
        runtime_fingerprint_sha256: SORTFORMER_RUNTIME_FINGERPRINT_SHA256.to_owned(),
        oracle_adapter_sha256: SORTFORMER_ORACLE_ADAPTER_SHA256.to_owned(),
        trainable_parameters: SORTFORMER_TRAINABLE_PARAMETERS,
        parameter_tensors: SORTFORMER_PARAMETER_TENSORS,
        state_tensors: SORTFORMER_STATE_TENSORS,
        state_elements: SORTFORMER_STATE_ELEMENTS,
        state_f32_tensors: SORTFORMER_STATE_F32_TENSORS,
        state_f32_elements: SORTFORMER_STATE_F32_ELEMENTS,
        state_f32_bytes: SORTFORMER_STATE_F32_BYTES,
        state_i64_tensors: SORTFORMER_STATE_I64_TENSORS,
        state_payload_bytes: SORTFORMER_STATE_PAYLOAD_BYTES,
    }
}

fn frozen_runtime_identity() -> SortformerRuntimeIdentity {
    SortformerRuntimeIdentity {
        python: "3.12.12".to_owned(),
        nemo: "3.1.0+40ace43c7c".to_owned(),
        torch: "2.7.1".to_owned(),
        torchaudio: "2.7.1".to_owned(),
        numpy: "2.4.6".to_owned(),
        safetensors: "0.8.0".to_owned(),
        librosa: "0.11.0".to_owned(),
        lhotse: "1.33.0".to_owned(),
        soundfile: "0.14.0".to_owned(),
        scipy: "1.18.0".to_owned(),
        omegaconf: "2.3.0".to_owned(),
        hydra_core: "1.3.2".to_owned(),
        lightning: "2.4.0".to_owned(),
    }
}

fn frozen_license_identity() -> SortformerLicenseIdentity {
    SortformerLicenseIdentity {
        model_license_id: LICENSE_ID.to_owned(),
        model_license_url: SORTFORMER_LICENSE_URL.to_owned(),
        model_license_snapshot_retrieved_date: "2026-08-06".to_owned(),
        model_license_last_modified: "Mon, 03 Aug 2026 17:46:28 GMT".to_owned(),
        model_license_etag: "4b001-658281e31650b".to_owned(),
        model_license_payload_sha256: SORTFORMER_MODEL_LICENSE_SNAPSHOT_SHA256.to_owned(),
        model_weight_distribution_policy: LICENSE_POLICY.to_owned(),
        nemo_source_license_spdx: "Apache-2.0".to_owned(),
        nemo_source_license_sha256: SORTFORMER_NEMO_LICENSE_SHA256.to_owned(),
        embedded_notice_source_path: "nemo/collections/asr/parts/preprocessing/features.py"
            .to_owned(),
        embedded_notice_source_sha256:
            "4290ed2d697362a68a6158fb8b7b8d1e2306b223b83172c63fc6b5d31b28ee69".to_owned(),
        embedded_notice_license_spdx: "MIT".to_owned(),
        embedded_notice_attribution: "Ryan Leary".to_owned(),
        embedded_notice_attribution_required: true,
    }
}

fn frozen_source_files() -> Vec<SortformerSourceFileIdentity> {
    [
        (
            "nemo/collections/asr/data/audio_to_diar_label.py",
            "f9b0d23bd52da417ac18418ea1c83aa1119f59e6b37d3b2b3159c8cb2f036234",
        ),
        (
            "nemo/collections/asr/models/sortformer_diar_models.py",
            "4978dba1a02b414893123f66905a1e523d5bb65766903269b325746c67f6920a",
        ),
        (
            "nemo/collections/asr/modules/audio_preprocessing.py",
            "c061f521e14978d22ad57fa5ddf08f1103c2d1f1a4e01aca6698bfad007e8e7c",
        ),
        (
            "nemo/collections/asr/modules/conformer_encoder.py",
            "a8b6f712cdf75a3be768848e8242ea9412ca7ff31ba2dda6b9602bcefc627cec",
        ),
        (
            "nemo/collections/asr/modules/sortformer_modules.py",
            "3d136c245e3bf7a88c47fdd2eae1edb9189bbeddc3ff779cb5679a29d890b7eb",
        ),
        (
            "nemo/collections/asr/modules/transformer/transformer_encoders.py",
            "a2859c86c8389f1954d5c8be04dc2bc422452517ef15e069cf42bfab5d304759",
        ),
        (
            "nemo/collections/asr/modules/transformer/transformer_modules.py",
            "2564d95365cfafd486b1a3d10e2e2f438702907076f3716dd4c42d568b3bcc72",
        ),
        (
            "nemo/collections/asr/parts/mixins/diarization.py",
            "5365e416ecab192cf59f1b9d6554ebce0ed3bdb2fee7575966ac1e3fca1a1408",
        ),
        (
            "nemo/collections/asr/parts/preprocessing/features.py",
            "4290ed2d697362a68a6158fb8b7b8d1e2306b223b83172c63fc6b5d31b28ee69",
        ),
        (
            "nemo/collections/asr/parts/preprocessing/segment.py",
            "a598d91b94110e0c12a1ba4a57894ce89109e597fa8e909cf7b5b6e7bb9369af",
        ),
        (
            "nemo/collections/asr/parts/submodules/causal_convs.py",
            "7cf505c8caef44a37a7dec10b51eb2d60ec2f1efc3a2badc3c20c37e427cbd42",
        ),
        (
            "nemo/collections/asr/parts/submodules/conformer_modules.py",
            "99bb846c51db028d6d30b3d844af22826068aeaa0e48eb586489a31a9cbacf9d",
        ),
        (
            "nemo/collections/asr/parts/submodules/multi_head_attention.py",
            "4999fd0d679fd7315ba275f7311fe6608c48e492bd337f2e220c99b8b9729c69",
        ),
        (
            "nemo/collections/asr/parts/submodules/subsampling.py",
            "4fbc689f3f66e4630b286196315a02b315ad53e8049c164fe40dd11168cf0834",
        ),
        (
            "nemo/collections/asr/parts/utils/speaker_utils.py",
            "6c247bdda26fd010190e1c96f8399f77a5265a180086e134d9b167b3c8019dc0",
        ),
        (
            "nemo/collections/asr/parts/utils/vad_utils.py",
            "7beb57efff5e08407f9f16afe9c0da7d0e2ddb9bd62e2a37424693e48c5f0437",
        ),
        (
            "nemo/collections/common/parts/transformer_utils.py",
            "47f5e337230e7b4e176877f01c2ae85f75c024942dc567f27d8429c3e60e67c0",
        ),
    ]
    .into_iter()
    .map(|(path, sha256)| SortformerSourceFileIdentity {
        path: path.to_owned(),
        sha256: sha256.to_owned(),
    })
    .collect()
}

fn frozen_activation_model_identity() -> SortformerActivationModelIdentity {
    SortformerActivationModelIdentity {
        model_id: SORTFORMER_MODEL_ID.to_owned(),
        model_revision: SORTFORMER_MODEL_REVISION.to_owned(),
        nemo_sha256: SORTFORMER_NEMO_SHA256.to_owned(),
        nemo_bytes: SORTFORMER_NEMO_BYTES,
        config_sha256: SORTFORMER_CONFIG_SHA256.to_owned(),
        checkpoint_sha256: SORTFORMER_CHECKPOINT_SHA256.to_owned(),
        state_inventory_sha256: SORTFORMER_STATE_INVENTORY_SHA256.to_owned(),
        nemo_source_revision: SORTFORMER_NEMO_SOURCE_REVISION.to_owned(),
        external_contract_sha256: SORTFORMER_EXTERNAL_CONTRACT_SHA256.to_owned(),
        conversion_receipt_sha256: SORTFORMER_CONVERSION_RECEIPT_SHA256.to_owned(),
        converted_package_sha256: SORTFORMER_PACKAGE_SHA256.to_owned(),
    }
}

fn frozen_activation_execution_identity() -> SortformerActivationExecutionIdentity {
    SortformerActivationExecutionIdentity {
        operating_system: "macOS-26.2-arm64-arm-64bit".to_owned(),
        machine_architecture: "arm64".to_owned(),
        device: "cpu".to_owned(),
        compute_dtype: "float32".to_owned(),
        autocast: false,
        quantization: "none".to_owned(),
        deterministic_algorithms: true,
        torch_intraop_thread_counts: vec![1, 8],
        torch_interop_threads: 1,
        data_loader_workers: 0,
        torch_blas_backend: "accelerate".to_owned(),
        torch_configuration_sha256:
            "ffc2f8b252a5c30391e838728b510b913f0063309951158cc33b7d363d345f2b".to_owned(),
        numpy_configuration_sha256:
            "b426b1270fa4f246379b5567f7c96ac171017279a8a3dfa8887b2eb51b455882".to_owned(),
        python_executable_sha256:
            "93c469a68969bd462e2ae6ebdcb595a1afe73aa7be31866de7a3e257948de9a0".to_owned(),
    }
}

fn frozen_activation_fixtures() -> Vec<SortformerActivationFixture> {
    [
        (
            "silence_320",
            "all_zero_i16_v1",
            "ebfc9c10ec42c1f6d524ebde6539160f344219286dee207f102e07fdd410ecac",
            320,
            2,
            3,
            "9e132485d5107211de325a45e7917cbe3e4b5b9cde3e4ee91d7d2102317759ee",
            "bfe492baf731a0dbf6e1e050f5bc3fe8c1b049383194dcdf82f023bfa409f462",
        ),
        (
            "impulse_480",
            "three_exact_impulses_i16_v1",
            "bfc0a95306d9e2bbf08d7c15838197bfa863b135d3001edb48d7ff2efe71fd2a",
            480,
            3,
            4,
            "b22c0f4537971eafdd326ea466245567f4506d051bcbae5fcfe710ad41a53212",
            "b3d74f0a8436daa0671e91fc16173a39eee5291df2b68ef12a911c26de9d5144",
        ),
        (
            "tone_640",
            "exact_i16_cycle_v1",
            "b4d07daaff3e5a9dff103fe461c0f15f4ffa50ea275eb97d18a63b33b52a68e2",
            640,
            4,
            5,
            "a7437b32c5d0d92d8f76be7848ad77379b505ce30490a5d035a92444a10f82b2",
            "f9cc66972a50c4007baf5a18e6e3337371a3c1a3980bb0a9b27208514ea3cdd8",
        ),
        (
            "partial_tail_321",
            "exact_integer_lcg_i16_v1",
            "8167c861a0da0339ec0762cc1e26452ed26060cc7a63079b1788b639a0406fdf",
            321,
            2,
            3,
            "3eb7dafa8d83ab68e1f82cb6fe9cdb8bf10547978828b93334a310649ed0a70e",
            "593f11442a170edf264f0b8b7ad6ab2615007550fbe486fc2885cc7d3e2c0e32",
        ),
    ]
    .into_iter()
    .map(
        |(
            name,
            generator,
            generator_parameters_sha256,
            sample_count,
            valid_frames,
            physical_frames,
            pcm16_sha256,
            decoded_f32_sha256,
        )| SortformerActivationFixture {
            name: name.to_owned(),
            generator: generator.to_owned(),
            generator_parameters_sha256: generator_parameters_sha256.to_owned(),
            sample_rate_hz: 16_000,
            channels: 1,
            sample_count,
            valid_frames,
            physical_frames,
            pcm16_sha256: pcm16_sha256.to_owned(),
            decoded_f32_sha256: decoded_f32_sha256.to_owned(),
        },
    )
    .collect()
}

#[derive(Debug, Clone)]
struct ActivationTensorContract {
    dtype: SortformerTensorDtype,
    shape: Vec<u64>,
    value_sha256: Option<String>,
}

fn expected_activation_tensor_contracts(
    fixtures: &[SortformerActivationFixture],
) -> BTreeMap<String, ActivationTensorContract> {
    let mut expected = BTreeMap::new();
    expected.insert(
        "analysis_window_f32".to_owned(),
        ActivationTensorContract {
            dtype: SortformerTensorDtype::F32,
            shape: vec![400],
            value_sha256: Some(
                "7d6b2ab4944b0b65650e1bba1132821fd1d2ed000df84dbd893316788d0ef062".to_owned(),
            ),
        },
    );
    expected.insert(
        "mel_filterbank_f32".to_owned(),
        ActivationTensorContract {
            dtype: SortformerTensorDtype::F32,
            shape: vec![1, 128, 257],
            value_sha256: Some(
                "82663f1145f6965d8b27a85f32a44fa4f3bffef9bd0d6c2d1902b334a012367b".to_owned(),
            ),
        },
    );

    for fixture in fixtures {
        let prefix = format!("fixture.{}", fixture.name);
        let definitions = [
            (
                "decoded_pcm_f32",
                SortformerTensorDtype::F32,
                vec![1, fixture.sample_count],
                Some(fixture.decoded_f32_sha256.clone()),
            ),
            (
                "frontend_padded_f32",
                SortformerTensorDtype::F32,
                vec![1, 128, 16],
                None,
            ),
            (
                "input_length_i64",
                SortformerTensorDtype::I64,
                vec![1],
                None,
            ),
            (
                "log_mel_f32",
                SortformerTensorDtype::F32,
                vec![1, 128, fixture.valid_frames],
                None,
            ),
            (
                "log_mel_physical_f32",
                SortformerTensorDtype::F32,
                vec![1, 128, fixture.physical_frames],
                None,
            ),
            (
                "mel_energy_f32",
                SortformerTensorDtype::F32,
                vec![1, 128, fixture.physical_frames],
                None,
            ),
            (
                "power_f32",
                SortformerTensorDtype::F32,
                vec![1, 257, fixture.physical_frames],
                None,
            ),
            (
                "preemphasis_f32",
                SortformerTensorDtype::F32,
                vec![1, fixture.sample_count],
                None,
            ),
            (
                "stft_complex_ri_f32",
                SortformerTensorDtype::F32,
                vec![1, 257, fixture.physical_frames, 2],
                None,
            ),
            (
                "valid_length_i64",
                SortformerTensorDtype::I64,
                vec![1],
                None,
            ),
            (
                "windowed_frames_f32",
                SortformerTensorDtype::F32,
                vec![1, fixture.physical_frames, 512],
                None,
            ),
        ];
        for (stage, dtype, shape, value_sha256) in definitions {
            expected.insert(
                format!("{prefix}.{stage}"),
                ActivationTensorContract {
                    dtype,
                    shape,
                    value_sha256,
                },
            );
        }
    }
    expected
}

fn verify_activation_receipt(
    receipt: &SortformerActivationReceipt,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    if receipt.schema_version != SORTFORMER_ACTIVATION_RECEIPT_SCHEMA
        || receipt.canonical_json_version != "lexicographic-json-v1"
    {
        return Err(sortformer_activation_error(
            "receipt_schema",
            "activation receipt is not the frozen canonical v1 schema",
        ));
    }
    if receipt.authority != "diagnostic_only"
        || receipt.equivalence_level != "partial_l1_synthetic_frontend"
        || receipt.fixture_set != "sortformer-synthetic-frontend-v1"
    {
        return Err(sortformer_activation_error(
            "authority",
            "activation receipt overstates or changes its diagnostic authority",
        ));
    }
    if receipt.model != frozen_activation_model_identity() {
        return Err(sortformer_activation_error(
            "model_identity",
            "activation receipt model identity does not match the converted-model trust chain",
        ));
    }
    if receipt.exporter
        != (SortformerActivationExporterIdentity {
            exporter_id: "franken-whisper-sortformer-activation-exporter".to_owned(),
            exporter_version: "1".to_owned(),
            source_sha256: SORTFORMER_ACTIVATION_EXPORTER_SHA256.to_owned(),
            conversion_helper_sha256: SORTFORMER_CONVERTER_SOURCE_SHA256.to_owned(),
        })
    {
        return Err(sortformer_activation_error(
            "exporter_identity",
            "activation exporter identity does not match the frozen executable source",
        ));
    }
    if receipt.runtime != frozen_runtime_identity()
        || receipt.source_files != frozen_source_files()
        || receipt.execution != frozen_activation_execution_identity()
    {
        return Err(sortformer_activation_error(
            "runtime_identity",
            "activation source or runtime identity does not match the frozen oracle",
        ));
    }
    let fixtures = frozen_activation_fixtures();
    if receipt.fixtures != fixtures {
        return Err(sortformer_activation_error(
            "fixture_identity",
            "activation fixtures do not match the predeclared non-human corpus",
        ));
    }
    verify_activation_package_identity(&receipt.package)?;
    let contracts = expected_activation_tensor_contracts(&fixtures);
    verify_activation_records(&receipt.records, &contracts, checkpoint)?;
    verify_activation_floor(&receipt.oracle_floor, &receipt.records, &contracts)
}

fn verify_activation_package_identity(
    package: &SortformerActivationPackageIdentity,
) -> FwResult<()> {
    require_sha256(
        "activation_package_identity",
        &package.sha256,
        SortformerArtifactDomain::Activation,
    )?;
    if package.format != PACKAGE_FORMAT
        || package.dtype_set != [SortformerTensorDtype::F32, SortformerTensorDtype::I64]
        || package.byte_order != PACKAGE_BYTE_ORDER
        || package.tensor_order != PACKAGE_TENSOR_ORDER
        || package.logical_layout != SOURCE_LAYOUT
        || package.metadata_policy != PACKAGE_METADATA_POLICY
        || package.tensor_count != SORTFORMER_ACTIVATION_TENSORS
        || package.f32_elements != SORTFORMER_ACTIVATION_F32_ELEMENTS
        || package.i64_elements != SORTFORMER_ACTIVATION_I64_ELEMENTS
        || package.payload_bytes != SORTFORMER_ACTIVATION_PAYLOAD_BYTES
        || package.bytes != SORTFORMER_ACTIVATION_PACKAGE_BYTES
        || package.sha256 != SORTFORMER_ACTIVATION_PACKAGE_SHA256
    {
        return Err(sortformer_activation_error(
            "package_identity",
            "activation package identity does not match the frozen synthetic truth pack",
        ));
    }
    Ok(())
}

fn verify_activation_records(
    records: &[SortformerActivationTensorRecord],
    contracts: &BTreeMap<String, ActivationTensorContract>,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    if records.len() != contracts.len()
        || u64::try_from(records.len()).ok() != Some(SORTFORMER_ACTIVATION_TENSORS)
    {
        return Err(sortformer_activation_error(
            "record_census",
            "activation record count does not match the frozen stage census",
        ));
    }
    let mut previous_name: Option<&str> = None;
    let mut f32_elements = 0u64;
    let mut i64_elements = 0u64;
    let mut payload_bytes = 0u64;
    for record in records {
        sortformer_checkpoint(checkpoint, SortformerArtifactDomain::Activation)?;
        validate_tensor_name(&record.name, SortformerArtifactDomain::Activation)?;
        if previous_name.is_some_and(|previous| previous >= record.name.as_str()) {
            return Err(sortformer_activation_error(
                "record_order",
                "activation records are not strictly lexicographic and unique",
            ));
        }
        previous_name = Some(&record.name);
        let contract = contracts.get(&record.name).ok_or_else(|| {
            sortformer_activation_error(
                "record_census",
                "activation record is outside the frozen stage census",
            )
        })?;
        if record.dtype != contract.dtype
            || record.shape != contract.shape
            || record.logical_layout != SOURCE_LAYOUT
        {
            return Err(sortformer_activation_error(
                "record_contract",
                "activation record dtype, shape, or layout changed",
            ));
        }
        require_sha256(
            "activation_record_identity",
            &record.value_sha256,
            SortformerArtifactDomain::Activation,
        )?;
        if contract
            .value_sha256
            .as_ref()
            .is_some_and(|expected| expected != &record.value_sha256)
        {
            return Err(sortformer_activation_error(
                "record_identity",
                "activation record checksum changed from its independent fixture root",
            ));
        }
        let elements = checked_elements(&record.shape, SortformerArtifactDomain::Activation)?;
        let width = match record.dtype {
            SortformerTensorDtype::F32 => F32_BYTES,
            SortformerTensorDtype::I64 => I64_BYTES,
        };
        let bytes = elements.checked_mul(width).ok_or_else(|| {
            sortformer_activation_error("record_census", "activation record byte count overflows")
        })?;
        if record.elements != elements || record.bytes != bytes {
            return Err(sortformer_activation_error(
                "record_census",
                "activation record shape does not account for its element and byte counts",
            ));
        }
        match record.dtype {
            SortformerTensorDtype::F32 => {
                f32_elements = f32_elements.checked_add(elements).ok_or_else(|| {
                    sortformer_activation_error(
                        "record_census",
                        "activation F32 element census overflows",
                    )
                })?;
            }
            SortformerTensorDtype::I64 => {
                i64_elements = i64_elements.checked_add(elements).ok_or_else(|| {
                    sortformer_activation_error(
                        "record_census",
                        "activation I64 element census overflows",
                    )
                })?;
            }
        }
        payload_bytes = payload_bytes.checked_add(bytes).ok_or_else(|| {
            sortformer_activation_error("record_census", "activation payload census overflows")
        })?;
    }
    if f32_elements != SORTFORMER_ACTIVATION_F32_ELEMENTS
        || i64_elements != SORTFORMER_ACTIVATION_I64_ELEMENTS
        || payload_bytes != SORTFORMER_ACTIVATION_PAYLOAD_BYTES
    {
        return Err(sortformer_activation_error(
            "record_census",
            "activation record totals do not match the frozen package census",
        ));
    }
    Ok(())
}

fn verify_activation_floor(
    floor: &SortformerActivationOracleFloor,
    records: &[SortformerActivationTensorRecord],
    contracts: &BTreeMap<String, ActivationTensorContract>,
) -> FwResult<()> {
    if floor.schema_version != SORTFORMER_ACTIVATION_FLOOR_SCHEMA
        || floor.baseline_threads != 1
        || floor.baseline_repetition != 0
        || floor.thread_counts != [1, 8]
        || floor.repetitions_per_thread != 5
        || !floor.all_byte_exact
        || floor.mismatch_count != 0
        || floor.comparison_rule != "exact_ieee_bits"
        || floor.absolute_tolerance_f32_bits != "0x00000000"
        || floor.relative_tolerance_f32_bits != "0x00000000"
        || floor.margin_basis != "deterministic_synthetic_preprocessing_zero_floor_no_margin"
    {
        return Err(sortformer_activation_error(
            "oracle_floor",
            "activation oracle floor is not the predeclared byte-exact thread-regime result",
        ));
    }

    let record_by_name = records
        .iter()
        .map(|record| (record.name.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    let expected_pairs = contracts
        .keys()
        .filter_map(|name| {
            name.strip_prefix("fixture.")
                .and_then(|suffix| suffix.split_once('.'))
                .map(|(fixture, stage)| (fixture.to_owned(), stage.to_owned()))
        })
        .collect::<BTreeSet<_>>();
    if floor.observations.len() != expected_pairs.len() || expected_pairs.len() != 44 {
        return Err(sortformer_activation_error(
            "oracle_floor_census",
            "activation oracle floor does not cover every fixture stage exactly once",
        ));
    }

    let mut observed_pairs = BTreeSet::new();
    let mut previous_pair: Option<(&str, &str)> = None;
    for observation in &floor.observations {
        let pair = (observation.fixture.as_str(), observation.stage.as_str());
        if previous_pair.is_some_and(|previous| previous >= pair) {
            return Err(sortformer_activation_error(
                "oracle_floor_order",
                "activation floor observations are not strictly lexicographic and unique",
            ));
        }
        previous_pair = Some(pair);
        if !observed_pairs.insert((observation.fixture.clone(), observation.stage.clone())) {
            return Err(sortformer_activation_error(
                "oracle_floor_census",
                "activation floor repeats a fixture stage",
            ));
        }
        let record_name = format!("fixture.{}.{}", observation.fixture, observation.stage);
        let record = record_by_name.get(record_name.as_str()).ok_or_else(|| {
            sortformer_activation_error(
                "oracle_floor_census",
                "activation floor names a stage absent from the tensor records",
            )
        })?;
        let compared_values = record.elements.checked_mul(45).ok_or_else(|| {
            sortformer_activation_error(
                "oracle_floor_census",
                "activation floor comparison count overflows",
            )
        })?;
        if observation.run_count != 10
            || observation.pair_count != 45
            || observation.compared_values != compared_values
            || observation.mismatch_count != 0
            || !observation.byte_exact
            || observation.max_abs_diff_f32_bits != "0x00000000"
            || observation.mean_abs_diff_f64_bits != "0x0000000000000000"
            || observation.relative_l2_f64_bits != "0x0000000000000000"
        {
            return Err(sortformer_activation_error(
                "oracle_floor_metric",
                "activation floor observation is not the exact ten-run zero floor",
            ));
        }
    }
    if observed_pairs != expected_pairs {
        return Err(sortformer_activation_error(
            "oracle_floor_census",
            "activation oracle floor fixture-stage coverage changed",
        ));
    }
    Ok(())
}

fn verify_activation_package(
    package: &SafetensorsFile,
    receipt: &SortformerActivationReceipt,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    if package.metadata().is_some() {
        return Err(sortformer_activation_error(
            "package_metadata",
            "activation package must not carry unaudited safetensors metadata",
        ));
    }
    if package.len() != receipt.records.len()
        || u64::try_from(package.len()).ok() != Some(SORTFORMER_ACTIVATION_TENSORS)
        || !package
            .names()
            .eq(receipt.records.iter().map(|record| record.name.as_str()))
    {
        return Err(sortformer_activation_error(
            "package_census",
            "activation package names do not exactly match the receipt records",
        ));
    }

    let fixtures = receipt
        .fixtures
        .iter()
        .map(|fixture| (fixture.name.as_str(), fixture))
        .collect::<BTreeMap<_, _>>();
    let mut f32_elements = 0u64;
    let mut i64_elements = 0u64;
    let mut payload_bytes = 0u64;
    for record in &receipt.records {
        sortformer_checkpoint(checkpoint, SortformerArtifactDomain::Activation)?;
        let expected_dtype = match record.dtype {
            SortformerTensorDtype::F32 => "F32",
            SortformerTensorDtype::I64 => "I64",
        };
        if package.dtype_name(&record.name).map_err(|_| {
            sortformer_activation_error("package_census", "activation tensor dtype is unavailable")
        })? != expected_dtype
        {
            return Err(sortformer_activation_error(
                "package_dtype",
                "activation tensor dtype does not match its receipt record",
            ));
        }
        let shape = package.shape(&record.name).map_err(|_| {
            sortformer_activation_error("package_census", "activation tensor shape is unavailable")
        })?;
        if shape.len() != record.shape.len()
            || !shape
                .iter()
                .zip(&record.shape)
                .all(|(left, right)| u64::try_from(*left).ok() == Some(*right))
        {
            return Err(sortformer_activation_error(
                "package_shape",
                "activation tensor shape does not match its receipt record",
            ));
        }
        let raw = package.tensor_raw_bytes(&record.name).map_err(|_| {
            sortformer_activation_error(
                "package_payload",
                "activation tensor payload could not be borrowed safely",
            )
        })?;
        if u64::try_from(raw.len()).ok() != Some(record.bytes) {
            return Err(sortformer_activation_error(
                "package_payload",
                "activation tensor byte length does not match its receipt record",
            ));
        }
        let observed_sha256 = match record.dtype {
            SortformerTensorDtype::F32 => {
                f32_elements = f32_elements.checked_add(record.elements).ok_or_else(|| {
                    sortformer_activation_error(
                        "package_census",
                        "activation F32 package census overflows",
                    )
                })?;
                finite_f32_sha256(raw, SortformerArtifactDomain::Activation, checkpoint)?
            }
            SortformerTensorDtype::I64 => {
                i64_elements = i64_elements.checked_add(record.elements).ok_or_else(|| {
                    sortformer_activation_error(
                        "package_census",
                        "activation I64 package census overflows",
                    )
                })?;
                verify_activation_i64_value(&record.name, raw, &fixtures)?;
                sha256_bytes(raw)
            }
        };
        if observed_sha256 != record.value_sha256 {
            return Err(sortformer_activation_error(
                "package_payload",
                "activation tensor bytes do not match the authenticated receipt",
            ));
        }
        payload_bytes = payload_bytes.checked_add(record.bytes).ok_or_else(|| {
            sortformer_activation_error("package_census", "activation package bytes overflow")
        })?;
    }
    if f32_elements != receipt.package.f32_elements
        || i64_elements != receipt.package.i64_elements
        || payload_bytes != receipt.package.payload_bytes
    {
        return Err(sortformer_activation_error(
            "package_census",
            "verified activation payload does not match the receipt totals",
        ));
    }
    Ok(())
}

fn verify_activation_i64_value(
    name: &str,
    raw: &[u8],
    fixtures: &BTreeMap<&str, &SortformerActivationFixture>,
) -> FwResult<()> {
    let (fixture_name, stage) = name
        .strip_prefix("fixture.")
        .and_then(|suffix| suffix.split_once('.'))
        .ok_or_else(|| {
            sortformer_activation_error(
                "package_i64",
                "activation I64 tensor name is outside the fixture-stage grammar",
            )
        })?;
    let fixture = fixtures.get(fixture_name).ok_or_else(|| {
        sortformer_activation_error(
            "package_i64",
            "activation I64 tensor names an unknown fixture",
        )
    })?;
    let expected = match stage {
        "input_length_i64" => fixture.sample_count,
        "valid_length_i64" => fixture.valid_frames,
        _ => {
            return Err(sortformer_activation_error(
                "package_i64",
                "activation package contains an unrecognized I64 stage",
            ));
        }
    };
    let bytes: [u8; 8] = raw.try_into().map_err(|_| {
        sortformer_activation_error(
            "package_i64",
            "activation I64 length tensor must contain exactly one value",
        )
    })?;
    let expected = i64::try_from(expected).map_err(|_| {
        sortformer_activation_error(
            "package_i64",
            "activation I64 fixture length does not fit its storage type",
        )
    })?;
    if i64::from_le_bytes(bytes) != expected {
        return Err(sortformer_activation_error(
            "package_i64",
            "activation I64 length value does not match its fixture contract",
        ));
    }
    Ok(())
}

fn load_verified_sortformer_package_against(
    receipt_path: &Path,
    package_path: &Path,
    expected_receipt_sha256: &str,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    expected: &ReceiptExpectations,
) -> FwResult<VerifiedSortformerPackage> {
    require_sha256(
        "receipt_trust_root",
        expected_receipt_sha256,
        SortformerArtifactDomain::Conversion,
    )?;
    sortformer_checkpoint(checkpoint, SortformerArtifactDomain::Conversion)?;
    let receipt_bytes = read_bounded_file(
        receipt_path,
        "receipt",
        MAX_RECEIPT_BYTES,
        None,
        SortformerArtifactDomain::Conversion,
        checkpoint,
    )?;
    if sha256_bytes(&receipt_bytes) != expected_receipt_sha256 {
        return Err(sortformer_error(
            "receipt_identity",
            "conversion receipt checksum does not match the independent trust root",
        ));
    }
    sortformer_checkpoint(checkpoint, SortformerArtifactDomain::Conversion)?;
    let receipt: SortformerConversionReceipt =
        serde_json::from_slice(&receipt_bytes).map_err(|_| {
            sortformer_error(
                "receipt_schema",
                "conversion receipt is not valid strict receipt JSON",
            )
        })?;
    let canonical = canonical_json_bytes(&receipt).map_err(|_| {
        sortformer_error(
            "receipt_schema",
            "conversion receipt could not be serialized canonically",
        )
    })?;
    if canonical != receipt_bytes {
        return Err(sortformer_error(
            "receipt_canonical",
            "conversion receipt bytes are not the canonical JSON encoding",
        ));
    }
    verify_receipt(&receipt, expected, checkpoint)?;

    let package_bytes = read_bounded_file(
        package_path,
        "package",
        MAX_PACKAGE_BYTES,
        Some((receipt.package.bytes, receipt.package.sha256.as_str())),
        SortformerArtifactDomain::Conversion,
        checkpoint,
    )?;
    verify_compact_safetensors_layout(
        &package_bytes,
        receipt.package.payload_bytes,
        SortformerArtifactDomain::Conversion,
        checkpoint,
    )?;
    sortformer_checkpoint(checkpoint, SortformerArtifactDomain::Conversion)?;
    let package = SafetensorsFile::from_owned_bytes(package_bytes).map_err(|_| {
        sortformer_error(
            "package_structure",
            "weight package is not structurally valid safetensors",
        )
    })?;
    verify_package(&package, &receipt, expected, checkpoint)?;
    sortformer_checkpoint(checkpoint, SortformerArtifactDomain::Conversion)?;
    Ok(VerifiedSortformerPackage { receipt, package })
}

fn verify_compact_safetensors_layout(
    bytes: &[u8],
    expected_payload_bytes: u64,
    domain: SortformerArtifactDomain,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    sortformer_checkpoint(checkpoint, domain)?;
    let length_prefix: [u8; 8] = bytes
        .get(..8)
        .and_then(|prefix| prefix.try_into().ok())
        .ok_or_else(|| {
            domain.error(
                "package_structure",
                "safetensors package lacks a complete header-length prefix",
            )
        })?;
    let header_bytes_u64 = u64::from_le_bytes(length_prefix);
    if header_bytes_u64 == 0 || header_bytes_u64 > MAX_PACKAGE_HEADER_BYTES {
        return Err(domain.error(
            "package_structure",
            "safetensors header size is outside the Sortformer envelope",
        ));
    }
    let header_bytes = usize::try_from(header_bytes_u64).map_err(|_| {
        domain.error(
            "package_structure",
            "safetensors header size does not fit this platform",
        )
    })?;
    let data_start = 8usize.checked_add(header_bytes).ok_or_else(|| {
        domain.error(
            "package_structure",
            "safetensors header offset overflows this platform",
        )
    })?;
    let data_len = bytes.len().checked_sub(data_start).ok_or_else(|| {
        domain.error(
            "package_structure",
            "safetensors header extends past the package bytes",
        )
    })?;
    if u64::try_from(data_len).ok() != Some(expected_payload_bytes) {
        return Err(domain.error(
            "package_layout",
            "safetensors data section is not exactly the frozen payload size",
        ));
    }
    let header: serde_json::Value =
        serde_json::from_slice(bytes.get(8..data_start).ok_or_else(|| {
            domain.error(
                "package_structure",
                "safetensors header span is unavailable",
            )
        })?)
        .map_err(|_| domain.error("package_structure", "safetensors header is not valid JSON"))?;
    sortformer_checkpoint(checkpoint, domain)?;
    let entries = header.as_object().ok_or_else(|| {
        domain.error(
            "package_structure",
            "safetensors header is not a JSON object",
        )
    })?;
    let mut spans = Vec::with_capacity(entries.len());
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue;
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| {
                domain.error(
                    "package_structure",
                    "safetensors tensor has invalid data offsets",
                )
            })?;
        let begin = offsets
            .first()
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                domain.error(
                    "package_structure",
                    "safetensors tensor begin offset is invalid",
                )
            })?;
        let end = offsets
            .get(1)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                domain.error(
                    "package_structure",
                    "safetensors tensor end offset is invalid",
                )
            })?;
        spans.push((name.as_str(), begin, end));
    }
    spans.sort_unstable_by_key(|(name, _, _)| *name);
    let mut cursor = 0u64;
    for (_, begin, end) in spans {
        sortformer_checkpoint(checkpoint, domain)?;
        if begin != cursor || end < begin || end > expected_payload_bytes {
            return Err(domain.error(
                "package_layout",
                "safetensors tensor spans are not a compact non-overlapping payload",
            ));
        }
        cursor = end;
    }
    if cursor != expected_payload_bytes {
        return Err(domain.error(
            "package_layout",
            "safetensors tensor spans do not cover the complete payload",
        ));
    }
    Ok(())
}

fn verify_receipt(
    receipt: &SortformerConversionReceipt,
    expected: &ReceiptExpectations,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    if receipt.schema_version != SORTFORMER_RECEIPT_SCHEMA {
        return Err(sortformer_error(
            "receipt_schema",
            "conversion receipt schema is not the frozen v1 schema",
        ));
    }
    if receipt.model != expected.model {
        return Err(sortformer_error(
            "model_identity",
            "conversion receipt model identity does not match the frozen source",
        ));
    }
    if receipt.execution != expected.execution {
        return Err(sortformer_error(
            "execution_config",
            "conversion receipt execution configuration does not match the frozen inference graph",
        ));
    }
    if receipt.source_files != expected.source_files {
        return Err(sortformer_error(
            "source_identity",
            "conversion receipt source-file identities do not match the frozen source",
        ));
    }
    if receipt.converter.converter_id != SORTFORMER_CONVERTER_ID
        || receipt.converter.converter_version != SORTFORMER_CONVERTER_VERSION
    {
        return Err(sortformer_error(
            "converter_identity",
            "conversion receipt converter identity is not the frozen v1 converter",
        ));
    }
    require_sha256(
        "converter_identity",
        &receipt.converter.source_sha256,
        SortformerArtifactDomain::Conversion,
    )?;
    if receipt.converter.source_sha256 != expected.converter_source_sha256 {
        return Err(sortformer_error(
            "converter_identity",
            "converter source checksum does not match the independently frozen converter",
        ));
    }
    if receipt.runtime != expected.runtime {
        return Err(sortformer_error(
            "runtime_identity",
            "conversion receipt runtime identity does not match the frozen runtime",
        ));
    }
    if receipt.license != expected.license {
        return Err(sortformer_error(
            "license_policy",
            "conversion receipt license policy does not match the conservative distribution boundary",
        ));
    }
    verify_package_identity(&receipt.package, expected)?;
    verify_records(&receipt.records, expected, checkpoint)
}

fn verify_package_identity(
    package: &SortformerPackageIdentity,
    expected: &ReceiptExpectations,
) -> FwResult<()> {
    require_sha256(
        "package_identity",
        &package.sha256,
        SortformerArtifactDomain::Conversion,
    )?;
    if package.sha256 != expected.package_sha256
        || package.bytes != expected.package_bytes
        || package.format != PACKAGE_FORMAT
        || package.payload_bytes != expected.package_payload_bytes
        || package.f32_elements != expected.package_f32_elements
        || package.tensor_count != expected.exported_tensors
        || package.dtype != PACKAGE_DTYPE
        || package.byte_order != PACKAGE_BYTE_ORDER
        || package.tensor_order != PACKAGE_TENSOR_ORDER
        || package.logical_layout != SOURCE_LAYOUT
        || package.metadata_policy != PACKAGE_METADATA_POLICY
    {
        return Err(sortformer_error(
            "package_identity",
            "converted package contract does not match the frozen F32 safetensors profile",
        ));
    }
    let derived_payload = package
        .f32_elements
        .checked_mul(F32_BYTES)
        .ok_or_else(|| sortformer_error("package_identity", "package payload size overflows"))?;
    if derived_payload != package.payload_bytes {
        return Err(sortformer_error(
            "package_identity",
            "converted package payload byte census is inconsistent",
        ));
    }
    if package.bytes <= package.payload_bytes || package.bytes > MAX_PACKAGE_BYTES {
        return Err(sortformer_error(
            "package_identity",
            "converted package file size is outside the bounded safetensors envelope",
        ));
    }
    Ok(())
}

#[derive(Default)]
struct ObservedCensus {
    state_tensors: u64,
    state_elements: u64,
    state_f32_tensors: u64,
    state_f32_elements: u64,
    state_f32_bytes: u64,
    state_i64_tensors: u64,
    state_payload_bytes: u64,
    exported_tensors: u64,
    exported_elements: u64,
    exported_bytes: u64,
    dropped_tensors: u64,
}

fn verify_records(
    records: &[SortformerTensorRecord],
    expected: &ReceiptExpectations,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    if u64::try_from(records.len()).ok() != Some(expected.source_records) {
        return Err(sortformer_error(
            "record_census",
            "conversion receipt source-record count does not match the frozen census",
        ));
    }

    let mut previous_name: Option<&str> = None;
    let mut destination_names = BTreeSet::new();
    let mut observed_counter_names = BTreeSet::new();
    let mut saw_position = false;
    let mut saw_sentinel = false;
    let mut census = ObservedCensus::default();

    for record in records {
        sortformer_checkpoint(checkpoint, SortformerArtifactDomain::Conversion)?;
        validate_tensor_name(&record.source_name, SortformerArtifactDomain::Conversion)?;
        if let Some(previous) = previous_name
            && previous >= record.source_name.as_str()
        {
            let code = if previous == record.source_name {
                "record_duplicate"
            } else {
                "record_order"
            };
            return Err(sortformer_error(
                code,
                "conversion records must have unique source names in strict lexicographic order",
            ));
        }
        previous_name = Some(&record.source_name);
        require_sha256(
            "record_hash",
            &record.source_value_sha256,
            SortformerArtifactDomain::Conversion,
        )?;
        if record.source_logical_layout != SOURCE_LAYOUT {
            return Err(sortformer_error(
                "record_layout",
                "source tensor layout is not the frozen contiguous row-major layout",
            ));
        }
        let source_elements =
            checked_elements(&record.source_shape, SortformerArtifactDomain::Conversion)?;
        let element_bytes = match record.source_dtype {
            SortformerTensorDtype::F32 => F32_BYTES,
            SortformerTensorDtype::I64 => I64_BYTES,
        };
        let source_bytes = source_elements.checked_mul(element_bytes).ok_or_else(|| {
            sortformer_error("record_shape", "source tensor byte count overflows")
        })?;
        if source_elements != record.source_elements || source_bytes != record.source_bytes {
            return Err(sortformer_error(
                "record_shape",
                "source tensor shape, element count, and byte count are inconsistent",
            ));
        }

        match record.source_origin {
            SortformerTensorOrigin::StateDict => {
                census.state_tensors = checked_add(census.state_tensors, 1, "record_census")?;
                census.state_elements = checked_add(
                    census.state_elements,
                    record.source_elements,
                    "record_census",
                )?;
                census.state_payload_bytes = checked_add(
                    census.state_payload_bytes,
                    record.source_bytes,
                    "record_census",
                )?;
                verify_state_record(
                    record,
                    expected,
                    &mut observed_counter_names,
                    &mut destination_names,
                    &mut census,
                )?;
            }
            SortformerTensorOrigin::NonpersistentBuffer => {
                verify_buffer_record(
                    record,
                    expected,
                    &mut saw_position,
                    &mut saw_sentinel,
                    &mut destination_names,
                    &mut census,
                )?;
            }
        }
    }

    if !saw_position || !saw_sentinel {
        return Err(sortformer_error(
            "buffer_census",
            "positional buffer and empty dtype sentinel must each appear exactly once",
        ));
    }
    if observed_counter_names != expected.counter_names {
        return Err(sortformer_error(
            "counter_census",
            "training-only batch counter names do not match the frozen census",
        ));
    }
    if census.state_tensors != expected.state_tensors
        || census.state_elements != expected.state_elements
        || census.state_f32_tensors != expected.state_f32_tensors
        || census.state_f32_elements != expected.state_f32_elements
        || census.state_f32_bytes != expected.state_f32_bytes
        || census.state_i64_tensors != expected.state_i64_tensors
        || census.state_payload_bytes != expected.state_payload_bytes
        || census.exported_tensors != expected.exported_tensors
        || census.exported_elements != expected.package_f32_elements
        || census.exported_bytes != expected.package_payload_bytes
        || census.dropped_tensors != expected.dropped_tensors
    {
        return Err(sortformer_error(
            "record_census",
            "conversion records do not match the frozen source and destination census",
        ));
    }
    if u64::try_from(destination_names.len()).ok() != Some(expected.exported_tensors) {
        return Err(sortformer_error(
            "destination_census",
            "exported destination names are not a unique complete census",
        ));
    }
    let observed_manifest_sha256 = tensor_manifest_sha256(&expected.model, records)?;
    if observed_manifest_sha256 != expected.model.tensor_manifest_sha256 {
        return Err(sortformer_error(
            "tensor_manifest",
            "conversion records do not match the frozen exact tensor topology",
        ));
    }
    Ok(())
}

fn verify_state_record(
    record: &SortformerTensorRecord,
    expected: &ReceiptExpectations,
    observed_counter_names: &mut BTreeSet<String>,
    destination_names: &mut BTreeSet<String>,
    census: &mut ObservedCensus,
) -> FwResult<()> {
    match record.source_dtype {
        SortformerTensorDtype::F32 => {
            census.state_f32_tensors = checked_add(census.state_f32_tensors, 1, "record_census")?;
            census.state_f32_elements = checked_add(
                census.state_f32_elements,
                record.source_elements,
                "record_census",
            )?;
            census.state_f32_bytes =
                checked_add(census.state_f32_bytes, record.source_bytes, "record_census")?;
            verify_exported_record(record, destination_names, census)
        }
        SortformerTensorDtype::I64 => {
            census.state_i64_tensors = checked_add(census.state_i64_tensors, 1, "record_census")?;
            if !expected.counter_names.contains(&record.source_name)
                || !record.source_shape.is_empty()
                || record.source_elements != 1
                || record.source_bytes != I64_BYTES
                || !matches!(
                    &record.disposition,
                    SortformerTensorDisposition::DroppedTrainOnly
                )
            {
                return Err(sortformer_error(
                    "counter_census",
                    "only the exact scalar I64 batch counters may be dropped as training-only",
                ));
            }
            if !observed_counter_names.insert(record.source_name.clone()) {
                return Err(sortformer_error(
                    "record_duplicate",
                    "training-only batch counter appears more than once",
                ));
            }
            census.dropped_tensors = checked_add(census.dropped_tensors, 1, "record_census")?;
            Ok(())
        }
    }
}

fn verify_buffer_record(
    record: &SortformerTensorRecord,
    expected: &ReceiptExpectations,
    saw_position: &mut bool,
    saw_sentinel: &mut bool,
    destination_names: &mut BTreeSet<String>,
    census: &mut ObservedCensus,
) -> FwResult<()> {
    match record.source_name.as_str() {
        SORTFORMER_POSITION_TENSOR => {
            if *saw_position
                || record.source_dtype != SortformerTensorDtype::F32
                || record.source_shape != expected.position_shape
            {
                return Err(sortformer_error(
                    "buffer_census",
                    "positional buffer identity, dtype, or shape is invalid",
                ));
            }
            *saw_position = true;
            verify_exported_record(record, destination_names, census)
        }
        SORTFORMER_DTYPE_SENTINEL => {
            if *saw_sentinel
                || record.source_dtype != SortformerTensorDtype::F32
                || record.source_shape != [0]
                || record.source_elements != 0
                || record.source_bytes != 0
                || record.source_value_sha256 != EMPTY_SHA256
                || !matches!(
                    &record.disposition,
                    SortformerTensorDisposition::DroppedRuntimeSentinel
                )
            {
                return Err(sortformer_error(
                    "buffer_census",
                    "empty dtype sentinel does not match the frozen dropped-buffer contract",
                ));
            }
            *saw_sentinel = true;
            census.dropped_tensors = checked_add(census.dropped_tensors, 1, "record_census")?;
            Ok(())
        }
        _ => Err(sortformer_error(
            "buffer_census",
            "conversion receipt contains an unexpected non-persistent buffer",
        )),
    }
}

fn verify_exported_record(
    record: &SortformerTensorRecord,
    destination_names: &mut BTreeSet<String>,
    census: &mut ObservedCensus,
) -> FwResult<()> {
    let SortformerTensorDisposition::Exported {
        transform,
        destination,
    } = &record.disposition
    else {
        return Err(sortformer_error(
            "record_disposition",
            "every F32 state tensor and the positional buffer must be exported",
        ));
    };
    if *transform != SortformerTensorTransform::IdentityContiguousF32
        || destination.name != record.source_name
        || destination.dtype != SortformerTensorDtype::F32
        || destination.shape != record.source_shape
        || destination.logical_layout != SOURCE_LAYOUT
        || destination.value_sha256 != record.source_value_sha256
        || destination.elements != record.source_elements
        || destination.bytes != record.source_bytes
    {
        return Err(sortformer_error(
            "record_transform",
            "v1 permits only byte-preserving identity-contiguous F32 exports",
        ));
    }
    require_sha256(
        "record_hash",
        &destination.value_sha256,
        SortformerArtifactDomain::Conversion,
    )?;
    if !destination_names.insert(destination.name.clone()) {
        return Err(sortformer_error(
            "destination_duplicate",
            "two source records map to the same destination tensor",
        ));
    }
    census.exported_tensors = checked_add(census.exported_tensors, 1, "record_census")?;
    census.exported_elements = checked_add(
        census.exported_elements,
        destination.elements,
        "record_census",
    )?;
    census.exported_bytes = checked_add(census.exported_bytes, destination.bytes, "record_census")?;
    Ok(())
}

fn verify_package(
    package: &SafetensorsFile,
    receipt: &SortformerConversionReceipt,
    expected: &ReceiptExpectations,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    if package.metadata().is_some() {
        return Err(sortformer_error(
            "package_metadata",
            "v1 converted package must not carry unaudited safetensors metadata",
        ));
    }
    if u64::try_from(package.len()).ok() != Some(expected.exported_tensors) {
        return Err(sortformer_error(
            "package_census",
            "safetensors tensor count does not match the frozen destination census",
        ));
    }

    let exported = receipt
        .records
        .iter()
        .filter_map(|record| match &record.disposition {
            SortformerTensorDisposition::Exported { destination, .. } => {
                Some((destination.name.as_str(), destination))
            }
            SortformerTensorDisposition::DroppedTrainOnly
            | SortformerTensorDisposition::DroppedRuntimeSentinel => None,
        })
        .collect::<BTreeMap<_, _>>();
    if exported.len() != package.len() {
        return Err(sortformer_error(
            "package_census",
            "receipt destination names are not a one-to-one package census",
        ));
    }
    if !package.names().eq(exported.keys().copied()) {
        return Err(sortformer_error(
            "package_census",
            "safetensors names do not exactly match receipt destinations",
        ));
    }

    let mut observed_elements = 0u64;
    let mut observed_bytes = 0u64;
    for (name, destination) in exported {
        sortformer_checkpoint(checkpoint, SortformerArtifactDomain::Conversion)?;
        if package.dtype_name(name).map_err(|_| {
            sortformer_error(
                "package_census",
                "receipt destination is absent from safetensors",
            )
        })? != "F32"
        {
            return Err(sortformer_error(
                "package_dtype",
                "every converted package tensor must be exact F32",
            ));
        }
        let package_shape = package.shape(name).map_err(|_| {
            sortformer_error(
                "package_census",
                "receipt destination shape is unavailable from safetensors",
            )
        })?;
        if package_shape.len() != destination.shape.len()
            || !package_shape
                .iter()
                .zip(&destination.shape)
                .all(|(left, right)| u64::try_from(*left).ok() == Some(*right))
        {
            return Err(sortformer_error(
                "package_shape",
                "safetensors shape does not match the authenticated receipt",
            ));
        }
        let raw = package.tensor_raw_bytes(name).map_err(|_| {
            sortformer_error(
                "package_payload",
                "safetensors tensor payload could not be borrowed safely",
            )
        })?;
        if u64::try_from(raw.len()).ok() != Some(destination.bytes) {
            return Err(sortformer_error(
                "package_payload",
                "safetensors tensor byte length does not match the authenticated receipt",
            ));
        }
        let observed_sha256 =
            finite_f32_sha256(raw, SortformerArtifactDomain::Conversion, checkpoint)?;
        if observed_sha256 != destination.value_sha256 {
            return Err(sortformer_error(
                "package_payload",
                "safetensors tensor bytes do not match the authenticated receipt",
            ));
        }
        observed_elements = checked_add(observed_elements, destination.elements, "package_census")?;
        observed_bytes = checked_add(observed_bytes, destination.bytes, "package_census")?;
    }
    if observed_elements != receipt.package.f32_elements
        || observed_bytes != receipt.package.payload_bytes
    {
        return Err(sortformer_error(
            "package_census",
            "verified safetensors payload does not match the receipt totals",
        ));
    }
    Ok(())
}

fn finite_f32_sha256(
    raw: &[u8],
    domain: SortformerArtifactDomain,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<String> {
    if !raw.len().is_multiple_of(4) {
        return Err(domain.error(
            "package_payload",
            "F32 tensor payload is not four-byte aligned",
        ));
    }
    let mut hasher = Sha256::new();
    for block in raw.chunks(READ_CHUNK_BYTES) {
        sortformer_checkpoint(checkpoint, domain)?;
        hasher.update(block);
        let (chunks, remainder) = block.as_chunks::<4>();
        if !remainder.is_empty() {
            return Err(domain.error("package_payload", "F32 tensor chunk has invalid width"));
        }
        for &bytes in chunks {
            if !f32::from_le_bytes(bytes).is_finite() {
                return Err(domain.error(
                    "package_nonfinite",
                    "authenticated package contains a non-finite F32 value",
                ));
            }
        }
    }
    Ok(hex_digest(hasher.finalize()))
}

fn open_regular_artifact(
    path: &Path,
    kind: &str,
    domain: SortformerArtifactDomain,
) -> FwResult<(File, u64)> {
    let before = std::fs::symlink_metadata(path).map_err(|_| {
        domain.error(
            &format!("{kind}_open"),
            &format!("{kind} artifact could not be opened"),
        )
    })?;
    if !before.file_type().is_file() {
        return Err(domain.error(
            &format!("{kind}_type"),
            &format!("{kind} artifact must be a regular file"),
        ));
    }
    open_prechecked_regular_artifact(path, kind, &before, domain)
}

fn open_prechecked_regular_artifact(
    path: &Path,
    kind: &str,
    before: &std::fs::Metadata,
    domain: SortformerArtifactDomain,
) -> FwResult<(File, u64)> {
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    let file = {
        use rustix::fs::{Mode, OFlags, open};

        let descriptor = open(
            path,
            OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
            Mode::empty(),
        )
        .map_err(|_| {
            domain.error(
                &format!("{kind}_open"),
                &format!("{kind} artifact could not be opened"),
            )
        })?;
        File::from(descriptor)
    };
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    let file = File::open(path).map_err(|_| {
        domain.error(
            &format!("{kind}_open"),
            &format!("{kind} artifact could not be opened"),
        )
    })?;

    let after = file.metadata().map_err(|_| {
        domain.error(
            &format!("{kind}_metadata"),
            &format!("{kind} artifact metadata is unavailable"),
        )
    })?;
    if !after.file_type().is_file() {
        return Err(domain.error(
            &format!("{kind}_type"),
            &format!("{kind} artifact must be a regular file"),
        ));
    }
    #[cfg(target_family = "unix")]
    {
        use std::os::unix::fs::MetadataExt;

        if before.dev() != after.dev() || before.ino() != after.ino() {
            return Err(domain.error(
                &format!("{kind}_identity"),
                &format!("{kind} artifact identity changed while it was opened"),
            ));
        }
    }
    Ok((file, after.len()))
}

fn read_bounded_file(
    path: &Path,
    kind: &str,
    maximum_bytes: u64,
    expected: Option<(u64, &str)>,
    domain: SortformerArtifactDomain,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<Vec<u8>> {
    sortformer_checkpoint(checkpoint, domain)?;
    let (file, metadata_bytes) = open_regular_artifact(path, kind, domain)?;
    if metadata_bytes == 0 || metadata_bytes > maximum_bytes {
        return Err(domain.error(
            &format!("{kind}_size"),
            &format!("{kind} artifact size is outside its bounded envelope"),
        ));
    }
    if let Some((expected_bytes, expected_sha256)) = expected {
        require_sha256(&format!("{kind}_identity"), expected_sha256, domain)?;
        if metadata_bytes != expected_bytes {
            return Err(domain.error(
                &format!("{kind}_identity"),
                &format!("{kind} artifact length does not match the authenticated receipt"),
            ));
        }
    }
    let capacity = usize::try_from(metadata_bytes).map_err(|_| {
        domain.error(
            &format!("{kind}_size"),
            &format!("{kind} artifact size does not fit this platform"),
        )
    })?;
    let mut bytes = reserved_byte_buffer(capacity, kind, domain, checkpoint)?;
    let mut hasher = Sha256::new();
    let mut reader = file;
    let mut buffer = [0u8; READ_CHUNK_BYTES];
    loop {
        sortformer_checkpoint(checkpoint, domain)?;
        let read = reader.read(&mut buffer).map_err(|_| {
            domain.error(
                &format!("{kind}_read"),
                &format!("{kind} artifact could not be read"),
            )
        })?;
        if read == 0 {
            break;
        }
        if read > capacity.saturating_sub(bytes.len()) {
            return Err(domain.error(
                &format!("{kind}_identity"),
                &format!("{kind} artifact length changed while it was read"),
            ));
        }
        let chunk = buffer.get(..read).ok_or_else(|| {
            domain.error(
                &format!("{kind}_read"),
                &format!("{kind} artifact read returned an invalid span"),
            )
        })?;
        hasher.update(chunk);
        bytes.extend_from_slice(chunk);
    }
    if bytes.len() != capacity {
        return Err(domain.error(
            &format!("{kind}_identity"),
            &format!("{kind} artifact length changed while it was read"),
        ));
    }
    let observed_sha256 = hex_digest(hasher.finalize());
    if let Some((_, expected_sha256)) = expected
        && observed_sha256 != expected_sha256
    {
        return Err(domain.error(
            &format!("{kind}_identity"),
            &format!("{kind} artifact checksum does not match the authenticated receipt"),
        ));
    }
    sortformer_checkpoint(checkpoint, domain)?;
    Ok(bytes)
}

fn reserved_byte_buffer(
    capacity: usize,
    kind: &str,
    domain: SortformerArtifactDomain,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<Vec<u8>> {
    sortformer_checkpoint(checkpoint, domain)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|_| {
        domain.error(
            &format!("{kind}_allocation"),
            &format!("{kind} artifact buffer could not be allocated within its bounded envelope"),
        )
    })?;
    sortformer_checkpoint(checkpoint, domain)?;
    Ok(bytes)
}

fn checked_elements(shape: &[u64], domain: SortformerArtifactDomain) -> FwResult<u64> {
    if shape.len() > 8 {
        return Err(domain.error(
            "record_shape",
            "tensor rank exceeds the bounded receipt schema",
        ));
    }
    shape.iter().try_fold(1u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| domain.error("record_shape", "tensor element count overflows"))
    })
}

fn checked_add(left: u64, right: u64, code: &str) -> FwResult<u64> {
    left.checked_add(right)
        .ok_or_else(|| sortformer_error(code, "tensor census sum overflows"))
}

fn tensor_manifest_sha256(
    model: &SortformerModelIdentity,
    records: &[SortformerTensorRecord],
) -> FwResult<String> {
    let projected_records = records
        .iter()
        .map(tensor_manifest_record)
        .collect::<Vec<_>>();
    let manifest = serde_json::json!({
        "schema_version": SORTFORMER_TENSOR_MANIFEST_SCHEMA,
        "model_id": model.model_id.as_str(),
        "model_revision": model.model_revision.as_str(),
        "nemo_sha256": model.nemo_sha256.as_str(),
        "config_sha256": model.config_sha256.as_str(),
        "checkpoint_sha256": model.checkpoint_sha256.as_str(),
        "records": projected_records,
    });
    canonical_json_bytes(&manifest)
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|_| {
            sortformer_error(
                "tensor_manifest",
                "tensor topology could not be encoded canonically",
            )
        })
}

fn tensor_manifest_record(record: &SortformerTensorRecord) -> serde_json::Value {
    let disposition = match &record.disposition {
        SortformerTensorDisposition::Exported {
            transform,
            destination,
        } => serde_json::json!({
            "kind": "exported",
            "transform": tensor_transform_name(*transform),
            "destination": {
                "name": destination.name.as_str(),
                "dtype": tensor_dtype_name(destination.dtype),
                "shape": &destination.shape,
                "logical_layout": destination.logical_layout.as_str(),
                "elements": destination.elements,
                "bytes": destination.bytes,
            },
        }),
        SortformerTensorDisposition::DroppedTrainOnly => {
            serde_json::json!({"kind": "dropped_train_only"})
        }
        SortformerTensorDisposition::DroppedRuntimeSentinel => {
            serde_json::json!({"kind": "dropped_runtime_sentinel"})
        }
    };
    serde_json::json!({
        "source_name": record.source_name.as_str(),
        "source_origin": tensor_origin_name(record.source_origin),
        "source_dtype": tensor_dtype_name(record.source_dtype),
        "source_shape": &record.source_shape,
        "source_logical_layout": record.source_logical_layout.as_str(),
        "source_elements": record.source_elements,
        "source_bytes": record.source_bytes,
        "disposition": disposition,
    })
}

const fn tensor_origin_name(origin: SortformerTensorOrigin) -> &'static str {
    match origin {
        SortformerTensorOrigin::StateDict => "state_dict",
        SortformerTensorOrigin::NonpersistentBuffer => "nonpersistent_buffer",
    }
}

const fn tensor_dtype_name(dtype: SortformerTensorDtype) -> &'static str {
    match dtype {
        SortformerTensorDtype::F32 => "f32",
        SortformerTensorDtype::I64 => "i64",
    }
}

const fn tensor_transform_name(transform: SortformerTensorTransform) -> &'static str {
    match transform {
        SortformerTensorTransform::IdentityContiguousF32 => "identity_contiguous_f32",
    }
}

fn validate_tensor_name(name: &str, domain: SortformerArtifactDomain) -> FwResult<()> {
    if name.is_empty()
        || name.len() > 512
        || name.starts_with('.')
        || name.ends_with('.')
        || name.contains("..")
        || name
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_')))
    {
        return Err(domain.error(
            "record_name",
            "tensor name is outside the bounded dotted-identifier grammar",
        ));
    }
    Ok(())
}

fn require_sha256(code: &str, value: &str, domain: SortformerArtifactDomain) -> FwResult<()> {
    if value.len() != HASH_HEX_LEN
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(domain.error(
            code,
            "SHA-256 identity must be exactly 64 lowercase hexadecimal characters",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortformerArtifactDomain {
    Conversion,
    Activation,
}

impl SortformerArtifactDomain {
    fn error(self, code: &str, message: &str) -> FwError {
        match self {
            Self::Conversion => sortformer_error(code, message),
            Self::Activation => sortformer_activation_error(code, message),
        }
    }

    const fn prefix(self) -> &'static str {
        match self {
            Self::Conversion => "sortformer_conversion",
            Self::Activation => "sortformer_activation",
        }
    }
}

fn sortformer_checkpoint(
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    domain: SortformerArtifactDomain,
) -> FwResult<()> {
    match checkpoint() {
        Ok(()) => Ok(()),
        Err(FwError::Cancelled(_)) => Err(FwError::Cancelled(format!(
            "{}.load_cancelled: cooperative checkpoint requested cancellation",
            domain.prefix()
        ))),
        Err(_) => Err(domain.error(
            "checkpoint_failure",
            "caller checkpoint returned a non-cancellation failure",
        )),
    }
}

fn sortformer_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("sortformer_conversion.{code}: {message}"))
}

fn sortformer_activation_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("sortformer_activation.{code}: {message}"))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_digest(hasher.finalize())
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> FwResult<Vec<u8>> {
    let value = serde_json::to_value(value)?;
    let mut output = Vec::new();
    write_lexicographic_json(&value, &mut output)?;
    Ok(output)
}

fn write_lexicographic_json(value: &serde_json::Value, output: &mut Vec<u8>) -> FwResult<()> {
    match value {
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_) => serde_json::to_writer(output, value)?,
        serde_json::Value::Array(values) => {
            output.push(b'[');
            for (index, item) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_lexicographic_json(item, output)?;
            }
            output.push(b']');
        }
        serde_json::Value::Object(values) => {
            output.push(b'{');
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| key.as_str());
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                serde_json::to_writer(&mut *output, key)?;
                output.push(b':');
                write_lexicographic_json(item, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::*;

    const TINY_TENSOR_MANIFEST_SHA256: &str =
        "f408e382179658671178ba767d2c27737f551c426543fea14e2cbddac53abfc8";

    struct TinyBundle {
        receipt: SortformerConversionReceipt,
        package_bytes: Vec<u8>,
        expected: ReceiptExpectations,
    }

    type CheckpointLoader =
        fn(&Path, &Path, &(dyn Fn() -> FwResult<()> + Sync)) -> FwResult<VerifiedSortformerPackage>;
    type ActivationCheckpointLoader = fn(
        &Path,
        &Path,
        &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<VerifiedSortformerActivationPack>;

    #[test]
    fn bounded_file_buffer_allocation_fails_closed() {
        let error = reserved_byte_buffer(
            usize::MAX,
            "package",
            SortformerArtifactDomain::Conversion,
            &|| Ok(()),
        )
        .expect_err("an impossible package capacity must not abort");
        assert_error(&error, "package_allocation");
    }

    #[test]
    fn bounded_file_buffer_checks_cancellation_after_reservation() {
        let checkpoints = AtomicUsize::new(0);
        let error =
            reserved_byte_buffer(16, "package", SortformerArtifactDomain::Conversion, &|| {
                if checkpoints.fetch_add(1, Ordering::SeqCst) == 0 {
                    Ok(())
                } else {
                    Err(FwError::Cancelled("sensitive test reason".to_owned()))
                }
            })
            .expect_err("post-reservation cancellation must be observed");
        assert_eq!(checkpoints.load(Ordering::SeqCst), 2);
        assert!(matches!(error, FwError::Cancelled(_)));
        assert!(!error.to_string().contains("sensitive test reason"));
    }

    #[test]
    fn bounded_file_rejects_directory_artifact() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let error = read_bounded_file(
            directory.path(),
            "receipt",
            MAX_RECEIPT_BYTES,
            None,
            SortformerArtifactDomain::Conversion,
            &|| Ok(()),
        )
        .expect_err("a directory must not be admitted as an artifact");
        assert_error(&error, "sortformer_conversion.receipt_type");
    }

    #[cfg(target_family = "unix")]
    #[test]
    fn bounded_file_rejects_symlink_artifact() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("temporary directory");
        let target = directory.path().join("receipt.json");
        let link = directory.path().join("receipt-link.json");
        std::fs::write(&target, b"{}").expect("write symlink target");
        symlink(&target, &link).expect("create artifact symlink");

        let error = read_bounded_file(
            &link,
            "receipt",
            MAX_RECEIPT_BYTES,
            None,
            SortformerArtifactDomain::Conversion,
            &|| Ok(()),
        )
        .expect_err("a symlink must not be admitted as an artifact");
        assert_error(&error, "sortformer_conversion.receipt_type");
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    #[test]
    fn prechecked_open_rejects_fifo_swap_without_blocking() -> Result<(), String> {
        let directory = tempfile::tempdir().expect("temporary directory");
        let sentinel = directory.path().join("receipt-before.json");
        let fifo = directory.path().join("receipt.fifo");
        std::fs::write(&sentinel, b"{}").expect("write regular sentinel");
        let before = std::fs::symlink_metadata(&sentinel).expect("sentinel metadata");
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            use rustix::fs::{Mode, mkfifoat};

            let directory_handle = File::open(directory.path()).expect("directory handle");
            mkfifoat(&directory_handle, "receipt.fifo", Mode::RUSR | Mode::WUSR)
                .expect("create FIFO fixture");
        }
        #[cfg(target_os = "macos")]
        {
            let status = std::process::Command::new("/usr/bin/mkfifo")
                .args(["-m", "600"])
                .arg(&fifo)
                .status()
                .expect("run system mkfifo for fixture");
            assert!(status.success(), "system mkfifo failed: {status}");
        }

        let (sender, receiver) = std::sync::mpsc::channel();
        let fifo_for_reader = fifo.clone();
        let reader = std::thread::spawn(move || {
            let outcome = open_prechecked_regular_artifact(
                &fifo_for_reader,
                "receipt",
                &before,
                SortformerArtifactDomain::Conversion,
            )
            .map(|_| ())
            .map_err(|error| error.to_string());
            let _ = sender.send(outcome);
        });
        let outcome = match receiver.recv_timeout(std::time::Duration::from_secs(2)) {
            Ok(outcome) => outcome,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                use rustix::fs::{Mode, OFlags, open};

                let rescue = open(
                    &fifo,
                    OFlags::RDWR | OFlags::CLOEXEC | OFlags::NONBLOCK,
                    Mode::empty(),
                )
                .expect("rescue a reader blocked by a regressed open flag");
                let _ = receiver.recv_timeout(std::time::Duration::from_secs(1));
                reader.join().expect("join rescued FIFO reader");
                drop(rescue);
                return Err(
                    "prechecked FIFO open blocked instead of failing immediately".to_owned(),
                );
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                reader.join().expect("join failed FIFO reader");
                return Err("prechecked FIFO reader exited without reporting an outcome".to_owned());
            }
        };
        reader.join().expect("join FIFO reader");
        let error = outcome
            .err()
            .ok_or_else(|| "a FIFO path swap was accepted".to_owned())?;
        assert!(
            error.contains("sortformer_conversion.receipt_type"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    #[test]
    fn prechecked_open_rejects_symlink_swap() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("temporary directory");
        let original = directory.path().join("receipt.json");
        let held = directory.path().join("receipt-held.json");
        std::fs::write(&original, b"{}").expect("write regular artifact");
        let before = std::fs::symlink_metadata(&original).expect("artifact metadata");
        std::fs::rename(&original, &held).expect("hold original artifact inode");
        symlink(&held, &original).expect("replace artifact path with symlink");

        let error = open_prechecked_regular_artifact(
            &original,
            "receipt",
            &before,
            SortformerArtifactDomain::Conversion,
        )
        .expect_err("a symlink path swap must not follow the held original inode");
        assert_error(&error, "sortformer_conversion.receipt_open");
    }

    #[test]
    fn tiny_identity_bound_bundle_is_accepted() {
        let bundle = tiny_bundle();
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let verified = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect("tiny authenticated bundle");

        assert_eq!(verified.receipt(), &bundle.receipt);
        assert_eq!(verified.safetensors().len(), 2);
        let rendered = format!("{verified:?}");
        assert!(rendered.contains("<authenticated model bytes redacted>"));
        assert!(!rendered.contains("SafetensorsFile"));
        assert!(!rendered.contains("package_bytes"));
    }

    #[test]
    fn missing_package_tensor_is_rejected() {
        let mut bundle = tiny_bundle();
        let merged_values = [0.25f32, 0.5, 0.75, 1.0, 1.5, -2.0];
        bundle.package_bytes = make_safetensors(&[(
            SORTFORMER_POSITION_TENSOR,
            vec![1, 3, 2],
            merged_values.as_slice(),
        )]);
        bind_package_file_identity(&mut bundle.receipt, &bundle.package_bytes);
        trust_test_package_identity(&mut bundle);
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("a package missing one receipt tensor must fail");
        assert_error(&error, "sortformer_conversion.package_census");
    }

    #[test]
    fn extra_package_tensor_is_rejected() {
        let mut bundle = tiny_bundle();
        let position_values = [0.25f32, 0.5, 0.75, 1.0];
        let weight_values = [1.5f32];
        let extra_values = [-2.0f32];
        bundle.package_bytes = make_safetensors(&[
            (
                SORTFORMER_POSITION_TENSOR,
                vec![1, 2, 2],
                position_values.as_slice(),
            ),
            ("encoder.weight", vec![1], weight_values.as_slice()),
            ("encoder.zzz", vec![1], extra_values.as_slice()),
        ]);
        bind_package_file_identity(&mut bundle.receipt, &bundle.package_bytes);
        trust_test_package_identity(&mut bundle);
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("a package with an extra tensor must fail");
        assert_error(&error, "sortformer_conversion.package_census");
    }

    #[test]
    fn independent_receipt_trust_root_rejects_tamper() {
        let mut bundle = tiny_bundle();
        let trusted_receipt = canonical_json_bytes(&bundle.receipt).expect("canonical receipt");
        let trusted_sha256 = sha256_bytes(&trusted_receipt);
        bundle.receipt.model.model_revision = "attacker-selected-revision".to_owned();
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, _) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &trusted_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("receipt mutation must not inherit trust");
        assert_error(&error, "sortformer_conversion.receipt_identity");
    }

    #[test]
    fn wrong_frozen_model_pin_is_rejected_even_with_a_matching_receipt_digest() {
        let mut bundle = tiny_bundle();
        bundle.receipt.model.model_revision = "wrong-but-well-formed-revision".to_owned();
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("wrong pin must fail semantic validation");
        assert_error(&error, "sortformer_conversion.model_identity");
    }

    #[test]
    fn wrong_oracle_adapter_pin_is_rejected() {
        let mut bundle = tiny_bundle();
        bundle.receipt.model.oracle_adapter_sha256 = sha256_bytes(b"different adapter");
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("wrong oracle adapter identity must fail");
        assert_error(&error, "sortformer_conversion.model_identity");
    }

    #[test]
    fn wrong_safetensors_runtime_pin_is_rejected() {
        let mut bundle = tiny_bundle();
        bundle.receipt.runtime.safetensors = "0.7.0".to_owned();
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("wrong safetensors runtime identity must fail");
        assert_error(&error, "sortformer_conversion.runtime_identity");
    }

    #[test]
    fn wrong_execution_config_is_rejected_even_with_a_matching_receipt_digest() {
        let mut bundle = tiny_bundle();
        bundle.receipt.execution.transformer_mask_future = true;
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("wrong execution configuration must fail semantic validation");
        assert_error(&error, "sortformer_conversion.execution_config");
    }

    #[test]
    fn noncanonical_receipt_bytes_are_rejected() {
        let bundle = tiny_bundle();
        let directory = tempfile::tempdir().expect("temporary directory");
        let receipt_path = directory.path().join("receipt.json");
        let package_path = directory.path().join("weights.safetensors");
        let noncanonical = serde_json::to_vec_pretty(&bundle.receipt).expect("pretty receipt");
        std::fs::write(&receipt_path, &noncanonical).expect("write receipt");
        std::fs::write(&package_path, &bundle.package_bytes).expect("write package");

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &sha256_bytes(&noncanonical),
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("noncanonical bytes must fail");
        assert_error(&error, "sortformer_conversion.receipt_canonical");
    }

    #[test]
    fn duplicate_source_record_is_rejected() {
        let mut bundle = tiny_bundle();
        bundle.receipt.records[1] = bundle.receipt.records[0].clone();
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("duplicate record must fail");
        assert_error(&error, "sortformer_conversion.record_duplicate");
    }

    #[test]
    fn nonlexicographic_record_order_is_rejected() {
        let mut bundle = tiny_bundle();
        bundle.receipt.records.swap(0, 1);
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("record order must fail");
        assert_error(&error, "sortformer_conversion.record_order");
    }

    #[test]
    fn renamed_f32_tensor_cannot_hide_behind_aggregate_census() -> Result<(), String> {
        let mut bundle = tiny_bundle();
        let record = bundle
            .receipt
            .records
            .iter_mut()
            .find(|record| record.source_name == "encoder.weight")
            .expect("tiny weight record");
        record.source_name = "encoder.xeight".to_owned();
        let destination = match &mut record.disposition {
            SortformerTensorDisposition::Exported { destination, .. } => destination,
            _ => return Err("tiny weight must be exported".to_owned()),
        };
        destination.name = "encoder.xeight".to_owned();
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("renamed tensor must not inherit the topology trust root");
        assert_error(&error, "sortformer_conversion.tensor_manifest");
        Ok(())
    }

    #[test]
    fn reshaped_f32_tensor_cannot_hide_behind_aggregate_census() -> Result<(), String> {
        let mut bundle = tiny_bundle();
        let record = bundle
            .receipt
            .records
            .iter_mut()
            .find(|record| record.source_name == "encoder.weight")
            .expect("tiny weight record");
        record.source_shape = vec![1, 2];
        let destination = match &mut record.disposition {
            SortformerTensorDisposition::Exported { destination, .. } => destination,
            _ => return Err("tiny weight must be exported".to_owned()),
        };
        destination.shape = vec![1, 2];
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("reshaped tensor must not inherit the topology trust root");
        assert_error(&error, "sortformer_conversion.tensor_manifest");
        Ok(())
    }

    #[test]
    fn consecutive_dot_tensor_name_is_rejected() {
        let mut bundle = tiny_bundle();
        let record = bundle
            .receipt
            .records
            .iter_mut()
            .find(|record| record.source_name == "encoder.weight")
            .expect("tiny weight record");
        record.source_name = "encoder..weight".to_owned();
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("empty tensor-name component must fail");
        assert_error(&error, "sortformer_conversion.record_name");
    }

    #[test]
    fn reviewed_converter_source_digest_is_mandatory() {
        let mut bundle = tiny_bundle();
        bundle.receipt.converter.source_sha256 = sha256_bytes(b"unreviewed converter source");
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("unreviewed converter source must fail");
        assert_error(&error, "sortformer_conversion.converter_identity");
    }

    #[test]
    fn production_loader_exposes_no_caller_supplied_trust_root() {
        let _: fn(&Path, &Path) -> FwResult<VerifiedSortformerPackage> =
            load_verified_sortformer_package;
        let _: CheckpointLoader = load_verified_sortformer_package_with_checkpoint;
        let _: fn(&Path, &Path) -> FwResult<VerifiedSortformerActivationPack> =
            load_verified_sortformer_activation_pack;
        let _: ActivationCheckpointLoader =
            load_verified_sortformer_activation_pack_with_checkpoint;
    }

    #[test]
    #[ignore = "requires operator-local licensed Sortformer package and receipt"]
    fn operator_local_real_sortformer_package_is_admitted() {
        let receipt = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_RECEIPT")
            .expect("set FRANKEN_WHISPER_SORTFORMER_RECEIPT");
        let package = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_PACKAGE")
            .expect("set FRANKEN_WHISPER_SORTFORMER_PACKAGE");
        let verified = load_verified_sortformer_package(Path::new(&receipt), Path::new(&package))
            .expect("operator-local frozen package must pass exact admission");
        assert_eq!(verified.receipt().records.len(), 992);
        assert_eq!(verified.safetensors().len(), 974);
        assert_eq!(
            verified.receipt().model.tensor_manifest_sha256,
            SORTFORMER_TENSOR_MANIFEST_SHA256
        );
    }

    #[test]
    #[ignore = "requires operator-local synthetic activation package and receipt"]
    fn operator_local_sortformer_activation_pack_is_admitted() {
        let receipt = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_ACTIVATION_RECEIPT")
            .expect("set FRANKEN_WHISPER_SORTFORMER_ACTIVATION_RECEIPT");
        let package = std::env::var_os("FRANKEN_WHISPER_SORTFORMER_ACTIVATION_PACKAGE")
            .expect("set FRANKEN_WHISPER_SORTFORMER_ACTIVATION_PACKAGE");
        let verified =
            load_verified_sortformer_activation_pack(Path::new(&receipt), Path::new(&package))
                .expect("operator-local synthetic activation pack must pass exact admission");
        assert_eq!(verified.receipt().records.len(), 46);
        assert_eq!(verified.safetensors().len(), 46);
        assert!(verified.receipt().oracle_floor.all_byte_exact);
        assert_eq!(verified.receipt().oracle_floor.observations.len(), 44);
        let rendered = format!("{verified:?}");
        assert!(rendered.contains("activations redacted"));
        assert!(!rendered.contains("decoded_pcm_f32"));
    }

    #[test]
    fn package_payload_swap_fails_against_unchanged_trust_root() {
        let mut bundle = tiny_bundle();
        let data_start = safetensors_data_start(&bundle.package_bytes);
        for offset in 0..4 {
            bundle
                .package_bytes
                .swap(data_start + offset, data_start + 16 + offset);
        }
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);
        let trusted_receipt_sha256 = sha256_bytes(
            &canonical_json_bytes(&bundle.receipt).expect("canonical trusted receipt"),
        );
        assert_eq!(receipt_sha256, trusted_receipt_sha256);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &trusted_receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("swapped package payload must fail");
        assert_error(&error, "sortformer_conversion.package_identity");
    }

    #[test]
    fn per_tensor_hash_rejects_package_tamper_after_whole_hash_is_updated() {
        let mut bundle = tiny_bundle();
        let data_start = safetensors_data_start(&bundle.package_bytes);
        bundle.package_bytes[data_start] ^= 1;
        bind_package_file_identity(&mut bundle.receipt, &bundle.package_bytes);
        trust_test_package_identity(&mut bundle);
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("per-tensor receipt hash must still detect tamper");
        assert_error(&error, "sortformer_conversion.package_payload");
    }

    #[test]
    fn duplicate_safetensors_header_key_is_rejected() {
        let mut bundle = tiny_bundle();
        bundle.package_bytes = duplicate_header_package();
        bind_package_file_identity(&mut bundle.receipt, &bundle.package_bytes);
        trust_test_package_identity(&mut bundle);
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("duplicate package tensor key must fail");
        assert_error(&error, "sortformer_conversion.package_structure");
    }

    #[test]
    fn ambiguous_safetensors_tensor_entries_are_rejected() {
        for violation in [r#""dtype":"F32","#, r#""strides":[1],"#] {
            let mut bundle = tiny_bundle();
            bundle.package_bytes = ambiguous_tensor_entry_package(violation);
            bind_package_file_identity(&mut bundle.receipt, &bundle.package_bytes);
            trust_test_package_identity(&mut bundle);
            let directory = tempfile::tempdir().expect("temporary directory");
            let (receipt_path, package_path, receipt_sha256) =
                write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

            let error = load_verified_sortformer_package_against(
                &receipt_path,
                &package_path,
                &receipt_sha256,
                &|| Ok(()),
                &bundle.expected,
            )
            .expect_err("ambiguous tensor entry must fail at the strict package boundary");
            assert_error(&error, "sortformer_conversion.package_structure");
        }
    }

    #[test]
    fn nonlexicographic_payload_offsets_are_rejected() {
        let mut bundle = tiny_bundle();
        bundle.package_bytes = nonlexicographic_payload_package();
        bind_package_file_identity(&mut bundle.receipt, &bundle.package_bytes);
        trust_test_package_identity(&mut bundle);
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("payload order must follow lexicographic tensor names");
        assert_error(&error, "sortformer_conversion.package_layout");
    }

    #[test]
    fn nonfinite_payload_is_rejected_even_when_all_hashes_match() {
        let mut bundle = tiny_bundle();
        let data_start = safetensors_data_start(&bundle.package_bytes);
        bundle.package_bytes[data_start..data_start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let position_payload = &bundle.package_bytes[data_start..data_start + 16];
        let position_sha256 = sha256_bytes(position_payload);
        let position_record = bundle
            .receipt
            .records
            .iter_mut()
            .find(|record| record.source_name == SORTFORMER_POSITION_TENSOR)
            .expect("position record");
        position_record
            .source_value_sha256
            .clone_from(&position_sha256);
        let destination = match &mut position_record.disposition {
            SortformerTensorDisposition::Exported { destination, .. } => Some(destination),
            _ => None,
        }
        .expect("tiny position fixture must be exported");
        destination.value_sha256 = position_sha256;
        bind_package_file_identity(&mut bundle.receipt, &bundle.package_bytes);
        trust_test_package_identity(&mut bundle);
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &|| Ok(()),
            &bundle.expected,
        )
        .expect_err("non-finite package value must fail");
        assert_error(&error, "sortformer_conversion.package_nonfinite");
    }

    #[test]
    fn cooperative_cancellation_is_preserved_and_sanitized() {
        let bundle = tiny_bundle();
        let directory = tempfile::tempdir().expect("temporary directory");
        let (receipt_path, package_path, receipt_sha256) =
            write_bundle(directory.path(), &bundle.receipt, &bundle.package_bytes);
        let calls = AtomicUsize::new(0);
        let checkpoint = || {
            if calls.fetch_add(1, Ordering::SeqCst) >= 2 {
                Err(FwError::Cancelled(
                    "caller detail must not escape".to_owned(),
                ))
            } else {
                Ok(())
            }
        };

        let error = load_verified_sortformer_package_against(
            &receipt_path,
            &package_path,
            &receipt_sha256,
            &checkpoint,
            &bundle.expected,
        )
        .expect_err("checkpoint must cancel load");
        match error {
            FwError::Cancelled(message) => {
                assert_eq!(
                    message,
                    "sortformer_conversion.load_cancelled: cooperative checkpoint requested cancellation"
                );
            }
            other => assert_eq!(other.error_code(), "FW-CANCELLED"),
        }
    }

    #[test]
    fn production_census_constants_are_internally_consistent() {
        let expected = pinned_model_expectations();
        assert_eq!(
            expected.converter_source_sha256,
            SORTFORMER_CONVERTER_SOURCE_SHA256
        );
        assert_eq!(expected.package_sha256, SORTFORMER_PACKAGE_SHA256);
        assert_eq!(expected.package_bytes, SORTFORMER_PACKAGE_BYTES);
        assert_eq!(
            expected.model.tensor_manifest_sha256,
            SORTFORMER_TENSOR_MANIFEST_SHA256
        );
        assert_eq!(expected.source_files.len(), 17);
        assert!(
            expected
                .source_files
                .windows(2)
                .all(|pair| pair[0].path < pair[1].path)
        );
        assert!(expected.source_files.iter().any(|source| {
            source.path == "nemo/collections/asr/data/audio_to_diar_label.py"
                && source.sha256
                    == "f9b0d23bd52da417ac18418ea1c83aa1119f59e6b37d3b2b3159c8cb2f036234"
        }));
        assert!(expected.source_files.iter().any(|source| {
            source.path == "nemo/collections/asr/parts/preprocessing/segment.py"
                && source.sha256
                    == "a598d91b94110e0c12a1ba4a57894ce89109e597fa8e909cf7b5b6e7bb9369af"
        }));
        assert!(expected.source_files.iter().any(|source| {
            source.path == "nemo/collections/asr/parts/utils/speaker_utils.py"
                && source.sha256
                    == "6c247bdda26fd010190e1c96f8399f77a5265a180086e134d9b167b3c8019dc0"
        }));
        assert_eq!(expected.counter_names.len(), 17);
        assert_eq!(
            checked_elements(
                &expected.position_shape,
                SortformerArtifactDomain::Conversion,
            )
            .unwrap(),
            5_119_488
        );
        assert_eq!(
            expected.state_f32_elements.checked_add(5_119_488).unwrap(),
            expected.package_f32_elements
        );
        assert_eq!(
            expected
                .package_f32_elements
                .checked_mul(F32_BYTES)
                .unwrap(),
            expected.package_payload_bytes
        );
        assert_eq!(
            expected.state_f32_bytes + expected.state_i64_tensors * I64_BYTES,
            expected.state_payload_bytes
        );
    }

    #[test]
    fn checked_in_converter_matches_the_compiled_trust_root() {
        assert_eq!(
            sha256_bytes(include_bytes!("../scripts/convert_to_safetensors.py")),
            SORTFORMER_CONVERTER_SOURCE_SHA256
        );
    }

    #[test]
    fn checked_in_activation_exporter_matches_the_compiled_trust_root() {
        assert_eq!(
            sha256_bytes(include_bytes!(
                "../scripts/export_sortformer_activations.py"
            )),
            SORTFORMER_ACTIVATION_EXPORTER_SHA256
        );
    }

    #[test]
    fn activation_receipt_semantics_cover_every_frozen_frontend_stage() {
        let receipt = synthetic_activation_receipt();
        verify_activation_receipt(&receipt, &|| Ok(())).expect("frozen activation semantics");
        assert_eq!(receipt.records.len(), 46);
        assert_eq!(receipt.oracle_floor.observations.len(), 44);
    }

    #[test]
    fn activation_receipt_rejects_authority_stage_and_floor_drift() {
        let mut receipt = synthetic_activation_receipt();
        receipt.authority = "production".to_owned();
        let error = verify_activation_receipt(&receipt, &|| Ok(()))
            .expect_err("diagnostic authority must not be promoted");
        assert_error(&error, "sortformer_activation.authority");

        let mut receipt = synthetic_activation_receipt();
        assert!(receipt.records.pop().is_some());
        let error =
            verify_activation_receipt(&receipt, &|| Ok(())).expect_err("missing stage must fail");
        assert_error(&error, "sortformer_activation.record_census");

        let mut receipt = synthetic_activation_receipt();
        receipt.oracle_floor.observations[0].mismatch_count = 1;
        let error = verify_activation_receipt(&receipt, &|| Ok(()))
            .expect_err("nonzero source floor must fail");
        assert_error(&error, "sortformer_activation.oracle_floor_metric");

        let mut receipt = synthetic_activation_receipt();
        receipt.records[0].shape = vec![401];
        let error = verify_activation_receipt(&receipt, &|| Ok(()))
            .expect_err("wrong activation shape must fail");
        assert_error(&error, "sortformer_activation.record_contract");
    }

    #[test]
    fn activation_i64_lengths_require_exact_little_endian_values() {
        let receipt = synthetic_activation_receipt();
        let fixtures = receipt
            .fixtures
            .iter()
            .map(|fixture| (fixture.name.as_str(), fixture))
            .collect::<BTreeMap<_, _>>();
        let fixture = receipt.fixtures.first().expect("frozen fixture");
        let input_name = format!("fixture.{}.input_length_i64", fixture.name);
        let valid_name = format!("fixture.{}.valid_length_i64", fixture.name);
        let input_length = i64::try_from(fixture.sample_count).expect("sample count fits I64");
        let valid_length = i64::try_from(fixture.valid_frames).expect("frame count fits I64");

        verify_activation_i64_value(&input_name, &input_length.to_le_bytes(), &fixtures)
            .expect("exact little-endian input length");
        verify_activation_i64_value(&valid_name, &valid_length.to_le_bytes(), &fixtures)
            .expect("exact little-endian valid length");

        let error =
            verify_activation_i64_value(&input_name, &input_length.to_be_bytes(), &fixtures)
                .expect_err("big-endian control data must fail");
        assert_error(&error, "sortformer_activation.package_i64");

        let error = verify_activation_i64_value(&input_name, &[0u8; 7], &fixtures)
            .expect_err("truncated I64 control data must fail");
        assert_error(&error, "sortformer_activation.package_i64");
    }

    #[test]
    fn activation_shared_failures_keep_the_activation_error_domain() {
        let mut receipt = synthetic_activation_receipt();
        receipt.records[0].name = "invalid..stage".to_owned();
        let error = verify_activation_receipt(&receipt, &|| Ok(()))
            .expect_err("invalid activation tensor names must fail");
        assert_error(&error, "sortformer_activation.record_name");

        let error = finite_f32_sha256(
            &f32::NAN.to_le_bytes(),
            SortformerArtifactDomain::Activation,
            &|| Ok(()),
        )
        .expect_err("non-finite activation data must fail");
        assert_error(&error, "sortformer_activation.package_nonfinite");

        let error = sortformer_checkpoint(
            &|| Err(FwError::Cancelled("sensitive reason".to_owned())),
            SortformerArtifactDomain::Activation,
        )
        .expect_err("activation cancellation must be normalized");
        assert_error(&error, "sortformer_activation.load_cancelled");
        assert!(!error.to_string().contains("sensitive reason"));

        let error = load_verified_sortformer_activation_pack_with_checkpoint(
            Path::new("unused-activation-receipt"),
            Path::new("unused-activation-package"),
            &|| Err(FwError::Cancelled("private caller reason".to_owned())),
        )
        .expect_err("public activation admission must preserve its error domain");
        assert_error(&error, "sortformer_activation.load_cancelled");
        assert!(!error.to_string().contains("private caller reason"));
    }

    #[test]
    fn activation_loader_rejects_untrusted_receipt_before_package_open() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let receipt_path = directory.path().join("activation-receipt.json");
        let absent_package = directory.path().join("absent.safetensors");
        std::fs::write(&receipt_path, b"{}").expect("write untrusted receipt");
        let error = load_verified_sortformer_activation_pack(&receipt_path, &absent_package)
            .expect_err("caller bytes must not select their own activation trust root");
        assert_error(&error, "sortformer_activation.receipt_identity");
    }

    fn synthetic_activation_receipt() -> SortformerActivationReceipt {
        let fixtures = frozen_activation_fixtures();
        let contracts = expected_activation_tensor_contracts(&fixtures);
        let records = contracts
            .iter()
            .map(|(name, contract)| {
                let elements =
                    checked_elements(&contract.shape, SortformerArtifactDomain::Activation)
                        .expect("activation elements");
                let width = match contract.dtype {
                    SortformerTensorDtype::F32 => F32_BYTES,
                    SortformerTensorDtype::I64 => I64_BYTES,
                };
                SortformerActivationTensorRecord {
                    name: name.clone(),
                    dtype: contract.dtype,
                    shape: contract.shape.clone(),
                    logical_layout: SOURCE_LAYOUT.to_owned(),
                    elements,
                    bytes: elements.checked_mul(width).expect("activation bytes"),
                    value_sha256: contract
                        .value_sha256
                        .clone()
                        .unwrap_or_else(|| EMPTY_SHA256.to_owned()),
                }
            })
            .collect::<Vec<_>>();
        let observations = records
            .iter()
            .filter_map(|record| {
                record
                    .name
                    .strip_prefix("fixture.")
                    .and_then(|suffix| suffix.split_once('.'))
                    .map(|(fixture, stage)| SortformerActivationFloorObservation {
                        fixture: fixture.to_owned(),
                        stage: stage.to_owned(),
                        run_count: 10,
                        pair_count: 45,
                        compared_values: record.elements.checked_mul(45).expect("comparisons"),
                        mismatch_count: 0,
                        byte_exact: true,
                        max_abs_diff_f32_bits: "0x00000000".to_owned(),
                        mean_abs_diff_f64_bits: "0x0000000000000000".to_owned(),
                        relative_l2_f64_bits: "0x0000000000000000".to_owned(),
                    })
            })
            .collect();
        SortformerActivationReceipt {
            schema_version: SORTFORMER_ACTIVATION_RECEIPT_SCHEMA.to_owned(),
            canonical_json_version: "lexicographic-json-v1".to_owned(),
            authority: "diagnostic_only".to_owned(),
            equivalence_level: "partial_l1_synthetic_frontend".to_owned(),
            fixture_set: "sortformer-synthetic-frontend-v1".to_owned(),
            model: frozen_activation_model_identity(),
            exporter: SortformerActivationExporterIdentity {
                exporter_id: "franken-whisper-sortformer-activation-exporter".to_owned(),
                exporter_version: "1".to_owned(),
                source_sha256: SORTFORMER_ACTIVATION_EXPORTER_SHA256.to_owned(),
                conversion_helper_sha256: SORTFORMER_CONVERTER_SOURCE_SHA256.to_owned(),
            },
            runtime: frozen_runtime_identity(),
            source_files: frozen_source_files(),
            execution: frozen_activation_execution_identity(),
            fixtures,
            oracle_floor: SortformerActivationOracleFloor {
                schema_version: SORTFORMER_ACTIVATION_FLOOR_SCHEMA.to_owned(),
                baseline_threads: 1,
                baseline_repetition: 0,
                thread_counts: vec![1, 8],
                repetitions_per_thread: 5,
                all_byte_exact: true,
                mismatch_count: 0,
                comparison_rule: "exact_ieee_bits".to_owned(),
                absolute_tolerance_f32_bits: "0x00000000".to_owned(),
                relative_tolerance_f32_bits: "0x00000000".to_owned(),
                margin_basis: "deterministic_synthetic_preprocessing_zero_floor_no_margin"
                    .to_owned(),
                observations,
            },
            package: SortformerActivationPackageIdentity {
                format: PACKAGE_FORMAT.to_owned(),
                dtype_set: vec![SortformerTensorDtype::F32, SortformerTensorDtype::I64],
                byte_order: PACKAGE_BYTE_ORDER.to_owned(),
                tensor_order: PACKAGE_TENSOR_ORDER.to_owned(),
                logical_layout: SOURCE_LAYOUT.to_owned(),
                metadata_policy: PACKAGE_METADATA_POLICY.to_owned(),
                tensor_count: SORTFORMER_ACTIVATION_TENSORS,
                f32_elements: SORTFORMER_ACTIVATION_F32_ELEMENTS,
                i64_elements: SORTFORMER_ACTIVATION_I64_ELEMENTS,
                payload_bytes: SORTFORMER_ACTIVATION_PAYLOAD_BYTES,
                bytes: SORTFORMER_ACTIVATION_PACKAGE_BYTES,
                sha256: SORTFORMER_ACTIVATION_PACKAGE_SHA256.to_owned(),
            },
            records,
        }
    }

    fn tiny_bundle() -> TinyBundle {
        let position_values = [0.25f32, 0.5, 0.75, 1.0];
        let weight_values = [1.5f32, -2.0];
        let tensors = [
            (
                SORTFORMER_POSITION_TENSOR,
                vec![1, 2, 2],
                position_values.as_slice(),
            ),
            ("encoder.weight", vec![2], weight_values.as_slice()),
        ];
        let package_bytes = make_safetensors(&tensors);
        let position_raw = f32_bytes(&position_values);
        let weight_raw = f32_bytes(&weight_values);

        let mut model = frozen_model_identity();
        model.state_tensors = 2;
        model.state_elements = 3;
        model.state_f32_tensors = 1;
        model.state_f32_elements = 2;
        model.state_f32_bytes = 8;
        model.state_i64_tensors = 1;
        model.state_payload_bytes = 16;

        let counter_name = "encoder.layers.0.conv.batch_norm.num_batches_tracked";
        let mut receipt = SortformerConversionReceipt {
            schema_version: SORTFORMER_RECEIPT_SCHEMA.to_owned(),
            model: model.clone(),
            execution: frozen_execution_config(),
            source_files: frozen_source_files(),
            converter: SortformerConverterIdentity {
                converter_id: SORTFORMER_CONVERTER_ID.to_owned(),
                converter_version: SORTFORMER_CONVERTER_VERSION.to_owned(),
                source_sha256: sha256_bytes(b"audited tiny converter source"),
            },
            runtime: frozen_runtime_identity(),
            license: frozen_license_identity(),
            package: SortformerPackageIdentity {
                format: PACKAGE_FORMAT.to_owned(),
                sha256: String::new(),
                bytes: 0,
                payload_bytes: 24,
                f32_elements: 6,
                tensor_count: 2,
                dtype: PACKAGE_DTYPE.to_owned(),
                byte_order: PACKAGE_BYTE_ORDER.to_owned(),
                tensor_order: PACKAGE_TENSOR_ORDER.to_owned(),
                logical_layout: SOURCE_LAYOUT.to_owned(),
                metadata_policy: PACKAGE_METADATA_POLICY.to_owned(),
            },
            records: vec![
                dropped_counter_record(counter_name),
                exported_f32_record(
                    SORTFORMER_POSITION_TENSOR,
                    SortformerTensorOrigin::NonpersistentBuffer,
                    vec![1, 2, 2],
                    &position_raw,
                ),
                exported_f32_record(
                    "encoder.weight",
                    SortformerTensorOrigin::StateDict,
                    vec![2],
                    &weight_raw,
                ),
                SortformerTensorRecord {
                    source_name: SORTFORMER_DTYPE_SENTINEL.to_owned(),
                    source_origin: SortformerTensorOrigin::NonpersistentBuffer,
                    source_dtype: SortformerTensorDtype::F32,
                    source_shape: vec![0],
                    source_logical_layout: SOURCE_LAYOUT.to_owned(),
                    source_value_sha256: EMPTY_SHA256.to_owned(),
                    source_elements: 0,
                    source_bytes: 0,
                    disposition: SortformerTensorDisposition::DroppedRuntimeSentinel,
                },
            ],
        };
        let tiny_manifest_sha256 =
            tensor_manifest_sha256(&model, &receipt.records).expect("tiny tensor manifest");
        assert_eq!(tiny_manifest_sha256, TINY_TENSOR_MANIFEST_SHA256);
        model
            .tensor_manifest_sha256
            .clone_from(&tiny_manifest_sha256);
        receipt
            .model
            .tensor_manifest_sha256
            .clone_from(&tiny_manifest_sha256);
        bind_package_file_identity(&mut receipt, &package_bytes);
        let expected = ReceiptExpectations {
            model,
            execution: frozen_execution_config(),
            converter_source_sha256: receipt.converter.source_sha256.clone(),
            package_sha256: receipt.package.sha256.clone(),
            package_bytes: receipt.package.bytes,
            source_files: frozen_source_files(),
            runtime: frozen_runtime_identity(),
            license: frozen_license_identity(),
            counter_names: BTreeSet::from([counter_name.to_owned()]),
            position_shape: vec![1, 2, 2],
            state_tensors: 2,
            state_elements: 3,
            state_f32_tensors: 1,
            state_f32_elements: 2,
            state_f32_bytes: 8,
            state_i64_tensors: 1,
            state_payload_bytes: 16,
            source_records: 4,
            exported_tensors: 2,
            dropped_tensors: 2,
            package_f32_elements: 6,
            package_payload_bytes: 24,
        };
        TinyBundle {
            receipt,
            package_bytes,
            expected,
        }
    }

    fn exported_f32_record(
        name: &str,
        origin: SortformerTensorOrigin,
        shape: Vec<u64>,
        raw: &[u8],
    ) -> SortformerTensorRecord {
        let elements =
            checked_elements(&shape, SortformerArtifactDomain::Conversion).expect("tiny shape");
        let bytes = u64::try_from(raw.len()).expect("tiny bytes");
        let value_sha256 = sha256_bytes(raw);
        SortformerTensorRecord {
            source_name: name.to_owned(),
            source_origin: origin,
            source_dtype: SortformerTensorDtype::F32,
            source_shape: shape.clone(),
            source_logical_layout: SOURCE_LAYOUT.to_owned(),
            source_value_sha256: value_sha256.clone(),
            source_elements: elements,
            source_bytes: bytes,
            disposition: SortformerTensorDisposition::Exported {
                transform: SortformerTensorTransform::IdentityContiguousF32,
                destination: SortformerDestinationTensor {
                    name: name.to_owned(),
                    dtype: SortformerTensorDtype::F32,
                    shape,
                    logical_layout: SOURCE_LAYOUT.to_owned(),
                    value_sha256,
                    elements,
                    bytes,
                },
            },
        }
    }

    fn dropped_counter_record(name: &str) -> SortformerTensorRecord {
        SortformerTensorRecord {
            source_name: name.to_owned(),
            source_origin: SortformerTensorOrigin::StateDict,
            source_dtype: SortformerTensorDtype::I64,
            source_shape: Vec::new(),
            source_logical_layout: SOURCE_LAYOUT.to_owned(),
            source_value_sha256: sha256_bytes(&0i64.to_le_bytes()),
            source_elements: 1,
            source_bytes: I64_BYTES,
            disposition: SortformerTensorDisposition::DroppedTrainOnly,
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn make_safetensors(tensors: &[(&str, Vec<u64>, &[f32])]) -> Vec<u8> {
        let mut header = BTreeMap::new();
        let mut payload = Vec::new();
        for (name, shape, values) in tensors {
            let begin = payload.len();
            payload.extend(f32_bytes(values));
            let end = payload.len();
            header.insert(
                (*name).to_owned(),
                json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [begin, end],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&header).expect("safetensors header");
        let mut output = Vec::with_capacity(8 + header_bytes.len() + payload.len());
        output.extend_from_slice(
            &u64::try_from(header_bytes.len())
                .expect("tiny header")
                .to_le_bytes(),
        );
        output.extend_from_slice(&header_bytes);
        output.extend_from_slice(&payload);
        output
    }

    fn duplicate_header_package() -> Vec<u8> {
        let position_entry = r#"{"dtype":"F32","shape":[1,2,2],"data_offsets":[0,16]}"#;
        let weight_entry = r#"{"dtype":"F32","shape":[2],"data_offsets":[16,24]}"#;
        let header = format!(
            r#"{{"encoder.pos_enc.pe":{position_entry},"encoder.pos_enc.pe":{position_entry},"encoder.weight":{weight_entry}}}"#
        );
        let mut output = Vec::new();
        output.extend_from_slice(
            &u64::try_from(header.len())
                .expect("tiny duplicate header")
                .to_le_bytes(),
        );
        output.extend_from_slice(header.as_bytes());
        output.extend(f32_bytes(&[0.25, 0.5, 0.75, 1.0, 1.5, -2.0]));
        output
    }

    fn ambiguous_tensor_entry_package(violation: &str) -> Vec<u8> {
        let position_entry =
            format!(r#"{{"dtype":"F32",{violation}"shape":[1,2,2],"data_offsets":[0,16]}}"#);
        let weight_entry = r#"{"dtype":"F32","shape":[2],"data_offsets":[16,24]}"#;
        let header =
            format!(r#"{{"encoder.pos_enc.pe":{position_entry},"encoder.weight":{weight_entry}}}"#);
        let mut output = Vec::new();
        output.extend_from_slice(
            &u64::try_from(header.len())
                .expect("tiny ambiguous header")
                .to_le_bytes(),
        );
        output.extend_from_slice(header.as_bytes());
        output.extend(f32_bytes(&[0.25, 0.5, 0.75, 1.0, 1.5, -2.0]));
        output
    }

    fn nonlexicographic_payload_package() -> Vec<u8> {
        let mut header = BTreeMap::new();
        header.insert(
            SORTFORMER_POSITION_TENSOR,
            json!({
                "dtype": "F32",
                "shape": [1, 2, 2],
                "data_offsets": [8, 24],
            }),
        );
        header.insert(
            "encoder.weight",
            json!({
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 8],
            }),
        );
        let header_bytes = serde_json::to_vec(&header).expect("nonlexicographic header");
        let mut output = Vec::new();
        output.extend_from_slice(
            &u64::try_from(header_bytes.len())
                .expect("tiny nonlexicographic header")
                .to_le_bytes(),
        );
        output.extend_from_slice(&header_bytes);
        output.extend(f32_bytes(&[1.5, -2.0, 0.25, 0.5, 0.75, 1.0]));
        output
    }

    fn bind_package_file_identity(receipt: &mut SortformerConversionReceipt, package_bytes: &[u8]) {
        receipt.package.bytes = u64::try_from(package_bytes.len()).expect("tiny package");
        receipt.package.sha256 = sha256_bytes(package_bytes);
    }

    fn trust_test_package_identity(bundle: &mut TinyBundle) {
        bundle
            .expected
            .package_sha256
            .clone_from(&bundle.receipt.package.sha256);
        bundle.expected.package_bytes = bundle.receipt.package.bytes;
    }

    fn safetensors_data_start(bytes: &[u8]) -> usize {
        let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("header length"));
        8 + usize::try_from(header_len).expect("tiny header length")
    }

    fn write_bundle(
        directory: &Path,
        receipt: &SortformerConversionReceipt,
        package_bytes: &[u8],
    ) -> (std::path::PathBuf, std::path::PathBuf, String) {
        let receipt_path = directory.join("receipt.json");
        let package_path = directory.join("weights.safetensors");
        let receipt_bytes = canonical_json_bytes(receipt).expect("canonical receipt");
        let receipt_sha256 = sha256_bytes(&receipt_bytes);
        std::fs::write(&receipt_path, receipt_bytes).expect("write receipt");
        std::fs::write(&package_path, package_bytes).expect("write package");
        (receipt_path, package_path, receipt_sha256)
    }

    fn assert_error(error: &FwError, expected: &str) {
        assert!(
            error.to_string().contains(expected),
            "expected `{expected}` in `{error}`"
        );
    }
}
