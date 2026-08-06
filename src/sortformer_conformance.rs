//! Fail-closed L0 artifact verifier for the native Streaming Sortformer port.
//!
//! This module does not run the model and does not admit a production route.
//! It authenticates an operator-local, non-executable safetensors conversion
//! against an independently reviewed canonical receipt. The caller supplies
//! the expected receipt SHA-256 as the trust root; a digest computed from the
//! same untrusted receipt is not authority.

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
/// Receipt schema label only. It is not a code identity; converter code is
/// bound by the receipt's source digest and the independently reviewed receipt
/// digest supplied to the loader.
pub const SORTFORMER_CONVERTER_ID: &str = "franken-whisper-native-sortformer-converter";
pub const SORTFORMER_CONVERTER_VERSION: &str = "1";
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

/// Authenticate the frozen pinned-model census using an independently supplied
/// canonical-receipt digest.
///
/// `expected_receipt_sha256` must come from independent conversion review. A
/// caller that hashes the untrusted `receipt_path` and passes that result here
/// has not established authenticity. There is deliberately no production
/// overload that omits this required trust root.
pub fn load_verified_sortformer_package(
    receipt_path: &Path,
    package_path: &Path,
    expected_receipt_sha256: &str,
) -> FwResult<VerifiedSortformerPackage> {
    load_verified_sortformer_package_with_checkpoint(
        receipt_path,
        package_path,
        expected_receipt_sha256,
        &|| Ok(()),
    )
}

/// Authenticate the frozen pinned-model census with cooperative cancellation.
/// The checkpoint runs before opening either file, between every bounded read,
/// and between tensor validations.
pub fn load_verified_sortformer_package_with_checkpoint(
    receipt_path: &Path,
    package_path: &Path,
    expected_receipt_sha256: &str,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<VerifiedSortformerPackage> {
    load_verified_sortformer_package_against(
        receipt_path,
        package_path,
        expected_receipt_sha256,
        checkpoint,
        &pinned_model_expectations(),
    )
}

#[derive(Debug, Clone)]
struct ReceiptExpectations {
    model: SortformerModelIdentity,
    execution: SortformerExecutionConfig,
    /// `None` until a real converter implementation receives an independent
    /// source review. The mandatory reviewed receipt digest still binds the
    /// converter-source digest in every accepted receipt.
    converter_source_sha256: Option<String>,
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
        converter_source_sha256: None,
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

fn load_verified_sortformer_package_against(
    receipt_path: &Path,
    package_path: &Path,
    expected_receipt_sha256: &str,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    expected: &ReceiptExpectations,
) -> FwResult<VerifiedSortformerPackage> {
    require_sha256("receipt_trust_root", expected_receipt_sha256)?;
    sortformer_checkpoint(checkpoint)?;
    let receipt_bytes =
        read_bounded_file(receipt_path, "receipt", MAX_RECEIPT_BYTES, None, checkpoint)?;
    if sha256_bytes(&receipt_bytes) != expected_receipt_sha256 {
        return Err(sortformer_error(
            "receipt_identity",
            "conversion receipt checksum does not match the independent trust root",
        ));
    }
    sortformer_checkpoint(checkpoint)?;
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
        checkpoint,
    )?;
    verify_compact_safetensors_layout(&package_bytes, receipt.package.payload_bytes, checkpoint)?;
    sortformer_checkpoint(checkpoint)?;
    let package = SafetensorsFile::from_owned_bytes(package_bytes).map_err(|_| {
        sortformer_error(
            "package_structure",
            "weight package is not structurally valid safetensors",
        )
    })?;
    verify_package(&package, &receipt, expected, checkpoint)?;
    sortformer_checkpoint(checkpoint)?;
    Ok(VerifiedSortformerPackage { receipt, package })
}

fn verify_compact_safetensors_layout(
    bytes: &[u8],
    expected_payload_bytes: u64,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<()> {
    sortformer_checkpoint(checkpoint)?;
    let length_prefix: [u8; 8] = bytes
        .get(..8)
        .and_then(|prefix| prefix.try_into().ok())
        .ok_or_else(|| {
            sortformer_error(
                "package_structure",
                "safetensors package lacks a complete header-length prefix",
            )
        })?;
    let header_bytes_u64 = u64::from_le_bytes(length_prefix);
    if header_bytes_u64 == 0 || header_bytes_u64 > MAX_PACKAGE_HEADER_BYTES {
        return Err(sortformer_error(
            "package_structure",
            "safetensors header size is outside the Sortformer envelope",
        ));
    }
    let header_bytes = usize::try_from(header_bytes_u64).map_err(|_| {
        sortformer_error(
            "package_structure",
            "safetensors header size does not fit this platform",
        )
    })?;
    let data_start = 8usize.checked_add(header_bytes).ok_or_else(|| {
        sortformer_error(
            "package_structure",
            "safetensors header offset overflows this platform",
        )
    })?;
    let data_len = bytes.len().checked_sub(data_start).ok_or_else(|| {
        sortformer_error(
            "package_structure",
            "safetensors header extends past the package bytes",
        )
    })?;
    if u64::try_from(data_len).ok() != Some(expected_payload_bytes) {
        return Err(sortformer_error(
            "package_layout",
            "safetensors data section is not exactly the frozen payload size",
        ));
    }
    let header: serde_json::Value =
        serde_json::from_slice(bytes.get(8..data_start).ok_or_else(|| {
            sortformer_error(
                "package_structure",
                "safetensors header span is unavailable",
            )
        })?)
        .map_err(|_| {
            sortformer_error("package_structure", "safetensors header is not valid JSON")
        })?;
    sortformer_checkpoint(checkpoint)?;
    let entries = header.as_object().ok_or_else(|| {
        sortformer_error(
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
                sortformer_error(
                    "package_structure",
                    "safetensors tensor has invalid data offsets",
                )
            })?;
        let begin = offsets
            .first()
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                sortformer_error(
                    "package_structure",
                    "safetensors tensor begin offset is invalid",
                )
            })?;
        let end = offsets
            .get(1)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                sortformer_error(
                    "package_structure",
                    "safetensors tensor end offset is invalid",
                )
            })?;
        spans.push((name.as_str(), begin, end));
    }
    spans.sort_unstable_by_key(|(name, _, _)| *name);
    let mut cursor = 0u64;
    for (_, begin, end) in spans {
        sortformer_checkpoint(checkpoint)?;
        if begin != cursor || end < begin || end > expected_payload_bytes {
            return Err(sortformer_error(
                "package_layout",
                "safetensors tensor spans are not a compact non-overlapping payload",
            ));
        }
        cursor = end;
    }
    if cursor != expected_payload_bytes {
        return Err(sortformer_error(
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
    require_sha256("converter_identity", &receipt.converter.source_sha256)?;
    if let Some(expected_converter_sha256) = &expected.converter_source_sha256
        && receipt.converter.source_sha256 != *expected_converter_sha256
    {
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
    require_sha256("package_identity", &package.sha256)?;
    if package.format != PACKAGE_FORMAT
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
        sortformer_checkpoint(checkpoint)?;
        validate_tensor_name(&record.source_name)?;
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
        require_sha256("record_hash", &record.source_value_sha256)?;
        if record.source_logical_layout != SOURCE_LAYOUT {
            return Err(sortformer_error(
                "record_layout",
                "source tensor layout is not the frozen contiguous row-major layout",
            ));
        }
        let source_elements = checked_elements(&record.source_shape)?;
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
    require_sha256("record_hash", &destination.value_sha256)?;
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
        sortformer_checkpoint(checkpoint)?;
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
        let observed_sha256 = finite_f32_sha256(raw, checkpoint)?;
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
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<String> {
    if !raw.len().is_multiple_of(4) {
        return Err(sortformer_error(
            "package_payload",
            "F32 tensor payload is not four-byte aligned",
        ));
    }
    let mut hasher = Sha256::new();
    for block in raw.chunks(READ_CHUNK_BYTES) {
        sortformer_checkpoint(checkpoint)?;
        hasher.update(block);
        let (chunks, remainder) = block.as_chunks::<4>();
        if !remainder.is_empty() {
            return Err(sortformer_error(
                "package_payload",
                "F32 tensor chunk has invalid width",
            ));
        }
        for &bytes in chunks {
            if !f32::from_le_bytes(bytes).is_finite() {
                return Err(sortformer_error(
                    "package_nonfinite",
                    "converted package contains a non-finite F32 value",
                ));
            }
        }
    }
    Ok(hex_digest(hasher.finalize()))
}

fn read_bounded_file(
    path: &Path,
    kind: &str,
    maximum_bytes: u64,
    expected: Option<(u64, &str)>,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<Vec<u8>> {
    sortformer_checkpoint(checkpoint)?;
    let file = File::open(path).map_err(|_| {
        sortformer_error(
            &format!("{kind}_open"),
            &format!("{kind} artifact could not be opened"),
        )
    })?;
    let metadata_bytes = file
        .metadata()
        .map_err(|_| {
            sortformer_error(
                &format!("{kind}_metadata"),
                &format!("{kind} artifact metadata is unavailable"),
            )
        })?
        .len();
    if metadata_bytes == 0 || metadata_bytes > maximum_bytes {
        return Err(sortformer_error(
            &format!("{kind}_size"),
            &format!("{kind} artifact size is outside its bounded envelope"),
        ));
    }
    if let Some((expected_bytes, expected_sha256)) = expected {
        require_sha256(&format!("{kind}_identity"), expected_sha256)?;
        if metadata_bytes != expected_bytes {
            return Err(sortformer_error(
                &format!("{kind}_identity"),
                &format!("{kind} artifact length does not match the authenticated receipt"),
            ));
        }
    }
    let capacity = usize::try_from(metadata_bytes).map_err(|_| {
        sortformer_error(
            &format!("{kind}_size"),
            &format!("{kind} artifact size does not fit this platform"),
        )
    })?;
    let mut bytes = reserved_byte_buffer(capacity, kind, checkpoint)?;
    let mut hasher = Sha256::new();
    let mut reader = file;
    let mut buffer = [0u8; READ_CHUNK_BYTES];
    loop {
        sortformer_checkpoint(checkpoint)?;
        let read = reader.read(&mut buffer).map_err(|_| {
            sortformer_error(
                &format!("{kind}_read"),
                &format!("{kind} artifact could not be read"),
            )
        })?;
        if read == 0 {
            break;
        }
        if read > capacity.saturating_sub(bytes.len()) {
            return Err(sortformer_error(
                &format!("{kind}_identity"),
                &format!("{kind} artifact length changed while it was read"),
            ));
        }
        let chunk = buffer.get(..read).ok_or_else(|| {
            sortformer_error(
                &format!("{kind}_read"),
                &format!("{kind} artifact read returned an invalid span"),
            )
        })?;
        hasher.update(chunk);
        bytes.extend_from_slice(chunk);
    }
    if bytes.len() != capacity {
        return Err(sortformer_error(
            &format!("{kind}_identity"),
            &format!("{kind} artifact length changed while it was read"),
        ));
    }
    let observed_sha256 = hex_digest(hasher.finalize());
    if let Some((_, expected_sha256)) = expected
        && observed_sha256 != expected_sha256
    {
        return Err(sortformer_error(
            &format!("{kind}_identity"),
            &format!("{kind} artifact checksum does not match the authenticated receipt"),
        ));
    }
    sortformer_checkpoint(checkpoint)?;
    Ok(bytes)
}

fn reserved_byte_buffer(
    capacity: usize,
    kind: &str,
    checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
) -> FwResult<Vec<u8>> {
    sortformer_checkpoint(checkpoint)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|_| {
        sortformer_error(
            &format!("{kind}_allocation"),
            &format!("{kind} artifact buffer could not be allocated within its bounded envelope"),
        )
    })?;
    sortformer_checkpoint(checkpoint)?;
    Ok(bytes)
}

fn checked_elements(shape: &[u64]) -> FwResult<u64> {
    if shape.len() > 8 {
        return Err(sortformer_error(
            "record_shape",
            "tensor rank exceeds the bounded receipt schema",
        ));
    }
    shape.iter().try_fold(1u64, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| sortformer_error("record_shape", "tensor element count overflows"))
    })
}

fn checked_add(left: u64, right: u64, code: &str) -> FwResult<u64> {
    left.checked_add(right)
        .ok_or_else(|| sortformer_error(code, "tensor census sum overflows"))
}

fn validate_tensor_name(name: &str) -> FwResult<()> {
    if name.is_empty()
        || name.len() > 512
        || name.starts_with('.')
        || name.ends_with('.')
        || name
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_')))
    {
        return Err(sortformer_error(
            "record_name",
            "tensor name is outside the bounded dotted-identifier grammar",
        ));
    }
    Ok(())
}

fn require_sha256(code: &str, value: &str) -> FwResult<()> {
    if value.len() != HASH_HEX_LEN
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(sortformer_error(
            code,
            "SHA-256 identity must be exactly 64 lowercase hexadecimal characters",
        ));
    }
    Ok(())
}

fn sortformer_checkpoint(checkpoint: &(dyn Fn() -> FwResult<()> + Sync)) -> FwResult<()> {
    match checkpoint() {
        Ok(()) => Ok(()),
        Err(FwError::Cancelled(_)) => Err(FwError::Cancelled(
            "sortformer_conversion.load_cancelled: cooperative checkpoint requested cancellation"
                .to_owned(),
        )),
        Err(_) => Err(sortformer_error(
            "checkpoint_failure",
            "caller checkpoint returned a non-cancellation failure",
        )),
    }
}

fn sortformer_error(code: &str, message: &str) -> FwError {
    FwError::InvalidRequest(format!("sortformer_conversion.{code}: {message}"))
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

    struct TinyBundle {
        receipt: SortformerConversionReceipt,
        package_bytes: Vec<u8>,
        expected: ReceiptExpectations,
    }

    #[test]
    fn bounded_file_buffer_allocation_fails_closed() {
        let error = reserved_byte_buffer(usize::MAX, "package", &|| Ok(()))
            .expect_err("an impossible package capacity must not abort");
        assert_error(&error, "package_allocation");
    }

    #[test]
    fn bounded_file_buffer_checks_cancellation_after_reservation() {
        let checkpoints = AtomicUsize::new(0);
        let error = reserved_byte_buffer(16, "package", &|| {
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
        assert!(expected.converter_source_sha256.is_none());
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
            checked_elements(&expected.position_shape).unwrap(),
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
        bind_package_file_identity(&mut receipt, &package_bytes);
        let expected = ReceiptExpectations {
            model,
            execution: frozen_execution_config(),
            converter_source_sha256: Some(receipt.converter.source_sha256.clone()),
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
        let elements = checked_elements(&shape).expect("tiny shape");
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
