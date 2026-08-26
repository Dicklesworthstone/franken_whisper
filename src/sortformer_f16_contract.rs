//! Shared, pin-free contract for the Sortformer f16 derivation receipt.
//!
//! The offline converter and every runtime verifier compile these exact types.
//! Admission roots intentionally live elsewhere so this complete schema source
//! can be hashed into the receipt without a self-referential digest cycle.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

pub const SORTFORMER_F16_DERIVATION_RECEIPT_SCHEMA: &str =
    "franken-whisper-sortformer-f16-derivation-receipt-v2";
pub const SORTFORMER_F16_DERIVATION_METHOD: &str = "rte-f16-downcast";
pub const SORTFORMER_F16_DERIVATION_METHOD_VERSION: &str = "1";

/// Strict lineage receipt for a deterministic f32-to-f16 derivation.
///
/// A receipt produced by the converter is not itself admission authority; a
/// runtime verifier must still bind its exact digest and package census through
/// independently compiled trust roots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerF16DerivationReceipt {
    pub schema_version: String,
    pub model_id: String,
    pub model_revision: String,
    pub derivation: SortformerF16DerivationIdentity,
    pub package: SortformerF16PackageIdentity,
    pub records: Vec<SortformerF16TensorRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerF16DerivationIdentity {
    pub parent_receipt_sha256: String,
    pub parent_package_sha256: String,
    pub method: String,
    pub method_version: String,
    pub rounding_mode: String,
    pub receipt_contract_source_sha256: String,
    pub downcaster_core_source_sha256: String,
    pub downcaster_cli_source_sha256: String,
    pub parent_verifier_source_sha256: String,
    pub safetensors_parser_source_sha256: String,
    pub cargo_manifest_sha256: String,
    pub cargo_lock_sha256: String,
    pub rust_toolchain_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerF16PackageIdentity {
    pub format: String,
    pub sha256: String,
    pub bytes: u64,
    pub payload_bytes: u64,
    pub tensor_count: u64,
    pub f16_tensors: u64,
    pub f16_elements: u64,
    pub f16_bytes: u64,
    pub i64_tensors: u64,
    pub i64_elements: u64,
    pub i64_bytes: u64,
    pub dtype_set: Vec<SortformerF16ArtifactDtype>,
    pub byte_order: String,
    pub tensor_order: String,
    pub metadata_policy: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SortformerF16ArtifactDtype {
    #[serde(rename = "F32")]
    F32,
    #[serde(rename = "F16")]
    F16,
    #[serde(rename = "I64")]
    I64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortformerF16TensorTransform {
    RoundToNearestTiesToEvenF32ToF16,
    IdentityI64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SortformerF16TensorRecord {
    pub name: String,
    pub shape: Vec<u64>,
    pub elements: u64,
    pub source_dtype: SortformerF16ArtifactDtype,
    pub destination_dtype: SortformerF16ArtifactDtype,
    pub transform: SortformerF16TensorTransform,
    pub source_bytes: u64,
    pub destination_bytes: u64,
    pub source_value_sha256: String,
    pub destination_value_sha256: String,
}
