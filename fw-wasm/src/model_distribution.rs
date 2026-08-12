//! Stub for the parent crate's `model_distribution` module.
//!
//! The canonical module (`../../src/model_distribution.rs`) resolves
//! hash-pinned model packages from an on-disk cache — meaningless in a
//! browser, where weights arrive as bytes via OPFS/fetch. `native_engine`
//! only calls these functions from its filesystem model-resolution paths
//! (`find_model_file` and friends), which the wasm embedding never uses:
//! every resolver here fails closed with `FwError::Unsupported` so a wasm
//! caller that strays onto a disk path gets a named error, not a trap.

use std::path::PathBuf;

use crate::error::{FwError, FwResult};

/// Mirror of the canonical artifact-version pin (used only in path joins on
/// resolution paths that always fail here).
pub const WHISPER_ARTIFACT_VERSION: &str = "whisper-large-v3-turbo-f16-v1";
/// Mirror of the canonical weights filename.
pub const WHISPER_WEIGHTS_FILENAME: &str = "ggml-large-v3-turbo.bin";

/// Mirror of the canonical package descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedWhisperPackage {
    pub weights_path: PathBuf,
    pub artifact_version: String,
    pub weights_sha256: String,
}

/// Browser builds have no model cache directory.
pub fn whisper_cache_dir() -> FwResult<PathBuf> {
    Err(FwError::Unsupported(
        "model cache directory is unavailable on wasm; load weights from bytes".to_string(),
    ))
}

/// Browser builds cannot resolve disk-cached weights.
pub fn resolve_cached_whisper() -> FwResult<CachedWhisperPackage> {
    Err(FwError::Unsupported(
        "disk model resolution is unavailable on wasm; load weights from bytes".to_string(),
    ))
}
