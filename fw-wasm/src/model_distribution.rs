//! Stub for the parent crate's `model_distribution` module.
//!
//! The canonical module (`../../src/model_distribution.rs`) resolves
//! hash-pinned model packages from an on-disk cache — meaningless in a
//! browser, where weights arrive as bytes via OPFS/fetch. `native_engine`
//! only calls these functions from its filesystem model-resolution paths
//! (`find_model_file` and friends), which the wasm embedding never uses:
//! every resolver here fails closed with `FwError::Unsupported` so a wasm
//! caller that strays onto a disk path gets a named error, not a trap.

use std::fs::File;
use std::path::PathBuf;

use crate::error::{FwError, FwResult};

/// Mirror of the canonical artifact-version pin (used only in path joins on
/// resolution paths that always fail here).
pub const WHISPER_ARTIFACT_VERSION: &str = "whisper-large-v3-turbo-f16-v1";
/// Mirrors the canonical English fast-lane artifact directory.
pub const TINY_EN_ARTIFACT_VERSION: &str = "whisper-tiny-en-f16-v1";
/// Mirrors the canonical multilingual fast-lane artifact directory.
pub const TINY_ARTIFACT_VERSION: &str = "whisper-tiny-f16-v1";
/// Mirror of the canonical weights filename.
pub const WHISPER_WEIGHTS_FILENAME: &str = "ggml-large-v3-turbo.bin";

/// Mirror of the canonical package descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedWhisperPackage {
    pub weights_path: PathBuf,
    pub artifact_version: String,
    pub weights_sha256: String,
}

impl CachedWhisperPackage {
    /// Embedded callers never receive a cache-authenticated package: wasm
    /// stages bytes and fw-ios opens the app-verified explicit model path.
    /// Keep the native engine's type-level seam complete while failing closed
    /// if a caller somehow constructs this descriptor and requests its file.
    pub(crate) fn try_clone_weights_file(&self) -> FwResult<File> {
        Err(FwError::Unsupported(
            "authenticated cache file handles are unavailable in this embedding".to_string(),
        ))
    }
}

/// Browser builds have no model cache directory.
pub fn whisper_cache_dir() -> FwResult<PathBuf> {
    Err(FwError::Unsupported(
        "model cache directory is unavailable on wasm; load weights from bytes".to_string(),
    ))
}

/// Browser builds cannot resolve disk-cached weights.
pub fn resolve_cached_whisper() -> FwResult<CachedWhisperPackage> {
    resolve_cached_whisper_with_cancel(|| false)
}

/// Embedded builds have no desktop model cache, but cancellation is still
/// authoritative when the shared native resolver reaches this seam.
pub fn resolve_cached_whisper_with_cancel<F>(is_cancelled: F) -> FwResult<CachedWhisperPackage>
where
    F: Fn() -> bool + Sync,
{
    if is_cancelled() {
        return Err(FwError::Cancelled(
            "model resolution was cancelled".to_string(),
        ));
    }
    Err(FwError::Unsupported(
        "disk cache model resolution is unavailable in this embedding; load weights from explicit bytes or a host-verified path".to_string(),
    ))
}
