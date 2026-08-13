//! Neural speech denoising as a default pipeline stage (bd-z6kz).
//!
//! The model is FastEnhancer-S (MIT, pinned upstream — see frankentts
//! `docs/DENOISER.md` for the conversion recipe), consumed through
//! `ftts_kernels::enhance::Enhancer`: a causal 48 kHz STFT complex-ratio-mask
//! network, ~838 KB of f32 safetensors. This module owns the whisper-side
//! plumbing: hydrate from the pinned artifact, and denoise 16 kHz mono PCM
//! length-preservingly (16 k → 48 k → network → 16 k, mirroring the
//! reference's own 24 kHz product path), chunked so the 48 kHz intermediate
//! never scales with recording length.
//!
//! Policy: ON whenever the artifact is present, OFF when it is absent, and
//! explicitly togglable (`FW_DENOISE=0` kills it; the browser has a
//! checkbox). CI and golden fixtures run without the artifact, so their
//! transcripts stay byte-identical; machines that pulled the artifact get
//! the stage by default. Same convention frankentts ships.

use std::path::PathBuf;
use std::sync::OnceLock;

use ftts_kernels::enhance::{Enhancer, SAMPLE_RATE_HZ, resample_lanczos6};

use crate::error::{FwError, FwResult};
use crate::native_engine::weights::SafetensorsFile;

/// Pinned artifact identity (frankentts release `model-qwen3-tts-v1`).
pub const DENOISER_FILENAME: &str = "fastenhancer-s-48k-denoise.safetensors";
pub const DENOISER_SHA256: &str =
    "28c1807fd9113e4ca09d3aacb2ecb07a742917321bfaced8b92598daffbd098b";
pub const DENOISER_BYTES: u64 = 838_440;

/// Chunking for the 48 kHz intermediate: 60 s of audio per chunk with 1 s of
/// context re-fed before each boundary (the network is causal; mask effects
/// of a boundary decay well inside a second). Keeps peak extra memory at
/// ~12 MB regardless of recording length.
const CHUNK_SEC: usize = 60;
const CTX_SEC: usize = 1;
const RATE_16K: usize = 16_000;

/// A hydrated denoiser.
pub struct Denoiser {
    enhancer: Enhancer,
}

impl Denoiser {
    /// Hydrate from safetensors bytes (the browser path, and the native path
    /// after its pin check).
    ///
    /// # Errors
    ///
    /// Malformed safetensors or a tensor set that does not describe
    /// FastEnhancer-S geometry.
    pub fn from_bytes(bytes: Vec<u8>) -> FwResult<Self> {
        let file = SafetensorsFile::from_owned_bytes(bytes)
            .map_err(|e| FwError::InvalidRequest(format!("denoiser artifact: {e}")))?;
        let mut tensors = std::collections::BTreeMap::new();
        let names: Vec<String> = file.names().map(str::to_owned).collect();
        for name in names {
            let (shape, data) = file.tensor_f32(&name)?;
            tensors.insert(name, (shape, data));
        }
        let enhancer = Enhancer::load(tensors)
            .map_err(|e| FwError::InvalidRequest(format!("denoiser artifact: {e}")))?;
        Ok(Self { enhancer })
    }

    /// Denoise 16 kHz mono PCM, preserving length exactly (so downstream
    /// timestamps need no bookkeeping). Chunked; see module docs.
    #[must_use]
    pub fn denoise_16k(&self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }
        let chunk = CHUNK_SEC * RATE_16K;
        let ctx = CTX_SEC * RATE_16K;
        if samples.len() <= chunk + ctx {
            return self.denoise_16k_whole(samples);
        }
        let mut out = Vec::with_capacity(samples.len());
        let mut start = 0;
        while start < samples.len() {
            let end = (start + chunk).min(samples.len());
            let ctx_start = start.saturating_sub(ctx);
            let denoised = self.denoise_16k_whole(&samples[ctx_start..end]);
            out.extend_from_slice(&denoised[start - ctx_start..]);
            start = end;
        }
        debug_assert_eq!(out.len(), samples.len());
        out
    }

    /// One un-chunked pass: the reference's own 24 kHz product shape, at
    /// 16 kHz (Lanczos-6 up to the model's native 48 kHz and back).
    fn denoise_16k_whole(&self, wav16k: &[f32]) -> Vec<f32> {
        let wav48 = resample_lanczos6(wav16k, RATE_16K as u32, SAMPLE_RATE_HZ);
        let enhanced = self.enhancer.enhance_48k(&wav48);
        let mut back = resample_lanczos6(&enhanced, SAMPLE_RATE_HZ, RATE_16K as u32);
        back.truncate(wav16k.len());
        // Length preservation is the timestamp contract; pad the (at most
        // one-sample) rounding shortfall rather than silently shrinking.
        back.resize(wav16k.len(), 0.0);
        back
    }
}

/// `FW_DENOISE=0` (or `off`/`false`/`no`) disables the stage even when the
/// artifact is present. Anything else — including unset — leaves it on.
pub fn denoise_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("FW_DENOISE")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "off" | "false" | "no"
        )
    })
}

/// Native artifact resolution: `FW_DENOISER_PATH` override, else the model
/// cache (`<cache>/models/denoiser/<file>`). Absent or unpinnable = None
/// (the stage silently skips — CI and golden fixtures rely on this).
#[cfg(not(target_arch = "wasm32"))]
fn artifact_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FW_DENOISER_PATH") {
        let path = PathBuf::from(path);
        return path.is_file().then_some(path);
    }
    let root = crate::model_distribution::whisper_cache_dir().ok()?;
    // `<cache>/models/whisper/<ver>` → `<cache>/models/denoiser/<file>`.
    let path = root
        .parent()?
        .parent()?
        .join("denoiser")
        .join(DENOISER_FILENAME);
    path.is_file().then_some(path)
}

/// The process-wide denoiser, hydrated once from the pinned artifact when
/// present and enabled; `None` otherwise. The sha256 pin is enforced before
/// a byte of it is trusted.
#[cfg(not(target_arch = "wasm32"))]
pub fn shared() -> Option<&'static Denoiser> {
    static SHARED: OnceLock<Option<Denoiser>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            if !denoise_enabled() {
                return None;
            }
            let path = artifact_path()?;
            let bytes = std::fs::read(&path).ok()?;
            if bytes.len() as u64 != DENOISER_BYTES {
                return None;
            }
            use sha2::Digest as _;
            let digest = format!("{:x}", sha2::Sha256::digest(&bytes));
            if digest != DENOISER_SHA256 {
                return None;
            }
            Denoiser::from_bytes(bytes).ok()
        })
        .as_ref()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    /// Garbage bytes must fail hydration loudly, not build a broken model.
    #[test]
    fn malformed_artifact_is_rejected() {
        assert!(super::Denoiser::from_bytes(vec![0u8; 64]).is_err());
    }
}
