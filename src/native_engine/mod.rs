//! Real in-process Whisper inference engine (pure Rust, no FFI).
//!
//! This module replaces the former mock "pilot" engines with genuine ASR:
//! it parses whisper.cpp ggml `.bin` model files, computes the log-mel
//! frontend, runs the encoder/decoder transformer forward passes on
//! `ft-kernel-cpu` (FrankenTorch) compute kernels, and decodes tokens with
//! whisper's timestamp rules.
//!
//! Module map (one bead per module; see `.beads/`):
//! - [`ggml`]      — model file parser (bd-s3y6)
//! - [`mel`]       — log-mel spectrogram frontend (bd-1eof)
//! - [`tokenizer`] — BPE decode + special-token map (bd-zpfy)
//! - [`nn`]        — inference kernels facade + KV-cache attention (bd-g3h4)
//! - [`encoder`]   — audio encoder forward (bd-9ycw)
//! - [`decoder`]   — text decoder forward (bd-hlpk)
//! - [`decode`]    — greedy decode loop / windowing (bd-szkq)
//!
//! This file (the module root, bd-hsbx) also hosts the **public engine API**:
//! [`NativeWhisperModel`] (a cached, reference-counted loaded model),
//! [`resolve_model`] / [`native_model_available`] (model-spec resolution and
//! honest, header-only availability sniffing — never any network access), and
//! the threading/default helpers [`default_threads`] / [`default_model_spec`].
//! Actual inference is delegated to [`decode::transcribe_samples`] against the
//! frozen [`decode::DecodeParams`] / [`decode::DecodeOutput`] contract.
//!
//! Numerical conventions shared by every submodule:
//! - All matrices are **row-major** `Mat { rows, cols, data }`.
//! - Mel spectrograms are mel-major: `data[mel_bin * n_frames + frame]`,
//!   mirroring whisper.cpp's `whisper_mel` layout.
//! - All forward passes are f32; f16 model weights are converted at load.

pub mod decode;
pub mod decoder;
pub mod dtw;
pub mod encoder;
pub mod ggml;
pub mod mel;
pub mod nn;
pub mod plat;
pub mod tokenizer;
pub mod weights;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use sha2::{Digest, Sha256};

use crate::error::{FwError, FwResult};

/// Whisper model hyper-parameters, in the exact order they appear in the
/// ggml `.bin` header (11 consecutive little-endian `i32` values following
/// the `0x67676d6c` magic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhisperHParams {
    pub n_vocab: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
    pub n_mels: i32,
    pub ftype: i32,
}

impl WhisperHParams {
    /// whisper.cpp convention: a vocab of >= 51865 entries marks a
    /// multilingual model (tiny.en etc. have 51864).
    #[must_use]
    pub fn is_multilingual(&self) -> bool {
        self.n_vocab >= 51865
    }
}

/// Tensor element type found in a ggml tensor directory entry.
// Variant names mirror ggml's `GGML_TYPE_*` C enum verbatim (Q8_0, Q6_K, ...).
// `non_camel_case_types` accepts `_` only between two digits (so `Q8_0` is fine)
// but rejects the k-quant `_K` suffix; keep the canonical ggml spelling.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDType {
    F32,
    F16,
    /// ggml `Q8_0` (per-tensor GGML_TYPE 8): blocks of 32 `int8` quants with one
    /// `f16` scale each (34 bytes/block, `x = q * scale`). Dequantized to f32 at
    /// load — lets the engine run whisper.cpp-quantized `q8_0` models on the
    /// existing f32 path.
    Q8_0,
    /// ggml `Q5_0` (per-tensor GGML_TYPE 6): blocks of 32 5-bit quants with one
    /// `f16` scale each (22 bytes/block: scale + 4-byte high-bit field + 16-byte
    /// nibbles; `x = ((nibble | hi<<4) - 16) * scale`). Dequantized to f32 at
    /// load, like [`Self::Q8_0`].
    Q5_0,
    /// ggml `Q4_0` (per-tensor GGML_TYPE 2): blocks of 32 4-bit quants with one
    /// `f16` scale each (18 bytes/block: scale + 16-byte nibbles; `x = (nibble -
    /// 8) * scale`). Dequantized to f32 at load, like [`Self::Q8_0`].
    Q4_0,
    /// ggml `Q4_1` (per-tensor GGML_TYPE 3): 4-bit quants with a per-block scale
    /// AND min (20 bytes/block: scale + min + 16-byte nibbles; `x = nibble *
    /// scale + min`). Dequantized to f32 at load.
    Q4_1,
    /// ggml `Q5_1` (per-tensor GGML_TYPE 7): 5-bit quants with a per-block scale
    /// AND min (24 bytes/block: scale + min + 4-byte high-bit field + 16-byte
    /// nibbles; `x = (nibble | hi<<4) * scale + min`). Dequantized to f32 at load.
    Q5_1,
    /// ggml `Q6_K` (per-tensor GGML_TYPE 14): k-quant 6-bit, 256-value
    /// super-blocks (210 bytes: 128-byte low-nibbles + 64-byte high-2-bits +
    /// 16-byte int8 sub-scales + `f16` super-scale; `x = d * sub_scale * (6bit −
    /// 32)`). Dequantized to f32 at load.
    Q6_K,
    /// ggml `Q4_K` (per-tensor GGML_TYPE 12): k-quant 4-bit, 256-value
    /// super-blocks (144 bytes: `f16 d, f16 dmin, scales[12] (8×6-bit packed
    /// scale+min), qs[128] (4-bit)`; per 32-value sub-block `x = d*sc*nibble −
    /// dmin*min`, sub-scales unpacked via `get_scale_min_k4`). Dequantized to f32
    /// at load.
    Q4_K,
    /// ggml `Q5_K` (per-tensor GGML_TYPE 13): k-quant 5-bit, 256-value
    /// super-blocks (176 bytes: `f16 d, f16 dmin, scales[12], qh[32] (high bit),
    /// qs[128] (low 4-bit)`; `x = d*sc*((nibble)+(high-bit?16:0)) − dmin*min`,
    /// same `get_scale_min_k4` sub-scales as `Q4_K` plus a per-group high-bit
    /// plane). Dequantized to f32 at load.
    Q5_K,
    /// ggml `Q3_K` (per-tensor GGML_TYPE 11): k-quant 3-bit, 256-value
    /// super-blocks (110 bytes: `hmask[32] (high bit), qs[64] (2-bit), scales[12]
    /// (bit-shuffled 6-bit), f16 d`; no per-block min — `x = d*(scale−32)*(2bit −
    /// (hmask-bit?0:4))`). Dequantized to f32 at load.
    Q3_K,
    /// ggml `Q2_K` (per-tensor GGML_TYPE 10): k-quant 2-bit, 256-value
    /// super-blocks (84 bytes: `scales[16] (4-bit scale | 4-bit min), qs[64]
    /// (2-bit), f16 d, f16 dmin`; `x = d*(sc&0xF)*2bit − dmin*(sc>>4)`).
    /// Dequantized to f32 at load — the coarsest quant native decodes.
    Q2_K,
}

/// Mel filterbank embedded in the ggml model file (`n_mel x n_fft_bins`,
/// row-major: `data[mel * n_fft_bins + bin]`). Using the file's own filters
/// guarantees our frontend matches whisper.cpp's bin weights exactly.
#[derive(Debug, Clone)]
pub struct MelFilterbank {
    pub n_mel: usize,
    pub n_fft_bins: usize,
    pub data: Vec<f32>,
}

/// Log-mel spectrogram, mel-major (`data[mel * n_frames + frame]`).
#[derive(Debug, Clone)]
pub struct Mel {
    pub n_mel: usize,
    pub n_frames: usize,
    pub data: Vec<f32>,
}

/// Row-major f32 matrix: `data[row * cols + col]`. The single tensor
/// currency of the inference path; weights are pre-transposed at load time
/// so every matmul is a contiguous `[m,k] x [k,n]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[must_use]
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        debug_assert_eq!(rows * cols, data.len(), "Mat shape/data mismatch");
        Self { rows, cols, data }
    }

    #[must_use]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

/// The ordered list of directories searched for ggml model files by short
/// name, derived from the current process environment.
///
/// Precedence (highest first):
/// 1. The FrankenWhisper release-package directory beneath
///    `$FRANKEN_WHISPER_MODEL_DIR`, when configured.
/// 2. `$FRANKEN_WHISPER_MODEL_DIR` — operator-chosen production model dir.
/// 3. `$FRANKEN_WHISPER_TEST_MODEL_DIR` — CI / dev fixtures.
/// 4. The verified FrankenWhisper release-package directory beneath the default
///    production cache.
/// 5. `~/.cache/franken_whisper/models` — default production cache.
/// 6. `~/.cache/franken_whisper/test-models` — default test cache.
/// 7. `~/models/whisper` — the conventional whisper.cpp download location.
///
/// Empty env vars are skipped. The home-relative entries are omitted entirely
/// when `$HOME` is unset (rather than rooting at the filesystem root). This is
/// the single source of truth shared by [`find_model_file`], [`resolve_model`],
/// and [`native_model_available`], so the search order can never drift between
/// them.
#[must_use]
fn model_search_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("FRANKEN_WHISPER_MODEL_DIR")
        && !dir.is_empty()
    {
        let root = PathBuf::from(dir);
        dirs.push(
            root.join("whisper")
                .join(crate::model_distribution::WHISPER_ARTIFACT_VERSION),
        );
        dirs.push(root);
    }
    if let Ok(dir) = std::env::var("FRANKEN_WHISPER_TEST_MODEL_DIR")
        && !dir.is_empty()
    {
        dirs.push(PathBuf::from(dir));
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        let cache_root = home.join(".cache").join("franken_whisper").join("models");
        dirs.push(
            cache_root
                .join("whisper")
                .join(crate::model_distribution::WHISPER_ARTIFACT_VERSION),
        );
        dirs.push(cache_root);
        dirs.push(
            home.join(".cache")
                .join("franken_whisper")
                .join("test-models"),
        );
        dirs.push(home.join("models").join("whisper"));
    }
    dirs
}

/// The on-disk filename a short model name maps to (`tiny.en` → `ggml-tiny.en.bin`).
#[must_use]
fn model_file_name(short_name: &str) -> String {
    format!("ggml-{short_name}.bin")
}

/// Measurement-only span emission for the profiling workflow
/// (`profiling-software-performance`). When `FRANKEN_WHISPER_PERF_SPANS=1`,
/// stage timings are written to stderr as one JSON object per line with the
/// `perf.profile.span_summary` event name. Never enabled by default; adds two
/// atomic loads when off. Do NOT use for production telemetry (events go
/// through the orchestrator's NDJSON stream instead).
pub(crate) fn perf_spans_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FRANKEN_WHISPER_PERF_SPANS").is_ok_and(|v| v == "1"))
}

/// Whether the f16-resident decoder compute path is enabled.
///
/// This is the runtime kill-switch for the f16-GEMV lever
/// (`FRANKEN_WHISPER_NATIVE_F16_COMPUTE`): when ON, decoder linear/logits
/// weights that are stored as f16 in the ggml file are kept f16-resident and
/// dequantized inside a fused GEMV (`out[o] = dot(W[o,:], x)` over natural
/// `[out, in]` rows) instead of being dequantized to a full f32 `Mat` and
/// pre-transposed at load. This halves the decoder weights' resident footprint
/// AND skips the load-time transpose; the value of every weight is **exact**
/// (the f16→f32 dequant changes nothing), only the dot-product accumulation
/// order differs from `matrixmultiply`'s blocked f32 sgemm — so it is a
/// **numerics-affecting** lever.
///
/// # Default is ON — vectorized dequant flips the pass-2 regression to a win
///
/// Pass 2 shipped this path **default OFF** because, although its conformance
/// gate passed, the interleaved e2e wall A/B *regressed* (+27% tiny / +12%
/// large): the per-element `half`→f32 dequant was interleaved with the FMA
/// inside the dot loop, which serialized BOTH the half widen and the multiply-
/// add and blocked autovectorization. Pass 3 fixed the kernel
/// ([`super::nn::gemv_f16`]): each weight row is now bulk-dequantized via the
/// SIMD `HalfFloatSliceExt::convert_to_f32_slice` (4-wide aarch64 `fp16` /
/// 8-wide x86 `f16c`) into a reused f32 scratch buffer, then dotted with the
/// vectorizable 8-lane [`super::nn::dot8`]. Measured on M4 Pro:
///
/// * isolated GEMV (`f16_gemv_dequant_1280x1280`): ~1080 µs → ~205 µs (≈5.3×;
///   3.0 → ~16 GFLOP/s),
/// * criterion vs round2-pre (f32): `decoder_token_step_large` −56.5%,
///   `logits_gemv_large` −19.3%, `decoder_token_step_tiny` −7.9%,
/// * interleaved e2e wall A/B (same binary, env toggled, jfk, ≥8 pairs,
///   min/p25): **large-v3-turbo min −11.5% / p25 −8.6%** (was +12% in pass 2),
///   **tiny.en min −0.3% / p25 −1.6%** (was +27%) — i.e. a clear win on large
///   and within noise (slightly better) on tiny.
///
/// The conformance gate passed again: f16-ON transcripts are
/// `ON == OFF == golden` (text and every segment timestamp identical) on BOTH
/// tiny.en and large-v3-turbo, and the native-engine lib tests are 195/195
/// green with the switch ON and OFF. The decision gate (e2e min improves on
/// both, or improves on one and within noise on the other) is satisfied, so the
/// lever is now shipped **default ON**. The env var remains as an opt-OUT kill
/// switch.
///
/// Read once via [`OnceLock`]; costs one atomic load. Accepts the usual
/// truthy spellings (`1`/`true`/`on`/`yes`) and falsy ones (`0`/`false`/`off`/
/// `no`); an unset or unrecognized value falls back to the compiled-in default.
pub(crate) fn f16_compute_enabled() -> bool {
    /// Compiled-in default when the env var is unset/unrecognized.
    ///
    /// **ON**: pass 3 vectorized the dequant (bulk SIMD `convert_to_f32_slice`
    /// then an 8-lane `dot8`), flipping the pass-2 e2e regression to a win
    /// (large-v3-turbo e2e min −11.5%, tiny.en within noise) while the
    /// conformance gate stayed clean (ON==OFF==golden, 195/195 lib tests green
    /// both states). See the function docs for the full evidence.
    const DEFAULT_ON: bool = true;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(
        || match std::env::var("FRANKEN_WHISPER_NATIVE_F16_COMPUTE") {
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "on" | "yes" => true,
                "0" | "false" | "off" | "no" => false,
                _ => DEFAULT_ON,
            },
            Err(_) => DEFAULT_ON,
        },
    )
}

/// Whether to run the tied-output (logits) projection through the int8/Q8
/// weight-quantized GEMV ([`nn::gemv_i8`]) instead of the f16 fused GEMV.
///
/// The logits stream is the model's largest tensor (`[n_vocab, n_state]`, 132 MB
/// f16 for large-v3-turbo) and is DRAM-bandwidth-bound in decode; int8 halves the
/// bytes (measured 1.86× single-thread, 3.5× tight-loop vs f16). It is a
/// NUMERICS-AFFECTING approximation (int8 ≈ 256 levels vs f16 ≈ 1000s), but the
/// logits are the FINAL projection — argmax-robust, and quantizing here leaves the
/// hidden state untouched. Validated transcript-identical to both the f16 path and
/// the whisper-cli golden reference on jfk for tiny.en AND large-v3-turbo across
/// every dispatch path (the whisper.cpp reference itself runs `MATMUL_INT8`), so it
/// is **ON by default**. Set `FRANKEN_WHISPER_INT8_LOGITS=0` to force the exact f16
/// path (bit-level A/B, or a hypothetical regressing input).
pub(crate) fn int8_logits_enabled() -> bool {
    const DEFAULT_ON: bool = true;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FRANKEN_WHISPER_INT8_LOGITS") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        ),
        Err(_) => DEFAULT_ON,
    })
}

/// Whether to run the decoder MLP up-projection (fc1 `[4·n_state, n_state]`,
/// `mlp_0`, feeding GELU) through the int8/Q8 GEMV ([`nn::gemv_i8`]) on the
/// per-token decode path (`tq == 1`). The down-projection (`mlp_2`) stays f16.
///
/// The MLP is ~28% of decode. In real per-token decode the working set (4×26 MB
/// MLP + 132 MB logits ≈ 250 MB) ≫ 128 MB L3, so the MLP weights are DRAM-resident
/// and int8 (half the bytes) is MEASURED 1.65–1.76× per linear (cache-cold probe).
///
/// **fc1-only** is the safe subset: `mlp_2` writes DIRECTLY into the residual
/// stream, so its per-token int8 rounding compounds across layers/tokens and was
/// the source of a turbo trailing-artifact under the both-quant variant (6c4b53d).
/// `mlp_0`'s error is instead absorbed by GELU saturation before it reaches the
/// residual, so quantizing ONLY it is transcript byte-exact vs the f16 baseline on
/// both tiny.en and large-v3-turbo (jfk golden) — hence **ON by default**. It still
/// captures ~1.20× on the MLP-GEMV span (~4.6% e2e decode). whisper.cpp's Q8_0
/// models quantize the whole MLP, so int8 here is a proven-safe class of target.
/// Disable with `FRANKEN_WHISPER_INT8_MLP=0`. Prefill (`tq > 1`) keeps the f16 path.
pub(crate) fn int8_mlp_enabled() -> bool {
    const DEFAULT_ON: bool = true;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FRANKEN_WHISPER_INT8_MLP") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        ),
        Err(_) => DEFAULT_ON,
    })
}

/// Whether to run the ENCODER linear GEMMs (attn q/k/v/out + mlp fc1/fc2) through
/// the maddubs 7-bit-weight int8 path ([`nn::matmul_bias_i7`]) instead of f32
/// sgemm. **DEFAULT OFF = f32 = byte-identical.** When on, each linear weight is
/// quantized to i7 ONCE at load; the per-window activation is quantized to u8 and
/// the GEMM runs `_mm256_maddubs_epi16` (measured 1.56-1.58x f32 on the MLP GEMMs,
/// integer-EXACT/non-saturating for i7, docs/NEGATIVE_EVIDENCE d8b8df6). NON-byte-
/// exact vs f32 sgemm (int8 quantization) -> owner-gated on a transcript A/B, hence
/// default off. Env: `FRANKEN_WHISPER_ENC_INT8=1`.
///
/// **e2e REALITY CHECK (turbo, 2026-07-13, `NEGATIVE_EVIDENCE` tick 13g): the
/// 1.56-1.58x is a per-GEMM MICROBENCH; it does NOT translate to e2e.** Measured on
/// real large-v3-turbo (jfk, 6 reps, alternating): `encoder_window` +3-4%,
/// `backend_run` +2.2% only — because attn_sdpa (42.9% of encoder) is external and
/// UNCHANGED, `attn_out` is already i8i32 by default, and the external f32 sgemm is
/// already fast (int8 barely wins on CPU without VNNI on this Zen3 box).
/// And it's non-byte-exact on real speech (track01: 2 word-diffs). ~2% e2e for a WER
/// risk is not worth a default flip — keep OFF. Do not cite "1.5x" as an e2e figure.
pub(crate) fn enc_int8_enabled() -> bool {
    const DEFAULT_ON: bool = false;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FRANKEN_WHISPER_ENC_INT8") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        ),
        Err(_) => DEFAULT_ON,
    })
}

/// Whether to FREE the f32 encoder weight copies (`attn_{q,k,v,out}_w`,
/// `mlp_{fc,proj}_w`) after they have been quantized to i7/i8 at load. In the
/// default full-int8 config every encoder linear runs through its i7/i8 quant
/// ([`encoder::enc_linear`] reads the f32 `w_t` **only** when the i7 field is
/// `None`), so once a linear is quantized its f32 copy is dead weight — ~2.5 GB
/// of steady-state RSS on turbo (78 MiB/layer × 32). Freeing it is **byte-exact**
/// (the forward never reads a freed weight) and load-time frees the f32 quant
/// source too, so the ~2.5 GB is never first-touched: MEASURED on turbo/jfk
/// −46% peak RSS (5.29→2.83 GB) AND **−12% single-shot wall** (2.77→2.43 s) via
/// −610 k page faults (1.77→1.16 M) — the retained f32 was pure page-fault tax.
/// `FW_ENC_FREE_F32`, **default ON** (kill-switch: `FW_ENC_FREE_F32=0` restores the
/// retained-f32 behavior byte-for-byte). Freeing is per-linear conditional on that
/// linear having an i7/i8 copy (an un-quantized linear keeps its f32 regardless),
/// and is NOT applied when the weight-roundtrip harness is active (it rewrites the
/// f32 in place) nor on macOS (the GPU encode stack reads the f32) — see the
/// `from_ggml` guard. Transcript verified byte-identical off-vs-on (turbo/jfk).
#[cfg(not(target_os = "macos"))]
pub(crate) fn enc_free_f32() -> bool {
    const DEFAULT_ON: bool = true;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FW_ENC_FREE_F32") {
        // Kill-switch: explicit 0/off/false/no restores the retained-f32 behavior.
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        Err(_) => DEFAULT_ON,
    })
}

/// Cap on how many model-weight tensors load+quantize concurrently across the
/// whole (encoder ∥ decoder) weight build. Applied as a scoped rayon pool around
/// the `rayon::join` in [`decode::LoadedModel::from_ggml`], so both builds' layer
/// `into_par_iter`s share it.
///
/// **Default = `host_parallelism()∧32`** (the all-core AVX freq-throttle knee —
/// the same optimum the encoder *compute* already uses). Uncapped, the load fans
/// across the full ambient pool (64-way on the 64-core box) AND each weight's
/// `thread::scope` workers pile on top → oversubscription + throttle. Capping to
/// 32 measured `model_weights` ~441 ms → ~394 ms (−11%, ~2% e2e single-shot),
/// **byte-exact** (thread count never changes the quantized output); it also cut
/// load-time voluntary context switches (122 k → far fewer). On ≤32-core hosts
/// `host∧32 == host`, so nothing changes there.
///
/// `FW_LOAD_WORKERS=<N>` overrides: a smaller `N` further bounds the transient
/// per-tensor buffers (most useful with [`ggml`]'s `FW_STREAM_LOAD` (bd-A14),
/// where each in-flight tensor is an owned pread buffer incl. the ~133 MB token
/// embedding — lower peak RSS, traded against a longer load). `FW_LOAD_WORKERS=0`
/// (or non-numeric) = **uncapped** kill-switch (restores the old ambient pool).
pub(crate) fn load_worker_cap() -> Option<usize> {
    static CAP: OnceLock<Option<usize>> = OnceLock::new();
    *CAP.get_or_init(|| match std::env::var("FW_LOAD_WORKERS") {
        // Explicit override: <N> caps to N; 0 / non-numeric = uncapped kill-switch.
        Ok(v) => v.trim().parse::<usize>().ok().filter(|&n| n >= 1),
        // Default: cap at the ~32-thread throttle knee (byte-exact; single scoped
        // pool covers both enc and dec builds via the join in from_ggml).
        Err(_) => Some(host_parallelism().min(32)),
    })
}

/// Whether to run ONLY the ENCODER MLP up-projection (`mlp.0`/fc1, feeding GELU) through the
/// maddubs i7 int8 GEMM, leaving attention (q/k/v/out) and fc2/`mlp.2` on f32 sgemm.
/// `FW_ENC_INT8_FC1`, **default OFF = f32 = byte-identical**.
///
/// This applies the PROVEN decode `mlp_0`/fc1-only recipe ([`int8_mlp_enabled`],
/// [[project_int8_mlp_fc1_default_on]]) to the encoder: only fc1 feeds GELU, whose saturation
/// absorbs the weight-quant error before it reaches the residual, so decode fc1-only int8 is
/// transcript byte-exact while both-quant / attention-quant is not. The prior encoder-int8 digs
/// quantized the WHOLE encoder (incl. attention) and hit intrinsic proper-noun errors from the
/// cross-attention alignment ([[project_turbo_encoder_dominates]]); fc1-only keeps attention f32
/// so that alignment is preserved. Open question this tests: whether GELU-absorption survives the
/// encoder's 32 stacked layers (vs the decoder's 4). Load-time i7 quantize only; the maddubs GEMV
/// kernel is unchanged. NON-byte-exact when ON (int8 quantization) ⇒ owner-gated on a transcript
/// A/B; hence default off. Mutually exclusive with the full [`enc_int8_enabled`] (that wins).
pub(crate) fn enc_int8_fc1_only() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("FW_ENC_INT8_FC1")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1" | "on" | "true" | "yes")
        )
    })
}

/// Whether to int8 the encoder attention INPUT projections (q/k/v) IN ADDITION to
/// `mlp.0`/fc1, keeping `attn.out` + `mlp.2`/fc2 on f32. `FW_ENC_INT8_ATTN_IN`,
/// **default OFF = byte-identical**. Q/K/V feed the attention SCORES → softmax
/// (error-robust; the DECODE runs qkv int8 default-on), while out/fc2 feed the
/// RESIDUAL (not GELU/softmax-absorbed — the [`enc_int8_fc1_only`] culprit class).
/// A faithfulness probe: does the prior whole-attention int8 proper-noun failure
/// ([[project_turbo_encoder_dominates]]) come from the in-projections or from the
/// residual-feeding out/fc2? NON-byte-exact when ON ⇒ owner-gated. Takes precedence
/// over [`enc_int8_fc1_only`]; subordinate to full [`enc_int8_enabled`].
pub(crate) fn enc_int8_attn_in() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("FW_ENC_INT8_ATTN_IN")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1" | "on" | "true" | "yes")
        )
    })
}

/// FULL quality-safe encoder int8: q/k/v/fc1/fc2 through the fast i7 maddubs
/// GEMM (each individually proven proper-noun-safe) AND the residual-feeding
/// `attn.out` through a **full i8** (per-output-channel amax/127, 8 bits) i32-
/// accumulate GEMM ([`encoder::matmul_bias_i8`]) instead of the i7 maddubs.
/// `FW_ENC_ATTN_OUT_I8I32`, **default OFF = f32 = byte-identical**.
///
/// Prior digs proved full-encoder int8 mangles proper nouns ONLY through the
/// residual-feeding `attn.out` ("Frank at"; [[project_turbo_encoder_dominates]]),
/// and attributed it to "the maddubs arithmetic." But franken's maddubs is
/// already i32-accumulate and non-saturating (i7 weight chosen so
/// `_mm256_maddubs_epi16` cannot i16-saturate) — so the untested variable is the
/// **1 bit of weight precision** (i7 vs i8). MEASURED (track01): the full config
/// PRESERVES `Franco`/`Franken`/`FrankenSearch` (45 benign word-diffs, NO
/// "Frank at"), where i7-maddubs `attn.out` mangles them (60 diffs → "Frank at").
/// This lifts the quality-fatal blocker on full-encoder int8: all 6 GEMMs are now
/// int8 ⇒ 0 f32 GEMMs/layer ⇒ the fast non-monotonic-mix state (the f32-mix
/// pessimum only bites at exactly 1 f32 GEMM) ⇒ **1.47× encoder_window** (jfk×3
/// window-1, min-of-5) vs f32, beating the prior quality-safe max `attn_in` (1.23×)
/// by ~20% at equal proper-noun fidelity. Owner-gated (non-byte-exact); default off.
pub(crate) const ENCODER_INT8_CALIBRATION_ID: &str = "encoder-int8-calibration-2026-07-10";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EncoderInt8PolicyAction {
    F32Encoder,
    QualitySafeInt8Encoder,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EncoderInt8PolicyDecision {
    pub action: EncoderInt8PolicyAction,
    pub reason: &'static str,
    pub calibration_id: &'static str,
    pub corpus_wer_delta_budget: f64,
    pub quant_rel_rmse_budget: f64,
}

impl EncoderInt8PolicyDecision {
    #[must_use]
    pub fn enabled(self) -> bool {
        self.action == EncoderInt8PolicyAction::QualitySafeInt8Encoder
    }
}

/// Expected-loss default policy for the quality-safe encoder int8 arm.
///
/// State: model hparams/family, compiled CPU feature class, calibration corpus
/// id, per-layer quantization-error budget, fixture WER/adversarial sentinels,
/// and the operator override. Actions: f32 encoder or quality-safe int8. Loss:
/// false-accepting int8 with WER/proper-noun drift is high loss; falling back to
/// f32 only pays speed. Posterior/calibration artifact:
/// [`ENCODER_INT8_CALIBRATION_ID`] in the performance ledger. Deterministic
/// fallback: f32 for unknown hparams or non-AVX2 builds, and f32 when the
/// kill-switch is set.
#[must_use]
pub(crate) fn encoder_int8_policy_decision(hparams: &WhisperHParams) -> EncoderInt8PolicyDecision {
    const WER_DELTA_BUDGET: f64 = 0.0;
    const QUANT_REL_RMSE_BUDGET: f64 = 0.09;

    if !encoder_i8_kernel_supported() {
        return EncoderInt8PolicyDecision {
            action: EncoderInt8PolicyAction::F32Encoder,
            reason: "cpu_feature_fallback",
            calibration_id: ENCODER_INT8_CALIBRATION_ID,
            corpus_wer_delta_budget: WER_DELTA_BUDGET,
            quant_rel_rmse_budget: QUANT_REL_RMSE_BUDGET,
        };
    }

    if !calibrated_encoder_int8_model(hparams) {
        return EncoderInt8PolicyDecision {
            action: EncoderInt8PolicyAction::F32Encoder,
            reason: "uncalibrated_model_fallback",
            calibration_id: ENCODER_INT8_CALIBRATION_ID,
            corpus_wer_delta_budget: WER_DELTA_BUDGET,
            quant_rel_rmse_budget: QUANT_REL_RMSE_BUDGET,
        };
    }

    EncoderInt8PolicyDecision {
        action: EncoderInt8PolicyAction::QualitySafeInt8Encoder,
        reason: "calibrated_model_budget_pass",
        calibration_id: ENCODER_INT8_CALIBRATION_ID,
        corpus_wer_delta_budget: WER_DELTA_BUDGET,
        quant_rel_rmse_budget: QUANT_REL_RMSE_BUDGET,
    }
}

/// Whether `hparams` is the `large-v3-turbo` checkpoint (as opposed to `tiny.en` or an
/// unknown model). Used both for the encoder int8 calibration set and for the poly-softmax
/// enablement, which is proven WER-neutral on this model but NOT on `tiny.en`.
#[must_use]
pub(crate) fn is_large_v3_turbo(hparams: &WhisperHParams) -> bool {
    hparams.n_vocab == 51_866
        && hparams.n_audio_ctx == 1_500
        && hparams.n_audio_state == 1_280
        && hparams.n_audio_head == 20
        && hparams.n_audio_layer == 32
        && hparams.n_text_state == 1_280
        && hparams.n_text_layer == 4
        && hparams.n_mels == 128
        && hparams.ftype == 1
}

#[must_use]
fn calibrated_encoder_int8_model(hparams: &WhisperHParams) -> bool {
    let tiny_en = hparams.n_vocab == 51_864
        && hparams.n_audio_ctx == 1_500
        && hparams.n_audio_state == 384
        && hparams.n_audio_head == 6
        && hparams.n_audio_layer == 4
        && hparams.n_text_state == 384
        && hparams.n_text_layer == 4
        && hparams.n_mels == 80
        && hparams.ftype == 1;
    tiny_en || is_large_v3_turbo(hparams)
}

/// Decide whether `ft_kernel_cpu`'s 8-lane poly softmax is admitted for a model.
///
/// **`large-v3-turbo` only.** Evidence (bd-bcm7, `docs/PROPOSAL_ft_sdpa_poly_exp_default_on.md`):
/// transcript **byte-identical** on jfk ×1/×3/×8, WER vs whisper.cpp **Δ 0.000**, e2e **1.0722×**
/// (cv 0.8%, 5/5 paired). `tiny.en` is **uncertified** (regressed on track01) and stays OFF.
/// The decision is stored on the loaded encoder and applied for the complete CPU
/// encoder forward. It must not be installed here as process-global state: two
/// different models can load or run concurrently in one embedding process.
///
/// Controls: `FW_SDPA_POLY_EXP=0` kills it even on turbo; `FT_SDPA_POLY_EXP=1` forces it on for any
/// model (operator override, e.g. for a certified fine-tune).
pub(crate) fn sdpa_poly_exp_for(hparams: &WhisperHParams) -> bool {
    let killed = std::env::var("FW_SDPA_POLY_EXP").as_deref() == Ok("0");
    let forced = std::env::var("FT_SDPA_POLY_EXP").as_deref() == Ok("1");
    forced || (is_large_v3_turbo(hparams) && !killed)
}

#[must_use]
fn encoder_i8_kernel_supported() -> bool {
    cfg!(all(target_arch = "x86_64", target_feature = "avx2"))
}

fn enc_attn_out_i8i32_override() -> Option<bool> {
    static OVERRIDE: OnceLock<Option<bool>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| match std::env::var("FW_ENC_ATTN_OUT_I8I32") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "on" | "true" | "yes" => Some(true),
            "0" | "off" | "false" | "no" => Some(false),
            _ => None,
        },
        Err(_) => None,
    })
}

pub(crate) fn enc_attn_out_i8i32_for(hparams: &WhisperHParams) -> bool {
    encoder_int8_effective_policy_decision(hparams).enabled()
}

#[must_use]
pub(crate) fn encoder_int8_effective_policy_decision(
    hparams: &WhisperHParams,
) -> EncoderInt8PolicyDecision {
    const WER_DELTA_BUDGET: f64 = 0.0;
    const QUANT_REL_RMSE_BUDGET: f64 = 0.09;
    match enc_attn_out_i8i32_override() {
        Some(true) => EncoderInt8PolicyDecision {
            action: EncoderInt8PolicyAction::QualitySafeInt8Encoder,
            reason: "operator_forced_quality_safe_int8",
            calibration_id: ENCODER_INT8_CALIBRATION_ID,
            corpus_wer_delta_budget: WER_DELTA_BUDGET,
            quant_rel_rmse_budget: QUANT_REL_RMSE_BUDGET,
        },
        Some(false) => EncoderInt8PolicyDecision {
            action: EncoderInt8PolicyAction::F32Encoder,
            reason: "operator_f32_kill_switch",
            calibration_id: ENCODER_INT8_CALIBRATION_ID,
            corpus_wer_delta_budget: WER_DELTA_BUDGET,
            quant_rel_rmse_budget: QUANT_REL_RMSE_BUDGET,
        },
        None => encoder_int8_policy_decision(hparams),
    }
}

/// When the int8 encoder attention-input path is active (q/k/v are i7), fuse the
/// SDPA gather into the maddubs GEMM: q/k/v are written DIRECTLY in head-major
/// layout ([`nn::attention_from_i7_qkv`]), skipping the separate
/// `sdpa_gather_head_major` transpose (a DRAM-latency floor, ledger 2026-07-04).
/// `FW_ENC_QKV_FUSED`, **default ON** (kill-switch `=0`). BYTE-IDENTICAL to the
/// two-step path (same maddubs dots + same head-major permutation ⇒ identical SDPA
/// inputs ⇒ identical transcript, verified on track01 + jfk×3) — a pure
/// speed/scheduling change WITHIN the already-gated int8 path, so default-on adds
/// zero risk to the default (int8-off) engine: the fused path only fires when
/// q/k/v are i7 (i.e. an encoder int8 gate is also on). MEASURED 1.082× on the int8
/// encoder_window (133 ms/window saved by dropping `sdpa_gather_head_major`),
/// lifting the full quality-safe int8 config to ~1.67× vs f32.
pub(crate) fn enc_qkv_fused() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("FW_ENC_QKV_FUSED")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("0" | "off" | "false" | "no")
        )
    })
}

/// Fold the encoder MLP GELU into fc2's int8 activation-quantize
/// (`nn::matmul_bias_i7_gelu`). `FW_ENC_GELU_FUSED`, **default ON** (kill-switch
/// `=0`). BYTE-IDENTICAL to the separate `nn::gelu` + `matmul_bias_i7` (same
/// GGML_GELU_FP16 table+clamp per element, same per-row quant), a pure
/// memory-traffic change WITHIN the already-gated int8 MLP: the classic form
/// writes a full `[1500, 5120]` GELU'd buffer and re-reads it to quantize; the
/// fused form GELUs each row into a per-worker L1 scratch during the quant,
/// eliminating that ~30 MiB (partly DRAM-resident) fc1-output round-trip. Only
/// fires when `mlp_proj` is i7 (an encoder int8 gate is on), so default-on adds
/// zero risk to the default (int8-off) engine. Kill-switch restores the separate
/// `gelu` + `enc_linear` path.
pub(crate) fn enc_gelu_fused() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("FW_ENC_GELU_FUSED")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("0" | "off" | "false" | "no")
        )
    })
}

/// Error-feedback weight quant for the DECODER per-row int8 weights (`nn::quantize_f16_to_i8`,
/// used by the default-ON `gemv_i8` path: logits, mlp_0, qkv, cross_q, self_out, cross_out).
/// `FW_DEC_EF`, default OFF = byte-identical. The decoder int8 is DEFAULT-ON and diverges from
/// the f32 decoder by a MEASURED ~32 word-diffs on track01 (the f32 decoder ≈ whisper.cpp's f16
/// reference), so this gap is a faithfulness cost paid for int8 speed. Error-feedback (the same
/// scheme validated strictly ≥ plain int8 for the ENCODER weights, [`enc_ef_quant`]) carries each
/// weight's rounding residual forward along the contraction dim, reducing accumulated dot bias —
/// STABLE here because the weight is a STATIC operand (the encoder lesson: EF only on static
/// operands; EF-activations was dynamic and regressed). Load-time only ⇒ ZERO runtime cost; the
/// int8 GEMV kernel is unchanged. If it reduces the decoder int8 gap it improves the DEFAULT
/// path's faithfulness for free (owner-gated to flip default since it changes the transcript).
pub(crate) fn dec_ef_quant() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("FW_DEC_EF").ok().as_deref().map(str::trim),
            Some("1" | "on" | "true" | "yes")
        )
    })
}

/// Feasibility harness (default OFF): `FW_ENC_WEIGHT_ROUNDTRIP` replaces every f32 encoder
/// GEMM weight with its i7 quantize→dequantize roundtrip at load, so the EXISTING f32 encoder
/// measures the WEIGHT-quant-granularity effect on the transcript (does block-wise recover the
/// int8 encoder's proper-noun errors?) without the block-wise maddubs kernel. Returns
/// `Some(None)` for `row` (per-output-column scale = current int8 granularity), `Some(Some(n))`
/// for a positive `n` (block size along the contraction dim, e.g. 32), `None` when unset/off.
pub(crate) fn enc_weight_roundtrip() -> Option<Option<usize>> {
    static V: OnceLock<Option<Option<usize>>> = OnceLock::new();
    *V.get_or_init(|| match std::env::var("FW_ENC_WEIGHT_ROUNDTRIP") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" => None,
            "row" | "col" | "percol" | "perrow" => Some(None),
            other => other.parse::<usize>().ok().filter(|&n| n > 0).map(Some),
        },
        Err(_) => None,
    })
}

/// Feasibility harness (default OFF): `FW_ENC_ACT_ROUNDTRIP` roundtrips every f32 encoder GEMM's
/// ACTIVATION input through the int8 path's u8 quant (symmetric `amax/127`) before the f32 matmul,
/// isolating the ACTIVATION-quant effect on the transcript (does block-wise activation granularity
/// recover the int8 encoder's proper-noun errors, or is it the u8 8-bit precision itself?).
/// `Some(None)` = `row` (per-row scale = current int8), `Some(Some(n))` = n-channel block, `None` = off.
pub(crate) fn enc_act_roundtrip() -> Option<Option<usize>> {
    static V: OnceLock<Option<Option<usize>>> = OnceLock::new();
    *V.get_or_init(|| match std::env::var("FW_ENC_ACT_ROUNDTRIP") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "off" => None,
            "row" | "col" | "percol" | "perrow" => Some(None),
            other => other.parse::<usize>().ok().filter(|&n| n > 0).map(Some),
        },
        Err(_) => None,
    })
}

/// Error-feedback WEIGHT quantization for the int8 encoder — now the DEFAULT within the (gated,
/// default-off) int8 path, kill-switch `FW_ENC_EF_QUANT=0`. Switches `nn::quantize_mat_to_i7` from
/// independent round-to-nearest to ERROR-FEEDBACK (error-diffusion / sigma-delta) rounding along the
/// contraction dim — each weight's rounding residual is carried into the next element, so the
/// per-output-column DOT `Σ q_i·a_i` has less accumulated quantization bias. Only affects the
/// load-time i7 weight table; the maddubs kernel (i7 format, per-col scale, colsum) is unchanged, so
/// this is ZERO e2e speed cost (the ~1.5× int8 encoder speedup is intact). MEASURED strictly ≥ plain
/// int8 on 4 clips: jfk/jfk_x3 BYTE-IDENTICAL to f32 golden, track01 44→41 diffs (+recovers
/// "Frank at"→"FrankenSearch"), sjobs_16k (13-min) 179→125 = 30% fewer errors (see
/// docs/NEGATIVE_EVIDENCE.md). Default-ON-within-int8 because it is validated strictly better; the
/// f32 default path (enc_int8 off) is UNAFFECTED. `FW_ENC_EF_QUANT=0` restores plain round-to-nearest.
pub(crate) fn enc_ef_quant() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FW_ENC_EF_QUANT") {
        // Kill-switch: explicit 0/off/false/no restores plain round-to-nearest int8.
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        Err(_) => true, // default: EF on (validated strictly ≥ plain int8, zero speed cost)
    })
}

/// Whether to run the decoder **attention** input projections (fused self `qkv`
/// and `cross_attn_q`) through the int8/Q8 GEMV on the per-token decode path
/// (`tq == 1`). The output projections (`self_out`, `cross_out`) stay f16.
///
/// Same DRAM-resident-bandwidth rationale as [`int8_mlp_enabled`]: in real decode
/// the 4-layer weight set ≫ L3, so these projection weights are streamed from DRAM
/// each token and int8 halves the bytes. Safety mirrors the fc1-only MLP finding —
/// the input projections feed attention SCORES → softmax (error-robust, like
/// `mlp_0`→GELU), whereas the output projections write the residual directly (the
/// `mlp_2` failure mode) and are excluded. The fused `qkv` also carries V (which
/// reaches the attention output), but softmax-weighted averaging bounds that error;
/// the byte-exact-vs-f16 golden gate is what validates the default flip. Disable
/// with `FRANKEN_WHISPER_INT8_ATTN=0`. Prefill (`tq > 1`) keeps the f16 path.
pub(crate) fn int8_attn_enabled() -> bool {
    const DEFAULT_ON: bool = true;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FRANKEN_WHISPER_INT8_ATTN") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        ),
        Err(_) => DEFAULT_ON,
    })
}

/// Route the per-window cross-K/V PROJECTIONS (encoder_out @ Wk/Wv, tq=1500)
/// through dequant-once f32 tiled sgemm instead of the f16 batched GEMV
/// ([`gemv_f16_batch`]). MEASURED 2.25× faster on the turbo cross shape
/// (233.96 → 103.95 ms for the 8 GEMMs, `cross_f16path_probe`) — the same reason
/// the ENCODER dequants-once to f32. NOT bit-exact (different accumulation order,
/// max|Δ| ~6.9e-6), so it is gated and was proven transcript-neutral before
/// defaulting on: jfk+tiny.en (the `ln.json` byte-exact gate asset) AND jfk×6
/// large-v3-turbo both produce a BYTE-IDENTICAL transcript gate-ON vs gate-OFF
/// (the 6.9e-6 divergence is fully absorbed). `=0` restores the f16 GEMV path.
pub(crate) fn cross_proj_f32_enabled() -> bool {
    const DEFAULT_ON: bool = true;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FRANKEN_WHISPER_CROSS_PROJ_F32") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => true,
            "0" | "false" | "off" | "no" => false,
            _ => DEFAULT_ON,
        },
        Err(_) => DEFAULT_ON,
    })
}

/// Route the prefill / multi-token (`tq > 1`) per-row-int8 projections through
/// the int8 BATCHED GEMV ([`nn::gemv_i8_batch`]) instead of the f16 batched GEMV
/// ([`nn::gemv_f16_batch`]). Reads HALF the weight bytes of the f16 path and is
/// BIT-IDENTICAL to running the prompt batch as `tq` separate per-token
/// [`nn::gemv_i8`] calls (same per-row activation quant + per-row weight scale).
/// Applies only to linears whose `tq == 1` path is already `gemv_i8` (qkv /
/// cross_q / self_out / cross_out, and mlp_0 when int4 is off) — i.e. those with
/// a `w_i8` copy and neither a block nor int4 copy — so prefill uses the SAME
/// quantization those linears already use per token. It also raises the
/// draft-decoding amortization ceiling (`examples/draft_amortization_probe.rs`).
/// Changes prefill numerics f16→int8, so gated + transcript-checked before any
/// default flip. Proven transcript BYTE-IDENTICAL gate-on vs gate-off (turbo
/// jfk×6, timestamp mode) AND bit-identical to per-token `gemv_i8` (0/15360
/// entries differ, `examples/gemv_i8_batch_probe.rs`), so defaulted ON. MEASURED
/// 3–12% faster on the cold-weight `tq>1` path (`examples/draft_amortization_probe.rs`,
/// best-of-60: K=2 +12%, K=4 +2.9%, K=8 +5.0%). `FRANKEN_WHISPER_INT8_BATCH=0` restores f16.
pub(crate) fn i8_batch_enabled() -> bool {
    const DEFAULT_ON: bool = true;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FRANKEN_WHISPER_INT8_BATCH") {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => true,
            "0" | "false" | "off" | "no" => false,
            _ => DEFAULT_ON,
        },
        Err(_) => DEFAULT_ON,
    })
}

/// DEAD PROBE (default off) — kept as a gated scaffold; do NOT re-attempt. int4
/// (block-wise, 4-bit weight × f32 activation) for `mlp_0`/fc1. fc1 feeds GELU,
/// whose saturation absorbs int8 weight error to byte-exactness (fc1-only int8,
/// default-on); 4-bit was the natural next byte-cut. Measured DEAD on BOTH axes:
///
/// 1. **NOT byte-exact on realistic audio.** `8ca4378` reported "byte-exact (GELU
///    absorbs 4-bit)" — but that was jfk single-window ONLY. Re-measured 2026-07-13
///    on track01 (124 s / 5-window real speech, tiny.en, no_ts): the transcript
///    DRIFTS materially vs int4-off (both deterministic; A/A null-control clean) —
///    e.g. "ranking this stuff" → "ranking and stuff like the video ranker". The
///    4-bit error escapes GELU absorption on ambiguous speech and compounds across
///    windows via the carried prompt (jfk-identical ≠ corpus-neutral).
/// 2. **Perf REGRESSION, not just the `60eb294` microbench wash.** Re-measured e2e
///    on that same decode-dominated clip: `decode_loop` +6% SLOWER int4-on vs off
///    (AVX2 nibble-unpack cost > the halved-bandwidth benefit — decode is dispatch/
///    latency-bound, not bandwidth-bound). See NEGATIVE_EVIDENCE 2026-07-13.
///
/// Stays default-off permanently. `FRANKEN_WHISPER_INT4_MLP0=1` to reproduce.
pub(crate) fn int4_mlp0_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("FRANKEN_WHISPER_INT4_MLP0")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "on" | "yes"
        )
    })
}

/// Whether to run `mlp_2` (fc2, the MLP down-projection) through the MIXED
/// BLOCK-WISE int8-weight × f32-activation GEMV ([`nn::gemv_i8w_f32a_blocked`]) on
/// the per-token decode path. fc2's weight is the bandwidth-bound operand (13 MB
/// f16 → 6.5 MB int8 per token), so quantizing ONLY the weight captures that win;
/// the activation stays f32. A per-ROW int8 weight scale was too coarse and broke
/// turbo (the trailing artifact, like full-int8 `mlp_2` at 6c4b53d), but per-BLOCK
/// scales (32-elt, whisper.cpp-Q8_0-class) are transcript BYTE-EXACT vs f16 on both
/// tiny.en and large-v3-turbo — hence **ON by default**. ~1.25× on the MLP-GEMV
/// span (~7% e2e decode). Disable with `FRANKEN_WHISPER_INT8_MLP_FC2=0`.
pub(crate) fn int8_mlp_fc2_enabled() -> bool {
    const DEFAULT_ON: bool = true;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FRANKEN_WHISPER_INT8_MLP_FC2") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        ),
        Err(_) => DEFAULT_ON,
    })
}

/// Whether to int8-quantize the attention **output** projections (`self_out`,
/// `cross_out`) on the per-token decode path. These write DIRECTLY into the
/// residual stream — the `mlp_2` failure mode — so they were expected to break
/// exactness. Empirically they do NOT: transcript is byte-exact vs f16 on both
/// tiny.en and large-v3-turbo (jfk), INCLUDING the exact turbo clip where the
/// both-quant MLP produced a trailing artifact. The difference is magnitude — the
/// attention output is a softmax-weighted average of value vectors (bounded, 1280-d),
/// so its per-token int8 rounding is far smaller than `mlp_2`'s 5120-d GELU-hidden
/// input and stays under the argmax margin. Hence **ON by default**; ~1.15× on the
/// two output-proj spans (~2.3% e2e decode). Disable with
/// `FRANKEN_WHISPER_INT8_ATTN_OUT=0`. Prefill (`tq > 1`) keeps the f16 path.
pub(crate) fn int8_attn_out_enabled() -> bool {
    const DEFAULT_ON: bool = true;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("FRANKEN_WHISPER_INT8_ATTN_OUT") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        ),
        Err(_) => DEFAULT_ON,
    })
}

/// Emit one measurement-only span line (see [`perf_spans_enabled`]).
pub(crate) fn perf_span(span: &str, ms: f64, extra: &str) {
    // wasm32 (bd-m2jm) + iOS (bd-n6wl): spans double as the engine's progress
    // heartbeat — the host registers a hook (`plat::set_span_hook`) and the
    // page/app derives its window-counted progress bar and remaining-time
    // estimate from `encoder_window` events. Desktop builds compile this away
    // entirely.
    #[cfg(any(target_arch = "wasm32", target_os = "ios"))]
    crate::native_engine::plat::emit_span(span, ms);
    if perf_spans_enabled() {
        static T0: std::sync::OnceLock<crate::native_engine::plat::Instant> =
            std::sync::OnceLock::new();
        let at_ms = T0
            .get_or_init(crate::native_engine::plat::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1e3;
        let comma = if extra.is_empty() { "" } else { "," };
        eprintln!(
            "{{\"event\":\"perf.profile.span_summary\",\"span\":\"{span}\",\"cumulative_ms\":{ms:.2},\"at_ms\":{at_ms:.0}{comma}{extra}}}"
        );
    }
}

/// Locate a test/dev model file by short name (e.g. `"tiny.en"`), checking the
/// shared [`model_search_dirs`] in precedence order. Returns `None` when absent
/// so gated tests can skip rather than fail (see bead bd-4slu).
///
/// This is the historical signature relied on by sibling tests; it now shares
/// the exact search-dir list used by [`resolve_model`].
#[must_use]
pub fn find_model_file(short_name: &str) -> Option<PathBuf> {
    if short_name.eq_ignore_ascii_case("large-v3-turbo") {
        // Availability is intentionally a header-only probe. Authentication
        // remains mandatory in `resolve_model` immediately before execution;
        // hashing 1.6 GB here made Auto routing hash the same package once per
        // native backend family before the authenticated execution lookup. The
        // canonical default must not fall through to an arbitrary same-named
        // file in a legacy/test search directory when the release package is
        // absent or malformed.
        let directory = crate::model_distribution::whisper_cache_dir().ok()?;
        let candidate = directory.join(crate::model_distribution::WHISPER_WEIGHTS_FILENAME);
        return (candidate.is_file() && header_ftype_ok(&candidate))
            .then(|| candidate.canonicalize().ok())
            .flatten();
    }
    let file_name = model_file_name(short_name);
    model_search_dirs()
        .into_iter()
        .map(|dir| dir.join(&file_name))
        .find(|path| path.is_file() && header_ftype_ok(path))
}

// ─────────────────────────────────────────────────────────────────────────
// Model resolution
// ─────────────────────────────────────────────────────────────────────────

/// Resolve a user-supplied model `spec` to a concrete, canonicalized path to a
/// ggml `.bin` file.
///
/// Two forms are accepted:
/// 1. **A filesystem path** (absolute or relative) that already exists — it is
///    canonicalized and returned verbatim. This lets callers point at any
///    `.bin` anywhere on disk.
/// 2. **A short model name** such as `"tiny.en"` or `"base"` — searched as
///    `ggml-{name}.bin` across [`model_search_dirs`] in precedence order. The
///    canonical `"large-v3-turbo"` name and an unspecified/default request are
///    reserved for the authenticated release package installed by `fw pull`.
///
/// # No network access during inference
///
/// This resolver never downloads anything. Model provisioning is performed by
/// the separate, explicit `fw pull` command (normally invoked by the installer),
/// so transcription remains offline and a missing model is a hard, actionable
/// error rather than a silent network operation.
///
/// # Errors
///
/// Returns [`FwError::InvalidRequest`] when the spec resolves to nothing. The
/// message is written for end users: it lists the expected filename and every
/// directory that was searched, so the fix (drop the file in one of them, or
/// set `$FRANKEN_WHISPER_MODEL_DIR`) is obvious. A canonicalization failure on
/// an existing path surfaces as [`FwError::Io`].
pub fn resolve_model(spec: &str) -> FwResult<PathBuf> {
    // Tolerate blank/whitespace specs — a common "dumb error" is `--model ""`
    // from an empty shell variable. Treat those as "no model specified" rather
    // than trying to open a file literally named "" (which fails obscurely).
    let spec = spec.trim();

    // Form 1: an existing path wins, even if it happens to look like a name.
    if !spec.is_empty() {
        let as_path = Path::new(spec);
        if as_path.is_file() {
            if header_ftype_ok(as_path) {
                return Ok(as_path.canonicalize()?);
            }
            return Err(FwError::InvalidRequest(format!(
                "Whisper model `{spec}` does not have a supported dense ggml f16/f32 header"
            )));
        }
    }

    // This canonical name is the release trust boundary, not an alias for an
    // arbitrary same-named file in a legacy search directory. Operators can
    // still request another compatible model by passing its explicit path.
    if spec.eq_ignore_ascii_case("large-v3-turbo") {
        let package = crate::model_distribution::resolve_cached_whisper()?;
        return package.weights_path.canonicalize().map_err(Into::into);
    }

    let dirs = model_search_dirs();

    // The shipped default is the hash-pinned release package. Never substitute
    // an arbitrary discovered model when that package is missing or corrupt;
    // doing so would make the default depend on unrelated files on the host.
    if spec.is_empty() || spec.eq_ignore_ascii_case("default") {
        let package = crate::model_distribution::resolve_cached_whisper()?;
        return package.weights_path.canonicalize().map_err(Into::into);
    }

    // Form 2: explicit short-name lookup across the shared search dirs.
    resolve_model_in_dirs(spec, &dirs)
}

/// Resolve a short-name `spec` against an explicit, ordered list of search
/// `dirs` (first match wins). Factored out of [`resolve_model`] so the
/// precedence logic is unit-testable without mutating process environment
/// variables (which is `unsafe` and crate-forbidden under edition 2024).
fn resolve_model_in_dirs(spec: &str, dirs: &[PathBuf]) -> FwResult<PathBuf> {
    let file_name = model_file_name(spec);
    for dir in dirs {
        let candidate = dir.join(&file_name);
        if candidate.is_file() && header_ftype_ok(&candidate) {
            return Ok(candidate.canonicalize()?);
        }
    }
    Err(FwError::InvalidRequest(model_resolution_error(
        spec, &file_name, dirs,
    )))
}

/// Build the actionable "model not found" message for [`resolve_model`].
#[must_use]
fn model_resolution_error(spec: &str, file_name: &str, dirs: &[PathBuf]) -> String {
    use std::fmt::Write as _;
    let mut msg = format!(
        "no whisper model found for `{spec}`: it is neither an existing file path \
         nor a short name resolvable to `{file_name}`.\n\
         Searched directories (in order):"
    );
    if dirs.is_empty() {
        msg.push_str(
            "\n  (none — set $FRANKEN_WHISPER_MODEL_DIR or $HOME to enable short-name lookup)",
        );
    } else {
        for dir in dirs {
            let _ = write!(msg, "\n  - {}", dir.join(file_name).display());
        }
    }
    msg.push_str(
        "\nFix: place the model file in one of the above directories, set \
         $FRANKEN_WHISPER_MODEL_DIR to its directory, pass an explicit path, or run \
         `fw pull whisper`. Transcription itself never accesses the network.",
    );
    msg
}

/// The explicitly configured default native model spec, or `None` when the
/// operator has not chosen one.
///
/// Reads `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL` if set and non-empty. When
/// unset this returns `None`. Call [`configured_or_release_model_spec`] when
/// execution should select the authenticated release package by default.
#[must_use]
pub fn default_model_spec() -> Option<String> {
    match std::env::var("FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL") {
        Ok(s) if !s.is_empty() => Some(s),
        _ => None,
    }
}

/// Select the native Whisper model used when a request did not provide one.
///
/// An explicit `$FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL` wins. Otherwise this
/// delegates to [`resolve_model`] with its `default` sentinel, which requires
/// the compiled-size/SHA-256 release package. The returned path is canonical,
/// no network access is performed, and a non-UTF-8 path fails with an
/// actionable error because downstream request model specifications are UTF-8
/// strings.
pub fn configured_or_release_model_spec() -> FwResult<String> {
    if let Some(spec) = default_model_spec() {
        return Ok(spec);
    }

    let path = resolve_model("default")?;
    path.into_os_string().into_string().map_err(|_| {
        FwError::InvalidRequest(
            "the release Whisper model path is not valid UTF-8; pass an explicit \
             UTF-8 --model path"
                .to_owned(),
        )
    })
}

// ─────────────────────────────────────────────────────────────────────────
// Real availability (header sniff, no tensor load)
// ─────────────────────────────────────────────────────────────────────────

/// Number of leading bytes that fully cover the ggml magic plus the eleven
/// `i32` hparams: `4 + 11 * 4 = 48`.
const HEADER_SNIFF_LEN: usize = 48;

/// ggml file magic (`"ggml"` as a little-endian `u32`), duplicated here from
/// the parser so availability sniffing needs no access to private parser
/// internals.
const GGML_MAGIC: u32 = 0x6767_6d6c;

/// Honestly report whether a native model is usable for `spec` **without
/// loading any tensors**.
///
/// This locates the explicit path or short-name candidate (no network) and
/// reads only the first [`HEADER_SNIFF_LEN`] bytes, checking the magic and that
/// `hparams.ftype` is a supported dense type (`0` = f32 or `1` = f16;
/// quantized models are rejected, matching the parser). The execution resolver
/// separately authenticates the pinned release package before loading it. This
/// function returns `false`—never panics or errors—for a miss, I/O failure, or
/// unsupported header.
///
/// This is the function the backend rollout machinery (bead bd-jryr) calls to
/// replace the previously dishonest `always true` availability constant: with
/// no resolvable, well-formed model the native engine reports itself
/// unavailable, so the router stays bridge-only instead of advertising a fake
/// recovery path.
#[must_use]
pub fn native_model_available(spec: &str) -> bool {
    model_probe_path(spec).is_some()
}

/// Resolve only enough of a model specification for route and capability
/// discovery.
///
/// This helper validates the 48-byte dense GGML header but deliberately does
/// not authenticate the complete release artifact. The execution resolver
/// remains responsible for the compiled size and SHA-256 check immediately
/// before loading tensors. Keeping those authority levels separate prevents a
/// health or routing probe from hashing the 1.62 GB default package repeatedly.
#[must_use]
pub(crate) fn model_probe_path(spec: &str) -> Option<PathBuf> {
    let spec = spec.trim();
    let explicit_path = Path::new(spec);
    if explicit_path.is_file() {
        return header_ftype_ok(explicit_path)
            .then(|| explicit_path.canonicalize().ok())
            .flatten();
    }
    let lookup = if spec.is_empty() || spec.eq_ignore_ascii_case("default") {
        "large-v3-turbo"
    } else {
        spec
    };
    find_model_file(lookup)
}

/// Cheap availability check for the exact default model selection policy.
///
/// An explicit configured model wins. With no override, only the header of the
/// pinned release-package path is inspected; arbitrary legacy search entries
/// cannot make the default route appear available.
#[must_use]
pub(crate) fn configured_or_release_model_available() -> bool {
    let spec = default_model_spec().unwrap_or_else(|| "large-v3-turbo".to_owned());
    native_model_available(&spec)
}

/// Read the first 48 bytes of `path` and validate magic + ftype. Any failure
/// (short read, bad magic, unsupported ftype) yields `false`.
#[must_use]
fn header_ftype_ok(path: &Path) -> bool {
    use std::io::Read as _;
    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut buf = [0u8; HEADER_SNIFF_LEN];
    if file.read_exact(&mut buf).is_err() {
        return false;
    }
    let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if magic != GGML_MAGIC {
        return false;
    }
    // ftype is the 11th i32 after the magic: bytes [44..48).
    let ftype = i32::from_le_bytes([buf[44], buf[45], buf[46], buf[47]]);
    ftype == 0 || ftype == 1
}

// ─────────────────────────────────────────────────────────────────────────
// Threading
// ─────────────────────────────────────────────────────────────────────────

/// The default inference thread count: the machine's available parallelism,
/// capped at 32.
///
/// The cap was 16, but that MEASURED as far too low for the large-v3-turbo
/// encoder (the dominant ~82% cost, big `[1500,1280]×[1280,K]` sgemms). Fresh
/// sweep on a 64-core box (`examples/encoder_scale_probe.rs`, min-of-N):
/// encoder::forward best 4100 ms/win @16 → **3022 ms/win @32 (1.34×)**, then
/// regresses (48 → 3304, 64 → 3991 ms; cross-CCD sync — same wall whisper.cpp
/// `-t64` hits). matmul thread-count does not change per-element k-accumulation
/// order, so this is BYTE-EXACT: turbo transcript IDENTICAL @16 vs @32 (jfk×3
/// and jfk×6, `e2e_probe`), e2e **~1.23–1.28×** (jfk×6 14.56 s → 11.3–12.3 s).
/// 32 is the perf optimum AND still leaves half a 64-core host free (48/64
/// regress anyway), preserving the "don't fully monopolize" intent. Falls back
/// to `1` when parallelism cannot be queried. Callers should plumb
/// `BackendParams.threads` through and only fall back to this when unset;
/// `RAYON_NUM_THREADS` still overrides the pool entirely.
///
/// NOTE (Threadripper / high-core hosts): the `min(32)` above was the ENCODER
/// optimum (encoder_scale_probe, sequential). But the e2e is decoder-dominated
/// (~87%) and, with **window pipelining** on by default (no_timestamps — the
/// prefetch encoder of window N+1 runs CONCURRENTLY with the decode of window N
/// on this shared pool), 32 threads makes the compute-bound encoder and the
/// bandwidth-bound decoder contend and serialize. Sizing the pool to the host's
/// PHYSICAL core count lets both phases overlap — measured consistently ~15–23%
/// faster e2e on a 64-core Threadripper 5995WX (large-v3-turbo), and the
/// concurrent_pipeline_probe confirms ~54% reclaim of the smaller phase. SMT
/// siblings only add bandwidth/scheduling contention on this memory-bound work,
/// so we count physical cores, not logical. Hosts ≤32 logical are unchanged;
/// `RAYON_NUM_THREADS` / `BackendParams.threads` still override entirely.
#[must_use]
pub fn default_threads() -> usize {
    let host = host_parallelism();
    if host <= 32 {
        return host;
    }
    physical_cores().unwrap_or(host).max(32)
}

/// Physical core count (SMT-aware), or `None` if it can't be determined.
///
/// On Linux, `cpu0`'s `thread_siblings_list` gives threads-per-core directly, so
/// `physical = logical / threads_per_core`. On other targets (and if the sysfs
/// read fails) returns `None`; the caller then uses logical parallelism, which is
/// correct on a non-SMT host. Queried once via [`host_parallelism`]'s cache path.
fn physical_cores() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let sibs =
            std::fs::read_to_string("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list")
                .ok()?;
        let tpc = sibs
            .trim()
            .split([',', '-'])
            .filter(|s| !s.is_empty())
            .count();
        if tpc >= 1 {
            return Some((host_parallelism() / tpc).max(1));
        }
    }
    None
}

/// Host parallelism, queried ONCE and cached for the process.
///
/// `std::thread::available_parallelism()` is NOT cached by std and, on Linux,
/// walks the cgroup CPU-quota hierarchy on every call — `/proc/self/cgroup` plus
/// a `cpu.max` open+read at each level (~8 file opens) on top of a
/// `sched_getaffinity` syscall. That cost was paid per worker-count decision in
/// the per-tensor load path (`ggml`) and per mel/decode setup; the value is a
/// process constant, so caching it is behavior-identical (worker counts, and
/// thus every band split, are unchanged) and removes the repeated cgroup I/O.
pub(crate) fn host_parallelism() -> usize {
    use std::sync::OnceLock;
    static P: OnceLock<usize> = OnceLock::new();
    *P.get_or_init(|| {
        crate::native_engine::plat::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1)
    })
}

/// Initialize Rayon before inference kernels touch the global pool.
///
/// Rayon defaults to the full host parallelism, which is a poor fit on large
/// shared machines: this crate's kernels are tuned around [`default_threads`],
/// and the caller can still opt into another value with `RAYON_NUM_THREADS`.
pub(crate) fn ensure_default_rayon_pool() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        if std::env::var_os("RAYON_NUM_THREADS").is_none() {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(default_threads())
                .build_global();
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────
// Loaded-model cache
// ─────────────────────────────────────────────────────────────────────────

/// A fully loaded, ready-to-run native Whisper model.
///
/// Holds the parsed weights (pre-transposed for inference), tokenizer, mel
/// filterbank, and hyper-parameters via [`decode::LoadedModel`]. Construct one
/// through [`NativeWhisperModel::load`], which deduplicates via a global cache
/// so repeated runs — and all three native engines (transcribe / shadow /
/// replay) — share a single in-memory copy.
///
/// # Memory model
///
/// A loaded large-v3 model is roughly **3 GB of f32 weights**; even
/// `large-v3-turbo` is well over 1 GB. The normal global cache holds [`Weak`]
/// references, so the weights live exactly as long as some caller holds an
/// `Arc<NativeWhisperModel>`. [`load_resident`](Self::load_resident) is the
/// explicit exception: it keeps one strong process-wide slot alive for API
/// servers that want loaded-model residency. When the last non-resident `Arc`
/// drops, the memory is freed immediately and the cache slot becomes a dangling
/// `Weak`. Every
/// [`load`](Self::load) that inserts a new entry also prunes **all** dead
/// `Weak`s from the cache (not just the slot being re-loaded), so the
/// `HashMap` cannot accumulate entries for models that have all been dropped —
/// even when those models are never loaded again. Hold an `Arc` for the
/// duration of a run (or longer to keep the model warm); drop it to reclaim the
/// RAM.
pub struct NativeWhisperModel {
    /// Parsed, inference-ready weights + tokenizer + filters + hparams.
    inner: decode::LoadedModel,
    /// The canonical path this model was loaded from (cache key).
    pub model_path: PathBuf,
    /// Lazily-computed, cached engine version tag (see [`Self::version_tag`]).
    version_tag: OnceLock<String>,
}

/// Global, process-wide model cache keyed by canonical path.
///
/// `Weak` values mean normal [`load`](NativeWhisperModel::load) never keeps a
/// model alive on its own. [`load_resident`](NativeWhisperModel::load_resident)
/// adds one bounded strong slot for in-process API users who explicitly want
/// OpenAI-style loaded-model residency without mmap/unsafe.
#[derive(Default)]
struct ModelCache {
    weak: HashMap<PathBuf, Weak<NativeWhisperModel>>,
    resident: Option<(PathBuf, Arc<NativeWhisperModel>)>,
    /// In-flight cold loads keyed by canonical path (`FW_LOAD_DEDUP`). A peer parsing
    /// the same model holds this per-path lock so concurrent cold loads serialize on
    /// the parse rather than all parsing (the default path re-checks + discards the
    /// redundant parse — correct but N× parse work + peak RSS on a cold burst). Empty
    /// unless the flag is set.
    loading: HashMap<PathBuf, Arc<Mutex<()>>>,
}

static MODEL_CACHE: Mutex<Option<ModelCache>> = Mutex::new(None);

/// `FW_LOAD_DEDUP=1` — serialize concurrent COLD loads of the same model on a per-path
/// lock so only one thread parses (peers wait then hit the cache), avoiding the default
/// double-parse race (both parse, one discarded). Byte-exact; default OFF (a server that
/// loads a model resident-once never races, so it is opt-in for lazy/burst deployments).
fn load_dedup_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FW_LOAD_DEDUP").ok().as_deref() == Some("1"))
}

impl NativeWhisperModel {
    /// Load (or fetch from the global cache) the model at `path`.
    ///
    /// `path` is canonicalized to form the cache key, so two specs pointing at
    /// the same file via different relative/symlinked paths share one instance.
    /// If a live `Arc` already exists for that canonical path it is returned
    /// directly (no re-parse, no extra RAM). Otherwise the file is parsed once
    /// and the resulting `Arc` is both returned and stashed as a `Weak` in the
    /// cache.
    ///
    /// # Errors
    ///
    /// - [`FwError::Io`] if `path` cannot be canonicalized or read.
    /// - Whatever [`ggml::GgmlModel::load`] / [`decode::LoadedModel::from_ggml`]
    ///   return for a malformed or unsupported model.
    pub fn load(path: &Path) -> FwResult<Arc<Self>> {
        Self::load_inner(path, false)
    }

    /// Load a model and keep one process-wide strong resident slot alive.
    ///
    /// This is the safe, bounded residency path for reusable in-process API
    /// servers: repeated calls for the same canonical path return an `Arc` clone
    /// even if the previous caller dropped its handle. Loading a different path
    /// replaces the resident slot, so memory retention is capped to one model.
    /// Plain [`load`](Self::load) keeps the original Weak-only semantics.
    pub fn load_resident(path: &Path) -> FwResult<Arc<Self>> {
        Self::load_inner(path, true)
    }

    /// Load a resident model from an already-canonicalized absolute path.
    ///
    /// This is the hot-path companion to [`load_resident`](Self::load_resident)
    /// for servers/model registries that resolve a model once and then acquire
    /// it repeatedly. It deliberately skips the filesystem `canonicalize()` on
    /// every acquire; callers are responsible for passing the canonical path
    /// returned by [`resolve_model`] or `Path::canonicalize`. If a merely
    /// absolute but non-canonical path is passed, correctness is unchanged but
    /// it may occupy a separate cache slot from the file's canonical spelling.
    ///
    /// # Errors
    ///
    /// - [`FwError::InvalidRequest`] when `canonical_path` is relative.
    /// - Whatever [`ggml::GgmlModel::load`] / [`decode::LoadedModel::from_ggml`]
    ///   return for an unreadable, malformed, or unsupported model file.
    pub fn load_resident_canonical(canonical_path: &Path) -> FwResult<Arc<Self>> {
        if !canonical_path.is_absolute() {
            return Err(FwError::InvalidRequest(
                "resident canonical model path must be absolute".to_owned(),
            ));
        }
        Self::load_canonical(canonical_path.to_path_buf(), true)
    }

    fn load_inner(path: &Path, keep_resident: bool) -> FwResult<Arc<Self>> {
        let canonical = path.canonicalize()?;
        Self::load_canonical(canonical, keep_resident)
    }

    fn load_canonical(canonical: PathBuf, keep_resident: bool) -> FwResult<Arc<Self>> {
        // Fast path: a live cached instance.
        {
            let mut guard = lock_cache();
            let cache = guard.get_or_insert_with(ModelCache::default);
            if let Some((resident_path, resident)) = &cache.resident
                && resident_path == &canonical
            {
                return Ok(Arc::clone(resident));
            }
            if let Some(weak) = cache.weak.get(&canonical)
                && let Some(existing) = weak.upgrade()
            {
                if keep_resident {
                    cache.resident = Some((canonical.clone(), Arc::clone(&existing)));
                }
                return Ok(existing);
            }
        }

        // Serialize concurrent COLD loads of the same model (FW_LOAD_DEDUP): peers
        // wait on a per-path lock and then hit the freshly-published cache, instead
        // of all parsing. The default path below re-checks + discards the redundant
        // parse (correct, but pays N× parse work + peak RSS on a cold burst).
        if load_dedup_enabled() {
            let plock = {
                let mut guard = lock_cache();
                let cache = guard.get_or_insert_with(ModelCache::default);
                Arc::clone(
                    cache
                        .loading
                        .entry(canonical.clone())
                        .or_insert_with(|| Arc::new(Mutex::new(()))),
                )
            };
            // Held across the parse below — `plock` and `_held` are both locals of
            // this block (no self-referential borrow); dropped on return. Lock order
            // is always per-path THEN cache (cache is only taken briefly), no deadlock.
            let _held = plock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // A peer may have published while we waited on the per-path lock.
            {
                let mut guard = lock_cache();
                let cache = guard.get_or_insert_with(ModelCache::default);
                if let Some((rp, r)) = &cache.resident
                    && rp == &canonical
                {
                    return Ok(Arc::clone(r));
                }
                if let Some(w) = cache.weak.get(&canonical)
                    && let Some(existing) = w.upgrade()
                {
                    if keep_resident {
                        cache.resident = Some((canonical.clone(), Arc::clone(&existing)));
                    }
                    return Ok(existing);
                }
            }
            let model = Self::do_parse_and_publish(canonical.clone(), keep_resident)?;
            // Drop our in-flight marker (waiting peers hold their own `Arc` clone).
            if let Some(cache) = lock_cache().as_mut() {
                cache.loading.remove(&canonical);
            }
            return Ok(model);
        }
        Self::do_parse_and_publish(canonical, keep_resident)
    }

    /// Parse the ggml model, quantize weights, publish to the cache, and warm the
    /// version tag on a background thread. Shared by the plain and `FW_LOAD_DEDUP`
    /// load paths; its own re-check-under-lock still handles a racing publisher on the
    /// plain (non-deduped) path.
    fn do_parse_and_publish(canonical: PathBuf, keep_resident: bool) -> FwResult<Arc<Self>> {
        // Parse outside the lock so a slow load doesn't block other paths.
        let t_parse = crate::native_engine::plat::Instant::now();
        let ggml = ggml::GgmlModel::load(&canonical)?;
        perf_span("model_parse", t_parse.elapsed().as_secs_f64() * 1e3, "");
        let t_weights = crate::native_engine::plat::Instant::now();
        let inner = decode::LoadedModel::from_ggml(ggml)?;
        perf_span("model_weights", t_weights.elapsed().as_secs_f64() * 1e3, "");
        let model = Arc::new(Self {
            inner,
            model_path: canonical.clone(),
            version_tag: OnceLock::new(),
        });

        // Re-check under the lock: a racing thread may have populated the slot
        // while we were parsing. If so, prefer the already-published instance.
        let mut guard = lock_cache();
        let cache = guard.get_or_insert_with(ModelCache::default);
        if let Some((resident_path, resident)) = &cache.resident
            && resident_path == &canonical
        {
            return Ok(Arc::clone(resident));
        }
        if let Some(weak) = cache.weak.get(&canonical)
            && let Some(existing) = weak.upgrade()
        {
            if keep_resident {
                cache.resident = Some((canonical.clone(), Arc::clone(&existing)));
            }
            return Ok(existing);
        }
        // Prune every dead `Weak` (models whose last `Arc` has dropped), not
        // just the slot we are about to overwrite, so the cache cannot grow
        // unbounded with stale entries for models that are never reloaded.
        cache.weak.retain(|_, w| w.strong_count() > 0);
        cache.weak.insert(canonical.clone(), Arc::downgrade(&model));
        if keep_resident {
            cache.resident = Some((canonical, Arc::clone(&model)));
        }
        drop(guard);

        // Warm the version tag on a background thread: the SHA-256 of a
        // multi-GB model file costs seconds, and every successful run needs it
        // for `raw_output` / replay envelopes. Hashing here overlaps with
        // encode/decode instead of stalling output assembly; `OnceLock`
        // blocks any concurrent `version_tag()` caller until the value is
        // ready, so observable behavior (the tag itself) is unchanged. The
        // clone keeps the model alive until the hash finishes (bounded by
        // hash time; documented tradeoff).
        let warm = Arc::clone(&model);
        let _ = std::thread::Builder::new()
            .name("fw-model-hash".into())
            .spawn(move || {
                let _ = warm.version_tag();
            });
        Ok(model)
    }

    /// Run transcription over mono 16 kHz f32 `samples`, delegating to the
    /// frozen [`decode::transcribe_samples`] contract.
    ///
    /// `checkpoint` is invoked periodically by the decode loop for cooperative
    /// cancellation / deadline enforcement; an `Err` from it aborts the run.
    ///
    /// # Errors
    ///
    /// Propagates any error from the decode loop, including a `checkpoint`
    /// cancellation.
    pub fn transcribe(
        &self,
        samples: &[f32],
        params: &decode::DecodeParams,
        checkpoint: &(dyn Fn() -> FwResult<()> + Sync),
    ) -> FwResult<decode::DecodeOutput> {
        decode::transcribe_samples(&self.inner, samples, params, checkpoint)
    }

    /// A stable identity string for this model's weights, of the form
    /// `"fw-native-v1+sha256:{first 12 hex of the model file's sha256}"`.
    ///
    /// Computed lazily on first call (streaming the file through SHA-256) and
    /// cached for the life of the `Arc`, so it is cheap on repeat calls. This
    /// feeds [`ReplayEnvelope`](crate::conformance) `backend_identity`, letting
    /// conformance drift detection distinguish runs across model file changes
    /// while remaining stable for an unchanged file.
    #[must_use]
    pub fn version_tag(&self) -> String {
        self.version_tag
            .get_or_init(|| {
                let prefix = file_sha256_prefix(&self.model_path)
                    .unwrap_or_else(|| "unavailable".to_owned());
                format!("fw-native-v1+sha256:{prefix}")
            })
            .clone()
    }

    /// Borrow the underlying loaded model (parsed weights / tokenizer / etc.).
    #[must_use]
    pub fn loaded(&self) -> &decode::LoadedModel {
        &self.inner
    }
}

/// Lock the global cache, recovering from a poisoned mutex (a panic in another
/// thread while holding the lock must not wedge the whole engine — the cache is
/// pure dedup state and safe to keep using).
fn lock_cache() -> std::sync::MutexGuard<'static, Option<ModelCache>> {
    MODEL_CACHE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Stream a file through SHA-256 and return the first 12 hex chars of the
/// digest, or `None` on I/O failure.
#[must_use]
fn file_sha256_prefix(path: &Path) -> Option<String> {
    use std::io::Read as _;
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(12);
    for byte in digest.iter().take(6) {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    Some(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multilingual_threshold_matches_whisper_cpp() {
        let mut hp = WhisperHParams {
            n_vocab: 51864,
            n_audio_ctx: 1500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            n_mels: 80,
            ftype: 1,
        };
        assert!(!hp.is_multilingual(), "tiny.en (51864) is English-only");
        hp.n_vocab = 51865;
        assert!(hp.is_multilingual());
        hp.n_vocab = 51866;
        assert!(hp.is_multilingual(), "large-v3 family (51866)");
    }

    #[test]
    fn encoder_int8_policy_allows_calibrated_model_shapes() {
        let tiny = WhisperHParams {
            n_vocab: 51_864,
            n_audio_ctx: 1_500,
            n_audio_state: 384,
            n_audio_head: 6,
            n_audio_layer: 4,
            n_text_ctx: 448,
            n_text_state: 384,
            n_text_head: 6,
            n_text_layer: 4,
            n_mels: 80,
            ftype: 1,
        };
        let large_turbo = WhisperHParams {
            n_vocab: 51_866,
            n_audio_ctx: 1_500,
            n_audio_state: 1_280,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_text_ctx: 448,
            n_text_state: 1_280,
            n_text_head: 20,
            n_text_layer: 4,
            n_mels: 128,
            ftype: 1,
        };

        for hp in [tiny, large_turbo] {
            let decision = encoder_int8_policy_decision(&hp);
            if encoder_i8_kernel_supported() {
                assert_eq!(
                    decision.action,
                    EncoderInt8PolicyAction::QualitySafeInt8Encoder
                );
                assert_eq!(decision.reason, "calibrated_model_budget_pass");
            } else {
                assert_eq!(decision.action, EncoderInt8PolicyAction::F32Encoder);
                assert_eq!(decision.reason, "cpu_feature_fallback");
            }
            assert_eq!(decision.calibration_id, ENCODER_INT8_CALIBRATION_ID);
            assert_eq!(decision.corpus_wer_delta_budget, 0.0);
            assert_eq!(decision.quant_rel_rmse_budget, 0.09);
        }
    }

    #[test]
    fn is_large_v3_turbo_discriminates_models_for_poly_exp() {
        // bd-bcm7: poly softmax is admitted for turbo only. Verify the model
        // discriminator: turbo -> true, tiny.en -> false (uncertified), unknown -> false.
        let turbo = WhisperHParams {
            n_vocab: 51_866,
            n_audio_ctx: 1_500,
            n_audio_state: 1_280,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_text_ctx: 448,
            n_text_state: 1_280,
            n_text_head: 20,
            n_text_layer: 4,
            n_mels: 128,
            ftype: 1,
        };
        assert!(is_large_v3_turbo(&turbo));
        let mut tiny = turbo;
        tiny.n_vocab = 51_864;
        tiny.n_audio_state = 384;
        tiny.n_audio_head = 6;
        tiny.n_audio_layer = 4;
        tiny.n_text_state = 384;
        tiny.n_text_head = 6;
        tiny.n_mels = 80;
        assert!(
            !is_large_v3_turbo(&tiny),
            "tiny.en must NOT enable poly (uncertified)"
        );
        let mut unknown = turbo;
        unknown.n_audio_state = 1_024;
        assert!(
            !is_large_v3_turbo(&unknown),
            "unknown model must NOT enable poly"
        );
    }

    #[test]
    fn encoder_int8_policy_falls_back_for_uncalibrated_shape() {
        let unknown = WhisperHParams {
            n_vocab: 51_866,
            n_audio_ctx: 1_500,
            n_audio_state: 768,
            n_audio_head: 12,
            n_audio_layer: 12,
            n_text_ctx: 448,
            n_text_state: 768,
            n_text_head: 12,
            n_text_layer: 12,
            n_mels: 80,
            ftype: 1,
        };
        let decision = encoder_int8_policy_decision(&unknown);
        assert_eq!(decision.action, EncoderInt8PolicyAction::F32Encoder);
        if encoder_i8_kernel_supported() {
            assert_eq!(decision.reason, "uncalibrated_model_fallback");
        } else {
            assert_eq!(decision.reason, "cpu_feature_fallback");
        }
    }

    #[test]
    fn mat_row_access() {
        let m = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test helpers
    // ─────────────────────────────────────────────────────────────────────

    use std::io::Write as _;
    use std::sync::Mutex as StdMutex;

    /// Serializes any test that reads or depends on process-wide environment
    /// state. We never *mutate* env vars (that is `unsafe`/forbidden under
    /// edition 2024 — see [`tests/e2e_pipeline_tests.rs`]); this guard simply
    /// keeps env-reading tests from interleaving with each other in case the
    /// surrounding process env is changed by an outer harness.
    static ENV_TEST_MUTEX: StdMutex<()> = StdMutex::new(());

    /// A unique temp dir under the system temp root, created fresh.
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let path = std::env::temp_dir().join(format!("fw_native_{tag}_{pid}_{n}"));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// Write a 48-byte ggml-style header into `bytes`: magic followed by the
    /// eleven hparams, with `ftype` controllable.
    fn push_valid_header(bytes: &mut Vec<u8>, ftype: i32) {
        bytes.extend_from_slice(&GGML_MAGIC.to_le_bytes());
        // n_vocab .. n_mels (ten i32), then ftype.
        for v in [51865i32, 1500, 384, 6, 4, 448, 384, 6, 4, 80] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&ftype.to_le_bytes());
    }

    /// Tiny hyper-parameters for the full synthetic model. These are the exact
    /// dimensions every emitted tensor is sized against in
    /// [`full_synthetic_model_bytes`]; keeping them as named constants lets the
    /// builder and any future assertion stay in lock-step.
    ///
    /// `N_VOCAB` is the real tiny.en value (51864) so the tokenizer is built as
    /// an English-only model exactly as production does; everything else is
    /// shrunk to the smallest sizes the encoder/decoder loaders accept.
    const SYN_N_VOCAB: i32 = 51864;
    const SYN_N_AUDIO_CTX: i32 = 4;
    const SYN_N_AUDIO_STATE: i32 = 8;
    const SYN_N_AUDIO_HEAD: i32 = 2;
    const SYN_N_AUDIO_LAYER: i32 = 1;
    const SYN_N_TEXT_CTX: i32 = 4;
    const SYN_N_TEXT_STATE: i32 = 8;
    const SYN_N_TEXT_HEAD: i32 = 2;
    const SYN_N_TEXT_LAYER: i32 = 1;
    const SYN_N_MELS: i32 = 4;
    /// Filterbank FFT-bin count (whisper.cpp uses 201; the mel module doesn't
    /// validate it against hparams here, but we keep the real value for realism).
    const SYN_N_FFT_BINS: i32 = 201;

    /// Number of byte-level vocab tokens stored *in the file*. Smaller than
    /// `SYN_N_VOCAB`; the gap becomes synthetic special tokens, exactly as in a
    /// real model. The tokenizer/loader never validate this count against
    /// `n_vocab`, so a small set keeps the blob tiny.
    const SYN_FILE_VOCAB: usize = 16;

    /// Build a **complete, fully valid** ggml model blob that
    /// `decode::LoadedModel::from_ggml` accepts: it emits every tensor the
    /// encoder and decoder loaders require, at shapes consistent with the tiny
    /// synthetic hyper-parameters above.
    ///
    /// Layout, in file order (all little-endian):
    /// - 48-byte header: magic + 11 `i32` hparams (`ftype = 0`, f32 tensors).
    /// - mel filterbank `n_mel x n_fft_bins` (`4 x 201`) of zeros.
    /// - vocab: `SYN_FILE_VOCAB` single-byte tokens.
    /// - encoder tensors: conv1/conv2 (+biases), positional embedding,
    ///   one transformer block (attn q/k/v/out, MLP, layer norms), `ln_post`.
    /// - decoder tensors: token embedding `[51864, 8]`, positional embedding,
    ///   one block (self-attn, cross-attn, MLP, layer norms), final `ln`.
    ///
    /// All weights are zeros — `from_ggml` only validates *names and shapes*,
    /// never values, so zero data is sufficient for a successful load. Linear
    /// weights are written in ggml `ne` order (the reverse of the logical
    /// `[out, in]` row-major shape the loaders assert), since the parser
    /// reverses dims on read. The result is ~1.7 MB (dominated by the token
    /// embedding); it is built once and memoised in [`synthetic_model_bytes`].
    fn build_full_synthetic_model() -> Vec<u8> {
        let n_state = SYN_N_AUDIO_STATE as usize; // == n_text_state == 8
        let n_mels = SYN_N_MELS as usize;
        let n_audio_ctx = SYN_N_AUDIO_CTX as usize;
        let n_text_ctx = SYN_N_TEXT_CTX as usize;
        let n_vocab = SYN_N_VOCAB as usize;
        let mlp_hidden = 4 * n_state;
        let conv_k = 3usize;

        let mut b = Vec::new();
        // ── header (ftype = 0 => all "big" tensors are f32) ──
        b.extend_from_slice(&GGML_MAGIC.to_le_bytes());
        for v in [
            SYN_N_VOCAB,
            SYN_N_AUDIO_CTX,
            SYN_N_AUDIO_STATE,
            SYN_N_AUDIO_HEAD,
            SYN_N_AUDIO_LAYER,
            SYN_N_TEXT_CTX,
            SYN_N_TEXT_STATE,
            SYN_N_TEXT_HEAD,
            SYN_N_TEXT_LAYER,
            SYN_N_MELS,
            0, // ftype = f32
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }

        // ── mel filterbank: n_mel x n_fft_bins, zeros ──
        b.extend_from_slice(&SYN_N_MELS.to_le_bytes());
        b.extend_from_slice(&SYN_N_FFT_BINS.to_le_bytes());
        for _ in 0..(n_mels * SYN_N_FFT_BINS as usize) {
            b.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // ── vocab: SYN_FILE_VOCAB single-byte tokens ──
        b.extend_from_slice(&(SYN_FILE_VOCAB as i32).to_le_bytes());
        for i in 0..SYN_FILE_VOCAB {
            let tok = [b'!'.wrapping_add(i as u8)];
            b.extend_from_slice(&(tok.len() as u32).to_le_bytes());
            b.extend_from_slice(&tok);
        }

        // ── encoder tensors ──
        // Conv weights are flat [Cout, Cin, K] (logical, row-major) — the loader
        // asserts that exact 3-D shape, so write ggml ne = reverse = [K, Cin, Cout].
        push_tensor_f32_logical(&mut b, "encoder.conv1.weight", &[n_state, n_mels, conv_k]);
        push_tensor_f32_logical(&mut b, "encoder.conv1.bias", &[n_state]);
        push_tensor_f32_logical(&mut b, "encoder.conv2.weight", &[n_state, n_state, conv_k]);
        push_tensor_f32_logical(&mut b, "encoder.conv2.bias", &[n_state]);
        push_tensor_f32_logical(
            &mut b,
            "encoder.positional_embedding",
            &[n_audio_ctx, n_state],
        );
        for i in 0..SYN_N_AUDIO_LAYER {
            let p = |s: &str| format!("encoder.blocks.{i}.{s}");
            push_tensor_f32_logical(&mut b, &p("attn_ln.weight"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("attn_ln.bias"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.query.weight"), &[n_state, n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.query.bias"), &[n_state]);
            // whisper key projection has NO bias.
            push_tensor_f32_logical(&mut b, &p("attn.key.weight"), &[n_state, n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.value.weight"), &[n_state, n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.value.bias"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.out.weight"), &[n_state, n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.out.bias"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("mlp_ln.weight"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("mlp_ln.bias"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("mlp.0.weight"), &[mlp_hidden, n_state]);
            push_tensor_f32_logical(&mut b, &p("mlp.0.bias"), &[mlp_hidden]);
            push_tensor_f32_logical(&mut b, &p("mlp.2.weight"), &[n_state, mlp_hidden]);
            push_tensor_f32_logical(&mut b, &p("mlp.2.bias"), &[n_state]);
        }
        push_tensor_f32_logical(&mut b, "encoder.ln_post.weight", &[n_state]);
        push_tensor_f32_logical(&mut b, "encoder.ln_post.bias", &[n_state]);

        // ── decoder tensors ──
        push_tensor_f32_logical(
            &mut b,
            "decoder.token_embedding.weight",
            &[n_vocab, n_state],
        );
        push_tensor_f32_logical(
            &mut b,
            "decoder.positional_embedding",
            &[n_text_ctx, n_state],
        );
        push_tensor_f32_logical(&mut b, "decoder.ln.weight", &[n_state]);
        push_tensor_f32_logical(&mut b, "decoder.ln.bias", &[n_state]);
        for i in 0..SYN_N_TEXT_LAYER {
            let p = |s: &str| format!("decoder.blocks.{i}.{s}");
            // self-attention
            push_tensor_f32_logical(&mut b, &p("attn_ln.weight"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("attn_ln.bias"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.query.weight"), &[n_state, n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.query.bias"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.key.weight"), &[n_state, n_state]); // no bias
            push_tensor_f32_logical(&mut b, &p("attn.value.weight"), &[n_state, n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.value.bias"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.out.weight"), &[n_state, n_state]);
            push_tensor_f32_logical(&mut b, &p("attn.out.bias"), &[n_state]);
            // cross-attention
            push_tensor_f32_logical(&mut b, &p("cross_attn_ln.weight"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("cross_attn_ln.bias"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("cross_attn.query.weight"), &[n_state, n_state]);
            push_tensor_f32_logical(&mut b, &p("cross_attn.query.bias"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("cross_attn.key.weight"), &[n_state, n_state]); // no bias
            push_tensor_f32_logical(&mut b, &p("cross_attn.value.weight"), &[n_state, n_state]);
            push_tensor_f32_logical(&mut b, &p("cross_attn.value.bias"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("cross_attn.out.weight"), &[n_state, n_state]);
            push_tensor_f32_logical(&mut b, &p("cross_attn.out.bias"), &[n_state]);
            // MLP
            push_tensor_f32_logical(&mut b, &p("mlp_ln.weight"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("mlp_ln.bias"), &[n_state]);
            push_tensor_f32_logical(&mut b, &p("mlp.0.weight"), &[mlp_hidden, n_state]);
            push_tensor_f32_logical(&mut b, &p("mlp.0.bias"), &[mlp_hidden]);
            push_tensor_f32_logical(&mut b, &p("mlp.2.weight"), &[n_state, mlp_hidden]);
            push_tensor_f32_logical(&mut b, &p("mlp.2.bias"), &[n_state]);
        }

        b
    }

    /// Memoised `~1.7 MB` synthetic model blob (built once, shared by every
    /// test that needs a loadable model — see [`build_full_synthetic_model`]).
    fn synthetic_model_bytes() -> &'static [u8] {
        static BLOB: OnceLock<Vec<u8>> = OnceLock::new();
        BLOB.get_or_init(build_full_synthetic_model)
    }

    /// Emit one f32 tensor whose **logical** (row-major / PyTorch) shape is
    /// `logical_shape`, with all-zero data. The ggml file stores dims in
    /// reversed (`ne[0]` = fastest axis) order and the parser reverses them on
    /// read, so we write `logical_shape` reversed; the parser then recovers
    /// exactly `logical_shape`, which is what the encoder/decoder loaders
    /// assert against.
    fn push_tensor_f32_logical(bytes: &mut Vec<u8>, name: &str, logical_shape: &[usize]) {
        let n_dims = logical_shape.len();
        bytes.extend_from_slice(&(n_dims as i32).to_le_bytes());
        bytes.extend_from_slice(&(name.len() as i32).to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes()); // ttype = f32
        // ggml ne order = reverse of logical row-major shape.
        for &d in logical_shape.iter().rev() {
            bytes.extend_from_slice(&(d as i32).to_le_bytes());
        }
        bytes.extend_from_slice(name.as_bytes());
        let n_elements: usize = logical_shape.iter().product();
        bytes.extend(std::iter::repeat_n(0u8, n_elements * 4));
    }

    fn write_file(dir: &Path, name: &str, contents: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).expect("create file");
        f.write_all(contents).expect("write file");
        path
    }

    // ─────────────────────────────────────────────────────────────────────
    // resolve_model
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn resolve_model_existing_path_is_canonicalized() {
        let dir = TempDir::new("path");
        let mut valid = Vec::new();
        push_valid_header(&mut valid, 1);
        let path = write_file(dir.path(), "ggml-tiny.en.bin", &valid);
        // Pass a relative-ish/non-canonical spec via the absolute path; result
        // must be the canonical form of the same file.
        let resolved = resolve_model(path.to_str().expect("utf8")).expect("resolve");
        assert_eq!(resolved, path.canonicalize().expect("canon"));
    }

    #[test]
    fn resolve_in_dirs_precedence_first_match_wins() {
        let high = TempDir::new("hi");
        let low = TempDir::new("lo");
        let mut valid = Vec::new();
        push_valid_header(&mut valid, 1);
        // Same short name present in both dirs; the first dir must win.
        let hi_path = write_file(high.path(), "ggml-base.bin", &valid);
        let _lo_path = write_file(low.path(), "ggml-base.bin", &valid);

        let dirs = vec![high.path().to_path_buf(), low.path().to_path_buf()];
        let resolved = resolve_model_in_dirs("base", &dirs).expect("resolve");
        assert_eq!(resolved, hi_path.canonicalize().expect("canon"));

        // Reversed order resolves to the other dir.
        let dirs_rev = vec![low.path().to_path_buf(), high.path().to_path_buf()];
        let resolved_rev = resolve_model_in_dirs("base", &dirs_rev).expect("resolve");
        assert_eq!(resolved_rev, _lo_path.canonicalize().expect("canon"));
    }

    #[test]
    fn resolve_in_dirs_falls_through_to_later_dir() {
        let empty = TempDir::new("empty");
        let real = TempDir::new("real");
        let mut valid = Vec::new();
        push_valid_header(&mut valid, 1);
        let path = write_file(real.path(), "ggml-small.bin", &valid);
        let dirs = vec![empty.path().to_path_buf(), real.path().to_path_buf()];
        let resolved = resolve_model_in_dirs("small", &dirs).expect("resolve");
        assert_eq!(resolved, path.canonicalize().expect("canon"));
    }

    #[test]
    fn resolve_in_dirs_skips_corrupt_higher_precedence_candidate() {
        let corrupt = TempDir::new("corrupt");
        let valid_dir = TempDir::new("valid");
        write_file(corrupt.path(), "ggml-small.bin", b"not a ggml model");
        let mut valid = Vec::new();
        push_valid_header(&mut valid, 1);
        let expected = write_file(valid_dir.path(), "ggml-small.bin", &valid);

        let dirs = vec![corrupt.path().to_path_buf(), valid_dir.path().to_path_buf()];
        let resolved = resolve_model_in_dirs("small", &dirs).expect("resolve valid fallback");
        assert_eq!(resolved, expected.canonicalize().expect("canon"));
    }

    #[test]
    fn resolve_model_rejects_explicit_corrupt_file() {
        let dir = TempDir::new("explicit_corrupt");
        let path = write_file(dir.path(), "ggml-tiny.en.bin", b"not a ggml model");
        let error = resolve_model(path.to_str().expect("utf8")).expect_err("reject corrupt model");
        assert!(matches!(error, FwError::InvalidRequest(_)));
    }

    #[test]
    fn resolve_miss_error_lists_dirs_and_filename() {
        let a = TempDir::new("a");
        let b = TempDir::new("b");
        let dirs = vec![a.path().to_path_buf(), b.path().to_path_buf()];
        let err = resolve_model_in_dirs("large-v3-turbo", &dirs).expect_err("should miss");
        let msg = err.to_string();
        assert!(
            msg.contains("ggml-large-v3-turbo.bin"),
            "names expected file: {msg}"
        );
        assert!(
            msg.contains(&a.path().display().to_string()),
            "lists first dir: {msg}"
        );
        assert!(
            msg.contains(&b.path().display().to_string()),
            "lists second dir: {msg}"
        );
        assert!(matches!(err, FwError::InvalidRequest(_)));
    }

    #[test]
    fn resolve_model_uses_env_search_dirs() {
        // Reads process env via model_search_dirs(); serialize against other
        // env-sensitive tests. We don't mutate env, so we just assert behavior
        // is consistent with find_model_file (same search list).
        let _guard = ENV_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        // A name almost certainly absent everywhere resolves to an error, not
        // a panic, and the error enumerates the live search dirs.
        let err = resolve_model("definitely-not-a-real-model-xyz").expect_err("miss");
        let msg = err.to_string();
        assert!(msg.contains("ggml-definitely-not-a-real-model-xyz.bin"));
    }

    // ─────────────────────────────────────────────────────────────────────
    // native_model_available  (header sniff)
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn available_false_for_missing_model() {
        let _guard = ENV_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!native_model_available("no-such-model-zzz"));
    }

    #[test]
    fn available_false_for_bad_magic() {
        let dir = TempDir::new("badmagic");
        // 48 bytes but wrong magic.
        let mut bytes = vec![0u8; HEADER_SNIFF_LEN];
        bytes[0..4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        let path = write_file(dir.path(), "ggml-x.bin", &bytes);
        assert!(!native_model_available(path.to_str().expect("utf8")));
    }

    #[test]
    fn available_false_for_short_file() {
        let dir = TempDir::new("short");
        // Fewer than 48 bytes => read_exact fails.
        let path = write_file(dir.path(), "ggml-x.bin", b"ggml-too-short");
        assert!(!native_model_available(path.to_str().expect("utf8")));
    }

    #[test]
    fn available_false_for_quantized_ftype() {
        let dir = TempDir::new("quant");
        let mut bytes = Vec::new();
        push_valid_header(&mut bytes, 2); // ftype 2 = quantized, unsupported.
        let path = write_file(dir.path(), "ggml-x.bin", &bytes);
        assert!(!native_model_available(path.to_str().expect("utf8")));
    }

    #[test]
    fn available_true_for_crafted_valid_header() {
        let dir = TempDir::new("valid");
        let mut bytes = Vec::new();
        push_valid_header(&mut bytes, 1); // ftype 1 = f16, supported.
        // Pad past 48 bytes to be realistic; only the header is sniffed.
        bytes.extend_from_slice(&[0u8; 16]);
        let path = write_file(dir.path(), "ggml-x.bin", &bytes);
        assert!(native_model_available(path.to_str().expect("utf8")));

        // ftype 0 (f32) is also valid.
        let mut bytes0 = Vec::new();
        push_valid_header(&mut bytes0, 0);
        let path0 = write_file(dir.path(), "ggml-y.bin", &bytes0);
        assert!(native_model_available(path0.to_str().expect("utf8")));
    }

    // ─────────────────────────────────────────────────────────────────────
    // default_model_spec / default_threads
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn default_model_spec_reflects_env() {
        let _guard = ENV_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        // We can't mutate env (forbidden unsafe under edition 2024), so assert
        // the documented mapping for the ambient value: Some(non-empty) or None.
        match std::env::var("FRANKEN_WHISPER_NATIVE_DEFAULT_MODEL") {
            Ok(v) if !v.is_empty() => assert_eq!(default_model_spec(), Some(v)),
            _ => assert_eq!(default_model_spec(), None),
        }
    }

    #[test]
    fn default_threads_in_bounds() {
        let n = default_threads();
        assert!((1..=32).contains(&n), "threads {n} must be 1..=32");
    }

    // ─────────────────────────────────────────────────────────────────────
    // NativeWhisperModel cache identity + version_tag
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn cache_returns_same_arc_then_reloads_after_drop() {
        let dir = TempDir::new("cache");
        let path = write_file(dir.path(), "ggml-cache.bin", synthetic_model_bytes());

        let a = NativeWhisperModel::load(&path).expect("load a");
        let b = NativeWhisperModel::load(&path).expect("load b");
        assert!(
            Arc::ptr_eq(&a, &b),
            "two loads of the same path must share one Arc"
        );

        // `load` spawns a background warm thread holding an `Arc` clone to
        // compute the version tag (SHA-256 of the file). Force that `OnceLock`
        // init to complete now so the warm thread's `version_tag()` returns
        // immediately and drops its clone promptly; without this the thread may
        // still be hashing when we drop our refs, keeping the `Weak` alive.
        let _ = a.version_tag();

        let weak = Arc::downgrade(&a);
        drop(a);
        drop(b);

        // The warm thread may not have dropped its `Arc` clone yet even after
        // `version_tag()` returned to us (it still has to return from its own
        // call and unwind). Poll with a deadline for the last strong ref to
        // drop so the `Weak` expires before we assert the reload is fresh.
        let deadline =
            crate::native_engine::plat::Instant::now() + std::time::Duration::from_secs(5);
        while weak.upgrade().is_some() {
            assert!(
                crate::native_engine::plat::Instant::now() < deadline,
                "all strong refs dropped => Weak must expire (memory freed) within 5s"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            weak.upgrade().is_none(),
            "all strong refs dropped => Weak must expire (memory freed)"
        );

        let c = NativeWhisperModel::load(&path).expect("reload c");
        // Fresh instance after the cache slot's Weak expired.
        assert!(weak.upgrade().is_none());
        assert_eq!(c.model_path, path.canonicalize().expect("canon"));
    }

    /// The resident cache is deliberately a SINGLE global slot
    /// (`ModelCache::resident`), so the two resident tests below evict each
    /// other's slot when the harness schedules them on concurrent threads —
    /// the bd-0ivd secondary flap (remote workers, 2026-07-22: `drop(a)` →
    /// sibling's resident load lands → `weak.upgrade()` finds the slot gone).
    /// The engine is behaving as designed; the tests assume slot exclusivity,
    /// so they serialize on this lock.
    static RESIDENT_SLOT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn resident_cache_keeps_one_model_alive_after_drop() {
        let _slot = RESIDENT_SLOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = TempDir::new("resident_cache");
        let path = write_file(dir.path(), "ggml-resident.bin", synthetic_model_bytes());

        let a = NativeWhisperModel::load_inner(&path, true).expect("resident load a");
        let _ = a.version_tag();
        let weak = Arc::downgrade(&a);
        drop(a);

        let retained = weak
            .upgrade()
            .expect("resident cache must keep the model alive after caller drop");
        let b = NativeWhisperModel::load_inner(&path, true).expect("resident load b");
        assert!(
            Arc::ptr_eq(&retained, &b),
            "resident reload must return the retained Arc"
        );
    }

    #[test]
    fn resident_canonical_path_reuses_resident_slot_without_recanonicalizing() {
        let _slot = RESIDENT_SLOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = TempDir::new("resident_canonical");
        let path = write_file(
            dir.path(),
            "ggml-resident-canonical.bin",
            synthetic_model_bytes(),
        );
        let canonical = path.canonicalize().expect("canon");

        let a = NativeWhisperModel::load_resident(&path).expect("resident load");
        let b = NativeWhisperModel::load_resident_canonical(&canonical)
            .expect("canonical resident load");
        assert!(
            Arc::ptr_eq(&a, &b),
            "canonical resident acquire must return the existing resident Arc"
        );
        assert_eq!(b.model_path, canonical);
    }

    #[test]
    fn resident_canonical_path_rejects_relative_path() {
        assert!(
            matches!(
                NativeWhisperModel::load_resident_canonical(Path::new("ggml-relative.bin")),
                Err(FwError::InvalidRequest(_))
            ),
            "relative canonical path must be rejected as invalid request"
        );
    }

    #[test]
    fn version_tag_is_stable_and_well_formed() {
        let dir = TempDir::new("vtag");
        let path = write_file(dir.path(), "ggml-vtag.bin", synthetic_model_bytes());
        let model = NativeWhisperModel::load(&path).expect("load");

        let t1 = model.version_tag();
        let t2 = model.version_tag();
        assert_eq!(t1, t2, "version_tag must be stable across calls");

        let prefix = "fw-native-v1+sha256:";
        assert!(t1.starts_with(prefix), "got {t1}");
        let hex = &t1[prefix.len()..];
        assert_eq!(hex.len(), 12, "sha prefix must be 12 hex chars: {hex}");
        assert!(
            hex.chars().all(|c| c.is_ascii_hexdigit()),
            "non-hex in {hex}"
        );
    }

    #[test]
    fn version_tag_changes_with_file_contents() {
        let dir = TempDir::new("vtag2");
        let bytes = synthetic_model_bytes();
        let p1 = write_file(dir.path(), "ggml-one.bin", bytes);
        // A different but still-parseable model: flip one byte INSIDE the tensor
        // data. The final bytes belong to the last decoder tensor's all-zero
        // f32 payload, so mutating them keeps every name/shape valid (the parser
        // never inspects values) while changing the file's SHA-256.
        let mut mutated = bytes.to_vec();
        let len = mutated.len();
        mutated[len - 4..].copy_from_slice(&9.0f32.to_le_bytes());
        let p2 = write_file(dir.path(), "ggml-two.bin", &mutated);

        let m1 = NativeWhisperModel::load(&p1).expect("load 1");
        let m2 = NativeWhisperModel::load(&p2).expect("load 2");
        assert_ne!(
            m1.version_tag(),
            m2.version_tag(),
            "distinct file contents must hash differently"
        );
    }

    #[test]
    fn find_model_file_shares_resolve_search_dirs() {
        // find_model_file and resolve_model must agree for a present file.
        // Use an explicit dir via resolve_model_in_dirs to confirm the shared
        // filename mapping (ggml-{name}.bin).
        let dir = TempDir::new("share");
        let path = write_file(dir.path(), "ggml-base.bin", synthetic_model_bytes());
        let dirs = vec![dir.path().to_path_buf()];
        let resolved = resolve_model_in_dirs("base", &dirs).expect("resolve");
        assert_eq!(resolved, path.canonicalize().expect("canon"));
        assert_eq!(model_file_name("base"), "ggml-base.bin");
    }
}
