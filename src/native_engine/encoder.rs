//! Whisper audio encoder forward pass (pure Rust, exact whisper.cpp semantics).
//!
//! This module ports whisper.cpp's `whisper_encode_internal` /
//! `whisper_build_graph_conv` + `whisper_build_graph_encoder` (see
//! `src/whisper.cpp`, the `embd_conv` and encoder-block sections) into the
//! row-major [`Mat`] world established by [`super::nn`]. The encoder turns a
//! 30 s log-mel window (`[n_mel, 3000]`) into a `[1500, n_state]` acoustic
//! embedding that the decoder cross-attends to.
//!
//! # Pipeline (exact whisper architecture)
//!
//! 1. **conv stem.** `mel` arrives mel-major (`[n_mel, n_frames]`); we
//!    transpose it to time-major `[n_frames, n_mel]` (the `[T, Cin]` layout
//!    [`nn::conv1d`] expects), then:
//!    - `conv1`: `Conv1d(n_mel -> n_state, k=3, stride=1, pad=1)` + GELU.
//!    - `conv2`: `Conv1d(n_state -> n_state, k=3, stride=2, pad=1)` + GELU.
//!
//!    With a 3000-frame input the stride-2 second conv halves time to
//!    `n_ctx = 1500`, yielding `[1500, n_state]` time-major.
//! 2. **positional embedding.** We add the **file tensor**
//!    `encoder.positional_embedding` (logical shape `[n_audio_ctx, n_state]`),
//!    sliced to the first `n_ctx` rows. whisper.cpp adds `e_pe` directly (it
//!    does *not* recompute sinusoids at inference), so we use the stored
//!    values verbatim for bit-fidelity.
//! 3. **`n_audio_layer` residual transformer blocks**, each:
//!    - `x = x + attn_out(attn(ln_attn(x)))` with **bidirectional**
//!      (un-masked) multi-head self-attention. The projections are
//!      `query (W+b)`, `key (W, NO bias)`, `value (W+b)`, `out (W+b)` —
//!      whisper deliberately omits the key bias.
//!    - `x = x + mlp(ln_mlp(x))` with
//!      `mlp = Linear(n, 4n) + GELU + Linear(4n, n)`.
//! 4. **final `ln_post`** layer-norm.
//!
//! Layer norms use `eps = 1e-5` (whisper.cpp `hparams.eps`).
//!
//! # Weight layout conventions
//!
//! Linear weights in ggml are stored `[out, in]` (PyTorch order).
//! [`nn::matmul_bias`] wants the pre-transposed `[in, out]`, so
//! [`EncoderWeights::from_ggml`] transposes **every** linear weight **once**
//! at load. Conv weights keep their flat `[Cout, Cin, K]` ggml order, which is
//! exactly what [`nn::conv1d`] consumes. Layer-norm `weight`/`bias` and conv
//! biases are loaded as-is.
//!
//! # Cross K/V boundary (decoder scope)
//!
//! whisper precomputes per-layer cross-attention K/V from the encoder output
//! for the decoder. Crucially, those projections use **decoder** weights
//! (`decoder.blocks.{i}.cross_attn.{key,value}` applied to *this* module's
//! output), so they belong to the decoder bead (bd-hlpk), **not** here. This
//! module's sole numeric output is the encoder embedding [`Mat`]; the decoder
//! consumes it and runs the cross projections itself.

#![allow(clippy::module_name_repetitions)]

use ft_core::Float16;
use rayon::prelude::*;

use super::ggml::GgmlModel;
use super::nn;
use super::{Mat, Mel, WhisperHParams};
use crate::error::{FwError, FwResult};

/// Layer-norm epsilon (whisper.cpp `whisper_hparams::eps`).
const LN_EPS: f32 = 1e-5;
/// Convolution kernel width for both encoder convs (whisper fixes `k = 3`).
const CONV_K: usize = 3;
/// Convolution padding for both encoder convs (`pad = 1`, "same" for `k=3`).
const CONV_PAD: usize = 1;
/// MLP inner-dimension expansion factor (`Linear(n, 4n)` then `Linear(4n, n)`).
const MLP_RATIO: usize = 4;

/// Pre-transposed weights for a single encoder transformer block.
///
/// All linear weights are stored in `[in, out]` order (transposed once from
/// ggml's `[out, in]`) so [`nn::matmul_bias`] is a contiguous `x @ w_t`.
#[derive(Debug, Clone)]
struct EncoderLayer {
    /// `attn_ln` (pre-attention layer-norm) scale/shift, length `n_state`.
    attn_ln_w: Vec<f32>,
    attn_ln_b: Vec<f32>,
    /// Query projection `[n_state, n_state]` (`[in, out]`) + bias.
    attn_q_w: Mat,
    attn_q_b: Vec<f32>,
    /// Key projection `[n_state, n_state]` (`[in, out]`); **no bias** (whisper).
    attn_k_w: Mat,
    /// Value projection `[n_state, n_state]` (`[in, out]`) + bias.
    attn_v_w: Mat,
    attn_v_b: Vec<f32>,
    /// Output projection `[n_state, n_state]` (`[in, out]`) + bias.
    attn_out_w: Mat,
    attn_out_b: Vec<f32>,
    /// `mlp_ln` (pre-MLP layer-norm) scale/shift, length `n_state`.
    mlp_ln_w: Vec<f32>,
    mlp_ln_b: Vec<f32>,
    /// MLP up projection `[n_state, 4*n_state]` (`[in, out]`) + bias.
    mlp_fc_w: Mat,
    mlp_fc_b: Vec<f32>,
    /// MLP down projection `[4*n_state, n_state]` (`[in, out]`) + bias.
    mlp_proj_w: Mat,
    mlp_proj_b: Vec<f32>,
    /// Optional 7-bit int8 (maddubs) copies of the six linear weights, built ONCE
    /// at load iff [`super::enc_int8_enabled`]. `None` = f32 path = byte-identical.
    attn_q_i7: Option<nn::I7Mat>,
    attn_k_i7: Option<nn::I7Mat>,
    attn_v_i7: Option<nn::I7Mat>,
    attn_out_i7: Option<nn::I7Mat>,
    mlp_fc_i7: Option<nn::I7Mat>,
    mlp_proj_i7: Option<nn::I7Mat>,
    /// Optional cached **i8** (full 8-bit) `attn.out` weight, built ONCE at load
    /// iff [`super::enc_attn_out_i8i32`]. Routes `attn.out` through the i8×i8
    /// i32-accumulate GEMM ([`matmul_bias_i8`]) instead of the i7 maddubs — the
    /// extra weight bit preserves proper nouns on this residual-feeding path.
    /// `None` = f32/i7 path.
    attn_out_i8: Option<EncI8Mat>,
}

/// Fully loaded, pre-transposed encoder weights for one whisper model.
///
/// Build with [`EncoderWeights::from_ggml`]; consume with [`forward`]. Every
/// tensor's shape is validated against the model hyper-parameters at load, so
/// a malformed or mismatched file fails fast with a tensor-named error rather
/// than producing silent garbage during inference.
#[derive(Debug, Clone)]
pub struct EncoderWeights {
    /// Number of mel input channels (`hparams.n_mels`).
    n_mels: usize,
    /// Hidden width (`hparams.n_audio_state`).
    n_state: usize,
    /// Attention head count (`hparams.n_audio_head`).
    n_head: usize,
    /// Maximum audio context (`hparams.n_audio_ctx`, e.g. 1500).
    n_ctx: usize,
    /// `conv1` weight PRE-TRANSPOSED to `[n_mels*K, n_state]` (the `[Cin*K, Cout]` layout
    /// `nn::conv1d_wt` / `matmul_bias` consume) + bias `[n_state]`. Transposed once at load
    /// so the per-window encode never re-transposes it.
    conv1_wt: Mat,
    conv1_b: Vec<f32>,
    /// `conv2` weight PRE-TRANSPOSED to `[n_state*K, n_state]` + bias `[n_state]`.
    conv2_wt: Mat,
    conv2_b: Vec<f32>,
    /// Positional embedding `[n_ctx, n_state]` (file tensor, row-major).
    pos_emb: Mat,
    /// Per-layer transformer weights.
    layers: Vec<EncoderLayer>,
    /// Final `ln_post` scale/shift, length `n_state`.
    ln_post_w: Vec<f32>,
    ln_post_b: Vec<f32>,
}

/// Decoder-owned cross-attention K/V cache (see module docs).
///
/// This type only documents the encoder→decoder boundary: the actual
/// projection uses *decoder* weights and is implemented in the decoder bead.
/// It is exposed here purely so the boundary has a name; the encoder never
/// constructs one.
#[derive(Debug, Clone, Default)]
pub struct CrossKv {
    /// Per-layer cross keys (decoder fills these from the encoder output).
    pub k: Vec<Mat>,
    /// Per-layer cross values.
    pub v: Vec<Mat>,
}

/// Fetch a tensor and assert its logical (row-major) shape, naming it on error.
fn load_shaped(model: &GgmlModel, name: &str, want: &[usize]) -> FwResult<Vec<f32>> {
    let (shape, data) = model.tensor_f32(name)?;
    if shape != want {
        return Err(FwError::InvalidRequest(format!(
            "encoder tensor '{name}' has shape {shape:?}, expected {want:?}"
        )));
    }
    Ok(data)
}

/// Load a 1-D tensor (e.g. a bias / layer-norm vector) of length `len`.
///
/// ggml stores some vectors as a genuine 1-D `[len]` and others (notably the
/// conv biases) as `[len, 1]`. Both describe the same `len` contiguous f32s,
/// so we accept either: the element count must equal `len` and, when 2-D, the
/// trailing dims must all be `1`. Any other shape names the tensor in the
/// error.
fn load_vec(model: &GgmlModel, name: &str, len: usize) -> FwResult<Vec<f32>> {
    let (shape, data) = model.tensor_f32(name)?;
    let n_elements: usize = shape.iter().product();
    let trailing_ones = shape.iter().skip(1).all(|&d| d == 1);
    let leading_ok = shape.first().copied() == Some(len);
    if n_elements != len || !leading_ok || !trailing_ones {
        return Err(FwError::InvalidRequest(format!(
            "encoder tensor '{name}' has shape {shape:?}, expected a length-{len} vector"
        )));
    }
    Ok(data)
}

/// Load a ggml linear weight `[out, in]` and pre-transpose it to `[in, out]`.
///
/// The returned [`Mat`] is `[in, out]`, ready for [`nn::matmul_bias`].
fn load_linear_transposed(
    model: &GgmlModel,
    name: &str,
    out_dim: usize,
    in_dim: usize,
) -> FwResult<Mat> {
    // FUSED dequant-transpose (cc, 2026-06-29): read the raw f16 bytes straight
    // from the blob and convert to f32 DIRECTLY into the transposed `[in, out]`
    // slot in ONE tiled pass — no intermediate `Vec<u16>` (the old `tensor_f16`
    // copy) and no separate transpose read. MEASURED 1.33× vs the `Vec<u16>` path
    // on the large encoder load (238→179 ms): one fewer linear pass over the
    // ~1.25 GB of encoder weights on this bandwidth-bound load, plus no per-weight
    // allocation. Bit-identical to dequant-then-`transpose_serial` (same
    // `Float16::from_bits` of the same LE byte pairs, just written transposed).
    // f32-stored tensors keep the two-step f32 path (nothing to dequantize).
    if let Ok((shape, raw)) = model.tensor_f16_bytes(name) {
        if shape != [out_dim, in_dim] {
            return Err(FwError::InvalidRequest(format!(
                "encoder tensor '{name}' has shape {shape:?}, expected {:?}",
                [out_dim, in_dim]
            )));
        }
        return Ok(Mat::from_vec(
            in_dim,
            out_dim,
            dequant_transpose_f16_bytes(raw, out_dim, in_dim),
        ));
    }

    // f32-stored fallback. SERIAL transpose: `from_ggml` parallelizes across
    // layers (rayon), so a per-weight `thread::scope` transpose here would nest
    // and spawn-thrash. The coarse (layer) parallelism keeps all cores busy.
    let data = load_shaped(model, name, &[out_dim, in_dim])?;
    Ok(Mat::from_vec(
        in_dim,
        out_dim,
        nn::transpose_serial(&data, out_dim, in_dim),
    ))
}

/// Weight-quant mode for the encoder `attn.out` linear (the residual-feeding one):
/// f32 sgemm, i7 maddubs, or the full-i8 i32-accumulate GEMM.
#[derive(Clone, Copy)]
enum OutQuant {
    F32,
    I7,
    I8,
}

/// Per-linear encoder weight-quant plan, derived ONCE from the `FW_*` flags +
/// hparams before the parallel layer load, so each linear can be quantized the
/// instant it is loaded (fusing the former separate post-load quant pass). Its f32
/// is then a per-linear transient (freed immediately when `free`) instead of the
/// whole f32 weight set being resident until a post-load quant — which is what lets
/// PEAK RSS drop below the full f32 floor. See [`load_linear_maybe_i7`].
#[derive(Clone, Copy)]
struct EncQuantPlan {
    q: bool,
    k: bool,
    v: bool,
    out: OutQuant,
    fc: bool,
    proj: bool,
}

/// Load a linear's f32 `[in, out]` weight and, when `to_i7`, quantize it to i7. When
/// `free`, the f32 is dropped in place (an empty `Mat` is returned) so it never
/// outlives this call — bounding the transient f32 to ~one linear per rayon worker
/// during the parallel layer load. Byte-exact: identical `load_linear_transposed`
/// f32 and identical `nn::quantize_mat_to_i7` to the historical two-phase path; only
/// the f32's lifetime differs. When `!free` (flag off / macOS / roundtrip harness)
/// the f32 is retained exactly as before, so the shipping default is unchanged.
fn load_linear_maybe_i7(
    model: &GgmlModel,
    name: &str,
    out_dim: usize,
    in_dim: usize,
    to_i7: bool,
    free: bool,
) -> FwResult<(Mat, Option<nn::I7Mat>)> {
    let w = load_linear_transposed(model, name, out_dim, in_dim)?;
    if !to_i7 {
        return Ok((w, None));
    }
    let i7 = nn::quantize_mat_to_i7(&w);
    let w = if free { Mat::from_vec(0, 0, Vec::new()) } else { w };
    Ok((w, Some(i7)))
}

/// Fused dequant-transpose reading raw little-endian f16 bytes (`raw`,
/// row-major `[rows, cols]` = ggml's `[out, in]`) DIRECTLY — no `Vec<u16>`
/// intermediate. Output is row-major `[cols, rows]` (`[in, out]`) f32, ready for
/// [`nn::matmul_bias`]. The 64×64 tiling keeps the strided read/write in cache
/// exactly as [`nn::transpose_serial`]; bit-identical to dequantizing then
/// transposing (`Float16::from_bits(le u16)` per element).
fn dequant_transpose_f16_bytes(raw: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(raw.len(), rows * cols * 2, "transpose byte/shape mismatch");
    const TILE: usize = 64;
    let mut out = vec![0.0f32; rows * cols];
    for r0 in (0..rows).step_by(TILE) {
        let r1 = (r0 + TILE).min(rows);
        for c0 in (0..cols).step_by(TILE) {
            let c1 = (c0 + TILE).min(cols);
            for r in r0..r1 {
                let src_row = r * cols;
                for c in c0..c1 {
                    let i = (src_row + c) * 2;
                    let bits = u16::from_le_bytes([raw[i], raw[i + 1]]);
                    out[c * rows + r] = Float16::from_bits(bits).to_f32();
                }
            }
        }
    }
    out
}

/// Transpose a row-major `[rows, cols]` buffer into a `[cols, rows]` [`Mat`].
///
/// Used to flip ggml's `[out, in]` linear weights into the `[in, out]` layout
/// [`nn::matmul_bias`] requires, and to turn the mel-major encoder input into
/// the time-major `[T, Cin]` layout [`nn::conv1d`] expects. Kept private to
/// this module: `mod.rs`/`nn.rs` are owned by other beads.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Mat {
    Mat::from_vec(cols, rows, nn::transpose_parallel(data, rows, cols))
}

/// Convert a compact mel-major encoder window to time-major conv input.
///
/// This is the exact preparation [`forward`] historically performed after
/// [`super::mel::chunk_frames`]: `[n_mels, n_frames]` mel-major in, `[n_frames,
/// n_mels]` time-major out.
#[must_use]
pub fn time_major_mel_window(mel_window: &Mel) -> Mat {
    transpose(&mel_window.data, mel_window.n_mel, mel_window.n_frames)
}

/// Slice a window from a full mel spectrogram directly into time-major layout.
///
/// This is equivalent to `time_major_mel_window(&mel::chunk_frames(...))`, but
/// skips materializing the intermediate compact mel-major [`Mel`] window. Frames
/// beyond `full_mel.n_frames` are filled with [`mel::SILENCE_FLOOR`], matching
/// [`super::mel::chunk_frames`].
#[must_use]
pub fn time_major_mel_window_from_full_mel(
    full_mel: &Mel,
    frame_offset: usize,
    n_frames: usize,
) -> Mat {
    let copy_frames = full_mel.n_frames.saturating_sub(frame_offset).min(n_frames);
    let fill = if copy_frames == n_frames {
        0.0
    } else {
        super::mel::SILENCE_FLOOR
    };
    let mut data = vec![fill; n_frames * full_mel.n_mel];

    const FRAME_TILE: usize = 64;
    const MEL_TILE: usize = 80;
    for f0 in (0..copy_frames).step_by(FRAME_TILE) {
        let f1 = (f0 + FRAME_TILE).min(copy_frames);
        for m0 in (0..full_mel.n_mel).step_by(MEL_TILE) {
            let m1 = (m0 + MEL_TILE).min(full_mel.n_mel);
            for m in m0..m1 {
                let src = m * full_mel.n_frames + frame_offset + f0;
                let row = &full_mel.data[src..src + (f1 - f0)];
                for (df, &v) in row.iter().enumerate() {
                    data[(f0 + df) * full_mel.n_mel + m] = v;
                }
            }
        }
    }

    Mat::from_vec(n_frames, full_mel.n_mel, data)
}

fn validate_mel_window_shape(w: &EncoderWeights, n_mel: usize, n_frames: usize) -> FwResult<()> {
    if n_mel != w.n_mels {
        return Err(FwError::InvalidRequest(format!(
            "encoder: mel has {} channels, model expects {}",
            n_mel, w.n_mels
        )));
    }
    // The conv stem (stride-2 conv2) halves time, so the frame count must be
    // even and at most `2 * n_audio_ctx`. The full-window case is the common
    // `2 * 1500 = 3000`; a smaller even count is the tail-window truncation
    // (whisper.cpp `audio_ctx`: conv input is `2*n_ctx` wide, 1982/1995). An
    // odd count would yield a fractional ctx; an oversized one would overrun
    // the positional embedding (re-checked after conv below).
    let max_frames = 2 * w.n_ctx;
    if n_frames == 0 || !n_frames.is_multiple_of(2) || n_frames > max_frames {
        return Err(FwError::InvalidRequest(format!(
            "encoder: mel window has {n_frames} frames, expected a positive even count \
             ≤ {max_frames} (= 2*n_audio_ctx; use mel::chunk_frames)",
        )));
    }
    Ok(())
}

impl EncoderWeights {
    /// The encoder embedding width (`n_audio_state`).
    #[must_use]
    pub fn n_state(&self) -> usize {
        self.n_state
    }

    /// The number of transformer layers (`n_audio_layer`).
    #[must_use]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Load and pre-transpose every encoder tensor from a parsed ggml model.
    ///
    /// Validates each tensor's shape against the model hyper-parameters and
    /// returns a tensor-named [`FwError::InvalidRequest`] on any mismatch, so
    /// a corrupt or unexpected model fails immediately rather than silently
    /// mis-computing.
    ///
    /// # Errors
    /// - [`FwError::InvalidRequest`] if hyper-parameters are non-positive /
    ///   inconsistent (e.g. `n_head` does not divide `n_state`), or any tensor
    ///   is missing or mis-shaped.
    /// - Propagates [`super::ggml::GgmlModel::tensor_f32`] decode errors.
    pub fn from_ggml(model: &GgmlModel) -> FwResult<Self> {
        let hp: &WhisperHParams = &model.hparams;
        let n_mels = positive(hp.n_mels, "n_mels")?;
        let n_state = positive(hp.n_audio_state, "n_audio_state")?;
        let n_head = positive(hp.n_audio_head, "n_audio_head")?;
        let n_layer = positive(hp.n_audio_layer, "n_audio_layer")?;
        let n_ctx = positive(hp.n_audio_ctx, "n_audio_ctx")?;

        if !n_state.is_multiple_of(n_head) {
            return Err(FwError::InvalidRequest(format!(
                "encoder: n_audio_head {n_head} does not divide n_audio_state {n_state}"
            )));
        }
        let mlp_hidden = n_state * MLP_RATIO;

        // Conv stem: ggml shapes are [Cout, Cin, K] = flat [Cout, Cin*K]. Pre-transpose
        // ONCE here to [Cin*K, Cout] (what `nn::conv1d_wt`/`matmul_bias` consume) so the
        // per-window encode skips the (redundant, ~15 ms/window on turbo conv2) transpose.
        // `transpose_serial([Cout, Cin*K])` is bit-identical to conv1d's inline transpose.
        let conv1_patch = n_mels * CONV_K;
        let conv1_raw = load_shaped(model, "encoder.conv1.weight", &[n_state, n_mels, CONV_K])?;
        let conv1_wt = Mat::from_vec(
            conv1_patch,
            n_state,
            nn::transpose_serial(&conv1_raw, n_state, conv1_patch),
        );
        let conv1_b = load_vec(model, "encoder.conv1.bias", n_state)?;
        let conv2_patch = n_state * CONV_K;
        let conv2_raw = load_shaped(model, "encoder.conv2.weight", &[n_state, n_state, CONV_K])?;
        let conv2_wt = Mat::from_vec(
            conv2_patch,
            n_state,
            nn::transpose_serial(&conv2_raw, n_state, conv2_patch),
        );
        let conv2_b = load_vec(model, "encoder.conv2.bias", n_state)?;

        // Positional embedding: file tensor [n_ctx, n_state], used verbatim.
        let pos_data = load_shaped(model, "encoder.positional_embedding", &[n_ctx, n_state])?;
        let pos_emb = Mat::from_vec(n_ctx, n_state, pos_data);

        // Build the per-block weights ACROSS layers in parallel (rayon's
        // persistent pool). The dominant load cost is the per-weight transpose
        // (`model_weights` ≈ 1.97 s on large = 32 layers); each layer is
        // independent, reads disjoint tensors from the (shared, read-only)
        // `model`, and now transposes SERIALLY, so this fans the 32 layers across
        // cores with no nested spawn. Order is preserved (`map`+`collect`), so the
        // assembled weights are byte-identical to the serial loop.
        // FW_ENC_FREE_F32 (default OFF): fuse the encoder weight quant INTO the
        // parallel layer load. Each linear is quantized the instant it is loaded and
        // its f32 dropped in place (`free_f32_now`), so at most ~one f32 linear per
        // rayon worker is transient — instead of the FULL f32 weight set (~2.5 GB
        // turbo) staying resident until a separate post-load quant pass. That drops
        // PEAK RSS below the f32 floor the earlier interleaved post-load free
        // (2ac1257) could not touch. Byte-exact: identical load + identical quantize
        // funcs. Retaining the f32 (flag off / macOS / weight-roundtrip harness)
        // reproduces the previous behavior exactly, so the shipping default (flag off)
        // is unchanged — the f32 is only dropped early under the opt-in flag.
        #[cfg(target_os = "macos")]
        let free_f32_now = false;
        #[cfg(not(target_os = "macos"))]
        let free_f32_now = super::enc_free_f32() && super::enc_weight_roundtrip().is_none();
        // Which linears get which quant — the EXACT per-branch policy the former
        // post-load pass applied (see the flag docs at each `enc_linear` call site).
        let plan = if super::enc_int8_enabled() {
            // FRANKEN_WHISPER_ENC_INT8: every linear i7.
            EncQuantPlan { q: true, k: true, v: true, out: OutQuant::I7, fc: true, proj: true }
        } else if super::enc_int8_attn_in() {
            // FW_ENC_INT8_ATTN_IN: q/k/v/fc1 i7, attn_out + fc2 stay f32.
            EncQuantPlan { q: true, k: true, v: true, out: OutQuant::F32, fc: true, proj: false }
        } else if super::enc_attn_out_i8i32_for(&model.hparams) {
            // Default quality-safe int8 for calibrated models: q/k/v/fc1/fc2 i7,
            // residual-feeding attn_out through the full-i8 i32-accumulate GEMM.
            EncQuantPlan { q: true, k: true, v: true, out: OutQuant::I8, fc: true, proj: true }
        } else if super::enc_int8_fc1_only() {
            // FW_ENC_INT8_FC1: fc1 only (GELU absorbs the quant error).
            EncQuantPlan { q: false, k: false, v: false, out: OutQuant::F32, fc: true, proj: false }
        } else {
            // No int8: all f32, byte-identical to the pre-lever encoder.
            EncQuantPlan { q: false, k: false, v: false, out: OutQuant::F32, fc: false, proj: false }
        };

        let mut layers = (0..n_layer)
            .into_par_iter()
            .map(|i| -> FwResult<EncoderLayer> {
                let p = |suffix: &str| format!("encoder.blocks.{i}.{suffix}");
                let (attn_q_w, attn_q_i7) = load_linear_maybe_i7(
                    model, &p("attn.query.weight"), n_state, n_state, plan.q, free_f32_now,
                )?;
                // whisper key projection has NO bias.
                let (attn_k_w, attn_k_i7) = load_linear_maybe_i7(
                    model, &p("attn.key.weight"), n_state, n_state, plan.k, free_f32_now,
                )?;
                let (attn_v_w, attn_v_i7) = load_linear_maybe_i7(
                    model, &p("attn.value.weight"), n_state, n_state, plan.v, free_f32_now,
                )?;
                let (attn_out_w, attn_out_i7, attn_out_i8) = match plan.out {
                    OutQuant::F32 => (
                        load_linear_transposed(model, &p("attn.out.weight"), n_state, n_state)?,
                        None,
                        None,
                    ),
                    OutQuant::I7 => {
                        let (w, i7) = load_linear_maybe_i7(
                            model, &p("attn.out.weight"), n_state, n_state, true, free_f32_now,
                        )?;
                        (w, i7, None)
                    }
                    OutQuant::I8 => {
                        let w =
                            load_linear_transposed(model, &p("attn.out.weight"), n_state, n_state)?;
                        let i8 = quantize_enc_i8(&w);
                        let w = if free_f32_now { Mat::from_vec(0, 0, Vec::new()) } else { w };
                        (w, None, Some(i8))
                    }
                };
                let (mlp_fc_w, mlp_fc_i7) = load_linear_maybe_i7(
                    model, &p("mlp.0.weight"), mlp_hidden, n_state, plan.fc, free_f32_now,
                )?;
                let (mlp_proj_w, mlp_proj_i7) = load_linear_maybe_i7(
                    model, &p("mlp.2.weight"), n_state, mlp_hidden, plan.proj, free_f32_now,
                )?;
                Ok(EncoderLayer {
                    attn_ln_w: load_vec(model, &p("attn_ln.weight"), n_state)?,
                    attn_ln_b: load_vec(model, &p("attn_ln.bias"), n_state)?,
                    attn_q_w,
                    attn_q_b: load_vec(model, &p("attn.query.bias"), n_state)?,
                    attn_k_w,
                    attn_v_w,
                    attn_v_b: load_vec(model, &p("attn.value.bias"), n_state)?,
                    attn_out_w,
                    attn_out_b: load_vec(model, &p("attn.out.bias"), n_state)?,
                    mlp_ln_w: load_vec(model, &p("mlp_ln.weight"), n_state)?,
                    mlp_ln_b: load_vec(model, &p("mlp_ln.bias"), n_state)?,
                    mlp_fc_w,
                    mlp_fc_b: load_vec(model, &p("mlp.0.bias"), mlp_hidden)?,
                    mlp_proj_w,
                    mlp_proj_b: load_vec(model, &p("mlp.2.bias"), n_state)?,
                    attn_q_i7,
                    attn_k_i7,
                    attn_v_i7,
                    attn_out_i7,
                    mlp_fc_i7,
                    mlp_proj_i7,
                    attn_out_i8,
                })
            })
            .collect::<FwResult<Vec<_>>>()?;

        // (Weight quant is now FUSED into the parallel layer load above — see the
        // `EncQuantPlan` / `load_linear_maybe_i7` block — so the former separate
        // post-load `par_iter_mut` quant pass, and the interleaved post-load free it
        // grew into, are gone. Fusing lets FW_ENC_FREE_F32 free each f32 while only
        // ~one linear per worker is transient, dropping PEAK below the f32 floor.)

        // bd-bcm7: enable ft_kernel_cpu's poly softmax for large-v3-turbo (proven WER-neutral:
        // byte-identical transcript, WER Δ 0.000, 1.0722× e2e). tiny.en stays off (uncertified).
        // Kill-switch FW_SDPA_POLY_EXP=0; operator force FT_SDPA_POLY_EXP=1.
        super::configure_sdpa_poly_exp(&model.hparams);

        // FEASIBILITY HARNESS (off by default): `FW_ENC_WEIGHT_ROUNDTRIP=row|<N>` replaces
        // every f32 GEMM weight with its i7 quantize→dequantize roundtrip, so the EXISTING
        // f32 encoder measures the WEIGHT-quant-granularity effect on the transcript.
        // `row` = per-output-column scale (current int8 encoder granularity); `<N>` = block
        // size along the contraction dim (e.g. `32` = the proposed block-wise scheme). Lets a
        // track01 A/B answer "does block-wise recover the int8 encoder's proper-noun errors?"
        // WITHOUT the block-wise maddubs kernel. Run with FRANKEN_WHISPER_ENC_INT8 unset.
        if let Some(mode) = super::enc_weight_roundtrip() {
            let block = mode; // None => per-column, Some(n) => n-block
            layers.par_iter_mut().for_each(|l| {
                l.attn_q_w = nn::i7_roundtrip(&l.attn_q_w, block);
                l.attn_k_w = nn::i7_roundtrip(&l.attn_k_w, block);
                l.attn_v_w = nn::i7_roundtrip(&l.attn_v_w, block);
                l.attn_out_w = nn::i7_roundtrip(&l.attn_out_w, block);
                l.mlp_fc_w = nn::i7_roundtrip(&l.mlp_fc_w, block);
                l.mlp_proj_w = nn::i7_roundtrip(&l.mlp_proj_w, block);
            });
        }

        let ln_post_w = load_vec(model, "encoder.ln_post.weight", n_state)?;
        let ln_post_b = load_vec(model, "encoder.ln_post.bias", n_state)?;

        Ok(Self {
            n_mels,
            n_state,
            n_head,
            n_ctx,
            conv1_wt,
            conv1_b,
            conv2_wt,
            conv2_b,
            pos_emb,
            layers,
            ln_post_w,
            ln_post_b,
        })
    }
}

/// Convert a positive ggml hyper-parameter `i32` to `usize`, naming it on error.
fn positive(value: i32, what: &str) -> FwResult<usize> {
    if value <= 0 {
        return Err(FwError::InvalidRequest(format!(
            "encoder hparam '{what}' must be positive, got {value}"
        )));
    }
    Ok(value as usize)
}

/// Run the whisper audio encoder over one 30 s mel window.
///
/// `mel_window` must be the model's mel-major spectrogram for a single window:
/// `[n_mels, n_frames]` (slice with [`super::mel::chunk_frames`]). `n_frames`
/// is normally the full `FRAMES_PER_CHUNK = 3000` (30 s); it may also be a
/// **smaller even** count `2*enc_ctx` for the **tail-window truncation**
/// optimization (mirrors whisper.cpp's `audio_ctx` / `-ac` feature, where the
/// conv input is `2*n_ctx` wide with `n_ctx = exp_n_audio_ctx`; whisper.cpp
/// 1982/1995). The output is the `[n_ctx, n_state]` acoustic embedding (e.g.
/// `[1500, 384]` for a full tiny.en window, or `[enc_ctx, 384]` for a
/// truncated tail), reused across every decoder token of this window.
///
/// `n_threads_hint` is currently informational: the heavy matmuls run on
/// FrankenTorch's internally-rayon-parallel sgemm via [`nn`], which manages
/// its own thread pool. The parameter is kept for forward-compatibility and a
/// stable signature.
///
/// `checkpoint` is invoked **between** transformer layers (the cancellation
/// contract; see [`nn`] module docs): returning `Err` aborts the forward pass
/// promptly with that error, so a cancelled pipeline doesn't pay for the
/// remaining layers.
///
/// # Errors
/// - [`FwError::InvalidRequest`] if the mel channel count does not match the
///   model (`n_mels`), the frame count is not a positive even number
///   `≤ 2 * n_audio_ctx` (the conv stem halves time, so an odd count would
///   produce a fractional ctx and a count `> 2*n_ctx` would overrun the
///   positional embedding), or if any inner op rejects a shape.
/// - Whatever error `checkpoint` returns (e.g. [`FwError::Cancelled`]).
pub fn forward(
    w: &EncoderWeights,
    mel_window: &Mel,
    n_threads_hint: usize,
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<Mat> {
    let _ = n_threads_hint; // ft kernels manage their own rayon pool.

    validate_mel_window_shape(w, mel_window.n_mel, mel_window.n_frames)?;

    // mel is mel-major [n_mel, n_frames]; conv1d wants time-major [T, Cin].
    let x = time_major_mel_window(mel_window);

    forward_time_major(w, x, checkpoint)
}

/// Run the whisper audio encoder over a window sliced from a full mel buffer.
///
/// This is numerically equivalent to:
///
/// ```text
/// let mel_window = mel::chunk_frames(full_mel, frame_offset, n_frames);
/// encoder::forward(w, &mel_window, n_threads_hint, checkpoint)
/// ```
///
/// but fuses the window slice with the encoder's required mel-major to
/// time-major transpose, avoiding an intermediate compact mel buffer in the
/// decode loop.
pub fn forward_from_full_mel_window(
    w: &EncoderWeights,
    full_mel: &Mel,
    frame_offset: usize,
    n_frames: usize,
    n_threads_hint: usize,
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<Mat> {
    let _ = n_threads_hint; // ft kernels manage their own rayon pool.

    validate_mel_window_shape(w, full_mel.n_mel, n_frames)?;
    let x = time_major_mel_window_from_full_mel(full_mel, frame_offset, n_frames);

    forward_time_major(w, x, checkpoint)
}

// ── THROWAWAY encoder sub-op profiler (gated on FRANKEN_WHISPER_PERF_SPANS) ──
// Accumulates wall-time per sub-op across all layers into a thread-local, emitted
// once at the end of forward_time_major. Zero cost when perf spans are off.
thread_local! {
    static ENC_PROF: std::cell::RefCell<[u128; 12]> = const { std::cell::RefCell::new([0; 12]) };
}
const ENC_PROF_LABELS: [&str; 12] = [
    "conv_stem",
    "pos_emb",
    "attn_ln",
    "qkv_proj",
    "attn_sdpa",
    "attn_out",
    "attn_resid",
    "mlp_ln",
    "mlp_fc",
    "gelu",
    "mlp_proj",
    "mlp_resid",
];
fn enc_prof_add(i: usize, ns: u128) {
    ENC_PROF.with(|p| p.borrow_mut()[i] += ns);
}

fn forward_time_major(
    w: &EncoderWeights,
    x: Mat,
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<Mat> {
    let measure = crate::native_engine::perf_spans_enabled();
    macro_rules! et {
        ($i:expr, $b:expr) => {{
            if measure {
                let __t = std::time::Instant::now();
                let __r = $b;
                enc_prof_add($i, __t.elapsed().as_nanos());
                __r
            } else {
                $b
            }
        }};
    }
    // conv1: [3000, n_mel] -> [3000, n_state], +gelu.
    let x = et!(0, {
        let mut x = nn::conv1d_wt(&x, &w.conv1_wt, w.n_mels, CONV_K, &w.conv1_b, 1, CONV_PAD)?;
        nn::gelu(&mut x);
        x
    });

    // conv2 (stride 2): [3000, n_state] -> [1500, n_state], +gelu.
    let mut x = et!(0, {
        let mut x = nn::conv1d_wt(&x, &w.conv2_wt, w.n_state, CONV_K, &w.conv2_b, 2, CONV_PAD)?;
        nn::gelu(&mut x);
        x
    });

    let n_ctx = x.rows;
    if n_ctx > w.n_ctx {
        return Err(FwError::InvalidRequest(format!(
            "encoder: conv produced {n_ctx} ctx rows > positional embedding capacity {}",
            w.n_ctx
        )));
    }

    // Add positional embedding (file tensor), sliced to the first n_ctx rows.
    et!(1, add_pos_emb(&mut x, &w.pos_emb, n_ctx));

    // Residual transformer blocks. On Apple Silicon with a large model the whole
    // stack runs on the GPU (fused: activations resident, one command buffer/sync
    // per layer instead of per-op CPU<->GPU ping-pong); every other case uses the
    // CPU blocks. `FRANKEN_WHISPER_GPU=0` forces the CPU path.
    if !gpu_encode_stack(&mut x, w) {
        // Optional depth truncation (`FW_ENCODER_LAYERS=N`): run only the first N
        // of the model's encoder transformer blocks. NON-byte-exact (fewer
        // refinements → different encoder output) — a VIABILITY PROBE for encoder
        // layer-pruning, default = all layers (byte-identical). If a truncated
        // depth keeps the transcript within conformance it is a direct encoder
        // FLOP win (≈ pruned_layers / n_layers of the block stack).
        let n_run = encoder_layer_limit()
            .unwrap_or(w.layers.len())
            .min(w.layers.len());
        for layer in w.layers.iter().take(n_run) {
            encoder_block(&mut x, layer, w.n_head)?;
            checkpoint()?;
        }
    }

    // Final ln_post.
    nn::layer_norm(&mut x, &w.ln_post_w, &w.ln_post_b, LN_EPS);

    if measure {
        ENC_PROF.with(|p| {
            let a = p.borrow();
            let total: u128 = a.iter().sum();
            eprintln!("--- encoder sub-op breakdown (sum over layers, one window) ---");
            for (i, lbl) in ENC_PROF_LABELS.iter().enumerate() {
                if a[i] > 0 {
                    let ms = a[i] as f64 / 1e6;
                    let pct = if total > 0 {
                        a[i] as f64 / total as f64 * 100.0
                    } else {
                        0.0
                    };
                    eprintln!("  {lbl:<12} {ms:>8.1} ms  {pct:>5.1}%");
                }
            }
            eprintln!(
                "  {:<12} {:>8.1} ms (encoder sub-op total)",
                "SUM",
                total as f64 / 1e6
            );
        });
        ENC_PROF.with(|p| *p.borrow_mut() = [0; 12]);
        // Split attn_sdpa into gather / kernel / scatter (sum over layers).
        let sp = nn::drain_sdpa_split();
        let stot: u128 = sp.iter().sum();
        if stot > 0 {
            let labels = ["sdpa_gather", "sdpa_kernel", "sdpa_scatter"];
            eprintln!("  -- attn_sdpa internal split --");
            for (i, lbl) in labels.iter().enumerate() {
                let ms = sp[i] as f64 / 1e6;
                let pct = sp[i] as f64 / stot as f64 * 100.0;
                eprintln!("  {lbl:<14} {ms:>8.1} ms  {pct:>5.1}% of attn_sdpa");
            }
        }
    }

    Ok(x)
}

/// Add the first `n_ctx` rows of the positional embedding into `x` in place.
fn add_pos_emb(x: &mut Mat, pos_emb: &Mat, n_ctx: usize) {
    let cols = x.cols;
    for r in 0..n_ctx {
        let pe = pos_emb.row(r);
        let dst = &mut x.data[r * cols..(r + 1) * cols];
        for (v, &p) in dst.iter_mut().zip(pe) {
            *v += p;
        }
    }
}

/// Run the encoder transformer stack on the GPU (Apple Silicon), keeping
/// activations resident and batching each layer into one command buffer (one
/// sync/layer). Returns `false` — so the caller uses the CPU blocks — off macOS,
/// with no GPU, on a small model, or when disabled via `FRANKEN_WHISPER_GPU=0`.
/// Weights are uploaded once and cached per model. All Metal lives in
/// `ft-kernel-metal`, so this crate keeps `#![deny(unsafe_code)]`.
#[cfg(target_os = "macos")]
fn gpu_encode_stack(x: &mut Mat, w: &EncoderWeights) -> bool {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    // `FRANKEN_WHISPER_FUSED_ENC=0` disables just the fused encoder (the per-matmul
    // GEMM offload in `nn` still applies) — for A/B against the fused path.
    if matches!(
        std::env::var("FRANKEN_WHISPER_FUSED_ENC").ok().as_deref(),
        Some("0")
    ) {
        return false;
    }
    if w.n_state < GPU_ENCODER_MIN_N_STATE || !gpu_encoder_enabled() {
        return false;
    }

    static CACHE: OnceLock<Mutex<HashMap<u64, Arc<ft_kernel_metal::fused::EncoderGpu>>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // Identify the model by a cheap content fingerprint (shape + a few weight
    // samples), NOT by `w`'s address: this static cache outlives the borrowed
    // `EncoderWeights`, so a dropped model replaced by a DIFFERENT model reusing the
    // same address would otherwise alias a stale resident encoder. The fingerprint
    // also lets the same model loaded twice share one upload.
    let key: u64 = {
        fn mix(h: u64, v: u64) -> u64 {
            (h ^ v).wrapping_mul(0x0000_0100_0000_01b3)
        }
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        h = mix(h, w.n_state as u64);
        h = mix(h, w.n_head as u64);
        h = mix(h, w.layers.len() as u64);
        for l in [w.layers.first(), w.layers.last()].into_iter().flatten() {
            for s in [
                l.attn_q_w.data.first(),
                l.attn_q_w.data.last(),
                l.mlp_fc_w.data.first(),
                l.mlp_proj_w.data.last(),
                l.attn_ln_w.first(),
                l.attn_ln_w.last(),
            ]
            .into_iter()
            .flatten()
            {
                h = mix(h, u64::from(s.to_bits()));
            }
        }
        h
    };

    let enc = {
        let mut guard = match cache.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        if !guard.contains_key(&key) {
            let refs: Vec<ft_kernel_metal::fused::LayerWeightsRef> = w
                .layers
                .iter()
                .map(|l| ft_kernel_metal::fused::LayerWeightsRef {
                    ln1_g: &l.attn_ln_w,
                    ln1_b: &l.attn_ln_b,
                    wq: &l.attn_q_w.data,
                    bq: &l.attn_q_b,
                    wk: &l.attn_k_w.data,
                    wv: &l.attn_v_w.data,
                    bv: &l.attn_v_b,
                    wo: &l.attn_out_w.data,
                    bo: &l.attn_out_b,
                    ln2_g: &l.mlp_ln_w,
                    ln2_b: &l.mlp_ln_b,
                    w1: &l.mlp_fc_w.data,
                    b1: &l.mlp_fc_b,
                    w2: &l.mlp_proj_w.data,
                    b2: &l.mlp_proj_b,
                })
                .collect();
            match ft_kernel_metal::fused::EncoderGpu::new(w.n_state, w.n_head, w.n_state * 4, &refs)
            {
                Ok(enc) => {
                    guard.insert(key, Arc::new(enc));
                }
                Err(_) => return false,
            }
        }
        Arc::clone(guard.get(&key).expect("just inserted"))
    };

    match enc.forward(&x.data, x.rows) {
        Ok(out) => {
            x.data = out;
            true
        }
        Err(_) => false,
    }
}

#[cfg(not(target_os = "macos"))]
fn gpu_encode_stack(_x: &mut Mat, _w: &EncoderWeights) -> bool {
    false
}

/// Minimum `n_state` (model width) for the GPU encoder: medium/large whisper
/// models (large-v3 = 1280). Smaller models keep the CPU path — their encoder is
/// already fast and GPU launch overhead would not pay off.
#[cfg(target_os = "macos")]
const GPU_ENCODER_MIN_N_STATE: usize = 1024;

/// Whether the GPU encoder is usable and enabled (probed once, cached).
#[cfg(target_os = "macos")]
fn gpu_encoder_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        let disabled = matches!(
            std::env::var("FRANKEN_WHISPER_GPU").ok().as_deref(),
            Some("0") | Some("off") | Some("false") | Some("no")
        );
        !disabled && ft_kernel_metal::fused::is_available()
    })
}

/// Whether a real GPU encoder path exists in this build, on this machine, right
/// now: the Metal kernels are compiled in, the device reports a usable queue,
/// and `FRANKEN_WHISPER_GPU=0` has not disabled it.
///
/// Model-independent: the per-model [`GPU_ENCODER_MIN_N_STATE`] width gate is a
/// performance policy applied inside [`gpu_encode_stack`], not a statement about
/// whether the engine *can* reach a GPU. Off Apple Silicon this is always
/// `false` — the native engine is CPU-only there, with no CUDA/Vulkan path.
///
/// This is the ground truth behind the native engines' reported
/// `supports_gpu` capability flag.
#[must_use]
pub fn gpu_encoder_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        gpu_encoder_enabled()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// One residual encoder block, mutating `x` (`[n_ctx, n_state]`) in place.
///
/// `x = x + attn_out(attn(ln_attn(x)))` then `x = x + mlp(ln_mlp(x))`. The
/// attention is bidirectional (no causal mask): every output row depends on
/// every input row.
/// Default-ON gate for the fused `layer_norm`-into-uninit path (kill switch
/// `FW_ENCODER_FUSED_LN=0` restores `x.clone()` + in-place `layer_norm`).
fn fused_ln_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FW_ENCODER_FUSED_LN").as_deref() != Ok("0"))
}

/// Default-ON gate (`FW_ENC_FC_BIAS_GELU_FUSED=0` kill-switch) for folding the
/// **f32-path** `mlp_fc` bias-add into the subsequent GELU pass
/// ([`nn::gelu_add_bias`]). `matmul_bias` applies the fc1 bias as a SEPARATE
/// single-threaded RMW over the whole `[n_ctx, mlp_hidden]` output (`[1500,5120]`
/// = ~30 MiB, L3-borderline/DRAM), immediately before the parallel GELU pass over
/// the same buffer; the fold removes that serial pass. BYTE-IDENTICAL. Fires only
/// on the f32 fc1 path (`mlp_fc_i7 == None`) with the separate-GELU branch — the
/// int8 `matmul_bias_i7_gelu` path already folds GELU into fc2's quant and needs
/// the bias resident in `h`, so it is excluded.
fn fc_bias_gelu_fused_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FW_ENC_FC_BIAS_GELU_FUSED").as_deref() != Ok("0"))
}

/// Default-ON gate (`FW_ENC_PROJ_BIAS_RESID_FUSED=0` kill-switch) for folding the
/// **f32-path** `mlp.proj` (fc2) bias-add into the residual add
/// ([`nn::add_bias_residual`]). `matmul_bias` applies the fc2 bias as a SEPARATE
/// single-threaded RMW over the `[n_ctx, n_state]` output; unlike qkv/attn_out
/// (whose sgemm working sets fit L3 ⇒ output cache-warm), fc2's sgemm streams a
/// ~56 MiB working set (`[1500,5120]`+`[5120,1280]`) that evicts its own 7.68 MiB
/// output, so that serial bias pass reads partly-DRAM. Folding it into the
/// residual removes the pass (and the h write-back) and parallelizes it.
/// BYTE-IDENTICAL. Fires only on the f32 fc2 path (`mlp_proj_i7 == None`).
fn proj_bias_resid_fused_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FW_ENC_PROJ_BIAS_RESID_FUSED").as_deref() != Ok("0"))
}

/// Optional encoder-depth cap `FW_ENCODER_LAYERS=N` (viability probe for encoder
/// layer-pruning). `None` (unset / unparsable / `0`) ⇒ run all layers
/// (byte-identical default). Resolved once.
fn encoder_layer_limit() -> Option<usize> {
    use std::sync::OnceLock;
    static N: OnceLock<Option<usize>> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("FW_ENCODER_LAYERS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
    })
}

/// `layer_norm(x)` returning a fresh `[rows, cols]` `Mat`. Fused path writes the
/// normalized rows straight into an uninitialized buffer (no clone memcpy);
/// clone path is the byte-identical legacy `x.clone()` + in-place norm.
#[inline]
fn ln_into(x: &Mat, w: &[f32], b: &[f32]) -> Mat {
    if fused_ln_enabled() {
        let mut data = nn::gemv_out_buf(x.rows * x.cols);
        nn::layer_norm_into(x, &mut data, w, b, LN_EPS);
        Mat::from_vec(x.rows, x.cols, data)
    } else {
        let mut h = x.clone();
        nn::layer_norm(&mut h, w, b, LN_EPS);
        h
    }
}

/// Dispatch a linear layer to the maddubs 7-bit int8 GEMM when an [`nn::I7Mat`]
/// was built at load (`FRANKEN_WHISPER_ENC_INT8=1`), else the f32 sgemm. The
/// default (no i7) path is byte-identical to the pre-lever encoder.
#[inline]
fn enc_linear(x: &Mat, w_t: &Mat, w_i7: &Option<nn::I7Mat>, bias: Option<&[f32]>) -> FwResult<Mat> {
    match w_i7 {
        Some(w) => nn::matmul_bias_i7(x, w, bias),
        None => match super::enc_act_roundtrip() {
            // Feasibility harness: roundtrip the ACTIVATION through the int8 path's u8 quant
            // (per-row or block-wise) before the f32 GEMM, isolating the activation-quant
            // effect on the transcript. Default (None) = the true f32 path, byte-identical.
            Some(block) => nn::matmul_bias(&nn::u8_act_roundtrip(x, block), w_t, bias),
            None => nn::matmul_bias(x, w_t, bias),
        },
    }
}

/// Cached per-output-channel **i8** weight for the encoder `attn.out` GEMM
/// (gated by [`super::enc_attn_out_i8i32`]). `data` is `[out, inp]` row-major i8
/// (per-output-channel amax/127, FULL 8 bits — the extra bit vs franken's i7
/// maddubs is what preserves proper nouns on the residual-feeding `attn.out`;
/// see the flag doc), transposed once at load from franken's `[inp, out]` `w_t`.
#[derive(Debug, Clone)]
struct EncI8Mat {
    data: Vec<i8>,
    scale: Vec<f32>,
    inp: usize,
    out: usize,
}

/// Quantize franken's `[inp, out]` weight (element `[i*out + o]`) to a cached
/// `[out, inp]` i8 matrix, per-output-channel symmetric (amax/127). One-time load
/// cost, parallel over output channels (disjoint rows ⇒ order-invariant).
fn quantize_enc_i8(w_t: &Mat) -> EncI8Mat {
    let inp = w_t.rows;
    let out = w_t.cols;
    let mut data = vec![0i8; out * inp];
    let mut scale = vec![0.0f32; out];
    data.par_chunks_mut(inp)
        .zip(scale.par_iter_mut())
        .enumerate()
        .for_each(|(o, (drow, s))| {
            let mut amax = 1e-9f32;
            for i in 0..inp {
                amax = amax.max(w_t.data[i * out + o].abs());
            }
            let sc = amax / 127.0;
            *s = sc;
            let inv = 1.0 / sc;
            for (i, d) in drow.iter_mut().enumerate() {
                *d = (w_t.data[i * out + o] * inv).round().clamp(-127.0, 127.0) as i8;
            }
        });
    EncI8Mat {
        data,
        scale,
        inp,
        out,
    }
}

/// AVX2 i8×i8 → i32 dot (vpmovsxbw + vpmaddwd, 2 accumulators; sign-extend both
/// operands so there is NO `_mm256_maddubs_epi16` i16-saturation constraint —
/// this is why the weight can be full i8, unlike franken's i7 maddubs). A private
/// copy of `nn::dot_i8` (which is not `pub`), kept self-contained in encoder.rs to
/// avoid touching the shared-tree `nn.rs`. Integer-exact.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8_enc(w: &[i8], x: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    let n = w.len();
    let (wp, xp) = (w.as_ptr(), x.as_ptr());
    // SAFETY: avx2 is a base target feature; every 128-bit load is bounded by the
    // `i+32<=n` / `i+16<=n` guards; the `<16` tail runs scalar. Bit-identical to
    // the scalar reduction (integer add is order-independent).
    unsafe {
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i) as *const __m128i));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16) as *const __m128i));
            let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i + 16) as *const __m128i));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, x1));
            i += 32;
        }
        while i + 16 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i) as *const __m128i));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
            i += 16;
        }
        let s = _mm256_add_epi32(a0, a1);
        let lo = _mm256_castsi256_si128(s);
        let hi = _mm256_extracti128_si256::<1>(s);
        let q = _mm_add_epi32(lo, hi);
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
        let mut acc = _mm_cvtsi128_si32(q);
        while i < n {
            acc += (*w.get_unchecked(i) as i32) * (*x.get_unchecked(i) as i32);
            i += 1;
        }
        acc
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot_i8_enc(w: &[i8], x: &[i8]) -> i32 {
    w.iter().zip(x).map(|(&a, &b)| a as i32 * b as i32).sum()
}

/// M4×N2 register-blocked i8×i8 → i32: 4 activation rows × 2 weight rows = 8 dots,
/// each 16-i8 chunk sign-extended ONCE and reused across all 8 dots (8 accumulators
/// + 4 act + 2 weight = 14 ymm, fits Zen3's 16). This amortizes the vpmovsxbw
/// sign-extend + the loads that make the per-call [`dot_i8_enc`] effectively M1
/// (it re-loads+re-extends the weight row for every activation row). Mirrors the
/// maddubs `dot_maddubs_i7_m4n2`; integer-EXACT (associative i32 add ⇒ bit-identical
/// to per-element order). Returns `[x0·w0,x1·w0,x2·w0,x3·w0, x0·w1,x1·w1,x2·w1,x3·w1]`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code, clippy::too_many_arguments)]
fn dot_i8_m4n2(x0: &[i8], x1: &[i8], x2: &[i8], x3: &[i8], w0: &[i8], w1: &[i8]) -> [i32; 8] {
    use core::arch::x86_64::*;
    let k = w0.len();
    let (p0, p1, p2, p3) = (x0.as_ptr(), x1.as_ptr(), x2.as_ptr(), x3.as_ptr());
    let (q0, q1) = (w0.as_ptr(), w1.as_ptr());
    // SAFETY: avx2 base feature; all six slices have length k; 16-lane steps guarded
    // by `i+16<=k`, scalar tail after. vpmaddwd accumulates i16→i32 (no saturation).
    unsafe {
        let mut acc = [_mm256_setzero_si256(); 8];
        let mut i = 0;
        while i + 16 <= k {
            let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p0.add(i) as *const __m128i));
            let a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p1.add(i) as *const __m128i));
            let a2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p2.add(i) as *const __m128i));
            let a3 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p3.add(i) as *const __m128i));
            let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(q0.add(i) as *const __m128i));
            let b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(q1.add(i) as *const __m128i));
            acc[0] = _mm256_add_epi32(acc[0], _mm256_madd_epi16(a0, b0));
            acc[1] = _mm256_add_epi32(acc[1], _mm256_madd_epi16(a1, b0));
            acc[2] = _mm256_add_epi32(acc[2], _mm256_madd_epi16(a2, b0));
            acc[3] = _mm256_add_epi32(acc[3], _mm256_madd_epi16(a3, b0));
            acc[4] = _mm256_add_epi32(acc[4], _mm256_madd_epi16(a0, b1));
            acc[5] = _mm256_add_epi32(acc[5], _mm256_madd_epi16(a1, b1));
            acc[6] = _mm256_add_epi32(acc[6], _mm256_madd_epi16(a2, b1));
            acc[7] = _mm256_add_epi32(acc[7], _mm256_madd_epi16(a3, b1));
            i += 16;
        }
        let hsum = |v: __m256i| -> i32 {
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256::<1>(v);
            let q = _mm_add_epi32(lo, hi);
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
            _mm_cvtsi128_si32(q)
        };
        let mut r = [0i32; 8];
        for (j, a) in acc.iter().enumerate() {
            r[j] = hsum(*a);
        }
        while i < k {
            let (wx0, wx1) = (w0[i] as i32, w1[i] as i32);
            r[0] += x0[i] as i32 * wx0;
            r[1] += x1[i] as i32 * wx0;
            r[2] += x2[i] as i32 * wx0;
            r[3] += x3[i] as i32 * wx0;
            r[4] += x0[i] as i32 * wx1;
            r[5] += x1[i] as i32 * wx1;
            r[6] += x2[i] as i32 * wx1;
            r[7] += x3[i] as i32 * wx1;
            i += 1;
        }
        r
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot_i8_m4n2(x0: &[i8], x1: &[i8], x2: &[i8], x3: &[i8], w0: &[i8], w1: &[i8]) -> [i32; 8] {
    [
        dot_i8_enc(w0, x0),
        dot_i8_enc(w0, x1),
        dot_i8_enc(w0, x2),
        dot_i8_enc(w0, x3),
        dot_i8_enc(w1, x0),
        dot_i8_enc(w1, x1),
        dot_i8_enc(w1, x2),
        dot_i8_enc(w1, x3),
    ]
}

/// Byte-exact AVX2 symmetric-i8 activation quant `(v*inv).round().clamp(-127,127) as i8`.
/// `f32::round` has no AVX rounding mode (LLVM scalarizes to a per-element `roundf`), so the
/// AVX2 path emulates round-half-away via `+ copysign(0.5, v)` + round-to-zero, then clamps
/// and order-preserving-packs. (Unlike the common `trunc(v+0.5)` emulation — e.g. the one in
/// `nn::quantize_act_i8_into` — this is byte-EXACT: that shortcut mis-rounds `v` just below 0.5.)
/// MEASURED ~2.2–2.7× on the m=1500 encoder shape (`examples/enc_i8quant_probe`), and
/// unit-tested byte-identical to the scalar map over the ±127 clamp / half-away edges.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
fn quant_row_i8(xr: &[f32], inv: f32, out: &mut [i8]) {
    use core::arch::x86_64::*;
    let n = xr.len().min(out.len());
    let xp = xr.as_ptr();
    // SAFETY: avx2 guaranteed by cfg; every load/store is bounded by the `i+8<=n` guard,
    // and the `< 8` remainder runs the scalar map.
    unsafe {
        let vinv = _mm256_set1_ps(inv);
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let signmask = _mm256_set1_ps(-0.0); // 0x80000000
        let c127 = _mm256_set1_ps(127.0);
        let cm127 = _mm256_set1_ps(-127.0);
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_mul_ps(_mm256_loadu_ps(xp.add(i)), vinv);
            // Round half away from zero, byte-identical to `f32::round`: `trunc(v) +
            // (|v-trunc(v)| >= 0.5 ? copysign(1,v) : 0)`. `trunc` and `v-trunc(v)` are
            // EXACT for |v| <= 127, so this avoids the `trunc(v+0.5)` sub-0.5 add-rounding
            // bug (x=0.4999… would wrongly round to 1). NaN/huge fall through to the clamp.
            let tr = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(v);
            let frac = _mm256_sub_ps(v, tr);
            let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(_mm256_andnot_ps(signmask, frac), half);
            let sign1 = _mm256_or_ps(one, _mm256_and_ps(v, signmask)); // copysign(1, v)
            let r = _mm256_add_ps(tr, _mm256_and_ps(ge, sign1));
            let r = _mm256_min_ps(_mm256_max_ps(r, cm127), c127);
            let ri = _mm256_cvtps_epi32(r);
            let lo = _mm256_castsi256_si128(ri);
            let hi = _mm256_extracti128_si256::<1>(ri);
            let i16s = _mm_packs_epi32(lo, hi); // order-preserving: [lo0..3, hi0..3]
            let i8s = _mm_packs_epi16(i16s, i16s); // low 8 bytes = elems 0..7
            _mm_storel_epi64(out.as_mut_ptr().add(i) as *mut __m128i, i8s);
            i += 8;
        }
        while i < n {
            *out.get_unchecked_mut(i) =
                (*xr.get_unchecked(i) * inv).round().clamp(-127.0, 127.0) as i8;
            i += 1;
        }
    }
}

/// Scalar fallback (non-avx2): the exact reference the AVX2 path reproduces.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn quant_row_i8(xr: &[f32], inv: f32, out: &mut [i8]) {
    for (d, &v) in out.iter_mut().zip(xr) {
        *d = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
}

/// Affine projection `x @ w^T (+ bias)` via an i8×i8 i32-accumulate GEMM (the fast
/// path for the quality-safe `attn.out` int8 — see [`EncI8Mat`]). Activation is
/// per-row i8-symmetric-quantized (amax/127) inline; the GEMM is M4-register-
/// blocked (each weight row streamed once per 4 activation rows) and parallel over
/// row-blocks. NON-byte-exact vs f32 (int8 quantization) but quality-safe on
/// proper nouns (validated on track01).
fn matmul_bias_i8(x: &Mat, w: &EncI8Mat, bias: Option<&[f32]>) -> FwResult<Mat> {
    let m = x.rows;
    let inp = x.cols;
    let out = w.out;
    if inp != w.inp {
        return Err(FwError::InvalidRequest(format!(
            "matmul_bias_i8: x.cols {inp} != w.inp {}",
            w.inp
        )));
    }
    // Per-row i8 symmetric activation quant.
    let mut xq = vec![0i8; m * inp];
    let mut sa = vec![0.0f32; m];
    xq.par_chunks_mut(inp)
        .zip(sa.par_iter_mut())
        .enumerate()
        .for_each(|(r, (xr_i8, s))| {
            let xr = &x.data[r * inp..(r + 1) * inp];
            let amax = xr.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
            let rs = amax / 127.0;
            *s = rs;
            let inv = 1.0 / rs;
            // Byte-exact AVX2 quant (f32::round doesn't vectorize → per-element roundf);
            // MEASURED ~2.2–2.7× on the m=1500 shape, byte-identical (enc_i8quant_probe + test).
            quant_row_i8(xr, inv, xr_i8);
        });
    // M4×N2 register-blocked GEMM: a full 4-row block streams each weight-row PAIR
    // once, computing 8 dots (4 act × 2 weight) with the sign-extend amortized
    // (dot_i8_m4n2). Partial row-blocks / the odd final weight row fall back to the
    // per-element dot_i8_enc. Bit-identical to the per-element order (assoc. i32 add).
    let mut c = vec![0.0f32; m * out];
    c.par_chunks_mut(4 * out)
        .enumerate()
        .for_each(|(blk, cblk)| {
            let r0 = blk * 4;
            let rows = (m - r0).min(4);
            if rows == 4 {
                let x0 = &xq[r0 * inp..(r0 + 1) * inp];
                let x1 = &xq[(r0 + 1) * inp..(r0 + 2) * inp];
                let x2 = &xq[(r0 + 2) * inp..(r0 + 3) * inp];
                let x3 = &xq[(r0 + 3) * inp..(r0 + 4) * inp];
                let (s0, s1, s2, s3) = (sa[r0], sa[r0 + 1], sa[r0 + 2], sa[r0 + 3]);
                let mut o = 0;
                while o + 2 <= out {
                    let w0 = &w.data[o * inp..(o + 1) * inp];
                    let w1 = &w.data[(o + 1) * inp..(o + 2) * inp];
                    let raw = dot_i8_m4n2(x0, x1, x2, x3, w0, w1);
                    let (sc0, sc1) = (w.scale[o], w.scale[o + 1]);
                    let (bo0, bo1) = bias.map_or((0.0, 0.0), |b| (b[o], b[o + 1]));
                    cblk[o] = raw[0] as f32 * s0 * sc0 + bo0;
                    cblk[out + o] = raw[1] as f32 * s1 * sc0 + bo0;
                    cblk[2 * out + o] = raw[2] as f32 * s2 * sc0 + bo0;
                    cblk[3 * out + o] = raw[3] as f32 * s3 * sc0 + bo0;
                    cblk[o + 1] = raw[4] as f32 * s0 * sc1 + bo1;
                    cblk[out + o + 1] = raw[5] as f32 * s1 * sc1 + bo1;
                    cblk[2 * out + o + 1] = raw[6] as f32 * s2 * sc1 + bo1;
                    cblk[3 * out + o + 1] = raw[7] as f32 * s3 * sc1 + bo1;
                    o += 2;
                }
                while o < out {
                    let wr = &w.data[o * inp..(o + 1) * inp];
                    let (sc, bo) = (w.scale[o], bias.map_or(0.0, |b| b[o]));
                    cblk[o] = dot_i8_enc(wr, x0) as f32 * s0 * sc + bo;
                    cblk[out + o] = dot_i8_enc(wr, x1) as f32 * s1 * sc + bo;
                    cblk[2 * out + o] = dot_i8_enc(wr, x2) as f32 * s2 * sc + bo;
                    cblk[3 * out + o] = dot_i8_enc(wr, x3) as f32 * s3 * sc + bo;
                    o += 1;
                }
            } else {
                for o in 0..out {
                    let wr = &w.data[o * inp..(o + 1) * inp];
                    let sc = w.scale[o];
                    let bo = bias.map_or(0.0, |b| b[o]);
                    for j in 0..rows {
                        let r = r0 + j;
                        let xr = &xq[r * inp..(r + 1) * inp];
                        cblk[j * out + o] = dot_i8_enc(wr, xr) as f32 * sa[r] * sc + bo;
                    }
                }
            }
        });
    Ok(Mat::from_vec(m, out, c))
}

fn encoder_block(x: &mut Mat, layer: &EncoderLayer, n_head: usize) -> FwResult<()> {
    let measure = crate::native_engine::perf_spans_enabled();
    macro_rules! et {
        ($i:expr, $b:expr) => {{
            if measure {
                let __t = std::time::Instant::now();
                let __r = $b;
                enc_prof_add($i, __t.elapsed().as_nanos());
                __r
            } else {
                $b
            }
        }};
    }
    // ── self-attention residual ──
    // `h = layer_norm(x)` into a fresh uninit buffer — byte-identical to
    // `x.clone()` + in-place `layer_norm`, minus the clone's redundant memcpy
    // (x is preserved for the residual `add_in_place` below regardless).
    // Kill switch FW_ENCODER_FUSED_LN=0 restores the clone path (A/B + escape).
    let h = et!(2, ln_into(x, &layer.attn_ln_w, &layer.attn_ln_b));

    // Bidirectional self-attention: causal_offset = None.
    let attn = if let (Some(qw), Some(kw), Some(vw)) =
        (&layer.attn_q_i7, &layer.attn_k_i7, &layer.attn_v_i7)
    {
        let hq = nn::quantize_act_i7(&h);
        if super::enc_qkv_fused() {
            // Fused: q/k/v written head-major inside the maddubs GEMM → external
            // SDPA → scatter, skipping the standalone gather. Byte-identical.
            et!(
                4,
                nn::attention_from_i7_qkv(
                    &hq,
                    qw,
                    Some(&layer.attn_q_b),
                    kw,
                    None,
                    vw,
                    Some(&layer.attn_v_b),
                    n_head,
                )?
            )
        } else {
            let (q, k, v) = et!(3, {
                let q = nn::matmul_bias_i7_quantized(&hq, qw, Some(&layer.attn_q_b))?;
                let k = nn::matmul_bias_i7_quantized(&hq, kw, None)?; // no key bias
                let v = nn::matmul_bias_i7_quantized(&hq, vw, Some(&layer.attn_v_b))?;
                (q, k, v)
            });
            et!(4, nn::attention(&q, &k, &v, n_head, None)?)
        }
    } else {
        let (q, k, v) = et!(3, {
            let q = enc_linear(&h, &layer.attn_q_w, &layer.attn_q_i7, Some(&layer.attn_q_b))?;
            let k = enc_linear(&h, &layer.attn_k_w, &layer.attn_k_i7, None)?; // no key bias
            let v = enc_linear(&h, &layer.attn_v_w, &layer.attn_v_i7, Some(&layer.attn_v_b))?;
            (q, k, v)
        });
        et!(4, nn::attention(&q, &k, &v, n_head, None)?)
    };
    let attn = et!(
        5,
        if let Some(w8) = &layer.attn_out_i8 {
            matmul_bias_i8(&attn, w8, Some(&layer.attn_out_b))?
        } else {
            enc_linear(
                &attn,
                &layer.attn_out_w,
                &layer.attn_out_i7,
                Some(&layer.attn_out_b),
            )?
        }
    );
    et!(6, add_in_place(x, &attn));

    // ── MLP residual ── (same fused layer_norm-into-uninit, no clone memcpy)
    let h = et!(7, ln_into(x, &layer.mlp_ln_w, &layer.mlp_ln_b));
    // f32-path fc1 bias→GELU fusion (see `fc_bias_gelu_fused_enabled`): fold the
    // separate serial 30 MiB bias RMW into the GELU pass. Only when fc1 is f32
    // (`mlp_fc_i7 == None`) AND the separate-GELU branch runs — the int8
    // `matmul_bias_i7_gelu` path folds GELU into fc2's quant and needs the bias
    // already in `h`, so it is excluded. Byte-identical.
    let int8_gelu_path = super::enc_gelu_fused() && layer.mlp_proj_i7.is_some();
    let fuse_fc_bias = !int8_gelu_path && layer.mlp_fc_i7.is_none() && fc_bias_gelu_fused_enabled();
    let fc_bias: Option<&[f32]> = if fuse_fc_bias {
        None
    } else {
        Some(&layer.mlp_fc_b)
    };
    let mut h = et!(
        8,
        enc_linear(&h, &layer.mlp_fc_w, &layer.mlp_fc_i7, fc_bias)?
    );
    // GELU-into-fc2-quant fusion: when fc2 is the int8 maddubs path, fold the
    // GELU into the activation quant so the big `[1500, 5120]` GELU'd buffer is
    // never materialized (byte-identical; see `super::enc_gelu_fused`). Otherwise
    // the classic separate GELU (label 9) + fc2 (label 10), with the f32 fc1 bias
    // folded into GELU when `fuse_fc_bias`.
    // f32-path fc2 bias→residual fusion (see `proj_bias_resid_fused_enabled`):
    // fold the separate serial bias RMW into the residual add. Only on the f32
    // fc2 path (`mlp_proj_i7 == None`); the int8 `matmul_bias_i7_gelu` path keeps
    // its own bias. Byte-identical.
    let fuse_proj_bias = layer.mlp_proj_i7.is_none() && proj_bias_resid_fused_enabled();
    let h = if super::enc_gelu_fused()
        && let Some(w) = layer.mlp_proj_i7.as_ref()
    {
        et!(10, nn::matmul_bias_i7_gelu(&h, w, Some(&layer.mlp_proj_b))?)
    } else {
        if fuse_fc_bias {
            et!(9, nn::gelu_add_bias(&mut h, &layer.mlp_fc_b));
        } else {
            et!(9, nn::gelu(&mut h));
        }
        let proj_bias: Option<&[f32]> = if fuse_proj_bias {
            None
        } else {
            Some(&layer.mlp_proj_b)
        };
        et!(
            10,
            enc_linear(&h, &layer.mlp_proj_w, &layer.mlp_proj_i7, proj_bias)?
        )
    };
    if fuse_proj_bias {
        et!(11, nn::add_bias_residual(x, &h, &layer.mlp_proj_b));
    } else {
        et!(11, add_in_place(x, &h));
    }

    Ok(())
}

/// In-place element-wise `x += y` for matrices of identical shape.
///
/// Kept SERIAL deliberately: parallelizing this was MEASURED a wash/slight-loss
/// (2026-07-02, BlackThrush) — the residual operands are cache-warm from the
/// matmul that just produced them and LLVM auto-vectorizes the loop, so rayon
/// dispatch overhead outweighs any bandwidth gain (unlike the fused-LN clone,
/// which was a *cold* memcpy pass). Do not re-parallelize.
fn add_in_place(x: &mut Mat, y: &Mat) {
    debug_assert_eq!(
        (x.rows, x.cols),
        (y.rows, y.cols),
        "add_in_place shape mismatch"
    );
    for (a, b) in x.data.iter_mut().zip(&y.data) {
        *a += b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::{find_model_file, mel};

    /// `quant_row_i8` (the AVX2 activation quant in `matmul_bias_i8`) must be
    /// byte-identical to the scalar `(v*inv).round().clamp(-127,127) as i8` map it
    /// replaces — including the ±127 clamp edges and round-half-away boundaries.
    #[test]
    fn quant_row_i8_is_byte_identical_to_scalar_round() {
        let scalar = |xr: &[f32], inv: f32, out: &mut [i8]| {
            for (d, &v) in out.iter_mut().zip(xr) {
                *d = (v * inv).round().clamp(-127.0, 127.0) as i8;
            }
        };
        for &inv in &[0.5f32, 1.0, 3.7, 42.0] {
            let mut xs: Vec<f32> = Vec::new();
            // Dense sweep across the clamp region + exact half-way / integer boundaries,
            // plus over-range values that must saturate identically.
            for k in -400..=400 {
                xs.push(k as f32 / inv * 0.5);
                xs.push((k as f32 + 0.5) / inv);
            }
            xs.extend_from_slice(&[0.0, -0.0, 1e30, -1e30, 127.4999 / inv, -127.5 / inv]);
            let n = xs.len();
            let mut a = vec![0i8; n];
            let mut b = vec![0i8; n];
            quant_row_i8(&xs, inv, &mut a);
            scalar(&xs, inv, &mut b);
            assert_eq!(a, b, "AVX2 quant_row_i8 != scalar round at inv={inv}");
        }
    }

    /// Deterministic LCG (Numerical Recipes constants) -> f32 in [-1, 1).
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next_f32() * scale).collect()
        }
        fn mat(&mut self, rows: usize, cols: usize, scale: f32) -> Mat {
            Mat::from_vec(rows, cols, self.vec(rows * cols, scale))
        }
    }

    /// Build a tiny but structurally-real `EncoderWeights` by hand.
    ///
    /// `n_state = 8`, `n_head = 2`, `n_layers = 2`, `n_mels = 4`, and a
    /// positional embedding sized for `pe_ctx` rows. Weights are small-scale
    /// random so the forward pass stays numerically tame (no overflow) yet
    /// genuinely depends on every input.
    fn synthetic_weights(pe_ctx: usize) -> EncoderWeights {
        let mut rng = Lcg::new(0xE0C0_DE01);
        let n_mels = 4;
        let n_state = 8;
        let n_head = 2;
        let n_layers = 2;
        let mlp_hidden = n_state * MLP_RATIO;
        let s = 0.2f32;

        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(EncoderLayer {
                attn_ln_w: vec![1.0; n_state],
                attn_ln_b: vec![0.0; n_state],
                attn_q_w: rng.mat(n_state, n_state, s),
                attn_q_b: rng.vec(n_state, s),
                attn_k_w: rng.mat(n_state, n_state, s),
                attn_v_w: rng.mat(n_state, n_state, s),
                attn_v_b: rng.vec(n_state, s),
                attn_out_w: rng.mat(n_state, n_state, s),
                attn_out_b: rng.vec(n_state, s),
                mlp_ln_w: vec![1.0; n_state],
                mlp_ln_b: vec![0.0; n_state],
                mlp_fc_w: rng.mat(n_state, mlp_hidden, s),
                mlp_fc_b: rng.vec(mlp_hidden, s),
                mlp_proj_w: rng.mat(mlp_hidden, n_state, s),
                mlp_proj_b: rng.vec(n_state, s),
                attn_q_i7: None,
                attn_k_i7: None,
                attn_v_i7: None,
                attn_out_i7: None,
                attn_out_i8: None,
                mlp_fc_i7: None,
                mlp_proj_i7: None,
            });
        }

        EncoderWeights {
            n_mels,
            n_state,
            n_head,
            n_ctx: pe_ctx,
            conv1_wt: rng.mat(n_mels * CONV_K, n_state, s),
            conv1_b: rng.vec(n_state, s),
            conv2_wt: rng.mat(n_state * CONV_K, n_state, s),
            conv2_b: rng.vec(n_state, s),
            pos_emb: rng.mat(pe_ctx, n_state, s),
            layers,
            ln_post_w: vec![1.0; n_state],
            ln_post_b: vec![0.0; n_state],
        }
    }

    /// A mel-major `[n_mel, n_frames]` window from an LCG seed.
    fn synthetic_mel(seed: u64, n_mel: usize, n_frames: usize) -> Mel {
        let mut rng = Lcg::new(seed);
        Mel {
            n_mel,
            n_frames,
            data: rng.vec(n_mel * n_frames, 1.0),
        }
    }

    /// A checkpoint closure that never cancels. The production `forward`
    /// enforces exactly 3000 mel frames, so all synthetic tests build a
    /// 3000-frame window (ctx = 1500) and assert against that.
    fn noop_checkpoint() -> FwResult<()> {
        Ok(())
    }

    #[test]
    fn fused_dequant_transpose_is_bit_identical_to_dequant_then_transpose() {
        // The fused load primitive must produce EXACTLY the f32 bytes that the
        // old two-step (dequant f16->f32, then `nn::transpose_serial`) produced,
        // for every shape — non-square, non-tile-aligned, and tile-aligned.
        for &(rows, cols) in &[
            (384usize, 1536usize),
            (37usize, 91usize),
            (64, 64),
            (1, 130),
        ] {
            // Deterministic spread of half bit patterns (normal + subnormal +
            // sign), enough to catch any index or conversion mistake.
            let bits: Vec<u16> = (0..rows * cols)
                .map(|i| (i as u16).wrapping_mul(7) ^ 0x3c00)
                .collect();
            // Reference: dequant in [out,in] order, then transpose to [in,out].
            let dequant: Vec<f32> = bits
                .iter()
                .map(|&b| Float16::from_bits(b).to_f32())
                .collect();
            let reference = nn::transpose_serial(&dequant, rows, cols);
            // Production path reads raw little-endian f16 bytes (as the blob holds).
            let raw: Vec<u8> = bits.iter().flat_map(|&b| b.to_le_bytes()).collect();
            let fused = dequant_transpose_f16_bytes(&raw, rows, cols);
            // Compare BIT patterns, not f32 values: the synthetic data includes
            // f16 NaN patterns, and `NaN != NaN` would spuriously fail float
            // equality even when both sides produced the identical NaN bits.
            let fused_bits: Vec<u32> = fused.iter().map(|x| x.to_bits()).collect();
            let reference_bits: Vec<u32> = reference.iter().map(|x| x.to_bits()).collect();
            assert_eq!(
                fused_bits, reference_bits,
                "fused dequant-transpose diverged at shape [{rows},{cols}]"
            );
        }
    }

    #[test]
    fn synthetic_forward_shape_and_finiteness() {
        // 24-frame conceptual window, but the production `forward` enforces
        // exactly 3000 frames; ctx = 3000/2 = 1500. Build a pe sized to 1500.
        let n_ctx = mel::FRAMES_PER_CHUNK / 2;
        let w = synthetic_weights(n_ctx);
        let melw = synthetic_mel(1, w.n_mels, mel::FRAMES_PER_CHUNK);

        let out = forward(&w, &melw, 4, &noop_checkpoint).expect("forward");
        assert_eq!(out.rows, n_ctx, "ctx = frames/2");
        assert_eq!(out.cols, w.n_state);
        assert!(out.data.iter().all(|v| v.is_finite()), "output finite");
    }

    #[test]
    fn synthetic_forward_depends_on_input() {
        let n_ctx = mel::FRAMES_PER_CHUNK / 2;
        let w = synthetic_weights(n_ctx);
        let mel_a = synthetic_mel(1, w.n_mels, mel::FRAMES_PER_CHUNK);
        let mel_b = synthetic_mel(2, w.n_mels, mel::FRAMES_PER_CHUNK);

        let out_a = forward(&w, &mel_a, 1, &noop_checkpoint).expect("a");
        let out_b = forward(&w, &mel_b, 1, &noop_checkpoint).expect("b");
        let max_diff = out_a
            .data
            .iter()
            .zip(&out_b.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "different inputs must yield different outputs (max_diff {max_diff})"
        );
    }

    #[test]
    fn synthetic_forward_is_bidirectional() {
        // Encoder attention is bidirectional: perturbing the LAST mel frame
        // must (almost surely) change output row 0. This catches an accidental
        // causal mask.
        let n_ctx = mel::FRAMES_PER_CHUNK / 2;
        let w = synthetic_weights(n_ctx);
        let base = synthetic_mel(7, w.n_mels, mel::FRAMES_PER_CHUNK);

        let mut perturbed = base.clone();
        // Last frame across every mel channel (mel-major index m*n_frames + f).
        let last = perturbed.n_frames - 1;
        for m in 0..perturbed.n_mel {
            perturbed.data[m * perturbed.n_frames + last] += 3.0;
        }

        let out_a = forward(&w, &base, 1, &noop_checkpoint).expect("base");
        let out_b = forward(&w, &perturbed, 1, &noop_checkpoint).expect("perturbed");

        let row0_changed = (0..w.n_state).any(|c| (out_a.data[c] - out_b.data[c]).abs() > 1e-5);
        assert!(
            row0_changed,
            "output row 0 must react to a change in the LAST input frame \
             (bidirectional attention — no causal mask)"
        );
    }

    #[test]
    fn checkpoint_cancellation_aborts() {
        let n_ctx = mel::FRAMES_PER_CHUNK / 2;
        let w = synthetic_weights(n_ctx);
        let melw = synthetic_mel(1, w.n_mels, mel::FRAMES_PER_CHUNK);
        let cancel = || Err(FwError::Cancelled("test".into()));
        let res = forward(&w, &melw, 1, &cancel);
        assert!(matches!(res, Err(FwError::Cancelled(_))));
    }

    #[test]
    fn odd_or_oversized_frame_count_is_rejected() {
        let w = synthetic_weights(mel::FRAMES_PER_CHUNK / 2);
        // Odd frame count: the stride-2 conv would yield a fractional ctx.
        let odd = synthetic_mel(1, w.n_mels, 23);
        assert!(
            forward(&w, &odd, 1, &noop_checkpoint).is_err(),
            "odd frame count must be rejected"
        );
        // Oversized: more than 2*n_ctx frames would overrun the pos embedding.
        let big = synthetic_mel(1, w.n_mels, mel::FRAMES_PER_CHUNK + 2);
        assert!(
            forward(&w, &big, 1, &noop_checkpoint).is_err(),
            "frame count > 2*n_audio_ctx must be rejected"
        );
        // Zero frames: rejected.
        let zero = synthetic_mel(1, w.n_mels, 0);
        assert!(
            forward(&w, &zero, 1, &noop_checkpoint).is_err(),
            "zero frame count must be rejected"
        );
    }

    #[test]
    fn truncated_even_frame_window_is_accepted() {
        // Tail-window truncation: a smaller even frame count yields ctx = n/2
        // rows, mirroring whisper.cpp's audio_ctx feature. 256-frame window →
        // 128 ctx rows.
        let w = synthetic_weights(mel::FRAMES_PER_CHUNK / 2);
        let melw = synthetic_mel(1, w.n_mels, 256);
        let out = forward(&w, &melw, 1, &noop_checkpoint).expect("truncated forward");
        assert_eq!(out.rows, 128, "ctx = frames/2 for a truncated window");
        assert_eq!(out.cols, w.n_state);
        assert!(out.data.iter().all(|v| v.is_finite()), "output finite");
    }

    #[test]
    fn wrong_mel_channels_is_rejected() {
        let w = synthetic_weights(mel::FRAMES_PER_CHUNK / 2);
        let melw = synthetic_mel(1, w.n_mels + 1, mel::FRAMES_PER_CHUNK);
        let res = forward(&w, &melw, 1, &noop_checkpoint);
        assert!(res.is_err(), "wrong mel channel count must be rejected");
    }

    #[test]
    fn transpose_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose(&data, 2, 3); // [2,3] -> [3,2]
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        // original [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let back = transpose(&t.data, 3, 2);
        assert_eq!(back.data, data);
    }

    #[test]
    fn fused_full_mel_window_matches_chunk_then_transpose() {
        let full = Mel {
            n_mel: 3,
            n_frames: 7,
            data: (0..21).map(|i| i as f32 + 0.25).collect(),
        };
        for &(offset, frames) in &[(0, 6), (2, 4), (5, 6), (7, 4), (9, 4)] {
            let compact = mel::chunk_frames(&full, offset, frames);
            let want = time_major_mel_window(&compact);
            let got = time_major_mel_window_from_full_mel(&full, offset, frames);
            assert_eq!(got.rows, want.rows);
            assert_eq!(got.cols, want.cols);
            assert_eq!(got.data, want.data, "offset={offset} frames={frames}");
        }
    }

    /// Minimal inline 16-bit PCM mono WAV reader (jfk.wav is 16 kHz mono i16).
    /// Returns f32 samples in [-1, 1]. `None` on any parse failure.
    fn read_wav_mono_f32(path: &std::path::Path) -> Option<Vec<f32>> {
        let reader = hound::WavReader::open(path).ok()?;
        let spec = reader.spec();
        if spec.channels != 1 || spec.sample_format != hound::SampleFormat::Int {
            return None;
        }
        let mut reader = reader;
        let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
        let samples: Vec<f32> = reader
            .samples::<i32>()
            .filter_map(Result::ok)
            .map(|s| s as f32 / max)
            .collect();
        Some(samples)
    }

    fn dequant_i8_for_test(w: &EncI8Mat) -> Mat {
        let mut data = vec![0.0f32; w.inp * w.out];
        for o in 0..w.out {
            let row = &w.data[o * w.inp..(o + 1) * w.inp];
            for (i, &q) in row.iter().enumerate() {
                data[i * w.out + o] = f32::from(q) * w.scale[o];
            }
        }
        Mat::from_vec(w.inp, w.out, data)
    }

    fn quant_error(original: &Mat, dequant: &Mat) -> (f64, f64) {
        assert_eq!(original.rows, dequant.rows);
        assert_eq!(original.cols, dequant.cols);
        let mut sum_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        let mut max_abs = 0.0f64;
        let mut ref_amax = 0.0f64;
        for (&a, &b) in original.data.iter().zip(&dequant.data) {
            let a = f64::from(a);
            let b = f64::from(b);
            let d = (a - b).abs();
            sum_sq += d * d;
            ref_sq += a * a;
            max_abs = max_abs.max(d);
            ref_amax = ref_amax.max(a.abs());
        }
        let rel_rmse = (sum_sq / ref_sq.max(1e-24)).sqrt();
        let max_abs_over_amax = max_abs / ref_amax.max(1e-12);
        (rel_rmse, max_abs_over_amax)
    }

    fn assert_quant_budget(
        model: &str,
        layer: usize,
        matrix: &str,
        original: &Mat,
        dequant: &Mat,
        max_rel_rmse: f64,
        max_abs_over_amax: f64,
    ) {
        let (rel_rmse, max_abs) = quant_error(original, dequant);
        eprintln!(
            "{model} layer={layer:02} {matrix:<12} rel_rmse={rel_rmse:.6} max_abs/amax={max_abs:.6}"
        );
        assert!(
            rel_rmse <= max_rel_rmse,
            "{model} layer {layer} {matrix} rel_rmse {rel_rmse:.6} exceeds {max_rel_rmse:.6}"
        );
        assert!(
            max_abs <= max_abs_over_amax,
            "{model} layer {layer} {matrix} max_abs/amax {max_abs:.6} exceeds {max_abs_over_amax:.6}"
        );
    }

    fn assert_quality_safe_int8_error_budget(model_name: &str) {
        let Some(path) = find_model_file(model_name) else {
            eprintln!("SKIP {model_name} encoder-int8 budget: model missing");
            return;
        };
        let model = GgmlModel::load(&path).expect("load model");
        let decision = crate::native_engine::encoder_int8_policy_decision(&model.hparams);
        if !decision.enabled() {
            eprintln!(
                "SKIP {model_name} encoder-int8 budget: policy reason={}",
                decision.reason
            );
            return;
        }

        let weights = EncoderWeights::from_ggml(&model).expect("encoder weights");
        assert_eq!(
            weights.layers.len(),
            model.hparams.n_audio_layer as usize,
            "loaded all encoder layers"
        );

        for (idx, layer) in weights.layers.iter().enumerate() {
            let attn_q =
                nn::dequant_i7_for_test(layer.attn_q_i7.as_ref().expect("quality-safe q i7"));
            let attn_k =
                nn::dequant_i7_for_test(layer.attn_k_i7.as_ref().expect("quality-safe k i7"));
            let attn_v =
                nn::dequant_i7_for_test(layer.attn_v_i7.as_ref().expect("quality-safe v i7"));
            let mlp_fc =
                nn::dequant_i7_for_test(layer.mlp_fc_i7.as_ref().expect("quality-safe fc1 i7"));
            let mlp_proj =
                nn::dequant_i7_for_test(layer.mlp_proj_i7.as_ref().expect("quality-safe fc2 i7"));
            let attn_out = dequant_i8_for_test(
                layer
                    .attn_out_i8
                    .as_ref()
                    .expect("quality-safe attn.out i8"),
            );
            assert!(
                layer.attn_out_i7.is_none(),
                "quality-safe policy must not use the rejected all-i7 attn.out path"
            );

            let max_i7_abs = 0.035;
            let max_i8_abs = 0.012;
            assert_quant_budget(
                model_name,
                idx,
                "attn_q_i7",
                &layer.attn_q_w,
                &attn_q,
                decision.quant_rel_rmse_budget,
                max_i7_abs,
            );
            assert_quant_budget(
                model_name,
                idx,
                "attn_k_i7",
                &layer.attn_k_w,
                &attn_k,
                decision.quant_rel_rmse_budget,
                max_i7_abs,
            );
            assert_quant_budget(
                model_name,
                idx,
                "attn_v_i7",
                &layer.attn_v_w,
                &attn_v,
                decision.quant_rel_rmse_budget,
                max_i7_abs,
            );
            assert_quant_budget(
                model_name,
                idx,
                "mlp_fc_i7",
                &layer.mlp_fc_w,
                &mlp_fc,
                decision.quant_rel_rmse_budget,
                max_i7_abs,
            );
            assert_quant_budget(
                model_name,
                idx,
                "mlp_proj_i7",
                &layer.mlp_proj_w,
                &mlp_proj,
                decision.quant_rel_rmse_budget,
                max_i7_abs,
            );
            assert_quant_budget(
                model_name,
                idx,
                "attn_out_i8",
                &layer.attn_out_w,
                &attn_out,
                decision.quant_rel_rmse_budget * 0.55,
                max_i8_abs,
            );
        }
    }

    #[test]
    fn real_tiny_en_quality_safe_int8_per_layer_error_budget() {
        assert_quality_safe_int8_error_budget("tiny.en");
    }

    #[test]
    fn real_large_v3_turbo_quality_safe_int8_per_layer_error_budget() {
        assert_quality_safe_int8_error_budget("large-v3-turbo");
    }

    // ── gated real-model test (skips when tiny.en is absent) ──

    #[test]
    fn real_tiny_en_encoder_stats() {
        let Some(path) = find_model_file("tiny.en") else {
            eprintln!("SKIP real_tiny_en_encoder_stats: ggml-tiny.en.bin not found");
            return;
        };
        let model = GgmlModel::load(&path).expect("load tiny.en");

        // Verify the exact encoder tensor names exist (fixture for bd-szkq).
        let mut expected: Vec<String> = vec![
            "encoder.conv1.weight".into(),
            "encoder.conv1.bias".into(),
            "encoder.conv2.weight".into(),
            "encoder.conv2.bias".into(),
            "encoder.positional_embedding".into(),
            "encoder.ln_post.weight".into(),
            "encoder.ln_post.bias".into(),
        ];
        for i in 0..model.hparams.n_audio_layer {
            for suf in [
                "attn_ln.weight",
                "attn_ln.bias",
                "attn.query.weight",
                "attn.query.bias",
                "attn.key.weight",
                "attn.value.weight",
                "attn.value.bias",
                "attn.out.weight",
                "attn.out.bias",
                "mlp_ln.weight",
                "mlp_ln.bias",
                "mlp.0.weight",
                "mlp.0.bias",
                "mlp.2.weight",
                "mlp.2.bias",
            ] {
                expected.push(format!("encoder.blocks.{i}.{suf}"));
            }
        }
        let all: std::collections::HashSet<&str> = model.tensor_names().collect();
        for name in &expected {
            assert!(
                all.contains(name.as_str()),
                "missing encoder tensor '{name}'"
            );
        }
        // tiny.en key projections must have NO bias.
        for i in 0..model.hparams.n_audio_layer {
            let key_bias = format!("encoder.blocks.{i}.attn.key.bias");
            assert!(
                model.tensor(&key_bias).is_none(),
                "whisper key projection must have no bias, found '{key_bias}'"
            );
        }

        let w = EncoderWeights::from_ggml(&model).expect("from_ggml");
        assert_eq!(w.n_state(), 384);
        assert_eq!(w.n_layers(), 4);

        // Real mel from /tmp/jfk.wav (skip if absent).
        let wav_path = std::path::Path::new("/tmp/jfk.wav");
        let Some(samples) = read_wav_mono_f32(wav_path) else {
            eprintln!("SKIP real mel forward: /tmp/jfk.wav not present/parseable");
            return;
        };
        let full = mel::log_mel(&samples, &model.filters, 4).expect("log_mel");
        let window = mel::chunk_frames(&full, 0, mel::FRAMES_PER_CHUNK);

        let noop = || Ok(());
        let out = forward(&w, &window, 4, &noop).expect("forward");
        assert_eq!(out.rows, 1500);
        assert_eq!(out.cols, 384);
        assert!(out.data.iter().all(|v| v.is_finite()), "output finite");

        // Stats fixture for bd-szkq.
        let n = out.data.len() as f64;
        let mean = out.data.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let var = out
            .data
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / n;
        let std = var.sqrt();
        assert!(
            std > 0.1 && std < 100.0,
            "encoder output std {std} outside sanity band (0.1, 100)"
        );
        eprintln!("tiny.en jfk encoder output: mean={mean:.6} std={std:.6}");
        eprint!("first 8 values: ");
        for v in &out.data[..8] {
            eprint!("{v:.6} ");
        }
        eprintln!();

        // Determinism: a second run must be bit-identical.
        let out2 = forward(&w, &window, 4, &noop).expect("forward 2");
        assert_eq!(out.data, out2.data, "encoder must be deterministic");
    }
}
