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
    /// `conv1` flat weight `[n_state, n_mels, K]` and bias `[n_state]`.
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    /// `conv2` flat weight `[n_state, n_state, K]` and bias `[n_state]`.
    conv2_w: Vec<f32>,
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

        // Conv stem: ggml shapes are [Cout, Cin, K]; nn::conv1d wants this flat
        // [Cout, Cin, K] order verbatim, so no transpose.
        let conv1_w = load_shaped(model, "encoder.conv1.weight", &[n_state, n_mels, CONV_K])?;
        let conv1_b = load_vec(model, "encoder.conv1.bias", n_state)?;
        let conv2_w = load_shaped(model, "encoder.conv2.weight", &[n_state, n_state, CONV_K])?;
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
        let layers = (0..n_layer)
            .into_par_iter()
            .map(|i| -> FwResult<EncoderLayer> {
                let p = |suffix: &str| format!("encoder.blocks.{i}.{suffix}");
                Ok(EncoderLayer {
                    attn_ln_w: load_vec(model, &p("attn_ln.weight"), n_state)?,
                    attn_ln_b: load_vec(model, &p("attn_ln.bias"), n_state)?,
                    attn_q_w: load_linear_transposed(
                        model,
                        &p("attn.query.weight"),
                        n_state,
                        n_state,
                    )?,
                    attn_q_b: load_vec(model, &p("attn.query.bias"), n_state)?,
                    // whisper key projection has NO bias.
                    attn_k_w: load_linear_transposed(
                        model,
                        &p("attn.key.weight"),
                        n_state,
                        n_state,
                    )?,
                    attn_v_w: load_linear_transposed(
                        model,
                        &p("attn.value.weight"),
                        n_state,
                        n_state,
                    )?,
                    attn_v_b: load_vec(model, &p("attn.value.bias"), n_state)?,
                    attn_out_w: load_linear_transposed(
                        model,
                        &p("attn.out.weight"),
                        n_state,
                        n_state,
                    )?,
                    attn_out_b: load_vec(model, &p("attn.out.bias"), n_state)?,
                    mlp_ln_w: load_vec(model, &p("mlp_ln.weight"), n_state)?,
                    mlp_ln_b: load_vec(model, &p("mlp_ln.bias"), n_state)?,
                    mlp_fc_w: load_linear_transposed(
                        model,
                        &p("mlp.0.weight"),
                        mlp_hidden,
                        n_state,
                    )?,
                    mlp_fc_b: load_vec(model, &p("mlp.0.bias"), mlp_hidden)?,
                    mlp_proj_w: load_linear_transposed(
                        model,
                        &p("mlp.2.weight"),
                        n_state,
                        mlp_hidden,
                    )?,
                    mlp_proj_b: load_vec(model, &p("mlp.2.bias"), n_state)?,
                })
            })
            .collect::<FwResult<Vec<_>>>()?;

        let ln_post_w = load_vec(model, "encoder.ln_post.weight", n_state)?;
        let ln_post_b = load_vec(model, "encoder.ln_post.bias", n_state)?;

        Ok(Self {
            n_mels,
            n_state,
            n_head,
            n_ctx,
            conv1_w,
            conv1_b,
            conv2_w,
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

fn forward_time_major(
    w: &EncoderWeights,
    x: Mat,
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<Mat> {
    // conv1: [3000, n_mel] -> [3000, n_state], +gelu.
    let mut x = nn::conv1d(
        &x, &w.conv1_w, w.n_state, w.n_mels, CONV_K, &w.conv1_b, 1, CONV_PAD,
    )?;
    nn::gelu(&mut x);

    // conv2 (stride 2): [3000, n_state] -> [1500, n_state], +gelu.
    let mut x = nn::conv1d(
        &x, &w.conv2_w, w.n_state, w.n_state, CONV_K, &w.conv2_b, 2, CONV_PAD,
    )?;
    nn::gelu(&mut x);

    let n_ctx = x.rows;
    if n_ctx > w.n_ctx {
        return Err(FwError::InvalidRequest(format!(
            "encoder: conv produced {n_ctx} ctx rows > positional embedding capacity {}",
            w.n_ctx
        )));
    }

    // Add positional embedding (file tensor), sliced to the first n_ctx rows.
    add_pos_emb(&mut x, &w.pos_emb, n_ctx);

    // Residual transformer blocks.
    for layer in &w.layers {
        encoder_block(&mut x, layer, w.n_head)?;
        checkpoint()?;
    }

    // Final ln_post.
    nn::layer_norm(&mut x, &w.ln_post_w, &w.ln_post_b, LN_EPS);

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

/// One residual encoder block, mutating `x` (`[n_ctx, n_state]`) in place.
///
/// `x = x + attn_out(attn(ln_attn(x)))` then `x = x + mlp(ln_mlp(x))`. The
/// attention is bidirectional (no causal mask): every output row depends on
/// every input row.
fn encoder_block(x: &mut Mat, layer: &EncoderLayer, n_head: usize) -> FwResult<()> {
    // ── self-attention residual ──
    let mut h = x.clone();
    nn::layer_norm(&mut h, &layer.attn_ln_w, &layer.attn_ln_b, LN_EPS);

    let q = nn::matmul_bias(&h, &layer.attn_q_w, Some(&layer.attn_q_b))?;
    let k = nn::matmul_bias(&h, &layer.attn_k_w, None)?; // no key bias
    let v = nn::matmul_bias(&h, &layer.attn_v_w, Some(&layer.attn_v_b))?;

    // Bidirectional self-attention: causal_offset = None.
    let attn = nn::attention(&q, &k, &v, n_head, None)?;
    let attn = nn::matmul_bias(&attn, &layer.attn_out_w, Some(&layer.attn_out_b))?;
    add_in_place(x, &attn);

    // ── MLP residual ──
    let mut h = x.clone();
    nn::layer_norm(&mut h, &layer.mlp_ln_w, &layer.mlp_ln_b, LN_EPS);
    let mut h = nn::matmul_bias(&h, &layer.mlp_fc_w, Some(&layer.mlp_fc_b))?;
    nn::gelu(&mut h);
    let h = nn::matmul_bias(&h, &layer.mlp_proj_w, Some(&layer.mlp_proj_b))?;
    add_in_place(x, &h);

    Ok(())
}

/// In-place element-wise `x += y` for matrices of identical shape.
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
            });
        }

        EncoderWeights {
            n_mels,
            n_state,
            n_head,
            n_ctx: pe_ctx,
            conv1_w: rng.vec(n_state * n_mels * CONV_K, s),
            conv1_b: rng.vec(n_state, s),
            conv2_w: rng.vec(n_state * n_state * CONV_K, s),
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
