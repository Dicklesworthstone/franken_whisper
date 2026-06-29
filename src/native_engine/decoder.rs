//! Whisper text decoder forward pass with an incremental KV cache.
//!
//! This module ports whisper.cpp's `whisper_build_graph_decoder()` (see
//! `src/whisper.cpp`, the `cur = d_te[token] + d_pe[pos]` token/position
//! embedding through the final `ggml_mul_mat(d_te, cur)` logits product). It
//! computes, for a batch of one or more newly-decoded tokens appended to a
//! running self-attention KV cache, the next-token logits for the **last**
//! position only.
//!
//! # Block structure (per decoder layer)
//!
//! Mirroring whisper exactly:
//!
//! ```text
//! x = token_emb[token] + pos_emb[cache_len + i]            (input embedding)
//! for each layer:
//!     x = x + self_out( self_attn( attn_ln(x) ) )          (causal, KV cache)
//!     x = x + cross_out( cross_attn( cross_attn_ln(x) ) )  (encoder K/V, no mask)
//!     x = x + mlp( mlp_ln(x) )
//! x = decoder.ln(x)
//! logits = x[last] @ token_embedding^T                     (weight tying)
//! ```
//!
//! Self-attention: query `w+b`, key `w` (**no bias**), value `w+b`, out
//! `w+b`; Q and K are each scaled by `d_head^-0.25` (handled inside
//! [`nn::attention`]); causal over the whole cache. Cross-attention: query
//! `w+b`, key `w` (no bias), value `w+b`, out `w+b`; the cross K/V are
//! **precomputed once per audio window** from the encoder output and reused
//! across every decode step. Cross-attention is **unmasked**.
//!
//! # Logits product / weight tying — memory rationale
//!
//! The output projection ties weights with the token embedding:
//! `logits = x @ token_embedding^T`. The embedding is stored naturally as
//! `[n_vocab, n_state]` (its ggml on-disk logical shape). Pre-transposing it
//! to `[n_state, n_vocab]` for a `matmul_bias`-style `x @ w_t` would store a
//! **second** full copy of the largest tensor in the model
//! (`51864 * 384 * 4 B ≈ 80 MB` for tiny, `≈ 265 MB` for large) — pure
//! waste. Instead we keep the single `[n_vocab, n_state]` copy and compute
//! the last row's logits as
//! `logits^T = token_emb @ x_last^T` → `[n_vocab, 1]` via one
//! [`nn::matmul`], then read off the column. This is exactly whisper.cpp's
//! `ggml_mul_mat(d_te, cur)` (`d_te` is `[n_state, n_vocab]` in ggml's
//! column-fastest convention, i.e. our `[n_vocab, n_state]`), and we only
//! ever materialize logits for the **last** position, not the whole batch.
//!
//! # Cross-attention weight recording (DTW word timestamps)
//!
//! Word-level timestamps (bead bd-rjsx) align text to audio via DTW over the
//! softmaxed cross-attention weights of selected heads. [`nn::attention`]
//! does not surface those intermediate weights, so the cross path is
//! implemented **locally** in this file (`q·k^T → softmax → @v`) rather than
//! by calling [`nn::attention`]: that lets us capture the per-head
//! `[tokens, enc_frames]` softmax matrices when recording is enabled. The
//! recording is opt-in ([`DecoderState::record_cross_attn`]) and off by
//! default, so the steady-state decode path pays nothing.

#![allow(clippy::module_name_repetitions)]

use rayon::prelude::*;

use super::nn::{self, KvCache, WeightMat};
use super::{Mat, WhisperHParams};
use crate::error::{FwError, FwResult};

/// Layer-norm epsilon used throughout whisper (`hparams.eps = 1e-5f`).
const LN_EPS: f32 = 1e-5;

// ─────────────────────────────────────────────────────────────────────────
// Per-sub-part attribution (measurement-only; OFF unless perf spans enabled)
// ─────────────────────────────────────────────────────────────────────────

/// The decoder-step sub-parts attributed by [`forward_step`] when perf-span
/// measurement is enabled. Index order is fixed so the accumulator is a flat
/// array (no hashing on the hot path). Kept in declaration order of the step.
#[derive(Clone, Copy)]
pub enum Sub {
    /// `token_emb[token] + pos_emb` lookup/add.
    Embed = 0,
    /// Self-attention input layer-norm + the residual `x.clone()`.
    SelfLn = 1,
    /// Self-attention Q/K/V projections (the threaded `project_qkv`).
    SelfQkv = 2,
    /// Causal cache-append + scaled-dot-product self-attention.
    SelfAttn = 3,
    /// Self-attention output projection + residual add.
    SelfOut = 4,
    /// Cross-attention input layer-norm + the residual `x.clone()`.
    CrossLn = 5,
    /// Cross-attention query projection.
    CrossQ = 6,
    /// Cross-attention (q·k^T over enc_frames → softmax → @v).
    CrossAttn = 7,
    /// Cross-attention output projection + residual add.
    CrossOut = 8,
    /// MLP input layer-norm + residual `x.clone()`.
    MlpLn = 9,
    /// MLP fc + gelu + proj.
    Mlp = 10,
    /// Final pre-logits layer-norm.
    FinalLn = 11,
    /// Tied output projection (the `[n_vocab, n_state]` logits GEMV).
    Logits = 12,
}

/// Number of attributed sub-parts (= `Sub` variant count).
pub const SUB_COUNT: usize = 13;

/// Human-readable labels for [`Sub`], in index order (for report tables).
pub const SUB_LABELS: [&str; SUB_COUNT] = [
    "embed+pos",
    "self_ln+clone",
    "self_qkv_proj",
    "self_attn",
    "self_out_proj",
    "cross_ln+clone",
    "cross_q_proj",
    "cross_attn",
    "cross_out_proj",
    "mlp_ln+clone",
    "mlp_fc_gelu_proj",
    "final_ln",
    "logits_gemv",
];

thread_local! {
    /// Per-sub-part cumulative nanoseconds, summed across every [`forward_step`]
    /// call on this thread. Only written when [`super::perf_spans_enabled`] is
    /// set; otherwise the instrumentation is skipped entirely (no `Instant`).
    static SUB_NS: std::cell::RefCell<[u128; SUB_COUNT]> =
        const { std::cell::RefCell::new([0u128; SUB_COUNT]) };
}

/// Add `dt` nanoseconds to sub-part `s`'s thread-local accumulator.
#[inline]
fn sub_add(s: Sub, dt: u128) {
    SUB_NS.with(|c| c.borrow_mut()[s as usize] += dt);
}

/// Drain and return this thread's per-sub-part cumulative nanoseconds, zeroing
/// the accumulator. Measurement-only: pairs with [`forward_step`]'s perf-span
/// instrumentation so an attribution harness can read the split for a batch of
/// steps then reset between phases. Returns all-zero when measurement was off.
#[must_use]
pub fn take_sub_ns() -> [u128; SUB_COUNT] {
    SUB_NS.with(|c| {
        let mut b = c.borrow_mut();
        let out = *b;
        *b = [0u128; SUB_COUNT];
        out
    })
}

/// One worker band's cross-K/V precompute result: the band's starting layer
/// index plus its layers' `(Kcross, Vcross)` pairs, in layer order. Used to
/// reassemble [`DecoderState::new`]'s parallel layer fan-out deterministically.
type CrossKvBand = FwResult<(usize, Vec<(Mat, Mat)>)>;

/// Linear weight (in one of two representations) plus its optional bias `[out]`.
///
/// Whisper linear layers are `y = x @ W^T + b` with `W` shaped `[out, in]`.
/// Two storage strategies, chosen once at load time (see [`load_linear`]):
///
/// * **f32 path** ([`WeightMat::F32`]): the weight is pre-transposed to
///   `[in, out]` so the forward is a contiguous `x @ w_t` ([`nn::matmul_bias`]).
/// * **f16 path** ([`WeightMat::F16`]): the weight is kept f16-resident in its
///   NATURAL `[out, in]` row-major layout (the transpose is skipped entirely),
///   and the forward is a fused dequant-in-GEMV
///   ([`nn::gemv_f16_batch`]: `out[t,o] = dot(W[o,:], x[t,:]) + b[o]`). Half the
///   resident bytes and half the weight-memory traffic per token.
#[derive(Debug, Clone)]
struct Linear {
    /// Weight in f32-transposed or f16-natural form.
    w: WeightMat,
    /// Optional bias, length `out`.
    bias: Option<Vec<f32>>,
}

impl Linear {
    /// Apply `y = x @ W^T + b` over `x` (`[tq, in]`), returning `[tq, out]`.
    fn forward(&self, x: &Mat) -> FwResult<Mat> {
        match &self.w {
            WeightMat::F32(w_t) => nn::matmul_bias(x, w_t, self.bias.as_deref()),
            WeightMat::F16 { data, out, inp } => {
                if x.cols != *inp {
                    return Err(FwError::InvalidRequest(format!(
                        "Linear(f16) forward: x.cols {} != in {inp}",
                        x.cols
                    )));
                }
                let tq = x.rows;
                let mut y = nn::gemv_out_buf(tq * out);
                nn::gemv_f16_batch(data, *out, *inp, &x.data, tq, self.bias.as_deref(), &mut y);
                Ok(Mat::from_vec(tq, *out, y))
            }
        }
    }
}

/// Affine layer-norm parameters (`weight`, `bias`), each length `n_state`.
#[derive(Debug, Clone)]
struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl LayerNorm {
    /// In-place normalize each row of `x` then apply the affine transform.
    fn apply(&self, x: &mut Mat) {
        nn::layer_norm(x, &self.w, &self.b, LN_EPS);
    }
}

/// All weights of a single decoder block.
#[derive(Debug, Clone)]
struct DecoderLayer {
    attn_ln: LayerNorm,
    attn_q: Linear,
    /// Self-attention key projection — **no bias** in whisper.
    attn_k: Linear,
    attn_v: Linear,
    attn_out: Linear,

    cross_attn_ln: LayerNorm,
    cross_attn_q: Linear,
    /// Cross-attention key projection — **no bias** in whisper.
    cross_attn_k: Linear,
    cross_attn_v: Linear,
    cross_attn_out: Linear,

    mlp_ln: LayerNorm,
    mlp_0: Linear,
    mlp_2: Linear,
}

/// Fully-loaded decoder weights, ready for [`forward_step`].
#[derive(Debug, Clone)]
pub struct DecoderWeights {
    /// Token embedding, kept in its natural `[n_vocab, n_state]` orientation
    /// and reused for both the per-token embedding lookup and the tied output
    /// projection (the logits GEMV). See the module-level "Logits product" note
    /// for the memory rationale.
    ///
    /// Under the f16-compute switch (and an f16-stored embedding) this is held
    /// as raw f16 bits ([`WeightMat::F16`], natural `[n_vocab, n_state]`),
    /// halving the resident footprint of the model's single largest tensor:
    /// the per-token `embed_tokens` lookup dequantizes just the one row it
    /// reads, and `logits_last` runs the fused dequant-GEMV directly over the
    /// natural rows. Otherwise it is a plain f32 [`WeightMat::F32`] in the same
    /// `[n_vocab, n_state]` orientation.
    token_embedding: WeightMat,
    /// Learned positional embedding `[n_text_ctx, n_state]`.
    positional_embedding: Mat,
    layers: Vec<DecoderLayer>,
    /// Final pre-logits layer norm (`decoder.ln`).
    ln: LayerNorm,

    n_state: usize,
    n_head: usize,
    n_vocab: usize,
    n_text_ctx: usize,
}

/// Pull a tensor and assert its logical shape, returning the flat f32 values.
fn tensor_shaped(
    model: &crate::native_engine::ggml::GgmlModel,
    name: &str,
    expected: &[usize],
) -> FwResult<Vec<f32>> {
    let (shape, values) = model.tensor_f32(name)?;
    if shape != expected {
        return Err(FwError::InvalidRequest(format!(
            "decoder tensor '{name}' shape {shape:?} != expected {expected:?}"
        )));
    }
    Ok(values)
}

/// Load a `[rows, cols]` matrix tensor with shape validation.
fn load_mat(
    model: &crate::native_engine::ggml::GgmlModel,
    name: &str,
    rows: usize,
    cols: usize,
) -> FwResult<Mat> {
    let values = tensor_shaped(model, name, &[rows, cols])?;
    Ok(Mat::from_vec(rows, cols, values))
}

/// Load a 1-D vector tensor (e.g. a bias) with length validation.
fn load_vec(
    model: &crate::native_engine::ggml::GgmlModel,
    name: &str,
    len: usize,
) -> FwResult<Vec<f32>> {
    tensor_shaped(model, name, &[len])
}

/// Transpose a `[rows, cols]` matrix into a fresh `[cols, rows]` matrix.
fn transpose(m: &Mat) -> Mat {
    Mat::from_vec(
        m.cols,
        m.rows,
        nn::transpose_parallel(&m.data, m.rows, m.cols),
    )
}

/// Load a whisper linear layer (`[out, in]` weight) plus an optional `[out]`
/// bias.
///
/// When the f16-compute switch ([`crate::native_engine::f16_compute_enabled`])
/// is on AND the weight tensor is stored as f16 in the file, the weight is kept
/// f16-resident in its NATURAL `[out, in]` layout (no transpose, no f32
/// materialization) for the fused dequant-GEMV path. Otherwise — switch off, or
/// an f32-stored tensor — it is dequantized/loaded and pre-transposed to
/// `[in, out]` for the contiguous-sgemm f32 path, exactly as before.
fn load_linear(
    model: &crate::native_engine::ggml::GgmlModel,
    weight_name: &str,
    bias_name: Option<&str>,
    out_dim: usize,
    in_dim: usize,
) -> FwResult<Linear> {
    let want_f16 = crate::native_engine::f16_compute_enabled()
        && model
            .tensor(weight_name)
            .is_some_and(|t| t.dtype == crate::native_engine::GgmlDType::F16);

    let w = if want_f16 {
        // Natural [out, in] f16 bits — skip the transpose entirely. Decode the
        // raw bytes straight to f16-resident `Vec<Float16>` in one parallel pass.
        let (shape, data) = model.tensor_f16_halves(weight_name)?;
        if shape != [out_dim, in_dim] {
            return Err(FwError::InvalidRequest(format!(
                "decoder f16 tensor '{weight_name}' shape {shape:?} != expected [{out_dim}, {in_dim}]"
            )));
        }
        WeightMat::F16 {
            data,
            out: out_dim,
            inp: in_dim,
        }
    } else {
        let w = load_mat(model, weight_name, out_dim, in_dim)?; // [out, in]
        WeightMat::F32(transpose(&w)) // [in, out]
    };

    let bias = match bias_name {
        Some(name) => Some(load_vec(model, name, out_dim)?),
        None => None,
    };
    Ok(Linear { w, bias })
}

/// Load the token embedding `[n_vocab, n_state]` in its NATURAL orientation
/// (no transpose — it is reused as-is for both the per-token lookup and the
/// tied logits GEMV).
///
/// Under the f16-compute switch and an f16-stored embedding, it is kept as raw
/// f16 bits ([`WeightMat::F16`]); otherwise a plain f32 [`WeightMat::F32`].
fn load_embedding(
    model: &crate::native_engine::ggml::GgmlModel,
    name: &str,
    n_vocab: usize,
    n_state: usize,
) -> FwResult<WeightMat> {
    let want_f16 = crate::native_engine::f16_compute_enabled()
        && model
            .tensor(name)
            .is_some_and(|t| t.dtype == crate::native_engine::GgmlDType::F16);
    if want_f16 {
        let (shape, data) = model.tensor_f16_halves(name)?;
        if shape != [n_vocab, n_state] {
            return Err(FwError::InvalidRequest(format!(
                "decoder f16 tensor '{name}' shape {shape:?} != expected [{n_vocab}, {n_state}]"
            )));
        }
        Ok(WeightMat::F16 {
            data,
            out: n_vocab,
            inp: n_state,
        })
    } else {
        Ok(WeightMat::F32(load_mat(model, name, n_vocab, n_state)?))
    }
}

/// Load a layer-norm (`weight`, `bias`), each length `n_state`.
fn load_layer_norm(
    model: &crate::native_engine::ggml::GgmlModel,
    prefix: &str,
    n_state: usize,
) -> FwResult<LayerNorm> {
    let w = load_vec(model, &format!("{prefix}.weight"), n_state)?;
    let b = load_vec(model, &format!("{prefix}.bias"), n_state)?;
    Ok(LayerNorm { w, b })
}

impl DecoderWeights {
    /// Load and pre-transpose every decoder weight from a parsed ggml model.
    ///
    /// All linear weights are transposed to `[in, out]`; the token embedding
    /// is kept `[n_vocab, n_state]`. Shapes are validated against the model's
    /// hyper-parameters, so a malformed/mismatched file is rejected here
    /// rather than corrupting the forward pass.
    ///
    /// # Errors
    /// [`FwError::InvalidRequest`] if any expected tensor is missing or has an
    /// unexpected shape; propagates [`ggml::GgmlModel::tensor_f32`] decode
    /// errors.
    pub fn from_ggml(model: &crate::native_engine::ggml::GgmlModel) -> FwResult<Self> {
        let hp: WhisperHParams = model.hparams;
        let n_state = usize::try_from(hp.n_text_state)
            .map_err(|_| FwError::InvalidRequest("negative n_text_state".into()))?;
        let n_head = usize::try_from(hp.n_text_head)
            .map_err(|_| FwError::InvalidRequest("negative n_text_head".into()))?;
        let n_layer = usize::try_from(hp.n_text_layer)
            .map_err(|_| FwError::InvalidRequest("negative n_text_layer".into()))?;
        let n_vocab = usize::try_from(hp.n_vocab)
            .map_err(|_| FwError::InvalidRequest("negative n_vocab".into()))?;
        let n_text_ctx = usize::try_from(hp.n_text_ctx)
            .map_err(|_| FwError::InvalidRequest("negative n_text_ctx".into()))?;
        if n_head == 0 || !n_state.is_multiple_of(n_head) {
            return Err(FwError::InvalidRequest(format!(
                "n_text_head {n_head} must divide n_text_state {n_state}"
            )));
        }

        let token_embedding =
            load_embedding(model, "decoder.token_embedding.weight", n_vocab, n_state)?;
        let positional_embedding =
            load_mat(model, "decoder.positional_embedding", n_text_ctx, n_state)?;
        let ln = load_layer_norm(model, "decoder.ln", n_state)?;

        let mlp_hidden = 4 * n_state; // whisper MLP expansion factor is 4.
        // PARALLEL layer load (cc, 2026-06-29): the old serial layer loop loaded
        // each layer's weights one after another (each weight using ≤16
        // within-weight workers), leaving most cores idle — DecoderWeights::
        // from_ggml ran at ~1.5 GB/s, ~9× under the ~13.5 GB/s the parallel read
        // achieves. Spreading few layers across rayon (see the gate below) recovers
        // it (measured 1.27× on large-v3-turbo). Each layer is independent.
        let build_layer = |i: usize| -> FwResult<DecoderLayer> {
            let p = format!("decoder.blocks.{i}");
            Ok(DecoderLayer {
                attn_ln: load_layer_norm(model, &format!("{p}.attn_ln"), n_state)?,
                attn_q: load_linear(
                    model,
                    &format!("{p}.attn.query.weight"),
                    Some(&format!("{p}.attn.query.bias")),
                    n_state,
                    n_state,
                )?,
                attn_k: load_linear(
                    model,
                    &format!("{p}.attn.key.weight"),
                    None, // key has no bias
                    n_state,
                    n_state,
                )?,
                attn_v: load_linear(
                    model,
                    &format!("{p}.attn.value.weight"),
                    Some(&format!("{p}.attn.value.bias")),
                    n_state,
                    n_state,
                )?,
                attn_out: load_linear(
                    model,
                    &format!("{p}.attn.out.weight"),
                    Some(&format!("{p}.attn.out.bias")),
                    n_state,
                    n_state,
                )?,
                cross_attn_ln: load_layer_norm(model, &format!("{p}.cross_attn_ln"), n_state)?,
                cross_attn_q: load_linear(
                    model,
                    &format!("{p}.cross_attn.query.weight"),
                    Some(&format!("{p}.cross_attn.query.bias")),
                    n_state,
                    n_state,
                )?,
                cross_attn_k: load_linear(
                    model,
                    &format!("{p}.cross_attn.key.weight"),
                    None, // key has no bias
                    n_state,
                    n_state,
                )?,
                cross_attn_v: load_linear(
                    model,
                    &format!("{p}.cross_attn.value.weight"),
                    Some(&format!("{p}.cross_attn.value.bias")),
                    n_state,
                    n_state,
                )?,
                cross_attn_out: load_linear(
                    model,
                    &format!("{p}.cross_attn.out.weight"),
                    Some(&format!("{p}.cross_attn.out.bias")),
                    n_state,
                    n_state,
                )?,
                mlp_ln: load_layer_norm(model, &format!("{p}.mlp_ln"), n_state)?,
                mlp_0: load_linear(
                    model,
                    &format!("{p}.mlp.0.weight"),
                    Some(&format!("{p}.mlp.0.bias")),
                    mlp_hidden,
                    n_state,
                )?,
                mlp_2: load_linear(
                    model,
                    &format!("{p}.mlp.2.weight"),
                    Some(&format!("{p}.mlp.2.bias")),
                    n_state,
                    mlp_hidden,
                )?,
            })
        };
        // Spread the layers across rayon when there are FEW of them: each
        // `build_layer` still uses ≤16 within-weight workers (`thread::scope`), so
        // the product (n_layer × 16) must stay near the core count to avoid
        // oversubscription. ≤8 layers (turbo/tiny/base = 4) → ~64 threads, the
        // measured 1.27× win; >8 (medium 24 / large-v3 32) keeps the serial loop
        // (32×16 = 512 would thrash). Output is identical (layers independent;
        // `collect` preserves order).
        const MAX_PAR_LAYERS: usize = 8;
        let layers: Vec<DecoderLayer> = if n_layer <= MAX_PAR_LAYERS {
            use rayon::prelude::*;
            (0..n_layer)
                .into_par_iter()
                .map(build_layer)
                .collect::<FwResult<Vec<_>>>()?
        } else {
            (0..n_layer)
                .map(build_layer)
                .collect::<FwResult<Vec<_>>>()?
        };

        Ok(Self {
            token_embedding,
            positional_embedding,
            layers,
            ln,
            n_state,
            n_head,
            n_vocab,
            n_text_ctx,
        })
    }

    /// Number of decoder layers.
    #[must_use]
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }

    /// Number of attention heads.
    #[must_use]
    pub fn n_head(&self) -> usize {
        self.n_head
    }

    /// Hidden width.
    #[must_use]
    pub fn n_state(&self) -> usize {
        self.n_state
    }

    /// Vocabulary size (logits length).
    #[must_use]
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Maximum decode context length (KV-cache capacity).
    #[must_use]
    pub fn n_text_ctx(&self) -> usize {
        self.n_text_ctx
    }
}

/// Per-window decoder state: self-attention KV cache, precomputed cross K/V,
/// and an optional cross-attention weight recording buffer.
///
/// One [`DecoderState`] is built per audio window from that window's encoder
/// output (which fixes the cross-attention keys/values for the whole window)
/// and is then advanced token-by-token via [`forward_step`].
#[derive(Debug)]
pub struct DecoderState {
    /// One self-attention KV cache per layer.
    kv: Vec<KvCache>,
    /// Per-`(layer, head)` pre-transposed cross-key head `[d_head, enc_frames]`
    /// (in `layer * n_head + head` order). The cross K/V are fixed for the
    /// whole window, so the per-step cross path's key transpose and value
    /// gather are hoisted here once at construction instead of being rebuilt
    /// every decode step (a large per-step scatter on wide models — `enc_frames`
    /// is ~1500). Byte-for-byte the per-step gather the old path produced.
    cross_kh_t: Vec<Mat>,
    /// Per-`(layer, head)` gathered cross-value head `[enc_frames, d_head]`
    /// (in `layer * n_head + head` order). See [`Self::cross_kh_t`].
    cross_vh: Vec<Mat>,
    /// Number of tokens currently in the self-attention cache.
    len: usize,
    /// Number of encoder frames (cross-attention key/value length).
    enc_frames: usize,
    n_state: usize,
    n_head: usize,
    /// When set, [`forward_step`] records the softmaxed cross-attention
    /// weights of the most recent call into [`Self::cross_attn_weights`].
    pub record_cross_attn: bool,
    /// Recorded cross-attention weights from the last [`forward_step`], one
    /// `[tokens, enc_frames]` matrix per `(layer, head)` in
    /// `layer * n_head + head` order. Empty unless [`Self::record_cross_attn`]
    /// was set for that call.
    cross_attn_weights: Vec<Mat>,
}

impl DecoderState {
    /// Build per-window state, precomputing cross-attention K/V for every
    /// layer from `encoder_out` (`[enc_frames, n_state]`).
    ///
    /// The cross keys are scaled by `d_head^-0.25` here (matching whisper's
    /// `whisper_build_graph_cross`, which scales `Kcross` once at precompute
    /// time); the query is scaled by the same factor at decode time, so the
    /// `q·k` product carries the full `d_head^-0.5` factor.
    ///
    /// # Errors
    /// [`FwError::InvalidRequest`] if `encoder_out` width != `n_state`;
    /// propagates projection matmul errors.
    pub fn new(w: &DecoderWeights, encoder_out: &Mat) -> FwResult<Self> {
        if encoder_out.cols != w.n_state {
            return Err(FwError::InvalidRequest(format!(
                "encoder_out width {} != decoder n_state {}",
                encoder_out.cols, w.n_state
            )));
        }
        let enc_frames = encoder_out.rows;
        let d_head = w.n_state / w.n_head;
        let k_scale = (d_head as f32).powf(-0.25);

        // Per-layer cross K/V precompute. Each layer's K/V is an independent
        // pair of `[enc_frames, n_state]` projections of the same encoder_out;
        // the per-layer arithmetic is identical to the serial loop (same
        // forward → same scale), only fanned out across layer bands. The
        // results are reassembled in layer order, so `cross_k[li]`/`cross_v[li]`
        // are bit-identical to the serial build. Threshold keeps tiny shapes
        // (and the unit-test synthetic models) serial.
        let n_layer = w.layers.len();
        let compute_layer = |li: usize| -> FwResult<(Mat, Mat)> {
            let layer = &w.layers[li];
            // Kcross = (encoder_out @ Wk^T) * d_head^-0.25  (no bias).
            let mut k = layer.cross_attn_k.forward(encoder_out)?;
            for val in &mut k.data {
                *val *= k_scale;
            }
            // Vcross = encoder_out @ Wv^T + bv.
            let v = layer.cross_attn_v.forward(encoder_out)?;
            Ok((k, v))
        };

        const PAR_THRESHOLD: usize = 1 << 16; // enc_frames*n_state per projection
        let work = enc_frames.saturating_mul(w.n_state);
        let workers = nn::worker_count();
        let mut cross_k: Vec<Mat> = Vec::with_capacity(n_layer);
        let mut cross_v: Vec<Mat> = Vec::with_capacity(n_layer);
        if n_layer < 2 || work < PAR_THRESHOLD || workers < 2 {
            for li in 0..n_layer {
                let (k, v) = compute_layer(li)?;
                cross_k.push(k);
                cross_v.push(v);
            }
        } else {
            let workers = workers.min(n_layer);
            let band = n_layer.div_ceil(workers).max(1);
            let bands: Vec<CrossKvBand> = std::thread::scope(|s| {
                let compute_layer = &compute_layer;
                let mut handles = Vec::new();
                let mut l0 = 0;
                while l0 < n_layer {
                    let l1 = (l0 + band).min(n_layer);
                    handles.push(s.spawn(move || -> CrossKvBand {
                        let mut local = Vec::with_capacity(l1 - l0);
                        for li in l0..l1 {
                            local.push(compute_layer(li)?);
                        }
                        Ok((l0, local))
                    }));
                    l0 = l1;
                }
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            // Reassemble in layer order.
            let mut ordered: Vec<Option<(Mat, Mat)>> = (0..n_layer).map(|_| None).collect();
            for b in bands {
                let (l0, local) = b?;
                for (off, kv) in local.into_iter().enumerate() {
                    ordered[l0 + off] = Some(kv);
                }
            }
            for kv in ordered {
                let (k, v) = kv.expect("every layer computed");
                cross_k.push(k);
                cross_v.push(v);
            }
        }

        let kv = (0..w.layers.len())
            .map(|_| KvCache::new(w.n_text_ctx, w.n_state))
            .collect();

        // Hoist the per-head cross-key transpose ([d_head, enc_frames]) and
        // cross-value gather ([enc_frames, d_head]) out of the per-step loop:
        // the cross K/V are constant for the whole window, so building these
        // once here makes the per-step cross path a pure matmul→softmax→matmul
        // with no scatter. The gathers reproduce exactly the bytes the old
        // per-step path computed (same indexing, same order → bit-identical).
        let n_head = w.n_head;
        let d_head = w.n_state / n_head;
        let mut cross_kh_t: Vec<Mat> = Vec::with_capacity(n_layer * n_head);
        let mut cross_vh: Vec<Mat> = Vec::with_capacity(n_layer * n_head);
        for li in 0..n_layer {
            let ck = &cross_k[li];
            let cv = &cross_v[li];
            for h in 0..n_head {
                let base = h * d_head;
                // kh_t [d_head, enc_frames]: kh_t[d][j] = ck.row(j)[base + d].
                let mut kh_t = vec![0.0f32; d_head * enc_frames];
                for j in 0..enc_frames {
                    let src = &ck.row(j)[base..base + d_head];
                    for (d, &s) in src.iter().enumerate() {
                        kh_t[d * enc_frames + j] = s;
                    }
                }
                cross_kh_t.push(Mat::from_vec(d_head, enc_frames, kh_t));
                // vh [enc_frames, d_head]: row j = cv.row(j)[base..base+d_head].
                let mut vh = vec![0.0f32; enc_frames * d_head];
                for j in 0..enc_frames {
                    let src = &cv.row(j)[base..base + d_head];
                    vh[j * d_head..(j + 1) * d_head].copy_from_slice(src);
                }
                cross_vh.push(Mat::from_vec(enc_frames, d_head, vh));
            }
        }

        Ok(Self {
            kv,
            cross_kh_t,
            cross_vh,
            len: 0,
            enc_frames,
            n_state: w.n_state,
            n_head: w.n_head,
            record_cross_attn: false,
            cross_attn_weights: Vec::new(),
        })
    }

    /// Reset the self-attention cache (and recorded weights) for reuse within
    /// the same window. The precomputed cross K/V are retained.
    pub fn reset(&mut self) {
        for c in &mut self.kv {
            c.reset();
        }
        self.len = 0;
        self.cross_attn_weights.clear();
    }

    /// Number of tokens currently in the self-attention cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no tokens have been decoded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of encoder frames the cross-attention attends over.
    #[must_use]
    pub fn enc_frames(&self) -> usize {
        self.enc_frames
    }

    /// Recorded cross-attention weights from the most recent [`forward_step`]
    /// (when [`Self::record_cross_attn`] was set): one `[tokens, enc_frames]`
    /// matrix per `(layer, head)` in `layer * n_head + head` order.
    #[must_use]
    pub fn cross_attn_weights(&self) -> &[Mat] {
        &self.cross_attn_weights
    }
}

/// Build the per-step input embedding `token_emb[token] + pos_emb[cache_len+i]`
/// for every token in the batch, returning a `[tokens, n_state]` matrix.
fn embed_tokens(w: &DecoderWeights, tokens: &[i32], cache_len: usize) -> FwResult<Mat> {
    let n_state = w.n_state;
    let mut x = vec![0.0f32; tokens.len() * n_state];
    for (i, &tok) in tokens.iter().enumerate() {
        let tok_idx = usize::try_from(tok)
            .map_err(|_| FwError::InvalidRequest(format!("negative token id {tok}")))?;
        if tok_idx >= w.n_vocab {
            return Err(FwError::InvalidRequest(format!(
                "token id {tok_idx} >= n_vocab {}",
                w.n_vocab
            )));
        }
        let pos = cache_len + i;
        if pos >= w.n_text_ctx {
            return Err(FwError::InvalidRequest(format!(
                "decode position {pos} >= n_text_ctx {}",
                w.n_text_ctx
            )));
        }
        let dst = &mut x[i * n_state..(i + 1) * n_state];
        let pe = w.positional_embedding.row(pos);
        // Token embedding row + positional embedding row. For the f16-resident
        // embedding we dequantize just this one row on the fly (one row per
        // token — cheap); the f32 arm borrows the row directly. Either way the
        // value added is the exact stored embedding value.
        match &w.token_embedding {
            WeightMat::F32(emb) => {
                let te = emb.row(tok_idx);
                for ((d, &t), &p) in dst.iter_mut().zip(te).zip(pe) {
                    *d = t + p;
                }
            }
            WeightMat::F16 { data, inp, .. } => {
                let row = &data[tok_idx * inp..(tok_idx + 1) * inp];
                for ((d, &tb), &p) in dst.iter_mut().zip(row).zip(pe) {
                    *d = tb.to_f32() + p;
                }
            }
        }
    }
    Ok(Mat::from_vec(tokens.len(), n_state, x))
}

/// Local multi-head cross-attention with optional weight capture.
///
/// `q` is `[tokens, n_state]`; `cross_k`/`cross_v` are
/// `[enc_frames, n_state]` (the keys already pre-scaled by `d_head^-0.25`).
/// The query is scaled by `d_head^-0.25` so the `q·k` scores carry the full
/// `d_head^-0.5` factor — identical to [`nn::attention`]'s convention but
/// implemented here so the softmaxed `[tokens, enc_frames]` per-head weights
/// can be recorded. There is **no mask** (cross-attention is bidirectional
/// over the encoder frames).
///
/// When `record` is set, `recorded` is populated with one
/// `[tokens, enc_frames]` matrix per head (in head order).
fn cross_attention(
    q: &Mat,
    cross_kh_t: &[Mat],
    cross_vh: &[Mat],
    tk: usize,
    n_head: usize,
    record: bool,
    recorded: &mut Vec<Mat>,
) -> FwResult<Mat> {
    let n_state = q.cols;
    if !n_state.is_multiple_of(n_head) || n_head == 0 {
        return Err(FwError::InvalidRequest(format!(
            "cross_attention: n_head {n_head} must divide n_state {n_state}"
        )));
    }
    let tq = q.rows;
    let d_head = n_state / n_head;
    let q_scale = (d_head as f32).powf(-0.25);

    // Compute one head's scores [tq, tk] and output [tq, d_head]. Per-head
    // math (scaled q·k^T → softmax → @v) is byte-for-byte the serial path;
    // returned so the caller can either record `scores` (serial path) or
    // scatter `out_h` (parallel path) deterministically. The key transpose
    // ([d_head, tk]) and value gather ([tk, d_head]) are precomputed once per
    // window in `DecoderState::new` (cross K/V are window-constant) and passed
    // in as `cross_kh_t[h]` / `cross_vh[h]`, so the per-step path only scales
    // the query and runs the two matmuls + softmax.
    let compute_head = |h: usize| -> FwResult<(Mat, Mat)> {
        let base = h * d_head;

        // Scaled query head [tq, d_head].
        let mut qh = vec![0.0f32; tq * d_head];
        for i in 0..tq {
            let src = &q.row(i)[base..base + d_head];
            let dst = &mut qh[i * d_head..(i + 1) * d_head];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * q_scale;
            }
        }
        let qh = Mat::from_vec(tq, d_head, qh);

        // scores = qh @ kh^T -> [tq, tk], softmax per query row (no mask).
        let mut scores = nn::matmul(&qh, &cross_kh_t[h])?;
        nn::softmax_rows(&mut scores);

        // out_h = scores @ vh  ([tk, d_head] precomputed).
        let out_h = nn::matmul(&scores, &cross_vh[h])?; // [tq, d_head]
        Ok((scores, out_h))
    };

    let scatter = |out: &mut [f32], h: usize, out_h: &Mat| {
        let base = h * d_head;
        for i in 0..tq {
            let src = &out_h.data[i * d_head..(i + 1) * d_head];
            out[i * n_state + base..i * n_state + base + d_head].copy_from_slice(src);
        }
    };

    let mut out = vec![0.0f32; tq * n_state];

    // When recording (DTW word timestamps), the per-head softmax `scores` must
    // land in `recorded` in head order. We still parallelize the per-head COMPUTE
    // via rayon's persistent pool (L12: the per-token spawn, not the compute, was
    // the cost — timestamps are the realistic default, so this path matters), then
    // push scores + scatter SERIALLY in head order. `compute_head` captures only
    // shared refs (it does not touch `recorded`), so it's Sync; ordering and the
    // disjoint scatter are unchanged → bit-identical to the serial loop.
    if record {
        if n_head < 2 || nn::worker_count() < 2 {
            for h in 0..n_head {
                let (scores, out_h) = compute_head(h)?;
                recorded.push(scores);
                scatter(&mut out, h, &out_h);
            }
        } else {
            let heads: Vec<(Mat, Mat)> = (0..n_head)
                .into_par_iter()
                .map(&compute_head)
                .collect::<FwResult<Vec<(Mat, Mat)>>>()?;
            for (h, (scores, out_h)) in heads.into_iter().enumerate() {
                recorded.push(scores);
                scatter(&mut out, h, &out_h);
            }
        }
        return Ok(Mat::from_vec(tq, n_state, out));
    }

    // Steady-state decode path (recording off): parallelize over heads. Each
    // head owns a disjoint column band of `out`; like nn::attention, the
    // merged output is strided per head, so workers scatter into private
    // buffers which we then disjoint-merge (every position written by exactly
    // one head, so `0.0 + x == x` exactly — bit-identical). The work metric is
    // `tq*tk*n_head`; at decode time (tq=1) the wide cross frames (tk≈1500) on
    // many-head models (n_head≈20) still make per-step head parallelism worth
    // the spawn, so the threshold is tuned to engage there while keeping the
    // cheap few-head cases (e.g. tiny's 6 heads) serial.
    // NB (bd-6qih, BlackThrush): raising this to 1<<14 (tiny's 6-head cross-attn
    // serial) REGRESSED the no-timestamps e2e +2.7% (p<0.05) — the parallel path
    // is genuinely faster here. `decoder_attrib`'s tight 400-step loop over-states
    // this sub's spawn cost vs the real e2e (decode interspersed with mel/encode).
    // Kept at 1<<13. (Only the MLP GEMV threshold, L9, was a real e2e spawn win.)
    const PAR_THRESHOLD: usize = 1 << 13; // tq*tk*n_head MACs
    let work = tq.saturating_mul(tk).saturating_mul(n_head);
    let workers = nn::worker_count();
    if n_head < 2 || work < PAR_THRESHOLD || workers < 2 {
        for h in 0..n_head {
            let (_, out_h) = compute_head(h)?;
            scatter(&mut out, h, &out_h);
        }
        return Ok(Mat::from_vec(tq, n_state, out));
    }

    let workers = workers.min(n_head);
    let band = n_head.div_ceil(workers).max(1);
    // Dispatch head bands via rayon's persistent pool (no per-token spawn — the
    // L11 lever, applied to the cross-attn wrapper). Each band scatters its heads
    // into a private buffer; we disjoint-merge below (every position written by
    // exactly one head → bit-identical). compute_head/scatter capture only shared
    // refs, so they're Sync.
    let band_starts: Vec<usize> = (0..n_head).step_by(band).collect();
    let results: Vec<FwResult<Vec<f32>>> = band_starts
        .into_par_iter()
        .map(|h0| -> FwResult<Vec<f32>> {
            let h1 = (h0 + band).min(n_head);
            let mut local = vec![0.0f32; tq * n_state];
            for h in h0..h1 {
                let (_, out_h) = compute_head(h)?;
                scatter(&mut local, h, &out_h);
            }
            Ok(local)
        })
        .collect();
    for r in results {
        let local = r?;
        for (o, l) in out.iter_mut().zip(local.iter()) {
            *o += *l;
        }
    }
    Ok(Mat::from_vec(tq, n_state, out))
}

/// Run one decoder forward step over `tokens`, returning the next-token
/// logits for the **last** token only (length `n_vocab`).
///
/// `tokens` is the newly-decoded batch: typically a single token for an
/// incremental step, or the whole prompt for the initial prefill. Each
/// token's self-attention key/value is appended to `st`'s per-layer KV cache,
/// and positional embeddings are indexed at `st.len() + i`. The checkpoint
/// closure is invoked **between** layers so a caller can cancel a long
/// decode; [`nn`] kernels themselves are uncancellable (see that module's
/// cancellation contract).
///
/// When [`DecoderState::record_cross_attn`] is set, the softmaxed
/// cross-attention weights of this call are captured into
/// [`DecoderState::cross_attn_weights`] (one `[tokens, enc_frames]` per
/// `(layer, head)`), for downstream DTW word-timestamp alignment.
///
/// # Errors
/// [`FwError::InvalidRequest`] on an empty batch, an out-of-range token id,
/// a decode position past `n_text_ctx`, or a width mismatch; propagates the
/// checkpoint closure's error and any kernel error.
pub fn forward_step(
    w: &DecoderWeights,
    st: &mut DecoderState,
    tokens: &[i32],
    checkpoint: &dyn Fn() -> FwResult<()>,
) -> FwResult<Vec<f32>> {
    if tokens.is_empty() {
        return Err(FwError::InvalidRequest(
            "forward_step: empty token batch".into(),
        ));
    }
    if st.n_state != w.n_state || st.n_head != w.n_head {
        return Err(FwError::InvalidRequest(
            "forward_step: state/weights shape mismatch".into(),
        ));
    }

    // Per-sub-part attribution is measurement-only: when perf spans are off
    // (the default) this is `false` and not a single `Instant` is taken, so the
    // production hot path is byte-for-byte the un-instrumented step. When on, a
    // small `timed` helper accumulates each sub-part's wall time into a
    // thread-local (drained by `take_sub_ns`).
    let measure = super::perf_spans_enabled();
    macro_rules! timed {
        ($sub:expr, $body:expr) => {{
            if measure {
                let __t = std::time::Instant::now();
                let __r = $body;
                sub_add($sub, __t.elapsed().as_nanos());
                __r
            } else {
                $body
            }
        }};
    }

    let cache_len = st.len;
    let mut x = timed!(Sub::Embed, embed_tokens(w, tokens, cache_len)?);

    if st.record_cross_attn {
        st.cross_attn_weights.clear();
    }

    for (li, layer) in w.layers.iter().enumerate() {
        // ── self-attention (causal, over the KV cache) ──
        let h = timed!(Sub::SelfLn, {
            let mut h = x.clone();
            layer.attn_ln.apply(&mut h);
            h
        });
        // Q/K/V are three independent projections of the same `h`. On wide
        // models each is a serial GEMV (`[1,n_state] x [n_state,n_state]`, below
        // the kernel's parallel threshold), so running them on separate threads
        // turns three sequential single-core matmuls into one concurrent pass.
        // Each `forward` is unchanged, so the outputs are bit-identical.
        let (q, k, v) = timed!(
            Sub::SelfQkv,
            project_qkv(&layer.attn_q, &layer.attn_k, &layer.attn_v, &h)?
        );
        let attn = timed!(
            Sub::SelfAttn,
            nn::attention_with_cache(&q, &k, &v, w.n_head, &mut st.kv[li])?
        );
        timed!(Sub::SelfOut, {
            let attn_out = layer.attn_out.forward(&attn)?;
            add_into(&mut x, &attn_out);
        });

        // ── cross-attention (encoder K/V, no mask, optional recording) ──
        let hc = timed!(Sub::CrossLn, {
            let mut hc = x.clone();
            layer.cross_attn_ln.apply(&mut hc);
            hc
        });
        let qc = timed!(Sub::CrossQ, layer.cross_attn_q.forward(&hc)?);
        let h0 = li * w.n_head;
        let cross = timed!(
            Sub::CrossAttn,
            cross_attention(
                &qc,
                &st.cross_kh_t[h0..h0 + w.n_head],
                &st.cross_vh[h0..h0 + w.n_head],
                st.enc_frames,
                w.n_head,
                st.record_cross_attn,
                &mut st.cross_attn_weights,
            )?
        );
        timed!(Sub::CrossOut, {
            let cross_out = layer.cross_attn_out.forward(&cross)?;
            add_into(&mut x, &cross_out);
        });

        // ── MLP ──
        let hm = timed!(Sub::MlpLn, {
            let mut hm = x.clone();
            layer.mlp_ln.apply(&mut hm);
            hm
        });
        timed!(Sub::Mlp, {
            let mut ff = layer.mlp_0.forward(&hm)?;
            nn::gelu(&mut ff);
            let ff = layer.mlp_2.forward(&ff)?;
            add_into(&mut x, &ff);
        });

        // Per-layer cancellation point (nn kernels are uncancellable).
        if li + 1 < w.layers.len() {
            checkpoint()?;
        }
    }

    // Advance the logical cache length once (all layers appended in lockstep).
    st.len += tokens.len();

    // Final layer norm, then logits for the last position only.
    timed!(Sub::FinalLn, w.ln.apply(&mut x));
    let last = x.rows - 1;
    let x_last = Mat::from_vec(1, w.n_state, x.row(last).to_vec());
    timed!(Sub::Logits, logits_last(w, &x_last))
}

/// Tied output projection for a single position.
///
/// Computes `logits^T = token_emb @ x_last^T` → `[n_vocab, 1]` from the
/// `[n_vocab, n_state]` embedding and the `[1, n_state]` last hidden row,
/// avoiding a second transposed embedding copy (see the module-level memory
/// note). Returns the flat `[n_vocab]` logits.
///
/// Exposed `pub` (bd-2th6, round-2 bench pass) so the criterion
/// `logits_gemv_large` bench can isolate this `[n_vocab, n_state]` tied product
/// — the direct instrument for the upcoming f16-compute GEMV lever — against a
/// fixed hidden vector. Visibility-only: behavior is unchanged.
pub fn logits_last(w: &DecoderWeights, x_last: &Mat) -> FwResult<Vec<f32>> {
    let n_vocab = w.n_vocab;
    let n_state = w.n_state;

    // f16-resident embedding: fused dequant-GEMV directly over the natural
    // `[n_vocab, n_state]` rows. `out[o] = dot(emb[o, :], x_last)`, contiguous
    // rows, dequant-in-loop — half the weight-memory traffic of the f32 path,
    // and it skips the `x^T` / band-copy bookkeeping entirely (gemv_f16 already
    // parallelizes over output-row bands with the same worker count). This is
    // the numerics-affecting arm gated by `f16_compute_enabled`.
    if let WeightMat::F16 { data, out, inp } = &w.token_embedding {
        debug_assert_eq!((*out, *inp), (n_vocab, n_state));
        let mut logits = nn::gemv_out_buf(*out);
        nn::gemv_f16(data, *out, *inp, &x_last.data, None, &mut logits);
        return Ok(logits);
    }
    let WeightMat::F32(emb_mat) = &w.token_embedding else {
        unreachable!("token_embedding is F32 or F16");
    };

    // x_last^T is [n_state, 1].
    let x_t = Mat::from_vec(w.n_state, 1, x_last.data.clone());

    // This `[n_vocab, n_state] @ [n_state, 1]` GEMV is ~57% of tiny's
    // per-token MACs (51864×384). Each output logit is an independent dot
    // product over the n_state contraction; splitting the *vocab* dimension
    // into row bands and running each band through the SAME `nn::matmul`
    // (ft sgemm) leaves every output element's accumulation order untouched
    // — bit-identical to the single full call — while distributing the rows
    // across worker threads (the GEMV's n=1 inner shape barely rayon-splits
    // on its own, hence the explicit row-band fan-out). Small n_vocab stays
    // serial via the threshold.
    const PAR_THRESHOLD: usize = 1 << 14; // n_vocab*n_state MACs
    let workers = nn::worker_count();
    if n_vocab.saturating_mul(n_state) < PAR_THRESHOLD || workers < 2 {
        let logits = nn::matmul(emb_mat, &x_t)?; // [n_vocab, 1]
        return Ok(logits.data);
    }

    let band = n_vocab.div_ceil(workers).max(1);
    let emb = &emb_mat.data; // [n_vocab, n_state], row-major
    let bands: Vec<FwResult<(usize, Vec<f32>)>> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut r0 = 0;
        while r0 < n_vocab {
            let r1 = (r0 + band).min(n_vocab);
            let x_t = &x_t;
            handles.push(s.spawn(move || -> FwResult<(usize, Vec<f32>)> {
                // Multiply the embedding row band in place (no per-token copy of
                // the band — the embedding is the model's largest tensor, so
                // copying each band out per step dominated the step's memmove).
                let sub = &emb[r0 * n_state..r1 * n_state];
                let part = nn::matmul_raw_lhs(sub, r1 - r0, x_t)?; // [(r1-r0), 1]
                Ok((r0, part.data))
            }));
            r0 = r1;
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut logits = vec![0.0f32; n_vocab];
    for b in bands {
        let (r0, part) = b?;
        logits[r0..r0 + part.len()].copy_from_slice(&part);
    }
    Ok(logits)
}

/// Project `h` through three independent linears (`q`, `k`, `v`) concurrently.
///
/// The self-attention Q/K/V projections share the same input `h` and have no
/// data dependency on each other, so they run on three threads instead of
/// serially. Each [`Linear::forward`] is the unchanged matmul, so the results
/// are bit-identical to computing them in sequence. `k` returns first in the
/// tuple-build order but ordering is irrelevant (disjoint outputs).
fn project_qkv(
    q_lin: &Linear,
    k_lin: &Linear,
    v_lin: &Linear,
    h: &Mat,
) -> FwResult<(Mat, Mat, Mat)> {
    // Two workers + the current thread: k and v on spawned threads while q
    // computes here, so all three run concurrently.
    // NB (bd-6qih, BlackThrush): serializing these for tiny.en was MEASURED ~0 at
    // e2e (566 vs 571 ms, p=0.55) — the L9 MLP-threshold fix already captured the
    // decoder's spawn-bound win, and `project_qkv`'s contention doesn't bite in
    // the real e2e (decode interspersed with mel/encode) the way it did in
    // decoder_attrib's tight loop. Kept concurrent because it helps large models
    // (the 3 projections are [1280,1280]=1.6 M MACs each there).
    std::thread::scope(|s| {
        let kh = s.spawn(|| k_lin.forward(h));
        let vh = s.spawn(|| v_lin.forward(h));
        let q = q_lin.forward(h)?;
        Ok((q, kh.join().unwrap()?, vh.join().unwrap()?))
    })
}

/// In-place `dst += src` (same shape assumed; residual add).
fn add_into(dst: &mut Mat, src: &Mat) {
    for (d, &s) in dst.data.iter_mut().zip(src.data.iter()) {
        *d += s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_engine::find_model_file;
    use crate::native_engine::ggml::GgmlModel;

    /// Deterministic LCG -> f32 in [-1, 1), matching nn.rs's generator style.
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
        fn mat(&mut self, rows: usize, cols: usize) -> Mat {
            let data = (0..rows * cols).map(|_| self.next_f32() * 0.2).collect();
            Mat::from_vec(rows, cols, data)
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32() * 0.2).collect()
        }
    }

    /// Synthetic decoder hyper-parameters for hermetic tests.
    const N_STATE: usize = 8;
    const N_HEAD: usize = 2;
    const N_LAYER: usize = 2;
    const N_VOCAB: usize = 16;
    const N_CTX: usize = 32;

    /// Build a small but structurally complete [`DecoderWeights`] in memory.
    fn synthetic_weights(seed: u64) -> DecoderWeights {
        let mut rng = Lcg::new(seed);
        let lin = |rng: &mut Lcg, out: usize, inp: usize, bias: bool| {
            // Store pre-transposed [in, out] directly (the f32 arm).
            Linear {
                w: WeightMat::F32(rng.mat(inp, out)),
                bias: if bias { Some(rng.vec(out)) } else { None },
            }
        };
        let ln = |rng: &mut Lcg| LayerNorm {
            w: {
                let mut v = rng.vec(N_STATE);
                for x in &mut v {
                    *x += 1.0; // keep scales near 1 for sane activations
                }
                v
            },
            b: rng.vec(N_STATE),
        };
        let mlp_hidden = 4 * N_STATE;
        let mut layers = Vec::new();
        for _ in 0..N_LAYER {
            layers.push(DecoderLayer {
                attn_ln: ln(&mut rng),
                attn_q: lin(&mut rng, N_STATE, N_STATE, true),
                attn_k: lin(&mut rng, N_STATE, N_STATE, false),
                attn_v: lin(&mut rng, N_STATE, N_STATE, true),
                attn_out: lin(&mut rng, N_STATE, N_STATE, true),
                cross_attn_ln: ln(&mut rng),
                cross_attn_q: lin(&mut rng, N_STATE, N_STATE, true),
                cross_attn_k: lin(&mut rng, N_STATE, N_STATE, false),
                cross_attn_v: lin(&mut rng, N_STATE, N_STATE, true),
                cross_attn_out: lin(&mut rng, N_STATE, N_STATE, true),
                mlp_ln: ln(&mut rng),
                mlp_0: lin(&mut rng, mlp_hidden, N_STATE, true),
                mlp_2: lin(&mut rng, N_STATE, mlp_hidden, true),
            });
        }
        DecoderWeights {
            token_embedding: WeightMat::F32(rng.mat(N_VOCAB, N_STATE)),
            positional_embedding: rng.mat(N_CTX, N_STATE),
            layers,
            ln: ln(&mut rng),
            n_state: N_STATE,
            n_head: N_HEAD,
            n_vocab: N_VOCAB,
            n_text_ctx: N_CTX,
        }
    }

    fn noop_checkpoint() -> FwResult<()> {
        Ok(())
    }

    #[test]
    fn forward_step_shapes() {
        let w = synthetic_weights(1);
        let mut rng = Lcg::new(99);
        let enc = rng.mat(6, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        assert_eq!(st.enc_frames(), 6);
        assert!(st.is_empty());

        let logits = forward_step(&w, &mut st, &[1], &noop_checkpoint).unwrap();
        assert_eq!(logits.len(), N_VOCAB);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(st.len(), 1);

        let logits2 = forward_step(&w, &mut st, &[2], &noop_checkpoint).unwrap();
        assert_eq!(logits2.len(), N_VOCAB);
        assert_eq!(st.len(), 2);
    }

    #[test]
    fn incremental_equals_batch_for_last_logits() {
        let w = synthetic_weights(2);
        let mut rng = Lcg::new(7);
        let enc = rng.mat(6, N_STATE);
        let (a, b) = (3i32, 5i32);

        // Incremental: feed [a] then [b]; keep the b-logits.
        let mut st_inc = DecoderState::new(&w, &enc).unwrap();
        let _ = forward_step(&w, &mut st_inc, &[a], &noop_checkpoint).unwrap();
        let inc_b = forward_step(&w, &mut st_inc, &[b], &noop_checkpoint).unwrap();

        // Batch: feed [a, b] in one shot; logits are for the last token (b).
        let mut st_batch = DecoderState::new(&w, &enc).unwrap();
        let batch_b = forward_step(&w, &mut st_batch, &[a, b], &noop_checkpoint).unwrap();

        let max = inc_b
            .iter()
            .zip(&batch_b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-4, "incremental vs batch last-logit diff {max}");
    }

    #[test]
    fn causality_earlier_logits_independent_of_future() {
        // Batch [a, b1] vs [a, b2]: the logits *for position a* (recomputed by
        // taking the first row) must be identical regardless of the later
        // token, since self-attention is causal. We verify by comparing the
        // a-logits obtained from two separate single-token incremental decodes
        // that share the same prefix but diverge afterward.
        let w = synthetic_weights(3);
        let mut rng = Lcg::new(11);
        let enc = rng.mat(6, N_STATE);
        let a = 4i32;

        let mut st1 = DecoderState::new(&w, &enc).unwrap();
        let a_logits_1 = forward_step(&w, &mut st1, &[a], &noop_checkpoint).unwrap();
        // Alter a LATER token in st1.
        let _ = forward_step(&w, &mut st1, &[7], &noop_checkpoint).unwrap();

        let mut st2 = DecoderState::new(&w, &enc).unwrap();
        let a_logits_2 = forward_step(&w, &mut st2, &[a], &noop_checkpoint).unwrap();
        let _ = forward_step(&w, &mut st2, &[9], &noop_checkpoint).unwrap();

        // The a-logits captured *before* the divergent token are identical.
        for (x, y) in a_logits_1.iter().zip(&a_logits_2) {
            assert!((x - y).abs() < 1e-6, "a-logits changed: {x} vs {y}");
        }
    }

    #[test]
    fn cross_attn_dependence_on_encoder_out() {
        let w = synthetic_weights(4);
        let mut rng = Lcg::new(13);
        let enc_a = rng.mat(6, N_STATE);
        let mut enc_b = enc_a.clone();
        for v in &mut enc_b.data {
            *v += 0.5; // perturb the encoder output
        }

        let mut st_a = DecoderState::new(&w, &enc_a).unwrap();
        let la = forward_step(&w, &mut st_a, &[1], &noop_checkpoint).unwrap();

        let mut st_b = DecoderState::new(&w, &enc_b).unwrap();
        let lb = forward_step(&w, &mut st_b, &[1], &noop_checkpoint).unwrap();

        let changed = la.iter().zip(&lb).any(|(x, y)| (x - y).abs() > 1e-4);
        assert!(changed, "logits should depend on encoder_out");
    }

    #[test]
    fn cross_attn_recording_shape_and_normalization() {
        let w = synthetic_weights(5);
        let mut rng = Lcg::new(17);
        let enc_frames = 6;
        let enc = rng.mat(enc_frames, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        st.record_cross_attn = true;

        let n_tokens = 3;
        let toks = [1i32, 2, 3];
        let _ = forward_step(&w, &mut st, &toks, &noop_checkpoint).unwrap();

        let rec = st.cross_attn_weights();
        // One [tokens, enc_frames] matrix per (layer, head).
        assert_eq!(rec.len(), N_LAYER * N_HEAD);
        for m in rec {
            assert_eq!(m.rows, n_tokens);
            assert_eq!(m.cols, enc_frames);
            // Each query row is a softmax distribution: sums to ~1.
            for r in 0..m.rows {
                let s: f32 = m.row(r).iter().sum();
                assert!((s - 1.0).abs() < 1e-4, "cross-attn row sum {s} != 1");
                assert!(m.row(r).iter().all(|&v| v >= 0.0));
            }
        }
    }

    #[test]
    fn recording_off_by_default() {
        let w = synthetic_weights(6);
        let mut rng = Lcg::new(19);
        let enc = rng.mat(6, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        let _ = forward_step(&w, &mut st, &[1, 2], &noop_checkpoint).unwrap();
        assert!(st.cross_attn_weights().is_empty());
    }

    #[test]
    fn reset_clears_cache() {
        let w = synthetic_weights(7);
        let mut rng = Lcg::new(23);
        let enc = rng.mat(6, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        let l1 = forward_step(&w, &mut st, &[3], &noop_checkpoint).unwrap();
        let _ = forward_step(&w, &mut st, &[4], &noop_checkpoint).unwrap();
        assert_eq!(st.len(), 2);
        st.reset();
        assert!(st.is_empty());
        // After reset, decoding the same first token reproduces its logits.
        let l1b = forward_step(&w, &mut st, &[3], &noop_checkpoint).unwrap();
        for (x, y) in l1.iter().zip(&l1b) {
            assert!((x - y).abs() < 1e-6, "reset should restore initial state");
        }
    }

    #[test]
    fn checkpoint_cancellation_propagates() {
        let w = synthetic_weights(8);
        let mut rng = Lcg::new(29);
        let enc = rng.mat(6, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        let cancel = || -> FwResult<()> { Err(FwError::Cancelled("stop".into())) };
        let err = forward_step(&w, &mut st, &[1], &cancel);
        // With N_LAYER=2, the checkpoint fires once (between layer 0 and 1).
        assert!(matches!(err, Err(FwError::Cancelled(_))));
    }

    #[test]
    fn rejects_bad_inputs() {
        let w = synthetic_weights(9);
        let mut rng = Lcg::new(31);
        let enc = rng.mat(6, N_STATE);
        let mut st = DecoderState::new(&w, &enc).unwrap();
        // Empty batch.
        assert!(forward_step(&w, &mut st, &[], &noop_checkpoint).is_err());
        // Out-of-range token id.
        assert!(forward_step(&w, &mut st, &[N_VOCAB as i32], &noop_checkpoint).is_err());
        // Negative token id.
        assert!(forward_step(&w, &mut st, &[-1], &noop_checkpoint).is_err());
        // Encoder width mismatch.
        let bad_enc = Mat::zeros(6, N_STATE + 1);
        assert!(DecoderState::new(&w, &bad_enc).is_err());
    }

    /// Gated: validate `from_ggml` against the real tiny.en model and run a
    /// zero-encoder forward, asserting a finite, valid-id logits vector.
    #[test]
    fn real_tiny_en_from_ggml_and_forward() {
        let Some(path) = find_model_file("tiny.en") else {
            eprintln!("SKIP real_tiny_en_from_ggml_and_forward: ggml-tiny.en.bin not found");
            return;
        };
        let model = GgmlModel::load(&path).expect("load tiny.en");
        let w = DecoderWeights::from_ggml(&model).expect("from_ggml");
        assert_eq!(w.n_vocab(), 51864);
        assert_eq!(w.n_state(), 384);
        assert_eq!(w.n_head(), 6);
        assert_eq!(w.n_layer(), 4);
        assert_eq!(w.n_text_ctx(), 448);

        // Zero encoder output [1500, 384]; build state and forward the SOT
        // token (50257 is whisper's <|startoftranscript|> for tiny.en).
        let enc = Mat::zeros(1500, 384);
        let mut st = DecoderState::new(&w, &enc).expect("decoder state");
        let sot = 50257i32;
        let logits = forward_step(&w, &mut st, &[sot], &noop_checkpoint).expect("forward");
        assert_eq!(logits.len(), 51864);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must be finite"
        );
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(argmax < 51864, "argmax {argmax} must be a valid id");
    }
}
