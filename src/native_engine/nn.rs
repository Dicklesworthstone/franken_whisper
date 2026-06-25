//! Inference micro-op kernels facade + KV-cache multi-head attention.
//!
//! This module is the numerical heart of the native engine: a thin,
//! row-major-`Mat`-centric set of building blocks that the encoder and
//! decoder compose into transformer forward passes.
//!
//! # Kernel integration choices (frankentorch `ft-kernel-cpu`)
//!
//! The single big win from FrankenTorch is its rayon-parallel,
//! `matrixmultiply`-backed sgemm. We delegate **all** matrix multiplies to
//! [`ft_kernel_cpu::matmul_tensor_contiguous_f32`], constructing the
//! required [`ft_core::TensorMeta`] for a contiguous 2-D f32 CPU tensor via
//! [`TensorMeta::from_shape`]`(vec![rows, cols], DType::F32, Device::Cpu)`
//! (this is exactly how `ft-dispatch` builds contiguous metas — no custom
//! strides or storage offset are needed for our row-major `Mat`).
//!
//! For the small, per-row activation ops (`layer_norm`, `softmax_rows`,
//! `gelu`) we implement locally rather than routing through the ft kernels.
//! Rationale, documented per-op below: the ft entry points (e.g.
//! [`ft_kernel_cpu::softmax_dim_tensor_contiguous_f32`]) operate over a
//! generic strided `(outer, dim, inner)` decomposition and return a fresh
//! `Vec`, whereas our hot path wants in-place row updates with f64
//! accumulation (layer_norm) and exact whisper.cpp tanh-GELU semantics
//! (the ft `gelu_value_f32` is the *erf* form — wrong for whisper). Keeping
//! these local avoids an allocation + copy round-trip and a semantic
//! mismatch, while the asymptotically dominant matmuls still get the
//! parallel kernel.
//!
//! # Cancellation contract
//!
//! This module is intentionally **pure**: no function here takes a
//! cancellation / checkpoint closure and none can be cancelled mid-call.
//! The project's `&dyn Fn() -> FwResult<()>` checkpoint contract is honored
//! by *callers* (encoder/decoder), which invoke the checkpoint **between**
//! layer calls — every individual op here is bounded and fast enough that
//! per-op cancellation would only add noise. Keeping nn.rs pure also makes
//! every function trivially testable and free of hidden control flow.
//!
//! # Scaling convention (attention)
//!
//! [`attention`] follows the openai/whisper convention of scaling **both**
//! Q and K by `d_head^-0.25` before the QK^T product, which is numerically
//! equivalent to whisper.cpp's single `1/sqrt(d_head)` factor on the QK
//! scores: `(q·d^-0.25)·(k·d^-0.25) = q·k·d^-0.5 = q·k / sqrt(d)`. See the
//! [`attention`] docs for the whisper.cpp citation.

#![allow(clippy::module_name_repetitions)]

use std::simd::{Simd, StdFloat};

use ft_core::{DType, Device, Float16, TensorMeta};
use half::slice::HalfFloatSliceExt;

use super::Mat;
use crate::error::{FwError, FwResult};

/// Build a contiguous 2-D f32 CPU `TensorMeta` for a `[rows, cols]` tensor.
///
/// Mirrors how `ft-dispatch` constructs metas for a plain contiguous
/// tensor: `from_shape` fills in row-major strides and zero storage offset,
/// which is exactly the layout of our row-major [`Mat`].
fn meta_2d(rows: usize, cols: usize) -> TensorMeta {
    TensorMeta::from_shape(vec![rows, cols], DType::F32, Device::Cpu)
}

/// House-style worker count: available parallelism capped at 8.
///
/// All the parallel-glue kernels below fan out across at most this many
/// `std::thread::scope` workers, mirroring [`transpose_parallel`]. The cap
/// keeps us from oversubscribing the (already rayon-parallel) inner sgemm and
/// matches the empirically-tuned ceiling used elsewhere in this module.
#[inline]
pub(crate) fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(8)
}

/// Worker count for the fused-dequant f16 GEMV ([`gemv_f16`] / its batch form),
/// as a function of the output dimension `out`.
///
/// Unlike the other parallel-glue kernels, the f16 GEMV does **not** nest a
/// rayon-parallel sgemm inside each band (it is a pure per-row
/// `convert_to_f32_slice` + [`dot8`]), so there is no inner pool to
/// oversubscribe — the 8-cap that protects the sgemm kernels buys nothing here.
///
/// The right width is **size-dependent**, and pass-5 criterion measured both
/// regimes on the M4 Pro (10 perf + 4 efficiency cores):
///
/// * **Huge `out` (logits, `out = 51866`, ~133 MB of f16 reads/token):**
///   memory-bandwidth-bound at ~50 GB/s, well under the controller ceiling.
///   Going from 8 → 12 load-issuing threads saturates it better: −2.8% on
///   `logits_gemv_large`. There are ~4300 rows/band even at 12 workers, so band
///   overhead stays negligible.
/// * **Moderate `out` (per-token Linears, `out = 1280`, ~3.3 MB):** NOT
///   bandwidth-bound; only ~107 rows/band at 12 workers, so the extra threads
///   (including the slower efficiency cores) add pure spawn/scheduling overhead
///   — measured **+29%** on `f16_gemv_dequant_1280x1280`. Capping at 8 keeps the
///   prior (good) behavior here.
///
/// So we widen to 12 ONLY past a row threshold where each band still carries
/// substantial work AND the read volume is bandwidth-class (the vocab GEMV is
/// the only decoder shape that qualifies); everything else keeps the 8-cap.
///
/// Row bands are disjoint and each output row's [`dot8`] is independent of the
/// band split, so the worker count is **bit-identical** (order-preserving) —
/// only scheduling changes. The split is purely a performance knob.
#[inline]
fn gemv_worker_count(out: usize) -> usize {
    let avail = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    // Only the vocab-class GEMV (tens of thousands of rows) is bandwidth-bound
    // enough to want >8 threads; below that the 8-cap wins (see fn docs).
    const WIDE_OUT_THRESHOLD: usize = 1 << 14; // 16384 rows
    let cap = if out >= WIDE_OUT_THRESHOLD { 12 } else { 8 };
    avail.min(cap)
}

/// Map a FrankenTorch `KernelError` into [`FwError`].
///
/// Kernel failures here are almost always shape/contract violations from
/// our own callers (mismatched dimensions), so [`FwError::InvalidRequest`]
/// is the right bucket; the kernel's `Display` carries the specifics.
fn kernel_err(e: ft_kernel_cpu::KernelError) -> FwError {
    FwError::InvalidRequest(format!("ft-kernel-cpu: {e}"))
}

/// `[m,k] x [k,n] -> [m,n]`, delegating to FrankenTorch's parallel sgemm.
///
/// # Errors
/// Returns [`FwError::InvalidRequest`] if the inner dimensions disagree
/// (`a.cols != b.rows`) or the kernel rejects the shapes.
pub fn matmul(a: &Mat, b: &Mat) -> FwResult<Mat> {
    if a.cols != b.rows {
        return Err(FwError::InvalidRequest(format!(
            "matmul inner dim mismatch: [{},{}] x [{},{}]",
            a.rows, a.cols, b.rows, b.cols
        )));
    }
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let lhs_meta = meta_2d(m, k);
    let rhs_meta = meta_2d(k, n);
    let data = ft_kernel_cpu::matmul_tensor_contiguous_f32(&a.data, &b.data, &lhs_meta, &rhs_meta)
        .map_err(kernel_err)?;
    Ok(Mat::from_vec(m, n, data))
}

/// `[m,k] x [k,n] -> [m,n]` where the LHS is a **raw row-major slice**.
///
/// Identical to [`matmul`] but the left operand is a flat `[m, k]` slice
/// rather than a [`Mat`], so a caller holding the LHS as a sub-band of a
/// larger backing buffer (e.g. a row band of the token-embedding matrix in
/// the tied logits product) can multiply without first copying the band out
/// into its own `Mat`. The sgemm sees the identical contiguous bytes, so the
/// output is bit-identical to `matmul(&Mat::from_vec(m, k, lhs.to_vec()), b)`.
///
/// # Errors
/// [`FwError::InvalidRequest`] if `lhs.len() != m * b.rows` or the kernel
/// rejects the shapes.
pub fn matmul_raw_lhs(lhs: &[f32], m: usize, b: &Mat) -> FwResult<Mat> {
    let k = b.rows;
    if lhs.len() != m * k {
        return Err(FwError::InvalidRequest(format!(
            "matmul_raw_lhs: lhs len {} != m*k {}",
            lhs.len(),
            m * k
        )));
    }
    let n = b.cols;
    let lhs_meta = meta_2d(m, k);
    let rhs_meta = meta_2d(k, n);
    let data = ft_kernel_cpu::matmul_tensor_contiguous_f32(lhs, &b.data, &lhs_meta, &rhs_meta)
        .map_err(kernel_err)?;
    Ok(Mat::from_vec(m, n, data))
}

/// Affine projection `x @ w_t (+ bias)`.
///
/// Whisper linear layers are `y = x @ W^T + b` with `W` shaped
/// `[out, in]`. To keep every matmul a **contiguous** `[m,k] x [k,n]`, the
/// model loader pre-transposes `W` to `w_t` of shape `[in, out]` once at
/// load time, so this function is a plain `x @ w_t` plus a broadcast bias
/// add over rows. `bias` (when present) must have length `w_t.cols`
/// (= `out`).
///
/// # Errors
/// [`FwError::InvalidRequest`] on a dimension mismatch (`x.cols != w_t.rows`
/// or `bias.len() != w_t.cols`) or kernel rejection.
pub fn matmul_bias(x: &Mat, w_t: &Mat, bias: Option<&[f32]>) -> FwResult<Mat> {
    let mut out = matmul(x, w_t)?;
    if let Some(b) = bias {
        if b.len() != out.cols {
            return Err(FwError::InvalidRequest(format!(
                "matmul_bias: bias len {} != out cols {}",
                b.len(),
                out.cols
            )));
        }
        let cols = out.cols;
        for row in out.data.chunks_mut(cols) {
            for (v, &bv) in row.iter_mut().zip(b.iter()) {
                *v += bv;
            }
        }
    }
    Ok(out)
}

/// A linear-layer weight matrix in EITHER representation.
///
/// The f32 path stores a pre-transposed `[in, out]` [`Mat`] (so the forward is
/// a contiguous `x @ w_t`); the f16 path keeps the weight in its **natural**
/// ggml `[out, in]` row-major layout as raw f16 bit patterns and runs a fused
/// dequant-in-GEMV ([`gemv_f16`]) — `out[o] = dot(W[o, :], x)`, contiguous rows,
/// no transpose, half the resident bytes.
///
/// Which arm a given weight uses is decided once at load time by the
/// [`super::f16_compute_enabled`] switch AND the source dtype: only tensors
/// that are f16 in the ggml file ever take the [`Self::F16`] arm; f32-stored
/// tensors (and the whole model when the switch is off) stay [`Self::F32`].
#[derive(Debug, Clone)]
pub enum WeightMat {
    /// Pre-transposed `[in, out]` f32 weight for the contiguous-sgemm path.
    F32(Mat),
    /// Natural `[out, in]` f16 weight (typed [`Float16`] = `half::f16`,
    /// row-major), dequantized on the fly by [`gemv_f16`]. Stored as typed
    /// halves (not raw `u16`) so the GEMV kernels can use the SIMD bulk
    /// [`HalfFloatSliceExt::convert_to_f32_slice`] dequant (4-wide aarch64
    /// `fp16` / 8-wide x86 `f16c`) instead of a per-element scalar widen — the
    /// per-element widen inside the dot loop blocked FMA vectorization and was
    /// the pass-2 e2e regression root cause (see module/kernel docs).
    F16 {
        /// Typed IEEE-754 halves, `out * in` elements row-major.
        data: Vec<Float16>,
        /// Output dimension (number of rows of the natural weight).
        out: usize,
        /// Input dimension (contraction length; number of columns).
        inp: usize,
    },
}

/// Vectorizable f32 dot product `sum(a[i] * b[i])` over equal-length slices,
/// using eight independent partial accumulators so LLVM lowers the body to a
/// SIMD multiply-add over 8-lane chunks (the scalar single-accumulator form is
/// a serial dependency chain that does NOT vectorize). The remainder past the
/// last full chunk is summed scalar.
///
/// Numerics: the chunk-of-8 partial layout fixes a specific, deterministic
/// summation tree (lane `i` accumulates elements `i, i+8, i+16, …`, then the
/// eight lanes are reduced left-to-right) — bit-reproducible for a given length
/// regardless of build, but a *different* order than a single running f32
/// accumulator. The f16 GEMV is already a numerics-affecting path vs the
/// f32-sgemm reference (gated by [`super::f16_compute_enabled`]); this only
/// changes which non-reference f32 order it uses, and is conformance-gated.
#[inline]
#[must_use]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let mut ac = a.chunks_exact(8);
    let mut bc = b.chunks_exact(8);
    for (ach, bch) in ac.by_ref().zip(bc.by_ref()) {
        for i in 0..8 {
            acc[i] += ach[i] * bch[i];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (&av, &bv) in ac.remainder().iter().zip(bc.remainder().iter()) {
        s += av * bv;
    }
    s
}

/// Fused dequant + GEMV: `out[o] = bias[o] + dot(W[o, :], x)` for a natural
/// `[out, in]` row-major f16 weight `w_f16` and an `[in]` activation `x`.
///
/// Each output row is an independent dot product over the `in`-dim contraction.
/// The weight row is first dequantized **in bulk** into a small reused f32
/// scratch buffer via the SIMD [`HalfFloatSliceExt::convert_to_f32_slice`]
/// (4-wide aarch64 `fp16` / 8-wide x86 `f16c`), then dotted against `x` with the
/// vectorizable [`dot8`] (8-lane FMA, f32 accumulator). This split — bulk SIMD
/// dequant, then a separate vectorized f32 dot — is ~4x the per-element
/// dequant-inside-the-dot-loop form it replaces (which serialized both the half
/// widen and the FMA): measured 3.0 → 13.5 GFLOP/s on a `[1280,1280]` row on
/// M4 Pro. Output rows are disjoint, so we fan out over contiguous row bands
/// with the house worker count; each worker owns a private scratch row buffer.
/// Tiny shapes stay serial via the element threshold so they never pay spawn
/// overhead.
///
/// Numerics: the per-element dequant is exact (bit-identical to the f32
/// loader's `half` widen — see the exhaustive 65536-value test); the f32 dot
/// uses [`dot8`]'s deterministic chunk-of-8 order, which differs from
/// `matrixmultiply`'s blocked kernel, so this stays a numerics-affecting path
/// (gated by [`super::f16_compute_enabled`]).
///
/// # Panics (debug) / contract
/// `w_f16.len()` must equal `out * inp`, `x.len() == inp`, `out_slice.len() ==
/// out`, and `bias` (if present) length `out`. Callers are model-shaped, so a
/// mismatch is a load bug; the debug asserts catch it in tests.
pub fn gemv_f16(
    w_f16: &[Float16],
    out: usize,
    inp: usize,
    x: &[f32],
    bias: Option<&[f32]>,
    out_slice: &mut [f32],
) {
    debug_assert_eq!(w_f16.len(), out * inp, "gemv_f16 weight shape mismatch");
    debug_assert_eq!(x.len(), inp, "gemv_f16 x length mismatch");
    debug_assert_eq!(out_slice.len(), out, "gemv_f16 out length mismatch");
    debug_assert!(
        bias.is_none_or(|b| b.len() == out),
        "gemv_f16 bias length mismatch"
    );

    // One output row: bulk-SIMD dequant into `scratch`, then vectorized dot.
    let row_dot = |o: usize, scratch: &mut [f32]| -> f32 {
        let w_row = &w_f16[o * inp..(o + 1) * inp];
        w_row.convert_to_f32_slice(scratch);
        let acc = dot8(scratch, x);
        match bias {
            Some(b) => acc + b[o],
            None => acc,
        }
    };

    // MACs of real work = out * inp. Below the threshold the spawn cost
    // dominates, so stay serial. The crossover is measured (M4 Pro, 10P+4E,
    // a serial-vs-`thread::scope` sweep): at `out*inp` up to ~393 k the serial
    // path is 1.5–3.7× faster (the [384,384] tiny per-token Linears were 3.7×
    // slower in the 8-way scope path — pure spawn/join overhead on ~9 µs of
    // compute); the two paths break even at ~590 k and the scope path pulls
    // clearly ahead by ~1.6 M ([1280,1280] large) and at the vocab GEMV
    // ([51864,384]). `1 << 19` (524 288) sits in that break-even band: it keeps
    // EVERY tiny [384,384] per-token Linear (self q/k/v, self_out, cross_q,
    // cross_out) serial — the round-3 self_qkv spawn-bound hotspot — while still
    // spawning the large-model Linears and the logits GEMV. The split is a pure
    // scheduling knob (disjoint row bands, each row's [`dot8`] order is
    // band-independent), so it is bit-identical either way.
    const PAR_THRESHOLD: usize = 1 << 19;
    let workers = gemv_worker_count(out);
    if out * inp < PAR_THRESHOLD || workers < 2 {
        let mut scratch = vec![0.0f32; inp];
        for (o, slot) in out_slice.iter_mut().enumerate() {
            *slot = row_dot(o, &mut scratch);
        }
        return;
    }
    let band = out.div_ceil(workers).max(1);
    std::thread::scope(|s| {
        let row_dot = &row_dot;
        for (w, band_slice) in out_slice.chunks_mut(band).enumerate() {
            let o_base = w * band;
            s.spawn(move || {
                let mut scratch = vec![0.0f32; inp];
                for (i, slot) in band_slice.iter_mut().enumerate() {
                    *slot = row_dot(o_base + i, &mut scratch);
                }
            });
        }
    });
}

/// Batched fused dequant + GEMV: `out[t, o] = bias[o] + dot(W[o, :], x[t, :])`
/// for `tq` activation rows `x` (`[tq, in]` row-major) against a natural
/// `[out, in]` f16 weight, producing `[tq, out]` row-major.
///
/// Used by the prefill (multi-token batch). Each `(t, o)` is an independent
/// dot product; we parallelize over the OUTPUT-row dimension (disjoint output
/// columns across the whole `[tq, out]` block). Within a band we dequantize
/// each weight row ONCE (bulk SIMD [`HalfFloatSliceExt::convert_to_f32_slice`]
/// into a reused scratch buffer) and then dot it against all `tq` token rows
/// with the vectorizable [`dot8`], amortizing the dequant over the batch. The
/// per-`(t,o)` math is identical to calling [`gemv_f16`] once per token (same
/// [`dot8`] order), so results match; for `tq == 1` this reduces to a single
/// GEMV.
///
/// # Contract
/// `w_f16.len() == out * inp`, `x.len() == tq * inp`, `out_slice.len() ==
/// tq * out`, `bias` (if present) length `out`.
pub fn gemv_f16_batch(
    w_f16: &[Float16],
    out: usize,
    inp: usize,
    x: &[f32],
    tq: usize,
    bias: Option<&[f32]>,
    out_slice: &mut [f32],
) {
    debug_assert_eq!(
        w_f16.len(),
        out * inp,
        "gemv_f16_batch weight shape mismatch"
    );
    debug_assert_eq!(x.len(), tq * inp, "gemv_f16_batch x length mismatch");
    debug_assert_eq!(
        out_slice.len(),
        tq * out,
        "gemv_f16_batch out length mismatch"
    );

    if tq == 1 {
        gemv_f16(w_f16, out, inp, x, bias, out_slice);
        return;
    }

    // Compute the column band [o0, o1) for every token row. `out_slice` is
    // `[tq, out]` row-major, so a column band is strided per token; we write it
    // directly (each band owns disjoint columns → no overlap across workers).
    let compute_band = |o0: usize, o1: usize, dst: &mut [f32]| {
        // dst is the FULL [tq, out] buffer in serial mode, or in parallel mode a
        // per-worker private [tq, out] buffer it later disjoint-merges. Either
        // way we write only columns [o0, o1). One reused dequant scratch row.
        let mut scratch = vec![0.0f32; inp];
        for o in o0..o1 {
            let w_row = &w_f16[o * inp..(o + 1) * inp];
            w_row.convert_to_f32_slice(&mut scratch);
            let b = bias.map_or(0.0, |bb| bb[o]);
            for t in 0..tq {
                let xr = &x[t * inp..(t + 1) * inp];
                dst[t * out + o] = dot8(&scratch, xr) + b;
            }
        }
    };

    // Same measured crossover as [`gemv_f16`] (see its `PAR_THRESHOLD` note),
    // but the work metric carries the batch dimension: each weight row is
    // dequantized once and dotted against all `tq` token rows, so the spawn is
    // amortized over `tq * out * inp` MACs. `1 << 19` keeps small prefills
    // serial while still parallelizing the realistic multi-token prompt batches.
    const PAR_THRESHOLD: usize = 1 << 19;
    let workers = gemv_worker_count(out);
    if tq * out * inp < PAR_THRESHOLD || workers < 2 {
        compute_band(0, out, out_slice);
        return;
    }

    // Parallelize over output-column bands; each worker fills a private
    // [tq, out] buffer (writing only its band), then we disjoint-merge them
    // (every column written by exactly one worker → `0.0 + x == x` exactly).
    let band = out.div_ceil(workers).max(1);
    let parts: Vec<(usize, usize, Vec<f32>)> = std::thread::scope(|s| {
        let compute_band = &compute_band;
        let mut handles = Vec::new();
        let mut o0 = 0;
        while o0 < out {
            let o1 = (o0 + band).min(out);
            handles.push(s.spawn(move || {
                let mut local = vec![0.0f32; tq * out];
                compute_band(o0, o1, &mut local);
                (o0, o1, local)
            }));
            o0 = o1;
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (o0, o1, local) in parts {
        for t in 0..tq {
            let dst = &mut out_slice[t * out + o0..t * out + o1];
            dst.copy_from_slice(&local[t * out + o0..t * out + o1]);
        }
    }
}

/// In-place per-row layer normalization with affine scale/shift.
///
/// For each row: subtract the row mean, divide by `sqrt(var + eps)`, then
/// apply `w * x + b` element-wise. Mean and variance use **f64**
/// accumulation for numerical stability (whisper hidden dims of 384..1280
/// make naive f32 sums lossy). Rows are independent.
///
/// Implemented locally (not via an ft kernel) because we want the in-place,
/// fused mean/var/affine pass over each row rather than an allocating
/// reduction; see module docs.
///
/// `w` and `b` must each have length `x.cols`; on mismatch this is a no-op
/// guard (callers pass model-shaped weights, so a mismatch is a load bug).
pub fn layer_norm(x: &mut Mat, w: &[f32], b: &[f32], eps: f32) {
    let cols = x.cols;
    if cols == 0 || w.len() != cols || b.len() != cols {
        return;
    }
    let eps = f64::from(eps);
    // Rows are independent, so we fan out over contiguous row bands; each band
    // owns a disjoint slice of `x.data`. PAR_THRESHOLD is in elements
    // (rows*cols) so tiny decoder shapes ([1..7, 384]) stay serial and never
    // pay spawn overhead. Within each band [`norm_rows`] vectorizes 8 rows at a
    // time (one row per f64 lane).
    const PAR_THRESHOLD: usize = 1 << 16;
    let rows = x.rows;
    if rows * cols < PAR_THRESHOLD || worker_count() < 2 {
        norm_rows(&mut x.data, cols, w, b, eps);
        return;
    }
    let band_rows = rows.div_ceil(worker_count()).max(1);
    std::thread::scope(|s| {
        for band in x.data.chunks_mut(band_rows * cols) {
            s.spawn(move || {
                norm_rows(band, cols, w, b, eps);
            });
        }
    });
}

/// Layer-norm a contiguous block of `block.len() / cols` rows in place.
///
/// Processes 8 rows at a time with **vertical SIMD** — one row per `f64x8` lane,
/// so each lane reduces its own row in the same ascending order as the scalar
/// loop. IEEE-754 f64 lane ops, plus correctly-rounded `sqrt`/division, are
/// bit-identical to scalar f64, so the result is **byte-for-byte** the same as
/// the per-row scalar path (proven by `layer_norm_simd_matches_scalar`). The
/// `< 8`-row tail runs scalar. Mean/var in f64 mirrors whisper.cpp.
fn norm_rows(block: &mut [f32], cols: usize, w: &[f32], b: &[f32], eps: f64) {
    const L: usize = 8;
    type V = Simd<f64, L>;
    let n = cols as f64;
    let nrows = block.len() / cols;
    let nfull = nrows - nrows % L;

    let mut soa = vec![V::splat(0.0); cols]; // reused per 8-row group
    let mut g = 0;
    while g < nfull {
        // Gather 8 rows into structure-of-arrays: soa[j] = element j of 8 rows.
        for (j, s) in soa.iter_mut().enumerate() {
            let mut a = [0.0f64; L];
            for (lane, al) in a.iter_mut().enumerate() {
                *al = f64::from(block[(g + lane) * cols + j]);
            }
            *s = V::from_array(a);
        }
        let mut sum = V::splat(0.0);
        for s in &soa {
            sum += *s;
        }
        let mean = sum / V::splat(n);
        let mut var = V::splat(0.0);
        for s in &soa {
            let d = *s - mean;
            var += d * d;
        }
        var /= V::splat(n);
        let inv = V::splat(1.0) / (var + V::splat(eps)).sqrt();
        for (j, s) in soa.iter().enumerate() {
            let normed =
                (*s - mean) * inv * V::splat(f64::from(w[j])) + V::splat(f64::from(b[j]));
            let arr = normed.to_array();
            for (lane, &val) in arr.iter().enumerate() {
                block[(g + lane) * cols + j] = val as f32;
            }
        }
        g += L;
    }

    // Scalar tail (< 8 remaining rows) — identical per-row f64 math.
    for r in nfull..nrows {
        let row = &mut block[r * cols..(r + 1) * cols];
        let mut sum = 0.0f64;
        for &v in row.iter() {
            sum += f64::from(v);
        }
        let mean = sum / n;
        let mut var = 0.0f64;
        for &v in row.iter() {
            let d = f64::from(v) - mean;
            var += d * d;
        }
        var /= n;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ((v, &wi), &bi) in row.iter_mut().zip(w.iter()).zip(b.iter()) {
            let normed = (f64::from(*v) - mean) * inv_std;
            *v = (normed * f64::from(wi) + f64::from(bi)) as f32;
        }
    }
}

/// whisper.cpp coefficient `sqrt(2/pi)` (`SQRT_2_OVER_PI` in ggml `vec.h`).
const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
/// whisper.cpp `GELU_COEF_A`.
const GELU_COEF_A: f32 = 0.044_715;

/// In-place exact whisper.cpp tanh-approximation GELU.
///
/// Matches ggml `ggml_gelu_f32` in
/// `ggml/src/ggml-cpu/vec.h`:
/// `0.5*x*(1 + tanh(sqrt(2/pi) * x * (1 + 0.044715*x*x)))`,
/// which is the standard `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3)))`
/// (note `x*(1 + a*x*x) == x + a*x^3`). We deliberately use this tanh form
/// rather than ft's `gelu_value_f32`, which is the *erf* GELU and would
/// diverge from whisper's activations.
pub fn gelu(x: &mut Mat) {
    // Pure elementwise: each output depends only on its own input, so we
    // split `data` into disjoint contiguous chunks across workers. The tanh
    // transcendental dominates, so this scales well; threshold keeps small
    // activations serial.
    let apply = |v: &mut f32| {
        let x = *v;
        *v = 0.5 * x * (1.0 + (GELU_SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh());
    };

    const PAR_THRESHOLD: usize = 1 << 15;
    let n = x.data.len();
    if n < PAR_THRESHOLD || worker_count() < 2 {
        for v in &mut x.data {
            apply(v);
        }
        return;
    }
    let chunk = n.div_ceil(worker_count()).max(1);
    std::thread::scope(|s| {
        let apply = &apply;
        for band in x.data.chunks_mut(chunk) {
            s.spawn(move || {
                for v in band.iter_mut() {
                    apply(v);
                }
            });
        }
    });
}

/// In-place numerically-stable per-row softmax (max-subtract).
///
/// Each row is softmaxed independently: subtract the row max before
/// exponentiating (so large logits like `1e30` don't overflow to `inf`),
/// then normalize by the row sum. Implemented locally to operate in place
/// over `Mat` rows; see module docs.
pub fn softmax_rows(x: &mut Mat) {
    let cols = x.cols;
    if cols == 0 {
        return;
    }
    // Per-row max-subtract / exp / normalize, order unchanged. Rows are
    // independent, so fan out over contiguous row bands (disjoint slices of
    // `x.data`). Threshold in elements keeps small score matrices serial.
    let softmax_row = |row: &mut [f32]| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            // All -inf (e.g. fully masked row): leave as-is to avoid NaNs.
            return;
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            let e = (*v - max).exp();
            *v = e;
            sum += e;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    };

    const PAR_THRESHOLD: usize = 1 << 16;
    let rows = x.rows;
    if rows * cols < PAR_THRESHOLD || worker_count() < 2 {
        for row in x.data.chunks_mut(cols) {
            softmax_row(row);
        }
        return;
    }
    let band_rows = rows.div_ceil(worker_count()).max(1);
    std::thread::scope(|s| {
        let softmax_row = &softmax_row;
        for band in x.data.chunks_mut(band_rows * cols) {
            s.spawn(move || {
                for row in band.chunks_mut(cols) {
                    softmax_row(row);
                }
            });
        }
    });
}

/// 1-D convolution via im2col + sgemm.
///
/// - `x` is `[T, Cin]` (time-major, channel-minor — whisper's mel/conv
///   activation layout).
/// - `w` is the flat `[Cout, Cin, K]` weight (row-major: index
///   `co*Cin*K + ci*K + kk`).
/// - `bias` has length `Cout`.
/// - Output is `[T_out, Cout]` with `T_out = (T + 2*pad - K)/stride + 1`.
///
/// We build the im2col matrix `[T_out, Cin*K]` (each output position's
/// receptive field flattened in `(ci, kk)` order), reshape the weights to
/// `[Cin*K, Cout]` (transposing `[Cout, Cin*K]`), and a single
/// [`matmul`] yields `[T_out, Cout]`; the bias is then broadcast-added.
///
/// # Errors
/// [`FwError::InvalidRequest`] if `x.cols != cin`, `w.len() != cout*cin*k`,
/// `bias.len() != cout`, `stride == 0`, or the padded input is shorter than
/// the kernel (empty output).
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    x: &Mat,
    w: &[f32],
    cout: usize,
    cin: usize,
    k: usize,
    bias: &[f32],
    stride: usize,
    pad: usize,
) -> FwResult<Mat> {
    if x.cols != cin {
        return Err(FwError::InvalidRequest(format!(
            "conv1d: x.cols {} != cin {cin}",
            x.cols
        )));
    }
    if w.len() != cout * cin * k {
        return Err(FwError::InvalidRequest(format!(
            "conv1d: weight len {} != cout*cin*k = {}",
            w.len(),
            cout * cin * k
        )));
    }
    if bias.len() != cout {
        return Err(FwError::InvalidRequest(format!(
            "conv1d: bias len {} != cout {cout}",
            bias.len()
        )));
    }
    if stride == 0 {
        return Err(FwError::InvalidRequest("conv1d: stride must be > 0".into()));
    }
    let t_in = x.rows;
    let padded = t_in + 2 * pad;
    if padded < k {
        return Err(FwError::InvalidRequest(format!(
            "conv1d: padded length {padded} < kernel {k}"
        )));
    }
    let t_out = (padded - k) / stride + 1;

    // im2col: [T_out, Cin*K], column index = ci*K + kk.
    // Pure gather: each output-time row `o` writes only its own
    // `cols[o*patch..(o+1)*patch]` band, so the construction fans out over
    // contiguous output-row bands. Each row reads disjoint output but shared
    // (read-only) `x`. Threshold in elements keeps small convs serial.
    let patch = cin * k;
    let mut cols = vec![0.0f32; t_out * patch];
    let fill_row = |o: usize, row: &mut [f32]| {
        let start = o * stride; // position in the padded input
        for kk in 0..k {
            let p = start + kk; // padded index
            // map padded index back to real input index
            if p < pad {
                continue; // left zero-pad
            }
            let ti = p - pad;
            if ti >= t_in {
                continue; // right zero-pad
            }
            let src = x.row(ti); // [Cin]
            for ci in 0..cin {
                row[ci * k + kk] = src[ci];
            }
        }
    };

    const PAR_THRESHOLD: usize = 1 << 16;
    if t_out * patch < PAR_THRESHOLD || worker_count() < 2 {
        for (o, row) in cols.chunks_mut(patch).enumerate() {
            fill_row(o, row);
        }
    } else {
        let band_rows = t_out.div_ceil(worker_count()).max(1);
        std::thread::scope(|s| {
            let fill_row = &fill_row;
            for (w, band) in cols.chunks_mut(band_rows * patch).enumerate() {
                let o_base = w * band_rows;
                s.spawn(move || {
                    for (i, row) in band.chunks_mut(patch).enumerate() {
                        fill_row(o_base + i, row);
                    }
                });
            }
        });
    }
    let im2col = Mat::from_vec(t_out, patch, cols);

    // Reshape weights [Cout, Cin*K] -> w_t [Cin*K, Cout] (transpose).
    let mut w_t = vec![0.0f32; patch * cout];
    for co in 0..cout {
        for j in 0..patch {
            w_t[j * cout + co] = w[co * patch + j];
        }
    }
    let w_t = Mat::from_vec(patch, cout, w_t);

    // [T_out, Cin*K] x [Cin*K, Cout] -> [T_out, Cout], then add bias.
    matmul_bias(&im2col, &w_t, Some(bias))
}

/// Incremental key/value cache for autoregressive self-attention.
///
/// Stores keys and values as contiguous `[capacity_tokens, n_state]` row-
/// major buffers; [`KvCache::append`] copies new per-token rows in and
/// advances `len`. [`KvCache::keys`] / [`KvCache::values`] expose the
/// populated prefix as a `[len, n_state]` [`Mat`] for [`attention`].
#[derive(Debug, Clone)]
pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    len: usize,
    capacity_tokens: usize,
    n_state: usize,
}

impl KvCache {
    /// Allocate a cache for up to `capacity_tokens` tokens of width
    /// `n_state`.
    #[must_use]
    pub fn new(capacity_tokens: usize, n_state: usize) -> Self {
        Self {
            k: vec![0.0; capacity_tokens * n_state],
            v: vec![0.0; capacity_tokens * n_state],
            len: 0,
            capacity_tokens,
            n_state,
        }
    }

    /// Number of tokens currently cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Per-token width.
    #[must_use]
    pub fn n_state(&self) -> usize {
        self.n_state
    }

    /// Clear the cache (retains the allocation).
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Append `k`/`v` rows (`[t, n_state]` each) to the cache.
    ///
    /// # Errors
    /// [`FwError::InvalidRequest`] if widths disagree with `n_state` or the
    /// append would exceed `capacity_tokens`.
    pub fn append(&mut self, k: &Mat, v: &Mat) -> FwResult<()> {
        if k.cols != self.n_state || v.cols != self.n_state {
            return Err(FwError::InvalidRequest(format!(
                "KvCache::append width mismatch: n_state={}, k.cols={}, v.cols={}",
                self.n_state, k.cols, v.cols
            )));
        }
        if k.rows != v.rows {
            return Err(FwError::InvalidRequest(format!(
                "KvCache::append row mismatch: k.rows={}, v.rows={}",
                k.rows, v.rows
            )));
        }
        let t = k.rows;
        if self.len + t > self.capacity_tokens {
            return Err(FwError::InvalidRequest(format!(
                "KvCache::append overflow: len {} + {t} > capacity {}",
                self.len, self.capacity_tokens
            )));
        }
        let off = self.len * self.n_state;
        let span = t * self.n_state;
        self.k[off..off + span].copy_from_slice(&k.data);
        self.v[off..off + span].copy_from_slice(&v.data);
        self.len += t;
        Ok(())
    }

    /// View of the cached keys as a `[len, n_state]` matrix.
    #[must_use]
    pub fn keys(&self) -> Mat {
        Mat::from_vec(
            self.len,
            self.n_state,
            self.k[..self.len * self.n_state].to_vec(),
        )
    }

    /// View of the cached values as a `[len, n_state]` matrix.
    #[must_use]
    pub fn values(&self) -> Mat {
        Mat::from_vec(
            self.len,
            self.n_state,
            self.v[..self.len * self.n_state].to_vec(),
        )
    }

    /// Borrow the populated key prefix as a contiguous `[len, n_state]`
    /// row-major slice (no copy). Same bytes as [`Self::keys`]`.data`.
    #[must_use]
    pub fn key_slice(&self) -> &[f32] {
        &self.k[..self.len * self.n_state]
    }

    /// Borrow the populated value prefix as a contiguous `[len, n_state]`
    /// row-major slice (no copy). Same bytes as [`Self::values`]`.data`.
    #[must_use]
    pub fn value_slice(&self) -> &[f32] {
        &self.v[..self.len * self.n_state]
    }
}

/// Multi-head scaled-dot-product attention.
///
/// - `q` is `[Tq, n_state]`, `k`/`v` are `[Tk, n_state]`.
/// - `n_head` must divide `n_state`; per-head width is `d_head =
///   n_state / n_head`.
/// - `causal_offset`: `None` for full (cross / bidirectional) attention;
///   `Some(cache_len)` for causal self-attention where query position `i`
///   attends to all keys with index `<= cache_len + i`. (For a fresh decode
///   over the whole prompt `cache_len = 0`; for an incremental single-token
///   step `cache_len = past_len` and `Tq = 1`.)
///
/// Heads are split along the state dimension, each head's `q`/`k` rows are
/// scaled by `d_head^-0.25`, then per-head `qk^T` scores are masked (causal,
/// if requested), softmaxed per query row, and multiplied by `v`; the head
/// outputs are concatenated back to `[Tq, n_state]`.
///
/// # Scaling convention
/// Scaling both Q and K by `d_head^-0.25` reproduces the openai/whisper
/// scaling and is numerically equal to whisper.cpp's single
/// `KQscale = 1/sqrt(d_head)` applied to the QK scores
/// (`whisper.cpp` decoder path scales Qcur and Kcur each by
/// `pow(n_state_head, -0.25)` — lines ~2506/2550/2557 of `src/whisper.cpp`;
/// the encoder uses the algebraically identical single `1/sqrtf(d)` factor
/// at line ~2069). The identity is
/// `(q·d^-0.25)·(k·d^-0.25) = q·k·d^-0.5 = q·k / sqrt(d)`.
///
/// # Errors
/// [`FwError::InvalidRequest`] if `n_head == 0`, `n_state % n_head != 0`,
/// the q/k/v widths disagree, or `k.rows != v.rows`.
pub fn attention(
    q: &Mat,
    k: &Mat,
    v: &Mat,
    n_head: usize,
    causal_offset: Option<usize>,
) -> FwResult<Mat> {
    if k.cols != q.cols || v.cols != q.cols {
        return Err(FwError::InvalidRequest(format!(
            "attention: width mismatch q={} k={} v={}",
            q.cols, k.cols, v.cols
        )));
    }
    if k.rows != v.rows {
        return Err(FwError::InvalidRequest(format!(
            "attention: k.rows {} != v.rows {}",
            k.rows, v.rows
        )));
    }
    attention_raw(q, &k.data, &v.data, k.rows, n_head, causal_offset)
}

/// Core multi-head attention over **raw row-major K/V slices**.
///
/// Identical math to [`attention`], but `k`/`v` are flat `[tk, n_state]`
/// row-major slices rather than [`Mat`]s, so a caller holding the K/V in a
/// larger backing buffer (e.g. a [`KvCache`]'s populated prefix) can attend
/// without first copying out a `[len, n_state]` `Mat`. Every per-head gather
/// reads the exact same bytes in the exact same order as [`attention`], so
/// results are bit-identical to the `Mat`-based path.
///
/// # Errors
/// [`FwError::InvalidRequest`] if `n_head == 0`, `n_state % n_head != 0`, or
/// the K/V slice lengths disagree with `tk * n_state`.
fn attention_raw(
    q: &Mat,
    k: &[f32],
    v: &[f32],
    tk: usize,
    n_head: usize,
    causal_offset: Option<usize>,
) -> FwResult<Mat> {
    let n_state = q.cols;
    if n_head == 0 || !n_state.is_multiple_of(n_head) {
        return Err(FwError::InvalidRequest(format!(
            "attention: n_head {n_head} must divide n_state {n_state}"
        )));
    }
    if k.len() != tk * n_state || v.len() != tk * n_state {
        return Err(FwError::InvalidRequest(format!(
            "attention: k/v slice len {}/{} != tk*n_state {}",
            k.len(),
            v.len(),
            tk * n_state
        )));
    }
    let tq = q.rows;
    let d_head = n_state / n_head;
    if d_head == 0 {
        return Err(FwError::InvalidRequest("attention: d_head == 0".into()));
    }
    let scale = (d_head as f32).powf(-0.25);
    let cache_len = causal_offset.unwrap_or(0);

    let mut out = vec![0.0f32; tq * n_state];

    // Compute one head's [Tq, d_head] output. Each head is independent and
    // its math (gather → scaled qk^T → mask → softmax → @v) is byte-for-byte
    // the serial computation; only the scheduling changes. The inner matmuls
    // go through the (rayon-parallel) sgemm — see the parallelism note below.
    let compute_head = |h: usize| -> FwResult<Mat> {
        let base = h * d_head;

        // Gather this head's scaled Q [Tq, d_head] and K [Tk, d_head].
        let mut qh = vec![0.0f32; tq * d_head];
        for i in 0..tq {
            let src = &q.row(i)[base..base + d_head];
            let dst = &mut qh[i * d_head..(i + 1) * d_head];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * scale;
            }
        }
        let mut kh = vec![0.0f32; tk * d_head];
        for j in 0..tk {
            let src = &k[j * n_state + base..j * n_state + base + d_head];
            let dst = &mut kh[j * d_head..(j + 1) * d_head];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * scale;
            }
        }
        let qh = Mat::from_vec(tq, d_head, qh);

        // scores = qh @ kh^T -> [Tq, Tk]. kh^T is [d_head, Tk]; build it
        // explicitly so the matmul stays contiguous.
        let mut kh_t = vec![0.0f32; d_head * tk];
        for j in 0..tk {
            for d in 0..d_head {
                kh_t[d * tk + j] = kh[j * d_head + d];
            }
        }
        let kh_t = Mat::from_vec(d_head, tk, kh_t);
        let mut scores = matmul(&qh, &kh_t)?; // [Tq, Tk]

        // Causal mask: query i attends to keys <= cache_len + i.
        if causal_offset.is_some() {
            for i in 0..tq {
                let limit = cache_len + i;
                let row = &mut scores.data[i * tk..(i + 1) * tk];
                for (j, s) in row.iter_mut().enumerate() {
                    if j > limit {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }
        }

        softmax_rows(&mut scores); // [Tq, Tk]

        // Gather this head's V [Tk, d_head] (unscaled), out_h = scores @ V.
        let mut vh = vec![0.0f32; tk * d_head];
        for j in 0..tk {
            let src = &v[j * n_state + base..j * n_state + base + d_head];
            let dst = &mut vh[j * d_head..(j + 1) * d_head];
            dst.copy_from_slice(src);
        }
        let vh = Mat::from_vec(tk, d_head, vh);
        matmul(&scores, &vh) // [Tq, d_head]
    };

    let scatter = |out: &mut [f32], h: usize, out_h: &Mat| {
        let base = h * d_head;
        for i in 0..tq {
            let src = &out_h.data[i * d_head..(i + 1) * d_head];
            out[i * n_state + base..i * n_state + base + d_head].copy_from_slice(src);
        }
    };

    // Parallelize over heads when the work is large enough to amortize the
    // spawn (encoder windows: Tk≈1500, n_head 6..20). We split heads across
    // workers and let each compute its head serially; the inner sgemm may
    // still rayon-split, but head-level threads are the bigger win for the
    // many small per-head matmuls and we accept the nested-pool interplay
    // (measured net positive — see HOTSPOTS run). Small/decode-step shapes
    // (Tq=1, tiny Tk) fall below the threshold and stay fully serial so they
    // never pay spawn overhead.
    //
    // The merged `out` is strided per head (each head owns a column band,
    // not a contiguous slice), so threads can't borrow disjoint `&mut out`
    // sub-slices directly; instead each worker scatters its own heads into a
    // private buffer and we sum the buffers (every position is written by
    // exactly one head, so the "sum" is just a disjoint merge — no overlap,
    // order-independent, bit-identical).
    const PAR_THRESHOLD: usize = 1 << 18; // tq*tk*n_head elements of real work
    let work = tq.saturating_mul(tk).saturating_mul(n_head);
    if n_head < 2 || work < PAR_THRESHOLD || worker_count() < 2 {
        for h in 0..n_head {
            let out_h = compute_head(h)?;
            scatter(&mut out, h, &out_h);
        }
        return Ok(Mat::from_vec(tq, n_state, out));
    }

    let workers = worker_count().min(n_head);
    let band = n_head.div_ceil(workers);
    let results: Vec<FwResult<Vec<f32>>> = std::thread::scope(|s| {
        let compute_head = &compute_head;
        let scatter = &scatter;
        let mut handles = Vec::with_capacity(workers);
        let mut h0 = 0;
        while h0 < n_head {
            let h1 = (h0 + band).min(n_head);
            handles.push(s.spawn(move || -> FwResult<Vec<f32>> {
                let mut local = vec![0.0f32; tq * n_state];
                for h in h0..h1 {
                    let out_h = compute_head(h)?;
                    scatter(&mut local, h, &out_h);
                }
                Ok(local)
            }));
            h0 = h1;
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in results {
        let local = r?;
        for (o, l) in out.iter_mut().zip(local.iter()) {
            *o += *l;
        }
    }

    Ok(Mat::from_vec(tq, n_state, out))
}

/// Incremental self-attention that extends a [`KvCache`].
///
/// Appends the step's `k_new`/`v_new` (`[Tq, n_state]`) to `cache`, then
/// runs causal [`attention`] of `q` against the *entire* cached K/V with
/// `causal_offset = past_len` (the cache length before this append). For a
/// single-token decode step `Tq == 1` and every cached key is visible; for a
/// multi-token prompt prefill each query `i` still only sees keys up to
/// `past_len + i`.
///
/// # Errors
/// Propagates [`KvCache::append`] and [`attention`] errors.
pub fn attention_with_cache(
    q: &Mat,
    k_new: &Mat,
    v_new: &Mat,
    n_head: usize,
    cache: &mut KvCache,
) -> FwResult<Mat> {
    let past_len = cache.len();
    cache.append(k_new, v_new)?;
    // Attend directly over the cache's populated prefix — no `[len, n_state]`
    // copy-out per step (the old `keys()`/`values()` each `.to_vec()`'d the
    // whole prefix, the dominant per-step memmove on wide models). The raw
    // path reads the identical bytes in the identical order, so the result is
    // bit-identical to the `Mat`-based attention.
    let tk = cache.len();
    attention_raw(
        q,
        cache.key_slice(),
        cache.value_slice(),
        tk,
        n_head,
        Some(past_len),
    )
}

/// Cache-blocked, multi-threaded out-of-place transpose: `data` viewed as
/// row-major `[rows, cols]` becomes row-major `[cols, rows]`.
///
/// Used at model-load time to pre-transpose every linear weight (ggml stores
/// PyTorch's `[out, in]`; the inference matmuls want `[in, out]`). The naive
/// column-strided serial loop dominated `model_weights` time on large models
/// (hotspot #5, tests/artifacts/perf/20260605T0218Z): ~3 GB of strided writes.
/// 64x64 tiles keep both source reads and destination writes inside cache
/// lines; independent row-bands fan out across threads.
///
/// Isomorphism: a pure permutation — every output element is the same
/// `data[r * cols + c]` the serial loop wrote, so results are bit-identical.
pub(crate) fn transpose_parallel(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(data.len(), rows * cols, "transpose shape/data mismatch");
    const TILE: usize = 64;
    const PAR_THRESHOLD: usize = 1 << 20;
    let mut out = vec![0.0f32; rows * cols];

    let tile_band = |band_rows: std::ops::Range<usize>, out: &mut [f32]| {
        // `out` here is the FULL output buffer for serial mode, or a row-band
        // is not separable for transpose outputs (writes scatter across all
        // of `out`), so parallel mode splits by output row bands (i.e. source
        // column bands) instead — see below.
        for r0 in band_rows.clone().step_by(TILE) {
            let r1 = (r0 + TILE).min(band_rows.end);
            for c0 in (0..cols).step_by(TILE) {
                let c1 = (c0 + TILE).min(cols);
                for r in r0..r1 {
                    for c in c0..c1 {
                        out[c * rows + r] = data[r * cols + c];
                    }
                }
            }
        }
    };

    let workers = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
        .min(8);
    if rows * cols < PAR_THRESHOLD || workers < 2 {
        tile_band(0..rows, &mut out);
        return out;
    }

    // Parallel split: each worker owns a contiguous band of OUTPUT rows
    // (= source columns c in [c0, c1)), so output slices are disjoint.
    let band = cols.div_ceil(workers);
    std::thread::scope(|s| {
        for (w, out_band) in out.chunks_mut(band * rows).enumerate() {
            let c_start = w * band;
            s.spawn(move || {
                let c_end = (c_start + band).min(cols);
                for c0 in (c_start..c_end).step_by(TILE) {
                    let c1 = (c0 + TILE).min(c_end);
                    for r0 in (0..rows).step_by(TILE) {
                        let r1 = (r0 + TILE).min(rows);
                        for c in c0..c1 {
                            let dst_row = c - c_start;
                            for r in r0..r1 {
                                out_band[dst_row * rows + r] = data[r * cols + c];
                            }
                        }
                    }
                }
            });
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
        fn mat(&mut self, rows: usize, cols: usize) -> Mat {
            let data = (0..rows * cols).map(|_| self.next_f32()).collect();
            Mat::from_vec(rows, cols, data)
        }
    }

    fn naive_matmul(a: &Mat, b: &Mat) -> Mat {
        let (m, k, n) = (a.rows, a.cols, b.cols);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                // Accumulate the reference dot product in f64: this is the
                // *true* value the f32 sgemm approximates. Summing the
                // reference in f32 would itself drift (a different, worse,
                // accumulation order than matrixmultiply's blocked kernel),
                // so a near-zero element where ±1 terms cancel would show a
                // large spurious relative error against an equally-noisy
                // reference. f64 here makes the reference the gold standard.
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += f64::from(a.data[i * k + p]) * f64::from(b.data[p * n + j]);
                }
                out[i * n + j] = acc as f32;
            }
        }
        Mat::from_vec(m, n, out)
    }

    /// Combined absolute + relative error: an element passes if it is within
    /// `1e-5` relative OR within an absolute floor scaled to the magnitude of
    /// the computation. For dot products of `k` values in `[-1, 1)` the f32
    /// rounding noise grows like `~sqrt(k) * eps`, so we scale the absolute
    /// floor by `sqrt(k)`; this judges near-zero (cancelling) elements by
    /// absolute error rather than a meaningless relative one.
    fn max_rel_err_k(a: &[f32], b: &[f32], k: usize) -> f32 {
        let abs_floor = 1e-5 * (k.max(1) as f32).sqrt();
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let abs = (x - y).abs();
                let denom = x.abs().max(y.abs()).max(1e-6);
                // Pass if within abs floor; otherwise report the relative error.
                if abs <= abs_floor { 0.0 } else { abs / denom }
            })
            .fold(0.0f32, f32::max)
    }

    fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
        max_rel_err_k(a, b, 1)
    }

    #[test]
    fn matmul_matches_naive_various_shapes() {
        let mut rng = Lcg::new(1);
        // Includes the decoder-step [1,k]x[k,n] shape and an encoder-sized one.
        let shapes = [(1, 384, 384), (1500, 384, 384), (7, 13, 5), (32, 64, 48)];
        for (m, k, n) in shapes {
            let a = rng.mat(m, k);
            let b = rng.mat(k, n);
            let got = matmul(&a, &b).unwrap();
            let want = naive_matmul(&a, &b);
            assert_eq!(got.rows, m);
            assert_eq!(got.cols, n);
            assert!(
                max_rel_err_k(&got.data, &want.data, k) < 1e-5,
                "shape {m}x{k}x{n} rel err too high"
            );
        }
    }

    /// Build a natural `[out, in]` f16 weight (typed [`Float16`]) plus the exact
    /// f32 matrix it dequantizes to, from the LCG.
    fn rand_f16_weight(rng: &mut Lcg, out: usize, inp: usize) -> (Vec<Float16>, Vec<f32>) {
        let mut halves = Vec::with_capacity(out * inp);
        let mut f32s = Vec::with_capacity(out * inp);
        for _ in 0..out * inp {
            let h = ft_core::Float16::from_f32(rng.next_f32());
            halves.push(h);
            f32s.push(h.to_f32()); // the EXACT value the f16 stores
        }
        (halves, f32s)
    }

    /// EXHAUSTIVE bit-exactness gate: the SIMD bulk `convert_to_f32_slice`
    /// dequant the GEMV kernels use must produce, for ALL 65536 possible u16
    /// half bit patterns, EXACTLY the same f32 (bit-for-bit) as the scalar
    /// `half`-crate `from_bits().to_f32()` widen the f32 loader uses. This is
    /// the load-bearing correctness proof for the f16-resident path: dequant is
    /// a lossless widening, so the conversion must be exact everywhere
    /// (normals, subnormals, +/-0, +/-inf, every NaN payload), not merely close.
    #[test]
    fn f16_dequant_bulk_is_bit_exact_for_all_65536() {
        let halves: Vec<Float16> = (0..=u16::MAX).map(Float16::from_bits).collect();
        let mut bulk = vec![0.0f32; halves.len()];
        halves.convert_to_f32_slice(&mut bulk);
        for (i, (&h, &b)) in halves.iter().zip(&bulk).enumerate() {
            let scalar = h.to_f32();
            assert_eq!(
                b.to_bits(),
                scalar.to_bits(),
                "bulk dequant of bits {i:#06x} = {b:?} (bits {:#010x}) != scalar {scalar:?} (bits {:#010x})",
                b.to_bits(),
                scalar.to_bits()
            );
        }
    }

    #[test]
    fn gemv_f16_matches_dequant_then_matmul() {
        let mut rng = Lcg::new(11);
        // Covers the decoder Linear shapes ([out,in]) and the logits-sized one.
        for (out, inp) in [
            (1usize, 4usize),
            (384, 384),
            (5, 64),
            (2048, 1280),
            (51866, 16),
        ] {
            let (w_h, w_f32) = rand_f16_weight(&mut rng, out, inp);
            let x: Vec<f32> = (0..inp).map(|_| rng.next_f32()).collect();
            let bias: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect();

            // Reference: dequant the weight to f32, then run it through the SAME
            // ft sgemm the f32 path uses. The f32-path Linear pre-transposes
            // [out,in] -> [in,out] and computes x[1,in] @ w_t[in,out]; reproduce
            // that exactly so we compare like-for-like accumulation environments.
            let w_t = {
                let mut t = vec![0.0f32; inp * out];
                for o in 0..out {
                    for i in 0..inp {
                        t[i * out + o] = w_f32[o * inp + i];
                    }
                }
                Mat::from_vec(inp, out, t)
            };
            let x_mat = Mat::from_vec(1, inp, x.clone());
            let want = matmul_bias(&x_mat, &w_t, Some(&bias)).unwrap();

            let mut got = vec![0.0f32; out];
            gemv_f16(&w_h, out, inp, &x, Some(&bias), &mut got);

            // Both accumulate in f32 over the same exact weight values; only the
            // summation order differs (row-dot vs sgemm block), so the diff is
            // tiny rounding noise. Spec gate: max abs diff < 1e-4.
            let max = got
                .iter()
                .zip(&want.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max < 1e-4,
                "gemv_f16 vs dequant-matmul diff {max} at [{out},{inp}]"
            );
        }
    }

    #[test]
    fn gemv_f16_dequant_is_exact() {
        // Every f16 value must dequantize to EXACTLY the same f32 used inside
        // the kernel: a single-element identity weight reads back the stored
        // value bit-for-bit (x = 1, no bias).
        let vals = [1.0f32, 0.5, -2.0, 0.0, 65504.0, 6.103_515_6e-5];
        for &v in &vals {
            let h = ft_core::Float16::from_f32(v);
            let halves = vec![h];
            let mut got = [0.0f32];
            gemv_f16(&halves, 1, 1, &[1.0], None, &mut got);
            assert_eq!(
                got[0].to_bits(),
                h.to_f32().to_bits(),
                "dequant of f16({v}) must be exact"
            );
        }
    }

    #[test]
    fn gemv_f16_no_bias_and_threshold_paths() {
        let mut rng = Lcg::new(13);
        // A shape above the parallel threshold exercises the threaded bands.
        let (out, inp) = (4096usize, 256usize);
        let (w_h, w_f32) = rand_f16_weight(&mut rng, out, inp);
        let x: Vec<f32> = (0..inp).map(|_| rng.next_f32()).collect();

        let mut got = vec![0.0f32; out];
        gemv_f16(&w_h, out, inp, &x, None, &mut got);

        // Reference: plain row-dot in f32 over the exact dequantized weight.
        for o in 0..out {
            let mut acc = 0.0f32;
            for i in 0..inp {
                acc += w_f32[o * inp + i] * x[i];
            }
            assert!((got[o] - acc).abs() < 1e-3, "row {o} mismatch");
        }
    }

    #[test]
    fn gemv_f16_batch_equals_per_token_gemv() {
        let mut rng = Lcg::new(17);
        let (out, inp, tq) = (300usize, 128usize, 5usize);
        let (w_h, _w_f32) = rand_f16_weight(&mut rng, out, inp);
        let x: Vec<f32> = (0..tq * inp).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect();

        let mut batch = vec![0.0f32; tq * out];
        gemv_f16_batch(&w_h, out, inp, &x, tq, Some(&bias), &mut batch);

        // Per-token gemv must be byte-identical to the batch (same math).
        for t in 0..tq {
            let mut row = vec![0.0f32; out];
            gemv_f16(
                &w_h,
                out,
                inp,
                &x[t * inp..(t + 1) * inp],
                Some(&bias),
                &mut row,
            );
            for (o, &r) in row.iter().enumerate() {
                assert_eq!(
                    batch[t * out + o].to_bits(),
                    r.to_bits(),
                    "batch[{t},{o}] != per-token gemv"
                );
            }
        }
    }

    #[test]
    fn matmul_inner_dim_mismatch_errors() {
        let a = Mat::zeros(2, 3);
        let b = Mat::zeros(4, 5);
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn matmul_bias_matches_manual() {
        let mut rng = Lcg::new(2);
        let x = rng.mat(4, 6); // [m=4, in=6]
        let w_t = rng.mat(6, 3); // [in=6, out=3]
        let bias = [0.5f32, -0.25, 1.0];
        let got = matmul_bias(&x, &w_t, Some(&bias)).unwrap();
        let base = naive_matmul(&x, &w_t);
        for i in 0..4 {
            for (j, b) in bias.iter().enumerate() {
                let expected = base.data[i * 3 + j] + b;
                assert!((got.data[i * 3 + j] - expected).abs() < 1e-5);
            }
        }
        // No-bias path == plain matmul.
        let no_bias = matmul_bias(&x, &w_t, None).unwrap();
        assert!(max_rel_err(&no_bias.data, &base.data) < 1e-5);
    }

    #[test]
    fn matmul_bias_wrong_bias_len_errors() {
        let x = Mat::zeros(2, 3);
        let w_t = Mat::zeros(3, 4);
        assert!(matmul_bias(&x, &w_t, Some(&[1.0, 2.0])).is_err());
    }

    #[test]
    fn layer_norm_simd_matches_scalar() {
        // norm_rows vectorizes 8 rows at a time; verify byte-identical to an
        // independent scalar per-row f64 reference across SIMD groups + the
        // < 8-row tail, for several row counts.
        let cols = 384usize;
        let eps_f32 = 1e-5f32;
        for rows in [1usize, 7, 8, 9, 20, 33] {
            let mut lcg = Lcg::new(0x000A_17E5 ^ rows as u64);
            let w: Vec<f32> = (0..cols).map(|_| lcg.next_f32() * 0.5 + 1.0).collect();
            let b: Vec<f32> = (0..cols).map(|_| lcg.next_f32() * 0.1).collect();
            let data: Vec<f32> = (0..rows * cols).map(|_| lcg.next_f32()).collect();

            let mut m = Mat::from_vec(rows, cols, data.clone());
            layer_norm(&mut m, &w, &b, eps_f32);

            // Independent scalar per-row f64 reference.
            let mut want = data;
            let eps = f64::from(eps_f32);
            for row in want.chunks_mut(cols) {
                let mut sum = 0.0f64;
                for &v in row.iter() {
                    sum += f64::from(v);
                }
                let mean = sum / cols as f64;
                let mut var = 0.0f64;
                for &v in row.iter() {
                    let d = f64::from(v) - mean;
                    var += d * d;
                }
                var /= cols as f64;
                let inv = 1.0 / (var + eps).sqrt();
                for ((v, &wi), &bi) in row.iter_mut().zip(w.iter()).zip(b.iter()) {
                    let normed = (f64::from(*v) - mean) * inv;
                    *v = (normed * f64::from(wi) + f64::from(bi)) as f32;
                }
            }
            for (i, (got, exp)) in m.data.iter().zip(want.iter()).enumerate() {
                assert_eq!(got.to_bits(), exp.to_bits(), "rows={rows} idx={i}");
            }
        }
    }

    #[test]
    fn layer_norm_known_small_case() {
        // Row [1,2,3,4]: mean=2.5, var=1.25, std=sqrt(1.25).
        let mut x = Mat::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let w = [1.0f32; 4];
        let b = [0.0f32; 4];
        layer_norm(&mut x, &w, &b, 0.0);
        let std = 1.25f32.sqrt();
        let expected = [
            (1.0 - 2.5) / std,
            (2.0 - 2.5) / std,
            (3.0 - 2.5) / std,
            (4.0 - 2.5) / std,
        ];
        for (g, e) in x.data.iter().zip(expected) {
            assert!((g - e).abs() < 1e-5, "got {g}, want {e}");
        }
    }

    #[test]
    fn layer_norm_property_zero_mean_unit_var() {
        let mut rng = Lcg::new(3);
        let mut x = rng.mat(5, 64);
        let w = vec![1.0f32; 64];
        let b = vec![0.0f32; 64];
        layer_norm(&mut x, &w, &b, 1e-5);
        for r in 0..5 {
            let row = x.row(r);
            let mean: f64 = row.iter().map(|&v| f64::from(v)).sum::<f64>() / 64.0;
            let var: f64 = row
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / 64.0;
            assert!(mean.abs() < 1e-4, "row {r} mean {mean}");
            assert!((var - 1.0).abs() < 1e-2, "row {r} var {var}");
        }
    }

    #[test]
    fn layer_norm_affine_applied() {
        let mut x = Mat::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let w = [2.0f32, 2.0, 2.0, 2.0];
        let b = [1.0f32, 1.0, 1.0, 1.0];
        layer_norm(&mut x, &w, &b, 0.0);
        let std = 1.25f32.sqrt();
        let expected = (1.0 - 2.5) / std * 2.0 + 1.0;
        assert!((x.data[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn gelu_known_values() {
        let mut x = Mat::from_vec(1, 3, vec![0.0, 1.0, -1.0]);
        gelu(&mut x);
        // Compute expected from the tanh-approx formula directly.
        let f = |v: f32| {
            0.5 * v * (1.0 + (GELU_SQRT_2_OVER_PI * v * (1.0 + GELU_COEF_A * v * v)).tanh())
        };
        assert!((x.data[0] - 0.0).abs() < 1e-6, "gelu(0)=0");
        assert!((x.data[1] - f(1.0)).abs() < 1e-6);
        assert!((x.data[2] - f(-1.0)).abs() < 1e-6);
        // Spec reference magnitudes.
        assert!(
            (x.data[1] - 0.8412).abs() < 1e-3,
            "gelu(1)~0.8412, got {}",
            x.data[1]
        );
        assert!(
            (x.data[2] - (-0.1588)).abs() < 1e-3,
            "gelu(-1)~-0.1588, got {}",
            x.data[2]
        );
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let mut rng = Lcg::new(4);
        let mut x = rng.mat(6, 11);
        softmax_rows(&mut x);
        for r in 0..6 {
            let s: f32 = x.row(r).iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row {r} sum {s}");
            assert!(x.row(r).iter().all(|&v| v >= 0.0));
        }
    }

    #[test]
    fn softmax_rows_max_stability() {
        // Large values must not overflow to NaN/inf.
        let mut x = Mat::from_vec(1, 3, vec![1e30, 1e30 + 1.0, 0.0]);
        softmax_rows(&mut x);
        let s: f32 = x.row(0).iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum {s}");
        assert!(x.data.iter().all(|v| v.is_finite()));
    }

    #[allow(clippy::too_many_arguments)]
    fn naive_conv1d(
        x: &Mat,
        w: &[f32],
        cout: usize,
        cin: usize,
        k: usize,
        bias: &[f32],
        stride: usize,
        pad: usize,
    ) -> Mat {
        let t_in = x.rows;
        let t_out = (t_in + 2 * pad - k) / stride + 1;
        let mut out = vec![0.0f32; t_out * cout];
        for o in 0..t_out {
            for co in 0..cout {
                // f64 reference accumulation; see `naive_matmul`.
                let mut acc = f64::from(bias[co]);
                for kk in 0..k {
                    let p = o * stride + kk;
                    if p < pad {
                        continue;
                    }
                    let ti = p - pad;
                    if ti >= t_in {
                        continue;
                    }
                    for ci in 0..cin {
                        acc += f64::from(w[co * cin * k + ci * k + kk])
                            * f64::from(x.data[ti * cin + ci]);
                    }
                }
                out[o * cout + co] = acc as f32;
            }
        }
        Mat::from_vec(t_out, cout, out)
    }

    #[test]
    fn conv1d_matches_naive() {
        let mut rng = Lcg::new(5);
        // (t_in, cin, cout, k, stride, pad)
        let cases = [
            (10, 3, 4, 3, 1, 1),
            (10, 3, 4, 3, 2, 1),
            (16, 5, 2, 3, 1, 1),
            (8, 2, 6, 5, 2, 1),
        ];
        for (t_in, cin, cout, k, stride, pad) in cases {
            let x = rng.mat(t_in, cin);
            let w: Vec<f32> = (0..cout * cin * k).map(|_| rng.next_f32()).collect();
            let bias: Vec<f32> = (0..cout).map(|_| rng.next_f32()).collect();
            let got = conv1d(&x, &w, cout, cin, k, &bias, stride, pad).unwrap();
            let want = naive_conv1d(&x, &w, cout, cin, k, &bias, stride, pad);
            assert_eq!(got.rows, want.rows, "t_out mismatch");
            assert_eq!(got.cols, cout);
            assert!(
                max_rel_err_k(&got.data, &want.data, cin * k) < 1e-5,
                "conv case {t_in},{cin},{cout},{k},{stride},{pad}"
            );
        }
    }

    /// Reference single-head attention (no cache, optional causal).
    fn naive_attention_single_head(q: &Mat, k: &Mat, v: &Mat, causal: bool) -> Mat {
        let d = q.cols;
        let tq = q.rows;
        let tk = k.rows;
        let scale = (d as f32).powf(-0.25);
        let mut out = vec![0.0f32; tq * d];
        for i in 0..tq {
            // scores
            let mut scores = vec![0.0f32; tk];
            for (j, score) in scores.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for c in 0..d {
                    acc += (q.data[i * d + c] * scale) * (k.data[j * d + c] * scale);
                }
                *score = if causal && j > i {
                    f32::NEG_INFINITY
                } else {
                    acc
                };
            }
            // softmax
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum;
            }
            // weighted sum of v
            for c in 0..d {
                let mut acc = 0.0f32;
                for (j, &sj) in scores.iter().enumerate() {
                    acc += sj * v.data[j * d + c];
                }
                out[i * d + c] = acc;
            }
        }
        Mat::from_vec(tq, d, out)
    }

    #[test]
    fn attention_matches_single_head_reference() {
        let mut rng = Lcg::new(6);
        let q = rng.mat(5, 8);
        let k = rng.mat(7, 8);
        let v = rng.mat(7, 8);
        let got = attention(&q, &k, &v, 1, None).unwrap();
        let want = naive_attention_single_head(&q, &k, &v, false);
        assert!(max_rel_err(&got.data, &want.data) < 1e-4);
    }

    #[test]
    fn attention_multi_head_merge() {
        // With n_head heads each head is an independent single-head attention
        // over its slice; concatenating reference per-head results must match.
        let mut rng = Lcg::new(7);
        let n_head = 4;
        let d_head = 6;
        let n_state = n_head * d_head;
        let tq = 3;
        let tk = 5;
        let q = rng.mat(tq, n_state);
        let k = rng.mat(tk, n_state);
        let v = rng.mat(tk, n_state);
        let got = attention(&q, &k, &v, n_head, None).unwrap();

        let mut want = vec![0.0f32; tq * n_state];
        for h in 0..n_head {
            let base = h * d_head;
            let slice = |m: &Mat, rows: usize| {
                let mut out = vec![0.0f32; rows * d_head];
                for r in 0..rows {
                    out[r * d_head..(r + 1) * d_head]
                        .copy_from_slice(&m.row(r)[base..base + d_head]);
                }
                Mat::from_vec(rows, d_head, out)
            };
            let qh = slice(&q, tq);
            let kh = slice(&k, tk);
            let vh = slice(&v, tk);
            let oh = naive_attention_single_head(&qh, &kh, &vh, false);
            for r in 0..tq {
                want[r * n_state + base..r * n_state + base + d_head].copy_from_slice(oh.row(r));
            }
        }
        assert!(max_rel_err(&got.data, &want).is_finite());
        assert!(max_rel_err(&got.data, &want) < 1e-4);
    }

    #[test]
    fn attention_causal_mask_property() {
        // Changing future keys/values must not change earlier query outputs.
        let mut rng = Lcg::new(8);
        let n_head = 2;
        let n_state = 8;
        let tq = 4;
        let q = rng.mat(tq, n_state);
        let mut k = rng.mat(tq, n_state);
        let mut v = rng.mat(tq, n_state);
        let out_a = attention(&q, &k, &v, n_head, Some(0)).unwrap();

        // Perturb the LAST key/value row (a "future" token for queries 0..2).
        for c in 0..n_state {
            k.data[(tq - 1) * n_state + c] += 5.0;
            v.data[(tq - 1) * n_state + c] += 5.0;
        }
        let out_b = attention(&q, &k, &v, n_head, Some(0)).unwrap();

        // Rows 0..tq-1 (which cannot attend to the last key) are unchanged.
        for i in 0..tq - 1 {
            for c in 0..n_state {
                assert!(
                    (out_a.data[i * n_state + c] - out_b.data[i * n_state + c]).abs() < 1e-6,
                    "row {i} changed under future-key perturbation"
                );
            }
        }
        // The last row DOES change (it attends to the perturbed key).
        let last_changed = (0..n_state).any(|c| {
            (out_a.data[(tq - 1) * n_state + c] - out_b.data[(tq - 1) * n_state + c]).abs() > 1e-4
        });
        assert!(
            last_changed,
            "last query row should react to its own key change"
        );
    }

    #[test]
    fn kv_cache_incremental_equals_full_recompute() {
        let mut rng = Lcg::new(9);
        let n_head = 3;
        let d_head = 5;
        let n_state = n_head * d_head;
        let n_tokens = 5;

        // Full per-token q/k/v (each token contributes one row).
        let q_all = rng.mat(n_tokens, n_state);
        let k_all = rng.mat(n_tokens, n_state);
        let v_all = rng.mat(n_tokens, n_state);

        // Full recompute: causal attention over all tokens at once.
        let full = attention(&q_all, &k_all, &v_all, n_head, Some(0)).unwrap();

        // Incremental: feed one token at a time through a KvCache.
        let mut cache = KvCache::new(n_tokens, n_state);
        let mut inc = vec![0.0f32; n_tokens * n_state];
        for t in 0..n_tokens {
            let qi = Mat::from_vec(1, n_state, q_all.row(t).to_vec());
            let ki = Mat::from_vec(1, n_state, k_all.row(t).to_vec());
            let vi = Mat::from_vec(1, n_state, v_all.row(t).to_vec());
            let step = attention_with_cache(&qi, &ki, &vi, n_head, &mut cache).unwrap();
            inc[t * n_state..(t + 1) * n_state].copy_from_slice(&step.data);
        }
        assert_eq!(cache.len(), n_tokens);
        assert!(
            max_rel_err(&inc, &full.data) < 1e-4,
            "incremental != full recompute"
        );
    }

    #[test]
    fn kv_cache_append_overflow_and_reset() {
        let mut cache = KvCache::new(2, 4);
        let row = Mat::from_vec(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        cache.append(&row, &row).unwrap();
        cache.append(&row, &row).unwrap();
        assert_eq!(cache.len(), 2);
        assert!(cache.append(&row, &row).is_err(), "third append overflows");
        cache.reset();
        assert!(cache.is_empty());
        cache.append(&row, &row).unwrap();
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn matmul_raw_lhs_bit_identical_to_matmul() {
        // The tied-logits band path multiplies an embedding row band in place
        // via `matmul_raw_lhs`; it must be byte-for-byte the copy-then-matmul.
        let mut rng = Lcg::new(101);
        for (m, k, n) in [(1usize, 384usize, 1usize), (6483, 1280, 1), (5, 64, 3)] {
            let a = rng.mat(m, k);
            let b = rng.mat(k, n);
            let raw = matmul_raw_lhs(&a.data, m, &b).unwrap();
            let copied = matmul(&a, &b).unwrap();
            assert_eq!(raw.rows, m);
            assert_eq!(raw.cols, n);
            assert_eq!(
                raw.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                copied.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "matmul_raw_lhs differs bitwise at {m}x{k}x{n}"
            );
        }
    }

    #[test]
    fn matmul_raw_lhs_len_mismatch_errors() {
        let b = Mat::zeros(3, 2);
        assert!(matmul_raw_lhs(&[1.0, 2.0], 1, &b).is_err());
    }

    #[test]
    fn attention_raw_bit_identical_to_mat_path() {
        // `attention_with_cache` attends over the KvCache's raw prefix slice via
        // `attention_raw`; it must be byte-for-byte the `Mat`-based `attention`.
        let mut rng = Lcg::new(202);
        let n_head = 4;
        let n_state = 32;
        let q = rng.mat(1, n_state);
        let k = rng.mat(7, n_state);
        let v = rng.mat(7, n_state);
        for off in [None, Some(0usize), Some(3usize)] {
            let viamat = attention(&q, &k, &v, n_head, off).unwrap();
            let viaraw = attention_raw(&q, &k.data, &v.data, k.rows, n_head, off).unwrap();
            assert_eq!(
                viamat.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                viaraw.data.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "attention_raw differs bitwise (offset {off:?})"
            );
        }
    }

    #[test]
    fn key_value_slice_match_keys_values() {
        let mut cache = KvCache::new(4, 6);
        let mut rng = Lcg::new(303);
        let r = rng.mat(2, 6);
        cache.append(&r, &r).unwrap();
        assert_eq!(cache.key_slice(), cache.keys().data.as_slice());
        assert_eq!(cache.value_slice(), cache.values().data.as_slice());
    }

    #[test]
    fn attention_rejects_bad_head_count() {
        let q = Mat::zeros(2, 6);
        let k = Mat::zeros(2, 6);
        let v = Mat::zeros(2, 6);
        assert!(attention(&q, &k, &v, 0, None).is_err());
        assert!(
            attention(&q, &k, &v, 4, None).is_err(),
            "4 does not divide 6"
        );
    }
}
