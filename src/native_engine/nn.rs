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
use rayon::prelude::*;

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
/// Host parallelism, queried ONCE and cached for the process.
///
/// `std::thread::available_parallelism()` is a `sched_getaffinity` syscall on
/// Linux and is **not** cached by std — every GEMV-dispatch call in the decode
/// hot path (~70 m=1 GEMVs/token via [`gemv_worker_count`] / [`gemv_f16_batch`])
/// otherwise re-pays it. The value is a process constant (the kernels are tuned
/// around it; the `FW_*_GEMV_CAP` overrides are already `OnceLock`-cached), so
/// caching is bit-identical — the derived worker count, and thus the GEMV band
/// split, is unchanged.
fn avail_parallelism() -> usize {
    use std::sync::OnceLock;
    static A: OnceLock<usize> = OnceLock::new();
    *A.get_or_init(|| {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
    })
}

#[inline]
pub(crate) fn worker_count() -> usize {
    avail_parallelism().min(8)
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
    let avail = avail_parallelism();
    // Only the vocab-class GEMV (tens of thousands of rows) is bandwidth-bound
    // enough to want >8 threads; below that the 8-cap wins (see fn docs).
    const WIDE_OUT_THRESHOLD: usize = 1 << 14; // 16384 rows
    let cap = if out >= WIDE_OUT_THRESHOLD { wide_gemv_cap() } else { 8 };
    avail.min(cap)
}

/// Worker cap for the vocab-class (bandwidth-bound) GEMV. **32** (measured optimum):
/// the old 12 left the logits GEMV at ~16 GB/s, well under the controller ceiling —
/// raising to 32 saturates ~4 CCDs' memory channels for ~1.4–1.8x on logits_gemv_large
/// and ~6–8% e2e (and far more load-robust). 48/64 regress (cross-CCD sync), and a
/// 24-thread CCD-split is a local minimum. Overridable via `FW_WIDE_GEMV_CAP`.
fn wide_gemv_cap() -> usize {
    use std::sync::OnceLock;
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("FW_WIDE_GEMV_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&c: &usize| c >= 1)
            .unwrap_or(32)
    })
}

/// Cached `FW_BATCH_GEMV_CAP` override (env read ONCE, not per batched-GEMV call).
/// `None` ⇒ no override; same value the per-call `env::var` returned, so the
/// derived worker count is unchanged (bit-identical band split).
fn batch_gemv_cap() -> Option<usize> {
    use std::sync::OnceLock;
    static CAP: OnceLock<Option<usize>> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("FW_BATCH_GEMV_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&c| c >= 1)
    })
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

    // m=1 fast path: the per-token decode attention matmuls (cross/self attn at
    // tq=1: `[1,d]x[d,tk]` scores and `[1,tk]x[tk,d]` out) are GEMV-shaped, but
    // ft sgemm packs/dispatches its full microkernel for them — MEASURED ~8–10×
    // slower than a direct GEMV for these shapes (`[1,64]x[64,1500]`: sgemm 46 µs
    // vs gemv 4.5 µs; `[1,1500]x[1500,64]`: 48 vs 6.3 µs; x86-64-v3). This is the
    // franken-vs-whisper.cpp decoder gap (bd-6qih): GGML uses a dedicated dot, we
    // routed everything through sgemm. A row-broadcast SAXPY accumulation over k
    // (LLVM lowers the inner `out += a[k]*b[k,:]` to AVX2 FMA) avoids all the
    // packing. NOT bit-identical to sgemm (different summation order; measured
    // max abs diff ~1e-6/2.7e-5), so it relies on the transcription-level
    // conformance contract — verified green (native_engine_e2e 6/6).
    if m == 1 {
        let mut out = vec![0.0f32; n];
        for kk in 0..k {
            let av = a.data[kk];
            let brow = &b.data[kk * n..(kk + 1) * n];
            for (o, &bv) in out.iter_mut().zip(brow) {
                *o += av * bv;
            }
        }
        return Ok(Mat::from_vec(1, n, out));
    }

    let lhs_meta = meta_2d(m, k);
    let rhs_meta = meta_2d(k, n);
    let data = matmul_into_uninit(&a.data, &b.data, &lhs_meta, &rhs_meta, m * n)?;
    Ok(Mat::from_vec(m, n, data))
}

/// Run the ft sgemm into a freshly-allocated **uninitialized** `[numel]` buffer.
///
/// The allocating `ft_kernel_cpu::matmul_tensor_contiguous_f32` does
/// `Vec::new()` then `resize(numel, 0.0)` — zero-initializing the entire output
/// — before the GEMM (which runs with `beta = 0`) overwrites every element. That
/// zero-init is pure dead work: MEASURED **~0.33 ms / 12.8%** of the call on the
/// `[1500,384]x[384,1536]` encoder MLP shape (bit-identical output; the encoder's
/// ~36 matmuls/window are a chunk of the profiled `__memset_avx2`). We instead
/// size the buffer to `numel` and call the buffer-reusing `_into` variant, whose
/// `resize` is then a no-op (no zero fill); the GEMM fills all `numel` outputs.
/// The escape hatch `FW_MATMUL_ZEROINIT` restores the old zero-init path.
fn matmul_into_uninit(
    lhs: &[f32],
    rhs: &[f32],
    lhs_meta: &TensorMeta,
    rhs_meta: &TensorMeta,
    numel: usize,
) -> FwResult<Vec<f32>> {
    use std::sync::OnceLock;
    static FORCE_ZEROINIT: OnceLock<bool> = OnceLock::new();
    let force_zeroinit =
        *FORCE_ZEROINIT.get_or_init(|| std::env::var_os("FW_MATMUL_ZEROINIT").is_some());
    if force_zeroinit {
        return ft_kernel_cpu::matmul_tensor_contiguous_f32(lhs, rhs, lhs_meta, rhs_meta)
            .map_err(kernel_err);
    }
    let mut data: Vec<f32> = Vec::with_capacity(numel);
    // SAFETY: `numel` elements of capacity are reserved just above. The beta=0
    // sgemm below overwrites all `numel` outputs before `data` is read, so no
    // uninitialized value is ever observed (f32 has no Drop and no invalid bit
    // patterns; on a kernel error the Vec is dropped without reading elements).
    // `clippy::uninit_vec` flags the with_capacity+set_len shape generically; it
    // is sound here precisely because the GEMM fully initializes the buffer.
    #[allow(unsafe_code, clippy::uninit_vec)]
    unsafe {
        data.set_len(numel);
    }
    ft_kernel_cpu::matmul_tensor_contiguous_f32_into(&mut data, lhs, rhs, lhs_meta, rhs_meta)
        .map_err(kernel_err)?;
    Ok(data)
}

/// Allocate an `[n]` f32 buffer that the caller **fully overwrites** before any
/// read, skipping the dead serial zero-init — the same dead-work elision as
/// [`matmul_into_uninit`] (d44f1fa). Used for the decode's per-token GEMV/logits
/// outputs ([`gemv_f16`]/[`gemv_f16_batch`] assign every slot) and the encoder
/// SDPA gather buffers (qa/ka/va, each `copy_from_slice`-filled in full). NOT for
/// accumulator buffers (the parallel per-head `out`, `+=`-merged — keep those
/// zeroed). Gated by `FW_DECODE_ZEROINIT` (set => zero-init: an A/B and safety
/// fallback covering all uninit-output sites).
pub fn gemv_out_buf(n: usize) -> Vec<f32> {
    use std::sync::OnceLock;
    static FORCE_ZEROINIT: OnceLock<bool> = OnceLock::new();
    if *FORCE_ZEROINIT.get_or_init(|| std::env::var_os("FW_DECODE_ZEROINIT").is_some()) {
        return vec![0.0f32; n];
    }
    let mut v: Vec<f32> = Vec::with_capacity(n);
    // SAFETY: `n` elements of capacity are reserved just above; the caller's GEMV
    // writes every one of the `n` outputs before any read (gemv_f16 assigns `*slot`
    // for all rows; gemv_f16_batch assigns `dst[t*out+o]` for all t,o). f32 has no
    // Drop and no invalid bit patterns. Mirrors `matmul_into_uninit`'s contract.
    #[allow(unsafe_code, clippy::uninit_vec)]
    unsafe {
        v.set_len(n);
    }
    v
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
    if m == 1 {
        let mut out = vec![0.0f32; n];
        for (kk, &av) in lhs.iter().take(k).enumerate() {
            let brow = &b.data[kk * n..(kk + 1) * n];
            for (o, &bv) in out.iter_mut().zip(brow) {
                *o += av * bv;
            }
        }
        return Ok(Mat::from_vec(1, n, out));
    }

    let lhs_meta = meta_2d(m, k);
    let rhs_meta = meta_2d(k, n);
    let data = matmul_into_uninit(lhs, &b.data, &lhs_meta, &rhs_meta, m * n)?;
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
    let (a_chunks, a_remainder) = a.as_chunks::<8>();
    let (b_chunks, b_remainder) = b.as_chunks::<8>();
    for (ach, bch) in a_chunks.iter().zip(b_chunks.iter()) {
        for i in 0..8 {
            acc[i] += ach[i] * bch[i];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (&av, &bv) in a_remainder.iter().zip(b_remainder.iter()) {
        s += av * bv;
    }
    s
}

/// Whether the **fused f16c dot** ([`dot_f16c`]) is compiled in and enabled. True
/// only when the *build target* has `f16c`+`fma` — franken's `x86-64-v3` baseline
/// does (`.cargo/config.toml`, lever L7) — and the ops escape hatch is unset.
/// Otherwise the GEMV uses the portable two-pass (`convert_to_f32_slice`+[`dot8`]),
/// so output on non-f16c builds/CPUs is unchanged. Because the binary already
/// requires `x86-64-v3` to run, this is a compile-time fact, not a CPUID gamble.
#[inline]
fn f16c_dot_available() -> bool {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "fma"
    ))]
    {
        use std::sync::OnceLock;
        static AVAIL: OnceLock<bool> = OnceLock::new();
        // Ops/debug escape hatch: force the portable two-pass.
        *AVAIL.get_or_init(|| std::env::var_os("FW_DISABLE_F16C_DOT").is_none())
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "fma"
    )))]
    {
        false
    }
}

/// Fused dequant-in-register f16 dot: `sum(f16→f32(w[i]) * x[i])` using
/// `vcvtph2ps` (`_mm256_cvtph_ps`) + FMA over four independent 8-lane
/// accumulators, with **no f32 scratch roundtrip**. This is the GGML-style dot
/// the safe two-pass ([`HalfFloatSliceExt::convert_to_f32_slice`] then [`dot8`])
/// emulates under the crate's `deny(unsafe_code)`; it is the measured **2.5–5×**
/// decoder-GEMV lever (NEGATIVE_EVIDENCE 2026-06-25). The result differs from
/// [`dot8`] only in f32 FMA/reduction order (rel ≈ 3e-6 on whisper shapes), well
/// inside the [`gemv_f16`] tolerance gate (`gemv_f16_matches_dequant_then_matmul`,
/// `< 1e-4`); the GEMV is already a numerics-affecting path vs the f32 sgemm
/// reference. All whisper `inp` (n_state/mlp_hidden) are multiples of 32; the
/// 8-lane and scalar tails are defensive for arbitrary lengths.
///
/// Compiled only under `target_feature = "f16c"`+`"fma"` (so the intrinsics are
/// available **without** a `#[target_feature]` attribute → this fn fully inlines,
/// unlike a feature-boundary call). Safe to call with any valid slices: the
/// internal `unsafe` only does in-bounds raw loads (`Float16` is
/// `repr(transparent)` over `u16`; the `i+32`/`i+8`/`i<n` guards bound every
/// access), so no caller precondition.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "f16c",
    target_feature = "fma"
))]
#[inline]
#[allow(unsafe_code)]
fn dot_f16c(w: &[Float16], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = w.len().min(x.len());
    let xp = x.as_ptr();
    // SAFETY: every load is in-bounds by the i+32 / i+8 / i<n guards over
    // n = min(w.len, x.len); Float16 is repr(transparent) u16 so a 128-bit load
    // reads 8 contiguous lanes; f16c/fma are guaranteed by this fn's target_feature cfg.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 32 <= n {
            let w0 = _mm_loadu_si128(w.as_ptr().add(i).cast());
            let w1 = _mm_loadu_si128(w.as_ptr().add(i + 8).cast());
            let w2 = _mm_loadu_si128(w.as_ptr().add(i + 16).cast());
            let w3 = _mm_loadu_si128(w.as_ptr().add(i + 24).cast());
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(w0), _mm256_loadu_ps(xp.add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(w1), _mm256_loadu_ps(xp.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(w2), _mm256_loadu_ps(xp.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(w3), _mm256_loadu_ps(xp.add(i + 24)), a3);
            i += 32;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut s =
            ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
        while i + 8 <= n {
            let p = _mm256_mul_ps(
                _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i).cast())),
                _mm256_loadu_ps(xp.add(i)),
            );
            let mut t = [0.0f32; 8];
            _mm256_storeu_ps(t.as_mut_ptr(), p);
            s += ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
            i += 8;
        }
        while i < n {
            s += w[i].to_f32() * x[i];
            i += 1;
        }
        s
    }
}

/// Two GEMV row dots over **two** contiguous f16 weight rows against the **same**
/// activation `x` (register-blocked GEMV). Each row keeps its OWN four AVX/F16C
/// accumulators reduced in the byte-identical order of [`dot_f16c`], so
/// `dot_f16c_2row(w0, w1, x) == (dot_f16c(w0, x), dot_f16c(w1, x))` **bit-for-bit**
/// — this is purely an instruction-scheduling reshape, NOT a numerics change.
///
/// The win over two separate [`dot_f16c`] calls: the `x[i..]` loads are issued
/// once and reused for both rows (halving activation L1 traffic), and the two
/// rows' independent FMA chains + horizontal-reduction tails interleave, hiding
/// the per-row reduction latency that dominates SHORT contractions (n_state/d_head
/// = 64..1280). Measured 1.17–1.27× on the cache-resident decode GEMVs (mlp fc/proj,
/// self/cross projections, turbo attention). Restricted by the caller to weights
/// that fit cache — the 40–130 MB logits stream is memory-bound and REGRESSES
/// ~2× under the two-stream access pattern, so it stays on the single-row path.
///
/// Compiled only under `f16c`+`fma` (same cfg/inlining contract as [`dot_f16c`]).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "f16c",
    target_feature = "fma"
))]
#[inline]
#[allow(unsafe_code)]
fn dot_f16c_2row(w0: &[Float16], w1: &[Float16], x: &[f32]) -> (f32, f32) {
    use core::arch::x86_64::*;
    let n = w0.len().min(w1.len()).min(x.len());
    let xp = x.as_ptr();
    let p0 = w0.as_ptr();
    let p1 = w1.as_ptr();
    // SAFETY: identical in-bounds contract to `dot_f16c`, applied to both rows;
    // every load is bounded by the i+32 / i+8 / i<n guards over n = min of the
    // three lengths. Float16 is repr(transparent) u16; f16c/fma guaranteed by cfg.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut b0 = _mm256_setzero_ps();
        let mut b1 = _mm256_setzero_ps();
        let mut b2 = _mm256_setzero_ps();
        let mut b3 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 32 <= n {
            let xa = _mm256_loadu_ps(xp.add(i));
            let xb = _mm256_loadu_ps(xp.add(i + 8));
            let xc = _mm256_loadu_ps(xp.add(i + 16));
            let xd = _mm256_loadu_ps(xp.add(i + 24));
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p0.add(i).cast())), xa, a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p0.add(i + 8).cast())), xb, a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p0.add(i + 16).cast())), xc, a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p0.add(i + 24).cast())), xd, a3);
            b0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p1.add(i).cast())), xa, b0);
            b1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p1.add(i + 8).cast())), xb, b1);
            b2 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p1.add(i + 16).cast())), xc, b2);
            b3 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p1.add(i + 24).cast())), xd, b3);
            i += 32;
        }
        let acc0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let acc1 = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
        let mut t = [0.0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), acc0);
        let mut s0 =
            ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
        _mm256_storeu_ps(t.as_mut_ptr(), acc1);
        let mut s1 =
            ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
        while i + 8 <= n {
            let xv = _mm256_loadu_ps(xp.add(i));
            let p = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(p0.add(i).cast())), xv);
            let mut u = [0.0f32; 8];
            _mm256_storeu_ps(u.as_mut_ptr(), p);
            s0 += ((u[0] + u[1]) + (u[2] + u[3])) + ((u[4] + u[5]) + (u[6] + u[7]));
            let q = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(p1.add(i).cast())), xv);
            _mm256_storeu_ps(u.as_mut_ptr(), q);
            s1 += ((u[0] + u[1]) + (u[2] + u[3])) + ((u[4] + u[5]) + (u[6] + u[7]));
            i += 8;
        }
        while i < n {
            let xv = x[i];
            s0 += w0[i].to_f32() * xv;
            s1 += w1[i].to_f32() * xv;
            i += 1;
        }
        (s0, s1)
    }
}

/// One GEMV row dot over an f16 weight row and f32 activation: the fused f16c
/// path when available ([`dot_f16c`], a safe call — its `unsafe` is internal),
/// else the portable two-pass (`convert_to_f32_slice` into `scratch`, then
/// [`dot8`]). `use_fused` is hoisted by the caller from [`f16c_dot_available`].
#[inline]
fn dequant_row_dot(w_row: &[Float16], x: &[f32], scratch: &mut [f32], use_fused: bool) -> f32 {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "fma"
    ))]
    if use_fused {
        return dot_f16c(w_row, x);
    }
    let _ = use_fused;
    w_row.convert_to_f32_slice(scratch);
    dot8(scratch, x)
}

/// Whether the register-blocked two-row fused dot ([`dot_f16c_2row`]) is both
/// available (f16c/fma) and beneficial for a `[out, inp]` weight. The win is on
/// cache-resident weights; the 40–130 MB logits projection is memory-bandwidth
/// bound and regresses ~2× under the two-stream pattern, so weights at/above the
/// threshold stay on the single-row path. `1<<22` elements = 8 MB of f16 weight
/// cleanly separates the per-token decode GEMVs (mlp 590 k, qkv 147 k, turbo attn
/// 1.6 M — all blocked) from the logits GEMV (tiny 20 M / turbo 66 M — single-row).
#[inline]
fn two_row_blocked(out: usize, inp: usize, use_fused: bool) -> bool {
    const TWO_ROW_MAX_ELEMS: usize = 1 << 22;
    let _ = (out, inp);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "fma"
    ))]
    {
        return use_fused && out * inp < TWO_ROW_MAX_ELEMS;
    }
    #[allow(unreachable_code)]
    {
        let _ = use_fused;
        false
    }
}

/// Two contiguous GEMV row dots against the same `x`, register-blocked when the
/// fused f16c path is available ([`dot_f16c_2row`], bit-identical to two
/// [`dequant_row_dot`] calls), else the portable two-pass per row. Callers gate
/// the blocked path via [`two_row_blocked`], so the fallback here is only reached
/// on non-f16c targets (where it preserves correctness).
#[inline]
fn dequant_2row_dot(
    w0: &[Float16],
    w1: &[Float16],
    x: &[f32],
    scratch: &mut [f32],
    use_fused: bool,
) -> (f32, f32) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "fma"
    ))]
    if use_fused {
        return dot_f16c_2row(w0, w1, x);
    }
    let s0 = dequant_row_dot(w0, x, scratch, use_fused);
    let s1 = dequant_row_dot(w1, x, scratch, use_fused);
    (s0, s1)
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

    // One output row: fused f16c dot when the CPU supports it (measured 2.5–5×),
    // else the portable two-pass (`convert_to_f32_slice` into `scratch`, [`dot8`]).
    // The CPUID check is hoisted here so it is out of the per-row loop.
    let use_fused = f16c_dot_available();
    let row_dot = |o: usize, scratch: &mut [f32]| -> f32 {
        let w_row = &w_f16[o * inp..(o + 1) * inp];
        let acc = dequant_row_dot(w_row, x, scratch, use_fused);
        match bias {
            Some(b) => acc + b[o],
            None => acc,
        }
    };

    // Register-blocked two-row dot for cache-resident weights (mlp/qkv/attn
    // projections): bit-identical to two `row_dot`s, but shares the `x` loads and
    // interleaves the two reduction tails (1.17–1.27× on the short-contraction
    // decode GEMVs). Gated OFF for the memory-bound logits stream by
    // `two_row_blocked`. Fills output rows `[o_base, o_base+slice.len())`.
    let use_2row = two_row_blocked(out, inp, use_fused);
    let fill_rows = |o_base: usize, slice: &mut [f32], scratch: &mut [f32]| {
        if use_2row {
            let n = slice.len();
            let mut i = 0usize;
            while i + 2 <= n {
                let o = o_base + i;
                let (mut s0, mut s1) = dequant_2row_dot(
                    &w_f16[o * inp..(o + 1) * inp],
                    &w_f16[(o + 1) * inp..(o + 2) * inp],
                    x,
                    scratch,
                    use_fused,
                );
                if let Some(b) = bias {
                    s0 += b[o];
                    s1 += b[o + 1];
                }
                slice[i] = s0;
                slice[i + 1] = s1;
                i += 2;
            }
            if i < n {
                slice[i] = row_dot(o_base + i, scratch);
            }
        } else {
            for (i, slot) in slice.iter_mut().enumerate() {
                *slot = row_dot(o_base + i, scratch);
            }
        }
    };

    // MACs of real work = out * inp; below the threshold, parallel dispatch isn't
    // worth it. History (bd-6qih): the original M4 Pro sweep chose `1<<19`; L9
    // raised it to `1<<21` because the per-token mid GEMVs (`[384,1536]`=590 k)
    // were SPAWN-BOUND under load on the old `std::thread::scope` path (per-call
    // spawn/join dominated ~20 µs of compute), so serial beat spawning (−9.5% e2e).
    // L11 fixes the *real* problem: dispatch via rayon's PERSISTENT global pool
    // (no per-call spawn — what whisper.cpp's pool does), so the mid GEMVs can
    // parallelize again. Standalone (contended host): rayon beats serial 1.40×
    // (`[1536,384]`) / 1.35× (`[384,1536]`), bit-identical (disjoint output-row
    // bands, each row's [`dot8`] order unchanged). Threshold back to `1<<19`:
    // mlp (590 k) + logits (20 M) parallelize, the tiny `[384,384]`=147 k stay
    // serial (rayon task overhead not worth it there).
    const PAR_THRESHOLD: usize = 1 << 19;
    let workers = gemv_worker_count(out);
    if out * inp < PAR_THRESHOLD || workers < 2 {
        // The fused f16c path (`dequant_row_dot` with `use_fused`) reads the f16
        // weights directly and never touches `scratch`; only the portable
        // two-pass dequantizes into it. So skip the per-call alloc+zero entirely
        // when fused (output is unaffected — `scratch` is dead on that path).
        let mut scratch = if use_fused {
            Vec::new()
        } else {
            vec![0.0f32; inp]
        };
        fill_rows(0, out_slice, &mut scratch);
        return;
    }
    let band = out.div_ceil(workers).max(1);
    out_slice
        .par_chunks_mut(band)
        .enumerate()
        .for_each(|(w, band_slice)| {
            let o_base = w * band;
            // See the serial branch: `scratch` is dead on the fused f16c path.
            let mut scratch = if use_fused {
                Vec::new()
            } else {
                vec![0.0f32; inp]
            };
            fill_rows(o_base, band_slice, &mut scratch);
        });
}

/// Per-output-row symmetric-int8-quantized weight (`[out, in]` row-major) with a
/// per-row f32 scale. Cuts the resident bytes in HALF vs [`WeightMat::F16`] — the
/// lever for the memory-bandwidth-bound vocab-class logits GEMV (measured 1.86×
/// single-thread vs f16 on `[51866,1280]`; the logits stream is DRAM-bandwidth-
/// bound, so halving the bytes ~halves the time). Numerics-affecting (int8 ≈ 256
/// levels): built + used only behind [`super::int8_logits_enabled`].
#[derive(Debug, Clone)]
pub struct I8Mat {
    /// Quantized weights, `out * in` elements row-major, each in `[-127, 127]`.
    pub data: Vec<i8>,
    /// Per-output-row dequant scale (`amax_row / 127`), `out` elements.
    pub scales: Vec<f32>,
    /// Output dimension (rows).
    pub out: usize,
    /// Input dimension (contraction length).
    pub inp: usize,
}

/// Per-output-row symmetric int8 quantization of a natural `[out, in]` f16
/// weight: `scale[o] = max_i |w[o,i]| / 127`, `q[o,i] = round(w[o,i]/scale[o])`.
/// Parallel over rows (each independent). The inverse `w ≈ q * scale` is what
/// [`gemv_i8`] reconstructs.
pub fn quantize_f16_to_i8(w: &[Float16], out: usize, inp: usize) -> I8Mat {
    debug_assert_eq!(w.len(), out * inp);
    let mut data = vec![0i8; out * inp];
    let mut scales = vec![0.0f32; out];
    data.par_chunks_mut(inp)
        .zip(scales.par_iter_mut())
        .enumerate()
        .for_each(|(o, (drow, s))| {
            let wrow = &w[o * inp..(o + 1) * inp];
            let amax = wrow
                .iter()
                .map(|h| h.to_f32().abs())
                .fold(0.0f32, f32::max)
                .max(1e-9);
            let sc = amax / 127.0;
            *s = sc;
            let inv = 1.0 / sc;
            for (d, h) in drow.iter_mut().zip(wrow) {
                *d = (h.to_f32() * inv).round().clamp(-127.0, 127.0) as i8;
            }
        });
    I8Mat { data, scales, out, inp }
}

/// Signed int8 dot. LLVM lowers this to `vpmovsxbw`+`vpmaddwd` under x86-64-v3;
/// the int8 compute far outruns the DRAM read rate, so the GEMV stays memory-
/// bound (the point: half the weight bytes of the f16 path).
#[inline]
fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    for (a, b) in w.iter().zip(x.iter()) {
        acc += (*a as i32) * (*b as i32);
    }
    acc
}

/// Fused int8 GEMV: `out[o] = (Σ_i q_w[o,i] · q_x[i]) · scale_w[o] · scale_x`.
/// Quantizes the activation `x` to int8 per-vector (symmetric), then dots each
/// weight row. Parallelizes over output-row bands exactly like [`gemv_f16`]
/// (wide worker cap for the vocab-class logits). A numerics-affecting int8
/// approximation of the f16 GEMV — the caller gates it ([`super::int8_logits_enabled`]).
pub fn gemv_i8(w: &I8Mat, x: &[f32], bias: Option<&[f32]>, out_slice: &mut [f32]) {
    let (out, inp) = (w.out, w.inp);
    debug_assert_eq!(w.data.len(), out * inp, "gemv_i8 weight shape mismatch");
    debug_assert_eq!(x.len(), inp, "gemv_i8 x length mismatch");
    debug_assert_eq!(out_slice.len(), out, "gemv_i8 out length mismatch");
    debug_assert!(bias.is_none_or(|b| b.len() == out), "gemv_i8 bias length mismatch");
    // Quantize the activation once (per-vector symmetric), shared by all rows.
    let xamax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    let xs = xamax / 127.0;
    let xinv = 1.0 / xs;
    let xi8: Vec<i8> = x
        .iter()
        .map(|v| (v * xinv).round().clamp(-127.0, 127.0) as i8)
        .collect();

    let fill = |o_base: usize, slice: &mut [f32]| {
        for (i, slot) in slice.iter_mut().enumerate() {
            let o = o_base + i;
            let acc = dot_i8(&w.data[o * inp..(o + 1) * inp], &xi8) as f32 * w.scales[o] * xs;
            *slot = acc + bias.map_or(0.0, |b| b[o]);
        }
    };

    // Parallelize only GEMVs whose `out*inp` clears this bar. At `1<<19` the small
    // decode projections (`self_out`/`cross_q`/`cross_out` = n_state² = 1.64 M for
    // large/turbo) were parallelized, but their per-row int8 dot is ~0.03 ms of
    // compute — `par_chunks_mut`'s rayon coordination cost DOMINATED it (MEASURED
    // serial 1.3–1.8× faster on those spans, min-of-8). `1<<21` (2.10 M) keeps them
    // serial while `qkv` (4.9 M), `mlp_0` (6.5 M) and the vocab logits (66 M) — which
    // genuinely amortize the spawn — stay parallel (also for medium's 3.15 M `qkv`).
    // Bit-identical: parallel vs serial is a disjoint output-row partition, same math.
    // Escape hatch / tuner: `FW_GEMV_I8_PAR`.
    let par_threshold = {
        use std::sync::OnceLock;
        static T: OnceLock<usize> = OnceLock::new();
        *T.get_or_init(|| {
            std::env::var("FW_GEMV_I8_PAR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1 << 21)
        })
    };
    let workers = gemv_worker_count(out);
    if out * inp < par_threshold || workers < 2 {
        fill(0, out_slice);
        return;
    }
    let band = out.div_ceil(workers).max(1);
    out_slice
        .par_chunks_mut(band)
        .enumerate()
        .for_each(|(wk, band_slice)| fill(wk * band, band_slice));
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
    // Fused f16c dot per (o, t) when available, else the two-pass. Matches
    // [`gemv_f16`]'s `row_dot` exactly (same [`dequant_row_dot`]), so the batch
    // path is bit-for-bit identical to per-token gemv. (The fused dot dequants
    // in-register, so there is no whole-row dequant to amortize across `tq`; the
    // two-pass fallback re-dequants per token — a minor cost only on the rare
    // pre-f16c CPU that takes that path.)
    let use_fused = f16c_dot_available();
    let compute_band = |o0: usize, o1: usize, dst: &mut [f32]| {
        // dst is the FULL [tq, out] buffer in serial mode, or in parallel mode a
        // per-worker private [tq, out] buffer it later disjoint-merges. Either
        // way we write only columns [o0, o1).
        let mut scratch = vec![0.0f32; inp];
        for o in o0..o1 {
            let w_row = &w_f16[o * inp..(o + 1) * inp];
            let b = bias.map_or(0.0, |bb| bb[o]);
            for t in 0..tq {
                let xr = &x[t * inp..(t + 1) * inp];
                dst[t * out + o] = dequant_row_dot(w_row, xr, &mut scratch, use_fused) + b;
            }
        }
    };

    // Same measured crossover as [`gemv_f16`] (see its `PAR_THRESHOLD` note),
    // but the work metric carries the batch dimension: each weight row is
    // dequantized once and dotted against all `tq` token rows, so the spawn is
    // amortized over `tq * out * inp` MACs. `1 << 19` keeps small prefills
    // serial while still parallelizing the realistic multi-token prompt batches.
    const PAR_THRESHOLD: usize = 1 << 21;
    // Unlike the m=1 gemv (dispatch-bound, cap8), a COMPUTE-bound BATCHED gemv
    // (tq>1, large work — e.g. cross-KV at tq=1500, ~2.4 GFLOP, and long prompt
    // prefills) scales past the m=1 cap: MEASURED 1.50× at 16 vs 8 (32 plateaus).
    // Use 16 once the work clears a compute-bound bar; small prefills keep the
    // m=1 cap (`gemv_worker_count`). `FW_BATCH_GEMV_CAP` overrides.
    const COMPUTE_BOUND_MACS: usize = 1 << 26; // ~67M: cross-KV (2.4G) + long prompts
    let avail = avail_parallelism();
    let work = tq.saturating_mul(out).saturating_mul(inp);
    let workers = batch_gemv_cap()
        .map(|c| avail.min(c))
        .unwrap_or_else(|| {
            if work >= COMPUTE_BOUND_MACS {
                avail.min(16)
            } else {
                gemv_worker_count(out)
            }
        });
    if work < PAR_THRESHOLD || workers < 2 {
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
    // Persistent rayon pool instead of `thread::scope` (which spawned/joined N OS
    // threads PER CALL — ~12 layer_norms/encoder-window × 16 = a clone3 storm at
    // ~30 µs each, often dwarfing this cheap O(elements) op). Same contiguous band
    // split ⇒ byte-identical (`layer_norm_simd_matches_scalar` + conformance gate).
    x.data
        .par_chunks_mut(band_rows * cols)
        .for_each(|band| norm_rows(band, cols, w, b, eps));
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
            let normed = (*s - mean) * inv * V::splat(f64::from(w[j])) + V::splat(f64::from(b[j]));
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

/// The `1 << 16`-entry f16 GELU lookup table, precomputed once, EXACTLY as ggml
/// builds `ggml_table_gelu_f16` (`ggml-cpu.c`): for every f16 bit pattern `i`,
/// `table[i] = f16→f32( f32→f16( gelu_tanh( f16→f32(i) ) ) )` — i.e. the tanh
/// GELU of the dequantized half, re-rounded to f16, then widened back to f32 (the
/// value ggml's `GGML_GELU_FP16` path returns). Stored pre-widened to f32 so the
/// hot lookup is one `f32→f16` index + one load, no per-element `tanh`.
///
/// `f16::from_bits`/`from_f32` use IEEE round-to-nearest-even, identical to ggml's
/// `GGML_CPU_FP16_TO_FP32` / `GGML_CPU_FP32_TO_FP16` (f16c), so the table is
/// bit-identical to whisper.cpp's — see [`gelu_slice`].
fn gelu_table() -> &'static [f32; 1 << 16] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Box<[f32; 1 << 16]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = vec![0.0f32; 1 << 16].into_boxed_slice();
        for (i, slot) in t.iter_mut().enumerate() {
            let f = Float16::from_bits(i as u16).to_f32();
            let g = 0.5 * f * (1.0 + (GELU_SQRT_2_OVER_PI * f * (1.0 + GELU_COEF_A * f * f)).tanh());
            *slot = Float16::from_f32(g).to_f32();
        }
        // Vec<f32> of exactly 1<<16 elements → Box<[f32; 1<<16]> (infallible).
        t.try_into().expect("gelu table length 1<<16")
    })
}

/// In-place GELU, bit-identical to whisper.cpp's shipped `ggml_vec_gelu_f32`.
///
/// whisper.cpp builds with `GGML_GELU_FP16` (see `ggml-cpu/vec.h`), so its GELU is
/// NOT the live tanh but a **f16 lookup table** with a saturating clamp:
/// `x <= -10 → 0`, `x >= 10 → x`, else `table[f16(x)]`. franken previously computed
/// the live tanh form, which DIVERGED from ORIG (more accurate, but not what
/// whisper actually runs). This matches whisper exactly — restoring
/// bit-exact-with-whisper on the activation — and is far cheaper (a `vcvtps2ph` +
/// table load per element vs a scalar `tanh` per lane). GELU is on the
/// transcription-tolerance encoder/decoder path (never the bit-exact mel path).
///
/// x86-64-v3 path: 8-wide `vcvtps2ph` (round-to-nearest-even → the same f16 index
/// as the scalar `Float16::from_f32`) → widen → AVX2 gather from the table → blend
/// the clamp. Bit-identical to the scalar fallback (verified max|Δ|=0 in
/// `examples/gelu_probe`), 1.38× faster than it / 4.4× faster than the old tanh.
#[cfg(all(target_arch = "x86_64", target_feature = "f16c", target_feature = "avx2"))]
#[allow(unsafe_code)]
fn gelu_slice(data: &mut [f32]) {
    use core::arch::x86_64::*;
    let table = gelu_table();
    let tp = table.as_ptr();
    let n = data.len();
    // SAFETY: all loads/stores are bounded by the `i+8<=n` guard; the gather index
    // is a widened f16 bit pattern (always 0..=65535, in-bounds for the 1<<16 table);
    // f16c/avx2 are guaranteed by this fn's target_feature cfg.
    unsafe {
        let neg10 = _mm256_set1_ps(-10.0);
        let pos10 = _mm256_set1_ps(10.0);
        let zero = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let x = _mm256_loadu_ps(data.as_ptr().add(i));
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x);
            let idx = _mm256_cvtepu16_epi32(h);
            let g = _mm256_i32gather_ps::<4>(tp, idx);
            // Clamp (ggml GGML_GELU_FP16): x>=10 → x, x<=-10 → 0, else gathered.
            let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(x, pos10);
            let le = _mm256_cmp_ps::<_CMP_LE_OQ>(x, neg10);
            let r = _mm256_blendv_ps(g, x, ge);
            let r = _mm256_blendv_ps(r, zero, le);
            _mm256_storeu_ps(data.as_mut_ptr().add(i), r);
            i += 8;
        }
        for v in &mut data[i..] {
            let x = *v;
            *v = if x <= -10.0 {
                0.0
            } else if x >= 10.0 {
                x
            } else {
                table[Float16::from_f32(x).to_bits() as usize]
            };
        }
    }
}

/// Scalar fallback (non-x86 / no f16c+avx2): exact ggml `GGML_GELU_FP16` branch + clamp.
#[cfg(not(all(target_arch = "x86_64", target_feature = "f16c", target_feature = "avx2")))]
fn gelu_slice(data: &mut [f32]) {
    let table = gelu_table();
    for v in data.iter_mut() {
        let x = *v;
        *v = if x <= -10.0 {
            0.0
        } else if x >= 10.0 {
            x
        } else {
            table[Float16::from_f32(x).to_bits() as usize]
        };
    }
}

pub fn gelu(x: &mut Mat) {
    // Pure elementwise: each output depends only on its own input, so we
    // split `data` into disjoint contiguous chunks across workers; threshold
    // keeps small activations serial.
    const PAR_THRESHOLD: usize = 1 << 15;
    let n = x.data.len();
    if n < PAR_THRESHOLD || worker_count() < 2 {
        gelu_slice(&mut x.data);
        return;
    }
    let chunk = n.div_ceil(worker_count()).max(1);
    // Persistent rayon pool, not a per-call `thread::scope` spawn/join (same
    // disjoint contiguous chunks ⇒ byte-identical). gelu is ~4/encoder-window.
    x.data.par_chunks_mut(chunk).for_each(gelu_slice);
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
            // A NaN score (e.g. from an upstream overflow) would make
            // `(*v - max).exp()` NaN, poison `sum`, skip normalization, and
            // leave NaN in the row. Treat non-finite contributions as 0.
            let e = (*v - max).exp();
            let e = if e.is_finite() { e } else { 0.0 };
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
/// Whether the bidirectional (encoder) attention uses the fused
/// `ft_kernel_cpu::sdpa_forward_f32` kernel (default ON; escape hatch
/// `FW_ATTN_NO_SDPA` restores the per-head path). Faithful: MEASURED max|Δ| ~1.2e-7
/// vs the per-head path, far inside the f16c-dot tolerance.
fn use_sdpa_attn() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| std::env::var_os("FW_ATTN_NO_SDPA").is_none())
}

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
    // Fused SDPA path (DEFAULT for the bidirectional encoder attention; escape
    // hatch FW_ATTN_NO_SDPA): `ft_kernel_cpu::sdpa_forward_f32` computes the whole
    // scores+softmax+×V row-tiled in one parallel call (over heads×query-blocks),
    // never materializing the full [tq,tk] scores — MEASURED 2.35x faster than the
    // per-head scheme with max|Δ| ~1.2e-7 (well inside the f16c-dot tolerance).
    // Encoder-only (causal_offset.is_none()): the decode's cached causal attention
    // has a cache_len offset the kernel's square-causal flag does not model.
    if causal_offset.is_none() && use_sdpa_attn() && n_head >= 2 && tq >= 64 {
        let hh = n_head;
        let mut qa = gemv_out_buf(hh * tq * d_head);
        let mut ka = gemv_out_buf(hh * tk * d_head);
        let mut va = gemv_out_buf(hh * tk * d_head);
        qa.par_chunks_mut(tq * d_head).enumerate().for_each(|(h, blk)| {
            let base = h * d_head;
            for i in 0..tq {
                blk[i * d_head..(i + 1) * d_head].copy_from_slice(&q.row(i)[base..base + d_head]);
            }
        });
        ka.par_chunks_mut(tk * d_head).enumerate().for_each(|(h, blk)| {
            let base = h * d_head;
            for j in 0..tk {
                blk[j * d_head..(j + 1) * d_head]
                    .copy_from_slice(&k[j * n_state + base..j * n_state + base + d_head]);
            }
        });
        va.par_chunks_mut(tk * d_head).enumerate().for_each(|(h, blk)| {
            let base = h * d_head;
            for j in 0..tk {
                blk[j * d_head..(j + 1) * d_head]
                    .copy_from_slice(&v[j * n_state + base..j * n_state + base + d_head]);
            }
        });
        let sdpa_scale = (d_head as f32).powf(-0.5);
        let o = ft_kernel_cpu::sdpa_forward_f32(
            &qa, &ka, &va, hh, tq, tk, d_head, d_head, sdpa_scale, false,
        );
        out.par_chunks_mut(n_state).enumerate().for_each(|(i, orow)| {
            for h in 0..hh {
                orow[h * d_head..(h + 1) * d_head].copy_from_slice(
                    &o[h * tq * d_head + i * d_head..h * tq * d_head + (i + 1) * d_head],
                );
            }
        });
        return Ok(Mat::from_vec(tq, n_state, out));
    }

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
/// Escape hatch `FW_FAST_SELF_ATTN=0` restores the `attention_raw` decode path.
fn fast_self_attn_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("FW_FAST_SELF_ATTN").as_deref() != Ok("0"))
}

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
    // Per-token (`tq == 1`) fast path: read K/V straight out of the cache with a
    // per-key dot / per-key SAXPY, so no `kh`/`kh_t`/`vh` gather+transpose+alloc
    // per head (`attention_raw` allocates ~6 buffers/head/token and transposes K).
    // The summation order is identical to `attention_raw`'s m=1 SAXPY (sum over
    // `d_head` ascending for scores, over `tk` ascending for the output), so the
    // f32 result is BIT-IDENTICAL — verified byte-exact. Decode at `tq==1` attends
    // to the whole cache (`limit == tk-1`), so the causal mask is a no-op and is
    // skipped. Prefill (`tq > 1`) keeps `attention_raw`.
    if q.rows == 1 && fast_self_attn_enabled() {
        return attention_decode_step(q, cache.key_slice(), cache.value_slice(), tk, n_head);
    }
    attention_raw(
        q,
        cache.key_slice(),
        cache.value_slice(),
        tk,
        n_head,
        Some(past_len),
    )
}

/// Allocation-light single-token (`tq == 1`) causal self-attention over a cache
/// prefix. Bit-identical to [`attention_raw`] with `causal_offset == Some(tk-1)`.
fn attention_decode_step(
    q: &Mat,
    k: &[f32],
    v: &[f32],
    tk: usize,
    n_head: usize,
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
    let d_head = n_state / n_head;
    if d_head == 0 {
        return Err(FwError::InvalidRequest("attention: d_head == 0".into()));
    }
    let scale = (d_head as f32).powf(-0.25);
    let q0 = q.row(0);
    let mut out = vec![0.0f32; n_state];
    let mut qh = vec![0.0f32; d_head];
    let mut scores = vec![0.0f32; tk];
    for h in 0..n_head {
        let base = h * d_head;
        // Scaled query head (`qh[d] = q[d] * scale`), matching `attention_raw`.
        for (d, slot) in qh.iter_mut().enumerate() {
            *slot = q0[base + d] * scale;
        }
        // scores[j] = sum_d qh[d] * (k[j,base+d] * scale). Same per-term product
        // and same summation order (d ascending) as the m=1 SAXPY over `kh_t`.
        for (j, sj) in scores.iter_mut().enumerate() {
            let krow = &k[j * n_state + base..j * n_state + base + d_head];
            let mut acc = 0.0f32;
            for (d, &qd) in qh.iter().enumerate() {
                acc += qd * (krow[d] * scale);
            }
            *sj = acc;
        }
        // No causal mask: at tq==1 the query attends to every cached key.
        let mut sm = Mat::from_vec(1, tk, std::mem::take(&mut scores));
        softmax_rows(&mut sm);
        scores = sm.data;
        // out[base+d] += sum_j scores[j] * v[j,base+d] (j ascending == m=1 SAXPY).
        for (j, &sj) in scores.iter().enumerate() {
            let vrow = &v[j * n_state + base..j * n_state + base + d_head];
            let orow = &mut out[base..base + d_head];
            for (o, &vd) in orow.iter_mut().zip(vrow) {
                *o += sj * vd;
            }
        }
    }
    Ok(Mat::from_vec(1, n_state, out))
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
/// Cache-blocked SERIAL transpose (no thread spawn). Same tiled permutation as
/// [`transpose_parallel`]'s serial fallback, but never parallel — for callers
/// that already parallelize at a coarser grain (e.g. model load fanning out
/// across layers via rayon: a per-weight `thread::scope` here would nest and
/// spawn-thrash). Pure permutation → bit-identical to `transpose_parallel`.
pub(crate) fn transpose_serial(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(data.len(), rows * cols, "transpose shape/data mismatch");
    const TILE: usize = 64;
    let mut out = vec![0.0f32; rows * cols];
    for r0 in (0..rows).step_by(TILE) {
        let r1 = (r0 + TILE).min(rows);
        for c0 in (0..cols).step_by(TILE) {
            let c1 = (c0 + TILE).min(cols);
            for r in r0..r1 {
                for c in c0..c1 {
                    out[c * rows + r] = data[r * cols + c];
                }
            }
        }
    }
    out
}

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

    let workers = avail_parallelism().min(8);
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
                    "batch[{t},{o}] differs from per-token gemv"
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
        // Expected: whisper.cpp's shipped f16-table GELU (GGML_GELU_FP16), i.e. the
        // tanh form re-rounded through f16 at both the input index and the value.
        let f = |v: f32| {
            let f = Float16::from_f32(v).to_f32();
            let g = 0.5 * f * (1.0 + (GELU_SQRT_2_OVER_PI * f * (1.0 + GELU_COEF_A * f * f)).tanh());
            Float16::from_f32(g).to_f32()
        };
        assert_eq!(x.data[0], f(0.0), "gelu(0) table-exact");
        assert_eq!(x.data[1], f(1.0), "gelu(1) table-exact");
        assert_eq!(x.data[2], f(-1.0), "gelu(-1) table-exact");
        // Spec reference magnitudes (f16 table is within ~1e-3 of the exact tanh).
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

    #[test]
    fn softmax_rows_sanitizes_nan() {
        // A NaN score must not leave NaN in the output row (upstream overflow
        // could otherwise poison the whole decoder residual stream).
        let mut x = Mat::from_vec(1, 3, vec![f32::NAN, 1.0, 0.0]);
        softmax_rows(&mut x);
        assert!(
            x.data.iter().all(|v| v.is_finite()),
            "no NaN/inf in output row"
        );
        // The NaN lane contributes 0; the finite lanes normalize to sum 1.
        let s: f32 = x.row(0).iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum {s}");
        assert_eq!(x.row(0)[0], 0.0, "NaN lane maps to 0");
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
