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
    // enough to want the wide cap; below that the sub-vocab cap applies.
    const WIDE_OUT_THRESHOLD: usize = 1 << 14; // 16384 rows
    let cap = if out >= WIDE_OUT_THRESHOLD {
        wide_gemv_cap()
    } else {
        mid_gemv_cap()
    };
    avail.min(cap)
}

/// Worker cap for every sub-vocab GEMV — i.e. EVERY decoder-layer projection
/// (`qkv` 3840x1280, `mlp_0` 5120x1280, `mlp_2` 1280x5120, `attn_out`/`cross_q`/
/// `cross_out` 1280x1280). Overridable via `FW_MID_GEMV_CAP`.
///
/// The 8 it defaults to came from criterion runs on an **M4 Pro (10 performance
/// + 4 efficiency cores)**, where widening past 8 recruited the efficiency cores
/// and measured +29% on `f16_gemv_dequant_1280x1280`. That is a statement about
/// a heterogeneous 14-core laptop, not about an 8-CCD server part, and it is the
/// same reasoning the vocab head already outgrew when its own cap went 12 -> 32
/// for ~1.4-1.8x (see [`wide_gemv_cap`]). Re-screen it per host rather than
/// inheriting the laptop number.
///
/// Row bands are disjoint and each output row's dot is independent of the split,
/// so this is **bit-identical** at any value — purely a scheduling knob.
fn mid_gemv_cap() -> usize {
    use std::sync::OnceLock;
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("FW_MID_GEMV_CAP")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&c: &usize| c >= 1)
            .unwrap_or(8)
    })
/// Worker width for a cohort of independent one-token decoder streams.
///
/// A scalar GEMV deliberately caps moderate projections at eight workers, but a
/// cohort carries `tq` independent dot products per output row.  Scale that
/// proven per-stream width by the cohort size so five simultaneous windows can
/// occupy roughly forty cores, while the vocabulary projection may use the
/// entire physical-core pool.
#[inline]
fn cohort_gemv_worker_count(out: usize, tq: usize) -> usize {
    avail_parallelism().min(gemv_worker_count(out).saturating_mul(tq).max(1))
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

/// Default-ON row-morsel scheduler for compute-bound batched f16 GEMV.
/// `FW_BATCH_GEMV_ROW_MORSEL=0` restores the legacy output-band/private-buffer
/// scheduler for A/B and rollback.
fn batch_gemv_row_morsel_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("FW_BATCH_GEMV_ROW_MORSEL").ok().as_deref() != Some("0"))
}

/// Default-ON M2 activation-column tile for the row-morsel f16 batch GEMV: pairs of
/// activation rows share each weight-row f16→f32 conversion (`dot_f16c_2col`), which
/// is BYTE-IDENTICAL to the M1 path and MEASURED 1.26× on the cross-K/V projection
/// shape (`examples/f16batch_m2col_probe`). `FW_F16_BATCH_M2COL=0` restores the M1
/// per-row `dequant_row_dot` loop for A/B and rollback.
fn f16_batch_m2col_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("FW_F16_BATCH_M2COL").ok().as_deref() != Some("0"))
}

/// Default-ON 2-token activation-column tile for the int8 batched GEMV
/// (`gemv_i8_batch`, prefill tq>1 + draft): pairs of tokens share each weight row's
/// `vpmovsxbw` (`dot_i8_2col`), BYTE-IDENTICAL to the M1 `dot_i8` loop (integer-exact)
/// and MEASURED 1.15-1.19× at tq=64 (`examples/i8batch_2col_probe`). `FW_I8_BATCH_2COL=0`
/// restores the per-token `dot_i8` loop for A/B and rollback.
fn i8_batch_2col_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("FW_I8_BATCH_2COL").ok().as_deref() != Some("0"))
}

/// Default-ON 4-token activation-column tile for the int8 batched GEMV, layered on
/// top of [`i8_batch_2col_enabled`] (the 4-tile handles groups of 4 tokens, then the
/// 2col tile the ≤3-token remainder, then a 1col tail). Shares each weight-row
/// `vpmovsxbw` across FOUR tokens (0.25 cvt/token vs 2col's 0.5); BYTE-IDENTICAL to
/// the `dot_i8` loop (integer-exact — a ULP-free lever).
///
/// SIZED on the 64-core box (build-remote/run-local, `examples/i8batch_4col_probe`,
/// same-binary order-alternated min-of-80, 3 reps, byte-id=true 12/12): **1.11-1.14×
/// pure-kernel (workers=1, 6/6 always faster)** and **1.03-1.18× at the shipped
/// 16-worker cap for tq≥64**; the tq=8/16t corner oscillates 0.96-1.08× (dispatch
/// noise on a sub-ms op, not a stable regression).
///
/// FLIPPED default-ON (bd-8wq6): the kernel is a strict, integer-exact win on the
/// decode prefill/draft batched GEMV, and the long-form turbo transcript diff of the
/// `compute_band` wire-in indexing (4-tile → 2col remainder → 1col tail) is
/// byte-identical (unset == `FW_I8_BATCH_4COL=0`, jfk ×1/×3/×8), so the previously
/// held-back risk is retired. Kill-switch: `FW_I8_BATCH_4COL=0` restores the 2col path.
fn i8_batch_4col_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("FW_I8_BATCH_4COL").ok().as_deref() != Some("0"))
}

/// Default-ON alternate register tile for square/non-expanding encoder i7 GEMMs.
/// `FW_I7_M2N4=0` restores the legacy M4xN2 tile for A/B and rollback.
fn i7_m2n4_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("FW_I7_M2N4").ok().as_deref() != Some("0"))
}

/// Default-ON row-block co-scheduler for fused head-major encoder Q/K/V i7 GEMMs.
/// `FW_I7_QKV_HEADMAJOR_ROWCO=0` restores the legacy three independent
/// `maddubs_i7_headmajor` passes for A/B and rollback.
fn i7_qkv_headmajor_rowco_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("FW_I7_QKV_HEADMAJOR_ROWCO").ok().as_deref() != Some("0"))
}

#[allow(clippy::too_many_arguments)]
fn gemv_f16_batch_rows(
    w_f16: &[Float16],
    out: usize,
    inp: usize,
    x: &[f32],
    tq: usize,
    bias: Option<&[f32]>,
    out_slice: &mut [f32],
    workers: usize,
    use_fused: bool,
) {
    let row_band = tq.div_ceil(workers).max(1);
    std::thread::scope(|s| {
        let mut t0 = 0;
        let mut remaining = out_slice;
        while t0 < tq {
            let rows = row_band.min(tq - t0);
            let (dst_rows, tail) = remaining.split_at_mut(rows * out);
            remaining = tail;
            s.spawn(move || {
                let mut scratch = vec![0.0f32; inp];
                // M2 activation-column tile: pairs of adjacent activation rows share
                // each weight-row f16→f32 conversion (`dequant_row_dot_2col`), the
                // cvtph-halving win. BYTE-IDENTICAL to the per-row M1 loop below (each
                // row keeps `dot_f16c`'s exact reduction). The odd tail row and the
                // `FW_F16_BATCH_M2COL=0` A/B path fall back to the M1 `dequant_row_dot`.
                let m2 = f16_batch_m2col_enabled();
                let mut local_t = 0;
                while local_t < rows {
                    if m2 && local_t + 2 <= rows {
                        let t = t0 + local_t;
                        let x0 = &x[t * inp..(t + 1) * inp];
                        let x1 = &x[(t + 1) * inp..(t + 2) * inp];
                        let (d0, d1) =
                            dst_rows[local_t * out..(local_t + 2) * out].split_at_mut(out);
                        for o in 0..out {
                            let w_row = &w_f16[o * inp..(o + 1) * inp];
                            let b = bias.map_or(0.0, |bb| bb[o]);
                            let (s0, s1) =
                                dequant_row_dot_2col(w_row, x0, x1, &mut scratch, use_fused);
                            d0[o] = s0 + b;
                            d1[o] = s1 + b;
                        }
                        local_t += 2;
                    } else {
                        let t = t0 + local_t;
                        let xr = &x[t * inp..(t + 1) * inp];
                        let dst_row = &mut dst_rows[local_t * out..(local_t + 1) * out];
                        for o in 0..out {
                            let w_row = &w_f16[o * inp..(o + 1) * inp];
                            let b = bias.map_or(0.0, |bb| bb[o]);
                            dst_row[o] = dequant_row_dot(w_row, xr, &mut scratch, use_fused) + b;
                        }
                        local_t += 1;
                    }
                }
            });
            t0 += rows;
        }
    });
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
    // GPU auto-offload (Apple Silicon): route large GEMMs to the Metal tiled
    // kernel when a GPU is present. Falls through to the CPU sgemm on any error,
    // for small matmuls (launch overhead), or when disabled by
    // `FRANKEN_WHISPER_GPU=0`. All Metal/unsafe is isolated in ft-kernel-metal, so
    // this crate keeps `#![deny(unsafe_code)]`.
    #[cfg(target_os = "macos")]
    {
        let m = lhs_meta.shape()[0];
        let k = lhs_meta.shape()[1];
        let n = rhs_meta.shape()[1];
        // Size check first (cheap) so small/short models never even probe for a
        // GPU — `metal_offload_enabled()` builds the Metal context (device +
        // kernel compile) on first call, which would otherwise tax a fast tiny.en
        // run that has no GEMM large enough to offload anyway.
        if m.saturating_mul(k).saturating_mul(n) >= METAL_MIN_MKN
            && metal_offload_enabled()
            && ft_kernel_metal::sgemm(lhs, rhs, &mut data, m, k, n).is_ok()
        {
            return Ok(data);
        }
    }
    ft_kernel_cpu::matmul_tensor_contiguous_f32_into(&mut data, lhs, rhs, lhs_meta, rhs_meta)
        .map_err(kernel_err)?;
    Ok(data)
}

/// Runtime gate for GPU matmul offload (Apple Silicon). `true` iff a Metal GPU is
/// present and the user hasn't set `FRANKEN_WHISPER_GPU=0`. Probed once, cached.
#[cfg(target_os = "macos")]
fn metal_offload_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let disabled = matches!(
            std::env::var("FRANKEN_WHISPER_GPU").ok().as_deref(),
            Some("0") | Some("off") | Some("false") | Some("no")
        );
        !disabled && ft_kernel_metal::is_available()
    })
}

/// Below this `m*k*n`, per-call GPU launch/sync + buffer-copy overhead outweighs
/// the compute win, so the matmul stays on the (already fast, multi-threaded) CPU.
/// MEASURED on an M4 Pro: tiny.en GEMMs (≤9e8) run *slower* on the GPU, while
/// large-v3-class GEMMs (≥2.5e9) win ~15% wall-clock and offload ~3× the CPU work.
/// The threshold sits between them so only large models auto-offload — small
/// models keep the optimal CPU path, with no regression. (This is a GEMM-only
/// first cut; a batched/overlapped GPU pipeline would widen the win and lower the
/// break-even, at which point this constant should drop.)
#[cfg(target_os = "macos")]
const METAL_MIN_MKN: usize = 2_000_000_000;

/// Allocate an `[n]` f32 buffer that the caller **fully overwrites** before any
/// read, skipping the dead serial zero-init — the same dead-work elision as
/// [`matmul_into_uninit`] (d44f1fa). Used for the decode's per-token GEMV/logits
/// outputs ([`gemv_f16`]/[`gemv_f16_batch`] assign every slot), the encoder
/// SDPA gather buffers (qa/ka/va, each `copy_from_slice`-filled in full), and
/// exact int8/i7 GEMM outputs that assign every matrix element. NOT for
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

/// `[m,k] x [k,n] -> [m,n]` on the allocating FrankenTorch CPU f32 path.
///
/// Unlike [`matmul_raw_lhs`], this entry point never takes the local GEMV fast
/// path, the uninitialized-output path, or Metal auto-offload. It is intended
/// for inference contracts that report an explicitly CPU-only compute path.
/// Every storage/product bound is checked before constructing kernel metadata.
pub(crate) fn matmul_raw_lhs_cpu(lhs: &[f32], m: usize, b: &Mat) -> FwResult<Mat> {
    let k = b.rows;
    let n = b.cols;
    if m == 0 || k == 0 || n == 0 {
        return Err(FwError::InvalidRequest(
            "matmul_raw_lhs_cpu: dimensions must be non-zero".to_owned(),
        ));
    }
    let lhs_len = m.checked_mul(k).ok_or_else(|| {
        FwError::InvalidRequest("matmul_raw_lhs_cpu: lhs dimensions overflow".to_owned())
    })?;
    let rhs_len = k.checked_mul(n).ok_or_else(|| {
        FwError::InvalidRequest("matmul_raw_lhs_cpu: rhs dimensions overflow".to_owned())
    })?;
    let output_len = m.checked_mul(n).ok_or_else(|| {
        FwError::InvalidRequest("matmul_raw_lhs_cpu: output dimensions overflow".to_owned())
    })?;
    if lhs.len() != lhs_len || b.data.len() != rhs_len {
        return Err(FwError::InvalidRequest(
            "matmul_raw_lhs_cpu: storage length does not match dimensions".to_owned(),
        ));
    }
    let lhs_meta = meta_2d(m, k);
    let rhs_meta = meta_2d(k, n);
    let data = ft_kernel_cpu::matmul_tensor_contiguous_f32(lhs, &b.data, &lhs_meta, &rhs_meta)
        .map_err(kernel_err)?;
    if data.len() != output_len {
        return Err(FwError::InvalidRequest(
            "matmul_raw_lhs_cpu: kernel returned an invalid output length".to_owned(),
        ));
    }
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

/// A linear weight quantized to 7-bit for the maddubs int8 encoder GEMM path
/// ([`matmul_bias_i7`]). Stored in NATURAL `[out, in]` row-major layout (each
/// output's `in` weights contiguous) so the maddubs dot is a contiguous stream.
///
/// 7-bit (not 8) is deliberate: `_mm256_maddubs_epi16` sums two `u8·i8` products
/// into an int16, which SATURATES for full int8 (docs/NEGATIVE_EVIDENCE 4cfcd56).
/// With the weight clamped to `[-63, 63]` the pair-sum stays in `[-32130, 32130]`
/// ⊂ int16 → NON-saturating / integer-EXACT (probe `sat_diff=0`, d8b8df6).
/// `scale[o]` (= amax/63) dequantizes; `colsum[o]` (= Σ row-o i7 weights) applies
/// the u8 activation sign-offset (`Σ(a+128)·w = maddubs − 128·Σw`).
#[derive(Debug, Clone)]
pub struct I7Mat {
    /// `[out, in]` row-major i7 weights (values in `[-63, 63]`).
    data: Vec<i8>,
    /// Per output row: `amax / 63`.
    scale: Vec<f32>,
    /// Per output row: sum of the i7 weights (for the +128 u8 sign offset).
    colsum: Vec<i32>,
    /// Output dimension (rows of the natural weight).
    out: usize,
    /// Input / contraction dimension (columns).
    inp: usize,
}

/// Shared inner quantizer for [`I7Mat`]: for output row `o`, `weight(o, i)` yields
/// the f32 weight at contraction index `i` (`0..inp`). Parallel over output rows;
/// identical amax/scale/EF/colsum arithmetic for every caller — the ONLY thing that
/// varies is WHERE the f32 comes from (a pre-transposed `[in, out]` `Mat`, or the raw
/// ggml `[out, in]` f16 bytes read directly). `weight` is monomorphized so it inlines
/// with no indirection. This is the single source of the quant arithmetic — the two
/// public entry points below must stay bit-identical by sharing it.
#[must_use]
fn quantize_rows_to_i7(
    out: usize,
    inp: usize,
    weight: impl Fn(usize, usize) -> f32 + Sync,
) -> I7Mat {
    let mut data = vec![0i8; out * inp];
    let mut scale = vec![0.0f32; out];
    let mut colsum = vec![0i32; out];
    // Deterministic flag (OnceLock env read) — hoisted out of the row loop; constant
    // across rows, so the branch taken is identical to reading it per row.
    let ef = crate::native_engine::enc_ef_quant();
    data.par_chunks_mut(inp)
        .zip(scale.par_iter_mut())
        .zip(colsum.par_iter_mut())
        .enumerate()
        .for_each(|(o, ((drow, s), cs))| {
            let mut amax = 1e-9f32;
            for i in 0..inp {
                amax = amax.max(weight(o, i).abs());
            }
            let sc = amax / 63.0;
            *s = sc;
            let inv = 1.0 / sc;
            let mut acc = 0i32;
            if ef {
                // Error-feedback (error-diffusion) rounding: carry each element's
                // rounding residual (in QUANTIZED units) into the next, so the
                // per-column dot Σ q_i·a_i has less accumulated quantization bias
                // than independent round-to-nearest. Same i7 format/scale/colsum.
                let mut err = 0.0f32;
                for (i, d) in drow.iter_mut().enumerate() {
                    let target = weight(o, i) * inv + err;
                    let q = target.round().clamp(-63.0, 63.0);
                    err = target - q; // residual carried forward
                    let qi = q as i32;
                    *d = qi as i8;
                    acc += qi;
                }
            } else {
                for (i, d) in drow.iter_mut().enumerate() {
                    let q = (weight(o, i) * inv).round().clamp(-63.0, 63.0) as i32;
                    *d = q as i8;
                    acc += q;
                }
            }
            *cs = acc;
        });
    I7Mat {
        data,
        scale,
        colsum,
        out,
        inp,
    }
}

/// Quantize a pre-transposed `[in, out]` f32 weight (the [`matmul_bias`] layout)
/// to a 7-bit `[out, in]` [`I7Mat`] for [`matmul_bias_i7`]. One-time, at load;
/// parallel over output rows. The strided gather of column `o` is a load-time
/// cost (amortized over every window's GEMM).
#[must_use]
pub fn quantize_mat_to_i7(w_t: &Mat) -> I7Mat {
    let inp = w_t.rows;
    let out = w_t.cols;
    quantize_rows_to_i7(out, inp, |o, i| w_t.data[i * out + o])
}

/// Quantize DIRECTLY from ggml's raw `[out, in]` f16 bytes (little-endian, row-major
/// — `raw` is borrowed from the resident model blob) to a 7-bit `[out, in]` [`I7Mat`],
/// WITHOUT ever materializing the intermediate transposed f32 `Mat`. **Bit-identical**
/// to `quantize_mat_to_i7(&load_linear_transposed(..))`: output row `o` reads ggml row
/// `o` (`raw[(o*inp + i)*2]`), which is exactly the value sequence the transposed path
/// gathers for column `o` — the same `Float16::from_bits` of the same LE pair (see
/// `encoder::dequant_transpose_f16_bytes`). It skips BOTH the transpose and the f32
/// round-trip (the load-time win) and leaves no per-linear f32 transient (the peak
/// win) — the i7's natural `[out, in]` layout IS ggml's, so no transpose is needed.
/// `raw.len()` must equal `out * inp * 2`.
#[must_use]
pub fn quantize_f16_bytes_to_i7(raw: &[u8], out: usize, inp: usize) -> I7Mat {
    debug_assert_eq!(raw.len(), out * inp * 2, "f16 byte length != out*inp*2");
    quantize_rows_to_i7(out, inp, |o, i| {
        let off = (o * inp + i) * 2;
        Float16::from_bits(u16::from_le_bytes([raw[off], raw[off + 1]])).to_f32()
    })
}

#[cfg(test)]
pub(crate) fn dequant_i7_for_test(w: &I7Mat) -> Mat {
    let mut data = vec![0.0f32; w.inp * w.out];
    for o in 0..w.out {
        let row = &w.data[o * w.inp..(o + 1) * w.inp];
        for (i, &q) in row.iter().enumerate() {
            data[i * w.out + o] = f32::from(q) * w.scale[o];
        }
    }
    Mat::from_vec(w.inp, w.out, data)
}

/// Quantize `w_t` (`[inp, out]`, element `w_t.data[i*out + o]`) to i7 then DEQUANTIZE
/// back to f32, returning the perturbed weight. `block == None` ⇒ one scale per output
/// column `o` over the whole `inp` dim (== [`quantize_mat_to_i7`] granularity = the current
/// int8 encoder). `block == Some(b)` ⇒ one scale per `b`-element block along `inp` (the
/// Q8_0 / block-wise granularity). Feasibility harness ONLY: running the EXISTING f32 GEMM
/// on the roundtripped weights isolates the WEIGHT-quant-granularity effect on the transcript
/// (does block-wise recover the int8 encoder's proper-noun errors?) without the multi-hour
/// block-wise maddubs kernel. Serial: one-time load cost. See `examples/blockwise_i7_quant_probe`.
pub fn i7_roundtrip(w_t: &Mat, block: Option<usize>) -> Mat {
    let inp = w_t.rows;
    let out = w_t.cols;
    let b = block.unwrap_or(inp).max(1);
    let mut data = vec![0.0f32; inp * out];
    for o in 0..out {
        let mut i0 = 0;
        while i0 < inp {
            let i1 = (i0 + b).min(inp);
            let mut amax = 1e-9f32;
            for i in i0..i1 {
                amax = amax.max(w_t.data[i * out + o].abs());
            }
            let sc = amax / 63.0;
            let inv = 1.0 / sc;
            for i in i0..i1 {
                let q = (w_t.data[i * out + o] * inv).round().clamp(-63.0, 63.0);
                data[i * out + o] = q * sc;
            }
            i0 = i1;
        }
    }
    Mat::from_vec(inp, out, data)
}

/// Roundtrip each row of `x` ([m, inp]) through the int8 encoder's ACTIVATION quantization
/// (symmetric u8: per-row `amax/127`, `round().clamp(-127,127)`, dequant) — the EXACT scheme
/// [`matmul_bias_i7`] applies to activations. `block == None` ⇒ per-row scale (== the current
/// int8 path); `block == Some(b)` ⇒ per-`b`-channel-block amax along the contraction dim.
/// Feasibility harness ONLY: running the EXISTING f32 GEMM on activation-roundtripped inputs
/// isolates the ACTIVATION-quant effect on the transcript (last cycle 03b55db proved the int8
/// encoder's proper-noun error is activation- not weight-quant; this tests whether block-wise
/// activation granularity recovers it, or whether it is the u8 8-bit precision itself).
pub fn u8_act_roundtrip(x: &Mat, block: Option<usize>) -> Mat {
    let m = x.rows;
    let inp = x.cols;
    let b = block.unwrap_or(inp).max(1);
    let mut data = vec![0.0f32; m * inp];
    data.par_chunks_mut(inp).enumerate().for_each(|(r, drow)| {
        let xr = &x.data[r * inp..(r + 1) * inp];
        let mut c0 = 0;
        while c0 < inp {
            let c1 = (c0 + b).min(inp);
            let amax = xr[c0..c1]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-9);
            let scale = amax / 127.0;
            let inv = 1.0 / scale;
            for c in c0..c1 {
                let q = (xr[c] * inv).round().clamp(-127.0, 127.0);
                drow[c] = q * scale;
            }
            c0 = c1;
        }
    });
    Mat::from_vec(m, inp, data)
}

/// maddubs `u8·i7` dot `Σ a[x]·w[x]`. Non-saturating for i7 weights (see [`I7Mat`]).
/// Base target-features include AVX2 (`target-cpu=x86-64-v3`), so this inlines
/// without a `#[target_feature]` attribute (same pattern as [`dot_f16c`]).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
#[inline]
fn dot_maddubs_i7(a: &[u8], w: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let k = a.len();
    let ap = a.as_ptr();
    let wp = w.as_ptr();
    // SAFETY: AVX2 is in the base target features; pointers stay in-bounds (128-
    // then 32-lane steps guarded by `x + N <= k`, scalar tail after).
    unsafe {
        let ones = _mm256_set1_epi16(1);
        // FOUR independent accumulators unroll the maddubs+madd by 128 elements so
        // the `add_epi32` latency is hidden (the single-accumulator form was a
        // serial dependency chain = latency-bound, ~25% of the maddubs compute
        // ceiling). Integer add is associative + commutative, so each lane sums the
        // SAME set of i32 products => the result is BIT-IDENTICAL to the 1-acc order
        // (the int8 transcript is unchanged; this is a pure speed lever).
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut x = 0;
        let dot32 = |o: usize| {
            _mm256_madd_epi16(
                _mm256_maddubs_epi16(
                    _mm256_loadu_si256(ap.add(o) as *const __m256i),
                    _mm256_loadu_si256(wp.add(o) as *const __m256i),
                ),
                ones,
            )
        };
        while x + 128 <= k {
            a0 = _mm256_add_epi32(a0, dot32(x));
            a1 = _mm256_add_epi32(a1, dot32(x + 32));
            a2 = _mm256_add_epi32(a2, dot32(x + 64));
            a3 = _mm256_add_epi32(a3, dot32(x + 96));
            x += 128;
        }
        let mut acc = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
        while x + 32 <= k {
            acc = _mm256_add_epi32(acc, dot32(x));
            x += 32;
        }
        let mut t = [0i32; 8];
        _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, acc);
        let mut s: i32 = t.iter().sum();
        while x < k {
            s += (a[x] as i32) * (w[x] as i32);
            x += 1;
        }
        s
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot_maddubs_i7(a: &[u8], w: &[i8]) -> i32 {
    a.iter()
        .zip(w)
        .map(|(&x, &y)| (x as i32) * (y as i32))
        .sum()
}

/// M4-register-blocked `u8·i7` dot: FOUR activation rows against ONE weight row,
/// the weight vector loaded ONCE per 32-lane chunk (the naive m-outer loop re-reads
/// the whole weight matrix once PER activation row = m× L3 traffic; M4 cuts weight
/// re-reads 4×). Each lane's result is BIT-IDENTICAL to [`dot_maddubs_i7`] (same set
/// of i32 products, same accumulation) so the int8 transcript is UNCHANGED — this is
/// a pure weight-bandwidth lever. 4 independent maddubs→madd→add chains hide the op
/// latency (same ILP as the 1-row 4-accumulator form). Measured ~1.11–1.31× over M1
/// on the encoder linear GEMMs (`encoder_maddubs_i7_gemm_probe`, `m4_diff=0`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
#[inline]
fn dot_maddubs_i7_m4(a0: &[u8], a1: &[u8], a2: &[u8], a3: &[u8], w: &[i8]) -> [i32; 4] {
    use std::arch::x86_64::*;
    let k = w.len();
    let (p0, p1, p2, p3, pw) = (
        a0.as_ptr(),
        a1.as_ptr(),
        a2.as_ptr(),
        a3.as_ptr(),
        w.as_ptr(),
    );
    // SAFETY: AVX2 in the base target features; all four activation slices and the
    // weight slice have length k (caller invariant); 32-lane steps guarded by
    // `x + 32 <= k`, scalar tail after.
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut c0 = _mm256_setzero_si256();
        let mut c1 = _mm256_setzero_si256();
        let mut c2 = _mm256_setzero_si256();
        let mut c3 = _mm256_setzero_si256();
        let mut x = 0;
        while x + 32 <= k {
            let wv = _mm256_loadu_si256(pw.add(x) as *const __m256i);
            c0 = _mm256_add_epi32(
                c0,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_loadu_si256(p0.add(x) as *const __m256i), wv),
                    ones,
                ),
            );
            c1 = _mm256_add_epi32(
                c1,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_loadu_si256(p1.add(x) as *const __m256i), wv),
                    ones,
                ),
            );
            c2 = _mm256_add_epi32(
                c2,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_loadu_si256(p2.add(x) as *const __m256i), wv),
                    ones,
                ),
            );
            c3 = _mm256_add_epi32(
                c3,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_loadu_si256(p3.add(x) as *const __m256i), wv),
                    ones,
                ),
            );
            x += 32;
        }
        let hsum = |acc: __m256i| -> i32 {
            let mut t = [0i32; 8];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, acc);
            t.iter().sum()
        };
        let mut r = [hsum(c0), hsum(c1), hsum(c2), hsum(c3)];
        while x < k {
            let wx = w[x] as i32;
            r[0] += (a0[x] as i32) * wx;
            r[1] += (a1[x] as i32) * wx;
            r[2] += (a2[x] as i32) * wx;
            r[3] += (a3[x] as i32) * wx;
            x += 1;
        }
        r
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot_maddubs_i7_m4(a0: &[u8], a1: &[u8], a2: &[u8], a3: &[u8], w: &[i8]) -> [i32; 4] {
    [
        dot_maddubs_i7(a0, w),
        dot_maddubs_i7(a1, w),
        dot_maddubs_i7(a2, w),
        dot_maddubs_i7(a3, w),
    ]
}

/// M4×N2 2D register tile: 4 activation rows × 2 weight rows = 8 dots per pass.
/// The L1-hot activation is reused across both weight rows (and each weight across
/// all 4 activation rows), improving the maddubs/load ratio once M4 has removed the
/// weight-L3 bottleneck. Returns `[w0: r0..r3, w1: r0..r3]`; each lane is
/// BIT-IDENTICAL to [`dot_maddubs_i7`]. 8 accumulators (fits Zen3's 16 ymm).
/// Measured ~1.3–1.5× over M4 on the non-expanding shapes (`out ≤ in`: attn
/// projections + mlp fc2); on the wide fc1 (`out ≫ in`) register pressure makes it a
/// wash/slight-loss, so the caller gates on `out ≤ in`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
#[inline]
fn dot_maddubs_i7_m4n2(
    a0: &[u8],
    a1: &[u8],
    a2: &[u8],
    a3: &[u8],
    w0: &[i8],
    w1: &[i8],
) -> [i32; 8] {
    use std::arch::x86_64::*;
    let k = w0.len();
    let (p0, p1, p2, p3) = (a0.as_ptr(), a1.as_ptr(), a2.as_ptr(), a3.as_ptr());
    let (q0, q1) = (w0.as_ptr(), w1.as_ptr());
    // SAFETY: AVX2 in base target features; all four activation slices and both
    // weight slices have length k; 32-lane steps guarded by `x + 32 <= k`, scalar
    // tail after.
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_si256(); 8];
        let mut x = 0;
        while x + 32 <= k {
            let wv0 = _mm256_loadu_si256(q0.add(x) as *const __m256i);
            let wv1 = _mm256_loadu_si256(q1.add(x) as *const __m256i);
            let av0 = _mm256_loadu_si256(p0.add(x) as *const __m256i);
            let av1 = _mm256_loadu_si256(p1.add(x) as *const __m256i);
            let av2 = _mm256_loadu_si256(p2.add(x) as *const __m256i);
            let av3 = _mm256_loadu_si256(p3.add(x) as *const __m256i);
            acc[0] = _mm256_add_epi32(
                acc[0],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av0, wv0), ones),
            );
            acc[1] = _mm256_add_epi32(
                acc[1],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av1, wv0), ones),
            );
            acc[2] = _mm256_add_epi32(
                acc[2],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av2, wv0), ones),
            );
            acc[3] = _mm256_add_epi32(
                acc[3],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av3, wv0), ones),
            );
            acc[4] = _mm256_add_epi32(
                acc[4],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av0, wv1), ones),
            );
            acc[5] = _mm256_add_epi32(
                acc[5],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av1, wv1), ones),
            );
            acc[6] = _mm256_add_epi32(
                acc[6],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av2, wv1), ones),
            );
            acc[7] = _mm256_add_epi32(
                acc[7],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av3, wv1), ones),
            );
            x += 32;
        }
        let hsum = |v: __m256i| -> i32 {
            let mut t = [0i32; 8];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, v);
            t.iter().sum()
        };
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = hsum(acc[i]);
        }
        while x < k {
            let (wx0, wx1) = (w0[x] as i32, w1[x] as i32);
            r[0] += (a0[x] as i32) * wx0;
            r[1] += (a1[x] as i32) * wx0;
            r[2] += (a2[x] as i32) * wx0;
            r[3] += (a3[x] as i32) * wx0;
            r[4] += (a0[x] as i32) * wx1;
            r[5] += (a1[x] as i32) * wx1;
            r[6] += (a2[x] as i32) * wx1;
            r[7] += (a3[x] as i32) * wx1;
            x += 1;
        }
        r
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot_maddubs_i7_m4n2(
    a0: &[u8],
    a1: &[u8],
    a2: &[u8],
    a3: &[u8],
    w0: &[i8],
    w1: &[i8],
) -> [i32; 8] {
    [
        dot_maddubs_i7(a0, w0),
        dot_maddubs_i7(a1, w0),
        dot_maddubs_i7(a2, w0),
        dot_maddubs_i7(a3, w0),
        dot_maddubs_i7(a0, w1),
        dot_maddubs_i7(a1, w1),
        dot_maddubs_i7(a2, w1),
        dot_maddubs_i7(a3, w1),
    ]
}

/// M2xN4 register tile: 2 activation rows x 4 weight rows = 8 dots per pass.
/// This is the dual of M4xN2: same dot count and exact integer arithmetic, but
/// it spends registers on four neighboring outputs so each activation vector is
/// reused across four weight rows. Returns `[w0:r0,r1, w1:r0,r1, ... w3:r0,r1]`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
#[inline]
fn dot_maddubs_i7_m2n4(
    a0: &[u8],
    a1: &[u8],
    w0: &[i8],
    w1: &[i8],
    w2: &[i8],
    w3: &[i8],
) -> [i32; 8] {
    use std::arch::x86_64::*;
    let k = w0.len();
    let (p0, p1) = (a0.as_ptr(), a1.as_ptr());
    let (pw0, pw1, pw2, pw3) = (w0.as_ptr(), w1.as_ptr(), w2.as_ptr(), w3.as_ptr());
    // SAFETY: AVX2 is in the base target features; all activation and weight slices
    // have length k (caller invariant); 32-lane steps are guarded by `x + 32 <= k`.
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_si256(); 8];
        let mut x = 0;
        while x + 32 <= k {
            let av0 = _mm256_loadu_si256(p0.add(x) as *const __m256i);
            let av1 = _mm256_loadu_si256(p1.add(x) as *const __m256i);
            let wv0 = _mm256_loadu_si256(pw0.add(x) as *const __m256i);
            let wv1 = _mm256_loadu_si256(pw1.add(x) as *const __m256i);
            let wv2 = _mm256_loadu_si256(pw2.add(x) as *const __m256i);
            let wv3 = _mm256_loadu_si256(pw3.add(x) as *const __m256i);
            acc[0] = _mm256_add_epi32(
                acc[0],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av0, wv0), ones),
            );
            acc[1] = _mm256_add_epi32(
                acc[1],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av1, wv0), ones),
            );
            acc[2] = _mm256_add_epi32(
                acc[2],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av0, wv1), ones),
            );
            acc[3] = _mm256_add_epi32(
                acc[3],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av1, wv1), ones),
            );
            acc[4] = _mm256_add_epi32(
                acc[4],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av0, wv2), ones),
            );
            acc[5] = _mm256_add_epi32(
                acc[5],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av1, wv2), ones),
            );
            acc[6] = _mm256_add_epi32(
                acc[6],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av0, wv3), ones),
            );
            acc[7] = _mm256_add_epi32(
                acc[7],
                _mm256_madd_epi16(_mm256_maddubs_epi16(av1, wv3), ones),
            );
            x += 32;
        }
        let hsum = |v: __m256i| -> i32 {
            let mut tmp = [0i32; 8];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v);
            tmp.iter().sum()
        };
        let mut r = acc.map(hsum);
        while x < k {
            let a0x = a0[x] as i32;
            let a1x = a1[x] as i32;
            let w0x = w0[x] as i32;
            let w1x = w1[x] as i32;
            let w2x = w2[x] as i32;
            let w3x = w3[x] as i32;
            r[0] += a0x * w0x;
            r[1] += a1x * w0x;
            r[2] += a0x * w1x;
            r[3] += a1x * w1x;
            r[4] += a0x * w2x;
            r[5] += a1x * w2x;
            r[6] += a0x * w3x;
            r[7] += a1x * w3x;
            x += 1;
        }
        r
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot_maddubs_i7_m2n4(
    a0: &[u8],
    a1: &[u8],
    w0: &[i8],
    w1: &[i8],
    w2: &[i8],
    w3: &[i8],
) -> [i32; 8] {
    [
        dot_maddubs_i7(a0, w0),
        dot_maddubs_i7(a1, w0),
        dot_maddubs_i7(a0, w1),
        dot_maddubs_i7(a1, w1),
        dot_maddubs_i7(a0, w2),
        dot_maddubs_i7(a1, w2),
        dot_maddubs_i7(a0, w3),
        dot_maddubs_i7(a1, w3),
    ]
}

/// Quantized activation buffer for [`matmul_bias_i7_quantized`].
///
/// Rows are symmetric-quantized to u8 (`amax/127`, then `+128` offset) with a
/// separate f32 scale per row. Q/K/V encoder projections share the same input,
/// so reusing this buffer avoids two duplicate quantize passes without changing
/// the maddubs dot inputs.
#[derive(Debug, Clone)]
pub struct I7Activation {
    rows: usize,
    inp: usize,
    data: Vec<u8>,
    scale: Vec<f32>,
}

/// Quantize `x` for the maddubs 7-bit int8 GEMM.
#[must_use]
pub fn quantize_act_i7(x: &Mat) -> I7Activation {
    let mut data = vec![0u8; x.rows * x.cols];
    let mut scale = vec![0.0f32; x.rows];
    data.par_chunks_mut(x.cols)
        .zip(scale.par_iter_mut())
        .enumerate()
        .for_each(|(r, (xr_u8, s))| {
            let xr = &x.data[r * x.cols..(r + 1) * x.cols];
            let amax = xr.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
            let row_scale = amax / 127.0;
            *s = row_scale;
            let inv = 1.0 / row_scale;
            quantize_row_i7_u8_into(xr, inv, xr_u8);
        });
    I7Activation {
        rows: x.rows,
        inp: x.cols,
        data,
        scale,
    }
}

/// Affine projection from a pre-quantized activation via the maddubs 7-bit int8 GEMM.
///
/// `x` is a [`I7Activation`] built from `[m, in]` f32; `w` is an [`I7Mat`]
/// (`[out, in]`). Each output is `dot_maddubs - 128*sum(w)`, dequantized by
/// `x.scale[row] * w.scale[o]` (+bias). **NON-byte-exact** vs [`matmul_bias`]
/// because the activation and weight are quantized, but byte-identical to
/// [`matmul_bias_i7`] for the same source activation.
///
/// # Errors
/// [`FwError::InvalidRequest`] on `x.inp != w.inp` or `bias.len() != w.out`.
pub fn matmul_bias_i7_quantized(
    x: &I7Activation,
    w: &I7Mat,
    bias: Option<&[f32]>,
) -> FwResult<Mat> {
    if bias.is_some() {
        matmul_bias_i7_quantized_impl::<true, true>(x, w, bias)
    } else {
        matmul_bias_i7_quantized_impl::<true, false>(x, w, bias)
    }
}

/// Const-specialized implementation behind a non-inlined boundary for the
/// same-ELF bias-specialization A/B. Keeping both A/B entry points non-inlined
/// prevents thin LTO from specializing the historical runtime `Option` at the
/// benchmark call site while preserving the production entry point above.
#[doc(hidden)]
#[inline(never)]
pub fn matmul_bias_i7_quantized_specialized_ab(
    x: &I7Activation,
    w: &I7Mat,
    bias: Option<&[f32]>,
) -> FwResult<Mat> {
    if bias.is_some() {
        matmul_bias_i7_quantized_impl::<true, true>(x, w, bias)
    } else {
        matmul_bias_i7_quantized_impl::<true, false>(x, w, bias)
    }
}

/// Historical runtime-`Option` implementation retained for the same-ELF
/// bias-specialization A/B. Production callers use
/// [`matmul_bias_i7_quantized`]; this entry point exists only so the reverted
/// candidate can be measured against its exact former branch shape without a
/// cross-binary comparison.
#[doc(hidden)]
#[inline(never)]
pub fn matmul_bias_i7_quantized_unspecialized(
    x: &I7Activation,
    w: &I7Mat,
    bias: Option<&[f32]>,
) -> FwResult<Mat> {
    matmul_bias_i7_quantized_impl::<false, false>(x, w, bias)
}

#[inline(always)]
fn i7_projection_bias<const SPECIALIZED: bool, const HAS_BIAS: bool>(
    runtime_bias: Option<&[f32]>,
    specialized_bias: &[f32],
    output: usize,
) -> Option<f32> {
    if SPECIALIZED {
        if HAS_BIAS {
            Some(specialized_bias[output])
        } else {
            None
        }
    } else {
        runtime_bias.map(|bias| bias[output])
    }
}

fn matmul_bias_i7_quantized_impl<const SPECIALIZED: bool, const HAS_BIAS: bool>(
    x: &I7Activation,
    w: &I7Mat,
    bias: Option<&[f32]>,
) -> FwResult<Mat> {
    let m = x.rows;
    let inp = x.inp;
    if inp != w.inp {
        return Err(FwError::InvalidRequest(format!(
            "matmul_bias_i7_quantized: x.cols {inp} != w.inp {}",
            w.inp
        )));
    }
    if let Some(b) = bias
        && b.len() != w.out
    {
        return Err(FwError::InvalidRequest(format!(
            "matmul_bias_i7_quantized: bias len {} != out {}",
            b.len(),
            w.out
        )));
    }
    debug_assert!(!SPECIALIZED || HAS_BIAS == bias.is_some());
    let specialized_bias = if SPECIALIZED && HAS_BIAS {
        bias.expect("specialized bias arm requires a validated bias slice")
    } else {
        &[]
    };
    let out = w.out;
    let xu = &x.data;
    let sa = &x.scale;
    // M4-register-blocked: process activation rows in blocks of 4, streaming each
    // weight row ONCE per block (4× less weight L3 traffic than the m-outer naive
    // loop). Bit-identical to the per-row form (same i32 dot + same f32 dequant
    // order), so the int8 transcript is unchanged. Parallel over row-blocks.
    let mut c = gemv_out_buf(m * out);
    c.par_chunks_mut(4 * out)
        .enumerate()
        .for_each(|(blk, cblk)| {
            let r0 = blk * 4;
            let rows = (m - r0).min(4);
            if rows == 4 {
                let x0 = &xu[r0 * inp..(r0 + 1) * inp];
                let x1 = &xu[(r0 + 1) * inp..(r0 + 2) * inp];
                let x2 = &xu[(r0 + 2) * inp..(r0 + 3) * inp];
                let x3 = &xu[(r0 + 3) * inp..(r0 + 4) * inp];
                let (s0, s1, s2, s3) = (sa[r0], sa[r0 + 1], sa[r0 + 2], sa[r0 + 3]);
                if out <= inp {
                    // M4×N2 tile: 2 weight rows per pass. Measured ~1.3–1.5× over plain
                    // M4 on the non-expanding shapes (attn projections `out==in`, mlp
                    // fc2 `out<in`). Gated to `out ≤ in` because the wide fc1
                    // (`out ≫ in`) makes N2 a wash/slight-loss (register pressure), so
                    // this is strictly ≥ M4 on every shape. Bit-identical.
                    let mut o = 0;
                    if i7_m2n4_enabled() {
                        while o + 4 <= out {
                            let w0r = &w.data[o * inp..(o + 1) * inp];
                            let w1r = &w.data[(o + 1) * inp..(o + 2) * inp];
                            let w2r = &w.data[(o + 2) * inp..(o + 3) * inp];
                            let w3r = &w.data[(o + 3) * inp..(o + 4) * inp];
                            let raw01 = dot_maddubs_i7_m2n4(x0, x1, w0r, w1r, w2r, w3r);
                            let raw23 = dot_maddubs_i7_m2n4(x2, x3, w0r, w1r, w2r, w3r);
                            let off0 = 128 * w.colsum[o];
                            let off1 = 128 * w.colsum[o + 1];
                            let off2 = 128 * w.colsum[o + 2];
                            let off3 = 128 * w.colsum[o + 3];
                            let sc0 = w.scale[o];
                            let sc1 = w.scale[o + 1];
                            let sc2 = w.scale[o + 2];
                            let sc3 = w.scale[o + 3];
                            let (bo0, bo1, bo2, bo3) = (
                                i7_projection_bias::<SPECIALIZED, HAS_BIAS>(
                                    bias,
                                    specialized_bias,
                                    o,
                                )
                                .unwrap_or(0.0),
                                i7_projection_bias::<SPECIALIZED, HAS_BIAS>(
                                    bias,
                                    specialized_bias,
                                    o + 1,
                                )
                                .unwrap_or(0.0),
                                i7_projection_bias::<SPECIALIZED, HAS_BIAS>(
                                    bias,
                                    specialized_bias,
                                    o + 2,
                                )
                                .unwrap_or(0.0),
                                i7_projection_bias::<SPECIALIZED, HAS_BIAS>(
                                    bias,
                                    specialized_bias,
                                    o + 3,
                                )
                                .unwrap_or(0.0),
                            );
                            cblk[o] = (raw01[0] - off0) as f32 * s0 * sc0 + bo0;
                            cblk[out + o] = (raw01[1] - off0) as f32 * s1 * sc0 + bo0;
                            cblk[2 * out + o] = (raw23[0] - off0) as f32 * s2 * sc0 + bo0;
                            cblk[3 * out + o] = (raw23[1] - off0) as f32 * s3 * sc0 + bo0;
                            cblk[o + 1] = (raw01[2] - off1) as f32 * s0 * sc1 + bo1;
                            cblk[out + o + 1] = (raw01[3] - off1) as f32 * s1 * sc1 + bo1;
                            cblk[2 * out + o + 1] = (raw23[2] - off1) as f32 * s2 * sc1 + bo1;
                            cblk[3 * out + o + 1] = (raw23[3] - off1) as f32 * s3 * sc1 + bo1;
                            cblk[o + 2] = (raw01[4] - off2) as f32 * s0 * sc2 + bo2;
                            cblk[out + o + 2] = (raw01[5] - off2) as f32 * s1 * sc2 + bo2;
                            cblk[2 * out + o + 2] = (raw23[4] - off2) as f32 * s2 * sc2 + bo2;
                            cblk[3 * out + o + 2] = (raw23[5] - off2) as f32 * s3 * sc2 + bo2;
                            cblk[o + 3] = (raw01[6] - off3) as f32 * s0 * sc3 + bo3;
                            cblk[out + o + 3] = (raw01[7] - off3) as f32 * s1 * sc3 + bo3;
                            cblk[2 * out + o + 3] = (raw23[6] - off3) as f32 * s2 * sc3 + bo3;
                            cblk[3 * out + o + 3] = (raw23[7] - off3) as f32 * s3 * sc3 + bo3;
                            o += 4;
                        }
                    }
                    while o + 2 <= out {
                        let w0r = &w.data[o * inp..(o + 1) * inp];
                        let w1r = &w.data[(o + 1) * inp..(o + 2) * inp];
                        let raw = dot_maddubs_i7_m4n2(x0, x1, x2, x3, w0r, w1r);
                        let off0 = 128 * w.colsum[o];
                        let off1 = 128 * w.colsum[o + 1];
                        let sc0 = w.scale[o];
                        let sc1 = w.scale[o + 1];
                        let mut a0v = (raw[0] - off0) as f32 * s0 * sc0;
                        let mut a1v = (raw[1] - off0) as f32 * s1 * sc0;
                        let mut a2v = (raw[2] - off0) as f32 * s2 * sc0;
                        let mut a3v = (raw[3] - off0) as f32 * s3 * sc0;
                        let mut b0v = (raw[4] - off1) as f32 * s0 * sc1;
                        let mut b1v = (raw[5] - off1) as f32 * s1 * sc1;
                        let mut b2v = (raw[6] - off1) as f32 * s2 * sc1;
                        let mut b3v = (raw[7] - off1) as f32 * s3 * sc1;
                        if let (Some(bo0), Some(bo1)) = (
                            i7_projection_bias::<SPECIALIZED, HAS_BIAS>(bias, specialized_bias, o),
                            i7_projection_bias::<SPECIALIZED, HAS_BIAS>(
                                bias,
                                specialized_bias,
                                o + 1,
                            ),
                        ) {
                            a0v += bo0;
                            a1v += bo0;
                            a2v += bo0;
                            a3v += bo0;
                            b0v += bo1;
                            b1v += bo1;
                            b2v += bo1;
                            b3v += bo1;
                        }
                        cblk[o] = a0v;
                        cblk[out + o] = a1v;
                        cblk[2 * out + o] = a2v;
                        cblk[3 * out + o] = a3v;
                        cblk[o + 1] = b0v;
                        cblk[out + o + 1] = b1v;
                        cblk[2 * out + o + 1] = b2v;
                        cblk[3 * out + o + 1] = b3v;
                        o += 2;
                    }
                    while o < out {
                        let wrow = &w.data[o * inp..(o + 1) * inp];
                        let raw = dot_maddubs_i7_m4(x0, x1, x2, x3, wrow);
                        let off = 128 * w.colsum[o];
                        let sc = w.scale[o];
                        let mut v0 = (raw[0] - off) as f32 * s0 * sc;
                        let mut v1 = (raw[1] - off) as f32 * s1 * sc;
                        let mut v2 = (raw[2] - off) as f32 * s2 * sc;
                        let mut v3 = (raw[3] - off) as f32 * s3 * sc;
                        if let Some(bo) =
                            i7_projection_bias::<SPECIALIZED, HAS_BIAS>(bias, specialized_bias, o)
                        {
                            v0 += bo;
                            v1 += bo;
                            v2 += bo;
                            v3 += bo;
                        }
                        cblk[o] = v0;
                        cblk[out + o] = v1;
                        cblk[2 * out + o] = v2;
                        cblk[3 * out + o] = v3;
                        o += 1;
                    }
                } else {
                    for o in 0..out {
                        let wrow = &w.data[o * inp..(o + 1) * inp];
                        let raw = dot_maddubs_i7_m4(x0, x1, x2, x3, wrow);
                        let off = 128 * w.colsum[o];
                        let sc = w.scale[o];
                        let mut v0 = (raw[0] - off) as f32 * s0 * sc;
                        let mut v1 = (raw[1] - off) as f32 * s1 * sc;
                        let mut v2 = (raw[2] - off) as f32 * s2 * sc;
                        let mut v3 = (raw[3] - off) as f32 * s3 * sc;
                        if let Some(bo) =
                            i7_projection_bias::<SPECIALIZED, HAS_BIAS>(bias, specialized_bias, o)
                        {
                            v0 += bo;
                            v1 += bo;
                            v2 += bo;
                            v3 += bo;
                        }
                        cblk[o] = v0;
                        cblk[out + o] = v1;
                        cblk[2 * out + o] = v2;
                        cblk[3 * out + o] = v3;
                    }
                }
            } else {
                for j in 0..rows {
                    let r = r0 + j;
                    let xr = &xu[r * inp..(r + 1) * inp];
                    let sar = sa[r];
                    for o in 0..out {
                        let wrow = &w.data[o * inp..(o + 1) * inp];
                        let dot = dot_maddubs_i7(xr, wrow) - 128 * w.colsum[o];
                        let mut val = dot as f32 * sar * w.scale[o];
                        if let Some(bo) =
                            i7_projection_bias::<SPECIALIZED, HAS_BIAS>(bias, specialized_bias, o)
                        {
                            val += bo;
                        }
                        cblk[j * out + o] = val;
                    }
                }
            }
        });
    Ok(Mat::from_vec(m, out, c))
}

/// Affine projection `x @ w^T (+ bias)` via the maddubs 7-bit int8 GEMM.
///
/// `x` is `[m, in]` f32; `w` is an [`I7Mat`] (`[out, in]`). The activation is
/// symmetric-quantized to u8 per row (`amax/127`, then `+128` offset), and each
/// output is `dot_maddubs - 128*sum(w)`, dequantized by `sa_row * scale[o]`
/// (+bias). **NON-byte-exact** vs [`matmul_bias`] (int8 quantization).
///
/// # Errors
/// [`FwError::InvalidRequest`] on `x.cols != w.inp` or `bias.len() != w.out`.
pub fn matmul_bias_i7(x: &Mat, w: &I7Mat, bias: Option<&[f32]>) -> FwResult<Mat> {
    let xq = quantize_act_i7(x);
    matmul_bias_i7_quantized(&xq, w, bias)
}

/// GELU-fused activation quantize: quantizes `gelu(x)` to i7 WITHOUT ever
/// materializing the `[rows, in]` GELU'd activation.
///
/// The encoder MLP runs `fc1 → gelu → fc2`; when fc2 is the int8 maddubs path,
/// the classic form writes a full `[1500, 5120]` GELU'd buffer (`nn::gelu`) and
/// then re-reads it here to quantize — a ~30 MiB in-place pass plus a ~30 MiB
/// re-read, both on the (partly DRAM-resident) fc1 output. This folds the GELU
/// into the quant's own read: each row is GELU'd into a per-worker scratch (via
/// the SAME [`gelu_slice`] — pure elementwise, so byte-identical to the flat
/// `nn::gelu`), then amax'd + quantized exactly as [`quantize_act_i7`]. The big
/// GELU pass disappears; only the tiny per-row L1 scratch remains. **Byte-identical**
/// to `gelu` + [`quantize_act_i7`] on the same fc1 output (same table+clamp GELU,
/// same per-row scale + round + affine). Gated behind `super::enc_gelu_fused`.
pub fn quantize_act_i7_gelu(x: &Mat) -> I7Activation {
    let cols = x.cols;
    let mut data = vec![0u8; x.rows * cols];
    let mut scale = vec![0.0f32; x.rows];
    // Per-row parallel (disjoint rows ⇒ order-invariant). `for_each_init` gives
    // each worker ONE reusable `cols`-wide GELU scratch (allocated once per
    // thread, not once per row) so the fusion adds no per-row heap churn.
    data.par_chunks_mut(cols)
        .zip(scale.par_iter_mut())
        .enumerate()
        .for_each_init(
            || vec![0.0f32; cols],
            |g, (r, (xr_u8, s))| {
                let xr = &x.data[r * cols..(r + 1) * cols];
                g.copy_from_slice(xr);
                gelu_slice(g); // same GGML_GELU_FP16 table+clamp as nn::gelu
                let amax = g.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
                let row_scale = amax / 127.0;
                *s = row_scale;
                let inv = 1.0 / row_scale;
                quantize_row_i7_u8_into(g, inv, xr_u8);
            },
        );
    I7Activation {
        rows: x.rows,
        inp: cols,
        data,
        scale,
    }
}

/// [`matmul_bias_i7`] with the GELU folded into the activation quantize (see
/// [`quantize_act_i7_gelu`]). `x` is the RAW fc1 output (pre-GELU); the result is
/// `gelu(x) @ w`, byte-identical to `matmul_bias_i7(&{let mut h=x.clone(); gelu(&mut h); h}, w, bias)`.
pub fn matmul_bias_i7_gelu(x: &Mat, w: &I7Mat, bias: Option<&[f32]>) -> FwResult<Mat> {
    let xq = quantize_act_i7_gelu(x);
    matmul_bias_i7_quantized(&xq, w, bias)
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
            a1 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128(p0.add(i + 8).cast())),
                xb,
                a1,
            );
            a2 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128(p0.add(i + 16).cast())),
                xc,
                a2,
            );
            a3 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128(p0.add(i + 24).cast())),
                xd,
                a3,
            );
            b0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128(p1.add(i).cast())), xa, b0);
            b1 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128(p1.add(i + 8).cast())),
                xb,
                b1,
            );
            b2 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128(p1.add(i + 16).cast())),
                xc,
                b2,
            );
            b3 = _mm256_fmadd_ps(
                _mm256_cvtph_ps(_mm_loadu_si128(p1.add(i + 24).cast())),
                xd,
                b3,
            );
            i += 32;
        }
        let acc0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let acc1 = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
        let mut t = [0.0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), acc0);
        let mut s0 = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
        _mm256_storeu_ps(t.as_mut_ptr(), acc1);
        let mut s1 = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
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

/// Two GEMV row dots over **one** f16 weight row against **two** activation rows
/// `x0`/`x1` — the ACTIVATION-column transpose of [`dot_f16c_2row`]. The four
/// f16→f32 weight-chunk conversions (`vcvtph2ps`) are done ONCE per 32-lane chunk
/// and REUSED across both activation rows; each row keeps its OWN four accumulators
/// reduced in the byte-identical order of [`dot_f16c`], so
/// `dot_f16c_2col(w, x0, x1) == (dot_f16c(w, x0), dot_f16c(w, x1))` **bit-for-bit**.
///
/// The win is on the batched (tq>1) row-morsel GEMV, where the weight is streamed
/// once per activation row: pairing rows halves the weight conversion (the
/// cvtph-throughput bottleneck at [1280,1280]) — MEASURED 1.26× cold on the
/// per-window cross-K/V projection shape, itself FASTER than a dequant-to-f32 ft
/// sgemm (which reads 2× the weight bytes and is non-byte-exact); see
/// `examples/f16batch_m2col_probe`. Register budget: 4 converted-weight + 4+4
/// accumulators + transient x = 16 ymm (fits Zen3).
///
/// Compiled only under `f16c`+`fma` (same cfg/inlining contract as [`dot_f16c`]).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "f16c",
    target_feature = "fma"
))]
#[inline]
#[allow(unsafe_code)]
fn dot_f16c_2col(w: &[Float16], x0: &[f32], x1: &[f32]) -> (f32, f32) {
    use core::arch::x86_64::*;
    let n = w.len().min(x0.len()).min(x1.len());
    let (p0, p1) = (x0.as_ptr(), x1.as_ptr());
    // SAFETY: every load is in-bounds by the i+32 / i+8 / i<n guards over
    // n = min(w.len, x0.len, x1.len); Float16 is repr(transparent) u16 so a 128-bit
    // load reads 8 contiguous lanes; f16c/fma are guaranteed by this fn's cfg.
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
            // Convert the four weight chunks ONCE, reuse across both activation rows.
            let wc0 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i).cast()));
            let wc1 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i + 8).cast()));
            let wc2 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i + 16).cast()));
            let wc3 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i + 24).cast()));
            a0 = _mm256_fmadd_ps(wc0, _mm256_loadu_ps(p0.add(i)), a0);
            a1 = _mm256_fmadd_ps(wc1, _mm256_loadu_ps(p0.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(wc2, _mm256_loadu_ps(p0.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(wc3, _mm256_loadu_ps(p0.add(i + 24)), a3);
            b0 = _mm256_fmadd_ps(wc0, _mm256_loadu_ps(p1.add(i)), b0);
            b1 = _mm256_fmadd_ps(wc1, _mm256_loadu_ps(p1.add(i + 8)), b1);
            b2 = _mm256_fmadd_ps(wc2, _mm256_loadu_ps(p1.add(i + 16)), b2);
            b3 = _mm256_fmadd_ps(wc3, _mm256_loadu_ps(p1.add(i + 24)), b3);
            i += 32;
        }
        let acca = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let accb = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
        let mut ta = [0.0f32; 8];
        let mut tb = [0.0f32; 8];
        _mm256_storeu_ps(ta.as_mut_ptr(), acca);
        _mm256_storeu_ps(tb.as_mut_ptr(), accb);
        let mut s0 = ((ta[0] + ta[1]) + (ta[2] + ta[3])) + ((ta[4] + ta[5]) + (ta[6] + ta[7]));
        let mut s1 = ((tb[0] + tb[1]) + (tb[2] + tb[3])) + ((tb[4] + tb[5]) + (tb[6] + tb[7]));
        while i + 8 <= n {
            let wc = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i).cast()));
            let pa = _mm256_mul_ps(wc, _mm256_loadu_ps(p0.add(i)));
            let pb = _mm256_mul_ps(wc, _mm256_loadu_ps(p1.add(i)));
            _mm256_storeu_ps(ta.as_mut_ptr(), pa);
            _mm256_storeu_ps(tb.as_mut_ptr(), pb);
            s0 += ((ta[0] + ta[1]) + (ta[2] + ta[3])) + ((ta[4] + ta[5]) + (ta[6] + ta[7]));
            s1 += ((tb[0] + tb[1]) + (tb[2] + tb[3])) + ((tb[4] + tb[5]) + (tb[6] + tb[7]));
            i += 8;
        }
        while i < n {
            let wv = w[i].to_f32();
            s0 += wv * x0[i];
            s1 += wv * x1[i];
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

/// Two GEMV row dots of ONE f16 weight row against TWO activation rows, sharing the
/// weight's f16→f32 conversion — the batched twin of [`dequant_row_dot`]. Fused f16c
/// path ([`dot_f16c_2col`]) when available; else the portable two-pass converts the
/// weight ONCE into `scratch` then [`dot8`]s each activation. BYTE-IDENTICAL to two
/// separate [`dequant_row_dot`] calls in BOTH paths (same per-row reduction; the
/// convert is row-independent so doing it once changes nothing).
#[inline]
fn dequant_row_dot_2col(
    w_row: &[Float16],
    x0: &[f32],
    x1: &[f32],
    scratch: &mut [f32],
    use_fused: bool,
) -> (f32, f32) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "fma"
    ))]
    if use_fused {
        return dot_f16c_2col(w_row, x0, x1);
    }
    let _ = use_fused;
    w_row.convert_to_f32_slice(scratch);
    (dot8(scratch, x0), dot8(scratch, x1))
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

/// AVX2+F16C symmetric int8 quantize of ONE f16 weight row into `out`, returning
/// the per-row `amax/127` scale. Two vectorized passes via `_mm256_cvtph_ps`
/// (`vcvtph2ps` — EXACT f16→f32, bit-identical to `Float16::to_f32` for finite
/// weights, same conversion [`dot_f16c`] relies on): (1) `amax = max_i |w[i]|`,
/// (2) `out[i] = (w[i]·127/amax).round().clamp(-127,127) as i8` with round-HALF-
/// AWAY (`trunc(v+copysign(0.5,v))` = `f32::round`) then saturating-pack to i8.
/// **Byte-identical** to the scalar non-EF loop below (same exact f16→f32, amax is
/// order-invariant for finite values, same round+clamp) but avoids the software
/// f16→f32 AND the scalarized `f32::round` — neither of which LLVM autovectorizes
/// (`project_round_doesnt_vectorize`). Mirrors [`quantize_act_i8_into`]'s pack.
/// See `quantize_f16_row_to_i8_matches_scalar`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "f16c"
))]
#[allow(unsafe_code)]
fn quantize_f16_row_to_i8_into(w: &[Float16], out: &mut [i8]) -> f32 {
    use core::arch::x86_64::*;
    debug_assert_eq!(
        w.len(),
        out.len(),
        "quantize_f16_row_to_i8_into len mismatch"
    );
    let n = w.len();
    let wp = w.as_ptr();
    // SAFETY: avx2+f16c guaranteed by cfg; every 128-bit f16 load is bounded by the
    // `i+8<=n` / `j+8<=n` guard, the `<8` remainders run scalar. Float16 is
    // repr(transparent) over u16 so a 128-bit load reads 8 contiguous lanes.
    unsafe {
        // pass 1: amax = max over |f16→f32(w[i])| (tree-reduce max == scalar fold
        // for finite values; abs = clear sign bit, matching f32::abs).
        let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let mut m = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_cvtph_ps(_mm_loadu_si128(wp.add(i).cast()));
            m = _mm256_max_ps(m, _mm256_and_ps(v, absmask));
            i += 8;
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), m);
        let mut amax = tmp.iter().copied().fold(0.0f32, f32::max);
        while i < n {
            amax = amax.max(w.get_unchecked(i).to_f32().abs());
            i += 1;
        }
        let scale = amax.max(1e-9) / 127.0;
        let inv = 1.0 / scale;
        // pass 2: (f16→f32(w)·inv) round-half-away, clamp ±127, pack to i8.
        let vinv = _mm256_set1_ps(inv);
        let half = _mm256_set1_ps(0.5);
        let signmask = _mm256_set1_ps(-0.0);
        let c127 = _mm256_set1_ps(127.0);
        let cm127 = _mm256_set1_ps(-127.0);
        let mut j = 0;
        while j + 8 <= n {
            let v = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(wp.add(j).cast())), vinv);
            let vh = _mm256_add_ps(v, _mm256_or_ps(half, _mm256_and_ps(v, signmask)));
            let r = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(vh);
            let r = _mm256_min_ps(_mm256_max_ps(r, cm127), c127);
            let ri = _mm256_cvtps_epi32(r);
            let lo = _mm256_castsi256_si128(ri);
            let hi = _mm256_extracti128_si256::<1>(ri);
            let i16s = _mm_packs_epi32(lo, hi); // [lo0..3, hi0..3]
            let i8s = _mm_packs_epi16(i16s, i16s); // low 8 bytes = elems 0..7
            _mm_storel_epi64(out.as_mut_ptr().add(j) as *mut __m128i, i8s);
            j += 8;
        }
        while j < n {
            *out.get_unchecked_mut(j) = (w.get_unchecked(j).to_f32() * inv)
                .round()
                .clamp(-127.0, 127.0) as i8;
            j += 1;
        }
        scale
    }
}

/// Scalar fallback (non-avx2/f16c): the exact reference the AVX2 path reproduces.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "f16c"
)))]
fn quantize_f16_row_to_i8_into(w: &[Float16], out: &mut [i8]) -> f32 {
    let amax = w
        .iter()
        .map(|h| h.to_f32().abs())
        .fold(0.0f32, f32::max)
        .max(1e-9);
    let scale = amax / 127.0;
    let inv = 1.0 / scale;
    for (d, h) in out.iter_mut().zip(w) {
        *d = (h.to_f32() * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Per-output-row symmetric int8 quantization of a natural `[out, in]` f16
/// weight: `scale[o] = max_i |w[o,i]| / 127`, `q[o,i] = round(w[o,i]/scale[o])`.
/// Parallel over rows (each independent). The inverse `w ≈ q * scale` is what
/// [`gemv_i8`] reconstructs.
pub fn quantize_f16_to_i8(w: &[Float16], out: usize, inp: usize) -> I8Mat {
    debug_assert_eq!(w.len(), out * inp);
    let ef = crate::native_engine::dec_ef_quant();
    let mut data = vec![0i8; out * inp];
    let mut scales = vec![0.0f32; out];
    data.par_chunks_mut(inp)
        .zip(scales.par_iter_mut())
        .enumerate()
        .for_each(|(o, (drow, s))| {
            let wrow = &w[o * inp..(o + 1) * inp];
            if ef {
                // Error-feedback weight quant (FW_DEC_EF; same scheme as encoder
                // EF-weights): carry each weight's rounding residual forward along the
                // contraction dim so the per-row dot has less accumulated quant bias.
                // Static operand ⇒ stable. Same i8 format/scale ⇒ gemv_i8 kernel unchanged.
                // SERIAL dependency (err chain) ⇒ not vectorizable; kept scalar.
                let amax = wrow
                    .iter()
                    .map(|h| h.to_f32().abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-9);
                let sc = amax / 127.0;
                *s = sc;
                let inv = 1.0 / sc;
                let mut err = 0.0f32;
                for (d, h) in drow.iter_mut().zip(wrow) {
                    let target = h.to_f32() * inv + err;
                    let q = target.round().clamp(-127.0, 127.0);
                    err = target - q; // residual carried forward
                    *d = q as i8;
                }
            } else {
                // Default path: AVX2+F16C amax + round (byte-identical to the scalar
                // non-EF loop). Returns the same `amax/127` scale.
                *s = quantize_f16_row_to_i8_into(wrow, drow);
            }
        });
    I8Mat {
        data,
        scales,
        out,
        inp,
    }
}

/// Signed int8 dot → i32. Explicit AVX2 (`vpmovsxbw`+`vpmaddwd`+`vpaddd`, 2
/// independent i32 accumulators) instead of the scalar loop: LLVM's autovec of the
/// scalar reduction caps at **~28 GB/s** cache-resident (compute-bound, too few
/// accumulators to hide `vpmaddwd` latency), while this hits **~50 GB/s** = **1.8×**
/// cache-resident and **1.13×** on the DRAM-bound `[51866,1280]` logits stream
/// (`examples/dot_i8_probe`). **Bit-identical to the scalar loop**: i8·i8 ∈
/// `[-16129,16129]` and decode contraction `≤5120` ⇒ `|Σ| ≤ 82.6M < 2³¹`, so there
/// is NO i32 overflow and integer add is associative — the vectorized pairwise-i32
/// sum equals the scalar sum exactly (verified 0-diff over the full logits matrix).
/// Feeds every int8 decode GEMV ([`gemv_i8`], [`gemv_i8_batch`]) at half the f16
/// weight bytes.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    debug_assert_eq!(w.len(), x.len(), "dot_i8 length mismatch");
    let n = w.len();
    let (wp, xp) = (w.as_ptr(), x.as_ptr());
    // SAFETY: avx2 guaranteed by this fn's cfg; every 128-bit load is bounded by the
    // `i+16<=n` / `i+32<=n` guard and the `< 16` remainder runs scalar; no overflow
    // (see doc), so the reduction is bit-identical to the scalar path.
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
        // Horizontal sum (exact integer add, order-independent).
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

/// Two i8 dots of ONE weight row against TWO activation rows (`xa`/`xb`), sign-extending
/// the weight (`vpmovsxbw`) ONCE per 16-chunk and reusing it for both — the int8
/// analogue of [`dot_f16c_2col`], for the WEIGHT-OUTER batched GEMV where `w[o]` is
/// re-sign-extended per token. Each token keeps [`dot_i8`]'s exact 2-accumulator layout;
/// i32 sums are integer-exact (order-independent), so `dot_i8_2col(w,xa,xb) ==
/// (dot_i8(w,xa), dot_i8(w,xb))` bit-for-bit. Halves the weight sign-extend (MEASURED
/// 1.15-1.19× at tq=64, `examples/i8batch_2col_probe`, byte-identical).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8_2col(w: &[i8], xa: &[i8], xb: &[i8]) -> (i32, i32) {
    use core::arch::x86_64::*;
    let n = w.len().min(xa.len()).min(xb.len());
    let (wp, ap, bp) = (w.as_ptr(), xa.as_ptr(), xb.as_ptr());
    // SAFETY: avx2 by cfg; every 128-bit load bounded by the i+32<=n / i+16<=n guards
    // over n = min(lens); tail runs scalar. No i32 overflow (see dot_i8 doc).
    unsafe {
        let mut aa0 = _mm256_setzero_si256();
        let mut aa1 = _mm256_setzero_si256();
        let mut ab0 = _mm256_setzero_si256();
        let mut ab1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16) as *const __m128i));
            let xa0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i) as *const __m128i));
            let xa1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i + 16) as *const __m128i));
            let xb0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i) as *const __m128i));
            let xb1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i + 16) as *const __m128i));
            aa0 = _mm256_add_epi32(aa0, _mm256_madd_epi16(w0, xa0));
            aa1 = _mm256_add_epi32(aa1, _mm256_madd_epi16(w1, xa1));
            ab0 = _mm256_add_epi32(ab0, _mm256_madd_epi16(w0, xb0));
            ab1 = _mm256_add_epi32(ab1, _mm256_madd_epi16(w1, xb1));
            i += 32;
        }
        while i + 16 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
            let xa0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i) as *const __m128i));
            let xb0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i) as *const __m128i));
            aa0 = _mm256_add_epi32(aa0, _mm256_madd_epi16(w0, xa0));
            ab0 = _mm256_add_epi32(ab0, _mm256_madd_epi16(w0, xb0));
            i += 16;
        }
        let hsum = |s: __m256i| -> i32 {
            let lo = _mm256_castsi256_si128(s);
            let hi = _mm256_extracti128_si256::<1>(s);
            let q = _mm_add_epi32(lo, hi);
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
            _mm_cvtsi128_si32(q)
        };
        let mut acc_a = hsum(_mm256_add_epi32(aa0, aa1));
        let mut acc_b = hsum(_mm256_add_epi32(ab0, ab1));
        while i < n {
            let wv = *w.get_unchecked(i) as i32;
            acc_a += wv * (*xa.get_unchecked(i) as i32);
            acc_b += wv * (*xb.get_unchecked(i) as i32);
            i += 1;
        }
        (acc_a, acc_b)
    }
}

/// 4-token activation-column tile: one weight row, FOUR activation columns, sharing
/// each weight-row `vpmovsxbw` (`_mm256_cvtepi8_epi16`) across all four tokens
/// (0.25 weight-cvt/token vs `dot_i8_2col`'s 0.5). 8 i32 accumulators (4 cols × 2
/// halves) + 2 weight regs = 10 YMM (fits Zen3's 16, no spill). Each column keeps
/// [`dot_i8`]'s EXACT `madd_epi16` pairing + 2-accumulator reduction, so
/// `dot_i8_4col(w,xa,xb,xc,xd) == (dot_i8(w,xa), dot_i8(w,xb), dot_i8(w,xc),
/// dot_i8(w,xd))` bit-for-bit (i32 sums are integer-exact ⇒ order-independent — a
/// ULP-FREE lever, not merely WER-neutral). Reference impl + measurement:
/// `examples/i8batch_4col_probe` (1.03-1.11× pure-kernel over 2col, byte-id 12/12).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8_4col(w: &[i8], xa: &[i8], xb: &[i8], xc: &[i8], xd: &[i8]) -> (i32, i32, i32, i32) {
    use core::arch::x86_64::*;
    let n = w
        .len()
        .min(xa.len())
        .min(xb.len())
        .min(xc.len())
        .min(xd.len());
    let (wp, ap, bp, cp, dp) = (
        w.as_ptr(),
        xa.as_ptr(),
        xb.as_ptr(),
        xc.as_ptr(),
        xd.as_ptr(),
    );
    // SAFETY: avx2 by cfg; every 128-bit load bounded by the i+32<=n / i+16<=n guards
    // over n = min(lens); tail runs scalar. No i32 overflow (see dot_i8 doc).
    unsafe {
        let mut aa0 = _mm256_setzero_si256();
        let mut aa1 = _mm256_setzero_si256();
        let mut ab0 = _mm256_setzero_si256();
        let mut ab1 = _mm256_setzero_si256();
        let mut ac0 = _mm256_setzero_si256();
        let mut ac1 = _mm256_setzero_si256();
        let mut ad0 = _mm256_setzero_si256();
        let mut ad1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16) as *const __m128i));
            aa0 = _mm256_add_epi32(
                aa0,
                _mm256_madd_epi16(
                    w0,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i) as *const __m128i)),
                ),
            );
            aa1 = _mm256_add_epi32(
                aa1,
                _mm256_madd_epi16(
                    w1,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i + 16) as *const __m128i)),
                ),
            );
            ab0 = _mm256_add_epi32(
                ab0,
                _mm256_madd_epi16(
                    w0,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i) as *const __m128i)),
                ),
            );
            ab1 = _mm256_add_epi32(
                ab1,
                _mm256_madd_epi16(
                    w1,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i + 16) as *const __m128i)),
                ),
            );
            ac0 = _mm256_add_epi32(
                ac0,
                _mm256_madd_epi16(
                    w0,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i) as *const __m128i)),
                ),
            );
            ac1 = _mm256_add_epi32(
                ac1,
                _mm256_madd_epi16(
                    w1,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i + 16) as *const __m128i)),
                ),
            );
            ad0 = _mm256_add_epi32(
                ad0,
                _mm256_madd_epi16(
                    w0,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i) as *const __m128i)),
                ),
            );
            ad1 = _mm256_add_epi32(
                ad1,
                _mm256_madd_epi16(
                    w1,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i + 16) as *const __m128i)),
                ),
            );
            i += 32;
        }
        while i + 16 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
            aa0 = _mm256_add_epi32(
                aa0,
                _mm256_madd_epi16(
                    w0,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i) as *const __m128i)),
                ),
            );
            ab0 = _mm256_add_epi32(
                ab0,
                _mm256_madd_epi16(
                    w0,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i) as *const __m128i)),
                ),
            );
            ac0 = _mm256_add_epi32(
                ac0,
                _mm256_madd_epi16(
                    w0,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i) as *const __m128i)),
                ),
            );
            ad0 = _mm256_add_epi32(
                ad0,
                _mm256_madd_epi16(
                    w0,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i) as *const __m128i)),
                ),
            );
            i += 16;
        }
        let hsum = |s: __m256i| -> i32 {
            let lo = _mm256_castsi256_si128(s);
            let hi = _mm256_extracti128_si256::<1>(s);
            let q = _mm_add_epi32(lo, hi);
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
            _mm_cvtsi128_si32(q)
        };
        let mut acc_a = hsum(_mm256_add_epi32(aa0, aa1));
        let mut acc_b = hsum(_mm256_add_epi32(ab0, ab1));
        let mut acc_c = hsum(_mm256_add_epi32(ac0, ac1));
        let mut acc_d = hsum(_mm256_add_epi32(ad0, ad1));
        while i < n {
            let wv = *w.get_unchecked(i) as i32;
            acc_a += wv * (*xa.get_unchecked(i) as i32);
            acc_b += wv * (*xb.get_unchecked(i) as i32);
            acc_c += wv * (*xc.get_unchecked(i) as i32);
            acc_d += wv * (*xd.get_unchecked(i) as i32);
            i += 1;
        }
        (acc_a, acc_b, acc_c, acc_d)
    }
}

/// Scalar fallback (non-x86 / no avx2): the reference the AVX2 path reproduces exactly.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    for (a, b) in w.iter().zip(x.iter()) {
        acc += (*a as i32) * (*b as i32);
    }
    acc
}

/// Scalar fallback for [`dot_i8_2col`].
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot_i8_2col(w: &[i8], xa: &[i8], xb: &[i8]) -> (i32, i32) {
    (dot_i8(w, xa), dot_i8(w, xb))
}

/// Scalar fallback for [`dot_i8_4col`].
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn dot_i8_4col(w: &[i8], xa: &[i8], xb: &[i8], xc: &[i8], xd: &[i8]) -> (i32, i32, i32, i32) {
    (dot_i8(w, xa), dot_i8(w, xb), dot_i8(w, xc), dot_i8(w, xd))
}

/// Symmetric int8 quantize of an activation into `out` (`len == x.len()`):
/// `out[i] = (x[i]·xinv).round().clamp(-127,127) as i8`. The AVX2 path computes
/// `round()` as `trunc(v + copysign(0.5, v))` (= round-HALF-AWAY = `f32::round`
/// for finite `v` — activations are always finite post-GEMM), then clamps and
/// saturating-packs to i8 — **byte-identical** to the scalar map but **~5× faster**
/// (`f32::round` has no direct AVX rounding mode, so LLVM scalarizes the map;
/// measured `examples/quant_i8_probe`, 0-diff over ±127 clamp edges). Runs ~7×/token
/// in decode.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
fn quantize_act_i8_into(x: &[f32], xinv: f32, out: &mut [i8]) {
    use core::arch::x86_64::*;
    debug_assert_eq!(x.len(), out.len(), "quantize_act_i8_into len mismatch");
    let n = x.len();
    let xp = x.as_ptr();
    // SAFETY: avx2 guaranteed by cfg; every load/store is bounded by the `i+8<=n`
    // guard and the `< 8` remainder runs scalar.
    unsafe {
        let vinv = _mm256_set1_ps(xinv);
        let half = _mm256_set1_ps(0.5);
        let signmask = _mm256_set1_ps(-0.0); // 0x80000000
        let c127 = _mm256_set1_ps(127.0);
        let cm127 = _mm256_set1_ps(-127.0);
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_mul_ps(_mm256_loadu_ps(xp.add(i)), vinv);
            let vh = _mm256_add_ps(v, _mm256_or_ps(half, _mm256_and_ps(v, signmask)));
            let r = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(vh);
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
                (*x.get_unchecked(i) * xinv).round().clamp(-127.0, 127.0) as i8;
            i += 1;
        }
    }
}

/// Scalar fallback (non-avx2): the exact reference the AVX2 path reproduces.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn quantize_act_i8_into(x: &[f32], xinv: f32, out: &mut [i8]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o = (v * xinv).round().clamp(-127.0, 127.0) as i8;
    }
}

/// Symmetric 7-bit activation quantize of one row into the maddubs `u8` layout
/// (`out[i] = ((src[i]·inv).round().clamp(-127,127) as i32 + 128) as u8`) — the
/// shared inner loop of [`quantize_act_i7`] and [`quantize_act_i7_gelu`]. The
/// AVX2 path reproduces `f32::round` as `trunc(v + copysign(0.5, v))` (round-
/// HALF-AWAY; activations are finite post-GEMM/GELU), clamps in f32, then adds
/// 128 in i32 and unsigned-saturating-packs to `u8` — **byte-identical** to the
/// scalar map but avoids LLVM scalarizing the per-element `f32::round` (no direct
/// AVX rounding mode), same win as [`quantize_act_i8_into`] (~5×). This runs once
/// per encoder layer over the `[n_ctx, n_state]` / `[n_ctx, 4·n_state]`
/// activation, 32×/window on turbo. See `quantize_row_i7_u8_matches_scalar`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
fn quantize_row_i7_u8_into(src: &[f32], inv: f32, out: &mut [u8]) {
    use core::arch::x86_64::*;
    debug_assert_eq!(src.len(), out.len(), "quantize_row_i7_u8_into len mismatch");
    let n = src.len();
    let sp = src.as_ptr();
    // SAFETY: avx2 guaranteed by cfg; every load/store is bounded by the `i+8<=n`
    // guard and the `< 8` remainder runs scalar.
    unsafe {
        let vinv = _mm256_set1_ps(inv);
        let half = _mm256_set1_ps(0.5);
        let signmask = _mm256_set1_ps(-0.0); // 0x80000000
        let c127 = _mm256_set1_ps(127.0);
        let cm127 = _mm256_set1_ps(-127.0);
        let c128 = _mm_set1_epi32(128);
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_mul_ps(_mm256_loadu_ps(sp.add(i)), vinv);
            let vh = _mm256_add_ps(v, _mm256_or_ps(half, _mm256_and_ps(v, signmask)));
            let r = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(vh);
            let r = _mm256_min_ps(_mm256_max_ps(r, cm127), c127);
            let ri = _mm256_cvtps_epi32(r); // 8×i32 in [-127,127]
            // +128 in i32 → [1,255], then unsigned-saturating pack i32→u16→u8.
            let lo = _mm_add_epi32(_mm256_castsi256_si128(ri), c128);
            let hi = _mm_add_epi32(_mm256_extracti128_si256::<1>(ri), c128);
            let u16s = _mm_packus_epi32(lo, hi); // [lo0..3, hi0..3] as u16, values [1,255]
            let u8s = _mm_packus_epi16(u16s, u16s); // low 8 bytes = elems 0..7
            _mm_storel_epi64(out.as_mut_ptr().add(i) as *mut __m128i, u8s);
            i += 8;
        }
        while i < n {
            let i8v = (*src.get_unchecked(i) * inv).round().clamp(-127.0, 127.0) as i32;
            *out.get_unchecked_mut(i) = (i8v + 128) as u8;
            i += 1;
        }
    }
}

/// Scalar fallback (non-avx2): the exact reference the AVX2 path reproduces.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn quantize_row_i7_u8_into(src: &[f32], inv: f32, out: &mut [u8]) {
    for (d, &v) in out.iter_mut().zip(src) {
        let i8v = (v * inv).round().clamp(-127.0, 127.0) as i32;
        *d = (i8v + 128) as u8;
    }
}

/// `o[d] += a * x[d]` for every `d`. BYTE-IDENTICAL to the scalar loop
/// `for (o, x) { *o += a * x }`: `*o += a*x` is mul-then-add (TWO IEEE roundings),
/// and the AVX2 path uses SEPARATE `_mm256_mul_ps` + `_mm256_add_ps` — **not**
/// `fmadd`, which would fuse to a single rounding and diverge in the last ULP.
/// The vectorization is across the INDEPENDENT output slots `d`, so for each
/// `o[d]` the accumulation order over successive calls (the decode `j`-ascending
/// score·V SAXPY) is preserved exactly. See `axpy_f32_into_matches_scalar`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
fn axpy_f32_into(o: &mut [f32], a: f32, x: &[f32]) {
    use core::arch::x86_64::*;
    debug_assert_eq!(o.len(), x.len(), "axpy_f32_into len mismatch");
    let n = o.len();
    let xp = x.as_ptr();
    let op = o.as_mut_ptr();
    // SAFETY: avx2 guaranteed by cfg; loads/stores bounded by `i+8<=n`, remainder scalar.
    unsafe {
        let va = _mm256_set1_ps(a);
        let mut i = 0;
        while i + 8 <= n {
            let ov = _mm256_loadu_ps(op.add(i));
            let xv = _mm256_loadu_ps(xp.add(i));
            // mul then add — two roundings, matching the scalar `*o += a*x`.
            let r = _mm256_add_ps(ov, _mm256_mul_ps(va, xv));
            _mm256_storeu_ps(op.add(i), r);
            i += 8;
        }
        while i < n {
            *o.get_unchecked_mut(i) += a * *x.get_unchecked(i);
            i += 1;
        }
    }
}

/// Scalar fallback (non-avx2): the exact reference the AVX2 path reproduces.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn axpy_f32_into(o: &mut [f32], a: f32, x: &[f32]) {
    for (oo, &xx) in o.iter_mut().zip(x) {
        *oo += a * xx;
    }
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
    debug_assert!(
        bias.is_none_or(|b| b.len() == out),
        "gemv_i8 bias length mismatch"
    );
    // Quantize the activation once (per-vector symmetric), shared by all rows.
    let xamax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    let xs = xamax / 127.0;
    let xinv = 1.0 / xs;
    let mut xi8 = vec![0i8; inp];
    quantize_act_i8_into(x, xinv, &mut xi8);

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

/// Batched fused int8 GEMV for the prefill / multi-token path (`tq > 1`) — the
/// int8 analog of [`gemv_f16_batch`]. Each weight row `w[o]` is read ONCE and
/// dotted against ALL `tq` activation rows, so the (bandwidth-bound) weight read
/// is amortized over the batch at HALF the bytes of the f16 batch path. Each
/// activation ROW is quantized per-vector by its OWN amax, exactly as [`gemv_i8`]
/// does per token, and each output row keeps its own weight scale — so every
/// `(t, o)` entry equals `gemv_i8`'s and the result is BIT-IDENTICAL to running
/// the batch as `tq` separate [`gemv_i8`] calls. `out_slice` is `[tq, out]`
/// row-major (same layout as [`gemv_f16_batch`]). Quantifies + captures the
/// draft-decoding amortization (`examples/draft_amortization_probe.rs`).
pub fn gemv_i8_batch(w: &I8Mat, x: &[f32], tq: usize, bias: Option<&[f32]>, out_slice: &mut [f32]) {
    let (out, inp) = (w.out, w.inp);
    debug_assert_eq!(
        w.data.len(),
        out * inp,
        "gemv_i8_batch weight shape mismatch"
    );
    debug_assert_eq!(x.len(), tq * inp, "gemv_i8_batch x length mismatch");
    debug_assert_eq!(
        out_slice.len(),
        tq * out,
        "gemv_i8_batch out length mismatch"
    );
    debug_assert!(
        bias.is_none_or(|b| b.len() == out),
        "gemv_i8_batch bias length mismatch"
    );
    if tq == 1 {
        gemv_i8(w, x, bias, out_slice);
        return;
    }

    // Per-ROW activation quantization (each row by its own amax → identical to the
    // per-token gemv_i8), shared by every output row.
    let mut xi8 = vec![0i8; tq * inp];
    let mut xs = vec![0.0f32; tq];
    for t in 0..tq {
        let row = &x[t * inp..(t + 1) * inp];
        let xamax = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
        let s = xamax / 127.0;
        xs[t] = s;
        let inv = 1.0 / s;
        // Same AVX2 copysign+trunc quantize as gemv_i8 (byte-identical to the scalar
        // `.round()` map for finite activations; ~5× — f32::round doesn't vectorize).
        quantize_act_i8_into(row, inv, &mut xi8[t * inp..(t + 1) * inp]);
    }

    // Disjoint-column-band structure identical to gemv_f16_batch; the only change
    // is the per-(o,t) value = `dot_i8(w[o], xi8[t]) * scale_w[o] * scale_x[t] + b`,
    // in the SAME product order as gemv_i8 (bit-identical).
    // 2-token activation-column tile (`dot_i8_2col`): share each weight row's
    // `vpmovsxbw` across a pair of tokens (byte-identical i32, then the SAME
    // per-(o,t) `* so * xs[t] + b`). Odd tail token + `FW_I8_BATCH_2COL=0` use `dot_i8`.
    let use_2col = i8_batch_2col_enabled();
    let use_4col = i8_batch_4col_enabled();
    let compute_band = |o0: usize, o1: usize, dst: &mut [f32]| {
        for o in o0..o1 {
            let wrow = &w.data[o * inp..(o + 1) * inp];
            let so = w.scales[o];
            let b = bias.map_or(0.0, |bb| bb[o]);
            if use_4col {
                // 4-token tile, then 2col for the ≤3-token remainder, then 1col tail.
                // dot_i8_4col == (dot_i8,dot_i8,dot_i8,dot_i8) bit-for-bit, so this is
                // byte-identical to both the 2col and the plain dot_i8 branches.
                let mut t = 0;
                while t + 4 <= tq {
                    let (da, db, dc, dd) = dot_i8_4col(
                        wrow,
                        &xi8[t * inp..(t + 1) * inp],
                        &xi8[(t + 1) * inp..(t + 2) * inp],
                        &xi8[(t + 2) * inp..(t + 3) * inp],
                        &xi8[(t + 3) * inp..(t + 4) * inp],
                    );
                    dst[t * out + o] = da as f32 * so * xs[t] + b;
                    dst[(t + 1) * out + o] = db as f32 * so * xs[t + 1] + b;
                    dst[(t + 2) * out + o] = dc as f32 * so * xs[t + 2] + b;
                    dst[(t + 3) * out + o] = dd as f32 * so * xs[t + 3] + b;
                    t += 4;
                }
                while t + 2 <= tq {
                    let (da, db) = dot_i8_2col(
                        wrow,
                        &xi8[t * inp..(t + 1) * inp],
                        &xi8[(t + 1) * inp..(t + 2) * inp],
                    );
                    dst[t * out + o] = da as f32 * so * xs[t] + b;
                    dst[(t + 1) * out + o] = db as f32 * so * xs[t + 1] + b;
                    t += 2;
                }
                if t < tq {
                    dst[t * out + o] =
                        dot_i8(wrow, &xi8[t * inp..(t + 1) * inp]) as f32 * so * xs[t] + b;
                }
            } else if use_2col {
                let mut t = 0;
                while t + 2 <= tq {
                    let (da, db) = dot_i8_2col(
                        wrow,
                        &xi8[t * inp..(t + 1) * inp],
                        &xi8[(t + 1) * inp..(t + 2) * inp],
                    );
                    dst[t * out + o] = da as f32 * so * xs[t] + b;
                    dst[(t + 1) * out + o] = db as f32 * so * xs[t + 1] + b;
                    t += 2;
                }
                if t < tq {
                    dst[t * out + o] =
                        dot_i8(wrow, &xi8[t * inp..(t + 1) * inp]) as f32 * so * xs[t] + b;
                }
            } else {
                for t in 0..tq {
                    dst[t * out + o] =
                        dot_i8(wrow, &xi8[t * inp..(t + 1) * inp]) as f32 * so * xs[t] + b;
                }
            }
        }
    };

    const PAR_THRESHOLD: usize = 1 << 21;
    const COMPUTE_BOUND_MACS: usize = 1 << 26;
    let avail = avail_parallelism();
    let work = tq.saturating_mul(out).saturating_mul(inp);
    let workers = batch_gemv_cap().map(|c| avail.min(c)).unwrap_or_else(|| {
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

    // Each worker fills a private [tq, out] buffer for its band, then disjoint-merge
    // (each column written by exactly one worker → `0.0 + x == x`), same as gemv_f16_batch.
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
            out_slice[t * out + o0..t * out + o1]
                .copy_from_slice(&local[t * out + o0..t * out + o1]);
        }
    }
}

/// Int8 GEMV for a cohort of independent one-token decoder streams.
///
/// Unlike the prefill scheduler in [`gemv_i8_batch`], this uses the persistent
/// Rayon pool and scales the output-row fan-out with cohort width.  Results are
/// accumulated in output-major order so every worker owns one contiguous slice;
/// the final transpose is pure movement and preserves every scalar bit.
pub fn gemv_i8_cohort(
    w: &I8Mat,
    x: &[f32],
    tq: usize,
    bias: Option<&[f32]>,
    out_slice: &mut [f32],
) {
    let (out, inp) = (w.out, w.inp);
    debug_assert_eq!(x.len(), tq * inp, "i8 cohort x shape mismatch");
    debug_assert_eq!(out_slice.len(), tq * out, "i8 cohort out shape mismatch");
    debug_assert!(bias.is_none_or(|b| b.len() == out));
    if tq == 0 || out == 0 {
        return;
    }
    if tq == 1 {
        gemv_i8(w, x, bias, out_slice);
        return;
    }

    let mut xi8 = vec![0i8; tq * inp];
    let mut xs = vec![0.0f32; tq];
    for t in 0..tq {
        let row = &x[t * inp..(t + 1) * inp];
        let xamax = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
        let scale = xamax / 127.0;
        xs[t] = scale;
        quantize_act_i8_into(row, 1.0 / scale, &mut xi8[t * inp..(t + 1) * inp]);
    }

    let workers = cohort_gemv_worker_count(out, tq).min(out.max(1));
    let band = out.div_ceil(workers).max(1);
    let mut by_output = vec![0.0f32; out * tq];
    by_output
        .par_chunks_mut(band * tq)
        .enumerate()
        .for_each(|(wk, local)| {
            let o0 = wk * band;
            let width = local.len() / tq;
            for local_o in 0..width {
                let o = o0 + local_o;
                let wrow = &w.data[o * inp..(o + 1) * inp];
                let weight_scale = w.scales[o];
                let b = bias.map_or(0.0, |bb| bb[o]);
                let dst = &mut local[local_o * tq..(local_o + 1) * tq];
                let mut t = 0;
                while t + 4 <= tq {
                    let (da, db, dc, dd) = dot_i8_4col(
                        wrow,
                        &xi8[t * inp..(t + 1) * inp],
                        &xi8[(t + 1) * inp..(t + 2) * inp],
                        &xi8[(t + 2) * inp..(t + 3) * inp],
                        &xi8[(t + 3) * inp..(t + 4) * inp],
                    );
                    dst[t] = da as f32 * weight_scale * xs[t] + b;
                    dst[t + 1] = db as f32 * weight_scale * xs[t + 1] + b;
                    dst[t + 2] = dc as f32 * weight_scale * xs[t + 2] + b;
                    dst[t + 3] = dd as f32 * weight_scale * xs[t + 3] + b;
                    t += 4;
                }
                while t + 2 <= tq {
                    let (da, db) = dot_i8_2col(
                        wrow,
                        &xi8[t * inp..(t + 1) * inp],
                        &xi8[(t + 1) * inp..(t + 2) * inp],
                    );
                    dst[t] = da as f32 * weight_scale * xs[t] + b;
                    dst[t + 1] = db as f32 * weight_scale * xs[t + 1] + b;
                    t += 2;
                }
                if t < tq {
                    dst[t] =
                        dot_i8(wrow, &xi8[t * inp..(t + 1) * inp]) as f32 * weight_scale * xs[t]
                            + b;
                }
            }
        });
    for o in 0..out {
        for t in 0..tq {
            out_slice[t * out + o] = by_output[o * tq + t];
        }
    }
}

/// Dot of an int8 weight row against an **f32** activation (no activation
/// quantization): `Σ_i (w[i] as f32) · x[i]`. Scalar fallback.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
fn dot_i8w_f32(w: &[i8], x: &[f32]) -> f32 {
    let n = w.len().min(x.len());
    let mut acc = 0.0f32;
    for i in 0..n {
        acc += (w[i] as f32) * x[i];
    }
    acc
}

/// AVX2 int8-weight × f32-activation dot: sign-extend 8 int8 → i32
/// (`vpmovsxbd`) → f32 (`vcvtdq2ps`) → `vfmadd` against the f32 activation. Four
/// accumulators reduced in the same `((0+1)+(2+3))+((4+5)+(6+7))` order as
/// [`dot_f16c`], so it is the f16-dot with the weight source swapped from f16 to
/// int8 — the win is bandwidth (int8 weight = half the f16 bytes), the activation
/// stays full precision (unlike `gemv_i8`, which also quantizes `x`).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8w_f32(w: &[i8], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = w.len().min(x.len());
    let xp = x.as_ptr();
    // SAFETY: every load is bounded by the `i+32`/`i+8`/`i<n` guards over
    // n = min(w.len, x.len); avx2+fma are guaranteed by this fn's target_feature cfg.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 32 <= n {
            // Each 64-bit load holds 8 int8; cvtepi8_epi32 widens to 8 i32.
            let w0 = _mm_loadl_epi64(w.as_ptr().add(i).cast());
            let w1 = _mm_loadl_epi64(w.as_ptr().add(i + 8).cast());
            let w2 = _mm_loadl_epi64(w.as_ptr().add(i + 16).cast());
            let w3 = _mm_loadl_epi64(w.as_ptr().add(i + 24).cast());
            a0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0)),
                _mm256_loadu_ps(xp.add(i)),
                a0,
            );
            a1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1)),
                _mm256_loadu_ps(xp.add(i + 8)),
                a1,
            );
            a2 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w2)),
                _mm256_loadu_ps(xp.add(i + 16)),
                a2,
            );
            a3 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w3)),
                _mm256_loadu_ps(xp.add(i + 24)),
                a3,
            );
            i += 32;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut s =
            ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
        while i + 8 <= n {
            let p = _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                    w.as_ptr().add(i).cast(),
                ))),
                _mm256_loadu_ps(xp.add(i)),
            );
            let mut t = [0.0f32; 8];
            _mm256_storeu_ps(t.as_mut_ptr(), p);
            s += ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
            i += 8;
        }
        while i < n {
            s += (w[i] as f32) * x[i];
            i += 1;
        }
        s
    }
}

/// Block-wise int8 weight (Q8_0-style): int8 data plus ONE dequant scale per
/// `block` consecutive input columns per row, so a wide row with outliers keeps
/// fine resolution in the calm blocks. `scales` is `out * n_blocks` row-major
/// (`n_blocks = ceil(inp/block)`). Used for `mlp_2`/fc2, where a single per-row
/// scale (`I8Mat`) is too coarse (breaks turbo) but block scales are byte-exact.
#[derive(Debug, Clone)]
pub struct I8BlockMat {
    pub data: Vec<i8>,
    pub scales: Vec<f32>,
    pub out: usize,
    pub inp: usize,
    pub block: usize,
}

/// Quantize a natural `[out, inp]` f16 weight to block-wise int8 (`block`
/// columns share a symmetric `amax/127` scale). Mirrors [`quantize_f16_to_i8`]
/// but per block, so the whisper.cpp-Q8_0-class accuracy on wide rows.
pub fn quantize_f16_to_i8_blocked(
    w: &[Float16],
    out: usize,
    inp: usize,
    block: usize,
) -> I8BlockMat {
    quantize_f16_to_int_blocked(w, out, inp, block, 127.0)
}

/// Block-wise 4-bit variant (levels in `[-7, 7]`, stored one-per-`i8` — the
/// values still ride the [`gemv_i8w_f32a_blocked`] dot, so this measures 4-bit
/// PRECISION without the packed-nibble kernel). For probing whether a GELU-absorbed
/// weight (`mlp_0`) tolerates int4 byte-exactly before writing the packed kernel.
pub fn quantize_f16_to_i4_blocked(
    w: &[Float16],
    out: usize,
    inp: usize,
    block: usize,
) -> I8BlockMat {
    quantize_f16_to_int_blocked(w, out, inp, block, 7.0)
}

/// AVX2+F16C block-wise symmetric quantize of ONE f16 weight row: for each
/// `block`-wide span, `scale[b] = max|w|/max_level` and `q[i] = (w[i]/scale[b])
/// .round().clamp(±max_level) as i8`. Two vectorized passes/block via
/// `_mm256_cvtph_ps` (exact f16→f32) — amax tree-reduce (= scalar fold for finite
/// values) then round-HALF-AWAY (`f32::round`) + clamp + saturating-pack. Runtime
/// `max_level` (127 for i8, 7 for the i4-level variant). **Byte-identical** to the
/// scalar block loop; hoists the SIMD constants once per row. See
/// `quantize_f16_row_blocked_matches_scalar`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "f16c"
))]
#[allow(unsafe_code)]
fn quantize_f16_row_blocked_to_int_into(
    wrow: &[Float16],
    block: usize,
    max_level: f32,
    drow: &mut [i8],
    srow: &mut [f32],
) {
    use core::arch::x86_64::*;
    let inp = wrow.len();
    debug_assert_eq!(drow.len(), inp, "blocked quant drow len mismatch");
    let n_blocks = inp.div_ceil(block);
    let wp = wrow.as_ptr();
    // SAFETY: avx2+f16c by cfg. Every 128-bit f16 load / i64 store is bounded by an
    // `i+8<=e` / `j+8<=e` guard (`e<=inp`), the `<8` remainders run scalar; no store
    // crosses a block boundary (guard stops at `e`). Float16 is repr(transparent) u16.
    unsafe {
        let absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        let half = _mm256_set1_ps(0.5);
        let signmask = _mm256_set1_ps(-0.0);
        let vmax = _mm256_set1_ps(max_level);
        let vmin = _mm256_set1_ps(-max_level);
        for b in 0..n_blocks {
            let s = b * block;
            let e = ((b + 1) * block).min(inp);
            // pass 1: amax = max|f16→f32(w[i])| over [s,e)
            let mut m = _mm256_setzero_ps();
            let mut i = s;
            while i + 8 <= e {
                let v = _mm256_cvtph_ps(_mm_loadu_si128(wp.add(i).cast()));
                m = _mm256_max_ps(m, _mm256_and_ps(v, absmask));
                i += 8;
            }
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), m);
            let mut amax = tmp.iter().copied().fold(0.0f32, f32::max);
            while i < e {
                amax = amax.max(wrow.get_unchecked(i).to_f32().abs());
                i += 1;
            }
            let scale = amax.max(1e-9) / max_level;
            *srow.get_unchecked_mut(b) = scale;
            let inv = 1.0 / scale;
            // pass 2: (f16→f32(w)·inv) round-half-away, clamp ±max_level, pack to i8.
            let vinv = _mm256_set1_ps(inv);
            let mut j = s;
            while j + 8 <= e {
                let v = _mm256_mul_ps(_mm256_cvtph_ps(_mm_loadu_si128(wp.add(j).cast())), vinv);
                let vh = _mm256_add_ps(v, _mm256_or_ps(half, _mm256_and_ps(v, signmask)));
                let r = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(vh);
                let r = _mm256_min_ps(_mm256_max_ps(r, vmin), vmax);
                let ri = _mm256_cvtps_epi32(r);
                let lo = _mm256_castsi256_si128(ri);
                let hi = _mm256_extracti128_si256::<1>(ri);
                let i16s = _mm_packs_epi32(lo, hi);
                let i8s = _mm_packs_epi16(i16s, i16s);
                _mm_storel_epi64(drow.as_mut_ptr().add(j) as *mut __m128i, i8s);
                j += 8;
            }
            while j < e {
                *drow.get_unchecked_mut(j) = (wrow.get_unchecked(j).to_f32() * inv)
                    .round()
                    .clamp(-max_level, max_level)
                    as i8;
                j += 1;
            }
        }
    }
}

/// Scalar fallback (non-avx2/f16c): the exact reference the AVX2 path reproduces.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "f16c"
)))]
fn quantize_f16_row_blocked_to_int_into(
    wrow: &[Float16],
    block: usize,
    max_level: f32,
    drow: &mut [i8],
    srow: &mut [f32],
) {
    let inp = wrow.len();
    let n_blocks = inp.div_ceil(block);
    for b in 0..n_blocks {
        let s = b * block;
        let e = ((b + 1) * block).min(inp);
        let amax = wrow[s..e]
            .iter()
            .map(|h| h.to_f32().abs())
            .fold(0.0f32, f32::max)
            .max(1e-9);
        let sc = amax / max_level;
        srow[b] = sc;
        let inv = 1.0 / sc;
        for i in s..e {
            drow[i] = (wrow[i].to_f32() * inv)
                .round()
                .clamp(-max_level, max_level) as i8;
        }
    }
}

fn quantize_f16_to_int_blocked(
    w: &[Float16],
    out: usize,
    inp: usize,
    block: usize,
    max_level: f32,
) -> I8BlockMat {
    debug_assert_eq!(w.len(), out * inp);
    debug_assert!(block > 0);
    let n_blocks = inp.div_ceil(block);
    let mut data = vec![0i8; out * inp];
    let mut scales = vec![0.0f32; out * n_blocks];
    data.par_chunks_mut(inp)
        .zip(scales.par_chunks_mut(n_blocks))
        .enumerate()
        .for_each(|(o, (drow, srow))| {
            let wrow = &w[o * inp..(o + 1) * inp];
            quantize_f16_row_blocked_to_int_into(wrow, block, max_level, drow, srow);
        });
    I8BlockMat {
        data,
        scales,
        out,
        inp,
        block,
    }
}

/// Mixed block-wise GEMV: block-int8 weight × **f32** activation.
/// `out[o] = Σ_b (scale[o,b] · Σ_{i∈b} w_i8[o,i]·x[i]) + bias[o]`. Keeps
/// [`gemv_i8`]'s halved weight-bandwidth win but with per-block weight scales and
/// a full-precision activation, so it is accurate enough for the residual-feeding
/// `mlp_2`/fc2 (the per-row [`I8Mat`] variant broke turbo). Parallelization mirrors
/// [`gemv_i8`]; the per-block dot reuses [`dot_i8w_f32`].
pub fn gemv_i8w_f32a_blocked(
    w: &I8BlockMat,
    x: &[f32],
    bias: Option<&[f32]>,
    out_slice: &mut [f32],
) {
    let (out, inp, block) = (w.out, w.inp, w.block);
    let n_blocks = inp.div_ceil(block);
    debug_assert_eq!(
        w.data.len(),
        out * inp,
        "gemv_i8w_f32a_blocked weight shape mismatch"
    );
    debug_assert_eq!(x.len(), inp, "gemv_i8w_f32a_blocked x length mismatch");
    debug_assert_eq!(
        out_slice.len(),
        out,
        "gemv_i8w_f32a_blocked out length mismatch"
    );
    debug_assert_eq!(
        w.scales.len(),
        out * n_blocks,
        "gemv_i8w_f32a_blocked scales shape mismatch"
    );
    debug_assert!(
        bias.is_none_or(|b| b.len() == out),
        "gemv_i8w_f32a_blocked bias length mismatch"
    );
    let fill = |o_base: usize, slice: &mut [f32]| {
        for (i, slot) in slice.iter_mut().enumerate() {
            let o = o_base + i;
            let wrow = &w.data[o * inp..(o + 1) * inp];
            let srow = &w.scales[o * n_blocks..(o + 1) * n_blocks];
            let mut acc = 0.0f32;
            for (b, &sc) in srow.iter().enumerate() {
                let s = b * block;
                let e = ((b + 1) * block).min(inp);
                acc += dot_i8w_f32(&wrow[s..e], &x[s..e]) * sc;
            }
            *slot = acc + bias.map_or(0.0, |bb| bb[o]);
        }
    };
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

/// Batched block-int8-weight × f32-activation GEMV.
///
/// Each `(token, output)` follows [`gemv_i8w_f32a_blocked`]'s block and
/// accumulation order exactly, while output-row bands keep one weight row hot
/// across every token in the cohort.  The temporary band layout is contiguous;
/// the final scatter only reorders already-computed scalar results.
pub fn gemv_i8w_f32a_blocked_batch(
    w: &I8BlockMat,
    x: &[f32],
    tq: usize,
    bias: Option<&[f32]>,
    out_slice: &mut [f32],
) {
    let (out, inp, block) = (w.out, w.inp, w.block);
    let n_blocks = inp.div_ceil(block);
    debug_assert_eq!(x.len(), tq * inp, "blocked batch x shape mismatch");
    debug_assert_eq!(
        out_slice.len(),
        tq * out,
        "blocked batch out shape mismatch"
    );
    debug_assert!(bias.is_none_or(|b| b.len() == out));
    if tq == 0 || out == 0 {
        return;
    }

    let workers = cohort_gemv_worker_count(out, tq).min(out.max(1));
    let band = out.div_ceil(workers).max(1);
    let mut by_output = vec![0.0f32; out * tq];
    by_output
        .par_chunks_mut(band * tq)
        .enumerate()
        .for_each(|(wk, local)| {
            let o0 = wk * band;
            let width = local.len() / tq;
            for local_o in 0..width {
                let o = o0 + local_o;
                let wrow = &w.data[o * inp..(o + 1) * inp];
                let srow = &w.scales[o * n_blocks..(o + 1) * n_blocks];
                for t in 0..tq {
                    let xrow = &x[t * inp..(t + 1) * inp];
                    let mut acc = 0.0f32;
                    for (b, &sc) in srow.iter().enumerate() {
                        let s = b * block;
                        let e = ((b + 1) * block).min(inp);
                        acc += dot_i8w_f32(&wrow[s..e], &xrow[s..e]) * sc;
                    }
                    local[local_o * tq + t] = acc + bias.map_or(0.0, |bb| bb[o]);
                }
            }
        });
    for o in 0..out {
        for t in 0..tq {
            out_slice[t * out + o] = by_output[o * tq + t];
        }
    }
}

/// PACKED block-wise 4-bit weight (Q4_0-style), fixed `block == 32`. Two signed
/// nibbles ride in each byte: for block `b`, byte `j` (`0..16`) holds `w[b*32+j]`
/// in the low nibble and `w[b*32+j+16]` in the high nibble, each a 4-bit two's
/// complement of a value in `[-7, 7]` (symmetric `amax/7` scale, ONE per block).
/// So `data` is `out * inp/2` bytes — HALF the [`I8BlockMat`] weight read, the
/// point of the type: the int4-probe ([`quantize_f16_to_i4_blocked`]) proved
/// mlp_0/fc1 is 4-bit byte-exact (GELU-absorbed) but stored one value per byte
/// (no bandwidth win); this packs them so the DRAM-bound decode read actually
/// halves. `scales` is `out * n_blocks` row-major (`n_blocks = inp/32`).
#[derive(Debug, Clone)]
pub struct I4BlockMat {
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub out: usize,
    pub inp: usize,
}

/// Quantize a natural `[out, inp]` f16 weight to PACKED block-wise int4. Produces
/// the SAME per-element 4-bit values as [`quantize_f16_to_i4_blocked`] (same
/// `amax/7` block scale, same round+clamp), just packed two-per-byte in the Q4_0
/// layout — so [`gemv_i4_packed_f32a`] is byte-exact with the unpacked int4 probe.
/// Requires `inp % 32 == 0` (fc1 input is `d_model` ∈ {384,512,768,1024,1280}).
pub fn quantize_f16_to_i4_packed(w: &[Float16], out: usize, inp: usize) -> I4BlockMat {
    const BLOCK: usize = 32;
    debug_assert_eq!(w.len(), out * inp);
    assert_eq!(
        inp % BLOCK,
        0,
        "i4-packed requires inp % 32 == 0, got inp={inp}"
    );
    let n_blocks = inp / BLOCK;
    let row_bytes = inp / 2;
    let mut data = vec![0u8; out * row_bytes];
    let mut scales = vec![0.0f32; out * n_blocks];
    data.par_chunks_mut(row_bytes)
        .zip(scales.par_chunks_mut(n_blocks))
        .enumerate()
        .for_each(|(o, (drow, srow))| {
            let wrow = &w[o * inp..(o + 1) * inp];
            for b in 0..n_blocks {
                let base = b * BLOCK;
                let amax = wrow[base..base + BLOCK]
                    .iter()
                    .map(|h| h.to_f32().abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-9);
                let sc = amax / 7.0;
                srow[b] = sc;
                let inv = 1.0 / sc;
                let q = |k: usize| -> u8 {
                    let v = (wrow[base + k].to_f32() * inv).round().clamp(-7.0, 7.0) as i32;
                    (v & 0x0F) as u8 // 4-bit two's complement of v ∈ [-7,7]
                };
                for j in 0..16 {
                    drow[b * 16 + j] = q(j) | (q(j + 16) << 4);
                }
            }
        });
    I4BlockMat {
        data,
        scales,
        out,
        inp,
    }
}

/// Dot of ONE packed 32-block (16 bytes → 32 signed nibbles) against 32 f32
/// activations, in the EXACT accumulator/reduction order of [`dot_i8w_f32`] on a
/// 32-element slice, so the packed path is bit-identical to the int4 probe.
/// Scalar fallback: unpack into an `[i8; 32]` and defer to [`dot_i8w_f32`].
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
#[inline]
fn dot_i4_block_packed(packed: &[u8], x: &[f32]) -> f32 {
    let mut tmp = [0i8; 32];
    for j in 0..16 {
        let byte = packed[j] as i32;
        tmp[j] = (((byte & 0x0F) ^ 0x08) - 0x08) as i8;
        tmp[j + 16] = ((((byte >> 4) & 0x0F) ^ 0x08) - 0x08) as i8;
    }
    dot_i8w_f32(&tmp, x)
}

/// AVX2 packed-int4 block dot. Unpacks the 16 bytes into two `__m128i` of signed
/// int8 (low nibbles → w[0..16], high nibbles → w[16..32]) via `(nib ^ 8) - 8`
/// (4-bit sign-extend), then runs the same four-accumulator fmadd + tree reduce as
/// [`dot_i8w_f32`]. The unpack is fully vectorized (no per-nibble scalar work) so
/// the halved weight read is not eaten by decode-time compute.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[inline]
#[allow(unsafe_code)]
fn dot_i4_block_packed(packed: &[u8], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let xp = x.as_ptr();
    // SAFETY: caller guarantees packed.len() >= 16 and x.len() >= 32 (one 32-block);
    // avx2+fma are guaranteed by this fn's target_feature cfg.
    unsafe {
        let v = _mm_loadu_si128(packed.as_ptr().cast());
        let mask0f = _mm_set1_epi8(0x0F);
        let c8 = _mm_set1_epi8(8);
        // low nibbles = w[0..16], high nibbles = w[16..32], each 0..15 then
        // sign-extended from 4 bits via (nib ^ 8) - 8.
        let lo_u = _mm_and_si128(v, mask0f);
        let hi_u = _mm_and_si128(_mm_srli_epi16(v, 4), mask0f);
        let lo = _mm_sub_epi8(_mm_xor_si128(lo_u, c8), c8);
        let hi = _mm_sub_epi8(_mm_xor_si128(hi_u, c8), c8);
        // Four groups matching dot_i8w_f32's a0..a3 on a 32-slice:
        // w[0..8]=lo[0..8], w[8..16]=lo[8..16], w[16..24]=hi[0..8], w[24..32]=hi[8..16].
        let w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
        let w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        let w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
        let w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        // fmadd against zero (not mul) to mirror dot_i8w_f32 exactly — bit-identical
        // even in the signed-zero lane corner.
        let z = _mm256_setzero_ps();
        let a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp), z);
        let a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp.add(8)), z);
        let a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp.add(16)), z);
        let a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp.add(24)), z);
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]))
    }
}

/// Mixed PACKED-int4 GEMV: packed-int4 weight × **f32** activation. Same math as
/// [`gemv_i8w_f32a_blocked`] (`out[o] = Σ_b scale[o,b]·dot(block_b, x_b) + bias`)
/// but reads HALF the weight bytes. Byte-identical to the int4 probe on
/// `gemv_i8w_f32a_blocked`. Parallelization mirrors [`gemv_i8`]. For `mlp_0`/fc1
/// (GELU-absorbed; see [`super::int4_mlp0_enabled`]).
pub fn gemv_i4_packed_f32a(w: &I4BlockMat, x: &[f32], bias: Option<&[f32]>, out_slice: &mut [f32]) {
    const BLOCK: usize = 32;
    let (out, inp) = (w.out, w.inp);
    let n_blocks = inp / BLOCK;
    let row_bytes = inp / 2;
    debug_assert_eq!(
        w.data.len(),
        out * row_bytes,
        "gemv_i4_packed weight shape mismatch"
    );
    debug_assert_eq!(x.len(), inp, "gemv_i4_packed x length mismatch");
    debug_assert_eq!(out_slice.len(), out, "gemv_i4_packed out length mismatch");
    debug_assert_eq!(
        w.scales.len(),
        out * n_blocks,
        "gemv_i4_packed scales shape mismatch"
    );
    debug_assert!(
        bias.is_none_or(|b| b.len() == out),
        "gemv_i4_packed bias length mismatch"
    );
    let fill = |o_base: usize, slice: &mut [f32]| {
        for (i, slot) in slice.iter_mut().enumerate() {
            let o = o_base + i;
            let wrow = &w.data[o * row_bytes..(o + 1) * row_bytes];
            let srow = &w.scales[o * n_blocks..(o + 1) * n_blocks];
            let mut acc = 0.0f32;
            for (b, &sc) in srow.iter().enumerate() {
                acc += dot_i4_block_packed(
                    &wrow[b * 16..b * 16 + 16],
                    &x[b * BLOCK..b * BLOCK + BLOCK],
                ) * sc;
            }
            *slot = acc + bias.map_or(0.0, |bb| bb[o]);
        }
    };
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

/// Batched packed-int4-weight × f32-activation GEMV for independent decoder
/// streams.  One output-weight row is reused across every activation row, and
/// every scalar result is bit-identical to [`gemv_i4_packed_f32a`].
pub fn gemv_i4_packed_f32a_batch(
    w: &I4BlockMat,
    x: &[f32],
    tq: usize,
    bias: Option<&[f32]>,
    out_slice: &mut [f32],
) {
    const BLOCK: usize = 32;
    let (out, inp) = (w.out, w.inp);
    let n_blocks = inp / BLOCK;
    let row_bytes = inp / 2;
    debug_assert_eq!(x.len(), tq * inp, "i4 batch x shape mismatch");
    debug_assert_eq!(out_slice.len(), tq * out, "i4 batch out shape mismatch");
    debug_assert!(bias.is_none_or(|b| b.len() == out));
    if tq == 0 || out == 0 {
        return;
    }

    let workers = cohort_gemv_worker_count(out, tq).min(out.max(1));
    let band = out.div_ceil(workers).max(1);
    let mut by_output = vec![0.0f32; out * tq];
    by_output
        .par_chunks_mut(band * tq)
        .enumerate()
        .for_each(|(wk, local)| {
            let o0 = wk * band;
            let width = local.len() / tq;
            for local_o in 0..width {
                let o = o0 + local_o;
                let wrow = &w.data[o * row_bytes..(o + 1) * row_bytes];
                let srow = &w.scales[o * n_blocks..(o + 1) * n_blocks];
                for t in 0..tq {
                    let xrow = &x[t * inp..(t + 1) * inp];
                    let mut acc = 0.0f32;
                    for (b, &sc) in srow.iter().enumerate() {
                        acc += dot_i4_block_packed(
                            &wrow[b * 16..b * 16 + 16],
                            &xrow[b * BLOCK..b * BLOCK + BLOCK],
                        ) * sc;
                    }
                    local[local_o * tq + t] = acc + bias.map_or(0.0, |bb| bb[o]);
                }
            }
        });
    for o in 0..out {
        for t in 0..tq {
            out_slice[t * out + o] = by_output[o * tq + t];
        }
    }
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
    let workers = batch_gemv_cap().map(|c| avail.min(c)).unwrap_or_else(|| {
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

    // For the long compute-bound cross-K/V shape, split ownership by token rows
    // instead of output columns. Each worker writes disjoint contiguous
    // `[rows, out]` morsels directly into `out_slice`, eliminating the legacy
    // private `[tq, out]` buffers and merge while preserving each dot product's
    // exact summation order.
    if work >= COMPUTE_BOUND_MACS && batch_gemv_row_morsel_enabled() {
        gemv_f16_batch_rows(w_f16, out, inp, x, tq, bias, out_slice, workers, use_fused);
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

/// F16-weight GEMV for independent one-token decoder streams.
///
/// The ordinary batch kernel is tuned for long prefill matrices.  Decoder
/// cohorts are small and repeated hundreds of times, so spawning OS threads per
/// projection dominates.  This path keeps the weight row hot across all cohort
/// activations, uses the persistent Rayon pool, and scales row bands with the
/// number of live streams.
pub fn gemv_f16_cohort(
    w_f16: &[Float16],
    out: usize,
    inp: usize,
    x: &[f32],
    tq: usize,
    bias: Option<&[f32]>,
    out_slice: &mut [f32],
) {
    debug_assert_eq!(w_f16.len(), out * inp, "f16 cohort weight shape mismatch");
    debug_assert_eq!(x.len(), tq * inp, "f16 cohort x shape mismatch");
    debug_assert_eq!(out_slice.len(), tq * out, "f16 cohort out shape mismatch");
    debug_assert!(bias.is_none_or(|b| b.len() == out));
    if tq == 0 || out == 0 {
        return;
    }
    if tq == 1 {
        gemv_f16(w_f16, out, inp, x, bias, out_slice);
        return;
    }

    let workers = cohort_gemv_worker_count(out, tq).min(out.max(1));
    let band = out.div_ceil(workers).max(1);
    let use_fused = f16c_dot_available();
    let mut by_output = vec![0.0f32; out * tq];
    by_output
        .par_chunks_mut(band * tq)
        .enumerate()
        .for_each(|(wk, local)| {
            let o0 = wk * band;
            let width = local.len() / tq;
            let mut scratch = vec![0.0f32; inp];
            for local_o in 0..width {
                let o = o0 + local_o;
                let wrow = &w_f16[o * inp..(o + 1) * inp];
                let b = bias.map_or(0.0, |bb| bb[o]);
                let dst = &mut local[local_o * tq..(local_o + 1) * tq];
                for t in 0..tq {
                    let xrow = &x[t * inp..(t + 1) * inp];
                    dst[t] = dequant_row_dot(wrow, xrow, &mut scratch, use_fused) + b;
                }
            }
        });
    for o in 0..out {
        for t in 0..tq {
            out_slice[t * out + o] = by_output[o * tq + t];
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

/// Out-of-place [`layer_norm`]: read `src` `[rows, cols]`, write normalized rows
/// into `dst` (`len == rows*cols`). Byte-identical to `x.clone()` +
/// `layer_norm(&mut x)` (same per-row f64 SoA math, same contiguous band split),
/// but skips the intermediate clone's full-buffer memcpy — the encoder does two
/// `x.clone()`s per layer purely to preserve `x` for the residual, and that copy
/// is a redundant pass this fuses into the normalize (~985 MB/window of memcpy
/// removed on turbo). `dst` may be uninitialized (every element is written).
pub fn layer_norm_into(src: &Mat, dst: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let cols = src.cols;
    debug_assert_eq!(
        dst.len(),
        src.rows * cols,
        "layer_norm_into: dst len mismatch"
    );
    if cols == 0 || w.len() != cols || b.len() != cols {
        // Degenerate no-op mirrors in-place `layer_norm` (which leaves x
        // untouched) ⇒ dst must equal src, since the clone would have copied it.
        dst.copy_from_slice(&src.data);
        return;
    }
    let eps = f64::from(eps);
    const PAR_THRESHOLD: usize = 1 << 16;
    let rows = src.rows;
    if rows * cols < PAR_THRESHOLD || worker_count() < 2 {
        norm_rows_into(&src.data, dst, cols, w, b, eps);
        return;
    }
    let band_rows = rows.div_ceil(worker_count()).max(1);
    // Same contiguous band split as `layer_norm` (byte-identical), reading the
    // matching `src` band and writing the `dst` band.
    dst.par_chunks_mut(band_rows * cols)
        .zip(src.data.par_chunks(band_rows * cols))
        .for_each(|(dband, sband)| norm_rows_into(sband, dband, cols, w, b, eps));
}

/// Out-of-place `norm_rows`: identical f64 SoA math, reading `src` and writing
/// `dst` (both `[nrows, cols]`) instead of mutating in place.
fn norm_rows_into(src: &[f32], dst: &mut [f32], cols: usize, w: &[f32], b: &[f32], eps: f64) {
    const L: usize = 8;
    type V = Simd<f64, L>;
    let n = cols as f64;
    let nrows = src.len() / cols;
    let nfull = nrows - nrows % L;

    let mut soa = vec![V::splat(0.0); cols]; // reused per 8-row group
    let mut g = 0;
    while g < nfull {
        for (j, s) in soa.iter_mut().enumerate() {
            let mut a = [0.0f64; L];
            for (lane, al) in a.iter_mut().enumerate() {
                *al = f64::from(src[(g + lane) * cols + j]);
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
                dst[(g + lane) * cols + j] = val as f32;
            }
        }
        g += L;
    }

    for r in nfull..nrows {
        let srow = &src[r * cols..(r + 1) * cols];
        let drow = &mut dst[r * cols..(r + 1) * cols];
        let mut sum = 0.0f64;
        for &v in srow.iter() {
            sum += f64::from(v);
        }
        let mean = sum / n;
        let mut var = 0.0f64;
        for &v in srow.iter() {
            let d = f64::from(v) - mean;
            var += d * d;
        }
        var /= n;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ((d, &s), (&wi, &bi)) in drow.iter_mut().zip(srow.iter()).zip(w.iter().zip(b.iter())) {
            let normed = (f64::from(s) - mean) * inv_std;
            *d = (normed * f64::from(wi) + f64::from(bi)) as f32;
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
            let g =
                0.5 * f * (1.0 + (GELU_SQRT_2_OVER_PI * f * (1.0 + GELU_COEF_A * f * f)).tanh());
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
/// as the scalar `Float16::from_f32`) → widen to 8 u32 indices → **8 explicit
/// scalar table-loads** (NOT `vgatherdps`) → blend the clamp. On Zen3 (this box's
/// 5975WX) `vgatherdps ymm` is microcoded and caps at ~2.2 Gelem/s even cache-hot;
/// scalar loads from the L2-resident 256 KiB table pipeline at ~2-3/cyc and are
/// byte-identical (same indices, same lane order), for a measured **1.18×
/// cache-resident / 1.13× streaming** speedup over the gather (`examples/gelu_gather_probe`,
/// max|Δ|=0 over 7.68M elems spanning both clamp regions). Bit-identical to the
/// scalar fallback (max|Δ|=0 in `examples/gelu_probe`), ~4.4× faster than the old tanh.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "f16c",
    target_feature = "avx2"
))]
#[allow(unsafe_code)]
fn gelu_slice(data: &mut [f32]) {
    use core::arch::x86_64::*;
    let table = gelu_table();
    let n = data.len();
    // SAFETY: all loads/stores are bounded by the `i+8<=n` guard; each table index
    // is a widened f16 bit pattern (always 0..=65535, in-bounds for the 1<<16 table,
    // so `get_unchecked` is sound); f16c/avx2 are guaranteed by this fn's
    // target_feature cfg.
    unsafe {
        let neg10 = _mm256_set1_ps(-10.0);
        let pos10 = _mm256_set1_ps(10.0);
        let zero = _mm256_setzero_ps();
        let mut i = 0;
        let mut idxs = [0u32; 8];
        while i + 8 <= n {
            let x = _mm256_loadu_ps(data.as_ptr().add(i));
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x);
            let idx = _mm256_cvtepu16_epi32(h);
            // 8 scalar table-loads instead of `vgatherdps` (microcoded on Zen3).
            // `_mm256_set_ps` lane j = table[idxs[j]] ⇒ identical to the gather.
            _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, idx);
            let g = _mm256_set_ps(
                *table.get_unchecked(idxs[7] as usize),
                *table.get_unchecked(idxs[6] as usize),
                *table.get_unchecked(idxs[5] as usize),
                *table.get_unchecked(idxs[4] as usize),
                *table.get_unchecked(idxs[3] as usize),
                *table.get_unchecked(idxs[2] as usize),
                *table.get_unchecked(idxs[1] as usize),
                *table.get_unchecked(idxs[0] as usize),
            );
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
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "f16c",
    target_feature = "avx2"
)))]
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

/// Fused bias-add + GELU: `x[i][j] = gelu(x[i][j] + bias[j])` in ONE pass.
///
/// BYTE-IDENTICAL to a separate `matmul_bias` bias pass (`x[i][j] += bias[j]`)
/// followed by [`gelu`]: the intermediate `x[i][j] + bias[j]` is the same f32
/// value (identical operands, identical add) and [`gelu_slice`] is elementwise +
/// chunk-invariant (the landed f16-table GELU is `max|Δ|=0` regardless of slice
/// boundaries). The win is ELIMINATING the separate bias pass — `matmul_bias`
/// runs its bias RMW SINGLE-THREADED over the whole `[n_ctx, mlp_hidden]` fc1
/// output (`[1500,5120]` = ~30 MiB, L3-borderline ⇒ partly DRAM), leaving 31
/// cores idle, right before the (already-parallel) GELU pass over the SAME
/// buffer. Folding the bias into the GELU's read makes it a single parallel pass.
/// Parallel over whole-row bands so `bias[j]` alignment stays trivial.
pub fn gelu_add_bias(x: &mut Mat, bias: &[f32]) {
    debug_assert_eq!(bias.len(), x.cols, "gelu_add_bias: bias len != cols");
    let cols = x.cols;
    let add_gelu = move |band: &mut [f32]| {
        for row in band.chunks_mut(cols) {
            for (v, &b) in row.iter_mut().zip(bias) {
                *v += b;
            }
            gelu_slice(row);
        }
    };
    const PAR_THRESHOLD: usize = 1 << 15;
    let n = x.data.len();
    if n < PAR_THRESHOLD || worker_count() < 2 {
        add_gelu(&mut x.data);
        return;
    }
    let rows_per = x.rows.div_ceil(worker_count()).max(1);
    x.data.par_chunks_mut(rows_per * cols).for_each(add_gelu);
}

/// Fused bias-add + residual: `x[i][j] += proj[i][j] + bias[j]` in ONE parallel pass.
///
/// BYTE-IDENTICAL to `matmul_bias`'s serial bias pass (`proj[i][j] += bias[j]`)
/// followed by the serial residual [`add_in_place`](super::encoder)-style
/// `x[i][j] += proj[i][j]`: both compute `x[i][j] + (proj[i][j] + bias[j])` as
/// `t = proj+bias` (one rounding) then `x + t` (second rounding), identical
/// operand order. Merges the two SERIAL passes into one and reads `proj` ONCE,
/// eliminating the separate bias pass's write-back of the `[n_ctx, n_state]`
/// output. Pays ONLY when that output is partly-DRAM — its producing sgemm's
/// working set > L3 (e.g. `mlp.proj`/fc2: `[1500,5120]` input + `[5120,1280]`
/// weight = ~56 MiB stream evicts the 7.68 MiB output before the bias pass reads
/// row 0) — so the serial bias RMW is bandwidth-starved single-core. (The residual
/// add was ALSO DRAM-starved at turbo scale for the same reason — `encoder::
/// add_in_place` is now parallel above a 1<<20-elt threshold, `e43b50a`, −48% on
/// `attn_resid`; only tiny.en's L2-warm `[1500,384]` operand stays serial.) The win
/// here is removing the DRAM-bound bias pass; the fused add rides along for free.
/// Parallel over whole-row bands (bias alignment trivial, rows independent).
pub fn add_bias_residual(x: &mut Mat, proj: &Mat, bias: &[f32]) {
    debug_assert_eq!(
        (x.rows, x.cols),
        (proj.rows, proj.cols),
        "add_bias_residual shape mismatch"
    );
    debug_assert_eq!(bias.len(), x.cols, "add_bias_residual: bias len != cols");
    let cols = x.cols;
    let apply = move |xband: &mut [f32], pband: &[f32]| {
        for (xrow, prow) in xband.chunks_mut(cols).zip(pband.chunks(cols)) {
            for ((xv, &pv), &bv) in xrow.iter_mut().zip(prow).zip(bias) {
                *xv += pv + bv;
            }
        }
    };
    const PAR_THRESHOLD: usize = 1 << 15;
    let n = x.data.len();
    if n < PAR_THRESHOLD || worker_count() < 2 {
        apply(&mut x.data, &proj.data);
        return;
    }
    let band = x.rows.div_ceil(worker_count()).max(1) * cols;
    x.data
        .par_chunks_mut(band)
        .zip(proj.data.par_chunks(band))
        .for_each(|(xb, pb)| apply(xb, pb));
}

/// Poly-exp softmax gate — shares the sampler's `FW_SIMD_EXP` env var (default OFF,
/// an owner opt-in). When set, [`softmax_rows`] replaces the per-element scalar libm
/// `.exp()` with an AVX2 poly (same Taylor/`ln2`-range-reduced poly family as
/// `decode::logsumexp_sum_simd`). NON-byte-exact ⇒ owner/faithfulness-gated; default off
/// is byte-identical to the pure-libm path. Measured 6.25× on the per-token decode softmax
/// workload (`examples/decode_softmax_exp_probe`), cross-attn-dominated; the sampler's
/// existing flag only covered `compute_logprobs`, so this captures the larger cross/self
/// attention softmax exp (see docs/NEGATIVE_EVIDENCE.md 2026-07-03).
fn simd_exp_softmax_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FW_SIMD_EXP").is_some())
}

/// AVX2 poly-exp softmax numerator: writes `exp(row[i]-max)` into `row` and returns the
/// sum. `-inf`/NaN lanes map to 0 (via the `l > -inf` keep-mask, matching the scalar
/// finite-guard: `NaN > -inf` and `-inf > -inf` are both false). Only reached under
/// [`simd_exp_softmax_enabled`]; NON-byte-exact vs libm (poly + reordered sum).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[allow(unsafe_code)]
// `log2e`/`ln2` are tuned range-reduction literals of this poly-exp kernel, not
// free-standing math constants; replacing them with `f32::consts` would perturb
// the kernel's numerics, so `approx_constant` is suppressed deliberately.
#[allow(clippy::approx_constant)]
fn softmax_row_poly_numer(row: &mut [f32], max: f32) -> f32 {
    use core::arch::x86_64::*;
    let n = row.len();
    let p = row.as_mut_ptr();
    // SAFETY: avx2+fma guaranteed by this fn's cfg; every load/store is bounded by the
    // `i+8<=n` guard and the `< 8` remainder runs scalar. `x = row[i]-max <= 0` (max is the
    // row max) so the pow2 scale never overflows; only the low clamp `lo` is needed.
    unsafe {
        let vmax = _mm256_set1_ps(max);
        let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
        let log2e = _mm256_set1_ps(1.442_695_f32);
        let ln2 = _mm256_set1_ps(0.693_147_2_f32);
        let lo = _mm256_set1_ps(-87.3365_f32);
        let c0 = _mm256_set1_ps(1.0);
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(0.166_666_67_f32);
        let c4 = _mm256_set1_ps(0.041_666_66_f32);
        let c5 = _mm256_set1_ps(0.008_333_33_f32);
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let l = _mm256_loadu_ps(p.add(i));
            let keep = _mm256_cmp_ps::<_CMP_GT_OQ>(l, ninf); // false for -inf AND NaN
            let xv = _mm256_max_ps(_mm256_sub_ps(l, vmax), lo);
            let kf = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
                _mm256_mul_ps(xv, log2e),
            );
            let r = _mm256_fnmadd_ps(kf, ln2, xv);
            let mut pp = _mm256_fmadd_ps(c5, r, c4);
            pp = _mm256_fmadd_ps(pp, r, c3);
            pp = _mm256_fmadd_ps(pp, r, c2);
            pp = _mm256_fmadd_ps(pp, r, c1);
            pp = _mm256_fmadd_ps(pp, r, c0);
            let ki = _mm256_cvtps_epi32(kf);
            let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
                ki,
                _mm256_set1_epi32(127),
            )));
            let e = _mm256_and_ps(_mm256_mul_ps(pp, pow2), keep); // zero masked lanes
            _mm256_storeu_ps(p.add(i), e);
            acc = _mm256_add_ps(acc, e);
            i += 8;
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut s =
            ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
        while i < n {
            // <8-element tail: libm exp (negligible count; the sum is non-byte-exact anyway).
            let e = (row[i] - max).exp();
            let e = if e.is_finite() { e } else { 0.0 };
            row[i] = e;
            s += e;
            i += 1;
        }
        s
    }
}

/// Scalar fallback (non-avx2 build): identical to the default libm loop.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
)))]
fn softmax_row_poly_numer(row: &mut [f32], max: f32) -> f32 {
    let mut s = 0.0f32;
    for v in row.iter_mut() {
        let e = (*v - max).exp();
        let e = if e.is_finite() { e } else { 0.0 };
        *v = e;
        s += e;
    }
    s
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
    // Default: per-row max-subtract / scalar-libm exp / normalize, order unchanged
    // (byte-identical). `FW_SIMD_EXP` swaps the exp numerator for the AVX2 poly
    // (owner-gated, non-byte-exact). Rows are independent, so fan out over contiguous
    // row bands (disjoint slices of `x.data`); threshold in elements keeps small
    // score matrices serial.
    let use_simd = simd_exp_softmax_enabled();
    let softmax_row = move |row: &mut [f32]| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            // All -inf (e.g. fully masked row): leave as-is to avoid NaNs.
            return;
        }
        let sum = if use_simd {
            softmax_row_poly_numer(row, max)
        } else {
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
            sum
        };
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
/// 1-D convolution with the weight in ggml `[Cout, Cin*K]` order. Transposes the weight
/// to `[Cin*K, Cout]` and delegates to [`conv1d_wt`]. For the encoder stem the conv weights
/// are CONSTANT across windows — prefer pre-transposing once at load and calling
/// [`conv1d_wt`] directly; this entry re-transposes on every call (kept for the standalone
/// API and as the byte-exact reference for [`conv1d_wt`]).
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
    let patch = cin * k;
    if w.len() != cout * patch {
        return Err(FwError::InvalidRequest(format!(
            "conv1d: weight len {} != cout*cin*k = {}",
            w.len(),
            cout * patch
        )));
    }
    // Reshape/transpose weights [Cout, Cin*K] -> w_t [Cin*K, Cout].
    let mut w_t = vec![0.0f32; patch * cout];
    for co in 0..cout {
        for j in 0..patch {
            w_t[j * cout + co] = w[co * patch + j];
        }
    }
    conv1d_wt(
        x,
        &Mat::from_vec(patch, cout, w_t),
        cin,
        k,
        bias,
        stride,
        pad,
    )
}

/// 1-D convolution taking the weight ALREADY transposed to `[Cin*K, Cout]` (the layout
/// [`matmul_bias`] consumes) — skips the per-call weight transpose. The encoder stem
/// pre-transposes its constant conv weights once at load ([`EncoderWeights::from_ggml`])
/// and calls this, avoiding a redundant ~15 ms/window strided transpose on turbo (conv2).
/// Byte-identical to [`conv1d`] fed the same weight (the transpose is a pure permutation).
pub fn conv1d_wt(
    x: &Mat,
    w_t: &Mat,
    cin: usize,
    k: usize,
    bias: &[f32],
    stride: usize,
    pad: usize,
) -> FwResult<Mat> {
    let patch = cin * k;
    let cout = w_t.cols;
    if x.cols != cin {
        return Err(FwError::InvalidRequest(format!(
            "conv1d: x.cols {} != cin {cin}",
            x.cols
        )));
    }
    if w_t.rows != patch {
        return Err(FwError::InvalidRequest(format!(
            "conv1d: w_t rows {} != cin*k = {patch}",
            w_t.rows
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
            for (wi, band) in cols.chunks_mut(band_rows * patch).enumerate() {
                let o_base = wi * band_rows;
                s.spawn(move || {
                    for (i, row) in band.chunks_mut(patch).enumerate() {
                        fill_row(o_base + i, row);
                    }
                });
            }
        });
    }
    let im2col = Mat::from_vec(t_out, patch, cols);

    // [T_out, Cin*K] x [Cin*K, Cout] -> [T_out, Cout], then add bias.
    matmul_bias(&im2col, w_t, Some(bias))
}

/// f16 KV-cache storage gate (`FW_KV_F16=1`, default off). Stores the
/// self-attention key/value cache as f16 instead of f32 — HALF the per-step
/// DRAM read of the cache (self_attn = 10.7% of decode, 35% of it the cache
/// read). Proven TRANSCRIPT-NEUTRAL: the f16-rounded values are bit-identical to
/// the `FW_KV_F16_SIM` round-through-f16 probe (which was byte-identical to f32
/// in both modes), because `f16→f32` is exact — so `k16[d].to_f32()` equals the
/// sim's stored value exactly. f16 is finer than the already-neutral int8 decode
/// weights. Read at [`KvCache::new`]; process-wide.
fn kv_f16_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FW_KV_F16").is_some())
}

/// Incremental key/value cache for autoregressive self-attention.
///
/// Stores keys and values as contiguous `[capacity_tokens, n_state]` row-
/// major buffers; [`KvCache::append`] copies new per-token rows in and
/// advances `len`. [`KvCache::keys`] / [`KvCache::values`] expose the
/// populated prefix as a `[len, n_state]` [`Mat`] for [`attention`].
///
/// When [`kv_f16_enabled`] is on the storage is f16 (`k16`/`v16`, `k`/`v`
/// empty), halving the per-step cache-read bandwidth; otherwise f32
/// (`k`/`v`, `k16`/`v16` empty). The mode is fixed at construction.
#[derive(Debug, Clone)]
pub struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    // Optional key mirror `[state, capacity_tokens]`. The token-major copy is
    // retained for prefill and fallback; the mirror makes the single-token
    // score loop contiguous across independent cached tokens.
    k_columns: Vec<f32>,
    k16: Vec<Float16>,
    v16: Vec<Float16>,
    f16: bool,
    len: usize,
    capacity_tokens: usize,
    n_state: usize,
}

impl KvCache {
    /// Allocate a cache for up to `capacity_tokens` tokens of width
    /// `n_state`.
    #[must_use]
    pub fn new(capacity_tokens: usize, n_state: usize) -> Self {
        let f16 = kv_f16_enabled();
        Self::new_with_layout(capacity_tokens, n_state, f16, false)
    }

    /// Construct the historical f32 token-major cache for same-binary A/B.
    #[doc(hidden)]
    #[must_use]
    pub fn new_row_major_keys_for_bench(capacity_tokens: usize, n_state: usize) -> Self {
        Self::new_with_layout(capacity_tokens, n_state, false, false)
    }

    /// Construct the packed-key f32 candidate for same-binary A/B.
    #[doc(hidden)]
    #[must_use]
    pub fn new_column_major_keys_for_bench(capacity_tokens: usize, n_state: usize) -> Self {
        Self::new_with_layout(capacity_tokens, n_state, false, true)
    }

    fn new_with_layout(
        capacity_tokens: usize,
        n_state: usize,
        f16: bool,
        column_keys: bool,
    ) -> Self {
        let n = capacity_tokens * n_state;
        Self {
            k: if f16 { Vec::new() } else { vec![0.0; n] },
            v: if f16 { Vec::new() } else { vec![0.0; n] },
            k_columns: if column_keys && !f16 {
                vec![0.0; n]
            } else {
                Vec::new()
            },
            k16: if f16 {
                vec![Float16::from_f32(0.0); n]
            } else {
                Vec::new()
            },
            v16: if f16 {
                vec![Float16::from_f32(0.0); n]
            } else {
                Vec::new()
            },
            f16,
            len: 0,
            capacity_tokens,
            n_state,
        }
    }

    /// Whether this cache stores f16 (vs f32).
    #[must_use]
    pub fn is_f16(&self) -> bool {
        self.f16
    }

    /// Borrow the populated f16 key prefix (`[len, n_state]` row-major). Only
    /// valid when [`Self::is_f16`]; panics otherwise (empty `k16`).
    #[must_use]
    pub fn key_slice_f16(&self) -> &[Float16] {
        &self.k16[..self.len * self.n_state]
    }

    /// Borrow the populated f16 value prefix. See [`Self::key_slice_f16`].
    #[must_use]
    pub fn value_slice_f16(&self) -> &[Float16] {
        &self.v16[..self.len * self.n_state]
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
        if self.f16 {
            // f16 storage: convert on append (halves the per-step cache-read DRAM).
            for (dst, &src) in self.k16[off..off + span].iter_mut().zip(&k.data) {
                *dst = Float16::from_f32(src);
            }
            for (dst, &src) in self.v16[off..off + span].iter_mut().zip(&v.data) {
                *dst = Float16::from_f32(src);
            }
        } else {
            self.k[off..off + span].copy_from_slice(&k.data);
            self.v[off..off + span].copy_from_slice(&v.data);
            if !self.k_columns.is_empty() {
                for r in 0..t {
                    let token = self.len + r; // ubs:ignore — cache position, not a secret
                    let src = &k.data[r * self.n_state..(r + 1) * self.n_state];
                    for (d, &value) in src.iter().enumerate() {
                        self.k_columns[d * self.capacity_tokens + token] = value;
                    }
                }
            }
        }
        self.len += t;
        Ok(())
    }

    /// View of the cached keys as a `[len, n_state]` matrix (dequantized from
    /// f16 when the cache is f16).
    #[must_use]
    pub fn keys(&self) -> Mat {
        let span = self.len * self.n_state;
        let data = if self.f16 {
            self.k16[..span].iter().map(|h| h.to_f32()).collect()
        } else {
            self.k[..span].to_vec()
        };
        Mat::from_vec(self.len, self.n_state, data)
    }

    /// View of the cached values as a `[len, n_state]` matrix (dequantized from
    /// f16 when the cache is f16).
    #[must_use]
    pub fn values(&self) -> Mat {
        let span = self.len * self.n_state;
        let data = if self.f16 {
            self.v16[..span].iter().map(|h| h.to_f32()).collect()
        } else {
            self.v[..span].to_vec()
        };
        Mat::from_vec(self.len, self.n_state, data)
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

    fn key_columns(&self) -> Option<&[f32]> {
        (!self.k_columns.is_empty()).then_some(self.k_columns.as_slice())
    }

    /// Restore a populated prefix without modifying its bytes. Benchmark-only:
    /// repeated arms exclude cache construction while exercising real append.
    #[doc(hidden)]
    pub fn truncate_for_bench(&mut self, len: usize) {
        assert!(len <= self.len, "KvCache benchmark truncate grows cache");
        self.len = len;
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
/// Maddubs i7 GEMM (`x @ w^T + bias`) that writes its `[m, out]` result DIRECTLY in
/// head-major `[hh, m, d_head]` layout (`dst[(o/d_head)*m*d_head + r*d_head + o%d_head]`)
/// — the FUSED form of `matmul_bias_i7_quantized` + `sdpa_gather_head_major`, so the
/// external SDPA can consume it with NO separate transpose pass. Byte-IDENTICAL values
/// to the two-step path (same i32 maddubs dot + same f32 dequant order; only the write
/// target is permuted). Same M4×N2 register blocking; parallel over 4-row blocks with a
/// disjoint scatter (each block owns row-range `r0..r0+4`, whose `r*d_head` offset inside
/// every head is unique across blocks).
#[allow(unsafe_code)]
fn maddubs_i7_headmajor(
    x: &I7Activation,
    w: &I7Mat,
    bias: Option<&[f32]>,
    dst: &mut [f32],
    hh: usize,
    d_head: usize,
) -> FwResult<()> {
    let m = x.rows;
    let inp = x.inp;
    let out = w.out;
    if inp != w.inp {
        return Err(FwError::InvalidRequest(format!(
            "maddubs_i7_headmajor: x.inp {inp} != w.inp {}",
            w.inp
        )));
    }
    if out != hh * d_head {
        return Err(FwError::InvalidRequest(format!(
            "maddubs_i7_headmajor: out {out} != hh*d_head {}",
            hh * d_head
        )));
    }
    if dst.len() != hh * m * d_head {
        return Err(FwError::InvalidRequest(format!(
            "maddubs_i7_headmajor: dst len {} != hh*m*d_head {}",
            dst.len(),
            hh * m * d_head
        )));
    }
    let xu = &x.data;
    let sa = &x.scale;
    let tq = m;
    // Pass the base pointer as usize (Send+Sync, captured by copy) so the rayon
    // closure can scatter into disjoint head-major slots without a wrapper type.
    let base_addr = dst.as_mut_ptr() as usize;
    let n_blocks = m.div_ceil(4);
    (0..n_blocks).into_par_iter().for_each(|blk| {
        let base = base_addr as *mut f32;
        // Scatter (row r, col o) to head-major. SAFETY: h*tq*d_head + r*d_head + d is in
        // 0..hh*m*d_head; blocks own disjoint r ⇒ disjoint writes (no data race).
        let put = |r: usize, o: usize, val: f32| {
            let h = o / d_head;
            let d = o % d_head;
            unsafe { *base.add(h * tq * d_head + r * d_head + d) = val };
        };
        let r0 = blk * 4;
        let rows = (m - r0).min(4);
        if rows == 4 {
            let x0 = &xu[r0 * inp..(r0 + 1) * inp];
            let x1 = &xu[(r0 + 1) * inp..(r0 + 2) * inp];
            let x2 = &xu[(r0 + 2) * inp..(r0 + 3) * inp];
            let x3 = &xu[(r0 + 3) * inp..(r0 + 4) * inp];
            let (s0, s1, s2, s3) = (sa[r0], sa[r0 + 1], sa[r0 + 2], sa[r0 + 3]);
            let mut o = 0;
            if i7_m2n4_enabled() {
                while o + 4 <= out {
                    let w0r = &w.data[o * inp..(o + 1) * inp];
                    let w1r = &w.data[(o + 1) * inp..(o + 2) * inp];
                    let w2r = &w.data[(o + 2) * inp..(o + 3) * inp];
                    let w3r = &w.data[(o + 3) * inp..(o + 4) * inp];
                    let raw01 = dot_maddubs_i7_m2n4(x0, x1, w0r, w1r, w2r, w3r);
                    let raw23 = dot_maddubs_i7_m2n4(x2, x3, w0r, w1r, w2r, w3r);
                    let off0 = 128 * w.colsum[o];
                    let off1 = 128 * w.colsum[o + 1];
                    let off2 = 128 * w.colsum[o + 2];
                    let off3 = 128 * w.colsum[o + 3];
                    let (sc0, sc1, sc2, sc3) =
                        (w.scale[o], w.scale[o + 1], w.scale[o + 2], w.scale[o + 3]);
                    let (bo0, bo1, bo2, bo3) = bias.map_or((0.0, 0.0, 0.0, 0.0), |b| {
                        (b[o], b[o + 1], b[o + 2], b[o + 3])
                    });
                    put(r0, o, (raw01[0] - off0) as f32 * s0 * sc0 + bo0);
                    put(r0 + 1, o, (raw01[1] - off0) as f32 * s1 * sc0 + bo0);
                    put(r0 + 2, o, (raw23[0] - off0) as f32 * s2 * sc0 + bo0);
                    put(r0 + 3, o, (raw23[1] - off0) as f32 * s3 * sc0 + bo0);
                    put(r0, o + 1, (raw01[2] - off1) as f32 * s0 * sc1 + bo1);
                    put(r0 + 1, o + 1, (raw01[3] - off1) as f32 * s1 * sc1 + bo1);
                    put(r0 + 2, o + 1, (raw23[2] - off1) as f32 * s2 * sc1 + bo1);
                    put(r0 + 3, o + 1, (raw23[3] - off1) as f32 * s3 * sc1 + bo1);
                    put(r0, o + 2, (raw01[4] - off2) as f32 * s0 * sc2 + bo2);
                    put(r0 + 1, o + 2, (raw01[5] - off2) as f32 * s1 * sc2 + bo2);
                    put(r0 + 2, o + 2, (raw23[4] - off2) as f32 * s2 * sc2 + bo2);
                    put(r0 + 3, o + 2, (raw23[5] - off2) as f32 * s3 * sc2 + bo2);
                    put(r0, o + 3, (raw01[6] - off3) as f32 * s0 * sc3 + bo3);
                    put(r0 + 1, o + 3, (raw01[7] - off3) as f32 * s1 * sc3 + bo3);
                    put(r0 + 2, o + 3, (raw23[6] - off3) as f32 * s2 * sc3 + bo3);
                    put(r0 + 3, o + 3, (raw23[7] - off3) as f32 * s3 * sc3 + bo3);
                    o += 4;
                }
            }
            while o + 2 <= out {
                let w0r = &w.data[o * inp..(o + 1) * inp];
                let w1r = &w.data[(o + 1) * inp..(o + 2) * inp];
                let raw = dot_maddubs_i7_m4n2(x0, x1, x2, x3, w0r, w1r);
                let off0 = 128 * w.colsum[o];
                let off1 = 128 * w.colsum[o + 1];
                let (sc0, sc1) = (w.scale[o], w.scale[o + 1]);
                let (bo0, bo1) = bias.map_or((0.0, 0.0), |b| (b[o], b[o + 1]));
                put(r0, o, (raw[0] - off0) as f32 * s0 * sc0 + bo0);
                put(r0 + 1, o, (raw[1] - off0) as f32 * s1 * sc0 + bo0);
                put(r0 + 2, o, (raw[2] - off0) as f32 * s2 * sc0 + bo0);
                put(r0 + 3, o, (raw[3] - off0) as f32 * s3 * sc0 + bo0);
                put(r0, o + 1, (raw[4] - off1) as f32 * s0 * sc1 + bo1);
                put(r0 + 1, o + 1, (raw[5] - off1) as f32 * s1 * sc1 + bo1);
                put(r0 + 2, o + 1, (raw[6] - off1) as f32 * s2 * sc1 + bo1);
                put(r0 + 3, o + 1, (raw[7] - off1) as f32 * s3 * sc1 + bo1);
                o += 2;
            }
            while o < out {
                let wrow = &w.data[o * inp..(o + 1) * inp];
                let raw = dot_maddubs_i7_m4(x0, x1, x2, x3, wrow);
                let off = 128 * w.colsum[o];
                let sc = w.scale[o];
                let bo = bias.map_or(0.0, |b| b[o]);
                put(r0, o, (raw[0] - off) as f32 * s0 * sc + bo);
                put(r0 + 1, o, (raw[1] - off) as f32 * s1 * sc + bo);
                put(r0 + 2, o, (raw[2] - off) as f32 * s2 * sc + bo);
                put(r0 + 3, o, (raw[3] - off) as f32 * s3 * sc + bo);
                o += 1;
            }
        } else {
            for j in 0..rows {
                let r = r0 + j;
                let xr = &xu[r * inp..(r + 1) * inp];
                let sar = sa[r];
                for o in 0..out {
                    let wrow = &w.data[o * inp..(o + 1) * inp];
                    let dot = dot_maddubs_i7(xr, wrow) - 128 * w.colsum[o];
                    let mut val = dot as f32 * sar * w.scale[o];
                    if let Some(b) = bias {
                        val += b[o];
                    }
                    put(r, o, val);
                }
            }
        }
    });
    Ok(())
}

#[allow(unsafe_code)]
fn maddubs_i7_headmajor_block(
    x: &I7Activation,
    w: &I7Mat,
    bias: Option<&[f32]>,
    base_addr: usize,
    hh: usize,
    d_head: usize,
    r0: usize,
) {
    let m = x.rows;
    let inp = x.inp;
    let out = w.out;
    let xu = &x.data;
    let sa = &x.scale;
    let base = base_addr as *mut f32;
    let put = |r: usize, o: usize, val: f32| {
        let h = o / d_head;
        let d = o % d_head;
        unsafe { *base.add(h * m * d_head + r * d_head + d) = val };
    };
    let rows = (m - r0).min(4);
    if rows == 4 {
        let x0 = &xu[r0 * inp..(r0 + 1) * inp];
        let x1 = &xu[(r0 + 1) * inp..(r0 + 2) * inp];
        let x2 = &xu[(r0 + 2) * inp..(r0 + 3) * inp];
        let x3 = &xu[(r0 + 3) * inp..(r0 + 4) * inp];
        let (s0, s1, s2, s3) = (sa[r0], sa[r0 + 1], sa[r0 + 2], sa[r0 + 3]);
        let mut o = 0;
        if i7_m2n4_enabled() {
            while o + 4 <= out {
                let w0r = &w.data[o * inp..(o + 1) * inp];
                let w1r = &w.data[(o + 1) * inp..(o + 2) * inp];
                let w2r = &w.data[(o + 2) * inp..(o + 3) * inp];
                let w3r = &w.data[(o + 3) * inp..(o + 4) * inp];
                let raw01 = dot_maddubs_i7_m2n4(x0, x1, w0r, w1r, w2r, w3r);
                let raw23 = dot_maddubs_i7_m2n4(x2, x3, w0r, w1r, w2r, w3r);
                let off0 = 128 * w.colsum[o];
                let off1 = 128 * w.colsum[o + 1];
                let off2 = 128 * w.colsum[o + 2];
                let off3 = 128 * w.colsum[o + 3];
                let (sc0, sc1, sc2, sc3) =
                    (w.scale[o], w.scale[o + 1], w.scale[o + 2], w.scale[o + 3]);
                let (bo0, bo1, bo2, bo3) = bias.map_or((0.0, 0.0, 0.0, 0.0), |b| {
                    (b[o], b[o + 1], b[o + 2], b[o + 3])
                });
                put(r0, o, (raw01[0] - off0) as f32 * s0 * sc0 + bo0);
                put(r0 + 1, o, (raw01[1] - off0) as f32 * s1 * sc0 + bo0);
                put(r0 + 2, o, (raw23[0] - off0) as f32 * s2 * sc0 + bo0);
                put(r0 + 3, o, (raw23[1] - off0) as f32 * s3 * sc0 + bo0);
                put(r0, o + 1, (raw01[2] - off1) as f32 * s0 * sc1 + bo1);
                put(r0 + 1, o + 1, (raw01[3] - off1) as f32 * s1 * sc1 + bo1);
                put(r0 + 2, o + 1, (raw23[2] - off1) as f32 * s2 * sc1 + bo1);
                put(r0 + 3, o + 1, (raw23[3] - off1) as f32 * s3 * sc1 + bo1);
                put(r0, o + 2, (raw01[4] - off2) as f32 * s0 * sc2 + bo2);
                put(r0 + 1, o + 2, (raw01[5] - off2) as f32 * s1 * sc2 + bo2);
                put(r0 + 2, o + 2, (raw23[4] - off2) as f32 * s2 * sc2 + bo2);
                put(r0 + 3, o + 2, (raw23[5] - off2) as f32 * s3 * sc2 + bo2);
                put(r0, o + 3, (raw01[6] - off3) as f32 * s0 * sc3 + bo3);
                put(r0 + 1, o + 3, (raw01[7] - off3) as f32 * s1 * sc3 + bo3);
                put(r0 + 2, o + 3, (raw23[6] - off3) as f32 * s2 * sc3 + bo3);
                put(r0 + 3, o + 3, (raw23[7] - off3) as f32 * s3 * sc3 + bo3);
                o += 4;
            }
        }
        while o + 2 <= out {
            let w0r = &w.data[o * inp..(o + 1) * inp];
            let w1r = &w.data[(o + 1) * inp..(o + 2) * inp];
            let raw = dot_maddubs_i7_m4n2(x0, x1, x2, x3, w0r, w1r);
            let off0 = 128 * w.colsum[o];
            let off1 = 128 * w.colsum[o + 1];
            let (sc0, sc1) = (w.scale[o], w.scale[o + 1]);
            let (bo0, bo1) = bias.map_or((0.0, 0.0), |b| (b[o], b[o + 1]));
            put(r0, o, (raw[0] - off0) as f32 * s0 * sc0 + bo0);
            put(r0 + 1, o, (raw[1] - off0) as f32 * s1 * sc0 + bo0);
            put(r0 + 2, o, (raw[2] - off0) as f32 * s2 * sc0 + bo0);
            put(r0 + 3, o, (raw[3] - off0) as f32 * s3 * sc0 + bo0);
            put(r0, o + 1, (raw[4] - off1) as f32 * s0 * sc1 + bo1);
            put(r0 + 1, o + 1, (raw[5] - off1) as f32 * s1 * sc1 + bo1);
            put(r0 + 2, o + 1, (raw[6] - off1) as f32 * s2 * sc1 + bo1);
            put(r0 + 3, o + 1, (raw[7] - off1) as f32 * s3 * sc1 + bo1);
            o += 2;
        }
        while o < out {
            let wrow = &w.data[o * inp..(o + 1) * inp];
            let raw = dot_maddubs_i7_m4(x0, x1, x2, x3, wrow);
            let off = 128 * w.colsum[o];
            let sc = w.scale[o];
            let bo = bias.map_or(0.0, |b| b[o]);
            put(r0, o, (raw[0] - off) as f32 * s0 * sc + bo);
            put(r0 + 1, o, (raw[1] - off) as f32 * s1 * sc + bo);
            put(r0 + 2, o, (raw[2] - off) as f32 * s2 * sc + bo);
            put(r0 + 3, o, (raw[3] - off) as f32 * s3 * sc + bo);
            o += 1;
        }
    } else {
        for j in 0..rows {
            let r = r0 + j;
            let xr = &xu[r * inp..(r + 1) * inp];
            let sar = sa[r];
            for o in 0..out {
                let wrow = &w.data[o * inp..(o + 1) * inp];
                let dot = dot_maddubs_i7(xr, wrow) - 128 * w.colsum[o];
                let mut val = dot as f32 * sar * w.scale[o];
                if let Some(b) = bias {
                    val += b[o];
                }
                put(r, o, val);
            }
        }
    }
    debug_assert_eq!(out, hh * d_head);
}

#[allow(clippy::too_many_arguments)]
fn maddubs_i7_qkv_headmajor(
    hq: &I7Activation,
    qw: &I7Mat,
    q_bias: Option<&[f32]>,
    kw: &I7Mat,
    k_bias: Option<&[f32]>,
    vw: &I7Mat,
    v_bias: Option<&[f32]>,
    qa: &mut [f32],
    ka: &mut [f32],
    va: &mut [f32],
    hh: usize,
    d_head: usize,
) -> FwResult<()> {
    let m = hq.rows;
    let inp = hq.inp;
    let out = hh * d_head;
    for (name, w, bias) in [("q", qw, q_bias), ("k", kw, k_bias), ("v", vw, v_bias)] {
        if w.inp != inp {
            return Err(FwError::InvalidRequest(format!(
                "maddubs_i7_qkv_headmajor: {name}.inp {} != x.inp {inp}",
                w.inp
            )));
        }
        if w.out != out {
            return Err(FwError::InvalidRequest(format!(
                "maddubs_i7_qkv_headmajor: {name}.out {} != hh*d_head {out}",
                w.out
            )));
        }
        if let Some(b) = bias
            && b.len() != out
        {
            return Err(FwError::InvalidRequest(format!(
                "maddubs_i7_qkv_headmajor: {name} bias len {} != out {out}",
                b.len()
            )));
        }
    }
    let want = hh * m * d_head;
    for (name, dst) in [("q", qa.len()), ("k", ka.len()), ("v", va.len())] {
        if dst != want {
            return Err(FwError::InvalidRequest(format!(
                "maddubs_i7_qkv_headmajor: {name} dst len {dst} != hh*m*d_head {want}"
            )));
        }
    }

    let q_addr = qa.as_mut_ptr() as usize;
    let k_addr = ka.as_mut_ptr() as usize;
    let v_addr = va.as_mut_ptr() as usize;
    let n_blocks = m.div_ceil(4);
    (0..n_blocks).into_par_iter().for_each(|blk| {
        let r0 = blk * 4;
        maddubs_i7_headmajor_block(hq, qw, q_bias, q_addr, hh, d_head, r0);
        maddubs_i7_headmajor_block(hq, kw, k_bias, k_addr, hh, d_head, r0);
        maddubs_i7_headmajor_block(hq, vw, v_bias, v_addr, hh, d_head, r0);
    });
    Ok(())
}

/// FUSED int8-QKV + fused SDPA: run the three maddubs i7 GEMMs writing q/k/v DIRECTLY in
/// head-major `[hh, tq, d_head]` (eliminating `sdpa_gather_head_major`), feed the external
/// `ft_kernel_cpu::sdpa_forward_f32`, then scatter back to `[tq, n_state]`. Byte-IDENTICAL
/// to `matmul_bias_i7_quantized ×3` + [`attention`] on the SDPA branch (same dots, same
/// head-major permutation ⇒ identical SDPA inputs ⇒ identical output). Encoder-only
/// (bidirectional self-attention). The head-major GEMM write (stride `d_head`) sidesteps
/// the gather's strided READ (stride `n_state`) that the ledger measured as a DRAM-latency
/// floor. `hq` is the shared quantized activation; `*_bias` mirror q/k/v (k has none).
///
/// # Errors
/// [`FwError::InvalidRequest`] on shape mismatch.
pub fn attention_from_i7_qkv(
    hq: &I7Activation,
    qw: &I7Mat,
    q_bias: Option<&[f32]>,
    kw: &I7Mat,
    k_bias: Option<&[f32]>,
    vw: &I7Mat,
    v_bias: Option<&[f32]>,
    n_head: usize,
) -> FwResult<Mat> {
    let tq = hq.rows;
    let n_state = qw.out;
    if n_head == 0 || !n_state.is_multiple_of(n_head) {
        return Err(FwError::InvalidRequest(format!(
            "attention_from_i7_qkv: n_head {n_head} must divide n_state {n_state}"
        )));
    }
    let d_head = n_state / n_head;
    let hh = n_head;
    let mut qa = gemv_out_buf(hh * tq * d_head);
    let mut ka = gemv_out_buf(hh * tq * d_head);
    let mut va = gemv_out_buf(hh * tq * d_head);
    if i7_qkv_headmajor_rowco_enabled() {
        maddubs_i7_qkv_headmajor(
            hq, qw, q_bias, kw, k_bias, vw, v_bias, &mut qa, &mut ka, &mut va, hh, d_head,
        )?;
    } else {
        maddubs_i7_headmajor(hq, qw, q_bias, &mut qa, hh, d_head)?;
        maddubs_i7_headmajor(hq, kw, k_bias, &mut ka, hh, d_head)?;
        maddubs_i7_headmajor(hq, vw, v_bias, &mut va, hh, d_head)?;
    }
    let sdpa_scale = (d_head as f32).powf(-0.5);
    let o = ft_kernel_cpu::sdpa_forward_f32(
        &qa, &ka, &va, hh, tq, tq, d_head, d_head, sdpa_scale, false,
    );
    let mut out = gemv_out_buf(tq * n_state);
    sdpa_scatter_interleaved(&mut out, &o, hh, tq, d_head, n_state, sdpa_gather_chunks());
    Ok(Mat::from_vec(tq, n_state, out))
}

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

// THROWAWAY: split the fused-SDPA encoder attention into gather/kernel/scatter
// wall-time (gated on FRANKEN_WHISPER_PERF_SPANS). Drained + printed by the
// encoder profiler. Zero cost when perf spans are off.
thread_local! {
    static SDPA_SPLIT: std::cell::RefCell<[u128; 3]> = const { std::cell::RefCell::new([0; 3]) };
}
pub(crate) fn drain_sdpa_split() -> [u128; 3] {
    SDPA_SPLIT.with(|p| {
        let v = *p.borrow();
        *p.borrow_mut() = [0; 3];
        v
    })
}
fn sdpa_split_add(i: usize, ns: u128) {
    SDPA_SPLIT.with(|p| p.borrow_mut()[i] += ns);
}

/// Target chunk count for the fused-SDPA q/k/v gather AND output scatter, tunable via
/// `FW_SDPA_GATHER_CHUNKS`. **Default `16`** splits BOTH reshape passes into 16 balanced
/// row-bands. Setting `0` restores the historical per-op chunking (gather: one band per
/// head; scatter: one band per output row) — byte-identical output either way (pure data
/// movement, unit-tested chunk-invariant).
///
/// QUIET-BOX MEASURED, real turbo encoder (jfk, `FRANKEN_WHISPER_PERF_SPANS=1`, min-of-9
/// interleaved + a 12/16/24/32 min-of-5 sweep, 2026-07-04, BlackThrush): the win is
/// **entirely in the SCATTER**. The legacy per-row scatter uses `tq`=1500 fine rayon bands
/// (massive oversubscription) ⇒ 16 bands cut it **~1.6×** (42.4→25.9 ms summed over 32
/// layers). The GATHER is FLAT (~80 ms both ways: legacy already uses n_head=20 coarse
/// bands ≈ 16 — the cold-probe's "gather 1.73×" was the shared-box artifact flagged in
/// 470fb79, it does NOT reproduce on the real quiet encoder). Net reshape **1.159×**
/// (120.9→104.3 ms/window); 16 and 24 tie on the plateau (12 worse, 32 regresses). Reshape
/// is ~5% of the encoder so this is ~0.7% of encode ≈ ~0.5% e2e, byte-exact, encoder-side
/// (NOT pipelining-hidden), always-positive (never regressed in 14 reps). See
/// NEGATIVE_EVIDENCE 2026-07-04 and `sdpa_gather_head_major` / the chunk-invariant tests.
fn sdpa_gather_chunks() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("FW_SDPA_GATHER_CHUNKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16)
    })
}

/// Gather `src` (interleaved `[t, n_state]`, row stride `n_state`) into the head-major
/// `dst` (`[hh, t, d_head]`): `dst[h*t*d_head + i*d_head + d] = src[i*n_state + h*d_head + d]`.
/// Parallelized over `chunks` balanced row-bands of the flat `[hh*t, d_head]` output
/// (each output row copied whole). `chunks == 0` ⇒ one band per head (== the historical
/// `par_chunks_mut(t*d_head)`, bit-identical). Pure data movement — output is independent
/// of `chunks`. See [`sdpa_gather_chunks`] / `sdpa_gather_head_major_chunk_invariant`.
fn sdpa_gather_head_major(
    dst: &mut [f32],
    src: &[f32],
    hh: usize,
    t: usize,
    d_head: usize,
    n_state: usize,
    chunks: usize,
) {
    let total_rows = hh * t;
    let n = if chunks == 0 {
        hh
    } else {
        chunks.min(total_rows).max(1)
    };
    let chunk_rows = total_rows.div_ceil(n).max(1);
    dst.par_chunks_mut(chunk_rows * d_head)
        .enumerate()
        .for_each(|(c, blk)| {
            let row0 = c * chunk_rows;
            for (local, out_row) in blk.chunks_mut(d_head).enumerate() {
                let r = row0 + local;
                let h = r / t;
                let i = r % t;
                let base = i * n_state + h * d_head;
                out_row.copy_from_slice(&src[base..base + d_head]);
            }
        });
}

/// Inverse of [`sdpa_gather_head_major`]: scatter head-major `o` (`[hh, t, d_head]`) into
/// interleaved `out` (`[t, n_state]`): `out[i*n_state + h*d_head + d] = o[h*t*d_head + i*d_head + d]`.
/// Parallelized over `chunks` balanced output-row bands (`chunks == 0` ⇒ one band per output
/// row, == the historical `par_chunks_mut(n_state)` scatter, bit-identical). Pure data
/// movement — output independent of `chunks`. See `sdpa_scatter_interleaved_chunk_invariant`.
fn sdpa_scatter_interleaved(
    out: &mut [f32],
    o: &[f32],
    hh: usize,
    t: usize,
    d_head: usize,
    n_state: usize,
    chunks: usize,
) {
    let n = if chunks == 0 { t } else { chunks.min(t).max(1) };
    let rows_per = t.div_ceil(n).max(1);
    out.par_chunks_mut(rows_per * n_state)
        .enumerate()
        .for_each(|(c, blk)| {
            let i0 = c * rows_per;
            for (local, orow) in blk.chunks_mut(n_state).enumerate() {
                let i = i0 + local;
                for h in 0..hh {
                    orow[h * d_head..(h + 1) * d_head].copy_from_slice(
                        &o[h * t * d_head + i * d_head..h * t * d_head + i * d_head + d_head],
                    );
                }
            }
        });
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
        let split = crate::native_engine::perf_spans_enabled();
        macro_rules! st {
            ($i:expr, $b:expr) => {{
                if split {
                    let __t = std::time::Instant::now();
                    let __r = $b;
                    sdpa_split_add($i, __t.elapsed().as_nanos());
                    __r
                } else {
                    $b
                }
            }};
        }
        let hh = n_head;
        let mut qa = gemv_out_buf(hh * tq * d_head);
        let mut ka = gemv_out_buf(hh * tk * d_head);
        let mut va = gemv_out_buf(hh * tk * d_head);
        // The per-head gather/scatter is a strided memcpy transpose (interleaved
        // [tq, n_state] <-> head-major [hh, tq, d_head]). It is ~20% of attn_sdpa
        // and BANDWIDTH-bound: serial (one core) was MEASURED 4.5x SLOWER. Chunk count
        // is `FW_SDPA_GATHER_CHUNKS` (default 16, bit-identical to legacy; set 0 for the
        // historical per-op chunking). The gather is ~flat 16-vs-20-bands; the WIN is the
        // scatter (see below). Quiet-box measured — see `sdpa_gather_chunks` doc.
        let gchunks = sdpa_gather_chunks();
        st!(0, {
            sdpa_gather_head_major(&mut qa, &q.data, hh, tq, d_head, n_state, gchunks);
            sdpa_gather_head_major(&mut ka, k, hh, tk, d_head, n_state, gchunks);
            sdpa_gather_head_major(&mut va, v, hh, tk, d_head, n_state, gchunks);
        });
        let sdpa_scale = (d_head as f32).powf(-0.5);
        let o = st!(
            1,
            ft_kernel_cpu::sdpa_forward_f32(
                &qa, &ka, &va, hh, tq, tk, d_head, d_head, sdpa_scale, false,
            )
        );
        // Scatter head-major `o` back to interleaved `out` — same FW_SDPA_GATHER_CHUNKS
        // knob (default 16). This is where the win is: the legacy per-row scatter used
        // tq=1500 fine rayon bands (oversubscribed) → 16 bands is ~1.6× faster on the real
        // quiet encoder (42.4→25.9 ms/window, byte-identical). See NEGATIVE_EVIDENCE 2026-07-04.
        st!(
            2,
            sdpa_scatter_interleaved(&mut out, &o, hh, tq, d_head, n_state, gchunks)
        );
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
    if cache.is_f16() {
        // f16 storage: read the half-width cache DIRECTLY in the hot path
        // (`.to_f32()` per element is lossless), halving the per-step cache-read
        // DRAM. f16→f32 is exact, so `k16.to_f32()` equals the value the
        // `FW_KV_F16_SIM` probe fed the f32 kernel (round-through-f16) — hence
        // the transcript matches the proven-neutral SIM result bit-for-bit.
        if q.rows == 1 && fast_self_attn_enabled() {
            return attention_decode_step_f16(
                q,
                cache.key_slice_f16(),
                cache.value_slice_f16(),
                tk,
                n_head,
            );
        }
        // Prefill (rare `tq > 1`): dequant the f16 prefix to f32 scratch once,
        // then the standard raw path. This is the multi-token prefill, not the
        // per-step hot path, so the one-time dequant is amortized (and we do NOT
        // dequant-to-scratch per step — that would revive the memmove the
        // alloc-light rewrite killed).
        let k_f32: Vec<f32> = cache.key_slice_f16().iter().map(|h| h.to_f32()).collect();
        let v_f32: Vec<f32> = cache.value_slice_f16().iter().map(|h| h.to_f32()).collect();
        return attention_raw(q, &k_f32, &v_f32, tk, n_head, Some(past_len));
    }
    if q.rows == 1 && fast_self_attn_enabled() {
        if let Some(k_columns) = cache.key_columns() {
            return attention_decode_step_column_keys(
                q,
                k_columns,
                cache.value_slice(),
                tk,
                n_head,
                cache.capacity_tokens,
            );
        }
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
fn attention_decode_step(q: &Mat, k: &[f32], v: &[f32], tk: usize, n_head: usize) -> FwResult<Mat> {
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
        // AVX2 `axpy_f32_into` vectorizes across the INDEPENDENT output slots `d`
        // (separate mul+add, NOT fmadd) so the per-slot j-ascending sum is
        // bit-identical to the scalar `*o += sj*vd` loop it replaces.
        for (j, &sj) in scores.iter().enumerate() {
            let vrow = &v[j * n_state + base..j * n_state + base + d_head];
            let orow = &mut out[base..base + d_head];
            axpy_f32_into(orow, sj, vrow);
        }
    }
    Ok(Mat::from_vec(1, n_state, out))
}

/// Packed-key variant of [`attention_decode_step`]. Keys are mirrored as
/// `[n_state, capacity_tokens]`, so the d-outer score loop reads contiguous
/// tokens. Each independent `scores[j]` still receives the same products in
/// ascending `d` order, preserving the exact f32 reduction while exposing
/// vector parallelism across tokens.
#[inline(never)]
fn attention_decode_step_column_keys(
    q: &Mat,
    k_columns: &[f32],
    v: &[f32],
    tk: usize,
    n_head: usize,
    capacity_tokens: usize,
) -> FwResult<Mat> {
    let n_state = q.cols;
    if n_head == 0 || !n_state.is_multiple_of(n_head) {
        return Err(FwError::InvalidRequest(format!(
            "attention: n_head {n_head} must divide n_state {n_state}"
        )));
    }
    if k_columns.len() != capacity_tokens * n_state || v.len() != tk * n_state {
        return Err(FwError::InvalidRequest(format!(
            "attention: column-k/v slice len {}/{} != capacity*n_state/tk*n_state {}/{}",
            k_columns.len(),
            v.len(),
            capacity_tokens * n_state,
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
        for (d, slot) in qh.iter_mut().enumerate() {
            *slot = q0[base + d] * scale;
        }
        scores.fill(0.0);
        for (d, &qd) in qh.iter().enumerate() {
            let column =
                &k_columns[(base + d) * capacity_tokens..(base + d) * capacity_tokens + tk];
            for (score, &key) in scores.iter_mut().zip(column) {
                *score += qd * (key * scale);
            }
        }
        let mut sm = Mat::from_vec(1, tk, std::mem::take(&mut scores));
        softmax_rows(&mut sm);
        scores = sm.data;
        for (j, &sj) in scores.iter().enumerate() {
            let vrow = &v[j * n_state + base..j * n_state + base + d_head];
            let orow = &mut out[base..base + d_head];
            axpy_f32_into(orow, sj, vrow);
        }
    }
    Ok(Mat::from_vec(1, n_state, out))
}

/// f16-storage variant of [`attention_decode_step`]: reads the KV cache f16
/// slices directly, `.to_f32()`-ing each element inside the two dot loops.
/// f16→f32 is lossless, so every f32 arithmetic value is IDENTICAL to
/// [`attention_decode_step`] fed the same keys/values rounded through f16 — i.e.
/// exactly the `FW_KV_F16_SIM` probe path, which is proven transcript-neutral.
/// The win is bandwidth: the cache read is half-width (f16), and it is read
/// straight out of storage (no dequant-to-f32-scratch memmove per step).
fn attention_decode_step_f16(
    q: &Mat,
    k: &[Float16],
    v: &[Float16],
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
        for (d, slot) in qh.iter_mut().enumerate() {
            *slot = q0[base + d] * scale;
        }
        // Same per-term product and summation order as `attention_decode_step`,
        // reading k as f16 (lossless `.to_f32()`).
        for (j, sj) in scores.iter_mut().enumerate() {
            let krow = &k[j * n_state + base..j * n_state + base + d_head];
            let mut acc = 0.0f32;
            for (d, &qd) in qh.iter().enumerate() {
                acc += qd * (krow[d].to_f32() * scale);
            }
            *sj = acc;
        }
        let mut sm = Mat::from_vec(1, tk, std::mem::take(&mut scores));
        softmax_rows(&mut sm);
        scores = sm.data;
        for (j, &sj) in scores.iter().enumerate() {
            let vrow = &v[j * n_state + base..j * n_state + base + d_head];
            let orow = &mut out[base..base + d_head];
            for (o, &vd) in orow.iter_mut().zip(vrow) {
                *o += sj * vd.to_f32();
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

    /// The f16-direct encoder quant (`quantize_f16_bytes_to_i7`, `c701b45`) MUST
    /// produce a bit-identical [`I7Mat`] to the two-phase path
    /// `quantize_mat_to_i7(transpose(f16→f32))` — the byte-exactness the −2.31 GB /
    /// 1.68×-weight-build f16-direct win relies on. This guards it in the DEFAULT
    /// suite (the live path only runs under `FW_ENC_FREE_F32=1`, so nothing else here
    /// exercises it). Basis: ggml row `o` of the `[out, inp]` bytes IS column `o` of
    /// the transposed `[inp, out]` Mat (`w_t.data[i*out + o]`), so both, feeding the
    /// same `quantize_rows_to_i7`, must agree bit-for-bit.
    #[test]
    fn quantize_f16_bytes_matches_transposed_f32_path_byte_exact() {
        let mut rng = Lcg::new(0xF16D_12EC7);
        // Non-square + edge shapes exercise the per-column EF chain, scales, colsums.
        for &(out, inp) in &[(40usize, 24usize), (17, 31), (64, 16), (1, 8), (8, 1)] {
            // Synthetic ggml [out, inp] f16 raw bytes (finite, varied).
            let mut raw = vec![0u8; out * inp * 2];
            for b2 in raw.chunks_exact_mut(2) {
                let v = rng.next_f32() * 4.0; // finite f16 range
                b2.copy_from_slice(&Float16::from_f32(v).to_bits().to_le_bytes());
            }
            // Two-phase reference: transpose f16→f32 into [inp, out], then quantize.
            let mut f32t = vec![0.0f32; inp * out];
            for o in 0..out {
                for i in 0..inp {
                    let off = (o * inp + i) * 2;
                    let v =
                        Float16::from_bits(u16::from_le_bytes([raw[off], raw[off + 1]])).to_f32();
                    f32t[i * out + o] = v; // transposed [inp, out], element [i*out + o]
                }
            }
            let a = quantize_mat_to_i7(&Mat::from_vec(inp, out, f32t));
            let b = quantize_f16_bytes_to_i7(&raw, out, inp);
            assert_eq!(a.data, b.data, "i7 data mismatch at {out}x{inp}");
            assert_eq!(a.scale, b.scale, "scale mismatch at {out}x{inp}");
            assert_eq!(a.colsum, b.colsum, "colsum mismatch at {out}x{inp}");
            assert_eq!(
                (a.out, a.inp),
                (b.out, b.inp),
                "dims mismatch at {out}x{inp}"
            );
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

    #[test]
    fn i7_prequantized_activation_matches_inline_quantize() {
        let mut rng = Lcg::new(0x17);
        let x = rng.mat(5, 37);
        let w_t = rng.mat(37, 9);
        let bias: Vec<f32> = (0..9).map(|_| rng.next_f32()).collect();
        let w = quantize_mat_to_i7(&w_t);

        let inline = matmul_bias_i7(&x, &w, Some(&bias)).expect("inline i7");
        let xq = quantize_act_i7(&x);
        let reused = matmul_bias_i7_quantized(&xq, &w, Some(&bias)).expect("reused i7");

        assert_eq!(inline.rows, reused.rows);
        assert_eq!(inline.cols, reused.cols);
        assert_eq!(inline.data, reused.data);
    }

    #[test]
    fn i7_bias_specialization_matches_runtime_option_bit_exact() {
        let mut rng = Lcg::new(0xb1a5);
        for (rows, inp, out) in [(5, 37, 9), (8, 32, 32), (4, 17, 41)] {
            let x = rng.mat(rows, inp);
            let xq = quantize_act_i7(&x);
            let w = quantize_mat_to_i7(&rng.mat(inp, out));
            let bias: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect();

            for runtime_bias in [None, Some(bias.as_slice())] {
                let historical = matmul_bias_i7_quantized_unspecialized(&xq, &w, runtime_bias)
                    .expect("historical runtime-option projection");
                let specialized = matmul_bias_i7_quantized(&xq, &w, runtime_bias)
                    .expect("const-specialized projection");
                let specialized_ab = matmul_bias_i7_quantized_specialized_ab(&xq, &w, runtime_bias)
                    .expect("non-inlined const-specialized projection");

                assert_eq!(
                    historical
                        .data
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    specialized
                        .data
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "rows={rows} inp={inp} out={out} bias={}",
                    runtime_bias.is_some()
                );
                assert_eq!(
                    specialized.data, specialized_ab.data,
                    "A/B entry point must execute the production specialized arm"
                );
            }
        }
    }

    #[test]
    fn qkv_headmajor_rowco_matches_three_passes() {
        let mut rng = Lcg::new(0x71);
        let (rows, inp, out, heads) = (5, 37, 8, 2);
        let d_head = out / heads;
        let x = rng.mat(rows, inp);
        let xq = quantize_act_i7(&x);
        let qw = quantize_mat_to_i7(&rng.mat(inp, out));
        let kw = quantize_mat_to_i7(&rng.mat(inp, out));
        let vw = quantize_mat_to_i7(&rng.mat(inp, out));
        let qb: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect();
        let vb: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect();

        let mut q0 = vec![0.0f32; heads * rows * d_head];
        let mut k0 = vec![0.0f32; heads * rows * d_head];
        let mut v0 = vec![0.0f32; heads * rows * d_head];
        let mut q1 = vec![0.0f32; heads * rows * d_head];
        let mut k1 = vec![0.0f32; heads * rows * d_head];
        let mut v1 = vec![0.0f32; heads * rows * d_head];

        maddubs_i7_headmajor(&xq, &qw, Some(&qb), &mut q0, heads, d_head).expect("q");
        maddubs_i7_headmajor(&xq, &kw, None, &mut k0, heads, d_head).expect("k");
        maddubs_i7_headmajor(&xq, &vw, Some(&vb), &mut v0, heads, d_head).expect("v");
        maddubs_i7_qkv_headmajor(
            &xq,
            &qw,
            Some(&qb),
            &kw,
            None,
            &vw,
            Some(&vb),
            &mut q1,
            &mut k1,
            &mut v1,
            heads,
            d_head,
        )
        .expect("qkv rowco");

        assert_eq!(q0, q1);
        assert_eq!(k0, k1);
        assert_eq!(v0, v1);
    }

    #[test]
    fn maddubs_i7_m2n4_matches_scalar_dots() {
        let mut rng = Lcg::new(0x2a4);
        let k = 73;
        let a0: Vec<u8> = (0..k).map(|_| (rng.next_u32() % 256) as u8).collect();
        let a1: Vec<u8> = (0..k).map(|_| (rng.next_u32() % 256) as u8).collect();
        let w0: Vec<i8> = (0..k).map(|_| (rng.next_u32() % 127) as i8 - 63).collect();
        let w1: Vec<i8> = (0..k).map(|_| (rng.next_u32() % 127) as i8 - 63).collect();
        let w2: Vec<i8> = (0..k).map(|_| (rng.next_u32() % 127) as i8 - 63).collect();
        let w3: Vec<i8> = (0..k).map(|_| (rng.next_u32() % 127) as i8 - 63).collect();

        let got = dot_maddubs_i7_m2n4(&a0, &a1, &w0, &w1, &w2, &w3);
        let want = [
            dot_maddubs_i7(&a0, &w0),
            dot_maddubs_i7(&a1, &w0),
            dot_maddubs_i7(&a0, &w1),
            dot_maddubs_i7(&a1, &w1),
            dot_maddubs_i7(&a0, &w2),
            dot_maddubs_i7(&a1, &w2),
            dot_maddubs_i7(&a0, &w3),
            dot_maddubs_i7(&a1, &w3),
        ];
        assert_eq!(got, want);
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
    fn quantize_act_i8_matches_scalar_reference() {
        // The AVX2 quantize must be BIT-identical to the scalar map for finite
        // activations, across all code paths (SIMD body, <8 tail) and the ±127 clamp
        // edges, and must round HALF-AWAY (f32::round), NOT round-to-even.
        fn scalar(x: &[f32], xinv: f32) -> Vec<i8> {
            x.iter()
                .map(|v| (v * xinv).round().clamp(-127.0, 127.0) as i8)
                .collect()
        }
        let mut s = 0x243F_6A88_85A3_08D3u64;
        let mut nf = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0
        };
        for &n in &[0usize, 1, 5, 7, 8, 9, 15, 16, 17, 1280, 5120] {
            let x: Vec<f32> = (0..n).map(|_| nf()).collect();
            let amax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
            let xinv = 127.0 / amax; // maps amax -> ±127 (exercise the clamp edge)
            let mut got = vec![0i8; n];
            quantize_act_i8_into(&x, xinv, &mut got);
            assert_eq!(got, scalar(&x, xinv), "quantize mismatch at n={n}");
        }
        // Exact-.5 inputs: half-AWAY (2.5->3), not round-to-even (which gives 2).
        let x = vec![0.5f32, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5, -126.5];
        let mut got = vec![0i8; x.len()];
        quantize_act_i8_into(&x, 1.0, &mut got);
        assert_eq!(got, vec![1i8, 2, 3, -1, -2, -3, 127, -127]);
    }

    #[test]
    fn quantize_row_i7_u8_matches_scalar_reference() {
        // The AVX2 i7 activation quant (maddubs u8 layout: `(i8 + 128) as u8`) must
        // be BIT-identical to the scalar map for finite activations, across the SIMD
        // body + <8 tail + ±127 clamp edges, rounding HALF-AWAY (f32::round). This is
        // the inner loop of quantize_act_i7 / quantize_act_i7_gelu.
        fn scalar(x: &[f32], inv: f32) -> Vec<u8> {
            x.iter()
                .map(|&v| ((v * inv).round().clamp(-127.0, 127.0) as i32 + 128) as u8)
                .collect()
        }
        let mut s = 0x853C_49E6_748F_EA9Bu64;
        let mut nf = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0
        };
        for &n in &[0usize, 1, 5, 7, 8, 9, 15, 16, 17, 1280, 5120] {
            let x: Vec<f32> = (0..n).map(|_| nf()).collect();
            let amax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
            let inv = 127.0 / amax; // maps amax -> ±127 (exercise the clamp edge)
            let mut got = vec![0u8; n];
            quantize_row_i7_u8_into(&x, inv, &mut got);
            assert_eq!(got, scalar(&x, inv), "i7 quantize mismatch at n={n}");
        }
        // Exact-.5 inputs: half-AWAY (2.5->3), not round-to-even; +128 offset applied.
        let x = vec![0.5f32, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5, -126.5];
        let mut got = vec![0u8; x.len()];
        quantize_row_i7_u8_into(&x, 1.0, &mut got);
        assert_eq!(
            got,
            vec![129u8, 130, 131, 127, 126, 125, 255, 1],
            "i7 u8 offset/round mismatch"
        );
    }

    // Single-binary kernel A/B: AVX2 `quantize_row_i7_u8_into` vs the inline scalar
    // `.round()` map it replaces, on the real turbo QKV/fc1 activation-row width
    // (n_state=1280). Run with:
    //   cargo test --release --lib quantize_row_i7_u8_perf -- --ignored --nocapture
    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn quantize_row_i7_u8_perf() {
        use std::time::Instant;
        let cols = 1280usize; // turbo n_state (QKV + fc1 input row width)
        let rows = 1500usize; // n_ctx
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut nf = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0
        };
        let x: Vec<f32> = (0..rows * cols).map(|_| nf()).collect();
        let inv = 127.0 / 4.0;
        let mut out = vec![0u8; rows * cols];
        let scalar = |src: &[f32], out: &mut [u8]| {
            for (d, &v) in out.iter_mut().zip(src) {
                let i8v = (v * inv).round().clamp(-127.0, 127.0) as i32;
                *d = (i8v + 128) as u8;
            }
        };
        let reps = 200usize;
        // Warm + interleaved ABBA to blunt order/thermal bias; report min (least-noisy).
        // black_box on the input row + output buffer defeats LTO dead-code elimination
        // (the result is otherwise unused → the whole loop folds to nothing).
        use std::hint::black_box;
        let (mut best_scalar, mut best_avx2) = (f64::MAX, f64::MAX);
        for _ in 0..reps {
            let t = Instant::now();
            for r in 0..rows {
                scalar(
                    black_box(&x[r * cols..(r + 1) * cols]),
                    &mut out[r * cols..(r + 1) * cols],
                );
            }
            black_box(&out);
            let sc = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for r in 0..rows {
                quantize_row_i7_u8_into(
                    black_box(&x[r * cols..(r + 1) * cols]),
                    inv,
                    &mut out[r * cols..(r + 1) * cols],
                );
            }
            black_box(&out);
            let av = t.elapsed().as_secs_f64();
            best_scalar = best_scalar.min(sc);
            best_avx2 = best_avx2.min(av);
        }
        eprintln!(
            "quantize_row_i7_u8 [{rows}x{cols}] scalar={:.1}us avx2={:.1}us speedup={:.2}x",
            best_scalar * 1e6,
            best_avx2 * 1e6,
            best_scalar / best_avx2
        );
    }

    #[test]
    fn quantize_f16_row_to_i8_matches_scalar_reference() {
        // The AVX2+F16C decoder weight-quant helper (amax reduce + round, one row)
        // must be BIT-identical to the scalar non-EF loop in quantize_f16_to_i8:
        // exact f16→f32, order-invariant amax, round HALF-AWAY (f32::round), ±127
        // clamp. Also verify the returned scale matches. Exercise SIMD body + <8 tail.
        fn scalar(w: &[Float16]) -> (Vec<i8>, f32) {
            let amax = w
                .iter()
                .map(|h| h.to_f32().abs())
                .fold(0.0f32, f32::max)
                .max(1e-9);
            let scale = amax / 127.0;
            let inv = 1.0 / scale;
            let q = w
                .iter()
                .map(|h| (h.to_f32() * inv).round().clamp(-127.0, 127.0) as i8)
                .collect();
            (q, scale)
        }
        let mut s = 0xD1B5_4A32_D192_ED03u64;
        let mut nf = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            Float16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
        };
        for &n in &[0usize, 1, 5, 7, 8, 9, 15, 16, 17, 1280, 5120] {
            let w: Vec<Float16> = (0..n).map(|_| nf()).collect();
            let mut got = vec![0i8; n];
            let gs = quantize_f16_row_to_i8_into(&w, &mut got);
            let (want, ws) = scalar(&w);
            assert_eq!(got, want, "f16→i8 quant mismatch at n={n}");
            assert_eq!(gs.to_bits(), ws.to_bits(), "f16→i8 scale mismatch at n={n}");
        }
        // Clamp edge: a row whose amax maps other values across the ±127 saturation,
        // and exact-.5 half-away via f16-representable 0.5/1.5/2.5.
        let w: Vec<Float16> = [0.5f32, 1.5, 2.5, -0.5, -2.5, 3.0, -3.0, 0.0]
            .iter()
            .map(|&v| Float16::from_f32(v))
            .collect();
        let mut got = vec![0i8; w.len()];
        let _ = quantize_f16_row_to_i8_into(&w, &mut got);
        let (want, _) = scalar(&w);
        assert_eq!(got, want, "f16→i8 clamp/half-away mismatch");
    }

    // Single-binary kernel A/B: AVX2+F16C quantize_f16_row_to_i8_into vs the scalar
    // non-EF loop it replaces, on a real turbo decoder attention-projection shape
    // ([1280,1280]). Run: cargo test --release --lib quantize_f16_row_to_i8_perf --
    // --ignored --nocapture
    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn quantize_f16_row_to_i8_perf() {
        use std::hint::black_box;
        use std::time::Instant;
        let (rows, cols) = (1280usize, 1280usize); // decoder attn projection [out,in]
        let mut s = 0x14D4_9C2A_7B01_55F1u64;
        let mut nf = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            Float16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
        };
        let w: Vec<Float16> = (0..rows * cols).map(|_| nf()).collect();
        let mut out = vec![0i8; rows * cols];
        let scalar = |wr: &[Float16], o: &mut [i8]| {
            let amax = wr
                .iter()
                .map(|h| h.to_f32().abs())
                .fold(0.0f32, f32::max)
                .max(1e-9);
            let inv = 127.0 / amax;
            for (d, h) in o.iter_mut().zip(wr) {
                *d = (h.to_f32() * inv).round().clamp(-127.0, 127.0) as i8;
            }
        };
        let (mut best_scalar, mut best_avx2) = (f64::MAX, f64::MAX);
        for _ in 0..200 {
            let t = Instant::now();
            for r in 0..rows {
                scalar(
                    black_box(&w[r * cols..(r + 1) * cols]),
                    &mut out[r * cols..(r + 1) * cols],
                );
            }
            black_box(&out);
            let sc = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for r in 0..rows {
                let _ = quantize_f16_row_to_i8_into(
                    black_box(&w[r * cols..(r + 1) * cols]),
                    &mut out[r * cols..(r + 1) * cols],
                );
            }
            black_box(&out);
            let av = t.elapsed().as_secs_f64();
            best_scalar = best_scalar.min(sc);
            best_avx2 = best_avx2.min(av);
        }
        eprintln!(
            "quantize_f16_row_to_i8 [{rows}x{cols}] scalar={:.1}us avx2={:.1}us speedup={:.2}x",
            best_scalar * 1e6,
            best_avx2 * 1e6,
            best_scalar / best_avx2
        );
    }

    #[test]
    fn quantize_f16_row_blocked_matches_scalar_reference() {
        // AVX2+F16C block-wise weight quant (per-block amax + round) must be BIT-
        // identical to the scalar block loop in quantize_f16_to_int_blocked, for both
        // i8 (max_level=127) and the i4-level (max_level=7) variants, across full and
        // PARTIAL trailing blocks and the ±max clamp. Verify both q AND per-block scale.
        fn scalar(wrow: &[Float16], block: usize, max_level: f32) -> (Vec<i8>, Vec<f32>) {
            let inp = wrow.len();
            let n_blocks = inp.div_ceil(block);
            let mut drow = vec![0i8; inp];
            let mut srow = vec![0.0f32; n_blocks];
            for b in 0..n_blocks {
                let s = b * block;
                let e = ((b + 1) * block).min(inp);
                let amax = wrow[s..e]
                    .iter()
                    .map(|h| h.to_f32().abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-9);
                let sc = amax / max_level;
                srow[b] = sc;
                let inv = 1.0 / sc;
                for i in s..e {
                    drow[i] = (wrow[i].to_f32() * inv)
                        .round()
                        .clamp(-max_level, max_level) as i8;
                }
            }
            (drow, srow)
        }
        let mut s = 0x2B99_2DDF_A232_47D9u64;
        let mut nf = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            Float16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
        };
        for &max_level in &[127.0f32, 7.0] {
            // include partial trailing blocks (80 = 32+32+16, 100 = 32*3+4)
            for &inp in &[0usize, 4, 16, 32, 33, 64, 80, 100, 5120] {
                let wrow: Vec<Float16> = (0..inp).map(|_| nf()).collect();
                let n_blocks = inp.div_ceil(32);
                let mut drow = vec![0i8; inp];
                let mut srow = vec![0.0f32; n_blocks];
                quantize_f16_row_blocked_to_int_into(&wrow, 32, max_level, &mut drow, &mut srow);
                let (wd, ws) = scalar(&wrow, 32, max_level);
                assert_eq!(drow, wd, "blocked q mismatch inp={inp} max={max_level}");
                let sb: Vec<u32> = srow.iter().map(|v| v.to_bits()).collect();
                let wb: Vec<u32> = ws.iter().map(|v| v.to_bits()).collect();
                assert_eq!(sb, wb, "blocked scale mismatch inp={inp} max={max_level}");
            }
        }
    }

    // Single-binary kernel A/B: AVX2 blocked weight quant vs the scalar block loop,
    // real turbo fc2 shape [1280,5120], block=32. Run: cargo test --release --lib
    // quantize_f16_row_blocked_perf -- --ignored --nocapture
    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn quantize_f16_row_blocked_perf() {
        use std::hint::black_box;
        use std::time::Instant;
        let (rows, cols, block) = (1280usize, 5120usize, 32usize); // decoder fc2 [out,in]
        let mut s = 0x6C62_272E_07BB_0142u64;
        let mut nf = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            Float16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0)
        };
        let w: Vec<Float16> = (0..rows * cols).map(|_| nf()).collect();
        let n_blocks = cols.div_ceil(block);
        let mut drow = vec![0i8; rows * cols];
        let mut srow = vec![0.0f32; rows * n_blocks];
        let scalar = |wrow: &[Float16], drow: &mut [i8], srow: &mut [f32]| {
            for b in 0..n_blocks {
                let (st, e) = (b * block, ((b + 1) * block).min(cols));
                let amax = wrow[st..e]
                    .iter()
                    .map(|h| h.to_f32().abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-9);
                let inv = 127.0 / amax;
                srow[b] = amax / 127.0;
                for i in st..e {
                    drow[i] = (wrow[i].to_f32() * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        };
        let (mut best_scalar, mut best_avx2) = (f64::MAX, f64::MAX);
        for _ in 0..100 {
            let t = Instant::now();
            for r in 0..rows {
                scalar(
                    black_box(&w[r * cols..(r + 1) * cols]),
                    &mut drow[r * cols..(r + 1) * cols],
                    &mut srow[r * n_blocks..(r + 1) * n_blocks],
                );
            }
            black_box(&drow);
            let sc = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for r in 0..rows {
                quantize_f16_row_blocked_to_int_into(
                    black_box(&w[r * cols..(r + 1) * cols]),
                    block,
                    127.0,
                    &mut drow[r * cols..(r + 1) * cols],
                    &mut srow[r * n_blocks..(r + 1) * n_blocks],
                );
            }
            black_box(&drow);
            let av = t.elapsed().as_secs_f64();
            best_scalar = best_scalar.min(sc);
            best_avx2 = best_avx2.min(av);
        }
        eprintln!(
            "quantize_f16_row_blocked [{rows}x{cols} blk{block}] scalar={:.1}us avx2={:.1}us speedup={:.2}x",
            best_scalar * 1e6,
            best_avx2 * 1e6,
            best_scalar / best_avx2
        );
    }

    #[test]
    fn axpy_f32_into_matches_scalar_reference() {
        // `axpy_f32_into` (the decode score·V output SAXPY) must be BIT-identical to
        // the scalar `*o += a*x` loop — separate mul+add (two roundings), NOT fmadd.
        // Exercise the SIMD body, the <8 tail, and repeated accumulation (the real
        // use accumulates over j into the same `o`). d_head=64 is the live shape.
        fn scalar(o: &mut [f32], a: f32, x: &[f32]) {
            for (oo, &xx) in o.iter_mut().zip(x) {
                *oo += a * xx;
            }
        }
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut nf = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0
        };
        for &n in &[0usize, 1, 7, 8, 9, 15, 16, 64, 65, 128] {
            let x: Vec<f32> = (0..n).map(|_| nf()).collect();
            let init: Vec<f32> = (0..n).map(|_| nf()).collect();
            let mut got = init.clone();
            let mut want = init;
            // Accumulate several SAXPYs (as the real j-loop does) — order must match.
            for _ in 0..5 {
                let a = nf();
                axpy_f32_into(&mut got, a, &x);
                scalar(&mut want, a, &x);
            }
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "axpy_f32_into bit mismatch at n={n}"
            );
        }
    }

    #[test]
    fn sdpa_gather_head_major_chunk_invariant() {
        // The gather output must be IDENTICAL for any chunk count (pure data movement)
        // and must equal the historical per-head reference. Covers chunks < / == / > hh
        // and a non-divisor of hh*t.
        let (hh, t, d_head) = (20usize, 37usize, 64usize);
        let n_state = hh * d_head;
        let mut s = 0x243F_6A88_85A3_08D3u64;
        let src: Vec<f32> = (0..t * n_state)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect();
        // Reference: the historical per-head gather.
        let mut want = vec![0.0f32; hh * t * d_head];
        for h in 0..hh {
            let base = h * d_head;
            for i in 0..t {
                want[h * t * d_head + i * d_head..h * t * d_head + (i + 1) * d_head]
                    .copy_from_slice(&src[i * n_state + base..i * n_state + base + d_head]);
            }
        }
        for &chunks in &[0usize, 1, 3, 7, 16, 20, 23, 64, 100000] {
            let mut got = vec![0.0f32; hh * t * d_head];
            sdpa_gather_head_major(&mut got, &src, hh, t, d_head, n_state, chunks);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "gather diverged at chunks={chunks}"
            );
        }
    }

    #[test]
    fn conv1d_wt_matches_conv1d() {
        // conv1d_wt fed the externally-transposed weight must be BIT-identical to conv1d
        // fed the ggml-order weight (the encoder pre-transposes at load via transpose_serial).
        let (cout, cin, k) = (12usize, 5usize, 3usize);
        let (t_in, stride, pad) = (17usize, 2usize, 1usize);
        let patch = cin * k;
        let mut s = 0x243F_6A88_85A3_08D3u64;
        let mut nf = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        };
        let x = Mat::from_vec(t_in, cin, (0..t_in * cin).map(|_| nf()).collect());
        let w: Vec<f32> = (0..cout * patch).map(|_| nf()).collect();
        let bias: Vec<f32> = (0..cout).map(|_| nf()).collect();
        // Pre-transpose w [cout, patch] -> [patch, cout] (transpose_serial == conv1d's inline).
        let w_t = Mat::from_vec(patch, cout, transpose_serial(&w, cout, patch));
        let a = conv1d(&x, &w, cout, cin, k, &bias, stride, pad).unwrap();
        let b = conv1d_wt(&x, &w_t, cin, k, &bias, stride, pad).unwrap();
        assert_eq!((a.rows, a.cols), (b.rows, b.cols));
        assert_eq!(
            a.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "conv1d_wt diverged from conv1d"
        );
    }

    #[test]
    fn sdpa_scatter_interleaved_chunk_invariant() {
        // Scatter output must be IDENTICAL for any chunk count and equal the per-row
        // reference (pure data movement). Covers chunks < / == / > t and non-divisors.
        let (hh, t, d_head) = (20usize, 37usize, 64usize);
        let n_state = hh * d_head;
        let mut s = 0xD1B5_4A32_D192_ED03u64;
        let o: Vec<f32> = (0..hh * t * d_head)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32
            })
            .collect();
        // Reference: the historical per-row scatter.
        let mut want = vec![0.0f32; t * n_state];
        for i in 0..t {
            for h in 0..hh {
                want[i * n_state + h * d_head..i * n_state + (h + 1) * d_head].copy_from_slice(
                    &o[h * t * d_head + i * d_head..h * t * d_head + i * d_head + d_head],
                );
            }
        }
        for &chunks in &[0usize, 1, 3, 12, 16, 37, 50, 1000] {
            let mut got = vec![0.0f32; t * n_state];
            sdpa_scatter_interleaved(&mut got, &o, hh, t, d_head, n_state, chunks);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "scatter diverged at chunks={chunks}"
            );
        }
    }

    #[test]
    fn dot_i8_matches_scalar_reference() {
        // The AVX2 `dot_i8` (x86) must be BIT-identical to the scalar reference for
        // every decode contraction length, across all three code paths (32-wide,
        // 16-wide tail, <16 scalar tail). Integer-exact: i8·i8 ∈ [-16129,16129] and
        // n ≤ 5120 ⇒ |Σ| ≤ 82.6M < 2³¹, so no i32 overflow and the vectorized
        // pairwise sum equals the scalar sum exactly (guards the gemv_i8 win landing).
        fn scalar(w: &[i8], x: &[i8]) -> i32 {
            let mut acc = 0i32;
            for (a, b) in w.iter().zip(x) {
                acc += (*a as i32) * (*b as i32);
            }
            acc
        }
        let mut s = 0xC0FF_EE12_3456_789Au64;
        let mut ni8 = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 24) as i32 % 255 - 127) as i8
        };
        for &n in &[
            0usize, 1, 7, 15, 16, 17, 31, 32, 33, 47, 63, 64, 384, 1280, 5120,
        ] {
            let w: Vec<i8> = (0..n).map(|_| ni8()).collect();
            let x: Vec<i8> = (0..n).map(|_| ni8()).collect();
            assert_eq!(dot_i8(&w, &x), scalar(&w, &x), "dot_i8 mismatch at n={n}");
        }
        // Worst-case magnitude at the max decode length.
        let w = vec![127i8; 5120];
        let x = vec![-127i8; 5120];
        assert_eq!(dot_i8(&w, &x), scalar(&w, &x));
        assert_eq!(dot_i8(&w, &x), -82_580_480); // 5120 · 127 · (−127)
    }

    #[test]
    fn dot_i8_4col_matches_four_dot_i8() {
        // The 4-token column tile must be BIT-IDENTICAL to four independent `dot_i8`
        // calls, for every contraction length across all three code paths (32-wide,
        // 16-wide tail, <16 scalar tail). This is the byte-exactness guarantee the
        // `FW_I8_BATCH_4COL` wire-in relies on: `gemv_i8_batch`'s 4col branch reuses
        // the exact per-(o,t) dequant of the 2col/dot_i8 branches, so once each column
        // of `dot_i8_4col` equals `dot_i8`, the whole path is byte-identical.
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut ni8 = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 24) as i32 % 255 - 127) as i8
        };
        for &n in &[
            0usize, 1, 7, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 384, 1280, 5120,
        ] {
            let w: Vec<i8> = (0..n).map(|_| ni8()).collect();
            let xa: Vec<i8> = (0..n).map(|_| ni8()).collect();
            let xb: Vec<i8> = (0..n).map(|_| ni8()).collect();
            let xc: Vec<i8> = (0..n).map(|_| ni8()).collect();
            let xd: Vec<i8> = (0..n).map(|_| ni8()).collect();
            let (a, b, c, d) = dot_i8_4col(&w, &xa, &xb, &xc, &xd);
            assert_eq!(a, dot_i8(&w, &xa), "col a mismatch at n={n}");
            assert_eq!(b, dot_i8(&w, &xb), "col b mismatch at n={n}");
            assert_eq!(c, dot_i8(&w, &xc), "col c mismatch at n={n}");
            assert_eq!(d, dot_i8(&w, &xd), "col d mismatch at n={n}");
        }
        // Worst-case magnitude: all four columns at the ±127 clamp edge, max length.
        let w = vec![127i8; 5120];
        let xn = vec![-127i8; 5120];
        let xp = vec![127i8; 5120];
        let (a, b, c, d) = dot_i8_4col(&w, &xn, &xp, &xn, &xp);
        assert_eq!(
            (a, b, c, d),
            (-82_580_480, 82_580_480, -82_580_480, 82_580_480)
        );
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
    fn gemv_f16_batch_row_morsel_equals_per_token_gemv() {
        let mut rng = Lcg::new(18);
        let (out, inp, tq) = (17usize, 19usize, 7usize);
        let (w_h, _w_f32) = rand_f16_weight(&mut rng, out, inp);
        let x: Vec<f32> = (0..tq * inp).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect();

        let mut batch = vec![0.0f32; tq * out];
        gemv_f16_batch_rows(
            &w_h,
            out,
            inp,
            &x,
            tq,
            Some(&bias),
            &mut batch,
            3,
            f16c_dot_available(),
        );

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
                    "row-morsel batch[{t},{o}] differs from per-token gemv"
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
            let g =
                0.5 * f * (1.0 + (GELU_SQRT_2_OVER_PI * f * (1.0 + GELU_COEF_A * f * f)).tanh());
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

    /// The gated `FW_SIMD_EXP` poly softmax numerator must match the scalar libm
    /// softmax within a tight tolerance, over lengths that exercise the AVX2 body +
    /// the `< 8` scalar tail, and must map `-inf`/NaN lanes to 0 exactly like scalar.
    #[test]
    fn softmax_row_poly_numer_matches_scalar() {
        let mut rng = Lcg::new(0xB1AC_5017);
        // Reference scalar softmax over one row (in place), returns the row.
        let scalar = |v: &[f32]| -> Vec<f32> {
            let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if !max.is_finite() {
                return v.to_vec();
            }
            let mut row: Vec<f32> = v
                .iter()
                .map(|&x| {
                    let e = (x - max).exp();
                    if e.is_finite() { e } else { 0.0 }
                })
                .collect();
            let sum: f32 = row.iter().sum();
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for r in &mut row {
                    *r *= inv;
                }
            }
            row
        };
        // Lengths hit 1..=40 (tails 0..7) plus the 1500-wide cross-attn shape.
        for &len in &[1usize, 4, 7, 8, 9, 15, 16, 33, 64, 128, 1500] {
            let base: Vec<f32> = (0..len).map(|_| rng.next_f32() * 6.0).collect();
            let want = scalar(&base);
            let mut got = base.clone();
            let max = got.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum = super::softmax_row_poly_numer(&mut got, max);
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for g in &mut got {
                    *g *= inv;
                }
            }
            let maxd = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(maxd < 1e-5, "len {len}: poly vs scalar max|Δ|={maxd:e}");
        }
        // Masked/NaN lanes -> 0 in the poly numerator (matches scalar finite-guard).
        let mut row = vec![
            1.0f32,
            f32::NEG_INFINITY,
            2.0,
            f32::NAN,
            0.5,
            3.0,
            -1.0,
            4.0,
            0.0,
        ];
        let max = 4.0f32;
        super::softmax_row_poly_numer(&mut row, max);
        assert_eq!(row[1], 0.0, "-inf lane -> 0");
        assert_eq!(row[3], 0.0, "NaN lane -> 0");
        assert!(
            row.iter().all(|v| v.is_finite()),
            "no NaN/inf in poly output"
        );
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
    fn matmul_raw_lhs_cpu_is_checked_and_uses_expected_layout() {
        let b = Mat::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let output =
            matmul_raw_lhs_cpu(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, &b).expect("CPU matmul");
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 2);
        assert_eq!(output.data, vec![58.0, 64.0, 139.0, 154.0]);
        assert!(matmul_raw_lhs_cpu(&[1.0, 2.0], 1, &b).is_err());
        assert!(matmul_raw_lhs_cpu(&[], 0, &b).is_err());
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
    fn column_major_key_cache_matches_row_major_turbo_bit_exact() {
        // large-v3-turbo decoder geometry at the maximum text-cache depth.
        // This drives both layouts through production `attention_with_cache`,
        // including append, the score kernel, softmax, and score-times-V.
        const N_STATE: usize = 1280;
        const N_HEAD: usize = 20;
        const CAPACITY: usize = 448;
        const PREFILL: usize = CAPACITY - 1;

        let mut rng = Lcg::new(0xc011_0a7e);
        let prefill_k = rng.mat(PREFILL, N_STATE);
        let prefill_v = rng.mat(PREFILL, N_STATE);
        let q = rng.mat(1, N_STATE);
        let k_new = rng.mat(1, N_STATE);
        let v_new = rng.mat(1, N_STATE);

        let mut row_major = KvCache::new_row_major_keys_for_bench(CAPACITY, N_STATE);
        let mut column_major = KvCache::new_column_major_keys_for_bench(CAPACITY, N_STATE);
        row_major.append(&prefill_k, &prefill_v).unwrap();
        column_major.append(&prefill_k, &prefill_v).unwrap();

        let expected = attention_with_cache(&q, &k_new, &v_new, N_HEAD, &mut row_major).unwrap();
        let actual = attention_with_cache(&q, &k_new, &v_new, N_HEAD, &mut column_major).unwrap();
        assert_eq!(row_major.len(), CAPACITY);
        assert_eq!(column_major.len(), CAPACITY);
        assert_eq!(
            expected
                .data
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            actual
                .data
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "packed K changed a large-v3-turbo attention output bit"
        );
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

    /// The PACKED int4 GEMV must be BIT-IDENTICAL to the unpacked int4 probe
    /// (`quantize_f16_to_i4_blocked` + `gemv_i8w_f32a_blocked`): same 4-bit values,
    /// same block scales, same `dot_i8w_f32` accumulation order — only the storage
    /// (2 nibbles/byte vs 1 value/byte) differs. This is the load-independent proof
    /// that swapping mlp_0/fc1 to the packed kernel changes no output bit. Covers
    /// several `inp` widths (d_model ∈ {384, 768, 1280}) and includes a bias.
    #[test]
    fn i4_packed_gemv_bit_identical_to_probe() {
        let mut rng = Lcg::new(0xF16C_4B17);
        for &inp in &[384usize, 768, 1280] {
            let out = 96; // small out; must exceed the narrow worker cap? no — serial ok
            let w: Vec<Float16> = (0..out * inp)
                .map(|_| Float16::from_f32(rng.next_f32() * 0.4))
                .collect();
            let x: Vec<f32> = (0..inp).map(|_| rng.next_f32()).collect();
            let bias: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect();

            let probe = quantize_f16_to_i4_blocked(&w, out, inp, 32);
            let mut y_probe = vec![0.0f32; out];
            gemv_i8w_f32a_blocked(&probe, &x, Some(&bias), &mut y_probe);

            let packed = quantize_f16_to_i4_packed(&w, out, inp);
            let mut y_packed = vec![0.0f32; out];
            gemv_i4_packed_f32a(&packed, &x, Some(&bias), &mut y_packed);

            // Storage really is halved (one byte per two int4 weights).
            assert_eq!(packed.data.len(), out * inp / 2, "packed size (inp={inp})");
            assert_eq!(probe.data.len(), out * inp, "probe stores one value/byte");
            // Every output element bit-for-bit equal.
            for o in 0..out {
                assert_eq!(
                    y_packed[o].to_bits(),
                    y_probe[o].to_bits(),
                    "i4-packed != probe at o={o}, inp={inp}: {} vs {}",
                    y_packed[o],
                    y_probe[o]
                );
            }
        }
    }

    #[test]
    fn quantized_cohort_gemvs_match_scalar_rows_bit_exact() {
        let mut rng = Lcg::new(0x0C04_027B_A7C4);
        let (tq, out, inp) = (5usize, 96usize, 384usize);
        let weights: Vec<Float16> = (0..out * inp)
            .map(|_| Float16::from_f32(rng.next_f32() * 0.4))
            .collect();
        let x: Vec<f32> = (0..tq * inp).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..out).map(|_| rng.next_f32()).collect();

        let i8 = quantize_f16_to_i8(&weights, out, inp);
        let mut i8_scalar = vec![0.0f32; tq * out];
        for t in 0..tq {
            gemv_i8(
                &i8,
                &x[t * inp..(t + 1) * inp],
                Some(&bias),
                &mut i8_scalar[t * out..(t + 1) * out],
            );
        }
        let mut i8_cohort = vec![0.0f32; tq * out];
        gemv_i8_cohort(&i8, &x, tq, Some(&bias), &mut i8_cohort);
        assert_eq!(
            i8_cohort
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            i8_scalar
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "int8 cohort changed a scalar row"
        );

        let mut f16_scalar = vec![0.0f32; tq * out];
        for t in 0..tq {
            gemv_f16(
                &weights,
                out,
                inp,
                &x[t * inp..(t + 1) * inp],
                Some(&bias),
                &mut f16_scalar[t * out..(t + 1) * out],
            );
        }
        let mut f16_cohort = vec![0.0f32; tq * out];
        gemv_f16_cohort(&weights, out, inp, &x, tq, Some(&bias), &mut f16_cohort);
        assert_eq!(
            f16_cohort
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            f16_scalar
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "f16 cohort changed a scalar row"
        );

        let blocked = quantize_f16_to_i8_blocked(&weights, out, inp, 32);
        let mut blocked_scalar = vec![0.0f32; tq * out];
        for t in 0..tq {
            gemv_i8w_f32a_blocked(
                &blocked,
                &x[t * inp..(t + 1) * inp],
                Some(&bias),
                &mut blocked_scalar[t * out..(t + 1) * out],
            );
        }
        let mut blocked_batch = vec![0.0f32; tq * out];
        gemv_i8w_f32a_blocked_batch(&blocked, &x, tq, Some(&bias), &mut blocked_batch);
        assert_eq!(
            blocked_batch
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            blocked_scalar
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "block-int8 cohort changed a scalar row"
        );

        let packed = quantize_f16_to_i4_packed(&weights, out, inp);
        let mut packed_scalar = vec![0.0f32; tq * out];
        for t in 0..tq {
            gemv_i4_packed_f32a(
                &packed,
                &x[t * inp..(t + 1) * inp],
                Some(&bias),
                &mut packed_scalar[t * out..(t + 1) * out],
            );
        }
        let mut packed_batch = vec![0.0f32; tq * out];
        gemv_i4_packed_f32a_batch(&packed, &x, tq, Some(&bias), &mut packed_batch);
        assert_eq!(
            packed_batch
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            packed_scalar
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "packed-int4 cohort changed a scalar row"
        );
    }
}
