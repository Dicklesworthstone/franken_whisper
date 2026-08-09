#![feature(portable_simd)]
//! Throwaway attribution probe: WHERE does `ft_kernel_cpu::sdpa_forward_f32`
//! actually spend its time at the real turbo-encoder shape?
//!
//! The ledger (2026-07-07, NE:830) concluded "the encoder attention is NOT
//! exp-bound; the exp is a negligible fraction" — but the experiment behind that
//! claim swapped the WHOLE fused kernel for a franken per-head rewrite that
//! materializes the `[1500,1500]` scores. That rewrite lost 34% on materialization,
//! transpose, and allocation overhead, which MASKS the exp term. Adding poly-exp to the
//! slow rewrite moved nothing, and the corollary "no softmax-exp optimization can
//! speed encoder attention" was drawn from that.
//!
//! Nobody ever held the fused kernel's structure FIXED and swapped only its exp.
//! This probe does that, and also prices two dead stores the kernel performs:
//!
//!   `let mut sc = vec![0.0f32; br * seq_k];`   <- zeroed, then fully overwritten
//!                                                 by `sgemm_bt`, once PER BLOCK
//!   `let mut out = vec![0.0f32; ...];`         <- zeroed, then fully overwritten
//!
//! At the turbo shape that per-block calloc runs 20 heads x 24 blocks x 32 layers
//! = 15,360 times per encoder window, each 64 x 1500 x 4 B = 384 KiB.
//!
//! Measured quantities (all at the real shape, min-of-N, cv reported):
//!   T_alloc  : per-block `vec![0.0f32; br*seq_k]` alloc+zero, nothing else
//!   T_fill   : per-block memcpy of representative post-sgemm scores (the control)
//!   T_libm   : T_fill + the kernel's exact 3-pass softmax (scalar libm `exp`)
//!   T_noexp  : T_fill + the same 3 passes with `exp` replaced by a subtraction
//!   T_poly   : T_fill + the same 3 passes with an AVX2-width poly `exp`
//!
//! Derived:  softmax = T_libm - T_fill
//!           exp     = T_libm - T_noexp        (the term NE:830 called negligible)
//!           poly win= T_libm - T_poly
//!           dead0   = T_alloc
//!
//! A baseline call into the REAL `sdpa_forward_f32` validates the model against
//! the `sdpa_kernel` span from `FRANKEN_WHISPER_PERF_SPANS=1`.
//!
//! Usage: sdpa_softmax_attrib_probe [reps]

use std::hint::black_box;
use std::simd::num::{SimdFloat as _, SimdInt as _};
use std::simd::{Simd, StdFloat as _};
use std::time::Instant;

use rayon::prelude::*;

/// large-v3-turbo encoder attention shape.
const N_HEAD: usize = 20;
const SEQ: usize = 1500;
const D_HEAD: usize = 64;
const N_LAYER: usize = 32;
/// The kernel's row-block height (`sdpa_forward_f32`'s `BR`).
const BR: usize = 64;
/// Encoder rayon cap (see project_encoder_thread_cap_win).
const THREADS: usize = 32;

const LANES: usize = 8;
type F32s = Simd<f32, LANES>;
type I32s = Simd<i32, LANES>;

/// The `(br, seq_k)` shape of every block the kernel processes, in kernel order.
fn blocks() -> Vec<usize> {
    let per_head: Vec<usize> = (0..SEQ.div_ceil(BR))
        .map(|b| (b * BR + BR).min(SEQ) - b * BR)
        .collect();
    // heads x blocks x layers -- the full per-window block population.
    per_head
        .iter()
        .copied()
        .cycle()
        .take(per_head.len() * N_HEAD * N_LAYER)
        .collect()
}

/// `exp(x)` for `x <= 0`, 5th-order minimax on the reduced argument — the same
/// scheme as the crate's landed `softmax_row_poly_numer` (FW_SIMD_EXP).
#[inline]
#[allow(
    clippy::approx_constant,
    reason = "the attribution probe mirrors the production polynomial coefficients exactly"
)]
fn exp_poly(x: F32s) -> F32s {
    const LOG2E: f32 = 1.442_695;
    const LN2: f32 = 0.693_147_2;
    const LO: f32 = -87.3365;

    let x = x.simd_max(F32s::splat(LO));
    // round-half-up via floor(y + 0.5): one vroundps, no scalarized `roundf`
    // (see project_round_doesnt_vectorize).
    let kf = (x * F32s::splat(LOG2E) + F32s::splat(0.5)).floor();
    let r = x - kf * F32s::splat(LN2);

    let mut p = F32s::splat(0.008_333_33);
    p = p * r + F32s::splat(0.041_666_66);
    p = p * r + F32s::splat(0.166_666_67);
    p = p * r + F32s::splat(0.5);
    p = p * r + F32s::splat(1.0);
    p = p * r + F32s::splat(1.0);

    // 2^k by direct exponent construction.
    let k: I32s = kf.cast::<i32>();
    let two_k = F32s::from_bits(((k + I32s::splat(127)) << I32s::splat(23)).cast::<u32>());
    p * two_k
}

/// The kernel's exact softmax over one `[br, seq_k]` block (scalar libm `exp`).
fn softmax_libm(sc: &mut [f32], br: usize, scale: f32) {
    for r in 0..br {
        let row = &mut sc[r * SEQ..(r + 1) * SEQ];
        let mut m = f32::NEG_INFINITY;
        for s in row.iter_mut() {
            *s *= scale;
            if *s > m {
                m = *s;
            }
        }
        let mut sum = 0.0f32;
        for s in row.iter_mut() {
            let e = (*s - m).exp();
            *s = e;
            sum += e;
        }
        for s in row.iter_mut() {
            *s /= sum;
        }
    }
}

/// Identical control with `exp` removed: isolates the transcendental's cost from
/// the three memory passes that carry it.
fn softmax_noexp(sc: &mut [f32], br: usize, scale: f32) {
    for r in 0..br {
        let row = &mut sc[r * SEQ..(r + 1) * SEQ];
        let mut m = f32::NEG_INFINITY;
        for s in row.iter_mut() {
            *s *= scale;
            if *s > m {
                m = *s;
            }
        }
        let mut sum = 0.0f32;
        for s in row.iter_mut() {
            let e = *s - m;
            *s = e;
            sum += e;
        }
        for s in row.iter_mut() {
            *s /= sum;
        }
    }
}

/// Same 3 passes, poly `exp`, SIMD max/sum. Non-byte-exact vs libm.
fn softmax_poly(sc: &mut [f32], br: usize, scale: f32) {
    for r in 0..br {
        let row = &mut sc[r * SEQ..(r + 1) * SEQ];
        let vscale = F32s::splat(scale);
        let mut vmax = F32s::splat(f32::NEG_INFINITY);
        let mut i = 0;
        while i + LANES <= SEQ {
            let v = F32s::from_slice(&row[i..]) * vscale;
            v.copy_to_slice(&mut row[i..]);
            vmax = vmax.simd_max(v);
            i += LANES;
        }
        let mut m = vmax.reduce_max();
        for s in &mut row[i..] {
            *s *= scale;
            if *s > m {
                m = *s;
            }
        }

        let vm = F32s::splat(m);
        let mut vsum = F32s::splat(0.0);
        let mut i = 0;
        while i + LANES <= SEQ {
            let e = exp_poly(F32s::from_slice(&row[i..]) - vm);
            e.copy_to_slice(&mut row[i..]);
            vsum += e;
            i += LANES;
        }
        let mut sum = vsum.reduce_sum();
        for s in &mut row[i..] {
            let e = (*s - m).exp();
            *s = e;
            sum += e;
        }

        let vinv = F32s::splat(1.0 / sum);
        let mut i = 0;
        while i + LANES <= SEQ {
            (F32s::from_slice(&row[i..]) * vinv).copy_to_slice(&mut row[i..]);
            i += LANES;
        }
        for s in &mut row[i..] {
            *s /= sum;
        }
    }
}

/// Pass 3 with a reciprocal multiply instead of a divide. Isolates the cost of
/// the kernel's 1.44 G scalar `*s /= sum` per window.
fn softmax_poly_recip(sc: &mut [f32], br: usize, scale: f32) {
    for r in 0..br {
        let row = &mut sc[r * SEQ..(r + 1) * SEQ];
        let sum = softmax_poly_row_unnorm(row, scale);
        let vinv = F32s::splat(1.0 / sum);
        let mut i = 0;
        while i + LANES <= SEQ {
            (F32s::from_slice(&row[i..]) * vinv).copy_to_slice(&mut row[i..]);
            i += LANES;
        }
        for s in &mut row[i..] {
            *s *= 1.0 / sum;
        }
    }
}

/// The 2-pass form: `scale` is folded into `q` before the scores GEMM (so pass 1
/// is a read-only max) and the normalize pass is DELETED — the caller instead
/// scales each `d_v`-wide output row of `O = P_unnorm @ V` by `1/sum`. That is
/// 64 multiplies per row instead of 1500, and one fewer full pass over `sc`.
/// This is the FlashAttention accumulation identity, exact up to rounding.
fn softmax_poly_2pass(sc: &mut [f32], br: usize, _scale: f32) {
    for r in 0..br {
        let row = &mut sc[r * SEQ..(r + 1) * SEQ];
        // pass 1: max only, read-only (no scale multiply, no store)
        let mut vmax = F32s::splat(f32::NEG_INFINITY);
        let mut i = 0;
        while i + LANES <= SEQ {
            vmax = vmax.simd_max(F32s::from_slice(&row[i..]));
            i += LANES;
        }
        let mut m = vmax.reduce_max();
        for s in &row[i..] {
            if *s > m {
                m = *s;
            }
        }
        // pass 2: exp + sum, leave UNNORMALIZED
        let vm = F32s::splat(m);
        let mut vsum = F32s::splat(0.0);
        let mut i = 0;
        while i + LANES <= SEQ {
            let e = exp_poly(F32s::from_slice(&row[i..]) - vm);
            e.copy_to_slice(&mut row[i..]);
            vsum += e;
            i += LANES;
        }
        let mut sum = vsum.reduce_sum();
        for s in &mut row[i..] {
            let e = (*s - m).exp();
            *s = e;
            sum += e;
        }
        // The caller would now do `o_row *= 1/sum` over d_v=64 elements; emulate
        // that cost so the comparison is honest rather than flattering.
        let inv = 1.0 / sum;
        let mut o = [1.0f32; 64];
        for x in &mut o {
            *x *= inv;
        }
        black_box(&o);
    }
}

/// Scale + max + exp + sum, leaving the row unnormalized; returns the row sum.
#[inline]
fn softmax_poly_row_unnorm(row: &mut [f32], scale: f32) -> f32 {
    let vscale = F32s::splat(scale);
    let mut vmax = F32s::splat(f32::NEG_INFINITY);
    let mut i = 0;
    while i + LANES <= SEQ {
        let v = F32s::from_slice(&row[i..]) * vscale;
        v.copy_to_slice(&mut row[i..]);
        vmax = vmax.simd_max(v);
        i += LANES;
    }
    let mut m = vmax.reduce_max();
    for s in &mut row[i..] {
        *s *= scale;
        if *s > m {
            m = *s;
        }
    }
    let vm = F32s::splat(m);
    let mut vsum = F32s::splat(0.0);
    let mut i = 0;
    while i + LANES <= SEQ {
        let e = exp_poly(F32s::from_slice(&row[i..]) - vm);
        e.copy_to_slice(&mut row[i..]);
        vsum += e;
        i += LANES;
    }
    let mut sum = vsum.reduce_sum();
    for s in &mut row[i..] {
        let e = (*s - m).exp();
        *s = e;
        sum += e;
    }
    sum
}

/// min / mean / cv% over `reps` timings of `f`, in ms.
fn bench(reps: usize, mut f: impl FnMut()) -> (f64, f64, f64) {
    let mut ms: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let min = ms.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    let var = ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ms.len() as f64;
    (min, mean, 100.0 * var.sqrt() / mean)
}

/// Run `body` once per block, in parallel, with a reusable per-thread scratch.
fn over_blocks(bs: &[usize], src: &[f32], body: impl Fn(&mut [f32], usize) + Sync) {
    bs.par_iter().for_each_init(
        || vec![0.0f32; BR * SEQ],
        |scratch, &br| {
            scratch[..br * SEQ].copy_from_slice(&src[..br * SEQ]);
            body(&mut scratch[..br * SEQ], br);
        },
    );
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);

    rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build_global()
        .expect("build rayon pool");

    let scale = (D_HEAD as f32).powf(-0.5);
    let bs = blocks();
    println!(
        "shape: n_head={N_HEAD} seq={SEQ} d_head={D_HEAD} n_layer={N_LAYER} BR={BR} \
         threads={THREADS}\nblocks/window={} reps={reps}\n",
        bs.len()
    );

    // Representative post-sgemm scores: q,k ~ U[-1,1], d=64 => s ~ N(0, 64/3).
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / 8388608.0 - 1.0) * 4.6
    };
    let src: Vec<f32> = (0..BR * SEQ).map(|_| rnd()).collect();

    // ---- baseline: the REAL kernel, one window's worth of layers ------------
    let qkv: Vec<f32> = (0..N_HEAD * SEQ * D_HEAD).map(|_| rnd() * 0.2).collect();
    let (real_min, real_mean, real_cv) = bench(reps.min(3), || {
        for _ in 0..N_LAYER {
            black_box(ft_kernel_cpu::sdpa_forward_f32(
                &qkv, &qkv, &qkv, N_HEAD, SEQ, SEQ, D_HEAD, D_HEAD, scale, false,
            ));
        }
    });
    println!("REAL sdpa_forward_f32 x{N_LAYER} layers (= one encoder window)");
    println!("  min {real_min:8.1} ms   mean {real_mean:8.1} ms   cv {real_cv:.1}%\n");

    // ---- T_alloc: the dead per-block calloc ---------------------------------
    let (alloc_min, _, alloc_cv) = bench(reps, || {
        bs.par_iter().for_each(|&br| {
            let sc = vec![0.0f32; br * SEQ];
            black_box(&sc);
        });
    });

    // ---- controls + softmax variants ---------------------------------------
    // NOTE: the control must OBSERVE the scratch, or LLVM dead-store-eliminates the
    // whole `copy_from_slice` and the "control" reads ~0 ms — which would silently
    // fold the probe's own 5.9 GB of fill traffic into every softmax variant.
    let (fill_min, _, fill_cv) = bench(reps, || {
        over_blocks(&bs, &src, |sc, _| {
            black_box(&*sc);
        });
    });
    let (libm_min, _, libm_cv) = bench(reps, || {
        over_blocks(&bs, &src, |sc, br| softmax_libm(sc, br, scale));
    });
    let (noexp_min, _, noexp_cv) = bench(reps, || {
        over_blocks(&bs, &src, |sc, br| softmax_noexp(sc, br, scale));
    });
    let (poly_min, _, poly_cv) = bench(reps, || {
        over_blocks(&bs, &src, |sc, br| softmax_poly(sc, br, scale));
    });
    let (recip_min, _, recip_cv) = bench(reps, || {
        over_blocks(&bs, &src, |sc, br| softmax_poly_recip(sc, br, scale));
    });
    let (p2_min, _, p2_cv) = bench(reps, || {
        over_blocks(&bs, &src, |sc, br| softmax_poly_2pass(sc, br, scale));
    });

    println!("per-window totals (min-of-{reps}, ms):");
    println!("  T_alloc  (dead calloc)   {alloc_min:8.1}   cv {alloc_cv:.1}%");
    println!("  T_fill   (control)       {fill_min:8.1}   cv {fill_cv:.1}%");
    println!("  T_libm   (fill+softmax)  {libm_min:8.1}   cv {libm_cv:.1}%");
    println!("  T_noexp  (fill+no exp)   {noexp_min:8.1}   cv {noexp_cv:.1}%");
    println!("  T_poly   (fill+poly exp) {poly_min:8.1}   cv {poly_cv:.1}%");
    println!("  T_recip  (poly, *1/sum)  {recip_min:8.1}   cv {recip_cv:.1}%");
    println!("  T_2pass  (poly, no norm) {p2_min:8.1}   cv {p2_cv:.1}%\n");
    println!("NEXT-FRAME candidates (both inside the already-gated poly path):");
    println!(
        "  divide -> reciprocal      saves {:8.1} ms/window",
        poly_min - recip_min
    );
    println!(
        "  + fold scale, drop norm   saves {:8.1} ms/window (total vs T_poly)",
        poly_min - p2_min
    );

    let softmax = libm_min - fill_min;
    let exp = libm_min - noexp_min;
    let poly_win = libm_min - poly_min;
    println!("derived (per encoder window):");
    println!("  softmax total     {softmax:8.1} ms");
    println!(
        "  |- exp term       {exp:8.1} ms   ({:.1}% of softmax)",
        100.0 * exp / softmax
    );
    println!("  poly saving       {poly_win:8.1} ms");
    println!("  dead zero-init    {alloc_min:8.1} ms");
    println!(
        "\n  vs REAL kernel ({real_min:.1} ms): softmax = {:.1}%, exp = {:.1}%, dead0 = {:.1}%",
        100.0 * softmax / real_min,
        100.0 * exp / real_min,
        100.0 * alloc_min / real_min,
    );
    println!(
        "  byte-exact headroom (dead0) {alloc_min:.1} ms; +poly (non-byte-exact) {:.1} ms",
        alloc_min + poly_win
    );

    // Correctness: poly vs libm on one block.
    let mut a = src.clone();
    let mut b = src.clone();
    softmax_libm(&mut a, BR, scale);
    softmax_poly(&mut b, BR, scale);
    let maxabs = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let rowsum: f32 = b[..SEQ].iter().sum();
    println!("\npoly vs libm: max|delta| = {maxabs:.3e}   poly row-sum = {rowsum:.7}");
}
