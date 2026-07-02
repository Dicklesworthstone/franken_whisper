#![feature(portable_simd)]
//! Is a SIMD-exp softmax worth it for the ENCODER (1500×1500 scores), where the
//! memory's decode-based "softmax-exp ~0" rejection does NOT apply?
//!
//! Decode softmax is tiny (~cache_len keys) and overlaps the GEMVs → SIMD exp was
//! ~0 e2e there (rejected). The ENCODER self-attention softmax is `[1500,1500]`
//! per head × n_head × n_layer, compute-bound and NOT overlapped. This probes the
//! softmax KERNEL A/B on that shape, model-free, best-of-N: franken's current
//! scalar `libm expf` softmax vs an 8-lane Cephes-poly-exp softmax. Reports the
//! speedup + max|Δ| vs the scalar row (the poly's approximation error).
//! Usage: `softmax_probe [iters]` (default 30).
use std::hint::black_box;
use std::simd::{Simd, StdFloat, cmp::SimdPartialOrd, num::SimdFloat, num::SimdInt};
use std::time::Instant;

/// franken's current softmax row: scalar max / libm exp (+finite guard) / normalize.
fn softmax_scalar(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
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
}

const L: usize = 8;
type V = Simd<f32, L>;

/// 8-lane Cephes-style expf (same shape ggml_v_expf uses): range-reduce
/// x = n·ln2 + r, exp(r) via a degree-6 poly, scale by 2^n via exponent bits.
#[inline]
fn exp8(x: V) -> V {
    let ln2 = V::splat(0.693_147_18);
    let inv_ln2 = V::splat(1.442_695_04);
    // clamp to avoid overflow in 2^n
    let hi = V::splat(88.0);
    let lo = V::splat(-88.0);
    let x = x.simd_min(hi).simd_max(lo);
    let n = (x * inv_ln2).round();
    let r = x - n * ln2;
    // exp(r) ≈ 1 + r + r²/2 + ... (Cephes p-coeffs)
    let mut p = V::splat(1.986_5e-4);
    p = p * r + V::splat(1.398_2e-3);
    p = p * r + V::splat(8.333_45e-3);
    p = p * r + V::splat(4.166_580e-2);
    p = p * r + V::splat(1.666_666_6e-1);
    p = p * r + V::splat(5.0e-1);
    p = p * r + V::splat(1.0);
    p = p * r + V::splat(1.0);
    // scale by 2^n: add n to the f32 exponent field.
    let ni: Simd<i32, L> = n.cast();
    let bits = (ni + Simd::splat(127)) << Simd::splat(23);
    let two_n = V::from_bits(bits.cast());
    p * two_n
}

fn softmax_simd(row: &mut [f32]) {
    let n = row.len();
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let vmax = V::splat(max);
    let nl = n - n % L;
    let mut sum = 0.0f32;
    let mut i = 0;
    let mut acc = V::splat(0.0);
    while i < nl {
        let x = V::from_slice(&row[i..i + L]);
        let e = exp8(x - vmax);
        e.copy_to_slice(&mut row[i..i + L]);
        acc += e;
        i += L;
    }
    sum += acc.reduce_sum();
    for v in &mut row[nl..] {
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
}

fn best_of(iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..2 {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    best
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let (rows, cols) = (1500usize, 1500usize); // one encoder self-attn head's scores.
    let base: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let t = (i as f32 * 0.6180339887) % 1.0;
            (t - 0.5) * 8.0
        })
        .collect();

    let mut buf = base.clone();
    let scalar_ms = best_of(iters, || {
        buf.copy_from_slice(&base);
        for r in buf.chunks_mut(cols) {
            softmax_scalar(r);
        }
        black_box(&buf);
    }) * 1e3;

    let mut buf2 = base.clone();
    let simd_ms = best_of(iters, || {
        buf2.copy_from_slice(&base);
        for r in buf2.chunks_mut(cols) {
            softmax_simd(r);
        }
        black_box(&buf2);
    }) * 1e3;

    // Accuracy: max |Δ| between the two normalized softmax outputs.
    let mut a = base.clone();
    let mut b = base.clone();
    for r in a.chunks_mut(cols) {
        softmax_scalar(r);
    }
    for r in b.chunks_mut(cols) {
        softmax_simd(r);
    }
    let maxdiff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);

    println!("softmax {rows}x{cols} (encoder self-attn head) best-of-{iters}");
    println!("  scalar libm-exp   {scalar_ms:7.3} ms");
    println!("  simd cephes-exp   {simd_ms:7.3} ms   speedup={:.2}x", scalar_ms / simd_ms);
    println!("  max |Δ| normalized softmax = {maxdiff:.2e}");
}
