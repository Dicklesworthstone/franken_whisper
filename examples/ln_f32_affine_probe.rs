//! LayerNorm: full-f64 (franken) vs f64-mean/var + f32-normalize/affine (ggml-style)
//! (BlackThrush, 2026-07-03).
//!
//! franken's `layer_norm` accumulates mean/var in f64 (faithful) AND does the normalize +
//! affine `(x-mean)*inv*w+b` in f64 (4-wide SIMD). ggml (and PyTorch) accumulate mean/var
//! in f64/double but do the normalize + affine in f32 (8-wide). So franken is MORE precise
//! than the references here. This measures whether an f32 normalize/affine (mean/var still
//! f64) is faster — if LN is compute-bound, f32's 8-wide beats f64's 4-wide; if it's
//! memory-bound (x/out are f32 in DRAM regardless), there's no win. Also reports the max |Δ|
//! between the two so the faithfulness cost of a switch is on record (owner-gated numerics).
//! Turbo encoder LN shape: [1500 rows, 1280 cols].
//! Usage: `ln_f32_affine_probe [iters]` (default 400).
use std::hint::black_box;
use std::time::Instant;

const EPS: f32 = 1e-5;

/// Full-f64 normalize+affine (franken's scheme), scalar reference.
fn ln_f64(x: &[f32], w: &[f32], b: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    let n = cols as f64;
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mut sum = 0.0f64;
        for &v in row {
            sum += f64::from(v);
        }
        let mean = sum / n;
        let mut var = 0.0f64;
        for &v in row {
            let d = f64::from(v) - mean;
            var += d * d;
        }
        var /= n;
        let inv = 1.0 / (var + f64::from(EPS)).sqrt();
        let o = &mut out[r * cols..(r + 1) * cols];
        for j in 0..cols {
            o[j] = ((f64::from(row[j]) - mean) * inv * f64::from(w[j]) + f64::from(b[j])) as f32;
        }
    }
}

/// f64 mean/var, f32 normalize+affine (ggml-style), scalar.
fn ln_f32affine(x: &[f32], w: &[f32], b: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    let n = cols as f64;
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mut sum = 0.0f64;
        for &v in row {
            sum += f64::from(v);
        }
        let mean = (sum / n) as f32;
        let mut var = 0.0f64;
        for &v in row {
            let d = f64::from(v) - sum / n;
            var += d * d;
        }
        let variance = (var / n) as f32;
        let inv = 1.0f32 / (variance + EPS).sqrt();
        let o = &mut out[r * cols..(r + 1) * cols];
        for j in 0..cols {
            o[j] = (row[j] - mean) * inv * w[j] + b[j];
        }
    }
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let (rows, cols) = (1500usize, 1280usize);
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0
    };
    let x: Vec<f32> = (0..rows * cols).map(|_| nf()).collect();
    let w: Vec<f32> = (0..cols).map(|_| nf() * 0.3 + 1.0).collect();
    let b: Vec<f32> = (0..cols).map(|_| nf() * 0.1).collect();
    let mut o64 = vec![0.0f32; rows * cols];
    let mut o32 = vec![0.0f32; rows * cols];

    ln_f64(&x, &w, &b, rows, cols, &mut o64);
    ln_f32affine(&x, &w, &b, rows, cols, &mut o32);
    let maxd = o64
        .iter()
        .zip(&o32)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ndiff = o64
        .iter()
        .zip(&o32)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();

    type LnFn<'a> = &'a dyn Fn(&[f32], &[f32], &[f32], usize, usize, &mut [f32]);
    let run = |f: LnFn, out: &mut [f32]| -> f64 {
        for _ in 0..3 {
            f(&x, &w, &b, rows, cols, out);
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            f(&x, &w, &b, rows, cols, out);
            best = best.min(t.elapsed().as_secs_f64());
            black_box(&out[0]);
        }
        best
    };
    let t64 = run(&ln_f64, &mut o64);
    let t32 = run(&ln_f32affine, &mut o32);
    println!(
        "=== LayerNorm [1500,1280]: full-f64 (franken) vs f32-normalize/affine (ggml-style), 1 thread ==="
    );
    println!(
        "  numeric Δ: max|Δ|={maxd:.3e}, {ndiff} of {} differ (non-byte-exact — owner-gated numerics)",
        rows * cols
    );
    println!("  full-f64 (franken)        : {:>8.1} µs", t64 * 1e6);
    println!(
        "  f32-normalize (ggml-style): {:>8.1} µs  {:.2}x  [{}]",
        t32 * 1e6,
        t64 / t32,
        if t32 < t64 { "faster" } else { "not faster" }
    );
    println!(
        "  (this is the SCALAR shape; SIMD would be f64x4 vs f32x8. If ~1.0x here, LN is memory-bound ⇒ no win.)"
    );
}
