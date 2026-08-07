//! Is the encoder f64 LayerNorm (`nn::norm_rows_into`, SoA `Simd<f64,8>` over 8
//! rows) MEMORY-bound (⇒ an LN→quant fusion that drops the `h` round-trip could
//! win ~like gelu), or SoA-overhead / strided-gather-bound (⇒ fusion is dead)?
//!
//! The SoA reads column-j of 8 consecutive rows = 8 loads at stride `cols`
//! (=5120 B, the stride that defeats the HW prefetcher — same as the SDPA gather).
//! This replicates `norm_rows_into` EXACTLY and A/Bs it against (a) a memcpy
//! baseline (the pure 15 MB read+write floor a fusion could remove) and (b) a
//! variant that transposes each 8-row block CONTIGUOUSLY first (byte-identical
//! math, no strided gather) — isolating gather cost from compute cost.
//!
//! Run: cargo +nightly run --release --example ln_membound_probe
#![feature(portable_simd)]
use std::simd::{Simd, StdFloat};

const ROWS: usize = 1500;
const COLS: usize = 1280;
type V = Simd<f64, 8>;
const L: usize = 8;

/// EXACT replica of nn::norm_rows_into (SoA, strided gather of 8 rows).
fn ln_soa(src: &[f32], dst: &mut [f32], w: &[f32], b: &[f32], eps: f64) {
    let n = COLS as f64;
    let nrows = src.len() / COLS;
    let nfull = nrows - nrows % L;
    let mut soa = vec![V::splat(0.0); COLS];
    let mut g = 0;
    while g < nfull {
        for (j, s) in soa.iter_mut().enumerate() {
            let mut a = [0.0f64; L];
            for (lane, al) in a.iter_mut().enumerate() {
                *al = f64::from(src[(g + lane) * COLS + j]); // strided gather, stride COLS
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
                dst[(g + lane) * COLS + j] = val as f32;
            }
        }
        g += L;
    }
    // scalar remainder (matches norm_rows_into tail)
    for r in nfull..nrows {
        let row = &src[r * COLS..r * COLS + COLS];
        let mean: f64 = row.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let var: f64 = row
            .iter()
            .map(|&x| {
                let d = f64::from(x) - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let inv = 1.0 / (var + eps).sqrt();
        for (j, &x) in row.iter().enumerate() {
            dst[r * COLS + j] =
                ((f64::from(x) - mean) * inv * f64::from(w[j]) + f64::from(b[j])) as f32;
        }
    }
}

/// SAME math, but transpose each 8-row block into a [COLS,8] contiguous scratch
/// FIRST (contiguous reads), then run the SoA on the scratch. Byte-identical
/// (same f64 accumulation order per lane), no strided gather.
fn ln_soa_transposed(src: &[f32], dst: &mut [f32], w: &[f32], b: &[f32], eps: f64) {
    let n = COLS as f64;
    let nrows = src.len() / COLS;
    let nfull = nrows - nrows % L;
    let mut soa = vec![V::splat(0.0); COLS];
    let mut tblock = vec![0.0f64; L * COLS]; // [lane, col] contiguous
    let mut g = 0;
    while g < nfull {
        // contiguous read of 8 rows into tblock[lane*COLS + j]
        for lane in 0..L {
            let row = &src[(g + lane) * COLS..(g + lane) * COLS + COLS];
            for (j, &x) in row.iter().enumerate() {
                tblock[lane * COLS + j] = f64::from(x);
            }
        }
        for (j, s) in soa.iter_mut().enumerate() {
            let mut a = [0.0f64; L];
            for (lane, al) in a.iter_mut().enumerate() {
                *al = tblock[lane * COLS + j];
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
                dst[(g + lane) * COLS + j] = val as f32;
            }
        }
        g += L;
    }
    for r in nfull..nrows {
        let row = &src[r * COLS..r * COLS + COLS];
        let mean: f64 = row.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let var: f64 = row
            .iter()
            .map(|&x| {
                let d = f64::from(x) - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let inv = 1.0 / (var + eps).sqrt();
        for (j, &x) in row.iter().enumerate() {
            dst[r * COLS + j] =
                ((f64::from(x) - mean) * inv * f64::from(w[j]) + f64::from(b[j])) as f32;
        }
    }
}

fn memcpy_baseline(src: &[f32], dst: &mut [f32]) {
    dst.copy_from_slice(src); // pure 15 MB read+write floor
}

fn ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let mut s = 0x9e3779b97f4a7c15u64;
    let mut nx = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((s >> 33) as u32) as f32 / u32::MAX as f32 - 0.5) * 4.0
    };
    let src: Vec<f32> = (0..ROWS * COLS).map(|_| nx()).collect();
    let w: Vec<f32> = (0..COLS).map(|_| 1.0 + nx() * 0.1).collect();
    let b: Vec<f32> = (0..COLS).map(|_| nx() * 0.1).collect();
    let eps = 1e-5f64;

    let mut d1 = vec![0.0f32; ROWS * COLS];
    let mut d2 = vec![0.0f32; ROWS * COLS];
    ln_soa(&src, &mut d1, &w, &b, eps);
    ln_soa_transposed(&src, &mut d2, &w, &b, eps);
    println!("SoA == transposed (byte-identical): {}", d1 == d2);

    let reps = 60;
    let mut evict = vec![1.0f32; 40 * 1024 * 1024 / 4];
    let (mut b_soa, mut b_tr, mut b_mc) = (f64::MAX, f64::MAX, f64::MAX);
    let mut dst = vec![0.0f32; ROWS * COLS];
    for _ in 0..reps {
        for e in evict.iter_mut() {
            *e *= 1.0000001;
        }
        let t = std::time::Instant::now();
        ln_soa(&src, &mut dst, &w, &b, eps);
        b_soa = b_soa.min(ms(t));
        for e in evict.iter_mut() {
            *e *= 1.0000001;
        }
        let t = std::time::Instant::now();
        ln_soa_transposed(&src, &mut dst, &w, &b, eps);
        b_tr = b_tr.min(ms(t));
        for e in evict.iter_mut() {
            *e *= 1.0000001;
        }
        let t = std::time::Instant::now();
        memcpy_baseline(&src, &mut dst);
        b_mc = b_mc.min(ms(t));
    }
    std::hint::black_box(&dst);
    std::hint::black_box(&evict);
    println!("[{ROWS},{COLS}] single-thread cold, min-of-{reps}:");
    println!("   memcpy baseline (15MB r+w floor) = {b_mc:.3} ms");
    println!(
        "   LN SoA (real, strided gather)     = {b_soa:.3} ms  ({:.2}× memcpy)",
        b_soa / b_mc
    );
    println!(
        "   LN SoA transposed-first           = {b_tr:.3} ms  ({:.2}× vs SoA)",
        b_tr / b_soa
    );
    println!(
        "   => LN is {}",
        if b_soa < b_mc * 1.4 {
            "MEMORY-bound (fusion could help)"
        } else {
            "COMPUTE/SoA-OVERHEAD-bound (fusion dead)"
        }
    );
}
