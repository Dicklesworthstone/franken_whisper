//! Decode score·V output SAXPY: scalar `*o += a*x` vs AVX2 mul+add (BlackThrush, 2026-07-03).
//!
//! `attention_decode_step`'s output accumulation is `out[d] += scores[j]*v[j,d]` summed
//! over j (keys), with the inner d (=d_head) loop writing INDEPENDENT output slots. So it
//! vectorizes across d BYTE-EXACTLY: 8-wide `_mm256_mul_ps`+`_mm256_add_ps` (NOT fmadd —
//! the scalar `+=` is mul-then-add, two roundings) keeps each out[d]'s j-ascending sum
//! bit-for-bit. This A/Bs one head's full output pass (tk SAXPYs of length d_head) and
//! asserts bit-equality. (The QK scores dot reduces over contiguous d, so it CANNOT be
//! SIMD'd byte-exactly and stays scalar — see NEGATIVE_EVIDENCE / kv_f16c_probe.)
//! Usage: `saxpy_f32_probe [iters]` (default 8000).
#![allow(unsafe_code)]
#[cfg(target_arch = "x86_64")]
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Replica of the live scalar output loop: for each key j, out[d] += scores[j]*v[j,d].
#[cfg(target_arch = "x86_64")]
fn out_scalar(scores: &[f32], v: &[f32], _tk: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d];
    for (j, &sj) in scores.iter().enumerate() {
        let vrow = &v[j * d..(j + 1) * d];
        for (o, &vd) in out.iter_mut().zip(vrow) {
            *o += sj * vd;
        }
    }
    out
}

/// AVX2 mul+add (byte-identical: two roundings, vectorized over independent d slots).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn out_avx2(scores: &[f32], v: &[f32], _tk: usize, d: usize) -> Vec<f32> {
    unsafe {
        let mut out = vec![0.0f32; d];
        let op = out.as_mut_ptr();
        for (j, &sj) in scores.iter().enumerate() {
            let vp = v.as_ptr().add(j * d);
            let va = _mm256_set1_ps(sj);
            let mut i = 0;
            while i + 8 <= d {
                let ov = _mm256_loadu_ps(op.add(i));
                let xv = _mm256_loadu_ps(vp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_add_ps(ov, _mm256_mul_ps(va, xv)));
                i += 8;
            }
            while i < d {
                *out.get_unchecked_mut(i) += sj * *v.get_unchecked(j * d + i);
                i += 1;
            }
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
fn bench(tk: usize, d: usize, iters: usize) {
    let mut st = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        ((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    };
    let scores: Vec<f32> = (0..tk).map(|_| nf().abs() * 0.01).collect(); // softmax-like
    let v: Vec<f32> = (0..tk * d).map(|_| nf()).collect();

    let a = out_scalar(&scores, &v, tk, d);
    let b = unsafe { out_avx2(&scores, &v, tk, d) };
    let bad = a
        .iter()
        .zip(&b)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();

    macro_rules! time {
        ($f:expr) => {{
            for _ in 0..3 {
                black_box($f);
            }
            let mut best = f64::INFINITY;
            for _ in 0..iters {
                let t = Instant::now();
                black_box($f);
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }};
    }
    let ts = time!(out_scalar(&scores, &v, tk, d));
    let ta = time!(unsafe { out_avx2(&scores, &v, tk, d) });
    println!("tk={tk} d={d}  best-of-{iters}");
    println!(
        "  byte-exact: {bad} differing bit-patterns of {d}  [{}]",
        if bad == 0 { "IDENTICAL" } else { "DIVERGENT" }
    );
    println!("  scalar *o += a*x : {:>8.3} µs", ts * 1e6);
    println!(
        "  AVX2 mul+add     : {:>8.3} µs  {:.2}x  [{}]",
        ta * 1e6,
        ts / ta,
        if ta < ts { "WIN" } else { "loss" }
    );
}

#[cfg(target_arch = "x86_64")]
fn main() {
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("saxpy_f32_probe requires AVX2 support");
        return;
    }
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);
    println!(
        "=== decode score·V output SAXPY: scalar vs AVX2 mul+add (turbo d_head=64, 1 thread) ==="
    );
    bench(64, 64, iters);
    bench(256, 64, iters);
    bench(448, 64, iters);
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("saxpy_f32_probe requires an x86_64 processor");
}
