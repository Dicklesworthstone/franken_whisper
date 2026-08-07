//! Sampler max-reduce: scalar `fold(f32::max)` vs AVX2 `vmaxps` (BlackThrush, 2026-07-03).
//!
//! `compute_logprobs` (and the timestamp-rule) reduce the 51866-vocab logits with
//! `logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)` — per token. `f32::max`
//! carries NaN semantics (returns the non-NaN operand) that LLVM cannot lower to a
//! plain `vmaxps` without proving no-NaN, so it scalarizes the reduction. BUT the
//! logits are SANITIZED (NaN/+inf → -inf) before this max, so inputs are finite-or-(-inf)
//! with NO NaN, and a plain `vmaxps` tree-reduction is BYTE-IDENTICAL (max is exact and
//! order-independent; -inf < any finite). This A/Bs the two, asserting bit-equality.
//! Usage: `maxreduce_probe [iters]` (default 2000).
#![allow(unsafe_code)]
#[cfg(target_arch = "x86_64")]
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

/// Exact replica of the sampler's scalar max-reduce.
#[cfg(target_arch = "x86_64")]
fn max_scalar(x: &[f32]) -> f32 {
    x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// AVX2 vmaxps tree-reduction. Byte-identical to the scalar fold for NaN-free input.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn max_avx2(x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let n = x.len();
        let xp = x.as_ptr();
        let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut m0 = ninf;
        let mut m1 = ninf;
        let mut m2 = ninf;
        let mut m3 = ninf;
        let mut i = 0;
        while i + 32 <= n {
            m0 = _mm256_max_ps(m0, _mm256_loadu_ps(xp.add(i)));
            m1 = _mm256_max_ps(m1, _mm256_loadu_ps(xp.add(i + 8)));
            m2 = _mm256_max_ps(m2, _mm256_loadu_ps(xp.add(i + 16)));
            m3 = _mm256_max_ps(m3, _mm256_loadu_ps(xp.add(i + 24)));
            i += 32;
        }
        while i + 8 <= n {
            m0 = _mm256_max_ps(m0, _mm256_loadu_ps(xp.add(i)));
            i += 8;
        }
        let m = _mm256_max_ps(_mm256_max_ps(m0, m1), _mm256_max_ps(m2, m3));
        // horizontal max of 8 lanes
        let lo = _mm256_castps256_ps128(m);
        let hi = _mm256_extractf128_ps::<1>(m);
        let q = _mm_max_ps(lo, hi);
        let q = _mm_max_ps(q, _mm_shuffle_ps::<0b01_00_11_10>(q, q));
        let q = _mm_max_ps(q, _mm_shuffle_ps::<0b00_00_00_01>(q, q));
        let mut best = _mm_cvtss_f32(q);
        while i < n {
            let v = *x.get_unchecked(i);
            if v > best {
                best = v;
            }
            i += 1;
        }
        best
    }
}

#[cfg(target_arch = "x86_64")]
fn bench(name: &str, n: usize, iters: usize) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut nf = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let u = (s >> 40) as f32 / (1u64 << 24) as f32;
        if u < 0.1 {
            f32::NEG_INFINITY
        } else {
            -30.0 * u
        } // sanitized: finite or -inf
    };
    let mut x: Vec<f32> = (0..n).map(|_| nf()).collect();
    x[n / 3] = 0.0; // a finite max

    let a = max_scalar(&x);
    let b = unsafe { max_avx2(&x) };
    let bitmatch = a.to_bits() == b.to_bits();

    let run = |f: &dyn Fn(&[f32]) -> f32| -> f64 {
        for _ in 0..3 {
            black_box(f(&x));
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            black_box(f(&x));
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };
    let ts = run(&max_scalar);
    let ta = run(&|x| unsafe { max_avx2(x) });
    println!("{name}  n={n}  best-of-{iters}");
    println!(
        "  byte-exact: scalar={a} avx={b}  [{}]",
        if bitmatch {
            "BIT-IDENTICAL"
        } else {
            "DIVERGENT"
        }
    );
    println!("  scalar fold(f32::max) : {:>8.3} µs", ts * 1e6);
    println!(
        "  AVX2 vmaxps           : {:>8.3} µs  {:.2}x  [{}]",
        ta * 1e6,
        ts / ta,
        if ta < ts { "WIN" } else { "loss" }
    );
}

#[cfg(target_arch = "x86_64")]
fn main() {
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("maxreduce_probe requires AVX2 support");
        return;
    }
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    println!("=== sampler max-reduce: scalar fold(f32::max) vs AVX2 vmaxps (1 thread) ===");
    bench("vocab logits [51866]", 51866, iters);
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("maxreduce_probe requires an x86_64 processor");
}
