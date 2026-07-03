//! Is the sampler's `logit_max = logits.fold(NEG_INFINITY, f32::max)` (decode.rs:514)
//! scalar-bound, and does an AVX2 `maxps` tree reduction give a BYTE-IDENTICAL result
//! that's faster? (BlackThrush, 2026-07-03).
//!
//! `f32::max` has `fmaxf` semantics (if one operand is NaN, return the other), which
//! DIFFERS from `_mm256_max_ps` (returns the 2nd operand on NaN/unordered). So LLVM
//! cannot lower a `fold(f32::max)` to a bare `maxps` reduction while preserving
//! semantics — the SAME reason it leaves `argmax` scalar (index dep) and `round`
//! scalar (no AVX mode). BUT: in `compute_logprobs` the input is SANITIZED first
//! (every non-finite → NEG_INFINITY), so there are NO NaNs at the fold — and max is
//! EXACTLY associative over reals (unlike float +), so ANY reduction order (incl. an
//! AVX2 maxps tree) yields the bit-identical result. This probe (a) proves byte-identity
//! on sanitized input and (b) times scalar fold vs AVX2 maxps over the 51866 vocab.
//! Usage: `logit_max_reduce_probe [iters]`.
#![allow(unsafe_code)]
use std::hint::black_box;
use std::time::Instant;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const N: usize = 51866; // large-v3-turbo vocab

/// Scalar reference: exactly `logits.iter().copied().fold(NEG_INFINITY, f32::max)`.
fn max_scalar(logits: &[f32]) -> f32 {
    logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// AVX2 `maxps` tree reduction. On NaN-free input this equals `f32::max` fold bit-for-bit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn max_avx2(logits: &[f32]) -> f32 {
    let n = logits.len();
    let p = logits.as_ptr();
    // 4 independent accumulator vectors to hide the maxps latency (like the dot probes).
    let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
    let (mut a0, mut a1, mut a2, mut a3) = (ninf, ninf, ninf, ninf);
    let mut i = 0;
    while i + 32 <= n {
        a0 = _mm256_max_ps(a0, _mm256_loadu_ps(p.add(i)));
        a1 = _mm256_max_ps(a1, _mm256_loadu_ps(p.add(i + 8)));
        a2 = _mm256_max_ps(a2, _mm256_loadu_ps(p.add(i + 16)));
        a3 = _mm256_max_ps(a3, _mm256_loadu_ps(p.add(i + 24)));
        i += 32;
    }
    let mut acc = _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3));
    while i + 8 <= n {
        acc = _mm256_max_ps(acc, _mm256_loadu_ps(p.add(i)));
        i += 8;
    }
    // horizontal max of the 8 lanes
    let mut t = [0.0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), acc);
    let mut m = t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    while i < n {
        m = m.max(logits[i]);
        i += 1;
    }
    m
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20000);
    let mut st = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; ((st >> 40) as f32 / (1u64 << 24) as f32) * 40.0 - 20.0 };
    // realistic logit magnitudes; sanitized (finite) as compute_logprobs guarantees.
    let mut logits: Vec<f32> = (0..N).map(|_| nf()).collect();
    // simulate the masked lanes compute_logprobs produces (timestamp/suppressed → NEG_INF).
    for k in (0..N).step_by(97) { logits[k] = f32::NEG_INFINITY; }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let s = max_scalar(&logits);
        let v = max_avx2(&logits);
        println!("=== logit_max reduce: scalar fold(f32::max) vs AVX2 maxps tree (N={N}, sanitized) ===");
        println!("  scalar max = {s:.9}   avx2 max = {v:.9}   bits {}",
            if s.to_bits() == v.to_bits() { "IDENTICAL" } else { "DIFFER" });
        // warm
        for _ in 0..2000 { black_box(max_scalar(black_box(&logits))); }
        for _ in 0..2000 { black_box(max_avx2(black_box(&logits))); }
        let mut ts = f64::INFINITY;
        for _ in 0..7 { let t = Instant::now(); for _ in 0..iters { black_box(max_scalar(black_box(&logits))); } ts = ts.min(t.elapsed().as_secs_f64()); }
        let mut tv = f64::INFINITY;
        for _ in 0..7 { let t = Instant::now(); for _ in 0..iters { black_box(max_avx2(black_box(&logits))); } tv = tv.min(t.elapsed().as_secs_f64()); }
        let us_s = ts / iters as f64 * 1e6;
        let us_v = tv / iters as f64 * 1e6;
        println!("  scalar : {us_s:>7.2} µs/call");
        println!("  avx2   : {us_v:>7.2} µs/call   {:.2}× [{}]", us_s / us_v,
            if us_v < us_s * 0.85 { "WIN — LLVM left the fold scalar" } else { "no win — LLVM already vectorizes the fold" });
        println!("  (per-token serial sampler op; ~0.05-0.15% e2e like argmax if it's a real win.)");
    }
}
