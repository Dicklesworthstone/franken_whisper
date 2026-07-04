//! Activation int8 quantize: scalar `.round()` vs SIMD copysign+trunc (BlackThrush, 2026-07-03).
//!
//! `gemv_i8` quantizes its f32 activation once per call via
//! `x.iter().map(|v| (v*xinv).round().clamp(-127,127) as i8).collect()` — ~7 calls/
//! token in decode, plus a fresh Vec alloc each call. `f32::round` is round-HALF-AWAY,
//! which has NO direct AVX rounding mode (roundps only does nearest/floor/ceil/trunc),
//! so LLVM may lower the map to a scalar `roundf` per element. This A/Bs it against an
//! explicit AVX2 `trunc(v + copysign(0.5,v))` (= round-half-away, BYTE-IDENTICAL for
//! finite inputs — activations are always finite post-GEMM) writing into a reused
//! buffer (no per-call alloc). Asserts the i8 output matches exactly.
//! Usage: `quant_i8_probe [iters]` (default 2000).
#![allow(unsafe_code)]
use std::hint::black_box;
use std::time::Instant;

/// Exact replica of gemv_i8's scalar quantize (allocates, like the real code).
fn quant_scalar(x: &[f32], xinv: f32) -> Vec<i8> {
    x.iter()
        .map(|v| (v * xinv).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// AVX2 quantize into a reused buffer. trunc(v + copysign(0.5,v)) = round-half-away
/// = f32::round for finite v; clamp then saturating-pack to i8. Byte-identical.
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn quant_simd(x: &[f32], xinv: f32, out: &mut [i8]) {
    use core::arch::x86_64::*;
    unsafe {
        let n = x.len();
        let xp = x.as_ptr();
        let vinv = _mm256_set1_ps(xinv);
        let half = _mm256_set1_ps(0.5);
        let signmask = _mm256_set1_ps(-0.0); // 0x80000000
        let c127 = _mm256_set1_ps(127.0);
        let cm127 = _mm256_set1_ps(-127.0);
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_mul_ps(_mm256_loadu_ps(xp.add(i)), vinv);
            let sign = _mm256_and_ps(v, signmask);
            let vh = _mm256_add_ps(v, _mm256_or_ps(half, sign)); // v + copysign(0.5,v)
            let r = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(vh); // trunc
            let r = _mm256_min_ps(_mm256_max_ps(r, cm127), c127); // clamp [-127,127]
            let ri = _mm256_cvtps_epi32(r); // exact (integer-valued)
            // pack i32x8 -> i8x8 (order preserved): packs_epi32(lo,hi) -> i16x8, packs_epi16 -> i8.
            let lo = _mm256_castsi256_si128(ri);
            let hi = _mm256_extracti128_si256::<1>(ri);
            let i16s = _mm_packs_epi32(lo, hi); // [0..3 lo][4..7 hi] i16, order 0..7
            let i8s = _mm_packs_epi16(i16s, i16s); // low 8 bytes = elems 0..7
            _mm_storel_epi64(out.as_mut_ptr().add(i) as *mut __m128i, i8s);
            i += 8;
        }
        while i < n {
            let q = (*x.get_unchecked(i) * xinv).round().clamp(-127.0, 127.0);
            *out.get_unchecked_mut(i) = q as i8;
            i += 1;
        }
    }
}

fn bench(name: &str, n: usize, iters: usize) {
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0 // ~U(-3,3), finite
    };
    let x: Vec<f32> = (0..n).map(|_| nf()).collect();
    let xamax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    let xinv = 127.0 / xamax;

    // Byte-exactness (spans the clamp edges since amax maps to ±127).
    let a = quant_scalar(&x, xinv);
    let mut b = vec![0i8; n];
    unsafe { quant_simd(&x, xinv, &mut b) };
    let diff = a.iter().zip(b.iter()).filter(|(p, q)| p != q).count();

    let ts = {
        for _ in 0..3 {
            black_box(quant_scalar(&x, xinv));
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            let r = quant_scalar(&x, xinv);
            best = best.min(t.elapsed().as_secs_f64());
            black_box(r);
        }
        best
    };
    let ta = {
        let mut buf = vec![0i8; n];
        for _ in 0..3 {
            unsafe { quant_simd(&x, xinv, &mut buf) };
            black_box(&buf);
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            unsafe { quant_simd(&x, xinv, &mut buf) };
            best = best.min(t.elapsed().as_secs_f64());
            black_box(&buf);
        }
        best
    };
    println!("{name}  n={n}  best-of-{iters}");
    println!(
        "  byte-exact: {diff} differing of {n}  [{}]",
        if diff == 0 { "IDENTICAL" } else { "DIVERGENT" }
    );
    println!("  scalar .round()+alloc : {:>8.3} µs", ts * 1e6);
    println!(
        "  AVX2 copysign+trunc   : {:>8.3} µs  {:.2}x  [{}]",
        ta * 1e6,
        ts / ta,
        if ta < ts { "WIN" } else { "loss" }
    );
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    println!("=== gemv_i8 activation quantize: scalar .round() vs AVX2 (1 thread) ===");
    bench("qkv/mlp_0 act [1280]", 1280, iters);
    bench("fc2 act    [5120]", 5120, iters / 2);
}
