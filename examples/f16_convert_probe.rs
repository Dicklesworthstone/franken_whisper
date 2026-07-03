//! Cross-build f32→f16 conversion: scalar `half::f16::from_f32` vs batched F16C
//! `_mm256_cvtps_ph` (BlackThrush, 2026-07-03).
//!
//! `decoder.rs`'s per-window cross-attention build (12.7 ms/window, already 4.99×
//! parallelized) converts the encoder K/V to f16 with per-element
//! `Float16::from_f32` — ~15.4 M conversions/window (k_nat + v_t over 4×20 pairs ×
//! 1500 frames × 64 d_head). If `half::f16::from_f32` is SOFTWARE bit-twiddling, a
//! batched `_mm256_cvtps_ph` (8 f32→f16 per instr, round-to-nearest-even, byte-
//! identical to `from_f32`'s RNE) is a real win on the contiguous k_nat build. If
//! `half` already lowers to scalar hardware `vcvtps2ph`, the batch is a wash. This
//! resolves which, asserting the u16 output bits match exactly.
//! Usage: `f16_convert_probe [iters]` (default 3000).
#![allow(unsafe_code)]
use ft_core::Float16;
use std::hint::black_box;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Exact replica: the code's per-element `Float16::from_f32` into a contiguous f16 buf.
fn convert_scalar(x: &[f32]) -> Vec<Float16> {
    x.iter().map(|&s| Float16::from_f32(s)).collect()
}

/// Batched F16C: 8 f32→f16 per `_mm256_cvtps_ph::<0>` (round-to-nearest-even).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c,avx")]
unsafe fn convert_f16c(x: &[f32], out: &mut [u16]) {
    unsafe {
        let n = x.len();
        let xp = x.as_ptr();
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(xp.add(i));
            let h = _mm256_cvtps_ph::<0>(v); // 0 = _MM_FROUND_TO_NEAREST_INT (RNE)
            _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, h);
            i += 8;
        }
        while i < n {
            *out.get_unchecked_mut(i) = Float16::from_f32(*x.get_unchecked(i)).to_bits();
            i += 1;
        }
    }
}

fn bench(n: usize, iters: usize) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut nf = || {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0 // encoder-output-ish range
    };
    let x: Vec<f32> = (0..n).map(|_| nf()).collect();

    let a = convert_scalar(&x);
    let mut b = vec![0u16; n];
    unsafe { convert_f16c(&x, &mut b) };
    let bad = a.iter().zip(&b).filter(|(p, q)| p.to_bits() != **q).count();

    let ts = {
        for _ in 0..3 { black_box(convert_scalar(&x)); }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            black_box(convert_scalar(&x));
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };
    let ta = {
        let mut buf = vec![0u16; n];
        for _ in 0..3 { unsafe { convert_f16c(&x, &mut buf) }; black_box(&buf); }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            unsafe { convert_f16c(&x, &mut buf) };
            best = best.min(t.elapsed().as_secs_f64());
            black_box(&buf);
        }
        best
    };
    println!("n={n}  best-of-{iters}");
    println!("  byte-exact: {bad} differing of {n}  [{}]", if bad == 0 { "IDENTICAL" } else { "DIVERGENT" });
    println!("  scalar from_f32 (+alloc) : {:>8.3} µs", ts * 1e6);
    println!("  F16C _mm256_cvtps_ph     : {:>8.3} µs  {:.2}x  [{}]",
        ta * 1e6, ts / ta, if ta < ts { "WIN" } else { "loss" });
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    println!("=== cross-build f32→f16: scalar half::from_f32 vs batched F16C (1 thread) ===");
    bench(64, iters);   // one k_nat row (d_head)
    bench(1500, iters); // one head's k_nat column-run (enc_frames)
    bench(96000, iters / 4); // full (li,h) pair k_nat: enc_frames*d_head
}
