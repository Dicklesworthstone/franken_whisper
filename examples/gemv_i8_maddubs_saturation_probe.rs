//! CORRECTION probe: VPMADDUBSW (maddubs) SATURATES for full-range int8×int8 (BlackThrush, 2026-07-03).
//!
//! Last cycle (4cfcd56) I measured maddubs (VPMADDUBSW+VPMADDWD) at 1.79× the widening op
//! (VPMOVSXBW+VPMADDWD) and recommended it for the int8 encoder GEMM. THAT WAS WRONG. maddubs
//! sums two u8×i8 products into an **int16** intermediate, which SATURATES: with u8∈[0,255],
//! i8∈[-128,127], each product ∈[-32640,32385] and two summed ∈[-65280,64770] ≫ int16's
//! ±32767. VPMADDWD (the "widening" path) accumulates into **int32** ⇒ no saturation. So the
//! prior tiled kernel used widening *because maddubs is inaccurate*, not by oversight; the only
//! non-saturating single-instruction u8·i8→i32 op is VNNI's VPDPBUSD, which this box lacks.
//! This probe proves the saturation: maddubs+sign-offset vs the exact (widening) dot over K.
//! Usage: `gemv_i8_maddubs_saturation_probe`.
#![allow(unsafe_code)]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::hint::black_box;

const K: usize = 1280; // decode/encoder inner dim (n_state)

/// Exact Σ w[k]·x[k] via VPMOVSXBW+VPMADDWD (int32 accumulation — the correct path).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_widening(w: &[i8], x: &[i8]) -> i32 {
    let mut acc = _mm256_setzero_si256();
    let (pw, px) = (w.as_ptr(), x.as_ptr());
    let mut k = 0;
    while k + 16 <= K {
        let vw = _mm256_cvtepi8_epi16(_mm_loadu_si128(pw.add(k).cast()));
        let vx = _mm256_cvtepi8_epi16(_mm_loadu_si128(px.add(k).cast()));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(vw, vx));
        k += 16;
    }
    let mut t = [0i32; 8];
    _mm256_storeu_si256(t.as_mut_ptr().cast(), acc);
    t.iter().sum()
}

/// maddubs + sign-offset: Σ(x+128)·w via VPMADDUBSW (int16 intermediate — SATURATES).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_maddubs(w: &[i8], xu: &[u8], wsum: i32) -> i32 {
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    let (pw, px) = (w.as_ptr(), xu.as_ptr());
    let mut k = 0;
    while k + 32 <= K {
        let vx = _mm256_loadu_si256(px.add(k).cast());
        let vw = _mm256_loadu_si256(pw.add(k).cast());
        let p16 = _mm256_maddubs_epi16(vx, vw); // saturates to int16!
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p16, ones));
        k += 32;
    }
    let mut t = [0i32; 8];
    _mm256_storeu_si256(t.as_mut_ptr().cast(), acc);
    t.iter().sum::<i32>() - 128 * wsum
}

fn main() {
    let mut st = 0x243F_6A88_85A3_08D3u64;
    let mut b8 = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        (st >> 56) as i8
    };
    const ROWS: usize = 4096;
    let w: Vec<i8> = (0..ROWS * K).map(|_| b8()).collect();
    let x: Vec<i8> = (0..K).map(|_| b8()).collect();
    let xu: Vec<u8> = x.iter().map(|&v| (v as i16 + 128) as u8).collect();

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut differ = 0usize;
        let mut maxerr = 0i32;
        for n in 0..ROWS {
            let wr = &w[n * K..(n + 1) * K];
            let wsum: i32 = wr.iter().map(|&v| v as i32).sum();
            let exact = dot_widening(wr, &x);
            let mad = dot_maddubs(wr, &xu, wsum);
            if exact != mad {
                differ += 1;
                maxerr = maxerr.max((exact - mad).abs());
            }
            black_box(mad);
        }
        println!(
            "=== VPMADDUBSW saturation check: maddubs+sign-offset vs exact int8·int8 dot (K={K}) ==="
        );
        println!(
            "  rows where maddubs ≠ exact : {differ} / {ROWS}  ({:.0}%)",
            100.0 * differ as f64 / ROWS as f64
        );
        println!(
            "  max |error|                : {maxerr}  (int16 saturation of the u8·i8 pair-sums)"
        );
        println!(
            "  VERDICT: [{}]",
            if differ == 0 {
                "no saturation at this scale — maddubs viable"
            } else {
                "maddubs SATURATES ⇒ NOT usable for accurate int8×int8 GEMM without VNNI; widening (int32) is the correct op"
            }
        );
        println!(
            "  ⇒ last cycle's 'maddubs is the fix' is RETRACTED: the prior kernel used widening because it's correct."
        );
        println!(
            "  ⇒ int8 encoder GEMM on AVX2 (no VNNI) is stuck with widening (~1.27× blocked-f32 compute), eaten to 0.89×."
        );
    }
}
