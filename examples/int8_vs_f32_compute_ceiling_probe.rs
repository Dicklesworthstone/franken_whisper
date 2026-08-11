//! int8 vs f32 GEMM COMPUTE ceiling on this AVX2 (no-VNNI) Zen3 box (BlackThrush, 2026-07-03).
//!
//! The encoder is ~80% of e2e and ~79% of it is external f32 sgemm at the single-core AVX2
//! peak (clock-throttled at 32t). The ONE big remaining lever is the owner-gated int8×int8
//! encoder GEMM — memory has "naive int8 = 0.38× (needs a blocked microkernel + VNNI)", but
//! this box has NO VNNI (VPDPBUSD). Before anyone invests in a blocked int8 microkernel, size
//! the CEILING: peak int8 MAC/s (VPMADDUBSW + VPMADDWD, the best AVX2-no-VNNI GEMM path) vs
//! peak f32 MAC/s (VFMADD). This is a compute-bound, L1-resident, latency-hidden (8 independent
//! accumulators) throughput test — the UPPER BOUND on what a perfect int8 encoder GEMM could
//! gain over f32 on this hardware. If int8 ≈ f32 here, the owner-gated lever is DEAD on AVX2;
//! if int8 ≫ f32, a blocked microkernel is worth building.
//! Usage: `int8_vs_f32_compute_ceiling_probe [outer_iters]` (default 200000).
#![allow(unsafe_code)]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
const K: usize = 4096; // per-buffer length; int8 = 4 KB, f32 = 16 KB (L1/L2-resident)

/// Peak f32 MAC/s: 8 independent FMA accumulators over an L1-resident buffer.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn f32_peak(a: &[f32], b: &[f32], iters: usize) -> f32 {
    // SAFETY: callers gate AVX2/FMA and supply K readable elements in each
    // slice; every pointer offset below remains within those K elements.
    unsafe {
        let mut acc = [_mm256_setzero_ps(); 8];
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        for _ in 0..iters {
            let mut k = 0;
            while k + 64 <= K {
                for (j, ac) in acc.iter_mut().enumerate() {
                    let off = k + j * 8;
                    *ac = _mm256_fmadd_ps(
                        _mm256_loadu_ps(pa.add(off)),
                        _mm256_loadu_ps(pb.add(off)),
                        *ac,
                    );
                }
                k += 64;
            }
        }
        let mut s = _mm256_setzero_ps();
        for ac in acc {
            s = _mm256_add_ps(s, ac);
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), s);
        tmp.iter().sum()
    }
}

/// Peak int8 MAC/s: VPMADDUBSW + VPMADDWD (the best AVX2-no-VNNI int8 GEMM inner op),
/// 8 independent int32 accumulators over an L1-resident buffer.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn i8_peak(a: &[i8], b: &[i8], iters: usize) -> i32 {
    // SAFETY: callers gate AVX2 and supply K readable elements in each slice;
    // every pointer offset below remains within those K elements.
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_si256(); 8];
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        for _ in 0..iters {
            let mut k = 0;
            // each 32-byte load = 32 int8; maddubs → 16 int16 partial sums; madd(·,1) → 8 int32.
            while k + 256 <= K {
                for (j, ac) in acc.iter_mut().enumerate() {
                    let off = k + j * 32;
                    let va = _mm256_loadu_si256(pa.add(off).cast());
                    let vb = _mm256_loadu_si256(pb.add(off).cast());
                    let p16 = _mm256_maddubs_epi16(va, vb); // 32×(u8·i8)→16×i16
                    let p32 = _mm256_madd_epi16(p16, ones); // 16×i16 → 8×i32
                    *ac = _mm256_add_epi32(*ac, p32);
                }
                k += 256;
            }
        }
        let mut s = _mm256_setzero_si256();
        for ac in acc {
            s = _mm256_add_epi32(s, ac);
        }
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr().cast(), s);
        tmp.iter().sum()
    }
}

/// int8 via VPMOVSXBW + VPMADDWD — the WIDENING path `nn::dot_i8` and the prior TILED encoder
/// int8 GEMM (examples/encoder_int8_gemm_probe) use. Signed-symmetric int8 (no sign-offset
/// needed), but ~16 MAC per (2 loads + 2 sign-extends + 1 madd + 1 add) vs maddubs' ~32 MAC.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn i8_peak_widening(a: &[i8], b: &[i8], iters: usize) -> i32 {
    // SAFETY: callers gate AVX2 and supply K readable elements in each slice;
    // every pointer offset below remains within those K elements.
    unsafe {
        let mut acc = [_mm256_setzero_si256(); 8];
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        for _ in 0..iters {
            let mut k = 0;
            // each 128-bit load = 16 int8 → sign-extend to 16 int16 → madd → 8 int32.
            while k + 128 <= K {
                for (j, ac) in acc.iter_mut().enumerate() {
                    let off = k + j * 16;
                    let va = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(off).cast()));
                    let vb = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(off).cast()));
                    *ac = _mm256_add_epi32(*ac, _mm256_madd_epi16(va, vb));
                }
                k += 128;
            }
        }
        let mut s = _mm256_setzero_si256();
        for ac in acc {
            s = _mm256_add_epi32(s, ac);
        }
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr().cast(), s);
        tmp.iter().sum()
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        eprintln!("int8_vs_f32_compute_ceiling_probe requires AVX2 and FMA support");
        return;
    }
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let mut st = 0x243F_6A88_85A3_08D3u64;
    let mut b8 = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        (st >> 56) as i8
    };
    let ai: Vec<i8> = (0..K).map(|_| b8()).collect();
    let bi: Vec<i8> = (0..K).map(|_| b8()).collect();
    let af: Vec<f32> = ai.iter().map(|&x| x as f32).collect();
    let bf: Vec<f32> = bi.iter().map(|&x| x as f32).collect();

    #[cfg(target_arch = "x86_64")]
    unsafe {
        // f32: K MACs/iter (K/8 FMAs × 8 lanes). int8: K MACs/iter (K/32 maddubs × 32).
        let macs_f = (iters as f64) * (K as f64);
        let macs_i = (iters as f64) * (K as f64);
        black_box(f32_peak(&af, &bf, 2000));
        black_box(i8_peak(&ai, &bi, 2000));
        let mut tf = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            black_box(f32_peak(&af, &bf, iters));
            tf = tf.min(t.elapsed().as_secs_f64());
        }
        let mut ti = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            black_box(i8_peak(&ai, &bi, iters));
            ti = ti.min(t.elapsed().as_secs_f64());
        }
        black_box(i8_peak_widening(&ai, &bi, 2000));
        let mut tw = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            black_box(i8_peak_widening(&ai, &bi, iters));
            tw = tw.min(t.elapsed().as_secs_f64());
        }
        let gf = macs_f / tf / 1e9;
        let gi = macs_i / ti / 1e9;
        let gw = macs_i / tw / 1e9;
        println!(
            "=== int8 COMPUTE ceiling: maddubs vs widening vs f32 — AVX2, no VNNI, L1-resident, 1 core ==="
        );
        println!("  f32 VFMADD                        : {gf:>7.1} GMAC/s   (1.00x baseline)");
        println!(
            "  int8 VPMOVSXBW+VPMADDWD (widening) : {gw:>7.1} GMAC/s   {:.2}x  ← nn::dot_i8 / prior TILED int8 GEMM",
            gw / gf
        );
        println!(
            "  int8 VPMADDUBSW+VPMADDWD (maddubs) : {gi:>7.1} GMAC/s   {:.2}x  ← untried for the encoder",
            gi / gf
        );
        println!(
            "  maddubs / widening                : {:.2}x  [{}]",
            gi / gw,
            if gi / gw >= 1.4 {
                "maddubs is the missing ingredient — prior tiled int8 (0.89x) used the SLOW widening op"
            } else {
                "maddubs ~= widening — the int8 gap is NOT the inner op"
            }
        );
        println!(
            "  (upper bound: peak instruction throughput, no GEMM load/store/blocking overhead;"
        );
        println!(
            "   real blocked int8 GEMM would be <= this; multi-thread clock-throttle applies to all.)"
        );
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("int8_vs_f32_compute_ceiling_probe requires an x86_64 processor");
}
