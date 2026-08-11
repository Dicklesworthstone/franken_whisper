//! Logits-GEMV bandwidth probe: is the [51866,1280] int8 vocab GEMV (biggest
//! DEFAULT-ON decode op, per-token serial critical path) DRAM-bandwidth-SATURATED
//! or LATENCY-bound (headroom)? Compares the real `nn::gemv_i8` against a
//! software-prefetch variant and a 2-row (2 concurrent weight streams => more MLP)
//! variant, reporting effective GB/s. All variants are bit-identical (integer i8
//! dot). If prefetch/2-row raise GB/s, the GEMV is latency-bound and there's a
//! byte-exact default-on lever; if flat, it's bandwidth-saturated (closed).
#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use franken_whisper::native_engine::nn::{self, I8Mat};

#[cfg(target_arch = "x86_64")]
fn fill_i8(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 40) as i32 & 0xff) - 128).clamp(-127, 127) as i8
        })
        .collect()
}

/// int8 dot, 2 accumulators — the same math as nn::dot_i8 (bit-identical).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    unsafe {
        let n = w.len();
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i) as *const __m128i));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16) as *const __m128i));
            let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i + 16) as *const __m128i));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, x1));
            i += 32;
        }
        let s = _mm256_add_epi32(a0, a1);
        let lo = _mm256_castsi256_si128(s);
        let hi = _mm256_extracti128_si256::<1>(s);
        let q = _mm_add_epi32(lo, hi);
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
        let mut acc = _mm_cvtsi128_si32(q);
        while i < n {
            acc += (*w.get_unchecked(i) as i32) * (*x.get_unchecked(i) as i32);
            i += 1;
        }
        acc
    }
}

/// int8 dot with software prefetch AHEAD cache lines into the current row stream.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn dot_i8_pf(w: &[i8], x: &[i8], ahead: usize) -> i32 {
    unsafe {
        let n = w.len();
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            if i + ahead < n {
                _mm_prefetch::<_MM_HINT_T0>(wp.add(i + ahead));
            }
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i) as *const __m128i));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16) as *const __m128i));
            let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i + 16) as *const __m128i));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, x1));
            i += 32;
        }
        let s = _mm256_add_epi32(a0, a1);
        let lo = _mm256_castsi256_si128(s);
        let hi = _mm256_extracti128_si256::<1>(s);
        let q = _mm_add_epi32(lo, hi);
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
        let mut acc = _mm_cvtsi128_si32(q);
        while i < n {
            acc += (*w.get_unchecked(i) as i32) * (*x.get_unchecked(i) as i32);
            i += 1;
        }
        acc
    }
}

/// 2-row dot: interleave loads from TWO consecutive weight rows => 2 concurrent
/// DRAM streams (more MLP). Returns (dot0, dot1). Bit-identical per lane.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn dot_i8_2row(w0: &[i8], w1: &[i8], x: &[i8]) -> (i32, i32) {
    unsafe {
        let n = w0.len();
        let (p0, p1, xp) = (w0.as_ptr(), w1.as_ptr(), x.as_ptr());
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 16 <= n {
            let xv = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i) as *const __m128i));
            let v0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p0.add(i) as *const __m128i));
            let v1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(p1.add(i) as *const __m128i));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(v0, xv));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(v1, xv));
            i += 16;
        }
        let hsum = |s: __m256i| -> i32 {
            let lo = _mm256_castsi256_si128(s);
            let hi = _mm256_extracti128_si256::<1>(s);
            let q = _mm_add_epi32(lo, hi);
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
            _mm_cvtsi128_si32(q)
        };
        let (mut r0, mut r1) = (hsum(a0), hsum(a1));
        while i < n {
            r0 += (*w0.get_unchecked(i) as i32) * (*x.get_unchecked(i) as i32);
            r1 += (*w1.get_unchecked(i) as i32) * (*x.get_unchecked(i) as i32);
            i += 1;
        }
        (r0, r1)
    }
}

#[cfg(target_arch = "x86_64")]
fn gemv_pf(w: &I8Mat, xi8: &[i8], xs: f32, out_slice: &mut [f32], ahead: usize) {
    let (out, inp) = (w.out, w.inp);
    let workers = rayon::current_num_threads().max(1);
    let band = out.div_ceil(workers).max(1);
    out_slice
        .par_chunks_mut(band)
        .enumerate()
        .for_each(|(wk, bs)| {
            let base = wk * band;
            for (i, slot) in bs.iter_mut().enumerate() {
                let o = base + i;
                let row = &w.data[o * inp..(o + 1) * inp];
                #[cfg(target_arch = "x86_64")]
                let acc = unsafe { dot_i8_pf(row, xi8, ahead) } as f32 * w.scales[o] * xs;
                #[cfg(not(target_arch = "x86_64"))]
                let acc = 0.0f32;
                *slot = acc;
            }
        });
}

#[cfg(target_arch = "x86_64")]
fn gemv_2row(w: &I8Mat, xi8: &[i8], xs: f32, out_slice: &mut [f32]) {
    let (out, inp) = (w.out, w.inp);
    let workers = rayon::current_num_threads().max(1);
    let band = (out.div_ceil(workers).max(2) + 1) & !1; // even band
    out_slice
        .par_chunks_mut(band)
        .enumerate()
        .for_each(|(wk, bs)| {
            let base = wk * band;
            let mut i = 0;
            while i + 2 <= bs.len() {
                let o = base + i;
                let r0 = &w.data[o * inp..(o + 1) * inp];
                let r1 = &w.data[(o + 1) * inp..(o + 2) * inp];
                #[cfg(target_arch = "x86_64")]
                {
                    let (d0, d1) = unsafe { dot_i8_2row(r0, r1, xi8) };
                    bs[i] = d0 as f32 * w.scales[o] * xs;
                    bs[i + 1] = d1 as f32 * w.scales[o + 1] * xs;
                }
                i += 2;
            }
            while i < bs.len() {
                let o = base + i;
                let row = &w.data[o * inp..(o + 1) * inp];
                #[cfg(target_arch = "x86_64")]
                {
                    bs[i] = unsafe { dot_i8(row, xi8) } as f32 * w.scales[o] * xs;
                }
                i += 1;
            }
        });
}

#[cfg(target_arch = "x86_64")]
fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let out = 51866usize;
    let inp = 1280usize;
    let bytes = out * inp; // int8 weight bytes streamed/token
    println!(
        "== logits GEMV bandwidth: [{out},{inp}] int8 = {} MB/token, threads={}, best-of-{iters} ==",
        bytes / (1 << 20),
        rayon::current_num_threads()
    );

    let data = fill_i8(out * inp, 0x1111);
    let scales = vec![0.01f32; out];
    let w = I8Mat {
        data,
        scales,
        out,
        inp,
    };
    let x: Vec<f32> = (0..inp).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();

    // pre-quantize activation like gemv_i8 does
    let xamax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    let xs = xamax / 127.0;
    let xinv = 1.0 / xs;
    let xi8: Vec<i8> = x
        .iter()
        .map(|v| (v * xinv).round().clamp(-127.0, 127.0) as i8)
        .collect();

    let mut o_ref = vec![0.0f32; out];
    let mut o_pf = vec![0.0f32; out];
    let mut o_2r = vec![0.0f32; out];

    // reference: the real nn::gemv_i8
    for _ in 0..3 {
        nn::gemv_i8(&w, &x, None, &mut o_ref);
    }
    let mut best_ref = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        nn::gemv_i8(&w, &x, None, &mut o_ref);
        best_ref = best_ref.min(t0.elapsed().as_secs_f64());
    }
    println!(
        "  {:<16} {:.2} ms   {:.1} GB/s (real nn::gemv_i8)",
        "baseline",
        best_ref * 1e3,
        bytes as f64 / best_ref / 1e9
    );

    // prefetch variants (ahead in bytes/elements)
    for ahead in [256usize, 512, 1024] {
        for _ in 0..3 {
            gemv_pf(&w, &xi8, xs, &mut o_pf, ahead);
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t0 = Instant::now();
            gemv_pf(&w, &xi8, xs, &mut o_pf, ahead);
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let diff = o_pf
            .iter()
            .zip(&o_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "  pf(ahead={ahead:<4})     {:.2} ms   {:.1} GB/s  maxdiff={diff:.2e}",
            best * 1e3,
            bytes as f64 / best / 1e9
        );
    }

    // 2-row (2 concurrent streams)
    for _ in 0..3 {
        gemv_2row(&w, &xi8, xs, &mut o_2r);
    }
    let mut best_2r = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        gemv_2row(&w, &xi8, xs, &mut o_2r);
        best_2r = best_2r.min(t0.elapsed().as_secs_f64());
    }
    let diff2 = o_2r
        .iter()
        .zip(&o_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  2row             {:.2} ms   {:.1} GB/s  maxdiff={diff2:.2e}",
        best_2r * 1e3,
        bytes as f64 / best_2r / 1e9
    );

    black_box((o_ref, o_pf, o_2r));
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("logits_gemv_bw_probe requires an x86_64 processor with AVX2 support");
}
