//! Microbench: is the encoder i7 activation quantize (`nn::quantize_act_i7`)
//! round bottlenecked by a scalarized `f32::round()`?
//!
//! `quantize_act_i7` runs before EVERY encoder maddubs GEMM (q/k/v shared, fc1,
//! fc2) and its inner map is `(v*inv).round().clamp(-127,127) as i32`. Round-half-
//! away has no single AVX2 instruction, so LLVM MAY scalarize it (the documented
//! antipattern fixed for the decoder `gemv_i8` in 7939ee6 — but NOT this encoder
//! path). This probe replicates the exact scalar loop and a hand-AVX2 version
//! (round via `trunc(v + copysign(0.5,v))`, vectorized amax/clamp/pack), asserts
//! byte-identical u8 + scales, and cold-benches both on the real fc1/fc2 shapes.
//!
//! Run: cargo run --release --example quant_round_probe

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Exact replica of `nn::quantize_act_i7`'s inner (scalar `.round()`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn quant_scalar(x: &[f32], rows: usize, cols: usize, out: &mut [u8], scales: &mut [f32]) {
    for r in 0..rows {
        let xr = &x[r * cols..(r + 1) * cols];
        let amax = xr.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
        let row_scale = amax / 127.0;
        scales[r] = row_scale;
        let inv = 1.0 / row_scale;
        let outr = &mut out[r * cols..(r + 1) * cols];
        for (d, &v) in outr.iter_mut().zip(xr) {
            let i8v = (v * inv).round().clamp(-127.0, 127.0) as i32;
            *d = (i8v + 128) as u8;
        }
    }
}

/// Hand-AVX2 quant of ONE row (amax + round-half-away via `trunc(v+copysign(0.5,v))`),
/// returning the row scale. Byte-identical to the scalar reference.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
fn quant_row_avx2_one(xr: &[f32], outr: &mut [u8]) -> f32 {
    let cols = xr.len();
    unsafe {
        let sign = _mm256_set1_ps(-0.0);
        let half = _mm256_set1_ps(0.5);
        let clamp_hi = _mm256_set1_ps(127.0);
        let clamp_lo = _mm256_set1_ps(-127.0);
        let bias = _mm256_set1_epi32(128);
        let mut amx = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= cols {
            let v = _mm256_loadu_ps(xr.as_ptr().add(i));
            amx = _mm256_max_ps(amx, _mm256_andnot_ps(sign, v));
            i += 8;
        }
        let hi = _mm256_extractf128_ps(amx, 1);
        let lo = _mm256_castps256_ps128(amx);
        let m = _mm_max_ps(lo, hi);
        let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        let m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
        let mut amax = _mm_cvtss_f32(m);
        for &v in &xr[i..] {
            amax = amax.max(v.abs());
        }
        amax = amax.max(1e-9);
        let row_scale = amax / 127.0;
        let inv = 1.0 / row_scale;
        let inv_v = _mm256_set1_ps(inv);
        let mut j = 0;
        while j + 8 <= cols {
            let s = _mm256_mul_ps(_mm256_loadu_ps(xr.as_ptr().add(j)), inv_v);
            let cs = _mm256_or_ps(_mm256_and_ps(s, sign), half);
            let rounded =
                _mm256_round_ps(_mm256_add_ps(s, cs), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            let cl = _mm256_min_ps(_mm256_max_ps(rounded, clamp_lo), clamp_hi);
            let ii = _mm256_add_epi32(_mm256_cvttps_epi32(cl), bias);
            let mut tmp = [0i32; 8];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, ii);
            for k in 0..8 {
                outr[j + k] = tmp[k] as u8;
            }
            j += 8;
        }
        for k in j..cols {
            let i8v = (xr[k] * inv).round().clamp(-127.0, 127.0) as i32;
            outr[k] = (i8v + 128) as u8;
        }
        row_scale
    }
}

/// Whole-matrix hand-AVX2 quant (serial), for the byte-identity check + serial bench.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn quant_avx2(x: &[f32], rows: usize, cols: usize, out: &mut [u8], scales: &mut [f32]) {
    for r in 0..rows {
        let xr = &x[r * cols..(r + 1) * cols];
        scales[r] = quant_row_avx2_one(xr, &mut out[r * cols..(r + 1) * cols]);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn now() -> std::time::Instant {
    std::time::Instant::now()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn main() {
    let shapes = [
        (1500usize, 1280usize, "fc1/qkv [1500,1280]"),
        (1500, 5120, "fc2 [1500,5120]"),
    ];
    // deterministic pseudo-random input (no rng dep): a cheap LCG mapped to ~[-6,6]
    for (rows, cols, label) in shapes {
        let n = rows * cols;
        let mut x = vec![0.0f32; n];
        let mut s: u64 = 0x9e3779b97f4a7c15;
        for v in x.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as u32) as f32 / (u32::MAX as f32); // [0,1]
            *v = (u - 0.5) * 12.0;
        }
        // an evict buffer bigger than L3 to force cold reads
        let mut evict = vec![1.0f32; 40 * 1024 * 1024 / 4];

        let mut out_s = vec![0u8; n];
        let mut sc_s = vec![0.0f32; rows];
        let mut out_a = vec![0u8; n];
        let mut sc_a = vec![0.0f32; rows];

        quant_scalar(&x, rows, cols, &mut out_s, &mut sc_s);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        quant_avx2(&x, rows, cols, &mut out_a, &mut sc_a);

        // byte-identity
        let u8_ok = out_s == out_a;
        let sc_ok = sc_s == sc_a;
        let mut maxdiff = 0i32;
        for (a, b) in out_s.iter().zip(&out_a) {
            maxdiff = maxdiff.max((*a as i32 - *b as i32).abs());
        }
        println!(
            "== {label} ==  byte-identical u8: {u8_ok} (max|Δu8|={maxdiff}), scales identical: {sc_ok}"
        );

        let reps = 30;
        let mut best_s = f64::MAX;
        let mut best_a = f64::MAX;
        for _ in 0..reps {
            for e in evict.iter_mut() {
                *e *= 1.0000001;
            }
            let t = now();
            quant_scalar(&x, rows, cols, &mut out_s, &mut sc_s);
            best_s = best_s.min(t.elapsed().as_secs_f64() * 1e3);

            for e in evict.iter_mut() {
                *e *= 1.0000001;
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                let t = now();
                quant_avx2(&x, rows, cols, &mut out_a, &mut sc_a);
                best_a = best_a.min(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        std::hint::black_box(&out_s);
        std::hint::black_box(&out_a);
        std::hint::black_box(&evict);
        println!(
            "   SERIAL   scalar = {best_s:.3} ms   |   AVX2 = {best_a:.3} ms   |   speedup = {:.2}×",
            best_s / best_a
        );

        // --- PARALLEL (rayon par_chunks_mut, mirroring the real quantize_act_i7) ---
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            use rayon::prelude::*;
            let mut best_ps = f64::MAX;
            let mut best_pa = f64::MAX;
            for _ in 0..reps {
                for e in evict.iter_mut() {
                    *e *= 1.0000001;
                }
                let t = now();
                out_s
                    .par_chunks_mut(cols)
                    .zip(sc_s.par_iter_mut())
                    .enumerate()
                    .for_each(|(r, (o, s))| {
                        let xr = &x[r * cols..(r + 1) * cols];
                        let amax = xr.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
                        let inv = 127.0 / amax;
                        *s = amax / 127.0;
                        for (d, &v) in o.iter_mut().zip(xr) {
                            *d = ((v * inv).round().clamp(-127.0, 127.0) as i32 + 128) as u8;
                        }
                    });
                best_ps = best_ps.min(t.elapsed().as_secs_f64() * 1e3);

                for e in evict.iter_mut() {
                    *e *= 1.0000001;
                }
                let t = now();
                out_a
                    .par_chunks_mut(cols)
                    .zip(sc_a.par_iter_mut())
                    .enumerate()
                    .for_each(|(r, (o, s))| {
                        let xr = &x[r * cols..(r + 1) * cols];
                        *s = quant_row_avx2_one(xr, o);
                    });
                best_pa = best_pa.min(t.elapsed().as_secs_f64() * 1e3);
            }
            std::hint::black_box(&out_s);
            std::hint::black_box(&out_a);
            println!(
                "   PARALLEL scalar = {best_ps:.3} ms   |   AVX2 = {best_pa:.3} ms   |   speedup = {:.2}×",
                best_ps / best_pa
            );
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
fn main() {
    eprintln!("quant_round_probe requires an x86_64 processor with AVX2 support");
}
