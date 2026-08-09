//! Does a 4-token activation-column tile (`dot_i8_4col`) beat the SHIPPED 2-token
//! tile (`dot_i8_2col`, nn.rs, landed 85776f4) on the int8 batched GEMV?
//!
//! `gemv_i8_batch` is WEIGHT-OUTER (`for o { for t { dot_i8(w[o], xi8[t]) } }`), so the
//! L1-hot weight row `w[o]` is sign-extended (`vpmovsxbw`) once per token. 2col shares
//! the weight `cvtepi8` across 2 tokens (1 weight-cvt/token → 0.5). 4col shares it across
//! 4 (→ 0.25 weight-cvt/token), at the cost of 8 i32 accumulators (4 cols × 2 halves) —
//! register-pressure-risky on AVX2's 16 YMM. Diminishing amortization vs 2col; the open
//! question is whether the extra sharing beats the spill risk. MEASURE it.
//!
//! BYTE-EXACT by construction: each column keeps `dot_i8`'s exact madd pairing + 2-accum
//! reduction, so `dot_i8_4col(w,a,b,c,d) == (dot_i8(w,a),dot_i8(w,b),dot_i8(w,c),dot_i8(w,d))`
//! bit-for-bit — the probe asserts it. This is a PER-CORE compute lever, so the 2col↔4col
//! ratio is thread-count-invariant and admissible on any worker (unlike 32-thread
//! scheduling levers — see bd-transcript-gate-unrunnable-xu9g).
//!
//! Run: cargo run --release --example i8batch_4col_probe
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use rayon::prelude::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const INP: usize = 1280;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    let n = w.len().min(x.len());
    unsafe {
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let mut i = 0usize;
        while i + 32 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16).cast()));
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i).cast()));
            let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i + 16).cast()));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, x1));
            i += 32;
        }
        while i + 16 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i).cast()));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
            i += 16;
        }
        let mut acc = hsum256(_mm256_add_epi32(a0, a1));
        while i < n {
            acc += *w.get_unchecked(i) as i32 * *x.get_unchecked(i) as i32;
            i += 1;
        }
        acc
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
unsafe fn hsum256(s: core::arch::x86_64::__m256i) -> i32 {
    use core::arch::x86_64::*;
    unsafe {
        let lo = _mm256_castsi256_si128(s);
        let hi = _mm256_extracti128_si256::<1>(s);
        let q = _mm_add_epi32(lo, hi);
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
        _mm_cvtsi128_si32(q)
    }
}

/// Shipped 2col: one weight row, two activation columns; weight `cvtepi8` shared.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8_2col(w: &[i8], xa: &[i8], xb: &[i8]) -> (i32, i32) {
    use core::arch::x86_64::*;
    let n = w.len().min(xa.len()).min(xb.len());
    let (wp, ap, bp) = (w.as_ptr(), xa.as_ptr(), xb.as_ptr());
    unsafe {
        let mut aa0 = _mm256_setzero_si256();
        let mut aa1 = _mm256_setzero_si256();
        let mut ab0 = _mm256_setzero_si256();
        let mut ab1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16).cast()));
            aa0 = _mm256_add_epi32(
                aa0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i).cast()))),
            );
            aa1 = _mm256_add_epi32(
                aa1,
                _mm256_madd_epi16(
                    w1,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i + 16).cast())),
                ),
            );
            ab0 = _mm256_add_epi32(
                ab0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i).cast()))),
            );
            ab1 = _mm256_add_epi32(
                ab1,
                _mm256_madd_epi16(
                    w1,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i + 16).cast())),
                ),
            );
            i += 32;
        }
        while i + 16 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            aa0 = _mm256_add_epi32(
                aa0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i).cast()))),
            );
            ab0 = _mm256_add_epi32(
                ab0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i).cast()))),
            );
            i += 16;
        }
        let mut acc_a = hsum256(_mm256_add_epi32(aa0, aa1));
        let mut acc_b = hsum256(_mm256_add_epi32(ab0, ab1));
        while i < n {
            let wv = *w.get_unchecked(i) as i32;
            acc_a += wv * *xa.get_unchecked(i) as i32;
            acc_b += wv * *xb.get_unchecked(i) as i32;
            i += 1;
        }
        (acc_a, acc_b)
    }
}

/// NEW 4col: one weight row, FOUR activation columns; weight `cvtepi8` shared across 4.
/// 8 i32 accumulators (4 cols × 2 halves). Same per-column madd pairing as `dot_i8`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8_4col(w: &[i8], xa: &[i8], xb: &[i8], xc: &[i8], xd: &[i8]) -> (i32, i32, i32, i32) {
    use core::arch::x86_64::*;
    let n = w
        .len()
        .min(xa.len())
        .min(xb.len())
        .min(xc.len())
        .min(xd.len());
    let (wp, ap, bp, cp, dp) = (
        w.as_ptr(),
        xa.as_ptr(),
        xb.as_ptr(),
        xc.as_ptr(),
        xd.as_ptr(),
    );
    unsafe {
        let mut aa0 = _mm256_setzero_si256();
        let mut aa1 = _mm256_setzero_si256();
        let mut ab0 = _mm256_setzero_si256();
        let mut ab1 = _mm256_setzero_si256();
        let mut ac0 = _mm256_setzero_si256();
        let mut ac1 = _mm256_setzero_si256();
        let mut ad0 = _mm256_setzero_si256();
        let mut ad1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16).cast()));
            aa0 = _mm256_add_epi32(
                aa0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i).cast()))),
            );
            aa1 = _mm256_add_epi32(
                aa1,
                _mm256_madd_epi16(
                    w1,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i + 16).cast())),
                ),
            );
            ab0 = _mm256_add_epi32(
                ab0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i).cast()))),
            );
            ab1 = _mm256_add_epi32(
                ab1,
                _mm256_madd_epi16(
                    w1,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i + 16).cast())),
                ),
            );
            ac0 = _mm256_add_epi32(
                ac0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i).cast()))),
            );
            ac1 = _mm256_add_epi32(
                ac1,
                _mm256_madd_epi16(
                    w1,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i + 16).cast())),
                ),
            );
            ad0 = _mm256_add_epi32(
                ad0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i).cast()))),
            );
            ad1 = _mm256_add_epi32(
                ad1,
                _mm256_madd_epi16(
                    w1,
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i + 16).cast())),
                ),
            );
            i += 32;
        }
        while i + 16 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            aa0 = _mm256_add_epi32(
                aa0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i).cast()))),
            );
            ab0 = _mm256_add_epi32(
                ab0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i).cast()))),
            );
            ac0 = _mm256_add_epi32(
                ac0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i).cast()))),
            );
            ad0 = _mm256_add_epi32(
                ad0,
                _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i).cast()))),
            );
            i += 16;
        }
        let mut acc_a = hsum256(_mm256_add_epi32(aa0, aa1));
        let mut acc_b = hsum256(_mm256_add_epi32(ab0, ab1));
        let mut acc_c = hsum256(_mm256_add_epi32(ac0, ac1));
        let mut acc_d = hsum256(_mm256_add_epi32(ad0, ad1));
        while i < n {
            let wv = *w.get_unchecked(i) as i32;
            acc_a += wv * *xa.get_unchecked(i) as i32;
            acc_b += wv * *xb.get_unchecked(i) as i32;
            acc_c += wv * *xc.get_unchecked(i) as i32;
            acc_d += wv * *xd.get_unchecked(i) as i32;
            i += 1;
        }
        (acc_a, acc_b, acc_c, acc_d)
    }
}

/// 2col caller: weight-outer band, 2 tokens/step, 1col tail.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn gemv_m2(w: &[i8], xi8: &[i8], out: usize, tq: usize, dst: &mut [f32], workers: usize) {
    let band = out.div_ceil(workers).max(1);
    let bands: Vec<(usize, usize)> = (0..out)
        .step_by(band)
        .map(|o0| (o0, (o0 + band).min(out)))
        .collect();
    let parts: Vec<Vec<f32>> = bands
        .par_iter()
        .map(|&(o0, o1)| {
            let mut local = vec![0.0f32; (o1 - o0) * tq];
            for o in o0..o1 {
                let wrow = &w[o * INP..(o + 1) * INP];
                let mut t = 0;
                while t + 2 <= tq {
                    let (s0, s1) = dot_i8_2col(
                        wrow,
                        &xi8[t * INP..(t + 1) * INP],
                        &xi8[(t + 1) * INP..(t + 2) * INP],
                    );
                    local[(o - o0) * tq + t] = s0 as f32;
                    local[(o - o0) * tq + t + 1] = s1 as f32;
                    t += 2;
                }
                if t < tq {
                    local[(o - o0) * tq + t] = dot_i8(wrow, &xi8[t * INP..(t + 1) * INP]) as f32;
                }
            }
            local
        })
        .collect();
    for (bi, &(o0, o1)) in bands.iter().enumerate() {
        for o in o0..o1 {
            for t in 0..tq {
                dst[o * tq + t] = parts[bi][(o - o0) * tq + t];
            }
        }
    }
}

/// 4col caller: 4 tokens/step, then 2col, then 1col tail. Identical output to gemv_m2.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn gemv_m4(w: &[i8], xi8: &[i8], out: usize, tq: usize, dst: &mut [f32], workers: usize) {
    let band = out.div_ceil(workers).max(1);
    let bands: Vec<(usize, usize)> = (0..out)
        .step_by(band)
        .map(|o0| (o0, (o0 + band).min(out)))
        .collect();
    let parts: Vec<Vec<f32>> = bands
        .par_iter()
        .map(|&(o0, o1)| {
            let mut local = vec![0.0f32; (o1 - o0) * tq];
            for o in o0..o1 {
                let wrow = &w[o * INP..(o + 1) * INP];
                let base = (o - o0) * tq;
                let mut t = 0;
                while t + 4 <= tq {
                    let (s0, s1, s2, s3) = dot_i8_4col(
                        wrow,
                        &xi8[t * INP..(t + 1) * INP],
                        &xi8[(t + 1) * INP..(t + 2) * INP],
                        &xi8[(t + 2) * INP..(t + 3) * INP],
                        &xi8[(t + 3) * INP..(t + 4) * INP],
                    );
                    local[base + t] = s0 as f32;
                    local[base + t + 1] = s1 as f32;
                    local[base + t + 2] = s2 as f32;
                    local[base + t + 3] = s3 as f32;
                    t += 4;
                }
                while t + 2 <= tq {
                    let (s0, s1) = dot_i8_2col(
                        wrow,
                        &xi8[t * INP..(t + 1) * INP],
                        &xi8[(t + 1) * INP..(t + 2) * INP],
                    );
                    local[base + t] = s0 as f32;
                    local[base + t + 1] = s1 as f32;
                    t += 2;
                }
                if t < tq {
                    local[base + t] = dot_i8(wrow, &xi8[t * INP..(t + 1) * INP]) as f32;
                }
            }
            local
        })
        .collect();
    for (bi, &(o0, o1)) in bands.iter().enumerate() {
        for o in o0..o1 {
            for t in 0..tq {
                dst[o * tq + t] = parts[bi][(o - o0) * tq + t];
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        eprintln!("needs avx2");
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut ni = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 40) as i8
        };
        let avail = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let mut evict = vec![1.0f32; 48 * 1024 * 1024 / 4];
        println!("# i8batch 4col-vs-2col probe — available_parallelism={avail}");
        // workers=1 = PURE kernel ratio (maximally admissible, thread-invariant);
        // workers=min(avail,16) = the shipped band structure.
        for workers in [1usize, avail.min(16)] {
            for (out, label) in [(5120usize, "mlp_0[1280,5120]"), (3840, "qkv[1280,3840]")] {
                let w: Vec<i8> = (0..out * INP).map(|_| ni()).collect();
                for tq in [8usize, 64, 200] {
                    let xi8: Vec<i8> = (0..tq * INP).map(|_| ni()).collect();
                    let mut a = vec![0.0f32; out * tq];
                    let mut b = vec![0.0f32; out * tq];
                    gemv_m2(&w, &xi8, out, tq, &mut a, workers);
                    gemv_m4(&w, &xi8, out, tq, &mut b, workers);
                    let ident = a == b;
                    let reps = 80;
                    let (mut b2, mut b4) = (f64::MAX, f64::MAX);
                    for r in 0..reps {
                        // alternate arm order per rep to cancel order bias
                        if r % 2 == 0 {
                            for e in &mut evict {
                                *e *= 1.0000001;
                            }
                            let t = std::time::Instant::now();
                            gemv_m2(&w, &xi8, out, tq, &mut a, workers);
                            b2 = b2.min(ms(t));
                            for e in &mut evict {
                                *e *= 1.0000001;
                            }
                            let t = std::time::Instant::now();
                            gemv_m4(&w, &xi8, out, tq, &mut b, workers);
                            b4 = b4.min(ms(t));
                        } else {
                            for e in &mut evict {
                                *e *= 1.0000001;
                            }
                            let t = std::time::Instant::now();
                            gemv_m4(&w, &xi8, out, tq, &mut b, workers);
                            b4 = b4.min(ms(t));
                            for e in &mut evict {
                                *e *= 1.0000001;
                            }
                            let t = std::time::Instant::now();
                            gemv_m2(&w, &xi8, out, tq, &mut a, workers);
                            b2 = b2.min(ms(t));
                        }
                    }
                    std::hint::black_box(&a);
                    std::hint::black_box(&b);
                    let verdict = if b4 < b2 {
                        "4col FASTER"
                    } else {
                        "4col slower"
                    };
                    println!(
                        "{label} tq={tq:3} {workers:2}t min-{reps}: 2col={b2:.4} ms  4col={b4:.4} ms  ({:.3}× {verdict})  byte-id={ident}",
                        b2 / b4
                    );
                }
            }
        }
        std::hint::black_box(&evict);
    }
}
