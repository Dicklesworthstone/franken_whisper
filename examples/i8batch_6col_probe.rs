//! Pins the optimal N for the int8 batched-GEMV weight-conversion-sharing tile:
//! does 6col beat the measured 4col (`i8batch_4col_probe`, 1.03–1.11× over 2col)?
//!
//! `dot_i8` uses 2 i32 accumulators/column (i16 `madd` → i32), so N-col needs 2N + 2
//! weight YMM: 4col = 10 (fits), **6col = 14 (fits, 2 spare)**, 8col = 18 (spills).
//! 6col shares the weight `vpmovsxbw` across 6 tokens (0.167/tok) vs 4col's 0.25/tok —
//! recovers ~1/3 of the 2col→4col step, so expect a small, possibly-noise increment.
//! BYTE-EXACT (i32 dots order-independent). PER-CORE ⇒ thread-invariant ⇒ admissible on
//! any rch worker (workers=1 arm). Run: cargo run --release --example i8batch_6col_probe
use rayon::prelude::*;

const INP: usize = 1280;

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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    let n = w.len().min(x.len());
    let (wp, xp) = (w.as_ptr(), x.as_ptr());
    unsafe {
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut i = 0usize;
        while i + 32 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16).cast()));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i).cast()))));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i + 16).cast()))));
            i += 32;
        }
        while i + 16 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i).cast()))));
            i += 16;
        }
        let mut acc = hsum256(_mm256_add_epi32(a0, a1));
        while i < n { acc += *w.get_unchecked(i) as i32 * *x.get_unchecked(i) as i32; i += 1; }
        acc
    }
}

/// 4col: weight cvt shared across 4 tokens (8 accumulators). Same per-col madd as dot_i8.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8_4col(w: &[i8], xa: &[i8], xb: &[i8], xc: &[i8], xd: &[i8]) -> (i32, i32, i32, i32) {
    use core::arch::x86_64::*;
    let n = w.len().min(xa.len()).min(xb.len()).min(xc.len()).min(xd.len());
    let (wp, ap, bp, cp, dp) = (w.as_ptr(), xa.as_ptr(), xb.as_ptr(), xc.as_ptr(), xd.as_ptr());
    unsafe {
        let (mut aa0, mut aa1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let (mut ab0, mut ab1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let (mut ac0, mut ac1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let (mut ad0, mut ad1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let mut i = 0;
        while i + 32 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16).cast()));
            aa0 = _mm256_add_epi32(aa0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i).cast()))));
            aa1 = _mm256_add_epi32(aa1, _mm256_madd_epi16(w1, _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i + 16).cast()))));
            ab0 = _mm256_add_epi32(ab0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i).cast()))));
            ab1 = _mm256_add_epi32(ab1, _mm256_madd_epi16(w1, _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i + 16).cast()))));
            ac0 = _mm256_add_epi32(ac0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i).cast()))));
            ac1 = _mm256_add_epi32(ac1, _mm256_madd_epi16(w1, _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i + 16).cast()))));
            ad0 = _mm256_add_epi32(ad0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i).cast()))));
            ad1 = _mm256_add_epi32(ad1, _mm256_madd_epi16(w1, _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i + 16).cast()))));
            i += 32;
        }
        while i + 16 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            aa0 = _mm256_add_epi32(aa0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i).cast()))));
            ab0 = _mm256_add_epi32(ab0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i).cast()))));
            ac0 = _mm256_add_epi32(ac0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i).cast()))));
            ad0 = _mm256_add_epi32(ad0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i).cast()))));
            i += 16;
        }
        let (mut sa, mut sb) = (hsum256(_mm256_add_epi32(aa0, aa1)), hsum256(_mm256_add_epi32(ab0, ab1)));
        let (mut sc, mut sd) = (hsum256(_mm256_add_epi32(ac0, ac1)), hsum256(_mm256_add_epi32(ad0, ad1)));
        while i < n {
            let wv = *w.get_unchecked(i) as i32;
            sa += wv * *xa.get_unchecked(i) as i32; sb += wv * *xb.get_unchecked(i) as i32;
            sc += wv * *xc.get_unchecked(i) as i32; sd += wv * *xd.get_unchecked(i) as i32;
            i += 1;
        }
        (sa, sb, sc, sd)
    }
}

/// 6col: weight cvt shared across 6 tokens (12 accumulators + 2 weight = 14 YMM).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
fn dot_i8_6col(w: &[i8], xa: &[i8], xb: &[i8], xc: &[i8], xd: &[i8], xe: &[i8], xf: &[i8]) -> (i32, i32, i32, i32, i32, i32) {
    use core::arch::x86_64::*;
    let n = w.len().min(xa.len()).min(xb.len()).min(xc.len()).min(xd.len()).min(xe.len()).min(xf.len());
    let (wp, ap, bp, cp, dp, ep, fp) = (w.as_ptr(), xa.as_ptr(), xb.as_ptr(), xc.as_ptr(), xd.as_ptr(), xe.as_ptr(), xf.as_ptr());
    unsafe {
        let (mut aa0, mut aa1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let (mut ab0, mut ab1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let (mut ac0, mut ac1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let (mut ad0, mut ad1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let (mut ae0, mut ae1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let (mut af0, mut af1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let mut i = 0;
        while i + 32 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16).cast()));
            aa0 = _mm256_add_epi32(aa0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i).cast()))));
            aa1 = _mm256_add_epi32(aa1, _mm256_madd_epi16(w1, _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i + 16).cast()))));
            ab0 = _mm256_add_epi32(ab0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i).cast()))));
            ab1 = _mm256_add_epi32(ab1, _mm256_madd_epi16(w1, _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i + 16).cast()))));
            ac0 = _mm256_add_epi32(ac0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i).cast()))));
            ac1 = _mm256_add_epi32(ac1, _mm256_madd_epi16(w1, _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i + 16).cast()))));
            ad0 = _mm256_add_epi32(ad0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i).cast()))));
            ad1 = _mm256_add_epi32(ad1, _mm256_madd_epi16(w1, _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i + 16).cast()))));
            ae0 = _mm256_add_epi32(ae0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(ep.add(i).cast()))));
            ae1 = _mm256_add_epi32(ae1, _mm256_madd_epi16(w1, _mm256_cvtepi8_epi16(_mm_loadu_si128(ep.add(i + 16).cast()))));
            af0 = _mm256_add_epi32(af0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(fp.add(i).cast()))));
            af1 = _mm256_add_epi32(af1, _mm256_madd_epi16(w1, _mm256_cvtepi8_epi16(_mm_loadu_si128(fp.add(i + 16).cast()))));
            i += 32;
        }
        while i + 16 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i).cast()));
            aa0 = _mm256_add_epi32(aa0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i).cast()))));
            ab0 = _mm256_add_epi32(ab0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i).cast()))));
            ac0 = _mm256_add_epi32(ac0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(cp.add(i).cast()))));
            ad0 = _mm256_add_epi32(ad0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(dp.add(i).cast()))));
            ae0 = _mm256_add_epi32(ae0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(ep.add(i).cast()))));
            af0 = _mm256_add_epi32(af0, _mm256_madd_epi16(w0, _mm256_cvtepi8_epi16(_mm_loadu_si128(fp.add(i).cast()))));
            i += 16;
        }
        let (mut sa, mut sb) = (hsum256(_mm256_add_epi32(aa0, aa1)), hsum256(_mm256_add_epi32(ab0, ab1)));
        let (mut sc, mut sd) = (hsum256(_mm256_add_epi32(ac0, ac1)), hsum256(_mm256_add_epi32(ad0, ad1)));
        let (mut se, mut sf) = (hsum256(_mm256_add_epi32(ae0, ae1)), hsum256(_mm256_add_epi32(af0, af1)));
        while i < n {
            let wv = *w.get_unchecked(i) as i32;
            sa += wv * *xa.get_unchecked(i) as i32; sb += wv * *xb.get_unchecked(i) as i32;
            sc += wv * *xc.get_unchecked(i) as i32; sd += wv * *xd.get_unchecked(i) as i32;
            se += wv * *xe.get_unchecked(i) as i32; sf += wv * *xf.get_unchecked(i) as i32;
            i += 1;
        }
        (sa, sb, sc, sd, se, sf)
    }
}

fn row(xi8: &[i8], t: usize) -> &[i8] { &xi8[t * INP..(t + 1) * INP] }

fn gemv_m4(w: &[i8], xi8: &[i8], out: usize, tq: usize, dst: &mut [f32], workers: usize) {
    let band = out.div_ceil(workers).max(1);
    let bands: Vec<(usize, usize)> = (0..out).step_by(band).map(|o0| (o0, (o0 + band).min(out))).collect();
    let parts: Vec<Vec<f32>> = bands.par_iter().map(|&(o0, o1)| {
        let mut local = vec![0.0f32; (o1 - o0) * tq];
        for o in o0..o1 {
            let wrow = &w[o * INP..(o + 1) * INP];
            let base = (o - o0) * tq;
            let mut t = 0;
            while t + 4 <= tq {
                let (a, b, c, d) = dot_i8_4col(wrow, row(xi8, t), row(xi8, t + 1), row(xi8, t + 2), row(xi8, t + 3));
                local[base + t] = a as f32; local[base + t + 1] = b as f32; local[base + t + 2] = c as f32; local[base + t + 3] = d as f32;
                t += 4;
            }
            while t < tq { local[base + t] = dot_i8(wrow, row(xi8, t)) as f32; t += 1; }
        }
        local
    }).collect();
    for (bi, &(o0, o1)) in bands.iter().enumerate() {
        for o in o0..o1 { for t in 0..tq { dst[o * tq + t] = parts[bi][(o - o0) * tq + t]; } }
    }
}

fn gemv_m6(w: &[i8], xi8: &[i8], out: usize, tq: usize, dst: &mut [f32], workers: usize) {
    let band = out.div_ceil(workers).max(1);
    let bands: Vec<(usize, usize)> = (0..out).step_by(band).map(|o0| (o0, (o0 + band).min(out))).collect();
    let parts: Vec<Vec<f32>> = bands.par_iter().map(|&(o0, o1)| {
        let mut local = vec![0.0f32; (o1 - o0) * tq];
        for o in o0..o1 {
            let wrow = &w[o * INP..(o + 1) * INP];
            let base = (o - o0) * tq;
            let mut t = 0;
            while t + 6 <= tq {
                let (a, b, c, d, e, f) = dot_i8_6col(wrow, row(xi8, t), row(xi8, t + 1), row(xi8, t + 2), row(xi8, t + 3), row(xi8, t + 4), row(xi8, t + 5));
                local[base + t] = a as f32; local[base + t + 1] = b as f32; local[base + t + 2] = c as f32;
                local[base + t + 3] = d as f32; local[base + t + 4] = e as f32; local[base + t + 5] = f as f32;
                t += 6;
            }
            while t + 4 <= tq {
                let (a, b, c, d) = dot_i8_4col(wrow, row(xi8, t), row(xi8, t + 1), row(xi8, t + 2), row(xi8, t + 3));
                local[base + t] = a as f32; local[base + t + 1] = b as f32; local[base + t + 2] = c as f32; local[base + t + 3] = d as f32;
                t += 4;
            }
            while t < tq { local[base + t] = dot_i8(wrow, row(xi8, t)) as f32; t += 1; }
        }
        local
    }).collect();
    for (bi, &(o0, o1)) in bands.iter().enumerate() {
        for o in o0..o1 { for t in 0..tq { dst[o * tq + t] = parts[bi][(o - o0) * tq + t]; } }
    }
}

fn ms(t: std::time::Instant) -> f64 { t.elapsed().as_secs_f64() * 1e3 }

fn main() {
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    { eprintln!("needs avx2"); }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut ni = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 40) as i8 };
        let avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let mut evict = vec![1.0f32; 48 * 1024 * 1024 / 4];
        println!("# i8batch 6col-vs-4col probe — available_parallelism={avail}");
        for workers in [1usize, avail.min(16)] {
            for (out, label) in [(5120usize, "mlp_0[1280,5120]"), (3840, "qkv[1280,3840]")] {
                let w: Vec<i8> = (0..out * INP).map(|_| ni()).collect();
                for tq in [12usize, 64, 204] {
                    let xi8: Vec<i8> = (0..tq * INP).map(|_| ni()).collect();
                    let mut a = vec![0.0f32; out * tq];
                    let mut b = vec![0.0f32; out * tq];
                    gemv_m4(&w, &xi8, out, tq, &mut a, workers);
                    gemv_m6(&w, &xi8, out, tq, &mut b, workers);
                    let ident = a == b;
                    let reps = 80;
                    let (mut b4, mut b6) = (f64::MAX, f64::MAX);
                    for r in 0..reps {
                        if r % 2 == 0 {
                            for e in evict.iter_mut() { *e *= 1.0000001; }
                            let t = std::time::Instant::now(); gemv_m4(&w, &xi8, out, tq, &mut a, workers); b4 = b4.min(ms(t));
                            for e in evict.iter_mut() { *e *= 1.0000001; }
                            let t = std::time::Instant::now(); gemv_m6(&w, &xi8, out, tq, &mut b, workers); b6 = b6.min(ms(t));
                        } else {
                            for e in evict.iter_mut() { *e *= 1.0000001; }
                            let t = std::time::Instant::now(); gemv_m6(&w, &xi8, out, tq, &mut b, workers); b6 = b6.min(ms(t));
                            for e in evict.iter_mut() { *e *= 1.0000001; }
                            let t = std::time::Instant::now(); gemv_m4(&w, &xi8, out, tq, &mut a, workers); b4 = b4.min(ms(t));
                        }
                    }
                    std::hint::black_box(&a); std::hint::black_box(&b);
                    let v = if b6 < b4 { "6col FASTER" } else { "6col slower" };
                    println!("{label} tq={tq:3} {workers:2}t min-{reps}: 4col={b4:.4} ms  6col={b6:.4} ms  ({:.3}× {v})  byte-id={ident}", b4 / b6);
                }
            }
        }
        std::hint::black_box(&evict);
    }
}
