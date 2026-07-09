//! The batched f16 GEMV (`nn::gemv_f16_batch` → `gemv_f16_batch_rows`, the
//! row-morsel kernel) is the tq=1500 workhorse for the per-window cross-K/V
//! projections (`cross_attn_k/v.forward(encoder_out)`, ~2% e2e). Its inner loop is
//! M1×N1: for EACH of the tq=1500 activation rows it streams ALL `out` weight rows
//! through `dot_f16c`, which converts every weight chunk f16→f32 (`vcvtph2ps`) —
//! so the SAME weight is re-converted tq=1500 times. `dot_f16c` is ~cvtph-throughput
//! bound at [1280,1280] (4 cvtph vs 4 fmadd per 32-chunk), so that re-conversion is
//! the bottleneck, not just L3 weight bandwidth.
//!
//! This A/Bs the M1 kernel against an M2 ACTIVATION-column tile (`dot_f16c_2col`):
//! one weight row streamed once, its 4 converted chunks REUSED across 2 activation
//! rows → HALF the cvtph work. It is the transpose of the landed `dot_f16c_2row`
//! (which shares the activation across 2 weight rows for the per-token tq=1 GEMV).
//! Register budget: 4 converted-w + 4+4 accumulators + transient x = 16 ymm (fits
//! Zen3). BYTE-EXACT: each row keeps dot_f16c's exact 4-accumulator + 8-lane-tree
//! reduction, so out[t] and out[t+1] are bit-for-bit the M1 values.
//!
//! Run: cargo run --release --example f16batch_m2col_probe
use rayon::prelude::*;

const OUT: usize = 1280;
const INP: usize = 1280;
const TQ: usize = 1500;

/// EXACT replica of nn::dot_f16c: 4 f32 accumulators over f16 weight × f32 act,
/// reduced `((a0+a1)+(a2+a3))` then the 8-lane tree. Weight is raw u16 (f16 bits).
#[cfg(all(target_arch = "x86_64", target_feature = "f16c", target_feature = "fma"))]
#[inline]
#[allow(unsafe_code)]
fn dot_f16c(w: &[u16], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = w.len().min(x.len());
    let xp = x.as_ptr();
    unsafe {
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        );
        let mut i = 0usize;
        while i + 32 <= n {
            let w0 = _mm_loadu_si128(w.as_ptr().add(i).cast());
            let w1 = _mm_loadu_si128(w.as_ptr().add(i + 8).cast());
            let w2 = _mm_loadu_si128(w.as_ptr().add(i + 16).cast());
            let w3 = _mm_loadu_si128(w.as_ptr().add(i + 24).cast());
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(w0), _mm256_loadu_ps(xp.add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(w1), _mm256_loadu_ps(xp.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(w2), _mm256_loadu_ps(xp.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(w3), _mm256_loadu_ps(xp.add(i + 24)), a3);
            i += 32;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let mut t = [0.0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), acc);
        let mut s = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
        while i + 8 <= n {
            let p = _mm256_mul_ps(
                _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i).cast())),
                _mm256_loadu_ps(xp.add(i)),
            );
            _mm256_storeu_ps(t.as_mut_ptr(), p);
            s += ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
            i += 8;
        }
        while i < n {
            s += f16_to_f32(w[i]) * x[i];
            i += 1;
        }
        s
    }
}

/// M2 activation-column tile: ONE weight row `w`, TWO activation rows `x0`/`x1`.
/// The 4 f16→f32 weight-chunk conversions are done ONCE and reused for both rows,
/// each row keeping its OWN 4 accumulators in dot_f16c's exact reduction order ⇒
/// (dot_f16c(w,x0), dot_f16c(w,x1)) bit-for-bit. Halves the cvtph work.
#[cfg(all(target_arch = "x86_64", target_feature = "f16c", target_feature = "fma"))]
#[inline]
#[allow(unsafe_code)]
fn dot_f16c_2col(w: &[u16], x0: &[f32], x1: &[f32]) -> (f32, f32) {
    use core::arch::x86_64::*;
    let n = w.len().min(x0.len()).min(x1.len());
    let (p0, p1) = (x0.as_ptr(), x1.as_ptr());
    unsafe {
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        );
        let (mut b0, mut b1, mut b2, mut b3) = (
            _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
        );
        let mut i = 0usize;
        while i + 32 <= n {
            // Convert the 4 weight chunks ONCE, reuse across both activation rows.
            let wc0 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i).cast()));
            let wc1 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i + 8).cast()));
            let wc2 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i + 16).cast()));
            let wc3 = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i + 24).cast()));
            a0 = _mm256_fmadd_ps(wc0, _mm256_loadu_ps(p0.add(i)), a0);
            a1 = _mm256_fmadd_ps(wc1, _mm256_loadu_ps(p0.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(wc2, _mm256_loadu_ps(p0.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(wc3, _mm256_loadu_ps(p0.add(i + 24)), a3);
            b0 = _mm256_fmadd_ps(wc0, _mm256_loadu_ps(p1.add(i)), b0);
            b1 = _mm256_fmadd_ps(wc1, _mm256_loadu_ps(p1.add(i + 8)), b1);
            b2 = _mm256_fmadd_ps(wc2, _mm256_loadu_ps(p1.add(i + 16)), b2);
            b3 = _mm256_fmadd_ps(wc3, _mm256_loadu_ps(p1.add(i + 24)), b3);
            i += 32;
        }
        let acca = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let accb = _mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3));
        let mut ta = [0.0f32; 8];
        let mut tb = [0.0f32; 8];
        _mm256_storeu_ps(ta.as_mut_ptr(), acca);
        _mm256_storeu_ps(tb.as_mut_ptr(), accb);
        let mut s0 = ((ta[0] + ta[1]) + (ta[2] + ta[3])) + ((ta[4] + ta[5]) + (ta[6] + ta[7]));
        let mut s1 = ((tb[0] + tb[1]) + (tb[2] + tb[3])) + ((tb[4] + tb[5]) + (tb[6] + tb[7]));
        while i + 8 <= n {
            let wc = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i).cast()));
            let pa = _mm256_mul_ps(wc, _mm256_loadu_ps(p0.add(i)));
            let pb = _mm256_mul_ps(wc, _mm256_loadu_ps(p1.add(i)));
            _mm256_storeu_ps(ta.as_mut_ptr(), pa);
            _mm256_storeu_ps(tb.as_mut_ptr(), pb);
            s0 += ((ta[0] + ta[1]) + (ta[2] + ta[3])) + ((ta[4] + ta[5]) + (ta[6] + ta[7]));
            s1 += ((tb[0] + tb[1]) + (tb[2] + tb[3])) + ((tb[4] + tb[5]) + (tb[6] + tb[7]));
            i += 8;
        }
        while i < n {
            let wv = f16_to_f32(w[i]);
            s0 += wv * x0[i];
            s1 += wv * x1[i];
            i += 1;
        }
        (s0, s1)
    }
}

#[allow(unsafe_code)]
fn f16_to_f32(h: u16) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm_cvtph_ps(_mm_set1_epi16(h as i16));
        _mm_cvtss_f32(v)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "f16c")))]
    {
        let _ = h;
        0.0
    }
}

/// M1 kernel: exact replica of gemv_f16_batch_rows (row-morsel, direct-write by
/// tq-band; per act row, loop weight rows via dot_f16c).
fn gemv_m1(w: &[u16], x: &[f32], out_slice: &mut [f32], workers: usize) {
    let row_band = TQ.div_ceil(workers).max(1);
    out_slice
        .par_chunks_mut(row_band * OUT)
        .enumerate()
        .for_each(|(band, dst_rows)| {
            let t0 = band * row_band;
            let rows = dst_rows.len() / OUT;
            for local_t in 0..rows {
                let t = t0 + local_t;
                let xr = &x[t * INP..(t + 1) * INP];
                let dst = &mut dst_rows[local_t * OUT..(local_t + 1) * OUT];
                for o in 0..OUT {
                    dst[o] = dot_f16c(&w[o * INP..(o + 1) * INP], xr);
                }
            }
        });
}

/// M2 kernel: same tq-band partition + direct-write, but pairs of activation rows
/// share each weight-row conversion via dot_f16c_2col. Odd tail row → dot_f16c.
fn gemv_m2col(w: &[u16], x: &[f32], out_slice: &mut [f32], workers: usize) {
    let row_band = TQ.div_ceil(workers).max(1);
    out_slice
        .par_chunks_mut(row_band * OUT)
        .enumerate()
        .for_each(|(band, dst_rows)| {
            let t0 = band * row_band;
            let rows = dst_rows.len() / OUT;
            let mut lt = 0;
            while lt + 2 <= rows {
                let t = t0 + lt;
                let x0 = &x[t * INP..(t + 1) * INP];
                let x1 = &x[(t + 1) * INP..(t + 2) * INP];
                // split the two output rows so we can write both inside the loop
                let (d0, d1) = dst_rows[lt * OUT..(lt + 2) * OUT].split_at_mut(OUT);
                for o in 0..OUT {
                    let wr = &w[o * INP..(o + 1) * INP];
                    let (s0, s1) = dot_f16c_2col(wr, x0, x1);
                    d0[o] = s0;
                    d1[o] = s1;
                }
                lt += 2;
            }
            if lt < rows {
                let t = t0 + lt;
                let xr = &x[t * INP..(t + 1) * INP];
                let dst = &mut dst_rows[lt * OUT..(lt + 1) * OUT];
                for o in 0..OUT {
                    dst[o] = dot_f16c(&w[o * INP..(o + 1) * INP], xr);
                }
            }
        });
}

fn ms(t: std::time::Instant) -> f64 { t.elapsed().as_secs_f64() * 1e3 }

fn main() {
    #[cfg(not(all(target_arch = "x86_64", target_feature = "f16c", target_feature = "fma")))]
    {
        eprintln!("needs f16c+fma (build with RUSTFLAGS=-Ctarget-cpu=native)");
        return;
    }
    let mut s = 0x2545F4914F6CDD1Du64;
    let mut nf = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (((s >> 40) as u32 as f32) / (1u64 << 24) as f32 - 0.5) * 2.0 };
    // f16 weight bits: round a small f32 to f16 via the hardware path.
    #[allow(unsafe_code)]
    let f32_to_f16 = |v: f32| -> u16 {
        #[cfg(all(target_arch = "x86_64", target_feature = "f16c"))]
        unsafe {
            use core::arch::x86_64::*;
            let p = _mm_cvtps_ph(_mm_set1_ps(v), _MM_FROUND_TO_NEAREST_INT);
            _mm_extract_epi16(p, 0) as u16
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "f16c")))]
        { let _ = v; 0u16 }
    };
    let w: Vec<u16> = (0..OUT * INP).map(|_| f32_to_f16(nf() * 0.1)).collect();
    let x: Vec<f32> = (0..TQ * INP).map(|_| nf()).collect();

    let workers = num_cpus_min16();
    let mut a = vec![0.0f32; TQ * OUT];
    let mut b = vec![0.0f32; TQ * OUT];
    gemv_m1(&w, &x, &mut a, workers);
    gemv_m2col(&w, &x, &mut b, workers);
    let identical = a == b;
    println!("workers={workers}  byte-identical M1==M2col: {identical}");
    if !identical {
        let mut ndiff = 0usize;
        let mut maxd = 0.0f32;
        for (p, q) in a.iter().zip(&b) { if p != q { ndiff += 1; maxd = maxd.max((p - q).abs()); } }
        println!("  !! {ndiff} diffs, max|Δ|={maxd:e}");
    }

    let reps = 40;
    let mut evict = vec![1.0f32; 48 * 1024 * 1024 / 4];
    let (mut bm1, mut bm2) = (f64::MAX, f64::MAX);
    for _ in 0..reps {
        for e in evict.iter_mut() { *e *= 1.0000001; }
        let t = std::time::Instant::now(); gemv_m1(&w, &x, &mut a, workers); bm1 = bm1.min(ms(t));
        for e in evict.iter_mut() { *e *= 1.0000001; }
        let t = std::time::Instant::now(); gemv_m2col(&w, &x, &mut b, workers); bm2 = bm2.min(ms(t));
    }
    std::hint::black_box(&a); std::hint::black_box(&b); std::hint::black_box(&evict);
    println!("[{TQ},{INP}]x[{INP},{OUT}] {workers}t cold, min-of-{reps}:");
    println!("   M1 (current row-morsel)   = {bm1:.3} ms");
    println!("   M2col (shared cvtph)      = {bm2:.3} ms   ({:.3}× {})",
        bm1 / bm2, if bm2 < bm1 { "FASTER" } else { "slower ⇒ DEAD" });
}

fn num_cpus_min16() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(16).min(16)
}
