//! Decisive 3-way A/B for the tq=1500 cross-K/V projection path
//! (`cross_attn_k/v.forward(encoder_out)` → `gemv_f16_batch` → the row-morsel
//! `gemv_f16_batch_rows`). All at the real `[1500,1280]×[1280,1280]` shape, cold.
//!
//!   M1       — exact replica of the current row-morsel kernel (M1×N1: per activation
//!              row, stream ALL weight rows through `dot_f16c`, re-converting every
//!              weight f16→f32 for each of tq=1500 rows).
//!   M2col    — BYTE-EXACT: an M2 activation-column tile (`dot_f16c_2col`) that shares
//!              each weight-chunk `vcvtph2ps` across 2 activation rows (halving the
//!              cvtph work that ~bounds `dot_f16c`). Each row keeps `dot_f16c`'s exact
//!              4-accumulator + 8-lane-tree reduction ⇒ bit-for-bit == M1.
//!   tiled_f32 — BlackThrush's concurrent `FW_BATCH_GEMV_TILED_F32` route: tiled
//!              dequant of the whole f16 weight → f32, then the external
//!              `ft_kernel_cpu` sgemm. FASTER but NON-byte-exact (sgemm reorders the
//!              summation), the same non-exactness that keeps `cross_proj_f32` gated.
//!
//! Purpose: decide the cross-proj DEFAULT. If tiled_f32 barely beats the BYTE-EXACT
//! M2col, breaking faithfulness (the shipped transcript) buys ~nothing and M2col
//! should be the default; if it wins big, the non-exactness is an owner speed call.
//!
//! Run: cargo run --release --example f16batch_m2col_probe
use half::f16;
use half::slice::HalfFloatSliceExt;
use rayon::prelude::*;

const OUT: usize = 1280;
const INP: usize = 1280;
const TQ: usize = 1500;

/// EXACT replica of nn::dot_f16c: 4 f32 accumulators over f16 weight × f32 act,
/// reduced `((a0+a1)+(a2+a3))` then the 8-lane tree.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "f16c",
    target_feature = "fma"
))]
#[inline]
#[allow(unsafe_code)]
fn dot_f16c(w: &[f16], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = w.len().min(x.len());
    let xp = x.as_ptr();
    unsafe {
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
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
            s += w[i].to_f32() * x[i];
            i += 1;
        }
        s
    }
}

/// M2 activation-column tile: one weight row, two activation rows; the 4 weight-chunk
/// conversions done ONCE and reused across both rows. Byte-identical to two dot_f16c.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "f16c",
    target_feature = "fma"
))]
#[inline]
#[allow(unsafe_code)]
fn dot_f16c_2col(w: &[f16], x0: &[f32], x1: &[f32]) -> (f32, f32) {
    use core::arch::x86_64::*;
    let n = w.len().min(x0.len()).min(x1.len());
    let (p0, p1) = (x0.as_ptr(), x1.as_ptr());
    unsafe {
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let (mut b0, mut b1, mut b2, mut b3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let mut i = 0usize;
        while i + 32 <= n {
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
            let wv = w[i].to_f32();
            s0 += wv * x0[i];
            s1 += wv * x1[i];
            i += 1;
        }
        (s0, s1)
    }
}

/// M3 activation-column tile: one weight row shared across THREE activation rows.
/// 4 converted-weight + 3×4 accumulators = 16 ymm (x loads fold into the fmadd memory
/// operand). Byte-identical to three dot_f16c. Tests whether 3-way cvtph sharing beats
/// M2col's 2-way before register pressure bites.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "f16c",
    target_feature = "fma"
))]
#[inline]
#[allow(unsafe_code)]
fn dot_f16c_3col(w: &[f16], x0: &[f32], x1: &[f32], x2: &[f32]) -> (f32, f32, f32) {
    use core::arch::x86_64::*;
    let n = w.len().min(x0.len()).min(x1.len()).min(x2.len());
    let (p0, p1, p2) = (x0.as_ptr(), x1.as_ptr(), x2.as_ptr());
    unsafe {
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let (mut b0, mut b1, mut b2, mut b3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let (mut c0, mut c1, mut c2, mut c3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let mut i = 0usize;
        while i + 32 <= n {
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
            c0 = _mm256_fmadd_ps(wc0, _mm256_loadu_ps(p2.add(i)), c0);
            c1 = _mm256_fmadd_ps(wc1, _mm256_loadu_ps(p2.add(i + 8)), c1);
            c2 = _mm256_fmadd_ps(wc2, _mm256_loadu_ps(p2.add(i + 16)), c2);
            c3 = _mm256_fmadd_ps(wc3, _mm256_loadu_ps(p2.add(i + 24)), c3);
            i += 32;
        }
        let red = |acc: __m256| -> f32 {
            let mut t = [0.0f32; 8];
            _mm256_storeu_ps(t.as_mut_ptr(), acc);
            ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]))
        };
        let mut s0 = red(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        let mut s1 = red(_mm256_add_ps(_mm256_add_ps(b0, b1), _mm256_add_ps(b2, b3)));
        let mut s2 = red(_mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3)));
        while i + 8 <= n {
            let wc = _mm256_cvtph_ps(_mm_loadu_si128(w.as_ptr().add(i).cast()));
            s0 += red(_mm256_mul_ps(wc, _mm256_loadu_ps(p0.add(i))));
            s1 += red(_mm256_mul_ps(wc, _mm256_loadu_ps(p1.add(i))));
            s2 += red(_mm256_mul_ps(wc, _mm256_loadu_ps(p2.add(i))));
            i += 8;
        }
        while i < n {
            let wv = w[i].to_f32();
            s0 += wv * x0[i];
            s1 += wv * x1[i];
            s2 += wv * x2[i];
            i += 1;
        }
        (s0, s1, s2)
    }
}

fn gemv_m3col(w: &[f16], x: &[f32], out_slice: &mut [f32], workers: usize) {
    let row_band = TQ.div_ceil(workers).max(1);
    out_slice
        .par_chunks_mut(row_band * OUT)
        .enumerate()
        .for_each(|(band, dst_rows)| {
            let t0 = band * row_band;
            let rows = dst_rows.len() / OUT;
            let mut lt = 0;
            while lt + 3 <= rows {
                let t = t0 + lt;
                let x0 = &x[t * INP..(t + 1) * INP];
                let x1 = &x[(t + 1) * INP..(t + 2) * INP];
                let x2 = &x[(t + 2) * INP..(t + 3) * INP];
                for o in 0..OUT {
                    let (s0, s1, s2) = dot_f16c_3col(&w[o * INP..(o + 1) * INP], x0, x1, x2);
                    dst_rows[lt * OUT + o] = s0;
                    dst_rows[(lt + 1) * OUT + o] = s1;
                    dst_rows[(lt + 2) * OUT + o] = s2;
                }
                lt += 3;
            }
            while lt < rows {
                let xr = &x[(t0 + lt) * INP..(t0 + lt + 1) * INP];
                for o in 0..OUT {
                    dst_rows[lt * OUT + o] = dot_f16c(&w[o * INP..(o + 1) * INP], xr);
                }
                lt += 1;
            }
        });
}

/// dot_f16c with the weight ALREADY converted to f32 (skip `vcvtph2ps`): same four
/// accumulators + same `((a0+a1)+(a2+a3))`/8-lane-tree reduction, so for `wf32 =
/// convert(w_f16)` (lossless) it is `== dot_f16c(w_f16, x)` bit-for-bit.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "f16c",
    target_feature = "fma"
))]
#[inline]
#[allow(unsafe_code)]
fn dot_f16corder_f32(w: &[f32], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = w.len().min(x.len());
    let (wp, xp) = (w.as_ptr(), x.as_ptr());
    unsafe {
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
            _mm256_setzero_ps(),
        );
        let mut i = 0usize;
        while i + 32 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(wp.add(i)), _mm256_loadu_ps(xp.add(i)), a0);
            a1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(wp.add(i + 8)),
                _mm256_loadu_ps(xp.add(i + 8)),
                a1,
            );
            a2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(wp.add(i + 16)),
                _mm256_loadu_ps(xp.add(i + 16)),
                a2,
            );
            a3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(wp.add(i + 24)),
                _mm256_loadu_ps(xp.add(i + 24)),
                a3,
            );
            i += 32;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let mut t = [0.0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), acc);
        let mut s = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
        while i + 8 <= n {
            let p = _mm256_mul_ps(_mm256_loadu_ps(wp.add(i)), _mm256_loadu_ps(xp.add(i)));
            _mm256_storeu_ps(t.as_mut_ptr(), p);
            s += ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
            i += 8;
        }
        while i < n {
            s += w[i] * x[i];
            i += 1;
        }
        s
    }
}

/// Token-block-B tile: convert each weight row f16→f32 ONCE per B-token block into an
/// L1 scratch, then do B byte-exact f32 dots against it. Cuts BOTH cvtph and weight
/// L3 re-reads by B× vs the M1 token-outer path (vs M2col's 2×). Weight-stationary
/// within the block (strided output writes); byte-identical (scratch = lossless f16→f32).
fn gemv_mblock(w: &[f16], x: &[f32], out_slice: &mut [f32], workers: usize, b: usize) {
    let row_band = TQ.div_ceil(workers).max(1);
    out_slice
        .par_chunks_mut(row_band * OUT)
        .enumerate()
        .for_each(|(band, dst_rows)| {
            let t0 = band * row_band;
            let rows = dst_rows.len() / OUT;
            let mut scratch = vec![0.0f32; INP];
            let mut bs = 0;
            while bs < rows {
                let be = (bs + b).min(rows);
                for o in 0..OUT {
                    w[o * INP..(o + 1) * INP].convert_to_f32_slice(&mut scratch);
                    for lt in bs..be {
                        let xr = &x[(t0 + lt) * INP..(t0 + lt + 1) * INP];
                        dst_rows[lt * OUT + o] = dot_f16corder_f32(&scratch, xr);
                    }
                }
                bs = be;
            }
        });
}

/// M1 kernel: exact replica of gemv_f16_batch_rows (row-morsel, tq-band direct-write).
fn gemv_m1(w: &[f16], x: &[f32], out_slice: &mut [f32], workers: usize) {
    let row_band = TQ.div_ceil(workers).max(1);
    out_slice
        .par_chunks_mut(row_band * OUT)
        .enumerate()
        .for_each(|(band, dst_rows)| {
            let t0 = band * row_band;
            let rows = dst_rows.len() / OUT;
            for lt in 0..rows {
                let xr = &x[(t0 + lt) * INP..(t0 + lt + 1) * INP];
                let dst = &mut dst_rows[lt * OUT..(lt + 1) * OUT];
                for o in 0..OUT {
                    dst[o] = dot_f16c(&w[o * INP..(o + 1) * INP], xr);
                }
            }
        });
}

/// M2col kernel: pairs of activation rows share each weight-row conversion.
fn gemv_m2col(w: &[f16], x: &[f32], out_slice: &mut [f32], workers: usize) {
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
                let (d0, d1) = dst_rows[lt * OUT..(lt + 2) * OUT].split_at_mut(OUT);
                for o in 0..OUT {
                    let (s0, s1) = dot_f16c_2col(&w[o * INP..(o + 1) * INP], x0, x1);
                    d0[o] = s0;
                    d1[o] = s1;
                }
                lt += 2;
            }
            if lt < rows {
                let xr = &x[(t0 + lt) * INP..(t0 + lt + 1) * INP];
                let dst = &mut dst_rows[lt * OUT..(lt + 1) * OUT];
                for o in 0..OUT {
                    dst[o] = dot_f16c(&w[o * INP..(o + 1) * INP], xr);
                }
            }
        });
}

/// Tiled dequant f16→f32 (BlackThrush's exact TILE_O/TILE_I blocking) into [inp,out].
fn dequant_tiled(w: &[f16]) -> Vec<f32> {
    const TILE_O: usize = 32;
    const TILE_I: usize = 64;
    let mut w_t = vec![0.0f32; OUT * INP];
    let mut scratch = [0.0f32; TILE_I];
    for o0 in (0..OUT).step_by(TILE_O) {
        let o1 = (o0 + TILE_O).min(OUT);
        for i0 in (0..INP).step_by(TILE_I) {
            let i1 = (i0 + TILE_I).min(INP);
            let width = i1 - i0;
            for o in o0..o1 {
                let w_row = &w[o * INP + i0..o * INP + i1];
                w_row.convert_to_f32_slice(&mut scratch[..width]);
                for (di, &wv) in scratch[..width].iter().enumerate() {
                    w_t[(i0 + di) * OUT + o] = wv;
                }
            }
        }
    }
    w_t
}

/// ft sgemm on a pre-transposed f32 weight [inp,out]: `[tq,inp] @ [inp,out]`.
fn sgemm_f32(x: &[f32], w_t: &[f32], out_slice: &mut [f32]) -> bool {
    let lhs =
        ft_core::TensorMeta::from_shape(vec![TQ, INP], ft_core::DType::F32, ft_core::Device::Cpu);
    let rhs =
        ft_core::TensorMeta::from_shape(vec![INP, OUT], ft_core::DType::F32, ft_core::Device::Cpu);
    match ft_kernel_cpu::matmul_tensor_contiguous_f32(x, w_t, &lhs, &rhs) {
        Ok(data) => {
            out_slice.copy_from_slice(&data);
            true
        }
        Err(_) => false,
    }
}

/// BlackThrush's full route: PER-CALL tiled dequant + ft sgemm. NON-byte-exact.
fn gemv_tiled_f32(w: &[f16], x: &[f32], out_slice: &mut [f32]) -> bool {
    let w_t = dequant_tiled(w);
    sgemm_f32(x, &w_t, out_slice)
}

fn ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "fma"
    )))]
    {
        eprintln!("needs f16c+fma (build with RUSTFLAGS=-Ctarget-cpu=native)");
        return;
    }
    let mut s = 0x2545F4914F6CDD1Du64;
    let mut nf = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((s >> 40) as u32 as f32) / (1u64 << 24) as f32 - 0.5) * 2.0
    };
    let w: Vec<f16> = (0..OUT * INP).map(|_| f16::from_f32(nf() * 0.1)).collect();
    let x: Vec<f32> = (0..TQ * INP).map(|_| nf()).collect();

    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(16)
        .min(16);
    let mut a = vec![0.0f32; TQ * OUT];
    let mut b = vec![0.0f32; TQ * OUT];
    let mut c = vec![0.0f32; TQ * OUT];
    let mut e = vec![0.0f32; TQ * OUT];
    gemv_m1(&w, &x, &mut a, workers);
    gemv_m2col(&w, &x, &mut b, workers);
    gemv_mblock(&w, &x, &mut e, workers, 8);
    let mut f = vec![0.0f32; TQ * OUT];
    gemv_m3col(&w, &x, &mut f, workers);
    let tiled_ok = gemv_tiled_f32(&w, &x, &mut c);
    println!("workers={workers}");
    println!("  M2col byte-identical to M1: {}", a == b);
    println!("  mblock(8) byte-identical to M1: {}", a == e);
    println!("  M3col byte-identical to M1: {}", a == f);
    if tiled_ok {
        let mut ndiff = 0usize;
        let mut maxd = 0.0f32;
        for (p, q) in a.iter().zip(&c) {
            if p != q {
                ndiff += 1;
                maxd = maxd.max((p - q).abs());
            }
        }
        println!(
            "  tiled_f32 vs M1: {} diffs / {} (max|Δ|={:e})  => {}",
            ndiff,
            a.len(),
            maxd,
            if ndiff == 0 {
                "byte-exact"
            } else {
                "NON-byte-exact"
            }
        );
    }

    // Pre-dequant ONCE (load-time model): isolates the sgemm from the per-call dequant.
    let w_t = dequant_tiled(&w);
    let mut d = vec![0.0f32; TQ * OUT];

    let reps = 40;
    let mut evict = vec![1.0f32; 48 * 1024 * 1024 / 4];
    let (mut bm1, mut bm2, mut bt, mut bs) = (f64::MAX, f64::MAX, f64::MAX, f64::MAX);
    let (mut bb4, mut bb8, mut bb16, mut bb3) = (f64::MAX, f64::MAX, f64::MAX, f64::MAX);
    for _ in 0..reps {
        for e in evict.iter_mut() {
            *e *= 1.0000001;
        }
        let t = std::time::Instant::now();
        gemv_m1(&w, &x, &mut a, workers);
        bm1 = bm1.min(ms(t));
        for e in evict.iter_mut() {
            *e *= 1.0000001;
        }
        let t = std::time::Instant::now();
        gemv_m2col(&w, &x, &mut b, workers);
        bm2 = bm2.min(ms(t));
        for e in evict.iter_mut() {
            *e *= 1.0000001;
        }
        let t = std::time::Instant::now();
        gemv_mblock(&w, &x, &mut e, workers, 4);
        bb4 = bb4.min(ms(t));
        for ev in evict.iter_mut() {
            *ev *= 1.0000001;
        }
        let t = std::time::Instant::now();
        gemv_mblock(&w, &x, &mut e, workers, 8);
        bb8 = bb8.min(ms(t));
        for ev in evict.iter_mut() {
            *ev *= 1.0000001;
        }
        let t = std::time::Instant::now();
        gemv_mblock(&w, &x, &mut e, workers, 16);
        bb16 = bb16.min(ms(t));
        for ev in evict.iter_mut() {
            *ev *= 1.0000001;
        }
        let t = std::time::Instant::now();
        gemv_m3col(&w, &x, &mut f, workers);
        bb3 = bb3.min(ms(t));
        for ev in evict.iter_mut() {
            *ev *= 1.0000001;
        }
        let t = std::time::Instant::now();
        gemv_tiled_f32(&w, &x, &mut c);
        bt = bt.min(ms(t));
        for ev in evict.iter_mut() {
            *ev *= 1.0000001;
        }
        let t = std::time::Instant::now();
        sgemm_f32(&x, &w_t, &mut d);
        bs = bs.min(ms(t));
    }
    std::hint::black_box(&a);
    std::hint::black_box(&b);
    std::hint::black_box(&c);
    std::hint::black_box(&d);
    std::hint::black_box(&e);
    std::hint::black_box(&evict);
    println!("[{TQ},{INP}]x[{INP},{OUT}] {workers}t cold, min-of-{reps}:");
    println!("   M1 (current row-morsel, byte-exact)        = {bm1:.3} ms");
    println!(
        "   M2col (shared cvtph, BYTE-EXACT)           = {bm2:.3} ms   ({:.3}× vs M1)",
        bm1 / bm2
    );
    println!(
        "   M3col (3-token in-reg, BYTE-EXACT)         = {bb3:.3} ms   ({:.3}× vs M1, {:.3}× vs M2col)",
        bm1 / bb3,
        bm2 / bb3
    );
    println!(
        "   mblock B=4  (convert-once/4tok, BYTE-EXACT)= {bb4:.3} ms   ({:.3}× vs M1, {:.3}× vs M2col)",
        bm1 / bb4,
        bm2 / bb4
    );
    println!(
        "   mblock B=8  (convert-once/8tok, BYTE-EXACT)= {bb8:.3} ms   ({:.3}× vs M1, {:.3}× vs M2col)",
        bm1 / bb8,
        bm2 / bb8
    );
    println!(
        "   mblock B=16 (convert-once/16tk, BYTE-EXACT)= {bb16:.3} ms   ({:.3}× vs M1, {:.3}× vs M2col)",
        bm1 / bb16,
        bm2 / bb16
    );
    println!(
        "   tiled_f32 per-call dequant (NON-exact)     = {bt:.3} ms   ({:.3}× vs M1, {:.3}× vs M2col)",
        bm1 / bt,
        bm2 / bt
    );
    println!(
        "   sgemm-only, load-time dequant (NON-exact)  = {bs:.3} ms   ({:.3}× vs M1, {:.3}× vs M2col)",
        bm1 / bs,
        bm2 / bs
    );
}
