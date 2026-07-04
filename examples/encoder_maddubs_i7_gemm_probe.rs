//! maddubs 7-bit-weight int8 encoder GEMM: MEASURE the ledger's extrapolated ~1.6x (BlackThrush, 2026-07-04).
//!
//! Context: the int8 encoder GEMM is the #1 owner-gated lever (encoder = ~87% e2e,
//! compute-bound f32 sgemm). Two prior facts in docs/NEGATIVE_EVIDENCE.md:
//!   - the WIDENING inner op (vpmovsxbw+vpmaddwd, int32-safe) tiled GEMM = 0.89x f32 (a LOSS);
//!   - VPMADDUBSW (maddubs) is ~1.79x the widening op on raw throughput, BUT it SATURATES its
//!     int16 intermediate for full-range int8xint8 (retraction 4cfcd56) -> unusable at 8-bit.
//! The retraction noted maddubs is "only usable with <=7-bit weights" but DISMISSED that by
//! reasoning ("not worth it, accuracy"), never MEASURING its tiled-GEMM speed. The ledger's
//! "actionable ~1.6x f32" is an EXTRAPOLATION (0.70 efficiency x 2.28x compute). This probe
//! MEASURES the real maddubs-7bit tiled GEMM efficiency on the true turbo encoder shapes so the
//! owner has a number, not an extrapolation.
//!
//! Non-saturation proof: activation -> u8 in [0,255] (symmetric i8 + 128 offset), weight -> i7 in
//! [-63,63]. maddubs pair-product a*w in [-16065,16065], pair-SUM in [-32130,32130] c int16
//! [-32768,32767] => NEVER saturates. The probe asserts the maddubs dot is INTEGER-EXACT vs the
//! reference i7*u8 dot (0 diff) to prove it. Sign-offset: sum_i8(a*w) = maddubs(a+128,w) - 128*sum(w).
//!
//! Build release-perf; run in a CALM window. Usage: `encoder_maddubs_i7_gemm_probe [iters]`.
#![allow(unsafe_code)]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

/// Symmetric per-row int8 (amax/127) then +128 -> u8 in [1,255]. Returns (u8 q, per-row scale).
fn quant_rows_u8(a: &[f32], rows: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut q = vec![0u8; rows * k];
    let mut sc = vec![0.0f32; rows];
    q.par_chunks_mut(k)
        .zip(sc.par_iter_mut())
        .enumerate()
        .for_each(|(r, (qr, s))| {
            let row = &a[r * k..(r + 1) * k];
            let amax = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9);
            let scale = amax / 127.0;
            *s = scale;
            let inv = 1.0 / scale;
            for (d, &x) in qr.iter_mut().zip(row) {
                let i8v = (x * inv).round().clamp(-127.0, 127.0) as i32;
                *d = (i8v + 128) as u8;
            }
        });
    (q, sc)
}

/// Symmetric per-row 7-bit (amax/63, clamp [-63,63]) weight quant. Returns (i7-in-i8 q, scale,
/// per-row weight SUM for the sign-offset correction).
fn quant_rows_i7(b: &[f32], rows: usize, k: usize) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    let mut q = vec![0i8; rows * k];
    let mut sc = vec![0.0f32; rows];
    let mut wsum = vec![0i32; rows];
    q.par_chunks_mut(k)
        .zip(sc.par_iter_mut())
        .zip(wsum.par_iter_mut())
        .enumerate()
        .for_each(|(r, ((qr, s), ws))| {
            let row = &b[r * k..(r + 1) * k];
            let amax = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9);
            let scale = amax / 63.0;
            *s = scale;
            let inv = 1.0 / scale;
            let mut acc = 0i32;
            for (d, &x) in qr.iter_mut().zip(row) {
                let v = (x * inv).round().clamp(-63.0, 63.0) as i32;
                *d = v as i8;
                acc += v;
            }
            *ws = acc;
        });
    (q, sc, wsum)
}

/// Exact scalar reference dot over u8 activation x i7 weight (ground truth for the saturation check).
fn dot_ref(a: &[u8], w: &[i8]) -> i32 {
    a.iter().zip(w).map(|(&x, &y)| (x as i32) * (y as i32)).sum()
}

/// AVX2 maddubs dot: sum_x a[x](u8) * w[x](i7). maddubs -> i16 pair-sums (non-saturating for i7),
/// widened to i32 via madd(_, ones) then accumulated. Returns sum over the FULL u8*i7 products
/// (the -128*sum(w) sign-offset is applied by the caller).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_maddubs(a: &[u8], w: &[i8], k: usize) -> i32 {
    unsafe {
        // 4 independent accumulators (128-elem unroll) hide the add_epi32 latency;
        // integer add is associative => bit-identical to the 1-acc order. Mirrors
        // nn::dot_maddubs_i7 (landed 8996fcb, this cycle upgraded to 4-acc).
        let ones = _mm256_set1_epi16(1);
        let (mut a0, mut a1, mut a2, mut a3) = (
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
            _mm256_setzero_si256(),
        );
        let ap = a.as_ptr();
        let wp = w.as_ptr();
        let d32 = |o: usize| {
            _mm256_madd_epi16(
                _mm256_maddubs_epi16(
                    _mm256_loadu_si256(ap.add(o) as *const __m256i),
                    _mm256_loadu_si256(wp.add(o) as *const __m256i),
                ),
                ones,
            )
        };
        let mut x = 0;
        while x + 128 <= k {
            a0 = _mm256_add_epi32(a0, d32(x));
            a1 = _mm256_add_epi32(a1, d32(x + 32));
            a2 = _mm256_add_epi32(a2, d32(x + 64));
            a3 = _mm256_add_epi32(a3, d32(x + 96));
            x += 128;
        }
        let mut acc = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
        while x + 32 <= k {
            acc = _mm256_add_epi32(acc, d32(x));
            x += 32;
        }
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
        let mut s: i32 = tmp.iter().sum();
        while x < k {
            s += (a[x] as i32) * (w[x] as i32);
            x += 1;
        }
        s
    }
}

/// Widening int8 GEMM (the 0.89x baseline path): scalar i32 dot LLVM-autovec's to vpmovsxbw+vpmaddwd.
fn gemm_i8_widening(qa: &[i8], sa: &[f32], qbt: &[i8], sb: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, crow)| {
        let arow = &qa[i * k..(i + 1) * k];
        let sai = sa[i];
        for o in 0..n {
            let brow = &qbt[o * k..(o + 1) * k];
            let mut acc: i32 = 0;
            for x in 0..k {
                acc += (arow[x] as i32) * (brow[x] as i32);
            }
            crow[o] = acc as f32 * sai * sb[o];
        }
    });
    c
}

/// maddubs 7-bit GEMM with sign-offset. qa=u8 activation, qw=i7 weight (bt-layout [n,k]).
fn gemm_maddubs(
    qa: &[u8],
    sa: &[f32],
    qw: &[i8],
    sw: &[f32],
    wsum: &[i32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, crow)| {
        let arow = &qa[i * k..(i + 1) * k];
        let sai = sa[i];
        for o in 0..n {
            let wrow = &qw[o * k..(o + 1) * k];
            #[cfg(target_arch = "x86_64")]
            let raw = unsafe { dot_maddubs(arow, wrow, k) };
            #[cfg(not(target_arch = "x86_64"))]
            let raw = dot_ref(arow, wrow);
            // sign-offset: true i8-activation dot = raw - 128*sum(w)
            let dot = raw - 128 * wsum[o];
            crow[o] = dot as f32 * sai * sw[o];
        }
    });
    c
}

/// M4-blocked maddubs dot: 4 activation rows vs ONE weight row, the weight loaded
/// ONCE per 32-elem chunk (cuts weight L3 re-reads 4x). Same compute op-count and
/// same 4-independent-chain ILP as the M1 4-accumulator dot, so a speedup isolates
/// WEIGHT-STREAMING bandwidth as the bottleneck (compute-bound => neutral). Each
/// output is bit-identical to `dot_maddubs` (same i32 accumulation set).
#[cfg(target_arch = "x86_64")]
unsafe fn dot_maddubs_m4(a0: &[u8], a1: &[u8], a2: &[u8], a3: &[u8], w: &[i8], k: usize) -> [i32; 4] {
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();
        let (p0, p1, p2, p3, pw) =
            (a0.as_ptr(), a1.as_ptr(), a2.as_ptr(), a3.as_ptr(), w.as_ptr());
        let mut x = 0;
        while x + 32 <= k {
            let wv = _mm256_loadu_si256(pw.add(x) as *const __m256i);
            acc0 = _mm256_add_epi32(
                acc0,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_loadu_si256(p0.add(x) as *const __m256i), wv),
                    ones,
                ),
            );
            acc1 = _mm256_add_epi32(
                acc1,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_loadu_si256(p1.add(x) as *const __m256i), wv),
                    ones,
                ),
            );
            acc2 = _mm256_add_epi32(
                acc2,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_loadu_si256(p2.add(x) as *const __m256i), wv),
                    ones,
                ),
            );
            acc3 = _mm256_add_epi32(
                acc3,
                _mm256_madd_epi16(
                    _mm256_maddubs_epi16(_mm256_loadu_si256(p3.add(x) as *const __m256i), wv),
                    ones,
                ),
            );
            x += 32;
        }
        let hsum = |acc: __m256i| -> i32 {
            let mut t = [0i32; 8];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, acc);
            t.iter().sum()
        };
        let mut r = [hsum(acc0), hsum(acc1), hsum(acc2), hsum(acc3)];
        while x < k {
            let wx = w[x] as i32;
            r[0] += (a0[x] as i32) * wx;
            r[1] += (a1[x] as i32) * wx;
            r[2] += (a2[x] as i32) * wx;
            r[3] += (a3[x] as i32) * wx;
            x += 1;
        }
        r
    }
}

/// M4-blocked maddubs GEMM: process activation rows in blocks of 4, streaming each
/// weight row once per block. Bit-identical output to `gemm_maddubs`.
fn gemm_maddubs_m4(
    qa: &[u8],
    sa: &[f32],
    qw: &[i8],
    sw: &[f32],
    wsum: &[i32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(4 * n).enumerate().for_each(|(blk, cblk)| {
        let r0 = blk * 4;
        let rows = (m - r0).min(4);
        if rows == 4 {
            let a0 = &qa[r0 * k..(r0 + 1) * k];
            let a1 = &qa[(r0 + 1) * k..(r0 + 2) * k];
            let a2 = &qa[(r0 + 2) * k..(r0 + 3) * k];
            let a3 = &qa[(r0 + 3) * k..(r0 + 4) * k];
            for o in 0..n {
                let wrow = &qw[o * k..(o + 1) * k];
                #[cfg(target_arch = "x86_64")]
                let raw = unsafe { dot_maddubs_m4(a0, a1, a2, a3, wrow, k) };
                #[cfg(not(target_arch = "x86_64"))]
                let raw = [
                    dot_ref(a0, wrow),
                    dot_ref(a1, wrow),
                    dot_ref(a2, wrow),
                    dot_ref(a3, wrow),
                ];
                let off = 128 * wsum[o];
                for (j, &rj) in raw.iter().enumerate() {
                    cblk[j * n + o] = (rj - off) as f32 * sa[r0 + j] * sw[o];
                }
            }
        } else {
            for j in 0..rows {
                let arow = &qa[(r0 + j) * k..(r0 + j + 1) * k];
                for o in 0..n {
                    let wrow = &qw[o * k..(o + 1) * k];
                    #[cfg(target_arch = "x86_64")]
                    let raw = unsafe { dot_maddubs(arow, wrow, k) };
                    #[cfg(not(target_arch = "x86_64"))]
                    let raw = dot_ref(arow, wrow);
                    cblk[j * n + o] = (raw - 128 * wsum[o]) as f32 * sa[r0 + j] * sw[o];
                }
            }
        }
    });
    c
}

/// M8-blocked maddubs dot: 8 activation rows vs ONE weight row (cuts weight L3
/// re-reads 8x vs M1). For K>~4000 the 8 activation rows (8*K u8) spill L1.
#[cfg(target_arch = "x86_64")]
unsafe fn dot_maddubs_m8(a: [&[u8]; 8], w: &[i8], k: usize) -> [i32; 8] {
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_si256(); 8];
        let pw = w.as_ptr();
        let p: [*const u8; 8] = [
            a[0].as_ptr(), a[1].as_ptr(), a[2].as_ptr(), a[3].as_ptr(),
            a[4].as_ptr(), a[5].as_ptr(), a[6].as_ptr(), a[7].as_ptr(),
        ];
        let mut x = 0;
        while x + 32 <= k {
            let wv = _mm256_loadu_si256(pw.add(x) as *const __m256i);
            for i in 0..8 {
                acc[i] = _mm256_add_epi32(
                    acc[i],
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_loadu_si256(p[i].add(x) as *const __m256i), wv),
                        ones,
                    ),
                );
            }
            x += 32;
        }
        let hsum = |v: __m256i| -> i32 {
            let mut t = [0i32; 8];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, v);
            t.iter().sum()
        };
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = hsum(acc[i]);
        }
        while x < k {
            let wx = w[x] as i32;
            for i in 0..8 {
                r[i] += (a[i][x] as i32) * wx;
            }
            x += 1;
        }
        r
    }
}

/// M4xN2 2D register tile: 4 activation rows x 2 weight rows = 8 dots per pass.
/// The L1-hot activation is reused across 2 weight rows and each weight across 4
/// activation rows => improves the maddubs/load ratio once M4 has fixed weight
/// bandwidth. Returns [w0: r0..r3, w1: r0..r3]. 8 accumulators.
#[cfg(target_arch = "x86_64")]
unsafe fn dot_maddubs_m4n2(a: [&[u8]; 4], w0: &[i8], w1: &[i8], k: usize) -> [i32; 8] {
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_si256(); 8];
        let (p0, p1, p2, p3) = (a[0].as_ptr(), a[1].as_ptr(), a[2].as_ptr(), a[3].as_ptr());
        let (q0, q1) = (w0.as_ptr(), w1.as_ptr());
        let mut x = 0;
        while x + 32 <= k {
            let wv0 = _mm256_loadu_si256(q0.add(x) as *const __m256i);
            let wv1 = _mm256_loadu_si256(q1.add(x) as *const __m256i);
            let av = [
                _mm256_loadu_si256(p0.add(x) as *const __m256i),
                _mm256_loadu_si256(p1.add(x) as *const __m256i),
                _mm256_loadu_si256(p2.add(x) as *const __m256i),
                _mm256_loadu_si256(p3.add(x) as *const __m256i),
            ];
            for i in 0..4 {
                acc[i] = _mm256_add_epi32(
                    acc[i],
                    _mm256_madd_epi16(_mm256_maddubs_epi16(av[i], wv0), ones),
                );
                acc[4 + i] = _mm256_add_epi32(
                    acc[4 + i],
                    _mm256_madd_epi16(_mm256_maddubs_epi16(av[i], wv1), ones),
                );
            }
            x += 32;
        }
        let hsum = |v: __m256i| -> i32 {
            let mut t = [0i32; 8];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, v);
            t.iter().sum()
        };
        let mut r = [0i32; 8];
        for i in 0..8 {
            r[i] = hsum(acc[i]);
        }
        while x < k {
            let (wx0, wx1) = (w0[x] as i32, w1[x] as i32);
            for i in 0..4 {
                r[i] += (a[i][x] as i32) * wx0;
                r[4 + i] += (a[i][x] as i32) * wx1;
            }
            x += 1;
        }
        r
    }
}

fn gemm_maddubs_m8(
    qa: &[u8], sa: &[f32], qw: &[i8], sw: &[f32], wsum: &[i32], m: usize, k: usize, n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(8 * n).enumerate().for_each(|(blk, cblk)| {
        let r0 = blk * 8;
        let rows = (m - r0).min(8);
        if rows == 8 {
            let a: [&[u8]; 8] = std::array::from_fn(|j| &qa[(r0 + j) * k..(r0 + j + 1) * k]);
            for o in 0..n {
                let wrow = &qw[o * k..(o + 1) * k];
                #[cfg(target_arch = "x86_64")]
                let raw = unsafe { dot_maddubs_m8(a, wrow, k) };
                #[cfg(not(target_arch = "x86_64"))]
                let raw: [i32; 8] = std::array::from_fn(|j| dot_ref(a[j], wrow));
                let off = 128 * wsum[o];
                for (j, &rj) in raw.iter().enumerate() {
                    cblk[j * n + o] = (rj - off) as f32 * sa[r0 + j] * sw[o];
                }
            }
        } else {
            for j in 0..rows {
                let arow = &qa[(r0 + j) * k..(r0 + j + 1) * k];
                for o in 0..n {
                    let wrow = &qw[o * k..(o + 1) * k];
                    #[cfg(target_arch = "x86_64")]
                    let raw = unsafe { dot_maddubs(arow, wrow, k) };
                    #[cfg(not(target_arch = "x86_64"))]
                    let raw = dot_ref(arow, wrow);
                    cblk[j * n + o] = (raw - 128 * wsum[o]) as f32 * sa[r0 + j] * sw[o];
                }
            }
        }
    });
    c
}

fn gemm_maddubs_m4n2(
    qa: &[u8], sa: &[f32], qw: &[i8], sw: &[f32], wsum: &[i32], m: usize, k: usize, n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(4 * n).enumerate().for_each(|(blk, cblk)| {
        let r0 = blk * 4;
        let rows = (m - r0).min(4);
        if rows == 4 {
            let a: [&[u8]; 4] = std::array::from_fn(|j| &qa[(r0 + j) * k..(r0 + j + 1) * k]);
            let mut o = 0;
            while o + 2 <= n {
                let w0 = &qw[o * k..(o + 1) * k];
                let w1 = &qw[(o + 1) * k..(o + 2) * k];
                #[cfg(target_arch = "x86_64")]
                let raw = unsafe { dot_maddubs_m4n2(a, w0, w1, k) };
                #[cfg(not(target_arch = "x86_64"))]
                let raw: [i32; 8] = {
                    let mut rr = [0i32; 8];
                    for i in 0..4 { rr[i] = dot_ref(a[i], w0); rr[4 + i] = dot_ref(a[i], w1); }
                    rr
                };
                let (off0, off1) = (128 * wsum[o], 128 * wsum[o + 1]);
                for i in 0..4 {
                    cblk[i * n + o] = (raw[i] - off0) as f32 * sa[r0 + i] * sw[o];
                    cblk[i * n + o + 1] = (raw[4 + i] - off1) as f32 * sa[r0 + i] * sw[o + 1];
                }
                o += 2;
            }
            while o < n {
                let wrow = &qw[o * k..(o + 1) * k];
                for i in 0..4 {
                    #[cfg(target_arch = "x86_64")]
                    let raw = unsafe { dot_maddubs(a[i], wrow, k) };
                    #[cfg(not(target_arch = "x86_64"))]
                    let raw = dot_ref(a[i], wrow);
                    cblk[i * n + o] = (raw - 128 * wsum[o]) as f32 * sa[r0 + i] * sw[o];
                }
                o += 1;
            }
        } else {
            for j in 0..rows {
                let arow = &qa[(r0 + j) * k..(r0 + j + 1) * k];
                for o in 0..n {
                    let wrow = &qw[o * k..(o + 1) * k];
                    #[cfg(target_arch = "x86_64")]
                    let raw = unsafe { dot_maddubs(arow, wrow, k) };
                    #[cfg(not(target_arch = "x86_64"))]
                    let raw = dot_ref(arow, wrow);
                    cblk[j * n + o] = (raw - 128 * wsum[o]) as f32 * sa[r0 + j] * sw[o];
                }
            }
        }
    });
    c
}

/// L2 weight-panel cache-blocked maddubs GEMM (on top of the M4xN2 microkernel).
/// Each parallel TASK owns a contiguous row-range and loops L2-sized weight PANELS
/// outer, reusing each panel across all its 4-row sub-blocks => each core loads the
/// weight ~once (reuse factor = rows_per_task/4) instead of the naive once-per-row-
/// block (m/4x). Cuts L3 weight traffic ~(rows_per_task/4)x. Bit-identical (same
/// dots, reordered loops, each output written once).
fn gemm_maddubs_l2(
    qa: &[u8], sa: &[f32], qw: &[i8], sw: &[f32], wsum: &[i32], m: usize, k: usize, n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    let pc = ((256 * 1024 / k).max(2)) & !1usize; // even panel width, weight panel ~256KB in L2
    let threads = rayon::current_num_threads().max(1);
    let mut rpt = m.div_ceil(threads);
    rpt = rpt.div_ceil(4) * 4; // multiple of 4 rows
    rpt = rpt.max(4);
    c.par_chunks_mut(rpt * n).enumerate().for_each(|(ti, cblk)| {
        let r_base = ti * rpt;
        let task_rows = (m - r_base).min(rpt);
        let mut o0 = 0;
        while o0 < n {
            let pend = (o0 + pc).min(n);
            let mut rr = 0;
            while rr + 4 <= task_rows {
                let r0 = r_base + rr;
                let a: [&[u8]; 4] = std::array::from_fn(|j| &qa[(r0 + j) * k..(r0 + j + 1) * k]);
                let mut o = o0;
                while o + 2 <= pend {
                    let w0 = &qw[o * k..(o + 1) * k];
                    let w1 = &qw[(o + 1) * k..(o + 2) * k];
                    #[cfg(target_arch = "x86_64")]
                    let raw = unsafe { dot_maddubs_m4n2(a, w0, w1, k) };
                    #[cfg(not(target_arch = "x86_64"))]
                    let raw: [i32; 8] = {
                        let mut z = [0i32; 8];
                        for i in 0..4 { z[i] = dot_ref(a[i], w0); z[4 + i] = dot_ref(a[i], w1); }
                        z
                    };
                    let (off0, off1) = (128 * wsum[o], 128 * wsum[o + 1]);
                    for i in 0..4 {
                        cblk[(rr + i) * n + o] = (raw[i] - off0) as f32 * sa[r0 + i] * sw[o];
                        cblk[(rr + i) * n + o + 1] = (raw[4 + i] - off1) as f32 * sa[r0 + i] * sw[o + 1];
                    }
                    o += 2;
                }
                while o < pend {
                    let wrow = &qw[o * k..(o + 1) * k];
                    #[cfg(target_arch = "x86_64")]
                    let raw = unsafe { dot_maddubs_m4(a[0], a[1], a[2], a[3], wrow, k) };
                    #[cfg(not(target_arch = "x86_64"))]
                    let raw = [dot_ref(a[0], wrow), dot_ref(a[1], wrow), dot_ref(a[2], wrow), dot_ref(a[3], wrow)];
                    let off = 128 * wsum[o];
                    for i in 0..4 {
                        cblk[(rr + i) * n + o] = (raw[i] - off) as f32 * sa[r0 + i] * sw[o];
                    }
                    o += 1;
                }
                rr += 4;
            }
            while rr < task_rows {
                let r = r_base + rr;
                let arow = &qa[r * k..(r + 1) * k];
                let mut o = o0;
                while o < pend {
                    let wrow = &qw[o * k..(o + 1) * k];
                    #[cfg(target_arch = "x86_64")]
                    let raw = unsafe { dot_maddubs(arow, wrow, k) };
                    #[cfg(not(target_arch = "x86_64"))]
                    let raw = dot_ref(arow, wrow);
                    cblk[rr * n + o] = (raw - 128 * wsum[o]) as f32 * sa[r] * sw[o];
                    o += 1;
                }
                rr += 1;
            }
            o0 = pend;
        }
    });
    c
}

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y).abs();
        let denom = x.abs().max(y.abs()).max(1e-6);
        m = m.max(d / denom);
    }
    m
}

fn bench(name: &str, m: usize, k: usize, n: usize, iters: usize) {
    let a = fill(m * k, 0x1234);
    let b = fill(k * n, 0x5678); // [k, n] row-major
    let a_mat = Mat::from_vec(m, k, a.clone());
    let b_mat = Mat::from_vec(k, n, b.clone());

    // f32 sgemm (the real encoder path)
    let cf32 = nn::matmul(&a_mat, &b_mat).expect("matmul");
    for _ in 0..3 {
        black_box(nn::matmul(&a_mat, &b_mat).expect("matmul"));
    }
    let mut best_f32 = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = nn::matmul(&a_mat, &b_mat).expect("matmul");
        best_f32 = best_f32.min(t0.elapsed().as_secs_f64());
        black_box(r);
    }

    // pre-transpose B [k,n] -> bt [n,k]
    let mut bt = vec![0.0f32; n * k];
    for kk in 0..k {
        for o in 0..n {
            bt[o * k + kk] = b[kk * n + o];
        }
    }

    // widening int8 (per-row symmetric i8, the 0.89x baseline)
    let (qa_i8, sa_i8) = {
        // symmetric i8 for A (for the widening path)
        let mut q = vec![0i8; m * k];
        let mut sc = vec![0.0f32; m];
        for r in 0..m {
            let row = &a[r * k..(r + 1) * k];
            let amax = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9);
            let scale = amax / 127.0;
            sc[r] = scale;
            for x in 0..k {
                q[r * k + x] = (row[x] / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
        (q, sc)
    };
    let (qbt_i8, sbt_i8) = {
        let mut q = vec![0i8; n * k];
        let mut sc = vec![0.0f32; n];
        for r in 0..n {
            let row = &bt[r * k..(r + 1) * k];
            let amax = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9);
            let scale = amax / 127.0;
            sc[r] = scale;
            for x in 0..k {
                q[r * k + x] = (row[x] / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
        (q, sc)
    };
    let cwide = gemm_i8_widening(&qa_i8, &sa_i8, &qbt_i8, &sbt_i8, m, k, n);
    for _ in 0..3 {
        black_box(gemm_i8_widening(&qa_i8, &sa_i8, &qbt_i8, &sbt_i8, m, k, n));
    }
    let mut best_wide = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = gemm_i8_widening(&qa_i8, &sa_i8, &qbt_i8, &sbt_i8, m, k, n);
        best_wide = best_wide.min(t0.elapsed().as_secs_f64());
        black_box(r);
    }

    // maddubs 7-bit: A->u8, weight(bt)->i7
    let (qa_u8, sa_u8) = quant_rows_u8(&a, m, k);
    let (qw_i7, sw_i7, wsum) = quant_rows_i7(&bt, n, k);

    // saturation proof: maddubs dot == exact ref dot (row 0 x a few output cols)
    let mut sat_diff: i64 = 0;
    for o in 0..n.min(64) {
        let arow = &qa_u8[0..k];
        let wrow = &qw_i7[o * k..(o + 1) * k];
        #[cfg(target_arch = "x86_64")]
        let got = unsafe { dot_maddubs(arow, wrow, k) };
        #[cfg(not(target_arch = "x86_64"))]
        let got = dot_ref(arow, wrow);
        sat_diff = sat_diff.max(((got - dot_ref(arow, wrow)) as i64).abs());
    }

    let cmad = gemm_maddubs(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n);
    for _ in 0..3 {
        black_box(gemm_maddubs(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n));
    }
    let mut best_mad = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = gemm_maddubs(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n);
        best_mad = best_mad.min(t0.elapsed().as_secs_f64());
        black_box(r);
    }

    // M4-blocked maddubs (weight loaded once per 4 activation rows)
    let cmad4 = gemm_maddubs_m4(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n);
    for _ in 0..3 {
        black_box(gemm_maddubs_m4(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n));
    }
    let mut best_mad4 = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = gemm_maddubs_m4(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n);
        best_mad4 = best_mad4.min(t0.elapsed().as_secs_f64());
        black_box(r);
    }
    // bit-identity: M4 must equal M1 maddubs exactly (same i32 accumulation set)
    let mut m4_diff = 0.0f32;
    for (&x, &y) in cmad4.iter().zip(&cmad) {
        m4_diff = m4_diff.max((x - y).abs());
    }

    // M8-blocked (weight loaded once per 8 rows)
    let cmad8 = gemm_maddubs_m8(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n);
    for _ in 0..3 {
        black_box(gemm_maddubs_m8(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n));
    }
    let mut best_mad8 = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = gemm_maddubs_m8(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n);
        best_mad8 = best_mad8.min(t0.elapsed().as_secs_f64());
        black_box(r);
    }

    // M4xN2 2D tile (activation reused across 2 weight rows)
    let cmad4n2 = gemm_maddubs_m4n2(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n);
    for _ in 0..3 {
        black_box(gemm_maddubs_m4n2(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n));
    }
    let mut best_m4n2 = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = gemm_maddubs_m4n2(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n);
        best_m4n2 = best_m4n2.min(t0.elapsed().as_secs_f64());
        black_box(r);
    }
    // L2 weight-panel cache-blocked (on M4xN2 microkernel)
    let cl2 = gemm_maddubs_l2(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n);
    for _ in 0..3 {
        black_box(gemm_maddubs_l2(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n));
    }
    let mut best_l2 = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = gemm_maddubs_l2(&qa_u8, &sa_u8, &qw_i7, &sw_i7, &wsum, m, k, n);
        best_l2 = best_l2.min(t0.elapsed().as_secs_f64());
        black_box(r);
    }
    let mut m8_diff = 0.0f32;
    for (&x, &y) in cmad8.iter().zip(&cmad) {
        m8_diff = m8_diff.max((x - y).abs());
    }
    let mut m4n2_diff = 0.0f32;
    for (&x, &y) in cmad4n2.iter().zip(&cmad) {
        m4n2_diff = m4n2_diff.max((x - y).abs());
    }
    let mut l2_diff = 0.0f32;
    for (&x, &y) in cl2.iter().zip(&cmad) {
        l2_diff = l2_diff.max((x - y).abs());
    }

    let err_wide = max_rel_err(&cwide, &cf32.data);
    let err_mad = max_rel_err(&cmad, &cf32.data);
    println!(
        "{name:<12} [{m},{k}]x[{k},{n}]  f32 {:.1}ms  mad4 {:.2}x  m4n2 {:.2}x ({:.2}vs4)  L2 {:.1}ms {:.2}x ({:.2}vs-m4n2)  | diffs m4={m4_diff:.0} m8={m8_diff:.0} m4n2={m4n2_diff:.0} L2={l2_diff:.0} relmad={:.3}",
        best_f32 * 1e3,
        best_f32 / best_mad4,
        best_f32 / best_m4n2,
        best_mad4 / best_m4n2,
        best_l2 * 1e3,
        best_f32 / best_l2,
        best_m4n2 / best_l2,
        err_mad,
    );
    let _ = (best_wide, err_wide, best_mad, best_mad8);
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    println!("== turbo encoder GEMM: f32 vs widening-int8 vs maddubs-7bit, best-of-{iters} ==");
    println!("(speedup = f32/variant; sat_diff MUST be 0 = maddubs-7bit is exact/non-saturating)");
    bench("proj", 1500, 1280, 1280, iters);
    bench("mlp fc1", 1500, 1280, 5120, iters);
    bench("mlp fc2", 1500, 5120, 1280, iters);
    // SDPA (per-head, d_head=64) — the biggest IMPROVABLE encoder gap once the
    // linear GEMMs are int8'd. scores = Q@K^T (K=d_head=64, SHORT dot => maddubs
    // setup overhead likely dominates); out = probs@V (K=n_ctx=1500, LONG dot,
    // N=64 => V^T fits L2). Sizes whether int8 attention is worth un-fusing the
    // external f32 SDPA for.
    println!("-- SDPA per-head shapes (d_head=64) --");
    bench("sdpa_scores", 1500, 64, 1500, iters);
    bench("sdpa_out", 1500, 1500, 64, iters);
}
