//! Microbench: can the encoder `attn.out` GEMM (currently i8 `vpmaddwd` M4×N2,
//! ~2.6× the maddubs MAC rate but forced because full-i8 weight can't use the
//! saturating `vpmaddubsw`) be beaten by a BYTE-EXACT 2-pass maddubs decomposition
//! `w = w_a + w_b` (each |·|≤64, no i16 saturation) so `dot(w_a,a)+dot(w_b,a) =
//! dot(w,a)` exactly, using the faster 32-wide `vpmaddubsw`?
//!
//! Analysis predicts DEAD at the real M4×N2 blocking (2-pass = 0.1875 ops/MAC +
//! 2× weight bytes vs vpmaddwd 0.172, because M4×N2 amortizes vpmaddwd's activation
//! sign-extend and vpmaddwd fuses multiply+widen in ONE instr). This probe MEASURES
//! it on the real attn.out shape and asserts the 2-pass i32 dot == the vpmaddwd i32.
//!
//! Run: cargo run --release --example attn_out_2pass_probe

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const M: usize = 1500; // frames
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const K: usize = 1280; // n_state (contraction)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const N: usize = 1280; // n_state (output)

/// Scalar reference i32 dot of one output (o) for one activation row (r): Σ a[r,k]·w[o,k].
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn ref_dot(a: &[i8], w: &[i8], r: usize, o: usize) -> i32 {
    let ar = &a[r * K..r * K + K];
    let wr = &w[o * K..o * K + K];
    ar.iter().zip(wr).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// vpmaddwd M4×N2 (mirrors encoder::dot_i8_m4n2): sign-extend both i8 operands to
/// i16, vpmaddwd (16 MAC/instr, fused multiply+widen), 8 i32 accumulators.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
fn gemm_vpmaddwd(a: &[i8], w: &[i8], out: &mut [i32]) {
    use rayon::prelude::*;
    // parallel over blocks of 4 activation rows
    out.par_chunks_mut(4 * N)
        .enumerate()
        .for_each(|(blk, cout)| {
            let r0 = blk * 4;
            if r0 + 4 > M {
                // tail: scalar
                for rr in r0..M.min(r0 + 4) {
                    for o in 0..N {
                        cout[(rr - r0) * N + o] = ref_dot(a, w, rr, o);
                    }
                }
                return;
            }
            unsafe {
                let ap: [*const i8; 4] = [
                    a.as_ptr().add(r0 * K),
                    a.as_ptr().add((r0 + 1) * K),
                    a.as_ptr().add((r0 + 2) * K),
                    a.as_ptr().add((r0 + 3) * K),
                ];
                let mut o = 0;
                while o + 2 <= N {
                    let w0 = w.as_ptr().add(o * K);
                    let w1 = w.as_ptr().add((o + 1) * K);
                    let mut acc = [_mm256_setzero_si256(); 8]; // 4 rows × 2 cols
                    let mut k = 0;
                    while k + 16 <= K {
                        let wv0 =
                            _mm256_cvtepi8_epi16(_mm_loadu_si128(w0.add(k) as *const __m128i));
                        let wv1 =
                            _mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(k) as *const __m128i));
                        for r in 0..4 {
                            let av = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                                ap[r].add(k) as *const __m128i
                            ));
                            acc[r * 2] = _mm256_add_epi32(acc[r * 2], _mm256_madd_epi16(av, wv0));
                            acc[r * 2 + 1] =
                                _mm256_add_epi32(acc[r * 2 + 1], _mm256_madd_epi16(av, wv1));
                        }
                        k += 16;
                    }
                    for r in 0..4 {
                        cout[r * N + o] = hsum_i32(acc[r * 2]);
                        cout[r * N + o + 1] = hsum_i32(acc[r * 2 + 1]);
                    }
                    o += 2;
                }
            }
        });
}

/// 2-pass maddubs M4×N2: activation u8 (= i8+128), weight split w_a/w_b (each ≤64),
/// two vpmaddubsw+vpmaddwd passes accumulated, minus 128·Σ(w_a+w_b). Byte-exact i32.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
fn gemm_2pass_maddubs(au8: &[u8], wa: &[i8], wb: &[i8], colsum: &[i32], out: &mut [i32]) {
    use rayon::prelude::*;
    out.par_chunks_mut(4 * N)
        .enumerate()
        .for_each(|(blk, cout)| {
            let r0 = blk * 4;
            if r0 + 4 > M {
                return; // ignore tail for the bench (M=1500 divisible by 4)
            }
            unsafe {
                let ones = _mm256_set1_epi16(1);
                let ap: [*const u8; 4] = [
                    au8.as_ptr().add(r0 * K),
                    au8.as_ptr().add((r0 + 1) * K),
                    au8.as_ptr().add((r0 + 2) * K),
                    au8.as_ptr().add((r0 + 3) * K),
                ];
                let mut o = 0;
                while o + 2 <= N {
                    let wa0 = wa.as_ptr().add(o * K);
                    let wa1 = wa.as_ptr().add((o + 1) * K);
                    let wb0 = wb.as_ptr().add(o * K);
                    let wb1 = wb.as_ptr().add((o + 1) * K);
                    let mut acc = [_mm256_setzero_si256(); 8];
                    let mut k = 0;
                    while k + 32 <= K {
                        let wa0v = _mm256_loadu_si256(wa0.add(k) as *const __m256i);
                        let wa1v = _mm256_loadu_si256(wa1.add(k) as *const __m256i);
                        let wb0v = _mm256_loadu_si256(wb0.add(k) as *const __m256i);
                        let wb1v = _mm256_loadu_si256(wb1.add(k) as *const __m256i);
                        for r in 0..4 {
                            let av = _mm256_loadu_si256(ap[r].add(k) as *const __m256i);
                            // pass A (w_a) + pass B (w_b), widen each via vpmaddwd(·,ones)
                            let pa0 = _mm256_madd_epi16(_mm256_maddubs_epi16(av, wa0v), ones);
                            let pb0 = _mm256_madd_epi16(_mm256_maddubs_epi16(av, wb0v), ones);
                            acc[r * 2] = _mm256_add_epi32(acc[r * 2], _mm256_add_epi32(pa0, pb0));
                            let pa1 = _mm256_madd_epi16(_mm256_maddubs_epi16(av, wa1v), ones);
                            let pb1 = _mm256_madd_epi16(_mm256_maddubs_epi16(av, wb1v), ones);
                            acc[r * 2 + 1] =
                                _mm256_add_epi32(acc[r * 2 + 1], _mm256_add_epi32(pa1, pb1));
                        }
                        k += 32;
                    }
                    for r in 0..4 {
                        cout[r * N + o] = hsum_i32(acc[r * 2]) - 128 * colsum[o];
                        cout[r * N + o + 1] = hsum_i32(acc[r * 2 + 1]) - 128 * colsum[o + 1];
                    }
                    o += 2;
                }
            }
        });
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
#[inline]
fn hsum_i32(v: __m256i) -> i32 {
    unsafe {
        let hi = _mm256_extracti128_si256(v, 1);
        let lo = _mm256_castsi256_si128(v);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        _mm_cvtsi128_si32(s)
    }
}

fn main() {
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        println!("needs x86_64 + avx2");
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // synthetic i8 activation + i8 weight (deterministic LCG)
        let mut a = vec![0i8; M * K];
        let mut w = vec![0i8; N * K];
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut nextb = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as u32) & 0xff) as i32
        };
        for v in a.iter_mut() {
            *v = (nextb() - 128).clamp(-127, 127) as i8;
        }
        for v in w.iter_mut() {
            *v = (nextb() - 128).clamp(-127, 127) as i8;
        }
        // u8 activation (+128) and weight split w = w_a + w_b, |w_a|≤63, |w_b|≤64
        let au8: Vec<u8> = a.iter().map(|&x| (x as i16 + 128) as u8).collect();
        let mut wa = vec![0i8; N * K];
        let mut wb = vec![0i8; N * K];
        for i in 0..N * K {
            let x = w[i] as i32;
            let xa = x.clamp(-63, 63);
            wa[i] = xa as i8;
            wb[i] = (x - xa) as i8; // |·| ≤ 64
        }
        let mut colsum = vec![0i32; N];
        for o in 0..N {
            colsum[o] = w[o * K..o * K + K].iter().map(|&x| x as i32).sum();
        }

        let mut out_v = vec![0i32; M * N];
        let mut out_m = vec![0i32; M * N];
        gemm_vpmaddwd(&a, &w, &mut out_v);
        gemm_2pass_maddubs(&au8, &wa, &wb, &colsum, &mut out_m);

        // byte-exact check vs scalar reference + cross-check
        let mut ok = true;
        for &(r, o) in &[
            (0usize, 0usize),
            (1, 1),
            (7, 1023),
            (1499, 1279),
            (513, 777),
        ] {
            let rf = ref_dot(&a, &w, r, o);
            if out_v[r * N + o] != rf || out_m[r * N + o] != rf {
                ok = false;
                println!(
                    "MISMATCH r={r} o={o}: ref={rf} vpmaddwd={} 2pass={}",
                    out_v[r * N + o],
                    out_m[r * N + o]
                );
            }
        }
        let equal = out_v == out_m;
        println!(
            "i32-exact vpmaddwd==2pass==ref: {} (full-array equal: {})",
            ok, equal
        );

        let reps = 30;
        let mut evict = vec![1.0f32; 48 * 1024 * 1024 / 4];
        let (mut bv, mut bm) = (f64::MAX, f64::MAX);
        for _ in 0..reps {
            for e in evict.iter_mut() {
                *e *= 1.0000001;
            }
            let t = std::time::Instant::now();
            gemm_vpmaddwd(&a, &w, &mut out_v);
            bv = bv.min(t.elapsed().as_secs_f64() * 1e3);
            for e in evict.iter_mut() {
                *e *= 1.0000001;
            }
            let t = std::time::Instant::now();
            gemm_2pass_maddubs(&au8, &wa, &wb, &colsum, &mut out_m);
            bm = bm.min(t.elapsed().as_secs_f64() * 1e3);
        }
        std::hint::black_box(&out_v);
        std::hint::black_box(&out_m);
        std::hint::black_box(&evict);
        println!(
            "attn.out [{M},{K}]x[{K},{N}] (32t, cold, min-of-{reps}):  vpmaddwd = {bv:.3} ms  |  2-pass maddubs = {bm:.3} ms  |  2pass/vpmaddwd = {:.3}× {}",
            bm / bv,
            if bm < bv {
                "(2-pass FASTER)"
            } else {
                "(vpmaddwd faster ⇒ 2-pass DEAD)"
            }
        );
    }
}
