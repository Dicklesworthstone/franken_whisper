//! Decode self-attn KV read: f32 scalar dot vs f16 scalar-dequant vs f16 F16C-vectorized
//! dequant (BlackThrush, 2026-07-03).
//!
//! `attention_decode_step` reads the f32 KV cache and dots each cached key row with the
//! scaled query (scalar loop). `attention_decode_step_f16` stores the cache as f16 (half
//! the read bandwidth) but dequants each element with a SCALAR `.to_f32()` inside the dot —
//! MEASURED 2x slower (the scalar dequant swamps the bandwidth saving; see NEGATIVE_EVIDENCE
//! / project_self_attn_kv_cache_lever). The OPEN question this probe resolves: does replacing
//! that scalar dequant with hardware F16C `_mm256_cvtph_ps` (8 f16->f32 in ONE instruction)
//! turn the halved read into a net win?
//!
//! Models one decode head's QK^T: q [d_head], K [tk, d_head], scores[j] = <q, K[j]>.
//! Variants:
//!   (1) f32 scalar     — the live `attention_decode_step` inner dot.
//!   (2) f32 AVX2 FMA   — reference: how fast the f32 read goes fully vectorized.
//!   (3) f16 scalar     — the live `attention_decode_step_f16` naive dequant (the 2x-slower one).
//!   (4) f16 F16C AVX2  — THE LEVER: _mm256_cvtph_ps dequant + FMA.
//! f16 storage is lossy (owner-gated, transcript-neutral per FW_KV_F16_SIM); we report the
//! max |Δ| of (4) vs (1) so the precision cost is on record.
//! Usage: `kv_f16c_probe [iters]` (default 4000).
#![allow(unsafe_code)]
#[cfg(target_arch = "x86_64")]
mod x86_probe {
    use std::hint::black_box;
    use std::time::Instant;

    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    /// f32 -> f16 bits via F16C round-to-nearest, for building the f16 cache.
    /// Requires `x.len() % 8 == 0` (all probe cache sizes are multiples of 8).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "f16c,avx")]
    unsafe fn f32_to_f16_bits(x: &[f32]) -> Vec<u16> {
        unsafe {
            let n = x.len();
            assert_eq!(n % 8, 0, "probe cache sizes must be multiples of 8");
            let mut out = vec![0u16; n];
            let mut i = 0;
            while i + 8 <= n {
                let v = _mm256_loadu_ps(x.as_ptr().add(i));
                let h = _mm256_cvtps_ph::<0>(v); // 0 = _MM_FROUND_TO_NEAREST_INT
                _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, h);
                i += 8;
            }
            out
        }
    }

    /// Single f16 bit pattern -> f32 (scalar F16C), for the naive-dequant reference.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "f16c")]
    unsafe fn f16_to_f32(bits: u16) -> f32 {
        let h = _mm_cvtsi32_si128(bits as i32);
        _mm_cvtss_f32(_mm_cvtph_ps(h))
    }

    /// (1) f32 scalar dot — replica of the live `attention_decode_step` inner loop.
    fn scores_f32_scalar(q: &[f32], k: &[f32], tk: usize, d: usize) -> Vec<f32> {
        let mut s = vec![0.0f32; tk];
        for (j, sj) in s.iter_mut().enumerate() {
            let krow = &k[j * d..(j + 1) * d];
            let mut acc = 0.0f32;
            for (dd, &qd) in q.iter().enumerate() {
                acc += qd * krow[dd];
            }
            *sj = acc;
        }
        s
    }

    /// (2) f32 AVX2 FMA dot.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn scores_f32_avx2(q: &[f32], k: &[f32], tk: usize, d: usize) -> Vec<f32> {
        unsafe {
            let mut s = vec![0.0f32; tk];
            for (j, score) in s.iter_mut().enumerate().take(tk) {
                let krow = k.as_ptr().add(j * d);
                let mut acc = _mm256_setzero_ps();
                let mut dd = 0;
                while dd + 8 <= d {
                    let qv = _mm256_loadu_ps(q.as_ptr().add(dd));
                    let kv = _mm256_loadu_ps(krow.add(dd));
                    acc = _mm256_fmadd_ps(qv, kv, acc);
                    dd += 8;
                }
                *score = hsum256(acc);
            }
            s
        }
    }

    /// (3) f16 scalar dequant dot — replica of the live `attention_decode_step_f16`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "f16c")]
    unsafe fn scores_f16_scalar(q: &[f32], k: &[u16], tk: usize, d: usize) -> Vec<f32> {
        unsafe {
            let mut s = vec![0.0f32; tk];
            for (j, sj) in s.iter_mut().enumerate() {
                let krow = &k[j * d..(j + 1) * d];
                let mut acc = 0.0f32;
                for (dd, &qd) in q.iter().enumerate() {
                    acc += qd * f16_to_f32(krow[dd]);
                }
                *sj = acc;
            }
            s
        }
    }

    /// (4) f16 F16C-vectorized dequant dot — THE LEVER: `_mm256_cvtph_ps` dequant + FMA.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn scores_f16_f16c(q: &[f32], k: &[u16], tk: usize, d: usize) -> Vec<f32> {
        unsafe {
            let mut s = vec![0.0f32; tk];
            for (j, score) in s.iter_mut().enumerate().take(tk) {
                let krow = k.as_ptr().add(j * d);
                let mut acc = _mm256_setzero_ps();
                let mut dd = 0;
                while dd + 8 <= d {
                    let qv = _mm256_loadu_ps(q.as_ptr().add(dd));
                    let hv = _mm_loadu_si128(krow.add(dd) as *const __m128i); // 8 f16
                    let kv = _mm256_cvtph_ps(hv); // 8 f16 -> 8 f32, one instr
                    acc = _mm256_fmadd_ps(qv, kv, acc);
                    dd += 8;
                }
                *score = hsum256(acc);
            }
            s
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum256(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps::<1>(v);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<0b01>(s, s));
        _mm_cvtss_f32(s)
    }

    fn bench(tk: usize, d: usize, iters: usize) {
        let mut st = 0xD1B5_4A32_D192_ED03u64;
        let mut nf = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            ((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        };
        let q: Vec<f32> = (0..d).map(|_| nf()).collect();
        let k: Vec<f32> = (0..tk * d).map(|_| nf()).collect();
        let kh = unsafe { f32_to_f16_bits(&k) };

        // Byte-exactness / precision: the lever (4) vs the live f32 path (1).
        let s1 = scores_f32_scalar(&q, &k, tk, d);
        let s4 = unsafe { scores_f16_f16c(&q, &kh, tk, d) };
        let s3 = unsafe { scores_f16_scalar(&q, &kh, tk, d) };
        let maxd_f16 = s1
            .iter()
            .zip(&s4)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // (4) vs (3): both consume the SAME f16 values; difference is only FMA-vs-scalar
        // summation order over d=64 — report it so the reorder cost is on record.
        let maxd_ord = s3
            .iter()
            .zip(&s4)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        macro_rules! time {
            ($f:expr) => {{
                for _ in 0..3 {
                    black_box($f);
                }
                let mut best = f64::INFINITY;
                for _ in 0..iters {
                    let t = Instant::now();
                    black_box($f);
                    best = best.min(t.elapsed().as_secs_f64());
                }
                best
            }};
        }
        let t1 = time!(scores_f32_scalar(&q, &k, tk, d));
        let t2 = time!(unsafe { scores_f32_avx2(&q, &k, tk, d) });
        let t3 = time!(unsafe { scores_f16_scalar(&q, &kh, tk, d) });
        let t4 = time!(unsafe { scores_f16_f16c(&q, &kh, tk, d) });

        println!("tk={tk} d={d}  best-of-{iters}");
        println!(
            "  precision: max|Δ| (4)f16c vs (1)f32 = {maxd_f16:.3e}   (4)vs(3) reorder = {maxd_ord:.3e}"
        );
        println!(
            "  (1) f32 scalar      : {:>8.3} µs  1.00x  [baseline = live decode dot]",
            t1 * 1e6
        );
        println!(
            "  (2) f32 AVX2 FMA    : {:>8.3} µs  {:.2}x",
            t2 * 1e6,
            t1 / t2
        );
        println!(
            "  (3) f16 scalar dq   : {:>8.3} µs  {:.2}x  [live _f16, the 2x-slower naive]",
            t3 * 1e6,
            t1 / t3
        );
        println!(
            "  (4) f16 F16C AVX2   : {:>8.3} µs  {:.2}x  [THE LEVER]  {}",
            t4 * 1e6,
            t1 / t4,
            if t4 < t1 { "WIN vs f32" } else { "loss vs f32" }
        );
    }

    pub fn run() {
        let iters: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(4000);
        println!(
            "=== decode self-attn KV dot: f32 vs f16-scalar vs f16-F16C (turbo d_head=64, 1 thread) ==="
        );
        // turbo: n_state=1280, n_head=20 -> d_head=64. cache_len grows over decode.
        bench(64, 64, iters);
        bench(256, 64, iters);
        bench(448, 64, iters);
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    if !std::is_x86_feature_detected!("avx2")
        || !std::is_x86_feature_detected!("f16c")
        || !std::is_x86_feature_detected!("fma")
    {
        eprintln!("kv_f16c_probe requires AVX2, F16C, and FMA support");
        return;
    }
    x86_probe::run();
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("kv_f16c_probe requires an x86_64 processor");
}
