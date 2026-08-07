//! Self-attn scores dot: f32-KV (current default) vs f16-KV naive scalar `to_f32`
//! (the measured-2×-slower dead path) vs f16-KV with VECTORIZED F16C batch dequant
//! (the missing piece per project_self_attn_kv_cache_lever). (BlackThrush, 2026-07-03).
//!
//! `attention_decode_step` reads an f32 KV cache and does a byte-exact scalar dot over
//! d_head=64 per cached key j, per head. It's BANDWIDTH-bound on the f32 cache read
//! (memory: "self_attn = 35% f32-KV read"). The f16-KV cache halves that read and is
//! transcript-neutral (proven via FW_KV_F16_SIM), BUT the built f16 path
//! (`attention_decode_step_f16`) does `krow[d].to_f32()` SCALAR per element → memory
//! measured it 2× SLOWER ("DEAD absent vectorized F16C dequant"). This probe tests the
//! missing ingredient: batch-convert the f16 k-row via `_mm256_cvtph_ps` (8 f16→f32/instr)
//! into f32 scratch, then the SAME byte-exact d-ascending scalar dot. That is bit-identical
//! to the scalar f16 path (f16→f32 is lossless, same sum order), so it inherits the f16
//! path's transcript-neutrality — the only question is whether it's finally FASTER than f32.
//! Realistic turbo self-attn: n_state=1280, n_head=20, d_head=64, varying tk.
//! Usage: `self_attn_f16c_dequant_probe [iters]`.
#![allow(unsafe_code)]
#[cfg(target_arch = "x86_64")]
mod x86_probe {
    use std::arch::x86_64::*;
    use std::hint::black_box;
    use std::time::Instant;

    const N_STATE: usize = 1280;
    const N_HEAD: usize = 20;
    const D_HEAD: usize = N_STATE / N_HEAD; // 64

    /// Correct scalar f16(bits)→f32 (IEEE, matches `half`/F16C), for the naive path.
    fn f16_to_f32(h: u16) -> f32 {
        let sign = (h >> 15) & 1;
        let exp = (h >> 10) & 0x1f;
        let mant = h & 0x3ff;
        let bits: u32 = if exp == 0 {
            if mant == 0 {
                (sign as u32) << 31
            } else {
                // subnormal
                let mut e = -1i32;
                let mut m = mant as u32;
                loop {
                    e += 1;
                    m <<= 1;
                    if m & 0x400 != 0 {
                        break;
                    }
                }
                let m = m & 0x3ff;
                ((sign as u32) << 31) | (((127 - 15 - e) as u32) << 23) | (m << 13)
            }
        } else if exp == 0x1f {
            ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
        } else {
            ((sign as u32) << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | ((mant as u32) << 13)
        };
        f32::from_bits(bits)
    }

    /// f32-KV: byte-exact scalar dot over the f32 cache (current default path).
    fn scores_f32(k: &[f32], qh: &[f32], scale: f32, tk: usize, base: usize, out: &mut [f32]) {
        for (j, sj) in out.iter_mut().enumerate().take(tk) {
            let krow = &k[j * N_STATE + base..j * N_STATE + base + D_HEAD];
            let mut acc = 0.0f32;
            for (d, &qd) in qh.iter().enumerate() {
                acc += qd * (krow[d] * scale);
            }
            *sj = acc;
        }
    }

    /// f16-KV naive: scalar `to_f32()` per element inside the dot (the 2×-slower dead path).
    fn scores_f16_naive(
        k: &[u16],
        qh: &[f32],
        scale: f32,
        tk: usize,
        base: usize,
        out: &mut [f32],
    ) {
        for (j, sj) in out.iter_mut().enumerate().take(tk) {
            let krow = &k[j * N_STATE + base..j * N_STATE + base + D_HEAD];
            let mut acc = 0.0f32;
            for (d, &qd) in qh.iter().enumerate() {
                acc += qd * (f16_to_f32(krow[d]) * scale);
            }
            *sj = acc;
        }
    }

    /// f16-KV + F16C batch dequant: `_mm256_cvtph_ps` converts 8 f16→f32 at once into scratch,
    /// then the SAME d-ascending scalar dot. Bit-identical to `scores_f16_naive`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn scores_f16_f16c(
        k: &[u16],
        qh: &[f32],
        scale: f32,
        tk: usize,
        base: usize,
        out: &mut [f32],
    ) {
        // SAFETY: callers gate AVX2/F16C and construct `k` with at least
        // `tk * N_STATE` elements; `base + D_HEAD <= N_STATE` in this probe.
        unsafe {
            let mut buf = [0.0f32; D_HEAD];
            for (j, sj) in out.iter_mut().enumerate().take(tk) {
                let krow = k.as_ptr().add(j * N_STATE + base);
                // D_HEAD=64 = 8 chunks of 8; convert f16→f32 with F16C.
                let mut d = 0;
                while d < D_HEAD {
                    let h8 = _mm_loadu_si128(krow.add(d).cast());
                    _mm256_storeu_ps(buf.as_mut_ptr().add(d), _mm256_cvtph_ps(h8));
                    d += 8;
                }
                // byte-exact scalar dot in the SAME order as the f32/naive paths
                let mut acc = 0.0f32;
                for (dd, &qd) in qh.iter().enumerate() {
                    acc += qd * (buf[dd] * scale);
                }
                *sj = acc;
            }
        }
    }

    /// f32-KV NON-byte-exact SIMD dot: 4 independent AVX2 FMA accumulators (breaks the
    /// scalar dependent-add chain) + horizontal reduce. Reorders the 64-elem sum ⇒ ~1e-6
    /// delta vs scalar (needs a transcript A/B). Sizes the latency-bound lever's ceiling.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn scores_f32_simd(
        k: &[f32],
        qh: &[f32],
        scale: f32,
        tk: usize,
        base: usize,
        out: &mut [f32],
    ) {
        // SAFETY: callers gate AVX2/FMA and construct `k` with at least
        // `tk * N_STATE` elements; `base + D_HEAD <= N_STATE` in this probe.
        unsafe {
            // pre-scale q once (scalar path folds scale into each term; here fold into qh*scale·k...
            // we replicate qd*(krow*scale) = (qd*scale)*krow to keep one mul in the FMA)
            let qs: Vec<f32> = qh.iter().map(|&q| q * scale).collect();
            for (j, sj) in out.iter_mut().enumerate().take(tk) {
                let krow = k.as_ptr().add(j * N_STATE + base);
                let (mut a0, mut a1, mut a2, mut a3) = (
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                );
                let mut d = 0;
                while d < D_HEAD {
                    a0 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(qs.as_ptr().add(d)),
                        _mm256_loadu_ps(krow.add(d)),
                        a0,
                    );
                    a1 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(qs.as_ptr().add(d + 8)),
                        _mm256_loadu_ps(krow.add(d + 8)),
                        a1,
                    );
                    a2 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(qs.as_ptr().add(d + 16)),
                        _mm256_loadu_ps(krow.add(d + 16)),
                        a2,
                    );
                    a3 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(qs.as_ptr().add(d + 24)),
                        _mm256_loadu_ps(krow.add(d + 24)),
                        a3,
                    );
                    d += 32;
                }
                let s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
                let mut t = [0f32; 8];
                _mm256_storeu_ps(t.as_mut_ptr(), s);
                *sj = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
            }
        }
    }

    pub fn run() {
        let iters: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(4000);
        let mut st = 0x243F_6A88_85A3_08D3u64;
        let mut nf = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            ((st >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        #[cfg(target_arch = "x86_64")]
        for &tk in &[64usize, 128, 256] {
            let kf: Vec<f32> = (0..tk * N_STATE).map(|_| nf()).collect();
            // f16 storage: round each f32 to f16 bits (via F16C for correctness).
            let kh: Vec<u16> = kf
                .iter()
                .map(|&x| unsafe {
                    let v = _mm256_set1_ps(x);
                    let p = _mm256_cvtps_ph::<0>(v); // RNE
                    _mm_extract_epi16::<0>(p) as u16
                })
                .collect();
            let qh: Vec<f32> = (0..D_HEAD).map(|_| nf()).collect();
            let scale = (D_HEAD as f32).powf(-0.25);
            let (mut o0, mut o1, mut o2) = (vec![0f32; tk], vec![0f32; tk], vec![0f32; tk]);

            // byte-identity: f16-naive vs f16-F16C must match exactly (both read the SAME f16 cache)
            scores_f16_naive(&kh, &qh, scale, tk, 0, &mut o1);
            unsafe {
                scores_f16_f16c(&kh, &qh, scale, tk, 0, &mut o2);
            }
            let diff = o1
                .iter()
                .zip(&o2)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();

            let bench = |f: &mut dyn FnMut()| {
                for _ in 0..500 {
                    f();
                }
                let mut best = f64::INFINITY;
                for _ in 0..7 {
                    let t = Instant::now();
                    for _ in 0..iters {
                        f();
                    }
                    best = best.min(t.elapsed().as_secs_f64());
                }
                best / iters as f64 * 1e6
            };
            // sum over all heads (realistic per-token self-attn cost)
            let tf = bench(&mut || {
                for h in 0..N_HEAD {
                    scores_f32(&kf, &qh, scale, tk, h * D_HEAD, &mut o0);
                    black_box(&o0[0]);
                }
            });
            let tn = bench(&mut || {
                for h in 0..N_HEAD {
                    scores_f16_naive(&kh, &qh, scale, tk, h * D_HEAD, &mut o1);
                    black_box(&o1[0]);
                }
            });
            let tc = bench(&mut || {
                for h in 0..N_HEAD {
                    unsafe {
                        scores_f16_f16c(&kh, &qh, scale, tk, h * D_HEAD, &mut o2);
                    }
                    black_box(&o2[0]);
                }
            });
            let mut o3 = vec![0f32; tk];
            unsafe {
                scores_f32_simd(&kf, &qh, scale, tk, 0, &mut o3);
            }
            scores_f32(&kf, &qh, scale, tk, 0, &mut o0);
            let mut maxrel = 0f32;
            for (a, b) in o0.iter().zip(&o3) {
                let d = (a - b).abs() / a.abs().max(1e-6);
                maxrel = maxrel.max(d);
            }
            let ts = bench(&mut || {
                for h in 0..N_HEAD {
                    unsafe {
                        scores_f32_simd(&kf, &qh, scale, tk, h * D_HEAD, &mut o3);
                    }
                    black_box(&o3[0]);
                }
            });

            println!(
                "=== self-attn scores, ALL {N_HEAD} heads, tk={tk} (cache {} KB f32 / {} KB f16) ===",
                tk * N_STATE * 4 / 1024,
                tk * N_STATE * 2 / 1024
            );
            println!(
                "  f16-naive vs f16-F16C byte-diff: {diff}/{tk}  [{}]",
                if diff == 0 { "IDENTICAL" } else { "DIVERGENT" }
            );
            println!("  f32-KV scalar dot        : {tf:>7.2} µs/token   1.00×");
            println!(
                "  f16-KV naive to_f32/elem : {tn:>7.2} µs/token   {:.2}×  (memory's dead path)",
                tf / tn
            );
            println!(
                "  f16-KV F16C batch dequant: {tc:>7.2} µs/token   {:.2}×  [{}]",
                tf / tc,
                if tc < tf * 0.9 {
                    "WIN vs f32 — the missing ingredient works"
                } else {
                    "no win vs f32"
                }
            );
            println!(
                "  f32-KV SIMD 4-accum dot  : {ts:>7.2} µs/token   {:.2}×  [{}] max-rel-Δ={maxrel:.1e} (non-byte-exact)",
                tf / ts,
                if ts < tf * 0.7 {
                    "WIN — latency-bound chain broken; needs transcript A/B"
                } else {
                    "marginal"
                }
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    if !std::is_x86_feature_detected!("avx2")
        || !std::is_x86_feature_detected!("f16c")
        || !std::is_x86_feature_detected!("fma")
    {
        eprintln!("self_attn_f16c_dequant_probe requires AVX2, F16C, and FMA support");
        return;
    }
    x86_probe::run();
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("self_attn_f16c_dequant_probe requires an x86_64 processor");
}
