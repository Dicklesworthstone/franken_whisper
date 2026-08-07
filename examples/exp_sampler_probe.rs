//! Sampler logsumexp: scalar libm `expf` vs AVX2-poly exp (BlackThrush, 2026-07-02).
//!
//! The decode sampler's dominant remaining cost is the vocab-wide `exp` in the
//! log-softmax (`compute_logprobs`: `logsumexp = Σ exp(l - max)` over n_vocab=51866,
//! per token) — the ledger's standing owner-gated lever #2, estimated "~3.5% e2e"
//! but NEVER isolated-measured. frankentorch HAS a SIMD-exp helper the owner
//! deliberately left unwired. This QUANTIFIES the lever with hard data: how much
//! faster is a SIMD-poly exp, and what is its numerical delta vs libm (the
//! transcript-risk gauge — the exp feeds logprobs → timestamp-rule / no_speech,
//! so a poly is NON-byte-exact and would land gated + transcript-measured, per
//! feedback_execute_dont_ask).
//!
//! Small isolated per-crate microbench: one 51866-logit logsumexp pass, single
//! thread, best-of-N (load-insensitive). Realistic input: post-subtract-max
//! logits (all ≤ 0, most very negative), ~10% masked to -inf (exp→0), like the
//! real sampler after masking. Usage: `exp_sampler_probe [iters]` (default 200).
#![allow(unsafe_code)]
#[cfg(target_arch = "x86_64")]
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

/// Scalar libm logsumexp — the CURRENT franken path (`compute_logprobs`).
#[cfg(target_arch = "x86_64")]
fn logsumexp_scalar(x: &[f32], max: f32) -> f32 {
    let mut s = 0.0f32;
    for &l in x {
        if l > f32::NEG_INFINITY {
            s += (l - max).exp();
        }
    }
    s
}

/// AVX2 exp poly (range-reduce x=k·ln2+r, exp(r)≈degree-5 minimax, scale by 2^k via
/// float-bit construction). ~1e-6 rel error — a fair stand-in for a wired SIMD exp.
/// -inf lanes (x-max = -inf) underflow to 0 via the x<=-87 clamp, matching the
/// scalar `l > -inf` guard's contribution of 0.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn logsumexp_avx2(x: &[f32], max: f32) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let n = x.len();
        let xp = x.as_ptr();
        let vmax = _mm256_set1_ps(max);
        let log2e = _mm256_set1_ps(1.442_695_f32);
        let ln2 = _mm256_set1_ps(0.693_147_2_f32);
        let lo = _mm256_set1_ps(-87.3365_f32); // exp underflows to 0 below this
        // degree-5 exp(r) minimax on r∈[-ln2/2, ln2/2]
        let c0 = _mm256_set1_ps(1.0);
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(0.166_666_67_f32);
        let c4 = _mm256_set1_ps(0.041_666_66_f32);
        let c5 = _mm256_set1_ps(0.008_333_33_f32);
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            let l = _mm256_loadu_ps(xp.add(i));
            let mut xv = _mm256_sub_ps(l, vmax); // ≤ 0; -inf stays -inf
            xv = _mm256_max_ps(xv, lo); // clamp -inf / very-neg → -87.3 (exp→~0)
            // k = round(x * log2e); r = x - k*ln2
            let kf = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
                _mm256_mul_ps(xv, log2e),
            );
            let r = _mm256_fnmadd_ps(kf, ln2, xv); // x - k*ln2
            // exp(r) ≈ ((((c5 r + c4) r + c3) r + c2) r + c1) r + c0
            let mut p = _mm256_fmadd_ps(c5, r, c4);
            p = _mm256_fmadd_ps(p, r, c3);
            p = _mm256_fmadd_ps(p, r, c2);
            p = _mm256_fmadd_ps(p, r, c1);
            p = _mm256_fmadd_ps(p, r, c0);
            // 2^k : (k + 127) << 23 as float bits
            let ki = _mm256_cvtps_epi32(kf);
            let bias = _mm256_set1_epi32(127);
            let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(ki, bias)));
            acc = _mm256_fmadd_ps(p, pow2, acc);
            i += 8;
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut s =
            ((tmp[0] + tmp[1]) + (tmp[2] + tmp[3])) + ((tmp[4] + tmp[5]) + (tmp[6] + tmp[7]));
        while i < n {
            let l = *x.get_unchecked(i);
            if l > f32::NEG_INFINITY {
                s += (l - max).exp();
            }
            i += 1;
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
        eprintln!("exp_sampler_probe requires AVX2 and FMA support");
        return;
    }
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let n = 51866usize;
    // Realistic post-softmax-input logits: a few high, most low, ~10% masked -inf.
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut nf = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 // U(0,1)
    };
    let mut x: Vec<f32> = (0..n)
        .map(|_| {
            let u = nf();
            if u < 0.10 {
                f32::NEG_INFINITY // masked lane
            } else {
                -30.0 * nf() // logits after some shift, mostly negative
            }
        })
        .collect();
    // A realistic max (the true row max, ~0 after subtract).
    x[123] = 0.0;
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Numerical delta.
    let ref_s = logsumexp_scalar(&x, max);
    let avx_s = unsafe { logsumexp_avx2(&x, max) };
    let rel = ((ref_s - avx_s).abs() / ref_s.abs().max(1e-30)) as f64;
    // logsumexp = ln(sum) + max; the logprob delta is ln(sum) delta.
    let ln_delta = (ref_s.ln() - avx_s.ln()).abs();

    let bench = |f: &dyn Fn() -> f32| -> f64 {
        for _ in 0..3 {
            black_box(f());
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            black_box(f());
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };
    let ts = bench(&|| logsumexp_scalar(&x, max));
    let ta = bench(&|| unsafe { logsumexp_avx2(&x, max) });
    println!(
        "=== sampler logsumexp: scalar libm expf vs AVX2-poly exp (n_vocab={n}, 1 thread) ==="
    );
    println!("best-of-{iters}");
    println!("  scalar libm : {:>8.3} µs", ts * 1e6);
    println!("  AVX2 poly   : {:>8.3} µs  {:.2}x", ta * 1e6, ts / ta);
    println!("  numerical delta: sum rel-err {rel:.2e}  |  logprob (ln-sum) delta {ln_delta:.2e}");
    println!("  scalar sum={ref_s:.6}  avx sum={avx_s:.6}");
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("exp_sampler_probe requires an x86_64 processor");
}
