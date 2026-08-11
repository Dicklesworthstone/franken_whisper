//! Self-attention SCORE-DOT lever (BlackThrush, 2026-07-04).
//!
//! Measurement probe only — never linked into the library. Its AVX2/FMA kernel
//! is isolated behind one runtime-checked capability token; the rest of the
//! probe remains under the workspace `unsafe_code = "deny"` lint.
//!
//! `attention_decode_step` (nn.rs:3205-3212) computes the per-token self-attn
//! scores with a SCALAR sequential f32 reduction:
//!   for j in 0..tk { let mut acc=0; for d in 0..d_head { acc += qh[d]*(k[j,d]*scale) } scores[j]=acc }
//! The `acc +=` chain is loop-carried, so LLVM keeps it a serial `vaddss` chain
//! (latency-bound: ~d_head × add-latency per dot) and does NOT multi-accumulate
//! (that would reorder the float sum). The output SAXPY below it is already AVX2;
//! this dot is the last scalar piece of self-attn (~65% "softmax/dots" per
//! project_self_attn_kv_cache_lever). Same latency-bound-reduction class as
//! dot_i8 / dot_f16c where hand-AVX2 beat autovec — memory tagged this
//! "owner-gated / non-byte-exact" by REASONING; this probe MEASURES it.
//!
//! Two AVX2 variants, both over the CONTIGUOUS d_head run (no gather/transpose):
//!   A) multi-accumulator (4×f32x8) horizontal-reduce dot — FAST, NON-byte-exact
//!      (reordered sum). Quantifies the owner-gated speed ceiling.
//!   B) ordered single-f32x8-accumulator (still one horizontal reduce at the end)
//!      — measures whether even a mild reorder helps and by how much its |Δ|.
//! Reports per-token score-dot µs (20 heads × tk keys), ratio, and max|Δ| vs scalar.
//! Usage: `self_attn_score_dot_probe [iters]`  (turbo shapes: n_head=20, d_head=64).
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
const N_STATE: usize = 1280;
#[cfg(target_arch = "x86_64")]
const N_HEAD: usize = 20;
#[cfg(target_arch = "x86_64")]
const D_HEAD: usize = N_STATE / N_HEAD; // 64

/// Current engine code: scalar sequential dot (byte-exact reference).
#[cfg(target_arch = "x86_64")]
fn scalar_scores(qh: &[f32], k: &[f32], scale: f32, base: usize, tk: usize, out: &mut [f32]) {
    debug_assert_eq!(qh.len(), D_HEAD);
    debug_assert!(k.len() >= tk * N_STATE);
    debug_assert_eq!(out.len(), tk);
    for (j, sj) in out.iter_mut().enumerate() {
        let krow = &k[j * N_STATE + base..j * N_STATE + base + D_HEAD];
        let mut acc = 0.0f32;
        for (d, &qd) in qh.iter().enumerate() {
            acc += qd * (krow[d] * scale);
        }
        *sj = acc;
    }
}

/// A) 4-accumulator AVX2 dot over contiguous d_head (D_HEAD=64 = 8×f32x8).
/// NON-byte-exact: 4 partial sums reduced at the end (reordered).
///
/// # Safety
///
/// The caller must verify AVX2 and FMA support and provide `qh`, `k`, and
/// `out` slices large enough for `D_HEAD`, `tk * N_STATE`, and `tk`; `base`
/// through `base + D_HEAD` must stay inside each `N_STATE` row.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn avx2_scores_4acc(
    qh: &[f32],
    k: &[f32],
    scale: f32,
    base: usize,
    tk: usize,
    out: &mut [f32],
) {
    // SAFETY: the function's contract establishes the ISA features and every
    // pointer range used below. Unaligned loads are intentional; all offsets
    // stay inside the validated slice lengths.
    unsafe {
        let vscale = _mm256_set1_ps(scale);
        let qp = qh.as_ptr();
        for (j, sj) in out.iter_mut().take(tk).enumerate() {
            let kp = k.as_ptr().add(j * N_STATE + base);
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();
            // D_HEAD=64 => 8 vectors; unroll 4 accumulators × 2 rounds.
            let mut d = 0;
            while d + 32 <= D_HEAD {
                let q0 = _mm256_loadu_ps(qp.add(d));
                let q1 = _mm256_loadu_ps(qp.add(d + 8));
                let q2 = _mm256_loadu_ps(qp.add(d + 16));
                let q3 = _mm256_loadu_ps(qp.add(d + 24));
                let k0 = _mm256_mul_ps(_mm256_loadu_ps(kp.add(d)), vscale);
                let k1 = _mm256_mul_ps(_mm256_loadu_ps(kp.add(d + 8)), vscale);
                let k2 = _mm256_mul_ps(_mm256_loadu_ps(kp.add(d + 16)), vscale);
                let k3 = _mm256_mul_ps(_mm256_loadu_ps(kp.add(d + 24)), vscale);
                a0 = _mm256_fmadd_ps(q0, k0, a0);
                a1 = _mm256_fmadd_ps(q1, k1, a1);
                a2 = _mm256_fmadd_ps(q2, k2, a2);
                a3 = _mm256_fmadd_ps(q3, k3, a3);
                d += 32;
            }
            let s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            // horizontal sum of the 8 lanes
            let hi = _mm256_extractf128_ps(s, 1);
            let lo = _mm256_castps256_ps128(s);
            let mut sum128 = _mm_add_ps(hi, lo);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            *sj = _mm_cvtss_f32(sum128);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Avx2Fma;

#[cfg(target_arch = "x86_64")]
impl Avx2Fma {
    fn detect() -> Option<Self> {
        (std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"))
            .then_some(Self)
    }

    #[inline]
    #[allow(unsafe_code)]
    fn scores(self, qh: &[f32], k: &[f32], scale: f32, base: usize, tk: usize, out: &mut [f32]) {
        debug_assert_eq!(std::mem::size_of_val(&self), 0);
        let required_k = tk
            .checked_mul(N_STATE)
            .expect("self-attention key-cache length overflow");
        assert_eq!(qh.len(), D_HEAD);
        assert!(base <= N_STATE - D_HEAD);
        assert!(k.len() >= required_k);
        assert!(out.len() >= tk);
        // SAFETY: `detect` proved the required ISA features, and the slice
        // and row bounds required by the kernel are checked above.
        unsafe { avx2_scores_4acc(qh, k, scale, base, tk, out) }
    }
}

#[cfg(target_arch = "x86_64")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("self_attn_score_dot_probe requires x86_64 with AVX2 and FMA");
    std::process::exit(2);
}

#[cfg(target_arch = "x86_64")]
fn main() {
    let avx2 =
        Avx2Fma::detect().expect("self_attn_score_dot_probe requires CPU support for AVX2 and FMA");
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let scale = (D_HEAD as f32).powf(-0.25);
    // deterministic synthetic q (one token) + K cache
    let mut st = 0x9E37_79B9_7F4A_7C15u64;
    let mut nf = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        ((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
    };
    let qfull: Vec<f32> = (0..N_STATE).map(|_| nf()).collect();

    println!(
        "self-attn score dot: scalar vs AVX2 4-accum (turbo n_head={N_HEAD}, d_head={D_HEAD})"
    );
    for &tk in &[32usize, 64, 128, 224] {
        let k: Vec<f32> = (0..tk * N_STATE).map(|_| nf()).collect();
        let mut sc_scalar = vec![0.0f32; N_HEAD * tk];
        let mut sc_avx = vec![0.0f32; N_HEAD * tk];

        // correctness + warm
        for h in 0..N_HEAD {
            let base = h * D_HEAD;
            scalar_scores(
                &qfull[base..base + D_HEAD],
                &k,
                scale,
                base,
                tk,
                &mut sc_scalar[h * tk..(h + 1) * tk],
            );
            avx2.scores(
                &qfull[base..base + D_HEAD],
                &k,
                scale,
                base,
                tk,
                &mut sc_avx[h * tk..(h + 1) * tk],
            );
        }
        let mad = max_abs_diff(&sc_scalar, &sc_avx);

        let t = Instant::now();
        for _ in 0..iters {
            for h in 0..N_HEAD {
                let base = h * D_HEAD;
                scalar_scores(
                    &qfull[base..base + D_HEAD],
                    black_box(&k),
                    scale,
                    base,
                    tk,
                    &mut sc_scalar[h * tk..(h + 1) * tk],
                );
            }
            black_box(sc_scalar[0]);
        }
        let t_scalar = t.elapsed().as_secs_f64() / iters as f64;

        let t = Instant::now();
        for _ in 0..iters {
            for h in 0..N_HEAD {
                let base = h * D_HEAD;
                avx2.scores(
                    &qfull[base..base + D_HEAD],
                    black_box(&k),
                    scale,
                    base,
                    tk,
                    &mut sc_avx[h * tk..(h + 1) * tk],
                );
            }
            black_box(sc_avx[0]);
        }
        let t_avx = t.elapsed().as_secs_f64() / iters as f64;

        println!(
            "  tk={tk:>3}: scalar {:>7.2} µs  avx4 {:>7.2} µs  = {:>4.2}×   max|Δ|={:.2e} ({})",
            t_scalar * 1e6,
            t_avx * 1e6,
            t_scalar / t_avx,
            mad,
            if mad == 0.0 {
                "BYTE-EXACT"
            } else {
                "non-byte-exact (reordered sum)"
            }
        );
    }
    println!(
        "\nNOTE: per-token score-dot time above is 1 head-group pass. In decode it is 4 layers ×\n\
         this, and self_attn is ~10.7% of decode which is ~15% of e2e (ts; ~0 no_ts pipeline-hidden).\n\
         If non-byte-exact, this is an owner-gated FW flag; if the win is large it may be worth it."
    );
}
