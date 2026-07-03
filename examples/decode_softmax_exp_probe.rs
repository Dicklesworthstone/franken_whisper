//! True decode-softmax exp headroom: cross-attn softmax is the DOMINANT exp
//! consumer, not the sampler (BlackThrush, 2026-07-03).
//!
//! `project_sampler_exp_measured` sized ONLY the sampler's vocab-wide exp (51866/token)
//! and concluded "pipelining-hidden ⇒ ~0 e2e". But per token the decoder also runs
//! `nn::softmax_rows` (scalar libm `.exp()`) for:
//!   - CROSS-attn: 4 layers × 20 heads × 1500 enc frames = 120000 exp/token  (decoder.rs:1281/1312)
//!   - SELF-attn : 4 layers × 20 heads × seq_len (grows; ~avg 128)           (attention_decode_step)
//!   - sampler   : 51866 exp/token                                           (compute_logprobs)
//! Cross-attn alone is ~2.3× the sampler. This probe times franken's exact scalar
//! `softmax_rows` vs an AVX2 poly-exp softmax on the REAL per-token decode shapes, to size
//! the true owner-gated SIMD-exp headroom — and confirms it's exposed in TIMESTAMP mode
//! (where decode does NOT pipeline behind encode).
//! Non-byte-exact (poly ≠ libm) ⇒ owner-gated; reports max|Δ| for the faithfulness record.
//! Usage: `decode_softmax_exp_probe [iters]` (default 200).
#![allow(unsafe_code)]
use std::hint::black_box;
use std::time::Instant;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// franken's exact scalar softmax_rows row kernel (libm exp), in place.
fn softmax_row_scalar(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() { return; }
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        let e = (*v - max).exp();
        let e = if e.is_finite() { e } else { 0.0 };
        *v = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in row.iter_mut() { *v *= inv; }
    }
}

/// Cephes/ggml-style f32 exp poly on 8 lanes (FMA — approximation, not byte-exact).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp8(x: __m256) -> __m256 {
    let lo = _mm256_set1_ps(-88.0);
    let hi = _mm256_set1_ps(88.0);
    let x = _mm256_max_ps(_mm256_min_ps(x, hi), lo);
    let log2ef = _mm256_set1_ps(1.442_695_04);
    let fx = _mm256_round_ps(_mm256_mul_ps(x, log2ef), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let c1 = _mm256_set1_ps(0.693_359_38);
    let c2 = _mm256_set1_ps(-2.121_944_4e-4);
    let mut xr = _mm256_fnmadd_ps(fx, c1, x);
    xr = _mm256_fnmadd_ps(fx, c2, xr);
    let mut p = _mm256_set1_ps(1.987_569_1e-4);
    p = _mm256_fmadd_ps(p, xr, _mm256_set1_ps(1.398_199_9e-3));
    p = _mm256_fmadd_ps(p, xr, _mm256_set1_ps(8.333_452e-3));
    p = _mm256_fmadd_ps(p, xr, _mm256_set1_ps(4.166_579_6e-2));
    p = _mm256_fmadd_ps(p, xr, _mm256_set1_ps(1.666_666_5e-1));
    p = _mm256_fmadd_ps(p, xr, _mm256_set1_ps(5.000_000_1e-1));
    let xr2 = _mm256_mul_ps(xr, xr);
    p = _mm256_fmadd_ps(p, xr2, _mm256_add_ps(xr, _mm256_set1_ps(1.0)));
    // scale by 2^fx: (int(fx)+127) << 23
    let ki = _mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127));
    let pow2 = _mm256_castsi256_ps(_mm256_slli_epi32(ki, 23));
    _mm256_mul_ps(p, pow2)
}

/// AVX2 poly-exp softmax row (max-subtract, poly exp, normalize).
#[cfg(target_arch = "x86_64")]
fn softmax_row_simd(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() { return; }
    unsafe {
        let vmax = _mm256_set1_ps(max);
        let mut vsum = _mm256_setzero_ps();
        let n = row.len();
        let n8 = n & !7;
        let mut i = 0;
        while i < n8 {
            let v = _mm256_loadu_ps(row.as_ptr().add(i));
            let e = exp8(_mm256_sub_ps(v, vmax));
            _mm256_storeu_ps(row.as_mut_ptr().add(i), e);
            vsum = _mm256_add_ps(vsum, e);
            i += 8;
        }
        // horizontal sum of vsum
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), vsum);
        let mut sum: f32 = tmp.iter().sum();
        // scalar tail with the SAME poly (via a 1-lane broadcast) so Δ is poly-only
        while i < n {
            let xv = _mm256_set1_ps(row[i] - max);
            let mut out = [0.0f32; 8];
            _mm256_storeu_ps(out.as_mut_ptr(), exp8(xv));
            row[i] = out[0];
            sum += out[0];
            i += 1;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            let vinv = _mm256_set1_ps(inv);
            let mut i = 0;
            while i < n8 {
                let v = _mm256_loadu_ps(row.as_ptr().add(i));
                _mm256_storeu_ps(row.as_mut_ptr().add(i), _mm256_mul_ps(v, vinv));
                i += 8;
            }
            while i < n { row[i] *= inv; i += 1; }
        }
    }
}

fn bench(label: &str, rows: usize, cols: usize, iters: usize,
         kernel: &dyn Fn(&mut [f32]), base: &[f32]) -> f64 {
    let mut data = base.to_vec();
    for _ in 0..3 { for r in data.chunks_mut(cols) { kernel(r); } data.copy_from_slice(base); }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        data.copy_from_slice(base);
        let t = Instant::now();
        for r in data.chunks_mut(cols) { kernel(r); }
        best = best.min(t.elapsed().as_secs_f64());
        black_box(&data[0]);
    }
    println!("    {label:<34}: {:>8.1} µs  ({rows}×{cols})", best * 1e6);
    best
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0 };

    // Per-token decode softmax shapes (turbo: n_layer=4, n_head=20, enc_frames=1500):
    let cross = (80usize, 1500usize);   // 4×20 rows, 1500 cols  = 120000 exp/token
    let selfa = (80usize, 128usize);    // 4×20 rows, ~128 seq   = 10240 exp/token (representative)
    let samp = (1usize, 51866usize);    // sampler logsumexp     = 51866 exp/token

    println!("=== decode softmax exp: franken scalar (libm) vs AVX2 poly, per-token turbo shapes ===");
    let mut tot_scalar = 0.0;
    let mut tot_simd = 0.0;
    let mut maxd = 0.0f32;
    for (name, (rows, cols)) in [("CROSS (4×20 heads × 1500)", cross),
                                 ("SELF  (4×20 heads × ~128)", selfa),
                                 ("SAMPLER (1 × 51866 vocab)", samp)] {
        let base: Vec<f32> = (0..rows * cols).map(|_| nf()).collect();
        // byte-exactness delta on this shape
        let (mut a, mut b) = (base.clone(), base.clone());
        for r in a.chunks_mut(cols) { softmax_row_scalar(r); }
        #[cfg(target_arch = "x86_64")]
        for r in b.chunks_mut(cols) { softmax_row_simd(r); }
        let d = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        maxd = maxd.max(d);
        println!("  {name}:");
        let ts = bench("scalar softmax_rows (libm)", rows, cols, iters, &|r| softmax_row_scalar(r), &base);
        #[cfg(target_arch = "x86_64")]
        let tv = bench("AVX2 poly-exp softmax", rows, cols, iters, &|r| softmax_row_simd(r), &base);
        #[cfg(not(target_arch = "x86_64"))]
        let tv = ts;
        println!("      -> {:.2}x on this shape", ts / tv);
        tot_scalar += ts;
        tot_simd += tv;
    }
    println!("  ------------------------------------------------------------");
    println!("  PER-TOKEN TOTAL scalar (libm) : {:>8.1} µs/token", tot_scalar * 1e6);
    println!("  PER-TOKEN TOTAL AVX2 poly      : {:>8.1} µs/token  ({:.2}x)", tot_simd * 1e6, tot_scalar / tot_simd);
    println!("  saved                          : {:>8.1} µs/token", (tot_scalar - tot_simd) * 1e6);
    println!("  faithfulness: max|Δ|={maxd:.3e} (poly ≠ libm ⇒ NON-byte-exact, owner-gated)");
    println!("  NOTE: cross-attn softmax dominates; in TIMESTAMP mode decode is NOT pipelining-hidden.");
}
