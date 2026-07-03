//! Sampler max-reduction + argmax over the vocab: scalar vs byte-exact AVX2
//! (BlackThrush, 2026-07-03).
//!
//! `compute_logprobs` (decode.rs:507) takes `logit_max = logits.fold(-inf, f32::max)` and
//! `argmax` (decode.rs:538) scans for the first index achieving the max — both SERIAL passes
//! over n_vocab=51866 per token, on the sampler critical path (NOT parallelized, unlike
//! attention, so full wall-clock weight). `argmax` is definitely scalar (index-tracking data
//! dependency defeats autovec); the max fold may or may not autovec. Post-sanitize the logits
//! contain NO NaN (only finite or -inf), so `_mm256_max_ps` == the `f32::max` fold BYTE-EXACT,
//! and a SIMD argmax that keeps the FIRST index of the max is byte-exact too. This measures
//! whether they are scalar levers (byte-exact, default-on-able) or already vectorized.
//! Usage: `sampler_maxargmax_probe [iters]` (default 3000).
#![allow(unsafe_code)]
use std::hint::black_box;
use std::time::Instant;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// franken's exact scalar max fold (decode.rs:507).
fn max_scalar(l: &[f32]) -> f32 {
    l.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// franken's exact scalar argmax: first index with strict `>` (decode.rs:538-546).
fn argmax_scalar(l: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > best {
            best = v;
            best_i = i;
        }
    }
    best_i
}

/// AVX2 horizontal max. Byte-exact vs `max_scalar` when no lane is NaN (post-sanitize holds).
#[cfg(target_arch = "x86_64")]
fn max_simd(l: &[f32]) -> f32 {
    let n = l.len();
    let n8 = n & !7;
    unsafe {
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut i = 0;
        while i < n8 {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(l.as_ptr().add(i)));
            i += 8;
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), vmax);
        let mut m = tmp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        while i < n {
            m = m.max(l[i]);
            i += 1;
        }
        m
    }
}

/// AVX2 argmax returning the FIRST index of the max (byte-exact vs `argmax_scalar`).
/// Per lane j, strict-`>` update keeps the first index in lane j's stream (j, j+8, ...);
/// the horizontal reduce then takes the max value and the MIN index among ties ⇒ global
/// first index. Tail is scalar with strict `>` against the reduced best.
#[cfg(target_arch = "x86_64")]
fn argmax_simd(l: &[f32]) -> usize {
    let n = l.len();
    let n8 = n & !7;
    unsafe {
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let mut vidx = _mm256_setzero_si256();
        let mut i = 0;
        while i < n8 {
            let v = _mm256_loadu_ps(l.as_ptr().add(i));
            let gt = _mm256_cmp_ps::<_CMP_GT_OQ>(v, vmax); // strict > ⇒ keep first
            vmax = _mm256_blendv_ps(vmax, v, gt);
            let idx = _mm256_add_epi32(_mm256_set1_epi32(i as i32), lane);
            vidx = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(vidx),
                _mm256_castsi256_ps(idx),
                gt,
            ));
            i += 8;
        }
        let mut vals = [0.0f32; 8];
        let mut idxs = [0i32; 8];
        _mm256_storeu_ps(vals.as_mut_ptr(), vmax);
        _mm256_storeu_si256(idxs.as_mut_ptr().cast(), vidx);
        let mut best = f32::NEG_INFINITY;
        let mut best_i = usize::MAX;
        for k in 0..8 {
            let (v, ix) = (vals[k], idxs[k] as usize);
            if v > best || (v == best && ix < best_i) {
                best = v;
                best_i = ix;
            }
        }
        while i < n {
            if l[i] > best {
                best = l[i];
                best_i = i;
            }
            i += 1;
        }
        best_i
    }
}

fn timeit<T>(f: impl Fn(&[f32]) -> T, l: &[f32], iters: usize) -> f64 {
    for _ in 0..5 { black_box(f(l)); }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        black_box(f(l));
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let n = 51866usize; // whisper large-v3 n_vocab
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 30.0 };
    let mut logits: Vec<f32> = (0..n).map(|_| nf()).collect();
    // sprinkle a few -inf (suppressed tokens) — must NOT break byte-exactness (no NaN post-sanitize)
    for &j in &[7usize, 100, 5000, 51865] { logits[j] = f32::NEG_INFINITY; }

    println!("=== sampler over n_vocab={n}: scalar vs byte-exact AVX2 (serial critical path) ===");
    #[cfg(target_arch = "x86_64")]
    {
        let (ms, mv) = (max_scalar(&logits), max_simd(&logits));
        let (as_, av) = (argmax_scalar(&logits), argmax_simd(&logits));
        println!("  byte-exact: max scalar={ms:.6} simd={mv:.6} [{}] | argmax scalar={as_} simd={av} [{}]",
            if ms.to_bits() == mv.to_bits() { "IDENTICAL" } else { "DIVERGENT" },
            if as_ == av { "IDENTICAL" } else { "DIVERGENT" });
        let tms = timeit(max_scalar, &logits, iters);
        let tmv = timeit(max_simd, &logits, iters);
        let tas = timeit(argmax_scalar, &logits, iters);
        let tav = timeit(argmax_simd, &logits, iters);
        println!("  max fold : scalar {:>6.2} µs | AVX2 {:>6.2} µs | {:.2}x [{}]",
            tms * 1e6, tmv * 1e6, tms / tmv, if tmv < tms { "WIN" } else { "already-vectorized/loss" });
        println!("  argmax   : scalar {:>6.2} µs | AVX2 {:>6.2} µs | {:.2}x [{}]",
            tas * 1e6, tav * 1e6, tas / tav, if tav < tas { "WIN" } else { "loss" });
        println!("  (per-token, SERIAL sampler critical path — full wall-clock weight, not parallelized.)");
    }
}
