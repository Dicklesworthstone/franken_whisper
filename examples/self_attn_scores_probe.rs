//! Self-attn f32 scores dot: scalar contiguous dot vs byte-exact loop-swapped AXPY
//! (BlackThrush, 2026-07-03).
//!
//! `attention_decode_step` (nn.rs, the DEFAULT f32 self-attn path — f16 KV was rejected)
//! computes `scores[j] = Σ_d qh[d]·(k[j][d]·scale)` with a SCALAR dot over d_head, per key j,
//! per head. It is NOT vectorized (float reduction ⇒ LLVM won't autovec; the register-blocked
//! GEMV landed only on the f16 path). A BYTE-EXACT vectorization exists: swap to d-outer /
//! j-inner, so each `scores[j] += qh[d]·(k[j][d]·scale)` still accumulates d-ascending
//! (bit-identical sum order) but the inner j-loop is an AXPY (vectorizes across the
//! INDEPENDENT output slots j). The tradeoff: the swap reads k strided (n_state stride) vs the
//! scalar's contiguous d_head. This measures whether the vectorization beats the cache penalty
//! at representative decode KV lengths. Byte-exactness is asserted (must be 0 differing bits).
//! Turbo self-attn: n_state=1280, n_head=20, d_head=64. Usage: `self_attn_scores_probe [iters]`.
#![allow(unsafe_code)]
use std::hint::black_box;
use std::time::Instant;

const N_STATE: usize = 1280;
const N_HEAD: usize = 20;
const D_HEAD: usize = N_STATE / N_HEAD;

/// franken's exact scalar scores dot (contiguous over d, d-ascending sum).
fn scores_scalar(k: &[f32], qh_all: &[f32], scale: f32, tk: usize, out: &mut [f32]) {
    for h in 0..N_HEAD {
        let base = h * D_HEAD;
        let qh = &qh_all[base..base + D_HEAD];
        let so = &mut out[h * tk..(h + 1) * tk];
        for (j, sj) in so.iter_mut().enumerate() {
            let krow = &k[j * N_STATE + base..j * N_STATE + base + D_HEAD];
            let mut acc = 0.0f32;
            for (d, &qd) in qh.iter().enumerate() {
                acc += qd * (krow[d] * scale);
            }
            *sj = acc;
        }
    }
}

/// Byte-exact loop-swap: d outer, j inner AXPY. Each scores[j] accumulates d-ascending
/// exactly as the scalar path ⇒ identical bits; inner j-loop autovectorizes (strided k read).
fn scores_swap(k: &[f32], qh_all: &[f32], scale: f32, tk: usize, out: &mut [f32]) {
    for h in 0..N_HEAD {
        let base = h * D_HEAD;
        let qh = &qh_all[base..base + D_HEAD];
        let so = &mut out[h * tk..(h + 1) * tk];
        for s in so.iter_mut() {
            *s = 0.0;
        }
        for (d, &qd) in qh.iter().enumerate() {
            for (j, sj) in so.iter_mut().enumerate() {
                *sj += qd * (k[j * N_STATE + base + d] * scale);
            }
        }
    }
}

fn run(f: &dyn Fn(&[f32], &[f32], f32, usize, &mut [f32]), k: &[f32], qh: &[f32],
       scale: f32, tk: usize, out: &mut [f32], iters: usize) -> f64 {
    for _ in 0..5 { f(k, qh, scale, tk, out); }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        f(k, qh, scale, tk, out);
        best = best.min(t.elapsed().as_secs_f64());
        black_box(&out[0]);
    }
    best
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let scale = (D_HEAD as f32).powf(-0.25);
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) };
    let qh: Vec<f32> = (0..N_STATE).map(|_| nf()).collect();

    println!("=== self-attn f32 scores dot: scalar contiguous vs byte-exact loop-swap AXPY ===");
    println!("    (turbo n_state={N_STATE}, n_head={N_HEAD}, d_head={D_HEAD}; per-token = ALL heads)");
    for &tk in &[64usize, 128, 224] {
        let k: Vec<f32> = (0..tk * N_STATE).map(|_| nf()).collect();
        let mut a = vec![0.0f32; N_HEAD * tk];
        let mut b = vec![0.0f32; N_HEAD * tk];
        scores_scalar(&k, &qh, scale, tk, &mut a);
        scores_swap(&k, &qh, scale, tk, &mut b);
        let ndiff = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let ts = run(&scores_scalar, &k, &qh, scale, tk, &mut a, iters);
        let tw = run(&scores_swap, &k, &qh, scale, tk, &mut b, iters);
        println!("  tk={tk:>3}: scalar {:>6.2} µs | swap {:>6.2} µs | {:.2}x [{}] | byte-diff={ndiff}",
            ts * 1e6, tw * 1e6, ts / tw, if tw < ts { "WIN" } else { "loss" });
    }
    println!("  (per-token self-attn scores for ALL 20 heads at the given KV length. byte-diff MUST be 0.)");
}
