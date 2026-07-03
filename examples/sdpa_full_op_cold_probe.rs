//! Full encoder SDPA op (gather + kernel + scatter) at cold: does the FW_SDPA_GATHER_CHUNKS
//! reshape win survive once the external kernel time is included? (BlackThrush, 2026-07-03).
//!
//! The isolated reshape probes showed 16 balanced chunks beats the legacy per-head gather
//! (1.73×) and per-row scatter (1.6×) on cold data. But the reshape is only ~part of the
//! attn_sdpa op — the `ft_kernel_cpu::sdpa_forward_f32` kernel dominates. This measures the
//! WHOLE op (gather + kernel + scatter) for one turbo encoder layer, cold, comparing the
//! legacy chunking (per-head gather / per-row scatter) vs 16 balanced chunks, to get the
//! realistic LAYER-level speedup — the number that actually maps to e2e. Byte-identical
//! output (the chunking is pure data movement; the kernel input is the same qa/ka/va).
//! Usage: `sdpa_full_op_cold_probe [iters]` (default 120).
#![allow(unsafe_code)]
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Gather interleaved `src [t, n_state]` -> head-major `dst [hh, t, d_head]`, `chunks`
/// balanced row-bands (chunks==0 ⇒ one band per head = legacy).
fn gather(dst: &mut [f32], src: &[f32], hh: usize, t: usize, d_head: usize, n_state: usize, chunks: usize) {
    let total = hh * t;
    let n = if chunks == 0 { hh } else { chunks.min(total).max(1) };
    let cr = total.div_ceil(n).max(1);
    dst.par_chunks_mut(cr * d_head).enumerate().for_each(|(c, blk)| {
        for (l, row) in blk.chunks_mut(d_head).enumerate() {
            let r = c * cr + l;
            let (h, i) = (r / t, r % t);
            row.copy_from_slice(&src[i * n_state + h * d_head..i * n_state + h * d_head + d_head]);
        }
    });
}

/// Scatter head-major `o [hh, t, d_head]` -> interleaved `out [t, n_state]`, `chunks` bands
/// (chunks==0 ⇒ one band per output row = legacy).
fn scatter(out: &mut [f32], o: &[f32], hh: usize, t: usize, d_head: usize, n_state: usize, chunks: usize) {
    let n = if chunks == 0 { t } else { chunks.min(t).max(1) };
    let rp = t.div_ceil(n).max(1);
    out.par_chunks_mut(rp * n_state).enumerate().for_each(|(c, blk)| {
        for (l, orow) in blk.chunks_mut(n_state).enumerate() {
            let i = c * rp + l;
            for h in 0..hh {
                orow[h * d_head..(h + 1) * d_head]
                    .copy_from_slice(&o[h * t * d_head + i * d_head..h * t * d_head + i * d_head + d_head]);
            }
        }
    });
}

fn full_op(q: &[f32], k: &[f32], v: &[f32], qa: &mut [f32], ka: &mut [f32], va: &mut [f32],
           out: &mut [f32], hh: usize, tq: usize, tk: usize, d_head: usize, n_state: usize, chunks: usize) {
    gather(qa, q, hh, tq, d_head, n_state, chunks);
    gather(ka, k, hh, tk, d_head, n_state, chunks);
    gather(va, v, hh, tk, d_head, n_state, chunks);
    let scale = (d_head as f32).powf(-0.5);
    let o = ft_kernel_cpu::sdpa_forward_f32(qa, ka, va, hh, tq, tk, d_head, d_head, scale, false);
    scatter(out, &o, hh, tq, d_head, n_state, chunks);
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(120);
    let (hh, tq, tk, d_head) = (20usize, 1500usize, 1500usize, 64usize);
    let n_state = hh * d_head;
    let one = tq * n_state;
    let pool_n = 12usize; // 12 * 3 inputs * 7.68MB ≈ 276MB > L3
    println!("=== full encoder SDPA op (gather+kernel+scatter) cold: legacy chunking vs 16 ===");

    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) };
    let qs: Vec<Vec<f32>> = (0..pool_n).map(|_| (0..one).map(|_| nf()).collect()).collect();
    let ks: Vec<Vec<f32>> = (0..pool_n).map(|_| (0..one).map(|_| nf()).collect()).collect();
    let vs: Vec<Vec<f32>> = (0..pool_n).map(|_| (0..one).map(|_| nf()).collect()).collect();
    let mut qa = vec![0.0f32; hh * tq * d_head];
    let mut ka = vec![0.0f32; hh * tk * d_head];
    let mut va = vec![0.0f32; hh * tk * d_head];
    let mut outs: Vec<Vec<f32>> = (0..pool_n).map(|_| vec![0.0f32; tq * n_state]).collect();

    // byte-exactness: legacy vs 16 produce identical out.
    full_op(&qs[0], &ks[0], &vs[0], &mut qa, &mut ka, &mut va, &mut outs[0], hh, tq, tk, d_head, n_state, 0);
    let ref0 = outs[0].clone();
    full_op(&qs[0], &ks[0], &vs[0], &mut qa, &mut ka, &mut va, &mut outs[0], hh, tq, tk, d_head, n_state, 16);
    let bad = ref0.iter().zip(&outs[0]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();

    let run = |chunks: usize, qa: &mut [f32], ka: &mut [f32], va: &mut [f32], outs: &mut [Vec<f32>]| -> f64 {
        for _ in 0..3 { full_op(&qs[0], &ks[0], &vs[0], qa, ka, va, &mut outs[0], hh, tq, tk, d_head, n_state, chunks); }
        let mut total = 0.0f64;
        for it in 0..iters {
            let kk = it % pool_n;
            let t = Instant::now();
            full_op(&qs[kk], &ks[kk], &vs[kk], qa, ka, va, &mut outs[kk], hh, tq, tk, d_head, n_state, chunks);
            total += t.elapsed().as_secs_f64();
            black_box(outs[kk][0]);
        }
        total / iters as f64
    };
    let legacy = run(0, &mut qa, &mut ka, &mut va, &mut outs);
    let c16 = run(16, &mut qa, &mut ka, &mut va, &mut outs);
    println!("  byte-exact (legacy vs 16): {bad} differing  [{}]", if bad == 0 { "IDENTICAL" } else { "DIVERGENT" });
    println!("  legacy (per-head/per-row) : {:>8.1} µs/op", legacy * 1e6);
    println!("  16 balanced chunks        : {:>8.1} µs/op  {:.3}x  [{}]",
        c16 * 1e6, legacy / c16, if c16 < legacy { "WIN" } else { "loss" });
    println!("  => per-LAYER SDPA speedup from the chunk flip (× {} encoder layers = e2e-relevant)", 32);
}
