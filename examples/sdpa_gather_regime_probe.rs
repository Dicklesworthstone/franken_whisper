//! Is the SDPA gather chunk flip Pareto-safe across cache regimes? (BlackThrush, 2026-07-03).
//!
//! The gate on FW_SDPA_GATHER_CHUNKS was "thread-count is regime/load-dependent, can't settle
//! on a shared box without a quiet box + model" (per project_decode_overthreaded_rayon_lead's
//! 3 reverts). But those reverts were ~0-gain, LOAD-DEPENDENT decode changes. This tests
//! whether the gather flip is instead PARETO-SAFE: if 16 balanced chunks beats the legacy
//! per-head (n_head=20) chunking in BOTH the cache-HOT (single L3-resident buffer, reused) and
//! cache-COLD (rotating pool >> L3) regimes, then the regime/load dependence is irrelevant —
//! 16 wins everywhere, so contention (which only shifts the regime) can't flip the conclusion.
//! Byte-identical throughout. Reports both regimes side-by-side.
//! Usage: `sdpa_gather_regime_probe [iters]` (default 500).
#![allow(unsafe_code)]
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

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

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(500);
    let (hh, t, d_head) = (20usize, 1500usize, 64usize);
    let n_state = hh * d_head;
    let one = t * n_state; // 7.68 MB (fits one CCD's 32 MB L3 when hot)
    let pool_n = 30usize;
    println!("=== SDPA gather chunk flip: Pareto check across HOT vs COLD regimes (legacy=0 ⇒ per-head 20) ===");

    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 40) as f32 / (1u64 << 24) as f32 };
    let qs: Vec<Vec<f32>> = (0..pool_n).map(|_| (0..one).map(|_| nf()).collect()).collect();
    let mut qas: Vec<Vec<f32>> = (0..pool_n).map(|_| vec![0.0f32; hh * t * d_head]).collect();
    let mut hotqa = vec![0.0f32; hh * t * d_head];

    // byte-exactness across chunk counts.
    let mut refb = vec![0.0f32; hh * t * d_head];
    gather(&mut refb, &qs[0], hh, t, d_head, n_state, 0);
    let mut bad = 0usize;
    for &ch in &[8usize, 16, 20, 32] {
        let mut g = vec![0.0f32; hh * t * d_head];
        gather(&mut g, &qs[0], hh, t, d_head, n_state, ch);
        bad += g.iter().zip(&refb).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    }

    let chunk_opts = [0usize, 8, 12, 16, 20, 32];
    // HOT: reuse ONE q buffer (stays L3-resident), best-of (steady hot).
    let mut hot = Vec::new();
    for &ch in &chunk_opts {
        for _ in 0..5 { gather(&mut hotqa, &qs[0], hh, t, d_head, n_state, ch); }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let tm = Instant::now();
            gather(&mut hotqa, &qs[0], hh, t, d_head, n_state, ch);
            best = best.min(tm.elapsed().as_secs_f64());
            black_box(hotqa[0]);
        }
        hot.push(best);
    }
    // COLD: rotate >L3 pool, MEAN (steady DRAM).
    let mut cold = Vec::new();
    for &ch in &chunk_opts {
        gather(&mut qas[0], &qs[0], hh, t, d_head, n_state, ch);
        let mut total = 0.0f64;
        for it in 0..iters {
            let k = it % pool_n;
            let tm = Instant::now();
            gather(&mut qas[k], &qs[k], hh, t, d_head, n_state, ch);
            total += tm.elapsed().as_secs_f64();
            black_box(qas[k][0]);
        }
        cold.push(total / iters as f64);
    }

    let li = chunk_opts.iter().position(|&c| c == 0).unwrap();
    println!("  byte-exact across chunk counts: {bad} differing  [{}]", if bad == 0 { "IDENTICAL" } else { "DIVERGENT" });
    println!("  chunks | HOT µs (best) vs legacy | COLD µs (mean) vs legacy");
    for (i, &ch) in chunk_opts.iter().enumerate() {
        let label = if ch == 0 { "0(=20)".to_string() } else { ch.to_string() };
        println!("  {label:>6} | {:>9.1}  {:>6.2}x       | {:>9.1}  {:>6.2}x",
            hot[i] * 1e6, hot[li] / hot[i], cold[i] * 1e6, cold[li] / cold[i]);
    }
    let c16 = chunk_opts.iter().position(|&c| c == 16).unwrap();
    let pareto = hot[c16] <= hot[li] && cold[c16] <= cold[li];
    println!("  => 16 vs legacy(20): HOT {:.2}x, COLD {:.2}x  ⇒ {}",
        hot[li] / hot[c16], cold[li] / cold[c16],
        if pareto { "PARETO-SAFE (16 wins in BOTH regimes)" } else { "NOT Pareto (regime-dependent)" });
}
