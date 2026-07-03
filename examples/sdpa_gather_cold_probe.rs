//! SDPA gather thread-count on COLD (DRAM-bound) data (BlackThrush, 2026-07-03).
//!
//! The prior `sdpa_gather_probe` was confounded: its tight loop kept q+qa (~15 MB)
//! L3-resident on ONE CCD, so it measured the cache-hot regime (serial won) — which
//! contradicts the real encoder where q is freshly scattered across CCDs by the
//! parallel QKV matmul (DRAM/cross-CCD bound). This probe forces the DRAM-bound regime
//! by rotating through a POOL of q/qa buffers whose total footprint (>200 MB) far
//! exceeds the 128 MB L3, so every iteration's q is cold. It then sweeps the rayon
//! pool size (the live gather chunks per-head = 20 tasks; pool size caps concurrency)
//! to find the true optimum, resolving whether the live per-head (~20-way) gather is
//! at its thread-count ceiling. All outputs are byte-identical (pure data movement).
//! Reports MEAN (steady-state DRAM), not min (which would catch a lucky cache hit).
//! Usage: `sdpa_gather_cold_probe [iters]` (default 400).
#![allow(unsafe_code)]
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Live gather: par over heads (chunk = one head), per-head strided read of q.
fn gather_headmajor(q: &[f32], qa: &mut [f32], hh: usize, tq: usize, d_head: usize) {
    let n_state = hh * d_head;
    qa.par_chunks_mut(tq * d_head).enumerate().for_each(|(h, blk)| {
        let base = h * d_head;
        for i in 0..tq {
            blk[i * d_head..(i + 1) * d_head]
                .copy_from_slice(&q[i * n_state + base..i * n_state + base + d_head]);
        }
    });
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(400);
    let (hh, tq, d_head) = (20usize, 1500usize, 64usize);
    let n_state = hh * d_head;
    let one = tq * n_state; // 1.92M f32 = 7.68 MB
    // Pool > L3 (128 MB): 30 * 7.68 MB = 230 MB of q, plus 230 MB of qa.
    let pool_n = 30usize;
    println!("=== SDPA gather thread-count on COLD data (pool {}×{:.1}MB q + same qa = {:.0}MB > 128MB L3) ===",
        pool_n, one as f64 * 4.0 / 1e6, pool_n as f64 * one as f64 * 8.0 / 1e6);

    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 40) as f32 / (1u64 << 24) as f32 };
    let qs: Vec<Vec<f32>> = (0..pool_n).map(|_| (0..one).map(|_| nf()).collect()).collect();
    let mut qas: Vec<Vec<f32>> = (0..pool_n).map(|_| vec![0.0f32; hh * tq * d_head]).collect();

    // Correctness: byte-identical across pool sizes (reference = serial-equivalent).
    let mut refbuf = vec![0.0f32; hh * tq * d_head];
    {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| gather_headmajor(&qs[0], &mut refbuf, hh, tq, d_head));
    }

    let threads = [1usize, 2, 4, 8, 12, 16, 20, 32, 64];
    let mut means = Vec::new();
    let mut bads = Vec::new();
    for &nt in &threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
        pool.install(|| gather_headmajor(&qs[0], &mut qas[0], hh, tq, d_head)); // warm
        let mut total = 0.0f64;
        pool.install(|| {
            for it in 0..iters {
                let k = it % pool_n; // rotate through >L3 pool → q[k] cold on reuse
                let t = Instant::now();
                gather_headmajor(&qs[k], &mut qas[k], hh, tq, d_head);
                total += t.elapsed().as_secs_f64();
                black_box(qas[k][0] + qas[k][hh * tq * d_head - 1]);
            }
        });
        gather_headmajor(&qs[0], &mut qas[0], hh, tq, d_head);
        bads.push(qas[0].iter().zip(&refbuf).filter(|(a, b)| a.to_bits() != b.to_bits()).count());
        means.push(total / iters as f64);
    }
    let live_mean = means[threads.iter().position(|&t| t == 20).unwrap()];
    let bytes = (one as f64 * 4.0) * 2.0; // read q + write qa
    println!("  threads |   mean µs | vs live(20) | GB/s(r+w) | byte-exact");
    for (i, &nt) in threads.iter().enumerate() {
        println!("  {nt:>7} | {:>9.2} | {:>10.2}x | {:>8.1} | {}",
            means[i] * 1e6, live_mean / means[i], bytes / means[i] / 1e9,
            if bads[i] == 0 { "IDENTICAL" } else { "DIVERGENT" });
    }
    println!("  (vs live: >1.00x means that thread count BEATS the live per-head 20-way gather)");
}
