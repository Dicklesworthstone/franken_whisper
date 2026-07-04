//! SDPA scatter chunk-count on COLD (DRAM-bound) data (BlackThrush, 2026-07-03).
//!
//! Sibling of `sdpa_gather_cold_probe`. The fused-SDPA output scatter reshapes the
//! kernel's head-major `o` [hh, tq, d_head] back into interleaved `out` [tq, n_state]
//! via `out.par_chunks_mut(n_state)` — tq=1500 FINE chunks (one output row each),
//! run ~32-way on the global pool. The gather probe showed this 8-channel box's
//! bandwidth-bound transposes peak near ~16 concurrent streams and DEGRADE at high
//! fan-out. This sweeps the scatter's chunk COUNT (concurrency ≈ min(chunks, pool)) on
//! cold data (pool of q-sized buffers >> 128 MB L3) to find its optimum. Byte-identical
//! for every chunk count (pure data movement). Reports MEAN (steady-state DRAM).
//! Usage: `sdpa_scatter_cold_probe [iters]` (default 400).
#![allow(unsafe_code)]
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Scatter head-major `o` [hh, t, d_head] into interleaved `out` [t, n_state], over
/// `chunks` balanced row-bands (chunks==0 ⇒ per-row == the live `par_chunks_mut(n_state)`).
fn scatter(
    out: &mut [f32],
    o: &[f32],
    hh: usize,
    t: usize,
    d_head: usize,
    n_state: usize,
    chunks: usize,
) {
    let rows_per = if chunks == 0 {
        1
    } else {
        t.div_ceil(chunks.min(t).max(1)).max(1)
    };
    out.par_chunks_mut(rows_per * n_state)
        .enumerate()
        .for_each(|(c, blk)| {
            let i0 = c * rows_per;
            for (local, orow) in blk.chunks_mut(n_state).enumerate() {
                let i = i0 + local;
                for h in 0..hh {
                    orow[h * d_head..(h + 1) * d_head].copy_from_slice(
                        &o[h * t * d_head + i * d_head..h * t * d_head + i * d_head + d_head],
                    );
                }
            }
        });
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let (hh, t, d_head) = (20usize, 1500usize, 64usize);
    let n_state = hh * d_head;
    let one = t * n_state; // 7.68 MB
    let pool_n = 30usize;
    println!(
        "=== SDPA scatter chunk-count on COLD data (pool {}×{:.1}MB = {:.0}MB > 128MB L3) ===",
        pool_n,
        one as f64 * 4.0 / 1e6,
        pool_n as f64 * one as f64 * 8.0 / 1e6
    );

    let mut s = 0xD1B5_4A32_D192_ED03u64;
    let mut nf = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let os: Vec<Vec<f32>> = (0..pool_n)
        .map(|_| (0..hh * t * d_head).map(|_| nf()).collect())
        .collect();
    let mut outs: Vec<Vec<f32>> = (0..pool_n).map(|_| vec![0.0f32; t * n_state]).collect();

    // Reference: per-row scatter.
    let mut refbuf = vec![0.0f32; t * n_state];
    scatter(&mut refbuf, &os[0], hh, t, d_head, n_state, 0);

    let chunk_opts = [0usize /*per-row*/, 64, 32, 20, 16, 12, 8, 4];
    let mut means = Vec::new();
    let mut bads = Vec::new();
    for &ch in &chunk_opts {
        scatter(&mut outs[0], &os[0], hh, t, d_head, n_state, ch); // warm
        let mut total = 0.0f64;
        for it in 0..iters {
            let k = it % pool_n;
            let tm = Instant::now();
            scatter(&mut outs[k], &os[k], hh, t, d_head, n_state, ch);
            total += tm.elapsed().as_secs_f64();
            black_box(outs[k][0] + outs[k][t * n_state - 1]);
        }
        scatter(&mut outs[0], &os[0], hh, t, d_head, n_state, ch);
        bads.push(
            outs[0]
                .iter()
                .zip(&refbuf)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count(),
        );
        means.push(total / iters as f64);
    }
    let live = means[0]; // per-row (current)
    let bytes = (one as f64 * 4.0) * 2.0; // read o + write out
    println!("  chunks |   mean µs | vs live(row) | GB/s(r+w) | byte-exact");
    for (i, &ch) in chunk_opts.iter().enumerate() {
        let label = if ch == 0 {
            "row(1500)".to_string()
        } else {
            ch.to_string()
        };
        println!(
            "  {label:>9} | {:>9.2} | {:>11.2}x | {:>8.1} | {}",
            means[i] * 1e6,
            live / means[i],
            bytes / means[i] / 1e9,
            if bads[i] == 0 {
                "IDENTICAL"
            } else {
                "DIVERGENT"
            }
        );
    }
    println!("  (vs live: >1.00x means that chunk count BEATS the current per-row scatter)");
}
