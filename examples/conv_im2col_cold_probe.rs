//! Conv2 im2col chunk-count on COLD data (BlackThrush, 2026-07-03).
//!
//! Completes the encoder reshape audit. After hoisting the conv weight transpose to load
//! (108d3cd), the remaining per-window conv cost is the im2col gather (derived from the
//! input, so legitimately per-window). It parallelizes over ~worker_count() bands. Unlike
//! the SDPA gather/scatter (5 KB stride ⇒ heavily oversubscribed), the im2col writes are
//! stride-K=3 (quasi-contiguous), so it MAY already be fine at high fan-out. This sweeps
//! the chunk count on cold data (pool >> L3) to settle it — turbo conv2 shape.
//! Byte-identical for every chunk count. Reports MEAN (steady-state DRAM).
//! Usage: `conv_im2col_cold_probe [iters]` (default 300).
#![allow(unsafe_code)]
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Build the conv im2col [t_out, cin*k] over `chunks` balanced output-row bands
/// (chunks==0 ⇒ per worker_count()-style band; here we test explicit counts).
/// Mirrors nn::conv1d_wt's fill_row exactly (stride/pad handling).
fn im2col(x: &[f32], t_in: usize, cin: usize, k: usize, stride: usize, pad: usize, t_out: usize, chunks: usize, out: &mut [f32]) {
    let patch = cin * k;
    let rows_per = t_out.div_ceil(chunks.max(1)).max(1);
    out.par_chunks_mut(rows_per * patch).enumerate().for_each(|(c, band)| {
        let o0 = c * rows_per;
        for (local, row) in band.chunks_mut(patch).enumerate() {
            let o = o0 + local;
            let start = o * stride;
            for kk in 0..k {
                let p = start + kk;
                if p < pad { continue; }
                let ti = p - pad;
                if ti >= t_in { continue; }
                let src = &x[ti * cin..(ti + 1) * cin];
                for ci in 0..cin {
                    row[ci * k + kk] = src[ci];
                }
            }
        }
    });
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    // turbo conv2: input [3000, 1280], stride 2, k 3, pad 1 -> t_out=1500, patch=3840.
    let (t_in, cin, k, stride, pad) = (3000usize, 1280usize, 3usize, 2usize, 1usize);
    let patch = cin * k;
    let t_out = (t_in + 2 * pad - k) / stride + 1;
    let xlen = t_in * cin; // 3.84M f32 = 15.4 MB
    let outlen = t_out * patch; // 5.76M f32 = 23 MB
    let pool_n = 16usize; // 16*(15.4+23) = 614 MB > 128 MB L3
    println!("=== conv2 im2col chunk-count on COLD data (t_out={t_out}, patch={patch}, pool {} = {:.0}MB) ===",
        pool_n, pool_n as f64 * (xlen + outlen) as f64 * 4.0 / 1e6);

    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 40) as f32 / (1u64 << 24) as f32 };
    let xs: Vec<Vec<f32>> = (0..pool_n).map(|_| (0..xlen).map(|_| nf()).collect()).collect();
    let mut outs: Vec<Vec<f32>> = (0..pool_n).map(|_| vec![0.0f32; outlen]).collect();

    let mut refbuf = vec![0.0f32; outlen];
    im2col(&xs[0], t_in, cin, k, stride, pad, t_out, 1, &mut refbuf);

    let chunk_opts = [1usize, 4, 8, 12, 16, 24, 32, 48, 64];
    let mut means = Vec::new();
    let mut bads = Vec::new();
    for &ch in &chunk_opts {
        im2col(&xs[0], t_in, cin, k, stride, pad, t_out, ch, &mut outs[0]); // warm
        let mut total = 0.0f64;
        for it in 0..iters {
            let kk = it % pool_n;
            let tm = Instant::now();
            im2col(&xs[kk], t_in, cin, k, stride, pad, t_out, ch, &mut outs[kk]);
            total += tm.elapsed().as_secs_f64();
            black_box(outs[kk][0] + outs[kk][outlen - 1]);
        }
        im2col(&xs[0], t_in, cin, k, stride, pad, t_out, ch, &mut outs[0]);
        bads.push(outs[0].iter().zip(&refbuf).filter(|(a, b)| a.to_bits() != b.to_bits()).count());
        means.push(total / iters as f64);
    }
    // current code uses worker_count() bands (~32 on the encoder pool).
    let live = means[chunk_opts.iter().position(|&c| c == 32).unwrap()];
    let bytes = (xlen as f64 * 4.0) + (outlen as f64 * 4.0);
    println!("  chunks |   mean µs | vs ~live(32) | GB/s(r+w) | byte-exact");
    for (i, &ch) in chunk_opts.iter().enumerate() {
        println!("  {ch:>6} | {:>9.1} | {:>11.2}x | {:>8.1} | {}",
            means[i] * 1e6, live / means[i], bytes / means[i] / 1e9,
            if bads[i] == 0 { "IDENTICAL" } else { "DIVERGENT" });
    }
    println!("  (the conv encode path uses ~worker_count()=~32 bands; >1.00x means a lower count is faster)");
}
