//! Conv weight transpose cost: the per-window redundant `w_t` build (BlackThrush, 2026-07-03).
//!
//! `nn::conv1d` transposes its weight `[Cout, Cin*K] -> [Cin*K, Cout]` on EVERY call
//! (serial nested loop, nn.rs ~2221) to feed `matmul_bias`. But conv weights are CONSTANT,
//! and conv1d runs once per encoder window — so this transpose is recomputed every window
//! for no reason (unlike the encoder's LINEAR weights, which are pre-transposed once at
//! load in EncoderWeights::from_ggml). This measures the per-window cost for the two turbo
//! conv layers to decide whether pre-transposing at load is worth it.
//!   conv1: Cout=1280, Cin=80,  K=3 -> patch=240,  w_t [240, 1280]
//!   conv2: Cout=1280, Cin=1280, K=3 -> patch=3840, w_t [3840, 1280]
//! Usage: `conv_wt_transpose_probe [iters]` (default 2000).
use std::hint::black_box;
use std::time::Instant;

/// Exact replica of conv1d's serial weight transpose.
fn transpose_wt(w: &[f32], cout: usize, patch: usize) -> Vec<f32> {
    let mut w_t = vec![0.0f32; patch * cout];
    for co in 0..cout {
        for j in 0..patch {
            w_t[j * cout + co] = w[co * patch + j];
        }
    }
    w_t
}

fn bench(name: &str, cout: usize, patch: usize, iters: usize) {
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let w: Vec<f32> = (0..cout * patch)
        .map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 40) as f32 / (1u64 << 24) as f32 })
        .collect();
    for _ in 0..3 { black_box(transpose_wt(&w, cout, patch)); }
    let mut best = f64::INFINITY;
    let mut sum = 0.0f64;
    for _ in 0..iters {
        let t = Instant::now();
        black_box(transpose_wt(&w, cout, patch));
        let e = t.elapsed().as_secs_f64();
        best = best.min(e);
        sum += e;
    }
    println!("{name}: w_t [{patch}, {cout}] = {} elems", patch * cout);
    println!("  best {:>8.1} µs   mean {:>8.1} µs   (per window, serial strided transpose)",
        best * 1e6, sum / iters as f64 * 1e6);
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    println!("=== conv1d per-call weight transpose cost (redundant per window) ===");
    bench("conv1", 1280, 240, iters);
    bench("conv2", 1280, 3840, iters / 2);
    println!("  (both run EVERY encoder window; pre-transposing at load removes them entirely)");
}
