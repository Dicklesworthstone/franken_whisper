//! Effective-bandwidth probe for the decode logits GEMV (`nn::gemv_f16`).
//!
//! The decode's dominant cost is the tied-output projection `[51864,384]` f16
//! (~40 MB read per token). This times the LANDED fused-f16c-dot `gemv_f16` on
//! that shape and reports GB/s (f16 weight bytes / time) + GFLOP/s, to check it
//! against the memory ceiling. CAVEAT: a tight loop keeps the 40 MB weight
//! L3-resident (this box has 128 MB L3), so this reports the L3-resident rate;
//! in real decode the weight may be partly DRAM-streamed between tokens, so the
//! production figure can be lower. Run in a calm window. Usage: `[iters]` (50).
use franken_whisper::native_engine::nn;
use ft_core::Float16;
use std::hint::black_box;
use std::time::Instant;

fn bench(name: &str, out: usize, inp: usize, iters: usize) {
    let w: Vec<Float16> = (0..out * inp)
        .map(|i| Float16::from_f32(((i % 101) as f32 / 101.0) - 0.5))
        .collect();
    let x: Vec<f32> = (0..inp).map(|i| (i as f32 / inp as f32) - 0.5).collect();
    let mut buf = vec![0.0f32; out];
    for _ in 0..3 {
        nn::gemv_f16(&w, out, inp, &x, None, &mut buf);
        black_box(&buf);
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        nn::gemv_f16(&w, out, inp, &x, None, &mut buf);
        let dt = t0.elapsed().as_secs_f64();
        black_box(&buf);
        if dt < best {
            best = dt;
        }
    }
    let gbs = (out * inp * 2) as f64 / best / 1e9;
    let gflops = 2.0 * (out as f64) * (inp as f64) / best / 1e9;
    println!(
        "{name:<12} [{out},{inp}] f16  best={:.3}ms  {gbs:5.0} GB/s  {gflops:5.0} GFLOP/s",
        best * 1e3
    );
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    println!("== decode f16 GEMV, best-of-{iters} ==");
    bench("logits", 51864, 384, iters); // tied-output projection (the ~40MB/token cost)
    bench("mlp fc1", 1536, 384, iters); // per-token MLP up-proj
    bench("attn proj", 384, 384, iters); // Q/K/V/out projection
}
