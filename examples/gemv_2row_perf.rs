//! Fixed-iteration perf driver for the 2-row register-blocked GEMV A/B.
//!
//! Wall-clock best-of is unreliable on a contended box; this driver runs a FIXED
//! number of `nn::gemv_f16` calls on a SINGLE shape so external `perf stat` can
//! count instructions-retired (contention-independent, deterministic) and cycles
//! (a clean work proxy for an L1/L2-resident compute-bound kernel). Same call
//! count on candidate vs baseline → the delta is the kernel change.
//!
//! Usage: `gemv_2row_perf <shape> [iters]`  shape ∈ {attn,mlp,logits}  (iters dflt 20000)
use franken_whisper::native_engine::nn;
use ft_core::Float16;
use std::hint::black_box;

fn main() {
    let shape = std::env::args().nth(1).unwrap_or_else(|| "attn".into());
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20000);
    let (out, inp) = match shape.as_str() {
        "attn" => (384usize, 384usize),   // Q/K/V/out proj  (2-row path, L2-resident)
        "mlp" => (1536usize, 384usize),   // MLP up-proj     (2-row path)
        "logits" => (51864usize, 384usize), // tied output   (single-row path, mem-bound)
        other => panic!("unknown shape {other}"),
    };
    let w: Vec<Float16> = (0..out * inp)
        .map(|i| Float16::from_f32(((i % 101) as f32 / 101.0) - 0.5))
        .collect();
    let x: Vec<f32> = (0..inp).map(|i| (i as f32 / inp as f32) - 0.5).collect();
    let mut buf = vec![0.0f32; out];
    for _ in 0..iters {
        nn::gemv_f16(black_box(&w), out, inp, black_box(&x), None, &mut buf);
        black_box(&buf);
    }
    // Print a checksum so the optimizer cannot elide the work.
    let s: f32 = buf.iter().copied().sum();
    println!("shape={shape} out={out} inp={inp} iters={iters} checksum={s:.4}");
}
