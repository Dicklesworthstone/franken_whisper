//! Per-shape GFLOP/s probe for the encoder GEMMs (the e2e-dominant cost).
//!
//! Times `nn::matmul` (which dispatches to `ft_kernel_cpu`'s `matrixmultiply`
//! sgemm for M>1) on each encoder GEMM shape, to MEASURE — not estimate — which
//! shapes are `matrixmultiply`'s weak spots. Validates the bd-4hc0 sub-target
//! hypothesis: the small-inner-dim ATTENTION GEMMs (`[1500,64]x[64,1500]` K=64,
//! `[1500,1500]x[1500,64]` N=64) should show far lower GFLOP/s than the K>=384
//! projection/MLP shapes that bd-4hc0 actually measured.
//!
//! Build release for production codegen; run in a CALM window (timing-based).
//! Usage: `gemm_shape_probe [iters]` (default 50, best-of).
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;
use std::hint::black_box;
use std::time::Instant;

fn fill(m: usize, k: usize, seed: u64) -> Mat {
    // Deterministic non-trivial f32 fill; GFLOP/s is data-independent for a GEMM.
    let mut s = seed;
    let data: Vec<f32> = (0..m * k)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32) - 0.5
        })
        .collect();
    Mat::from_vec(m, k, data)
}

fn bench(name: &str, m: usize, k: usize, n: usize, iters: usize) {
    let a = fill(m, k, 0x1234);
    let b = fill(k, n, 0x5678);
    // warmup (also faults in pages / primes the rayon pool)
    for _ in 0..3 {
        black_box(nn::matmul(&a, &b).expect("matmul"));
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = nn::matmul(&a, &b).expect("matmul");
        let dt = t0.elapsed().as_secs_f64();
        black_box(r);
        if dt < best {
            best = dt;
        }
    }
    let gflops = 2.0 * (m as f64) * (k as f64) * (n as f64) / best / 1e9;
    println!(
        "{name:<16} [{m},{k}]x[{k},{n}]  best={:.3}ms  {gflops:6.0} GFLOP/s",
        best * 1e3
    );
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    // tiny.en encoder shapes (n_state=384, head_dim=64, n_mlp=1536, n_ctx=1500).
    println!("== encoder GEMM shapes, best-of-{iters} ==");
    bench("proj QKV/out", 1500, 384, 384, iters);
    bench("attn scores", 1500, 64, 1500, iters); // K=64  (small inner)
    bench("attn xV", 1500, 1500, 64, iters); // N=64  (small output)
    bench("mlp fc1", 1500, 384, 1536, iters);
    bench("mlp fc2", 1500, 1536, 384, iters);
}
