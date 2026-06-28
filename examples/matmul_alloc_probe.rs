//! Measure the DEAD output zero-init in `nn::matmul`. The allocating
//! `ft_kernel_cpu::matmul_tensor_contiguous_f32` does `Vec::new()` ->
//! `resize(m*n, 0.0)` (zero-inits the whole output) then the GEMM (beta=0)
//! OVERWRITES every element — so the zero-init is dead. The buffer-reusing
//! `_into` variant skips it when the buffer is reused. This A/Bs the two on the
//! tiny.en encoder MLP fc1 shape [1500,384]x[384,1536] (output 9.2 MB). Model-free.
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;
use ft_core::{DType, Device, TensorMeta};
use std::hint::black_box;
use std::time::Instant;

fn fill(m: usize, k: usize, seed: u64) -> Mat {
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

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let (m, k, n) = (1500usize, 384usize, 1536usize); // encoder MLP fc1
    let a = fill(m, k, 0x11);
    let b = fill(k, n, 0x22);
    let lhs_meta = TensorMeta::from_shape(vec![m, k], DType::F32, Device::Cpu);
    let rhs_meta = TensorMeta::from_shape(vec![k, n], DType::F32, Device::Cpu);

    let best = |f: &mut dyn FnMut() -> (f64, f64)| {
        let mut bt = f64::INFINITY;
        let mut cs = 0.0;
        for _ in 0..iters {
            let (t, c) = f();
            if t < bt {
                bt = t;
            }
            cs = c;
        }
        (bt, cs)
    };

    // (a) allocating nn::matmul: malloc + dead zero-init + GEMM, every call.
    for _ in 0..3 {
        black_box(nn::matmul(&a, &b).expect("matmul"));
    }
    let (alloc_t, alloc_cs) = best(&mut || {
        let t0 = Instant::now();
        let r = nn::matmul(&a, &b).expect("matmul");
        let dt = t0.elapsed().as_secs_f64();
        let c: f64 = r.data.iter().map(|&x| x as f64).sum();
        black_box(&r);
        (dt, c)
    });

    // (b) reused-buffer _into: no realloc / no zero-init after the first call.
    let mut reused: Vec<f32> = Vec::new();
    for _ in 0..3 {
        ft_kernel_cpu::matmul_tensor_contiguous_f32_into(
            &mut reused, &a.data, &b.data, &lhs_meta, &rhs_meta,
        )
        .expect("into");
    }
    let (into_t, into_cs) = best(&mut || {
        let t0 = Instant::now();
        ft_kernel_cpu::matmul_tensor_contiguous_f32_into(
            &mut reused, &a.data, &b.data, &lhs_meta, &rhs_meta,
        )
        .expect("into");
        let dt = t0.elapsed().as_secs_f64();
        let c: f64 = reused.iter().map(|&x| x as f64).sum();
        black_box(&reused);
        (dt, c)
    });

    println!("== encoder MLP fc1 [{m},{k}]x[{k},{n}] (output {:.1} MB), best-of-{iters} ==", (m * n * 4) as f64 / 1e6);
    println!("(a) allocating nn::matmul (zero-init each call): {:.3} ms", alloc_t * 1e3);
    println!("(b) reused-buffer _into  (no zero-init)        : {:.3} ms", into_t * 1e3);
    println!("dead-zero-init+malloc per matmul = {:.3} ms  ({:.1}% of the call)", (alloc_t - into_t) * 1e3, 100.0 * (alloc_t - into_t) / alloc_t);
    println!("checksums: alloc={alloc_cs:.3} into={into_cs:.3} (equal => bit-identical GEMM)");
}
