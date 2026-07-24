//! int8 vs f32 GEMM throughput on the TURBO encoder shapes (BlackThrush, 2026-07-02).
//!
//! Quantifies the top turbo lever: the encoder is compute-bound f32 `matmul`
//! (matrixmultiply sgemm) and dominates per-window cost (~3.1 s/window, see
//! NEGATIVE_EVIDENCE 2026-07-02). A byte-exact speedup is impossible (sum-order),
//! but TRUE int8xint8 GEMM (`vpmaddwd`, LLVM-autovec of a scalar i32 dot under
//! target-cpu=x86-64-v3) is denser compute — this is how whisper.cpp-Q8 encodes
//! fast. This probe MEASURES that int8-vs-f32 ratio on the real turbo GEMM shapes
//! so the owner can decide the (non-byte-exact) quality tradeoff with a number.
//!
//! The int8 GEMM here is a faithful row/col-symmetric-quantized GEMM (per-row A
//! scale, per-col B scale), parallelized over output rows exactly like the real
//! encoder GEMM would be — NOT a per-row `gemv_i8` (which pays 1500x rayon
//! dispatch). So the ratio is representative, not a pessimistic proxy.
//!
//! Build release-perf; run in a CALM window. Usage: `encoder_int8_gemm_probe [iters]`.
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

/// Symmetric per-row int8 quantization of a `[rows, k]` row-major matrix.
/// Returns (q, scale-per-row). Same amax/127 scheme as `nn::quantize_f16_to_i8`.
fn quant_rows(a: &[f32], rows: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * k];
    let mut sc = vec![0.0f32; rows];
    q.par_chunks_mut(k)
        .zip(sc.par_iter_mut())
        .enumerate()
        .for_each(|(r, (qr, s))| {
            let row = &a[r * k..(r + 1) * k];
            let amax = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9);
            let scale = amax / 127.0;
            *s = scale;
            let inv = 1.0 / scale;
            for (d, &x) in qr.iter_mut().zip(row) {
                *d = (x * inv).round().clamp(-127.0, 127.0) as i8;
            }
        });
    (q, sc)
}

/// int8 GEMM `C[m,n] = A[m,k] @ B[k,n]`, both symmetric-quantized. `bt` is B
/// pre-transposed to `[n,k]` row-major (so each output dot is contiguous). The
/// inner scalar i32 dot is what LLVM lowers to `vpmovsxbw`+`vpmaddwd` under
/// target-cpu=x86-64-v3 (see `nn::dot_i8`). Parallel over output rows.
fn gemm_i8(
    qa: &[i8],
    sa: &[f32],
    qbt: &[i8],
    sb: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(i, crow)| {
        let arow = &qa[i * k..(i + 1) * k];
        let sai = sa[i];
        for o in 0..n {
            let brow = &qbt[o * k..(o + 1) * k];
            let mut acc: i32 = 0;
            for x in 0..k {
                acc += (arow[x] as i32) * (brow[x] as i32);
            }
            crow[o] = acc as f32 * sai * sb[o];
        }
    });
    c
}

fn bench(name: &str, m: usize, k: usize, n: usize, iters: usize) {
    let a = fill(m * k, 0x1234);
    let b = fill(k * n, 0x5678); // [k, n] row-major
    let a_mat = Mat::from_vec(m, k, a.clone());
    let b_mat = Mat::from_vec(k, n, b.clone());

    // --- f32 sgemm (the current encoder path) ---
    for _ in 0..3 {
        black_box(nn::matmul(&a_mat, &b_mat).expect("matmul"));
    }
    let mut best_f32 = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = nn::matmul(&a_mat, &b_mat).expect("matmul");
        let dt = t0.elapsed().as_secs_f64();
        black_box(r);
        best_f32 = best_f32.min(dt);
    }

    // --- int8 GEMM (quantize once, then dot) ---
    // Pre-transpose B [k,n] -> bt [n,k] and quantize per output row (= per B col).
    let mut bt = vec![0.0f32; n * k];
    for kk in 0..k {
        for o in 0..n {
            bt[o * k + kk] = b[kk * n + o];
        }
    }
    let (qa, sa) = quant_rows(&a, m, k);
    let (qbt, sb) = quant_rows(&bt, n, k);
    for _ in 0..3 {
        black_box(gemm_i8(&qa, &sa, &qbt, &sb, m, k, n));
    }
    let mut best_i8 = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        let r = gemm_i8(&qa, &sa, &qbt, &sb, m, k, n);
        let dt = t0.elapsed().as_secs_f64();
        black_box(r);
        best_i8 = best_i8.min(dt);
    }

    let g32 = 2.0 * (m * k * n) as f64 / best_f32 / 1e9;
    let g8 = 2.0 * (m * k * n) as f64 / best_i8 / 1e9;
    println!(
        "{name:<14} [{m},{k}]x[{k},{n}]  f32 {:.2}ms ({g32:.0} GF/s)  int8 {:.2}ms ({g8:.0} GF/s)  speedup {:.2}x",
        best_f32 * 1e3,
        best_i8 * 1e3,
        best_f32 / best_i8
    );
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    // large-v3-turbo encoder shapes: n_state=1280, n_mlp=5120, n_ctx=1500.
    println!("== turbo encoder GEMM: int8 vs f32, best-of-{iters} ==");
    bench("proj QKV/out", 1500, 1280, 1280, iters);
    bench("mlp fc1", 1500, 1280, 5120, iters);
    bench("mlp fc2", 1500, 5120, 1280, iters);
}
