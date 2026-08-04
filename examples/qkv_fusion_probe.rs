//! QKV-fusion micro-probe (BlackThrush, 2026-07-02).
//!
//! The encoder self-attention computes Q/K/V as THREE separate matmuls over the
//! same activation `h` ([n_ctx, n_state]) — so the activation is packed 3× by
//! sgemm (encoder.rs:610-612). This probe MEASURES whether fusing them into ONE
//! `[n_ctx, n_state] x [n_state, 3*n_state]` GEMM (concatenated weight) beats the
//! 3 separate calls, both raw and after de-interleaving the [n_ctx, 3*n_state]
//! output back into 3 contiguous [n_ctx, n_state] buffers (what the attention
//! path needs). Also asserts byte-exactness (fused columns == separate outputs),
//! since column-concatenation must not change any per-element k-accumulation.
//!
//! Usage: `qkv_fusion_probe [iters] [n_state]`  (default 50, 1280=turbo).
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;
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
    let mut a = std::env::args().skip(1);
    let iters: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(50);
    let n_state: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1280);
    let n_ctx = 1500usize;

    let h = fill(n_ctx, n_state, 0x1111);
    let wq = fill(n_state, n_state, 0x2222);
    let wk = fill(n_state, n_state, 0x3333);
    let wv = fill(n_state, n_state, 0x4444);

    // Fused weight [n_state, 3*n_state] = [Wq | Wk | Wv] (row-major hstack).
    let n3 = 3 * n_state;
    let mut wqkv = vec![0.0f32; n_state * n3];
    for r in 0..n_state {
        let dst = &mut wqkv[r * n3..(r + 1) * n3];
        dst[..n_state].copy_from_slice(&wq.data[r * n_state..(r + 1) * n_state]);
        dst[n_state..2 * n_state].copy_from_slice(&wk.data[r * n_state..(r + 1) * n_state]);
        dst[2 * n_state..].copy_from_slice(&wv.data[r * n_state..(r + 1) * n_state]);
    }
    let wqkv = Mat::from_vec(n_state, n3, wqkv);

    // Byte-exactness: fused output columns must equal the separate outputs.
    let q = nn::matmul(&h, &wq).unwrap();
    let k = nn::matmul(&h, &wk).unwrap();
    let v = nn::matmul(&h, &wv).unwrap();
    let fused = nn::matmul(&h, &wqkv).unwrap();
    let mut max_abs = 0.0f32;
    for i in 0..n_ctx {
        let frow = &fused.data[i * n3..(i + 1) * n3];
        for j in 0..n_state {
            max_abs = max_abs.max((frow[j] - q.data[i * n_state + j]).abs());
            max_abs = max_abs.max((frow[n_state + j] - k.data[i * n_state + j]).abs());
            max_abs = max_abs.max((frow[2 * n_state + j] - v.data[i * n_state + j]).abs());
        }
    }
    println!("byte-exact check: fused-vs-separate max|Δ| = {max_abs:.3e}  (0 = bit-exact)");

    // Warmup.
    for _ in 0..3 {
        black_box(nn::matmul(&h, &wq).unwrap());
        black_box(nn::matmul(&h, &wqkv).unwrap());
    }

    // A) 3 separate matmuls.
    let mut best_sep = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let q = nn::matmul(&h, &wq).unwrap();
        let k = nn::matmul(&h, &wk).unwrap();
        let v = nn::matmul(&h, &wv).unwrap();
        let dt = t.elapsed().as_secs_f64();
        black_box((q, k, v));
        best_sep = best_sep.min(dt);
    }

    // B) fused matmul only (raw GEMM+pack savings, no de-interleave).
    let mut best_fused = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let f = nn::matmul(&h, &wqkv).unwrap();
        let dt = t.elapsed().as_secs_f64();
        black_box(f);
        best_fused = best_fused.min(dt);
    }

    // C) fused matmul + de-interleave into 3 contiguous [n_ctx, n_state] buffers.
    let mut best_fused_di = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let f = nn::matmul(&h, &wqkv).unwrap();
        let mut qo = vec![0.0f32; n_ctx * n_state];
        let mut ko = vec![0.0f32; n_ctx * n_state];
        let mut vo = vec![0.0f32; n_ctx * n_state];
        for i in 0..n_ctx {
            let frow = &f.data[i * n3..(i + 1) * n3];
            qo[i * n_state..(i + 1) * n_state].copy_from_slice(&frow[..n_state]);
            ko[i * n_state..(i + 1) * n_state].copy_from_slice(&frow[n_state..2 * n_state]);
            vo[i * n_state..(i + 1) * n_state].copy_from_slice(&frow[2 * n_state..]);
        }
        let dt = t.elapsed().as_secs_f64();
        black_box((qo, ko, vo));
        best_fused_di = best_fused_di.min(dt);
    }

    println!("n_state={n_state} n_ctx={n_ctx}, best-of-{iters}:");
    println!("  A) 3× separate matmul      : {:.3} ms", best_sep * 1e3);
    println!(
        "  B) 1× fused matmul (raw)   : {:.3} ms   ({:.3}× vs A)",
        best_fused * 1e3,
        best_sep / best_fused
    );
    println!(
        "  C) fused + de-interleave   : {:.3} ms   ({:.3}× vs A)",
        best_fused_di * 1e3,
        best_sep / best_fused_di
    );
}
