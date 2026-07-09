//! Encoder fc2 (mlp_proj) int8 DECOMPOSITION probe (2026-07-08, AshHeron).
//!
//! Last turn measured attn_in+fc2-int8 = 0.988× (SLOWER than f32) and attributed it to
//! the [1500,5120] GELU-output activation u8-quant done every layer. This probe SETTLES
//! the crux: does the maddubs fc2 GEMM (isolated from the activation quant) actually beat
//! f32 sgemm? If YES, a cheaper/fused activation quant could unlock fc2 int8 (+23.7% of
//! encoder at proven-safe quality). If NO (maddubs GEMM ≥ f32), fc2 is closed forever —
//! no quant trick recovers it.
//!
//! Decomposes fc2 [1500,5120]×[5120,1280] into:
//!   (a) f32:        matmul_bias(x, w_f32)                      — the current baseline
//!   (b) act-quant:  quantize_act_i7(x)                         — the per-layer overhead
//!   (c) maddubs:    matmul_bias_i7_quantized(x_i7, w_i7)       — GEMM ONLY (no quant)
//!   (d) full int8:  matmul_bias_i7(x, w_i7)                    — quant + GEMM (== b+c)
//! Verdict from (c) vs (a): int8-GEMM-only faster ⇒ fc2 recoverable; slower ⇒ dead.
//! Compares also the SMALLER fc1 shape [1500,1280]×[1280,5120] (attn_in's WORKING case)
//! as a control — fc1 int8 IS a win, so its maddubs GEMM must beat f32.

use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;
use std::time::Instant;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as i32 as f32) / (1i64 << 23) as f32 * 0.1
        })
        .collect()
}

fn bench<F: FnMut()>(label: &str, reps: usize, mut f: F) -> f64 {
    // warmup
    f();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms < best {
            best = ms;
        }
    }
    println!("  {label:<34} {best:8.3} ms (min of {reps})");
    best
}

fn probe(name: &str, m: usize, k: usize, n: usize) {
    println!("\n=== {name}: [{m},{k}] @ [{k},{n}] -> [{m},{n}] ===");
    let x = Mat::from_vec(m, k, fill(m * k, 0x1111));
    let w_t = Mat::from_vec(k, n, fill(k * n, 0x2222));
    let w_i7 = nn::quantize_mat_to_i7(&w_t);
    let bias = fill(n, 0x3333);

    let a = bench("(a) f32 matmul_bias", 20, || {
        let _ = nn::matmul_bias(&x, &w_t, Some(&bias)).unwrap();
    });
    let b = bench("(b) quantize_act_i7 (quant only)", 20, || {
        let _ = nn::quantize_act_i7(&x);
    });
    let xq = nn::quantize_act_i7(&x);
    let c = bench("(c) matmul_bias_i7_quantized (GEMM)", 20, || {
        let _ = nn::matmul_bias_i7_quantized(&xq, &w_i7, Some(&bias)).unwrap();
    });
    let d = bench("(d) matmul_bias_i7 (quant+GEMM)", 20, || {
        let _ = nn::matmul_bias_i7(&x, &w_i7, Some(&bias)).unwrap();
    });
    println!("  --------------------------------------------------");
    println!("  int8-GEMM-only (c) vs f32 (a):  {:.3}×  {}", a / c,
        if c < a { "int8 GEMM FASTER ✓" } else { "int8 GEMM SLOWER ✗" });
    println!("  full int8 (d) vs f32 (a):       {:.3}×  {}", a / d,
        if d < a { "int8 net FASTER ✓" } else { "int8 net SLOWER ✗" });
    println!("  quant overhead (b) as % of f32: {:.0}%", b / a * 100.0);
    println!("  best-case (free quant) = (c):   {:.3}× vs f32", a / c);
}

fn main() {
    println!("Encoder GEMM int8 decomposition (32 threads, warm, min-of-20)");
    // fc2 / mlp_proj: the target — big [1500,5120] activation
    probe("fc2 / mlp_proj", 1500, 5120, 1280);
    // fc1 / mlp_fc: the WORKING control (attn_in int8s this, gets a win)
    probe("fc1 / mlp_fc (control)", 1500, 1280, 5120);
    // qkv: another working control
    probe("qkv (control)", 1500, 1280, 1280);
}
