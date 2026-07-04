//! Logits-GEMV worker-cap sweep (BlackThrush, 2026-07-02).
//!
//! The decode bottleneck is the tied-logits GEMV: int8 [n_vocab, n_state] weight
//! streamed from DRAM every token (bandwidth-bound). `gemv_i8` caps workers at
//! `wide_gemv_cap()` = 32 (`FW_WIDE_GEMV_CAP`), tuned "48/64 regress" on a ~4-CCD
//! box. This box is 64c / likely 8 CCDs, where more workers may reach more memory
//! channels. The band split is order-preserving ⇒ BYTE-IDENTICAL for any cap, so
//! the fastest cap is a free pick. Run this under several FW_WIDE_GEMV_CAP values
//! (the cap is read once via OnceLock, so one value per process):
//!   for c in 8 16 24 32 48 64; do FW_WIDE_GEMV_CAP=$c logits_cap_probe; done
//!
//! Usage: `logits_cap_probe [iters]`  (default 300).
use franken_whisper::native_engine::nn;
use half::f16;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let (n_vocab, n_state) = (51866usize, 1280usize); // turbo tied-logits shape
    // Synthetic f16 weight → int8 (the exact layout logits_last consumes).
    let mut s = 0x1234u64;
    let wf16: Vec<f16> = (0..n_vocab * n_state)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
        })
        .collect();
    let w_i8 = nn::quantize_f16_to_i8(&wf16, n_vocab, n_state);
    let x: Vec<f32> = (0..n_state).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut y = nn::gemv_out_buf(n_vocab);

    let cap = std::env::var("FW_WIDE_GEMV_CAP").unwrap_or_else(|_| "default(32)".into());
    let bytes = (n_vocab * n_state) as f64; // int8 weight bytes streamed per call

    for _ in 0..5 {
        nn::gemv_i8(&w_i8, &x, None, &mut y);
        black_box(&y);
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        nn::gemv_i8(&w_i8, &x, None, &mut y);
        let dt = t.elapsed().as_secs_f64();
        black_box(&y);
        best = best.min(dt);
    }
    println!(
        "FW_WIDE_GEMV_CAP={cap:<12} logits gemv_i8 [{n_vocab},{n_state}]: best {:.3} ms  {:.1} GB/s",
        best * 1e3,
        bytes / best / 1e9
    );
}
