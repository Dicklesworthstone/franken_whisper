//! `gemv_i8_batch` correctness + speed probe (BlackThrush, 2026-07-02).
//!
//! Verifies the new batched int8 GEMV is (a) BIT-IDENTICAL to running the batch
//! as `tq` separate per-token `gemv_i8` calls (the byte-exactness claim), and
//! (b) faster than the current `tq>1` path `gemv_f16_batch` (it reads HALF the
//! weight bytes). Shape = the turbo fused-QKV projection `[3*1280, 1280]` at a
//! representative prefill batch `tq=4`. Run at `RAYON_NUM_THREADS=32`.
//! Usage: `gemv_i8_batch_probe [iters]` (default 300).
use franken_whisper::native_engine::nn;
use half::f16;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    let (out, inp) = (3 * 1280usize, 1280usize); // turbo fused QKV
    let tq = 4usize;

    // Synthetic f16 weight + f32 activation batch.
    let mut s = 0x51EDu64;
    let mut nextf = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let wf16: Vec<f16> = (0..out * inp).map(|_| f16::from_f32(nextf())).collect();
    let x: Vec<f32> = (0..tq * inp).map(|_| nextf() * 0.7).collect();
    let bias: Vec<f32> = (0..out).map(|_| nextf() * 0.1).collect();

    let w_i8 = nn::quantize_f16_to_i8(&wf16, out, inp);

    // (a) Correctness: batch output must equal per-token gemv_i8 for every row.
    let mut y_batch = nn::gemv_out_buf(tq * out);
    nn::gemv_i8_batch(&w_i8, &x, tq, Some(&bias), &mut y_batch);
    let mut max_abs_diff = 0.0f32;
    let mut mismatches = 0usize;
    for t in 0..tq {
        let mut y_row = nn::gemv_out_buf(out);
        nn::gemv_i8(&w_i8, &x[t * inp..(t + 1) * inp], Some(&bias), &mut y_row);
        for o in 0..out {
            let d = (y_batch[t * out + o] - y_row[o]).abs();
            if d != 0.0 {
                mismatches += 1;
                max_abs_diff = max_abs_diff.max(d);
            }
        }
    }
    println!(
        "correctness: {} / {} entries differ from per-token gemv_i8 (max|Δ|={:.3e}) => {}",
        mismatches,
        tq * out,
        max_abs_diff,
        if mismatches == 0 { "BIT-IDENTICAL ✓" } else { "MISMATCH ✗" }
    );

    // (b) Speed: int8 batch vs f16 batch (same weight), best-of-iters (min).
    let mut y_i8 = nn::gemv_out_buf(tq * out);
    let mut y_f16 = nn::gemv_out_buf(tq * out);
    for _ in 0..10 {
        nn::gemv_i8_batch(&w_i8, &x, tq, Some(&bias), &mut y_i8);
        nn::gemv_f16_batch(&wf16, out, inp, &x, tq, Some(&bias), &mut y_f16);
        black_box((&y_i8, &y_f16));
    }
    let (mut b_i8, mut b_f16) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..iters {
        let t = Instant::now();
        nn::gemv_i8_batch(&w_i8, &x, tq, Some(&bias), &mut y_i8);
        b_i8 = b_i8.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        nn::gemv_f16_batch(&wf16, out, inp, &x, tq, Some(&bias), &mut y_f16);
        b_f16 = b_f16.min(t.elapsed().as_secs_f64());
        black_box((&y_i8, &y_f16));
    }
    let bytes_i8 = (out * inp) as f64;
    let bytes_f16 = (out * inp * 2) as f64;
    println!(
        "QKV [{out},{inp}] tq={tq} best-of-{iters} @ {} threads:",
        rayon::current_num_threads()
    );
    println!("  f16 batch: {:.4} ms  ({:.1} GB/s weight)", b_f16 * 1e3, bytes_f16 / b_f16 / 1e9);
    println!(
        "  int8 batch: {:.4} ms  ({:.1} GB/s weight)  speedup {:.2}x",
        b_i8 * 1e3,
        bytes_i8 / b_i8 / 1e9,
        b_f16 / b_i8
    );
}
