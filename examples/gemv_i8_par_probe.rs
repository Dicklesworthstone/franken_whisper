//! Per-token `gemv_i8` serial-vs-parallel threshold probe (BlackThrush, 2026-07-02).
//!
//! Closes the standing open lead (NEGATIVE_EVIDENCE "Live lead for next cycle,
//! unmeasured"): the int8 decode projections parallelize at `out*inp >= FW_GEMV_I8_PAR`
//! (default `1<<21` = 2.10 M). That keeps the small projections
//! (`self_out`/`cross_q`/`cross_out` = 1.64 M) SERIAL — MEASURED win — but
//! `qkv` (4.9 M) and `mlp_0` (6.5 M) still PARALLELIZE on the ASSUMPTION they
//! "amortize the spawn". That was reasoned, never measured. Each per-token GEMV
//! is ~0.03 ms of real dot compute, so `par_chunks_mut`'s rayon coordination may
//! DOMINATE even at 4.9–6.5 M. Serial vs parallel is BIT-IDENTICAL (disjoint
//! output-row bands, each row's `dot_i8` order unchanged), so if serial is faster
//! it is a clean default-on win across ~35% of decode.
//!
//! Method: time `gemv_i8` min-of-N (min iter = least-contended ⇒ contention-robust
//! on this shared box) at each real turbo decode projection shape. Run TWICE:
//!   FW_GEMV_I8_PAR=999999999  -> all shapes SERIAL
//!   FW_GEMV_I8_PAR=1          -> all shapes PARALLEL
//! then compare per shape. (The threshold is a per-process OnceLock, hence 2 runs.)
//!
//! Usage: `FW_GEMV_I8_PAR=<t> RAYON_NUM_THREADS=32 gemv_i8_par_probe [iters]` (default 200).
use franken_whisper::native_engine::nn;
use half::f16;
use std::hint::black_box;
use std::time::Instant;

fn bench(name: &str, out: usize, inp: usize, iters: usize) {
    let mut s = 0xBEEF_1234u64;
    let mut nextf = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let wf16: Vec<f16> = (0..out * inp).map(|_| f16::from_f32(nextf() * 0.1)).collect();
    let w_i8 = nn::quantize_f16_to_i8(&wf16, out, inp);
    let x: Vec<f32> = (0..inp).map(|_| nextf()).collect();
    let bias: Vec<f32> = (0..out).map(|_| nextf() * 0.1).collect();
    let mut y = nn::gemv_out_buf(out);

    for _ in 0..20 {
        nn::gemv_i8(&w_i8, &x, Some(&bias), &mut y);
        black_box(&y);
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        nn::gemv_i8(&w_i8, &x, Some(&bias), &mut y);
        best = best.min(t.elapsed().as_secs_f64());
        black_box(&y);
    }
    let macs = out * inp;
    println!(
        "  {name:<10} out={out:<6} inp={inp:<5} (MACs={:.1}M)  min={:>7.1} us",
        macs as f64 / 1e6,
        best * 1e6
    );
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let thr = std::env::var("FW_GEMV_I8_PAR").unwrap_or_else(|_| "(default 1<<21)".into());
    println!(
        "=== gemv_i8 per-token min-of-{iters} @ {}t, FW_GEMV_I8_PAR={thr} ===",
        rayon::current_num_threads()
    );
    // Real turbo decode projection shapes (n_state=1280, mlp inner=5120, vocab=51866):
    bench("self_out", 1280, 1280, iters); // 1.64M — currently SERIAL (baseline check)
    bench("qkv", 3840, 1280, iters); // 4.9M  — currently PARALLEL (the question)
    bench("mlp_0", 5120, 1280, iters); // 6.5M  — currently PARALLEL (the question)
    bench("mlp_2", 1280, 5120, iters); // 6.5M  — currently PARALLEL
    bench("logits", 51866, 1280, iters); // 66M   — currently PARALLEL (should stay)
}
