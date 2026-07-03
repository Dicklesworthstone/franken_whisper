//! Encoder GEMM column-panel cache-blocking probe (BlackThrush, 2026-07-02).
//!
//! ft_kernel_cpu's sgemm splits the output over ROW blocks (`m.div_ceil(threads)`)
//! and runs single-threaded `matrixmultiply` on each; EVERY core therefore reads
//! the FULL weight `B` ([K,N]). For the encoder MLP GEMMs `B` is ~26 MB
//! ([1280,5120] / [5120,1280]) — far bigger than any single CCD's L3 — so on a
//! multi-CCD box the 1-D row split streams `B` from DRAM on all cores at once.
//! That is the measured "37% multi-thread scaling wall" (single-core is 64-81% of
//! peak = compute-bound; the wall only appears at 32 threads = bandwidth).
//!
//! LEVER (graveyard 9.6 communication-avoiding / 7.2 cache-oblivious): split the
//! GEMM into P column panels `B[:, panel]` ([K, N/P]). Each panel's `B`-slice is
//! 1/P the size, so for P large enough it becomes L3-resident and the DRAM stream
//! turns into a cache read. Column panels are disjoint output columns → the
//! k-accumulation for each `out[i,j]` is unchanged ⇒ BIT-IDENTICAL.
//!
//! The catch (cf. the rejected encoder QKV-fusion): each ft call writes a
//! CONTIGUOUS [1500, N/P]; assembling the [1500, N] output needs a strided
//! scatter (~mn copy). We measure THREE things so the trade-off is explicit:
//!   (a) baseline  : one ft matmul  [1500,K]@[K,N]              [current path]
//!   (b) panel raw : P ft matmuls over pre-packed [K,N/P] panels, NO scatter
//!                   (= the ceiling if the consumer read panel-major)
//!   (c) panel+scat: (b) + scatter each panel into the [1500,N] output
//!                   (= the deployable cost with a row-major consumer)
//! (c) is verified BIT-EXACT vs (a). Weights are pre-packed once (weights are
//! static in the real model, so packing is a load-time cost, not per-window).
//!
//! Run at RAYON_NUM_THREADS=32 (the encoder's tuned thread count).
//! Usage: `enc_gemm_colpanel_probe [iters]` (default 40).
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;
use std::hint::black_box;
use std::time::Instant;

/// Pre-pack column panel `p` of a [K, N] row-major weight into a contiguous
/// [K, N/P] buffer (one-time, load-amortized in the real model).
fn pack_panel(wt: &[f32], k: usize, n: usize, j0: usize, pw: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; k * pw];
    for row in 0..k {
        out[row * pw..(row + 1) * pw].copy_from_slice(&wt[row * n + j0..row * n + j0 + pw]);
    }
    out
}

fn bench(name: &str, k: usize, n: usize, seq: usize, panels: &[usize], iters: usize) {
    let mut s = 0x1234_5678u64;
    let mut nextf = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    // wt [K, N] row-major (as nn::matmul's rhs wants: inner dim K = rows).
    let wt: Vec<f32> = (0..k * n).map(|_| nextf() * 0.1).collect();
    let wt_mat = Mat::from_vec(k, n, wt.clone());
    let h: Vec<f32> = (0..seq * k).map(|_| nextf()).collect();
    let h_mat = Mat::from_vec(seq, k, h.clone());
    let bmb = (k * n * 4) as f64 / 1e6;
    let gflop = 2.0 * seq as f64 * k as f64 * n as f64 / 1e9;

    println!(
        "\n{name}  [seq={seq}] K={k} N={n}  (B={bmb:.1} MB, {gflop:.2} GFLOP)  best-of-{iters} @ {}t:",
        rayon::current_num_threads()
    );

    // (a) baseline: one ft matmul.
    let mut bbase = f64::INFINITY;
    let reference = nn::matmul(&h_mat, &wt_mat).unwrap();
    for _ in 0..3 {
        black_box(nn::matmul(&h_mat, &wt_mat).unwrap());
    }
    for _ in 0..iters {
        let t = Instant::now();
        let r = nn::matmul(&h_mat, &wt_mat).unwrap();
        bbase = bbase.min(t.elapsed().as_secs_f64());
        black_box(r);
    }
    println!(
        "  (a) baseline 1x ft   : {:>7.3} ms  {:>6.0} GFLOP/s   [current encoder path]",
        bbase * 1e3,
        gflop / bbase
    );

    for &p in panels {
        if n % p != 0 {
            println!("  P={p}: skip (N={n} not divisible)");
            continue;
        }
        let pw = n / p;
        // Pre-pack panels (load-time in the real model — not timed in the loop).
        let packed: Vec<Vec<f32>> = (0..p)
            .map(|pi| pack_panel(&wt, k, n, pi * pw, pw))
            .collect();
        let panel_mats: Vec<Mat> = packed.iter().map(|pk| Mat::from_vec(k, pw, pk.clone())).collect();

        // (b) panel raw: P ft matmuls, keep panel outputs separate (no scatter).
        let mut braw = f64::INFINITY;
        for _ in 0..3 {
            let outs: Vec<Mat> = panel_mats.iter().map(|pm| nn::matmul(&h_mat, pm).unwrap()).collect();
            black_box(outs);
        }
        for _ in 0..iters {
            let t = Instant::now();
            let outs: Vec<Mat> = panel_mats.iter().map(|pm| nn::matmul(&h_mat, pm).unwrap()).collect();
            braw = braw.min(t.elapsed().as_secs_f64());
            black_box(outs);
        }

        // (c) panel + scatter into a [seq, N] row-major output.
        let mut bscat = f64::INFINITY;
        let mut assembled = vec![0.0f32; seq * n];
        let scatter = |outs: &[Mat], dst: &mut [f32]| {
            for (pi, om) in outs.iter().enumerate() {
                let j0 = pi * pw;
                for i in 0..seq {
                    dst[i * n + j0..i * n + j0 + pw]
                        .copy_from_slice(&om.data[i * pw..(i + 1) * pw]);
                }
            }
        };
        for _ in 0..iters {
            let t = Instant::now();
            let outs: Vec<Mat> = panel_mats.iter().map(|pm| nn::matmul(&h_mat, pm).unwrap()).collect();
            scatter(&outs, &mut assembled);
            bscat = bscat.min(t.elapsed().as_secs_f64());
            black_box(&assembled);
        }
        // Bit-exactness of the assembled (scattered) output vs the baseline.
        let maxd = assembled
            .iter()
            .zip(reference.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!(
            "  P={p:<2} raw {:>7.3} ms {:>6.0} GF/s {:.2}x | +scatter {:>7.3} ms {:>6.0} GF/s {:.2}x | max|d|={:.1e}{}",
            braw * 1e3,
            gflop / braw,
            bbase / braw,
            bscat * 1e3,
            gflop / bscat,
            bbase / bscat,
            maxd,
            if maxd == 0.0 { " OK" } else { " !!" }
        );
    }
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    println!(
        "=== encoder GEMM column-panel cache-blocking @ {} threads ===",
        rayon::current_num_threads()
    );
    println!("raw = P panels no scatter (ceiling) | +scatter = deployable (row-major consumer)");
    // Real turbo encoder shapes (n_state=1280, n_ctx=1500):
    bench("QKV/out proj", 1280, 1280, 1500, &[2, 4, 8], iters);
    bench("MLP fc1", 1280, 5120, 1500, &[2, 4, 8, 16], iters);
    bench("MLP fc2", 5120, 1280, 1500, &[2, 4, 8], iters);
}
