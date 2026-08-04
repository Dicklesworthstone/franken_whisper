//! Cross-K/V projection GEMM scheduling probe (BlackThrush, 2026-07-02).
//!
//! The per-window cross K/V projections (encoder_out @ Wk, @ Wv for each decoder
//! layer) run in decoder.rs via `thread::scope` band-concurrency: n_layer bands
//! run CONCURRENTLY, each calling nn::matmul → the dep sgemm, which is ITSELF
//! rayon-parallel over the global pool. This probe MEASURES whether that outer
//! concurrency actually helps at turbo shapes (enc_out [1500,1280] × [1280,1280],
//! 4 layers × {K,V} = 8 GEMMs) or just contends with the already-saturating
//! sgemm — i.e. whether plain SEQUENTIAL (each GEMM full-8-core) is equal/faster.
//! The GEMM outputs are byte-identical either way (deterministic sgemm), so the
//! faster schedule is a free pick.
//!
//! Usage: `cross_gemm_probe [iters]`  (default 50).
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;
use std::hint::black_box;
use std::thread;
use std::time::Instant;

fn fill(m: usize, k: usize, seed: u64) -> Mat {
    let mut s = seed;
    let data = (0..m * k)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    Mat::from_vec(m, k, data)
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let (n_layer, enc, n_state) = (4usize, 1500usize, 1280usize);
    let encoder_out = fill(enc, n_state, 0x1);
    // Pre-transposed [in,out] weights, as nn::matmul consumes.
    let wk: Vec<Mat> = (0..n_layer)
        .map(|l| fill(n_state, n_state, 0x100 + l as u64))
        .collect();
    let wv: Vec<Mat> = (0..n_layer)
        .map(|l| fill(n_state, n_state, 0x200 + l as u64))
        .collect();
    println!("turbo cross proj: enc[{enc},{n_state}] × {n_layer}×2 [{n_state},{n_state}] GEMMs");

    let seq = || {
        let mut out = Vec::with_capacity(n_layer * 2);
        for li in 0..n_layer {
            out.push(nn::matmul(&encoder_out, &wk[li]).unwrap());
            out.push(nn::matmul(&encoder_out, &wv[li]).unwrap());
        }
        out
    };
    let band = || {
        let (eo, wkr, wvr) = (&encoder_out, &wk, &wv);
        thread::scope(|s| {
            let handles: Vec<_> = (0..n_layer)
                .map(|li| {
                    s.spawn(move || {
                        let k = nn::matmul(eo, &wkr[li]).unwrap();
                        let v = nn::matmul(eo, &wvr[li]).unwrap();
                        (k, v)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        })
    };

    // Byte-exact sanity: same K/V values regardless of schedule.
    let a = seq();
    let b = band();
    let mut ok = true;
    for li in 0..n_layer {
        ok &= a[li * 2].data == b[li].0.data && a[li * 2 + 1].data == b[li].1.data;
    }
    println!("byte-exact: seq == band -> {ok}");

    for _ in 0..3 {
        black_box(seq());
        black_box(band());
    }

    let mut best_seq = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let r = seq();
        best_seq = best_seq.min(t.elapsed().as_secs_f64());
        black_box(r);
    }
    let mut best_band = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let r = band();
        best_band = best_band.min(t.elapsed().as_secs_f64());
        black_box(r);
    }

    println!("best-of-{iters}:");
    println!("  band-concurrency (current) : {:.3} ms", best_band * 1e3);
    println!(
        "  sequential (full-parallel) : {:.3} ms   ({:.2}× vs band)",
        best_seq * 1e3,
        best_band / best_seq
    );
}
