//! Cross-K/V per-head build (transpose+gather+f16) granularity probe
//! (BlackThrush, 2026-07-02).
//!
//! decoder.rs:892-924 builds the window-constant per-head cross K/V buffers
//! (cross_kh_t [d_head,enc], cross_kh_f16 [enc,d_head], cross_vh [enc,d_head],
//! cross_vh_f16 [d_head,enc]) in a SERIAL nested loop over n_layer×n_head pairs
//! (turbo 4×20 = 80), each doing 4 buffer builds with f16 conversions over
//! enc×d_head = 96K elements. This probe MEASURES parallelizing the OUTER loop
//! over the 80 independent (layer,head) pairs (the granularity-flip that landed
//! 1.87× on the sibling quantize, b22f8ae), and proves the parallel build is
//! byte-identical to the serial one.
//!
//! Usage: `cross_build_probe [iters]`  (default 60).
use franken_whisper::native_engine::Mat;
use half::f16;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

type Built = (Vec<f32>, Vec<f16>, Vec<f32>, Vec<f16>); // (kh_t, k_nat, vh, v_t)

// Build one (layer,head)'s four buffers — the exact arithmetic/layout of
// decoder.rs:895-923.
fn build_head(ck: &Mat, cv: &Mat, base: usize, d_head: usize, enc: usize) -> Built {
    let mut kh_t = vec![0.0f32; d_head * enc];
    let mut k_nat = Vec::<f16>::with_capacity(enc * d_head);
    for j in 0..enc {
        let src = &ck.row(j)[base..base + d_head];
        for (d, &s) in src.iter().enumerate() {
            kh_t[d * enc + j] = s;
            k_nat.push(f16::from_f32(s));
        }
    }
    let mut vh = vec![0.0f32; enc * d_head];
    let mut v_t = vec![f16::from_bits(0); d_head * enc];
    for j in 0..enc {
        let src = &cv.row(j)[base..base + d_head];
        vh[j * d_head..(j + 1) * d_head].copy_from_slice(src);
        for (d, &s) in src.iter().enumerate() {
            v_t[d * enc + j] = f16::from_f32(s);
        }
    }
    (kh_t, k_nat, vh, v_t)
}

fn fill(rows: usize, cols: usize, seed: u64) -> Mat {
    let mut s = seed;
    let data = (0..rows * cols)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    Mat::from_vec(rows, cols, data)
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let (n_layer, n_head, d_head, enc) = (4usize, 20usize, 64usize, 1500usize);
    let n_state = n_head * d_head;
    let cross_k: Vec<Mat> = (0..n_layer).map(|l| fill(enc, n_state, 0x10 + l as u64)).collect();
    let cross_v: Vec<Mat> = (0..n_layer).map(|l| fill(enc, n_state, 0x90 + l as u64)).collect();
    let npairs = n_layer * n_head;
    println!("turbo cross build: {npairs} (layer,head) pairs, each 4 bufs over {}×{}", enc, d_head);

    // A) serial nested loop (current decoder.rs).
    let serial = || {
        let mut out: Vec<Built> = Vec::with_capacity(npairs);
        for li in 0..n_layer {
            for h in 0..n_head {
                out.push(build_head(&cross_k[li], &cross_v[li], h * d_head, d_head, enc));
            }
        }
        out
    };
    // B) par outer over the npairs independent pairs, collected in order.
    let parallel = || {
        (0..npairs)
            .into_par_iter()
            .map(|idx| {
                let (li, h) = (idx / n_head, idx % n_head);
                build_head(&cross_k[li], &cross_v[li], h * d_head, d_head, enc)
            })
            .collect::<Vec<Built>>()
    };

    // Byte-exact: parallel == serial for all four buffers of every pair.
    let a0 = serial();
    let b0 = parallel();
    let mut ok = a0.len() == b0.len();
    for (a, b) in a0.iter().zip(&b0) {
        ok &= a.0 == b.0 && a.1 == b.1 && a.2 == b.2 && a.3 == b.3;
    }
    println!("byte-exact: parallel == serial  -> {ok}");

    for _ in 0..3 { black_box(serial()); black_box(parallel()); }

    let mut best_s = f64::INFINITY;
    for _ in 0..iters { let t = Instant::now(); let r = serial(); best_s = best_s.min(t.elapsed().as_secs_f64()); black_box(r); }
    let mut best_p = f64::INFINITY;
    for _ in 0..iters { let t = Instant::now(); let r = parallel(); best_p = best_p.min(t.elapsed().as_secs_f64()); black_box(r); }

    println!("best-of-{iters}:");
    println!("  A) serial nested : {:.3} ms", best_s * 1e3);
    println!("  B) par outer     : {:.3} ms   ({:.2}× vs A)", best_p * 1e3, best_s / best_p);
}
