//! Randomized / sketched approximate-GEMM probe for the encoder MLP+QKV GEMMs
//! (land-or-dig cycle, 2026-07-05).
//!
//! The ledger established the encoder GEMM is COMPUTE-bound per core (single-core
//! 64-81% of FMA peak) and only frequency/power-throttled at 32t — so a *genuine*
//! FLOP reduction with *small* overhead helps proportionally. Strassen was the one
//! sub-cubic lever tried and it LOST (0.27-0.34x) because its O(n^2) extraction/
//! assembly overhead dwarfed the 12.5% multiply saving at these shapes.
//!
//! NEW PRIMITIVE (not previously in NEGATIVE_EVIDENCE.md): CountSketch / compressed
//! matrix multiplication (Pagh 2013; randomized numerical linear algebra). Reduce
//! the CONTRACTION dim K -> K' < K with a sparse +-1 sketch S[K,K'] (one nonzero
//! per row). Then h@W ~= (h@S) @ (S^T @ W) = hs[M,K'] @ Ws[K',N]. The weight sketch
//! Ws is STATIC (built once at load, like the existing pretranspose/i7 quant), so
//! the per-window cost is just the cheap O(M*K) activation sketch + the SMALLER
//! O(M*K'*N) GEMM. Unlike Strassen the overhead is O(M*K) (a scatter-add), far below
//! the O(M*K*N) GEMM, so if it wins on SPEED the only question is ACCURACY.
//!
//! (hs@Ws)[m,n] = sum_{k,j : bucket(k)=bucket(j)} s(k)s(j) h[m,k] W[j,n]. The k=j
//! terms give the exact product (s^2=1); k!=j collisions are mean-zero over random
//! signs => UNBIASED estimator, variance ~ (K/K') * ||h_row||^2 ||W_col||^2. We also
//! measure d-way sketch AVERAGING (variance/d at d x cost) to map the speed/accuracy
//! frontier.
//!
//! We report SPEED (vs baseline nn::matmul, weight-sketch load-amortized/untimed)
//! and ACCURACY (max|d|, mean|d|, Frobenius relerr). relerr is the transcript-impact
//! predictor: encoder depth-pruning is FATAL at even 4/32 layers, so any per-GEMM
//! relerr above ~1% almost certainly breaks the (no-slack, distilled-turbo) transcript.
//!
//! Run at RAYON_NUM_THREADS=32 (the encoder's tuned thread count).
//! Usage: `sketched_gemm_probe [iters]` (default 40).
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Deterministic sketch map for contraction dim `k` -> `kp` buckets.
/// Returns (bucket[k] in [0,kp), sign[k] in {-1,+1}) as parallel arrays.
fn sketch_map(k: usize, kp: usize, seed: u64) -> (Vec<u32>, Vec<f32>) {
    let mut s = seed | 1;
    let mut nextu = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s >> 33
    };
    let mut bucket = vec![0u32; k];
    let mut sign = vec![0.0f32; k];
    for i in 0..k {
        bucket[i] = (nextu() as usize % kp) as u32;
        sign[i] = if nextu() & 1 == 0 { 1.0 } else { -1.0 };
    }
    (bucket, sign)
}

/// One-time (load-amortized) weight sketch: Ws[kp,n] = sum_{j:bucket(j)=k'} sign(j)*W[j,n].
/// W is [k,n] row-major.
fn sketch_weight(w: &[f32], k: usize, n: usize, kp: usize, bucket: &[u32], sign: &[f32]) -> Mat {
    let mut ws = vec![0.0f32; kp * n];
    for j in 0..k {
        let b = bucket[j] as usize;
        let sg = sign[j];
        let (src, dst) = (&w[j * n..(j + 1) * n], &mut ws[b * n..(b + 1) * n]);
        for (d, &sv) in dst.iter_mut().zip(src) {
            *d += sg * sv;
        }
    }
    Mat::from_vec(kp, n, ws)
}

/// Per-window (TIMED) activation sketch: hs[m,k'] = sum_{k:bucket(k)=k'} sign(k)*h[m,k].
/// h is [m,k] row-major. Parallel over rows (independent), matching encoder threading.
fn sketch_act(h: &[f32], m: usize, k: usize, kp: usize, bucket: &[u32], sign: &[f32]) -> Mat {
    let mut hs = vec![0.0f32; m * kp];
    hs.par_chunks_mut(kp)
        .zip(h.par_chunks(k))
        .for_each(|(out_row, in_row)| {
            for (kk, &v) in in_row.iter().enumerate() {
                out_row[bucket[kk] as usize] += sign[kk] * v;
            }
        });
    Mat::from_vec(m, kp, hs)
}

fn frob(a: &[f32]) -> f64 {
    a.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

fn bench(name: &str, k: usize, n: usize, seq: usize, ratios: &[f64], davg: &[usize], iters: usize) {
    let mut s = 0x1234_5678u64;
    let mut nextf = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    // W [K,N] row-major (nn::matmul rhs: inner dim K = rows), h [seq,K] row-major.
    let w: Vec<f32> = (0..k * n).map(|_| nextf() * 0.1).collect();
    let w_mat = Mat::from_vec(k, n, w.clone());
    let h: Vec<f32> = (0..seq * k).map(|_| nextf()).collect();
    let h_mat = Mat::from_vec(seq, k, h.clone());
    let gflop = 2.0 * seq as f64 * k as f64 * n as f64 / 1e9;

    println!(
        "\n{name}  [seq={seq}] K={k} N={n}  ({gflop:.2} GFLOP)  best-of-{iters} @ {}t:",
        rayon::current_num_threads()
    );

    // baseline: one ft matmul.
    let reference = nn::matmul(&h_mat, &w_mat).unwrap();
    let ref_frob = frob(&reference.data);
    let mut bbase = f64::INFINITY;
    for _ in 0..3 {
        black_box(nn::matmul(&h_mat, &w_mat).unwrap());
    }
    for _ in 0..iters {
        let t = Instant::now();
        let r = nn::matmul(&h_mat, &w_mat).unwrap();
        bbase = bbase.min(t.elapsed().as_secs_f64());
        black_box(r);
    }
    println!(
        "  baseline 1x ft   : {:>7.3} ms  {:>6.0} GFLOP/s   [current encoder path]",
        bbase * 1e3,
        gflop / bbase
    );

    for &r in ratios {
        let kp = ((k as f64 * r).round() as usize).clamp(1, k);
        for &d in davg {
            // Build d independent sketches (weights sketched once = load-time, untimed).
            let maps: Vec<(Vec<u32>, Vec<f32>)> = (0..d)
                .map(|di| sketch_map(k, kp, 0xA53Fu64 + di as u64 * 2654435761))
                .collect();
            let ws_mats: Vec<Mat> = maps
                .iter()
                .map(|(b, sg)| sketch_weight(&w, k, n, kp, b, sg))
                .collect();

            // Approx (for accuracy): average d sketch products.
            let compute_approx = || {
                let mut acc = vec![0.0f32; seq * n];
                for di in 0..d {
                    let (b, sg) = &maps[di];
                    let hs = sketch_act(&h, seq, k, kp, b, sg);
                    let prod = nn::matmul(&hs, &ws_mats[di]).unwrap();
                    for (a, p) in acc.iter_mut().zip(&prod.data) {
                        *a += *p;
                    }
                }
                if d > 1 {
                    let inv = 1.0 / d as f32;
                    for a in &mut acc {
                        *a *= inv;
                    }
                }
                acc
            };

            let approx = compute_approx();
            let mut maxd = 0.0f64;
            let mut sumd = 0.0f64;
            let mut diff2 = 0.0f64;
            for (a, e) in approx.iter().zip(&reference.data) {
                let dd = (*a as f64 - *e as f64).abs();
                maxd = maxd.max(dd);
                sumd += dd;
                diff2 += dd * dd;
            }
            let meand = sumd / approx.len() as f64;
            let relerr = diff2.sqrt() / ref_frob;

            // Timed sketched path.
            for _ in 0..3 {
                black_box(compute_approx());
            }
            let mut bsk = f64::INFINITY;
            for _ in 0..iters {
                let t = Instant::now();
                let out = compute_approx();
                bsk = bsk.min(t.elapsed().as_secs_f64());
                black_box(out);
            }
            let approx_gflop = 2.0 * seq as f64 * kp as f64 * n as f64 / 1e9 * d as f64;
            println!(
                "  K'={kp:<4} r={r:.2} d={d}: {:>7.3} ms  {:>5.2}x speed  (sketchGF={approx_gflop:.2}) | relerr={:>7.3}% max|d|={:.2e} mean|d|={:.2e}",
                bsk * 1e3,
                bbase / bsk,
                relerr * 100.0,
                maxd,
                meand,
            );
        }
    }
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    println!(
        "=== sketched (CountSketch / compressed) approximate encoder GEMM @ {} threads ===",
        rayon::current_num_threads()
    );
    println!(
        "speed = baseline / sketched (weight sketch load-amortized, untimed); relerr = Frobenius ||approx-exact||/||exact||"
    );
    let ratios = [0.5, 0.75];
    let davg = [1usize, 2, 4];
    // Real turbo encoder shapes (n_state=1280, n_ctx=1500):
    bench("QKV/out proj", 1280, 1280, 1500, &ratios, &davg, iters);
    bench("MLP fc1", 1280, 5120, 1500, &ratios, &davg, iters);
    bench("MLP fc2", 5120, 1280, 1500, &ratios, &davg, iters);
}
