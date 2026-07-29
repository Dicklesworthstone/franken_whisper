//! Product-Quantized (table-lookup) GEMM probe for the encoder GEMMs (land-or-dig, 2026-07-05).
//!
//! The 4th and final major reduced-cost GEMM class after this session's Strassen
//! (935a883-pre, O(n²) overhead), CountSketch (935a883, variance+slower), and
//! low-rank (e2ee176, near-full-rank weights). PQ-GEMM is mechanistically DIFFERENT
//! from all three: it replaces FMA multiplies with CODEBOOK TABLE LOOKUPS.
//!   - Split the contraction `in` into n_sub subspaces of width D. Quantize each
//!     output column's subvector to one of K centroids (codes[out][n_sub], u8).
//!   - Per activation row x: precompute table[s][c] = dot(x_sub_s, centroid[s][c]).
//!   - dot(x, W_col) = sum_s table[s][codes[col][s]]  -> n_sub ADDS + LOOKUPS, no mults.
//! (Jégou PQ / additive-quantization GEMM; the CPU billion-scale-ANN primitive.)
//!
//! The whole bet is a SPEED question with a built-in tension: small K (table L1-
//! resident, fast lookups) is coarse (inaccurate); large K (accurate) blows the L1
//! table (slow lookups). And lookups don't vectorize (AVX2 gather is microcoded on
//! Zen3, per the gelu-gather finding) while the baseline is 16 flops/cycle of FMA.
//! CHEAPEST KILL FIRST (extreme-opt method): if the lookup-sum can't even BEAT FMA,
//! PQ-GEMM is dead on speed and accuracy is moot. We measure lookup-sum speed with
//! random codes/table at real turbo shapes, then (for one shape) the REAL-weight VQ
//! reconstruction error via k-means so both axes are on record.
//!
//! Run at RAYON_NUM_THREADS=32. Needs FRANKEN_WHISPER_MODEL_DIR (for the accuracy leg).
//! Usage: `pq_gemm_probe [layer] [iters]` (default layer=15, iters=20).
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::nn;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// PQ lookup-sum GEMM: out[m,o] = sum_s table_m[s][codes[o*n_sub+s]].
/// table precomputed PER ROW: table_m[s][c] = dot(x[m, s*d ..], centroid[s*K*d + c*d ..]).
/// Parallel over m rows (each row independent). Returns wall time (best-of-iters).
fn pq_time(
    x: &[f32],
    m: usize,
    in_d: usize,
    out_d: usize,
    d: usize,
    k: usize,
    centroids: &[f32],
    codes: &[u8],
    iters: usize,
) -> f64 {
    let n_sub = in_d / d;
    let run = || {
        let mut out = vec![0.0f32; m * out_d];
        out.par_chunks_mut(out_d)
            .zip(x.par_chunks(in_d))
            .for_each(|(orow, xrow)| {
                // per-row table [n_sub][k]
                let mut table = vec![0.0f32; n_sub * k];
                for s in 0..n_sub {
                    let xs = &xrow[s * d..s * d + d];
                    let cbase = s * k * d;
                    for c in 0..k {
                        let cen = &centroids[cbase + c * d..cbase + c * d + d];
                        let mut acc = 0.0f32;
                        for j in 0..d {
                            acc += xs[j] * cen[j];
                        }
                        table[s * k + c] = acc;
                    }
                }
                // lookup-sum over outputs
                for o in 0..out_d {
                    let cb = o * n_sub;
                    let mut acc = 0.0f32;
                    for s in 0..n_sub {
                        acc += table[s * k + codes[cb + s] as usize];
                    }
                    orow[o] = acc;
                }
            });
        out
    };
    for _ in 0..2 {
        black_box(run());
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let r = run();
        best = best.min(t.elapsed().as_secs_f64());
        black_box(r);
    }
    best
}

fn frob(v: &[f32]) -> f64 {
    v.iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()
}

/// Real-weight VQ accuracy: per-subspace k-means over the out columns' subvectors,
/// return reconstruction Frobenius relerr. wt is [in,out] row-major.
fn vq_relerr(wt: &[f32], in_d: usize, out_d: usize, d: usize, k: usize, iters: usize) -> f64 {
    let n_sub = in_d / d;
    let mut err2 = 0.0f64;
    let wt_frob = frob(wt);
    // process each subspace independently
    for s in 0..n_sub {
        // gather out subvectors [out_d][d] for subspace s
        let mut pts = vec![0.0f32; out_d * d];
        for o in 0..out_d {
            for j in 0..d {
                pts[o * d + j] = wt[(s * d + j) * out_d + o];
            }
        }
        // init centroids: first k points (deterministic)
        let mut cen = vec![0.0f32; k * d];
        for c in 0..k {
            cen[c * d..c * d + d]
                .copy_from_slice(&pts[(c * (out_d / k)) * d..(c * (out_d / k)) * d + d]);
        }
        let mut assign = vec![0usize; out_d];
        for _ in 0..iters {
            // assign
            for o in 0..out_d {
                let p = &pts[o * d..o * d + d];
                let (mut bi, mut bd) = (0usize, f32::INFINITY);
                for c in 0..k {
                    let cc = &cen[c * d..c * d + d];
                    let mut dd = 0.0f32;
                    for j in 0..d {
                        let e = p[j] - cc[j];
                        dd += e * e;
                    }
                    if dd < bd {
                        bd = dd;
                        bi = c;
                    }
                }
                assign[o] = bi;
            }
            // update
            cen.iter_mut().for_each(|v| *v = 0.0);
            let mut cnt = vec![0u32; k];
            for o in 0..out_d {
                let c = assign[o];
                cnt[c] += 1;
                for j in 0..d {
                    cen[c * d + j] += pts[o * d + j];
                }
            }
            for c in 0..k {
                if cnt[c] > 0 {
                    let inv = 1.0 / cnt[c] as f32;
                    for j in 0..d {
                        cen[c * d + j] *= inv;
                    }
                }
            }
        }
        // reconstruction error for this subspace
        for o in 0..out_d {
            let c = assign[o];
            for j in 0..d {
                let e = pts[o * d + j] - cen[c * d + j];
                err2 += (e as f64) * (e as f64);
            }
        }
    }
    err2.sqrt() / wt_frob
}

fn load_wt(model: &GgmlModel, name: &str) -> (Vec<f32>, usize, usize) {
    let (shape, data) = model.tensor_f32(name).expect("tensor");
    let (out_d, in_d) = (shape[0], shape[1]);
    let mut wt = vec![0.0f32; in_d * out_d];
    for o in 0..out_d {
        for i in 0..in_d {
            wt[i * out_d + o] = data[o * in_d + i];
        }
    }
    (wt, in_d, out_d)
}

fn bench(
    model: &GgmlModel,
    label: &str,
    name: &str,
    seq: usize,
    configs: &[(usize, usize)],
    iters: usize,
) {
    let (wt, in_d, out_d) = load_wt(model, name);
    let wt_mat = Mat::from_vec(in_d, out_d, wt.clone());
    let gflop = 2.0 * seq as f64 * in_d as f64 * out_d as f64 / 1e9;
    let mut s = 0xC0FFEEu64;
    let mut af = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let x: Vec<f32> = (0..seq * in_d).map(|_| af()).collect();
    let x_mat = Mat::from_vec(seq, in_d, x.clone());

    let mut bbase = f64::INFINITY;
    for _ in 0..3 {
        black_box(nn::matmul(&x_mat, &wt_mat).unwrap());
    }
    for _ in 0..iters {
        let t = Instant::now();
        black_box(nn::matmul(&x_mat, &wt_mat).unwrap());
        bbase = bbase.min(t.elapsed().as_secs_f64());
    }
    println!(
        "\n{label}  W_t[{in_d},{out_d}] seq={seq} ({gflop:.2} GFLOP)  FMA baseline {:.3} ms {:.0} GF/s @ {}t",
        bbase * 1e3,
        gflop / bbase,
        rayon::current_num_threads()
    );
    for &(d, k) in configs {
        if in_d % d != 0 {
            println!("  D={d} K={k}: skip (in%D!=0)");
            continue;
        }
        let n_sub = in_d / d;
        // random centroids + codes for the SPEED leg (speed is data-independent).
        let centroids: Vec<f32> = (0..n_sub * k * d).map(|_| af()).collect();
        let codes: Vec<u8> = (0..out_d * n_sub)
            .map(|_| (((af() + 0.5) * k as f32) as usize % k) as u8)
            .collect();
        let tpq = pq_time(&x, seq, in_d, out_d, d, k, &centroids, &codes, iters);
        let tbl_kb = (n_sub * k * 4) as f64 / 1024.0;
        // real-weight VQ accuracy (k-means, 6 iters) — one measure per config.
        let relerr = vq_relerr(&wt, in_d, out_d, d, k, 6);
        // effective bits/weight = log2(K)/D * 8 ... report as codebook bytes/col + relerr.
        let bits_per_w = (k as f64).log2() / d as f64;
        println!(
            "  D={d:<2} K={k:<4}: {:.3} ms  {:.2}x speed (per-row table {:.0} KB) | VQ relerr={:.2}%  ({:.2} bits/weight)",
            tpq * 1e3,
            bbase / tpq,
            tbl_kb,
            relerr * 100.0,
            bits_per_w
        );
    }
}

fn main() {
    let layer: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let path = find_model_file("large-v3-turbo").expect("set FRANKEN_WHISPER_MODEL_DIR");
    let model = GgmlModel::load(&path).expect("load large-v3-turbo");
    println!(
        "=== PQ (table-lookup) GEMM vs FMA, encoder layer {layer}, real turbo weights @ {}t ===",
        rayon::current_num_threads()
    );
    println!(
        "speed = FMA / PQ (higher=better); VQ relerr = real-weight k-means reconstruction; L1 data cache = 32 KB"
    );
    let cfgs = [(4usize, 16usize), (8, 16), (8, 256), (4, 256)];
    let p = |s: &str| format!("encoder.blocks.{layer}.{s}");
    bench(
        &model,
        "attn.query",
        &p("attn.query.weight"),
        1500,
        &cfgs,
        iters,
    );
    bench(&model, "mlp.0 fc1 ", &p("mlp.0.weight"), 1500, &cfgs, iters);
    bench(&model, "mlp.2 fc2 ", &p("mlp.2.weight"), 1500, &cfgs, iters);
}
