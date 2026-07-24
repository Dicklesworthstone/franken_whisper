//! Low-rank WEIGHT factorization probe for the encoder GEMMs (land-or-dig, 2026-07-05).
//!
//! DISTINCT from the ledger's dismissed "low-rank ATTENTION" (a data-dependent
//! softmax(QK^T) approximation that breaks the faithful port). Here we factor the
//! STATIC linear weight `W_t[in,out] ~= U[in,r] V[r,out]` ONCE at load (like the
//! existing pretranspose / i7 quant), turning the per-window `x@W_t` into TWO clean
//! DENSE GEMMs `(x@U)@V`. This sidesteps the wall that killed last round's CountSketch
//! probe: that lost on the memory-bound O(M*K) scatter-add, not the GEMM. Low-rank has
//! NO scatter — both factors are dense matmuls the tuned `ft_kernel_cpu` runs fast — and
//! its error is DETERMINISTIC (Eckart-Young optimal truncation), not sketch variance.
//!
//! The whole bet rides on whether the REAL distilled-turbo weights are low-rank, so we
//! load actual model tensors (NOT random — a random matrix is full-rank and would make
//! this trivially fail). We estimate the rank-r captured energy via randomized range
//! finding (Halko-Martinsson-Tropp) with 1 power iteration:
//!   Y = W_t (W_t^T (W_t Omega));  Q = orth(Y);  V = Q^T W_t;  U = Q
//!   captured = ||V||_F / ||W_t||_F;  weight relerr = sqrt(1 - captured^2).
//! We ALSO measure the applied OUTPUT relerr on a real-sized activation x[1500,in] and
//! the two-GEMM SPEED vs the baseline single matmul. FLOP cut = K*N / (r*(K+N)).
//!
//! Viability gate: encoder depth-pruning is FATAL at even 4/32 layers
//! (project_encoder_flop_reduction_mapped) -> the no-slack transcript tolerates <<1%
//! per-GEMM error, so we need captured energy ~>99.5% AT a rank r small enough to also
//! be FASTER. Reports let both be judged.
//!
//! Run at RAYON_NUM_THREADS=32. Needs FRANKEN_WHISPER_MODEL_DIR.
//! Usage: `lowrank_gemm_probe [layer] [iters]`  (default layer=15, iters=30).
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::nn;
use std::hint::black_box;
use std::time::Instant;

/// Load ggml linear weight `name` (shape [out,in]) and return its transpose
/// `W_t[in,out]` as a Mat (the layout the encoder GEMM consumes), plus (in,out).
fn load_wt(model: &GgmlModel, name: &str) -> (Mat, usize, usize) {
    let (shape, data) = model.tensor_f32(name).expect("tensor");
    assert_eq!(shape.len(), 2, "{name} not 2-D: {shape:?}");
    let (out_d, in_d) = (shape[0], shape[1]);
    // transpose [out,in] -> [in,out]
    let mut wt = vec![0.0f32; in_d * out_d];
    for o in 0..out_d {
        for i in 0..in_d {
            wt[i * out_d + o] = data[o * in_d + i];
        }
    }
    (Mat::from_vec(in_d, out_d, wt), in_d, out_d)
}

fn frob(v: &[f32]) -> f64 {
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

/// Extract column `j` (length `rows`) of a row-major [rows,cols] matrix.
fn col(m: &[f32], rows: usize, cols: usize, j: usize) -> Vec<f32> {
    (0..rows).map(|i| m[i * cols + j]).collect()
}

/// Randomized range finding: return U=Q [in,r] (Mat) and captured energy fraction.
fn range_find(wt: &Mat, wt_t_as_ggml: &Mat, r: usize, seed: u64) -> (Mat, f64) {
    let (ind, outd) = (wt.rows, wt.cols);
    // Omega [out, r] random +-1.
    let mut s = seed | 1;
    let mut nextf = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if (s >> 40) & 1 == 0 { 1.0f32 } else { -1.0f32 }
    };
    let omega = Mat::from_vec(outd, r, (0..outd * r).map(|_| nextf()).collect());
    // Y = W_t (W_t^T (W_t Omega)) — 1 power iteration. W_t^T == wt_t_as_ggml [out,in].
    let y0 = nn::matmul(wt, &omega).unwrap(); // [in,r]
    let z = nn::matmul(wt_t_as_ggml, &y0).unwrap(); // [out,in]@[in,r]=[out,r]
    let y = nn::matmul(wt, &z).unwrap(); // [in,out]@[out,r]=[in,r]
    // Modified Gram-Schmidt over the r columns of Y[in,r].
    let mut qcols: Vec<Vec<f32>> = Vec::with_capacity(r);
    for j in 0..r {
        let mut v = col(&y.data, ind, r, j);
        for q in &qcols {
            let dot: f64 = v.iter().zip(q).map(|(&a, &b)| a as f64 * b as f64).sum();
            let d = dot as f32;
            for (vi, &qi) in v.iter_mut().zip(q) {
                *vi -= d * qi;
            }
        }
        let nrm = frob(&v);
        if nrm > 1e-6 {
            let inv = (1.0 / nrm) as f32;
            for vi in v.iter_mut() {
                *vi *= inv;
            }
            qcols.push(v);
        }
    }
    let rk = qcols.len();
    // Build Q [in, rk] row-major and Q^T [rk, in].
    let mut q_rowmajor = vec![0.0f32; ind * rk];
    let mut qt = vec![0.0f32; rk * ind];
    for (j, qc) in qcols.iter().enumerate() {
        for i in 0..ind {
            q_rowmajor[i * rk + j] = qc[i];
            qt[j * ind + i] = qc[i];
        }
    }
    let qt_mat = Mat::from_vec(rk, ind, qt);
    let v_mat = nn::matmul(&qt_mat, wt).unwrap(); // [rk,out] = Q^T W_t
    let captured = frob(&v_mat.data) / frob(&wt.data);
    (Mat::from_vec(ind, rk, q_rowmajor), captured)
}

fn bench_weight(model: &GgmlModel, label: &str, name: &str, ranks: &[usize], seq: usize, iters: usize) {
    let (wt, ind, outd) = load_wt(model, name);
    // W_t^T in [out,in] layout for the power iteration = re-load ggml raw (row-major [out,in]).
    let (shape, data) = model.tensor_f32(name).unwrap();
    let wt_t = Mat::from_vec(shape[0], shape[1], data); // [out,in]
    let base_flop = 2.0 * seq as f64 * ind as f64 * outd as f64 / 1e9;

    // Activation x[seq,in] (random, representative for GEMM shape/speed).
    let mut s = 0xBEEFu64;
    let mut af = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let x = Mat::from_vec(seq, ind, (0..seq * ind).map(|_| af()).collect());
    let exact = nn::matmul(&x, &wt).unwrap();
    let exact_frob = frob(&exact.data);

    // baseline speed.
    let mut bbase = f64::INFINITY;
    for _ in 0..3 { black_box(nn::matmul(&x, &wt).unwrap()); }
    for _ in 0..iters {
        let t = Instant::now();
        black_box(nn::matmul(&x, &wt).unwrap());
        bbase = bbase.min(t.elapsed().as_secs_f64());
    }
    println!(
        "\n{label}  W_t[{ind},{outd}] seq={seq}  ({base_flop:.2} GFLOP)  baseline {:.3} ms {:.0} GF/s @ {}t",
        bbase * 1e3, base_flop / bbase, rayon::current_num_threads()
    );
    for &r in ranks {
        if r >= ind.min(outd) { println!("  r={r}: skip (>= min dim)"); continue; }
        let (u, captured) = range_find(&wt, &wt_t, r, 0x51A7 + r as u64);
        let rk = u.cols;
        let v = nn::matmul(&{
            // V = U^T W_t = [r,in]@[in,out]; build U^T.
            let mut ut = vec![0.0f32; rk * ind];
            for i in 0..ind { for j in 0..rk { ut[j * ind + i] = u.data[i * rk + j]; } }
            Mat::from_vec(rk, ind, ut)
        }, &wt).unwrap();
        // applied output accuracy: (x@U)@V vs exact.
        let approx = nn::matmul(&nn::matmul(&x, &u).unwrap(), &v).unwrap();
        let out_relerr = {
            let d2: f64 = approx.data.iter().zip(&exact.data)
                .map(|(&a, &e)| { let dd = a as f64 - e as f64; dd * dd }).sum();
            d2.sqrt() / exact_frob
        };
        let w_relerr = (1.0 - captured * captured).max(0.0).sqrt();
        // two-GEMM speed.
        let mut blr = f64::INFINITY;
        for _ in 0..3 { black_box(nn::matmul(&nn::matmul(&x, &u).unwrap(), &v).unwrap()); }
        for _ in 0..iters {
            let t = Instant::now();
            black_box(nn::matmul(&nn::matmul(&x, &u).unwrap(), &v).unwrap());
            blr = blr.min(t.elapsed().as_secs_f64());
        }
        let flop_cut = (ind as f64 * outd as f64) / (rk as f64 * (ind + outd) as f64);
        println!(
            "  r={rk:<4}: {:.3} ms  {:.2}x speed  (FLOPcut {:.2}x) | captured={:.3}% w_relerr={:.3}% OUT_relerr={:.3}%",
            blr * 1e3, bbase / blr, flop_cut, captured * 100.0, w_relerr * 100.0, out_relerr * 100.0,
        );
    }
}

fn main() {
    let layer: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(15);
    let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let path = find_model_file("large-v3-turbo").expect("set FRANKEN_WHISPER_MODEL_DIR");
    let model = GgmlModel::load(&path).expect("load large-v3-turbo");
    println!("=== low-rank WEIGHT factorization, encoder layer {layer}, real turbo weights @ {}t ===",
             rayon::current_num_threads());
    println!("captured = rank-r Frobenius energy of REAL weight; OUT_relerr = (x@U)@V vs x@W_t; speed weight-factor load-amortized");
    let seq = 1500;
    let p = |s: &str| format!("encoder.blocks.{layer}.{s}");
    bench_weight(&model, "attn.query", &p("attn.query.weight"), &[128, 256, 512, 768], seq, iters);
    bench_weight(&model, "attn.out  ", &p("attn.out.weight"), &[128, 256, 512, 768], seq, iters);
    bench_weight(&model, "mlp.0 fc1 ", &p("mlp.0.weight"), &[256, 512, 768, 1024], seq, iters);
    bench_weight(&model, "mlp.2 fc2 ", &p("mlp.2.weight"), &[256, 512, 768, 1024], seq, iters);
}
