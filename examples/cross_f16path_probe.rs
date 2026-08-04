//! Cross projection: real f16 path vs f32-sgemm (BlackThrush, 2026-07-02).
//!
//! The cross_attn_k/v projections are F16-stored + unquantized, so
//! Linear::forward runs `nn::gemv_f16_batch` for the tq=1500 cross precompute.
//! For a GEMM-shaped (tq=1500) problem an f16 batched-GEMV can be memory-bound
//! vs a tiled f32 sgemm (the ENCODER already dequants-once to f32 for exactly
//! this reason). This probe MEASURES the real f16 path against the f32-sgemm
//! path on the turbo cross shape, and reports max|Δ| between them (f32-sgemm is
//! NOT bit-identical to the f16 GEMV — different accumulation — so this
//! quantifies both the speed gap AND the numeric divergence, to judge whether a
//! switch is a byte-exact win, an owner-gated speed/quality trade, or neither).
//!
//! Usage: `cross_f16path_probe [iters]`  (default 50).
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;
use half::f16;
use std::hint::black_box;
use std::time::Instant;

fn fill_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let (n_layer, enc, n_state) = (4usize, 1500usize, 1280usize);
    let x = fill_f32(enc * n_state, 0x1); // encoder_out [enc, n_state]

    // Per-layer weights: f16 (real path) and its f32 view (sgemm path). The f32
    // view is the EXACT dequant of the f16 weight, so both consume identical
    // numeric weights — only the GEMM accumulation order differs.
    let wf16: Vec<Vec<f16>> = (0..n_layer * 2)
        .map(|i| {
            fill_f32(n_state * n_state, 0x100 + i as u64)
                .iter()
                .map(|&v| f16::from_f32(v))
                .collect()
        })
        .collect();
    let wf32: Vec<Mat> = wf16
        .iter()
        .map(|w| {
            // f16 weight is [out, in] row-major; matmul wants [in, out] (transposed).
            let mut t = vec![0.0f32; n_state * n_state];
            for o in 0..n_state {
                for ii in 0..n_state {
                    t[ii * n_state + o] = w[o * n_state + ii].to_f32();
                }
            }
            Mat::from_vec(n_state, n_state, t)
        })
        .collect();
    let xmat = Mat::from_vec(enc, n_state, x.clone());
    println!(
        "turbo cross proj: enc[{enc},{n_state}] × {} [{n_state},{n_state}] GEMMs",
        n_layer * 2
    );

    // numeric divergence on one projection.
    let mut yf16 = nn::gemv_out_buf(enc * n_state);
    nn::gemv_f16_batch(&wf16[0], n_state, n_state, &x, enc, None, &mut yf16);
    let yf32 = nn::matmul(&xmat, &wf32[0]).unwrap();
    let maxd = yf16
        .iter()
        .zip(&yf32.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "max|Δ| f16-gemv vs f32-sgemm = {maxd:.3e}  (0 = bit-exact; >0 = NOT byte-exact to swap)"
    );

    let f16path = || {
        let mut out = Vec::with_capacity(n_layer * 2);
        for w in &wf16 {
            let mut y = nn::gemv_out_buf(enc * n_state);
            nn::gemv_f16_batch(w, n_state, n_state, &x, enc, None, &mut y);
            out.push(y);
        }
        out
    };
    let f32path = || {
        let mut out = Vec::with_capacity(n_layer * 2);
        for w in &wf32 {
            out.push(nn::matmul(&xmat, w).unwrap());
        }
        out
    };

    for _ in 0..3 {
        black_box(f16path());
        black_box(f32path());
    }
    let mut b16 = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let r = f16path();
        b16 = b16.min(t.elapsed().as_secs_f64());
        black_box(r);
    }
    let mut b32 = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        let r = f32path();
        b32 = b32.min(t.elapsed().as_secs_f64());
        black_box(r);
    }

    println!("best-of-{iters} (8 GEMMs):");
    println!("  f16 gemv_f16_batch (real) : {:.3} ms", b16 * 1e3);
    println!(
        "  f32 sgemm (dequant-once)  : {:.3} ms   ({:.2}× vs f16)",
        b32 * 1e3,
        b16 / b32
    );
}
