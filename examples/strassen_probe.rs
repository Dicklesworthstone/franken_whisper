//! One-level Strassen for the encoder MLP GEMMs (BlackThrush, 2026-07-02).
//!
//! Every prior encoder lever held the FLOP count fixed (n^3 FMAs) and tried to
//! execute them faster (int8, blocking, threads) — all hit the CPU power ceiling
//! (this session MEASURED the encoder is FREQUENCY/POWER-bound at 32t = 2022 MHz,
//! NOT bandwidth: DRAM traffic <6 GB/s). Strassen does FEWER mults (7 vs 8 per
//! 2x2 block = 12.5% fewer multiply-FLOPs per level), so on a power-capped
//! machine it can beat the ceiling that every FLOP-preserving lever accepted.
//!
//! Prior ledger dismissed Strassen ONLY for the K=64 attention GEMM (contraction
//! too small) — never for the MLP GEMMs (K=1280/5120, large contraction = the
//! right target). Implemented FRANKEN-SIDE (recursion calls `nn::matmul` for the
//! 7 sub-products), so it is NOT an ft_kernel_cpu change. NON-byte-exact (extra
//! adds reorder f32 rounding) — this probe reports max|Δ| to gauge transcript
//! risk; a speed win would then be gated + transcript-measured before landing.
//!
//! Method: one-level Strassen-Winograd on [M,K]@[K,N], base case = `nn::matmul`
//! (the current blocked ft path). min-of-N vs baseline single `nn::matmul`.
//! Usage: `RAYON_NUM_THREADS=32 strassen_probe [iters]` (default 40).
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;
use std::hint::black_box;
use std::time::Instant;

/// Extract contiguous [br,bc] block at (r0,c0) from row-major `src` [_,cols].
fn block(src: &[f32], cols: usize, r0: usize, c0: usize, br: usize, bc: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; br * bc];
    for r in 0..br {
        let s = (r0 + r) * cols + c0;
        out[r * bc..(r + 1) * bc].copy_from_slice(&src[s..s + bc]);
    }
    out
}
fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}
fn mm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    nn::matmul(
        &Mat::from_vec(m, k, a.to_vec()),
        &Mat::from_vec(k, n, b.to_vec()),
    )
    .unwrap()
    .data
}

/// One-level Strassen: C[M,N] = A[M,K] @ B[K,N]. M,K,N must be even.
fn strassen1(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let (m2, k2, n2) = (m / 2, k / 2, n / 2);
    // A quadrants [m2,k2], B quadrants [k2,n2].
    let a11 = block(a, k, 0, 0, m2, k2);
    let a12 = block(a, k, 0, k2, m2, k2);
    let a21 = block(a, k, m2, 0, m2, k2);
    let a22 = block(a, k, m2, k2, m2, k2);
    let b11 = block(b, n, 0, 0, k2, n2);
    let b12 = block(b, n, 0, n2, k2, n2);
    let b21 = block(b, n, k2, 0, k2, n2);
    let b22 = block(b, n, k2, n2, k2, n2);
    // 7 products (Strassen).
    let m1 = mm(&add(&a11, &a22), m2, k2, &add(&b11, &b22), n2);
    let m2p = mm(&add(&a21, &a22), m2, k2, &b11, n2);
    let m3 = mm(&a11, m2, k2, &sub(&b12, &b22), n2);
    let m4 = mm(&a22, m2, k2, &sub(&b21, &b11), n2);
    let m5 = mm(&add(&a11, &a12), m2, k2, &b22, n2);
    let m6 = mm(&sub(&a21, &a11), m2, k2, &add(&b11, &b12), n2);
    let m7 = mm(&sub(&a12, &a22), m2, k2, &add(&b21, &b22), n2);
    // C quadrants.
    let c11 = add(&sub(&add(&m1, &m4), &m5), &m7); // m1+m4-m5+m7
    let c12 = add(&m3, &m5);
    let c21 = add(&m2p, &m4);
    let c22 = add(&sub(&add(&m1, &m3), &m2p), &m6); // m1-m2+m3+m6
    // Scatter quadrants into [M,N].
    let mut c = vec![0.0f32; m * n];
    for r in 0..m2 {
        c[r * n..r * n + n2].copy_from_slice(&c11[r * n2..(r + 1) * n2]);
        c[r * n + n2..(r + 1) * n].copy_from_slice(&c12[r * n2..(r + 1) * n2]);
        let rr = m2 + r;
        c[rr * n..rr * n + n2].copy_from_slice(&c21[r * n2..(r + 1) * n2]);
        c[rr * n + n2..(rr + 1) * n].copy_from_slice(&c22[r * n2..(r + 1) * n2]);
    }
    c
}

fn bench(name: &str, m: usize, k: usize, n: usize, iters: usize) {
    let mut s = 0x51A5_1234u64;
    let mut nf = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let a: Vec<f32> = (0..m * k).map(|_| nf()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| nf() * 0.1).collect();
    let am = Mat::from_vec(m, k, a.clone());
    let bm = Mat::from_vec(k, n, b.clone());
    let reference = nn::matmul(&am, &bm).unwrap();

    let mut bbase = f64::INFINITY;
    for _ in 0..3 {
        black_box(nn::matmul(&am, &bm).unwrap());
    }
    for _ in 0..iters {
        let t = Instant::now();
        let r = nn::matmul(&am, &bm).unwrap();
        bbase = bbase.min(t.elapsed().as_secs_f64());
        black_box(r);
    }
    let mut bstr = f64::INFINITY;
    let mut cstr = Vec::new();
    for _ in 0..3 {
        black_box(strassen1(&a, m, k, &b, n));
    }
    for _ in 0..iters {
        let t = Instant::now();
        cstr = strassen1(&a, m, k, &b, n);
        bstr = bstr.min(t.elapsed().as_secs_f64());
        black_box(&cstr);
    }
    let maxd = cstr
        .iter()
        .zip(reference.data.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let relmax = reference
        .data
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-9);
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
    println!(
        "{name}  [{m}x{k}]@[{k}x{n}]  ({gflop:.1} GF)  best-of-{iters} @ {}t:",
        rayon::current_num_threads()
    );
    println!(
        "  baseline nn::matmul : {:>7.2} ms  {:>6.0} GF/s",
        bbase * 1e3,
        gflop / bbase
    );
    println!(
        "  1-level Strassen    : {:>7.2} ms  {:>6.0} GF/s  {:.2}x  |  max|d|={:.2e} (rel {:.1e})  {}",
        bstr * 1e3,
        gflop / bstr,
        bbase / bstr,
        maxd,
        maxd / relmax,
        if bstr < bbase { "WIN" } else { "loss" }
    );
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    println!("=== one-level Strassen vs blocked nn::matmul, encoder MLP shapes ===");
    bench("MLP fc1", 1500, 1280, 5120, iters);
    bench("MLP fc2", 1500, 5120, 1280, iters);
    bench("QKV/out", 1500, 1280, 1280, iters);
}
