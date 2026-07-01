#![feature(portable_simd)]
//! Is the whisper.cpp f16-table GELU faster than franken's old live-tanh GELU?
//!
//! whisper.cpp ships `GGML_GELU_FP16` — a `1<<16` f16 lookup table (vec.h), NOT
//! the live `0.5x(1+tanh(...))` franken used to compute. The table is both closer
//! to ORIG (bit-exact with whisper) and cheaper: a `vcvtps2ph` + load per element
//! vs a scalar `tanh` per lane. This probes the kernel A/B directly on an
//! encoder-sized buffer (1500 frames × mlp_hidden), model-free, best-of-N.
//! It also reports the max |Δ| between the two forms (the f16-quantization error,
//! ~1e-3) as an accuracy canary. Usage: `gelu_probe [iters]` (default 50).
use std::hint::black_box;
use std::time::Instant;

const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
const GELU_COEF_A: f32 = 0.044_715;

/// Old franken GELU: 16-lane SIMD polynomial with scalar `tanh` per lane.
fn gelu_tanh(data: &mut [f32]) {
    use std::simd::Simd;
    const L: usize = 16;
    type V = Simd<f32, L>;
    let n = data.len();
    let nl = n - n % L;
    let coef_a = V::splat(GELU_COEF_A);
    let sqrt_2pi = V::splat(GELU_SQRT_2_OVER_PI);
    let one = V::splat(1.0);
    let half = V::splat(0.5);
    let mut i = 0;
    while i < nl {
        let xv = V::from_slice(&data[i..i + L]);
        let arg = (sqrt_2pi * xv) * (one + (coef_a * xv) * xv);
        let aa = arg.to_array();
        let tanh = V::from_array(std::array::from_fn(|k| aa[k].tanh()));
        ((half * xv) * (one + tanh)).copy_to_slice(&mut data[i..i + L]);
        i += L;
    }
    for v in &mut data[nl..] {
        let x = *v;
        *v = 0.5 * x * (1.0 + (GELU_SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh());
    }
}

/// whisper.cpp `GGML_GELU_FP16` f16-table GELU (the new franken path).
fn build_table() -> Vec<f32> {
    use ft_core::Float16;
    let mut t = vec![0.0f32; 1 << 16];
    for (i, slot) in t.iter_mut().enumerate() {
        let f = Float16::from_bits(i as u16).to_f32();
        let g = 0.5 * f * (1.0 + (GELU_SQRT_2_OVER_PI * f * (1.0 + GELU_COEF_A * f * f)).tanh());
        *slot = Float16::from_f32(g).to_f32();
    }
    t
}

fn gelu_table(data: &mut [f32], table: &[f32]) {
    use ft_core::Float16;
    for v in data.iter_mut() {
        let x = *v;
        *v = if x <= -10.0 {
            0.0
        } else if x >= 10.0 {
            x
        } else {
            table[Float16::from_f32(x).to_bits() as usize]
        };
    }
}

fn best_of(iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..2 {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    best
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let n = 1500 * 5120; // one large-v3-turbo encoder MLP activation panel.
    // Realistic post-layernorm activations: roughly unit-scale, spread over [-6,6].
    let base: Vec<f32> = (0..n)
        .map(|i| {
            let t = (i as f32 * 0.6180339887) % 1.0; // low-discrepancy in [0,1)
            (t - 0.5) * 12.0
        })
        .collect();

    let table = build_table();

    let mut buf = base.clone();
    let tanh_ms = best_of(iters, || {
        buf.copy_from_slice(&base);
        gelu_tanh(&mut buf);
        black_box(&buf);
    }) * 1e3;

    let mut buf2 = base.clone();
    let table_ms = best_of(iters, || {
        buf2.copy_from_slice(&base);
        gelu_table(&mut buf2, &table);
        black_box(&buf2);
    }) * 1e3;

    // Accuracy canary: max |Δ| tanh-form vs table-form over the full panel.
    let mut a = base.clone();
    let mut b = base.clone();
    gelu_tanh(&mut a);
    gelu_table(&mut b, &table);
    let maxdiff = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);

    let mb = (n * 4) as f64 / 1e6;
    println!("gelu panel n={n} ({mb:.1} MB f32) best-of-{iters}");
    println!("  tanh (old)  {tanh_ms:7.3} ms");
    println!("  table (new) {table_ms:7.3} ms   speedup={:.2}x", tanh_ms / table_ms);
    println!("  max |Δ| tanh-vs-table = {maxdiff:.5}  (f16-quant error, expect ~1e-3)");
}
