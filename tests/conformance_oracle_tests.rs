//! Differential-oracle conformance tests (feature `fj-oracle`, bd-ohra).
//!
//! Cross-checks the native engine's `nn.rs` primitives — matmul, softmax,
//! layer-norm, conv1d — against the independent frankenjax interpreter
//! (`fj-lax::eval_primitive`) on seeded random tensors, tolerance 1e-4 max
//! absolute element difference. This is the FrankenSuite verification story:
//! oracle-verified kernels under the engine.
//!
//! Run: `cargo test --features fj-oracle --test conformance_oracle_tests`
//! (slow; intended for CI with the feature enabled and local verification).
//!
//! Divergence behavior: any mismatch beyond `TOLERANCE` fails the test after
//! writing a full JSON artifact (inputs, expected, actual, per-element diffs)
//! beneath `target/` for debugging.
//!
//! Determinism: all randomness comes from a seeded xorshift64* generator;
//! every test pins its seed in its name. No wall-clock, no threads, no model.

#![cfg(feature = "fj-oracle")]

use fj_core::{Primitive, Shape, TensorValue, Value};
use fj_lax::eval_primitive;
use franken_whisper::native_engine::{Mat, nn};
use std::collections::BTreeMap;

/// Maximum absolute element difference accepted between the native kernel and
/// the fj interpreter (bead-specified).
const TOLERANCE: f32 = 1e-4;

// ---------------------------------------------------------------------------
// Seeded randomness (xorshift64*, deterministic across platforms)
// ---------------------------------------------------------------------------

struct XorShift(u64);

impl XorShift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform f32 in [-1, 1).
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 bits of mantissa entropy
        let unit = f32::from_bits(0x3F80_0000 | (bits & 0x007F_FFFF)) - 1.0;
        unit * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

// ---------------------------------------------------------------------------
// fj harness helpers
// ---------------------------------------------------------------------------

fn tensor(shape: &[u32], data: &[f32]) -> Value {
    Value::Tensor(
        TensorValue::new_f32_values(
            Shape {
                dims: shape.to_vec(),
            },
            data.to_vec(),
        )
        .expect("oracle input tensor"),
    )
}

fn params(pairs: &[(&str, String)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), v.clone()))
        .collect()
}

/// Evaluate one primitive and return its F32 tensor payload flattened.
fn eval_f32(primitive: Primitive, inputs: &[Value], p: &BTreeMap<String, String>) -> Vec<f32> {
    match eval_primitive(primitive, inputs, p).expect("fj eval succeeds") {
        Value::Tensor(t) => t
            .elements
            .as_f32_slice()
            .map(<[f32]>::to_vec)
            .unwrap_or_else(|| {
                t.to_f64_vec()
                    .expect("numeric tensor")
                    .iter()
                    .map(|&v| v as f32)
                    .collect()
            }),
        other => panic!("expected tensor from {primitive:?}, got {other:?}"),
    }
}

fn broadcast(src: &Value, target_shape: &[u32], broadcast_dims: &str) -> Value {
    let shape_str = target_shape
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut p = BTreeMap::new();
    p.insert("shape".to_owned(), shape_str);
    p.insert("broadcast_dimensions".to_owned(), broadcast_dims.to_owned());
    eval_primitive(Primitive::BroadcastInDim, std::slice::from_ref(src), &p)
        .expect("broadcast_in_dim")
}

fn reduce_axes(op: Primitive, src: &Value, axes: &str) -> Vec<f32> {
    eval_f32(
        op,
        std::slice::from_ref(src),
        &params(&[("axes", axes.to_owned())]),
    )
}

// ---------------------------------------------------------------------------
// Oracle compositions (each mirrors the published JAX/fj definition built ONLY
// from fj primitives, so a native-kernel divergence is an engine bug or a
// composition bug — never two implementations drifting silently)
// ---------------------------------------------------------------------------

/// jnp.dot for 2-D operands: [M,K] x [K,N] -> [M,N].
fn oracle_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: u32) -> Vec<f32> {
    let lhs = tensor(&[m as u32, k as u32], a);
    let rhs = tensor(&[k as u32, n], b);
    eval_f32(Primitive::Dot, &[lhs, rhs], &BTreeMap::new())
}

/// Row-wise softmax over [R,K]: exp(x - max) / sum(exp(x - max)).
fn oracle_softmax(x: &[f32], r: u32, k: u32) -> Vec<f32> {
    let xt = tensor(&[r, k], x);
    let row_max = reduce_axes(Primitive::ReduceMax, &xt, "1");
    let max_b = broadcast(&tensor(&[r, 1], &row_max), &[r, k], "0,1");
    let z = eval_f32(Primitive::Sub, &[xt.clone(), max_b], &BTreeMap::new());
    let e_t = tensor(&[r, k], &z);
    let e = eval_f32(Primitive::Exp, std::slice::from_ref(&e_t), &BTreeMap::new());
    let et = tensor(&[r, k], &e);
    let denom = reduce_axes(Primitive::ReduceSum, &et, "1");
    let denom_b = broadcast(&tensor(&[r, 1], &denom), &[r, k], "0,1");
    eval_f32(Primitive::Div, &[et, denom_b], &BTreeMap::new())
}

/// Row-wise layer norm over [R,K] with per-channel gamma/beta.
fn oracle_layer_norm(x: &[f32], r: u32, k: u32, gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let xt = tensor(&[r, k], x);
    // mean = sum(x)/K per row, then center.
    let row_sum = reduce_axes(Primitive::ReduceSum, &xt, "1");
    let k_col = vec![k as f32; r as usize];
    let mean = {
        let s = tensor(&[r, 1], &row_sum);
        let kk = tensor(&[r, 1], &k_col);
        eval_f32(Primitive::Div, &[s, kk], &BTreeMap::new())
    };
    let mean_b = broadcast(&tensor(&[r, 1], &mean), &[r, k], "0,1");
    let centered_v = eval_f32(
        Primitive::Sub,
        &[xt.clone(), mean_b.clone()],
        &BTreeMap::new(),
    );
    let centered = tensor(&[r, k], &centered_v);
    // var = sum(centered^2)/K per row.
    let sq_v = eval_f32(
        Primitive::Mul,
        &[centered.clone(), centered.clone()],
        &BTreeMap::new(),
    );
    let sq = tensor(&[r, k], &sq_v);
    let sq_sum = reduce_axes(Primitive::ReduceSum, &sq, "1");
    let var = {
        let s = tensor(&[r, 1], &sq_sum);
        let kk = tensor(&[r, 1], &k_col);
        eval_f32(Primitive::Div, &[s, kk], &BTreeMap::new())
    };
    // rsqrt(var + eps): Add against an explicit column of eps values (pure
    // tensor-tensor arithmetic; no scalar-broadcast path differences).
    let eps_col = vec![eps; r as usize];
    let var_eps = {
        let v = tensor(&[r, 1], &var);
        let e = tensor(&[r, 1], &eps_col);
        eval_f32(Primitive::Add, &[v, e], &BTreeMap::new())
    };
    let inv_std = {
        let v = tensor(&[r, 1], &var_eps);
        eval_f32(Primitive::Rsqrt, std::slice::from_ref(&v), &BTreeMap::new())
    };
    let inv_b = broadcast(&tensor(&[r, 1], &inv_std), &[r, k], "0,1");
    let normed_v = eval_f32(Primitive::Mul, &[centered, inv_b], &BTreeMap::new());
    let normed = tensor(&[r, k], &normed_v);
    let gamma_b = broadcast(&tensor(&[k], gamma), &[r, k], "1");
    let scaled_v = eval_f32(Primitive::Mul, &[normed, gamma_b], &BTreeMap::new());
    let scaled = tensor(&[r, k], &scaled_v);
    let beta_b = broadcast(&tensor(&[k], beta), &[r, k], "1");
    eval_f32(Primitive::Add, &[scaled, beta_b], &BTreeMap::new())
}

/// 1-D convolution over an explicitly padded input, matching
/// `nn::conv1d_wt`'s symmetric zero-pad semantics with VALID windows:
/// lhs [1,Tp,Cin], rhs [K,Cin,Cout], stride S.
fn oracle_conv1d(
    x_padded: &[f32],
    t_padded: usize,
    cin: usize,
    w_fj: &[f32],
    k: usize,
    cout: usize,
    bias: &[f32],
    stride: usize,
) -> Vec<f32> {
    let lhs = tensor(&[1, t_padded as u32, cin as u32], x_padded);
    let rhs = tensor(&[k as u32, cin as u32, cout as u32], w_fj);
    let mut out = eval_f32(
        Primitive::Conv,
        &[lhs, rhs],
        &params(&[
            ("strides", stride.to_string()),
            ("padding", "VALID".to_owned()),
        ]),
    );
    // fj output is [1,T_out,Cout]; add bias via explicit broadcast add.
    let t_out = out.len() / cout;
    let out_t = tensor(&[1, t_out as u32, cout as u32], &out);
    let bias_b = broadcast(
        &tensor(&[cout as u32], bias),
        &[1, t_out as u32, cout as u32],
        "2",
    );
    out = eval_f32(Primitive::Add, &[out_t, bias_b], &BTreeMap::new());
    out
}

// ---------------------------------------------------------------------------
// Comparison + divergence artifact
// ---------------------------------------------------------------------------

struct Diff {
    op: &'static str,
    seed: u64,
    max_abs: f32,
    first_bad: Option<(usize, f32, f32)>,
    bad_count: usize,
    expected: Vec<f32>,
    actual: Vec<f32>,
}

fn compare(op: &'static str, seed: u64, expected: &[f32], actual: &[f32]) -> Result<(), Diff> {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{op}: oracle/native output lengths diverge ({} vs {})",
        expected.len(),
        actual.len()
    );
    let mut max_abs = 0.0f32;
    let mut first_bad = None;
    let mut bad_count = 0usize;
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let d = (e - a).abs();
        if d > max_abs {
            max_abs = d;
        }
        if d > TOLERANCE {
            bad_count += 1;
            if first_bad.is_none() {
                first_bad = Some((i, *e, *a));
            }
        }
    }
    if bad_count == 0 {
        return Ok(());
    }
    Err(Diff {
        op,
        seed,
        max_abs,
        first_bad,
        bad_count,
        expected: expected.to_vec(),
        actual: actual.to_vec(),
    })
}

impl Diff {
    /// Write the full debugging artifact and panic with its path.
    fn fail(self) -> ! {
        let dir = std::path::PathBuf::from(
            std::env::var("CARGO_TARGET_DIR")
                .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/target").to_owned()),
        );
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(format!(
            "fj_oracle_divergence_{}_seed{}.json",
            self.op, self.seed
        ));
        let artifact = serde_json::json!({
            "op": self.op,
            "seed": self.seed,
            "tolerance": TOLERANCE,
            "max_abs_diff": self.max_abs,
            "bad_count": self.bad_count,
            "first_bad": self.first_bad.map(|(i, e, a)| serde_json::json!({"index": i, "expected": e, "actual": a})),
            "expected": self.expected,
            "actual": self.actual,
        });
        let wrote = std::fs::write(
            &path,
            serde_json::to_string_pretty(&artifact).expect("artifact json"),
        );
        let note = match wrote {
            Ok(()) => format!("divergence artifact written to {}", path.display()),
            Err(e) => format!(
                "FAILED to write divergence artifact {}: {e}",
                path.display()
            ),
        };
        panic!(
            "{}: {} elements exceed tolerance {TOLERANCE} (max |diff| = {}, first at {:?}); {note}",
            self.op, self.bad_count, self.max_abs, self.first_bad
        );
    }
}

// ---------------------------------------------------------------------------
// Tests (each pins its own seed; no model required)
// ---------------------------------------------------------------------------

#[test]
fn oracle_matmul_matches_native_matmul_seed_20260823() {
    const M: usize = 16;
    const K: usize = 32;
    const N: usize = 24;
    let mut rng = XorShift(0x2026_0823_DEAD_BEEF);
    let a = rng.vec(M * K);
    let b = rng.vec(K * N);
    let native = nn::matmul(
        &Mat::from_vec(M, K, a.clone()),
        &Mat::from_vec(K, N, b.clone()),
    )
    .expect("native matmul");
    let oracle = oracle_matmul(&a, &b, M, K, N as u32);
    if let Err(d) = compare("matmul", 0x2026_0823_DEAD_BEEF, &oracle, &native.data) {
        d.fail();
    }
}

#[test]
fn oracle_softmax_rows_matches_native_seed_20260823() {
    const R: usize = 8;
    const K: usize = 64;
    let mut rng = XorShift(0x2026_0823_CAFE_F00D);
    let x = rng.vec(R * K);
    let mut native_mat = Mat::from_vec(R, K, x.clone());
    nn::softmax_rows(&mut native_mat);
    let oracle = oracle_softmax(&x, R as u32, K as u32);
    if let Err(d) = compare("softmax", 0x2026_0823_CAFE_F00D, &oracle, &native_mat.data) {
        d.fail();
    }
}

#[test]
fn oracle_layer_norm_matches_native_seed_20260823() {
    const R: usize = 6;
    const K: usize = 48;
    let mut rng = XorShift(0x2026_0823_FEED_C0DE);
    let x = rng.vec(R * K);
    let gamma: Vec<f32> = (0..K).map(|i| 0.5 + (i % 7) as f32 * 0.125).collect();
    let beta: Vec<f32> = (0..K).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
    let eps = 1e-5_f32;
    let mut native_mat = Mat::from_vec(R, K, x.clone());
    nn::layer_norm(&mut native_mat, &gamma, &beta, eps);
    let oracle = oracle_layer_norm(&x, R as u32, K as u32, &gamma, &beta, eps);
    if let Err(d) = compare(
        "layer_norm",
        0x2026_0823_FEED_C0DE,
        &oracle,
        &native_mat.data,
    ) {
        d.fail();
    }
}

#[test]
fn oracle_conv1d_matches_native_seed_20260823() {
    const T: usize = 40;
    const CIN: usize = 4;
    const K: usize = 3;
    const COUT: usize = 6;
    const STRIDE: usize = 2;
    const PAD: usize = 1;
    let mut rng = XorShift(0x2026_0823_BADC_0DE0);
    let x = rng.vec(T * CIN);
    let w = rng.vec(COUT * CIN * K);
    let bias: Vec<f32> = (0..COUT).map(|i| (i as f32) * 0.1 - 0.25).collect();

    let native = nn::conv1d(
        &Mat::from_vec(T, CIN, x.clone()),
        &w,
        COUT,
        CIN,
        K,
        &bias,
        STRIDE,
        PAD,
    )
    .expect("native conv1d");

    // Mirror conv1d's symmetric zero-pad on the time axis, then hand fj the
    // VALID-window problem so both sides see identical inputs.
    let tp = T + 2 * PAD;
    let mut xp = vec![0.0f32; tp * CIN];
    xp[PAD * CIN..(PAD + T) * CIN].copy_from_slice(&x);
    // Native weights are [Cout, Cin*K] with patch index ci*K + kk; fj Conv 1-D
    // wants [K, Cin, Cout]. Pure permutation, done once, test-side.
    let mut w_fj = vec![0.0f32; COUT * CIN * K];
    for co in 0..COUT {
        for ci in 0..CIN {
            for kk in 0..K {
                w_fj[(kk * CIN + ci) * COUT + co] = w[co * (CIN * K) + ci * K + kk];
            }
        }
    }
    let oracle = oracle_conv1d(&xp, tp, CIN, &w_fj, K, COUT, &bias, STRIDE);
    if let Err(d) = compare("conv1d", 0x2026_0823_BADC_0DE0, &oracle, &native.data) {
        d.fail();
    }
}
