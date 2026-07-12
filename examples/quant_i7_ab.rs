//! Same-binary ABBA A/B for the encoder activation quantize (`nn::quantize_act_i7`).
//!
//! NEW arm = the shipped AVX2 `quantize_act_i7` (round-half-away via `quantize_i7_row_into_u8`).
//! OLD arm = an inline copy of the pre-change scalar loop (per-row amax + scalar
//! `(v*inv).round().clamp(-127,127) as i32 + 128`), same rayon parallelism.
//! Byte-exactness is proven separately by `quantize_i7_row_u8_matches_scalar_reference`
//! + the transcript diff; this example only measures the kernel speedup.

use franken_whisper::native_engine::nn;
use franken_whisper::native_engine::Mat;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn old_quant(x: &Mat) -> (Vec<u8>, Vec<f32>) {
    let mut data = vec![0u8; x.rows * x.cols];
    let mut scale = vec![0.0f32; x.rows];
    data.par_chunks_mut(x.cols)
        .zip(scale.par_iter_mut())
        .enumerate()
        .for_each(|(r, (xr_u8, s))| {
            let xr = &x.data[r * x.cols..(r + 1) * x.cols];
            let amax = xr.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
            let row_scale = amax / 127.0;
            *s = row_scale;
            let inv = 1.0 / row_scale;
            for (d, &v) in xr_u8.iter_mut().zip(xr) {
                let i8v = (v * inv).round().clamp(-127.0, 127.0) as i32;
                *d = (i8v + 128) as u8;
            }
        });
    (data, scale)
}

fn build(rows: usize, cols: usize) -> Mat {
    // Deterministic xorshift, LN-scale activations (~N(0,1)-ish, some outliers).
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0
    };
    Mat::from_vec(rows, cols, (0..rows * cols).map(|_| nf()).collect())
}

fn bench(name: &str, rows: usize, cols: usize) {
    let m = build(rows, cols);
    let reps = 60usize;
    let (mut tn, mut to) = (f64::MAX, f64::MAX);
    // ABBA interleave: NEW, OLD, OLD, NEW per pair — cancels linear drift.
    for _ in 0..reps {
        let t = Instant::now();
        black_box(nn::quantize_act_i7(black_box(&m)));
        tn = tn.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(old_quant(black_box(&m)));
        to = to.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(old_quant(black_box(&m)));
        to = to.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        black_box(nn::quantize_act_i7(black_box(&m)));
        tn = tn.min(t.elapsed().as_secs_f64());
    }
    println!(
        "{name:22} [{rows}x{cols}]  new(AVX2) {:.1} us   old(scalar) {:.1} us   speedup {:.2}x",
        tn * 1e6,
        to * 1e6,
        to / tn
    );
}

fn main() {
    // Real encoder activation shapes: seq=1500 rows.
    bench("tiny.en qkv (n=384)", 1500, 384);
    bench("tiny.en fc1  (n=1536)", 1500, 1536);
    bench("turbo   qkv (n=1280)", 1500, 1280);
    bench("turbo   fc1  (n=5120)", 1500, 5120);
}
