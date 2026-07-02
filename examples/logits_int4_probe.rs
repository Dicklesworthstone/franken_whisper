//! int4 vs int8 logits GEMV kernel bench (BlackThrush, 2026-07-02).
//!
//! The decode's tied-logits GEMV streams the int8 `[n_vocab, n_state]` embedding
//! from DRAM every token (~66 MB, bandwidth-bound). int4 would halve that (~33 MB).
//! But memory records int4-fc1 as a "packed kernel wash" (unpack overhead ate the
//! bandwidth win) — HOWEVER fc1 is 6.5 MB (cache-friendlier), while logits is 10×
//! bigger and pure-DRAM-stream, so int4 may win HERE where it didn't for fc1.
//! This bench answers ONLY the speed question (is the int4 kernel faster?) before
//! any full load-path build; the quality/transcript question is separate.
//!
//! `gemv_i4` mirrors `nn::gemv_i8` exactly (quantize x once → per-row dot × row
//! scale × x scale, parallel over output rows) but with packed int4 weights.
//! Usage: `logits_int4_probe [iters]` (default 200). Run at RAYON_NUM_THREADS=32.
use franken_whisper::native_engine::nn;
use half::f16;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Sign-extend the two 4-bit nibbles of `b` and MAC against two int8 inputs.
#[inline(always)]
fn nib_lo(b: u8) -> i32 {
    (((b << 4) as i8) >> 4) as i32 // low nibble, sign-extended
}
#[inline(always)]
fn nib_hi(b: u8) -> i32 {
    ((b as i8) >> 4) as i32 // high nibble, sign-extended
}

/// int4 GEMV: `wdata` = per-row packed int4 (inp/2 bytes/row), `scales` per row.
fn gemv_i4(wdata: &[u8], scales: &[f32], out: usize, inp: usize, x: &[f32], y: &mut [f32]) {
    let xamax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    let xs = xamax / 127.0;
    let xinv = 1.0 / xs;
    let xi8: Vec<i8> = x.iter().map(|v| (v * xinv).round().clamp(-127.0, 127.0) as i8).collect();
    let row_bytes = inp / 2;
    let dot = |wrow: &[u8]| -> i32 {
        let mut acc: i32 = 0;
        for (k, &b) in wrow.iter().enumerate() {
            acc += nib_lo(b) * (xi8[2 * k] as i32) + nib_hi(b) * (xi8[2 * k + 1] as i32);
        }
        acc
    };
    let band = out.div_ceil(32).max(1);
    y.par_chunks_mut(band).enumerate().for_each(|(w, slice)| {
        let o0 = w * band;
        for (i, slot) in slice.iter_mut().enumerate() {
            let o = o0 + i;
            *slot = dot(&wdata[o * row_bytes..(o + 1) * row_bytes]) as f32 * scales[o] * xs;
        }
    });
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let (out, inp) = (51866usize, 1280usize); // turbo tied-logits shape

    // Synthetic f16 embedding.
    let mut s = 0x1234u64;
    let wf16: Vec<f16> = (0..out * inp)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            f16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
        })
        .collect();
    // int8 path (the real kernel).
    let w_i8 = nn::quantize_f16_to_i8(&wf16, out, inp);
    // int4 path: per-row symmetric quant, scale = amax/7, packed 2 nibbles/byte.
    let mut w_i4 = vec![0u8; out * (inp / 2)];
    let mut sc4 = vec![0.0f32; out];
    w_i4.par_chunks_mut(inp / 2).zip(sc4.par_iter_mut()).enumerate().for_each(|(o, (row, sco))| {
        let src = &wf16[o * inp..(o + 1) * inp];
        let amax = src.iter().map(|v| v.to_f32().abs()).fold(0.0f32, f32::max).max(1e-9);
        let scale = amax / 7.0;
        *sco = scale;
        let inv = 1.0 / scale;
        for (k, byte) in row.iter_mut().enumerate() {
            let q = |v: f16| (v.to_f32() * inv).round().clamp(-7.0, 7.0) as i8 as u8 & 0x0F;
            *byte = q(src[2 * k]) | (q(src[2 * k + 1]) << 4);
        }
    });

    let x: Vec<f32> = (0..inp).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut y8 = nn::gemv_out_buf(out);
    let mut y4 = nn::gemv_out_buf(out);
    let bytes8 = (out * inp) as f64;
    let bytes4 = (out * inp / 2) as f64;

    for _ in 0..5 {
        nn::gemv_i8(&w_i8, &x, None, &mut y8);
        gemv_i4(&w_i4, &sc4, out, inp, &x, &mut y4);
        black_box((&y8, &y4));
    }
    let (mut b8, mut b4) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..iters {
        let t = Instant::now();
        nn::gemv_i8(&w_i8, &x, None, &mut y8);
        b8 = b8.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        gemv_i4(&w_i4, &sc4, out, inp, &x, &mut y4);
        b4 = b4.min(t.elapsed().as_secs_f64());
        black_box((&y8, &y4));
    }
    println!("logits [{out},{inp}] best-of-{iters} @ {} threads:", rayon::current_num_threads());
    println!("  int8: {:.3} ms  {:.1} GB/s", b8 * 1e3, bytes8 / b8 / 1e9);
    println!("  int4: {:.3} ms  {:.1} GB/s  (speedup {:.2}x)", b4 * 1e3, bytes4 / b4 / 1e9, b8 / b4);
}
