//! Cross-K/V int8-quantize granularity probe (BlackThrush, 2026-07-02).
//!
//! The per-window cross-attention cache is int8-quantized at
//! `decoder.rs:930-938` with a SERIAL outer loop over ~160 small per-head mats
//! (turbo: 4 layers × 20 heads × {K:[1500,64], V:[64,1500]}), each quantized by
//! `nn::quantize_f16_to_i8` which is INTERNALLY parallel (`par_chunks_mut` over
//! rows). For the K mats that inner parallelism is over 1500 tiny 64-element
//! rows — likely rayon-overhead-bound. This probe MEASURES whether flipping the
//! granularity to a PARALLEL outer loop with a SERIAL inner quantize (coarse:
//! ~160 mats / 8 cores, each done whole) beats the current fine-grained form.
//! All variants produce byte-identical i8+scales (same per-row arithmetic).
//!
//!   A) serial outer, par inner  (current decoder.rs behavior)
//!   B) par outer,    par inner  (minimal .iter()->.par_iter() change; nested)
//!   C) par outer,    serial inner (coarse — the hypothesized correct grain)
//!
//! Usage: `cross_quant_probe [iters]`  (default 100).
use franken_whisper::native_engine::nn;
use half::f16;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

// Serial mirror of nn::quantize_f16_to_i8 (same per-row symmetric int8 arithmetic,
// no internal rayon). Returns (data, scales) so we can checksum byte-exactness.
fn quant_serial(w: &[f16], out: usize, inp: usize) -> (Vec<i8>, Vec<f32>) {
    let mut data = vec![0i8; out * inp];
    let mut scales = vec![0.0f32; out];
    for o in 0..out {
        let wrow = &w[o * inp..(o + 1) * inp];
        let amax = wrow
            .iter()
            .map(|h| h.to_f32().abs())
            .fold(0.0f32, f32::max)
            .max(1e-9);
        let sc = amax / 127.0;
        scales[o] = sc;
        let inv = 1.0 / sc;
        for (d, h) in data[o * inp..(o + 1) * inp].iter_mut().zip(wrow) {
            *d = (h.to_f32() * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (data, scales)
}

fn mk(out: usize, inp: usize, seed: u64) -> Vec<f16> {
    let mut s = seed;
    (0..out * inp)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            f16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
        })
        .collect()
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    // turbo cross cache: 4 layers × 20 heads, K=[1500,64], V=[64,1500].
    let (n_layer, n_head, d_head, enc) = (4usize, 20usize, 64usize, 1500usize);
    let nmats = n_layer * n_head; // 80 K + 80 V
    let ks: Vec<Vec<f16>> = (0..nmats)
        .map(|i| mk(enc, d_head, 0x100 + i as u64))
        .collect();
    let vs: Vec<Vec<f16>> = (0..nmats)
        .map(|i| mk(d_head, enc, 0x900 + i as u64))
        .collect();
    println!(
        "turbo cross cache: {nmats} K[{enc},{d_head}] + {nmats} V[{d_head},{enc}]  ({} elems)",
        nmats * enc * d_head * 2
    );

    // Byte-exactness: serial-inner must equal the crate's par-inner quantize.
    let a0 = nn::quantize_f16_to_i8(&ks[0], enc, d_head);
    let (cd, cs) = quant_serial(&ks[0], enc, d_head);
    let i8_ok = a0.data == cd;
    let sc_ok = a0.scales == cs;
    println!("byte-exact: serial-inner == par-inner  data={i8_ok} scales={sc_ok}");

    for _ in 0..3 {
        black_box(
            ks.iter()
                .map(|k| nn::quantize_f16_to_i8(k, enc, d_head))
                .collect::<Vec<_>>(),
        );
    }

    let run = |label: &str, f: &dyn Fn() -> usize| {
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            let acc = f();
            best = best.min(t.elapsed().as_secs_f64());
            black_box(acc);
        }
        println!("  {label:<28}: {:.3} ms", best * 1e3);
        best
    };

    // A) serial outer, par inner (current).
    let a = run("A) serial-outer par-inner", &|| {
        let k: Vec<_> = ks
            .iter()
            .map(|k| nn::quantize_f16_to_i8(k, enc, d_head))
            .collect();
        let v: Vec<_> = vs
            .iter()
            .map(|v| nn::quantize_f16_to_i8(v, d_head, enc))
            .collect();
        k.len() + v.len()
    });
    // B) par outer, par inner (minimal change; nested rayon).
    let b = run("B) par-outer par-inner", &|| {
        let k: Vec<_> = ks
            .par_iter()
            .map(|k| nn::quantize_f16_to_i8(k, enc, d_head))
            .collect();
        let v: Vec<_> = vs
            .par_iter()
            .map(|v| nn::quantize_f16_to_i8(v, d_head, enc))
            .collect();
        k.len() + v.len()
    });
    // C) par outer, serial inner (coarse grain).
    let c = run("C) par-outer serial-inner", &|| {
        let k: Vec<_> = ks
            .par_iter()
            .map(|k| quant_serial(k, enc, d_head))
            .collect();
        let v: Vec<_> = vs
            .par_iter()
            .map(|v| quant_serial(v, d_head, enc))
            .collect();
        k.len() + v.len()
    });

    println!("speedups vs A:  B={:.2}×  C={:.2}×", a / b, a / c);
}
