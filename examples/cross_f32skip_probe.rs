//! Cross-build: skip the unused f32 kh_t/vh buffers when FW_CROSS_F16 is on (BlackThrush, 2026-07-03).
//!
//! `DecoderState::new`'s per-window cross-attention build makes FOUR buffers per (layer,head):
//! kh_t [d_head, enc_frames] f32 (a LARGE-STRIDE transpose), k_nat [enc_frames, d_head] f16,
//! vh [enc_frames, d_head] f32, v_t [d_head, enc_frames] f16. But the f32 kh_t/vh are consumed
//! ONLY by the `FW_CROSS_F16=0` escape-hatch path (decoder.rs:1284/1288); the DEFAULT f16/i8
//! path uses only k_nat/v_t (+ DTW `scores_all`, built in the f16 path too). So when
//! FW_CROSS_F16 is on (default), kh_t (strided transpose, ~expensive) + vh are built every
//! window but NEVER read. This A/Bs the full 80-pair build (turbo: 4 layers × 20 heads) with
//! all-4 vs f16-only, to size the redundant f32 work. f16 buffers are byte-identical either way.
//! Usage: `cross_f32skip_probe [iters]` (default 400).
#![allow(unsafe_code)]
use ft_core::Float16;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

type CrossHeadBuffers = (Vec<f32>, Vec<Float16>, Vec<f32>, Vec<Float16>);

#[allow(
    clippy::too_many_arguments,
    reason = "the probe exposes each cross-cache geometry and output mode independently"
)]
fn build(
    cross_k: &[Vec<f32>],
    cross_v: &[Vec<f32>],
    n_layer: usize,
    n_head: usize,
    enc_frames: usize,
    n_state: usize,
    d_head: usize,
    need_f32: bool,
) -> Vec<CrossHeadBuffers> {
    (0..n_layer * n_head)
        .into_par_iter()
        .map(|idx| {
            let (li, h) = (idx / n_head, idx % n_head);
            let ck = &cross_k[li];
            let cv = &cross_v[li];
            let base = h * d_head;
            let mut kh_t = if need_f32 {
                vec![0.0f32; d_head * enc_frames]
            } else {
                Vec::new()
            };
            let mut k_nat = Vec::<Float16>::with_capacity(enc_frames * d_head);
            if need_f32 {
                for j in 0..enc_frames {
                    let src = &ck[j * n_state + base..j * n_state + base + d_head];
                    for (d, &s) in src.iter().enumerate() {
                        kh_t[d * enc_frames + j] = s;
                        k_nat.push(Float16::from_f32(s));
                    }
                }
            } else {
                for j in 0..enc_frames {
                    let src = &ck[j * n_state + base..j * n_state + base + d_head];
                    for &s in src {
                        k_nat.push(Float16::from_f32(s));
                    }
                }
            }
            let mut vh = if need_f32 {
                vec![0.0f32; enc_frames * d_head]
            } else {
                Vec::new()
            };
            let mut v_t = vec![Float16::from_bits(0); d_head * enc_frames];
            for j in 0..enc_frames {
                let src = &cv[j * n_state + base..j * n_state + base + d_head];
                if need_f32 {
                    vh[j * d_head..(j + 1) * d_head].copy_from_slice(src);
                }
                for (d, &s) in src.iter().enumerate() {
                    v_t[d * enc_frames + j] = Float16::from_f32(s);
                }
            }
            (kh_t, k_nat, vh, v_t)
        })
        .collect()
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(400);
    let (n_layer, n_head, enc_frames, n_state) = (4usize, 20usize, 1500usize, 1280usize);
    let d_head = n_state / n_head;
    println!(
        "=== cross-build: all-4 buffers vs f16-only (skip unused f32 kh_t/vh), turbo 80 pairs ==="
    );

    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let cross_k: Vec<Vec<f32>> = (0..n_layer)
        .map(|_| (0..enc_frames * n_state).map(|_| nf()).collect())
        .collect();
    let cross_v: Vec<Vec<f32>> = (0..n_layer)
        .map(|_| (0..enc_frames * n_state).map(|_| nf()).collect())
        .collect();

    // byte-exactness of the f16 buffers (k_nat, v_t) across need_f32 true/false.
    let a = build(
        &cross_k, &cross_v, n_layer, n_head, enc_frames, n_state, d_head, true,
    );
    let b = build(
        &cross_k, &cross_v, n_layer, n_head, enc_frames, n_state, d_head, false,
    );
    let mut bad = 0usize;
    for (x, y) in a.iter().zip(&b) {
        if x.1
            .iter()
            .zip(&y.1)
            .any(|(p, q)| p.to_bits() != q.to_bits())
        {
            bad += 1;
        }
        if x.3
            .iter()
            .zip(&y.3)
            .any(|(p, q)| p.to_bits() != q.to_bits())
        {
            bad += 1;
        }
    }

    let run = |need_f32: bool| -> f64 {
        for _ in 0..3 {
            black_box(build(
                &cross_k, &cross_v, n_layer, n_head, enc_frames, n_state, d_head, need_f32,
            ));
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            let r = build(
                &cross_k, &cross_v, n_layer, n_head, enc_frames, n_state, d_head, need_f32,
            );
            best = best.min(t.elapsed().as_secs_f64());
            black_box(r);
        }
        best
    };
    let all4 = run(true);
    let f16only = run(false);
    println!(
        "  f16 buffers byte-exact across skip: {bad} differing pairs  [{}]",
        if bad == 0 { "IDENTICAL" } else { "DIVERGENT" }
    );
    println!(
        "  all-4 (current, need_f32=true)  : {:>8.2} ms/window",
        all4 * 1e3
    );
    println!(
        "  f16-only (skip kh_t/vh)         : {:>8.2} ms/window  {:.2}x  [{}]",
        f16only * 1e3,
        all4 / f16only,
        if f16only < all4 { "WIN" } else { "loss" }
    );
    println!(
        "  => saved {:.2} ms/window when FW_CROSS_F16 on (default); kh_t is a large-stride transpose",
        (all4 - f16only) * 1e3
    );
}
