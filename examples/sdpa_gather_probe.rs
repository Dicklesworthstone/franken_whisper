//! SDPA gather access-order: head-band-major (current) vs row-major (BlackThrush, 2026-07-03).
//!
//! `attention_raw`'s fused-SDPA path gathers q/k/v [tq, n_state] into head-major
//! [hh, tq, d_head] before the kernel. The LIVE gather parallelizes over heads and, per
//! head, reads `q.row(i)[base..base+d_head]` for every row i — i.e. it strides through q
//! by n_state (5 KB) reading a 256 B band each step (20 strided passes over q). Memory
//! already closed the PARALLELISM-COUNT question (serial 4.5x slower — bandwidth wants
//! many channels). This probes a different axis: ACCESS ORDER. A row-major gather reads
//! each q row CONTIGUOUSLY (one 5 KB linear pass, prefetcher-friendly) and scatters to the
//! strided head-major dst. Both produce byte-identical qa. Which locality wins on Zen3?
//! Usage: `sdpa_gather_probe [iters]` (default 200).
#![allow(unsafe_code)]
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

/// Current LIVE gather: par over heads, per-head strided read of q.
fn gather_headmajor(q: &[f32], hh: usize, tq: usize, d_head: usize) -> Vec<f32> {
    let n_state = hh * d_head;
    let mut qa = vec![0.0f32; hh * tq * d_head];
    qa.par_chunks_mut(tq * d_head)
        .enumerate()
        .for_each(|(h, blk)| {
            let base = h * d_head;
            for i in 0..tq {
                blk[i * d_head..(i + 1) * d_head]
                    .copy_from_slice(&q[i * n_state + base..i * n_state + base + d_head]);
            }
        });
    qa
}

/// Row-major gather: par over row-bands, per-row CONTIGUOUS read of q (one linear pass
/// over the row), scattered into the head-major dst. Byte-identical output.
fn gather_rowmajor(q: &[f32], hh: usize, tq: usize, d_head: usize) -> Vec<f32> {
    let n_state = hh * d_head;
    let mut qa = vec![0.0f32; hh * tq * d_head];
    // qa[h*tq*d_head + i*d_head + d] = q[i*n_state + h*d_head + d].
    // Parallelize over row-bands of the SOURCE; each task owns disjoint source rows i and
    // writes disjoint (i-indexed) slots across all heads. Use a raw pointer for the strided
    // head-major writes (each (h,i) slot is disjoint across tasks — no aliasing).
    let qa_ptr = qa.as_mut_ptr() as usize;
    let nthreads = rayon::current_num_threads().max(1);
    let band = tq.div_ceil(nthreads).max(1);
    (0..tq)
        .step_by(band)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|i0| {
            let i1 = (i0 + band).min(tq);
            let qa_base = qa_ptr as *mut f32;
            for i in i0..i1 {
                let qrow = &q[i * n_state..(i + 1) * n_state]; // contiguous read
                for h in 0..hh {
                    let dst = qa_base.wrapping_add(h * tq * d_head + i * d_head);
                    // SAFETY: (h,i) slot is unique to this task's row range; disjoint writes.
                    unsafe {
                        core::ptr::copy_nonoverlapping(qrow.as_ptr().add(h * d_head), dst, d_head);
                    }
                }
            }
        });
    qa
}

fn gather_serial(q: &[f32], hh: usize, tq: usize, d_head: usize) -> Vec<f32> {
    let n_state = hh * d_head;
    let mut qa = vec![0.0f32; hh * tq * d_head];
    for h in 0..hh {
        let base = h * d_head;
        for i in 0..tq {
            qa[h * tq * d_head + i * d_head..h * tq * d_head + (i + 1) * d_head]
                .copy_from_slice(&q[i * n_state + base..i * n_state + base + d_head]);
        }
    }
    qa
}

fn bench(hh: usize, tq: usize, d_head: usize, iters: usize) {
    let n_state = hh * d_head;
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let q: Vec<f32> = (0..tq * n_state)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect();

    let a = gather_headmajor(&q, hh, tq, d_head);
    let b = gather_rowmajor(&q, hh, tq, d_head);
    let bad = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();

    macro_rules! time {
        ($f:expr) => {{
            for _ in 0..5 {
                black_box($f);
            }
            let mut best = f64::INFINITY;
            for _ in 0..iters {
                let t = Instant::now();
                black_box($f);
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }};
    }
    let th = time!(gather_headmajor(&q, hh, tq, d_head));
    let tr = time!(gather_rowmajor(&q, hh, tq, d_head));
    let tser = time!(gather_serial(&q, hh, tq, d_head));
    println!(
        "hh={hh} tq={tq} d_head={d_head} (n_state={n_state}, {} threads)  best-of-{iters}",
        rayon::current_num_threads()
    );
    println!(
        "  byte-exact: {bad} differing of {}  [{}]",
        a.len(),
        if bad == 0 { "IDENTICAL" } else { "DIVERGENT" }
    );
    println!("  head-major (LIVE)  : {:>8.2} µs  1.00x", th * 1e6);
    println!(
        "  row-major          : {:>8.2} µs  {:.2}x  [{}]",
        tr * 1e6,
        th / tr,
        if tr < th { "WIN" } else { "loss" }
    );
    println!(
        "  serial (ref)       : {:>8.2} µs  {:.2}x",
        tser * 1e6,
        th / tser
    );
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    println!("=== SDPA gather access-order: head-major (live) vs row-major (multi-thread) ===");
    // turbo encoder self-attn: n_state=1280, n_head=20 -> d_head=64, tq=tk=1500.
    bench(20, 1500, 64, iters);
}
