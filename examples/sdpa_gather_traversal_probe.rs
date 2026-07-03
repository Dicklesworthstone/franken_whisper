//! SDPA gather traversal order: h-outer (current) vs i-outer vs prefetched — byte-exact
//! (BlackThrush, 2026-07-03).
//!
//! `sdpa_gather_head_major` (nn.rs:2661) reshapes q/k/v `[t, n_state]` → head-major
//! `[hh, t, d_head]` per encoder window (the last un-cache-audited encoder franken-side op;
//! the gather+scatter is ~6% of attn_sdpa ≈ ~0.5% encoder). Current traversal is h-outer /
//! i-inner: **strided src read** (stride n_state) + **contiguous dst write**. Two byte-exact
//! alternatives: (a) i-outer / h-inner = **contiguous src read** (whole [hh·d_head] row) +
//! strided dst write; (b) current order + software prefetch of the next strided src row. Pure
//! data movement ⇒ identical output regardless of order; this measures which memory pattern is
//! fastest single-thread (isolating the pattern from shared-box thread noise). Turbo:
//! t=1500, n_state=1280, hh=20, d_head=64. Usage: `sdpa_gather_traversal_probe [iters]`.
#![allow(unsafe_code)]
use std::hint::black_box;
use std::time::Instant;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const T: usize = 1500;
const N_STATE: usize = 1280;
const HH: usize = 20;
const D_HEAD: usize = N_STATE / HH; // 64

/// Current: h-outer / i-inner — strided src read, contiguous dst write.
fn gather_h_outer(dst: &mut [f32], src: &[f32]) {
    for r in 0..HH * T {
        let (h, i) = (r / T, r % T);
        let base = i * N_STATE + h * D_HEAD;
        dst[r * D_HEAD..(r + 1) * D_HEAD].copy_from_slice(&src[base..base + D_HEAD]);
    }
}

/// i-outer / h-inner — contiguous src read (whole row), strided dst write.
fn gather_i_outer(dst: &mut [f32], src: &[f32]) {
    for i in 0..T {
        let row = &src[i * N_STATE..(i + 1) * N_STATE];
        for h in 0..HH {
            let r = h * T + i;
            dst[r * D_HEAD..(r + 1) * D_HEAD].copy_from_slice(&row[h * D_HEAD..(h + 1) * D_HEAD]);
        }
    }
}

/// Current order + software prefetch of the next strided src row.
#[cfg(target_arch = "x86_64")]
fn gather_h_outer_prefetch(dst: &mut [f32], src: &[f32]) {
    const PF: usize = 8; // rows ahead
    let sp = src.as_ptr();
    for r in 0..HH * T {
        let (h, i) = (r / T, r % T);
        let base = i * N_STATE + h * D_HEAD;
        let ni = i + PF;
        if ni < T {
            unsafe { _mm_prefetch::<_MM_HINT_T0>(sp.add(ni * N_STATE + h * D_HEAD).cast()); }
        }
        dst[r * D_HEAD..(r + 1) * D_HEAD].copy_from_slice(&src[base..base + D_HEAD]);
    }
}

fn run(f: &dyn Fn(&mut [f32], &[f32]), dst: &mut [f32], src: &[f32], iters: usize) -> f64 {
    for _ in 0..5 { f(dst, src); }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        f(dst, src);
        best = best.min(t.elapsed().as_secs_f64());
        black_box(&dst[0]);
    }
    best
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let mut nf = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 40) as f32 / (1u64 << 24) as f32 };
    let src: Vec<f32> = (0..T * N_STATE).map(|_| nf()).collect();
    let mut a = vec![0.0f32; HH * T * D_HEAD];
    let mut b = vec![0.0f32; HH * T * D_HEAD];
    gather_h_outer(&mut a, &src);
    gather_i_outer(&mut b, &src);
    let diff = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();

    println!("=== SDPA gather [1500,1280]→[20,1500,64] traversal order (1 thread, byte-exact) ===");
    println!("  byte-diff (h-outer vs i-outer): {diff}  [{}]", if diff == 0 { "IDENTICAL" } else { "DIVERGENT" });
    let th = run(&gather_h_outer, &mut a, &src, iters);
    let ti = run(&gather_i_outer, &mut b, &src, iters);
    println!("  h-outer (current: strided read, contiguous write): {:>6.1} µs", th * 1e6);
    println!("  i-outer (contiguous read, strided write)         : {:>6.1} µs  {:.2}x [{}]",
        ti * 1e6, th / ti, if ti < th { "WIN" } else { "loss" });
    #[cfg(target_arch = "x86_64")]
    {
        let mut c = vec![0.0f32; HH * T * D_HEAD];
        gather_h_outer_prefetch(&mut c, &src);
        let pdiff = a.iter().zip(&c).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let tp = run(&gather_h_outer_prefetch, &mut c, &src, iters);
        println!("  h-outer + SW prefetch (T0, 8 rows ahead)         : {:>6.1} µs  {:.2}x [{}] byte-diff={pdiff}",
            tp * 1e6, th / tp, if tp < th { "WIN" } else { "loss" });
    }
    println!("  (gather+scatter ≈ ~6% of attn_sdpa ≈ ~0.5% encoder; byte-exact traversal reorder.)");
}
