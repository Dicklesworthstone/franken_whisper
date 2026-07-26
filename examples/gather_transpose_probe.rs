// Encoder SDPA gather-transpose micro-probe. Standalone (no crate deps) — build with
//   rustc -O -C target-cpu=native --edition 2021 examples/gather_transpose_probe.rs
// or `cargo build --release --example gather_transpose_probe`.
//
// Turbo encoder SDPA gather dims: hh=20, t=1500, d_head=64, n_state=1280 (=hh*d_head).
// src interleaved [t, n_state] -> dst head-major [hh, t, d_head]:
//   dst[(h*t+i)*d_head + d] = src[i*n_state + h*d_head + d].
// ALL variants produce byte-identical dst (pure data movement) — verified + timed.
//
// FINDING (NEGATIVE_EVIDENCE 2026-07-04, BlackThrush): the current dest-order gather
// (nn.rs `sdpa_gather_head_major`) has NO byte-exact algorithmic lever.
//   * WARM: source-order (v1, contiguous reads) is ~1.3x faster than dest-order (v0)
//     single-thread — but this is a pure L2/prefetch artifact.
//   * COLD (cache evicted between reps — the REAL in-engine regime, src just written by
//     the QKV GEMM & evicted): v1 ~= v0 (~2.5%, noise) — DRAM latency dominates, read
//     order is irrelevant. v2 (tiled) is WORSE cold.
//   * In-engine chunk sweep (FW_SDPA_GATHER_CHUNKS on the real model, PERF_SPANS):
//     gather 1->428ms, 8->98ms, 16->80ms, 24->79.6, 32->85, 64->79.3 => FLAT past 16.
// The gather is cold/DRAM-latency-bound at ~80ms/window, parallelism-saturated at the
// default 16 chunks. Do not "optimize" it based on its low apparent GB/s.
//
// Measurement probe only — never linked into the library. It hand-writes the AVX2
// intrinsics it is timing, so it opts out of the workspace `unsafe_code = "deny"`
// lint the same way `native_engine::nn` does for its own kernels.
#![allow(unsafe_code)]

use std::thread;
use std::time::Instant;

const HH: usize = 20;
const T: usize = 1500;
const D: usize = 64; // d_head
const NS: usize = 1280; // n_state = HH*D

// V0: current impl — dest-order, strided READ, contiguous WRITE.
fn gather_v0(dst: &mut [f32], src: &[f32]) {
    for h in 0..HH {
        for i in 0..T {
            let base = i * NS + h * D;
            let o = (h * T + i) * D;
            dst[o..o + D].copy_from_slice(&src[base..base + D]);
        }
    }
}

// V1: source-order — contiguous READ (full src rows), strided WRITE.
fn gather_v1(dst: &mut [f32], src: &[f32]) {
    for i in 0..T {
        let row = &src[i * NS..i * NS + NS];
        for h in 0..HH {
            let o = (h * T + i) * D;
            dst[o..o + D].copy_from_slice(&row[h * D..h * D + D]);
        }
    }
}

// V2: tiled over i (block of BI rows) — read a tile of src rows, write into all heads.
// Keeps the src tile (BI*NS f32) hot while scattering to HH dst locations.
fn gather_v2(dst: &mut [f32], src: &[f32]) {
    const BI: usize = 32;
    let mut i0 = 0;
    while i0 < T {
        let i1 = (i0 + BI).min(T);
        for h in 0..HH {
            for i in i0..i1 {
                let base = i * NS + h * D;
                let o = (h * T + i) * D;
                dst[o..o + D].copy_from_slice(&src[base..base + D]);
            }
        }
        i0 = i1;
    }
}

// Multi-thread wrappers: split work into `nt` bands.
// V0-mt: bands over the flat [hh*t] output rows (== current par_chunks_mut).
fn gather_v0_mt(dst: &mut [f32], src: &[f32], nt: usize) {
    let total = HH * T;
    let per = (total + nt - 1) / nt;
    thread::scope(|s| {
        for (c, blk) in dst.chunks_mut(per * D).enumerate() {
            let row0 = c * per;
            let src = &src;
            s.spawn(move || {
                for (local, out_row) in blk.chunks_mut(D).enumerate() {
                    let r = row0 + local;
                    let h = r / T;
                    let i = r % T;
                    let base = i * NS + h * D;
                    out_row.copy_from_slice(&src[base..base + D]);
                }
            });
        }
    });
}

// V1-mt: bands over i (source rows). Each thread reads a contiguous src slab,
// writes strided across heads. dst is written at (h*T+i) — disjoint per i, so
// we can't take a simple contiguous dst chunk; use raw pointer with disjoint i.
fn gather_v1_mt(dst: &mut [f32], src: &[f32], nt: usize) {
    let per = (T + nt - 1) / nt;
    let dst_ptr = dst.as_mut_ptr() as usize;
    thread::scope(|s| {
        for c in 0..nt {
            let i0 = c * per;
            if i0 >= T {
                break;
            }
            let i1 = (i0 + per).min(T);
            let src = &src;
            s.spawn(move || {
                let dp = dst_ptr as *mut f32;
                for i in i0..i1 {
                    let row = &src[i * NS..i * NS + NS];
                    for h in 0..HH {
                        let o = (h * T + i) * D;
                        unsafe {
                            let d = std::slice::from_raw_parts_mut(dp.add(o), D);
                            d.copy_from_slice(&row[h * D..h * D + D]);
                        }
                    }
                }
            });
        }
    });
}

fn bench<F: FnMut(&mut [f32], &[f32])>(
    name: &str,
    src: &[f32],
    reps: usize,
    truth: &[f32],
    mut f: F,
) {
    let mut dst = vec![0f32; HH * T * D];
    // warm
    f(&mut dst, src);
    assert_eq!(&dst[..], truth, "{name} not byte-identical!");
    let mut best = f64::MAX;
    for _ in 0..reps {
        for v in dst.iter_mut() {
            *v = 0.0;
        }
        let t = Instant::now();
        f(&mut dst, src);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < best {
            best = e;
        }
    }
    // bytes moved: read 7.68MB + write 7.68MB = 15.36MB
    let gbps = (2.0 * (HH * T * D) as f64 * 4.0) / (best / 1e3) / 1e9;
    println!("  {name:<16} {best:8.3} ms   {gbps:6.1} GB/s");
}

// COLD bench: evict caches with a >L3 sweep before each timed rep (latency-bound,
// mimics the in-engine gather whose src was just written by the QKV GEMM & evicted).
fn bench_cold<F: FnMut(&mut [f32], &[f32])>(
    name: &str,
    src: &[f32],
    reps: usize,
    truth: &[f32],
    mut f: F,
) {
    let mut dst = vec![0f32; HH * T * D];
    let mut evict = vec![0u8; 256 * 1024 * 1024]; // 256MB >> 128MB L3
    f(&mut dst, src);
    assert_eq!(&dst[..], truth, "{name} not byte-identical!");
    let mut times = Vec::new();
    for _ in 0..reps {
        for (j, v) in evict.iter_mut().enumerate() {
            *v = (j & 0xff) as u8;
        }
        std::hint::black_box(&evict);
        let t = Instant::now();
        f(&mut dst, src);
        std::hint::black_box(&dst);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    println!("  {name:<16} {med:8.3} ms (cold median)");
}

fn main() {
    let src: Vec<f32> = (0..T * NS).map(|x| (x % 997) as f32 * 0.5).collect();
    let mut truth = vec![0f32; HH * T * D];
    gather_v0(&mut truth, &src);
    let reps = 200;
    println!("=== single-thread WARM (best of {reps}) ===");
    bench("v0_current", &src, reps, &truth, gather_v0);
    bench("v1_srcorder", &src, reps, &truth, gather_v1);
    bench("v2_tiled", &src, reps, &truth, gather_v2);
    println!("=== single-thread COLD (median of 30, cache-evicted) ===");
    bench_cold("v0_current", &src, 30, &truth, gather_v0);
    bench_cold("v1_srcorder", &src, 30, &truth, gather_v1);
    bench_cold("v2_tiled", &src, 30, &truth, gather_v2);
    for nt in [16usize, 24, 32] {
        println!("=== {nt} threads WARM (best of {reps}) ===");
        bench("v0_mt", &src, reps, &truth, |d, s| gather_v0_mt(d, s, nt));
        bench("v1_mt", &src, reps, &truth, |d, s| gather_v1_mt(d, s, nt));
    }
}
