//! Probe: is the load-time f16→f32 weight dequant (`ggml::dequant_f16_parallel`,
//! currently a SCALAR per-element `half::f16::to_f32` loop) compute-bound (so the
//! SIMD `HalfFloatSliceExt::convert_to_f32_slice`, which uses F16C `vcvtph2ps` on
//! x86, would win) or already bandwidth-bound (wash)?
//!
//! Both paths are parallelised across 8 workers (matching production's `min(8)`),
//! start from the same `&[f16]`, and MUST produce bit-identical `f32` (f16→f32 is
//! lossless; the `half` crate exhaustively bit-tests `convert_to_f32_slice` vs the
//! scalar `to_f32`, so any diff here is a probe bug). Reports best-of-N wall time.
//!
//! Run (build remote via rch, RUN LOCAL — example bins sync back):
//!   cargo run --release --example f16_dequant_probe

use std::time::Instant;

use half::f16;
use half::slice::HalfFloatSliceExt;

const N: usize = 100_000_000; // 100M f16 = 200 MB in, 400 MB f32 out — big-tensor scale
const WORKERS: usize = 8; // matches dequant_f16_parallel's `min(8)`
const REPS: usize = 7;

fn scalar_parallel(src: &[f16], dst: &mut [f32]) {
    let chunk = src.len().div_ceil(WORKERS);
    std::thread::scope(|s| {
        for (sc, dc) in src.chunks(chunk).zip(dst.chunks_mut(chunk)) {
            s.spawn(move || {
                for (o, &f) in dc.iter_mut().zip(sc) {
                    *o = f.to_f32();
                }
            });
        }
    });
}

fn simd_parallel(src: &[f16], dst: &mut [f32]) {
    let chunk = src.len().div_ceil(WORKERS);
    std::thread::scope(|s| {
        for (sc, dc) in src.chunks(chunk).zip(dst.chunks_mut(chunk)) {
            s.spawn(move || sc.convert_to_f32_slice(dc));
        }
    });
}

fn best_ms(label: &str, src: &[f16], dst: &mut [f32], f: fn(&[f16], &mut [f32])) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let t = Instant::now();
        f(src, dst);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        best = best.min(ms);
    }
    eprintln!("  {label:<20} best {best:.2} ms  ({:.2} GB/s in+out)", (N as f64 * 6.0) / (best / 1e3) / 1e9);
    best
}

fn main() {
    // Deterministic weight-like f16 values (finite, no NaN/inf) via an LCG → f32 → f16.
    let mut st = 0x2545_F491_4F6C_DD1Du64;
    let mut nf = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        ((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2 // ~[-0.1, 0.1], weight-scale
    };
    eprintln!("building {N} f16 values …");
    let src: Vec<f16> = (0..N).map(|_| f16::from_f32(nf())).collect();
    let mut a = vec![0.0f32; N];
    let mut b = vec![0.0f32; N];

    // Byte-exactness FIRST (both start from the same f16).
    scalar_parallel(&src, &mut a);
    simd_parallel(&src, &mut b);
    let diffs = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    eprintln!("byte-exactness: {diffs} differing bit-patterns of {N} (MUST be 0)");
    assert_eq!(diffs, 0, "SIMD convert diverged from scalar to_f32");

    eprintln!("timing (best-of-{REPS}, {WORKERS} workers):");
    let s = best_ms("scalar to_f32", &src, &mut a, scalar_parallel);
    let v = best_ms("SIMD convert", &src, &mut b, simd_parallel);
    eprintln!(
        "SPEEDUP scalar/SIMD = {:.2}×  (>1 ⇒ SIMD faster ⇒ dequant is compute-bound ⇒ real load lever)",
        s / v
    );
}
