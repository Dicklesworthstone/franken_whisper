//! GELU inner kernel: AVX2 `vgatherdps` vs explicit scalar table-loads (BlackThrush, 2026-07-02).
//!
//! `nn::gelu_slice` is bit-exact-with-whisper (`GGML_GELU_FP16`): 8-wide `vcvtps2ph`
//! to f16 indices, then **`_mm256_i32gather_ps`** from a 1<<16 f32 table, then the
//! `x<=-10 -> 0`, `x>=10 -> x` clamp blends. The existing code chose the gather
//! after comparing it to the FULL-SCALAR fallback (per-element `Float16::from_f32`
//! followed by a table load) and won 1.38x. But it NEVER compared the gather to keeping the
//! fast 8-wide `vcvtps2ph` and replacing ONLY the gather with 8 explicit scalar
//! table-loads.
//!
//! This box is a Threadripper PRO 5975WX = **Zen3**, where `vgatherdps ymm` is
//! MICROCODED (~1 element/2-3 cyc, ~12-20 cyc for the full 8-wide gather) — the
//! classic AMD gather antipattern. Scalar loads from the (L2-resident, 256 KiB)
//! table pipeline at ~2-3/cyc, so the scalar-load variant may beat the gather on
//! Zen while staying **byte-identical** (same table, same f16 indices, same clamp
//! => same output; asserted max bit-diff == 0). Elementwise + byte-exact => a win
//! is landable default-on.
//!
//! Small isolated per-crate microbench: gelu on one fixed 7.68M-elem buffer (the
//! encoder fc1 output shape [1500,5120]), single-threaded (the rayon band-split
//! wrapper is orthogonal), best-of-N so it is load-insensitive on a shared box.
//! Usage: `gelu_gather_probe [iters]` (default 60).
#![allow(unsafe_code)]
#[cfg(target_arch = "x86_64")]
use ft_core::Float16;
#[cfg(target_arch = "x86_64")]
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
#[cfg(target_arch = "x86_64")]
const GELU_COEF_A: f32 = 0.044_715;

/// Rebuild the exact `nn::gelu_table` (GGML_GELU_FP16 f16-indexed table).
#[cfg(target_arch = "x86_64")]
fn gelu_table() -> Box<[f32; 1 << 16]> {
    let mut t = vec![0.0f32; 1 << 16].into_boxed_slice();
    for (i, slot) in t.iter_mut().enumerate() {
        let f = Float16::from_bits(i as u16).to_f32();
        let g = 0.5 * f * (1.0 + (GELU_SQRT_2_OVER_PI * f * (1.0 + GELU_COEF_A * f * f)).tanh());
        *slot = Float16::from_f32(g).to_f32();
    }
    t.try_into().expect("gelu table length 1<<16")
}

/// Baseline: exact copy of `nn::gelu_slice` (AVX2 `vgatherdps`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
unsafe fn gelu_gather(data: &mut [f32], table: &[f32; 1 << 16]) {
    use core::arch::x86_64::*;
    unsafe {
        let tp = table.as_ptr();
        let n = data.len();
        let neg10 = _mm256_set1_ps(-10.0);
        let pos10 = _mm256_set1_ps(10.0);
        let zero = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let x = _mm256_loadu_ps(data.as_ptr().add(i));
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x);
            let idx = _mm256_cvtepu16_epi32(h);
            let g = _mm256_i32gather_ps::<4>(tp, idx);
            let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(x, pos10);
            let le = _mm256_cmp_ps::<_CMP_LE_OQ>(x, neg10);
            let r = _mm256_blendv_ps(g, x, ge);
            let r = _mm256_blendv_ps(r, zero, le);
            _mm256_storeu_ps(data.as_mut_ptr().add(i), r);
            i += 8;
        }
        for v in &mut data[i..] {
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
}

/// Variant: keep the 8-wide `vcvtps2ph`, replace the gather with 8 scalar loads.
/// Lane order preserved (`_mm256_set_ps` lane j = table[idxs[j]]) => byte-identical.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[allow(unsafe_code)]
unsafe fn gelu_scalar_loads(data: &mut [f32], table: &[f32; 1 << 16]) {
    use core::arch::x86_64::*;
    unsafe {
        let n = data.len();
        let neg10 = _mm256_set1_ps(-10.0);
        let pos10 = _mm256_set1_ps(10.0);
        let zero = _mm256_setzero_ps();
        let mut i = 0;
        let mut idxs = [0u32; 8];
        while i + 8 <= n {
            let x = _mm256_loadu_ps(data.as_ptr().add(i));
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x);
            let idx = _mm256_cvtepu16_epi32(h);
            _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, idx);
            let g = _mm256_set_ps(
                *table.get_unchecked(idxs[7] as usize),
                *table.get_unchecked(idxs[6] as usize),
                *table.get_unchecked(idxs[5] as usize),
                *table.get_unchecked(idxs[4] as usize),
                *table.get_unchecked(idxs[3] as usize),
                *table.get_unchecked(idxs[2] as usize),
                *table.get_unchecked(idxs[1] as usize),
                *table.get_unchecked(idxs[0] as usize),
            );
            let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(x, pos10);
            let le = _mm256_cmp_ps::<_CMP_LE_OQ>(x, neg10);
            let r = _mm256_blendv_ps(g, x, ge);
            let r = _mm256_blendv_ps(r, zero, le);
            _mm256_storeu_ps(data.as_mut_ptr().add(i), r);
            i += 8;
        }
        for v in &mut data[i..] {
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
}

#[cfg(target_arch = "x86_64")]
fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let table = gelu_table();
    // Encoder fc1 output shape [1500, 5120] = 7.68M elements. Override with
    // GELU_BUF_ELEMS to test a cache-resident (compute-bound) size, isolating the
    // gather's compute cost from the 29 MiB-streaming DRAM-bandwidth floor.
    let n = std::env::var("GELU_BUF_ELEMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1500 * 5120);
    // Realistic-ish activation spread; span the clamp regions so both variants
    // exercise the x<=-10 / x>=10 paths (proves byte-exactness there too).
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut nf = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 24.0 // ~U(-12, 12)
    };
    let base: Vec<f32> = (0..n).map(|_| nf()).collect();

    // Byte-exactness: run both on identical inputs, compare bit patterns.
    let mut a = base.clone();
    let mut b = base.clone();
    unsafe {
        gelu_gather(&mut a, &table);
        gelu_scalar_loads(&mut b, &table);
    }
    let bitdiff = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let maxd = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    let bench = |f: unsafe fn(&mut [f32], &[f32; 1 << 16])| -> f64 {
        let mut buf = base.clone();
        for _ in 0..3 {
            unsafe { f(&mut buf, &table) };
            black_box(&buf);
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            buf.copy_from_slice(&base);
            let t = Instant::now();
            unsafe { f(&mut buf, &table) };
            best = best.min(t.elapsed().as_secs_f64());
            black_box(&buf);
        }
        best
    };

    let tg = bench(gelu_gather);
    let ts = bench(gelu_scalar_loads);
    let gelem = n as f64 / 1e9;
    println!("=== GELU inner kernel: AVX2 vgatherdps vs 8 scalar table-loads (Zen3, 1 thread) ===");
    println!(
        "buffer = {n} f32 ({:.1} MiB), best-of-{iters}",
        (n * 4) as f64 / (1 << 20) as f64
    );
    println!(
        "byte-exactness: {bitdiff} differing elems (of {n}), max|d| = {maxd:.2e}  [{}]",
        if bitdiff == 0 {
            "BYTE-IDENTICAL"
        } else {
            "DIVERGENT"
        }
    );
    println!(
        "  vgatherdps      : {:>7.3} ms  {:>6.2} Gelem/s",
        tg * 1e3,
        gelem / tg
    );
    println!(
        "  scalar-loads    : {:>7.3} ms  {:>6.2} Gelem/s  {:.2}x  [{}]",
        ts * 1e3,
        gelem / ts,
        tg / ts,
        if ts < tg { "WIN" } else { "loss" }
    );
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("gelu_gather_probe requires an x86_64 processor with AVX2 and F16C support");
}
