//! Does int8/Q8 weight quantization ~2× the memory-bound decoder MLP GEMV?
//!
//! The per-token decode MLP is two linears — fc1 `[4·n_state, n_state]` and fc2
//! `[n_state, 4·n_state]` (large-v3-turbo: `[5120,1280]` + `[1280,5120]` f16 =
//! 26.2 MB/token, ~28% of decode). Like the logits stream it is DRAM-bandwidth-
//! bound in the m=1 regime, so halving the weight bytes (int8) should ~2× it. This
//! probes that DIRECTLY on BOTH MLP shapes: single-threaded (isolates bytes/
//! bandwidth from rayon dispatch), best-of-N, f16c dot vs an autovectorized int8
//! dot — the same instrument as `int8_logits_probe`.
//! Usage: `int8_mlp_probe [iters]` (default 60).
use ft_core::Float16;
use std::hint::black_box;
use std::time::Instant;

/// f16c fused dot — a copy of `nn::dot_f16c` (4 accumulators, cvtph inline).
#[cfg(all(target_arch = "x86_64", target_feature = "f16c", target_feature = "fma"))]
#[allow(unsafe_code)]
fn dot_f16c(w: &[Float16], x: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = w.len().min(x.len());
    let xp = x.as_ptr();
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 32 <= n {
            let w0 = _mm_loadu_si128(w.as_ptr().add(i).cast());
            let w1 = _mm_loadu_si128(w.as_ptr().add(i + 8).cast());
            let w2 = _mm_loadu_si128(w.as_ptr().add(i + 16).cast());
            let w3 = _mm_loadu_si128(w.as_ptr().add(i + 24).cast());
            a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(w0), _mm256_loadu_ps(xp.add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(w1), _mm256_loadu_ps(xp.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtph_ps(w2), _mm256_loadu_ps(xp.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtph_ps(w3), _mm256_loadu_ps(xp.add(i + 24)), a3);
            i += 32;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let mut t = [0.0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), acc);
        let mut s = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]));
        while i < n {
            s += w[i].to_f32() * x[i];
            i += 1;
        }
        s
    }
}

/// Signed int8 dot — LLVM autovectorizes to `vpmovsxbw`+`vpmaddwd` under
/// x86-64-v3; compute is far above the DRAM read rate so it stays bandwidth-bound.
fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    for (a, b) in w.iter().zip(x.iter()) {
        acc += (*a as i32) * (*b as i32);
    }
    acc
}

fn best_of(iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..2 {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    best
}

/// Best-of with L3 EVICTION before each timed rep: stream a >L3 buffer (untimed)
/// so the weights are DRAM-resident when `f()` runs — the REAL per-token decode
/// regime (working set 4×MLP + logits ≈ 250 MB ≫ 128 MB L3), unlike the hot loop.
fn best_of_cold(iters: usize, pollute: &[f32], mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let mut s = 0.0f32;
        let mut j = 0;
        while j < pollute.len() {
            s += pollute[j];
            j += 16; // touch one per cache line, sweep the whole >L3 buffer
        }
        black_box(s);
        let t0 = Instant::now();
        f();
        let dt = t0.elapsed().as_secs_f64();
        if dt < best {
            best = dt;
        }
    }
    best
}

fn probe(label: &str, out: usize, inp: usize, iters: usize) {
    // f16 weights + f32 activation (the current path). Structured but non-cancelling.
    let wf16: Vec<Float16> = (0..out * inp)
        .map(|i| Float16::from_f32((((i * 2654435761) % 211) as f32 / 211.0) - 0.5))
        .collect();
    let xf: Vec<f32> = (0..inp).map(|i| ((i % 97) as f32 / 97.0) - 0.5).collect();
    // int8 weights (per-row symmetric quant) + int8 activation.
    let mut wi8: Vec<i8> = vec![0; out * inp];
    let mut wscale: Vec<f32> = vec![0.0; out];
    for o in 0..out {
        let row = &wf16[o * inp..(o + 1) * inp];
        let amax = row.iter().map(|h| h.to_f32().abs()).fold(0.0f32, f32::max).max(1e-9);
        let s = amax / 127.0;
        wscale[o] = s;
        for i in 0..inp {
            wi8[o * inp + i] = (row[i].to_f32() / s).round().clamp(-127.0, 127.0) as i8;
        }
    }
    let xamax = xf.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    let xs = xamax / 127.0;
    let xi8: Vec<i8> = xf.iter().map(|v| (v / xs).round().clamp(-127.0, 127.0) as i8).collect();

    let mut buf = vec![0.0f32; out];
    // >L3 eviction buffer (256 MB f32) to force DRAM-resident weights (real decode).
    let pollute: Vec<f32> = (0..64 * 1024 * 1024).map(|i| (i & 0xff) as f32).collect();

    let f16_ms = best_of(iters, || {
        for o in 0..out {
            buf[o] = dot_f16c(&wf16[o * inp..(o + 1) * inp], &xf);
        }
        black_box(&buf);
    }) * 1e3;

    let i8_ms = best_of(iters, || {
        for o in 0..out {
            buf[o] = dot_i8(&wi8[o * inp..(o + 1) * inp], &xi8) as f32 * wscale[o] * xs;
        }
        black_box(&buf);
    }) * 1e3;

    // COLD (DRAM-resident weights): the real per-token decode regime.
    let f16_cold = best_of_cold(iters, &pollute, || {
        for o in 0..out {
            buf[o] = dot_f16c(&wf16[o * inp..(o + 1) * inp], &xf);
        }
        black_box(&buf);
    }) * 1e3;

    let i8_cold = best_of_cold(iters, &pollute, || {
        for o in 0..out {
            buf[o] = dot_i8(&wi8[o * inp..(o + 1) * inp], &xi8) as f32 * wscale[o] * xs;
        }
        black_box(&buf);
    }) * 1e3;
    println!(
        "  COLD (DRAM-resident): f16 {f16_cold:6.3} ms  int8 {i8_cold:6.3} ms  speedup={:.2}x",
        f16_cold / i8_cold
    );

    let f16_gb = (out * inp * 2) as f64 / (f16_ms / 1e3) / 1e9;
    let i8_gb = (out * inp) as f64 / (i8_ms / 1e3) / 1e9;
    println!("{label} [{out},{inp}] single-thread best-of-{iters}");
    println!("  f16  {f16_ms:6.3} ms  {f16_gb:5.1} GB/s");
    println!("  int8 {i8_ms:6.3} ms  {i8_gb:5.1} GB/s   speedup={:.2}x", f16_ms / i8_ms);
    let mut maxrel = 0.0f32;
    for o in 0..out.min(2000) {
        let f = dot_f16c(&wf16[o * inp..(o + 1) * inp], &xf);
        let q = dot_i8(&wi8[o * inp..(o + 1) * inp], &xi8) as f32 * wscale[o] * xs;
        let rel = (f - q).abs() / f.abs().max(1e-6);
        if rel > maxrel {
            maxrel = rel;
        }
    }
    println!("  int8 max rel err (first 2000 rows) = {maxrel:.4}");
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    // large-v3-turbo MLP: n_state=1280, mlp_hidden=4*1280=5120.
    probe("mlp.fc1", 5120, 1280, iters);
    probe("mlp.fc2", 1280, 5120, iters);
}
