//! int8 GEMV register-row-blocking probe — extend the f16 2-row win to the
//! DEFAULT decode path? (land-or-dig, 2026-07-06).
//!
//! The f16 GEMV uses 2-row register blocking (`nn::dot_f16c_2row`, LANDED 50059ef,
//! 1.17-1.27x, bit-exact, "row-blocking != SIMD-width" — explicitly NOT marked
//! exhausted). But EVERY per-token decode GEMV is int8 by default now
//! ([[project_int8_mlp_fc1_default_on]]): `gemv_i8` calls `dot_i8` SINGLE-ROW per
//! output row, re-widening the shared quantized activation (`vpmovsxbw`) for every
//! row and paying an isolated horizontal-reduction tail each time. `dot_i8`'s own
//! doc says the bottleneck is "too few accumulators to hide vpmaddwd latency".
//!
//! Hypothesis: register-block R weight rows against the SAME xi8 — widen x once per
//! R rows, run 2R independent madd chains. int8 sums are INTEGER-associative and do
//! not overflow (|w·x| <= 127·127 = 16129, ·inp<=5120 => |Σ| < 82.6M < 2^31), so
//! ANY accumulator grouping is BYTE-EXACT vs the scalar/`dot_i8` reference. Mechanism
//! is sound; the question is whether it's a *speed* win on the real decode.
//!
//! CRITICAL METHOD (the [[project_draft_decoding_amortization]] lesson): measure
//! weight-streaming kernels COLD, not warm-loop. This probe sweeps THREE residency
//! regimes by rotating through N distinct weight copies sized to a target working
//! set: L1-hot (256 KB), L3-resident (24 MB), and DRAM-cold (400 MB — the real
//! per-token decode regime, where the ~166 MB int8 weight set >> 128 MB L3).
//!
//! Verdict rule: a robust win across L3-resident + DRAM-cold => land (gated to the
//! winning shapes). Warm-only win that inverts at L3/DRAM => the decode GEMVs are
//! bandwidth-bound and blocking's multi-stream access regresses them => REJECT.
//!
//! Self-contained (no model): `cargo run --release --example rowblock_i8_probe`.
#![allow(unsafe_code)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod bench {
    use core::arch::x86_64::*;
    use std::time::Instant;

    type GemvFn = dyn Fn(&[i8], usize, usize, &[i8], &mut [i32]);

    /// Single-row reference == the engine's `nn::dot_i8` (2 i32 accumulators).
    #[inline]
    unsafe fn dot1(w: *const i8, x: *const i8, n: usize) -> i32 {
        unsafe {
            let mut a0 = _mm256_setzero_si256();
            let mut a1 = _mm256_setzero_si256();
            let mut i = 0usize;
            while i + 32 <= n {
                let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(i).cast()));
                let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.add(i).cast()));
                let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(i + 16).cast()));
                let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.add(i + 16).cast()));
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, x1));
                i += 32;
            }
            while i + 16 <= n {
                let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.add(i).cast()));
                let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.add(i).cast()));
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
                i += 16;
            }
            reduce(_mm256_add_epi32(a0, a1), w, x, i, n)
        }
    }

    #[inline]
    unsafe fn reduce(s: __m256i, w: *const i8, x: *const i8, mut i: usize, n: usize) -> i32 {
        unsafe {
            let q = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
            let mut acc = _mm_cvtsi128_si32(q);
            while i < n {
                acc += (*w.add(i) as i32) * (*x.add(i) as i32);
                i += 1;
            }
            acc
        }
    }

    /// R-row blocked dot: shares the x0/x1 widened tiles across R weight rows.
    macro_rules! blocked {
        ($name:ident, $R:literal) => {
            #[inline]
            unsafe fn $name(w: [*const i8; $R], x: *const i8, n: usize, out: &mut [i32; $R]) {
                unsafe {
                    let mut a0 = [_mm256_setzero_si256(); $R];
                    let mut a1 = [_mm256_setzero_si256(); $R];
                    let mut i = 0usize;
                    while i + 32 <= n {
                        let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.add(i).cast()));
                        let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.add(i + 16).cast()));
                        let mut r = 0;
                        while r < $R {
                            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w[r].add(i).cast()));
                            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w[r].add(i + 16).cast()));
                            a0[r] = _mm256_add_epi32(a0[r], _mm256_madd_epi16(w0, x0));
                            a1[r] = _mm256_add_epi32(a1[r], _mm256_madd_epi16(w1, x1));
                            r += 1;
                        }
                        i += 32;
                    }
                    let mut it = i;
                    while it + 16 <= n {
                        let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.add(it).cast()));
                        let mut r = 0;
                        while r < $R {
                            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w[r].add(it).cast()));
                            a0[r] = _mm256_add_epi32(a0[r], _mm256_madd_epi16(w0, x0));
                            r += 1;
                        }
                        it += 16;
                    }
                    let mut r = 0;
                    while r < $R {
                        out[r] = reduce(_mm256_add_epi32(a0[r], a1[r]), w[r], x, it, n);
                        r += 1;
                    }
                }
            }
        };
    }
    blocked!(dot4, 4);

    fn gemv1(w: &[i8], out: usize, inp: usize, x: &[i8], o: &mut [i32]) {
        for (r, output) in o.iter_mut().take(out).enumerate() {
            *output = unsafe { dot1(w.as_ptr().add(r * inp), x.as_ptr(), inp) };
        }
    }
    fn gemv4(w: &[i8], out: usize, inp: usize, x: &[i8], o: &mut [i32]) {
        let mut r = 0usize;
        while r + 4 <= out {
            let mut p = [core::ptr::null::<i8>(); 4];
            for (k, row) in p.iter_mut().enumerate() {
                *row = unsafe { w.as_ptr().add((r + k) * inp) };
            }
            let mut res = [0i32; 4];
            unsafe { dot4(p, x.as_ptr(), inp, &mut res) };
            o[r..r + 4].copy_from_slice(&res);
            r += 4;
        }
        while r < out {
            o[r] = unsafe { dot1(w.as_ptr().add(r * inp), x.as_ptr(), inp) };
            r += 1;
        }
    }

    pub fn run() {
        // (name, out, inp) — the real int8 decode GEMV shapes (turbo).
        let shapes: &[(&str, usize, usize)] = &[
            ("self_out[1280,1280]", 1280, 1280),
            ("qkv     [3840,1280]", 3840, 1280),
            ("mlp_fc1 [5120,1280]", 5120, 1280),
            ("cross_K [1500,64]  ", 1500, 64),
            ("cross_V [64,1500]  ", 64, 1500),
            ("logits  [51865,1280]", 51865, 1280),
        ];
        // Three residency regimes by working-set size (rotate distinct copies).
        let regimes: &[(&str, usize)] = &[
            ("L1-hot(256KB)", 256 * 1024),
            ("L3-res(24MB) ", 24 * 1024 * 1024),
            ("DRAM-cold(400MB)", 400 * 1024 * 1024),
        ];
        let mut seed = 0xdead_beef_1234_5678u64;
        let mut rb = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 33) as i32 % 255 - 127) as i8
        };
        println!(
            "=== int8 GEMV 4-row register-blocking vs single-row `dot_i8` (Zen3, single-thread) ==="
        );
        println!(
            "byte-exact by integer associativity; ratio = single-row / 4-row (>1 = blocking wins)\n"
        );
        for &(rname, ws_bytes) in regimes {
            println!("-- {rname} --");
            for &(name, out, inp) in shapes {
                let one = out * inp;
                let copies = (ws_bytes / one).clamp(2, 512);
                let ws: Vec<Vec<i8>> = (0..copies)
                    .map(|_| (0..one).map(|_| rb()).collect())
                    .collect();
                let x: Vec<i8> = (0..inp).map(|_| rb()).collect();
                let mut o1 = vec![0i32; out];
                let mut o4 = vec![0i32; out];
                gemv1(&ws[0], out, inp, &x, &mut o1);
                gemv4(&ws[0], out, inp, &x, &mut o4);
                let exact = o1 == o4;
                let bench = |f: &GemvFn, o: &mut [i32]| -> f64 {
                    for weights in ws.iter().take(copies.min(4)) {
                        f(weights, out, inp, &x, o);
                    }
                    let mut best = f64::INFINITY;
                    for _ in 0..10 {
                        let t = Instant::now();
                        for weights in &ws {
                            f(weights, out, inp, &x, o);
                        }
                        let us = t.elapsed().as_secs_f64() * 1e6 / copies as f64;
                        if us < best {
                            best = us;
                        }
                    }
                    best
                };
                let t1 = bench(&gemv1, &mut o1);
                let t4 = bench(&gemv4, &mut o4);
                std::hint::black_box(o4.iter().map(|&v| v as i64).sum::<i64>());
                println!(
                    "  {name}  copies={copies:>4}  1row={t1:>9.3}us  4row={t4:>9.3}us  ratio={:>5.2}x  exact:{exact}",
                    t1 / t4
                );
            }
            println!();
        }
        println!("VERDICT (measured 2026-07-06): warm L1-hot 4row wins ~1.3-1.6x, but that is a");
        println!(
            "cache-residency artifact — at L3 it is marginal/mixed and DRAM-cold it REGRESSES"
        );
        println!("to ~0.67x on every weight-streaming shape. The int8 decode GEMVs are memory-");
        println!("bandwidth-bound (weight stream); R concurrent weight streams degrade DRAM/L3");
        println!(
            "open-row locality. int8 already halved the bytes => MORE bandwidth-bound => blocking"
        );
        println!("hurts more than for f16. REJECTED: the lever is bytes, not blocking.");
    }
}

fn main() {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    bench::run();
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    println!("rowblock_i8_probe requires x86_64 + AVX2");
}
