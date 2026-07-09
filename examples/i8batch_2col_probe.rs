//! `gemv_i8_batch` (nn.rs, the int8 batched GEMV for prefill tq>1 + draft) is
//! WEIGHT-OUTER: `for o { for t { dot_i8(w[o], xi8[t]) } }`. `dot_i8` sign-extends
//! BOTH operands (`vpmovsxbw`) every call, so the weight row `w[o]` — fixed across
//! the tq tokens, L1-hot — is RE-sign-extended tq times. This is the int8 analogue
//! of the redundant per-token f16→f32 cvtph that M2col fixed for the f16 batch.
//!
//! A 2-token activation-column tile (`dot_i8_2col`) sign-extends `w[o]` ONCE and
//! reuses it across 2 tokens, halving the weight sign-extend. BYTE-EXACT: the i32
//! integer dot is order-independent, and both share the SAME madd pairing/reduction.
//!
//! Distinct from the rejected int8 ROW-blocking (R concurrent WEIGHT streams, DRAM-
//! cold, tq=1): here it's ONE weight stream shared across 2 ACTIVATIONS, weight-outer,
//! prefill shapes (weight L1-hot). Prediction (int8-bandwidth-bound argument): if the
//! kernel is weight-DRAM-bound the tile is sub-noise; if sign-extend-compute-bound
//! (small tq, weight L1-hot) it wins ~like M2col. Measure it.
//!
//! Run: cargo run --release --example i8batch_2col_probe
use rayon::prelude::*;

const INP: usize = 1280;

/// i32 dot of two int8 vectors: cvtepi8_epi16 both + madd_epi16 (16 MAC/instr) into
/// one i32 accumulator lane-set, horizontal-summed. Integer-exact.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8(w: &[i8], x: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    let n = w.len().min(x.len());
    unsafe {
        let mut acc = _mm256_setzero_si256();
        let mut i = 0usize;
        while i + 16 <= n {
            let wv = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(i).cast()));
            let xv = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.as_ptr().add(i).cast()));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(wv, xv));
            i += 16;
        }
        let mut t = [0i32; 8];
        _mm256_storeu_si256(t.as_mut_ptr().cast(), acc);
        let mut s = t.iter().sum::<i32>();
        while i < n {
            s += w[i] as i32 * x[i] as i32;
            i += 1;
        }
        s
    }
}

/// One weight row, TWO activation rows: sign-extend `w[o]` ONCE per 16-chunk, reuse
/// for both tokens. Byte-identical i32 to two `dot_i8` calls (same madd pairing).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[allow(unsafe_code)]
fn dot_i8_2col(w: &[i8], x0: &[i8], x1: &[i8]) -> (i32, i32) {
    use core::arch::x86_64::*;
    let n = w.len().min(x0.len()).min(x1.len());
    unsafe {
        let mut a = _mm256_setzero_si256();
        let mut b = _mm256_setzero_si256();
        let mut i = 0usize;
        while i + 16 <= n {
            let wv = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(i).cast())); // ONCE
            let x0v = _mm256_cvtepi8_epi16(_mm_loadu_si128(x0.as_ptr().add(i).cast()));
            let x1v = _mm256_cvtepi8_epi16(_mm_loadu_si128(x1.as_ptr().add(i).cast()));
            a = _mm256_add_epi32(a, _mm256_madd_epi16(wv, x0v));
            b = _mm256_add_epi32(b, _mm256_madd_epi16(wv, x1v));
            i += 16;
        }
        let mut ta = [0i32; 8];
        let mut tb = [0i32; 8];
        _mm256_storeu_si256(ta.as_mut_ptr().cast(), a);
        _mm256_storeu_si256(tb.as_mut_ptr().cast(), b);
        let (mut s0, mut s1) = (ta.iter().sum::<i32>(), tb.iter().sum::<i32>());
        while i < n {
            s0 += w[i] as i32 * x0[i] as i32;
            s1 += w[i] as i32 * x1[i] as i32;
            i += 1;
        }
        (s0, s1)
    }
}

/// M1: exact structure of gemv_i8_batch compute_band, weight-outer over an out-band.
fn gemv_m1(w: &[i8], xi8: &[i8], out: usize, tq: usize, dst: &mut [f32], workers: usize) {
    let band = out.div_ceil(workers).max(1);
    let bands: Vec<(usize, usize)> = (0..out).step_by(band).map(|o0| (o0, (o0 + band).min(out))).collect();
    let parts: Vec<Vec<f32>> = bands.par_iter().map(|&(o0, o1)| {
        let mut local = vec![0.0f32; (o1 - o0) * tq];
        for o in o0..o1 {
            let wrow = &w[o * INP..(o + 1) * INP];
            for t in 0..tq {
                local[(o - o0) * tq + t] = dot_i8(wrow, &xi8[t * INP..(t + 1) * INP]) as f32;
            }
        }
        local
    }).collect();
    for (bi, &(o0, o1)) in bands.iter().enumerate() {
        for o in o0..o1 {
            for t in 0..tq { dst[o * tq + t] = parts[bi][(o - o0) * tq + t]; }
        }
    }
}

fn gemv_m2(w: &[i8], xi8: &[i8], out: usize, tq: usize, dst: &mut [f32], workers: usize) {
    let band = out.div_ceil(workers).max(1);
    let bands: Vec<(usize, usize)> = (0..out).step_by(band).map(|o0| (o0, (o0 + band).min(out))).collect();
    let parts: Vec<Vec<f32>> = bands.par_iter().map(|&(o0, o1)| {
        let mut local = vec![0.0f32; (o1 - o0) * tq];
        for o in o0..o1 {
            let wrow = &w[o * INP..(o + 1) * INP];
            let mut t = 0;
            while t + 2 <= tq {
                let (s0, s1) = dot_i8_2col(wrow, &xi8[t * INP..(t + 1) * INP], &xi8[(t + 1) * INP..(t + 2) * INP]);
                local[(o - o0) * tq + t] = s0 as f32;
                local[(o - o0) * tq + t + 1] = s1 as f32;
                t += 2;
            }
            if t < tq {
                local[(o - o0) * tq + t] = dot_i8(wrow, &xi8[t * INP..(t + 1) * INP]) as f32;
            }
        }
        local
    }).collect();
    for (bi, &(o0, o1)) in bands.iter().enumerate() {
        for o in o0..o1 {
            for t in 0..tq { dst[o * tq + t] = parts[bi][(o - o0) * tq + t]; }
        }
    }
}

fn ms(t: std::time::Instant) -> f64 { t.elapsed().as_secs_f64() * 1e3 }

fn main() {
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    { eprintln!("needs avx2"); return; }
    let mut s = 0x2545F4914F6CDD1Du64;
    let mut ni = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); ((s >> 40) as i8) };
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(16).min(16);
    let mut evict = vec![1.0f32; 48 * 1024 * 1024 / 4];

    // Representative prefill / draft shapes: mlp_0 [inp,5120], qkv [inp,3840]; tq small.
    for (out, label) in [(5120usize, "mlp_0[1280,5120]"), (3840, "qkv[1280,3840]")] {
        let w: Vec<i8> = (0..out * INP).map(|_| ni()).collect();
        for tq in [8usize, 64, 200] {
            let xi8: Vec<i8> = (0..tq * INP).map(|_| ni()).collect();
            let mut a = vec![0.0f32; out * tq];
            let mut b = vec![0.0f32; out * tq];
            gemv_m1(&w, &xi8, out, tq, &mut a, workers);
            gemv_m2(&w, &xi8, out, tq, &mut b, workers);
            let ident = a == b;
            let reps = 60;
            let (mut b1, mut b2) = (f64::MAX, f64::MAX);
            for _ in 0..reps {
                for e in evict.iter_mut() { *e *= 1.0000001; }
                let t = std::time::Instant::now(); gemv_m1(&w, &xi8, out, tq, &mut a, workers); b1 = b1.min(ms(t));
                for e in evict.iter_mut() { *e *= 1.0000001; }
                let t = std::time::Instant::now(); gemv_m2(&w, &xi8, out, tq, &mut b, workers); b2 = b2.min(ms(t));
            }
            std::hint::black_box(&a); std::hint::black_box(&b);
            println!("{label} tq={tq:3} {workers}t cold min-{reps}: M1={b1:.4} ms  M2col={b2:.4} ms  ({:.3}× {})  byte-id={ident}",
                b1 / b2, if b2 < b1 { "FASTER" } else { "slower" });
        }
    }
    std::hint::black_box(&evict);
}
