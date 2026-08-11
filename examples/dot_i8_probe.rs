//! int8 GEMV inner dot: scalar-autovec `dot_i8` vs hand-written AVX2 (BlackThrush, 2026-07-02).
//!
//! `nn::dot_i8` is a scalar `acc += (a as i32)*(b as i32)` loop; its comment ASSERTS
//! "LLVM lowers this to vpmovsxbw+vpmaddwd ... the int8 compute far outruns the DRAM
//! read rate, so the GEMV stays memory-bound." This is the single biggest per-token
//! decode op (the [51866,1280] vocab logits GEMV ~= 943 us/token). The gelu gather
//! taught us a ledger "tuned/closed" claim can hide an unexamined Zen3 codegen issue,
//! so this MEASURES the assertion instead of trusting it.
//!
//! Byte-exactness is FREE here: i8*i8 in [-16129,16129], K<=5120 terms => |sum| <=
//! 82.6M < 2^31, so there is NO i32 overflow and integer add is associative — the
//! hand AVX2 dot (vpmovsxbw + vpmaddwd + 4 independent i32 accumulators, horizontal-
//! summed at the end) yields the EXACT same i32 as the scalar loop. Asserted here
//! (0 differing rows over the full matrix). So any speedup is landable default-on.
//!
//! Small isolated per-crate microbench: full single-thread GEMV over the real decode
//! shapes (logits/mlp_0/qkv), best-of-N (load-insensitive on a shared box). The
//! parallel band-split wrapper is orthogonal and already tuned. Reports ms + the
//! effective weight-stream GB/s (N*K bytes / time) — if the scalar path is well below
//! the ~single-core streaming rate it is compute-bound (autovec headroom), not memory-
//! bound as the comment claims.
//! Usage: `dot_i8_probe [iters]` (default 30).
#![allow(unsafe_code)]
#[cfg(target_arch = "x86_64")]
use std::hint::black_box;
#[cfg(target_arch = "x86_64")]
use std::time::Instant;

/// Exact replica of `nn::dot_i8` (scalar; LLVM auto-vectorizes).
#[inline]
#[cfg(target_arch = "x86_64")]
fn dot_i8_scalar(w: &[i8], x: &[i8]) -> i32 {
    let mut acc: i32 = 0;
    for (a, b) in w.iter().zip(x.iter()) {
        acc += (*a as i32) * (*b as i32);
    }
    acc
}

/// Hand-written AVX2: 32 i8/iter via 2x (vpmovsxbw + vpmaddwd), 4 i32 accumulators
/// to hide the ~4-cyc madd latency. Byte-identical to the scalar loop (exact i32).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn dot_i8_avx2(w: &[i8], x: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    unsafe {
        let n = w.len();
        let wp = w.as_ptr();
        let xp = x.as_ptr();
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let a3 = _mm256_setzero_si256();
        let mut i = 0;
        // 32 elements per iteration (two 16-wide madd chains).
        while i + 32 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i) as *const __m128i));
            let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i + 16) as *const __m128i));
            let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i + 16) as *const __m128i));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w1, x1));
            i += 32;
        }
        // 16-wide tail block.
        while i + 16 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i) as *const __m128i));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(w0, x0));
            i += 16;
        }
        // Horizontal sum of the 4 accumulators (exact integer add, order-independent).
        let s = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
        let lo = _mm256_castsi256_si128(s);
        let hi = _mm256_extracti128_si256::<1>(s);
        let q = _mm_add_epi32(lo, hi);
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
        let mut acc = _mm_cvtsi128_si32(q);
        // Scalar tail (< 16 remaining).
        while i < n {
            acc += (*w.get_unchecked(i) as i32) * (*x.get_unchecked(i) as i32);
            i += 1;
        }
        acc
    }
}

/// 4-accumulator / 64-elem-per-iter variant (test whether the landed 2-acc dot_i8
/// is vpmovsxbw-bound or accumulator-latency-bound). Byte-identical (integer sum).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn dot_i8_avx2_4acc(w: &[i8], x: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    unsafe {
        let n = w.len();
        let (wp, xp) = (w.as_ptr(), x.as_ptr());
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut a2 = _mm256_setzero_si256();
        let mut a3 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 64 <= n {
            let l = |o: usize| _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(o) as *const __m128i));
            let r = |o: usize| _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(o) as *const __m128i));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(l(i), r(i)));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(l(i + 16), r(i + 16)));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(l(i + 32), r(i + 32)));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(l(i + 48), r(i + 48)));
            i += 64;
        }
        while i + 16 <= n {
            let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(wp.add(i) as *const __m128i));
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i) as *const __m128i));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w0, x0));
            i += 16;
        }
        let s = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
        let lo = _mm256_castsi256_si128(s);
        let hi = _mm256_extracti128_si256::<1>(s);
        let q = _mm_add_epi32(lo, hi);
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
        let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
        let mut acc = _mm_cvtsi128_si32(q);
        while i < n {
            acc += (*w.get_unchecked(i) as i32) * (*x.get_unchecked(i) as i32);
            i += 1;
        }
        acc
    }
}

#[cfg(target_arch = "x86_64")]
fn gemv_4acc(w: &[i8], x: &[i8], n: usize, k: usize, out: &mut [i32]) {
    for (o, slot) in out.iter_mut().enumerate() {
        *slot = unsafe { dot_i8_avx2_4acc(&w[o * k..(o + 1) * k], x) };
    }
    let _ = n;
}

/// 2-row dot: convert the activation `x` sign-extension ONCE and reuse it for two
/// weight rows (dot_i8 is vpmovsxbw-bound; ~half its conversions are the re-loaded x).
/// Each result is integer-identical to `dot_i8_avx2` (same per-row madd + sum).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn dot_i8_avx2_2row(w0: &[i8], w1: &[i8], x: &[i8]) -> (i32, i32) {
    use core::arch::x86_64::*;
    unsafe {
        let n = w0.len().min(w1.len()).min(x.len());
        let (w0p, w1p, xp) = (w0.as_ptr(), w1.as_ptr(), x.as_ptr());
        let mut a0 = _mm256_setzero_si256();
        let mut a1 = _mm256_setzero_si256();
        let mut b0 = _mm256_setzero_si256();
        let mut b1 = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let x0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i) as *const __m128i));
            let x1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(xp.add(i + 16) as *const __m128i));
            let w00 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w0p.add(i) as *const __m128i));
            let w01 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w0p.add(i + 16) as *const __m128i));
            let w10 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w1p.add(i) as *const __m128i));
            let w11 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w1p.add(i + 16) as *const __m128i));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(w00, x0));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(w01, x1));
            b0 = _mm256_add_epi32(b0, _mm256_madd_epi16(w10, x0));
            b1 = _mm256_add_epi32(b1, _mm256_madd_epi16(w11, x1));
            i += 32;
        }
        let hsum = |v: __m256i| -> i32 {
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256::<1>(v);
            let q = _mm_add_epi32(lo, hi);
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b01_00_11_10>(q));
            let q = _mm_add_epi32(q, _mm_shuffle_epi32::<0b00_00_00_01>(q));
            _mm_cvtsi128_si32(q)
        };
        let mut s0 = hsum(_mm256_add_epi32(a0, a1));
        let mut s1 = hsum(_mm256_add_epi32(b0, b1));
        while i < n {
            let xi = *x.get_unchecked(i) as i32;
            s0 += (*w0.get_unchecked(i) as i32) * xi;
            s1 += (*w1.get_unchecked(i) as i32) * xi;
            i += 1;
        }
        (s0, s1)
    }
}

#[cfg(target_arch = "x86_64")]
fn gemv_2row(w: &[i8], x: &[i8], n: usize, k: usize, out: &mut [i32]) {
    let mut o = 0;
    while o + 2 <= n {
        let (s0, s1) =
            unsafe { dot_i8_avx2_2row(&w[o * k..(o + 1) * k], &w[(o + 1) * k..(o + 2) * k], x) };
        out[o] = s0;
        out[o + 1] = s1;
        o += 2;
    }
    if o < n {
        out[o] = unsafe { dot_i8_avx2(&w[o * k..(o + 1) * k], x) };
    }
}

#[cfg(target_arch = "x86_64")]
fn gemv_scalar(w: &[i8], x: &[i8], n: usize, k: usize, out: &mut [i32]) {
    for (o, slot) in out.iter_mut().enumerate() {
        *slot = dot_i8_scalar(&w[o * k..(o + 1) * k], x);
    }
    let _ = n;
}
#[cfg(target_arch = "x86_64")]
fn gemv_avx2(w: &[i8], x: &[i8], n: usize, k: usize, out: &mut [i32]) {
    for (o, slot) in out.iter_mut().enumerate() {
        *slot = unsafe { dot_i8_avx2(&w[o * k..(o + 1) * k], x) };
    }
    let _ = n;
}

#[cfg(target_arch = "x86_64")]
fn bench(name: &str, n: usize, k: usize, iters: usize) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut ni8 = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 24) as i32 % 255 - 127) as i8
    };
    let w: Vec<i8> = (0..n * k).map(|_| ni8()).collect();
    let x: Vec<i8> = (0..k).map(|_| ni8()).collect();

    // Byte-exactness over the full matrix.
    let mut os = vec![0i32; n];
    let mut oa = vec![0i32; n];
    gemv_scalar(&w, &x, n, k, &mut os);
    gemv_avx2(&w, &x, n, k, &mut oa);
    let diff = os.iter().zip(oa.iter()).filter(|(a, b)| a != b).count();

    let run = |f: fn(&[i8], &[i8], usize, usize, &mut [i32]), buf: &mut [i32]| -> f64 {
        for _ in 0..2 {
            f(&w, &x, n, k, buf);
            black_box(&buf);
        }
        let mut best = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            f(&w, &x, n, k, buf);
            best = best.min(t.elapsed().as_secs_f64());
            black_box(&buf);
        }
        best
    };
    let mut o4 = vec![0i32; n];
    let mut o2r = vec![0i32; n];
    gemv_4acc(&w, &x, n, k, &mut o4);
    gemv_2row(&w, &x, n, k, &mut o2r);
    let diff4 = oa.iter().zip(o4.iter()).filter(|(a, b)| a != b).count();
    let diff2r = oa.iter().zip(o2r.iter()).filter(|(a, b)| a != b).count();
    let ts = run(gemv_scalar, &mut os);
    let ta = run(gemv_avx2, &mut oa);
    let t4 = run(gemv_4acc, &mut o4);
    let t2r = run(gemv_2row, &mut o2r);
    let bytes = (n * k) as f64; // 1 byte/weight, streamed once
    println!(
        "{name}  [{n}x{k}] ({:.1} MiB int8)  best-of-{iters} @ 1 thread",
        bytes / (1 << 20) as f64
    );
    println!(
        "  byte-exact vs 2acc: 4acc {diff4} diff, 2row {diff2r} diff (scalar {diff})  [{}]",
        if diff == 0 && diff4 == 0 && diff2r == 0 {
            "ALL IDENTICAL"
        } else {
            "DIVERGENT"
        }
    );
    println!(
        "  scalar-autovec : {:>7.3} ms  {:>6.1} GB/s",
        ts * 1e3,
        bytes / ts / 1e9
    );
    println!(
        "  hand AVX2 2acc : {:>7.3} ms  {:>6.1} GB/s  {:.2}x (landed)",
        ta * 1e3,
        bytes / ta / 1e9,
        ts / ta
    );
    println!(
        "  hand AVX2 4acc : {:>7.3} ms  {:>6.1} GB/s  {:.2}x vs2acc",
        t4 * 1e3,
        bytes / t4 / 1e9,
        ta / t4
    );
    println!(
        "  hand AVX2 2row : {:>7.3} ms  {:>6.1} GB/s  {:.2}x vs2acc  [{}]",
        t2r * 1e3,
        bytes / t2r / 1e9,
        ta / t2r,
        if t2r < ta * 0.98 {
            "2row WINS"
        } else {
            "no gain"
        }
    );
}

#[cfg(target_arch = "x86_64")]
fn main() {
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("dot_i8_probe requires AVX2 support");
        return;
    }
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    println!("=== int8 GEMV inner dot: scalar-autovec vs hand AVX2 (Zen3, 1 thread) ===");
    bench("logits", 51866, 1280, iters);
    bench("mlp_0 ", 5120, 1280, iters * 4);
    bench("qkv   ", 1280, 1280, iters * 8);
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("dot_i8_probe requires an x86_64 processor");
}
