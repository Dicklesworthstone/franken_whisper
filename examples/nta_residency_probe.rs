//! NTA-residency feasibility probe for the "keep decoder weights L3-resident by
//! streaming the logits non-cachingly" decode lever. Per token the decode reads the
//! ~92 MB decoder-layer weights then the 66 MB logits weight; the logits read has NO
//! reuse yet evicts the decoder weights from L3 (working set 158 MB > 128 MB L3), so
//! the next token re-reads decoder weights from DRAM. IF the logits stream could
//! bypass/limit L3 pollution, the decoder weights would stay resident.
//!
//! This tests whether Zen3 can do that: allocate R (resident, smaller than the CCD
//! L3) + S (stream, larger than L3), and measure R's read bandwidth after streaming
//! S three ways — normal loads,
//! MOVNTDQA (`_mm256_stream_load`, the x86 NT load — HINT IGNORED on WB memory), and
//! PREFETCHNTA-hinted loads. If any keeps R L3-warm (R BW stays high) the lever is
//! alive; if all pollute (R BW drops to DRAM) it's dead. Single-thread, one CCD's L3.
#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::hint::black_box;
use std::time::Instant;

const MB: usize = 1 << 20;

fn sum_normal(buf: &[i8]) -> i64 {
    // 32-wide accumulate to keep it memory-bound, not add-latency-bound
    let mut acc = [0i64; 4];
    let ch = buf.len() / 4;
    for j in 0..4 {
        let s = &buf[j * ch..(j + 1) * ch];
        let mut a = 0i64;
        for &b in s {
            a += b as i64;
        }
        acc[j] = a;
    }
    acc.iter().sum()
}

/// Read `buf` via MOVNTDQA (`_mm256_stream_load_si256`). On WB memory the NT hint is
/// architecturally IGNORED (behaves as a normal cached load) — this measures whether
/// that is actually so on this Zen3.
#[cfg(target_arch = "x86_64")]
fn sum_stream(buf: &[i8]) -> i64 {
    unsafe {
        let n = buf.len();
        let p = buf.as_ptr();
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            let v = _mm256_stream_load_si256(p.add(i) as *const __m256i);
            // widen i8->i16->i32 pairs and accumulate (values tiny, no overflow over this test)
            let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lo, _mm256_set1_epi16(1)));
            i += 32;
        }
        let mut t = [0i32; 8];
        _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, acc);
        t.iter().map(|&x| x as i64).sum()
    }
}

/// Read `buf` with PREFETCHNTA ahead (non-temporal prefetch hint) + normal loads.
#[cfg(target_arch = "x86_64")]
fn sum_nta(buf: &[i8]) -> i64 {
    unsafe {
        let n = buf.len();
        let p = buf.as_ptr();
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        while i + 32 <= n {
            if i + 512 < n {
                _mm_prefetch::<_MM_HINT_NTA>(p.add(i + 512) as *const i8);
            }
            let v = _mm256_loadu_si256(p.add(i) as *const __m256i);
            let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(lo, _mm256_set1_epi16(1)));
            i += 32;
        }
        let mut t = [0i32; 8];
        _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, acc);
        t.iter().map(|&x| x as i64).sum()
    }
}

fn best_read_bw(r: &[i8], iters: usize, f: &dyn Fn(&[i8]) -> i64) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let t0 = Instant::now();
        black_box(f(r));
        best = best.min(t0.elapsed().as_secs_f64());
    }
    r.len() as f64 / best / 1e9
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let r_mb = 24usize; // resident (decoder-share), < 32 MB CCD L3
    let s_mb = 96usize; // stream (logits + overflow), > L3
    let r: Vec<i8> = (0..r_mb * MB).map(|i| (i as i8).wrapping_mul(3)).collect();
    let s: Vec<i8> = (0..s_mb * MB).map(|i| (i as i8).wrapping_mul(7)).collect();
    println!(
        "== NTA residency: R={r_mb}MB (resident target) S={s_mb}MB (stream), best-of-{iters}, 1 thread =="
    );

    // Baseline: R alone, kept warm (repeated reads). Upper bound on R read BW (L3/L2).
    let warm = best_read_bw(&r, iters, &sum_normal);
    println!("  R warm (L3-resident)              {:6.1} GB/s", warm);

    // After a normal full-S read (should evict R => R goes DRAM-cold on next read).
    let mut bw_after_normal = f64::INFINITY;
    for _ in 0..iters {
        black_box(sum_normal(&s)); // pollute
        let t0 = Instant::now();
        black_box(sum_normal(&r));
        bw_after_normal = bw_after_normal.min(t0.elapsed().as_secs_f64());
    }
    let bw_after_normal = r.len() as f64 / bw_after_normal / 1e9;
    println!(
        "  R after normal-S read (baseline)  {:6.1} GB/s",
        bw_after_normal
    );

    // MOVNTDQA (_mm256_stream_load) needs 32B alignment (Vec<i8> is 1B-aligned => #GP);
    // and on WB memory the NT-load hint is architecturally IGNORED (== normal load), so it
    // is uninformative. The decisive test is PREFETCHNTA, below.
    #[cfg(target_arch = "x86_64")]
    {
        let mut bw2 = f64::INFINITY;
        for _ in 0..iters {
            black_box(sum_nta(&s));
            let t0 = Instant::now();
            black_box(sum_normal(&r));
            bw2 = bw2.min(t0.elapsed().as_secs_f64());
        }
        let bw2 = r.len() as f64 / bw2 / 1e9;
        println!(
            "  R after PREFETCHNTA-S read        {:6.1} GB/s  ({})",
            bw2,
            if bw2 > 1.3 * bw_after_normal {
                "KEEPS R WARM => LEVER ALIVE"
            } else {
                "pollutes => lever dead"
            }
        );
    }
    black_box((r, s));
}
