#![allow(unsafe_code)] // probe-only: mmap/madvise for the huge-page A/B (crate stays deny-unsafe)
//! Huge-page (2 MB, THP/madvise) weight-streaming probe for the bandwidth-bound
//! decode GEMVs (land-or-dig, 2026-07-05).
//!
//! MOTIVATION (measured, not assumed): `perf stat` on the REAL turbo decode shows
//! **dTLB-load-misses = 17.76% of all dTLB accesses** (1.3 B misses), IPC 0.50 —
//! classic TLB thrashing. The decoder streams ~158 MB of int8 weights PER TOKEN
//! across ~40 K 4 KB pages, but the L2 dTLB holds only ~2 K entries, so the page
//! table is walked constantly. With 2 MB pages the same 158 MB is ~79 pages — it
//! fits the TLB, eliminating the walks. This is a SYSTEMS primitive (not arithmetic):
//! IDENTICAL bytes, IDENTICAL math → byte-exact. THP mode here is `[madvise]` and
//! AnonHugePages=0, so plain `Vec` weights get NO huge pages; an explicit
//! `madvise(MADV_HUGEPAGE)` on the weight blob is the (load-time, one-shot) lever.
//!
//! This probe A/Bs the crate's real `nn::gemv_i8` over the SAME int8 weights backed
//! by (A) a normal `Vec` (4 KB pages) vs (B) a 2 MB-aligned `mmap`+`MADV_HUGEPAGE`
//! buffer, at the real turbo decode streaming shapes. Run it under
//!   perf stat -e ls_l1_d_tlb_miss.tlb_reload_4k_l2_miss,ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit
//! to confirm the 4 KB page-walks convert to 2 MB reloads.
//!
//! Run at RAYON_NUM_THREADS matching decode (probe uses whatever rayon picks; the
//! logits GEMV parallelizes). Usage: `hugepage_gemv_probe [iters]` (default 60).
use franken_whisper::native_engine::nn;
use franken_whisper::native_engine::nn::I8Mat;
use std::hint::black_box;
use std::time::Instant;

/// Allocate a 2 MB-aligned, MADV_HUGEPAGE-advised copy of `src` as a `Vec<i8>`.
/// SAFETY: the returned Vec is backed by `mmap`, NOT the global allocator — it must
/// be `mem::forget`-leaked, never dropped/freed. This is a probe; leaking is fine.
fn hugepage_copy_i8(src: &[i8]) -> Vec<i8> {
    let len = src.len();
    const ALIGN: usize = 2 * 1024 * 1024;
    let map_len = len + ALIGN;
    unsafe {
        let raw = libc::mmap(
            std::ptr::null_mut(),
            map_len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        assert!(raw != libc::MAP_FAILED, "mmap failed");
        let aligned = ((raw as usize + ALIGN - 1) & !(ALIGN - 1)) as *mut u8;
        // Request transparent huge pages for the aligned region.
        libc::madvise(aligned as *mut libc::c_void, len, libc::MADV_HUGEPAGE);
        // Fault every byte in (copy the weights) so THP collapses to 2 MB pages now.
        std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, aligned, len);
        Vec::from_raw_parts(aligned as *mut i8, len, len)
    }
}

fn make_i8(out: usize, inp: usize, seed: u64) -> (Vec<i8>, Vec<f32>) {
    let mut s = seed | 1;
    let mut nb = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 40) as i32
    };
    let data: Vec<i8> = (0..out * inp).map(|_| (nb() % 255 - 127) as i8).collect();
    let scales: Vec<f32> = (0..out)
        .map(|_| 0.01 + (nb() % 100) as f32 * 1e-4)
        .collect();
    (data, scales)
}

fn bench(label: &str, out: usize, inp: usize, iters: usize) {
    let (data, scales) = make_i8(out, inp, 0xA11CE ^ out as u64);
    // Activation x[inp] (reused; small, cache-resident — isolates the weight stream).
    let mut s = 0x1234u64;
    let x: Vec<f32> = (0..inp)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();

    let w_normal = I8Mat {
        data: data.clone(),
        scales: scales.clone(),
        out,
        inp,
    };
    let hp = hugepage_copy_i8(&data);
    let w_huge = I8Mat {
        data: hp,
        scales: scales.clone(),
        out,
        inp,
    };

    let mut y = vec![0.0f32; out];
    let mut best = |w: &I8Mat| -> f64 {
        for _ in 0..3 {
            nn::gemv_i8(w, &x, None, &mut y);
            black_box(&y);
        }
        let mut b = f64::INFINITY;
        for _ in 0..iters {
            let t = Instant::now();
            nn::gemv_i8(w, &x, None, &mut y);
            b = b.min(t.elapsed().as_secs_f64());
            black_box(&y);
        }
        b
    };
    // Interleave A/B/A/B to fight drift on a shared box.
    let bn1 = best(&w_normal);
    let bh1 = best(&w_huge);
    let bn2 = best(&w_normal);
    let bh2 = best(&w_huge);
    let bn = bn1.min(bn2);
    let bh = bh1.min(bh2);
    let mb = (out * inp) as f64 / 1e6;
    println!(
        "  {label:<14} [{out}x{inp}] {mb:>6.1} MB | normal {:>7.3} ms {:>5.0} GB/s | huge {:>7.3} ms {:>5.0} GB/s | speedup {:.3}x",
        bn * 1e3,
        mb / 1e3 / bn,
        bh * 1e3,
        mb / 1e3 / bh,
        bn / bh,
    );
    // Leak the mmap-backed I8Mat (its Vec must never hit the global free).
    std::mem::forget(w_huge);
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    println!(
        "=== huge-page (2 MB THP) vs normal (4 KB) weight-stream A/B, real turbo decode GEMV shapes @ {}t ===",
        rayon::current_num_threads()
    );
    println!(
        "THP=madvise, AnonHugePages baseline 0 → normal Vec gets NO huge pages; huge = mmap+MADV_HUGEPAGE. byte-exact (identical weights)."
    );
    // Real turbo decode per-token streaming set (n_state=1280, mlp_hidden=5120, vocab=51865):
    bench("mlp_fc/fc1", 5120, 1280, iters);
    bench("mlp_proj/fc2", 1280, 5120, iters);
    bench("logits", 51865, 1280, iters);
    bench("qkv(fused)", 3840, 1280, iters);
}
