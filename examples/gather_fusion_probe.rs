//! SDPA gather-FUSION hypothesis probe (2026-07-09, AshHeron).
//!
//! The int8 QKV GEMM writes its `[t, n_state]` output, then `sdpa_gather_head_major`
//! transposes it to head-major `[hh, t, d_head]` for the external SDPA. The ledger
//! (2026-07-04, BlackThrush) rejected TILING / access-order variants of that STANDALONE
//! gather (DRAM-latency floor). This probe tests a DIFFERENT primitive it never covered:
//! FUSING the transpose into the GEMM's output write (write head-major directly).
//!
//! The GEMM writes SOMETHING regardless, so fusion wins iff the head-major write is
//! cheaper than [contiguous write + separate gather]:
//!     t_headmajor_write  <  t_contiguous_write + t_gather
//! Head-major write stride = d_head (64 f32 = 256 B); the gather's strided READ stride
//! = n_state (1280 f32 = 5120 B). Tighter stride ⇒ plausibly less latency-bound.
//!
//! COLD regime (the ledger's key lesson: a WARM reuse-loop over-states transpose wins) —
//! every rep touches a >L3 eviction buffer first so `val`/`out` are DRAM-cold.
//! Byte-identity of both paths' `hm` output is asserted.

use std::time::Instant;

const T: usize = 1500; // enc frames (tq)
const NH: usize = 20; // n_head
const DH: usize = 64; // d_head
const NSTATE: usize = NH * DH; // 1280

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as i32 as f32) / (1i64 << 23) as f32 * 0.1
        })
        .collect()
}

/// Contiguous copy `val[t,nstate] -> out[t,nstate]` (models the GEMM's contiguous write).
fn write_contiguous(val: &[f32], out: &mut [f32]) {
    use rayon::prelude::*;
    out.par_chunks_mut(NSTATE)
        .zip(val.par_chunks(NSTATE))
        .for_each(|(o, v)| o.copy_from_slice(v));
}

/// Standalone gather `out[t,nstate] -> hm[nh,t,dh]` (== nn::sdpa_gather_head_major, 16 chunks).
fn gather(out: &[f32], hm: &mut [f32]) {
    use rayon::prelude::*;
    let total_rows = NH * T;
    let chunks = 16usize;
    let chunk_rows = total_rows.div_ceil(chunks).max(1);
    hm.par_chunks_mut(chunk_rows * DH).enumerate().for_each(|(c, blk)| {
        let row0 = c * chunk_rows;
        for (local, out_row) in blk.chunks_mut(DH).enumerate() {
            let r = row0 + local;
            let h = r / T;
            let i = r % T;
            let base = i * NSTATE + h * DH;
            out_row.copy_from_slice(&out[base..base + DH]);
        }
    });
}

/// FUSED head-major write `val[t,nstate] -> hm[nh,t,dh]` (models a head-major GEMM write).
/// Parallel over heads (each head owns a disjoint [t*dh] region ⇒ safe, no unsafe).
fn write_headmajor(val: &[f32], hm: &mut [f32]) {
    use rayon::prelude::*;
    hm.par_chunks_mut(T * DH).enumerate().for_each(|(h, hblk)| {
        for i in 0..T {
            let src = &val[i * NSTATE + h * DH..i * NSTATE + h * DH + DH];
            hblk[i * DH..(i + 1) * DH].copy_from_slice(src);
        }
    });
}

fn evict(buf: &mut [f32]) {
    // Touch a >L3 buffer to push val/out/hm out of cache (DRAM-cold regime).
    for (j, v) in buf.iter_mut().enumerate() {
        *v += (j & 0xff) as f32;
    }
}

fn bench_cold<F: FnMut()>(label: &str, reps: usize, evbuf: &mut [f32], mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        evict(evbuf);
        let t = Instant::now();
        f();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms < best {
            best = ms;
        }
    }
    println!("  {label:<40} {best:8.4} ms (min of {reps}, COLD)");
    best
}

fn main() {
    let reps = 30;
    let val = fill(T * NSTATE, 0x1234);
    let mut out = vec![0f32; T * NSTATE];
    let mut hm_a = vec![0f32; NH * T * DH];
    let mut hm_b = vec![0f32; NH * T * DH];
    // >L3 eviction buffer (64 MB > 32 MB CCD L3).
    let mut evbuf = vec![0f32; 16 * 1024 * 1024];

    // Byte-identity: path A (contiguous + gather) vs path B (head-major) must match.
    write_contiguous(&val, &mut out);
    gather(&out, &mut hm_a);
    write_headmajor(&val, &mut hm_b);
    assert_eq!(hm_a, hm_b, "path A != path B — layout math wrong");
    println!("byte-identity: hm_a == hm_b ✓\n");

    println!("COLD per-op costs (32 threads, {reps} reps, min):");
    let t_contig = bench_cold("contiguous write (GEMM does this anyway)", reps, &mut evbuf, || {
        write_contiguous(&val, &mut out);
    });
    let t_gather = bench_cold("gather (out -> head-major)", reps, &mut evbuf, || {
        gather(&out, &mut hm_a);
    });
    let t_hm = bench_cold("FUSED head-major write (val -> head-major)", reps, &mut evbuf, || {
        write_headmajor(&val, &mut hm_b);
    });

    println!("\n--- verdict (per QKV tensor, one layer) ---");
    let path_a = t_contig + t_gather;
    println!("  path A  = contiguous_write + gather = {t_contig:.4} + {t_gather:.4} = {path_a:.4} ms");
    println!("  path B  = fused head-major write     = {t_hm:.4} ms");
    if t_hm < path_a {
        println!("  ⇒ FUSION WINS by {:.4} ms ({:.2}×) — wire it (nn.rs head-major GEMM)", path_a - t_hm, path_a / t_hm);
    } else {
        println!("  ⇒ FUSION LOSES ({:.2}×) — head-major write ≥ contig+gather; the transpose", path_a / t_hm);
        println!("     latency is inherent, fusion just moves it. Ledger the rejection.");
    }
    println!("  (×3 tensors ×32 layers scales to the window; gather span was ~74 ms/window)");
}
