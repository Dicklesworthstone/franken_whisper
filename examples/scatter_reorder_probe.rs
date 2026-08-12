//! Is the SDPA scatter (`sdpa_scatter_interleaved`) faster with a HEAD-MAJOR read
//! order? The current loop is POSITION-major (i outer, h inner): per output row it
//! copies 20 head-blocks from `o` at a 384 KB inter-head stride = a strided READ of
//! `o`. A head-major reorder within each worker's output band (h outer, i inner)
//! reads `o` CONTIGUOUSLY per head (prefetcher-friendly) and moves the stride to the
//! WRITE (into the L2-resident output band, write-combined). Same partitioning, same
//! copies ⇒ BYTE-EXACT. The mirror of the landed gather-fusion (favor the contiguous
//! access on the DRAM/L3 operand). BlackThrush tested the GATHER's ordering (ledger
//! 3536) but not the scatter's read-order.
//!
//! Run: cargo run --release --example scatter_reorder_probe
use rayon::prelude::*;

const HH: usize = 20;
const T: usize = 1500;
const DH: usize = 64;
const NS: usize = HH * DH; // 1280

/// Exact replica of nn::sdpa_scatter_interleaved (position-major, i outer / h inner).
fn scatter_pos_major(out: &mut [f32], o: &[f32], chunks: usize) {
    let n = if chunks == 0 { T } else { chunks.clamp(1, T) };
    let rows_per = T.div_ceil(n).max(1);
    out.par_chunks_mut(rows_per * NS)
        .enumerate()
        .for_each(|(c, blk)| {
            let i0 = c * rows_per;
            for (local, orow) in blk.chunks_mut(NS).enumerate() {
                let i = i0 + local;
                for h in 0..HH {
                    orow[h * DH..(h + 1) * DH]
                        .copy_from_slice(&o[h * T * DH + i * DH..h * T * DH + i * DH + DH]);
                }
            }
        });
}

/// Head-major reorder: same output partitioning + same copies, but iterate h OUTER,
/// i INNER within each band ⇒ contiguous read of `o` per head, strided write into the
/// (L2-resident) output band. Byte-identical.
fn scatter_head_major(out: &mut [f32], o: &[f32], chunks: usize) {
    let n = if chunks == 0 { T } else { chunks.clamp(1, T) };
    let rows_per = T.div_ceil(n).max(1);
    out.par_chunks_mut(rows_per * NS)
        .enumerate()
        .for_each(|(c, blk)| {
            let i0 = c * rows_per;
            let band = blk.len() / NS; // rows in this band (last band may be short)
            for h in 0..HH {
                for local in 0..band {
                    let i = i0 + local;
                    let src = &o[h * T * DH + i * DH..h * T * DH + i * DH + DH];
                    blk[local * NS + h * DH..local * NS + h * DH + DH].copy_from_slice(src);
                }
            }
        });
}

fn ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let mut s = 0x2545F4914F6CDD1Du64;
    let o: Vec<f32> = (0..HH * T * DH)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 40) as f32
        })
        .collect();

    let mut a = vec![0.0f32; T * NS];
    let mut b = vec![0.0f32; T * NS];
    // real default chunk count = 16 (FW_SDPA_GATHER_CHUNKS)
    for ch in [16usize, 0] {
        scatter_pos_major(&mut a, &o, ch);
        scatter_head_major(&mut b, &o, ch);
        println!("chunks={ch}: byte-identical pos==head: {}", a == b);
    }

    let reps = 50;
    let mut evict = vec![1.0f32; 40 * 1024 * 1024 / 4];
    for ch in [16usize, 0] {
        let (mut bp, mut bh) = (f64::MAX, f64::MAX);
        for _ in 0..reps {
            for e in &mut *evict {
                *e *= 1.0000001;
            }
            let t = std::time::Instant::now();
            scatter_pos_major(&mut a, &o, ch);
            bp = bp.min(ms(t));
            for e in &mut *evict {
                *e *= 1.0000001;
            }
            let t = std::time::Instant::now();
            scatter_head_major(&mut b, &o, ch);
            bh = bh.min(ms(t));
        }
        std::hint::black_box(&a);
        std::hint::black_box(&b);
        println!(
            "chunks={ch:2} (32t, cold, min-of-{reps}): pos-major = {bp:.3} ms | head-major = {bh:.3} ms | head/pos = {:.3}× {}",
            bh / bp,
            if bh < bp {
                "(head-major FASTER)"
            } else {
                "(pos-major faster ⇒ reorder DEAD)"
            }
        );
    }
    std::hint::black_box(&evict);
}
