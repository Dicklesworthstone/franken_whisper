//! Encoder alloc-churn isolation probe (BlackThrush, 2026-07-02).
//!
//! `encoder_block` allocates fresh buffers every layer (2× x.clone() + fresh
//! matmul_bias outputs for q/k/v/attn/attn_out + a [n_ctx,n_mlp] mlp_h), ~92 MB
//! per turbo layer × 32 = ~2.9 GB alloc+free/window, each fresh block first-touch
//! faulted. This probe ISOLATES the alloc+free+fault overhead a buffer-reuse
//! rewrite would remove: it replicates the per-layer alloc sizes and the write
//! (matmul beta=0 overwrite) touch pattern, comparing
//!   A) alloc fresh every layer, touch, drop   (current encoder behavior)
//!   B) alloc ONCE, touch every layer           (buffer-reuse behavior)
//! The A−B delta is the upper bound on what a byte-exact reuse rewrite can save.
//! The touch (a full write pass) happens in BOTH, so the delta isolates malloc.
//!
//! Usage: `encoder_alloc_probe [layers] [iters]`  (default 32 layers, 20 iters).
use std::hint::black_box;
use std::time::Instant;

const N_CTX: usize = 1500;
const N_STATE: usize = 1280; // turbo
const N_MLP: usize = 5120; // turbo (4× n_state)

// Per-layer buffer element counts, mirroring encoder_block's allocations.
fn layer_buf_sizes() -> Vec<usize> {
    let s = N_CTX * N_STATE;
    vec![
        s,             // h = x.clone() (attn)
        s,             // q
        s,             // k
        s,             // v
        s,             // attn
        s,             // attn_out
        s,             // h = x.clone() (mlp)
        N_CTX * N_MLP, // mlp fc h
        s,             // mlp proj h
    ]
}

// Simulate a matmul beta=0 overwrite: write every element (first-touch faults a
// fresh buffer; a reused buffer is already resident). Non-trivial value so the
// optimizer cannot elide the store.
#[inline(never)]
fn touch(buf: &mut [f32], k: usize) {
    for (i, x) in buf.iter_mut().enumerate() {
        *x = (i as f32).mul_add(1.000_001, k as f32);
    }
    black_box(&buf[buf.len() - 1]);
}

fn main() {
    let mut a = std::env::args().skip(1);
    let layers: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let iters: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(20);
    let sizes = layer_buf_sizes();
    let mb_per_layer: f64 = sizes.iter().sum::<usize>() as f64 * 4.0 / 1e6;
    println!(
        "layers={layers} iters={iters}  per-layer alloc={mb_per_layer:.1} MB  window={:.0} MB",
        mb_per_layer * layers as f64
    );

    // A) fresh alloc every layer (current behavior).
    let mut best_fresh = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        for l in 0..layers {
            for &sz in &sizes {
                let mut buf = vec![0.0f32; sz];
                touch(&mut buf, l);
                black_box(&buf);
                drop(buf);
            }
        }
        best_fresh = best_fresh.min(t.elapsed().as_secs_f64());
    }

    // B) alloc ONCE, reuse across layers (buffer-reuse behavior).
    let mut best_reuse = f64::INFINITY;
    for _ in 0..iters {
        let mut bufs: Vec<Vec<f32>> = sizes.iter().map(|&sz| vec![0.0f32; sz]).collect();
        let t = Instant::now();
        for l in 0..layers {
            for buf in &mut bufs {
                touch(buf, l);
                black_box(&buf);
            }
        }
        best_reuse = best_reuse.min(t.elapsed().as_secs_f64());
        black_box(&bufs);
    }

    println!("best-of-{iters}:");
    println!("  A) fresh alloc/layer : {:.2} ms", best_fresh * 1e3);
    println!("  B) reuse buffers     : {:.2} ms", best_reuse * 1e3);
    println!(
        "  A−B (alloc+fault cost reuse removes): {:.2} ms/window  ({:.1}%)",
        (best_fresh - best_reuse) * 1e3,
        (best_fresh - best_reuse) / best_fresh * 100.0
    );
}
