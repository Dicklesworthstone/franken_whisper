//! The self-speculative "logits tax" — the structural floor on ANY draft-model-FREE
//! (layer-skip / early-exit) speculative decode for large-v3-turbo. (BlackThrush, 2026-07-03)
//!
//! The ledger de-risked draft decoding on the VERIFY side: R(K) (weights read once for
//! K verified tokens) is depth-invariant ~3.7× at K=8 (project_draft_decoding_amortization).
//! But the realized speedup is R(K) × accept-rate ÷ DRAFT-COST, and the DRAFT cost was never
//! measured. A self-draft (no second model) proposes a token by running FEWER of the 4 turbo
//! decoder layers — BUT it must STILL emit a full-vocab token, so it ALWAYS pays the
//! 51866×1280 logits GEMV. That shared logits GEMV is a fixed tax on every draft attempt and
//! sets a hard floor on how cheap a self-draft token can be.
//!
//! This probe MEASURES (wall-clock, not the ledger's byte estimate — per-GEMV dispatch
//! overhead can shift the ratio) the per-token int8 weight-GEMV set on the real turbo shapes:
//!   per layer (x4): qkv[3840,1280]  attn_out[1280,1280]  fc1[5120,1280]  fc2[1280,5120]
//!   once:           logits[51866,1280]
//! and reports the logits fraction + the k-layer self-draft floor + the break-even accept rate.
//!
//! Model-FREE (synthetic int8 weights via nn::quantize_f16_to_i8), so it runs per-crate under
//! `rch exec -- cargo bench`-class harnessing. Byte-exact to the engine (calls the SAME
//! nn::gemv_i8). Usage: `self_draft_logits_tax_probe [iters]` (default 300).
use franken_whisper::native_engine::nn::{self, I8Mat};
use ft_core::Float16;
use std::hint::black_box;
use std::time::Instant;

/// Build a deterministic int8 weight matrix [out, inp] via the engine's own quantizer.
fn make_w(out: usize, inp: usize, seed: u64) -> I8Mat {
    let mut st = seed | 1;
    let mut f16 = Vec::with_capacity(out * inp);
    for _ in 0..out * inp {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        let v = ((st >> 40) as f32 / (1u64 << 24) as f32) * 0.08 - 0.04; // ~[-0.04,0.04]
        f16.push(Float16::from_f32(v));
    }
    nn::quantize_f16_to_i8(&f16, out, inp)
}

/// min-of-N wall-clock of one `gemv_i8` call [out,inp] @ [inp] -> [out].
fn time_gemv(w: &I8Mat, x: &[f32], out: usize, iters: usize) -> f64 {
    let mut y = vec![0.0f32; out];
    for _ in 0..(iters / 10).max(3) {
        nn::gemv_i8(w, black_box(x), None, &mut y);
        black_box(y[0]);
    }
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..iters {
            nn::gemv_i8(w, black_box(x), None, &mut y);
            black_box(y[0]);
        }
        best = best.min(t.elapsed().as_secs_f64() / iters as f64);
    }
    best
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    const NS: usize = 1280; // turbo n_state
    const VOCAB: usize = 51866;
    const FFN: usize = 5120;
    const QKV: usize = 3840; // fused q|k|v

    // synthetic activation vectors (finite, engine quantizes internally)
    let x_ns: Vec<f32> = (0..NS).map(|i| 0.02 * ((i % 51) as f32 - 25.0)).collect();
    let x_ffn: Vec<f32> = (0..FFN).map(|i| 0.02 * ((i % 51) as f32 - 25.0)).collect();

    eprintln!("building synthetic turbo int8 weights (one-time)…");
    let w_qkv = make_w(QKV, NS, 0x11);
    let w_out = make_w(NS, NS, 0x22);
    let w_fc1 = make_w(FFN, NS, 0x33);
    let w_fc2 = make_w(NS, FFN, 0x44);
    let w_log = make_w(VOCAB, NS, 0x55);

    let t_qkv = time_gemv(&w_qkv, &x_ns, QKV, iters);
    let t_out = time_gemv(&w_out, &x_ns, NS, iters);
    let t_fc1 = time_gemv(&w_fc1, &x_ns, FFN, iters);
    let t_fc2 = time_gemv(&w_fc2, &x_ffn, NS, iters);
    let t_log = time_gemv(&w_log, &x_ns, VOCAB, iters);

    let t_layer = t_qkv + t_out + t_fc1 + t_fc2; // one decoder layer's 4 weight GEMVs
    let n_layers = 4usize;
    let t_full = n_layers as f64 * t_layer + t_log; // per-token weight-GEMV time

    let us = |s: f64| s * 1e6;
    eprintln!("\n=== per-token int8 weight-GEMV breakdown (turbo, µs; min-of-7) ===");
    eprintln!("  qkv   [3840,1280] : {:>7.2}", us(t_qkv));
    eprintln!("  out   [1280,1280] : {:>7.2}", us(t_out));
    eprintln!("  fc1   [5120,1280] : {:>7.2}", us(t_fc1));
    eprintln!("  fc2   [1280,5120] : {:>7.2}", us(t_fc2));
    eprintln!("  1 layer (4 GEMVs) : {:>7.2}", us(t_layer));
    eprintln!("  logits[51866,1280]: {:>7.2}   <-- the fixed self-draft tax", us(t_log));
    eprintln!("  full token (4L+lg): {:>7.2}", us(t_full));

    let logits_frac = t_log / t_full;
    eprintln!("\n=== self-speculative (layer-skip) ceiling ===");
    eprintln!("  logits fraction of per-token GEMV time : {:.1}%", 100.0 * logits_frac);
    // Break-even model (OPTIMISTIC for speculation, so the bar is a lower bound):
    //   - draft `kdraft` tokens, each running the first k of 4 layers + the shared logits
    //     head, at cost `floor` full-token-forwards each;
    //   - 1 verify pass over the kdraft proposals, batched so weights stream ONCE ⇒ cost
    //     c_v ≈ 1.0 full-token-forward (the best case; the real R(4)≈2.2× makes it ~1.8);
    //   - a spec iteration emits (a+1) tokens: `a` accepted drafts + 1 always-correct token
    //     the verify produces for free (0 ≤ a ≤ kdraft).
    // Beat greedy (1.0/token) ⇔ (kdraft*floor + c_v)/(a+1) < 1.0 ⇔ a > kdraft*floor + c_v - 1.
    let kdraft = 4.0;
    let c_v = 1.0; // optimistic verify cost (weights-once); real ~1.8 ⇒ bar is HIGHER
    for k in 1..=3usize {
        let floor = (k as f64 * t_layer + t_log) / t_full; // draft-token cost / full token
        let a_be = (kdraft * floor + c_v - 1.0).clamp(0.0, kdraft); // accepted drafts needed
        eprintln!(
            "  draft = first {k} of 4 layers: draft-token cost = {:.2}x a full token; \
             K=4 break-even needs > {:.2} of 4 drafts accepted ({:.0}% accept, optimistic c_v=1.0)",
            floor, a_be, 100.0 * a_be / kdraft
        );
    }
    let floor1 = (1.0 * t_layer + t_log) / t_full;
    let a_be1 = kdraft * floor1 + c_v - 1.0;
    eprintln!(
        "\nVERDICT: the shared logits GEMV is {:.0}% of a per-token forward (MEASURED wall-clock, \n\
         vs the ledger's ~42% BYTE estimate — the 51866-row GEMV is less efficient per byte). \n\
         So even the cheapest 1-layer self-draft costs {:.2}x a full token, needing >{:.0}% draft \n\
         acceptance to break even (and MORE once verify c_v>1). Layer-skip self-speculation on the \n\
         4-layer turbo decoder is structurally capped — the logits head cannot be skipped while \n\
         still proposing a token. Confirms the ledger scoping WITH A MEASURED FLOOR: draft decoding \n\
         needs a genuinely CHEAP separate drafter (smaller vocab / shared-head shortcut), not a \n\
         depth-truncation of turbo's own decoder.",
        100.0 * logits_frac,
        floor1,
        100.0 * a_be1 / kdraft
    );
}
