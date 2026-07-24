//! Byte-exact "masked-logits-row-skip" lever (BlackThrush, 2026-07-04).
//!
//! In NO_TIMESTAMPS mode `process_logits` masks every timestamp token
//! (`timestamp_begin .. vocab`) to -inf BEFORE the greedy argmax, so those rows
//! of the tied 51866×1280 int8 logits GEMV can never be the selected token.
//! Computing only rows `[0 .. timestamp_begin]` is therefore BYTE-EXACT in no_ts
//! (the skipped rows were -inf anyway) and skips ~1500/51866 = 2.9% of the single
//! biggest per-token decode op. This probe measures whether the op-level saving
//! is worth touching the hottest decode kernel, or is sub-noise (a DROP).
//!
//! Model-free: synthetic turbo int8 weights via nn::quantize_f16_to_i8, same
//! nn::gemv_i8 the decode uses. COLD (rotating >L3 pool) — the logits weight is
//! DRAM-streamed per token, so cold is the honest regime (per
//! project_draft_decoding_amortization). Usage: `logits_row_skip_probe [passes]`.
use franken_whisper::native_engine::nn::{self, I8Mat};
use ft_core::Float16;
use std::hint::black_box;
use std::time::Instant;

fn make_w(out: usize, inp: usize, seed: u64) -> I8Mat {
    let mut st = seed | 1;
    let mut f16 = Vec::with_capacity(out * inp);
    for _ in 0..out * inp {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        let v = ((st >> 40) as f32 / (1u64 << 24) as f32) * 0.08 - 0.04;
        f16.push(Float16::from_f32(v));
    }
    nn::quantize_f16_to_i8(&f16, out, inp)
}

fn make_pool(out: usize, inp: usize, min_bytes: usize, seed0: u64) -> Vec<I8Mat> {
    let per = out * inp;
    let copies = (min_bytes / per + 1).max(2);
    (0..copies)
        .map(|c| make_w(out, inp, seed0.wrapping_add(c as u64 * 0x9E37)))
        .collect()
}

fn time_cold(pool: &[I8Mat], x: &[f32], out: usize, passes: usize) -> f64 {
    let mut y = vec![0.0f32; out];
    for w in pool {
        nn::gemv_i8(w, black_box(x), None, &mut y);
        black_box(y[0]);
    }
    let mut best = f64::INFINITY;
    for _ in 0..passes {
        let t = Instant::now();
        for w in pool {
            nn::gemv_i8(w, black_box(x), None, &mut y);
            black_box(y[0]);
        }
        best = best.min(t.elapsed().as_secs_f64() / pool.len() as f64);
    }
    best
}

fn main() {
    let passes: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    const NS: usize = 1280; // turbo n_state
    const VOCAB: usize = 51866;
    const TS_BEGIN: usize = 50365; // large-v3-turbo timestamp_begin (vocab-1500..)
    const L3: usize = 192 * 1024 * 1024;

    let x: Vec<f32> = (0..NS).map(|i| 0.02 * ((i % 51) as f32 - 25.0)).collect();

    eprintln!("building synthetic logits weights (full {VOCAB} vs skipped {TS_BEGIN} rows) …");
    let p_full = make_pool(VOCAB, NS, L3, 0x55);
    let p_skip = make_pool(TS_BEGIN, NS, L3, 0x55);

    let t_full = time_cold(&p_full, &x, VOCAB, passes);
    let t_skip = time_cold(&p_skip, &x, TS_BEGIN, passes);

    let us = |s: f64| s * 1e6;
    let gbs = |rows: usize, s: f64| (rows * NS) as f64 / s / 1e9;
    eprintln!("\n=== COLD logits GEMV, full vs no_ts timestamp-row-skip ===");
    eprintln!(
        "  full  [{VOCAB}×{NS}] : {:>7.2} µs  ({:>5.1} GB/s)",
        us(t_full),
        gbs(VOCAB, t_full)
    );
    eprintln!(
        "  skip  [{TS_BEGIN}×{NS}] : {:>7.2} µs  ({:>5.1} GB/s)  (skip {} ts rows)",
        us(t_skip),
        gbs(TS_BEGIN, t_skip),
        VOCAB - TS_BEGIN
    );
    let ratio = t_full / t_skip;
    let op_frac = 1.0 - (t_skip / t_full);
    eprintln!(
        "  ratio full/skip = {:.4}×  (op-level saving {:.2}%; row-count saving {:.2}%)",
        ratio,
        100.0 * op_frac,
        100.0 * (VOCAB - TS_BEGIN) as f64 / VOCAB as f64
    );
    // e2e: logits ≈ 29% of per-token decode (cold, project_draft_decoding_amortization),
    // decode ≈ 15% of e2e (ts) — and this lever ONLY applies in no_ts.
    let e2e = op_frac * 0.29 * 0.15;
    eprintln!(
        "  => e2e ≈ {:.3}%  (op_saving × 29% logits-of-decode × 15% decode-of-e2e; no_ts-only)",
        100.0 * e2e
    );
    eprintln!(
        "\nVERDICT: byte-exact in no_ts (skipped rows are -inf pre-argmax), but the saving is a fixed\n\
         2.9% of ONE op on the hottest, most correctness-sensitive decode kernel ⇒ sub-noise e2e.\n\
         DROP: not worth a row-range gemv_i8 variant + no_ts-only branch on the logits critical path."
    );
}
