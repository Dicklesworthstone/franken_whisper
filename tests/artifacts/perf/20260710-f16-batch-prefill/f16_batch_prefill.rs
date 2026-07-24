//! Re-attack of the REOPENED `weight-stationary f16 GEMV` family, on the input where the
//! code actually executes (bd-f16-gemv-weight-stationary-reopen-ugyh).
//!
//! **Why the old REJECT was invalid.** It benched `[1500,1280]x[1280,1280]` — the cross-K/V
//! shape — claiming `cross_attn_k/v` route to `gemv_f16_batch`. They do not: `a674b49`
//! (2026-07-02) flipped `cross_proj_f32_enabled()` default-ON, sending cross-K/V through f32
//! sgemm. `gemv_f16_batch` has ONE production caller (`decoder.rs:345`), reached only by
//! `WeightMat::F16` linears at `tq>1` without `w_i8`. Per `decoder.rs:312` that is **mlp_2
//! (fc2) at prefill**, which is excluded from the int8 batch path by its `w_i8_block`.
//! Real shape: `out = n_state = 1280`, `inp = n_mlp = 5120`, `tq` = prompt length.
//!
//! **The lever.** `gemv_f16_batch` has two schedulers:
//!   column-band (`compute_band`, o-outer): weight streamed ONCE. Intensity `tq` flop/wbyte.
//!   row-morsel  (t-outer per band):        weight re-streamed by EVERY band -> `workers`
//!                                          passes. Intensity `tq / workers`.
//! Row-morsel is chosen when `work = tq*out*inp >= 1<<26`. That gate conflates "compute-bound"
//! with "big weight". fc2's weight is 1280*5120*2 = 13.1 MB (4x cross-K/V's 3.3 MB), so at a
//! realistic prefill `tq` the row-morsel schedule re-streams ~200 MB for ~0.65 GFLOP.
//!
//! **BOTH ARMS CALL THE REAL `nn::gemv_f16_batch`**, differing only by
//! `nn::set_batch_gemv_row_morsel(bool)`. Nothing here replicates the kernel. Both schedulers
//! are proven bit-identical to per-token `gemv_f16`, so the knob cannot change results.
//!
//! **SUBSTRATE.** Criterion group members run SEQUENTIALLY and do NOT cancel drift. The
//! reported ratios come from `paired()`, which alternates the two arms WITHIN one measured
//! routine and forms a per-rep paired ratio, so a thermal/neighbour spike hits both arms in a
//! rep and cancels. The keep-gate statistic is cv of that paired ratio.
//!
//! **black_box discipline.** Every input is fed through `black_box` and the FULL output is
//! consumed through `black_box` (summed), so no arm can be dead-code-eliminated. A `NULL
//! CONTROL` runs arm-vs-itself: the ratio must be ~1.000x and the win rate ~50%.
//!
//! Run:  env -u CARGO_TARGET_DIR rch exec -- cargo bench -p franken_whisper --bench f16_batch_prefill

use criterion::{criterion_group, criterion_main, Criterion};
use franken_whisper::native_engine::nn;
use half::f16 as Float16;
use std::hint::black_box;
use std::time::Instant;

fn fill_f32(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 16_777_216.0) - 0.5
        })
        .collect()
}
fn fill_f16(seed: u64, n: usize) -> Vec<Float16> {
    fill_f32(seed, n).into_iter().map(Float16::from_f32).collect()
}

/// consume the WHOLE output so LLVM cannot DCE the call
#[inline]
fn consume(v: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for chunk in v.chunks(64) {
        acc += chunk[0];
    }
    black_box(acc)
}

fn run_once(morsel: bool, w: &[Float16], out: usize, inp: usize, x: &[f32], tq: usize, y: &mut [f32]) -> f64 {
    nn::set_batch_gemv_row_morsel(Some(morsel));
    let t0 = Instant::now();
    nn::gemv_f16_batch(
        black_box(w),
        black_box(out),
        black_box(inp),
        black_box(x),
        black_box(tq),
        None,
        y,
    );
    let dt = t0.elapsed().as_secs_f64() * 1e3;
    black_box(consume(y));
    dt
}

struct Stat {
    med: f64,
    cv: f64,
    wins: usize,
    n: usize,
}

/// TRUE interleaving: both arms timed back-to-back inside one rep, order alternated.
fn paired(w: &[Float16], out: usize, inp: usize, x: &[f32], tq: usize, arm_a: bool, arm_b: bool, reps: usize) -> (Stat, f64, f64) {
    let mut ya = vec![0.0f32; tq * out];
    let mut yb = vec![0.0f32; tq * out];
    let warm = 3usize;
    let (mut va, mut vb, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..(reps + warm) {
        let (ta, tb) = if r % 2 == 0 {
            let a = run_once(arm_a, w, out, inp, x, tq, &mut ya);
            let b = run_once(arm_b, w, out, inp, x, tq, &mut yb);
            (a, b)
        } else {
            let b = run_once(arm_b, w, out, inp, x, tq, &mut yb);
            let a = run_once(arm_a, w, out, inp, x, tq, &mut ya);
            (a, b)
        };
        if r >= warm {
            va.push(ta);
            vb.push(tb);
            ratios.push(ta / tb);
        }
    }
    // bit-exactness of the two schedulers (only meaningful when arms differ)
    let bit = ya.iter().zip(yb.iter()).all(|(p, q)| p.to_bits() == q.to_bits());
    assert!(bit, "the two schedulers must be bit-identical");
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|p, q| p.partial_cmp(q).unwrap());
        v[v.len() / 2]
    };
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let sd = (ratios.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ratios.len() as f64).sqrt();
    let wins = ratios.iter().filter(|r| **r > 1.0).count();
    let n = ratios.len();
    let mut rs = ratios.clone();
    (Stat { med: med(&mut rs), cv: 100.0 * sd / mean, wins, n }, med(&mut va), med(&mut vb))
}

fn bench(_c: &mut Criterion) {
    let avail = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_string()).unwrap_or_else(|_| "?".into());
    let reps: usize = std::env::var("F16_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    println!("\n===== f16 batch GEMV: row-morsel vs column-band, REAL nn::gemv_f16_batch =====");
    println!("host={host} available_parallelism={avail} reps={reps}");
    println!("arms differ ONLY by nn::set_batch_gemv_row_morsel(); both schedulers are bit-identical");
    println!("COMPUTE_BOUND_MACS = 1<<26 = {} ; row-morsel fires when tq*out*inp >= that\n", 1u64 << 26);

    // (label, out, inp, tq, is_real_consumer)
    let cases: &[(&str, usize, usize, usize, bool)] = &[
        ("fc2 prefill tq=12  [1280x5120]", 1280, 5120, 12, true),
        ("fc2 prefill tq=24  [1280x5120]", 1280, 5120, 24, true),
        ("fc2 prefill tq=50  [1280x5120]", 1280, 5120, 50, true),
        ("fc2 prefill tq=100 [1280x5120]", 1280, 5120, 100, true),
        ("cross-KV tq=1500   [1280x1280]  (NOT a live consumer since a674b49)", 1280, 1280, 1500, false),
    ];

    println!("{:<62} {:>9} {:>9} {:>8} {:>7} {:>8} {:>7}", "case", "morsel ms", "colband ms", "ratio", "cv%", "wins", "gate");
    for (label, out, inp, tq, real) in cases {
        let (out, inp, tq) = (*out, *inp, *tq);
        let w = fill_f16(1, out * inp);
        let x = fill_f32(7, tq * inp);
        let work = tq * out * inp;
        let fires = work >= (1usize << 26);
        let (s, ma, mb) = paired(&w, out, inp, &x, tq, true, false, reps);
        let wmb = (out * inp * 2) as f64 / 1e6;
        println!(
            "{:<62} {ma:>9.2} {mb:>9.2} {:>7.3}x {:>6.1} {:>4}/{:<3} {:>7}",
            format!("{label}{}", if *real { "" } else { " [control]" }),
            s.med, s.cv, s.wins, s.n,
            if s.cv < 5.0 { "PASS" } else { "FAIL" }
        );
        println!("      weight {wmb:.1} MB; row-morsel re-streams it once per band; morsel path active: {fires}");
    }

    // NULL CONTROL: same arm twice. ratio must be ~1.000x, wins ~50%, and it calibrates cv.
    println!("\n--- NULL CONTROL (row-morsel vs row-morsel: identical code) ---");
    let (out, inp, tq) = (1280usize, 5120usize, 50usize);
    let w = fill_f16(1, out * inp);
    let x = fill_f32(7, tq * inp);
    let (s, ma, mb) = paired(&w, out, inp, &x, tq, true, true, reps);
    println!(
        "  fc2 tq=50 self-vs-self: a {ma:.2} ms  b {mb:.2} ms  ratio {:.3}x  cv {:.1}%  wins {}/{}  -> harness floor",
        s.med, s.cv, s.wins, s.n
    );
    println!("\nNOTE: criterion group members run SEQUENTIALLY and would NOT cancel drift; every ratio");
    println!("above comes from paired(), which alternates the arms inside one measured routine.");
}

criterion_group!(benches, bench);
criterion_main!(benches);
