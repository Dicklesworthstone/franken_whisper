//! Feasibility probe for BLOCK-WISE int7 weight scales in the int8 ENCODER GEMM.
//!
//! The gated maddubs int8 encoder ([[project_turbo_encoder_dominates]]) is ~1.5× e2e
//! (a power-throttle dodge) and BYTE-IDENTICAL on easy audio (jfk), but makes occasional
//! PROPER-NOUN errors on hard audio (track01: "Franken"->"Frank at"). The ledger's
//! SKIP_FIRST=2 experiment (first 2 encoder layers kept f32) RECOVERS the proper nouns —
//! i.e. the damage is early-layer WEIGHT-QUANTIZATION error — but SKIP_FIRST forfeits the
//! whole speed win (any f32 layer re-triggers the all-core power throttle). The one un-tried
//! fix: BLOCK-WISE int7 scales (one amax/63 per 32-weight block, the Q8_0 layout that made
//! decode fc2 byte-exact where per-row broke it) instead of the current PER-ROW `I7Mat.scale`.
//!
//! Mechanism this probe measures: transformer projection rows have occasional LARGE OUTLIER
//! weights. A per-ROW amax is inflated by one outlier → the whole row quantizes coarsely
//! (large step) → the many small weights lose precision → proper-noun-class errors. A
//! per-BLOCK amax isolates the outlier to its 32-wide block, sparing the other blocks. This
//! probe quantifies the RMS-error reduction on realistic (Gaussian + sparse-outlier) rows —
//! the quantitative basis for whether a block-wise maddubs GEMM (multi-hour) is worth building.
#![allow(unsafe_code)]

// Deterministic LCG (Date/rand are unavailable and we want reproducibility).
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
    /// Box-Muller-ish standard normal from two uniforms.
    fn normal(&mut self) -> f32 {
        let u1 = (self.unit()).max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Quantize one row to i7 [-63,63] with a SINGLE (per-row) scale = amax/63; return RMS
/// dequant error over the row.
fn per_row_rms(row: &[f32]) -> f64 {
    let amax = row.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    if amax == 0.0 { return 0.0; }
    let scale = amax / 63.0;
    let inv = 1.0 / scale;
    let mut se = 0.0f64;
    for &w in row {
        let q = (w * inv).round().clamp(-63.0, 63.0);
        let dq = q * scale;
        se += ((w - dq) as f64).powi(2);
    }
    (se / row.len() as f64).sqrt()
}

/// Quantize one row to i7 with a PER-BLOCK scale (block = `blk` weights); RMS dequant error.
fn per_block_rms(row: &[f32], blk: usize) -> f64 {
    let mut se = 0.0f64;
    for chunk in row.chunks(blk) {
        let amax = chunk.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        if amax == 0.0 { continue; }
        let scale = amax / 63.0;
        let inv = 1.0 / scale;
        for &w in chunk {
            let q = (w * inv).round().clamp(-63.0, 63.0);
            let dq = q * scale;
            se += ((w - dq) as f64).powi(2);
        }
    }
    (se / row.len() as f64).sqrt()
}

fn main() {
    let inp = 1280usize; // encoder contraction dim (n_state)
    let rows = 1280usize; // one projection's output rows
    let blk = 32usize;    // Q8_0 block size
    let mut rng = Lcg(0x9E3779B97F4A7C15);

    // Two regimes: (A) pure Gaussian weights (no outliers), (B) Gaussian + sparse large
    // outliers (~1 per row at 8-15σ), the realistic transformer-projection case that
    // inflates a per-row amax.
    for (label, outliers) in [("gaussian (no outliers)", false), ("gaussian + sparse outliers", true)] {
        let mut sum_row = 0.0f64;
        let mut sum_blk = 0.0f64;
        let mut worst_ratio = 0.0f64;
        for _r in 0..rows {
            let mut row: Vec<f32> = (0..inp).map(|_| rng.normal() * 0.02).collect();
            if outliers {
                // one big outlier in a random position (8-15σ of the 0.02 scale)
                let pos = (rng.next_u32() as usize) % inp;
                let mag = (8.0 + rng.unit() * 7.0) * 0.02;
                row[pos] = if rng.unit() < 0.5 { mag } else { -mag };
            }
            let pr = per_row_rms(&row);
            let pb = per_block_rms(&row, blk);
            sum_row += pr;
            sum_blk += pb;
            if pb > 0.0 { worst_ratio = worst_ratio.max(pr / pb); }
        }
        let mr = sum_row / rows as f64;
        let mb = sum_blk / rows as f64;
        println!("== {label} ==");
        println!("  per-row  RMS dequant err : {:.3e}", mr);
        println!("  per-block RMS dequant err: {:.3e}   (block={blk})", mb);
        println!("  mean error reduction (per-row / per-block): {:.2}×", mr / mb.max(1e-30));
        println!("  worst-row error reduction:                  {:.2}×", worst_ratio);
    }
    println!("\nInterpretation: a >1× reduction in the OUTLIER regime = block-wise isolates the");
    println!("outlier's inflated amax to its 32-wide block, sparing the rest of the row = the");
    println!("mechanism that would recover proper-noun accuracy while staying all-int8 (power dodge).");
}
