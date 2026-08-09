// Standalone probe: does ERROR-FEEDBACK (error-diffusion) weight quantization make the FULL
// int8 encoder dot (u8-activation × i7-weight) closer to the f32 truth than independent
// round-to-nearest? This is the numerical gate for the FW_ENC_EF_QUANT lever — if EF does
// NOT reduce the int8 dot error vs f32, the transcript test can't recover proper nouns.
// rustc -O -C target-cpu=native --edition 2021
struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    fn f32(&mut self) -> f32 {
        (self.u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn gauss(&mut self, scale: f32) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..4 {
            s += self.f32();
        }
        (s / 2.0) * scale
    }
}

// quantize one weight column (len inp) to i7 with per-col scale. ef=true => error-feedback.
fn quant_w(col: &[f32], ef: bool) -> (Vec<i32>, f32) {
    let amax = col.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    let sc = amax / 63.0;
    let inv = 1.0 / sc;
    let mut q = vec![0i32; col.len()];
    if ef {
        let mut err = 0.0f32;
        for (d, &w) in q.iter_mut().zip(col) {
            let t = w * inv + err;
            let r = t.round().clamp(-63.0, 63.0);
            err = t - r;
            *d = r as i32;
        }
    } else {
        for (d, &w) in q.iter_mut().zip(col) {
            *d = (w * inv).round().clamp(-63.0, 63.0) as i32;
        }
    }
    (q, sc)
}

fn main() {
    let inp = 1280usize;
    let ncol = 4000usize; // sample output columns
    let nact = 40usize; // activation rows per column
    let mut rng = Lcg::new(0xBADC_0FFE_1234_5678);

    let mut sum_indep = 0.0f64;
    let mut sum_ef = 0.0f64;
    let mut max_indep = 0.0f64;
    let mut max_ef = 0.0f64;
    let mut ef_closer = 0u64;
    let mut indep_closer = 0u64;
    let mut n = 0u64;
    // also track error on the SUBSET of dots whose truth is near a "decision boundary"
    // (small |truth|), where a proper-noun logit flip is most likely.

    for _c in 0..ncol {
        // weight column: gaussian + occasional sparse outlier (real transformer weights)
        let mut col = vec![0.0f32; inp];
        for v in &mut col {
            *v = rng.gauss(0.06);
        }
        let oi = (rng.u32() as usize) % inp;
        col[oi] *= 6.0;
        let (qw_i, sw_i) = quant_w(&col, false);
        let (qw_e, sw_e) = quant_w(&col, true);

        for _a in 0..nact {
            // activation row (post-LN encoder hidden): zero-mean-ish gaussian
            // u8-quantized (symmetric amax/127) EXACTLY as matmul_bias_i7 does
            let mut a = vec![0.0f32; inp];
            for v in &mut a {
                *v = rng.gauss(1.0);
            }
            let aamax = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
            let sa = aamax / 127.0;
            let ainv = 1.0 / sa;
            let qa: Vec<i32> = a
                .iter()
                .map(|&v| (v * ainv).round().clamp(-127.0, 127.0) as i32)
                .collect();

            // f32 truth (the golden dot, full precision f32 operands)
            let mut truth = 0.0f64;
            for i in 0..inp {
                truth += (col[i] as f64) * (a[i] as f64);
            }

            // int8 indep: sw*sa * Σ qw_i*qa
            let mut d_i = 0i64;
            let mut d_e = 0i64;
            for i in 0..inp {
                d_i += (qw_i[i] as i64) * (qa[i] as i64);
                d_e += (qw_e[i] as i64) * (qa[i] as i64);
            }
            let v_i = (d_i as f64) * (sw_i as f64) * (sa as f64);
            let v_e = (d_e as f64) * (sw_e as f64) * (sa as f64);
            let ei = (v_i - truth).abs();
            let ee = (v_e - truth).abs();
            sum_indep += ei;
            sum_ef += ee;
            if ei > max_indep {
                max_indep = ei;
            }
            if ee > max_ef {
                max_ef = ee;
            }
            if ee < ei {
                ef_closer += 1;
            } else if ei < ee {
                indep_closer += 1;
            }
            n += 1;
        }
    }
    println!("samples                 = {}", n);
    println!(
        "indep  mean|err|        = {:.4e}   max = {:.4e}",
        sum_indep / n as f64,
        max_indep
    );
    println!(
        "EF     mean|err|        = {:.4e}   max = {:.4e}",
        sum_ef / n as f64,
        max_ef
    );
    let r = (sum_indep / n as f64) / (sum_ef / n as f64);
    println!(
        "indep/EF mean-err ratio = {:.3}x   (>1 => EF MORE accurate => promising)",
        r
    );
    println!(
        "EF closer to truth      = {} / {}  ({:.1}%)",
        ef_closer,
        n,
        100.0 * ef_closer as f64 / n as f64
    );
    println!(
        "indep closer            = {} / {}  ({:.1}%)",
        indep_closer,
        n,
        100.0 * indep_closer as f64 / n as f64
    );
}
