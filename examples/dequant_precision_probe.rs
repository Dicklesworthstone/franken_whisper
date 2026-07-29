// Standalone probe: is the maddubs int8 dequant (integer-dot-then-cast-then-scale)
// MORE or LESS accurate than the f32-accumulate roundtrip, relative to an f64 ground
// truth? Resolves the track01 anomaly: int8 maddubs -> "Frank at" (wrong), both-row
// f32-roundtrip -> "FrankenSearch" (correct). If maddubs is CLOSER to f64-truth, the
// roundtrip's correct proper noun is f32-accumulation NOISE, and the int8 error is
// fundamental (not a fixable dequant bug).  rustc -O -C target-cpu=native --edition 2021
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
    // approx gaussian via sum-of-uniforms (central limit), scaled
    fn gauss(&mut self, scale: f32) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..4 {
            s += self.f32();
        }
        (s / 2.0) * scale
    }
}

fn main() {
    let inp = 1280usize;
    let out = 1280usize;
    let m = 1500usize; // encoder ctx rows
    let mut rng = Lcg::new(0x1234_5678_9abc_def0);

    // Weights [inp,out] row-major as data[i*out+o] (matches Mat layout), gaussian.
    let mut w = vec![0.0f32; inp * out];
    for v in w.iter_mut() {
        *v = rng.gauss(0.06);
    } // typical linear weight magnitude
    // occasional outlier weights (real weights have heavy tails)
    for o in 0..out {
        let oi = (rng.u32() as usize) % inp;
        w[oi * out + o] *= 6.0;
    }

    // Per-output-column i7 quant: scale=amax/63, round, clamp; colsum.
    let mut wq = vec![0i32; inp * out];
    let mut wsc = vec![0.0f32; out];
    let mut wcs = vec![0i32; out];
    for o in 0..out {
        let mut amax = 1e-9f32;
        for i in 0..inp {
            amax = amax.max(w[i * out + o].abs());
        }
        let sc = amax / 63.0;
        wsc[o] = sc;
        let inv = 1.0 / sc;
        let mut cs = 0i32;
        for i in 0..inp {
            let q = (w[i * out + o] * inv).round().clamp(-63.0, 63.0) as i32;
            wq[i * out + o] = q;
            cs += q;
        }
        wcs[o] = cs;
    }

    // Activations [m,inp], gaussian-ish (encoder hidden states after LN/gelu).
    let mut x = vec![0.0f32; m * inp];
    for v in x.iter_mut() {
        *v = rng.gauss(1.0);
    }
    // per-row u8 quant: scale=amax/127, round, clamp; store i8v (a_int).
    let mut xq = vec![0i32; m * inp];
    let mut xsc = vec![0.0f32; m];
    for r in 0..m {
        let mut amax = 1e-9f32;
        for c in 0..inp {
            amax = amax.max(x[r * inp + c].abs());
        }
        let sc = amax / 127.0;
        xsc[r] = sc;
        let inv = 1.0 / sc;
        for c in 0..inp {
            xq[r * inp + c] = (x[r * inp + c] * inv).round().clamp(-127.0, 127.0) as i32;
        }
    }

    // For each (r,o): three dequants of the SAME quantized operands.
    // 1) maddubs: D = sum a_int*w (exact i64), then (D as f32) * xsc[r] * wsc[o]
    // 2) roundtrip: sum over c of (a_int*xsc[r]) * (w*wsc[o]) accumulated in f32
    // 3) f64 truth: (D as f64) * xsc64 * wsc64
    let mut sum_madd_err = 0.0f64;
    let mut max_madd_err = 0.0f64;
    let mut sum_rt_err = 0.0f64;
    let mut max_rt_err = 0.0f64;
    let mut madd_closer = 0u64;
    let mut rt_closer = 0u64;
    let mut n = 0u64;
    let mut max_d: i64 = 0;

    // sample a subset of (r,o) for speed but representative
    let r_step = 7usize;
    let o_step = 3usize;
    for r in (0..m).step_by(r_step) {
        let sa = xsc[r];
        let sa64 = sa as f64;
        let xrow = &xq[r * inp..(r + 1) * inp];
        for o in (0..out).step_by(o_step) {
            let sc = wsc[o];
            let sc64 = sc as f64;
            // integer dot (exact)
            let mut d: i64 = 0;
            // f32-accumulate roundtrip
            let mut rt: f32 = 0.0;
            for c in 0..inp {
                let a = xrow[c];
                let wv = wq[c * out + o];
                d += (a as i64) * (wv as i64);
                rt += (a as f32 * sa) * (wv as f32 * sc);
            }
            if d.abs() > max_d {
                max_d = d.abs();
            }
            let madd = (d as f32) * sa * sc;
            let truth = (d as f64) * sa64 * sc64;
            let me = (madd as f64 - truth).abs();
            let re = (rt as f64 - truth).abs();
            sum_madd_err += me;
            if me > max_madd_err {
                max_madd_err = me;
            }
            sum_rt_err += re;
            if re > max_rt_err {
                max_rt_err = re;
            }
            if me < re {
                madd_closer += 1;
            } else if re < me {
                rt_closer += 1;
            }
            n += 1;
        }
    }
    println!("samples                = {}", n);
    println!(
        "max |D| (integer dot)  = {}  (2^24 = {})",
        max_d,
        1i64 << 24
    );
    println!(
        "maddubs  mean|err|     = {:.3e}   max|err| = {:.3e}",
        sum_madd_err / n as f64,
        max_madd_err
    );
    println!(
        "roundtrip mean|err|    = {:.3e}   max|err| = {:.3e}",
        sum_rt_err / n as f64,
        max_rt_err
    );
    println!(
        "maddubs closer to f64  = {} / {}  ({:.1}%)",
        madd_closer,
        n,
        100.0 * madd_closer as f64 / n as f64
    );
    println!(
        "roundtrip closer       = {} / {}  ({:.1}%)",
        rt_closer,
        n,
        100.0 * rt_closer as f64 / n as f64
    );
    let ratio = (sum_rt_err / n as f64) / (sum_madd_err / n as f64);
    println!(
        "roundtrip/maddubs err  = {:.2}x  (>1 => maddubs MORE accurate => 'Frank at' is truer int8)",
        ratio
    );
}
