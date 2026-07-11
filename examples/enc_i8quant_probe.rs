//! `matmul_bias_i8` (encoder.rs, the DEFAULT-ON attn.out i8 GEMM) quantizes activations
//! with a SCALAR `.round().clamp()` loop — `f32::round` has no AVX rounding mode, so LLVM
//! emits a per-element `roundf`. nn.rs already has a byte-identical AVX2 quantizer
//! (`quantize_act_i8_into`, "~5×", tested 0-diff over ±127 edges) but it's module-private,
//! and consolidating it means editing cod-dirty nn.rs. Replicating its proven algorithm
//! LOCALLY in encoder.rs is byte-exact and shippable. This probe measures the quant
//! speedup on the real shape (m=1500 × inp=1280) and asserts byte-identical output.
//!
//! BYTE-EXACT: `v + copysign(0.5,v)` then round-to-zero = round-half-away (== f32::round),
//! then clamp[-127,127], order-preserving pack. PER-CORE ⇒ thread-invariant ⇒ admissible
//! on any rch worker. Run: cargo run --release --example enc_i8quant_probe
use rayon::prelude::*;

const INP: usize = 1280;

/// Scalar reference: exactly the current encoder.rs:1256 inner loop.
fn quant_row_scalar(xr: &[f32], inv: f32, out: &mut [i8]) {
    for (d, &v) in out.iter_mut().zip(xr) {
        *d = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
}

/// AVX2 replica of nn::quantize_act_i8_into — byte-identical to the scalar map.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[allow(unsafe_code)]
fn quant_row_avx2(xr: &[f32], inv: f32, out: &mut [i8]) {
    use core::arch::x86_64::*;
    let n = xr.len().min(out.len());
    let xp = xr.as_ptr();
    unsafe {
        let vinv = _mm256_set1_ps(inv);
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let signmask = _mm256_set1_ps(-0.0);
        let c127 = _mm256_set1_ps(127.0);
        let cm127 = _mm256_set1_ps(-127.0);
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_mul_ps(_mm256_loadu_ps(xp.add(i)), vinv);
            // byte-exact round-half-away: trunc(v) + (|v-trunc(v)|>=0.5 ? copysign(1,v):0)
            let tr = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(v);
            let frac = _mm256_sub_ps(v, tr);
            let ge = _mm256_cmp_ps::<_CMP_GE_OQ>(_mm256_andnot_ps(signmask, frac), half);
            let sign1 = _mm256_or_ps(one, _mm256_and_ps(v, signmask));
            let r = _mm256_add_ps(tr, _mm256_and_ps(ge, sign1));
            let r = _mm256_min_ps(_mm256_max_ps(r, cm127), c127);
            let ri = _mm256_cvtps_epi32(r);
            let lo = _mm256_castsi256_si128(ri);
            let hi = _mm256_extracti128_si256::<1>(ri);
            let i16s = _mm_packs_epi32(lo, hi);
            let i8s = _mm_packs_epi16(i16s, i16s);
            _mm_storel_epi64(out.as_mut_ptr().add(i) as *mut __m128i, i8s);
            i += 8;
        }
        while i < n {
            *out.get_unchecked_mut(i) = (*xr.get_unchecked(i) * inv).round().clamp(-127.0, 127.0) as i8;
            i += 1;
        }
    }
}

/// Full per-row activation quant (amax → scale → quant), par over rows — as in matmul_bias_i8.
fn quant_all(x: &[f32], m: usize, out: &mut [i8], sa: &mut [f32], avx2: bool) {
    out.par_chunks_mut(INP)
        .zip(sa.par_iter_mut())
        .enumerate()
        .for_each(|(r, (xr_i8, s))| {
            let xr = &x[r * INP..(r + 1) * INP];
            let amax = xr.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
            let rs = amax / 127.0;
            *s = rs;
            let inv = 1.0 / rs;
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            if avx2 { quant_row_avx2(xr, inv, xr_i8); } else { quant_row_scalar(xr, inv, xr_i8); }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            { let _ = avx2; quant_row_scalar(xr, inv, xr_i8); }
        });
    let _ = m;
}

fn ms(t: std::time::Instant) -> f64 { t.elapsed().as_secs_f64() * 1e3 }

fn main() {
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    { eprintln!("needs avx2"); }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut nf = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); ((s >> 33) as i32 as f32) / 4.0e8 };
        let avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        println!("# enc i8 activation-quant AVX2-vs-scalar — available_parallelism={avail}");
        for workers in [1usize, avail.min(16)] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
            for m in [1500usize, 200] {
                let x: Vec<f32> = (0..m * INP).map(|_| nf()).collect();
                let (mut a, mut b) = (vec![0i8; m * INP], vec![0i8; m * INP]);
                let (mut sa, mut sb) = (vec![0.0f32; m], vec![0.0f32; m]);
                pool.install(|| quant_all(&x, m, &mut a, &mut sa, false));
                pool.install(|| quant_all(&x, m, &mut b, &mut sb, true));
                let ident = a == b && sa == sb;
                let reps = 100;
                let (mut bs, mut bv) = (f64::MAX, f64::MAX);
                for r in 0..reps {
                    if r % 2 == 0 {
                        let t = std::time::Instant::now(); pool.install(|| quant_all(&x, m, &mut a, &mut sa, false)); bs = bs.min(ms(t));
                        let t = std::time::Instant::now(); pool.install(|| quant_all(&x, m, &mut b, &mut sb, true)); bv = bv.min(ms(t));
                    } else {
                        let t = std::time::Instant::now(); pool.install(|| quant_all(&x, m, &mut b, &mut sb, true)); bv = bv.min(ms(t));
                        let t = std::time::Instant::now(); pool.install(|| quant_all(&x, m, &mut a, &mut sa, false)); bs = bs.min(ms(t));
                    }
                }
                std::hint::black_box(&a); std::hint::black_box(&b);
                let v = if bv < bs { "AVX2 FASTER" } else { "AVX2 slower" };
                println!("m={m:4} {workers:2}t min-{reps}: scalar={bs:.4} ms  avx2={bv:.4} ms  ({:.2}× {v})  byte-id={ident}", bs / bv);
            }
        }
    }
}
