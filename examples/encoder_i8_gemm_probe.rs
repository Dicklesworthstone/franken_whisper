//! Encoder int8-GEMM viability probe (BlackThrush, 2026-07-02).
//!
//! The ENCODER (76% of turbo e2e) runs f32 GEMMs through matrixmultiply (blocked,
//! at-peak FMA). Memory flags "int8×int8 GEMM" as the owner-gated encoder unlock,
//! but that assumes a naive int8 GEMM would even be FASTER than the blocked f32
//! one — never measured at the encoder shape (tq=1500). This settles the SPEED
//! question (transcript question is separate + owner-gated) by racing, at the
//! real encoder projection shapes:
//!   - f32:  `nn::matmul` (matrixmultiply, BLOCKED)      [current encoder path]
//!   - int8: `gemv_i8_batch` (naive row×row int8 dot)    [tq=1500 = encoder GEMM]
//!   - f16:  `gemv_f16_batch` (naive row×row f16 dot)     [reference]
//! If naive int8 already beats blocked f32, the encoder int8 lever is worth the
//! transcript work; if it loses (naive has no cache-blocking, cf. cross_kv where
//! blocked-f32 beat naive-f16 2.25×), a blocked int8 microkernel is required
//! (bigger, owner-scoped). Run at RAYON_NUM_THREADS=32.
//! Usage: `encoder_i8_gemm_probe [iters]` (default 40).
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::nn;
use half::f16;
use std::hint::black_box;
use std::time::Instant;

fn bench(name: &str, out: usize, inp: usize, seq: usize, iters: usize) {
    let mut s = 0x1234_5678u64;
    let mut nextf = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let wf16: Vec<f16> = (0..out * inp).map(|_| f16::from_f32(nextf() * 0.1)).collect();
    // Wt [inp, out] f32 = transpose of the natural [out, inp] weight, as nn::matmul wants.
    let mut wt = vec![0.0f32; inp * out];
    for o in 0..out {
        for i in 0..inp {
            wt[i * out + o] = wf16[o * inp + i].to_f32();
        }
    }
    let wt_mat = Mat::from_vec(inp, out, wt);
    let w_i8 = nn::quantize_f16_to_i8(&wf16, out, inp);
    let h: Vec<f32> = (0..seq * inp).map(|_| nextf()).collect();
    let h_mat = Mat::from_vec(seq, inp, h.clone());
    let bias: Vec<f32> = (0..out).map(|_| nextf() * 0.1).collect();

    let mut y_i8 = nn::gemv_out_buf(seq * out);
    let mut y_f16 = nn::gemv_out_buf(seq * out);
    for _ in 0..3 {
        black_box(nn::matmul(&h_mat, &wt_mat).unwrap());
        nn::gemv_i8_batch(&w_i8, &h, seq, Some(&bias), &mut y_i8);
        nn::gemv_f16_batch(&wf16, out, inp, &h, seq, Some(&bias), &mut y_f16);
        black_box((&y_i8, &y_f16));
    }
    let (mut bf32, mut bi8, mut bf16) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    for _ in 0..iters {
        let t = Instant::now();
        let r = nn::matmul(&h_mat, &wt_mat).unwrap();
        bf32 = bf32.min(t.elapsed().as_secs_f64());
        black_box(r);
        let t = Instant::now();
        nn::gemv_i8_batch(&w_i8, &h, seq, Some(&bias), &mut y_i8);
        bi8 = bi8.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        nn::gemv_f16_batch(&wf16, out, inp, &h, seq, Some(&bias), &mut y_f16);
        bf16 = bf16.min(t.elapsed().as_secs_f64());
        black_box((&y_i8, &y_f16));
    }
    let gflop = 2.0 * seq as f64 * out as f64 * inp as f64 / 1e9;
    let verdict = if bi8 < bf32 { "int8 WINS — pursue encoder int8" } else { "int8 LOSES — needs blocked int8 kernel" };
    println!("{name}  [seq={seq}] out={out} inp={inp}  ({gflop:.2} GFLOP)  best-of-{iters}:");
    println!("  f32 matmul (BLOCKED): {:>7.3} ms  {:>6.0} GFLOP/s   [encoder path]", bf32 * 1e3, gflop / bf32);
    println!("  int8 batch (naive):   {:>7.3} ms  {:>6.0} GFLOP/s   {:.2}x vs f32   [{}]", bi8 * 1e3, gflop / bi8, bf32 / bi8, verdict);
    println!("  f16 batch (naive):    {:>7.3} ms  {:>6.0} GFLOP/s   {:.2}x vs f32", bf16 * 1e3, gflop / bf16, bf32 / bf16);
}

fn main() {
    let iters: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    println!("=== encoder GEMM shapes @ {} threads (f32=blocked matrixmultiply, int8/f16=naive) ===", rayon::current_num_threads());
    bench("QKV/out proj", 1280, 1280, 1500, iters);
    bench("MLP fc1", 5120, 1280, 1500, iters);
    bench("MLP fc2", 1280, 5120, 1500, iters);
}
