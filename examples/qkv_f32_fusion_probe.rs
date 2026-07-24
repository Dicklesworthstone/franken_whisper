//! Probe: does fusing the encoder's 3 f32 QKV sgemms (which all share the same
//! activation A=[1500,1280]) into ONE [1280,3840] sgemm (a) stay byte-identical
//! and (b) beat 3 separate [1280,1280] sgemms? matrixmultiply packs A per call,
//! so the fused form packs A once instead of 3×.
//!
//! Run: cargo run --profile release --example qkv_f32_fusion_probe
use franken_whisper::native_engine::nn;
use franken_whisper::native_engine::Mat;

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    fn mat(&mut self, r: usize, c: usize) -> Mat {
        Mat::from_vec(r, c, (0..r * c).map(|_| self.next_f32()).collect())
    }
}

fn main() {
    let (m, k, n) = (1500usize, 1280usize, 1280usize);
    let mut rng = Lcg(0x1234_5678);
    let h = rng.mat(m, k);
    let wq = rng.mat(k, n);
    let wk = rng.mat(k, n);
    let wv = rng.mat(k, n);

    // Fused weight [k, 3n]: row kk = [wq[kk,:] | wk[kk,:] | wv[kk,:]]
    let mut wqkv = vec![0.0f32; k * 3 * n];
    for kk in 0..k {
        wqkv[kk * 3 * n..kk * 3 * n + n].copy_from_slice(&wq.data[kk * n..kk * n + n]);
        wqkv[kk * 3 * n + n..kk * 3 * n + 2 * n].copy_from_slice(&wk.data[kk * n..kk * n + n]);
        wqkv[kk * 3 * n + 2 * n..kk * 3 * n + 3 * n].copy_from_slice(&wv.data[kk * n..kk * n + n]);
    }
    let wqkv = Mat::from_vec(k, 3 * n, wqkv);

    // Separate
    let q = nn::matmul(&h, &wq).unwrap();
    let kk_ = nn::matmul(&h, &wk).unwrap();
    let v = nn::matmul(&h, &wv).unwrap();
    // Fused
    let f = nn::matmul(&h, &wqkv).unwrap();

    // Byte-exactness: f[:, 0:n]==q, [n:2n]==k, [2n:3n]==v
    let mut diffs = 0usize;
    let mut maxabs = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let (fq, fk, fv) = (
                f.data[i * 3 * n + j],
                f.data[i * 3 * n + n + j],
                f.data[i * 3 * n + 2 * n + j],
            );
            for (a, b) in [(fq, q.data[i * n + j]), (fk, kk_.data[i * n + j]), (fv, v.data[i * n + j])] {
                if a.to_bits() != b.to_bits() {
                    diffs += 1;
                    maxabs = maxabs.max((a - b).abs());
                }
            }
        }
    }
    println!("byte-exact check: {diffs} differing / {} total  max|Δ|={maxabs:.3e}", m * n * 3);

    // Timing: min-of-N, interleaved
    let reps = std::env::var("REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let mut best_sep = f64::MAX;
    let mut best_fus = f64::MAX;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        let _q = nn::matmul(&h, &wq).unwrap();
        let _k = nn::matmul(&h, &wk).unwrap();
        let _v = nn::matmul(&h, &wv).unwrap();
        best_sep = best_sep.min(t.elapsed().as_secs_f64() * 1e3);

        let t = std::time::Instant::now();
        let _f = nn::matmul(&h, &wqkv).unwrap();
        best_fus = best_fus.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("min-of-{reps}: 3× separate = {best_sep:.3} ms   fused [1280,3840] = {best_fus:.3} ms   ratio(sep/fus) = {:.4}×", best_sep / best_fus);
    println!("(fused must ALSO beat sep by more than the downstream Q/K/V split cost to be a net win)");
}
