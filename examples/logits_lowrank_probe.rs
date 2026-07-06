//! Token-embedding (logits head) effective-rank probe (land-or-dig, 2026-07-06).
//!
//! The decode logits GEMV `emb[n_vocab,n_state] @ x` is the biggest single decode op
//! (66 MB int8/token, MEMORY-BANDWIDTH-bound — [[project_2row_gemv_landed]]). If the
//! tied token embedding is LOW-RANK (`emb ≈ U[n_vocab,r]·Vᵀ[r,n_state]`, r≪n_state),
//! the logits become `(x·V)·Uᵀ` = O(n_vocab·r + r·n_state) FLOPs AND — crucially —
//! `U` is `n_vocab·r` bytes vs `n_vocab·n_state`, so it CUTS THE DRAM STREAM the head
//! is bound on. This is a DIFFERENT bet than the encoder low-rank digs (which failed
//! partly because the encoder is COMPUTE-bound so a FLOP cut doesn't help): here the
//! byte cut attacks the actual bottleneck. The open question is purely the RANK.
//!
//! Method: form the Gram `G = EᵀE` [n_state,n_state] (one big matmul), then measure
//! captured Frobenius energy at rank r = (Σ top-r σ²(E))/(Σ σ²(E)) = trace(Qᵣᵀ G Qᵣ)/
//! trace(G), where Qᵣ is the rank-r range of the SMALL PSD `G` via randomized range
//! finding + 2 power iterations (MGS on n_state-length vectors — cheap). σ²(E) =
//! eig(G), so this is the exact rank-r Frobenius capture.
//!
//! Verdict: rank-r captures ≥99% at r≪n_state (e.g. r≤512 ⇒ ≤0.4× bytes) ⇒ a low-rank
//! logits head is a real bandwidth lever (owner-gated on argmax-flip accuracy, like
//! int8 logits). Near-full-rank ⇒ closes it (embedding uses all n_state dims).
//! Needs FRANKEN_WHISPER_MODEL_DIR. `logits_lowrank_probe`.
use franken_whisper::native_engine::Mat;
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use franken_whisper::native_engine::nn;

fn dot(a: &[f32], b: &[f32]) -> f64 { a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum() }
fn nrm(a: &[f32]) -> f64 { dot(a, a).sqrt() }

fn main() {
    let path = find_model_file("large-v3-turbo").expect("set FRANKEN_WHISPER_MODEL_DIR");
    let model = GgmlModel::load(&path).expect("ggml");
    let (shape, data) = model
        .tensor_f32("decoder.token_embedding.weight")
        .expect("token_embedding");
    let (n_vocab, n_state) = (shape[0], shape[1]);
    println!("=== token-embedding (logits head) effective rank | [{n_vocab},{n_state}] ===");
    println!("captured = (Σ top-r σ²)/(Σ σ²) = Frobenius energy a rank-r factorization keeps\n");

    // G = EᵀE  [n_state, n_state]  (via Eᵀ[n_state,n_vocab] @ E[n_vocab,n_state]).
    let e = Mat::from_vec(n_vocab, n_state, data);
    let mut et = vec![0.0f32; n_state * n_vocab];
    for r in 0..n_vocab {
        let row = &e.data[r * n_state..(r + 1) * n_state];
        for c in 0..n_state { et[c * n_vocab + r] = row[c]; }
    }
    let et = Mat::from_vec(n_state, n_vocab, et);
    let g = nn::matmul(&et, &e).expect("gram"); // [n_state, n_state], symmetric PSD
    let ns = n_state;
    let trace_g: f64 = (0..ns).map(|d| g.data[d * ns + d] as f64).sum();

    // Randomized range finding on G with 2 power iterations → Qᵣ [ns, r].
    let mut seed = 0x51D5_10AD_BEEF_0001u64;
    let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); if (seed >> 40) & 1 == 0 { 1.0f32 } else { -1.0f32 } };

    for &r in &[32usize, 64, 128, 256, 512, 768, 1024] {
        if r >= ns { continue; }
        let omega = Mat::from_vec(ns, r, (0..ns * r).map(|_| rnd()).collect());
        // Z = G·G·Ω  (2 power iterations to sharpen the top-r eigenspace)
        let z1 = nn::matmul(&g, &omega).expect("z1"); // [ns, r]
        let z = nn::matmul(&g, &z1).expect("z"); // [ns, r]
        // MGS on Z's r columns (each length ns=1280 — cheap).
        let mut q: Vec<Vec<f32>> = Vec::with_capacity(r);
        for j in 0..r {
            let mut v: Vec<f32> = (0..ns).map(|i| z.data[i * r + j]).collect();
            for qc in &q {
                let d = dot(&v, qc) as f32;
                for (vi, &qi) in v.iter_mut().zip(qc) { *vi -= d * qi; }
            }
            let nn_ = nrm(&v);
            if nn_ > 1e-6 { let inv = (1.0 / nn_) as f32; for vi in v.iter_mut() { *vi *= inv; } q.push(v); }
        }
        // captured = Σ_i (qᵢᵀ G qᵢ) / trace(G).
        let mut cap = 0.0f64;
        for qc in &q {
            // Gq = G·qc  [ns]
            let mut gq = vec![0.0f32; ns];
            for i in 0..ns {
                let grow = &g.data[i * ns..(i + 1) * ns];
                gq[i] = grow.iter().zip(qc).map(|(&a, &b)| a * b).sum();
            }
            cap += dot(qc, &gq);
        }
        let pct = cap / trace_g * 100.0;
        let byte_ratio = r as f64 / ns as f64;
        println!("  rank {r:>4}: captured {pct:6.2}%   (factorized head ≈ {byte_ratio:.2}× the DRAM bytes)");
    }
    println!("\nVIABLE iff ≥99% at r≪{ns} (r≤512 ⇒ ≤0.4× bytes on the bandwidth-bound logits head).");
    println!("Near-full-rank ⇒ embedding uses all {ns} dims ⇒ low-rank logits head DEAD.");
}
