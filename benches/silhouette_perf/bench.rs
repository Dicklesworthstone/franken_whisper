// A/B for a byte-exact restructure of orchestrator::silhouette_score (the
// diarization quality metric). The pairwise distance matrix is symmetric
// (d(i,j) == d(j,i) bit-for-bit, since (a-b)^2 == (b-a)^2), yet the original
// computes every ordered pair — n(n-1) distances. The candidate computes each
// unordered pair once (n(n-1)/2 distances) and accumulates into per-point
// per-cluster sums; the summation ORDER per (point, cluster) is preserved
// (increasing other-index), so the score is bit-identical. Halves the sqrt.
use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct SpeakerEmbedding {
    features: [f64; 6],
}

impl SpeakerEmbedding {
    fn euclidean_distance(&self, other: &SpeakerEmbedding) -> f64 {
        self.features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

// Verbatim copy of the production silhouette_score.
fn silhouette_original(
    embeddings: &[SpeakerEmbedding],
    assignments: &[usize],
    num_clusters: usize,
) -> Option<f64> {
    if num_clusters < 2 || embeddings.len() < 2 {
        return None;
    }
    let n = embeddings.len();
    let mut sum = 0.0_f64;
    for i in 0..n {
        let ci = assignments[i];
        let mut a_sum = 0.0_f64;
        let mut a_count = 0u64;
        for (j, emb_j) in embeddings.iter().enumerate() {
            if j != i && assignments[j] == ci {
                a_sum += embeddings[i].euclidean_distance(emb_j);
                a_count += 1;
            }
        }
        let a_i = if a_count > 0 { a_sum / a_count as f64 } else { 0.0 };
        let mut b_i = f64::INFINITY;
        for cj in 0..num_clusters {
            if cj == ci {
                continue;
            }
            let mut b_sum = 0.0_f64;
            let mut b_count = 0u64;
            for (j, emb_j) in embeddings.iter().enumerate() {
                if assignments[j] == cj {
                    b_sum += embeddings[i].euclidean_distance(emb_j);
                    b_count += 1;
                }
            }
            if b_count > 0 {
                let mean_dist = b_sum / b_count as f64;
                if mean_dist < b_i {
                    b_i = mean_dist;
                }
            }
        }
        let denom = a_i.max(b_i);
        let s_i = if denom < 1e-15 { 0.0 } else { (b_i - a_i) / denom };
        sum += s_i;
    }
    Some(sum / n as f64)
}

// Candidate: one distance per unordered pair, symmetric accumulation.
fn silhouette_symmetric(
    embeddings: &[SpeakerEmbedding],
    assignments: &[usize],
    num_clusters: usize,
) -> Option<f64> {
    if num_clusters < 2 || embeddings.len() < 2 {
        return None;
    }
    let n = embeddings.len();
    let mut cluster_sum = vec![0.0_f64; n * num_clusters];
    let mut cluster_count = vec![0u64; n * num_clusters];
    for i in 0..n {
        let ci = assignments[i];
        for j in (i + 1)..n {
            let d = embeddings[i].euclidean_distance(&embeddings[j]);
            let cj = assignments[j];
            cluster_sum[i * num_clusters + cj] += d;
            cluster_count[i * num_clusters + cj] += 1;
            cluster_sum[j * num_clusters + ci] += d;
            cluster_count[j * num_clusters + ci] += 1;
        }
    }
    let mut sum = 0.0_f64;
    for i in 0..n {
        let ci = assignments[i];
        let base = i * num_clusters;
        let a_count = cluster_count[base + ci];
        let a_i = if a_count > 0 {
            cluster_sum[base + ci] / a_count as f64
        } else {
            0.0
        };
        let mut b_i = f64::INFINITY;
        for cj in 0..num_clusters {
            if cj == ci {
                continue;
            }
            let b_count = cluster_count[base + cj];
            if b_count > 0 {
                let mean_dist = cluster_sum[base + cj] / b_count as f64;
                if mean_dist < b_i {
                    b_i = mean_dist;
                }
            }
        }
        let denom = a_i.max(b_i);
        let s_i = if denom < 1e-15 { 0.0 } else { (b_i - a_i) / denom };
        sum += s_i;
    }
    Some(sum / n as f64)
}

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn fixture(n: usize, num_clusters: usize) -> (Vec<SpeakerEmbedding>, Vec<usize>) {
    let mut lcg = Lcg(0x51_5100 + n as u64);
    let embeddings: Vec<SpeakerEmbedding> = (0..n)
        .map(|_| SpeakerEmbedding {
            features: std::array::from_fn(|_| lcg.next_f64() * 10.0 - 5.0),
        })
        .collect();
    // Balanced-ish assignment; clustering quality is irrelevant to timing/exactness.
    let assignments: Vec<usize> = (0..n).map(|i| (i * 7 + i / 3) % num_clusters).collect();
    (embeddings, assignments)
}

fn timed(
    embeddings: &[SpeakerEmbedding],
    assignments: &[usize],
    num_clusters: usize,
    implementation: fn(&[SpeakerEmbedding], &[usize], usize) -> Option<f64>,
) -> (Duration, u64) {
    let started = Instant::now();
    let score = implementation(black_box(embeddings), black_box(assignments), num_clusters);
    let elapsed = started.elapsed();
    (elapsed, score.map(|s| s.to_bits()).unwrap_or(0))
}

fn percentile(values: &[f64], percent: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * percent / 100]
}

fn main() {
    const SIZES: [usize; 4] = [200, 600, 1500, 3000];
    const NUM_CLUSTERS: usize = 4;
    const PAIRS: usize = 25;
    const WARMUP: usize = 6;

    // Edge-case byte-exactness: a singleton cluster (a_count==0 branch) and an
    // empty cluster id must not perturb the bit-identity of the two forms.
    {
        let mut lcg = Lcg(0xED9E);
        let embeddings: Vec<SpeakerEmbedding> = (0..40)
            .map(|_| SpeakerEmbedding {
                features: std::array::from_fn(|_| lcg.next_f64() * 8.0 - 4.0),
            })
            .collect();
        // point 0 alone in cluster 4 (singleton), cluster 3 left empty.
        let assignments: Vec<usize> =
            (0..40).map(|i| if i == 0 { 4 } else { i % 3 }).collect();
        let (_, bo) = timed(&embeddings, &assignments, 5, silhouette_original);
        let (_, bs) = timed(&embeddings, &assignments, 5, silhouette_symmetric);
        assert_eq!(bo, bs, "silhouette bits differ on singleton/empty-cluster edge");
    }

    for n in SIZES {
        let (embeddings, assignments) = fixture(n, NUM_CLUSTERS);

        // Byte-exactness.
        let (_, bits_o) = timed(&embeddings, &assignments, NUM_CLUSTERS, silhouette_original);
        let (_, bits_s) = timed(&embeddings, &assignments, NUM_CLUSTERS, silhouette_symmetric);
        assert_eq!(bits_o, bits_s, "silhouette score bits differ at n={n}");

        for _ in 0..WARMUP {
            black_box(timed(&embeddings, &assignments, NUM_CLUSTERS, silhouette_original));
            black_box(timed(&embeddings, &assignments, NUM_CLUSTERS, silhouette_symmetric));
        }

        let mut null_ratios = Vec::with_capacity(PAIRS);
        let mut speedups = Vec::with_capacity(PAIRS);
        let mut orig_ns = Vec::with_capacity(PAIRS);
        let mut cand_ns = Vec::with_capacity(PAIRS);
        for pair in 0..PAIRS {
            let a = timed(&embeddings, &assignments, NUM_CLUSTERS, silhouette_original).0;
            let b = timed(&embeddings, &assignments, NUM_CLUSTERS, silhouette_original).0;
            null_ratios.push(if pair.is_multiple_of(2) {
                a.as_secs_f64() / b.as_secs_f64()
            } else {
                b.as_secs_f64() / a.as_secs_f64()
            });

            let (orig, cand) = if pair.is_multiple_of(2) {
                let orig = timed(&embeddings, &assignments, NUM_CLUSTERS, silhouette_original).0;
                let cand = timed(&embeddings, &assignments, NUM_CLUSTERS, silhouette_symmetric).0;
                (orig, cand)
            } else {
                let cand = timed(&embeddings, &assignments, NUM_CLUSTERS, silhouette_symmetric).0;
                let orig = timed(&embeddings, &assignments, NUM_CLUSTERS, silhouette_original).0;
                (orig, cand)
            };
            orig_ns.push(orig.as_nanos());
            cand_ns.push(cand.as_nanos());
            speedups.push(orig.as_secs_f64() / cand.as_secs_f64());
        }

        let wins = speedups.iter().filter(|&&r| r > 1.0).count();
        println!(
            "n={n} clusters={NUM_CLUSTERS} pairs={PAIRS} null_p10={:.4} null_median={:.4} null_p90={:.4} orig_median_ns={} cand_median_ns={} speedup_p10={:.4} speedup_median={:.4} speedup_p90={:.4} wins={wins}/{PAIRS}",
            percentile(&null_ratios, 10),
            percentile(&null_ratios, 50),
            percentile(&null_ratios, 90),
            { let mut v = orig_ns.clone(); v.sort_unstable(); v[v.len() / 2] },
            { let mut v = cand_ns.clone(); v.sort_unstable(); v[v.len() / 2] },
            percentile(&speedups, 10),
            percentile(&speedups, 50),
            percentile(&speedups, 90),
        );
    }
}
