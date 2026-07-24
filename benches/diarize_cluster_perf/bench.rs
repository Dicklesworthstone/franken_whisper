// A/B for a byte-exact O(1) running-sum centroid update in diarize_segments'
// greedy clustering. The original recomputes centroid(&cluster_members[cid]) —
// summing ALL members — on every assignment, so a growing cluster costs
// O(cluster_size) per add = O(n^2) for a dominant speaker. A running per-cluster
// sum (accumulated in the same push order the recompute re-sums) makes each
// centroid update O(1) and bit-identical, so every subsequent cosine decision —
// and thus the whole assignment sequence — is byte-exact.
use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct SpeakerEmbedding {
    features: [f64; 6],
}

impl SpeakerEmbedding {
    fn cosine_similarity(&self, other: &SpeakerEmbedding) -> f64 {
        let dot: f64 = self
            .features
            .iter()
            .zip(other.features.iter())
            .map(|(a, b)| a * b)
            .sum();
        let mag_a: f64 = self.features.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = other.features.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag_a < 1e-10 || mag_b < 1e-10 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    fn centroid(embeddings: &[SpeakerEmbedding]) -> SpeakerEmbedding {
        let n = embeddings.len() as f64;
        if n < 1.0 {
            return SpeakerEmbedding { features: [0.0; 6] };
        }
        let mut features = [0.0_f64; 6];
        for emb in embeddings {
            for (i, val) in emb.features.iter().enumerate() {
                features[i] += val;
            }
        }
        for f in &mut features {
            *f /= n;
        }
        SpeakerEmbedding { features }
    }
}

const THRESHOLD: f64 = 0.92;

// Original: recompute centroid from all members on every assignment.
fn greedy_recompute(embeddings: &[SpeakerEmbedding]) -> (Vec<usize>, Vec<SpeakerEmbedding>) {
    let mut cluster_members: Vec<Vec<SpeakerEmbedding>> = Vec::new();
    let mut centroids: Vec<SpeakerEmbedding> = Vec::new();
    let mut assignments: Vec<usize> = Vec::with_capacity(embeddings.len());
    for emb in embeddings {
        let mut best_cluster = None;
        let mut best_sim = f64::NEG_INFINITY;
        for (cid, centroid) in centroids.iter().enumerate() {
            let sim = emb.cosine_similarity(centroid);
            if sim > best_sim {
                best_sim = sim;
                best_cluster = Some(cid);
            }
        }
        if best_sim >= THRESHOLD && best_cluster.is_some() {
            let cid = best_cluster.unwrap();
            assignments.push(cid);
            cluster_members[cid].push(emb.clone());
            centroids[cid] = SpeakerEmbedding::centroid(&cluster_members[cid]);
        } else {
            let nid = centroids.len();
            centroids.push(emb.clone());
            cluster_members.push(vec![emb.clone()]);
            assignments.push(nid);
        }
    }
    (assignments, centroids)
}

// Candidate: O(1) running-sum centroid update.
fn greedy_running(embeddings: &[SpeakerEmbedding]) -> (Vec<usize>, Vec<SpeakerEmbedding>) {
    let mut cluster_sums: Vec<[f64; 6]> = Vec::new();
    let mut cluster_counts: Vec<usize> = Vec::new();
    let mut centroids: Vec<SpeakerEmbedding> = Vec::new();
    let mut assignments: Vec<usize> = Vec::with_capacity(embeddings.len());
    for emb in embeddings {
        let mut best_cluster = None;
        let mut best_sim = f64::NEG_INFINITY;
        for (cid, centroid) in centroids.iter().enumerate() {
            let sim = emb.cosine_similarity(centroid);
            if sim > best_sim {
                best_sim = sim;
                best_cluster = Some(cid);
            }
        }
        if best_sim >= THRESHOLD && best_cluster.is_some() {
            let cid = best_cluster.unwrap();
            assignments.push(cid);
            for k in 0..6 {
                cluster_sums[cid][k] += emb.features[k];
            }
            cluster_counts[cid] += 1;
            let cnt = cluster_counts[cid] as f64;
            centroids[cid] = SpeakerEmbedding {
                features: std::array::from_fn(|k| cluster_sums[cid][k] / cnt),
            };
        } else {
            let nid = centroids.len();
            centroids.push(emb.clone());
            cluster_sums.push(emb.features);
            cluster_counts.push(1);
            assignments.push(nid);
        }
    }
    (assignments, centroids)
}

struct Lcg(u64);
impl Lcg {
    fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

// Clustered fixture: a few well-separated "speaker" directions + small noise, so
// same-speaker points exceed the 0.92 cosine threshold and grow big clusters
// (exercising the O(n^2) recompute), while speakers stay distinct.
fn fixture(n: usize) -> Vec<SpeakerEmbedding> {
    let bases: [[f64; 6]; 3] = [
        [1.0, 0.15, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.15, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.15],
    ];
    let mut lcg = Lcg(0xC1_05 + n as u64);
    (0..n)
        .map(|i| {
            let b = bases[i % 3];
            SpeakerEmbedding {
                features: std::array::from_fn(|k| b[k] + (lcg.next_unit() - 0.5) * 0.02),
            }
        })
        .collect()
}

fn centroids_bits(cs: &[SpeakerEmbedding]) -> Vec<u64> {
    cs.iter().flat_map(|c| c.features.iter().map(|f| f.to_bits())).collect()
}

fn timed(
    embeddings: &[SpeakerEmbedding],
    implementation: fn(&[SpeakerEmbedding]) -> (Vec<usize>, Vec<SpeakerEmbedding>),
) -> Duration {
    let started = Instant::now();
    let out = implementation(black_box(embeddings));
    let elapsed = started.elapsed();
    black_box(out.0.len());
    black_box(out.1.len());
    elapsed
}

fn percentile(values: &[f64], percent: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * percent / 100]
}

fn median_ns(values: &[u128]) -> u128 {
    let mut v = values.to_vec();
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() {
    const SIZES: [usize; 4] = [500, 1500, 3000, 6000];
    const PAIRS: usize = 21;
    const WARMUP: usize = 5;

    for n in SIZES {
        let embeddings = fixture(n);

        // Byte-exactness: identical assignments AND bit-identical centroids.
        let (asg_r, cen_r) = greedy_recompute(&embeddings);
        let (asg_s, cen_s) = greedy_running(&embeddings);
        assert_eq!(asg_r, asg_s, "assignments differ at n={n}");
        assert_eq!(centroids_bits(&cen_r), centroids_bits(&cen_s), "centroid bits differ at n={n}");
        let clusters = cen_r.len();

        for _ in 0..WARMUP {
            black_box(timed(&embeddings, greedy_recompute));
            black_box(timed(&embeddings, greedy_running));
        }

        let mut null_ratios = Vec::with_capacity(PAIRS);
        let mut speedups = Vec::with_capacity(PAIRS);
        let mut recompute_ns = Vec::with_capacity(PAIRS);
        let mut running_ns = Vec::with_capacity(PAIRS);
        for pair in 0..PAIRS {
            let a = timed(&embeddings, greedy_recompute);
            let b = timed(&embeddings, greedy_recompute);
            null_ratios.push(if pair.is_multiple_of(2) {
                a.as_secs_f64() / b.as_secs_f64()
            } else {
                b.as_secs_f64() / a.as_secs_f64()
            });

            let (recompute, running) = if pair.is_multiple_of(2) {
                (timed(&embeddings, greedy_recompute), timed(&embeddings, greedy_running))
            } else {
                let running = timed(&embeddings, greedy_running);
                let recompute = timed(&embeddings, greedy_recompute);
                (recompute, running)
            };
            recompute_ns.push(recompute.as_nanos());
            running_ns.push(running.as_nanos());
            speedups.push(recompute.as_secs_f64() / running.as_secs_f64());
        }

        let wins = speedups.iter().filter(|&&r| r > 1.0).count();
        println!(
            "n={n} clusters={clusters} pairs={PAIRS} null_p10={:.4} null_median={:.4} null_p90={:.4} recompute_median_ns={} running_median_ns={} speedup_p10={:.4} speedup_median={:.4} speedup_p90={:.4} wins={wins}/{PAIRS}",
            percentile(&null_ratios, 10),
            percentile(&null_ratios, 50),
            percentile(&null_ratios, 90),
            median_ns(&recompute_ns),
            median_ns(&running_ns),
            percentile(&speedups, 10),
            percentile(&speedups, 50),
            percentile(&speedups, 90),
        );
    }
}
