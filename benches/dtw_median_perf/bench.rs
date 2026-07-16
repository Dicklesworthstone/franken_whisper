use std::hint::black_box;
use std::time::{Duration, Instant};

// Per-call workload. 6 attention heads x 64 tokens = 384 rows is one real DTW
// median-filter invocation for a single 30 s (1500-frame) window, so each timed
// arm does the work of one production `align_tokens` median-filter pass rather
// than 1/6th of it. Larger per-arm work shrinks the fixed per-call overhead
// (the `row.to_vec()` scratch, branch-predictor warm-up) as a fraction of the
// measured span, which is what a *valid null* (BASE/BASE ~ 1.0) requires.
const ROWS: usize = 384;
const FRAMES: usize = 1500;
const PAIRS: usize = 31;
// Warm-up rounds before any timed pair, run until the CPU reaches all-core
// frequency steady-state so later pairs do not drift slower than earlier ones
// (the 0.957 null bias in the 2026-07-15 HOLD was a monotonic thermal droop
// over 15 un-warmed pairs).
const WARMUP_ROUNDS: usize = 12;

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
        (self.0 as f32 / (1u32 << 30) as f32) - 1.0
    }
}

#[inline(always)]
fn compare_swap_tagged(values: &mut [(f32, u8); 7], left: usize, right: usize) {
    let order = values[right]
        .0
        .partial_cmp(&values[left].0)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| values[right].1.cmp(&values[left].1));
    if order.is_lt() {
        values.swap(left, right);
    }
}

#[inline(always)]
fn median_seven_stable(mut values: [(f32, u8); 7]) -> f32 {
    compare_swap_tagged(&mut values, 0, 5);
    compare_swap_tagged(&mut values, 0, 3);
    compare_swap_tagged(&mut values, 1, 6);
    compare_swap_tagged(&mut values, 2, 4);
    compare_swap_tagged(&mut values, 0, 1);
    compare_swap_tagged(&mut values, 3, 5);
    compare_swap_tagged(&mut values, 2, 6);
    compare_swap_tagged(&mut values, 2, 3);
    compare_swap_tagged(&mut values, 3, 6);
    compare_swap_tagged(&mut values, 4, 5);
    compare_swap_tagged(&mut values, 1, 4);
    compare_swap_tagged(&mut values, 1, 3);
    compare_swap_tagged(&mut values, 3, 4);
    values[3].0
}

fn historical_sort(row: &mut [f32]) {
    let len = row.len() as i64;
    let src = row.to_vec();
    let mut window = Vec::with_capacity(7);
    for (k, out) in row.iter_mut().enumerate() {
        window.clear();
        for off in -3i64..=3 {
            let mut idx = k as i64 + off;
            if idx < 0 {
                idx = -idx;
            } else if idx >= len {
                idx = 2 * (len - 1) - idx;
            }
            let idx = idx.clamp(0, len - 1) as usize;
            window.push(src[idx]);
        }
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        *out = window[3];
    }
}

fn median7_network(row: &mut [f32]) {
    let len = row.len() as i64;
    let src = row.to_vec();
    for (k, out) in row.iter_mut().enumerate() {
        let window: [(f32, u8); 7] = std::array::from_fn(|slot| {
            let mut idx = k as i64 + slot as i64 - 3;
            if idx < 0 {
                idx = -idx;
            } else if idx >= len {
                idx = 2 * (len - 1) - idx;
            }
            let idx = idx.clamp(0, len - 1) as usize;
            (src[idx], slot as u8)
        });
        *out = if window.iter().all(|&(value, _)| value.is_finite()) {
            median_seven_stable(window)
        } else {
            let mut fallback = window.map(|(value, _)| value);
            fallback.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            fallback[3]
        };
    }
}

fn fixture() -> Vec<Vec<f32>> {
    let mut lcg = Lcg(0xd7_0007);
    (0..ROWS)
        .map(|_| (0..FRAMES).map(|_| lcg.next_f32()).collect())
        .collect()
}

/// Restore `work` from `pristine` (a straight per-row `copy_from_slice` memcpy,
/// reusing `work`'s existing allocation) and then time only the transform. The
/// restore is deliberately OUTSIDE the `Instant` and allocation-free, so — unlike
/// the prior `fixture.clone()`-per-call harness — no 1.5 MB allocation churns the
/// allocator/cache asymmetrically between the two arms. Both arms see identically
/// prepared memory; the only difference timed is the median implementation.
fn timed(work: &mut [Vec<f32>], pristine: &[Vec<f32>], implementation: fn(&mut [f32])) -> Duration {
    for (dst, src) in work.iter_mut().zip(pristine) {
        dst.copy_from_slice(src);
    }
    let started = Instant::now();
    for row in work.iter_mut() {
        implementation(black_box(row));
    }
    let elapsed = started.elapsed();
    black_box(&*work);
    elapsed
}

fn percentile(values: &[f64], percentile: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn output_hash(rows: &[Vec<f32>]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for bits in rows.iter().flatten().map(|value| value.to_bits()) {
        for byte in bits.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

fn main() {
    let fixture = fixture();

    // Byte-exactness: the network must reproduce the historical stable-sort
    // median bit-for-bit (same FNV-1a digest across refactors of the harness).
    let mut expected = fixture.clone();
    let mut actual = fixture.clone();
    for row in expected.iter_mut() {
        historical_sort(row);
    }
    for row in actual.iter_mut() {
        median7_network(row);
    }
    assert_eq!(
        actual
            .iter()
            .flatten()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .flatten()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    println!("DTW_MEDIAN_OUTPUT_FNV64={:016x}", output_hash(&actual));

    // Single reusable working buffer; restored from `fixture` before every timed
    // call inside `timed` (allocation-free memcpy, outside the timer).
    let mut work = fixture.clone();

    for _ in 0..WARMUP_ROUNDS {
        black_box(timed(&mut work, &fixture, historical_sort));
        black_box(timed(&mut work, &fixture, median7_network));
    }

    let mut null_ratios = Vec::with_capacity(PAIRS);
    let mut candidate_ratios = Vec::with_capacity(PAIRS);
    let mut historical_ns = Vec::with_capacity(PAIRS);
    let mut candidate_ns = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        // Null (BASE/BASE): two adjacent historical calls, order-alternated.
        let null_first = timed(&mut work, &fixture, historical_sort);
        let null_second = timed(&mut work, &fixture, historical_sort);
        let null_ratio = if pair.is_multiple_of(2) {
            null_first.as_secs_f64() / null_second.as_secs_f64()
        } else {
            null_second.as_secs_f64() / null_first.as_secs_f64()
        };
        null_ratios.push(null_ratio);

        // Candidate: adjacent historical vs network, order-alternated, so both
        // arms share whatever residual drift remains across the pair.
        let (historical, candidate) = if pair.is_multiple_of(2) {
            (
                timed(&mut work, &fixture, historical_sort),
                timed(&mut work, &fixture, median7_network),
            )
        } else {
            let candidate = timed(&mut work, &fixture, median7_network);
            let historical = timed(&mut work, &fixture, historical_sort);
            (historical, candidate)
        };
        historical_ns.push(historical.as_nanos());
        candidate_ns.push(candidate.as_nanos());
        candidate_ratios.push(historical.as_secs_f64() / candidate.as_secs_f64());
    }

    let null_p10 = percentile(&null_ratios, 10);
    let null_median = percentile(&null_ratios, 50);
    let null_p90 = percentile(&null_ratios, 90);
    let candidate_p10 = percentile(&candidate_ratios, 10);
    let candidate_median = percentile(&candidate_ratios, 50);
    let candidate_p90 = percentile(&candidate_ratios, 90);
    let wins = candidate_ratios
        .iter()
        .filter(|&&ratio| ratio > 1.0)
        .count();
    println!("ROWS={ROWS} FRAMES={FRAMES} PAIRS={PAIRS} WARMUP={WARMUP_ROUNDS}");
    println!("BASE_BASE_RATIOS={null_ratios:?}");
    println!("HISTORICAL_NETWORK_RATIOS={candidate_ratios:?}");
    println!("HISTORICAL_NS={historical_ns:?}");
    println!("NETWORK_NS={candidate_ns:?}");
    println!("NULL_P10={null_p10:.6} NULL_MEDIAN={null_median:.6} NULL_P90={null_p90:.6}");
    println!(
        "CANDIDATE_P10={candidate_p10:.6} CANDIDATE_MEDIAN={candidate_median:.6} CANDIDATE_P90={candidate_p90:.6} WINS={wins}/{PAIRS}"
    );
}
