use std::hint::black_box;
use std::time::{Duration, Instant};

const ROWS: usize = 64;
const FRAMES: usize = 1500;
const PAIRS: usize = 15;

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

fn transform(rows: &mut [Vec<f32>], implementation: fn(&mut [f32])) {
    for row in rows.iter_mut() {
        implementation(black_box(row));
    }
    black_box(rows);
}

fn timed(mut rows: Vec<Vec<f32>>, implementation: fn(&mut [f32])) -> Duration {
    let started = Instant::now();
    transform(&mut rows, implementation);
    let elapsed = started.elapsed();
    black_box(rows);
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
    let mut expected = fixture.clone();
    let mut actual = fixture.clone();
    transform(&mut expected, historical_sort);
    transform(&mut actual, median7_network);
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

    for _ in 0..3 {
        black_box(timed(fixture.clone(), historical_sort));
        black_box(timed(fixture.clone(), median7_network));
    }

    let mut null_ratios = Vec::with_capacity(PAIRS);
    let mut candidate_ratios = Vec::with_capacity(PAIRS);
    let mut historical_ns = Vec::with_capacity(PAIRS);
    let mut candidate_ns = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        let null_first = timed(fixture.clone(), historical_sort);
        let null_second = timed(fixture.clone(), historical_sort);
        let null_ratio = if pair.is_multiple_of(2) {
            null_first.as_secs_f64() / null_second.as_secs_f64()
        } else {
            null_second.as_secs_f64() / null_first.as_secs_f64()
        };
        null_ratios.push(null_ratio);

        let (historical, candidate) = if pair.is_multiple_of(2) {
            (
                timed(fixture.clone(), historical_sort),
                timed(fixture.clone(), median7_network),
            )
        } else {
            let candidate = timed(fixture.clone(), median7_network);
            let historical = timed(fixture.clone(), historical_sort);
            (historical, candidate)
        };
        historical_ns.push(historical.as_nanos());
        candidate_ns.push(candidate.as_nanos());
        candidate_ratios.push(historical.as_secs_f64() / candidate.as_secs_f64());
    }

    let null_median = percentile(&null_ratios, 50);
    let null_p90 = percentile(&null_ratios, 90);
    let candidate_p10 = percentile(&candidate_ratios, 10);
    let candidate_median = percentile(&candidate_ratios, 50);
    let candidate_p90 = percentile(&candidate_ratios, 90);
    let wins = candidate_ratios
        .iter()
        .filter(|&&ratio| ratio > 1.0)
        .count();
    println!("BASE_BASE_RATIOS={null_ratios:?}");
    println!("HISTORICAL_NETWORK_RATIOS={candidate_ratios:?}");
    println!("HISTORICAL_NS={historical_ns:?}");
    println!("NETWORK_NS={candidate_ns:?}");
    println!("NULL_MEDIAN={null_median:.6} NULL_P90={null_p90:.6}");
    println!(
        "CANDIDATE_P10={candidate_p10:.6} CANDIDATE_MEDIAN={candidate_median:.6} CANDIDATE_P90={candidate_p90:.6} WINS={wins}/{PAIRS}"
    );
}
