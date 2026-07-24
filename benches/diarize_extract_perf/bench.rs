// A/B for a byte-exact fusion of diarize_segments' Step-1 embedding extraction.
// The original makes FOUR passes over the segments (max_seg_duration,
// max_word_count, max_text_len, then the embedding map) and calls
// `text.split_whitespace()` TWICE per segment (once to count words for
// max_word_count, once inside the embedding). The fused form makes TWO passes:
// pass 1 splits each segment ONCE, storing (word_count, total_chars) and folding
// all three maxes; pass 2 builds the embedding reusing the stored counts. Maxes
// fold over the same values in the same order, and the stored counts equal the
// re-split counts, so every feature is bit-identical.
use std::hint::black_box;
use std::time::{Duration, Instant};

struct Seg {
    start_sec: Option<f64>,
    end_sec: Option<f64>,
    text: String,
}

fn extract_current(segs: &[Seg], duration: f64) -> Vec<[f64; 6]> {
    let max_seg_duration = segs
        .iter()
        .map(|s| {
            let start = s.start_sec.unwrap_or(0.0);
            let end = s.end_sec.unwrap_or(start);
            (end - start).max(0.0)
        })
        .fold(0.0_f64, f64::max)
        .max(1e-6);
    let max_word_count = segs
        .iter()
        .map(|s| s.text.split_whitespace().count() as f64)
        .fold(1.0_f64, f64::max);
    let max_text_len = segs.iter().map(|s| s.text.len() as f64).fold(1.0_f64, f64::max);

    segs.iter()
        .enumerate()
        .map(|(i, seg)| {
            let start = seg.start_sec.unwrap_or(0.0);
            let end = seg.end_sec.unwrap_or(start);
            let seg_duration = (end - start).max(0.0);
            let midpoint_norm = ((start + end) / 2.0) / duration;
            let duration_norm = seg_duration / max_seg_duration;
            let gap = if i > 0 {
                let prev_end = segs[i - 1].end_sec.unwrap_or(0.0);
                ((start - prev_end).max(0.0) / duration).min(1.0)
            } else {
                0.0
            };
            let mut word_count = 0usize;
            let mut total_chars = 0usize;
            for word in seg.text.split_whitespace() {
                word_count += 1;
                total_chars += word.len();
            }
            let word_count_norm = word_count as f64 / max_word_count;
            let avg_word_len = if word_count == 0 {
                0.0
            } else {
                (total_chars as f64 / word_count as f64) / 12.0
            };
            let text_len_norm = seg.text.len() as f64 / max_text_len;
            [
                midpoint_norm,
                duration_norm,
                gap,
                word_count_norm,
                avg_word_len,
                text_len_norm,
            ]
        })
        .collect()
}

fn extract_fused(segs: &[Seg], duration: f64) -> Vec<[f64; 6]> {
    // Pass 1: split each segment once; fold all three maxes.
    let mut word_counts: Vec<usize> = Vec::with_capacity(segs.len());
    let mut char_counts: Vec<usize> = Vec::with_capacity(segs.len());
    let mut max_seg_duration = 0.0_f64;
    let mut max_word_count = 1.0_f64;
    let mut max_text_len = 1.0_f64;
    for s in segs {
        let start = s.start_sec.unwrap_or(0.0);
        let end = s.end_sec.unwrap_or(start);
        max_seg_duration = max_seg_duration.max((end - start).max(0.0));
        let mut wc = 0usize;
        let mut tc = 0usize;
        for word in s.text.split_whitespace() {
            wc += 1;
            tc += word.len();
        }
        word_counts.push(wc);
        char_counts.push(tc);
        max_word_count = max_word_count.max(wc as f64);
        max_text_len = max_text_len.max(s.text.len() as f64);
    }
    let max_seg_duration = max_seg_duration.max(1e-6);

    // Pass 2: embedding, reusing stored counts.
    segs.iter()
        .enumerate()
        .map(|(i, seg)| {
            let start = seg.start_sec.unwrap_or(0.0);
            let end = seg.end_sec.unwrap_or(start);
            let seg_duration = (end - start).max(0.0);
            let midpoint_norm = ((start + end) / 2.0) / duration;
            let duration_norm = seg_duration / max_seg_duration;
            let gap = if i > 0 {
                let prev_end = segs[i - 1].end_sec.unwrap_or(0.0);
                ((start - prev_end).max(0.0) / duration).min(1.0)
            } else {
                0.0
            };
            let word_count = word_counts[i];
            let total_chars = char_counts[i];
            let word_count_norm = word_count as f64 / max_word_count;
            let avg_word_len = if word_count == 0 {
                0.0
            } else {
                (total_chars as f64 / word_count as f64) / 12.0
            };
            let text_len_norm = seg.text.len() as f64 / max_text_len;
            [
                midpoint_norm,
                duration_norm,
                gap,
                word_count_norm,
                avg_word_len,
                text_len_norm,
            ]
        })
        .collect()
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
}

const WORDS: [&str; 12] = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello", "world",
    "transcription", "segment",
];

fn fixture(n: usize) -> Vec<Seg> {
    let mut lcg = Lcg(0xEC_7A + n as u64);
    let mut cursor = 0.0_f64;
    (0..n)
        .map(|_| {
            let words = 3 + (lcg.next() % 13) as usize;
            let mut text = String::new();
            for w in 0..words {
                if w > 0 {
                    text.push(' ');
                }
                text.push_str(WORDS[(lcg.next() % 12) as usize]);
            }
            let dur = 0.5 + (lcg.next() % 40) as f64 / 10.0;
            let start = cursor;
            let end = cursor + dur;
            cursor = end + (lcg.next() % 5) as f64 / 10.0;
            Seg {
                start_sec: Some(start),
                end_sec: Some(end),
                text,
            }
        })
        .collect()
}

fn bits(v: &[[f64; 6]]) -> Vec<u64> {
    v.iter().flat_map(|f| f.iter().map(|x| x.to_bits())).collect()
}

fn timed(segs: &[Seg], duration: f64, f: fn(&[Seg], f64) -> Vec<[f64; 6]>) -> Duration {
    let started = Instant::now();
    let out = f(black_box(segs), black_box(duration));
    let elapsed = started.elapsed();
    black_box(out.len());
    black_box(out.last().map(|x| x[5].to_bits()));
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
    const SIZES: [usize; 3] = [200, 1000, 3000];
    const PAIRS: usize = 25;
    const WARMUP: usize = 6;

    for n in SIZES {
        let segs = fixture(n);
        let duration = segs
            .iter()
            .filter_map(|s| s.end_sec)
            .fold(1.0_f64, f64::max)
            .max(1e-6);

        assert_eq!(
            bits(&extract_current(&segs, duration)),
            bits(&extract_fused(&segs, duration)),
            "extraction bits differ at n={n}"
        );

        for _ in 0..WARMUP {
            black_box(timed(&segs, duration, extract_current));
            black_box(timed(&segs, duration, extract_fused));
        }

        let mut null_ratios = Vec::with_capacity(PAIRS);
        let mut speedups = Vec::with_capacity(PAIRS);
        let mut cur_ns = Vec::with_capacity(PAIRS);
        let mut fus_ns = Vec::with_capacity(PAIRS);
        for pair in 0..PAIRS {
            let a = timed(&segs, duration, extract_current);
            let b = timed(&segs, duration, extract_current);
            null_ratios.push(if pair.is_multiple_of(2) {
                a.as_secs_f64() / b.as_secs_f64()
            } else {
                b.as_secs_f64() / a.as_secs_f64()
            });

            let (cur, fus) = if pair.is_multiple_of(2) {
                (timed(&segs, duration, extract_current), timed(&segs, duration, extract_fused))
            } else {
                let fus = timed(&segs, duration, extract_fused);
                let cur = timed(&segs, duration, extract_current);
                (cur, fus)
            };
            cur_ns.push(cur.as_nanos());
            fus_ns.push(fus.as_nanos());
            speedups.push(cur.as_secs_f64() / fus.as_secs_f64());
        }

        let wins = speedups.iter().filter(|&&r| r > 1.0).count();
        println!(
            "n={n} pairs={PAIRS} null_p10={:.4} null_median={:.4} null_p90={:.4} current_median_ns={} fused_median_ns={} speedup_p10={:.4} speedup_median={:.4} speedup_p90={:.4} wins={wins}/{PAIRS}",
            percentile(&null_ratios, 10),
            percentile(&null_ratios, 50),
            percentile(&null_ratios, 90),
            median_ns(&cur_ns),
            median_ns(&fus_ns),
            percentile(&speedups, 10),
            percentile(&speedups, 50),
            percentile(&speedups, 90),
        );
    }
}
