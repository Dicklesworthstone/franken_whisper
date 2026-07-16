// A/B for a byte-exact change to robot::emit_line: instead of
// `serde_json::to_string(value)` (allocate an intermediate String + validate
// UTF-8) then write it + a newline, use `serde_json::to_writer(sink, value)`
// which streams the same JSON bytes straight into the sink with no intermediate
// allocation. The emitted bytes are identical. Robot mode emits one NDJSON line
// per event (per segment / stage / progress), so this is a per-event win.
use std::hint::black_box;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

// Current: to_string + write String + newline.
fn emit_to_string(events: &[Value], sink: &mut Vec<u8>) {
    sink.clear();
    for value in events {
        let s = serde_json::to_string(value).unwrap();
        sink.extend_from_slice(s.as_bytes());
        sink.push(b'\n');
    }
}

// Candidate: to_writer directly + newline.
fn emit_to_writer(events: &[Value], sink: &mut Vec<u8>) {
    sink.clear();
    for value in events {
        serde_json::to_writer(&mut *sink, value).unwrap();
        sink.push(b'\n');
    }
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

// A realistic robot `transcript.partial`-shaped event.
fn events(n: usize) -> Vec<Value> {
    let mut lcg = Lcg(0xB0B07 + n as u64);
    (0..n)
        .map(|i| {
            let start = i as f64 * 1.37;
            json!({
                "event": "transcript.partial",
                "schema_version": "1.2.0",
                "run_id": "run-01HXYZ8Q9K3M4N5P6R7S8T9V0W",
                "seq": i as u64,
                "ts": "2026-07-16T07:05:00.123456Z",
                "segment": {
                    "start_sec": start,
                    "end_sec": start + 1.25,
                    "text": format!("segment {i:04}: the quick brown fox jumps over the lazy dog"),
                    "speaker": if lcg.next() % 2 == 0 { "SPEAKER_00" } else { "SPEAKER_01" },
                    "confidence": (lcg.next() % 1000) as f64 / 1000.0,
                }
            })
        })
        .collect()
}

fn timed(events: &[Value], sink: &mut Vec<u8>, f: fn(&[Value], &mut Vec<u8>)) -> Duration {
    let started = Instant::now();
    f(black_box(events), sink);
    let elapsed = started.elapsed();
    black_box(sink.len());
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
    const SIZES: [usize; 3] = [64, 512, 4096];
    const PAIRS: usize = 25;
    const WARMUP: usize = 6;

    for n in SIZES {
        let evs = events(n);
        let mut a_sink = Vec::new();
        let mut b_sink = Vec::new();

        // Byte-exactness: both paths must emit identical NDJSON bytes.
        emit_to_string(&evs, &mut a_sink);
        emit_to_writer(&evs, &mut b_sink);
        assert_eq!(a_sink, b_sink, "NDJSON bytes differ at n={n}");

        for _ in 0..WARMUP {
            black_box(timed(&evs, &mut a_sink, emit_to_string));
            black_box(timed(&evs, &mut b_sink, emit_to_writer));
        }

        let mut null_ratios = Vec::with_capacity(PAIRS);
        let mut speedups = Vec::with_capacity(PAIRS);
        let mut str_ns = Vec::with_capacity(PAIRS);
        let mut wr_ns = Vec::with_capacity(PAIRS);
        for pair in 0..PAIRS {
            let x = timed(&evs, &mut a_sink, emit_to_string);
            let y = timed(&evs, &mut a_sink, emit_to_string);
            null_ratios.push(if pair.is_multiple_of(2) {
                x.as_secs_f64() / y.as_secs_f64()
            } else {
                y.as_secs_f64() / x.as_secs_f64()
            });

            let (s, w) = if pair.is_multiple_of(2) {
                (timed(&evs, &mut a_sink, emit_to_string), timed(&evs, &mut b_sink, emit_to_writer))
            } else {
                let w = timed(&evs, &mut b_sink, emit_to_writer);
                let s = timed(&evs, &mut a_sink, emit_to_string);
                (s, w)
            };
            str_ns.push(s.as_nanos());
            wr_ns.push(w.as_nanos());
            speedups.push(s.as_secs_f64() / w.as_secs_f64());
        }

        let wins = speedups.iter().filter(|&&r| r > 1.0).count();
        println!(
            "n={n} bytes={} pairs={PAIRS} null_p10={:.4} null_median={:.4} null_p90={:.4} to_string_median_ns={} to_writer_median_ns={} speedup_p10={:.4} speedup_median={:.4} speedup_p90={:.4} wins={wins}/{PAIRS}",
            a_sink.len(),
            percentile(&null_ratios, 10),
            percentile(&null_ratios, 50),
            percentile(&null_ratios, 90),
            median_ns(&str_ns),
            median_ns(&wr_ns),
            percentile(&speedups, 10),
            percentile(&speedups, 50),
            percentile(&speedups, 90),
        );
    }
}
