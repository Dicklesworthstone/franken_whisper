// A/B for the sync JSONL-export per-row write. The three export_table_* loops
// (runs / segments / events) each do
//   `writeln!(writer, "{}", serde_json::to_string(&obj)?)?`
// per row — allocating an intermediate String (+ UTF-8 validation) AND paying the
// `write!` fmt machinery on top — before writing to the buffered+hashing writer.
// Streaming with `to_writer(&mut writer, &obj); writer.write_all(b"\n")` emits the
// SAME bytes (so the HashingWriter hash is identical) with no per-row allocation.
use std::hint::black_box;
use std::io::Write;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

// Current: to_string + writeln! per row.
fn export_to_string(rows: &[Value], sink: &mut Vec<u8>) {
    sink.clear();
    let mut writer = std::io::BufWriter::new(sink);
    for obj in rows {
        writeln!(writer, "{}", serde_json::to_string(obj).unwrap()).unwrap();
    }
    writer.flush().unwrap();
}

// Candidate: serialize into a REUSED scratch Vec (no per-row String alloc), then
// one write_all of the whole line to the BufWriter (one big write, like the
// to_string path — avoids to_writer's many small BufWriter writes).
fn export_to_writer(rows: &[Value], sink: &mut Vec<u8>) {
    sink.clear();
    let mut writer = std::io::BufWriter::new(sink);
    let mut scratch: Vec<u8> = Vec::new();
    for obj in rows {
        scratch.clear();
        serde_json::to_writer(&mut scratch, obj).unwrap();
        scratch.push(b'\n');
        writer.write_all(&scratch).unwrap();
    }
    writer.flush().unwrap();
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

// A realistic `runs`-table row: 12 fields, several holding embedded JSON strings
// (request_json / result_json / transcript can be large — the full transcript).
fn rows(n: usize) -> Vec<Value> {
    let mut lcg = Lcg(0x5E_C0 + n as u64);
    (0..n)
        .map(|i| {
            let seg_count = 5 + (lcg.next() % 40) as usize;
            let result: String = {
                let mut s = String::from("{\"segments\":[");
                for s_i in 0..seg_count {
                    if s_i > 0 {
                        s.push(',');
                    }
                    s.push_str(&format!(
                        "{{\"start\":{}.5,\"end\":{}.0,\"text\":\"the quick brown fox {s_i}\"}}",
                        s_i, s_i + 1
                    ));
                }
                s.push_str("]}");
                s
            };
            json!({
                "id": format!("01HXYZ{i:08}"),
                "started_at": "2026-07-16T07:05:00Z",
                "finished_at": "2026-07-16T07:05:12Z",
                "backend": "native-whisper-cpp",
                "input_path": format!("/data/audio/clip_{i:05}.m4a"),
                "normalized_wav_path": format!("/data/tmp/clip_{i:05}.wav"),
                "request_json": "{\"model\":\"turbo\",\"language\":\"en\",\"diarize\":false}",
                "result_json": result,
                "warnings_json": Value::Null,
                "transcript": format!("segment text {i} the quick brown fox jumps over the lazy dog"),
                "replay_json": Value::Null,
                "acceleration_json": "{\"backend\":\"none\",\"normalized\":true}",
            })
        })
        .collect()
}

fn timed(rows: &[Value], sink: &mut Vec<u8>, f: fn(&[Value], &mut Vec<u8>)) -> Duration {
    let started = Instant::now();
    f(black_box(rows), sink);
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
    const SIZES: [usize; 3] = [100, 1000, 8000];
    const PAIRS: usize = 25;
    const WARMUP: usize = 5;

    for n in SIZES {
        let rs = rows(n);
        let mut a = Vec::new();
        let mut b = Vec::new();

        export_to_string(&rs, &mut a);
        export_to_writer(&rs, &mut b);
        assert_eq!(a, b, "JSONL bytes differ at n={n}");

        for _ in 0..WARMUP {
            black_box(timed(&rs, &mut a, export_to_string));
            black_box(timed(&rs, &mut b, export_to_writer));
        }

        let mut null_ratios = Vec::with_capacity(PAIRS);
        let mut speedups = Vec::with_capacity(PAIRS);
        let mut str_ns = Vec::with_capacity(PAIRS);
        let mut wr_ns = Vec::with_capacity(PAIRS);
        for pair in 0..PAIRS {
            let x = timed(&rs, &mut a, export_to_string);
            let y = timed(&rs, &mut a, export_to_string);
            null_ratios.push(if pair.is_multiple_of(2) {
                x.as_secs_f64() / y.as_secs_f64()
            } else {
                y.as_secs_f64() / x.as_secs_f64()
            });

            let (s, w) = if pair.is_multiple_of(2) {
                (timed(&rs, &mut a, export_to_string), timed(&rs, &mut b, export_to_writer))
            } else {
                let w = timed(&rs, &mut b, export_to_writer);
                let s = timed(&rs, &mut a, export_to_string);
                (s, w)
            };
            str_ns.push(s.as_nanos());
            wr_ns.push(w.as_nanos());
            speedups.push(s.as_secs_f64() / w.as_secs_f64());
        }

        let wins = speedups.iter().filter(|&&r| r > 1.0).count();
        println!(
            "rows={n} bytes={} pairs={PAIRS} null_p10={:.4} null_median={:.4} null_p90={:.4} to_string_median_ns={} to_writer_median_ns={} speedup_p10={:.4} speedup_median={:.4} speedup_p90={:.4} wins={wins}/{PAIRS}",
            a.len(),
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
