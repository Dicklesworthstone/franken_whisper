use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

// Payload-size sweep: the lever (retain the owned fs::read buffer vs copying the
// whole data section out) must win at every representative auxiliary-model size,
// not just 32 MiB.
// Sizes are chosen to sit clearly on one side of this box's ~32 MiB L3-per-CCD
// knee: 4 MiB (8 MiB working set, in-cache) and 32/64/128 MiB (64/128/256 MiB
// working sets, firmly DRAM-bound). A payload near 16 MiB (32 MiB working set)
// lands exactly on the knee and its BASE/BASE null goes bimodal (CV 8-15%) — a
// pure measurement artifact of this hardware, not the lever; the lever still
// wins ~2x/31-of-31 there (see the ledger). Fixed low ROUNDS_PER_ARM keeps each
// timed call a few ms without the frequency-drift noise that long many-round
// in-cache calls accrue.
const SIZES: [usize; 4] = [4 << 20, 32 << 20, 64 << 20, 128 << 20];
const MAX_DATA_BYTES: usize = 128 << 20;
const TENSORS: usize = 256;
const PAIRS: usize = 31;
const WARMUP_ROUNDS: usize = 12;
const ROUNDS_PER_ARM: usize = 2;

fn fixture(data_bytes: usize) -> Vec<u8> {
    let tensor_bytes = data_bytes / TENSORS;
    let mut header = serde_json::Map::new();
    for tensor in 0..TENSORS {
        let begin = tensor * tensor_bytes;
        let end = if tensor + 1 == TENSORS {
            data_bytes
        } else {
            begin + tensor_bytes
        };
        header.insert(
            format!("encoder.layer.{tensor:03}.weight"),
            serde_json::json!({
                "dtype": "F32",
                "shape": [(end - begin) / 4],
                "data_offsets": [begin, end],
            }),
        );
    }
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({"source": "representative auxiliary model"}),
    );

    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut bytes = Vec::with_capacity(8 + header.len() + data_bytes);
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend((0..data_bytes).map(|index| (index.wrapping_mul(131) & 0xff) as u8));
    bytes
}

fn parse_header(bytes: &[u8]) -> (usize, usize, u64) {
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let data_offset = 8 + header_len;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..data_offset]).unwrap();
    let mut tensors = BTreeMap::new();
    let mut signature = 0_u64;
    for (name, value) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let offsets = value["data_offsets"].as_array().unwrap();
        let begin = offsets[0].as_u64().unwrap() as usize;
        let end = offsets[1].as_u64().unwrap() as usize;
        assert!(begin <= end && end <= bytes.len() - data_offset);
        tensors.insert(name.clone(), (begin, end));
        signature = signature
            .wrapping_mul(31)
            .wrapping_add(name.len() as u64)
            .wrapping_add(begin as u64)
            .wrapping_add(end as u64);
    }
    (data_offset, tensors.len(), signature)
}

fn payload_signature(directory_signature: u64, tensor_count: usize, payload: &[u8]) -> u64 {
    directory_signature
        .wrapping_mul(31)
        .wrapping_add(tensor_count as u64)
        .wrapping_add(payload.len() as u64)
        .wrapping_add(payload[0] as u64)
        .wrapping_add((payload[payload.len() / 2] as u64) << 8)
        .wrapping_add((payload[payload.len() - 1] as u64) << 16)
}

/// One load, timed. `read_buf`/`copy_buf` are reused across every call — refilled
/// with `copy_from_slice` (allocation-free, resident pages) so no 33.5 MB `to_vec`
/// mmaps/munmaps churn the allocator per timed call, which was the source of the
/// prior 1.023x null bias. `read_buf` refill models the common `std::fs::read`;
/// the historical arm additionally fills `copy_buf` from the payload span (models
/// `from_bytes`' `data = bytes[header_end..].to_vec()`), while the owned arm reads
/// the payload straight out of `read_buf` with no second copy.
fn timed(read_buf: &mut [u8], copy_buf: &mut [u8], fixture: &[u8], owned: bool) -> (Duration, u64) {
    let fixture_len = fixture.len();
    let started = Instant::now();
    let mut signature = 0_u64;
    for _ in 0..ROUNDS_PER_ARM {
        let read = &mut read_buf[..fixture_len];
        read.copy_from_slice(fixture);
        let (data_offset, tensor_count, directory_signature) = parse_header(read);
        let payload_len = fixture_len - data_offset;
        let round = if owned {
            payload_signature(directory_signature, tensor_count, &read[data_offset..])
        } else {
            let copy = &mut copy_buf[..payload_len];
            copy.copy_from_slice(&read[data_offset..]);
            payload_signature(directory_signature, tensor_count, copy)
        };
        signature = signature.wrapping_add(round);
        black_box(&copy_buf[0]);
    }
    let elapsed = started.elapsed();
    black_box(&read_buf[0]);
    (elapsed, signature)
}

fn percentile(ratios: &[f64], percent: usize) -> f64 {
    let mut sorted = ratios.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = (sorted.len() * percent).div_ceil(100).saturating_sub(1);
    sorted[index.min(sorted.len() - 1)]
}

fn cv_percent(ratios: &[f64]) -> f64 {
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let variance = ratios
        .iter()
        .map(|ratio| (ratio - mean).powi(2))
        .sum::<f64>()
        / (ratios.len() - 1) as f64;
    variance.sqrt() / mean * 100.0
}

fn main() {
    let mut read_buf = vec![0u8; MAX_DATA_BYTES + (1 << 20)];
    let mut copy_buf = vec![0u8; MAX_DATA_BYTES + (1 << 20)];

    for data_bytes in SIZES {
        let fixture = fixture(data_bytes);

        // Exactness: owned and copy arms must produce the same signature.
        let (_, owned_sig) = timed(&mut read_buf, &mut copy_buf, &fixture, true);
        let (_, copy_sig) = timed(&mut read_buf, &mut copy_buf, &fixture, false);
        assert_eq!(owned_sig, copy_sig, "signature mismatch at {data_bytes} bytes");

        for _ in 0..WARMUP_ROUNDS {
            black_box(timed(&mut read_buf, &mut copy_buf, &fixture, false));
            black_box(timed(&mut read_buf, &mut copy_buf, &fixture, true));
        }

        let mut null_ratios = Vec::with_capacity(PAIRS);
        let mut candidate_ratios = Vec::with_capacity(PAIRS);
        for pair in 0..PAIRS {
            // Null: two adjacent copy-arm calls, order-alternated.
            let a = timed(&mut read_buf, &mut copy_buf, &fixture, false).0;
            let b = timed(&mut read_buf, &mut copy_buf, &fixture, false).0;
            null_ratios.push(if pair.is_multiple_of(2) {
                a.as_secs_f64() / b.as_secs_f64()
            } else {
                b.as_secs_f64() / a.as_secs_f64()
            });

            // Candidate: copy (historical) vs owned, order-alternated. Ratio is
            // historical / owned, so > 1.0 means the owned lever is faster.
            let (historical, owned) = if pair.is_multiple_of(2) {
                let historical = timed(&mut read_buf, &mut copy_buf, &fixture, false).0;
                let owned = timed(&mut read_buf, &mut copy_buf, &fixture, true).0;
                (historical, owned)
            } else {
                let owned = timed(&mut read_buf, &mut copy_buf, &fixture, true).0;
                let historical = timed(&mut read_buf, &mut copy_buf, &fixture, false).0;
                (historical, owned)
            };
            candidate_ratios.push(historical.as_secs_f64() / owned.as_secs_f64());
        }

        let null_p10 = percentile(&null_ratios, 10);
        let null_median = percentile(&null_ratios, 50);
        let null_p90 = percentile(&null_ratios, 90);
        let candidate_p10 = percentile(&candidate_ratios, 10);
        let candidate_median = percentile(&candidate_ratios, 50);
        let candidate_p90 = percentile(&candidate_ratios, 90);
        let wins = candidate_ratios.iter().filter(|&&r| r > 1.0).count();
        println!(
            "SIZE_MiB={} TENSORS={TENSORS} PAIRS={PAIRS} WARMUP={WARMUP_ROUNDS} ROUNDS_PER_CALL={ROUNDS_PER_ARM}",
            data_bytes >> 20
        );
        println!("  NULL_P10={null_p10:.6} NULL_MEDIAN={null_median:.6} NULL_P90={null_p90:.6} NULL_CV_PCT={:.3}", cv_percent(&null_ratios));
        println!("  CANDIDATE_P10={candidate_p10:.6} CANDIDATE_MEDIAN={candidate_median:.6} CANDIDATE_P90={candidate_p90:.6} CANDIDATE_CV_PCT={:.3} WINS={wins}/{PAIRS}", cv_percent(&candidate_ratios));
        println!("  BASE_BASE_RATIOS={null_ratios:?}");
        println!("  BASE_CANDIDATE_RATIOS={candidate_ratios:?}");
    }
}
