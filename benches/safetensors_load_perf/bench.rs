use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

const DATA_BYTES: usize = 32 * 1024 * 1024;
const TENSORS: usize = 256;
const PAIRS: usize = 15;
const ROUNDS_PER_ARM: usize = 2;

struct ParsedFile {
    data: Vec<u8>,
    data_offset: usize,
    tensor_count: usize,
    directory_signature: u64,
}

impl ParsedFile {
    fn signature(&self) -> u64 {
        let payload = &self.data[self.data_offset..];
        self.directory_signature
            .wrapping_mul(31)
            .wrapping_add(self.tensor_count as u64)
            .wrapping_add(payload.len() as u64)
            .wrapping_add(payload[0] as u64)
            .wrapping_add((payload[payload.len() / 2] as u64) << 8)
            .wrapping_add((payload[payload.len() - 1] as u64) << 16)
    }
}

fn fixture() -> Vec<u8> {
    let tensor_bytes = DATA_BYTES / TENSORS;
    let mut header = serde_json::Map::new();
    for tensor in 0..TENSORS {
        let begin = tensor * tensor_bytes;
        let end = if tensor + 1 == TENSORS {
            DATA_BYTES
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
    let mut bytes = Vec::with_capacity(8 + header.len() + DATA_BYTES);
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend((0..DATA_BYTES).map(|index| (index.wrapping_mul(131) & 0xff) as u8));
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

fn parse_copy(bytes: Vec<u8>) -> ParsedFile {
    let (data_offset, tensor_count, directory_signature) = parse_header(&bytes);
    let data = bytes[data_offset..].to_vec();
    ParsedFile {
        data,
        data_offset: 0,
        tensor_count,
        directory_signature,
    }
}

fn parse_owned(bytes: Vec<u8>) -> ParsedFile {
    let (data_offset, tensor_count, directory_signature) = parse_header(&bytes);
    ParsedFile {
        data: bytes,
        data_offset,
        tensor_count,
        directory_signature,
    }
}

fn parse_signature(fixture: &[u8], owned: bool, rounds: usize) -> u64 {
    let mut signature = 0_u64;
    for _ in 0..rounds {
        let bytes = black_box(fixture.to_vec());
        let parsed = if owned {
            parse_owned(bytes)
        } else {
            parse_copy(bytes)
        };
        signature = signature.wrapping_add(black_box(parsed.signature()));
        black_box(parsed);
    }
    black_box(signature)
}

fn timed(fixture: &[u8], owned: bool) -> (Duration, u64) {
    let started = Instant::now();
    let signature = parse_signature(fixture, owned, ROUNDS_PER_ARM);
    (started.elapsed(), signature)
}

fn paired_ratios(fixture: &[u8], owned_second: bool) -> Vec<f64> {
    let mut ratios = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        let (first_before, second_before, second_after, first_after) = if pair % 2 == 0 {
            let first_before = timed(fixture, false);
            let second_before = timed(fixture, owned_second);
            let second_after = timed(fixture, owned_second);
            let first_after = timed(fixture, false);
            (first_before, second_before, second_after, first_after)
        } else {
            let second_before = timed(fixture, owned_second);
            let first_before = timed(fixture, false);
            let first_after = timed(fixture, false);
            let second_after = timed(fixture, owned_second);
            (first_before, second_before, second_after, first_after)
        };
        assert_eq!(first_before.1, second_before.1);
        assert_eq!(first_after.1, second_after.1);
        ratios.push(
            (first_before.0 + first_after.0).as_secs_f64()
                / (second_before.0 + second_after.0).as_secs_f64(),
        );
    }
    ratios
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
    let fixture = fixture();
    for _ in 0..3 {
        assert_eq!(
            parse_signature(&fixture, false, 1),
            parse_signature(&fixture, true, 1)
        );
    }

    let null = paired_ratios(&fixture, false);
    let candidate = paired_ratios(&fixture, true);
    let null_median = percentile(&null, 50);
    let null_p90 = percentile(&null, 90);
    let candidate_p10 = percentile(&candidate, 10);
    let candidate_median = percentile(&candidate, 50);
    let candidate_wins = candidate.iter().filter(|&&ratio| ratio > 1.0).count();
    println!(
        "FIXTURE_BYTES={} DATA_BYTES={DATA_BYTES} TENSORS={TENSORS} PARSES_PER_TIMED_ARM={ROUNDS_PER_ARM}",
        fixture.len()
    );
    println!("BASE_BASE_RATIOS={null:?}");
    println!("BASE_CANDIDATE_RATIOS={candidate:?}");
    println!(
        "NULL_MEDIAN={null_median:.6} NULL_P90={null_p90:.6} NULL_CV_PCT={:.3} CANDIDATE_P10={candidate_p10:.6} CANDIDATE_MEDIAN={candidate_median:.6} CANDIDATE_CV_PCT={:.3} CANDIDATE_WINS={candidate_wins}/15",
        cv_percent(&null),
        cv_percent(&candidate)
    );
}
