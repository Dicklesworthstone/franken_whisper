use std::env;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const OWNED_FIELDS: usize = 12;
const PATH_FIELDS: usize = 4;

#[derive(Clone)]
struct ArgsShape {
    owned: [Option<String>; OWNED_FIELDS],
    paths: [Option<PathBuf>; PATH_FIELDS],
    gpu_device: Option<String>,
    timeout_s: Option<u64>,
    flags: u64,
}

struct RequestShape {
    owned: [Option<String>; OWNED_FIELDS],
    paths: [Option<PathBuf>; PATH_FIELDS],
    gpu_device: Option<String>,
    diarization_device: Option<String>,
    timeout_ms: Option<u64>,
    flags: u64,
}

impl ArgsShape {
    #[inline(never)]
    fn historical_request(&self) -> RequestShape {
        RequestShape {
            owned: self.owned.clone(),
            paths: self.paths.clone(),
            gpu_device: self.gpu_device.clone(),
            diarization_device: self.gpu_device.clone(),
            timeout_ms: self.timeout_s.map(|seconds| seconds.saturating_mul(1_000)),
            flags: self.flags,
        }
    }

    #[inline(never)]
    fn consuming_request(self) -> RequestShape {
        let diarization_device = self.gpu_device.clone();
        RequestShape {
            owned: self.owned,
            paths: self.paths,
            gpu_device: self.gpu_device,
            diarization_device,
            timeout_ms: self.timeout_s.map(|seconds| seconds.saturating_mul(1_000)),
            flags: self.flags,
        }
    }

    fn cloned_payload_bytes(&self) -> usize {
        self.owned.iter().flatten().map(String::len).sum::<usize>()
            + self
                .paths
                .iter()
                .flatten()
                .map(|path| path.as_os_str().len())
                .sum::<usize>()
            + self.gpu_device.as_ref().map_or(0, |value| value.len() * 2)
    }

    fn moved_payload_bytes(&self) -> usize {
        self.gpu_device.as_ref().map_or(0, String::len)
    }
}

#[derive(Clone, Copy)]
enum Shape {
    Minimal,
    FullShort,
    FullLarge,
}

impl Shape {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "minimal" => Some(Self::Minimal),
            "short" => Some(Self::FullShort),
            "large" => Some(Self::FullLarge),
            _ => None,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::FullShort => "short",
            Self::FullLarge => "large",
        }
    }

    const fn default_count(self) -> usize {
        match self {
            Self::Minimal => 100_000,
            Self::FullShort => 50_000,
            Self::FullLarge => 128,
        }
    }

    fn field_len(self) -> usize {
        match self {
            Self::Minimal => 0,
            Self::FullShort => 64,
            Self::FullLarge => 64 * 1_024,
        }
    }
}

#[derive(Clone, Copy)]
enum Variant {
    Historical,
    Consuming,
}

fn make_value(len: usize, seed: usize) -> String {
    let mut value = String::with_capacity(len);
    let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789";
    for index in 0..len {
        value.push(alphabet[(index + seed) % alphabet.len()] as char);
    }
    value
}

fn make_args(shape: Shape, seed: usize) -> ArgsShape {
    if matches!(shape, Shape::Minimal) {
        return ArgsShape {
            owned: std::array::from_fn(|_| None),
            paths: std::array::from_fn(|_| None),
            gpu_device: None,
            timeout_s: Some(seed as u64),
            flags: seed as u64,
        };
    }

    let len = shape.field_len();
    ArgsShape {
        owned: std::array::from_fn(|field| Some(make_value(len, seed + field))),
        paths: std::array::from_fn(|field| {
            Some(PathBuf::from(make_value(len, seed + OWNED_FIELDS + field)))
        }),
        gpu_device: Some(make_value(len, seed + OWNED_FIELDS + PATH_FIELDS)),
        timeout_s: Some(seed as u64),
        flags: seed as u64,
    }
}

fn make_pool(shape: Shape, count: usize) -> Vec<ArgsShape> {
    (0..count).map(|index| make_args(shape, index)).collect()
}

fn checksum(requests: &[RequestShape]) -> u64 {
    requests.iter().fold(0_u64, |sum, request| {
        let owned_len = request
            .owned
            .iter()
            .flatten()
            .map(String::len)
            .sum::<usize>();
        let path_len = request
            .paths
            .iter()
            .flatten()
            .map(|path| path.as_os_str().len())
            .sum::<usize>();
        sum.wrapping_add(owned_len as u64)
            .wrapping_add(path_len as u64)
            .wrapping_add(request.gpu_device.as_ref().map_or(0, String::len) as u64)
            .wrapping_add(request.diarization_device.as_ref().map_or(0, String::len) as u64)
            .wrapping_add(request.timeout_ms.unwrap_or(0))
            .wrapping_add(request.flags)
    })
}

#[inline(never)]
fn historical(pool: Vec<ArgsShape>) -> Vec<RequestShape> {
    let mut requests = Vec::with_capacity(pool.len());
    for args in &pool {
        requests.push(args.historical_request());
    }
    requests
}

#[inline(never)]
fn consuming(pool: Vec<ArgsShape>) -> Vec<RequestShape> {
    pool.into_iter().map(ArgsShape::consuming_request).collect()
}

fn time_once(variant: Variant, shape: Shape, count: usize) -> (Duration, u64) {
    let pool = make_pool(shape, count);
    let started = Instant::now();
    let requests = match variant {
        Variant::Historical => historical(pool),
        Variant::Consuming => consuming(pool),
    };
    let elapsed = started.elapsed();
    let digest = black_box(checksum(&requests));
    black_box(requests);
    (elapsed, digest)
}

fn median_ns(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn profile() {
    for shape in [Shape::Minimal, Shape::FullShort, Shape::FullLarge] {
        let count = shape.default_count();
        let sample = make_args(shape, 7);
        let historical_bytes = sample.cloned_payload_bytes();
        let consuming_bytes = sample.moved_payload_bytes();
        let mut historical_ns = Vec::with_capacity(5);
        let mut consuming_ns = Vec::with_capacity(5);
        let mut digest = None;
        for _ in 0..5 {
            let (elapsed, value) = time_once(Variant::Historical, shape, count);
            historical_ns.push(elapsed.as_nanos());
            digest.get_or_insert(value);
            assert_eq!(digest, Some(value));

            let (elapsed, value) = time_once(Variant::Consuming, shape, count);
            consuming_ns.push(elapsed.as_nanos());
            assert_eq!(digest, Some(value));
        }
        let historical = median_ns(&mut historical_ns);
        let consuming = median_ns(&mut consuming_ns);
        println!(
            "PROFILE shape={} count={} historical_ns={} consuming_ns={} ratio={:.4} historical_clone_bytes_per_request={} consuming_clone_bytes_per_request={} eliminated_bytes_pct={:.2} checksum={}",
            shape.name(),
            count,
            historical,
            consuming,
            historical as f64 / consuming as f64,
            historical_bytes,
            consuming_bytes,
            if historical_bytes == 0 {
                0.0
            } else {
                100.0 * (historical_bytes - consuming_bytes) as f64 / historical_bytes as f64
            },
            digest.unwrap_or(0),
        );
    }
}

fn measure(shape: Shape, pairs: usize) {
    let count = shape.default_count();
    let mut historical_ns = Vec::with_capacity(pairs * 2);
    let mut consuming_ns = Vec::with_capacity(pairs * 2);
    let mut null_first_ns = Vec::with_capacity(pairs);
    let mut null_second_ns = Vec::with_capacity(pairs);
    let mut digest = None;

    for _ in 0..pairs {
        let (a1, value) = time_once(Variant::Historical, shape, count);
        digest.get_or_insert(value);
        assert_eq!(digest, Some(value));
        let (b1, value) = time_once(Variant::Consuming, shape, count);
        assert_eq!(digest, Some(value));
        let (b2, value) = time_once(Variant::Consuming, shape, count);
        assert_eq!(digest, Some(value));
        let (a2, value) = time_once(Variant::Historical, shape, count);
        assert_eq!(digest, Some(value));
        historical_ns.extend([a1.as_nanos(), a2.as_nanos()]);
        consuming_ns.extend([b1.as_nanos(), b2.as_nanos()]);

        let (n1, value) = time_once(Variant::Historical, shape, count);
        assert_eq!(digest, Some(value));
        let (n2, value) = time_once(Variant::Historical, shape, count);
        assert_eq!(digest, Some(value));
        null_first_ns.push(n1.as_nanos());
        null_second_ns.push(n2.as_nanos());
    }

    let historical = median_ns(&mut historical_ns);
    let consuming = median_ns(&mut consuming_ns);
    let null_first = median_ns(&mut null_first_ns);
    let null_second = median_ns(&mut null_second_ns);
    println!(
        "RESULT shape={} count={} pairs={} historical_median_ns={} consuming_median_ns={} ab_ratio={:.4} reduction_pct={:.2} null_a_median_ns={} null_b_median_ns={} null_ratio={:.4} checksum={}",
        shape.name(),
        count,
        pairs,
        historical,
        consuming,
        historical as f64 / consuming as f64,
        100.0 * (historical - consuming) as f64 / historical as f64,
        null_first,
        null_second,
        null_first as f64 / null_second as f64,
        digest.unwrap_or(0),
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str).unwrap_or("profile") {
        "profile" => profile(),
        "measure" => {
            let shape = match args.get(2).map(String::as_str) {
                Some(value) => match Shape::parse(value) {
                    Some(shape) => shape,
                    None => {
                        eprintln!("unknown shape: {value}");
                        std::process::exit(2);
                    }
                },
                None => Shape::FullShort,
            };
            let pairs = match args.get(3) {
                Some(value) => match value.parse::<usize>() {
                    Ok(pairs) if pairs > 0 => pairs,
                    _ => {
                        eprintln!("pairs must be a non-zero usize");
                        std::process::exit(2);
                    }
                },
                None => 21,
            };
            measure(shape, pairs);
        }
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(2);
        }
    }
}
