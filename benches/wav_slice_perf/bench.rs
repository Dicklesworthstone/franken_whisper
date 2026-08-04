use std::hint::black_box;
use std::io::Cursor;
use std::time::{Duration, Instant};

const SAMPLE_RATE: u32 = 16_000;
const SOURCE_SECONDS: u32 = 10 * 60;
const SLICE_START_MS: u64 = 5 * 60 * 1_000;
const SLICE_END_MS: u64 = SLICE_START_MS + 30_000;
const PROFILE_REPS: usize = 9;
const PAIRED_REPS: usize = 21;
const TARGET_ARM_SECS: f64 = 0.050;

#[derive(Clone, Copy)]
struct FrameRange {
    start: usize,
    end: usize,
}

fn wav_spec() -> hound::WavSpec {
    hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    }
}

fn fixture() -> Vec<u8> {
    let sample_count = SAMPLE_RATE as usize * SOURCE_SECONDS as usize;
    let mut output = Cursor::new(Vec::with_capacity(44 + sample_count * 2));
    {
        let mut writer = hound::WavWriter::new(&mut output, wav_spec()).expect("create fixture");
        for index in 0..sample_count {
            let mixed = (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .rotate_left(17);
            writer
                .write_sample((mixed >> 48) as i16)
                .expect("write fixture sample");
        }
        writer.finalize().expect("finalize fixture");
    }
    output.into_inner()
}

fn frame_range(
    total_frames: u64,
    channels: u16,
    sample_rate: u32,
    start_ms: u64,
    end_ms: u64,
) -> FrameRange {
    let total_duration_ms = total_frames.saturating_mul(1_000) / u64::from(sample_rate);
    let clamped_start = start_ms.min(total_duration_ms);
    let clamped_end = end_ms.max(clamped_start).min(total_duration_ms);
    let start_frame = clamped_start.saturating_mul(u64::from(sample_rate)) / 1_000;
    let end_frame = clamped_end.saturating_mul(u64::from(sample_rate)) / 1_000;
    let channels = usize::from(channels);
    FrameRange {
        start: usize::try_from(start_frame).expect("start frame fits usize") * channels,
        end: usize::try_from(end_frame).expect("end frame fits usize") * channels,
    }
}

fn decode_all(source: &[u8]) -> (hound::WavSpec, Vec<i32>) {
    let reader = hound::WavReader::new(Cursor::new(source)).expect("open source WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    let samples = reader
        .into_samples::<i32>()
        .collect::<Result<Vec<_>, _>>()
        .expect("decode complete source WAV");
    (spec, samples)
}

fn write_wav(spec: hound::WavSpec, samples: &[i32]) -> Vec<u8> {
    let bytes_per_sample = usize::from(spec.bits_per_sample).div_ceil(8);
    let mut output = Cursor::new(Vec::with_capacity(44 + samples.len() * bytes_per_sample));
    {
        let mut writer = hound::WavWriter::new(&mut output, spec).expect("create output WAV");
        for &sample in samples {
            writer.write_sample(sample).expect("write output sample");
        }
        writer.finalize().expect("finalize output WAV");
    }
    output.into_inner()
}

fn historical_output(source: &[u8], start_ms: u64, end_ms: u64) -> Vec<u8> {
    let (spec, samples) = decode_all(source);
    let total_frames = samples.len() as u64 / u64::from(spec.channels);
    let range = frame_range(
        total_frames,
        spec.channels,
        spec.sample_rate,
        start_ms,
        end_ms,
    );
    write_wav(spec, &samples[range.start..range.end])
}

fn candidate_output(source: &[u8], start_ms: u64, end_ms: u64) -> Vec<u8> {
    let mut reader = hound::WavReader::new(Cursor::new(source)).expect("open source WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_format, hound::SampleFormat::Int);
    let total_frames = u64::from(reader.duration());
    let range = frame_range(
        total_frames,
        spec.channels,
        spec.sample_rate,
        start_ms,
        end_ms,
    );

    if total_frames > 0 {
        reader
            .seek(u32::try_from(total_frames - 1).expect("WAV duration fits u32"))
            .expect("seek to final frame");
        let mut tail = reader.samples::<i32>();
        for _ in 0..spec.channels {
            tail.next()
                .expect("declared final frame is present")
                .expect("decode final frame");
        }
    }

    let start_frame = range.start / usize::from(spec.channels);
    reader
        .seek(u32::try_from(start_frame).expect("WAV start frame fits u32"))
        .expect("seek to selected window");
    let selected_samples = range.end - range.start;
    let samples = reader
        .samples::<i32>()
        .take(selected_samples)
        .collect::<Result<Vec<_>, _>>()
        .expect("decode selected window");
    assert_eq!(
        samples.len(),
        selected_samples,
        "selected window is complete"
    );
    write_wav(spec, &samples)
}

fn open_and_bound(source: &[u8], start_ms: u64, end_ms: u64) -> FrameRange {
    let reader = hound::WavReader::new(Cursor::new(source)).expect("open source WAV");
    let spec = reader.spec();
    frame_range(
        u64::from(reader.duration()),
        spec.channels,
        spec.sample_rate,
        start_ms,
        end_ms,
    )
}

fn timed<T>(operation: impl FnOnce() -> T) -> (Duration, T) {
    let started = Instant::now();
    let output = operation();
    (started.elapsed(), output)
}

fn median_ns(values: &[u128]) -> u128 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn percentile(values: &[f64], percentile: usize) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf_29ce_4842_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

type SliceFn = fn(&[u8], u64, u64) -> Vec<u8>;

fn measure_arm(source: &[u8], implementation: SliceFn, inner_steps: usize) -> Duration {
    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..inner_steps {
        let output = implementation(
            black_box(source),
            black_box(SLICE_START_MS),
            black_box(SLICE_END_MS),
        );
        checksum ^= output.len();
        black_box(output);
    }
    black_box(checksum);
    started.elapsed()
}

struct PairedResults {
    ratios: Vec<f64>,
    numerator_ns: Vec<u128>,
    denominator_ns: Vec<u128>,
}

fn paired_ratios(
    source: &[u8],
    numerator: SliceFn,
    denominator: SliceFn,
    inner_steps: usize,
    repetitions: usize,
) -> PairedResults {
    let mut ratios = Vec::with_capacity(repetitions);
    let mut numerator_ns = Vec::with_capacity(repetitions);
    let mut denominator_ns = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let (numerator_elapsed, denominator_elapsed) = if repetition.is_multiple_of(2) {
            let numerator_before = measure_arm(source, numerator, inner_steps);
            let denominator_before = measure_arm(source, denominator, inner_steps);
            let denominator_after = measure_arm(source, denominator, inner_steps);
            let numerator_after = measure_arm(source, numerator, inner_steps);
            (
                numerator_before + numerator_after,
                denominator_before + denominator_after,
            )
        } else {
            let denominator_before = measure_arm(source, denominator, inner_steps);
            let numerator_before = measure_arm(source, numerator, inner_steps);
            let numerator_after = measure_arm(source, numerator, inner_steps);
            let denominator_after = measure_arm(source, denominator, inner_steps);
            (
                numerator_before + numerator_after,
                denominator_before + denominator_after,
            )
        };
        ratios.push(numerator_elapsed.as_secs_f64() / denominator_elapsed.as_secs_f64());
        numerator_ns.push(numerator_elapsed.as_nanos());
        denominator_ns.push(denominator_elapsed.as_nanos());
    }
    PairedResults {
        ratios,
        numerator_ns,
        denominator_ns,
    }
}

fn assert_valid_parity(source: &[u8]) {
    for &(start_ms, end_ms) in &[
        (0, 1_000),
        (SLICE_START_MS, SLICE_END_MS),
        (SLICE_END_MS, SLICE_START_MS),
        (u64::MAX, u64::MAX),
    ] {
        assert_eq!(
            candidate_output(source, start_ms, end_ms),
            historical_output(source, start_ms, end_ms),
            "WAV bytes differ for {start_ms}..{end_ms} ms"
        );
    }
}

fn profile(source: &[u8]) {
    let (spec, all_samples) = decode_all(source);
    let range = frame_range(
        all_samples.len() as u64 / u64::from(spec.channels),
        spec.channels,
        spec.sample_rate,
        SLICE_START_MS,
        SLICE_END_MS,
    );
    let selected = all_samples[range.start..range.end].to_vec();

    for _ in 0..2 {
        black_box(decode_all(black_box(source)));
        black_box(open_and_bound(
            black_box(source),
            SLICE_START_MS,
            SLICE_END_MS,
        ));
        black_box(write_wav(spec, black_box(&selected)));
        black_box(historical_output(
            black_box(source),
            SLICE_START_MS,
            SLICE_END_MS,
        ));
    }

    let mut decode_ns = Vec::with_capacity(PROFILE_REPS);
    let mut bound_ns = Vec::with_capacity(PROFILE_REPS);
    let mut write_ns = Vec::with_capacity(PROFILE_REPS);
    let mut historical_ns = Vec::with_capacity(PROFILE_REPS);
    for _ in 0..PROFILE_REPS {
        let (elapsed, decoded) = timed(|| decode_all(black_box(source)));
        black_box(decoded);
        decode_ns.push(elapsed.as_nanos());

        let (elapsed, bounds) =
            timed(|| open_and_bound(black_box(source), SLICE_START_MS, SLICE_END_MS));
        black_box(bounds);
        bound_ns.push(elapsed.as_nanos());

        let (elapsed, output) = timed(|| write_wav(spec, black_box(&selected)));
        black_box(output);
        write_ns.push(elapsed.as_nanos());

        let (elapsed, output) =
            timed(|| historical_output(black_box(source), SLICE_START_MS, SLICE_END_MS));
        black_box(output);
        historical_ns.push(elapsed.as_nanos());
    }

    let decode_median = median_ns(&decode_ns);
    let bound_median = median_ns(&bound_ns);
    let write_median = median_ns(&write_ns);
    let historical_median = median_ns(&historical_ns);
    println!("PROFILE_DECODE_ALL_NS={decode_ns:?}");
    println!("PROFILE_HEADER_BOUNDS_NS={bound_ns:?}");
    println!("PROFILE_SELECTED_WRITE_NS={write_ns:?}");
    println!("PROFILE_HISTORICAL_FULL_NS={historical_ns:?}");
    println!(
        "PROFILE_MEDIANS decode_all_ns={decode_median} header_bounds_ns={bound_median} selected_write_ns={write_median} historical_full_ns={historical_median} decode_share={:.6}",
        decode_median as f64 / historical_median as f64
    );
}

fn ab(source: &[u8]) {
    assert_valid_parity(source);
    let reference = historical_output(source, SLICE_START_MS, SLICE_END_MS);
    println!("SLICE_OUTPUT_FNV64={:016x}", fnv64(&reference));
    let executable = std::fs::read(std::env::current_exe().expect("benchmark path"))
        .expect("read benchmark binary");
    println!("SLICE_BINARY_FNV64={:016x}", fnv64(&executable));

    let calibration = measure_arm(source, historical_output, 1);
    let inner_steps = (TARGET_ARM_SECS / calibration.as_secs_f64()).ceil() as usize;
    let inner_steps = inner_steps.clamp(1, 8);

    black_box(paired_ratios(
        source,
        historical_output,
        historical_output,
        inner_steps,
        3,
    ));
    let null = paired_ratios(
        source,
        historical_output,
        historical_output,
        inner_steps,
        PAIRED_REPS,
    );
    black_box(paired_ratios(
        source,
        historical_output,
        candidate_output,
        inner_steps,
        3,
    ));
    let candidate = paired_ratios(
        source,
        historical_output,
        candidate_output,
        inner_steps,
        PAIRED_REPS,
    );

    let null_p10 = percentile(&null.ratios, 10);
    let null_median = percentile(&null.ratios, 50);
    let null_p90 = percentile(&null.ratios, 90);
    let null_cv = coefficient_of_variation(&null.ratios);
    let candidate_p10 = percentile(&candidate.ratios, 10);
    let candidate_median = percentile(&candidate.ratios, 50);
    let candidate_p90 = percentile(&candidate.ratios, 90);
    let candidate_cv = coefficient_of_variation(&candidate.ratios);
    let wins = candidate
        .ratios
        .iter()
        .filter(|&&ratio| ratio > 1.0)
        .count();
    let null_valid = (0.98..=1.02).contains(&null_median) && null_cv <= 0.03;
    let keep = null_valid && candidate_median >= 1.10 && candidate_p10 > null_p90 && wins >= 19;

    println!(
        "SLICE_CALIBRATION historical_ns={} inner_steps={} target_arm_ms={:.1}",
        calibration.as_nanos(),
        inner_steps,
        TARGET_ARM_SECS * 1_000.0
    );
    println!("BASE_BASE_RATIOS={:?}", null.ratios);
    println!("HISTORICAL_CANDIDATE_RATIOS={:?}", candidate.ratios);
    println!(
        "ABSOLUTE_MEDIANS historical_ns={} candidate_ns={}",
        median_ns(&candidate.numerator_ns),
        median_ns(&candidate.denominator_ns)
    );
    println!(
        "NULL p10={null_p10:.6} median={null_median:.6} p90={null_p90:.6} cv={null_cv:.6} valid={null_valid}"
    );
    println!(
        "CANDIDATE p10={candidate_p10:.6} median={candidate_median:.6} p90={candidate_p90:.6} cv={candidate_cv:.6} wins={wins}/{PAIRED_REPS} keep={keep}"
    );
}

fn main() -> std::process::ExitCode {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "profile".into());
    let source = fixture();
    println!(
        "SLICE_FIXTURE bytes={} frames={} slice_ms={}..{} profile=release lto=false",
        source.len(),
        SAMPLE_RATE as usize * SOURCE_SECONDS as usize,
        SLICE_START_MS,
        SLICE_END_MS
    );
    match mode.as_str() {
        "profile" => profile(&source),
        "ab" => ab(&source),
        other => {
            eprintln!("unknown mode {other:?}; expected profile or ab");
            return std::process::ExitCode::FAILURE;
        }
    }
    std::process::ExitCode::SUCCESS
}
