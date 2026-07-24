//! Criterion benches for low-bandwidth TTY audio paths.
//!
//! Covers:
//! - frame encode throughput (`encode_to_writer`)
//! - frame decode throughput (`decode_frames_to_raw_with_policy`)
//! - control-frame serialization throughput (`emit_control_frame_to_writer`)

use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::STANDARD_NO_PAD;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use flate2::Compression;
use flate2::write::ZlibEncoder;
use sha2::{Digest, Sha256};

use franken_whisper::tty_audio::{
    DecodeRecoveryPolicy, TtyAudioFrame, TtyControlFrame, decode_frames_to_raw_with_policy,
    emit_control_frame_to_writer, encode_to_writer, mic_stream_event_value, write_audio_frame_line,
    write_mic_stream_event_line,
};

const CHUNK_SIZES_MS: [u32; 2] = [20, 60];
const CODEC_MULAW_ZLIB_B64: &str = "mulaw+zlib+b64";

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn fixture_wav_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("audio")
        .join("test_1s_tone.wav")
}

fn encoded_fixture(chunk_ms: u32) -> Vec<u8> {
    let input = fixture_wav_path();
    let mut output = Vec::new();
    encode_to_writer(&input, chunk_ms, &mut output).expect("tty encode fixture should succeed");
    output
}

fn synthetic_frame(seq: u64, data: &[u8]) -> TtyAudioFrame {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder
        .write_all(data)
        .expect("synthetic frame compression should succeed");
    let compressed = encoder
        .finish()
        .expect("synthetic frame compression should finish");

    let mut crc = crc32fast::Hasher::new();
    crc.update(data);
    let digest = Sha256::digest(data);

    TtyAudioFrame {
        protocol_version: 1,
        seq,
        codec: CODEC_MULAW_ZLIB_B64.to_owned(),
        sample_rate_hz: 8_000,
        channels: 1,
        payload_b64: STANDARD_NO_PAD.encode(compressed),
        crc32: Some(crc.finalize()),
        payload_sha256: Some(format!("{digest:x}")),
    }
}

fn synthetic_ndjson(frame_count: usize, bytes_per_frame: usize) -> String {
    let mut out = String::new();
    let handshake = TtyControlFrame::Handshake {
        min_version: 1,
        max_version: 1,
        supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned()],
    };
    out.push_str(&serde_json::to_string(&handshake).expect("handshake serialization"));
    out.push('\n');

    for seq in 0..frame_count {
        let data: Vec<u8> = (0..bytes_per_frame)
            .map(|i| (seq as u8).wrapping_mul(17).wrapping_add(i as u8))
            .collect();
        let frame = synthetic_frame(seq as u64, &data);
        out.push_str(&serde_json::to_string(&frame).expect("frame serialization"));
        out.push('\n');
    }

    out
}

fn bench_tty_encode(c: &mut Criterion) {
    if !ffmpeg_available() {
        return;
    }

    let input = fixture_wav_path();
    let input_size = std::fs::metadata(&input).map(|m| m.len()).unwrap_or(0);
    let mut group = c.benchmark_group("tty/encode");
    group.throughput(Throughput::Bytes(input_size));

    for chunk_ms in CHUNK_SIZES_MS {
        group.bench_with_input(
            BenchmarkId::new("chunk_ms", chunk_ms),
            &chunk_ms,
            |b, &chunk| {
                b.iter(|| {
                    let mut output = Vec::new();
                    encode_to_writer(&input, chunk, &mut output)
                        .expect("tty encode should succeed");
                    output.len()
                });
            },
        );
    }

    group.finish();
}

fn bench_tty_decode(c: &mut Criterion) {
    if !ffmpeg_available() {
        return;
    }

    let mut group = c.benchmark_group("tty/decode");

    for chunk_ms in CHUNK_SIZES_MS {
        let encoded = encoded_fixture(chunk_ms);
        group.throughput(Throughput::Bytes(encoded.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("chunk_ms", chunk_ms),
            &encoded,
            |b, payload| {
                b.iter(|| {
                    let mut reader = Cursor::new(payload.as_slice());
                    let (report, raw) = decode_frames_to_raw_with_policy(
                        &mut reader,
                        DecodeRecoveryPolicy::FailClosed,
                    )
                    .expect("tty decode should succeed");
                    assert!(report.frames_decoded > 0);
                    raw.len()
                });
            },
        );
    }

    group.finish();
}

fn bench_tty_decode_synthetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("tty/decode_synthetic");

    for frame_count in [32usize, 128] {
        let payload = synthetic_ndjson(frame_count, 24);
        group.throughput(Throughput::Bytes(payload.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("frames", frame_count),
            &payload,
            |b, payload| {
                b.iter(|| {
                    let mut reader = Cursor::new(payload.as_bytes());
                    let (report, raw) = decode_frames_to_raw_with_policy(
                        &mut reader,
                        DecodeRecoveryPolicy::FailClosed,
                    )
                    .expect("synthetic tty decode should succeed");
                    assert_eq!(report.frames_decoded, frame_count as u64);
                    raw.len()
                });
            },
        );
    }

    group.finish();
}

#[derive(Debug)]
struct PairedRatioStats {
    median: f64,
    p10: f64,
    p90: f64,
    min: f64,
    max: f64,
    cv_pct: f64,
    wins: usize,
}

fn paired_ratio_stats(ratios: &[f64]) -> PairedRatioStats {
    assert!(!ratios.is_empty());
    let mut sorted = ratios.to_vec();
    sorted.sort_by(f64::total_cmp);
    let nearest_rank = |percent: usize| {
        let index = (sorted.len() * percent).div_ceil(100).saturating_sub(1);
        sorted[index.min(sorted.len() - 1)]
    };
    let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let variance = ratios
        .iter()
        .map(|ratio| (ratio - mean).powi(2))
        .sum::<f64>()
        / (ratios.len() - 1) as f64;
    PairedRatioStats {
        median: sorted[sorted.len() / 2],
        p10: nearest_rank(10),
        p90: nearest_rank(90),
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        cv_pct: variance.sqrt() / mean * 100.0,
        wins: ratios.iter().filter(|&&ratio| ratio > 1.0).count(),
    }
}

fn format_ratios(ratios: &[f64]) -> String {
    ratios
        .iter()
        .map(|ratio| format!("{ratio:.6}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn eager_tty_ab_selected(benchmark_name: &str, env_name: &str) -> bool {
    if std::env::var_os(env_name).is_some_and(|value| value == "1") {
        return true;
    }
    std::env::args()
        .find(|argument| argument.contains("tty/"))
        .is_none_or(|filter| filter.contains(benchmark_name) || benchmark_name.contains(&filter))
}

fn paired_ratios<F, S>(mut measure_first: F, mut measure_second: S, repetitions: usize) -> Vec<f64>
where
    F: FnMut() -> Duration,
    S: FnMut() -> Duration,
{
    let mut ratios = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        let (first_elapsed, second_elapsed) = if repetition % 2 == 0 {
            let first_before = measure_first();
            let second_before = measure_second();
            let second_after = measure_second();
            let first_after = measure_first();
            (first_before + first_after, second_before + second_after)
        } else {
            let second_before = measure_second();
            let first_before = measure_first();
            let first_after = measure_first();
            let second_after = measure_second();
            (first_before + first_after, second_before + second_after)
        };
        ratios.push(first_elapsed.as_secs_f64() / second_elapsed.as_secs_f64());
    }
    ratios
}

fn measure_mic_event_arm<const BORROWED_BUFFERED: bool>(
    frames: &[TtyAudioFrame],
    inner_steps: usize,
) -> Duration {
    let mut output = Vec::new();
    let mut line_buffer = Vec::new();
    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..inner_steps {
        output.clear();
        for frame in frames {
            if BORROWED_BUFFERED {
                write_mic_stream_event_line(&mut output, frame, &mut line_buffer)
                    .expect("borrowed mic event emit");
            } else {
                let event = mic_stream_event_value(frame);
                writeln!(
                    output,
                    "{}",
                    serde_json::to_string(&event).expect("owned mic event serialization")
                )
                .expect("owned mic event emit");
            }
        }
        checksum ^= output.len();
        black_box(output.as_slice());
    }
    black_box(checksum);
    started.elapsed()
}

fn paired_mic_event_ratios<const FIRST_BORROWED: bool, const SECOND_BORROWED: bool>(
    first_frames: &[TtyAudioFrame],
    second_frames: &[TtyAudioFrame],
    inner_steps: usize,
    repetitions: usize,
) -> Vec<f64> {
    paired_ratios(
        || measure_mic_event_arm::<FIRST_BORROWED>(first_frames, inner_steps),
        || measure_mic_event_arm::<SECOND_BORROWED>(second_frames, inner_steps),
        repetitions,
    )
}

fn bench_tty_mic_event_emit_ab(c: &mut Criterion) {
    if !eager_tty_ab_selected("tty/mic_event_emit_ab", "FW_BENCH_TTY_MIC_EVENT_AB") {
        return;
    }
    const FRAME_COUNT: usize = 128;
    const BYTES_PER_FRAME: usize = 1_600;
    const INNER_STEPS: usize = 32;
    const WARMUP_REPS: usize = 3;
    const PAIRED_REPS: usize = 31;
    const NULL_MEDIAN_MIN: f64 = 0.98;
    const NULL_MEDIAN_MAX: f64 = 1.02;

    let frames: Vec<TtyAudioFrame> = (0..FRAME_COUNT)
        .map(|seq| {
            let data: Vec<u8> = (0..BYTES_PER_FRAME)
                .map(|i| (seq as u8).wrapping_mul(17).wrapping_add(i as u8))
                .collect();
            synthetic_frame(seq as u64, &data)
        })
        .collect();
    let null_frames = frames.clone();
    let candidate_frames = frames.clone();

    let mut baseline_output = Vec::new();
    for frame in &frames {
        let event = mic_stream_event_value(frame);
        writeln!(
            baseline_output,
            "{}",
            serde_json::to_string(&event).expect("owned mic event serialization")
        )
        .expect("owned mic event emit");
    }
    let mut candidate_output = Vec::new();
    let mut line_buffer = Vec::new();
    for frame in &candidate_frames {
        write_mic_stream_event_line(&mut candidate_output, frame, &mut line_buffer)
            .expect("borrowed mic event emit");
    }
    assert_eq!(
        candidate_output, baseline_output,
        "mic event A/B byte parity"
    );

    black_box(paired_mic_event_ratios::<false, false>(
        &frames,
        &null_frames,
        INNER_STEPS,
        WARMUP_REPS,
    ));
    let null_ratios =
        paired_mic_event_ratios::<false, false>(&frames, &null_frames, INNER_STEPS, PAIRED_REPS);
    let null_stats = paired_ratio_stats(&null_ratios);

    black_box(paired_mic_event_ratios::<false, true>(
        &frames,
        &candidate_frames,
        INNER_STEPS,
        WARMUP_REPS,
    ));
    let candidate_ratios = paired_mic_event_ratios::<false, true>(
        &frames,
        &candidate_frames,
        INNER_STEPS,
        PAIRED_REPS,
    );
    let candidate_stats = paired_ratio_stats(&candidate_ratios);
    let null_valid = (NULL_MEDIAN_MIN..=NULL_MEDIAN_MAX).contains(&null_stats.median);
    let decision_eligible = null_valid
        && (candidate_stats.median < null_stats.p10 || candidate_stats.median > null_stats.p90);

    let executable = std::env::current_exe().expect("benchmark executable path");
    let executable_bytes = std::fs::read(&executable).expect("read benchmark executable");
    let binary_sha256 = format!("{:x}", Sha256::digest(&executable_bytes));
    let output_sha256 = format!("{:x}", Sha256::digest(&candidate_output));
    eprintln!(
        "TTY_MIC_EVENT_BINARY sha256={binary_sha256} output_sha256={output_sha256} path={}",
        executable.display()
    );
    eprintln!(
        "TTY_MIC_EVENT_NULL ratios=[{}] median={:.6} p10={:.6} p90={:.6} min={:.6} \
         max={:.6} cv_pct={:.3} wins={}/{} acceptance=[{NULL_MEDIAN_MIN:.2},{NULL_MEDIAN_MAX:.2}]",
        format_ratios(&null_ratios),
        null_stats.median,
        null_stats.p10,
        null_stats.p90,
        null_stats.min,
        null_stats.max,
        null_stats.cv_pct,
        null_stats.wins,
        PAIRED_REPS,
    );
    eprintln!(
        "TTY_MIC_EVENT_AB ratios=[{}] median={:.6} p10={:.6} p90={:.6} min={:.6} \
         max={:.6} cv_pct={:.3} wins={}/{} null_valid={} decision_eligible={} frames={} output_bytes={}",
        format_ratios(&candidate_ratios),
        candidate_stats.median,
        candidate_stats.p10,
        candidate_stats.p90,
        candidate_stats.min,
        candidate_stats.max,
        candidate_stats.cv_pct,
        candidate_stats.wins,
        PAIRED_REPS,
        null_valid,
        decision_eligible,
        FRAME_COUNT,
        candidate_output.len(),
    );

    c.bench_function("tty/mic_event_emit_ab/frames/128", |b| {
        b.iter(|| {
            let mut output = Vec::new();
            let mut scratch = Vec::new();
            for frame in black_box(&candidate_frames) {
                write_mic_stream_event_line(&mut output, frame, &mut scratch)
                    .expect("borrowed mic event emit");
            }
            black_box(output)
        });
    });
}

fn measure_audio_frame_arm<const BUFFERED: bool>(
    frames: &[TtyAudioFrame],
    inner_steps: usize,
    output_capacity: usize,
) -> Duration {
    let mut output = Vec::with_capacity(output_capacity);
    let mut line_buffer = Vec::new();
    let started = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..inner_steps {
        output.clear();
        for frame in frames {
            if BUFFERED {
                write_audio_frame_line(&mut output, frame, &mut line_buffer)
                    .expect("buffered audio frame emit");
            } else {
                writeln!(
                    output,
                    "{}",
                    serde_json::to_string(frame).expect("owned audio frame serialization")
                )
                .expect("owned audio frame emit");
            }
        }
        checksum ^= output.len();
        black_box(output.as_slice());
    }
    black_box(checksum);
    started.elapsed()
}

fn paired_audio_frame_ratios<const FIRST_BUFFERED: bool, const SECOND_BUFFERED: bool>(
    first_frames: &[TtyAudioFrame],
    second_frames: &[TtyAudioFrame],
    inner_steps: usize,
    output_capacity: usize,
    repetitions: usize,
) -> Vec<f64> {
    paired_ratios(
        || measure_audio_frame_arm::<FIRST_BUFFERED>(first_frames, inner_steps, output_capacity),
        || measure_audio_frame_arm::<SECOND_BUFFERED>(second_frames, inner_steps, output_capacity),
        repetitions,
    )
}

fn bench_tty_audio_frame_emit_ab(c: &mut Criterion) {
    if !eager_tty_ab_selected("tty/audio_frame_emit_ab", "FW_BENCH_TTY_AUDIO_FRAME_AB") {
        return;
    }

    const FRAME_COUNT: usize = 128;
    const BYTES_PER_FRAME: usize = 1_600;
    const WARMUP_REPS: usize = 3;
    const PAIRED_REPS: usize = 21;
    const NULL_MEDIAN_MIN: f64 = 0.98;
    const NULL_MEDIAN_MAX: f64 = 1.02;
    const REQUIRED_WINS: usize = 17;
    const TARGET_ARM_SECS: f64 = 0.100;

    let frames: Vec<TtyAudioFrame> = (0..FRAME_COUNT)
        .map(|seq| {
            let data: Vec<u8> = (0..BYTES_PER_FRAME)
                .map(|i| (seq as u8).wrapping_mul(17).wrapping_add(i as u8))
                .collect();
            synthetic_frame(seq as u64, &data)
        })
        .collect();
    let null_frames = frames.clone();
    let candidate_frames = frames.clone();

    let mut baseline_output = Vec::new();
    for frame in &frames {
        writeln!(
            baseline_output,
            "{}",
            serde_json::to_string(frame).expect("owned audio frame serialization")
        )
        .expect("owned audio frame emit");
    }
    let mut candidate_output = Vec::new();
    let mut line_buffer = Vec::new();
    for frame in &candidate_frames {
        write_audio_frame_line(&mut candidate_output, frame, &mut line_buffer)
            .expect("buffered audio frame emit");
    }
    assert_eq!(
        candidate_output, baseline_output,
        "audio frame A/B byte parity"
    );

    let output_capacity = baseline_output.len();
    let calibration = measure_audio_frame_arm::<false>(&frames, 1, output_capacity);
    let inner_steps = (TARGET_ARM_SECS / calibration.as_secs_f64()).ceil() as usize;
    let inner_steps = inner_steps.clamp(64, 4_096);

    black_box(paired_audio_frame_ratios::<false, false>(
        &frames,
        &null_frames,
        inner_steps,
        output_capacity,
        WARMUP_REPS,
    ));
    let null_ratios = paired_audio_frame_ratios::<false, false>(
        &frames,
        &null_frames,
        inner_steps,
        output_capacity,
        PAIRED_REPS,
    );
    let null_stats = paired_ratio_stats(&null_ratios);

    black_box(paired_audio_frame_ratios::<false, true>(
        &frames,
        &candidate_frames,
        inner_steps,
        output_capacity,
        WARMUP_REPS,
    ));
    let candidate_ratios = paired_audio_frame_ratios::<false, true>(
        &frames,
        &candidate_frames,
        inner_steps,
        output_capacity,
        PAIRED_REPS,
    );
    let candidate_stats = paired_ratio_stats(&candidate_ratios);
    let null_valid = (NULL_MEDIAN_MIN..=NULL_MEDIAN_MAX).contains(&null_stats.median);
    let decision_eligible = null_valid
        && (candidate_stats.median < null_stats.p10 || candidate_stats.median > null_stats.p90);
    let keep_eligible =
        decision_eligible && candidate_stats.median > 1.0 && candidate_stats.wins >= REQUIRED_WINS;

    let executable = std::env::current_exe().expect("benchmark executable path");
    let executable_bytes = std::fs::read(&executable).expect("read benchmark executable");
    let binary_sha256 = format!("{:x}", Sha256::digest(&executable_bytes));
    let output_sha256 = format!("{:x}", Sha256::digest(&candidate_output));
    eprintln!(
        "TTY_AUDIO_FRAME_BINARY sha256={binary_sha256} output_sha256={output_sha256} path={}",
        executable.display()
    );
    eprintln!(
        "TTY_AUDIO_FRAME_CALIBRATION one_pass_us={:.3} inner_steps={} target_arm_ms={:.1}",
        calibration.as_secs_f64() * 1_000_000.0,
        inner_steps,
        TARGET_ARM_SECS * 1_000.0,
    );
    eprintln!(
        "TTY_AUDIO_FRAME_NULL ratios=[{}] median={:.6} p10={:.6} p90={:.6} min={:.6} \
         max={:.6} cv_pct={:.3} wins={}/{} acceptance=[{NULL_MEDIAN_MIN:.2},{NULL_MEDIAN_MAX:.2}]",
        format_ratios(&null_ratios),
        null_stats.median,
        null_stats.p10,
        null_stats.p90,
        null_stats.min,
        null_stats.max,
        null_stats.cv_pct,
        null_stats.wins,
        PAIRED_REPS,
    );
    eprintln!(
        "TTY_AUDIO_FRAME_AB ratios=[{}] median={:.6} p10={:.6} p90={:.6} min={:.6} \
         max={:.6} cv_pct={:.3} wins={}/{} null_valid={} decision_eligible={} keep_eligible={} required_wins={} frames={} output_bytes={}",
        format_ratios(&candidate_ratios),
        candidate_stats.median,
        candidate_stats.p10,
        candidate_stats.p90,
        candidate_stats.min,
        candidate_stats.max,
        candidate_stats.cv_pct,
        candidate_stats.wins,
        PAIRED_REPS,
        null_valid,
        decision_eligible,
        keep_eligible,
        REQUIRED_WINS,
        FRAME_COUNT,
        candidate_output.len(),
    );

    c.bench_function("tty/audio_frame_emit_ab/frames/128", |b| {
        b.iter(|| {
            let mut output = Vec::new();
            let mut scratch = Vec::new();
            for frame in black_box(&candidate_frames) {
                write_audio_frame_line(&mut output, frame, &mut scratch)
                    .expect("buffered audio frame emit");
            }
            black_box(output)
        });
    });
}

fn bench_tty_control_emit(c: &mut Criterion) {
    let mut group = c.benchmark_group("tty/control_emit");

    let fixtures: [(&str, TtyControlFrame); 6] = [
        (
            "handshake",
            TtyControlFrame::Handshake {
                min_version: 1,
                max_version: 1,
                supported_codecs: vec!["mulaw+zlib+b64".to_owned()],
            },
        ),
        (
            "handshake_ack",
            TtyControlFrame::HandshakeAck {
                negotiated_version: 1,
                negotiated_codec: "mulaw+zlib+b64".to_owned(),
            },
        ),
        ("ack", TtyControlFrame::Ack { up_to_seq: 128 }),
        (
            "backpressure",
            TtyControlFrame::Backpressure {
                remaining_capacity: 32,
            },
        ),
        (
            "retransmit_request",
            TtyControlFrame::RetransmitRequest {
                sequences: vec![4, 5, 6, 99],
            },
        ),
        (
            "retransmit_response",
            TtyControlFrame::RetransmitResponse {
                sequences: vec![4, 5, 6, 99],
            },
        ),
    ];

    for (name, frame) in fixtures {
        group.bench_with_input(BenchmarkId::new("frame", name), &frame, |b, control| {
            b.iter(|| {
                let mut out = Vec::new();
                emit_control_frame_to_writer(&mut out, control)
                    .expect("control frame emit should succeed");
                out.len()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tty_encode,
    bench_tty_decode,
    bench_tty_decode_synthetic,
    bench_tty_mic_event_emit_ab,
    bench_tty_audio_frame_emit_ab,
    bench_tty_control_emit
);
criterion_main!(benches);
