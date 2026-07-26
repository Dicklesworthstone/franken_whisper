//! End-to-end native-engine probe: drive `transcribe_samples` directly on an
//! arbitrary audio/model/repeat-count to measure realistic wall-clock + RTF,
//! including long-form (tiled audio) and word-timestamp paths.
//!
//! Usage: e2e_probe <model_short> <wav> <repeat> [wordts]
//!   model_short: tiny.en | large-v3-turbo
//!   repeat:      tile the audio N times to synthesize long-form
//!   wordts:      pass the literal "wordts" to enable DTW word timestamps

use franken_whisper::native_engine::decode::{
    DecodeOutput, DecodeParams, LoadedModel, transcribe_samples,
};
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use sha2::{Digest, Sha256};
use std::hint::black_box;
use std::time::Instant;

const CONTEXT_AB_DEFAULT_PAIRS: usize = 21;
const CONTEXT_AB_BOOTSTRAP_RESAMPLES: usize = 20_000;

fn executable_identity() -> String {
    let Ok(path) = std::env::current_exe() else {
        return "unavailable".to_owned();
    };
    let Ok(bytes) = std::fs::read(&path) else {
        return "unavailable".to_owned();
    };
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    format!(
        "{:x} ({} bytes) {}",
        hasher.finalize(),
        bytes.len(),
        path.display()
    )
}

fn segment_oracle(output: &DecodeOutput) -> Vec<u8> {
    fn append_optional_f64(bytes: &mut Vec<u8>, value: Option<f64>) {
        match value {
            Some(value) => {
                bytes.push(1);
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            None => bytes.push(0),
        }
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(output.segments.len() as u64).to_le_bytes());
    for segment in &output.segments {
        append_optional_f64(&mut bytes, segment.start_sec);
        append_optional_f64(&mut bytes, segment.end_sec);
        bytes.extend_from_slice(&(segment.text.len() as u64).to_le_bytes());
        bytes.extend_from_slice(segment.text.as_bytes());
    }
    bytes
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
    let mut state = 0x510e_527f_ade6_82d1_u64 ^ values.len() as u64;
    let mut sample = Vec::with_capacity(values.len());
    let mut medians = Vec::with_capacity(CONTEXT_AB_BOOTSTRAP_RESAMPLES);
    for _ in 0..CONTEXT_AB_BOOTSTRAP_RESAMPLES {
        sample.clear();
        for _ in 0..values.len() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            sample.push(values[state as usize % values.len()]);
        }
        sample.sort_by(f64::total_cmp);
        medians.push(sample[sample.len() / 2]);
    }
    medians.sort_by(f64::total_cmp);
    (
        medians[CONTEXT_AB_BOOTSTRAP_RESAMPLES * 25 / 1_000],
        medians[CONTEXT_AB_BOOTSTRAP_RESAMPLES * 975 / 1_000],
    )
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

fn timed_transcribe(model: &LoadedModel, samples: &[f32], params: &DecodeParams) -> f64 {
    let started = Instant::now();
    let output =
        transcribe_samples(model, samples, params, &(|| Ok(()))).expect("timed transcribe");
    let elapsed_ns = started.elapsed().as_nanos() as f64;
    black_box(output.segments.len());
    black_box(
        output
            .segments
            .iter()
            .map(|segment| segment.text.len())
            .sum::<usize>(),
    );
    elapsed_ns
}

fn paired_context_times(
    model: &LoadedModel,
    samples: &[f32],
    left: &DecodeParams,
    right: &DecodeParams,
    pairs: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut left_ns = Vec::with_capacity(pairs);
    let mut right_ns = Vec::with_capacity(pairs);
    let mut ratios = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let (left_elapsed, right_elapsed) = if pair.is_multiple_of(2) {
            (
                timed_transcribe(model, samples, left),
                timed_transcribe(model, samples, right),
            )
        } else {
            let right_elapsed = timed_transcribe(model, samples, right);
            let left_elapsed = timed_transcribe(model, samples, left);
            (left_elapsed, right_elapsed)
        };
        left_ns.push(left_elapsed);
        right_ns.push(right_elapsed);
        ratios.push(left_elapsed / right_elapsed.max(f64::MIN_POSITIVE));
    }
    (left_ns, right_ns, ratios)
}

fn run_tiny_en_context_ab(
    model: &LoadedModel,
    samples: &[f32],
    params: &DecodeParams,
    pairs: usize,
) {
    assert!(
        pairs >= 3 && !pairs.is_multiple_of(2),
        "PROBE_CONTEXT_AB_PAIRS must be an odd integer >= 3"
    );
    for name in [
        "FW_NO_CONTEXT",
        "FW_TINY_EN_TS_CONTEXT",
        "FW_RETRY_FAILED_WINDOW",
        "PROBE_NO_TS",
    ] {
        assert!(
            std::env::var_os(name).is_none(),
            "{name} must be unset for the context A/B contract"
        );
    }
    assert!(
        params.timestamps && !params.word_timestamps,
        "context A/B is defined for segment timestamps only"
    );

    let mut historical_params = params.clone();
    historical_params.max_context = Some(-1);
    let candidate_params = params.clone();

    let historical = transcribe_samples(model, samples, &historical_params, &(|| Ok(())))
        .expect("historical parity transcribe");
    let candidate = transcribe_samples(model, samples, &candidate_params, &(|| Ok(())))
        .expect("candidate parity transcribe");
    let historical_oracle = segment_oracle(&historical);
    let candidate_oracle = segment_oracle(&candidate);
    let segments_exact = historical_oracle == candidate_oracle;
    let historical_sha = Sha256::digest(&historical_oracle);
    let candidate_sha = Sha256::digest(&candidate_oracle);
    let historical_chars = historical
        .segments
        .iter()
        .map(|segment| segment.text.len())
        .sum::<usize>();
    let candidate_chars = candidate
        .segments
        .iter()
        .map(|segment| segment.text.len())
        .sum::<usize>();
    println!(
        "CONTEXT_AB_PARITY segments_exact={segments_exact} historical_segments={} candidate_segments={} historical_chars={historical_chars} candidate_chars={candidate_chars} historical_sha256={historical_sha:x} candidate_sha256={candidate_sha:x}",
        historical.segments.len(),
        candidate.segments.len()
    );
    if std::env::var("PROBE_DUMP_TEXT").as_deref() == Ok("1") {
        let historical_text = historical
            .segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<String>();
        let candidate_text = candidate
            .segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<String>();
        eprintln!("HISTORICAL_TRANSCRIPT>>>{historical_text}<<<");
        eprintln!("CANDIDATE_TRANSCRIPT>>>{candidate_text}<<<");
    }
    assert!(
        segments_exact,
        "tiny.en context candidate changed the segment oracle; timing is inadmissible"
    );
    drop(historical);
    drop(candidate);

    black_box(timed_transcribe(model, samples, &historical_params));
    black_box(timed_transcribe(model, samples, &candidate_params));

    let (null_left_ns, null_right_ns, null_ratios) = paired_context_times(
        model,
        samples,
        &historical_params,
        &historical_params,
        pairs,
    );
    let (historical_ns, candidate_ns, candidate_ratios) =
        paired_context_times(model, samples, &historical_params, &candidate_params, pairs);
    let (null_ci_low, null_ci_high) = bootstrap_median_ci(&null_ratios);
    let (candidate_ci_low, candidate_ci_high) = bootstrap_median_ci(&candidate_ratios);
    let null_half_width = (1.0 - null_ci_low).abs().max((null_ci_high - 1.0).abs());
    let required_speedup = 1.0 + 2.0 * null_half_width;
    let candidate_median = median(&candidate_ratios);
    let verdict = if candidate_median >= required_speedup {
        "KEEP"
    } else {
        "REJECT"
    };

    println!(
        "CONTEXT_AB_NULL ratios={null_ratios:?} left_median_ms={:.6} right_median_ms={:.6} median={:.6} median_ci95=[{null_ci_low:.6},{null_ci_high:.6}] cv={:.6}",
        median(&null_left_ns) / 1e6,
        median(&null_right_ns) / 1e6,
        median(&null_ratios),
        coefficient_of_variation(&null_ratios)
    );
    println!(
        "CONTEXT_AB_CANDIDATE ratios={candidate_ratios:?} historical_median_ms={:.6} candidate_median_ms={:.6} median={candidate_median:.6} median_ci95=[{candidate_ci_low:.6},{candidate_ci_high:.6}] cv={:.6} wins={}/{}",
        median(&historical_ns) / 1e6,
        median(&candidate_ns) / 1e6,
        coefficient_of_variation(&candidate_ratios),
        candidate_ratios
            .iter()
            .filter(|ratio| **ratio > 1.0)
            .count(),
        candidate_ratios.len()
    );
    println!(
        "CONTEXT_AB_GATE method=median_vs_null_ci95_2x_margin null_half_width={null_half_width:.6} required_speedup={required_speedup:.6} candidate_median={candidate_median:.6} cv_is_provenance_only=true verdict={verdict}"
    );
}

/// Minimal robust WAV reader: locate the `data` chunk, parse PCM16 mono/stereo,
/// downmix to mono, return f32 in [-1,1]. Assumes 16 kHz (whisper standard).
fn read_wav_mono16k(path: &str) -> (Vec<f32>, u32, u16) {
    let bytes = std::fs::read(path).expect("read wav");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut pos = 12;
    let mut channels = 1u16;
    let mut rate = 16000u32;
    let mut bits = 16u16;
    let mut data: &[u8] = &[];
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
            rate = u32::from_le_bytes([
                bytes[body + 4],
                bytes[body + 5],
                bytes[body + 6],
                bytes[body + 7],
            ]);
            bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
        } else if id == b"data" {
            data = &bytes[body..(body + sz).min(bytes.len())];
        }
        pos = body + sz + (sz & 1);
    }
    assert_eq!(bits, 16, "expected PCM16");
    let n = data.len() / 2;
    let mut samples = Vec::with_capacity(n / channels as usize);
    let mut i = 0;
    while i + 2 * channels as usize <= data.len() {
        let mut acc = 0i32;
        for c in 0..channels as usize {
            let s = i16::from_le_bytes([data[i + 2 * c], data[i + 2 * c + 1]]);
            acc += s as i32;
        }
        samples.push((acc as f32 / channels as f32) / 32768.0);
        i += 2 * channels as usize;
    }
    (samples, rate, channels)
}

fn main() {
    println!("probe_elf_sha256={}", executable_identity());

    let args: Vec<String> = std::env::args().collect();
    let model_short = args.get(1).map(String::as_str).unwrap_or("tiny.en");
    let wav = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "tests/fixtures/native/jfk.wav".to_string());
    let repeat: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let wordts = args.get(4).map(|s| s == "wordts").unwrap_or(false);

    let path = find_model_file(model_short)
        .unwrap_or_else(|| panic!("model {model_short} not found in search dirs"));
    let t_load = Instant::now();
    let model = GgmlModel::load(&path)
        .and_then(LoadedModel::from_ggml)
        .expect("load model");
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let (base, rate, ch) = read_wav_mono16k(&wav);
    assert_eq!(rate, 16000, "probe assumes 16kHz input");
    let mut samples = Vec::with_capacity(base.len() * repeat);
    for _ in 0..repeat {
        samples.extend_from_slice(&base);
    }
    let audio_sec = samples.len() as f64 / 16000.0;

    let params = DecodeParams {
        language: Some("en".to_string()),
        translate: false,
        timestamps: std::env::var("PROBE_NO_TS").as_deref() != Ok("1"),
        n_threads: 0,
        max_text_ctx: None,
        word_timestamps: wordts,
        model_hint: Some(model_short.to_string()),
        ..DecodeParams::default()
    };

    if std::env::var("PROBE_CONTEXT_AB").as_deref() == Ok("1") {
        assert_eq!(
            model_short, "tiny.en",
            "context A/B is scoped to the tiny.en regression"
        );
        let pairs = std::env::var("PROBE_CONTEXT_AB_PAIRS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(CONTEXT_AB_DEFAULT_PAIRS);
        println!(
            "CONTEXT_AB_CONFIG pairs={pairs} order=alternating null_first=true candidate_policy=tiny_en_segment_ts_no_carry historical_override=max_context_negative"
        );
        run_tiny_en_context_ab(&model, &samples, &params, pairs);
        return;
    }

    // warm (mmap/page-in) then timed
    let t = Instant::now();
    let out = transcribe_samples(&model, &samples, &params, &(|| Ok(()))).expect("transcribe");
    let dt = t.elapsed().as_secs_f64();

    // Per-sub-part decode attribution (only populated under
    // FRANKEN_WHISPER_PERF_SPANS=1; thread-local on this calling thread).
    if std::env::var("FRANKEN_WHISPER_PERF_SPANS").as_deref() == Ok("1") {
        use franken_whisper::native_engine::decoder::{SUB_LABELS, take_sub_ns};
        let ns = take_sub_ns();
        let total: u128 = ns.iter().sum();
        eprintln!("--- forward_step sub-part breakdown (sum over all tokens) ---");
        let mut idx: Vec<usize> = (0..ns.len()).collect();
        idx.sort_by(|&a, &b| ns[b].cmp(&ns[a]));
        for i in idx {
            let ms = ns[i] as f64 / 1e6;
            let pct = if total > 0 {
                ns[i] as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            eprintln!("  {:<18} {:>8.1} ms  {:>5.1}%", SUB_LABELS[i], ms, pct);
        }
        eprintln!(
            "  {:<18} {:>8.1} ms (forward_step total)",
            "SUM",
            total as f64 / 1e6
        );
    }

    // Layer-skip self-draft accept rate (only when FW_DRAFT_ACCEPT_LAYERS is set).
    if let Ok(k) = std::env::var("FW_DRAFT_ACCEPT_LAYERS") {
        let (m, tot) = franken_whisper::native_engine::decoder::drain_draft_accept();
        let pct = if tot > 0 {
            100.0 * m as f64 / tot as f64
        } else {
            0.0
        };
        eprintln!(
            "DRAFT_ACCEPT k={k} layers: {m}/{tot} decode steps matched full argmax = {pct:.1}% accept"
        );
    }

    let chars: usize = out.segments.iter().map(|s| s.text.len()).sum();
    if std::env::var("PROBE_DUMP_TEXT").as_deref() == Ok("1") {
        let full: String = out.segments.iter().map(|s| s.text.as_str()).collect();
        eprintln!("TRANSCRIPT>>>{full}<<<");
    }
    // Segment-level timestamp dump (PROBE_DUMP_SEGS=1): one line per segment as
    // `SEG i [t0 -> t1] text`, matching whisper.cpp's `[HH:MM:SS.mmm -->
    // HH:MM:SS.mmm]` layout for a franken-vs-whisper.cpp timestamp diff.
    if std::env::var("PROBE_DUMP_SEGS").as_deref() == Ok("1") {
        for (i, s) in out.segments.iter().enumerate() {
            eprintln!(
                "SEG {i} [{:.3} -> {:.3}] {}",
                s.start_sec.unwrap_or(f64::NAN),
                s.end_sec.unwrap_or(f64::NAN),
                s.text.trim()
            );
        }
    }
    // Word-level DTW timestamp dump (PROBE_DUMP_WORDS=1, needs the `wordts` arg):
    // one line per word as `WORD [t0 -> t1] text`, for a franken-vs-whisper.cpp
    // (`-dtw <model> -ml 1`) word-timestamp diff.
    if std::env::var("PROBE_DUMP_WORDS").as_deref() == Ok("1") {
        match out.word_timings.as_ref() {
            Some(wt) => {
                for w in wt.iter().flatten() {
                    eprintln!("WORD [{:.3} -> {:.3}] {}", w.start_sec, w.end_sec, w.text);
                }
            }
            None => eprintln!("WORD (none — word_timestamps not enabled; pass `wordts`)"),
        }
    }
    let rtf = dt / audio_sec;
    println!(
        "model={model_short} wav={wav} repeat={repeat} wordts={wordts} ch={ch} | audio={audio_sec:.1}s load={load_ms:.0}ms | transcribe={:.3}s RTF={rtf:.4} | segs={} chars={chars}",
        dt,
        out.segments.len()
    );
}
