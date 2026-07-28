//! Live-incumbent A/B: franken_whisper vs `whisper.cpp`, both driven from **one
//! invocation**, order-alternating, with an A/A null for **each** arm.
//!
//! ## Why this exists
//!
//! A self-speedup (our own code, before vs after) is maintenance. A competitive
//! claim requires a measured ratio against the actual legacy incumbent, produced
//! by a harness that runs the incumbent **side by side in the same invocation** —
//! otherwise the two engines are measured under different machine states and the
//! ratio inherits an uncontrolled between-session drift.
//!
//! The prior tiny.en segment-timestamp result (1.35×) was measured with both
//! binaries in the same *session* but not interleaved, and carried no cross-tool
//! null. This harness closes exactly that gap.
//!
//! ## What is measured, and why it is the fair quantity
//!
//! Both sides are timed on **transcribe work, excluding one-time model load**:
//!
//! - `whisper.cpp` self-reports `load time` and `total time`; its transcribe
//!   time is `total − load`.
//! - franken is timed in-process around `transcribe_samples`, with the model
//!   already resident.
//!
//! Comparing full process wall would instead compare `whisper-cli`'s thin
//! inference binary against franken's *orchestrator* (routing, storage,
//! normalization), which is not the quantity in question and would understate
//! franken. Excluding load on both sides is the matched comparison.
//!
//! Residual asymmetry, disclosed rather than hidden: `whisper.cpp` additionally
//! pays process spawn and stdout formatting inside its `total`, on the order of
//! milliseconds against a ~1.7 s measurement. It is not subtracted.
//!
//! ## Statistic and gate
//!
//! Per round, both engines run once, **alternating which goes first**, so any
//! monotonic machine drift lands on both arms equally. The statistic is the
//! **median of per-round ratios** (`wc_transcribe / fw_transcribe`).
//!
//! Two A/A nulls run in the same invocation and the same alternating shape:
//! franken against itself, and `whisper.cpp` against itself. A claim is
//! decidable only when the comparison median lies outside **both** null CI95s
//! with a 2× margin, and when the comparison medians from lighter and heavier
//! rounds differ by at most 0.1×. `cv` is recorded as provenance and decides
//! nothing.
//!
//! ## Usage
//!
//! ```text
//! incumbent_ab <model_short> <wav> [rounds]
//! FW_INCUMBENT_BIN=/path/to/whisper-cli   (default: legacy_whispercpp/.../whisper-cli)
//! FW_INCUMBENT_THREADS=16                 (whisper.cpp's best tiny.en thread count)
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use franken_whisper::native_engine::decode::{DecodeParams, LoadedModel, transcribe_samples};
use franken_whisper::native_engine::find_model_file;
use franken_whisper::native_engine::ggml::GgmlModel;
use sha2::{Digest, Sha256};

const MAX_LOAD_SPLIT_GAP: f64 = 0.1;

/// Read a PCM16 WAV into mono f32. Mirrors `e2e_probe`'s reader.
fn read_wav_mono16k(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read wav");
    assert_eq!(&bytes[0..4], b"RIFF", "not a RIFF file");
    assert_eq!(&bytes[8..12], b"WAVE", "not a WAVE file");
    let mut pos = 12;
    let mut channels = 1u16;
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
            bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
        } else if id == b"data" {
            data = &bytes[body..(body + sz).min(bytes.len())];
        }
        pos = body + sz + (sz & 1);
    }
    assert_eq!(bits, 16, "expected PCM16");
    let step = 2 * channels as usize;
    let mut samples = Vec::with_capacity(data.len() / step);
    let mut i = 0;
    while i + step <= data.len() {
        let mut acc = 0i32;
        for c in 0..channels as usize {
            acc += i16::from_le_bytes([data[i + 2 * c], data[i + 2 * c + 1]]) as i32;
        }
        samples.push((acc as f32 / channels as f32) / 32768.0);
        i += step;
    }
    samples
}

fn sha256_file(path: &Path) -> String {
    match std::fs::read(path) {
        Ok(bytes) => format!("{:x}", Sha256::digest(&bytes)),
        Err(_) => "unreadable".to_owned(),
    }
}

/// Self-reported identity of this harness binary (campaign harness contract).
fn executable_identity() -> String {
    match std::env::current_exe() {
        Ok(path) => format!("{} {}", sha256_file(&path), path.display()),
        Err(error) => format!("unavailable ({error})"),
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        f64::midpoint(sorted[mid - 1], sorted[mid])
    }
}

/// Deterministic bootstrap CI95 of the median (fixed LCG seed — no `rand`, and
/// reproducible across runs so a reviewer can re-derive the interval).
fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
    if values.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut medians = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let mut sample = Vec::with_capacity(values.len());
        for _ in 0..values.len() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let idx = (state >> 33) as usize % values.len();
            sample.push(values[idx]);
        }
        medians.push(median(&sample));
    }
    medians.sort_by(f64::total_cmp);
    (medians[50], medians[1949])
}

fn cv(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    var.sqrt() / mean
}

/// Split rounds at the median total arm cost and return the comparison median
/// for the lighter and heavier halves plus their absolute gap. For an odd
/// number of rounds, the single middle-cost round is deliberately excluded.
fn load_split(fw_ms: &[f64], wc_ms: &[f64], compare: &[f64]) -> Option<(f64, f64, f64)> {
    if fw_ms.len() != wc_ms.len() || fw_ms.len() != compare.len() || fw_ms.len() < 3 {
        return None;
    }

    let mut totals: Vec<(f64, f64)> = fw_ms
        .iter()
        .zip(wc_ms)
        .zip(compare)
        .map(|((fw, wc), ratio)| (fw + wc, *ratio))
        .collect();
    totals.sort_by(|a, b| a.0.total_cmp(&b.0));

    let half = totals.len() / 2;
    let light: Vec<f64> = totals[..half].iter().map(|(_, ratio)| *ratio).collect();
    let heavy: Vec<f64> = totals[totals.len() - half..]
        .iter()
        .map(|(_, ratio)| *ratio)
        .collect();
    let light_median = median(&light);
    let heavy_median = median(&heavy);
    Some((
        light_median,
        heavy_median,
        (light_median - heavy_median).abs(),
    ))
}

/// One `whisper.cpp` run. Returns `(transcribe_ms, total_ms, load_ms, chars)`.
///
/// `transcribe_ms` is `total − load`, so a one-time model load is excluded on
/// this side exactly as it is on franken's.
fn run_incumbent(bin: &Path, model: &Path, wav: &str, threads: usize) -> (f64, f64, f64, usize) {
    let output = Command::new(bin)
        .args([
            "-m",
            &model.display().to_string(),
            "-f",
            wav,
            "-bs",
            "1",
            "-bo",
            "1",
            "-t",
            &threads.to_string(),
        ])
        .output()
        .expect("run whisper-cli");
    let text = String::from_utf8_lossy(&output.stderr).into_owned()
        + &String::from_utf8_lossy(&output.stdout);

    let field = |needle: &str| -> f64 {
        text.lines()
            .find(|line| line.contains(needle))
            .and_then(|line| {
                line.rsplit('=')
                    .next()
                    .map(|tail| tail.trim().trim_end_matches(" ms").trim())
                    .and_then(|value| value.split_whitespace().next())
                    .and_then(|value| value.parse::<f64>().ok())
            })
            .unwrap_or(f64::NAN)
    };
    let total = field("total time");
    let load = field("load time");
    let chars: usize = text
        .lines()
        .filter(|line| line.starts_with('['))
        .map(|line| line.split(']').nth(1).map_or(0, str::len))
        .sum();
    assert!(
        total.is_finite() && load.is_finite(),
        "could not parse whisper.cpp timings; is this whisper-cli?"
    );
    (total - load, total, load, chars)
}

/// One franken run: transcribe with the model already resident.
fn run_franken(model: &LoadedModel, samples: &[f32], params: &DecodeParams) -> (f64, usize) {
    let started = Instant::now();
    let out = transcribe_samples(model, samples, params, &(|| Ok(()))).expect("fw transcribe");
    let elapsed = started.elapsed().as_secs_f64() * 1e3;
    let chars: usize = out.segments.iter().map(|s| s.text.trim().len()).sum();
    (elapsed, chars)
}

fn report(label: &str, ratios: &[f64]) -> (f64, f64, f64) {
    let med = median(ratios);
    let (lo, hi) = bootstrap_median_ci(ratios);
    println!(
        "{label} median={med:.6} ci95=[{lo:.6},{hi:.6}] cv={:.6} n={}",
        cv(ratios),
        ratios.len()
    );
    (med, lo, hi)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_short = args.get(1).map(String::as_str).unwrap_or("tiny.en");
    let wav = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "tests/fixtures/native/jfk.wav".to_string());
    let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(11);
    assert!(
        rounds >= 3 && rounds % 2 == 1,
        "rounds must be odd and >= 3"
    );

    let incumbent = std::env::var("FW_INCUMBENT_BIN")
        .unwrap_or_else(|_| "legacy_whispercpp/whisper.cpp/build/bin/whisper-cli".to_string());
    let incumbent = PathBuf::from(incumbent);
    assert!(
        incumbent.is_file(),
        "incumbent binary not found at {} (set FW_INCUMBENT_BIN)",
        incumbent.display()
    );
    let threads: usize = std::env::var("FW_INCUMBENT_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);

    let model_path = find_model_file(model_short)
        .unwrap_or_else(|| panic!("model {model_short} not found in search dirs"));
    let model = GgmlModel::load(&model_path)
        .and_then(LoadedModel::from_ggml)
        .expect("load model");
    let samples = read_wav_mono16k(&wav);

    let params = DecodeParams {
        language: Some("en".to_string()),
        translate: false,
        timestamps: true,
        n_threads: 0,
        max_text_ctx: None,
        word_timestamps: false,
        model_hint: Some(model_short.to_string()),
        ..DecodeParams::default()
    };

    println!("harness_elf_sha256={}", executable_identity());
    println!(
        "incumbent_bin_sha256={} {}",
        sha256_file(&incumbent),
        incumbent.display()
    );
    println!(
        "model_sha256={} {}",
        sha256_file(&model_path),
        model_path.display()
    );
    println!(
        "INCUMBENT_AB_CONFIG rounds={rounds} order=alternating wav={wav} \
         audio_sec={:.1} incumbent_threads={threads} \
         measured=transcribe_excluding_model_load",
        samples.len() as f64 / 16000.0
    );

    // Warm both engines once; neither warm-up is timed.
    let (_, fw_chars) = run_franken(&model, &samples, &params);
    let (_, _, _, wc_chars) = run_incumbent(&incumbent, &model_path, &wav, threads);
    println!("INCUMBENT_AB_COVERAGE fw_chars={fw_chars} wc_chars={wc_chars}");

    let mut compare = Vec::with_capacity(rounds);
    let mut fw_null = Vec::with_capacity(rounds);
    let mut wc_null = Vec::with_capacity(rounds);
    let mut fw_ms = Vec::with_capacity(rounds);
    let mut wc_ms = Vec::with_capacity(rounds);

    for round in 0..rounds {
        // Alternate which engine runs first so monotonic drift hits both equally.
        let (fw_a, wc_a) = if round % 2 == 0 {
            let fw = run_franken(&model, &samples, &params).0;
            let wc = run_incumbent(&incumbent, &model_path, &wav, threads).0;
            (fw, wc)
        } else {
            let wc = run_incumbent(&incumbent, &model_path, &wav, threads).0;
            let fw = run_franken(&model, &samples, &params).0;
            (fw, wc)
        };
        // Second observation of each engine, opposite order: pairs with the
        // first to form each arm's own A/A null inside this same invocation.
        let (fw_b, wc_b) = if round % 2 == 0 {
            let wc = run_incumbent(&incumbent, &model_path, &wav, threads).0;
            let fw = run_franken(&model, &samples, &params).0;
            (fw, wc)
        } else {
            let fw = run_franken(&model, &samples, &params).0;
            let wc = run_incumbent(&incumbent, &model_path, &wav, threads).0;
            (fw, wc)
        };

        compare.push(wc_a / fw_a);
        fw_null.push(fw_a / fw_b);
        wc_null.push(wc_a / wc_b);
        fw_ms.push(fw_a);
        wc_ms.push(wc_a);
    }

    println!(
        "INCUMBENT_AB_TIMES fw_median_ms={:.3} wc_median_ms={:.3}",
        median(&fw_ms),
        median(&wc_ms)
    );
    // Raw per-round series. Interleaving cancels drift *over time*, but it does
    // NOT cancel one engine being more load-sensitive than the other — that bias
    // survives alternation and silently scales the ratio. Emitting the raw series
    // lets a reviewer regress ratio against absolute round cost (a proxy for
    // instantaneous load) and see whether the ratio moves with it.
    println!("INCUMBENT_AB_RAW fw_ms={fw_ms:?}");
    println!("INCUMBENT_AB_RAW wc_ms={wc_ms:?}");
    println!("INCUMBENT_AB_RAW compare={compare:?}");
    println!("INCUMBENT_AB_RAW null_fw={fw_null:?}");
    println!("INCUMBENT_AB_RAW null_wc={wc_null:?}");
    // This is part of the verdict, not commentary: differential load
    // sensitivity can survive order alternation and bias the cross-tool ratio.
    let (light_median, heavy_median, load_split_gap) =
        load_split(&fw_ms, &wc_ms, &compare).expect("odd rounds >= 3 form a load split");
    println!(
        "INCUMBENT_AB_LOAD_SPLIT lighter_rounds_median={light_median:.6} \
         heavier_rounds_median={heavy_median:.6} n_each={} gap={load_split_gap:.6} \
         max_gap={MAX_LOAD_SPLIT_GAP:.6}",
        rounds / 2
    );
    let (_, fw_lo, fw_hi) = report("INCUMBENT_AB_NULL_FW", &fw_null);
    let (_, wc_lo, wc_hi) = report("INCUMBENT_AB_NULL_WC", &wc_null);
    let (cmp_med, cmp_lo, cmp_hi) = report("INCUMBENT_AB_COMPARE", &compare);

    // Decidable only if the comparison clears BOTH nulls' worst edge by 2x.
    let fw_half = (fw_hi - 1.0).abs().max((1.0 - fw_lo).abs());
    let wc_half = (wc_hi - 1.0).abs().max((1.0 - wc_lo).abs());
    let required = 1.0 + 2.0 * fw_half.max(wc_half);
    let load_split_clear = load_split_gap <= MAX_LOAD_SPLIT_GAP;
    let verdict = if !load_split_clear {
        "UNDECIDABLE"
    } else if cmp_med > required && cmp_lo > 1.0 {
        "WIN"
    } else if cmp_med < 1.0 / required && cmp_hi < 1.0 {
        "LOSS"
    } else {
        "UNDECIDABLE"
    };
    println!(
        "INCUMBENT_AB_GATE method=median_vs_both_null_ci95_2x_margin \
         fw_null_half={fw_half:.6} wc_null_half={wc_half:.6} required={required:.6} \
         compare_median={cmp_med:.6} compare_ci95=[{cmp_lo:.6},{cmp_hi:.6}] \
         load_split_gap={load_split_gap:.6} load_split_max={MAX_LOAD_SPLIT_GAP:.6} \
         load_split_clear={load_split_clear} \
         cv_is_provenance_only=true class=vs_incumbent verdict={verdict}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_split_excludes_middle_round_and_reports_absolute_gap() {
        let fw_ms = [1.0, 2.0, 3.0, 4.0, 5.0];
        let wc_ms = [1.0, 2.0, 3.0, 4.0, 5.0];
        let compare = [1.1, 1.2, 9.9, 1.3, 1.4];

        let (light, heavy, gap) =
            load_split(&fw_ms, &wc_ms, &compare).expect("valid equal-length inputs");

        assert!((light - 1.15).abs() < f64::EPSILON);
        assert!((heavy - 1.35).abs() < f64::EPSILON);
        assert!((gap - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn load_split_rejects_mismatched_or_too_short_inputs() {
        assert!(load_split(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 2.0]).is_none());
        assert!(load_split(&[1.0, 2.0, 3.0], &[1.0, 2.0], &[1.0, 2.0, 3.0]).is_none());
    }
}
