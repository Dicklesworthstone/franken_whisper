use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::error::FwResult;
use crate::model::{OutputFormat, TranscriptionResult, TranscriptionSegment};

/// Write the transcription result into the requested output formats.
pub fn write_artifacts(
    formats: &[OutputFormat],
    result: &TranscriptionResult,
    output_prefix: &Path,
) -> FwResult<Vec<PathBuf>> {
    let mut artifacts = Vec::new();

    for fmt in formats {
        let ext = match fmt {
            OutputFormat::Txt => "txt",
            OutputFormat::Vtt => "vtt",
            OutputFormat::Srt => "srt",
            OutputFormat::Csv => "csv",
            OutputFormat::Json => "json",
            OutputFormat::JsonFull => "json_full",
            OutputFormat::Lrc => "lrc",
        };
        let candidate = Path::new(&format!("{}.{ext}", output_prefix.display())).to_path_buf();

        match fmt {
            OutputFormat::Txt => write_txt(&candidate, result)?,
            OutputFormat::Vtt => write_vtt(&candidate, result)?,
            OutputFormat::Srt => write_srt(&candidate, result)?,
            OutputFormat::Csv => write_csv(&candidate, result)?,
            OutputFormat::Json => write_json(&candidate, result)?,
            OutputFormat::JsonFull => write_json_full(&candidate, result)?,
            OutputFormat::Lrc => write_lrc(&candidate, result)?,
        }

        artifacts.push(candidate);
    }

    Ok(artifacts)
}

fn write_txt(path: &Path, result: &TranscriptionResult) -> FwResult<()> {
    // BufWriter batches the per-segment writeln! into ~8 KiB write() syscalls
    // instead of one syscall per line — byte-identical output, far fewer syscalls
    // on long transcripts. Explicit flush surfaces write errors (raw-File drop
    // would swallow them).
    let mut file = BufWriter::new(File::create(path)?);
    for seg in &result.segments {
        writeln!(file, "{}", seg.text)?;
    }
    file.flush()?;
    Ok(())
}

fn format_timestamp_vtt(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}

fn format_timestamp_srt(seconds: f64) -> String {
    let total_ms = (seconds * 1000.0).round() as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

fn write_vtt(path: &Path, result: &TranscriptionResult) -> FwResult<()> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "WEBVTT\n")?;
    for seg in &result.segments {
        if let (Some(start), Some(end)) = (seg.start_sec, seg.end_sec) {
            writeln!(
                file,
                "{} --> {}",
                format_timestamp_vtt(start),
                format_timestamp_vtt(end)
            )?;
            writeln!(file, "{}\n", seg.text)?;
        }
    }
    file.flush()?;
    Ok(())
}

fn write_srt(path: &Path, result: &TranscriptionResult) -> FwResult<()> {
    let mut file = BufWriter::new(File::create(path)?);
    for (i, seg) in result.segments.iter().enumerate() {
        if let (Some(start), Some(end)) = (seg.start_sec, seg.end_sec) {
            writeln!(file, "{}", i + 1)?;
            writeln!(
                file,
                "{} --> {}",
                format_timestamp_srt(start),
                format_timestamp_srt(end)
            )?;
            writeln!(file, "{}\n", seg.text)?;
        }
    }
    file.flush()?;
    Ok(())
}

fn write_csv(path: &Path, result: &TranscriptionResult) -> FwResult<()> {
    let mut file = BufWriter::new(File::create(path)?);
    writeln!(file, "start,end,speaker,text")?;
    for seg in &result.segments {
        let start = seg.start_sec.unwrap_or(0.0);
        let end = seg.end_sec.unwrap_or(0.0);
        let speaker = seg.speaker.as_deref().unwrap_or("");
        // simple CSV escaping: replace " with "" and wrap in "
        let escaped_speaker = speaker.replace('\"', "\"\"");
        let escaped_text = seg.text.replace('\"', "\"\"");
        writeln!(
            file,
            "{},{},\"{}\",\"{}\"",
            start, end, escaped_speaker, escaped_text
        )?;
    }
    file.flush()?;
    Ok(())
}

#[derive(Serialize)]
struct JsonTranscript<'a> {
    transcription: &'a [TranscriptionSegment],
}

fn write_json(path: &Path, result: &TranscriptionResult) -> FwResult<()> {
    let mut file = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(
        &mut file,
        &JsonTranscript {
            transcription: &result.segments,
        },
    )?;
    file.flush()?;
    Ok(())
}

fn write_json_full(path: &Path, result: &TranscriptionResult) -> FwResult<()> {
    let mut file = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut file, result)?;
    file.flush()?;
    Ok(())
}

fn write_lrc(path: &Path, result: &TranscriptionResult) -> FwResult<()> {
    let mut file = BufWriter::new(File::create(path)?);
    for seg in &result.segments {
        if let Some(start) = seg.start_sec {
            let total_ms = (start * 1000.0).round() as u64;
            let m = total_ms / 60_000;
            let s = (total_ms % 60_000) / 1000;
            let cs = (total_ms % 1000) / 10;
            writeln!(file, "[{:02}:{:02}.{:02}] {}", m, s, cs, seg.text)?;
        }
    }
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owned_json_transcript_bytes(segments: &[TranscriptionSegment]) -> Vec<u8> {
        serde_json::to_vec_pretty(&serde_json::json!({ "transcription": segments }))
            .expect("serialize owned JSON transcript")
    }

    fn borrowed_json_transcript_bytes(segments: &[TranscriptionSegment]) -> Vec<u8> {
        serde_json::to_vec_pretty(&JsonTranscript {
            transcription: segments,
        })
        .expect("serialize borrowed JSON transcript")
    }

    #[test]
    fn test_format_timestamp_vtt() {
        assert_eq!(format_timestamp_vtt(0.0), "00:00:00.000");
        assert_eq!(format_timestamp_vtt(1.5), "00:00:01.500");
        assert_eq!(format_timestamp_vtt(3661.123), "01:01:01.123");
    }

    #[test]
    fn test_format_timestamp_srt() {
        assert_eq!(format_timestamp_srt(0.0), "00:00:00,000");
        assert_eq!(format_timestamp_srt(1.5), "00:00:01,500");
        assert_eq!(format_timestamp_srt(3661.123), "01:01:01,123");
    }

    #[test]
    fn test_csv_escaping_speaker_and_text() {
        let result = TranscriptionResult {
            backend: crate::model::BackendKind::WhisperCpp,
            transcript: "".to_string(),
            language: None,
            segments: vec![TranscriptionSegment {
                start_sec: Some(1.0),
                end_sec: Some(2.0),
                text: "Hello, \"world\"".to_string(),
                speaker: Some("Speaker 1, \"Boss\"".to_string()),
                confidence: None,
            }],
            acceleration: None,
            raw_output: serde_json::json!({}),
            artifact_paths: vec![],
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.csv");
        write_csv(&path, &result).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            content,
            "start,end,speaker,text\n1,2,\"Speaker 1, \"\"Boss\"\"\",\"Hello, \"\"world\"\"\"\n"
        );
    }

    #[test]
    fn borrowed_json_transcript_matches_owned_value_bytes() {
        let cases = [
            Vec::new(),
            vec![TranscriptionSegment {
                start_sec: None,
                end_sec: None,
                text: String::new(),
                speaker: None,
                confidence: None,
            }],
            vec![
                TranscriptionSegment {
                    start_sec: Some(-0.0),
                    end_sec: Some(1.25),
                    text: "line one\nline two \\\"quoted\\\" λ 🎧".to_owned(),
                    speaker: Some("SPEAKER_00/話者".to_owned()),
                    confidence: Some(0.9375),
                },
                TranscriptionSegment {
                    start_sec: Some(123.456_789),
                    end_sec: Some(456.789_012),
                    text: "第二段".repeat(64),
                    speaker: None,
                    confidence: Some(0.999_999),
                },
            ],
        ];

        for segments in &cases {
            assert_eq!(
                borrowed_json_transcript_bytes(segments),
                owned_json_transcript_bytes(segments),
                "borrowed JSON transcript bytes differ"
            );
        }

        assert_eq!(
            String::from_utf8(borrowed_json_transcript_bytes(&[])).expect("UTF-8 JSON"),
            "{\n  \"transcription\": []\n}"
        );
    }

    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn borrowed_json_transcript_serialization_perf() {
        use sha2::{Digest as _, Sha256};
        use std::hint::black_box;
        use std::time::Instant;

        const SAMPLES: usize = 21;

        let executable = std::fs::read(std::env::current_exe().expect("test executable path"))
            .expect("read test executable");
        eprintln!(
            "export_json_serialization binary_sha256={:x}",
            Sha256::digest(executable)
        );

        fn time_owned(segments: &[TranscriptionSegment], iterations: usize) -> u128 {
            let started = Instant::now();
            for _ in 0..iterations {
                let envelope = serde_json::json!({
                    "transcription": black_box(segments),
                });
                let mut bytes = Vec::new();
                serde_json::to_writer_pretty(&mut bytes, black_box(&envelope))
                    .expect("serialize owned JSON transcript");
                drop(black_box(bytes));
            }
            started.elapsed().as_nanos()
        }

        fn time_borrowed(segments: &[TranscriptionSegment], iterations: usize) -> u128 {
            let started = Instant::now();
            for _ in 0..iterations {
                let envelope = JsonTranscript {
                    transcription: black_box(segments),
                };
                let mut bytes = Vec::new();
                serde_json::to_writer_pretty(&mut bytes, black_box(&envelope))
                    .expect("serialize borrowed JSON transcript");
                drop(black_box(bytes));
            }
            started.elapsed().as_nanos()
        }

        fn percentile(values: &[f64], percentile: usize) -> f64 {
            let mut sorted = values.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted[(sorted.len() - 1) * percentile / 100]
        }

        fn median_ns(values: &[u128]) -> u128 {
            let mut sorted = values.to_vec();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        }

        fn run_shape(name: &str, segments: &[TranscriptionSegment], iterations: usize) {
            let owned_bytes = owned_json_transcript_bytes(segments);
            let borrowed_bytes = borrowed_json_transcript_bytes(segments);
            assert_eq!(borrowed_bytes, owned_bytes, "{name} exact bytes");
            eprintln!(
                "export_json_serialization shape={name} segments={} output_bytes={} output_sha256={:x}",
                segments.len(),
                owned_bytes.len(),
                Sha256::digest(&owned_bytes)
            );

            for _ in 0..3 {
                black_box(time_owned(segments, iterations));
                black_box(time_borrowed(segments, iterations));
            }

            let mut null_ratios = Vec::with_capacity(SAMPLES);
            let mut speedups = Vec::with_capacity(SAMPLES);
            let mut owned_times = Vec::with_capacity(SAMPLES);
            let mut borrowed_times = Vec::with_capacity(SAMPLES);

            for sample in 0..SAMPLES {
                let null_first = time_owned(segments, iterations);
                let null_second = time_owned(segments, iterations);
                let (null_numerator, null_denominator) = if sample % 2 == 0 {
                    (null_first, null_second)
                } else {
                    (null_second, null_first)
                };
                null_ratios.push(null_numerator as f64 / null_denominator as f64);

                let (owned, borrowed) = if sample % 2 == 0 {
                    (
                        time_owned(segments, iterations),
                        time_borrowed(segments, iterations),
                    )
                } else {
                    let borrowed = time_borrowed(segments, iterations);
                    let owned = time_owned(segments, iterations);
                    (owned, borrowed)
                };
                owned_times.push(owned);
                borrowed_times.push(borrowed);
                speedups.push(owned as f64 / borrowed as f64);
            }

            let null_median = percentile(&null_ratios, 50);
            assert!(
                (0.95..=1.05).contains(&null_median),
                "{name} null median {null_median:.6} outside predeclared guard"
            );
            eprintln!(
                "export_json_serialization shape={name} samples={SAMPLES} iterations={iterations} null_p10={:.6} null_median={null_median:.6} null_p90={:.6} owned_arm_median_ns={} borrowed_arm_median_ns={} speedup_p10={:.6} speedup_median={:.6} speedup_p90={:.6}",
                percentile(&null_ratios, 10),
                percentile(&null_ratios, 90),
                median_ns(&owned_times),
                median_ns(&borrowed_times),
                percentile(&speedups, 10),
                percentile(&speedups, 50),
                percentile(&speedups, 90),
            );
            eprintln!(
                "export_json_serialization shape={name} null_ratios={null_ratios:?} speedups={speedups:?}"
            );
        }

        let current_like = (0..32)
            .map(|index| TranscriptionSegment {
                start_sec: Some(f64::from(index) * 1.25),
                end_sec: Some(f64::from(index + 1) * 1.25),
                text: format!(
                    "segment {index:02}: measured transcript output with Unicode λ and 🎧"
                ),
                speaker: (index % 4 == 0).then(|| format!("SPEAKER_{:02}", index % 3)),
                confidence: Some(0.85 + f64::from(index % 10) / 100.0),
            })
            .collect::<Vec<_>>();
        let long = (0..4_096)
            .map(|index| TranscriptionSegment {
                start_sec: Some(index as f64 * 0.75),
                end_sec: Some((index + 1) as f64 * 0.75),
                text: format!(
                    "long segment {index:04}: escaped \\\"text\\\", line-safe Unicode 日本語 {}",
                    "payload ".repeat(6)
                ),
                speaker: Some(format!("SPEAKER_{:02}", index % 16)),
                confidence: Some(0.8 + f64::from(index % 20) / 100.0),
            })
            .collect::<Vec<_>>();

        run_shape("current_like", &current_like, 1_000);
        run_shape("long", &long, 8);
    }

    /// Byte-exact output guard for the `BufWriter`-wrapped writers (srt/vtt/txt
    /// previously had no content test). Confirms the buffered+flushed writers emit
    /// the exact bytes across a multi-segment result — the buffering change must be
    /// transparent.
    #[test]
    fn writers_emit_byte_exact_content() {
        let result = TranscriptionResult {
            backend: crate::model::BackendKind::WhisperCpp,
            transcript: "hello world".to_string(),
            language: Some("en".to_string()),
            segments: vec![
                TranscriptionSegment {
                    start_sec: Some(0.0),
                    end_sec: Some(1.5),
                    text: "hello".to_string(),
                    speaker: None,
                    confidence: None,
                },
                TranscriptionSegment {
                    start_sec: Some(1.5),
                    end_sec: Some(3.0),
                    text: "world".to_string(),
                    speaker: None,
                    confidence: None,
                },
            ],
            acceleration: None,
            raw_output: serde_json::json!({}),
            artifact_paths: vec![],
        };
        let dir = tempfile::tempdir().unwrap();

        let srt = dir.path().join("o.srt");
        write_srt(&srt, &result).unwrap();
        assert_eq!(
            std::fs::read_to_string(&srt).unwrap(),
            "1\n00:00:00,000 --> 00:00:01,500\nhello\n\n2\n00:00:01,500 --> 00:00:03,000\nworld\n\n"
        );

        let vtt = dir.path().join("o.vtt");
        write_vtt(&vtt, &result).unwrap();
        assert_eq!(
            std::fs::read_to_string(&vtt).unwrap(),
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nhello\n\n00:00:01.500 --> 00:00:03.000\nworld\n\n"
        );

        let txt = dir.path().join("o.txt");
        write_txt(&txt, &result).unwrap();
        assert_eq!(std::fs::read_to_string(&txt).unwrap(), "hello\nworld\n");
    }
}
