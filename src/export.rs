use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::error::FwResult;
use crate::model::{OutputFormat, TranscriptionResult};

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

fn write_json(path: &Path, result: &TranscriptionResult) -> FwResult<()> {
    let mut file = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(
        &mut file,
        &serde_json::json!({ "transcription": result.segments }),
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
    use crate::model::TranscriptionSegment;

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
