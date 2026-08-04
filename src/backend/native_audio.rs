//! Shared waveform-aware audio analysis for native backend adapters.
//!
//! This module provides deterministic, allocation-safe WAV parsing and
//! energy-based speech-region extraction used by native backend runtimes.

use std::fs;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use serde_json::{Value, json};

const FRAME_MS: u32 = 20;
const MIN_THRESHOLD: f32 = 0.003;
const GAP_BRIDGE_MAX_FRAMES: usize = 2;
const MIN_REGION_FRAMES: usize = 2;

#[derive(Debug, Clone)]
pub(crate) struct AudioRegion {
    pub start_ms: u64,
    pub end_ms: u64,
    pub avg_rms: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct NativeAudioAnalysis {
    pub duration_ms: u64,
    pub sample_rate_hz: u32,
    pub frame_ms: u32,
    pub frame_count: usize,
    pub avg_rms: f32,
    pub max_rms: f32,
    pub activity_threshold: f32,
    pub active_regions: Vec<AudioRegion>,
}

impl NativeAudioAnalysis {
    pub(crate) fn as_json(&self) -> Value {
        json!({
            "sample_rate_hz": self.sample_rate_hz,
            "duration_ms": self.duration_ms,
            "frame_ms": self.frame_ms,
            "frame_count": self.frame_count,
            "avg_rms": self.avg_rms,
            "max_rms": self.max_rms,
            "activity_threshold": self.activity_threshold,
            "active_region_count": self.active_regions.len(),
            "active_regions": self.active_regions.iter().map(|r| {
                json!({
                    "start_ms": r.start_ms,
                    "end_ms": r.end_ms,
                    "avg_rms": r.avg_rms,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

#[derive(Debug, Clone)]
struct WavPcm16Mono {
    sample_rate_hz: u32,
    samples: Vec<i16>,
}

pub(crate) fn analyze_wav(
    path: &Path,
    duration_override_ms: Option<u64>,
) -> Result<NativeAudioAnalysis, String> {
    let wav = parse_pcm16_mono_wav(path)?;

    if wav.samples.is_empty() {
        return Ok(NativeAudioAnalysis {
            duration_ms: duration_override_ms.unwrap_or(0),
            sample_rate_hz: wav.sample_rate_hz,
            frame_ms: FRAME_MS,
            frame_count: 0,
            avg_rms: 0.0,
            max_rms: 0.0,
            activity_threshold: MIN_THRESHOLD,
            active_regions: Vec::new(),
        });
    }

    let frame_samples = (((wav.sample_rate_hz as u64) * (FRAME_MS as u64)) / 1_000)
        .max(1)
        .try_into()
        .unwrap_or(1usize);

    let frame_rms = compute_frame_rms(&wav.samples, frame_samples);
    let frame_count = frame_rms.len();

    let avg_rms = if frame_rms.is_empty() {
        0.0
    } else {
        frame_rms.iter().copied().sum::<f32>() / frame_rms.len() as f32
    };
    let max_rms = frame_rms
        .iter()
        .copied()
        .fold(0.0_f32, |acc, v| if v > acc { v } else { acc });

    // Deterministic adaptive threshold: scales with corpus energy while
    // preserving a conservative floor for quiet speech.
    let activity_threshold =
        ((avg_rms * 1.5).max(MIN_THRESHOLD)).min((max_rms * 0.8).max(MIN_THRESHOLD));

    let mut active = frame_rms
        .iter()
        .map(|rms| *rms >= activity_threshold)
        .collect::<Vec<_>>();

    let computed_duration_ms = (((wav.samples.len() as f64) / (wav.sample_rate_hz as f64))
        * 1_000.0)
        .round()
        .max(0.0) as u64;
    bridge_short_gaps(&mut active, GAP_BRIDGE_MAX_FRAMES);
    let active_regions = active_regions_from_frames(
        &frame_rms,
        &active,
        FRAME_MS,
        MIN_REGION_FRAMES,
        computed_duration_ms,
    );

    Ok(NativeAudioAnalysis {
        duration_ms: duration_override_ms
            .filter(|value| *value > 0)
            .unwrap_or(computed_duration_ms),
        sample_rate_hz: wav.sample_rate_hz,
        frame_ms: FRAME_MS,
        frame_count,
        avg_rms,
        max_rms,
        activity_threshold,
        active_regions,
    })
}

fn parse_pcm16_mono_wav(path: &Path) -> Result<WavPcm16Mono, String> {
    let file_len = fs::metadata(path)
        .map_err(|error| format!("failed to read wav `{}`: {error}", path.display()))?
        .len() as usize;
    if file_len < 44 {
        return Err(format!("wav too small: {file_len} bytes"));
    }

    let file = fs::File::open(path)
        .map_err(|error| format!("failed to read wav `{}`: {error}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 12];
    reader
        .read_exact(&mut header)
        .map_err(|error| format!("failed to read wav header: {error}"))?;
    if header.get(0..4) != Some(b"RIFF") || header.get(8..12) != Some(b"WAVE") {
        return Err("unsupported wav container; expected RIFF/WAVE".to_owned());
    }

    let mut sample_rate_hz: Option<u32> = None;
    let mut channels: Option<u16> = None;
    let mut bits_per_sample: Option<u16> = None;
    let mut audio_format: Option<u16> = None;
    let mut data_offset: Option<u64> = None;
    let mut data_len: Option<u32> = None;

    loop {
        let mut chunk_header = [0u8; 8];
        match reader.read_exact(&mut chunk_header) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => {
                return Err(format!("failed to read wav chunk header: {error}"));
            }
        }

        let chunk_id = &chunk_header[0..4];
        let chunk_size = u32::from_le_bytes([
            chunk_header[4],
            chunk_header[5],
            chunk_header[6],
            chunk_header[7],
        ]);
        let data_start = reader
            .stream_position()
            .map_err(|error| format!("failed to read wav chunk offset: {error}"))?;

        if chunk_id == b"fmt " {
            if chunk_size < 16 {
                return Err("invalid wav fmt chunk".to_owned());
            }
            let mut fmt = [0u8; 16];
            reader
                .read_exact(&mut fmt)
                .map_err(|error| format!("failed to read wav fmt chunk: {error}"))?;
            audio_format = Some(u16::from_le_bytes([fmt[0], fmt[1]]));
            channels = Some(u16::from_le_bytes([fmt[2], fmt[3]]));
            sample_rate_hz = Some(u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]));
            bits_per_sample = Some(u16::from_le_bytes([fmt[14], fmt[15]]));
            if chunk_size > 16 {
                reader
                    .seek(SeekFrom::Current((chunk_size - 16) as i64))
                    .map_err(|error| format!("failed to skip wav fmt padding: {error}"))?;
            }
        } else if chunk_id == b"data" {
            data_offset = Some(data_start);
            data_len = Some(chunk_size);
            reader
                .seek(SeekFrom::Current(chunk_size as i64))
                .map_err(|error| format!("failed to skip wav data chunk: {error}"))?;
        } else {
            reader
                .seek(SeekFrom::Current(chunk_size as i64))
                .map_err(|error| format!("failed to skip wav chunk: {error}"))?;
        }

        // Chunks are word-aligned.
        if chunk_size % 2 == 1 {
            reader
                .seek(SeekFrom::Current(1))
                .map_err(|error| format!("failed to skip wav padding: {error}"))?;
        }
    }

    let sample_rate_hz = sample_rate_hz.ok_or_else(|| "missing wav fmt sample_rate".to_owned())?;
    let channels = channels.ok_or_else(|| "missing wav fmt channels".to_owned())?;
    let bits_per_sample =
        bits_per_sample.ok_or_else(|| "missing wav fmt bits_per_sample".to_owned())?;
    let audio_format = audio_format.ok_or_else(|| "missing wav fmt audio_format".to_owned())?;
    let data_offset = data_offset.ok_or_else(|| "missing wav data chunk".to_owned())?;
    let mut data_len = data_len.ok_or_else(|| "missing wav data chunk".to_owned())?;

    if audio_format != 1 {
        return Err(format!(
            "unsupported wav audio_format={audio_format}; expected PCM (1)"
        ));
    }
    if channels != 1 {
        return Err(format!(
            "unsupported wav channels={channels}; expected mono (1)"
        ));
    }
    if bits_per_sample != 16 {
        return Err(format!(
            "unsupported wav bits_per_sample={bits_per_sample}; expected 16"
        ));
    }
    if sample_rate_hz == 0 {
        return Err("invalid wav sample_rate 0".to_owned());
    }

    let available = (file_len as u64).saturating_sub(data_offset);
    if available == 0 {
        return Ok(WavPcm16Mono {
            sample_rate_hz,
            samples: Vec::new(),
        });
    }
    if data_len as u64 > available {
        data_len = available as u32;
    }

    reader
        .seek(SeekFrom::Start(data_offset))
        .map_err(|error| format!("failed to seek wav data: {error}"))?;

    let mut samples = Vec::with_capacity((data_len as usize) / 2);
    let mut remaining = data_len as usize;
    let mut buf = [0u8; 8192];
    let mut leftover: Option<u8> = None;

    while remaining > 0 {
        let to_read = buf.len().min(remaining);
        let read = reader
            .read(&mut buf[..to_read])
            .map_err(|error| format!("failed to read wav data: {error}"))?;
        if read == 0 {
            break;
        }
        remaining = remaining.saturating_sub(read);

        let mut start = 0usize;
        if let Some(prev) = leftover.take() {
            if read > 0 {
                samples.push(i16::from_le_bytes([prev, buf[0]]));
                start = 1;
            } else {
                leftover = Some(prev);
            }
        }

        let chunk = &buf[start..read];
        let (pairs, remainder) = chunk.as_chunks::<2>();
        for pair in pairs {
            samples.push(i16::from_le_bytes(*pair));
        }
        if let Some(remainder) = remainder.first() {
            leftover = Some(*remainder);
        }
    }

    // If the data chunk is truncated, leftover bytes are ignored.

    Ok(WavPcm16Mono {
        sample_rate_hz,
        samples,
    })
}

fn compute_frame_rms(samples: &[i16], frame_samples: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for chunk in samples.chunks(frame_samples.max(1)) {
        if chunk.is_empty() {
            continue;
        }
        let sum_sq = chunk.iter().fold(0.0_f64, |acc, value| {
            let normalized = (*value as f64) / 32768.0;
            acc + (normalized * normalized)
        });
        let rms = (sum_sq / (chunk.len() as f64)).sqrt() as f32;
        out.push(rms);
    }
    out
}

fn bridge_short_gaps(active: &mut [bool], max_gap_frames: usize) {
    if active.is_empty() {
        return;
    }

    let mut idx = 0usize;
    while idx < active.len() {
        if active[idx] {
            idx += 1;
            continue;
        }

        let gap_start = idx;
        while idx < active.len() && !active[idx] {
            idx += 1;
        }
        let gap_end = idx;
        let gap_len = gap_end.saturating_sub(gap_start);

        if gap_start > 0 && gap_end < active.len() && gap_len <= max_gap_frames {
            for frame in &mut active[gap_start..gap_end] {
                *frame = true;
            }
        }
    }
}

fn active_regions_from_frames(
    frame_rms: &[f32],
    active: &[bool],
    frame_ms: u32,
    min_region_frames: usize,
    duration_ms: u64,
) -> Vec<AudioRegion> {
    let mut regions = Vec::new();
    let mut idx = 0usize;

    while idx < active.len() {
        if !active[idx] {
            idx += 1;
            continue;
        }

        let start = idx;
        while idx < active.len() && active[idx] {
            idx += 1;
        }
        let end = idx;

        if end.saturating_sub(start) < min_region_frames {
            continue;
        }

        let avg_rms = if end > start {
            frame_rms[start..end].iter().copied().sum::<f32>() / (end - start) as f32
        } else {
            0.0
        };

        let start_ms = ((start as u64) * (frame_ms as u64)).min(duration_ms);
        let end_ms = ((end as u64) * (frame_ms as u64)).min(duration_ms);
        if end_ms <= start_ms {
            continue;
        }

        regions.push(AudioRegion {
            start_ms,
            end_ms,
            avg_rms,
        });
    }

    regions
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::tempdir;

    use super::analyze_wav;

    fn write_pcm16_mono_wav(path: &Path, sample_rate: u32, samples: &[i16]) {
        let data_len = (samples.len() * 2) as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);

        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");

        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * 2;
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        bytes.extend_from_slice(&16u16.to_le_bytes());

        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        std::fs::write(path, bytes).expect("write wav");
    }

    #[test]
    fn silence_has_no_active_regions() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("silence.wav");
        let samples = vec![0i16; 16_000];
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let analysis = analyze_wav(&wav, None).expect("analysis should succeed");
        assert!(analysis.active_regions.is_empty());
        assert_eq!(analysis.duration_ms, 1_000);
    }

    #[test]
    fn tone_has_detectable_active_region() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("tone.wav");

        let mut samples = vec![0i16; 16_000];
        // 1s of moderate-amplitude pulse train.
        for i in 0..16_000usize {
            if (i / 40) % 2 == 0 {
                samples.push(8_000);
            } else {
                samples.push(-8_000);
            }
        }
        samples.extend(vec![0i16; 16_000]);

        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let analysis = analyze_wav(&wav, None).expect("analysis should succeed");
        assert!(!analysis.active_regions.is_empty());
        assert!(analysis.active_regions[0].start_ms <= 1_100);
        assert!(analysis.active_regions[0].end_ms >= 1_900);
    }

    #[test]
    fn analysis_is_deterministic_for_same_input() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("deterministic.wav");

        let mut samples = Vec::new();
        samples.extend(vec![0i16; 4_000]);
        samples.extend(vec![7_000i16; 8_000]);
        samples.extend(vec![0i16; 4_000]);
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let first = analyze_wav(&wav, None).expect("first analysis");
        let second = analyze_wav(&wav, None).expect("second analysis");

        assert_eq!(first.duration_ms, second.duration_ms);
        assert_eq!(first.frame_count, second.frame_count);
        assert_eq!(first.active_regions.len(), second.active_regions.len());
        for (a, b) in first
            .active_regions
            .iter()
            .zip(second.active_regions.iter())
        {
            assert_eq!(a.start_ms, b.start_ms);
            assert_eq!(a.end_ms, b.end_ms);
            assert!((a.avg_rms - b.avg_rms).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_header_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("invalid.wav");
        std::fs::write(&file, b"not-a-wav").expect("write invalid data");

        let error = analyze_wav(&file, None).expect_err("invalid wav should fail");
        assert!(
            error.contains("unsupported wav") || error.contains("too small"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn minimal_wav_with_no_samples_is_valid_and_empty() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("minimal.wav");
        write_pcm16_mono_wav(&wav, 16_000, &[]);

        let analysis = analyze_wav(&wav, None).expect("minimal wav should parse");
        assert_eq!(analysis.duration_ms, 0);
        assert_eq!(analysis.frame_count, 0);
        assert!(analysis.active_regions.is_empty());
    }

    #[test]
    fn corrupt_fmt_chunk_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("corrupt_fmt.wav");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&(8u32).to_le_bytes()); // invalid (<16 bytes)
        bytes.extend_from_slice(&[0u8; 8]);
        std::fs::write(&wav, bytes).expect("write corrupt wav");

        let error = analyze_wav(&wav, None).expect_err("corrupt fmt should fail");
        assert!(
            error.contains("invalid wav fmt chunk")
                || error.contains("missing wav")
                || error.contains("unsupported")
                || error.contains("too small"),
            "unexpected corrupt fmt error: {error}"
        );
    }

    #[test]
    fn as_json_contains_expected_fields() {
        use super::{AudioRegion, NativeAudioAnalysis};
        let analysis = NativeAudioAnalysis {
            duration_ms: 5000,
            sample_rate_hz: 16_000,
            frame_ms: 20,
            frame_count: 250,
            avg_rms: 0.05,
            max_rms: 0.3,
            activity_threshold: 0.075,
            active_regions: vec![AudioRegion {
                start_ms: 100,
                end_ms: 500,
                avg_rms: 0.1,
            }],
        };
        let json = analysis.as_json();
        assert_eq!(json["sample_rate_hz"], 16_000);
        assert_eq!(json["duration_ms"], 5000);
        assert_eq!(json["frame_count"], 250);
        assert_eq!(json["active_region_count"], 1);
        let regions = json["active_regions"].as_array().expect("array");
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0]["start_ms"], 100);
        assert_eq!(regions[0]["end_ms"], 500);
    }

    #[test]
    fn bridge_short_gaps_fills_short_preserves_long() {
        use super::bridge_short_gaps;
        // Gap of 2 frames (= GAP_BRIDGE_MAX_FRAMES) between two active runs → bridged.
        let mut active = vec![true, true, false, false, true, true];
        bridge_short_gaps(&mut active, 2);
        assert!(active.iter().all(|&v| v), "gap of 2 should be bridged");

        // Gap of 3 frames (> max 2) → NOT bridged.
        let mut active2 = vec![true, false, false, false, true];
        bridge_short_gaps(&mut active2, 2);
        assert!(!active2[1]);
        assert!(!active2[2]);
        assert!(!active2[3]);

        // Empty input → no panic.
        let mut empty: Vec<bool> = vec![];
        bridge_short_gaps(&mut empty, 2);
        assert!(empty.is_empty());
    }

    #[test]
    fn active_regions_filters_short_regions() {
        use super::active_regions_from_frames;
        // Active run of 1 frame with min_region_frames=2 → filtered out.
        let rms = vec![0.1, 0.0, 0.1, 0.1, 0.0];
        let active = vec![true, false, true, true, false];
        let regions = active_regions_from_frames(&rms, &active, 20, 2, 100);
        // Only the 2-frame run at index 2-3 should survive.
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start_ms, 40);
        assert_eq!(regions[0].end_ms, 80);
    }

    #[test]
    fn compute_frame_rms_max_amplitude() {
        use super::compute_frame_rms;
        // Max amplitude 16-bit: 32767/32768 ≈ 1.0 per sample → RMS ≈ 1.0.
        let samples = vec![i16::MAX; 160]; // one 10ms frame at 16kHz
        let rms = compute_frame_rms(&samples, 160);
        assert_eq!(rms.len(), 1);
        assert!(
            (rms[0] - 1.0).abs() < 0.001,
            "max amplitude RMS ≈ 1.0, got {}",
            rms[0]
        );

        // Silence → RMS 0.
        let silence = vec![0i16; 160];
        let rms_silent = compute_frame_rms(&silence, 160);
        assert!((rms_silent[0] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn duration_override_zero_falls_back_to_computed() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("override_zero.wav");
        let samples = vec![0i16; 16_000]; // 1 second at 16kHz
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        // Some(0) is filtered out by `.filter(|value| *value > 0)`.
        let analysis = analyze_wav(&wav, Some(0)).expect("analysis should succeed");
        assert_eq!(
            analysis.duration_ms, 1_000,
            "Some(0) should fall back to computed duration"
        );
    }

    #[test]
    fn duration_override_is_honored_when_positive() {
        let dir = tempdir().expect("tempdir");
        let wav = dir.path().join("override.wav");
        let samples = vec![0i16; 16_000];
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let analysis = analyze_wav(&wav, Some(2_500)).expect("analysis should succeed");
        assert_eq!(analysis.duration_ms, 2_500);
    }

    // ── Task #217 — native_audio.rs edge-case tests ────────────────

    /// Build a custom WAV file with configurable fmt fields.
    fn custom_wav(
        audio_format: u16,
        channels: u16,
        sample_rate: u32,
        bits_per_sample: u16,
        data_bytes: &[u8],
    ) -> Vec<u8> {
        let data_len = data_bytes.len() as u32;
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");

        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&audio_format.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * (channels as u32) * (bits_per_sample as u32 / 8);
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        let block_align = channels * (bits_per_sample / 8);
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits_per_sample.to_le_bytes());

        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        bytes.extend_from_slice(data_bytes);

        bytes
    }

    #[test]
    fn non_pcm_audio_format_rejected() {
        let dir = tempdir().unwrap();
        let wav_path = dir.path().join("float.wav");
        // audio_format=3 (IEEE float), otherwise valid mono 16-bit.
        let bytes = custom_wav(3, 1, 16_000, 16, &[0u8; 320]);
        std::fs::write(&wav_path, bytes).unwrap();

        let err = analyze_wav(&wav_path, None).unwrap_err();
        let msg = err.clone();
        assert!(
            msg.contains("audio_format=3"),
            "should reject non-PCM format: {msg}"
        );
    }

    #[test]
    fn stereo_and_non_16bit_rejected() {
        let dir = tempdir().unwrap();

        // Stereo file.
        let stereo_path = dir.path().join("stereo.wav");
        let bytes = custom_wav(1, 2, 16_000, 16, &[0u8; 640]);
        std::fs::write(&stereo_path, bytes).unwrap();
        let err = analyze_wav(&stereo_path, None).unwrap_err();
        assert!(err.contains("channels=2"), "should reject stereo: {}", err);

        // 8-bit file.
        let eight_bit_path = dir.path().join("8bit.wav");
        let bytes = custom_wav(1, 1, 16_000, 8, &[0u8; 320]);
        std::fs::write(&eight_bit_path, bytes).unwrap();
        let err = analyze_wav(&eight_bit_path, None).unwrap_err();
        assert!(
            err.contains("bits_per_sample=8"),
            "should reject 8-bit: {}",
            err
        );
    }

    #[test]
    fn missing_data_chunk_rejected() {
        let dir = tempdir().unwrap();
        let wav_path = dir.path().join("no_data.wav");
        // Valid RIFF/WAVE with fmt chunk + unknown filler chunk but no data chunk.
        // Must be ≥ 44 bytes to pass the early size check.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&44u32.to_le_bytes()); // file size - 8
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&32_000u32.to_le_bytes()); // byte_rate
        bytes.extend_from_slice(&2u16.to_le_bytes()); // block_align
        bytes.extend_from_slice(&16u16.to_le_bytes()); // bits
        // Filler unknown chunk to pad to ≥ 44 bytes.
        bytes.extend_from_slice(b"JUNK");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        std::fs::write(&wav_path, bytes).unwrap();

        let err = analyze_wav(&wav_path, None).unwrap_err();
        assert!(
            err.contains("data chunk"),
            "should reject missing data chunk: {}",
            err
        );
    }

    #[test]
    fn bridge_short_gaps_leading_and_trailing_gaps_not_filled() {
        use super::bridge_short_gaps;

        // Leading gap: should NOT be bridged.
        let mut leading = vec![false, false, true, true];
        bridge_short_gaps(&mut leading, 2);
        assert_eq!(leading, vec![false, false, true, true]);

        // Trailing gap: should NOT be bridged.
        let mut trailing = vec![true, true, false, false];
        bridge_short_gaps(&mut trailing, 2);
        assert_eq!(trailing, vec![true, true, false, false]);
    }

    #[test]
    fn compute_frame_rms_partial_last_frame() {
        use super::compute_frame_rms;

        // 5 samples with frame_size=3 → 2 frames: [s0,s1,s2] and [s3,s4].
        // Values normalized by /32768.0 internally.
        let samples: Vec<i16> = vec![16384, 16384, 16384, 32767, 32767];
        let rms = compute_frame_rms(&samples, 3);
        assert_eq!(rms.len(), 2, "should produce 2 frames (one partial)");

        // Frame 1: RMS of [0.5, 0.5, 0.5] = 0.5
        assert!((rms[0] - 0.5).abs() < 0.01, "frame 0 rms={}", rms[0]);
        // Frame 2: RMS of [~1.0, ~1.0] ≈ 1.0 (partial frame, only 2 samples)
        assert!(rms[1] > 0.99, "frame 1 should be ~1.0, got {}", rms[1]);
    }

    // ── Task #222 — native_audio pass 2 edge-case tests ────────────────

    #[test]
    fn zero_sample_rate_is_rejected() {
        let dir = tempdir().unwrap();
        let wav_path = dir.path().join("zero_rate.wav");
        let bytes = custom_wav(1, 1, 0, 16, &[0u8; 4]);
        std::fs::write(&wav_path, bytes).unwrap();

        let err = analyze_wav(&wav_path, None).unwrap_err();
        assert!(
            err.contains("sample_rate 0"),
            "expected sample_rate error, got: {err}"
        );
    }

    #[test]
    fn odd_sized_chunk_word_aligned_parser_continues() {
        let dir = tempdir().unwrap();
        let wav_path = dir.path().join("odd_chunk.wav");

        // RIFF/WAVE + fmt(16) + unknown "INFO" chunk (3 bytes, odd) + pad + data(4)
        let riff_content_size: u32 = 4 + 24 + 12 + 12;

        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&riff_content_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");

        // fmt chunk: PCM, mono, 16kHz, 16-bit
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&32_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());

        // Unknown chunk "INFO" with odd size = 3 bytes + 1 padding byte
        bytes.extend_from_slice(b"INFO");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(b"abc");
        bytes.push(0x00); // word-alignment padding

        // data chunk: 4 bytes = 2 i16 samples
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[0x00, 0x10, 0x00, 0x20]);

        std::fs::write(&wav_path, bytes).unwrap();

        let analysis = analyze_wav(&wav_path, None).expect("should find data past odd chunk");
        assert!(
            analysis.frame_count >= 1,
            "should parse samples from data chunk after odd-sized INFO chunk"
        );
    }

    #[test]
    fn active_regions_from_frames_avg_rms_computed_correctly() {
        use super::active_regions_from_frames;

        let rms = vec![0.2f32, 0.4f32, 0.0f32];
        let active = vec![true, true, false];
        // frame_ms=20, min_region_frames=1 (so 2-frame region qualifies)
        let regions = active_regions_from_frames(&rms, &active, 20, 1, 60);

        assert_eq!(regions.len(), 1);
        // avg_rms = (0.2 + 0.4) / 2 = 0.3
        assert!(
            (regions[0].avg_rms - 0.3).abs() < 1e-6,
            "expected avg_rms ~0.3, got {}",
            regions[0].avg_rms
        );
        assert_eq!(regions[0].start_ms, 0);
        assert_eq!(regions[0].end_ms, 40); // 2 frames * 20ms
    }

    #[test]
    fn duration_override_on_empty_samples_file() {
        let dir = tempdir().unwrap();
        let wav = dir.path().join("empty_override.wav");
        write_pcm16_mono_wav(&wav, 16_000, &[]);

        // Positive override on empty file — used verbatim.
        let analysis = analyze_wav(&wav, Some(3_000)).expect("should succeed");
        assert_eq!(
            analysis.duration_ms, 3_000,
            "positive override on empty file should be honoured"
        );

        // Some(0) on empty file — unwrap_or(0) path.
        let analysis_zero = analyze_wav(&wav, Some(0)).expect("should succeed");
        assert_eq!(
            analysis_zero.duration_ms, 0,
            "Some(0) on empty file should remain 0"
        );
    }

    #[test]
    fn bridge_short_gaps_boundary_at_max_gap_frames() {
        use super::bridge_short_gaps;

        // Gap of exactly 1 (== max) → bridged.
        let mut exactly_max = vec![true, false, true];
        bridge_short_gaps(&mut exactly_max, 1);
        assert_eq!(
            exactly_max,
            vec![true, true, true],
            "gap of 1 with max=1 should be bridged"
        );

        // Gap of 2 (== max + 1) → NOT bridged.
        let mut over_max = vec![true, false, false, true];
        bridge_short_gaps(&mut over_max, 1);
        assert_eq!(
            over_max,
            vec![true, false, false, true],
            "gap of 2 with max=1 should NOT be bridged"
        );
    }

    // ── Task #227 — native_audio pass 3 edge-case tests ────────────────

    #[test]
    fn parse_wav_with_max_chunk_size_does_not_panic() {
        let dir = tempdir().unwrap();
        let wav_path = dir.path().join("maxchunk.wav");

        // Build a WAV where the data chunk has chunk_size = u32::MAX.
        // The parser should clamp via .min(bytes.len()) and not panic or loop.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&100u32.to_le_bytes()); // file size (irrelevant)
        bytes.extend_from_slice(b"WAVE");

        // fmt chunk (16 bytes): PCM, mono, 16kHz, 16-bit
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&16000u32.to_le_bytes()); // sample rate
        bytes.extend_from_slice(&32000u32.to_le_bytes()); // byte rate
        bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

        // data chunk with u32::MAX size but only 4 real bytes
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]); // 2 samples

        std::fs::write(&wav_path, &bytes).unwrap();
        // Should not panic — data is clamped to actual file length.
        let result = analyze_wav(&wav_path, None);
        assert!(result.is_ok(), "max chunk_size must not panic: {result:?}");
    }

    #[test]
    fn odd_byte_data_chunk_trailing_byte_silently_dropped() {
        let dir = tempdir().unwrap();
        let wav_path = dir.path().join("odd_data.wav");

        // 5 bytes of data → chunks_exact(2) produces 2 samples, drops 1 trailing byte.
        let data_bytes: &[u8] = &[0x00, 0x40, 0x00, 0x40, 0xFF]; // 2 valid samples + 1 stray byte
        let bytes = custom_wav(1, 1, 16000, 16, data_bytes);
        std::fs::write(&wav_path, &bytes).unwrap();

        let analysis = analyze_wav(&wav_path, None).expect("should parse without error");
        // 2 samples at 16kHz → 0.000125 sec → 0 ms (rounds to 0), but duration_ms formula:
        // (2.0 / 16000.0) * 1000 = 0.125 → rounds to 0.
        // The key assertion: no panic, only 2 samples worth of data processed.
        assert_eq!(
            analysis.sample_rate_hz, 16000,
            "sample rate should be parsed correctly"
        );
    }

    #[test]
    fn bridge_short_gaps_all_inactive_frames_unchanged() {
        use super::bridge_short_gaps;

        // All false → no mutation, no panic.
        let mut all_inactive = vec![false, false, false, false, false];
        let original = all_inactive.clone();
        bridge_short_gaps(&mut all_inactive, 2);
        assert_eq!(
            all_inactive, original,
            "all-inactive frames should remain unchanged"
        );
    }

    #[test]
    fn active_regions_from_frames_min_region_zero_accepts_all_runs() {
        use super::active_regions_from_frames;

        // With min_region_frames=0, even single-frame active runs should be kept.
        let rms = vec![0.1f32, 0.0, 0.2, 0.0, 0.3];
        let active = vec![true, false, true, false, true];
        let regions = active_regions_from_frames(&rms, &active, 20, 0, 100);
        assert_eq!(
            regions.len(),
            3,
            "min_region_frames=0 should accept all 3 single-frame runs"
        );
        assert_eq!(regions[0].start_ms, 0);
        assert_eq!(regions[0].end_ms, 20);
        assert_eq!(regions[1].start_ms, 40);
        assert_eq!(regions[1].end_ms, 60);
        assert_eq!(regions[2].start_ms, 80);
        assert_eq!(regions[2].end_ms, 100);
    }

    #[test]
    fn compute_frame_rms_empty_samples_returns_empty() {
        use super::compute_frame_rms;

        let rms = compute_frame_rms(&[], 160);
        assert!(
            rms.is_empty(),
            "empty samples should produce no frames, got {} frames",
            rms.len()
        );
    }

    // -- bd-246: native_audio.rs edge-case tests pass 2 --

    #[test]
    fn missing_fmt_chunk_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("no_fmt.wav");
        // Construct a RIFF/WAVE with only a data chunk, no fmt chunk.
        let mut bytes = Vec::new();
        let data_payload = vec![0u8; 100];
        let data_chunk_size = data_payload.len() as u32;
        let file_size = 4 + 8 + data_chunk_size; // "WAVE" + data chunk header + data

        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&file_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_chunk_size.to_le_bytes());
        bytes.extend_from_slice(&data_payload);

        std::fs::write(&wav, bytes).unwrap();
        let result = analyze_wav(&wav, None);
        assert!(result.is_err(), "missing fmt chunk should cause error");
        let err = result.unwrap_err();
        assert!(
            err.contains("missing wav fmt"),
            "error should mention missing fmt, got: {err}"
        );
    }

    #[test]
    fn bridge_short_gaps_all_active_no_mutation() {
        use super::bridge_short_gaps;

        let mut active = vec![true, true, true, true, true];
        let original = active.clone();
        bridge_short_gaps(&mut active, 2);
        assert_eq!(active, original, "all-true input should not be mutated");
    }

    #[test]
    fn as_json_with_zero_active_regions() {
        use super::NativeAudioAnalysis;
        let analysis = NativeAudioAnalysis {
            duration_ms: 1000,
            sample_rate_hz: 16000,
            frame_ms: 20,
            frame_count: 50,
            avg_rms: 0.001,
            max_rms: 0.002,
            activity_threshold: 0.003,
            active_regions: vec![],
        };
        let json = analysis.as_json();
        assert_eq!(json["active_region_count"], 0);
        assert!(json["active_regions"].is_array());
        assert_eq!(json["active_regions"].as_array().unwrap().len(), 0);
        assert_eq!(json["duration_ms"], 1000);
    }

    #[test]
    fn negative_amplitude_samples_rms_is_positive() {
        use super::compute_frame_rms;

        // All negative samples including i16::MIN.
        let samples = vec![i16::MIN, -16384, -8192, -4096];
        let rms = compute_frame_rms(&samples, 4);
        assert_eq!(rms.len(), 1, "one frame expected");
        assert!(rms[0] > 0.0, "RMS should be positive for negative samples");
        // i16::MIN = -32768, normalized = -32768/32768 ≈ -1.0
        // RMS should be fairly large (close to 0.6-0.7 range).
        assert!(
            rms[0] > 0.3,
            "RMS of large negative samples should be substantial, got {}",
            rms[0]
        );
    }

    #[test]
    fn analyze_wav_silence_yields_no_active_regions() {
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("silence.wav");
        // Pure silence: all zero samples, 16kHz, 1 second.
        let samples = vec![0i16; 16_000];
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let analysis = analyze_wav(&wav, None).expect("should parse");
        assert!(
            analysis.active_regions.is_empty(),
            "pure silence should have no active regions, got {}",
            analysis.active_regions.len()
        );
        assert!((analysis.avg_rms - 0.0).abs() < 1e-6);
    }

    // ── Task #260 — native_audio pass 5 edge-case tests ──

    #[test]
    fn analyze_wav_low_sample_rate_exercises_frame_samples_max_guard() {
        // Sample rate of 100 Hz → frame_samples = (100 * 20) / 1000 = 2.
        // Very small frame size, exercises edge of frame computation.
        // All existing tests use 16 kHz (frame_samples = 320).
        let dir = tempdir().unwrap();
        let wav = dir.path().join("low_rate.wav");
        // 100 samples at 100 Hz = 1 second.
        let samples: Vec<i16> = (0..100)
            .map(|i| if i % 2 == 0 { 10_000 } else { -10_000 })
            .collect();
        write_pcm16_mono_wav(&wav, 100, &samples);

        let analysis = analyze_wav(&wav, None).expect("low sample rate should parse");
        assert_eq!(analysis.sample_rate_hz, 100);
        assert_eq!(analysis.duration_ms, 1_000);
        // frame_samples = 2, 100 samples → 50 frames
        assert_eq!(
            analysis.frame_count, 50,
            "100 samples / 2 per frame = 50 frames"
        );
    }

    #[test]
    fn active_regions_from_frames_all_active_single_region() {
        use super::active_regions_from_frames;

        // All frames active → single region spanning the full range.
        // Existing tests always have mixed active/inactive; none test all-active.
        let rms = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let active = vec![true, true, true, true, true];
        let regions = active_regions_from_frames(&rms, &active, 20, 1, 100);
        assert_eq!(regions.len(), 1, "all-active should produce single region");
        assert_eq!(regions[0].start_ms, 0);
        assert_eq!(regions[0].end_ms, 100); // 5 frames * 20ms
        // avg_rms = (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5 = 0.3
        assert!(
            (regions[0].avg_rms - 0.3).abs() < 1e-6,
            "avg_rms should be 0.3, got {}",
            regions[0].avg_rms
        );
    }

    #[test]
    fn analyze_wav_activity_threshold_matches_formula() {
        // Verify that activity_threshold is computed as:
        //   ((avg_rms * 1.5).max(MIN_THRESHOLD)).min((max_rms * 0.8).max(MIN_THRESHOLD))
        // No existing test asserts the specific threshold value.
        let dir = tempdir().unwrap();
        let wav = dir.path().join("threshold.wav");
        // Constant amplitude (not silence) so we get predictable RMS.
        let samples = vec![4096i16; 16_000]; // ~0.125 normalized amplitude, 1 second
        write_pcm16_mono_wav(&wav, 16_000, &samples);

        let analysis = analyze_wav(&wav, None).expect("should parse");
        // All frames have same RMS → avg_rms == max_rms ≈ 0.125.
        let expected_rms = 4096.0 / 32768.0; // 0.125
        assert!(
            (analysis.avg_rms - expected_rms as f32).abs() < 0.01,
            "avg_rms should be ~{}, got {}",
            expected_rms,
            analysis.avg_rms
        );
        // activity_threshold = min(max(avg*1.5, 0.003), max(max_rms*0.8, 0.003))
        //                    = min(max(0.1875, 0.003), max(0.100, 0.003))
        //                    = min(0.1875, 0.100) = 0.100
        let expected_threshold = (analysis.max_rms * 0.8).max(super::MIN_THRESHOLD);
        assert!(
            (analysis.activity_threshold - expected_threshold).abs() < 0.01,
            "activity_threshold should match formula: expected ~{}, got {}",
            expected_threshold,
            analysis.activity_threshold
        );
    }

    #[test]
    fn compute_frame_rms_single_sample_frames() {
        use super::compute_frame_rms;

        // frame_samples=1 → each sample is its own frame.
        // No existing test uses frame_samples=1.
        let samples: Vec<i16> = vec![0, 16384, -16384, 32767];
        let rms = compute_frame_rms(&samples, 1);
        assert_eq!(rms.len(), 4, "4 samples with frame_size=1 → 4 frames");
        // Frame 0: 0/32768 = 0.0 → RMS = 0.0
        assert!(
            (rms[0] - 0.0).abs() < 1e-6,
            "silence sample RMS should be 0"
        );
        // Frame 1: 16384/32768 = 0.5 → RMS = 0.5
        assert!(
            (rms[1] - 0.5).abs() < 0.01,
            "half-amplitude RMS should be ~0.5, got {}",
            rms[1]
        );
        // Frame 2: -16384/32768 = -0.5 → RMS = 0.5 (squared then sqrt)
        assert!(
            (rms[2] - 0.5).abs() < 0.01,
            "negative half-amplitude RMS should be ~0.5"
        );
        // Frame 3: 32767/32768 ≈ 1.0 → RMS ≈ 1.0
        assert!(
            rms[3] > 0.99,
            "max amplitude RMS should be ~1.0, got {}",
            rms[3]
        );
    }

    #[test]
    fn active_regions_clamp_partial_trailing_frame_to_audio_duration() {
        use super::active_regions_from_frames;

        let rms = vec![0.2f32];
        let active = vec![true];
        let regions = active_regions_from_frames(&rms, &active, 20, 1, 10);

        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].start_ms, 0);
        assert_eq!(regions[0].end_ms, 10);
    }

    #[test]
    fn as_json_multiple_active_regions_serialized() {
        use super::{AudioRegion, NativeAudioAnalysis};

        // Existing tests have 0 or 1 region; none test multiple.
        let analysis = NativeAudioAnalysis {
            duration_ms: 10_000,
            sample_rate_hz: 16_000,
            frame_ms: 20,
            frame_count: 500,
            avg_rms: 0.1,
            max_rms: 0.4,
            activity_threshold: 0.05,
            active_regions: vec![
                AudioRegion {
                    start_ms: 0,
                    end_ms: 2000,
                    avg_rms: 0.2,
                },
                AudioRegion {
                    start_ms: 4000,
                    end_ms: 6000,
                    avg_rms: 0.3,
                },
                AudioRegion {
                    start_ms: 8000,
                    end_ms: 10_000,
                    avg_rms: 0.15,
                },
            ],
        };
        let json = analysis.as_json();
        assert_eq!(json["active_region_count"], 3);
        let regions = json["active_regions"].as_array().expect("array");
        assert_eq!(regions.len(), 3);
        assert_eq!(regions[0]["start_ms"], 0);
        assert_eq!(regions[0]["end_ms"], 2000);
        assert_eq!(regions[1]["start_ms"], 4000);
        assert_eq!(regions[2]["start_ms"], 8000);
        assert_eq!(regions[2]["end_ms"], 10_000);
    }
}
