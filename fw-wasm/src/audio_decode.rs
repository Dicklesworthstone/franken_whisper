//! In-memory audio decode for the browser: bytes (mp3/m4a/wav) → 16 kHz mono
//! f32, via symphonia (pure Rust, wasm-clean).
//!
//! The parent crate's `src/audio.rs` covers this for FILES (plus ffmpeg
//! fallbacks and temp-dir plumbing the browser can't use), so the two pure
//! transforms are mirrored here against their canonical, test-guarded
//! reference semantics:
//! - downmix: mean over channels per frame (`audio::downmix_to_mono`),
//! - resample: linear interpolation at `idx * src/dst`, clamp-on-load
//!   (`audio::resample_mono_linear`'s bit-exact scalar reference).

#![cfg(target_arch = "wasm32")]

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

use crate::error::{FwError, FwResult};

pub const TARGET_RATE: u32 = 16_000;

/// Decode an audio container held in memory to 16 kHz mono f32 samples.
/// `ext` is a lowercase extension hint from the picked file's name ("mp3",
/// "m4a", "wav", ...); empty lets symphonia probe blind.
pub fn decode_to_16k_mono(bytes: Vec<u8>, ext: &str) -> FwResult<Vec<f32>> {
    let mss = MediaSourceStream::new(
        Box::new(std::io::Cursor::new(bytes)),
        symphonia::core::io::MediaSourceStreamOptions::default(),
    );
    let mut hint = Hint::new();
    if !ext.is_empty() {
        hint.with_extension(ext);
    }
    let probed = get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| FwError::InvalidRequest(format!("audio probe failed: {e}")))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| FwError::InvalidRequest("no decodable audio track".to_string()))?;
    let track_id = track.id;
    let mut decoder = get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| FwError::InvalidRequest(format!("unsupported codec: {e}")))?;

    let mut interleaved: Vec<f32> = Vec::new();
    let mut channels: usize = 0;
    let mut src_rate: u32 = 0;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // End of stream: symphonia signals it as an IoError with
            // UnexpectedEof (or ResetRequired on some containers).
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => {
                return Err(FwError::InvalidRequest(format!("demux failed: {e}")));
            }
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let spec = *audio_buf.spec();
                channels = spec.channels.count();
                src_rate = spec.rate;
                let mut sample_buf = SampleBuffer::<f32>::new(audio_buf.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(audio_buf);
                interleaved.extend_from_slice(sample_buf.samples());
            }
            // Recoverable per-packet decode errors: skip the packet, keep the
            // stream (matches the parent decoder's tolerance for damaged
            // frames in otherwise-valid files).
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => {
                return Err(FwError::InvalidRequest(format!("decode failed: {e}")));
            }
        }
    }
    if interleaved.is_empty() || channels == 0 || src_rate == 0 {
        return Err(FwError::InvalidRequest(
            "decoded zero audio samples".to_string(),
        ));
    }

    let mono = downmix_to_mono(&interleaved, channels);
    Ok(resample_mono_linear(&mono, src_rate, TARGET_RATE))
}

/// Mirror of `audio::downmix_to_mono`'s reference semantics: mean over
/// channels per frame.
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let frames = interleaved.len() / channels;
    let mut out = vec![0.0f32; frames];
    if channels == 2 {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = (interleaved[2 * i] + interleaved[2 * i + 1]) * 0.5;
        }
        return out;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let frame = &interleaved[i * channels..(i + 1) * channels];
        let sum: f32 = frame.iter().copied().sum();
        *slot = sum / channels as f32;
    }
    out
}

/// Mirror of `audio::resample_mono_linear`'s bit-exact scalar reference:
/// linear interpolation at f64 position `idx * src/dst`, indices clamped.
fn resample_mono_linear(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == 0 || dst_rate == 0 {
        return Vec::new();
    }
    if src_rate == dst_rate {
        return input.to_vec();
    }
    let ratio = f64::from(src_rate) / f64::from(dst_rate);
    let output_len =
        (((input.len() as f64) * f64::from(dst_rate)) / f64::from(src_rate)).ceil() as usize;
    let total = output_len.max(1);
    let last = input.len() - 1;
    let mut output = vec![0.0f32; total];
    for (idx, slot) in output.iter_mut().enumerate() {
        let src_pos = idx as f64 * ratio;
        let left_idx = (src_pos.floor() as usize).min(last);
        let right_idx = (left_idx + 1).min(last);
        let frac = (src_pos - src_pos.floor()) as f32;
        let left = input[left_idx];
        let right = input[right_idx];
        *slot = left + (right - left) * frac;
    }
    output
}
