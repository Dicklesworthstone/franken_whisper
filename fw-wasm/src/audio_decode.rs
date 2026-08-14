//! In-memory audio decode for the browser: bytes (mp3/m4a/wav) → 16 kHz mono
//! f32, via symphonia (pure Rust, wasm-clean).
//!
//! STREAMING by construction: each decoded packet is downmixed to mono and
//! fed through an incremental linear resampler immediately, so the only
//! buffer that grows with duration is the 16 kHz mono OUTPUT (~230 MB/hour).
//! The first cut accumulated the full interleaved source stream (~1.3
//! GB/hour for 44.1 kHz stereo), which pushed a tab holding 2.4 GB of model
//! weights past the 4 GB wasm ceiling — the allocator aborts with an opaque
//! `unreachable`, which is exactly the trap doctrine says to never ship.
//!
//! The resampler emits bit-identical output to the parent's whole-buffer
//! `audio::resample_mono_linear` reference: the same `idx * src/dst` f64
//! position, the same `floor`/f32-frac interpolation, the same end clamps —
//! only the availability of source samples is chunked, and an output index
//! is emitted only once its right neighbor exists (or the stream has ended,
//! which is when the reference clamps too).
//!
//! The parent's `src/audio.rs` covers file-based decode for the CLI; the
//! downmix here mirrors its reference semantics (mean over channels).

// Also mounted by fw-ios (the phone app's FFI crate), which needs the same
// streaming bytes→PCM decode for imported audio files.
#![cfg(any(target_arch = "wasm32", target_os = "ios"))]

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

/// Hard duration cap for the browser build, enforced on DECODED samples with
/// a named error (never an allocator abort). Two hours of 16 kHz mono f32 is
/// ~460 MB next to ~2.4 GB of resident model weights.
pub const MAX_OUTPUT_SAMPLES: usize = 2 * 60 * 60 * TARGET_RATE as usize;

/// Incremental twin of `audio::resample_mono_linear` (bit-identical output;
/// see module docs). Feed mono source chunks with [`Self::push`], then call
/// [`Self::finish`] once for the clamped tail.
struct StreamingResampler {
    ratio: f64,
    dst_rate: u32,
    src_rate: u32,
    /// Next output index to emit (global, so positions match the reference).
    next_out: usize,
    /// Total source samples pushed so far.
    src_len: usize,
    /// Retained source tail; `tail[0]` is global source index `tail_start`.
    tail: Vec<f32>,
    tail_start: usize,
    out: Vec<f32>,
}

impl StreamingResampler {
    fn new(src_rate: u32, dst_rate: u32) -> Self {
        Self {
            ratio: f64::from(src_rate) / f64::from(dst_rate),
            dst_rate,
            src_rate,
            next_out: 0,
            src_len: 0,
            tail: Vec::new(),
            tail_start: 0,
            out: Vec::new(),
        }
    }

    fn src(&self, global: usize) -> f32 {
        self.tail[global - self.tail_start]
    }

    /// Emit every output whose interpolation window is fully available:
    /// `left + 1 <= src_len - 1`, i.e. the right neighbor exists. End-of-file
    /// clamping belongs to [`Self::finish`], exactly like the reference.
    fn emit_ready(&mut self) -> FwResult<()> {
        if self.src_len < 2 {
            return Ok(());
        }
        let last_full = self.src_len - 1;
        loop {
            let src_pos = self.next_out as f64 * self.ratio;
            let left = src_pos.floor() as usize;
            if left + 1 > last_full {
                break;
            }
            let frac = (src_pos - src_pos.floor()) as f32;
            let l = self.src(left);
            let r = self.src(left + 1);
            self.out.push(l + (r - l) * frac);
            if self.out.len() > MAX_OUTPUT_SAMPLES {
                return Err(FwError::InvalidRequest(format!(
                    "audio exceeds the browser build's {}-hour cap; trim the file or use the CLI",
                    MAX_OUTPUT_SAMPLES / (3600 * TARGET_RATE as usize),
                )));
            }
            self.next_out += 1;
        }
        // Drop the consumed prefix; keep from the next output's left index.
        let keep_from = (self.next_out as f64 * self.ratio).floor() as usize;
        let keep_from = keep_from.min(self.src_len.saturating_sub(1));
        if keep_from > self.tail_start {
            self.tail.drain(..keep_from - self.tail_start);
            self.tail_start = keep_from;
        }
        Ok(())
    }

    fn push(&mut self, mono: &[f32]) -> FwResult<()> {
        self.tail.extend_from_slice(mono);
        self.src_len += mono.len();
        self.emit_ready()
    }

    /// Clamped tail + exact output length, mirroring the reference: output
    /// length is `ceil(src_len * dst/src)` (min 1 for non-empty input), and
    /// past-the-end interpolation clamps both neighbors to the last sample.
    fn finish(mut self) -> FwResult<Vec<f32>> {
        if self.src_len == 0 {
            return Ok(Vec::new());
        }
        // Equal rates need no special case: ratio 1.0 already emitted
        // idx→idx copies in `push`; only the final sample (which has no
        // right neighbor until EOF) lands here — same values as the
        // reference's `to_vec` short-circuit.
        let output_len = (((self.src_len as f64) * f64::from(self.dst_rate))
            / f64::from(self.src_rate))
        .ceil() as usize;
        let total = output_len.max(1);
        if total > MAX_OUTPUT_SAMPLES {
            return Err(FwError::InvalidRequest(format!(
                "audio exceeds the browser build's {}-hour cap; trim the file or use the CLI",
                MAX_OUTPUT_SAMPLES / (3600 * TARGET_RATE as usize),
            )));
        }
        // Every remaining output's `left` is >= the retained tail's start:
        // `emit_ready` keeps the tail from `floor(next_out * ratio)` onward,
        // which is exactly the smallest index `finish` can touch.
        let last = self.src_len - 1;
        while self.next_out < total {
            let src_pos = self.next_out as f64 * self.ratio;
            let left = (src_pos.floor() as usize).min(last);
            let right = (left + 1).min(last);
            let frac = (src_pos - src_pos.floor()) as f32;
            let l = self.src(left);
            let r = self.src(right);
            self.out.push(l + (r - l) * frac);
            self.next_out += 1;
        }
        Ok(self.out)
    }
}

/// Decode an audio container held in memory to 16 kHz mono f32 samples,
/// streaming packet-by-packet (see module docs). `ext` is a lowercase
/// extension hint from the picked file's name; empty lets symphonia probe.
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

    let mut resampler: Option<StreamingResampler> = None;
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    let mut mono_chunk: Vec<f32> = Vec::new();
    let mut src_rate_seen: u32 = 0;
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
                let channels = spec.channels.count();
                if channels == 0 || spec.rate == 0 {
                    continue;
                }
                if src_rate_seen == 0 {
                    src_rate_seen = spec.rate;
                    resampler = Some(StreamingResampler::new(spec.rate, TARGET_RATE));
                } else if spec.rate != src_rate_seen {
                    return Err(FwError::InvalidRequest(
                        "mid-stream sample-rate changes are not supported".to_string(),
                    ));
                }
                let buf = sample_buf.get_or_insert_with(|| {
                    SampleBuffer::<f32>::new(audio_buf.capacity() as u64, spec)
                });
                if buf.capacity() < audio_buf.capacity() * channels {
                    *buf = SampleBuffer::<f32>::new(audio_buf.capacity() as u64, spec);
                }
                buf.copy_interleaved_ref(audio_buf);
                let interleaved = buf.samples();
                // Downmix THIS PACKET to mono (mean over channels — the
                // parent `audio::downmix_to_mono` reference semantics).
                let frames = interleaved.len() / channels;
                mono_chunk.clear();
                mono_chunk.reserve(frames);
                if channels == 1 {
                    mono_chunk.extend_from_slice(interleaved);
                } else if channels == 2 {
                    for f in 0..frames {
                        mono_chunk.push((interleaved[2 * f] + interleaved[2 * f + 1]) * 0.5);
                    }
                } else {
                    for f in 0..frames {
                        let frame = &interleaved[f * channels..(f + 1) * channels];
                        let sum: f32 = frame.iter().copied().sum();
                        mono_chunk.push(sum / channels as f32);
                    }
                }
                if let Some(rs) = resampler.as_mut() {
                    rs.push(&mono_chunk)?;
                }
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
    let Some(rs) = resampler else {
        return Err(FwError::InvalidRequest(
            "decoded zero audio samples".to_string(),
        ));
    };
    let out = rs.finish()?;
    if out.is_empty() {
        return Err(FwError::InvalidRequest(
            "decoded zero audio samples".to_string(),
        ));
    }
    Ok(out)
}

/// Leading-silence detection (bd-m2jm): whisper hallucinates on silence (a
/// silent first window free-runs the decoder, often in a random language), so
/// the playground trims leading silence BEFORE the model hears anything and
/// adds the trimmed offset back to every emitted timestamp.
///
/// Energy-based and conservative: 100 ms RMS frames; a frame counts as sound
/// above max(-60 dBFS, 2% of the loudest frame); half a second of pre-roll is
/// kept before the first sound so onsets are never clipped; nothing is
/// trimmed unless there is more than a second to skip.
pub fn leading_silence_samples(samples: &[f32]) -> usize {
    const FRAME: usize = 1_600; // 100 ms at 16 kHz
    const PRE_ROLL: usize = 8_000; // 0.5 s
    const MIN_TRIM: usize = 16_000; // only bother past 1 s
    if samples.len() < 2 * FRAME {
        return 0;
    }
    let mut rms = Vec::with_capacity(samples.len() / FRAME);
    for frame in samples.chunks(FRAME) {
        let energy: f64 = frame.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        rms.push((energy / frame.len() as f64).sqrt());
    }
    let peak = rms.iter().copied().fold(0.0f64, f64::max);
    let threshold = (peak * 0.02).max(1e-3); // 1e-3 ≈ -60 dBFS
    let first_sound = match rms.iter().position(|&v| v >= threshold) {
        Some(i) => i,
        None => return 0, // all silence: trim nothing, let the engine see it
    };
    let start = (first_sound * FRAME).saturating_sub(PRE_ROLL);
    if start < MIN_TRIM { 0 } else { start }
}
