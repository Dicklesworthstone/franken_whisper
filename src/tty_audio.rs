use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::STANDARD_NO_PAD;
use flate2::Compression;
use flate2::bufread::ZlibDecoder;
use flate2::write::ZlibEncoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::audio;
use crate::error::{FwError, FwResult};
use crate::process::run_command_with_timeout;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtyAudioFrame {
    #[serde(default = "default_protocol_version")]
    pub protocol_version: u32,
    pub seq: u64,
    pub codec: String,
    pub sample_rate_hz: u32,
    pub channels: u8,
    pub payload_b64: String,
    /// CRC32 of the raw (pre-compression) audio bytes for integrity validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crc32: Option<u32>,
    /// SHA-256 of the raw (pre-compression) audio bytes for stronger integrity checks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_sha256: Option<String>,
}

const MIN_PROTOCOL_VERSION: u32 = 1;
/// Protocol version that adds transcript correction frames.
pub const TRANSCRIPT_PROTOCOL_VERSION: u32 = 2;
// Audio frames remain protocol version 1; transcript control frames are negotiated separately.
const SUPPORTED_PROTOCOL_VERSION: u32 = MIN_PROTOCOL_VERSION;
const DEFAULT_PROTOCOL_VERSION: u32 = MIN_PROTOCOL_VERSION;
const CODEC_MULAW_ZLIB_B64: &str = "mulaw+zlib+b64";

/// Wire-efficient transcript segment for TTY protocol.
/// Field names are deliberately short to minimize bandwidth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegmentCompact {
    /// Start time in seconds.
    pub s: Option<f64>,
    /// End time in seconds.
    pub e: Option<f64>,
    /// Transcribed text.
    pub t: String,
    /// Speaker identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sp: Option<String>,
    /// Confidence score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub c: Option<f64>,
}

impl From<&crate::model::TranscriptionSegment> for TranscriptSegmentCompact {
    fn from(seg: &crate::model::TranscriptionSegment) -> Self {
        Self {
            s: seg.start_sec,
            e: seg.end_sec,
            t: seg.text.clone(),
            sp: seg.speaker.clone(),
            c: seg.confidence,
        }
    }
}

/// Control frame for protocol negotiation and retransmit signaling.
///
/// Control frames share the same NDJSON transport but are distinguished by a
/// `frame_type` field.  The v2 robust-mode spec uses these for:
/// - **handshake**: Version negotiation at stream start
/// - **retransmit_request**: Decoder requests specific missing sequences
/// - **retransmit_response**: Encoder re-sends requested frames
/// - **backpressure**: Decoder signals flow control
/// - **ack**: Decoder acknowledges receipt up to a sequence number
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "frame_type")]
pub enum TtyControlFrame {
    #[serde(rename = "handshake")]
    Handshake {
        min_version: u32,
        max_version: u32,
        supported_codecs: Vec<String>,
    },
    #[serde(rename = "handshake_ack")]
    HandshakeAck {
        negotiated_version: u32,
        negotiated_codec: String,
    },
    #[serde(rename = "retransmit_request")]
    RetransmitRequest { sequences: Vec<u64> },
    #[serde(rename = "retransmit_response")]
    RetransmitResponse { sequences: Vec<u64> },
    #[serde(rename = "ack")]
    Ack {
        /// Highest contiguous sequence number received.
        up_to_seq: u64,
    },
    #[serde(rename = "backpressure")]
    Backpressure {
        /// Decoder's remaining buffer capacity in frames.
        remaining_capacity: u64,
    },
    /// Speculative transcript partial from the fast model.
    #[serde(rename = "transcript_partial")]
    TranscriptPartial {
        seq: u64,
        window_id: u64,
        segments: Vec<TranscriptSegmentCompact>,
        model_id: String,
        speculative: bool,
    },
    /// Retraction of a previous speculative transcript.
    #[serde(rename = "transcript_retract")]
    TranscriptRetract {
        retracted_seq: u64,
        window_id: u64,
        reason: String,
    },
    /// Correction replacing a retracted transcript.
    #[serde(rename = "transcript_correct")]
    TranscriptCorrect {
        correction_id: u64,
        replaces_seq: u64,
        window_id: u64,
        segments: Vec<TranscriptSegmentCompact>,
        model_id: String,
        drift_wer: f64,
    },
    /// Session close with a reason code.
    #[serde(rename = "session_close")]
    SessionClose {
        reason: SessionCloseReason,
        /// The highest data-frame sequence number that was sent before close.
        last_data_seq: Option<u64>,
    },
}

/// Negotiate protocol version from a handshake frame.
///
/// Returns `Ok(version)` if a compatible version exists within both parties'
/// supported ranges, `Err` otherwise.
pub fn negotiate_version(
    local_min: u32,
    local_max: u32,
    remote_min: u32,
    remote_max: u32,
) -> Result<u32, String> {
    let effective_min = local_min.max(remote_min);
    let effective_max = local_max.min(remote_max);
    if effective_min > effective_max {
        return Err(format!(
            "no compatible protocol version: local [{local_min}, {local_max}] vs remote [{remote_min}, {remote_max}]"
        ));
    }
    Ok(effective_max)
}

/// Build retransmit candidates from decode report gaps.
pub fn retransmit_candidates(report: &DecodeReport) -> Vec<u64> {
    let mut missing = Vec::new();
    for gap in &report.gaps {
        for seq in gap.expected..gap.got {
            missing.push(seq);
        }
    }
    missing
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetransmitRange {
    pub start_seq: u64,
    pub end_seq: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetransmitPlan {
    pub protocol_version: u32,
    pub requested_sequences: Vec<u64>,
    pub requested_ranges: Vec<RetransmitRange>,
    pub gap_count: usize,
    pub integrity_failure_count: usize,
    pub dropped_frame_count: usize,
}

#[derive(Debug, Clone)]
enum FrameLine {
    Audio(TtyAudioFrame),
    Control(TtyControlFrame),
}

fn default_protocol_version() -> u32 {
    DEFAULT_PROTOCOL_VERSION
}

#[derive(Debug, Clone)]
pub struct DecodeReport {
    pub protocol_version: u32,
    pub frames_decoded: u64,
    pub gaps: Vec<SequenceGap>,
    pub duplicates: Vec<u64>,
    pub integrity_failures: Vec<u64>,
    pub dropped_frames: Vec<u64>,
    /// Highest contiguous successfully decoded data-frame sequence starting
    /// from seq 0. This is the correct value to surface in ACK frames.
    pub highest_contiguous_seq: Option<u64>,
    pub recovery_policy: DecodeRecoveryPolicy,
}

#[derive(Debug, Clone)]
pub struct SequenceGap {
    pub expected: u64,
    pub got: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeRecoveryPolicy {
    FailClosed,
    SkipMissing,
}

pub fn encode_to_stdout(input_audio: &Path, chunk_ms: u32) -> FwResult<()> {
    let mut stdout = std::io::stdout().lock();
    encode_to_writer(input_audio, chunk_ms, &mut stdout)
}

pub fn emit_control_frame_to_stdout(frame: &TtyControlFrame) -> FwResult<()> {
    let mut stdout = std::io::stdout().lock();
    emit_control_frame_to_writer(&mut stdout, frame)
}

pub fn emit_control_frame_to_writer<W: Write>(
    writer: &mut W,
    frame: &TtyControlFrame,
) -> FwResult<()> {
    write_control_frame(writer, frame)
}

pub fn encode_to_writer<W: Write>(
    input_audio: &Path,
    chunk_ms: u32,
    writer: &mut W,
) -> FwResult<()> {
    let temp_dir = tempfile::tempdir()?;
    let ulaw_path = temp_dir.path().join("tty_audio.ulaw");
    transcode_to_mulaw(input_audio, &ulaw_path)?;

    let bytes = fs::read(&ulaw_path)?;
    let chunk_size = mulaw_chunk_size(chunk_ms);

    let handshake = TtyControlFrame::Handshake {
        min_version: MIN_PROTOCOL_VERSION,
        max_version: SUPPORTED_PROTOCOL_VERSION,
        supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned()],
    };
    write_control_frame(writer, &handshake)?;

    for (index, chunk) in bytes.chunks(chunk_size).enumerate() {
        let crc = crc32_of(chunk);
        let compressed = compress_chunk(chunk)?;
        let frame = TtyAudioFrame {
            protocol_version: SUPPORTED_PROTOCOL_VERSION,
            seq: index as u64,
            codec: CODEC_MULAW_ZLIB_B64.to_owned(),
            sample_rate_hz: 8_000,
            channels: 1,
            payload_b64: STANDARD_NO_PAD.encode(compressed),
            crc32: Some(crc),
            payload_sha256: Some(sha256_hex(chunk)),
        };
        writeln!(writer, "{}", serde_json::to_string(&frame)?)?;
    }

    Ok(())
}

pub fn decode_from_stdin_to_wav(output_wav: &Path) -> FwResult<()> {
    decode_from_stdin_to_wav_with_policy(output_wav, DecodeRecoveryPolicy::FailClosed)
}

pub fn decode_from_stdin_to_wav_with_policy(
    output_wav: &Path,
    policy: DecodeRecoveryPolicy,
) -> FwResult<()> {
    let stdin = std::io::stdin();
    let mut locked = stdin.lock();
    decode_from_reader_to_wav_with_policy(&mut locked, output_wav, policy)
}

pub fn decode_from_reader_to_wav<R: Read>(reader: &mut R, output_wav: &Path) -> FwResult<()> {
    decode_from_reader_to_wav_with_policy(reader, output_wav, DecodeRecoveryPolicy::FailClosed)
}

pub fn decode_from_reader_to_wav_with_policy<R: Read>(
    reader: &mut R,
    output_wav: &Path,
    policy: DecodeRecoveryPolicy,
) -> FwResult<()> {
    let (_report, raw_bytes) = decode_frames_to_raw_with_policy(reader, policy)?;

    let temp_dir = tempfile::tempdir()?;
    let raw_ulaw = temp_dir.path().join("decoded.ulaw");
    fs::write(&raw_ulaw, raw_bytes)?;

    transcode_mulaw_to_wav(&raw_ulaw, output_wav)
}

pub fn decode_frames_to_raw<R: Read>(reader: &mut R) -> FwResult<(DecodeReport, Vec<u8>)> {
    decode_frames_to_raw_with_policy(reader, DecodeRecoveryPolicy::FailClosed)
}

pub fn decode_frames_to_raw_with_policy<R: Read>(
    reader: &mut R,
    policy: DecodeRecoveryPolicy,
) -> FwResult<(DecodeReport, Vec<u8>)> {
    let mut frames = parse_audio_frames_for_decode(reader)?;
    if frames.is_empty() {
        return Err(FwError::InvalidRequest(
            "no tty-audio frames were provided on stdin".to_owned(),
        ));
    }
    let protocol_version = frames
        .first()
        .map(|frame| frame.protocol_version)
        .unwrap_or(DEFAULT_PROTOCOL_VERSION);

    // Sort frames by sequence number so that retransmitted and out-of-order frames
    // are placed in their correct chronological position before decoding.
    frames.sort_by_key(|f| f.seq);

    let mut gaps = Vec::new();
    let mut duplicates = Vec::new();
    let mut integrity_failures = Vec::new();
    let mut dropped_frames = Vec::new();
    let mut raw = Vec::new();
    let mut seen = HashSet::new();
    let mut frames_decoded = 0u64;
    let mut highest_contiguous_seq = None;
    let mut contiguous_prefix_intact = true;

    let mut expected_seq = 0u64;

    for frame in &frames {
        validate_audio_frame_metadata(frame)?;

        if seen.contains(&frame.seq) {
            duplicates.push(frame.seq);
            dropped_frames.push(frame.seq);
            if policy == DecodeRecoveryPolicy::FailClosed {
                return Err(FwError::InvalidRequest(format!(
                    "duplicate tty-audio frame sequence {}",
                    frame.seq
                )));
            }
            continue;
        }

        // Gap detection.
        if frame.seq > expected_seq {
            gaps.push(SequenceGap {
                expected: expected_seq,
                got: frame.seq,
            });
            if policy == DecodeRecoveryPolicy::FailClosed {
                return Err(FwError::InvalidRequest(format!(
                    "missing tty-audio frame sequence: expected {}, got {}",
                    expected_seq, frame.seq
                )));
            }
            contiguous_prefix_intact = false;
            expected_seq = frame.seq;
        }
        if frame.seq < expected_seq {
            duplicates.push(frame.seq);
            dropped_frames.push(frame.seq);
            if policy == DecodeRecoveryPolicy::FailClosed {
                return Err(FwError::InvalidRequest(format!(
                    "out-of-order tty-audio frame: expected {}, got {}",
                    expected_seq, frame.seq
                )));
            }
            continue;
        }

        let compressed = STANDARD_NO_PAD
            .decode(&frame.payload_b64)
            .map_err(|error| FwError::InvalidRequest(format!("invalid base64 payload: {error}")));
        let compressed = match compressed {
            Ok(value) => value,
            Err(error) => {
                integrity_failures.push(frame.seq);
                dropped_frames.push(frame.seq);
                if policy == DecodeRecoveryPolicy::FailClosed {
                    return Err(error);
                }
                contiguous_prefix_intact = false;
                expected_seq = frame.seq + 1;
                seen.insert(frame.seq);
                continue;
            }
        };
        let decoded = match decompress_chunk(&compressed) {
            Ok(value) => value,
            Err(error) => {
                integrity_failures.push(frame.seq);
                dropped_frames.push(frame.seq);
                if policy == DecodeRecoveryPolicy::FailClosed {
                    return Err(error);
                }
                contiguous_prefix_intact = false;
                expected_seq = frame.seq + 1;
                seen.insert(frame.seq);
                continue;
            }
        };

        // CRC32 integrity check.
        if let Some(expected_crc) = frame.crc32 {
            let actual_crc = crc32_of(&decoded);
            if actual_crc != expected_crc {
                integrity_failures.push(frame.seq);
                dropped_frames.push(frame.seq);
                if policy == DecodeRecoveryPolicy::FailClosed {
                    return Err(FwError::InvalidRequest(format!(
                        "tty-audio CRC mismatch at seq {}: expected {}, got {}",
                        frame.seq, expected_crc, actual_crc
                    )));
                }
                contiguous_prefix_intact = false;
                expected_seq = frame.seq + 1;
                seen.insert(frame.seq);
                continue;
            }
        }
        if let Some(expected_sha) = frame.payload_sha256.as_deref() {
            let actual_sha = sha256_hex(&decoded);
            if !actual_sha.eq_ignore_ascii_case(expected_sha) {
                integrity_failures.push(frame.seq);
                dropped_frames.push(frame.seq);
                if policy == DecodeRecoveryPolicy::FailClosed {
                    return Err(FwError::InvalidRequest(format!(
                        "tty-audio SHA-256 mismatch at seq {}: expected {}, got {}",
                        frame.seq, expected_sha, actual_sha
                    )));
                }
                contiguous_prefix_intact = false;
                expected_seq = frame.seq + 1;
                seen.insert(frame.seq);
                continue;
            }
        }

        raw.extend_from_slice(&decoded);
        seen.insert(frame.seq);
        frames_decoded += 1;
        if contiguous_prefix_intact {
            highest_contiguous_seq = Some(frame.seq);
        }
        expected_seq = frame.seq + 1;
    }

    let report = DecodeReport {
        protocol_version,
        frames_decoded,
        gaps,
        duplicates,
        integrity_failures,
        dropped_frames,
        highest_contiguous_seq,
        recovery_policy: policy,
    };

    Ok((report, raw))
}

fn validate_audio_frame_metadata(frame: &TtyAudioFrame) -> FwResult<()> {
    if !(MIN_PROTOCOL_VERSION..=SUPPORTED_PROTOCOL_VERSION).contains(&frame.protocol_version) {
        return Err(FwError::InvalidRequest(format!(
            "unsupported tty-audio protocol_version {} at seq {} (supported: {}-{})",
            frame.protocol_version, frame.seq, MIN_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSION
        )));
    }
    if frame.codec != CODEC_MULAW_ZLIB_B64 {
        return Err(FwError::InvalidRequest(format!(
            "unsupported tty-audio codec `{}` at seq {}",
            frame.codec, frame.seq
        )));
    }
    if frame.sample_rate_hz != 8_000 || frame.channels != 1 {
        return Err(FwError::InvalidRequest(format!(
            "unsupported tty-audio shape at seq {}: sample_rate_hz={}, channels={}",
            frame.seq, frame.sample_rate_hz, frame.channels
        )));
    }
    Ok(())
}

pub fn retransmit_plan_from_stdin(policy: DecodeRecoveryPolicy) -> FwResult<RetransmitPlan> {
    let stdin = std::io::stdin();
    let mut locked = stdin.lock();
    retransmit_plan_from_reader(&mut locked, policy)
}

pub fn emit_retransmit_loop_from_stdin(policy: DecodeRecoveryPolicy, rounds: u32) -> FwResult<()> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut stdout = std::io::stdout().lock();
    emit_retransmit_loop_from_reader(&mut reader, policy, rounds, &mut stdout)
}

pub fn emit_retransmit_loop_from_reader<R: Read, W: Write>(
    reader: &mut R,
    policy: DecodeRecoveryPolicy,
    rounds: u32,
    writer: &mut W,
) -> FwResult<()> {
    let (report, _raw) = decode_frames_to_raw_with_policy(reader, policy)?;
    let plan = retransmit_plan_from_report(&report);
    if plan.requested_sequences.is_empty() {
        return emit_control_frame_to_writer(
            writer,
            &TtyControlFrame::Ack {
                up_to_seq: report.highest_contiguous_seq.unwrap_or(0),
            },
        );
    }

    let bounded_rounds = rounds.max(1);
    for _round in 0..bounded_rounds {
        emit_control_frame_to_writer(
            writer,
            &TtyControlFrame::RetransmitRequest {
                sequences: plan.requested_sequences.clone(),
            },
        )?;
    }

    emit_control_frame_to_writer(
        writer,
        &TtyControlFrame::RetransmitResponse {
            sequences: plan.requested_sequences,
        },
    )
}

pub fn retransmit_plan_from_reader<R: Read>(
    reader: &mut R,
    policy: DecodeRecoveryPolicy,
) -> FwResult<RetransmitPlan> {
    let (report, _raw) = decode_frames_to_raw_with_policy(reader, policy)?;
    Ok(retransmit_plan_from_report(&report))
}

#[must_use]
pub fn retransmit_plan_from_report(report: &DecodeReport) -> RetransmitPlan {
    let mut requested_sequences = retransmit_candidates(report);
    requested_sequences.extend(report.integrity_failures.iter().copied());
    requested_sequences.sort_unstable();
    requested_sequences.dedup();

    let requested_ranges = collapse_sequences_to_ranges(&requested_sequences);

    RetransmitPlan {
        protocol_version: report.protocol_version,
        requested_sequences,
        requested_ranges,
        gap_count: report.gaps.len(),
        integrity_failure_count: report.integrity_failures.len(),
        dropped_frame_count: report.dropped_frames.len(),
    }
}

fn collapse_sequences_to_ranges(sequences: &[u64]) -> Vec<RetransmitRange> {
    if sequences.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut start = sequences[0];
    let mut end = sequences[0];

    for seq in sequences.iter().copied().skip(1) {
        if seq == end.saturating_add(1) {
            end = seq;
            continue;
        }
        ranges.push(RetransmitRange {
            start_seq: start,
            end_seq: end,
        });
        start = seq;
        end = seq;
    }

    ranges.push(RetransmitRange {
        start_seq: start,
        end_seq: end,
    });

    ranges
}

fn parse_audio_frames_for_decode<R: Read>(reader: &mut R) -> FwResult<Vec<TtyAudioFrame>> {
    let entries = parse_frame_lines(reader)?;
    let mut frames = Vec::new();
    let mut handshake_seen = false;
    let mut audio_started = false;
    let mut negotiated_version = SUPPORTED_PROTOCOL_VERSION;
    let mut legacy_protocol_version: Option<u32> = None;
    let mut session_close_reason: Option<SessionCloseReason> = None;
    let mut session_close_last_seq: Option<u64> = None;

    for entry in entries {
        match entry {
            FrameLine::Audio(frame) => {
                if handshake_seen {
                    if frame.protocol_version != negotiated_version {
                        return Err(FwError::InvalidRequest(format!(
                            "frame protocol_version {} does not match negotiated version {} at seq {}",
                            frame.protocol_version, negotiated_version, frame.seq
                        )));
                    }
                } else if let Some(legacy_version) = legacy_protocol_version {
                    if frame.protocol_version != legacy_version {
                        return Err(FwError::InvalidRequest(format!(
                            "frame protocol_version {} does not match legacy stream version {} at seq {}",
                            frame.protocol_version, legacy_version, frame.seq
                        )));
                    }
                } else {
                    legacy_protocol_version = Some(frame.protocol_version);
                }
                audio_started = true;
                frames.push(frame);
            }
            FrameLine::Control(control) => match control {
                TtyControlFrame::Handshake {
                    min_version,
                    max_version,
                    supported_codecs,
                } => {
                    if audio_started {
                        return Err(FwError::InvalidRequest(
                            "tty-audio handshake must appear before audio frames".to_owned(),
                        ));
                    }
                    if handshake_seen {
                        return Err(FwError::InvalidRequest(
                            "duplicate tty-audio handshake".to_owned(),
                        ));
                    }

                    let negotiated = negotiate_version(
                        MIN_PROTOCOL_VERSION,
                        SUPPORTED_PROTOCOL_VERSION,
                        min_version,
                        max_version,
                    )
                    .map_err(FwError::InvalidRequest)?;
                    if !supported_codecs
                        .iter()
                        .any(|codec| codec == CODEC_MULAW_ZLIB_B64)
                    {
                        return Err(FwError::InvalidRequest(format!(
                            "no compatible tty-audio codec in handshake (expected `{CODEC_MULAW_ZLIB_B64}`)"
                        )));
                    }
                    negotiated_version = negotiated;
                    handshake_seen = true;
                }
                TtyControlFrame::HandshakeAck {
                    negotiated_version: ack_version,
                    negotiated_codec,
                } => {
                    if !handshake_seen {
                        return Err(FwError::InvalidRequest(
                            "tty-audio handshake_ack received before handshake".to_owned(),
                        ));
                    }
                    if ack_version != negotiated_version {
                        return Err(FwError::InvalidRequest(format!(
                            "unexpected negotiated_version {} in handshake_ack",
                            ack_version
                        )));
                    }
                    if negotiated_codec != CODEC_MULAW_ZLIB_B64 {
                        return Err(FwError::InvalidRequest(format!(
                            "unsupported negotiated codec `{negotiated_codec}` in handshake_ack"
                        )));
                    }
                }
                TtyControlFrame::SessionClose {
                    reason,
                    last_data_seq,
                } => {
                    if let Some(existing_reason) = session_close_reason {
                        if existing_reason != reason || session_close_last_seq != last_data_seq {
                            return Err(FwError::InvalidRequest(
                                "conflicting tty-audio session_close frames received".to_owned(),
                            ));
                        }
                    } else {
                        session_close_reason = Some(reason);
                        session_close_last_seq = last_data_seq;
                    }
                    if !handshake_seen && !audio_started {
                        return Err(FwError::InvalidRequest(
                            "tty-audio control frame received before handshake".to_owned(),
                        ));
                    }
                }
                TtyControlFrame::RetransmitRequest { .. }
                | TtyControlFrame::RetransmitResponse { .. }
                | TtyControlFrame::Ack { .. }
                | TtyControlFrame::Backpressure { .. }
                | TtyControlFrame::TranscriptPartial { .. }
                | TtyControlFrame::TranscriptRetract { .. }
                | TtyControlFrame::TranscriptCorrect { .. } => {
                    if !handshake_seen && !audio_started {
                        return Err(FwError::InvalidRequest(
                            "tty-audio control frame received before handshake".to_owned(),
                        ));
                    }
                }
            },
        }
    }

    if let Some(reason) = session_close_reason {
        let observed_last_seq = frames.iter().map(|frame| frame.seq).max();
        let close_frame = TtyControlFrame::SessionClose {
            reason,
            last_data_seq: session_close_last_seq,
        };
        validate_session_close(observed_last_seq, &close_frame)?;
    }

    Ok(frames)
}

#[cfg(test)]
fn parse_frames<R: Read>(reader: &mut R) -> FwResult<Vec<TtyAudioFrame>> {
    let entries = parse_frame_lines(reader)?;
    Ok(entries
        .into_iter()
        .filter_map(|entry| match entry {
            FrameLine::Audio(frame) => Some(frame),
            FrameLine::Control(_) => None,
        })
        .collect())
}

fn parse_frame_lines<R: Read>(reader: &mut R) -> FwResult<Vec<FrameLine>> {
    let buffered = BufReader::new(reader);
    let mut frames = Vec::new();

    for (line_number, line) in buffered.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let entry = parse_frame_line(line).map_err(|error| {
            FwError::InvalidRequest(format!(
                "invalid tty-audio frame at line {}: {}",
                line_number + 1,
                error
            ))
        })?;
        frames.push(entry);
    }

    Ok(frames)
}

fn parse_frame_line(line: &str) -> FwResult<FrameLine> {
    if line.contains("\"frame_type\"") {
        let control: TtyControlFrame = serde_json::from_str(line)?;
        Ok(FrameLine::Control(control))
    } else {
        let frame: TtyAudioFrame = serde_json::from_str(line)?;
        Ok(FrameLine::Audio(frame))
    }
}

fn write_control_frame<W: Write>(writer: &mut W, frame: &TtyControlFrame) -> FwResult<()> {
    match frame {
        TtyControlFrame::Handshake {
            min_version,
            max_version,
            supported_codecs,
        } => {
            write!(
                writer,
                "{{\"frame_type\":\"handshake\",\"min_version\":{min_version},\"max_version\":{max_version},\"supported_codecs\":"
            )?;
            write_json_string_array(writer, supported_codecs)?;
            writer.write_all(b"}\n")?;
        }
        TtyControlFrame::HandshakeAck {
            negotiated_version,
            negotiated_codec,
        } => {
            write!(
                writer,
                "{{\"frame_type\":\"handshake_ack\",\"negotiated_version\":{negotiated_version},\"negotiated_codec\":"
            )?;
            serde_json::to_writer(&mut *writer, negotiated_codec)?;
            writer.write_all(b"}\n")?;
        }
        TtyControlFrame::Ack { up_to_seq } => {
            writeln!(
                writer,
                "{{\"frame_type\":\"ack\",\"up_to_seq\":{up_to_seq}}}"
            )?;
        }
        TtyControlFrame::Backpressure { remaining_capacity } => {
            writeln!(
                writer,
                "{{\"frame_type\":\"backpressure\",\"remaining_capacity\":{remaining_capacity}}}"
            )?;
        }
        TtyControlFrame::RetransmitRequest { .. }
        | TtyControlFrame::RetransmitResponse { .. }
        | TtyControlFrame::SessionClose { .. }
        | TtyControlFrame::TranscriptPartial { .. }
        | TtyControlFrame::TranscriptRetract { .. }
        | TtyControlFrame::TranscriptCorrect { .. } => {
            writeln!(writer, "{}", serde_json::to_string(frame)?)?;
        }
    }
    Ok(())
}

fn write_json_string_array<W: Write>(writer: &mut W, values: &[String]) -> FwResult<()> {
    writer.write_all(b"[")?;
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            writer.write_all(b",")?;
        }
        serde_json::to_writer(&mut *writer, value)?;
    }
    writer.write_all(b"]")?;
    Ok(())
}

fn transcode_to_mulaw(input_audio: &Path, ulaw_path: &Path) -> FwResult<()> {
    let ffmpeg_program = audio::resolve_ffmpeg_program(None)?;
    let timeout = audio::ffmpeg_timeout();
    let args = vec![
        "-hide_banner".to_owned(),
        "-loglevel".to_owned(),
        "error".to_owned(),
        "-y".to_owned(),
        "-i".to_owned(),
        input_audio.display().to_string(),
        "-ar".to_owned(),
        "8000".to_owned(),
        "-ac".to_owned(),
        "1".to_owned(),
        "-f".to_owned(),
        "mulaw".to_owned(),
        ulaw_path.display().to_string(),
    ];
    run_command_with_timeout(&ffmpeg_program, &args, None, Some(timeout))?;
    Ok(())
}

fn ensure_parent_dir(path: &Path) -> FwResult<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() {
        return Ok(());
    }
    fs::create_dir_all(parent)?;
    Ok(())
}

fn transcode_mulaw_to_wav(raw_ulaw: &Path, output_wav: &Path) -> FwResult<()> {
    ensure_parent_dir(output_wav)?;

    let ffmpeg_program = audio::resolve_ffmpeg_program(None)?;
    let timeout = audio::ffmpeg_timeout();
    let args = vec![
        "-hide_banner".to_owned(),
        "-loglevel".to_owned(),
        "error".to_owned(),
        "-y".to_owned(),
        "-f".to_owned(),
        "mulaw".to_owned(),
        "-ar".to_owned(),
        "8000".to_owned(),
        "-ac".to_owned(),
        "1".to_owned(),
        "-i".to_owned(),
        raw_ulaw.display().to_string(),
        output_wav.display().to_string(),
    ];
    run_command_with_timeout(&ffmpeg_program, &args, None, Some(timeout))?;
    Ok(())
}

fn mulaw_chunk_size(chunk_ms: u32) -> usize {
    let clamped = chunk_ms.clamp(20, 5_000);
    // 8kHz * 1 byte/sample * duration
    ((8_000u64 * u64::from(clamped)) / 1_000) as usize
}

fn compress_chunk(input: &[u8]) -> FwResult<Vec<u8>> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(input)?;
    Ok(encoder.finish()?)
}

/// Maximum decompressed frame size: 5 seconds of 8 kHz mono mulaw = 40 000 bytes.
/// A 2× safety margin gives 80 000 bytes.  Anything larger is treated as a
/// decompression bomb and rejected.
const MAX_DECOMPRESSED_FRAME_BYTES: usize = 80_000;

fn decompress_chunk(input: &[u8]) -> FwResult<Vec<u8>> {
    // `input` is already a fully-buffered `&[u8]` (which implements `BufRead`), so
    // feed it straight to the `bufread` decoder. `flate2::read::ZlibDecoder` would
    // wrap it in an *additional* 32 KiB read-ahead `BufReader`, allocating + memmoving
    // a scratch buffer for every frame; `flate2::bufread::ZlibDecoder` reads directly
    // from the slice. Output is byte-identical — same inflate, no extra copy.
    let mut decoder = ZlibDecoder::new(input);
    let mut out = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = decoder.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf[..n]);
        if out.len() > MAX_DECOMPRESSED_FRAME_BYTES {
            return Err(FwError::InvalidRequest(format!(
                "decompressed frame exceeds {MAX_DECOMPRESSED_FRAME_BYTES} byte limit \
                 (possible decompression bomb)"
            )));
        }
    }
    Ok(out)
}

fn crc32_of(data: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    format!("{digest:x}")
}

// ---------------------------------------------------------------------------
// Real-time mic-to-NDJSON streaming (bead bd-2xe.1)
// ---------------------------------------------------------------------------

/// Configuration for real-time microphone-to-NDJSON streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicStreamConfig {
    /// Duration of each audio chunk in milliseconds.  Clamped to [20, 5000].
    pub chunk_ms: u32,
    /// Target sample rate for the mu-law encoding pipeline.  Currently only
    /// 8000 Hz is supported by the TTY audio codec.
    pub sample_rate_hz: u32,
    /// Number of audio channels.  Currently only mono (1) is supported.
    pub channels: u8,
    /// Optional audio device identifier (e.g. `"hw:1,0"` on ALSA).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
}

impl Default for MicStreamConfig {
    fn default() -> Self {
        Self {
            chunk_ms: 200,
            sample_rate_hz: 8_000,
            channels: 1,
            device: None,
        }
    }
}

impl MicStreamConfig {
    /// Validate this configuration, returning an error for unsupported
    /// sample rates or channel counts.
    pub fn validate(&self) -> FwResult<()> {
        if self.sample_rate_hz != 8_000 {
            return Err(FwError::InvalidRequest(format!(
                "unsupported mic stream sample_rate_hz={}: only 8000 is supported",
                self.sample_rate_hz
            )));
        }
        if self.channels != 1 {
            return Err(FwError::InvalidRequest(format!(
                "unsupported mic stream channels={}: only mono (1) is supported",
                self.channels
            )));
        }
        Ok(())
    }

    /// Compute the byte size of a single chunk based on `chunk_ms`.
    #[must_use]
    pub fn chunk_byte_size(&self) -> usize {
        mulaw_chunk_size(self.chunk_ms)
    }
}

/// NDJSON event emitted for each audio chunk captured from the microphone.
///
/// This wraps a [`TtyAudioFrame`] with the project's standard NDJSON event
/// envelope (event type + schema version), integrating with the existing
/// robot schema versioning defined in [`crate::robot`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicStreamEvent {
    /// Always `"mic_audio_chunk"`.
    pub event: String,
    /// Robot schema version for forward compatibility.
    pub schema_version: String,
    /// The TTY audio frame carrying the encoded audio payload.
    pub frame: TtyAudioFrame,
}

/// Required NDJSON fields for `mic_audio_chunk` events.
pub const MIC_AUDIO_CHUNK_REQUIRED_FIELDS: &[&str] = &["event", "schema_version", "frame"];

/// Construct a [`MicStreamEvent`] from a [`TtyAudioFrame`].
#[must_use]
pub fn mic_stream_event_value(frame: &TtyAudioFrame) -> MicStreamEvent {
    MicStreamEvent {
        event: "mic_audio_chunk".to_owned(),
        schema_version: crate::robot::ROBOT_SCHEMA_VERSION.to_owned(),
        frame: frame.clone(),
    }
}

/// Trait abstracting the raw audio source for mic streaming.
///
/// Implementing this trait allows callers to substitute real microphone
/// hardware with synthetic sources for testing.
pub trait MicAudioSource {
    /// Read the next chunk of **mu-law encoded** bytes from the audio source.
    ///
    /// Returns `Ok(Some(bytes))` on success, `Ok(None)` when the source is
    /// exhausted (end-of-stream), and `Err` on device failures.
    fn read_chunk(&mut self, chunk_size: usize) -> FwResult<Option<Vec<u8>>>;
}

/// Streams audio chunks from a [`MicAudioSource`] through the TTY audio
/// codec and writes NDJSON [`MicStreamEvent`] lines to `writer`.
///
/// This function:
/// 1. Validates the configuration
/// 2. Emits a handshake control frame
/// 3. Reads chunks from the source in a loop
/// 4. Encodes each chunk (compress + base64 + integrity hashes)
/// 5. Wraps each encoded frame in a `MicStreamEvent` envelope
/// 6. Writes one NDJSON line per chunk
///
/// Returns the total number of frames emitted.
pub fn stream_mic_to_ndjson<S: MicAudioSource, W: Write>(
    config: &MicStreamConfig,
    source: &mut S,
    writer: &mut W,
) -> FwResult<u64> {
    config.validate()?;

    let chunk_size = config.chunk_byte_size();

    // Emit protocol handshake.
    let handshake = TtyControlFrame::Handshake {
        min_version: MIN_PROTOCOL_VERSION,
        max_version: SUPPORTED_PROTOCOL_VERSION,
        supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned()],
    };
    write_control_frame(writer, &handshake)?;

    let mut seq: u64 = 0;

    loop {
        let chunk = match source.read_chunk(chunk_size) {
            Ok(Some(data)) => data,
            Ok(None) => break,
            Err(error) => return Err(error),
        };

        if chunk.is_empty() {
            continue;
        }

        let crc = crc32_of(&chunk);
        let sha = sha256_hex(&chunk);
        let compressed = compress_chunk(&chunk)?;

        let frame = TtyAudioFrame {
            protocol_version: SUPPORTED_PROTOCOL_VERSION,
            seq,
            codec: CODEC_MULAW_ZLIB_B64.to_owned(),
            sample_rate_hz: config.sample_rate_hz,
            channels: config.channels,
            payload_b64: STANDARD_NO_PAD.encode(compressed),
            crc32: Some(crc),
            payload_sha256: Some(sha),
        };

        let event = mic_stream_event_value(&frame);
        writeln!(writer, "{}", serde_json::to_string(&event)?)?;

        seq += 1;
    }

    Ok(seq)
}

/// A [`MicAudioSource`] backed by a byte slice — useful for testing and
/// for piping pre-recorded mu-law data through the streaming pipeline.
pub struct SliceMicSource<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> SliceMicSource<'a> {
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }
}

impl MicAudioSource for SliceMicSource<'_> {
    fn read_chunk(&mut self, chunk_size: usize) -> FwResult<Option<Vec<u8>>> {
        if self.offset >= self.data.len() {
            return Ok(None);
        }
        let end = (self.offset + chunk_size).min(self.data.len());
        let chunk = self.data[self.offset..end].to_vec();
        self.offset = end;
        Ok(Some(chunk))
    }
}

/// A [`MicAudioSource`] that always returns a device-unavailable error.
/// Useful for testing error-handling paths.
pub struct UnavailableMicSource {
    message: String,
}

impl UnavailableMicSource {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl MicAudioSource for UnavailableMicSource {
    fn read_chunk(&mut self, _chunk_size: usize) -> FwResult<Option<Vec<u8>>> {
        Err(FwError::BackendUnavailable(self.message.clone()))
    }
}

/// A [`MicAudioSource`] that yields a fixed number of chunks and then
/// signals end-of-stream.  Useful for bounded-duration test scenarios.
pub struct FixedCountMicSource {
    remaining: u64,
    fill_byte: u8,
}

impl FixedCountMicSource {
    #[must_use]
    pub fn new(count: u64, fill_byte: u8) -> Self {
        Self {
            remaining: count,
            fill_byte,
        }
    }
}

impl MicAudioSource for FixedCountMicSource {
    fn read_chunk(&mut self, chunk_size: usize) -> FwResult<Option<Vec<u8>>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        Ok(Some(vec![self.fill_byte; chunk_size]))
    }
}

// ---------------------------------------------------------------------------
// Pipeline adapter: TTY audio -> decoded PCM -> temp WAV (bead bd-2xe.2)
// ---------------------------------------------------------------------------

/// Adapter that bridges the TTY audio frame stream into the transcription
/// pipeline by decoding frames to raw PCM mu-law bytes and writing them to a
/// temporary WAV file that the pipeline can ingest.
///
/// Lifecycle:
/// 1. `new()` -- create the adapter (allocates a temp directory).
/// 2. `ingest_frames()` -- feed in `TtyAudioFrame`s; decoded bytes accumulate.
/// 3. `finalize_wav()` -- write the accumulated bytes to a WAV file and return
///    the path for the transcription pipeline.
pub struct TtyAudioPipelineAdapter {
    raw_pcm: std::collections::BTreeMap<u64, Vec<u8>>,
    temp_dir: tempfile::TempDir,
    finalized: bool,
    frames_ingested: u64,
}

impl TtyAudioPipelineAdapter {
    /// Create a new pipeline adapter.  Allocates a temporary directory that
    /// will hold intermediate files.
    pub fn new() -> FwResult<Self> {
        let temp_dir = tempfile::tempdir()?;
        Ok(Self {
            raw_pcm: std::collections::BTreeMap::new(),
            temp_dir,
            finalized: false,
            frames_ingested: 0,
        })
    }

    /// Ingest a slice of TTY audio frames.  Each frame is decoded (base64 ->
    /// zlib decompress) and stored by sequence number. Returns the
    /// number of new frames successfully decoded in this call.
    pub fn ingest_frames(&mut self, frames: &[TtyAudioFrame]) -> FwResult<u64> {
        if self.finalized {
            return Err(FwError::InvalidRequest(
                "pipeline adapter has already been finalized".to_owned(),
            ));
        }

        let mut sorted_frames = frames.to_vec();
        sorted_frames.sort_by_key(|f| f.seq);

        let mut count: u64 = 0;
        for frame in &sorted_frames {
            // Skip if we already have this frame
            if self.raw_pcm.contains_key(&frame.seq) {
                continue;
            }

            validate_audio_frame_metadata(frame)?;

            let compressed = STANDARD_NO_PAD
                .decode(&frame.payload_b64)
                .map_err(|e| FwError::InvalidRequest(format!("invalid base64 payload: {e}")))?;
            let decoded = decompress_chunk(&compressed)?;

            // Integrity check: CRC32.
            if let Some(expected_crc) = frame.crc32 {
                let actual_crc = crc32_of(&decoded);
                if actual_crc != expected_crc {
                    return Err(FwError::InvalidRequest(format!(
                        "CRC mismatch at seq {}: expected {expected_crc}, got {actual_crc}",
                        frame.seq
                    )));
                }
            }

            // Integrity check: SHA-256.
            if let Some(expected_sha) = frame.payload_sha256.as_deref() {
                let actual_sha = sha256_hex(&decoded);
                if !actual_sha.eq_ignore_ascii_case(expected_sha) {
                    return Err(FwError::InvalidRequest(format!(
                        "SHA-256 mismatch at seq {}: expected {expected_sha}, got {actual_sha}",
                        frame.seq
                    )));
                }
            }

            self.raw_pcm.insert(frame.seq, decoded);
            count += 1;
        }
        self.frames_ingested += count;
        Ok(count)
    }

    /// Finalize the adapter: write the accumulated raw mu-law PCM to a WAV
    /// file (via ffmpeg transcode) and return the path.  After this call the
    /// adapter is sealed and `ingest_frames` will error.
    pub fn finalize_wav(&mut self) -> FwResult<PathBuf> {
        if self.finalized {
            return Err(FwError::InvalidRequest(
                "pipeline adapter has already been finalized".to_owned(),
            ));
        }
        if self.raw_pcm.is_empty() {
            return Err(FwError::InvalidRequest(
                "no frames have been ingested".to_owned(),
            ));
        }

        self.finalized = true;

        let Some(highest_seq) = self.raw_pcm.keys().next_back().copied() else {
            self.finalized = false;
            return Err(FwError::InvalidRequest(
                "no frames have been ingested".to_owned(),
            ));
        };
        let expected_frame_count = highest_seq + 1;
        if self.raw_pcm.len() as u64 != expected_frame_count {
            self.finalized = false;
            return Err(FwError::InvalidRequest(format!(
                "cannot finalize tty-audio pipeline with missing frame sequences: have {}, expected contiguous 0..={highest_seq}",
                self.raw_pcm.len()
            )));
        }

        let mut contiguous_pcm = Vec::new();
        // Since it's a BTreeMap, values are yielded in sequence number order.
        for decoded in self.raw_pcm.values() {
            contiguous_pcm.extend_from_slice(decoded);
        }

        let raw_path = self.temp_dir.path().join("decoded.ulaw");
        let wav_path = self.temp_dir.path().join("pipeline_input.wav");
        fs::write(&raw_path, &contiguous_pcm)?;
        transcode_mulaw_to_wav(&raw_path, &wav_path)?;
        Ok(wav_path)
    }

    /// Number of frames ingested so far.
    #[must_use]
    pub fn frames_ingested(&self) -> u64 {
        self.frames_ingested
    }

    /// Whether `finalize_wav` has been called.
    #[must_use]
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Total bytes of decoded PCM accumulated.
    #[must_use]
    pub fn raw_pcm_len(&self) -> usize {
        self.raw_pcm.values().map(|v| v.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Adaptive bitrate controller (bead bd-2xe.3)
// ---------------------------------------------------------------------------

/// Tracks link quality and adjusts compression level dynamically.
///
/// The controller monitors frame delivery success rate and selects a
/// compression level accordingly:
/// - High quality link (loss < 1%) -> low compression (fast / level 1)
/// - Moderate link (loss 1%-10%) -> default compression (level 6)
/// - Poor link (loss > 10%) -> best compression (level 9)
///
/// It also provides simple FEC: critical frame types (handshake, EOF markers)
/// are duplicated a configurable number of times.
#[derive(Debug, Clone)]
pub struct AdaptiveBitrateController {
    /// Target bitrate in bits per second (informational / advisory).
    pub target_bitrate: u32,
    /// Current quality estimate in the range [0.0, 1.0] where 1.0 is perfect.
    pub current_quality: f64,
    /// Observed frame loss rate in [0.0, 1.0].
    pub frame_loss_rate: f64,
    frames_sent: u64,
    frames_lost: u64,
}

/// Quality thresholds for adaptive bitrate decisions.
const ABR_HIGH_QUALITY_LOSS_THRESHOLD: f64 = 0.01;
const ABR_MODERATE_QUALITY_LOSS_THRESHOLD: f64 = 0.10;

impl AdaptiveBitrateController {
    /// Create a new controller with a target bitrate.  Initial quality is
    /// assumed perfect (1.0) and loss rate is 0.
    #[must_use]
    pub fn new(target_bitrate: u32) -> Self {
        Self {
            target_bitrate,
            current_quality: 1.0,
            frame_loss_rate: 0.0,
            frames_sent: 0,
            frames_lost: 0,
        }
    }

    /// Record a frame delivery outcome.  `delivered` is true if the frame was
    /// acknowledged, false if it was lost / timed out.
    pub fn record_delivery(&mut self, delivered: bool) {
        self.frames_sent += 1;
        if !delivered {
            self.frames_lost += 1;
        }
        self.recalculate();
    }

    /// Record a batch of delivery outcomes.
    pub fn record_batch(&mut self, delivered_count: u64, lost_count: u64) {
        self.frames_sent += delivered_count + lost_count;
        self.frames_lost += lost_count;
        self.recalculate();
    }

    /// Return the recommended zlib `Compression` level based on current link
    /// quality.
    #[must_use]
    pub fn recommended_compression(&self) -> Compression {
        if self.frame_loss_rate <= ABR_HIGH_QUALITY_LOSS_THRESHOLD {
            Compression::fast() // level 1
        } else if self.frame_loss_rate <= ABR_MODERATE_QUALITY_LOSS_THRESHOLD {
            Compression::default() // level 6
        } else {
            Compression::best() // level 9
        }
    }

    /// Compress a chunk using the currently recommended compression level.
    pub fn compress_adaptive(&self, input: &[u8]) -> FwResult<Vec<u8>> {
        let level = self.recommended_compression();
        let mut encoder = ZlibEncoder::new(Vec::new(), level);
        encoder.write_all(input)?;
        Ok(encoder.finish()?)
    }

    /// Returns the number of times a critical frame (handshake, EOF) should
    /// be emitted for FEC purposes.  On a perfect link this is 1 (no
    /// duplication); on a poor link, critical frames are sent up to 3 times.
    #[must_use]
    pub fn critical_frame_redundancy(&self) -> u32 {
        if self.frame_loss_rate <= ABR_HIGH_QUALITY_LOSS_THRESHOLD {
            1
        } else if self.frame_loss_rate <= ABR_MODERATE_QUALITY_LOSS_THRESHOLD {
            2
        } else {
            3
        }
    }

    /// Emit a control frame with FEC duplication appropriate for the current
    /// link quality.  Critical control frames (handshake, session close) are
    /// written multiple times based on `critical_frame_redundancy()`.
    pub fn emit_critical_frame_with_fec<W: Write>(
        &self,
        writer: &mut W,
        frame: &TtyControlFrame,
    ) -> FwResult<u32> {
        let count = self.critical_frame_redundancy();
        for _ in 0..count {
            write_control_frame(writer, frame)?;
        }
        Ok(count)
    }

    fn recalculate(&mut self) {
        if self.frames_sent == 0 {
            self.frame_loss_rate = 0.0;
            self.current_quality = 1.0;
        } else {
            self.frame_loss_rate =
                (self.frames_lost as f64 / self.frames_sent as f64).clamp(0.0, 1.0);
            self.current_quality = 1.0 - self.frame_loss_rate;
        }
    }

    /// Total frames observed (sent + lost).
    #[must_use]
    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// Total frames lost.
    #[must_use]
    pub fn frames_lost(&self) -> u64 {
        self.frames_lost
    }
}

// ---------------------------------------------------------------------------
// TTY Control-Plane Closure Packet (bead bd-30v)
// ---------------------------------------------------------------------------

/// Reason code for a session close control frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionCloseReason {
    /// Normal end of stream -- all data frames have been sent.
    #[serde(rename = "normal")]
    Normal,
    /// The session is being closed due to an error condition.
    #[serde(rename = "error")]
    Error,
    /// The session is being closed due to a timeout.
    #[serde(rename = "timeout")]
    Timeout,
    /// The peer requested session termination.
    #[serde(rename = "peer_requested")]
    PeerRequested,
}

/// Emit a `session_close` control frame to the given writer.
pub fn emit_session_close<W: Write>(
    writer: &mut W,
    reason: SessionCloseReason,
    last_data_seq: Option<u64>,
) -> FwResult<()> {
    let frame = TtyControlFrame::SessionClose {
        reason,
        last_data_seq,
    };
    writeln!(writer, "{}", serde_json::to_string(&frame)?)?;
    Ok(())
}

/// Validate that a session close frame is properly sequenced: it should only
/// appear after all data frames.  Given the highest data-frame sequence
/// observed so far and the `last_data_seq` claimed in the close frame, this
/// returns `Ok(())` if they are consistent, or an error otherwise.
pub fn validate_session_close(
    observed_last_seq: Option<u64>,
    close_frame: &TtyControlFrame,
) -> FwResult<()> {
    match close_frame {
        TtyControlFrame::SessionClose {
            last_data_seq,
            reason: _,
        } => {
            match (observed_last_seq, last_data_seq) {
                (Some(observed), Some(claimed)) if observed != *claimed => {
                    return Err(FwError::InvalidRequest(format!(
                        "session_close last_data_seq mismatch: observed {observed}, claimed {claimed}"
                    )));
                }
                (None, Some(claimed)) => {
                    return Err(FwError::InvalidRequest(format!(
                        "session_close claims last_data_seq={claimed} but no data frames were observed"
                    )));
                }
                _ => {}
            }
            Ok(())
        }
        _ => Err(FwError::InvalidRequest(
            "validate_session_close requires a session_close control frame".to_owned(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Deterministic tty-audio retransmit-loop automation (bead bd-2xe.5)
// ---------------------------------------------------------------------------

/// Recovery strategy applied at each round of the retransmit loop.
///
/// The loop can escalate through strategies as rounds progress:
/// - `Simple` — retransmit frames as-is (no modification).
/// - `Redundant` — add forward error correction (duplicate each frame).
/// - `Escalate` — increase compression level for smaller frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Retransmit lost frames without modification.
    Simple,
    /// Retransmit with redundancy (each frame is sent twice).
    Redundant,
    /// Retransmit with increased compression to reduce frame size.
    Escalate,
}

impl RecoveryStrategy {
    /// Return the next escalation level.  `Escalate` is the ceiling.
    #[must_use]
    pub fn escalate(self) -> Self {
        match self {
            Self::Simple => Self::Redundant,
            Self::Redundant => Self::Escalate,
            Self::Escalate => Self::Escalate,
        }
    }
}

/// Summary report produced after a retransmit loop completes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetransmitReport {
    /// Total number of frames in the buffer.
    pub total_frames: usize,
    /// Number of frames that were marked as lost.
    pub lost_frames: usize,
    /// Number of lost frames that were recovered during the loop.
    pub recovered_frames: usize,
    /// Number of recovery rounds actually executed.
    pub rounds_used: u32,
    /// The strategy that was active in the final round.
    pub final_strategy: RecoveryStrategy,
}

/// Manages automatic retransmission of lost TTY audio frames.
///
/// The loop is fully deterministic: given the same input frames and the same
/// loss pattern (injected via [`inject_loss`](Self::inject_loss)), the output
/// and report are identical across runs.
///
/// # Lifecycle
///
/// 1. Create with [`new`](Self::new), providing the frame buffer and limits.
/// 2. Optionally call [`inject_loss`](Self::inject_loss) to simulate frame
///    drops for deterministic testing.
/// 3. Call [`run`](Self::run) to execute recovery rounds.
/// 4. Call [`report`](Self::report) to obtain a [`RetransmitReport`].
pub struct RetransmitLoop {
    /// Maximum number of recovery rounds before giving up.
    pub max_rounds: u32,
    /// Per-round timeout in milliseconds (advisory; used for reporting, not
    /// for actual sleeping, keeping the loop deterministic).
    pub timeout_ms: u64,
    /// The recovery strategy for the first round.  The loop escalates
    /// automatically on subsequent rounds if losses remain.
    pub recovery_strategy: RecoveryStrategy,
    /// The frame buffer managed by this loop.
    pub frame_buffer: Vec<TtyAudioFrame>,

    // -- internal bookkeeping --
    /// Set of sequence numbers marked as lost.
    lost_seqs: HashSet<u64>,
    /// Set of sequence numbers that have been recovered.
    recovered_seqs: HashSet<u64>,
    /// Number of rounds actually executed so far.
    rounds_executed: u32,
    /// The strategy used in the most recent round.
    current_strategy: RecoveryStrategy,
}

impl RetransmitLoop {
    /// Create a new retransmit loop.
    ///
    /// - `max_rounds`: ceiling on recovery iterations.
    /// - `timeout_ms`: advisory per-round timeout (deterministic — no sleep).
    /// - `recovery_strategy`: starting strategy for round 0.
    /// - `frame_buffer`: the full set of frames (including those that may
    ///   later be marked as lost).
    #[must_use]
    pub fn new(
        max_rounds: u32,
        timeout_ms: u64,
        recovery_strategy: RecoveryStrategy,
        frame_buffer: Vec<TtyAudioFrame>,
    ) -> Self {
        let current_strategy = recovery_strategy;
        Self {
            max_rounds,
            timeout_ms,
            recovery_strategy,
            frame_buffer,
            lost_seqs: HashSet::new(),
            recovered_seqs: HashSet::new(),
            rounds_executed: 0,
            current_strategy,
        }
    }

    /// Mark specific sequence numbers as lost for deterministic testing.
    ///
    /// Only sequences that actually exist in `frame_buffer` are recorded;
    /// unknown sequence numbers are silently ignored.
    pub fn inject_loss(&mut self, sequences: &[u64]) {
        let known: HashSet<u64> = self.frame_buffer.iter().map(|f| f.seq).collect();
        self.lost_seqs.clear();
        for &seq in sequences {
            if known.contains(&seq) {
                self.lost_seqs.insert(seq);
            }
        }
        // Clear any prior recovery state when loss pattern changes.
        self.recovered_seqs.clear();
        self.rounds_executed = 0;
        self.current_strategy = self.recovery_strategy;
    }

    /// Execute the retransmit loop.
    ///
    /// Each round:
    /// 1. Identify frames still lost (lost but not yet recovered).
    /// 2. If none remain, stop early.
    /// 3. Apply the current [`RecoveryStrategy`] to attempt recovery.
    /// 4. Escalate the strategy for the next round.
    ///
    /// The recovery model is deterministic:
    /// - `Simple` recovers 1 frame per round.
    /// - `Redundant` recovers up to 2 frames per round.
    /// - `Escalate` recovers up to 4 frames per round.
    ///
    /// Returns `Ok(())` on completion (even if some frames remain
    /// unrecovered after `max_rounds`).
    pub fn run(&mut self) -> FwResult<()> {
        self.rounds_executed = 0;
        self.recovered_seqs.clear();
        self.current_strategy = self.recovery_strategy;

        for _round in 0..self.max_rounds {
            let still_lost: Vec<u64> = self
                .lost_seqs
                .iter()
                .copied()
                .filter(|seq| !self.recovered_seqs.contains(seq))
                .collect();

            if still_lost.is_empty() {
                // All losses recovered — stop early.
                break;
            }

            self.rounds_executed += 1;

            // Deterministic recovery: strategy determines how many frames
            // we can recover per round.
            let recover_count = match self.current_strategy {
                RecoveryStrategy::Simple => 1,
                RecoveryStrategy::Redundant => 2,
                RecoveryStrategy::Escalate => 4,
            };

            // Recover frames in sequence-number order for determinism.
            let mut sorted_lost = still_lost;
            sorted_lost.sort_unstable();

            for &seq in sorted_lost.iter().take(recover_count) {
                self.recovered_seqs.insert(seq);
            }

            // Escalate for next round.
            self.current_strategy = self.current_strategy.escalate();
        }

        Ok(())
    }

    /// Produce a summary report of the retransmit loop state.
    ///
    /// Can be called before or after [`run`](Self::run).
    #[must_use]
    pub fn report(&self) -> RetransmitReport {
        RetransmitReport {
            total_frames: self.frame_buffer.len(),
            lost_frames: self.lost_seqs.len(),
            recovered_frames: self.recovered_seqs.len(),
            rounds_used: self.rounds_executed,
            final_strategy: self.current_strategy,
        }
    }
}

// ---------------------------------------------------------------------------
// bd-qlt.10: TTY transcript correction frame helpers
// ---------------------------------------------------------------------------

/// Emit a speculative transcript partial over the TTY protocol.
pub fn emit_tty_transcript_partial(
    writer: &mut impl Write,
    seq: u64,
    window_id: u64,
    segments: &[crate::model::TranscriptionSegment],
    model_id: &str,
    speculative: bool,
) -> FwResult<()> {
    let compact: Vec<TranscriptSegmentCompact> = segments.iter().map(Into::into).collect();
    let frame = TtyControlFrame::TranscriptPartial {
        seq,
        window_id,
        segments: compact,
        model_id: model_id.to_owned(),
        speculative,
    };
    let line = serde_json::to_string(&frame)?;
    writeln!(writer, "{line}")?;
    Ok(())
}

/// Emit a transcript retraction over the TTY protocol.
pub fn emit_tty_transcript_retract(
    writer: &mut impl Write,
    retracted_seq: u64,
    window_id: u64,
    reason: &str,
) -> FwResult<()> {
    let frame = TtyControlFrame::TranscriptRetract {
        retracted_seq,
        window_id,
        reason: reason.to_owned(),
    };
    let line = serde_json::to_string(&frame)?;
    writeln!(writer, "{line}")?;
    Ok(())
}

/// Emit a transcript correction over the TTY protocol.
pub fn emit_tty_transcript_correct(
    writer: &mut impl Write,
    correction: &crate::speculation::CorrectionEvent,
) -> FwResult<()> {
    let compact: Vec<TranscriptSegmentCompact> = correction
        .corrected_segments
        .iter()
        .map(Into::into)
        .collect();
    let frame = TtyControlFrame::TranscriptCorrect {
        correction_id: correction.correction_id,
        replaces_seq: correction.retracted_seq,
        window_id: correction.window_id,
        segments: compact,
        model_id: correction.quality_model_id.clone(),
        drift_wer: correction.drift.wer_approx,
    };
    let line = serde_json::to_string(&frame)?;
    writeln!(writer, "{line}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::error::{FwError, FwResult};
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD_NO_PAD;

    use super::{
        CODEC_MULAW_ZLIB_B64, DEFAULT_PROTOCOL_VERSION, DecodeRecoveryPolicy, MIN_PROTOCOL_VERSION,
        SUPPORTED_PROTOCOL_VERSION, SessionCloseReason, TranscriptSegmentCompact, TtyAudioFrame,
        compress_chunk, crc32_of, decode_frames_to_raw, decode_frames_to_raw_with_policy,
        decompress_chunk, emit_control_frame_to_writer, emit_retransmit_loop_from_reader,
        emit_tty_transcript_partial, emit_tty_transcript_retract, ensure_parent_dir,
        mulaw_chunk_size, parse_audio_frames_for_decode, parse_frame_line, parse_frames,
        retransmit_plan_from_reader, sha256_hex,
    };

    fn make_frame(seq: u64, data: &[u8]) -> TtyAudioFrame {
        let compressed = compress_chunk(data).expect("compress");
        TtyAudioFrame {
            protocol_version: SUPPORTED_PROTOCOL_VERSION,
            seq,
            codec: CODEC_MULAW_ZLIB_B64.to_owned(),
            sample_rate_hz: 8_000,
            channels: 1,
            payload_b64: STANDARD_NO_PAD.encode(compressed),
            crc32: Some(crc32_of(data)),
            payload_sha256: Some(sha256_hex(data)),
        }
    }

    fn frames_to_ndjson(frames: &[TtyAudioFrame]) -> String {
        frames
            .iter()
            .map(|f| serde_json::to_string(f).expect("serialize"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    #[test]
    fn chunk_size_is_bounded() {
        assert_eq!(mulaw_chunk_size(1), 160);
        assert_eq!(mulaw_chunk_size(20), 160);
        assert_eq!(mulaw_chunk_size(200), 1600);
    }

    #[test]
    fn compression_roundtrip() {
        let data = b"hello hello hello hello";
        let compressed = compress_chunk(data).expect("compression should work");
        let decompressed = decompress_chunk(&compressed).expect("decompression should work");
        assert_eq!(decompressed, data);
    }

    /// Byte-exactness certification for the `read::ZlibDecoder` ->
    /// `bufread::ZlibDecoder` swap in [`decompress_chunk`]. Both decoders run the
    /// identical inflate algorithm; only the input buffering differs (the `read`
    /// variant wraps the slice in an extra 32 KiB `BufReader`). This asserts the
    /// production `bufread` output is byte-identical to the legacy `read` reference
    /// across empty, small, boundary, and near-limit frame sizes and several
    /// content patterns, so the allocation-removal lever cannot alter any output.
    #[test]
    fn decompress_bufread_matches_read_reference_byte_exact() {
        use std::io::Read as _;

        fn decompress_read_reference(input: &[u8]) -> Vec<u8> {
            let mut decoder = flate2::read::ZlibDecoder::new(input);
            let mut out = Vec::new();
            decoder.read_to_end(&mut out).expect("reference inflate");
            out
        }

        // Sizes span 0, 1, byte/kib boundaries, a mulaw chunk, and just under the
        // 80 000-byte decompression-bomb limit.
        let sizes = [0usize, 1, 15, 16, 17, 160, 1600, 8192, 8193, 40_000, 79_999];
        for &n in &sizes {
            for pattern in 0u8..4 {
                let data: Vec<u8> = (0..n)
                    .map(|i| match pattern {
                        0 => 0,                                // all-zero (highly compressible)
                        1 => 0xAB,                             // constant
                        2 => (i % 251) as u8,                  // strided
                        _ => (i.wrapping_mul(2654435761)) as u8, // pseudo-random
                    })
                    .collect();
                let compressed = compress_chunk(&data).expect("compress");
                let production = decompress_chunk(&compressed).expect("bufread decompress");
                let reference = decompress_read_reference(&compressed);
                assert_eq!(
                    production, reference,
                    "bufread != read reference at size {n}, pattern {pattern}"
                );
                assert_eq!(production, data, "roundtrip failed at size {n}, pattern {pattern}");
            }
        }
    }

    #[test]
    fn ensure_parent_dir_ignores_empty_parent() {
        let path = Path::new("out.wav");
        ensure_parent_dir(path).expect("empty parent should be ignored");
    }

    #[test]
    fn parses_frame_lines() {
        let payload = serde_json::to_string(&TtyAudioFrame {
            protocol_version: SUPPORTED_PROTOCOL_VERSION,
            seq: 0,
            codec: CODEC_MULAW_ZLIB_B64.to_owned(),
            sample_rate_hz: 8_000,
            channels: 1,
            payload_b64: "abc".to_owned(),
            crc32: None,
            payload_sha256: None,
        })
        .expect("frame serialization should work");

        let content = format!("{payload}\n");
        let mut input = content.as_bytes();
        let frames = parse_frames(&mut input).expect("frame parsing should work");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].seq, 0);
    }

    #[test]
    fn crc32_field_is_optional_in_deserialization() {
        let json = r#"{"seq":0,"codec":"mulaw+zlib+b64","sample_rate_hz":8000,"channels":1,"payload_b64":"abc"}"#;
        let frame: TtyAudioFrame = serde_json::from_str(json).expect("should deserialize");
        assert!(frame.crc32.is_none());
        assert_eq!(frame.protocol_version, DEFAULT_PROTOCOL_VERSION);
    }

    #[test]
    fn crc32_field_roundtrips_when_present() {
        let frame = make_frame(0, b"test data");
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyAudioFrame = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.crc32, frame.crc32);
        assert!(parsed.crc32.is_some());
        assert_eq!(parsed.payload_sha256, frame.payload_sha256);
        assert!(parsed.payload_sha256.is_some());
    }

    #[test]
    fn decode_sequential_frames_no_gaps() {
        let frames = vec![
            make_frame(0, b"chunk-zero"),
            make_frame(1, b"chunk-one"),
            make_frame(2, b"chunk-two"),
        ];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode");
        assert_eq!(report.frames_decoded, 3);
        assert!(report.gaps.is_empty());
        assert!(report.duplicates.is_empty());
        assert!(report.integrity_failures.is_empty());
        assert!(report.dropped_frames.is_empty());
        assert_eq!(report.recovery_policy, DecodeRecoveryPolicy::FailClosed);
        assert_eq!(raw, b"chunk-zerochunk-onechunk-two");
    }

    #[test]
    fn decode_out_of_order_frames_succeeds_after_sort() {
        // The decoder sorts frames by sequence number before processing,
        // so out-of-order input [2, 0, 1] becomes [0, 1, 2] and decodes normally.
        let frames = vec![
            make_frame(2, b"chunk-two"),
            make_frame(0, b"chunk-zero"),
            make_frame(1, b"chunk-one"),
        ];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let (report, raw) =
            decode_frames_to_raw(&mut reader).expect("sorted decode should succeed");
        assert_eq!(report.frames_decoded, 3);
        assert_eq!(raw, b"chunk-zerochunk-onechunk-two");
    }

    #[test]
    fn decode_detects_sequence_gaps_as_error() {
        let frames = vec![make_frame(0, b"first"), make_frame(3, b"fourth")];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let error = decode_frames_to_raw(&mut reader).expect_err("gap should fail");
        assert!(
            error
                .to_string()
                .contains("missing tty-audio frame sequence")
        );
    }

    #[test]
    fn decode_detects_duplicate_frames_as_error() {
        let frames = vec![
            make_frame(0, b"first"),
            make_frame(1, b"second"),
            make_frame(1, b"second-dup"),
        ];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let error = decode_frames_to_raw(&mut reader).expect_err("duplicate should fail");
        assert!(
            error
                .to_string()
                .contains("duplicate tty-audio frame sequence")
        );
    }

    #[test]
    fn decode_detects_integrity_failure_as_error() {
        let mut frame = make_frame(0, b"original data");
        // Corrupt the CRC to simulate transmission error.
        frame.crc32 = Some(frame.crc32.unwrap() ^ 0xDEADBEEF);

        let ndjson = frames_to_ndjson(&[frame]);
        let mut reader = ndjson.as_bytes();

        let error = decode_frames_to_raw(&mut reader).expect_err("crc mismatch should fail");
        assert!(error.to_string().contains("tty-audio CRC mismatch"));
    }

    #[test]
    fn decode_skip_missing_policy_recovers_gap() {
        let frames = vec![
            make_frame(0, b"first"),
            make_frame(2, b"third"),
            make_frame(3, b"fourth"),
        ];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let (report, raw) =
            decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
                .expect("skip missing should recover");
        assert_eq!(report.frames_decoded, 3);
        assert_eq!(report.gaps.len(), 1);
        assert!(report.duplicates.is_empty());
        assert!(report.integrity_failures.is_empty());
        assert!(report.dropped_frames.is_empty());
        assert_eq!(raw, b"firstthirdfourth");
    }

    #[test]
    fn decode_skip_missing_policy_drops_corrupt_frame_and_continues() {
        let mut bad = make_frame(1, b"bad");
        bad.payload_sha256 = Some("deadbeef".to_owned());
        let frames = vec![make_frame(0, b"ok0"), bad, make_frame(2, b"ok2")];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let (report, raw) =
            decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
                .expect("skip missing should recover");
        assert_eq!(report.frames_decoded, 2);
        assert_eq!(report.integrity_failures, vec![1]);
        assert_eq!(report.dropped_frames, vec![1]);
        assert_eq!(raw, b"ok0ok2");
    }

    #[test]
    fn decode_passes_integrity_when_crc_correct() {
        let frames = vec![make_frame(0, b"valid data")];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let (report, _raw) = decode_frames_to_raw(&mut reader).expect("decode");
        assert!(report.integrity_failures.is_empty());
    }

    #[test]
    fn decode_empty_input_returns_error() {
        let ndjson = "";
        let mut reader = ndjson.as_bytes();

        let result = decode_frames_to_raw(&mut reader);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no tty-audio"));
    }

    #[test]
    fn decode_detects_sha256_integrity_failure_as_error() {
        let mut frame = make_frame(0, b"original data");
        frame.payload_sha256 = Some("deadbeef".to_owned());

        let ndjson = frames_to_ndjson(&[frame]);
        let mut reader = ndjson.as_bytes();

        let error = decode_frames_to_raw(&mut reader).expect_err("sha mismatch should fail");
        assert!(error.to_string().contains("tty-audio SHA-256 mismatch"));
    }

    #[test]
    fn decode_rejects_unsupported_protocol_version() {
        let mut frame = make_frame(0, b"chunk");
        frame.protocol_version = 99;

        let ndjson = frames_to_ndjson(&[frame]);
        let mut reader = ndjson.as_bytes();
        let error = decode_frames_to_raw(&mut reader).expect_err("unsupported protocol");
        assert!(
            error
                .to_string()
                .contains("unsupported tty-audio protocol_version")
        );
    }

    #[test]
    fn decode_rejects_unsupported_codec() {
        let mut frame = make_frame(0, b"chunk");
        frame.codec = "pcm16".to_owned();

        let ndjson = frames_to_ndjson(&[frame]);
        let mut reader = ndjson.as_bytes();
        let error = decode_frames_to_raw(&mut reader).expect_err("unsupported codec");
        assert!(error.to_string().contains("unsupported tty-audio codec"));
    }

    #[test]
    fn crc32_is_deterministic() {
        let data = b"deterministic test data";
        let crc1 = crc32_of(data);
        let crc2 = crc32_of(data);
        assert_eq!(crc1, crc2);
        assert_ne!(crc1, 0);
    }

    // --- K14.7: Robust-mode protocol tests ---

    use super::{
        RetransmitRange, TtyControlFrame, negotiate_version, retransmit_candidates,
        retransmit_plan_from_report,
    };

    #[test]
    fn handshake_frame_serializes_and_deserializes() {
        let frame = TtyControlFrame::Handshake {
            min_version: MIN_PROTOCOL_VERSION,
            max_version: SUPPORTED_PROTOCOL_VERSION,
            supported_codecs: vec!["mulaw+zlib+b64".to_owned()],
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");

        match parsed {
            TtyControlFrame::Handshake {
                min_version,
                max_version,
                supported_codecs,
            } => {
                assert_eq!(min_version, MIN_PROTOCOL_VERSION);
                assert_eq!(max_version, SUPPORTED_PROTOCOL_VERSION);
                assert_eq!(supported_codecs.len(), 1);
            }
            other => assert!(
                matches!(other, TtyControlFrame::Handshake { .. }),
                "expected Handshake variant"
            ),
        }
    }

    #[test]
    fn handshake_ack_frame_round_trips() {
        let frame = TtyControlFrame::HandshakeAck {
            negotiated_version: 1,
            negotiated_codec: "mulaw+zlib+b64".to_owned(),
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");

        match parsed {
            TtyControlFrame::HandshakeAck {
                negotiated_version,
                negotiated_codec,
            } => {
                assert_eq!(negotiated_version, 1);
                assert_eq!(negotiated_codec, "mulaw+zlib+b64");
            }
            other => assert!(
                matches!(other, TtyControlFrame::HandshakeAck { .. }),
                "expected HandshakeAck variant"
            ),
        }
    }

    #[test]
    fn retransmit_request_frame_round_trips() {
        let frame = TtyControlFrame::RetransmitRequest {
            sequences: vec![3, 5, 7],
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");

        match parsed {
            TtyControlFrame::RetransmitRequest { sequences } => {
                assert_eq!(sequences, vec![3, 5, 7]);
            }
            other => assert!(
                matches!(other, TtyControlFrame::RetransmitRequest { .. }),
                "expected RetransmitRequest variant"
            ),
        }
    }

    #[test]
    fn retransmit_response_frame_round_trips() {
        let frame = TtyControlFrame::RetransmitResponse {
            sequences: vec![10, 11],
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");

        match parsed {
            TtyControlFrame::RetransmitResponse { sequences } => {
                assert_eq!(sequences, vec![10, 11]);
            }
            other => assert!(
                matches!(other, TtyControlFrame::RetransmitResponse { .. }),
                "expected RetransmitResponse variant"
            ),
        }
    }

    #[test]
    fn ack_frame_round_trips() {
        let frame = TtyControlFrame::Ack { up_to_seq: 42 };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");

        match parsed {
            TtyControlFrame::Ack { up_to_seq } => {
                assert_eq!(up_to_seq, 42);
            }
            other => assert!(
                matches!(other, TtyControlFrame::Ack { .. }),
                "expected Ack variant"
            ),
        }
    }

    #[test]
    fn backpressure_frame_round_trips() {
        let frame = TtyControlFrame::Backpressure {
            remaining_capacity: 100,
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");

        match parsed {
            TtyControlFrame::Backpressure { remaining_capacity } => {
                assert_eq!(remaining_capacity, 100);
            }
            other => assert!(
                matches!(other, TtyControlFrame::Backpressure { .. }),
                "expected Backpressure variant"
            ),
        }
    }

    #[test]
    fn negotiate_version_overlapping_ranges() {
        // Local [1,2], Remote [1,3] → negotiate 2
        assert_eq!(negotiate_version(1, 2, 1, 3), Ok(2));
    }

    #[test]
    fn negotiate_version_exact_match() {
        assert_eq!(negotiate_version(1, 1, 1, 1), Ok(1));
    }

    #[test]
    fn negotiate_version_no_overlap() {
        let result = negotiate_version(1, 2, 3, 4);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .contains("no compatible protocol version")
        );
    }

    #[test]
    fn negotiate_version_picks_highest_compatible() {
        // Local [1,5], Remote [2,4] → negotiate 4
        assert_eq!(negotiate_version(1, 5, 2, 4), Ok(4));
    }

    #[test]
    fn retransmit_candidates_from_gaps() {
        let report = super::DecodeReport {
            protocol_version: DEFAULT_PROTOCOL_VERSION,
            frames_decoded: 5,
            gaps: vec![
                super::SequenceGap {
                    expected: 1,
                    got: 3,
                },
                super::SequenceGap {
                    expected: 5,
                    got: 8,
                },
            ],
            duplicates: vec![],
            integrity_failures: vec![],
            dropped_frames: vec![],
            highest_contiguous_seq: None,
            recovery_policy: DecodeRecoveryPolicy::SkipMissing,
        };

        let candidates = retransmit_candidates(&report);
        assert_eq!(candidates, vec![1, 2, 5, 6, 7]);
    }

    #[test]
    fn retransmit_candidates_empty_when_no_gaps() {
        let report = super::DecodeReport {
            protocol_version: DEFAULT_PROTOCOL_VERSION,
            frames_decoded: 3,
            gaps: vec![],
            duplicates: vec![],
            integrity_failures: vec![],
            dropped_frames: vec![],
            highest_contiguous_seq: Some(2),
            recovery_policy: DecodeRecoveryPolicy::FailClosed,
        };

        let candidates = retransmit_candidates(&report);
        assert!(candidates.is_empty());
    }

    #[test]
    fn decode_accepts_handshake_and_interleaved_control_frames() {
        let handshake = serde_json::to_string(&TtyControlFrame::Handshake {
            min_version: MIN_PROTOCOL_VERSION,
            max_version: SUPPORTED_PROTOCOL_VERSION,
            supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned()],
        })
        .expect("serialize handshake");
        let ack = serde_json::to_string(&TtyControlFrame::Ack { up_to_seq: 0 }).expect("ack");
        let backpressure = serde_json::to_string(&TtyControlFrame::Backpressure {
            remaining_capacity: 12,
        })
        .expect("backpressure");
        let frames = [make_frame(0, b"chunk-zero"), make_frame(1, b"chunk-one")];
        let ndjson = format!(
            "{handshake}\n{}\n{ack}\n{backpressure}\n{}\n",
            serde_json::to_string(&frames[0]).expect("serialize"),
            serde_json::to_string(&frames[1]).expect("serialize")
        );
        let mut reader = ndjson.as_bytes();

        let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode");
        assert_eq!(report.frames_decoded, 2);
        assert_eq!(raw, b"chunk-zerochunk-one");
    }

    #[test]
    fn decode_rejects_non_handshake_control_as_first_frame() {
        let first = serde_json::to_string(&TtyControlFrame::Ack { up_to_seq: 3 }).expect("ack");
        let second = serde_json::to_string(&make_frame(0, b"chunk")).expect("frame");
        let ndjson = format!("{first}\n{second}\n");
        let mut reader = ndjson.as_bytes();

        let error = decode_frames_to_raw(&mut reader).expect_err("control-first should fail");
        assert!(error.to_string().contains("received before handshake"));
    }

    #[test]
    fn decode_rejects_duplicate_handshake() {
        let handshake = serde_json::to_string(&TtyControlFrame::Handshake {
            min_version: MIN_PROTOCOL_VERSION,
            max_version: SUPPORTED_PROTOCOL_VERSION,
            supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned()],
        })
        .expect("serialize handshake");
        let frame = serde_json::to_string(&make_frame(0, b"chunk")).expect("serialize frame");
        let ndjson = format!("{handshake}\n{handshake}\n{frame}\n");
        let mut reader = ndjson.as_bytes();

        let error = decode_frames_to_raw(&mut reader).expect_err("duplicate handshake must fail");
        assert!(error.to_string().contains("duplicate tty-audio handshake"));
    }

    #[test]
    fn decode_rejects_handshake_without_supported_codec() {
        let handshake = serde_json::to_string(&TtyControlFrame::Handshake {
            min_version: MIN_PROTOCOL_VERSION,
            max_version: SUPPORTED_PROTOCOL_VERSION,
            supported_codecs: vec!["pcm16".to_owned()],
        })
        .expect("serialize handshake");
        let frame = serde_json::to_string(&make_frame(0, b"chunk")).expect("serialize frame");
        let ndjson = format!("{handshake}\n{frame}\n");
        let mut reader = ndjson.as_bytes();

        let error = decode_frames_to_raw(&mut reader).expect_err("codec mismatch should fail");
        assert!(error.to_string().contains("no compatible tty-audio codec"));
    }

    #[test]
    fn decode_accepts_session_close_with_matching_last_seq() {
        let handshake = serde_json::to_string(&TtyControlFrame::Handshake {
            min_version: MIN_PROTOCOL_VERSION,
            max_version: SUPPORTED_PROTOCOL_VERSION,
            supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned()],
        })
        .expect("serialize handshake");
        let frames = [make_frame(0, b"chunk-zero"), make_frame(1, b"chunk-one")];
        let close = serde_json::to_string(&TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Normal,
            last_data_seq: Some(1),
        })
        .expect("serialize close");
        let ndjson = format!(
            "{handshake}\n{}\n{}\n{close}\n",
            serde_json::to_string(&frames[0]).expect("serialize"),
            serde_json::to_string(&frames[1]).expect("serialize")
        );
        let mut reader = ndjson.as_bytes();

        let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode");
        assert_eq!(report.frames_decoded, 2);
        assert_eq!(raw, b"chunk-zerochunk-one");
    }

    #[test]
    fn decode_rejects_session_close_with_mismatched_last_seq() {
        let handshake = serde_json::to_string(&TtyControlFrame::Handshake {
            min_version: MIN_PROTOCOL_VERSION,
            max_version: SUPPORTED_PROTOCOL_VERSION,
            supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned()],
        })
        .expect("serialize handshake");
        let frame = make_frame(0, b"chunk-zero");
        let close = serde_json::to_string(&TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Normal,
            last_data_seq: Some(9),
        })
        .expect("serialize close");
        let ndjson = format!(
            "{handshake}\n{}\n{close}\n",
            serde_json::to_string(&frame).expect("serialize")
        );
        let mut reader = ndjson.as_bytes();

        let error = decode_frames_to_raw(&mut reader).expect_err("mismatch should fail");
        assert!(
            error
                .to_string()
                .contains("session_close last_data_seq mismatch")
        );
    }

    #[test]
    fn decode_accepts_duplicate_session_close_when_identical() {
        let handshake = serde_json::to_string(&TtyControlFrame::Handshake {
            min_version: MIN_PROTOCOL_VERSION,
            max_version: SUPPORTED_PROTOCOL_VERSION,
            supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned()],
        })
        .expect("serialize handshake");
        let frame = make_frame(0, b"chunk-zero");
        let close = TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Normal,
            last_data_seq: Some(0),
        };
        let close_json = serde_json::to_string(&close).expect("serialize close");
        let ndjson = format!(
            "{handshake}\n{}\n{close_json}\n{close_json}\n",
            serde_json::to_string(&frame).expect("serialize")
        );
        let mut reader = ndjson.as_bytes();

        let (report, _raw) = decode_frames_to_raw(&mut reader).expect("decode");
        assert_eq!(report.frames_decoded, 1);
    }

    #[test]
    fn parse_frame_line_classifies_control_and_audio_entries() {
        let control_line = serde_json::to_string(&TtyControlFrame::Ack { up_to_seq: 5 }).unwrap();
        let audio_line = serde_json::to_string(&make_frame(0, b"chunk")).unwrap();

        let control = parse_frame_line(&control_line).expect("control parse");
        assert!(matches!(control, super::FrameLine::Control(_)));

        let audio = parse_frame_line(&audio_line).expect("audio parse");
        assert!(matches!(audio, super::FrameLine::Audio(_)));
    }

    #[test]
    fn parse_audio_frames_for_decode_legacy_stream_without_handshake_still_works() {
        let frames = vec![make_frame(0, b"a"), make_frame(1, b"b")];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();
        let parsed = parse_audio_frames_for_decode(&mut reader).expect("legacy parse");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].seq, 0);
        assert_eq!(parsed[1].seq, 1);
    }

    #[test]
    fn parse_audio_frames_for_decode_rejects_mixed_versions_without_handshake() {
        let mut frame_v1 = make_frame(0, b"a");
        frame_v1.protocol_version = MIN_PROTOCOL_VERSION;
        let mut frame_v2 = make_frame(1, b"b");
        frame_v2.protocol_version = MIN_PROTOCOL_VERSION + 1;
        let ndjson = frames_to_ndjson(&[frame_v1, frame_v2]);
        let mut reader = ndjson.as_bytes();
        let err =
            parse_audio_frames_for_decode(&mut reader).expect_err("mixed versions should fail");
        assert!(
            err.to_string().contains("protocol_version"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn emit_control_frame_to_writer_outputs_single_ndjson_line() {
        let mut out = Vec::new();
        emit_control_frame_to_writer(&mut out, &TtyControlFrame::Ack { up_to_seq: 9 })
            .expect("emit");

        let text = String::from_utf8(out).expect("utf8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1);
        let parsed: serde_json::Value = serde_json::from_str(lines[0]).expect("json");
        assert_eq!(parsed["frame_type"], "ack");
        assert_eq!(parsed["up_to_seq"], 9);
    }

    #[test]
    fn emit_control_frame_to_writer_matches_serde_json_line() {
        let frames = [
            TtyControlFrame::Handshake {
                min_version: MIN_PROTOCOL_VERSION,
                max_version: SUPPORTED_PROTOCOL_VERSION,
                supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned(), "pcm16".to_owned()],
            },
            TtyControlFrame::Ack { up_to_seq: 9 },
            TtyControlFrame::Backpressure {
                remaining_capacity: 50,
            },
            TtyControlFrame::HandshakeAck {
                negotiated_version: 1,
                negotiated_codec: CODEC_MULAW_ZLIB_B64.to_owned(),
            },
            TtyControlFrame::RetransmitRequest {
                sequences: vec![2, 5, 8],
            },
            TtyControlFrame::RetransmitResponse {
                sequences: vec![13, 21],
            },
            TtyControlFrame::SessionClose {
                reason: SessionCloseReason::PeerRequested,
                last_data_seq: None,
            },
            TtyControlFrame::SessionClose {
                reason: SessionCloseReason::Normal,
                last_data_seq: Some(34),
            },
        ];

        for frame in frames {
            let mut out = Vec::new();
            emit_control_frame_to_writer(&mut out, &frame).expect("emit");

            let mut expected = serde_json::to_string(&frame).expect("serialize");
            expected.push('\n');
            assert_eq!(out, expected.into_bytes());
        }
    }

    #[test]
    fn emit_retransmit_loop_from_reader_emits_requests_then_response() {
        let mut corrupt = make_frame(4, b"bad-frame");
        corrupt.payload_sha256 = Some("deadbeef".to_owned());
        let frames = vec![
            make_frame(0, b"chunk-zero"),
            make_frame(3, b"chunk-three"),
            corrupt,
        ];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();
        let mut out = Vec::new();

        emit_retransmit_loop_from_reader(
            &mut reader,
            DecodeRecoveryPolicy::SkipMissing,
            2,
            &mut out,
        )
        .expect("emit loop");

        let lines = String::from_utf8(out).expect("utf8");
        let frames: Vec<TtyControlFrame> = lines
            .lines()
            .map(|line| serde_json::from_str(line).expect("control frame"))
            .collect();
        assert_eq!(frames.len(), 3, "2 requests + 1 response");
        assert!(matches!(
            &frames[0],
            TtyControlFrame::RetransmitRequest { sequences } if sequences == &vec![1,2,4]
        ));
        assert!(matches!(
            &frames[1],
            TtyControlFrame::RetransmitRequest { sequences } if sequences == &vec![1,2,4]
        ));
        assert!(matches!(
            &frames[2],
            TtyControlFrame::RetransmitResponse { sequences } if sequences == &vec![1,2,4]
        ));
    }

    #[test]
    fn emit_retransmit_loop_from_reader_emits_ack_when_no_missing_frames() {
        let frames = vec![make_frame(0, b"a"), make_frame(1, b"b")];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();
        let mut out = Vec::new();

        emit_retransmit_loop_from_reader(
            &mut reader,
            DecodeRecoveryPolicy::SkipMissing,
            4,
            &mut out,
        )
        .expect("emit loop");

        let lines = String::from_utf8(out).expect("utf8");
        let parsed: TtyControlFrame = serde_json::from_str(lines.trim()).expect("control frame");
        assert!(matches!(parsed, TtyControlFrame::Ack { up_to_seq: 1 }));
    }

    #[test]
    fn retransmit_plan_merges_gap_and_integrity_sequences_into_ranges() {
        let report = super::DecodeReport {
            protocol_version: DEFAULT_PROTOCOL_VERSION,
            frames_decoded: 6,
            gaps: vec![
                super::SequenceGap {
                    expected: 1,
                    got: 4,
                },
                super::SequenceGap {
                    expected: 8,
                    got: 9,
                },
            ],
            duplicates: vec![2],
            integrity_failures: vec![6, 7],
            dropped_frames: vec![2, 6, 7],
            highest_contiguous_seq: Some(0),
            recovery_policy: DecodeRecoveryPolicy::SkipMissing,
        };

        let plan = retransmit_plan_from_report(&report);
        assert_eq!(plan.protocol_version, DEFAULT_PROTOCOL_VERSION);
        assert_eq!(plan.requested_sequences, vec![1, 2, 3, 6, 7, 8]);
        assert_eq!(
            plan.requested_ranges,
            vec![
                RetransmitRange {
                    start_seq: 1,
                    end_seq: 3
                },
                RetransmitRange {
                    start_seq: 6,
                    end_seq: 8
                },
            ]
        );
        assert_eq!(plan.gap_count, 2);
        assert_eq!(plan.integrity_failure_count, 2);
        assert_eq!(plan.dropped_frame_count, 3);
    }

    // --- Additional edge case tests ---

    #[test]
    fn sha256_hex_produces_64_char_lowercase_hex() {
        let hash = sha256_hex(b"hello world");
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        // Known SHA-256 of "hello world"
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn sha256_hex_empty_input() {
        let hash = sha256_hex(b"");
        assert_eq!(hash.len(), 64);
        // Known SHA-256 of empty string
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_hex_is_deterministic() {
        let data = b"deterministic test";
        assert_eq!(sha256_hex(data), sha256_hex(data));
    }

    #[test]
    fn decode_report_construction_and_field_access() {
        let report = super::DecodeReport {
            protocol_version: DEFAULT_PROTOCOL_VERSION,
            frames_decoded: 10,
            gaps: vec![super::SequenceGap {
                expected: 2,
                got: 4,
            }],
            duplicates: vec![5, 5],
            integrity_failures: vec![7],
            dropped_frames: vec![5, 7],
            highest_contiguous_seq: Some(1),
            recovery_policy: DecodeRecoveryPolicy::SkipMissing,
        };
        assert_eq!(report.frames_decoded, 10);
        assert_eq!(report.gaps.len(), 1);
        assert_eq!(report.gaps[0].expected, 2);
        assert_eq!(report.gaps[0].got, 4);
        assert_eq!(report.duplicates, vec![5, 5]);
        assert_eq!(report.integrity_failures, vec![7]);
        assert_eq!(report.dropped_frames, vec![5, 7]);
        assert_eq!(report.recovery_policy, DecodeRecoveryPolicy::SkipMissing);
    }

    #[test]
    fn parse_frames_skips_blank_lines() {
        let frame = make_frame(0, b"data");
        let json = serde_json::to_string(&frame).expect("serialize");
        let content = format!("\n\n{json}\n\n\n");
        let mut reader = content.as_bytes();
        let frames = parse_frames(&mut reader).expect("parse");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].seq, 0);
    }

    #[test]
    fn parse_frames_empty_input_returns_empty_vec() {
        let content = "";
        let mut reader = content.as_bytes();
        let frames = parse_frames(&mut reader).expect("parse");
        assert!(frames.is_empty());
    }

    #[test]
    fn parse_frames_only_whitespace_returns_empty_vec() {
        let content = "   \n  \n\n";
        let mut reader = content.as_bytes();
        let frames = parse_frames(&mut reader).expect("parse");
        assert!(frames.is_empty());
    }

    #[test]
    fn control_frame_retransmit_request_empty_sequences() {
        let frame = TtyControlFrame::RetransmitRequest { sequences: vec![] };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            TtyControlFrame::RetransmitRequest { sequences } => {
                assert!(sequences.is_empty());
            }
            other => assert!(
                matches!(other, TtyControlFrame::RetransmitRequest { .. }),
                "expected RetransmitRequest"
            ),
        }
    }

    #[test]
    fn control_frame_handshake_empty_codecs() {
        let frame = TtyControlFrame::Handshake {
            min_version: MIN_PROTOCOL_VERSION,
            max_version: SUPPORTED_PROTOCOL_VERSION,
            supported_codecs: vec![],
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            TtyControlFrame::Handshake {
                supported_codecs, ..
            } => {
                assert!(supported_codecs.is_empty());
            }
            other => assert!(
                matches!(other, TtyControlFrame::Handshake { .. }),
                "expected Handshake"
            ),
        }
    }

    #[test]
    fn control_frame_backpressure_zero_capacity() {
        let frame = TtyControlFrame::Backpressure {
            remaining_capacity: 0,
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            TtyControlFrame::Backpressure { remaining_capacity } => {
                assert_eq!(remaining_capacity, 0);
            }
            other => assert!(
                matches!(other, TtyControlFrame::Backpressure { .. }),
                "expected Backpressure"
            ),
        }
    }

    #[test]
    fn control_frame_ack_zero_seq() {
        let frame = TtyControlFrame::Ack { up_to_seq: 0 };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            TtyControlFrame::Ack { up_to_seq } => {
                assert_eq!(up_to_seq, 0);
            }
            other => assert!(matches!(other, TtyControlFrame::Ack { .. }), "expected Ack"),
        }
    }

    #[test]
    fn decode_rejects_unsupported_sample_rate() {
        let mut frame = make_frame(0, b"chunk");
        frame.sample_rate_hz = 44_100;

        let ndjson = frames_to_ndjson(&[frame]);
        let mut reader = ndjson.as_bytes();
        let error = decode_frames_to_raw(&mut reader).expect_err("should reject");
        assert!(error.to_string().contains("unsupported tty-audio shape"));
    }

    #[test]
    fn decode_rejects_multi_channel() {
        let mut frame = make_frame(0, b"chunk");
        frame.channels = 2;

        let ndjson = frames_to_ndjson(&[frame]);
        let mut reader = ndjson.as_bytes();
        let error = decode_frames_to_raw(&mut reader).expect_err("should reject");
        assert!(error.to_string().contains("unsupported tty-audio shape"));
    }

    #[test]
    fn mulaw_chunk_size_clamped_at_upper_bound() {
        // 10_000ms > 5_000ms cap → 5_000 * 8 = 40_000
        assert_eq!(mulaw_chunk_size(10_000), 40_000);
    }

    #[test]
    fn compress_decompress_empty_data() {
        let compressed = compress_chunk(b"").expect("compress empty");
        let decompressed = decompress_chunk(&compressed).expect("decompress empty");
        assert!(decompressed.is_empty());
    }

    #[test]
    fn crc32_of_empty_data() {
        let crc = crc32_of(b"");
        assert_eq!(crc, 0);
    }

    #[test]
    fn crc32_of_different_data_produces_different_values() {
        let crc1 = crc32_of(b"hello");
        let crc2 = crc32_of(b"world");
        assert_ne!(crc1, crc2);
    }

    #[test]
    fn default_protocol_version_is_supported() {
        assert_eq!(super::default_protocol_version(), DEFAULT_PROTOCOL_VERSION);
    }

    #[test]
    fn skip_missing_policy_drops_duplicate_and_continues() {
        let frames = vec![
            make_frame(0, b"first"),
            make_frame(1, b"second"),
            make_frame(1, b"second-dup"),
            make_frame(2, b"third"),
        ];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let (report, raw) =
            decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
                .expect("skip-missing should recover from duplicates");
        assert_eq!(report.frames_decoded, 3);
        assert_eq!(report.duplicates, vec![1]);
        assert_eq!(report.dropped_frames, vec![1]);
        assert_eq!(raw, b"firstsecondthird");
    }

    #[test]
    fn skip_missing_policy_handles_out_of_order_input_via_sort() {
        // Frames arrive as [0, 2, 1, 3] but the decoder sorts by seq first,
        // producing [0, 1, 2, 3] — no gaps, no duplicates, all frames decoded.
        let frames = vec![
            make_frame(0, b"a"),
            make_frame(2, b"c"),
            make_frame(1, b"b"),
            make_frame(3, b"d"),
        ];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let (report, raw) =
            decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
                .expect("sorted decode should succeed");
        assert_eq!(report.gaps.len(), 0, "no gaps after sorting");
        assert!(report.duplicates.is_empty(), "no duplicates after sorting");
        assert_eq!(report.frames_decoded, 4);
        assert_eq!(raw, b"abcd");
    }

    #[test]
    fn skip_missing_policy_drops_bad_base64_and_continues() {
        let good0 = make_frame(0, b"ok");
        let mut bad1 = make_frame(1, b"will-be-corrupted");
        bad1.payload_b64 = "!!!not-valid-base64!!!".to_owned();
        bad1.crc32 = None;
        bad1.payload_sha256 = None;
        let good2 = make_frame(2, b"also-ok");

        let frames = vec![good0, bad1, good2];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let (report, raw) =
            decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
                .expect("skip-missing should recover from bad base64");
        assert_eq!(report.frames_decoded, 2);
        assert_eq!(report.integrity_failures, vec![1]);
        assert_eq!(report.dropped_frames, vec![1]);
        assert_eq!(raw, b"okalso-ok");
    }

    #[test]
    fn frame_without_crc_or_sha_still_decodes() {
        let data = b"no-integrity-fields";
        let compressed = compress_chunk(data).expect("compress");
        let frame = TtyAudioFrame {
            protocol_version: SUPPORTED_PROTOCOL_VERSION,
            seq: 0,
            codec: CODEC_MULAW_ZLIB_B64.to_owned(),
            sample_rate_hz: 8_000,
            channels: 1,
            payload_b64: STANDARD_NO_PAD.encode(compressed),
            crc32: None,
            payload_sha256: None,
        };
        let ndjson = frames_to_ndjson(&[frame]);
        let mut reader = ndjson.as_bytes();

        let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode without integrity");
        assert_eq!(report.frames_decoded, 1);
        assert!(report.integrity_failures.is_empty());
        assert_eq!(raw, data);
    }

    #[test]
    fn mulaw_chunk_size_max_u32_clamped() {
        // u32::MAX ms should be clamped to 5000ms → 40_000 bytes
        assert_eq!(mulaw_chunk_size(u32::MAX), 40_000);
    }

    #[test]
    fn mulaw_chunk_size_zero_clamped_to_minimum() {
        // 0ms should be clamped to 20ms → 160 bytes
        assert_eq!(mulaw_chunk_size(0), 160);
    }

    #[test]
    fn skip_missing_crc_corrupt_drops_and_continues() {
        let good0 = make_frame(0, b"ok");
        let mut bad1 = make_frame(1, b"will-corrupt-crc");
        bad1.crc32 = Some(bad1.crc32.unwrap() ^ 0xFFFF_FFFF);
        bad1.payload_sha256 = None; // remove SHA so CRC check triggers first
        let good2 = make_frame(2, b"also-ok");

        let frames = vec![good0, bad1, good2];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let (report, raw) =
            decode_frames_to_raw_with_policy(&mut reader, DecodeRecoveryPolicy::SkipMissing)
                .expect("skip-missing should recover from CRC corruption");
        assert_eq!(report.integrity_failures, vec![1]);
        assert_eq!(report.dropped_frames, vec![1]);
        assert_eq!(raw, b"okalso-ok");
    }

    // ---------------------------------------------------------------
    // Comprehensive TTY audio codec roundtrip and edge-case tests
    // ---------------------------------------------------------------

    // -- mu-law helpers (ITU-T G.711) for in-process roundtrip tests --

    const MULAW_BIAS: i32 = 0x84; // 132
    const MULAW_MAX: i32 = 0x7FFF - MULAW_BIAS; // 32635 (clamp ceiling before bias addition)

    /// Encode a single 16-bit linear PCM sample to 8-bit mu-law.
    fn mulaw_encode_sample(sample: i16) -> u8 {
        let sign: i32;
        let mut pcm: i32 = sample as i32;

        if pcm < 0 {
            sign = 0x80;
            pcm = -pcm;
        } else {
            sign = 0;
        }

        if pcm > MULAW_MAX {
            pcm = MULAW_MAX;
        }
        pcm += MULAW_BIAS;

        let mut exponent: i32 = 7;
        let mut mask: i32 = 0x4000;
        while pcm & mask == 0 && exponent > 0 {
            exponent -= 1;
            mask >>= 1;
        }

        let mantissa = (pcm >> (exponent + 3)) & 0x0F;
        let byte = sign | (exponent << 4) | mantissa;
        !(byte as u8)
    }

    /// Decode a single 8-bit mu-law sample back to 16-bit linear PCM.
    fn mulaw_decode_sample(byte: u8) -> i16 {
        let complement = !byte as i32;
        let sign = complement & 0x80;
        let exponent = (complement >> 4) & 0x07;
        let mantissa = complement & 0x0F;

        let mut magnitude = ((mantissa << 1) | 0x21) << (exponent + 2);
        magnitude -= MULAW_BIAS;

        if sign != 0 {
            -magnitude as i16
        } else {
            magnitude as i16
        }
    }

    fn mulaw_encode_samples(samples: &[i16]) -> Vec<u8> {
        samples.iter().map(|&s| mulaw_encode_sample(s)).collect()
    }

    fn mulaw_decode_samples(encoded: &[u8]) -> Vec<i16> {
        encoded.iter().map(|&b| mulaw_decode_sample(b)).collect()
    }

    // -- Test 1: mu-law encode/decode roundtrip with a sine wave --

    #[test]
    fn mulaw_encode_decode_roundtrip() {
        // Generate a 100-sample 400 Hz sine wave at 8 kHz sample rate.
        let num_samples = 100;
        let amplitude: f64 = 16_000.0;
        let freq_hz = 400.0_f64;
        let sample_rate = 8_000.0_f64;

        let samples: Vec<i16> = (0..num_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (amplitude * (2.0 * std::f64::consts::PI * freq_hz * t).sin()) as i16
            })
            .collect();

        let encoded = mulaw_encode_samples(&samples);
        assert_eq!(encoded.len(), num_samples);

        let decoded = mulaw_decode_samples(&encoded);
        assert_eq!(decoded.len(), num_samples);

        // mu-law is lossy; allow tolerance of up to 512 per sample (typical
        // for low-amplitude segments; large amplitudes are much more precise).
        let max_err: i16 = 512;
        for (i, (&orig, &dec)) in samples.iter().zip(decoded.iter()).enumerate() {
            let diff = (orig as i32 - dec as i32).unsigned_abs() as i16;
            assert!(
                diff <= max_err,
                "sample {i}: original={orig}, decoded={dec}, diff={diff} exceeds tolerance {max_err}"
            );
        }
    }

    // -- Test 2: mu-law silence roundtrip --

    #[test]
    fn mulaw_silence_roundtrip() {
        let silence: Vec<i16> = vec![0; 256];
        let encoded = mulaw_encode_samples(&silence);
        let decoded = mulaw_decode_samples(&encoded);

        for (i, &sample) in decoded.iter().enumerate() {
            assert_eq!(
                sample, 0,
                "silence sample {i} decoded to {sample}, expected 0"
            );
        }
    }

    // -- Test 3: mu-law max amplitude roundtrip --

    #[test]
    fn mulaw_max_amplitude_roundtrip() {
        let extremes: Vec<i16> = vec![i16::MAX, i16::MIN + 1, i16::MAX, i16::MIN + 1];
        let encoded = mulaw_encode_samples(&extremes);
        let decoded = mulaw_decode_samples(&encoded);

        // mu-law clips at 32767; tolerance is wider at extreme values.
        let tolerance: i32 = 1024;
        for (i, (&orig, &dec)) in extremes.iter().zip(decoded.iter()).enumerate() {
            let diff = (orig as i32 - dec as i32).abs();
            assert!(
                diff <= tolerance,
                "extreme sample {i}: original={orig}, decoded={dec}, diff={diff} exceeds tolerance {tolerance}"
            );
            // Sign must be preserved.
            if orig > 0 {
                assert!(dec > 0, "positive sample {i} decoded to non-positive {dec}");
            }
            if orig < 0 {
                assert!(dec < 0, "negative sample {i} decoded to non-negative {dec}");
            }
        }
    }

    // -- Test 4: zlib compress/decompress roundtrip --

    #[test]
    fn zlib_compress_decompress_roundtrip() {
        // Arbitrary byte pattern: pseudo-random via simple LCG.
        let mut data = vec![0u8; 4096];
        let mut state: u32 = 0xDEAD_BEEF;
        for byte in &mut data {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (state >> 16) as u8;
        }

        let compressed = compress_chunk(&data).expect("compress arbitrary bytes");
        let decompressed = decompress_chunk(&compressed).expect("decompress arbitrary bytes");
        assert_eq!(decompressed, data, "zlib roundtrip must be lossless");
    }

    // -- Test 5: zlib empty input --

    #[test]
    fn zlib_empty_input() {
        let compressed = compress_chunk(b"").expect("compress empty");
        assert!(
            !compressed.is_empty(),
            "zlib header should be non-empty even for empty input"
        );
        let decompressed = decompress_chunk(&compressed).expect("decompress empty");
        assert!(
            decompressed.is_empty(),
            "decompressed empty input must be empty"
        );
    }

    // -- Test 6: base64 encode/decode roundtrip --

    #[test]
    fn base64_encode_decode_roundtrip() {
        // Generate pseudo-random bytes via simple LCG.
        let mut data = vec![0u8; 1024];
        let mut state: u32 = 0xCAFE_BABE;
        for byte in &mut data {
            state = state.wrapping_mul(1103515245).wrapping_add(1);
            *byte = (state >> 24) as u8;
        }

        let encoded = STANDARD_NO_PAD.encode(&data);
        let decoded = STANDARD_NO_PAD
            .decode(&encoded)
            .expect("base64 decode must succeed");
        assert_eq!(decoded, data, "base64 roundtrip must be lossless");
    }

    // -- Test 7: full frame encode/decode roundtrip --

    #[test]
    fn frame_encode_decode_roundtrip() {
        // Simulate mu-law encoded audio: 800 bytes ≈ 100ms at 8 kHz.
        let sample_count = 800;
        let raw_mulaw: Vec<u8> = (0..sample_count).map(|i| (i % 256) as u8).collect();

        let frame = make_frame(0, &raw_mulaw);
        let ndjson = frames_to_ndjson(&[frame]);
        let mut reader = ndjson.as_bytes();

        let (report, decoded_raw) =
            decode_frames_to_raw(&mut reader).expect("frame roundtrip decode");

        assert_eq!(report.frames_decoded, 1);
        assert!(report.gaps.is_empty());
        assert!(report.integrity_failures.is_empty());
        assert_eq!(
            decoded_raw.len(),
            raw_mulaw.len(),
            "sample count must match after roundtrip"
        );
        assert_eq!(decoded_raw, raw_mulaw, "frame roundtrip must be lossless");

        // SNR check: since the frame codec (zlib + base64) is lossless for the
        // mu-law byte stream, every byte must match exactly — infinite SNR.
        let mismatch_count = decoded_raw
            .iter()
            .zip(raw_mulaw.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatch_count, 0,
            "lossless codec must produce zero mismatches"
        );
    }

    // -- Test 8: frame NDJSON is valid JSON with required fields --

    #[test]
    fn frame_ndjson_is_valid_json() {
        let frame = make_frame(42, b"json-check-payload");
        let json_str = serde_json::to_string(&frame).expect("serialize frame");

        // Must parse as valid JSON.
        let value: serde_json::Value =
            serde_json::from_str(&json_str).expect("frame must be valid JSON");

        // Required fields.
        assert!(value.get("seq").is_some(), "missing 'seq' field");
        assert!(value.get("codec").is_some(), "missing 'codec' field");
        assert!(
            value.get("sample_rate_hz").is_some(),
            "missing 'sample_rate_hz' field"
        );
        assert!(value.get("channels").is_some(), "missing 'channels' field");
        assert!(
            value.get("payload_b64").is_some(),
            "missing 'payload_b64' field"
        );

        // Verify field values.
        assert_eq!(value["seq"].as_u64(), Some(42));
        assert_eq!(value["codec"].as_str(), Some(CODEC_MULAW_ZLIB_B64));
        assert_eq!(value["sample_rate_hz"].as_u64(), Some(8_000));
        assert_eq!(value["channels"].as_u64(), Some(1));
        assert!(
            value["payload_b64"].as_str().is_some_and(|s| !s.is_empty()),
            "payload_b64 must be a non-empty string"
        );

        // NDJSON: the serialized frame must be a single line (no embedded newlines).
        assert!(
            !json_str.contains('\n'),
            "NDJSON frame must not contain embedded newlines"
        );
    }

    // -- Test 9: frame decode rejects invalid codec --

    #[test]
    fn frame_decode_rejects_invalid_codec() {
        let mut frame = make_frame(0, b"codec-rejection-test");
        frame.codec = "opus+lz4+b64".to_owned();

        let ndjson = frames_to_ndjson(&[frame]);
        let mut reader = ndjson.as_bytes();

        let error =
            decode_frames_to_raw(&mut reader).expect_err("unknown codec should be rejected");
        let msg = error.to_string();
        assert!(
            msg.contains("unsupported tty-audio codec"),
            "error message should mention unsupported codec, got: {msg}"
        );
        assert!(
            msg.contains("opus+lz4+b64"),
            "error message should include the invalid codec name, got: {msg}"
        );
    }

    // -- Test 10: silence frame compresses well --

    #[test]
    fn silence_frame_compresses_well() {
        // 8000 bytes of mu-law silence (0xFF is mu-law silence).
        let silence = vec![0xFFu8; 8_000];
        let raw_size = silence.len();

        let compressed = compress_chunk(&silence).expect("compress silence");
        let compressed_size = compressed.len();

        let ratio = compressed_size as f64 / raw_size as f64;
        assert!(
            ratio < 0.50,
            "silence should compress to < 50% of raw size, got {compressed_size}/{raw_size} = {ratio:.2}"
        );

        // Also verify the full frame pipeline compresses well.
        let frame = make_frame(0, &silence);
        let json = serde_json::to_string(&frame).expect("serialize");
        // The base64-encoded payload should still be significantly smaller than
        // the raw data (base64 adds ~33% overhead but zlib compression of
        // constant data is extreme).
        let payload_len = frame.payload_b64.len();
        let b64_raw_len = raw_size.div_ceil(3) * 4; // base64 of uncompressed
        assert!(
            payload_len < b64_raw_len / 2,
            "compressed+b64 payload ({payload_len}) should be much smaller than raw b64 ({b64_raw_len}), json len = {}",
            json.len()
        );
    }

    // -- Test 11: multiple frames roundtrip with ordering --

    #[test]
    fn multiple_frames_roundtrip() {
        let frame_count = 10;
        let frames: Vec<TtyAudioFrame> = (0..frame_count)
            .map(|i| {
                let data: Vec<u8> = vec![(i as u8).wrapping_mul(17); 160]; // 20ms chunk
                make_frame(i, &data)
            })
            .collect();

        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode 10 frames");

        assert_eq!(report.frames_decoded, frame_count);
        assert!(report.gaps.is_empty(), "no gaps expected");
        assert!(report.duplicates.is_empty(), "no duplicates expected");
        assert!(
            report.integrity_failures.is_empty(),
            "no integrity failures expected"
        );
        assert!(
            report.dropped_frames.is_empty(),
            "no dropped frames expected"
        );

        // Verify ordering: reconstruct expected raw and compare.
        let expected_raw: Vec<u8> = (0..frame_count)
            .flat_map(|i| vec![(i as u8).wrapping_mul(17); 160])
            .collect();
        assert_eq!(raw.len(), expected_raw.len());
        assert_eq!(
            raw, expected_raw,
            "decoded bytes must match in correct order"
        );
    }

    // -- Test 12: single sample frame --

    #[test]
    fn single_sample_frame() {
        // A frame with exactly 1 byte of mu-law data.
        let single = vec![0x42u8];
        let frame = make_frame(0, &single);
        let ndjson = frames_to_ndjson(&[frame]);
        let mut reader = ndjson.as_bytes();

        let (report, raw) = decode_frames_to_raw(&mut reader).expect("decode single-sample frame");
        assert_eq!(report.frames_decoded, 1);
        assert!(report.gaps.is_empty());
        assert!(report.integrity_failures.is_empty());
        assert_eq!(raw.len(), 1, "single sample must produce 1 byte");
        assert_eq!(raw[0], 0x42, "single sample value must roundtrip exactly");
    }

    // ---------------------------------------------------------------
    // Real-time mic-to-NDJSON streaming tests (bead bd-2xe.1)
    // ---------------------------------------------------------------

    use super::{
        FixedCountMicSource, MIC_AUDIO_CHUNK_REQUIRED_FIELDS, MicAudioSource, MicStreamConfig,
        MicStreamEvent, SliceMicSource, UnavailableMicSource, mic_stream_event_value,
        stream_mic_to_ndjson,
    };

    // -- MicStreamConfig tests --

    #[test]
    fn mic_stream_config_default_values() {
        let config = MicStreamConfig::default();
        assert_eq!(config.chunk_ms, 200);
        assert_eq!(config.sample_rate_hz, 8_000);
        assert_eq!(config.channels, 1);
        assert!(config.device.is_none());
    }

    #[test]
    fn mic_stream_config_validate_accepts_defaults() {
        let config = MicStreamConfig::default();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn mic_stream_config_validate_rejects_non_8khz_sample_rate() {
        let config = MicStreamConfig {
            sample_rate_hz: 16_000,
            ..MicStreamConfig::default()
        };
        let err = config.validate().expect_err("non-8kHz should fail");
        assert!(err.to_string().contains("sample_rate_hz=16000"));
    }

    #[test]
    fn mic_stream_config_validate_rejects_stereo_channels() {
        let config = MicStreamConfig {
            channels: 2,
            ..MicStreamConfig::default()
        };
        let err = config.validate().expect_err("stereo should fail");
        assert!(err.to_string().contains("channels=2"));
    }

    #[test]
    fn mic_stream_config_chunk_byte_size() {
        let config = MicStreamConfig::default();
        // 200ms at 8kHz = 1600 bytes
        assert_eq!(config.chunk_byte_size(), 1600);
    }

    #[test]
    fn mic_stream_config_chunk_byte_size_20ms() {
        let config = MicStreamConfig {
            chunk_ms: 20,
            ..MicStreamConfig::default()
        };
        assert_eq!(config.chunk_byte_size(), 160);
    }

    #[test]
    fn mic_stream_config_chunk_byte_size_clamped_low() {
        let config = MicStreamConfig {
            chunk_ms: 1,
            ..MicStreamConfig::default()
        };
        // Clamped to 20ms -> 160
        assert_eq!(config.chunk_byte_size(), 160);
    }

    #[test]
    fn mic_stream_config_serde_round_trip() {
        let config = MicStreamConfig {
            chunk_ms: 100,
            sample_rate_hz: 8_000,
            channels: 1,
            device: Some("hw:2,0".to_owned()),
        };
        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: MicStreamConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.chunk_ms, 100);
        assert_eq!(parsed.sample_rate_hz, 8_000);
        assert_eq!(parsed.channels, 1);
        assert_eq!(parsed.device.as_deref(), Some("hw:2,0"));
    }

    #[test]
    fn mic_stream_config_serde_without_device_field() {
        let json = r#"{"chunk_ms":200,"sample_rate_hz":8000,"channels":1}"#;
        let parsed: MicStreamConfig = serde_json::from_str(json).expect("deserialize");
        assert!(parsed.device.is_none());
    }

    // -- MicStreamEvent tests --

    #[test]
    fn mic_stream_event_value_has_correct_event_type() {
        let frame = make_frame(0, b"test-data");
        let event = mic_stream_event_value(&frame);
        assert_eq!(event.event, "mic_audio_chunk");
    }

    #[test]
    fn mic_stream_event_value_has_schema_version() {
        let frame = make_frame(0, b"test-data");
        let event = mic_stream_event_value(&frame);
        assert_eq!(event.schema_version, crate::robot::ROBOT_SCHEMA_VERSION);
    }

    #[test]
    fn mic_stream_event_value_preserves_frame() {
        let frame = make_frame(42, b"preserve-this");
        let event = mic_stream_event_value(&frame);
        assert_eq!(event.frame.seq, 42);
        assert_eq!(event.frame.codec, CODEC_MULAW_ZLIB_B64);
    }

    #[test]
    fn mic_stream_event_serializes_to_single_ndjson_line() {
        let frame = make_frame(0, b"ndjson-check");
        let event = mic_stream_event_value(&frame);
        let line = serde_json::to_string(&event).expect("serialize");
        assert!(
            !line.contains('\n'),
            "NDJSON event must not contain newlines"
        );
    }

    #[test]
    fn mic_stream_event_serde_round_trip() {
        let frame = make_frame(5, b"round-trip-data");
        let event = mic_stream_event_value(&frame);
        let json = serde_json::to_string(&event).expect("serialize");
        let parsed: MicStreamEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.event, "mic_audio_chunk");
        assert_eq!(parsed.frame.seq, 5);
        assert_eq!(parsed.frame.codec, CODEC_MULAW_ZLIB_B64);
    }

    #[test]
    fn mic_stream_event_json_has_required_fields() {
        let frame = make_frame(0, b"fields-check");
        let event = mic_stream_event_value(&frame);
        let value = serde_json::to_value(&event).expect("to_value");
        for field in MIC_AUDIO_CHUNK_REQUIRED_FIELDS {
            assert!(
                value.get(*field).is_some(),
                "missing required field `{field}` in mic_audio_chunk event"
            );
        }
    }

    // -- SliceMicSource tests --

    #[test]
    fn slice_mic_source_yields_exact_chunks() {
        let data = vec![0xAAu8; 320]; // Two chunks of 160
        let mut source = SliceMicSource::new(&data);

        let chunk1 = source.read_chunk(160).expect("read_chunk 1");
        assert!(chunk1.is_some());
        assert_eq!(chunk1.unwrap().len(), 160);

        let chunk2 = source.read_chunk(160).expect("read_chunk 2");
        assert!(chunk2.is_some());
        assert_eq!(chunk2.unwrap().len(), 160);

        let chunk3 = source.read_chunk(160).expect("read_chunk 3");
        assert!(chunk3.is_none(), "source should be exhausted");
    }

    #[test]
    fn slice_mic_source_partial_last_chunk() {
        let data = vec![0xBBu8; 250]; // 160 + 90
        let mut source = SliceMicSource::new(&data);

        let chunk1 = source.read_chunk(160).expect("read");
        assert_eq!(chunk1.as_ref().unwrap().len(), 160);

        let chunk2 = source.read_chunk(160).expect("read");
        assert_eq!(
            chunk2.as_ref().unwrap().len(),
            90,
            "last chunk should be partial"
        );

        let chunk3 = source.read_chunk(160).expect("read");
        assert!(chunk3.is_none());
    }

    #[test]
    fn slice_mic_source_empty_data() {
        let data: Vec<u8> = vec![];
        let mut source = SliceMicSource::new(&data);
        let result = source.read_chunk(160).expect("read");
        assert!(result.is_none(), "empty source yields None immediately");
    }

    // -- UnavailableMicSource tests --

    #[test]
    fn unavailable_mic_source_returns_error() {
        let mut source = UnavailableMicSource::new("no audio device found");
        let err = source.read_chunk(160).expect_err("should fail");
        assert!(err.to_string().contains("no audio device found"));
    }

    #[test]
    fn unavailable_mic_source_error_is_backend_unavailable() {
        let mut source = UnavailableMicSource::new("device missing");
        let err = source.read_chunk(160).expect_err("should fail");
        assert!(matches!(err, FwError::BackendUnavailable(_)));
    }

    // -- FixedCountMicSource tests --

    #[test]
    fn fixed_count_mic_source_yields_exact_count() {
        let mut source = FixedCountMicSource::new(3, 0xFF);
        for _ in 0..3 {
            let chunk = source.read_chunk(100).expect("read");
            assert!(chunk.is_some());
            let data = chunk.unwrap();
            assert_eq!(data.len(), 100);
            assert!(data.iter().all(|&b| b == 0xFF));
        }
        let final_chunk = source.read_chunk(100).expect("read");
        assert!(final_chunk.is_none(), "should be exhausted after 3 chunks");
    }

    #[test]
    fn fixed_count_mic_source_zero_count_yields_none_immediately() {
        let mut source = FixedCountMicSource::new(0, 0x00);
        let result = source.read_chunk(160).expect("read");
        assert!(result.is_none());
    }

    // -- stream_mic_to_ndjson tests --

    #[test]
    fn stream_mic_to_ndjson_empty_source_returns_zero() {
        let config = MicStreamConfig::default();
        let data: Vec<u8> = vec![];
        let mut source = SliceMicSource::new(&data);
        let mut output = Vec::new();

        let count =
            stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream should succeed");
        assert_eq!(count, 0, "no frames from empty source");
    }

    #[test]
    fn stream_mic_to_ndjson_emits_handshake_even_for_empty_source() {
        let config = MicStreamConfig::default();
        let data: Vec<u8> = vec![];
        let mut source = SliceMicSource::new(&data);
        let mut output = Vec::new();

        stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream should succeed");

        let output_str = String::from_utf8(output).expect("utf8");
        assert!(!output_str.is_empty(), "should have at least the handshake");
        let first_line = output_str.lines().next().expect("at least one line");
        let parsed: serde_json::Value = serde_json::from_str(first_line).expect("parse");
        assert_eq!(parsed["frame_type"], "handshake");
    }

    #[test]
    fn stream_mic_to_ndjson_single_chunk() {
        let config = MicStreamConfig {
            chunk_ms: 20,
            ..MicStreamConfig::default()
        };
        let chunk_size = config.chunk_byte_size();
        let data = vec![0xAAu8; chunk_size];
        let mut source = SliceMicSource::new(&data);
        let mut output = Vec::new();

        let count =
            stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream should succeed");
        assert_eq!(count, 1);

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        // Line 0: handshake, Line 1: audio event
        assert_eq!(lines.len(), 2);

        let event: MicStreamEvent = serde_json::from_str(lines[1]).expect("parse event");
        assert_eq!(event.event, "mic_audio_chunk");
        assert_eq!(event.frame.seq, 0);
        assert_eq!(event.frame.sample_rate_hz, 8_000);
        assert_eq!(event.frame.channels, 1);
        assert_eq!(event.frame.codec, CODEC_MULAW_ZLIB_B64);
    }

    #[test]
    fn stream_mic_to_ndjson_multiple_chunks() {
        let config = MicStreamConfig {
            chunk_ms: 20,
            ..MicStreamConfig::default()
        };
        let chunk_size = config.chunk_byte_size();
        let num_chunks = 5;
        let data = vec![0xCCu8; chunk_size * num_chunks];
        let mut source = SliceMicSource::new(&data);
        let mut output = Vec::new();

        let count =
            stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream should succeed");
        assert_eq!(count, num_chunks as u64);

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        // 1 handshake + 5 audio events
        assert_eq!(lines.len(), 6);

        // Verify sequential sequence numbers
        for (i, line) in lines.iter().skip(1).enumerate() {
            let event: MicStreamEvent = serde_json::from_str(line).expect("parse");
            assert_eq!(event.frame.seq, i as u64, "seq mismatch at frame {i}");
        }
    }

    #[test]
    fn stream_mic_to_ndjson_frames_have_integrity_fields() {
        let config = MicStreamConfig {
            chunk_ms: 20,
            ..MicStreamConfig::default()
        };
        let chunk_size = config.chunk_byte_size();
        let data = vec![0xDDu8; chunk_size];
        let mut source = SliceMicSource::new(&data);
        let mut output = Vec::new();

        stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");

        let output_str = String::from_utf8(output).expect("utf8");
        let event_line = output_str.lines().nth(1).expect("event line");
        let event: MicStreamEvent = serde_json::from_str(event_line).expect("parse");

        assert!(event.frame.crc32.is_some(), "should have CRC32");
        assert!(event.frame.payload_sha256.is_some(), "should have SHA-256");
    }

    #[test]
    fn stream_mic_to_ndjson_output_is_decodable() {
        // Verify that the audio frames emitted by the stream can be decoded
        // back by the existing decode pipeline.
        let config = MicStreamConfig {
            chunk_ms: 20,
            ..MicStreamConfig::default()
        };
        let chunk_size = config.chunk_byte_size();
        let num_chunks = 3;
        let original_data: Vec<u8> = (0..chunk_size * num_chunks)
            .map(|i| (i % 256) as u8)
            .collect();
        let mut source = SliceMicSource::new(&original_data);
        let mut output = Vec::new();

        let count = stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");
        assert_eq!(count, num_chunks as u64);

        // Extract just the audio frame NDJSON lines (skip handshake, unwrap events).
        let output_str = String::from_utf8(output).expect("utf8");
        let mut frame_lines = String::new();
        for line in output_str.lines().skip(1) {
            let event: MicStreamEvent = serde_json::from_str(line).expect("parse");
            frame_lines.push_str(&serde_json::to_string(&event.frame).expect("serialize frame"));
            frame_lines.push('\n');
        }

        let mut reader = frame_lines.as_bytes();
        let (report, decoded_raw) = decode_frames_to_raw(&mut reader).expect("decode");
        assert_eq!(report.frames_decoded, num_chunks as u64);
        assert!(report.gaps.is_empty());
        assert!(report.integrity_failures.is_empty());
        assert_eq!(decoded_raw, original_data);
    }

    #[test]
    fn stream_mic_to_ndjson_rejects_invalid_config() {
        let config = MicStreamConfig {
            sample_rate_hz: 44_100,
            ..MicStreamConfig::default()
        };
        let mut source = FixedCountMicSource::new(1, 0x00);
        let mut output = Vec::new();

        let err = stream_mic_to_ndjson(&config, &mut source, &mut output)
            .expect_err("invalid config should fail");
        assert!(err.to_string().contains("sample_rate_hz=44100"));
    }

    #[test]
    fn stream_mic_to_ndjson_device_unavailable_error() {
        let config = MicStreamConfig::default();
        let mut source = UnavailableMicSource::new("ALSA device hw:5,0 not found");
        let mut output = Vec::new();

        let err = stream_mic_to_ndjson(&config, &mut source, &mut output)
            .expect_err("device error should propagate");
        assert!(err.to_string().contains("ALSA device hw:5,0 not found"));
        assert!(matches!(err, FwError::BackendUnavailable(_)));
    }

    #[test]
    fn stream_mic_to_ndjson_with_fixed_count_source() {
        let config = MicStreamConfig::default();
        let num_frames = 10;
        let mut source = FixedCountMicSource::new(num_frames, 0x55);
        let mut output = Vec::new();

        let count = stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");
        assert_eq!(count, num_frames);

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        // 1 handshake + 10 events
        assert_eq!(lines.len(), 11);
    }

    #[test]
    fn stream_mic_to_ndjson_each_line_is_valid_json() {
        let config = MicStreamConfig {
            chunk_ms: 20,
            ..MicStreamConfig::default()
        };
        let mut source = FixedCountMicSource::new(3, 0xAB);
        let mut output = Vec::new();

        stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");

        let output_str = String::from_utf8(output).expect("utf8");
        for (i, line) in output_str.lines().enumerate() {
            let parsed = serde_json::from_str::<serde_json::Value>(line);
            assert!(parsed.is_ok(), "line {i} is not valid JSON: {parsed:?}");
        }
    }

    #[test]
    fn stream_mic_to_ndjson_schema_version_matches_robot() {
        let config = MicStreamConfig::default();
        let mut source = FixedCountMicSource::new(1, 0x00);
        let mut output = Vec::new();

        stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");

        let output_str = String::from_utf8(output).expect("utf8");
        let event_line = output_str.lines().nth(1).expect("event line");
        let event: MicStreamEvent = serde_json::from_str(event_line).expect("parse");
        assert_eq!(
            event.schema_version,
            crate::robot::ROBOT_SCHEMA_VERSION,
            "schema_version should match robot schema"
        );
    }

    #[test]
    fn stream_mic_to_ndjson_handshake_is_well_formed() {
        let config = MicStreamConfig::default();
        let mut source = FixedCountMicSource::new(1, 0x00);
        let mut output = Vec::new();

        stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");

        let output_str = String::from_utf8(output).expect("utf8");
        let handshake_line = output_str.lines().next().expect("handshake line");
        let parsed: TtyControlFrame =
            serde_json::from_str(handshake_line).expect("parse handshake");

        match parsed {
            TtyControlFrame::Handshake {
                min_version,
                max_version,
                supported_codecs,
            } => {
                assert_eq!(min_version, MIN_PROTOCOL_VERSION);
                assert_eq!(max_version, SUPPORTED_PROTOCOL_VERSION);
                assert!(supported_codecs.contains(&CODEC_MULAW_ZLIB_B64.to_owned()));
            }
            other => assert!(
                matches!(other, TtyControlFrame::Handshake { .. }),
                "expected Handshake"
            ),
        }
    }

    #[test]
    fn stream_mic_to_ndjson_partial_chunk_at_end_is_encoded() {
        let config = MicStreamConfig {
            chunk_ms: 20,
            ..MicStreamConfig::default()
        };
        let chunk_size = config.chunk_byte_size();
        // 1.5 chunks worth of data
        let data = vec![0xEEu8; chunk_size + chunk_size / 2];
        let mut source = SliceMicSource::new(&data);
        let mut output = Vec::new();

        let count = stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");
        assert_eq!(count, 2, "should emit 2 frames (1 full + 1 partial)");
    }

    #[test]
    fn stream_mic_to_ndjson_crc32_integrity_verified_on_decode() {
        let config = MicStreamConfig {
            chunk_ms: 20,
            ..MicStreamConfig::default()
        };
        let chunk_size = config.chunk_byte_size();
        let data = vec![0x77u8; chunk_size];
        let mut source = SliceMicSource::new(&data);
        let mut output = Vec::new();

        stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");

        let output_str = String::from_utf8(output).expect("utf8");
        let event_line = output_str.lines().nth(1).expect("event line");
        let event: MicStreamEvent = serde_json::from_str(event_line).expect("parse");

        // Verify the CRC matches the decoded payload.
        let compressed = STANDARD_NO_PAD
            .decode(&event.frame.payload_b64)
            .expect("base64 decode");
        let decompressed = decompress_chunk(&compressed).expect("decompress");
        let actual_crc = crc32_of(&decompressed);
        assert_eq!(
            event.frame.crc32,
            Some(actual_crc),
            "CRC should match decoded data"
        );
    }

    #[test]
    fn stream_mic_to_ndjson_sha256_integrity_verified_on_decode() {
        let config = MicStreamConfig {
            chunk_ms: 20,
            ..MicStreamConfig::default()
        };
        let chunk_size = config.chunk_byte_size();
        let data = vec![0x88u8; chunk_size];
        let mut source = SliceMicSource::new(&data);
        let mut output = Vec::new();

        stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");

        let output_str = String::from_utf8(output).expect("utf8");
        let event_line = output_str.lines().nth(1).expect("event line");
        let event: MicStreamEvent = serde_json::from_str(event_line).expect("parse");

        let compressed = STANDARD_NO_PAD
            .decode(&event.frame.payload_b64)
            .expect("base64 decode");
        let decompressed = decompress_chunk(&compressed).expect("decompress");
        let actual_sha = sha256_hex(&decompressed);
        assert_eq!(
            event.frame.payload_sha256.as_deref(),
            Some(actual_sha.as_str()),
            "SHA-256 should match decoded data"
        );
    }

    // -- Custom MicAudioSource that fails mid-stream --

    struct FailAfterNSource {
        remaining_ok: u64,
    }

    impl MicAudioSource for FailAfterNSource {
        fn read_chunk(&mut self, chunk_size: usize) -> FwResult<Option<Vec<u8>>> {
            if self.remaining_ok == 0 {
                return Err(FwError::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "audio device disconnected",
                )));
            }
            self.remaining_ok -= 1;
            Ok(Some(vec![0xAA; chunk_size]))
        }
    }

    #[test]
    fn stream_mic_to_ndjson_error_mid_stream_propagates() {
        let config = MicStreamConfig::default();
        let mut source = FailAfterNSource { remaining_ok: 2 };
        let mut output = Vec::new();

        let err = stream_mic_to_ndjson(&config, &mut source, &mut output)
            .expect_err("mid-stream error should propagate");
        assert!(
            err.to_string().contains("audio device disconnected"),
            "error message: {}",
            err
        );

        // Should have emitted handshake + 2 frames before the error.
        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        assert_eq!(lines.len(), 3, "handshake + 2 audio events before failure");
    }

    // -- MIC_AUDIO_CHUNK_REQUIRED_FIELDS constant tests --

    #[test]
    fn mic_audio_chunk_required_fields_are_non_empty_and_unique() {
        use std::collections::HashSet;
        assert!(!MIC_AUDIO_CHUNK_REQUIRED_FIELDS.is_empty());
        let unique: HashSet<_> = MIC_AUDIO_CHUNK_REQUIRED_FIELDS.iter().collect();
        assert_eq!(unique.len(), MIC_AUDIO_CHUNK_REQUIRED_FIELDS.len());
        for f in MIC_AUDIO_CHUNK_REQUIRED_FIELDS {
            assert!(!f.is_empty());
        }
    }

    // -- Large streaming test --

    #[test]
    fn stream_mic_to_ndjson_large_stream_100_chunks() {
        let config = MicStreamConfig {
            chunk_ms: 20,
            ..MicStreamConfig::default()
        };
        let mut source = FixedCountMicSource::new(100, 0x42);
        let mut output = Vec::new();

        let count =
            stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream 100 chunks");
        assert_eq!(count, 100);

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        assert_eq!(lines.len(), 101); // handshake + 100

        // Verify last frame has seq=99
        let last_event: MicStreamEvent = serde_json::from_str(lines[100]).expect("parse last");
        assert_eq!(last_event.frame.seq, 99);
    }

    // -- Varying chunk sizes test --

    #[test]
    fn stream_mic_to_ndjson_different_chunk_durations() {
        for chunk_ms in [20, 50, 100, 200, 500, 1000] {
            let config = MicStreamConfig {
                chunk_ms,
                ..MicStreamConfig::default()
            };
            let expected_size = config.chunk_byte_size();
            let data = vec![0x11u8; expected_size * 2];
            let mut source = SliceMicSource::new(&data);
            let mut output = Vec::new();

            let count = stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");
            assert_eq!(
                count, 2,
                "chunk_ms={chunk_ms}: expected 2 frames for 2 chunks of data"
            );
        }
    }

    // -- Custom source yielding empty chunks --

    struct EmptyChunkSource {
        empty_count: u64,
        then_real: u64,
    }

    impl MicAudioSource for EmptyChunkSource {
        fn read_chunk(&mut self, chunk_size: usize) -> FwResult<Option<Vec<u8>>> {
            if self.empty_count > 0 {
                self.empty_count -= 1;
                return Ok(Some(vec![]));
            }
            if self.then_real > 0 {
                self.then_real -= 1;
                return Ok(Some(vec![0xBB; chunk_size]));
            }
            Ok(None)
        }
    }

    #[test]
    fn stream_mic_to_ndjson_skips_empty_chunks() {
        let config = MicStreamConfig::default();
        let mut source = EmptyChunkSource {
            empty_count: 3,
            then_real: 2,
        };
        let mut output = Vec::new();

        let count = stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");
        // Empty chunks are skipped; only the 2 real chunks produce frames.
        assert_eq!(count, 2);

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        assert_eq!(lines.len(), 3, "handshake + 2 events");
    }

    // -- Full end-to-end roundtrip test --

    #[test]
    fn stream_mic_to_ndjson_full_roundtrip_integrity() {
        let config = MicStreamConfig {
            chunk_ms: 100,
            ..MicStreamConfig::default()
        };
        let chunk_size = config.chunk_byte_size();

        // Generate pseudo-random data
        let num_chunks = 7;
        let mut data = vec![0u8; chunk_size * num_chunks];
        let mut state: u32 = 0xBADC0FFE;
        for byte in &mut data {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (state >> 16) as u8;
        }

        let mut source = SliceMicSource::new(&data);
        let mut output = Vec::new();

        let count = stream_mic_to_ndjson(&config, &mut source, &mut output).expect("stream");
        assert_eq!(count, num_chunks as u64);

        // Parse all events, extract frames, and decode
        let output_str = String::from_utf8(output).expect("utf8");
        let mut frame_ndjson = String::new();
        for line in output_str.lines().skip(1) {
            let event: MicStreamEvent = serde_json::from_str(line).expect("parse");
            frame_ndjson.push_str(&serde_json::to_string(&event.frame).expect("frame json"));
            frame_ndjson.push('\n');
        }

        let mut reader = frame_ndjson.as_bytes();
        let (report, decoded) = decode_frames_to_raw(&mut reader).expect("decode");
        assert_eq!(report.frames_decoded, num_chunks as u64);
        assert!(report.gaps.is_empty());
        assert!(report.integrity_failures.is_empty());
        assert_eq!(decoded.len(), data.len());
        assert_eq!(decoded, data, "full roundtrip data must match exactly");
    }

    #[test]
    fn collapse_sequences_to_ranges_with_u64_max_boundary() {
        use super::{RetransmitRange, collapse_sequences_to_ranges};

        // Non-contiguous sequences with u64::MAX.
        let seqs = vec![0, 1, 2, u64::MAX - 1, u64::MAX];
        let ranges = collapse_sequences_to_ranges(&seqs);
        assert_eq!(ranges.len(), 2, "should produce two ranges");
        assert_eq!(
            ranges[0],
            RetransmitRange {
                start_seq: 0,
                end_seq: 2
            }
        );
        assert_eq!(
            ranges[1],
            RetransmitRange {
                start_seq: u64::MAX - 1,
                end_seq: u64::MAX
            }
        );
    }

    #[test]
    fn parse_rejects_handshake_ack_before_handshake() {
        use super::TtyControlFrame;

        let ack = serde_json::to_string(&TtyControlFrame::HandshakeAck {
            negotiated_version: 1,
            negotiated_codec: CODEC_MULAW_ZLIB_B64.to_owned(),
        })
        .expect("serialize");
        let frame = serde_json::to_string(&make_frame(0, b"data")).expect("frame");
        let ndjson = format!("{ack}\n{frame}\n");
        let mut reader = ndjson.as_bytes();

        let err = parse_audio_frames_for_decode(&mut reader)
            .expect_err("HandshakeAck before Handshake should fail");
        assert!(
            err.to_string()
                .contains("handshake_ack received before handshake"),
            "error should mention ack before handshake: {err}"
        );
    }

    #[test]
    fn parse_rejects_handshake_ack_version_mismatch() {
        use super::TtyControlFrame;

        let handshake = serde_json::to_string(&TtyControlFrame::Handshake {
            min_version: MIN_PROTOCOL_VERSION,
            max_version: SUPPORTED_PROTOCOL_VERSION,
            supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned()],
        })
        .expect("serialize handshake");
        // Send ack with different version (99) than negotiated (1).
        let ack = serde_json::to_string(&TtyControlFrame::HandshakeAck {
            negotiated_version: 99,
            negotiated_codec: CODEC_MULAW_ZLIB_B64.to_owned(),
        })
        .expect("serialize ack");
        let ndjson = format!("{handshake}\n{ack}\n");
        let mut reader = ndjson.as_bytes();

        let err =
            parse_audio_frames_for_decode(&mut reader).expect_err("version mismatch should fail");
        assert!(
            err.to_string().contains("unexpected negotiated_version"),
            "error should mention version mismatch: {err}"
        );
    }

    #[test]
    fn parse_rejects_handshake_ack_unsupported_codec() {
        use super::TtyControlFrame;

        let handshake = serde_json::to_string(&TtyControlFrame::Handshake {
            min_version: MIN_PROTOCOL_VERSION,
            max_version: SUPPORTED_PROTOCOL_VERSION,
            supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned()],
        })
        .expect("serialize handshake");
        // Send ack with unsupported codec.
        let ack = serde_json::to_string(&TtyControlFrame::HandshakeAck {
            negotiated_version: SUPPORTED_PROTOCOL_VERSION,
            negotiated_codec: "opus+webm+b64".to_owned(),
        })
        .expect("serialize ack");
        let ndjson = format!("{handshake}\n{ack}\n");
        let mut reader = ndjson.as_bytes();

        let err =
            parse_audio_frames_for_decode(&mut reader).expect_err("unsupported codec should fail");
        assert!(
            err.to_string().contains("unsupported negotiated codec"),
            "error should mention unsupported codec: {err}"
        );
    }

    #[test]
    fn emit_retransmit_loop_zero_rounds_sends_one_request() {
        use super::TtyControlFrame;

        // Create a stream with a gap: seq 0, seq 2 (missing seq 1).
        let frames = [make_frame(0, b"aa"), make_frame(2, b"bb")];
        let ndjson = format!(
            "{}\n{}\n",
            serde_json::to_string(&frames[0]).expect("f0"),
            serde_json::to_string(&frames[1]).expect("f1")
        );
        let mut reader = ndjson.as_bytes();
        let mut output = Vec::new();

        emit_retransmit_loop_from_reader(
            &mut reader,
            DecodeRecoveryPolicy::SkipMissing,
            0, // rounds=0 should be clamped to 1
            &mut output,
        )
        .expect("should succeed");

        let output_str = String::from_utf8(output).expect("utf8");
        let lines: Vec<&str> = output_str.lines().collect();
        // Should emit exactly 1 RetransmitRequest (not 0) + 1 RetransmitResponse = 2 lines.
        assert_eq!(
            lines.len(),
            2,
            "rounds=0 clamped to 1: 1 request + 1 response"
        );

        let req: TtyControlFrame =
            serde_json::from_str(lines[0]).expect("parse first control frame");
        assert!(
            matches!(req, TtyControlFrame::RetransmitRequest { .. }),
            "first line should be RetransmitRequest"
        );
        let resp: TtyControlFrame =
            serde_json::from_str(lines[1]).expect("parse second control frame");
        assert!(
            matches!(resp, TtyControlFrame::RetransmitResponse { .. }),
            "second line should be RetransmitResponse"
        );
    }

    // ---------------------------------------------------------------
    // TtyAudioPipelineAdapter tests (bead bd-2xe.2)
    // ---------------------------------------------------------------

    use super::TtyAudioPipelineAdapter;

    #[test]
    fn pipeline_adapter_new_creates_instance() {
        let adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        assert_eq!(adapter.frames_ingested(), 0);
        assert!(!adapter.is_finalized());
        assert_eq!(adapter.raw_pcm_len(), 0);
    }

    #[test]
    fn pipeline_adapter_ingest_single_frame() {
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        let frame = make_frame(0, b"test-data-for-pipeline");
        let count = adapter.ingest_frames(&[frame]).expect("ingest");
        assert_eq!(count, 1);
        assert_eq!(adapter.frames_ingested(), 1);
        assert_eq!(adapter.raw_pcm_len(), b"test-data-for-pipeline".len());
    }

    #[test]
    fn pipeline_adapter_ingest_multiple_frames() {
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        let frames = vec![
            make_frame(0, b"chunk-a"),
            make_frame(1, b"chunk-b"),
            make_frame(2, b"chunk-c"),
        ];
        let count = adapter.ingest_frames(&frames).expect("ingest");
        assert_eq!(count, 3);
        assert_eq!(adapter.frames_ingested(), 3);
        let expected_len = b"chunk-a".len() + b"chunk-b".len() + b"chunk-c".len();
        assert_eq!(adapter.raw_pcm_len(), expected_len);
    }

    #[test]
    fn pipeline_adapter_ingest_in_batches() {
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        adapter
            .ingest_frames(&[make_frame(0, b"batch-one")])
            .expect("first batch");
        adapter
            .ingest_frames(&[make_frame(1, b"batch-two")])
            .expect("second batch");
        assert_eq!(adapter.frames_ingested(), 2);
        assert_eq!(
            adapter.raw_pcm_len(),
            b"batch-one".len() + b"batch-two".len()
        );
    }

    #[test]
    fn pipeline_adapter_ingest_empty_slice() {
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        let count = adapter.ingest_frames(&[]).expect("ingest empty");
        assert_eq!(count, 0);
        assert_eq!(adapter.frames_ingested(), 0);
    }

    #[test]
    fn pipeline_adapter_ingest_rejects_corrupt_crc() {
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        let mut frame = make_frame(0, b"data");
        frame.crc32 = Some(frame.crc32.unwrap() ^ 0xDEAD);
        let err = adapter
            .ingest_frames(&[frame])
            .expect_err("corrupt CRC should fail");
        assert!(err.to_string().contains("CRC mismatch"));
    }

    #[test]
    fn pipeline_adapter_ingest_rejects_corrupt_sha256() {
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        let mut frame = make_frame(0, b"data");
        frame.payload_sha256 = Some("badbadbadbad".to_owned());
        let err = adapter
            .ingest_frames(&[frame])
            .expect_err("corrupt SHA should fail");
        assert!(err.to_string().contains("SHA-256 mismatch"));
    }

    #[test]
    fn pipeline_adapter_finalize_rejects_empty() {
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        let err = adapter
            .finalize_wav()
            .expect_err("empty adapter should fail");
        assert!(err.to_string().contains("no frames have been ingested"));
    }

    #[test]
    fn pipeline_adapter_finalize_rejects_double_finalize() {
        // We cannot actually call finalize_wav successfully without ffmpeg,
        // but we can test the double-finalize guard by ingesting a frame and
        // then simulating finalization state.
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        // Manually set finalized to test the guard.
        adapter.finalized = true;
        let err = adapter
            .finalize_wav()
            .expect_err("double finalize should fail");
        assert!(err.to_string().contains("already been finalized"));
    }

    #[test]
    fn pipeline_adapter_ingest_rejects_after_finalize() {
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        adapter.finalized = true;
        let err = adapter
            .ingest_frames(&[make_frame(0, b"late")])
            .expect_err("ingest after finalize should fail");
        assert!(err.to_string().contains("already been finalized"));
    }

    #[test]
    fn pipeline_adapter_ingest_frame_without_integrity_fields() {
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        let data = b"no-integrity-data";
        let compressed = compress_chunk(data).expect("compress");
        let frame = TtyAudioFrame {
            protocol_version: SUPPORTED_PROTOCOL_VERSION,
            seq: 0,
            codec: CODEC_MULAW_ZLIB_B64.to_owned(),
            sample_rate_hz: 8_000,
            channels: 1,
            payload_b64: STANDARD_NO_PAD.encode(compressed),
            crc32: None,
            payload_sha256: None,
        };
        let count = adapter.ingest_frames(&[frame]).expect("ingest");
        assert_eq!(count, 1);
        assert_eq!(adapter.raw_pcm_len(), data.len());
    }

    #[test]
    fn pipeline_adapter_ingest_rejects_unsupported_frame_metadata() {
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        let mut frame = make_frame(0, b"data");
        frame.sample_rate_hz = 16_000;

        let err = adapter
            .ingest_frames(&[frame])
            .expect_err("unsupported frame metadata should fail");
        assert!(err.to_string().contains("unsupported tty-audio shape"));
    }

    #[test]
    fn pipeline_adapter_finalize_rejects_sequence_gaps() {
        let mut adapter = TtyAudioPipelineAdapter::new().expect("new adapter");
        adapter
            .ingest_frames(&[make_frame(0, b"first"), make_frame(2, b"third")])
            .expect("ingest gapped frames");

        let err = adapter
            .finalize_wav()
            .expect_err("gapped frame set should fail finalize");
        assert!(err.to_string().contains("missing frame sequences"));
        assert!(
            !adapter.is_finalized(),
            "failed finalize should leave adapter open for retransmit repair"
        );
    }

    // ---------------------------------------------------------------
    // AdaptiveBitrateController tests (bead bd-2xe.3)
    // ---------------------------------------------------------------

    use super::AdaptiveBitrateController;

    #[test]
    fn abr_new_defaults() {
        let abr = AdaptiveBitrateController::new(64_000);
        assert_eq!(abr.target_bitrate, 64_000);
        assert!((abr.current_quality - 1.0).abs() < f64::EPSILON);
        assert!(abr.frame_loss_rate.abs() < f64::EPSILON);
        assert_eq!(abr.frames_sent(), 0);
        assert_eq!(abr.frames_lost(), 0);
    }

    #[test]
    fn abr_perfect_link_uses_fast_compression() {
        let mut abr = AdaptiveBitrateController::new(64_000);
        // 100 delivered, 0 lost
        abr.record_batch(100, 0);
        assert!(abr.frame_loss_rate.abs() < f64::EPSILON);
        assert_eq!(
            abr.recommended_compression().level(),
            flate2::Compression::fast().level()
        );
    }

    #[test]
    fn abr_moderate_link_uses_default_compression() {
        let mut abr = AdaptiveBitrateController::new(64_000);
        // 95 delivered, 5 lost => 5% loss
        abr.record_batch(95, 5);
        let expected_loss = 5.0 / 100.0;
        assert!((abr.frame_loss_rate - expected_loss).abs() < 0.001);
        assert_eq!(
            abr.recommended_compression().level(),
            flate2::Compression::default().level()
        );
    }

    #[test]
    fn abr_poor_link_uses_best_compression() {
        let mut abr = AdaptiveBitrateController::new(64_000);
        // 80 delivered, 20 lost => 20% loss
        abr.record_batch(80, 20);
        let expected_loss = 20.0 / 100.0;
        assert!((abr.frame_loss_rate - expected_loss).abs() < 0.001);
        assert_eq!(
            abr.recommended_compression().level(),
            flate2::Compression::best().level()
        );
    }

    #[test]
    fn abr_record_delivery_single_success() {
        let mut abr = AdaptiveBitrateController::new(64_000);
        abr.record_delivery(true);
        assert_eq!(abr.frames_sent(), 1);
        assert_eq!(abr.frames_lost(), 0);
        assert!(abr.frame_loss_rate.abs() < f64::EPSILON);
    }

    #[test]
    fn abr_record_delivery_single_loss() {
        let mut abr = AdaptiveBitrateController::new(64_000);
        abr.record_delivery(false);
        assert_eq!(abr.frames_sent(), 1);
        assert_eq!(abr.frames_lost(), 1);
        assert!((abr.frame_loss_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn abr_quality_is_complement_of_loss() {
        let mut abr = AdaptiveBitrateController::new(64_000);
        abr.record_batch(90, 10);
        assert!((abr.current_quality + abr.frame_loss_rate - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn abr_critical_frame_redundancy_perfect_link() {
        let abr = AdaptiveBitrateController::new(64_000);
        assert_eq!(abr.critical_frame_redundancy(), 1);
    }

    #[test]
    fn abr_critical_frame_redundancy_moderate_link() {
        let mut abr = AdaptiveBitrateController::new(64_000);
        abr.record_batch(95, 5); // 5% loss
        assert_eq!(abr.critical_frame_redundancy(), 2);
    }

    #[test]
    fn abr_critical_frame_redundancy_poor_link() {
        let mut abr = AdaptiveBitrateController::new(64_000);
        abr.record_batch(80, 20); // 20% loss
        assert_eq!(abr.critical_frame_redundancy(), 3);
    }

    #[test]
    fn abr_emit_critical_frame_with_fec_perfect_link() {
        let abr = AdaptiveBitrateController::new(64_000);
        let handshake = TtyControlFrame::Handshake {
            min_version: MIN_PROTOCOL_VERSION,
            max_version: SUPPORTED_PROTOCOL_VERSION,
            supported_codecs: vec![CODEC_MULAW_ZLIB_B64.to_owned()],
        };
        let mut out = Vec::new();
        let count = abr
            .emit_critical_frame_with_fec(&mut out, &handshake)
            .expect("emit");
        assert_eq!(count, 1);
        let output_str = String::from_utf8(out).expect("utf8");
        assert_eq!(output_str.lines().count(), 1);
    }

    #[test]
    fn abr_emit_critical_frame_with_fec_poor_link() {
        let mut abr = AdaptiveBitrateController::new(64_000);
        abr.record_batch(70, 30); // 30% loss
        let ack = TtyControlFrame::Ack { up_to_seq: 10 };
        let mut out = Vec::new();
        let count = abr
            .emit_critical_frame_with_fec(&mut out, &ack)
            .expect("emit");
        assert_eq!(count, 3);
        let output_str = String::from_utf8(out).expect("utf8");
        assert_eq!(output_str.lines().count(), 3);
    }

    #[test]
    fn abr_compress_adaptive_produces_decompressible_output() {
        let abr = AdaptiveBitrateController::new(64_000);
        let data = b"adaptive-compress-test-data";
        let compressed = abr.compress_adaptive(data).expect("compress");
        let decompressed = decompress_chunk(&compressed).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn abr_compress_adaptive_poor_link_still_decompressible() {
        let mut abr = AdaptiveBitrateController::new(64_000);
        abr.record_batch(50, 50); // 50% loss
        let data = b"poor-link-data";
        let compressed = abr.compress_adaptive(data).expect("compress");
        let decompressed = decompress_chunk(&compressed).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn abr_incremental_recording_updates_state() {
        let mut abr = AdaptiveBitrateController::new(64_000);
        for _ in 0..50 {
            abr.record_delivery(true);
        }
        assert_eq!(abr.frames_sent(), 50);
        assert_eq!(abr.frames_lost(), 0);
        assert!(abr.frame_loss_rate.abs() < f64::EPSILON);

        // Now lose 5 in a row
        for _ in 0..5 {
            abr.record_delivery(false);
        }
        assert_eq!(abr.frames_sent(), 55);
        assert_eq!(abr.frames_lost(), 5);
        // 5/55 ~ 9.09% -> moderate
        assert!(abr.frame_loss_rate > 0.09);
        assert!(abr.frame_loss_rate < 0.10);
    }

    // ---------------------------------------------------------------
    // Session close (TtyControlFrame) tests (bead bd-30v)
    // ---------------------------------------------------------------

    use super::{emit_session_close, validate_session_close};

    #[test]
    fn session_close_reason_serde_round_trip_normal() {
        let json = serde_json::to_string(&SessionCloseReason::Normal).expect("serialize");
        let parsed: SessionCloseReason = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, SessionCloseReason::Normal);
    }

    #[test]
    fn session_close_reason_serde_round_trip_error() {
        let json = serde_json::to_string(&SessionCloseReason::Error).expect("serialize");
        let parsed: SessionCloseReason = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, SessionCloseReason::Error);
    }

    #[test]
    fn session_close_reason_serde_round_trip_timeout() {
        let json = serde_json::to_string(&SessionCloseReason::Timeout).expect("serialize");
        let parsed: SessionCloseReason = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, SessionCloseReason::Timeout);
    }

    #[test]
    fn session_close_reason_serde_round_trip_peer_requested() {
        let json = serde_json::to_string(&SessionCloseReason::PeerRequested).expect("serialize");
        let parsed: SessionCloseReason = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, SessionCloseReason::PeerRequested);
    }

    #[test]
    fn control_frame_type_session_close_serde_round_trip() {
        let frame = TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Normal,
            last_data_seq: Some(42),
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            TtyControlFrame::SessionClose {
                reason,
                last_data_seq,
            } => {
                assert_eq!(reason, SessionCloseReason::Normal);
                assert_eq!(last_data_seq, Some(42));
            }
            other => assert!(
                matches!(other, TtyControlFrame::SessionClose { .. }),
                "expected SessionClose"
            ),
        }
    }

    #[test]
    fn control_frame_type_session_close_none_last_seq() {
        let frame = TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Error,
            last_data_seq: None,
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            TtyControlFrame::SessionClose {
                reason,
                last_data_seq,
            } => {
                assert_eq!(reason, SessionCloseReason::Error);
                assert!(last_data_seq.is_none());
            }
            other => assert!(
                matches!(other, TtyControlFrame::SessionClose { .. }),
                "expected SessionClose"
            ),
        }
    }

    #[test]
    fn control_frame_type_session_close_json_has_frame_type_field() {
        let frame = TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Normal,
            last_data_seq: Some(10),
        };
        let value = serde_json::to_value(&frame).expect("to_value");
        assert_eq!(value["frame_type"], "session_close");
        assert_eq!(value["reason"], "normal");
        assert_eq!(value["last_data_seq"], 10);
    }

    #[test]
    fn emit_session_close_writes_ndjson_line() {
        let mut out = Vec::new();
        emit_session_close(&mut out, SessionCloseReason::Normal, Some(99)).expect("emit");
        let text = String::from_utf8(out).expect("utf8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1);
        let parsed: serde_json::Value = serde_json::from_str(lines[0]).expect("json");
        assert_eq!(parsed["frame_type"], "session_close");
        assert_eq!(parsed["reason"], "normal");
        assert_eq!(parsed["last_data_seq"], 99);
    }

    #[test]
    fn emit_session_close_error_reason() {
        let mut out = Vec::new();
        emit_session_close(&mut out, SessionCloseReason::Error, None).expect("emit");
        let text = String::from_utf8(out).expect("utf8");
        let parsed: serde_json::Value = serde_json::from_str(text.trim()).expect("json");
        assert_eq!(parsed["reason"], "error");
        assert!(parsed["last_data_seq"].is_null());
    }

    #[test]
    fn emit_session_close_timeout_reason() {
        let mut out = Vec::new();
        emit_session_close(&mut out, SessionCloseReason::Timeout, Some(5)).expect("emit");
        let text = String::from_utf8(out).expect("utf8");
        let parsed: serde_json::Value = serde_json::from_str(text.trim()).expect("json");
        assert_eq!(parsed["reason"], "timeout");
        assert_eq!(parsed["last_data_seq"], 5);
    }

    #[test]
    fn validate_session_close_matching_seq() {
        let frame = TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Normal,
            last_data_seq: Some(42),
        };
        validate_session_close(Some(42), &frame).expect("should be valid");
    }

    #[test]
    fn validate_session_close_both_none() {
        let frame = TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Normal,
            last_data_seq: None,
        };
        validate_session_close(None, &frame).expect("should be valid");
    }

    #[test]
    fn validate_session_close_observed_none_claimed_zero_is_mismatch() {
        // Some(0) means "I sent a data frame with seq 0" — but observed=None means
        // the receiver never saw any data frames. This is a legitimate mismatch.
        let frame = TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Normal,
            last_data_seq: Some(0),
        };
        let err = validate_session_close(None, &frame)
            .expect_err("claiming data when none observed should fail");
        assert!(err.to_string().contains("no data frames were observed"));
    }

    #[test]
    fn validate_session_close_mismatch_error() {
        let frame = TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Normal,
            last_data_seq: Some(100),
        };
        let err =
            validate_session_close(Some(50), &frame).expect_err("mismatch should produce error");
        assert!(err.to_string().contains("last_data_seq mismatch"));
        assert!(err.to_string().contains("observed 50"));
        assert!(err.to_string().contains("claimed 100"));
    }

    #[test]
    fn validate_session_close_no_data_but_claims_data() {
        let frame = TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Error,
            last_data_seq: Some(10),
        };
        let err = validate_session_close(None, &frame)
            .expect_err("claiming data when none observed should fail");
        assert!(err.to_string().contains("no data frames were observed"));
    }

    #[test]
    fn validate_session_close_observed_some_claimed_none() {
        let frame = TtyControlFrame::SessionClose {
            reason: SessionCloseReason::Normal,
            last_data_seq: None,
        };
        // If close says None for last_data_seq, that is still valid
        // (the sender might not track seq numbers).
        validate_session_close(Some(10), &frame).expect("should be valid");
    }

    #[test]
    fn session_close_is_single_ndjson_line() {
        let mut out = Vec::new();
        emit_session_close(&mut out, SessionCloseReason::PeerRequested, Some(0)).expect("emit");
        let text = String::from_utf8(out).expect("utf8");
        assert!(
            !text.trim().contains('\n'),
            "session close must be a single NDJSON line"
        );
    }

    // ---------------------------------------------------------------
    // RetransmitLoop tests (bead bd-2xe.5)
    // ---------------------------------------------------------------

    use super::{RecoveryStrategy, RetransmitLoop};

    #[test]
    fn recovery_strategy_escalation_sequence() {
        assert_eq!(
            RecoveryStrategy::Simple.escalate(),
            RecoveryStrategy::Redundant
        );
        assert_eq!(
            RecoveryStrategy::Redundant.escalate(),
            RecoveryStrategy::Escalate
        );
        assert_eq!(
            RecoveryStrategy::Escalate.escalate(),
            RecoveryStrategy::Escalate
        );
    }

    #[test]
    fn retransmit_loop_zero_loss_path() {
        let frames = vec![
            make_frame(0, b"frame-zero"),
            make_frame(1, b"frame-one"),
            make_frame(2, b"frame-two"),
        ];
        let mut rl = RetransmitLoop::new(5, 100, RecoveryStrategy::Simple, frames);
        // No loss injected — run should complete immediately.
        rl.run().expect("run should succeed");

        let report = rl.report();
        assert_eq!(report.total_frames, 3);
        assert_eq!(report.lost_frames, 0);
        assert_eq!(report.recovered_frames, 0);
        assert_eq!(report.rounds_used, 0);
        // Strategy stays at initial value when no rounds executed.
        assert_eq!(report.final_strategy, RecoveryStrategy::Simple);
    }

    #[test]
    fn retransmit_loop_single_frame_loss() {
        let frames = vec![
            make_frame(0, b"aaa"),
            make_frame(1, b"bbb"),
            make_frame(2, b"ccc"),
        ];
        let mut rl = RetransmitLoop::new(5, 100, RecoveryStrategy::Simple, frames);
        rl.inject_loss(&[1]);
        rl.run().expect("run should succeed");

        let report = rl.report();
        assert_eq!(report.total_frames, 3);
        assert_eq!(report.lost_frames, 1);
        assert_eq!(report.recovered_frames, 1);
        // Simple recovers 1 per round, so 1 round needed.
        assert_eq!(report.rounds_used, 1);
        // After 1 round starting from Simple, strategy escalated to Redundant.
        assert_eq!(report.final_strategy, RecoveryStrategy::Redundant);
    }

    #[test]
    fn retransmit_loop_multiple_frame_loss_simple_strategy() {
        let frames = vec![
            make_frame(0, b"d0"),
            make_frame(1, b"d1"),
            make_frame(2, b"d2"),
            make_frame(3, b"d3"),
            make_frame(4, b"d4"),
        ];
        let mut rl = RetransmitLoop::new(10, 200, RecoveryStrategy::Simple, frames);
        rl.inject_loss(&[1, 3, 4]);
        rl.run().expect("run should succeed");

        let report = rl.report();
        assert_eq!(report.total_frames, 5);
        assert_eq!(report.lost_frames, 3);
        assert_eq!(report.recovered_frames, 3);
        // Round 1 (Simple): recovers 1 frame (seq 1)
        // Round 2 (Redundant): recovers 2 frames (seq 3, 4)
        // All recovered after 2 rounds.
        assert_eq!(report.rounds_used, 2);
        assert_eq!(report.final_strategy, RecoveryStrategy::Escalate);
    }

    #[test]
    fn retransmit_loop_escalation_through_all_strategies() {
        // Create enough lost frames to exercise all three strategies.
        let frames: Vec<TtyAudioFrame> = (0..10)
            .map(|i| make_frame(i, format!("payload-{i}").as_bytes()))
            .collect();
        let mut rl = RetransmitLoop::new(10, 50, RecoveryStrategy::Simple, frames);
        // Lose 7 frames: need Simple(1) + Redundant(2) + Escalate(4) = 7
        rl.inject_loss(&[0, 1, 2, 3, 4, 5, 6]);
        rl.run().expect("run should succeed");

        let report = rl.report();
        assert_eq!(report.lost_frames, 7);
        assert_eq!(report.recovered_frames, 7);
        // Exactly 3 rounds: Simple(1) + Redundant(2) + Escalate(4) = 7
        assert_eq!(report.rounds_used, 3);
        // After round 3 (Escalate), escalate() is ceiling => still Escalate.
        assert_eq!(report.final_strategy, RecoveryStrategy::Escalate);
    }

    #[test]
    fn retransmit_loop_max_rounds_exceeded() {
        let frames: Vec<TtyAudioFrame> = (0..20)
            .map(|i| make_frame(i, format!("data-{i}").as_bytes()))
            .collect();
        let mut rl = RetransmitLoop::new(2, 100, RecoveryStrategy::Simple, frames);
        // Lose 10 frames but only allow 2 rounds.
        // Round 1 (Simple): recovers 1
        // Round 2 (Redundant): recovers 2
        // Total recovered: 3 out of 10.
        rl.inject_loss(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        rl.run()
            .expect("run should succeed even when max rounds exceeded");

        let report = rl.report();
        assert_eq!(report.lost_frames, 10);
        assert_eq!(report.recovered_frames, 3);
        assert_eq!(report.rounds_used, 2);
        assert_eq!(report.final_strategy, RecoveryStrategy::Escalate);
    }

    #[test]
    fn retransmit_loop_deterministic_same_input_same_output() {
        let frames: Vec<TtyAudioFrame> = (0..8)
            .map(|i| make_frame(i, format!("det-{i}").as_bytes()))
            .collect();
        let loss_pattern = vec![2, 5, 7];

        // Run #1
        let mut rl1 = RetransmitLoop::new(5, 100, RecoveryStrategy::Simple, frames.clone());
        rl1.inject_loss(&loss_pattern);
        rl1.run().expect("run 1");
        let report1 = rl1.report();

        // Run #2 — identical input
        let mut rl2 = RetransmitLoop::new(5, 100, RecoveryStrategy::Simple, frames);
        rl2.inject_loss(&loss_pattern);
        rl2.run().expect("run 2");
        let report2 = rl2.report();

        assert_eq!(
            report1, report2,
            "deterministic: identical inputs must produce identical reports"
        );
    }

    #[test]
    fn retransmit_loop_inject_loss_ignores_unknown_seqs() {
        let frames = vec![make_frame(0, b"only")];
        let mut rl = RetransmitLoop::new(3, 100, RecoveryStrategy::Simple, frames);
        // Inject losses for sequences that do not exist in buffer.
        rl.inject_loss(&[5, 10, 999]);

        let report = rl.report();
        assert_eq!(report.lost_frames, 0, "unknown seqs should be ignored");
    }

    #[test]
    fn retransmit_loop_inject_loss_resets_state() {
        let frames = vec![make_frame(0, b"x"), make_frame(1, b"y")];
        let mut rl = RetransmitLoop::new(5, 100, RecoveryStrategy::Simple, frames);
        rl.inject_loss(&[0]);
        rl.run().expect("first run");
        assert_eq!(rl.report().recovered_frames, 1);

        // Re-inject a different loss pattern — should reset recovery state.
        rl.inject_loss(&[1]);
        assert_eq!(rl.report().rounds_used, 0, "rounds reset after re-inject");
        assert_eq!(
            rl.report().recovered_frames,
            0,
            "recovered reset after re-inject"
        );

        rl.run().expect("second run");
        let report = rl.report();
        assert_eq!(report.lost_frames, 1);
        assert_eq!(report.recovered_frames, 1);
    }

    #[test]
    fn retransmit_loop_empty_buffer() {
        let mut rl = RetransmitLoop::new(3, 100, RecoveryStrategy::Simple, vec![]);
        rl.run().expect("empty buffer run");

        let report = rl.report();
        assert_eq!(report.total_frames, 0);
        assert_eq!(report.lost_frames, 0);
        assert_eq!(report.recovered_frames, 0);
        assert_eq!(report.rounds_used, 0);
    }

    #[test]
    fn retransmit_loop_start_with_redundant_strategy() {
        let frames = vec![
            make_frame(0, b"r0"),
            make_frame(1, b"r1"),
            make_frame(2, b"r2"),
        ];
        let mut rl = RetransmitLoop::new(5, 100, RecoveryStrategy::Redundant, frames);
        rl.inject_loss(&[0, 1]);
        rl.run().expect("run");

        let report = rl.report();
        assert_eq!(report.recovered_frames, 2);
        // Redundant recovers 2 per round, so 1 round for 2 losses.
        assert_eq!(report.rounds_used, 1);
        // After round 1 with Redundant, escalated to Escalate.
        assert_eq!(report.final_strategy, RecoveryStrategy::Escalate);
    }

    #[test]
    fn retransmit_loop_start_with_escalate_strategy() {
        let frames: Vec<TtyAudioFrame> = (0..6)
            .map(|i| make_frame(i, format!("e-{i}").as_bytes()))
            .collect();
        let mut rl = RetransmitLoop::new(5, 100, RecoveryStrategy::Escalate, frames);
        rl.inject_loss(&[0, 1, 2, 3]);
        rl.run().expect("run");

        let report = rl.report();
        assert_eq!(report.recovered_frames, 4);
        // Escalate recovers 4 per round.
        assert_eq!(report.rounds_used, 1);
        assert_eq!(report.final_strategy, RecoveryStrategy::Escalate);
    }

    #[test]
    fn retransmit_loop_max_rounds_one() {
        let frames = vec![
            make_frame(0, b"m0"),
            make_frame(1, b"m1"),
            make_frame(2, b"m2"),
        ];
        let mut rl = RetransmitLoop::new(1, 100, RecoveryStrategy::Simple, frames);
        rl.inject_loss(&[0, 1, 2]);
        rl.run().expect("run");

        let report = rl.report();
        assert_eq!(report.lost_frames, 3);
        // Only 1 round with Simple => recovers 1 frame.
        assert_eq!(report.recovered_frames, 1);
        assert_eq!(report.rounds_used, 1);
    }

    #[test]
    fn retransmit_loop_report_before_run() {
        let frames = vec![make_frame(0, b"pre")];
        let rl = RetransmitLoop::new(3, 100, RecoveryStrategy::Simple, frames);
        let report = rl.report();
        assert_eq!(report.total_frames, 1);
        assert_eq!(report.lost_frames, 0);
        assert_eq!(report.recovered_frames, 0);
        assert_eq!(report.rounds_used, 0);
        assert_eq!(report.final_strategy, RecoveryStrategy::Simple);
    }

    #[test]
    fn retransmit_loop_max_rounds_zero() {
        let frames = vec![make_frame(0, b"z0"), make_frame(1, b"z1")];
        let mut rl = RetransmitLoop::new(0, 100, RecoveryStrategy::Simple, frames);
        rl.inject_loss(&[0]);
        rl.run().expect("run with zero max rounds");

        let report = rl.report();
        // max_rounds=0 means no rounds execute.
        assert_eq!(report.rounds_used, 0);
        assert_eq!(report.recovered_frames, 0);
        assert_eq!(report.lost_frames, 1);
    }

    #[test]
    fn recovery_strategy_serde_roundtrip() {
        for strategy in [
            RecoveryStrategy::Simple,
            RecoveryStrategy::Redundant,
            RecoveryStrategy::Escalate,
        ] {
            let json = serde_json::to_string(&strategy).expect("serialize");
            let parsed: RecoveryStrategy = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, strategy);
        }
    }

    #[test]
    fn transcript_segment_compact_from_transcription_segment() {
        use crate::model::TranscriptionSegment;
        let seg = TranscriptionSegment {
            start_sec: Some(1.5),
            end_sec: Some(3.0),
            text: "hello".to_owned(),
            speaker: Some("A".to_owned()),
            confidence: Some(0.95),
        };
        let compact: TranscriptSegmentCompact = (&seg).into();
        assert_eq!(compact.s, Some(1.5));
        assert_eq!(compact.e, Some(3.0));
        assert_eq!(compact.t, "hello");
        assert_eq!(compact.sp.as_deref(), Some("A"));
        assert_eq!(compact.c, Some(0.95));

        // None speaker/confidence should be omitted from JSON
        let seg_none = TranscriptionSegment {
            start_sec: None,
            end_sec: None,
            text: "x".to_owned(),
            speaker: None,
            confidence: None,
        };
        let compact2: TranscriptSegmentCompact = (&seg_none).into();
        let json = serde_json::to_string(&compact2).expect("serialize");
        assert!(!json.contains("\"sp\""), "None speaker should be omitted");
        assert!(!json.contains("\"c\""), "None confidence should be omitted");
    }

    #[test]
    fn transcript_partial_frame_round_trips() {
        let frame = TtyControlFrame::TranscriptPartial {
            seq: 7,
            window_id: 3,
            segments: vec![TranscriptSegmentCompact {
                s: Some(0.0),
                e: Some(1.5),
                t: "hello world".to_owned(),
                sp: Some("speaker_0".to_owned()),
                c: Some(0.92),
            }],
            model_id: "whisper-tiny".to_owned(),
            speculative: true,
        };
        let json = serde_json::to_string(&frame).expect("serialize");
        let parsed: TtyControlFrame = serde_json::from_str(&json).expect("deserialize");
        match parsed {
            TtyControlFrame::TranscriptPartial {
                seq,
                window_id,
                segments,
                speculative,
                ..
            } => {
                assert_eq!(seq, 7);
                assert_eq!(window_id, 3);
                assert_eq!(segments.len(), 1);
                assert_eq!(segments[0].t, "hello world");
                assert!(speculative);
            }
            other => assert!(
                matches!(other, TtyControlFrame::TranscriptPartial { .. }),
                "expected TranscriptPartial"
            ),
        }
    }

    #[test]
    fn emit_tty_transcript_partial_writes_correct_ndjson() {
        use crate::model::TranscriptionSegment;
        let segments = vec![TranscriptionSegment {
            start_sec: Some(0.0),
            end_sec: Some(2.0),
            text: "test text".to_owned(),
            speaker: None,
            confidence: Some(0.88),
        }];
        let mut out = Vec::new();
        emit_tty_transcript_partial(&mut out, 5, 2, &segments, "whisper-base", false)
            .expect("emit partial");

        let text = String::from_utf8(out).expect("utf8");
        let value: serde_json::Value = serde_json::from_str(text.trim()).expect("json");
        assert_eq!(value["frame_type"], "transcript_partial");
        assert_eq!(value["seq"], 5);
        assert_eq!(value["window_id"], 2);
        assert_eq!(value["model_id"], "whisper-base");
        assert_eq!(value["speculative"], false);
        assert_eq!(value["segments"][0]["t"], "test text");
    }

    #[test]
    fn emit_tty_transcript_retract_writes_correct_ndjson() {
        let mut out = Vec::new();
        emit_tty_transcript_retract(&mut out, 42, 7, "model_timeout").expect("emit retract");

        let text = String::from_utf8(out).expect("utf8");
        let value: serde_json::Value = serde_json::from_str(text.trim()).expect("json");
        assert_eq!(value["frame_type"], "transcript_retract");
        assert_eq!(value["retracted_seq"], 42);
        assert_eq!(value["window_id"], 7);
        assert_eq!(value["reason"], "model_timeout");
    }

    #[test]
    fn retransmit_plan_from_reader_identifies_gaps() {
        let frames = vec![
            make_frame(0, b"a"),
            make_frame(2, b"c"),
            make_frame(3, b"d"),
        ];
        let ndjson = frames_to_ndjson(&frames);
        let mut reader = ndjson.as_bytes();

        let plan = retransmit_plan_from_reader(&mut reader, DecodeRecoveryPolicy::SkipMissing)
            .expect("plan from reader");

        assert_eq!(plan.requested_sequences, vec![1]);
        assert_eq!(plan.gap_count, 1);
        assert_eq!(plan.integrity_failure_count, 0);
        assert_eq!(plan.dropped_frame_count, 0);
        assert_eq!(plan.protocol_version, SUPPORTED_PROTOCOL_VERSION);
        assert_eq!(plan.requested_ranges.len(), 1);
        assert_eq!(plan.requested_ranges[0].start_seq, 1);
        assert_eq!(plan.requested_ranges[0].end_seq, 1);
    }

    #[test]
    fn decompress_chunk_rejects_oversized_output() {
        // Create zlib-compressed data that decompresses to more than
        // MAX_DECOMPRESSED_FRAME_BYTES.  A run of zeros compresses very well.
        let big = vec![0u8; super::MAX_DECOMPRESSED_FRAME_BYTES + 1];
        let compressed = super::compress_chunk(&big).expect("compress should succeed");
        let result = super::decompress_chunk(&compressed);
        assert!(result.is_err(), "oversized decompression should fail");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("decompression bomb"),
            "error should mention decompression bomb, got: {msg}"
        );
    }

    #[test]
    fn decompress_chunk_accepts_normal_frame() {
        // 40 000 bytes = 5 seconds of 8 kHz mulaw (maximum normal frame size).
        let normal = vec![0u8; 40_000];
        let compressed = super::compress_chunk(&normal).expect("compress should succeed");
        let result = super::decompress_chunk(&compressed);
        assert!(
            result.is_ok(),
            "normal-sized frame should decompress successfully"
        );
        assert_eq!(result.unwrap().len(), 40_000);
    }

    #[test]
    fn decompress_chunk_accepts_exact_size_limit() {
        let normal = vec![0u8; super::MAX_DECOMPRESSED_FRAME_BYTES];
        let compressed = super::compress_chunk(&normal).expect("compress should succeed");
        let result = super::decompress_chunk(&compressed).expect("exact-limit frame should pass");
        assert_eq!(result.len(), super::MAX_DECOMPRESSED_FRAME_BYTES);
    }
}
