# TTY Audio Protocol

Low-bandwidth audio transport over TTY/PTY links using NDJSON framing.

Operator-focused replay/framing guarantees are documented separately in
[`docs/tty-replay-guarantees.md`](tty-replay-guarantees.md). This protocol doc
remains the wire-format source of truth.

---

## Version

- Audio frame protocol (`TtyAudioFrame.protocol_version`): `1`
- Transcript control extension protocol constant: `2` (`TRANSCRIPT_PROTOCOL_VERSION`)
- Field behavior:
  - audio frames carry `protocol_version`
  - control frames are selected by `frame_type`
- Decoder behavior for the audio decode path (`decode_tty_audio_from_reader`):
  - accepts only `protocol_version == 1`
  - rejects any other audio-frame version with a deterministic non-zero error
- Transcript correction control frames (`transcript_partial`, `transcript_retract`, `transcript_correct`) are parsed on the control channel and validated for framing/order; they are not negotiated via audio-frame `protocol_version`.

---

## Frame Schema (NDJSON)

Each line is one JSON frame:

- `protocol_version` (`u32`)
- `seq` (`u64`) -- strict contiguous sequence starting at `0`
- `codec` (`string`) -- currently must be `mulaw+zlib+b64`
- `sample_rate_hz` (`u32`) -- currently must be `8000`
- `channels` (`u8`) -- currently must be `1`
- `payload_b64` (`string`) -- base64 (no-pad) encoded zlib-compressed mu-law bytes
- `crc32` (`u32?`) -- optional CRC32 of the raw (pre-compression) audio bytes
- `payload_sha256` (`string?`) -- optional SHA-256 hex of the raw (pre-compression) audio bytes

Example audio frame:

```json
{"protocol_version":1,"seq":0,"codec":"mulaw+zlib+b64","sample_rate_hz":8000,"channels":1,"payload_b64":"eJxz...","crc32":3849212457,"payload_sha256":"a1b2c3..."}
```

---

## Control Frame Types

Control frames share the same NDJSON transport as audio frames but are distinguished by a `frame_type` field. The parser uses the presence of `frame_type` to route a line to control-frame handling rather than audio-frame handling.

### Handshake

Sent by the encoder before any audio frames to negotiate protocol version and codec.

| Field              | Type       | Description                                |
|--------------------|------------|--------------------------------------------|
| `frame_type`       | `string`   | Always `"handshake"`                       |
| `min_version`      | `u32`      | Lowest protocol version the sender supports|
| `max_version`      | `u32`      | Highest protocol version the sender supports|
| `supported_codecs` | `string[]` | List of codec identifiers the sender supports|

```json
{"frame_type":"handshake","min_version":1,"max_version":1,"supported_codecs":["mulaw+zlib+b64"]}
```

Version negotiation selects the highest version in the intersection of both sides' `[min_version, max_version]` ranges. If no overlap exists, the stream is rejected.

Rules:
- Handshake must appear before any audio frames.
- Duplicate handshake frames are rejected.
- The decoder validates that at least one offered codec matches `mulaw+zlib+b64`.

### Handshake Ack

Sent by the decoder to confirm the negotiated protocol version and codec.

| Field                | Type     | Description                              |
|----------------------|----------|------------------------------------------|
| `frame_type`         | `string` | Always `"handshake_ack"`                 |
| `negotiated_version` | `u32`    | The agreed-upon protocol version         |
| `negotiated_codec`   | `string` | The agreed-upon codec identifier         |

```json
{"frame_type":"handshake_ack","negotiated_version":1,"negotiated_codec":"mulaw+zlib+b64"}
```

Rules:
- Must appear after a handshake, never before.
- `negotiated_version` must match the version selected during handshake.
- `negotiated_codec` must be `mulaw+zlib+b64` for protocol 1.

### Retransmit Request

Sent by the decoder to request re-delivery of specific missing or corrupt frames.

| Field       | Type     | Description                                     |
|-------------|----------|-------------------------------------------------|
| `frame_type`| `string` | Always `"retransmit_request"`                   |
| `sequences` | `u64[]`  | Sequence numbers of frames to be retransmitted  |

```json
{"frame_type":"retransmit_request","sequences":[1,2,4]}
```

### Retransmit Response

Sent by the encoder to acknowledge which frames are being retransmitted.

| Field       | Type     | Description                                       |
|-------------|----------|---------------------------------------------------|
| `frame_type`| `string` | Always `"retransmit_response"`                    |
| `sequences` | `u64[]`  | Sequence numbers of frames included in the retransmission |

```json
{"frame_type":"retransmit_response","sequences":[1,2,4]}
```

### Ack

Sent by the decoder to acknowledge receipt of contiguous frames.

| Field        | Type     | Description                                  |
|--------------|----------|----------------------------------------------|
| `frame_type` | `string` | Always `"ack"`                               |
| `up_to_seq`  | `u64`    | Highest contiguous sequence number received  |

```json
{"frame_type":"ack","up_to_seq":42}
```

### Backpressure

Sent by the decoder to signal flow control.

| Field                | Type     | Description                                |
|----------------------|----------|--------------------------------------------|
| `frame_type`         | `string` | Always `"backpressure"`                    |
| `remaining_capacity` | `u64`    | Decoder's remaining buffer capacity in frames |

```json
{"frame_type":"backpressure","remaining_capacity":64}
```

### Transcript Partial

Sent by the speculative/fast transcript lane to provide provisional text for a window.

| Field         | Type       | Description                                      |
|---------------|------------|--------------------------------------------------|
| `frame_type`  | `string`   | Always `"transcript_partial"`                    |
| `seq`         | `u64`      | Transcript sequence number                        |
| `window_id`   | `u64`      | Correlation window identifier                     |
| `segments`    | `object[]` | Compact segments with short keys (`s`,`e`,`t`,`sp`,`c`) |
| `model_id`    | `string`   | Origin model identifier                           |
| `speculative` | `bool`     | Whether this partial is speculative               |

```json
{"frame_type":"transcript_partial","seq":42,"window_id":7,"segments":[{"s":0.0,"e":0.8,"t":"helo wrld","c":0.62}],"model_id":"fast-v1","speculative":true}
```

### Transcript Retract

Sent to retract a previous speculative partial when quality checks fail.

| Field           | Type     | Description                                |
|-----------------|----------|--------------------------------------------|
| `frame_type`    | `string` | Always `"transcript_retract"`              |
| `retracted_seq` | `u64`    | Transcript sequence being retracted        |
| `window_id`     | `u64`    | Correlation window identifier               |
| `reason`        | `string` | Deterministic reason for retraction         |

```json
{"frame_type":"transcript_retract","retracted_seq":42,"window_id":7,"reason":"quality_model_disagreed"}
```

### Transcript Correct

Sent after a retract to provide corrected segments from the quality lane.

| Field           | Type       | Description                                      |
|-----------------|------------|--------------------------------------------------|
| `frame_type`    | `string`   | Always `"transcript_correct"`                    |
| `correction_id` | `u64`      | Unique correction identifier                      |
| `replaces_seq`  | `u64`      | Retracted transcript sequence being replaced      |
| `window_id`     | `u64`      | Correlation window identifier                     |
| `segments`      | `object[]` | Corrected compact segments (`s`,`e`,`t`,`sp`,`c`)|
| `model_id`      | `string`   | Quality model identifier                          |
| `drift_wer`     | `f64`      | Approximate drift/error estimate                  |

```json
{"frame_type":"transcript_correct","correction_id":10,"replaces_seq":42,"window_id":7,"segments":[{"s":0.0,"e":0.8,"t":"hello world","c":0.89}],"model_id":"quality-v2","drift_wer":0.12}
```

### Session Close

Sent to signal session termination with a reason code (defined by `TtyControlFrame::SessionClose`).

| Field           | Type     | Description                                         |
|-----------------|----------|-----------------------------------------------------|
| `frame_type`    | `string` | Always `"session_close"`                            |
| `reason`        | `string` | One of `normal`, `error`, `timeout`, `peer_requested`|
| `last_data_seq` | `u64?`   | Highest data-frame sequence sent before close       |

```json
{"frame_type":"session_close","reason":"normal","last_data_seq":99}
```

The decoder validates that `last_data_seq` is consistent with the highest data-frame sequence actually observed in the stream.

---

## Decode Recovery Policies

Protocol `1` supports two explicit decode recovery policies, selected via the `--recovery` CLI flag:

### `fail_closed` (default for decode)

Decode fails immediately on any sequence or integrity violation:
- sequence gap (expected seq N, got seq M where M > N)
- duplicate or out-of-order sequence number
- base64 decode failure
- zlib decompression failure
- CRC32 mismatch (when `crc32` field is present)
- SHA-256 mismatch (when `payload_sha256` field is present)

Unsupported protocol version, codec, sample rate, or channel count always fail immediately regardless of policy.

### `skip_missing` (default for retransmit-plan and retransmit)

Decode continues processing later frames after encountering violations. Dropped frames are recorded in decode telemetry for downstream retransmit planning:

- `gaps` -- list of `{expected, got}` pairs recording sequence discontinuities
- `duplicates` -- list of sequence numbers seen more than once
- `integrity_failures` -- list of sequence numbers that failed CRC32, SHA-256, base64, or zlib checks
- `dropped_frames` -- union of all frames that could not be included in the output

Unsupported protocol/codec/shape still fail immediately even under `skip_missing`.

---

## Retransmit Plan

The retransmit plan is a deterministic artifact produced by scanning a frame stream with `skip_missing` policy and collecting all gap and integrity telemetry.

### Retransmit Plan Schema

| Field                      | Type                              | Description                                 |
|----------------------------|-----------------------------------|---------------------------------------------|
| `protocol_version`         | `u32`                             | Always `1`                                  |
| `requested_sequences`      | `u64[]`                           | Deduplicated, sorted list of missing/corrupt sequences |
| `requested_ranges`         | `{start_seq, end_seq}[]`          | Collapsed contiguous ranges for efficiency  |
| `gap_count`                | `usize`                           | Number of sequence gaps detected            |
| `integrity_failure_count`  | `usize`                           | Number of frames that failed integrity checks|
| `dropped_frame_count`      | `usize`                           | Total frames excluded from output           |

Example:

```json
{
  "protocol_version": 1,
  "requested_sequences": [1, 2, 4],
  "requested_ranges": [{"start_seq": 1, "end_seq": 2}, {"start_seq": 4, "end_seq": 4}],
  "gap_count": 1,
  "integrity_failure_count": 1,
  "dropped_frame_count": 1
}
```

The plan is built from:
1. Gap analysis: for each `{expected, got}` gap, every sequence in `[expected, got)` is added.
2. Integrity failures: every sequence that failed CRC32/SHA-256/base64/zlib is added.
3. The union is sorted and deduplicated.
4. Contiguous sequences are collapsed into `{start_seq, end_seq}` ranges.

---

## Retransmit-Loop Workflow

The retransmit loop automates the scan-request-respond cycle for recovering lost frames. It operates in two modes: the CLI `retransmit-loop` command and the programmatic `RetransmitLoop` struct.

### CLI Retransmit Loop

The CLI `retransmit-loop` command reads an NDJSON frame stream from stdin, builds a retransmit plan, and emits deterministic control frames:

1. Parse the frame stream using `skip_missing` policy (or `fail_closed` if overridden).
2. Build a retransmit plan from the decode report.
3. If no sequences need retransmission, emit a single `ack` frame and exit.
4. For each round (up to `--rounds`), emit a `retransmit_request` control frame listing all missing sequences.
5. After all rounds, emit a final `retransmit_response` control frame.

### Programmatic RetransmitLoop

The `RetransmitLoop` struct manages automatic retransmission with recovery-strategy escalation. It is fully deterministic: given the same input frames and the same loss pattern, the output and report are identical across runs.

Lifecycle:

1. **Create** with `RetransmitLoop::new(max_rounds, timeout_ms, recovery_strategy, frame_buffer)`.
2. **Inject loss** (optional) with `inject_loss(&[seq_numbers])` to mark specific frames as lost. Only sequences present in the frame buffer are recorded; unknown sequences are silently ignored. Calling `inject_loss` resets all prior recovery state.
3. **Run** with `run()` to execute recovery rounds.
4. **Report** with `report()` to obtain a `RetransmitReport` summary.

Each round of `run()`:
1. Identifies frames still lost (in the lost set but not yet recovered).
2. If none remain, stops early.
3. Applies the current `RecoveryStrategy` to recover frames (in sequence-number order for determinism).
4. Escalates the strategy for the next round.

---

## Recovery Strategy Semantics

The `RecoveryStrategy` enum controls how many frames are recovered per round and how the loop escalates under persistent loss.

### Simple

- Recovers **1 frame per round**.
- No modification to the retransmitted frame.
- This is the starting strategy when beginning with `RecoveryStrategy::Simple`.
- Escalates to `Redundant` for the next round.

### Redundant

- Recovers **up to 2 frames per round**.
- Conceptually adds forward error correction by duplicating each frame.
- Escalates to `Escalate` for the next round.

### Escalate

- Recovers **up to 4 frames per round**.
- Conceptually increases compression level to reduce frame size.
- `Escalate` is the ceiling -- it does not escalate further. Subsequent rounds remain at `Escalate`.

### Escalation Chain

```
Simple (1/round) -> Redundant (2/round) -> Escalate (4/round) -> Escalate (4/round) -> ...
```

Example: 7 lost frames starting with `Simple` strategy:
- Round 1 (`Simple`): recovers 1 frame. 6 remaining.
- Round 2 (`Redundant`): recovers 2 frames. 4 remaining.
- Round 3 (`Escalate`): recovers 4 frames. 0 remaining. Loop stops.
- Total: 3 rounds used.

Starting with a higher strategy skips the lower tiers:
- Starting at `Redundant`: recovers 2/round, escalates to `Escalate` (4/round).
- Starting at `Escalate`: recovers 4/round from the first round onward.

### RetransmitReport

The `report()` method returns a summary after the loop completes (or before `run()` is called for pre-run diagnostics):

| Field              | Type               | Description                                    |
|--------------------|--------------------|------------------------------------------------|
| `total_frames`     | `usize`            | Total frames in the buffer                     |
| `lost_frames`      | `usize`            | Number of frames marked as lost                |
| `recovered_frames` | `usize`            | Number of lost frames recovered during the loop|
| `rounds_used`      | `u32`              | Number of recovery rounds actually executed    |
| `final_strategy`   | `RecoveryStrategy` | The strategy active in the final round         |

---

## Adaptive Bitrate Controller

The `AdaptiveBitrateController` monitors link quality and adjusts compression dynamically based on observed frame loss rate:

| Loss Rate         | Link Quality | Compression Level  | Critical Frame FEC |
|-------------------|--------------|--------------------|---------------------|
| < 1%              | High         | Fast (level 1)     | 1x (no duplication) |
| 1% -- 10%         | Moderate     | Default (level 6)  | 2x                  |
| > 10%             | Poor         | Best (level 9)     | 3x                  |

Critical control frames (handshake, session close) are emitted multiple times based on the FEC redundancy factor via `emit_critical_frame_with_fec()`.

---

## Machine-Output Guarantees (Robot Mode)

All TTY audio CLI commands produce machine-readable output suitable for agent orchestration:

1. **NDJSON transport**: every output line is a self-contained JSON object. One JSON object per line, newline-delimited.
2. **Stable schema**: output shapes are versioned (`schema_version` follows the robot contract's `ROBOT_SCHEMA_VERSION`, currently `1.1.0`). Breaking changes require a version bump.
3. **Deterministic output**: given identical input and parameters, all commands produce byte-identical output. The retransmit loop is fully deterministic with no time-dependent behavior (`timeout_ms` is advisory/reporting only; no sleeps).
4. **Structured errors**: errors are reported as JSON with explicit error codes, never mixed with human-readable decoration.
5. **Pipe-friendly**: all data frames and control frames go to stdout. Diagnostic/error information goes to stderr. Output can be safely piped into downstream tools (`jq`, other `tty-audio` subcommands, agent orchestrators).
6. **Mic stream events**: real-time microphone capture wraps each `TtyAudioFrame` in a `MicStreamEvent` envelope with `event: "mic_audio_chunk"` and `schema_version`, integrating with the project's standard NDJSON event protocol.

Control frames and audio frames coexist in the same NDJSON stream. Consumers distinguish them by checking for the presence of `frame_type` (control) vs. `seq`/`codec` (audio).

---

## CLI Command Reference

### Encode

Transcode an audio file to NDJSON TTY audio frames on stdout.

```bash
cargo run -- tty-audio encode --input /path/to/audio.wav
cargo run -- tty-audio encode --input /path/to/audio.wav --chunk-ms 500
```

The encoder:
1. Emits a `handshake` control frame.
2. Transcodes the input to 8 kHz mono mu-law via ffmpeg.
3. Splits the mu-law bytes into chunks (default 200 ms, range 20--5000 ms).
4. Emits one NDJSON audio frame per chunk with CRC32 and SHA-256 integrity fields.

### Decode

Decode NDJSON TTY audio frames from stdin to a WAV file.

```bash
# strict mode (default): fail on any gap or corruption
cat frames.ndjson | cargo run -- tty-audio decode --output restored.wav

# tolerant mode: skip missing/corrupt frames
cat frames.ndjson | cargo run -- tty-audio decode --output restored.wav --recovery skip_missing
```

### Retransmit Plan

Scan a frame stream and emit a retransmit plan (JSON) listing missing/corrupt sequences.

```bash
cat frames.ndjson | cargo run -- tty-audio retransmit-plan
cat frames.ndjson | cargo run -- tty-audio retransmit-plan --recovery fail_closed
```

Default recovery policy for retransmit-plan is `skip_missing` (to collect full gap telemetry).

### Control Frame Commands

Emit individual control frames to stdout for scripted agent orchestration.

```bash
# handshake (defaults: version 1, codec mulaw+zlib+b64)
cargo run -- tty-audio control handshake
cargo run -- tty-audio control handshake --min-version 1 --max-version 1 --codec mulaw+zlib+b64

# handshake acknowledgment
cargo run -- tty-audio control handshake-ack --negotiated-version 1 --negotiated-codec mulaw+zlib+b64

# acknowledge receipt up to a sequence number
cargo run -- tty-audio control ack --up-to-seq 42

# signal decoder buffer pressure
cargo run -- tty-audio control backpressure --remaining-capacity 64

# request retransmission of specific sequences
cargo run -- tty-audio control retransmit-request --sequences 1,2,4

# signal retransmission of specific sequences
cargo run -- tty-audio control retransmit-response --sequences 1,2,4
```

### Retransmit Loop

Scan a frame stream from stdin and emit a deterministic sequence of retransmit request/response control frames.

```bash
# single round (default)
cat frames.ndjson | cargo run -- tty-audio control retransmit-loop

# multiple rounds
cat frames.ndjson | cargo run -- tty-audio control retransmit-loop --rounds 3

# with explicit recovery policy
cat frames.ndjson | cargo run -- tty-audio control retransmit-loop --recovery fail_closed --rounds 2
```

### Convenience Commands

Shorthand commands that mirror the `control` subcommands.

```bash
# emit a control frame by kind (handshake, eof, reset)
cargo run -- tty-audio send-control handshake
cargo run -- tty-audio send-control eof
cargo run -- tty-audio send-control reset

# run retransmit loop directly (equivalent to control retransmit-loop)
cat frames.ndjson | cargo run -- tty-audio retransmit
cat frames.ndjson | cargo run -- tty-audio retransmit --recovery fail_closed --rounds 3
```

---

## End-to-End Workflow Example

A typical agent-orchestrated TTY audio session:

```bash
# 1. Encode audio to NDJSON frames
cargo run -- tty-audio encode --input recording.wav > frames.ndjson

# 2. Simulate a lossy link (e.g., drop some lines)
#    In production this is the actual TTY/PTY transport.

# 3. On the receiving end, attempt decode
cat frames.ndjson | cargo run -- tty-audio decode --output restored.wav --recovery skip_missing

# 4. If frames were lost, generate a retransmit plan
cat frames.ndjson | cargo run -- tty-audio retransmit-plan
# Output: {"protocol_version":1,"requested_sequences":[3,7],...}

# 5. Run automated retransmit loop
cat frames.ndjson | cargo run -- tty-audio control retransmit-loop --rounds 3
# Output: retransmit_request lines followed by a retransmit_response line

# 6. Sender retransmits requested frames and receiver re-decodes
```

---

## Integrity Checks

Two optional integrity mechanisms are available, computed over raw (pre-compression) audio bytes:

- **CRC32**: fast integrity check. Mismatch causes the frame to be dropped (under `skip_missing`) or the stream to fail (under `fail_closed`).
- **SHA-256**: strong integrity check. Comparison is case-insensitive hex. Same drop/fail behavior as CRC32.

Both fields are optional for backward compatibility. When both are present, both are validated.

---

## Violations Covered by Policy Handling

- Sequence gaps (`expected != got`)
- Duplicate or out-of-order sequence numbers
- Payload base64 decode failure
- Payload zlib decompression failure
- `crc32` mismatch (when provided)
- `payload_sha256` mismatch (when provided)

Violations that always fail regardless of policy:
- Unsupported `protocol_version`
- Unsupported `codec`
- Unsupported `sample_rate_hz` or `channels`
- Handshake ordering violations (handshake after audio, duplicate handshake, handshake_ack before handshake)

---

## Compatibility Rules

- New optional fields may be added in future minor revisions.
- Behavior-changing protocol changes must increment `protocol_version`.
- Decoder for protocol `1` must reject unknown major versions rather than attempting heuristic decode.
- Control frames with unknown `frame_type` values should be ignored by consumers to allow forward-compatible extension.
